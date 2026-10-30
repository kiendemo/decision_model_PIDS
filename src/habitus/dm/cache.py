"""Answer cache of a decision model: one namespace (tag) per backend configuration and dataset.

    cache = AnswerCache("qwen3.5-4b", "CADETS_E3", base=<release answers .jsonl.gz>, overlay=<runs/...jsonl>)
    rec = cache.get(key)          # {"z", "lse", "n", "cov"} or None
    cache.put(key, rec); cache.flush(); cache.close()

Records are JSON lines {"k": sha1 of the prompt, "z": letter logits A..D (4 dp), "lse": logsumexp
(4 dp), "n": offered letters, "cov": coverage (6 dp)}. A line that does not parse or whose z / lse /
cov is not finite is skipped (counted in stats["bad"]; a warning names the file); the FIRST record
of a key wins (base before overlay, and in line order inside a file), as in memory.

The original Laya caches (encoder.py _Member.run) stored {"k", "z" (5 dp), "n"} only. Such a line
is accepted when len(z) == n and z is finite (the original _load_cache rule) and completed as
dm.laya.cache_record would write it: "lse" = logsumexp(z) (5 dp), "cov" = 1.0.

base     a shipped .jsonl or .jsonl.gz, opened read-only; no lock file is created next to it.
overlay  an append-only .jsonl under runs/. The first write takes a non-blocking flock on
         <overlay>.lock for the cache's lifetime (one writer process per (tag, ds)); every flush
         appends under an flock on the file itself, so a crash loses at most one partial line
         (skipped on load) and two writers never interleave lines. A reader never creates either.
"""

from __future__ import annotations

import fcntl
import gzip
import json
import math
import os
from pathlib import Path

FIELDS = ("z", "lse", "n", "cov")


def finite_record(r):
    """An answer record whose logits, logsumexp and coverage are all finite numbers."""
    return (all(math.isfinite(float(x)) for x in r["z"]) and math.isfinite(float(r["lse"]))
            and math.isfinite(float(r["cov"])))


def complete_record(r):
    """`r` itself, or, for an original-format Laya record {"z", "n"} (no "lse" and no "cov") with
    len(z) == n and finite z, `r` completed with "lse" = logsumexp(z) rounded to 5 dp and "cov" = 1.0
    (dm.laya.cache_record). Any other record is returned unchanged (finite_record judges it)."""
    if "lse" in r or "cov" in r or "n" not in r:
        return r
    z = [float(x) for x in r["z"]]
    if len(z) != int(r["n"]) or not z or not all(math.isfinite(x) for x in z):
        return r
    m = max(z)
    r["lse"] = round(m + math.log(sum(math.exp(x - m) for x in z)), 5)
    r["cov"] = 1.0
    return r


def _open_text(path):
    p = str(path)
    if p.endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8")
    return open(p, encoding="utf-8")


class AnswerCache:
    """First-record-wins answer store over a read-only base and a writable overlay (module docstring).

    read_only: no put / flush (cache-only replay). Files are read on the first lookup."""

    def __init__(self, tag, ds, base=None, overlay=None, read_only=False):
        self.tag, self.ds = tag, ds
        self.base = Path(base) if base is not None else None
        self.overlay = Path(overlay) if overlay is not None else None
        self.read_only = bool(read_only)
        self._answers = None
        self._pending = []
        self._lock = None
        self.stats = {"loaded": 0, "bad": 0, "dup": 0, "put": 0, "written": 0}

    # --- reading --------------------------------------------------------------------------------- #
    def _read(self, path):
        bad = dup = n = 0
        with _open_text(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    k = r.pop("k")
                    r = complete_record(r)
                    ok = finite_record(r)
                except (ValueError, KeyError, TypeError, AttributeError):
                    ok = False
                if not ok:
                    bad += 1
                elif k in self._answers:
                    dup += 1                # the first answer of a key wins, as in memory
                else:
                    self._answers[k] = r
                    n += 1
        self.stats["loaded"] += n
        self.stats["bad"] += bad
        self.stats["dup"] += dup
        if bad:
            from habitus.log import log
            log(f"answer cache {self.tag}/{self.ds}: {bad} line(s) of {path} skipped (unparsable, a missing "
                f"field or a non-finite value); {n} loaded", tag="cache")

    def _load(self):
        if self._answers is None:
            self._answers = {}
            for p in (self.base, self.overlay):
                if p is not None and p.is_file():
                    self._read(p)
        return self._answers

    def get(self, key):
        """The record of `key` ({"z", "lse", "n", "cov"}) or None."""
        return self._load().get(key)

    def __contains__(self, key):
        return key in self._load()

    def __len__(self):
        return len(self._load())

    # --- writing --------------------------------------------------------------------------------- #
    def _take_lock(self):
        if self._lock is not None:
            return
        path = str(self.overlay) + ".lock"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        f = open(path, "a")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            f.close()
            raise RuntimeError(f"answer cache {self.overlay} is held by another writer (one writer process "
                               f"per (tag, ds))") from None
        self._lock = f

    def put(self, key, rec):
        """Store one answer record (fields z, lse, n, cov); returns the record the cache now holds for
        `key` (an existing record of the key wins and nothing is queued)."""
        if self.read_only:
            raise PermissionError(f"answer cache {self.tag}/{self.ds} is read-only")
        answers = self._load()
        if key in answers:
            return answers[key]
        rec = {f: rec[f] for f in FIELDS}
        if not finite_record(rec):
            raise ValueError(f"non-finite answer record for {key}")
        if self.overlay is not None:
            self._take_lock()
        answers[key] = rec
        self._pending.append({"k": key, **rec})
        self.stats["put"] += 1
        return rec

    @property
    def pending(self):
        return len(self._pending)

    def flush(self):
        """Append the queued records to the overlay (nothing without an overlay)."""
        if not (self._pending and self.overlay is not None):
            self._pending = []
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.overlay)), exist_ok=True)
        with open(self.overlay, "a+b") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.seek(0, os.SEEK_END)
                lead = b""
                if f.tell() > 0:   # a crashed writer may have left a partial last line
                    f.seek(-1, os.SEEK_END)
                    lead = b"" if f.read(1) == b"\n" else b"\n"
                f.write(lead + "".join(json.dumps(r, separators=(",", ":")) + "\n"
                                       for r in self._pending).encode("utf-8"))
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        self.stats["written"] += len(self._pending)
        self._pending = []

    def close(self):
        if not self.read_only:
            self.flush()
        if self._lock is not None:
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            self._lock.close()
            self._lock = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def cache_paths(tag, ds, settings):
    """(base, overlay) of (tag, ds): base = <paths.answers>/<tag>/<ds>.jsonl.gz (or .jsonl) when it
    exists, else None; overlay = <paths.runs>/answers/<tag>/<ds>.jsonl (never created here)."""
    base = None
    answers = getattr(settings.paths, "answers", None)
    if answers is not None:
        for ext in (".jsonl.gz", ".jsonl"):
            p = Path(str(answers)) / tag / f"{ds}{ext}"
            if p.is_file():
                base = p
                break
    overlay = Path(str(settings.paths.runs)) / "answers" / tag / f"{ds}.jsonl"
    return base, overlay
