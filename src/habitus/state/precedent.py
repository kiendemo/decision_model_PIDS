"""Persistent precedent answers: RefIndex(host, ref).nearest per (ds, reference days, e).

A signature hashes its whole card and RefIndex.nearest answers each query card on its own
(row-wise TF-IDF products rounded to 1e-9, per-row top-e), so a cached answer is the fresh one and
the states are byte-identical with or without the cache.

Two layers, same file name <sha1(','.join(sorted(set(ref))))>_e<e>.json.gz and the same JSON
{"ds", "ref", "e", "nearest": {sig: [reference sigs]}} (sort_keys, compact):
  base     settings.paths.precedent/<ds>: the shipped cache. Read-only: never locked, never written.
  overlay  settings.paths.runs/precedent_cache/<ds>: answers computed on a miss. Writers merge
           under an flock on <file>.lock, never overwrite a stored answer, and raise on a
           disagreeing one (nearest() is deterministic per card: a disagreement is a bug).
set_roots() replaces either root (None disables that layer; both None = no persistent cache).
"""

import fcntl
import gzip
import hashlib
import json
import os
import time
from collections import OrderedDict
from pathlib import Path

from habitus.log import log
from habitus.state.profile import host_key

PREC_E, PREC_OPS = 2, 4      # neighbours per precedent section, top ops shown per neighbour
CHUNK_CELLS = 1e7            # dense (queries x reference cards) cells per RefIndex.nearest call
PREC_SUBDIR = "precedent_cache"

_REFINDEX = OrderedDict()
MAX_REFINDEX = 2             # a THEIA RefIndex holds ~10^5 reference cards x 2 TF-IDF matrices
_PCACHE = OrderedDict()
MAX_PCACHE = 8
_UNSET = object()
_ROOTS = {"base": _UNSET, "overlay": _UNSET}


def get_refindex(host, ref):
    from habitus.state.refindex import RefIndex
    key = (*host_key(host), tuple(sorted(set(ref))))
    if key in _REFINDEX:
        _REFINDEX.move_to_end(key)
        return _REFINDEX[key]
    t0 = time.time()
    ri = _REFINDEX[key] = RefIndex(host, list(key[2]))
    while len(_REFINDEX) > MAX_REFINDEX:
        _REFINDEX.popitem(last=False)
    log(f"{host.ds} RefIndex ref={list(key[2])}: {ri.n} reference cards ({time.time() - t0:.1f}s)",
        "state.precedent")
    return ri


def _settings():
    from habitus import config
    s = config.settings
    return s() if callable(s) and not hasattr(s, "paths") else s


def set_roots(base=_UNSET, overlay=_UNSET):
    """Override the base and/or overlay root (a directory, or None to disable that layer).
    Clears the in-memory cache objects so the next lookup uses the new roots."""
    if base is not _UNSET:
        _ROOTS["base"] = base
    if overlay is not _UNSET:
        _ROOTS["overlay"] = overlay
    _PCACHE.clear()


def reset_roots():
    """Back to the settings roots."""
    _ROOTS["base"] = _ROOTS["overlay"] = _UNSET
    _PCACHE.clear()


def precedent_roots(settings=None):
    """(base root, overlay root): settings.paths.precedent and settings.paths.runs/precedent_cache
    unless set_roots() replaced them; None = that layer is off."""
    base, overlay = _ROOTS["base"], _ROOTS["overlay"]
    if base is _UNSET or overlay is _UNSET:
        paths = (settings or _settings()).paths
        if base is _UNSET:
            base = getattr(paths, "precedent", None)
        if overlay is _UNSET:
            runs = getattr(paths, "runs", None)
            overlay = Path(runs) / PREC_SUBDIR if runs else None
    return (None if base is None else str(base)), (None if overlay is None else str(overlay))


def precedent_cache_dirs(host):
    """(base dir, overlay dir) of a host: <root>/<ds>, None when that layer is off."""
    base, overlay = precedent_roots()
    return (None if base is None else os.path.join(base, host.ds),
            None if overlay is None else os.path.join(overlay, host.ds))


class PrecedentCache:
    """RefIndex(host, ref).nearest(cards, type, e) answers of one (ds, reference, e) as
    {sig: [reference sigs]}, in memory (`map`: base and overlay merged) and in
    <directory>/<sha1(','.join(ref))>_e<e>.json.gz. `directory` is the writable overlay (None: misses
    are kept in memory only); `base` is a read-only directory holding the shipped file."""

    def __init__(self, directory, ds, ref, e, base=None):
        self.ds, self.ref, self.e = ds, tuple(ref), int(e)
        name = hashlib.sha1(",".join(self.ref).encode("utf-8")).hexdigest()
        self.name = f"{name}_e{self.e}.json.gz"
        self.path = os.path.join(directory, self.name) if directory else None
        self.base_path = os.path.join(base, self.name) if base else None
        self.map, self.own, self._stamp, self._base_stamp = {}, {}, None, None
        self.refresh()

    @staticmethod
    def _stat(path):
        if path is None:
            return None
        try:
            st = os.stat(path)
        except FileNotFoundError:
            return None
        return st.st_ino, st.st_size, st.st_mtime_ns

    def _read(self, path):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            obj = json.load(f)
        if obj.get("ds") != self.ds or tuple(obj.get("ref") or ()) != self.ref or obj.get("e") != self.e:
            raise ValueError(f"{path} holds ds {obj.get('ds')} ref {obj.get('ref')} e {obj.get('e')}, "
                             f"not {self.ds} {list(self.ref)} {self.e}")
        return obj["nearest"]

    def _absorb(self, other, where):
        bad = [s for s, v in other.items() if s in self.map and self.map[s] != v]
        if bad:
            raise RuntimeError(f"precedent cache {self.name} ({self.ds}): {len(bad)} answers disagree with "
                               f"{where} (e.g. {bad[0]}: {self.map[bad[0]]} vs {other[bad[0]]})")
        for s, v in other.items():
            self.map.setdefault(s, v)

    def refresh(self):
        """Merge the base and overlay files into memory when they changed since they were last read."""
        st = self._stat(self.base_path)
        if st is not None and st != self._base_stamp:
            self._absorb(self._read(self.base_path), "the shipped cache file")
            self._base_stamp = st
        st = self._stat(self.path)
        if st is not None and st != self._stamp:
            disk = self._read(self.path)
            self._absorb(disk, "the overlay cache file")
            for s, v in disk.items():
                self.own.setdefault(s, v)
            self._stamp = st

    def lookup(self, sigs):
        """The signatures of `sigs` without a cached answer (the files are re-read once if needed)."""
        miss = [s for s in sigs if s not in self.map]
        if miss:
            self.refresh()
            miss = [s for s in miss if s not in self.map]
        return miss

    def add(self, new):
        """Keep fresh answers and persist the overlay's union with whatever other writers stored."""
        if not new:
            return
        self._absorb(new, "a fresh nearest() answer")
        for s, v in new.items():
            self.own.setdefault(s, v)
        if self.path is None:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path + ".lock", "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                disk = self._read(self.path) if os.path.exists(self.path) else {}
                self._absorb(disk, "the overlay cache file")
                for s, v in disk.items():
                    self.own.setdefault(s, v)
                if len(disk) < len(self.own):
                    tmp = f"{self.path}.tmp{os.getpid()}"
                    with open(tmp, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
                        gz.write(json.dumps({"ds": self.ds, "ref": list(self.ref), "e": self.e,
                                             "nearest": self.own},
                                            separators=(",", ":"), sort_keys=True).encode("utf-8"))
                    os.replace(tmp, self.path)
                self._stamp = self._stat(self.path)
            finally:
                fcntl.flock(lk, fcntl.LOCK_UN)


def precedent_cache(host, ref, e=PREC_E):
    """The PrecedentCache of (host, ref, e), cached in memory (LRU of MAX_PCACHE); None when off."""
    base, overlay = precedent_cache_dirs(host)
    if base is None and overlay is None:
        return None
    key = (base, overlay, host.ds, tuple(sorted(set(ref))), int(e))
    if key in _PCACHE:
        _PCACHE.move_to_end(key)
        return _PCACHE[key]
    pc = _PCACHE[key] = PrecedentCache(overlay, host.ds, key[3], e, base=base)
    while len(_PCACHE) > MAX_PCACHE:
        _PCACHE.popitem(last=False)
    return pc


def nearest_sigs(host, ref, sigs, refindex=None):
    """{sig: [<= PREC_E reference sigs]} = RefIndex(host, ref).nearest of each sig's card. The
    persistent cache is read first; only its misses build the RefIndex (get_refindex) and are
    queried, per type in chunks of CHUNK_CELLS dense cells, then stored. An explicit `refindex`
    bypasses the persistent cache (it may not be RefIndex(host, ref))."""
    ref = tuple(sorted(set(ref)))
    sigs = list(dict.fromkeys(sigs))
    pc = None if refindex is not None else precedent_cache(host, ref)
    miss = sigs if pc is None else pc.lookup(sigs)
    new = {}
    if miss:
        ri = refindex if refindex is not None else get_refindex(host, ref)
        by_type = {}
        for s in miss:
            by_type.setdefault(host.card(s)["type"], []).append(s)
        for t, ss in by_type.items():
            n_ref = len(ri.rows.get(t, ()))
            step = max(1, int(CHUNK_CELLS // max(1, n_ref)))
            for a in range(0, len(ss), step):
                chunk = ss[a:a + step]
                new.update(zip(chunk, ri.nearest([host.card(s) for s in chunk], t, e=PREC_E)))
        if pc is not None:
            pc.add(new)
    return new if pc is None else {s: pc.map[s] for s in sigs}
