"""Score files and node scores: per-signature DecisionModel records -> node scores over the test
universe.

    node_scores(host, by_day, field)    node score = max over the test days the node appears on of
                                        its signature's value that day (head) or the fallback
                                        (non-head); a float array aligned to host.universe, no NaN
    day_values(host, day, recs, field)  the same per node-day on one day (null pools use it)

`by_day = {day: {sig: rec}}`, rec = an E1 record {"qv", "qt", "cov_min"}, an E2 record
{"qv", "qt", "qa", "hard", "cov_min", "base", "composite"} or a plain float (see composite.value).

A score file `<root>/<ds>/<level>_k<k>/<day>.json.gz` holds
{"meta", "head", "scores": {sig: rec}, "fallback": {sig: float}}, meta first (read_meta relies on
it). Test days and clean (null) days of the same (level, k) live side by side: a clean day's
reference is the LODO one, fixed by k. The "fallback" block is informational: node_scores
recomputes the fallback from the day's cards (select.fallback_value) and never reads it.

The default read root is settings.paths.scores (the read-only release). Writes go to runs/ only:
write_score refuses any path under the release data dir or the read-only source trees.

Per-day node arrays (ids, signature codes, fallback) are built once per (host, day) and reused for
every (level, k, field).
"""

import gzip
import json
import math
import os
import weakref

import numpy as np

from habitus.decide.composite import value
from habitus.state.select import fallback_value

META_BYTES = 1 << 16                     # read_meta reads at most this many characters

_DAYS = weakref.WeakKeyDictionary()      # host -> {day: (ids, inv, usigs, fallback per usig)}


# --------------------------------------------------------------------------- #
# Settings / roots
# --------------------------------------------------------------------------- #
def _settings(settings=None):
    if settings is not None:
        return settings
    from habitus import config
    s = config.settings
    return s() if callable(s) else s


def default_root(settings=None):
    """The read root: settings.paths.scores (the release score files, read-only)."""
    return str(_settings(settings).paths.scores)


def runs_root(settings=None):
    """The default write root: <settings.paths.runs>/scores."""
    return os.path.join(str(_settings(settings).paths.runs), "scores")


def _protected(settings=None):
    """Directories no score file may be written under: the release data dir (and its scores dir)
    and the read-only source trees named in the settings."""
    from habitus.config import READ_ONLY_PATHS, RELEASE_PATHS
    p = _settings(settings).paths
    out = []
    for key in (*RELEASE_PATHS, *READ_ONLY_PATHS):
        v = getattr(p, key, None)
        if v:
            out.append(os.path.realpath(str(v)))
    return out


def _under(path, root):
    path, root = os.path.realpath(path), os.path.realpath(root)
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def check_writable(path, settings=None):
    """Raise PermissionError when `path` lies under the release dir or a read-only source tree."""
    for root in _protected(settings):
        if _under(path, root):
            raise PermissionError(f"refusing to write {path}: it is under the read-only {root}; "
                                  "score files are written under runs/ only")
    return path


# --------------------------------------------------------------------------- #
# Labels of k
# --------------------------------------------------------------------------- #
def klabel(k):
    return "all" if k is None else str(k)


def parse_k(s):
    return [None if x == "all" else int(x) for x in str(s).split(",") if x != ""]


# --------------------------------------------------------------------------- #
# Score files
# --------------------------------------------------------------------------- #
def score_path(ds, level, k, day, root=None):
    root = default_root() if root is None else root
    return os.path.join(str(root), ds, f"{level}_k{klabel(k)}", f"{day}.json.gz")


def write_score(obj, path, settings=None):
    """Atomic write (tmp + rename) of a score file, compact JSON in dict order (meta first),
    deterministic gzip header. Refuses the release dir and the read-only source trees."""
    path = str(path)
    check_writable(path, settings)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    try:
        with open(tmp, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            gz.write(json.dumps(obj, separators=(",", ":")).encode("utf-8"))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def read_score(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


write_json_gz = write_score             # the original names (kcs1.score)
read_json_gz = read_score


def read_meta(path):
    """The "meta" block of a score file without parsing its scores / fallback (write_score puts
    it first): at most the first 64 KiB are read; any other layout falls back to a full read."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        text = f.read(META_BYTES)
    if text.startswith('{"meta":'):
        try:
            return json.JSONDecoder().raw_decode(text, 8)[0]
        except ValueError:
            pass
    return read_score(path).get("meta", {})


def load_by_day(ds, level, k, days, root=None, metas=None):
    """{day: {sig: rec}} for the days whose score file exists, plus the list of missing days.
    `metas`: a dict that receives {day: file meta} (caps, ref, engine, ...)."""
    out, missing = {}, []
    for d in days:
        p = score_path(ds, level, k, d, root)
        if os.path.exists(p):
            raw = read_score(p)
            out[d] = raw["scores"]
            if metas is not None:
                metas[d] = raw.get("meta", {})
        else:
            missing.append(d)
    return out, missing


# --------------------------------------------------------------------------- #
# Node scores
# --------------------------------------------------------------------------- #
def _per_host(store, host):
    try:
        return store.setdefault(host, {})
    except TypeError:                       # a test double that cannot be weak-referenced: no cache
        return {}


def upos(universe, ids):
    """Universe position of each id, -1 outside (Host.upos without needing a Host)."""
    u = np.asarray(universe, dtype=np.int64)
    pos = np.searchsorted(u, ids)
    ok = pos < len(u)
    ok[ok] = u[pos[ok]] == ids[ok]
    return np.where(ok, pos, -1)


def day_arrays(host, day):
    """(node ids, signature code per node, distinct signatures, fallback per distinct sig)."""
    per = _per_host(_DAYS, host)
    hit = per.get(day)
    if hit is not None:
        return hit
    nd = host.nodes(day)
    ids = np.fromiter(nd.keys(), dtype=np.int64, count=len(nd))
    usigs, inv, cnt = np.unique(np.array(list(nd.values()), dtype="<U16"), return_inverse=True,
                                return_counts=True)
    fb = fallback_value(cnt.astype(np.float64))
    per[day] = hit = (ids, inv.astype(np.int64), usigs.tolist(), fb)
    if len(per) > 12:                       # bounded: the oldest day goes first
        per.pop(next(iter(per)))
    return hit


def _sig_values(usigs, fb, recs, field):
    v = fb.copy()
    for i, s in enumerate(usigs):
        r = recs.get(s)
        if r is not None:
            x = value(r, field)
            if not math.isfinite(x) or not 0.0 <= x <= 1.0:
                raise ValueError(f"engine value {x!r} for {s} outside [0, 1]")
            v[i] = x
    return v


def day_values(host, day, recs, field="composite"):
    """(node ids on `day`, their node-day scores): head value or fallback."""
    ids, inv, usigs, fb = day_arrays(host, day)
    return ids, _sig_values(usigs, fb, recs, field)[inv]


def node_scores(host, by_day, field="composite", days=None):
    """Max over the test days of each universe node's node-day score. Every test day must be in
    `by_day` (a day without a score file would silently fall back to rarity)."""
    days = list(host.test) if days is None else list(days)
    missing = [d for d in days if d not in by_day]
    if missing:
        raise KeyError(f"no scores for test days {missing}")
    s = np.full(len(host.universe), -np.inf)
    for d in days:
        ids, v = day_values(host, d, by_day[d], field)
        pos = upos(host.universe, ids)
        ok = pos >= 0
        np.maximum.at(s, pos[ok], v[ok])
    if not np.isfinite(s).all():
        raise ValueError(f"{int((~np.isfinite(s)).sum())} universe nodes got no score from days {days}")
    return s


def head_mask(host, by_day, days=None):
    """Universe nodes whose signature was in the scored head on at least one test day."""
    days = list(host.test) if days is None else list(days)
    m = np.zeros(len(host.universe), dtype=bool)
    for d in days:
        ids, inv, usigs, _fb = day_arrays(host, d)
        recs = by_day.get(d, {})
        inh = np.fromiter((s in recs for s in usigs), dtype=bool, count=len(usigs))[inv]
        pos = upos(host.universe, ids[inh])
        m[pos[pos >= 0]] = True
    return m
