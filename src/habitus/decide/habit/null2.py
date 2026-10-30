"""Habit-channel nulls: k plans, the cross-fitted s_hat map, bars and their persistence.

    from habitus.decide.habit import null2
    null2.plan(host, "k2")                 # [Ref] of the null days (role "null") of a k plan
    null2.ref_for(host, test_day, "k2")    # the test day's Ref under the same plan
    m = null2.EMap.fit(rows)               # rows: (null day, ntype, E, base0 = max(qv, qt))
    m.s_hat("process", E, exclude_day=d)   # Q_base(F_E(E)), cross-fitted
    null2.bars(values, n_days)             # {0: tau_0, 1: tau_1, 20: tau_20} (decide.null.bar)

k plans (score directories, constants.KPLANS). Every reference and null day is a pre-deployment day
(types.predeploy: usable clean days strictly before the host's first test day; asserted).

  plan        test-day reference                       null days (each with its own reference)
  k0          none                                     the host's own pre-deployment days
  k2          shots(2) (the first 2 usable clean days) decide.null.lodo_plan(host, 2): each shot day
                                                       with the OTHER shot day (1-day reference)
  allc-pre    the whole pre-deployment window          LODO inside the window

The k2 null scores each shot day against a 1-day reference while test days use 2: the disclosed k2
null mismatch.

s_hat (EMap). Per node type t, F_E,t = the null distribution of E and Q_base,t = the quantile
function of base0 = max(qv, qt) over the SAME null node-days, both cross-fitted (a unit on null day
c uses every other null day). With the null sorted ascending (E_(1..N), b_(1..N)) and
m = #{null E < E} (strict, so the many ties at E = 0 map to the LOWEST base0, consistent with the
upper-tail p of surprise.FamilyNull):
    s_hat(E) = b_(m+1)                                           for E <= E_max
    s_hat(E) = b_max + (1 - b_max) (1 - exp(-(E - E_max)))       for E > E_max   (fixed a priori)
Non-decreasing in E, continuous at E_max when E_max is unique, in [b_min, 1). The map is fitted on
E itself (the max over families), never assumed from per-family calibration. None for a node type
without null units.

Label-free: plans read the frozen day lists; maps and bars read null-day score values only. Bars use
decide.null.bar (the label-free copy), never habitus.eval.
"""

import gzip
import io
import json
import math
import os

import numpy as np

from habitus.decide.habit.surprise import FamilyNull, fitmethod
from habitus.decide.null import bar, lodo_plan
from habitus.questions.habit.constants import BARS, H501_SELECTION_DAYS, KPLANS
from habitus.questions.habit.types import NODE_TYPES, NodeHabit, Ref, assert_predeploy, predeploy

H501 = "optc_h501"
KMODE_OF = {"k0": "k0", "k2": "k2", "allc-pre": "allc-pre"}


def _check_kplan(kplan):
    if kplan not in KPLANS:
        raise ValueError(f"k plan {kplan!r} not in {KPLANS}")


def default_root(settings=None):
    """The habit channel's output root: <settings.paths.runs>/habit."""
    if settings is None:
        from habitus import config
        s = config.settings
        settings = s() if callable(s) and not hasattr(s, "paths") else s
    return os.path.join(os.fspath(settings.paths.runs), "habit")


# --------------------------------------------------------------------------- #
# k plans
# --------------------------------------------------------------------------- #
def shots2(host):
    """shots(2) = the first two usable clean days, asserted pre-deployment."""
    s = list(host.shots(2))
    if len(s) != 2:
        raise ValueError(f"{host.ds}: k = 2 needs 2 usable clean days, have {s}")
    assert_predeploy(host, s)
    return s


def selection_days(host):
    """Days the arm-selection metrics score: optc_h501's parity half 09-19 / 09-21, else the host's
    whole pre-deployment window."""
    return list(H501_SELECTION_DAYS) if host.ds == H501 else predeploy(host)


def ref_for(host, day, kplan):
    """The Ref of a TEST day of `host` under `kplan` (a null day of the plan returns its plan Ref)."""
    _check_kplan(kplan)
    if day not in host.test:
        hit = [r for r in plan(host, kplan) if r.ds == host.ds and r.day == day]
        if hit:
            return hit[0]
        raise ValueError(f"{host.ds}: {day} is neither a test day nor a {kplan} null day")
    pre = predeploy(host)
    if kplan == "k0":
        return Ref(host.ds, day, (), "k0")
    if kplan == "k2":
        ref = shots2(host)
    else:                                               # allc-pre
        ref = pre
        if not ref:
            raise ValueError(f"{host.ds}: empty pre-deployment window")
    assert_predeploy(host, ref)
    return Ref(host.ds, day, ref, KMODE_OF[kplan])


def _lodo(host, days, k_mode):
    if len(days) < 2:
        raise ValueError(f"{host.ds}: a LODO null needs >= 2 days, have {days}")
    return [Ref(host.ds, c, [d for d in days if d != c], k_mode) for c in days]


def plan(host, kplan):
    """[Ref] of the null days of `kplan` (module docstring; role "null")."""
    _check_kplan(kplan)
    pre = predeploy(host)
    if kplan == "k2":
        refs = [Ref(host.ds, c, ref, "k2") for c, ref in lodo_plan(host, 2)]
    elif kplan == "allc-pre":
        refs = _lodo(host, pre, "allc-pre")
    else:
        if not pre:
            raise ValueError(f"{host.ds}: empty pre-deployment window")
        return [Ref(host.ds, c, (), "k0") for c in pre]
    assert_predeploy(host, sorted({d for r in refs for d in (r.day,) + r.ref_days}))
    return refs


def null_days(host, kplan):
    return [r.day for r in plan(host, kplan)]


def plan_refs(host, kplan):
    """[(types.Ref, role)] of one k plan: plan's null days (role "null"), then ref_for of every
    test day (role "test")."""
    out = [(r, "null") for r in plan(host, kplan)]
    out += [(ref_for(host, d, kplan), "test") for d in host.test]
    return out


# --------------------------------------------------------------------------- #
# s_hat map
# --------------------------------------------------------------------------- #
class EMap:
    """Cross-fitted monotone map of E onto the base0 scale, per node type (module doc)."""

    def __init__(self):
        self._rows = {}             # ntype -> {day: (sorted E, sorted base0)}
        self._fits = {}             # (ntype, exclude_day or None) -> (sorted E, sorted base0)
        self.days = ()

    @fitmethod
    def fit(self, rows):
        acc = {}
        for day, ntype, e, b in rows:
            if ntype not in NODE_TYPES:
                raise ValueError(f"node type {ntype!r}")
            e, b = float(e), float(b)
            if not (math.isfinite(e) and math.isfinite(b)) or e < 0:
                raise ValueError(f"EMap row E {e!r} / base0 {b!r}")
            if not -1e-9 <= b <= 1.0 + 1e-9:
                raise ValueError(f"base0 {b!r} outside [0, 1]")
            es, bs = acc.setdefault(ntype, {}).setdefault(str(day), ([], []))
            es.append(e)
            bs.append(min(1.0, max(0.0, b)))
        self._rows = {t: {d: (np.sort(np.asarray(es)), np.sort(np.asarray(bs)))
                          for d, (es, bs) in sorted(acc[t].items())} for t in sorted(acc)}
        self.days = tuple(sorted({d for by_day in acc.values() for d in by_day}))
        self._fits = {}
        return self

    def _sorted(self, ntype, exclude_day=None):
        key = (ntype, exclude_day if exclude_day in self.days else None)
        hit = self._fits.get(key)
        if hit is None:
            parts = [v for d, v in self._rows.get(ntype, {}).items() if d != key[1]]
            es = np.sort(np.concatenate([p[0] for p in parts])) if parts else np.zeros(0)
            bs = np.sort(np.concatenate([p[1] for p in parts])) if parts else np.zeros(0)
            hit = self._fits[key] = (es, bs)
        return hit

    def n(self, ntype, exclude_day=None):
        return len(self._sorted(ntype, exclude_day)[0])

    def cdf(self, ntype, E, exclude_day=None):
        """F_E,t(E) = #{null E < E} / N (strict; None without null units)."""
        es, _bs = self._sorted(ntype, exclude_day)
        if not len(es):
            return None
        return int(np.searchsorted(es, float(E), side="left")) / len(es)

    def quantile(self, ntype, q, exclude_day=None):
        """Q_base,t(q) = b_(floor(q N) + 1), capped at b_max (None without null units)."""
        _es, bs = self._sorted(ntype, exclude_day)
        if not len(bs):
            return None
        return float(bs[min(len(bs) - 1, max(0, int(math.floor(float(q) * len(bs) + 1e-9))))])

    def s_hat(self, ntype, E, exclude_day=None):
        """Q_base(F_E(E)) below the null maximum, the fixed exponential approach to 1 above it."""
        es, bs = self._sorted(ntype, exclude_day)
        if not len(es):
            return None
        E = float(E)
        if math.isnan(E):
            raise ValueError("E is NaN")
        e_max, b_max = float(es[-1]), float(bs[-1])
        if E > e_max:
            return b_max + (1.0 - b_max) * (1.0 - math.exp(-(E - e_max)))
        return float(bs[int(np.searchsorted(es, E, side="left"))])

    def report(self):
        out = {}
        for t in sorted(self._rows):
            es, bs = self._sorted(t)
            out[t] = {"n": len(es), "n_by_day": {d: len(v[0]) for d, v in self._rows[t].items()},
                      "E_max": float(es[-1]) if len(es) else None, "b_max": float(bs[-1]) if len(bs) else None,
                      "E_q": {q: float(np.quantile(es, q)) for q in (0.5, 0.9, 0.99)} if len(es) else {},
                      "share_E0": float(np.mean(es <= 0.0)) if len(es) else None}
        return {"days": list(self.days), "ntypes": out}

    def to_json(self):
        return {"kind": "EMap", "rows": {t: {d: {"E": es.tolist(), "base0": bs.tolist()}
                                             for d, (es, bs) in by_day.items()} for t, by_day in self._rows.items()}}

    @classmethod
    def from_json(cls, d):
        if d.get("kind") != "EMap":
            raise ValueError(f"not an EMap record: kind {d.get('kind')!r}")
        return cls().fit((day, t, e, b) for t, by_day in d["rows"].items() for day, v in by_day.items()
                         for e, b in zip(v["E"], v["base0"]))


def with_s_hat(h, emap, exclude_day=None):
    """A copy of NodeHabit `h` with s_hat = emap.s_hat(h.ntype, h.E, exclude_day)."""
    return NodeHabit(h.node, h.ntype, h.attested, h.fams, h.E, h.E_mono, emap.s_hat(h.ntype, h.E, exclude_day))


# --------------------------------------------------------------------------- #
# Bars
# --------------------------------------------------------------------------- #
def bars(values, n_days, budgets=BARS):
    """{B: tau_B} = decide.null.bar (the label-free bar): smallest t with
    #{null units >= t} / n_days <= B; -inf when B covers the whole null, None without a null."""
    v = np.asarray(values, dtype=np.float64)
    return {int(b): bar(v, int(n_days), b) for b in budgets}


# --------------------------------------------------------------------------- #
# Persistence under <root>/nulls/<ds>/
# --------------------------------------------------------------------------- #
def _name(name):
    name = str(name)
    return name if name.endswith(".json.gz") else name + ".json.gz"


def null_path(ds, name, root=None):
    """<root>/nulls/<ds>/<name>.json.gz (root None = default_root()); no directory is created."""
    return os.path.join(default_root() if root is None else os.fspath(root), "nulls", ds, _name(name))


def save(obj, ds, name, root=None):
    """Write a FamilyNull / EMap / plain dict as gzip JSON under <root>/nulls/<ds>/<name>.json.gz
    (atomic, byte-deterministic: sorted keys, gzip mtime 0). Returns the path."""
    from habitus.decide.scorefile import check_writable
    payload = obj.to_json() if hasattr(obj, "to_json") else obj
    path = check_writable(null_path(ds, name, root))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as g:
        g.write(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(buf.getvalue())
    os.replace(tmp, path)
    return path


def load(ds, name, root=None):
    """Read what save() wrote: a FamilyNull / EMap when the record says so, else the dict."""
    with gzip.open(null_path(ds, name, root), "rt", encoding="utf-8") as f:
        d = json.load(f)
    kind = d.get("kind") if isinstance(d, dict) else None
    if kind == "FamilyNull":
        return FamilyNull.from_json(d)
    if kind == "EMap":
        return EMap.from_json(d)
    return d


__all__ = ["KMODE_OF", "default_root", "shots2", "selection_days", "ref_for", "plan", "null_days", "plan_refs",
           "EMap", "with_s_hat", "bars", "null_path", "save", "load"]
