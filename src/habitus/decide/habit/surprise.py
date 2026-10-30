"""Habit surprise and the cross-fitted per-family null transform.

    from habitus.decide.habit import surprise as SU
    x = SU.node_x(obs, views, own, answers)          # {Family: x_f} for the node's applicable families
    null = SU.FamilyNull.fit(units)                  # units: NullUnit per null node-day and family
    h = SU.habit(obs, x, null, attested, exclude_day=day)   # z_f, E, E_mono (s_hat: null2.EMap)

Raw statistic (per node v, per family f; answers read at T = 1)
  inclusion (P2-P5, F1, F2, N1)  x_f(v) = max over v's present buckets e of -ln p_bar(yes | view, e);
                                 0 when v's present set is empty (at most MAX_BUCKETS buckets)
  choice (P1; N2 per owner)      PIT pi = sum of p_bar(o) over the options with p_bar(o) <= p_bar(observed),
                                 x = -ln pi; N2: the max over the node's owner buckets
A family whose view was refused (text.fits) is n/a and omitted; a present bucket of an available
view without an answer raises KeyError (an engine / pipeline bug, never a silent zero). Observed sets
are per node, so x never depends on n_s or on which members a signature's representatives are.

Null transform (FamilyNull; one per host, level and k plan)
  strata     s = (node type, family, identity-attested flag). For each (node type, family) the
             attested split is MERGED (flag None) when either side holds < NULL_MIN_STRATUM null
             node-days. The flag is attest.tier0 (link-free), never today's |S_f|, so padding cannot
             move a unit into a more lenient stratum.
  tail p     p_f(v) = (1 + #{u in Null_s: x_u >= x_v}) / (1 + |Null_s|) up to the TAIL_Q (95th) null
             percentile u (order statistic x_(ceil(.95 N))). Beyond u, when |Null_s| >= TAIL_FIT_MIN,
             an exponential (GPD shape 0) fitted by maximum likelihood to the excesses of the null
             values strictly above u:  p = p_u * exp(-(x - u) / beta),  beta = mean(x_u' - u),
             p_u = (1 + #{x_u' > u}) / (1 + N)  (the empirical value just above u, so p is continuous
             from the right and non-increasing in x). No tail fit when no value exceeds u.
  floor      p >= 1 / (1 + N_min), N_min = the smallest ADMITTED stratum of the node type (admitted =
             a non-empty null stratum; a family without null units is not admitted: p = z = None).
             The floor gives every family of a node type the same maximum z, so E = max_f z_f is not
             biased toward families with large strata.
  cross-fit  for a unit on null day c its stratum's SAMPLE is every OTHER null day (exclude_day =
             c); test days use the whole null (exclude_day None). The merge decisions (routes) and
             N_min are properties of the WHOLE null, shared by null and test units: recomputing them
             per left-out day gave null units other strata and another z ceiling ln(1 + N_min) than
             test units, so the null composite pool was not exchangeable with a clean test day
             (review finding: a merge flip under cross-fitting could make s_hat unable to reach
             tau_20; a smaller leave-one-out N_min put clean test units above every null unit).
  z_f = -ln p_f;  E = max over admitted families;  E_mono = the same over MONOTONE families (P1 out).

Label-free: every input is an answer to a code-built question, a code-observed bucket set or a null
node-day of an attack-free pre-deployment day; no score here ever sees a test-day rank or a label.
Deterministic: units are sorted on fit, strata are iterated in sorted order, JSON is sorted.
"""

import functools
import math
from collections import namedtuple

import numpy as np

from habitus.questions.habit.constants import MAX_BUCKETS, NULL_MIN_STRATUM, TAIL_FIT_MIN, TAIL_Q
from habitus.questions.habit.types import (FAMILIES, FAMILIES_OF, FAMILY_SPEC, MONOTONE, N2_OPTIONS, NODE_TYPES,
                                           P1_OPTIONS, Family, FamilyScore, NodeHabit, NullUnit, ViewKind, qkey)

P_EPS = 1e-300                  # probabilities are clamped here before -ln (x stays finite)
_EMPTY = np.zeros(0, dtype=np.float64)


def fitmethod(f):
    """Decorator for `fit`: `Cls.fit(data, **kw)` fits a default instance, `obj.fit(data)` fits
    `obj` in place (keeping its constructor parameters); both return the fitted instance."""
    class _Fit:
        def __get__(self, obj, cls):
            if obj is None:
                return lambda *a, **kw: f(cls(), *a, **kw)
            return functools.partial(f, obj)
    return _Fit()


# --------------------------------------------------------------------------- #
# Raw statistic x_f
# --------------------------------------------------------------------------- #
def _nll(p):
    p = float(p)
    if not (math.isfinite(p) and -1e-9 <= p <= 1.0 + 1e-9):
        raise ValueError(f"probability {p!r} outside [0, 1]")
    return max(0.0, -math.log(max(p, P_EPS)))


def x_inclusion(present, p_yes):
    """x_f = max over the present bucket keys of -ln p_yes[bucket]; 0.0 when nothing is present.
    `p_yes`: {bucket key: p_bar(yes)}; a present bucket without a probability raises KeyError."""
    present = sorted(set(present))
    if len(present) > MAX_BUCKETS:
        raise ValueError(f"{len(present)} present buckets > MAX_BUCKETS {MAX_BUCKETS}")
    return max((_nll(p_yes[b]) for b in present), default=0.0)


def x_choice(probs, observed_index):
    """x = -ln PIT, PIT = sum of the option probabilities <= the observed option's (ties included,
    so the most probable option gives PIT = 1 and x = 0)."""
    p = [float(v) for v in probs]
    i = int(observed_index)
    if not 0 <= i < len(p):
        raise IndexError(f"observed option {i} of {len(p)}")
    po = p[i]
    return _nll(min(1.0, sum(v for v in p if v <= po)))


def node_x(obs, views, own, answers, refused=()):
    """{Family: x_f} for the node's applicable families (types.NodeObs `obs`). `views` = {ViewKind:
    View} of the node's signature, `own` = {owner bucket key: V_own View}, `answers` = {qkey: Answer}.
    A family whose view is missing (refused by text.fits) is n/a and omitted; N2 is the max over
    the owner buckets whose view exists (n/a when none does; 0.0 when the node has no owner).
    `refused`: qkeys the answer source declined (a laya model over its length budget: DecisionModel.refused);
    an inclusion family with a refused present bucket and a refused P1 question are n/a, a refused N2
    owner is skipped like a missing V_own view. Any other missing answer still raises KeyError."""
    refused = frozenset(refused)
    out = {}
    for f in FAMILIES_OF[obs.ntype]:
        if not obs.applicable(f):
            continue
        sp = FAMILY_SPEC[f]
        if sp.kind == "inclusion":
            view = views.get(sp.view)
            if view is None:
                continue
            pres = obs.present[f]
            if any(qkey(view.sha1, f, b) in refused for b in pres):
                continue
            out[f] = x_inclusion(pres, {b: answers[qkey(view.sha1, f, b)].p_yes for b in pres})
        elif f is Family.P1_parent:
            view = views.get(ViewKind.V_par)
            if view is None or f not in obs.choice or qkey(view.sha1, f) in refused:
                continue
            ans = answers[qkey(view.sha1, f)]
            out[f] = x_choice(ans.probs, P1_OPTIONS.index(obs.choice[f]))
        elif f is Family.N2_address:
            if not obs.owners:
                out[f] = 0.0
                continue
            xs = []
            for o in obs.owners:
                view = own.get(o.bucket)
                if view is None or qkey(view.sha1, f) in refused:
                    continue
                ans = answers[qkey(view.sha1, f)]
                xs.append(x_choice(ans.probs, N2_OPTIONS.index(o.observed)))
            if xs:
                out[f] = max(xs)
    return out


def null_units(day, ntype, attested, x):
    """NullUnits of one null node-day from its {Family: x_f} (n/a families are simply absent)."""
    return [NullUnit(str(day), ntype, Family.of(f), bool(attested), float(v))
            for f, v in sorted(x.items(), key=lambda kv: Family.of(kv[0]).value) if v is not None]


# --------------------------------------------------------------------------- #
# Null transform
# --------------------------------------------------------------------------- #
_Stratum = namedtuple("_Stratum", "x n u beta p_u")        # u / beta / p_u None without a tail fit
_Fitted = namedtuple("_Fitted", "strata route nmin")        # stratum key -> _Stratum; unit key -> stratum key


def _ukey(ntype, family, attested):
    return ntype, Family.of(family), bool(attested)


class FamilyNull:
    """Per-family empirical null of one host, level and k plan, cross-fitted by day (module doc)."""

    def __init__(self, min_stratum=NULL_MIN_STRATUM, tail_min=TAIL_FIT_MIN, tail_q=TAIL_Q):
        self.min_stratum, self.tail_min, self.tail_q = int(min_stratum), int(tail_min), float(tail_q)
        if not 0.0 < self.tail_q < 1.0:
            raise ValueError(f"tail_q {tail_q} outside (0, 1)")
        self._units = {}            # (ntype, Family, attested) -> {day: sorted float64 x}
        self._fits = {}             # exclude_day (None = whole null) -> _Fitted
        self.days = ()

    @fitmethod
    def fit(self, units):
        acc = {}
        for u in units:
            u = NullUnit(*u)
            f = Family.of(u.family)
            if u.ntype not in NODE_TYPES or FAMILY_SPEC[f].node_type != u.ntype:
                raise ValueError(f"null unit of family {f.value} on a {u.ntype!r} node")
            x = float(u.x)
            if math.isnan(x) or x < 0:
                raise ValueError(f"null unit x {u.x!r} is not a non-negative number")
            acc.setdefault((u.ntype, f, bool(u.attested)), {}).setdefault(str(u.day), []).append(x)
        self._units = {k: {d: np.sort(np.asarray(v, dtype=np.float64)) for d, v in sorted(acc[k].items())}
                       for k in sorted(acc, key=lambda k: (k[0], k[1].value, k[2]))}
        self.days = tuple(sorted({d for by_day in acc.values() for d in by_day}))
        self._fits = {}
        return self

    # --- fitting one cross-fitted null --------------------------------------------------- #
    def _stratum(self, x):
        n = len(x)
        if n < self.tail_min:
            return _Stratum(x, n, None, None, None)
        u = float(x[max(0, math.ceil(self.tail_q * n - 1e-9) - 1)])
        above = x[np.searchsorted(x, u, side="right"):]
        beta = float(np.mean(above - u)) if len(above) else 0.0
        if not beta > 0.0:
            return _Stratum(x, n, None, None, None)
        return _Stratum(x, n, u, beta, (1.0 + len(above)) / (1.0 + n))

    def _fitted(self, exclude_day=None):
        key = exclude_day if exclude_day in self.days else None
        hit = self._fits.get(key)
        if hit is not None:
            return hit
        if key is not None:                     # cross-fitted: the whole null's routes and N_min,
            full = self._fitted(None)           # the stratum samples without day `key`
            strata = {}
            for sk in full.strata:
                t, fv, a = sk
                parts = [v for flag in ((True, False) if a is None else (a,))
                         for d, v in self._units.get((t, Family(fv), flag), {}).items() if d != key]
                strata[sk] = self._stratum(np.sort(np.concatenate(parts)) if parts else _EMPTY)
            fit = self._fits[key] = _Fitted(strata, full.route, full.nmin)
            return fit
        sides = {}
        for (t, f, a), by_day in self._units.items():
            parts = list(by_day.values())
            sides.setdefault((t, f), {})[a] = np.sort(np.concatenate(parts)) if parts else _EMPTY
        strata, route = {}, {}
        for (t, f), s in sorted(sides.items(), key=lambda kv: (kv[0][0], kv[0][1].value)):
            yes, no = s.get(True, _EMPTY), s.get(False, _EMPTY)
            if min(len(yes), len(no)) < self.min_stratum:
                sk = (t, f.value, None)
                strata[sk] = self._stratum(np.sort(np.concatenate([yes, no])))
                route[(t, f, True)] = route[(t, f, False)] = sk
            else:
                for a, v in ((False, no), (True, yes)):
                    sk = (t, f.value, a)
                    strata[sk] = self._stratum(v)
                    route[(t, f, a)] = sk
        nmin = {}
        for sk, st in strata.items():
            if st.n > 0:
                nmin[sk[0]] = min(nmin.get(sk[0], st.n), st.n)
        fit = _Fitted(strata, route, nmin)
        self._fits[key] = fit
        return fit

    # --- queries ------------------------------------------------------------------------- #
    def stratum(self, ntype, family, attested, exclude_day=None):
        """(node type, family value, attested flag or None = merged), or None when the family has
        no null unit at all on this host."""
        return self._fitted(exclude_day).route.get(_ukey(ntype, family, attested))

    def n(self, ntype, family, attested, exclude_day=None):
        """|Null_s| of the unit's stratum (0 when not admitted)."""
        fit = self._fitted(exclude_day)
        sk = fit.route.get(_ukey(ntype, family, attested))
        return fit.strata[sk].n if sk is not None else 0

    def floor(self, ntype, exclude_day=None):
        """1 / (1 + N_min) of the node type, None when no family of it is admitted."""
        nm = self._fitted(exclude_day).nmin.get(ntype)
        return None if nm is None else 1.0 / (1.0 + nm)

    def p(self, ntype, family, attested, x, exclude_day=None):
        """Null upper-tail p of a raw value x (module doc); None for x None or a family that is not
        admitted. Non-increasing in x, in [1 / (1 + N_min), 1]."""
        if x is None:
            return None
        x = float(x)
        if math.isnan(x):
            raise ValueError("x is NaN")
        fit = self._fitted(exclude_day)
        sk = fit.route.get(_ukey(ntype, family, attested))
        st = fit.strata.get(sk) if sk is not None else None
        if st is None or st.n == 0:
            return None
        if st.u is not None and x > st.u:
            p = st.p_u * math.exp(-(x - st.u) / st.beta)
        else:
            p = (1.0 + st.n - int(np.searchsorted(st.x, x, side="left"))) / (1.0 + st.n)
        return min(1.0, max(p, 1.0 / (1.0 + fit.nmin[ntype])))

    def z(self, ntype, family, attested, x, exclude_day=None):
        """-ln p (None when p is None)."""
        p = self.p(ntype, family, attested, x, exclude_day)
        return None if p is None else max(0.0, -math.log(p))

    def report(self):
        """Strata of the whole null (n, per-day n, merged, tail fit), N_min and floors per node type,
        and the parameters (published with every HB level's null)."""
        fit = self._fitted(None)
        rows = []
        for sk in sorted(fit.strata, key=lambda k: (k[0], k[1], -1 if k[2] is None else int(k[2]))):
            st = fit.strata[sk]
            t, fv, a = sk
            flags = (True, False) if a is None else (a,)
            per_day = {}
            for flag in flags:
                for d, v in self._units.get((t, Family(fv), flag), {}).items():
                    per_day[d] = per_day.get(d, 0) + len(v)
            rows.append({"ntype": t, "family": fv, "attested": a, "merged": a is None, "n": st.n,
                         "n_by_day": dict(sorted(per_day.items())), "tail": st.u is not None,
                         "u": st.u, "beta": st.beta, "p_u": st.p_u})
        return {"params": self.params(), "days": list(self.days), "strata": rows,
                "n_min": dict(sorted(fit.nmin.items())),
                "floor": {t: 1.0 / (1.0 + n) for t, n in sorted(fit.nmin.items())},
                "z_max": {t: math.log(1.0 + n) for t, n in sorted(fit.nmin.items())}}

    def params(self):
        return {"min_stratum": self.min_stratum, "tail_min": self.tail_min, "tail_q": self.tail_q}

    # --- JSON ---------------------------------------------------------------------------- #
    def to_json(self):
        units = [[t, f.value, a, {d: v.tolist() for d, v in by_day.items()}]
                 for (t, f, a), by_day in self._units.items()]
        return {"kind": "FamilyNull", "params": self.params(), "days": list(self.days), "units": units}

    @classmethod
    def from_json(cls, d):
        if d.get("kind") != "FamilyNull":
            raise ValueError(f"not a FamilyNull record: kind {d.get('kind')!r}")
        obj = cls(**d["params"])
        return obj.fit(NullUnit(day, t, Family(fv), bool(a), x)
                       for t, fv, a, by_day in d["units"] for day, xs in by_day.items() for x in xs)


# --------------------------------------------------------------------------- #
# Per-node habit statistics
# --------------------------------------------------------------------------- #
def combine(z):
    """(E, E_mono) from {Family: z_f} (None values skipped): max over all, max over MONOTONE
    families; 0.0 when empty."""
    vals = [(Family.of(f), float(v)) for f, v in z.items() if v is not None]
    e = max((v for _f, v in vals), default=0.0)
    em = max((v for f, v in vals if f in MONOTONE), default=0.0)
    return e, em


def habit(obs, x, null, attested, exclude_day=None):
    """types.NodeHabit of one node-day: per applicable family x / p / z / stratum against `null`
    (a FamilyNull; exclude_day = the unit's own day on a null day), E and E_mono over the admitted
    families. s_hat is None here (null2.EMap sets it)."""
    fams = {}
    for f in FAMILIES:
        if f not in x or x[f] is None:
            continue
        p = null.p(obs.ntype, f, attested, x[f], exclude_day)
        fams[f] = FamilyScore(f, float(x[f]), p, None if p is None else max(0.0, -math.log(p)),
                              null.stratum(obs.ntype, f, attested, exclude_day))
    e, em = combine({f: s.z for f, s in fams.items()})
    return NodeHabit(int(obs.node), obs.ntype, bool(attested), fams, e, em, None)


__all__ = ["P_EPS", "fitmethod", "x_inclusion", "x_choice", "node_x", "null_units", "FamilyNull", "combine",
           "habit"]
