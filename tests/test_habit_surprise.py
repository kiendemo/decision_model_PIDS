"""Habit surprise and nulls (CPU, synthetic arrays and the shared fake host): surprise (x_f, the
cross-fitted per-family null with stratum merging / exponential tail / floor, E / E_mono), null2 (k
plans, the h501 parity split, the cross-fitted s_hat map, bars, persistence). Property tests:
monotone in x, permutation invariance, stratum merging, cross-fitting = refit without the day."""

import gzip
import json
import math
import os
import random
import re
import subprocess
import sys

import numpy as np
import pytest

from conftest import REPO
from habit_fakes import CLEAN, TEST, FakeHabitHost, make_habit_host

from habitus.decide.habit import null2 as N2
from habitus.decide.habit import surprise as SU
from habitus.decide.null import bar as label_free_bar
from habitus.questions.habit import constants as H
from habitus.questions.habit import types as T

HYGIENE = re.compile(r"gt\(|ground_truth|host_labels|Ground_Truth|gt_")     # the label-token pattern
F = T.Family


@pytest.fixture
def hb_root(tmp_path):
    return str(tmp_path / "hb")


@pytest.fixture(scope="module")
def host():
    return make_habit_host()


def _view(kind, ntype, tag, **meta):
    """A View whose bytes depend on `tag` (distinct qkeys), meta filled with defaults."""
    m = {k: "" for k in T.VIEW_META_KEYS}
    m.update({"ident": f"id-{tag}", "tier0": "IDENTITY", "k_mode": "k2"})
    m.update(meta)
    sections = {"entity": {"type": ntype, "tag": str(tag)}}
    return T.View(kind, ntype, sections, (), m)


def _ans(p_yes=None, probs=None):
    probs = (p_yes, 1.0 - p_yes) if probs is None else tuple(probs)
    return T.Answer(probs, {"fwd": probs, "rev": probs}, 1.0)


# --------------------------------------------------------------------------- #
# 1. raw statistic x_f
# --------------------------------------------------------------------------- #
def test_x_inclusion_and_choice():
    assert SU.x_inclusion([], {}) == 0.0
    assert SU.x_inclusion(["a", "b"], {"a": 0.5, "b": 0.01, "c": 1e-9}) == pytest.approx(-math.log(0.01))
    with pytest.raises(KeyError):
        SU.x_inclusion(["a", "z"], {"a": 0.5})
    with pytest.raises(ValueError):                                 # at most MAX_BUCKETS present buckets
        SU.x_inclusion([f"b{i}" for i in range(H.MAX_BUCKETS + 1)], {f"b{i}": 0.5 for i in range(9)})
    with pytest.raises(ValueError):
        SU.x_inclusion(["a"], {"a": 1.5})
    assert math.isfinite(SU.x_inclusion(["a"], {"a": 0.0}))          # clamped, never inf
    ps = sorted(random.Random(0).random() for _ in range(20))
    xs = [SU.x_inclusion(["a"], {"a": p}) for p in ps]
    assert xs == sorted(xs, reverse=True)                           # lower p_yes -> larger x
    p = (0.5, 0.3, 0.15, 0.05)
    assert SU.x_choice(p, 0) == 0.0                                 # the most probable option: PIT = 1
    assert SU.x_choice(p, 3) == pytest.approx(-math.log(0.05))
    assert SU.x_choice(p, 2) == pytest.approx(-math.log(0.20))
    assert SU.x_choice((0.4, 0.2, 0.2, 0.2), 1) == pytest.approx(-math.log(0.6))      # ties included
    with pytest.raises(IndexError):
        SU.x_choice(p, 4)


def test_node_x_refused_questions_are_na_and_nothing_else():
    """A question the source declined (a laya head over its budget: DecisionModel.refused) makes its inclusion family
    / P1 n/a and drops its N2 owner, like a refused view; any OTHER missing answer still raises."""
    vb, vp = _view(T.ViewKind.V_beh, "process", "b"), _view(T.ViewKind.V_par, "process", "p")
    obs = T.NodeObs(1, "s", "process", "k", present={"P2": {"dir:home", "dir:staging"}, "P3": set(),
                                                    "P4": {"gen:never-seen"}, "P5": set()},
                    choice={"P1": "never-seen"})
    ans = {T.qkey(vb.sha1, "P2", "dir:home"): _ans(0.9), T.qkey(vb.sha1, "P4", "gen:never-seen"): _ans(0.001)}
    views = {T.ViewKind.V_beh: vb, T.ViewKind.V_par: vp}
    refused = {T.qkey(vb.sha1, "P2", "dir:staging"), T.qkey(vp.sha1, "P1")}
    x = SU.node_x(obs, views, {}, ans, refused=refused)
    assert F.P2_modify not in x and F.P1_parent not in x                       # one refused bucket: family n/a
    assert x[F.P4_children] == pytest.approx(-math.log(0.001)) and x[F.P3_peers] == 0.0
    with pytest.raises(KeyError):                                            # not refused: still a bug
        SU.node_x(obs, views, {}, ans, refused={T.qkey(vp.sha1, "P1")})
    vn = _view(T.ViewKind.V_net, "netflow", "n")
    vo1, vo2 = _view(T.ViewKind.V_own, "netflow", "o1"), _view(T.ViewKind.V_own, "netflow", "o2")
    nf = T.NodeObs(2, "s", "netflow", "k", present={"N1": set()},
                   owners=(T.OwnerObs("id:o1", "o1", "same-address"), T.OwnerObs("gen:typed", "o9", "no-program")))
    na = {T.qkey(vo1.sha1, "N2"): _ans(probs=(0.85, 0.1, 0.04, 0.01))}
    xn = SU.node_x(nf, {T.ViewKind.V_net: vn}, {"id:o1": vo1, "gen:typed": vo2}, na, refused={T.qkey(vo2.sha1, "N2")})
    assert xn[F.N2_address] == 0.0                                           # the refused owner is skipped


def test_node_x_families_views_and_owners():
    vb, vp = _view(T.ViewKind.V_beh, "process", "b"), _view(T.ViewKind.V_par, "process", "p")
    obs = T.NodeObs(1, "s", "process", "k", present={"P2": {"dir:home", "dir:staging"}, "P3": set(),
                                                    "P4": {"gen:never-seen"}, "P5": set()},
                    choice={"P1": "never-seen"})
    ans = {T.qkey(vb.sha1, "P2", "dir:home"): _ans(0.9), T.qkey(vb.sha1, "P2", "dir:staging"): _ans(0.02),
           T.qkey(vb.sha1, "P4", "gen:never-seen"): _ans(0.001),
           T.qkey(vp.sha1, "P1"): _ans(probs=(0.7, 0.2, 0.08, 0.02))}
    x = SU.node_x(obs, {T.ViewKind.V_beh: vb, T.ViewKind.V_par: vp}, {}, ans)
    assert set(x) == {F.P1_parent, F.P2_modify, F.P3_peers, F.P4_children, F.P5_exec_from}
    assert x[F.P2_modify] == pytest.approx(-math.log(0.02)) and x[F.P3_peers] == 0.0 == x[F.P5_exec_from]
    assert x[F.P1_parent] == pytest.approx(-math.log(0.1))
    x2 = SU.node_x(obs, {T.ViewKind.V_beh: vb}, {}, ans)                # V_par refused -> P1 n/a
    assert F.P1_parent not in x2 and x2[F.P4_children] == pytest.approx(-math.log(0.001))
    del ans[T.qkey(vb.sha1, "P2", "dir:home")]
    with pytest.raises(KeyError):                                   # a present bucket without an answer
        SU.node_x(obs, {T.ViewKind.V_beh: vb}, {}, ans)
    vn = _view(T.ViewKind.V_net, "netflow", "n")
    vo1, vo2 = _view(T.ViewKind.V_own, "netflow", "o1"), _view(T.ViewKind.V_own, "netflow", "o2")
    nf = T.NodeObs(2, "s", "netflow", "k", present={"N1": {"id:o1"}},
                   owners=(T.OwnerObs("id:o1", "o1", "same-address"), T.OwnerObs("gen:typed", "o9", "no-program"),
                           T.OwnerObs("gen:typed", "o8", "same-address")))
    na = {T.qkey(vn.sha1, "N1", "id:o1"): _ans(0.6), T.qkey(vo1.sha1, "N2"): _ans(probs=(0.85, 0.1, 0.04, 0.01)),
          T.qkey(vo2.sha1, "N2"): _ans(probs=(0.25, 0.25, 0.25, 0.25))}
    own = {"id:o1": vo1, "gen:typed": vo2}
    xn = SU.node_x(nf, {T.ViewKind.V_net: vn}, own, na)
    assert xn[F.N2_address] == 0.0                                   # uniform typed view: PIT 1; o1 usual: 0
    na[T.qkey(vo2.sha1, "N2")] = _ans(probs=(0.7, 0.2, 0.07, 0.03))
    xn = SU.node_x(nf, {T.ViewKind.V_net: vn}, own, na)
    assert xn[F.N2_address] == pytest.approx(-math.log(0.03))        # max over owner buckets
    assert SU.node_x(nf, {T.ViewKind.V_net: vn}, {"id:o1": vo1}, na)[F.N2_address] == 0.0   # partial views
    assert F.N2_address not in SU.node_x(nf, {T.ViewKind.V_net: vn}, {}, na)                 # no owner view
    empty = T.NodeObs(3, "s", "netflow", "k", present={"N1": set()})
    assert SU.node_x(empty, {T.ViewKind.V_net: vn}, {}, {}) == {F.N1_owner: 0.0, F.N2_address: 0.0}
    k0 = T.NodeObs(4, "s", "netflow", "k", present={"N1": set()}, na={"N2"})
    assert SU.node_x(k0, {T.ViewKind.V_net: vn}, {}, {}) == {F.N1_owner: 0.0}


# --------------------------------------------------------------------------- #
# 2. FamilyNull: strata, merging, tail, floor, cross-fitting
# --------------------------------------------------------------------------- #
def _units(rng, spec):
    """spec: [(day, ntype, family, attested, n, scale)] -> NullUnits with exponential x (40% zeros)."""
    out = []
    for day, t, f, a, n, scale in spec:
        x = rng.exponential(scale, n) * (rng.random(n) > 0.4)
        out += [T.NullUnit(day, t, F.of(f), a, float(v)) for v in x]
    return out


def test_family_null_small_stratum_is_exact_empirical_with_floor():
    units = [T.NullUnit("d1", "process", F.P2_modify, False, x) for x in (0.0, 0.0, 1.0, 2.0)] + \
            [T.NullUnit("d2", "process", F.P2_modify, True, x) for x in (0.0, 3.0)] + \
            [T.NullUnit("d1", "process", F.P3_peers, False, x) for x in [0.0] * 9 + [5.0]]
    null = SU.FamilyNull.fit(units)
    assert null.stratum("process", "P2", True) == ("process", "P2", None)          # merged (< 500 a side)
    assert null.stratum("process", "P2", False) == ("process", "P2", None)
    assert null.n("process", "P2", True) == 6 and null.n("process", "P3", False) == 10
    assert null.floor("process") == pytest.approx(1 / 7)                            # N_min = 6
    assert null.p("process", "P2", True, 0.0) == 1.0
    assert null.p("process", "P2", False, 1.5) == pytest.approx(3 / 7)              # (1 + #{>= 1.5}) / 7
    assert null.p("process", "P2", False, 99.0) == pytest.approx(1 / 7)
    assert null.p("process", "P3", False, 4.0) == pytest.approx(2 / 11)
    assert null.p("process", "P3", False, 9.0) == pytest.approx(1 / 7)              # floored at 1 / (1 + N_min)
    assert null.z("process", "P2", False, 1.5) == pytest.approx(-math.log(3 / 7))
    assert null.p("process", "P5", False, 1.0) is None                             # not admitted
    assert null.stratum("process", "P5", False) is None
    assert null.p("process", "P2", False, None) is None
    assert null.floor("file") is None
    with pytest.raises(ValueError):
        SU.FamilyNull.fit([T.NullUnit("d1", "file", F.P2_modify, False, 0.0)])     # P2 is a process family
    with pytest.raises(ValueError):
        SU.FamilyNull.fit([T.NullUnit("d1", "process", F.P2_modify, False, float("nan"))])


def test_family_null_stratum_merging_threshold():
    mk = lambda n_yes, n_no: [T.NullUnit("d1", "file", F.F1_writer, True, 1.0)] * n_yes + \
        [T.NullUnit("d1", "file", F.F1_writer, False, 2.0)] * n_no                  # noqa: E731
    split = SU.FamilyNull.fit(mk(500, 600))
    assert split.stratum("file", "F1", True) == ("file", "F1", True)
    assert split.stratum("file", "F1", False) == ("file", "F1", False)
    assert split.n("file", "F1", True) == 500 and split.floor("file") == pytest.approx(1 / 501)
    merged = SU.FamilyNull.fit(mk(499, 5000))
    assert merged.stratum("file", "F1", True) == merged.stratum("file", "F1", False) == ("file", "F1", None)
    assert merged.n("file", "F1", True) == 5499
    one_side = SU.FamilyNull.fit(mk(0, 5000))
    assert one_side.stratum("file", "F1", True) == ("file", "F1", None)
    custom = SU.FamilyNull(min_stratum=10).fit(mk(20, 30))                        # instance fit keeps params
    assert custom.min_stratum == 10 and custom.stratum("file", "F1", True) == ("file", "F1", True)
    rep = merged.report()
    assert rep["strata"][0]["merged"] and rep["n_min"] == {"file": 5499}
    assert rep["params"]["min_stratum"] == H.NULL_MIN_STRATUM


def test_family_null_exponential_tail_beyond_the_95th_percentile():
    rng = np.random.default_rng(1)
    x = rng.exponential(2.0, 4000)
    null = SU.FamilyNull(min_stratum=1).fit(T.NullUnit("d1", "netflow", F.N1_owner, False, float(v)) for v in x)
    st = null._fitted()
    s = st.strata[("netflow", "N1", None)]
    xs = np.sort(x)
    assert s.u == pytest.approx(xs[math.ceil(0.95 * 4000) - 1])
    assert s.beta == pytest.approx(np.mean(xs[xs > s.u] - s.u)) and 1.6 < s.beta < 2.4   # MLE of the scale
    assert s.p_u == pytest.approx((1 + np.sum(xs > s.u)) / 4001)
    # empirical below u
    v = float(xs[1000])
    assert null.p("netflow", "N1", False, v) == pytest.approx((1 + np.sum(xs >= v)) / 4001)
    # exponential above u, continuing past the null maximum (until the floor)
    a, b = s.u + 1.0, s.u + 3.0
    pa, pb = null.p("netflow", "N1", False, a), null.p("netflow", "N1", False, b)
    assert pa == pytest.approx(s.p_u * math.exp(-1.0 / s.beta)) and pb / pa == pytest.approx(math.exp(-2.0 / s.beta))
    assert null.p("netflow", "N1", False, 1e6) == pytest.approx(1 / 4001)             # the floor
    small = SU.FamilyNull(min_stratum=1).fit(T.NullUnit("d1", "netflow", F.N1_owner, False, float(v))
                                             for v in x[:150])
    assert small._fitted().strata[("netflow", "N1", None)].u is None               # < TAIL_FIT_MIN: no tail
    ties = SU.FamilyNull(min_stratum=1).fit([T.NullUnit("d1", "file", F.F2_executor, False, 0.0)] * 300)
    assert ties._fitted().strata[("file", "F2", None)].u is None                   # nothing above u


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_family_null_monotone_in_x_property(seed):
    rng = np.random.default_rng(seed)
    spec = [("d1", "process", "P2", True, 300, 1.0), ("d1", "process", "P2", False, 900, 2.0),
            ("d2", "process", "P2", True, 400, 1.5), ("d2", "process", "P2", False, 700, 0.5),
            ("d1", "process", "P1", False, 150, 1.0), ("d3", "file", "F1", True, 2500, 3.0)]
    null = SU.FamilyNull.fit(_units(rng, spec))
    grid = np.concatenate([[0.0], np.sort(rng.exponential(4.0, 300)), [50.0, 200.0]])
    for t, f, a in (("process", "P2", True), ("process", "P2", False), ("process", "P1", True), ("file", "F1", False)):
        for ex in (None, "d1", "d2", "d3"):
            ps = [null.p(t, f, a, v, ex) for v in grid]
            if ps[0] is None:
                continue
            assert all(p1 >= p2 for p1, p2 in zip(ps, ps[1:])), (t, f, a, ex)
            assert all(null.floor(t, ex) - 1e-15 <= p <= 1.0 for p in ps)
            zs = [null.z(t, f, a, v, ex) for v in grid]
            assert all(z1 <= z2 for z1, z2 in zip(zs, zs[1:]))


def test_family_null_permutation_invariance_and_json_roundtrip():
    rng = np.random.default_rng(3)
    units = _units(rng, [("d1", "process", "P3", True, 600, 1.0), ("d2", "process", "P3", False, 800, 1.0),
                         ("d2", "netflow", "N2", False, 260, 0.7), ("d3", "netflow", "N2", True, 40, 0.7)])
    a = SU.FamilyNull.fit(units)
    shuffled = list(units)
    random.Random(7).shuffle(shuffled)
    b = SU.FamilyNull.fit(shuffled)
    assert json.dumps(a.to_json(), sort_keys=True) == json.dumps(b.to_json(), sort_keys=True)
    c = SU.FamilyNull.from_json(json.loads(json.dumps(a.to_json())))
    for t, f in (("process", "P3"), ("netflow", "N2")):
        for att in (True, False):
            for ex in (None, "d1", "d2", "d3"):
                for v in (0.0, 0.3, 1.0, 2.5, 7.0, 30.0):
                    assert a.p(t, f, att, v, ex) == b.p(t, f, att, v, ex) == c.p(t, f, att, v, ex)
    assert a.report() == c.report()
    with pytest.raises(ValueError):
        SU.FamilyNull.from_json({"kind": "EMap"})


def test_family_null_cross_fitting_removes_the_day_from_the_sample_only():
    """Cross-fitting leaves the unit's own day out of its stratum SAMPLE; the routes (merge
    decisions) and N_min (floor) are the whole null's, shared by null and test units (review
    finding: per-day routes / floors made null and test z ceilings differ)."""
    rng = np.random.default_rng(4)
    spec = [("d1", "process", "P4", True, 520, 1.0), ("d1", "process", "P4", False, 300, 1.0),
            ("d2", "process", "P4", True, 250, 2.0), ("d2", "process", "P4", False, 700, 0.5),
            ("d3", "process", "P4", True, 280, 1.0), ("d3", "process", "P4", False, 400, 3.0)]
    units = _units(rng, spec)
    full = SU.FamilyNull.fit(units)
    assert full.stratum("process", "P4", True) == ("process", "P4", True)              # 1050 / 1400: split
    for day in ("d1", "d2", "d3"):
        ref = SU.FamilyNull.fit(u for u in units if u.day != day)
        assert full.floor("process", day) == full.floor("process")                    # whole-null N_min
        for att in (True, False):
            assert full.stratum("process", "P4", att, day) == full.stratum("process", "P4", att)
            assert full.n("process", "P4", att, day) == ref.n("process", "P4", att)  # sample without the day
            for v in (0.0, 0.5, 1.7, 4.0, 12.0):
                p, pr = full.p("process", "P4", att, v, day), ref.p("process", "P4", att, v)
                if pr > ref.floor("process"):                                          # unfloored: same sample p
                    assert p == pytest.approx(max(pr, full.floor("process")))
                else:
                    assert full.floor("process") <= p <= pr
    assert full.p("process", "P4", True, 1.0, "not-a-null-day") == full.p("process", "P4", True, 1.0)


def test_family_null_merge_flip_under_cross_fitting_keeps_one_route_and_ceiling():
    """Review finding (a): 2 null days of 5000 attested + 300 non-attested units. The whole null
    splits the flag (600 >= 500), every leave-one-day-out null alone would merge (300 < 500). Null
    and test units must share the split route and the z ceiling ln(1 + 600)."""
    rng = np.random.default_rng(5)
    units = _units(rng, [("d1", "process", "P2", True, 5000, 0.3), ("d1", "process", "P2", False, 300, 1.0),
                         ("d2", "process", "P2", True, 5000, 0.3), ("d2", "process", "P2", False, 300, 1.0)])
    full = SU.FamilyNull.fit(units)
    assert SU.FamilyNull.fit(u for u in units if u.day != "d1").stratum("process", "P2", False)[2] is None
    for day in ("d1", "d2", None):
        assert full.stratum("process", "P2", False, day) == ("process", "P2", False)
        assert full.floor("process", day) == pytest.approx(1 / 601)
        assert full.z("process", "P2", False, 1e9, day) == pytest.approx(math.log(601))


def test_exchangeable_test_days_cross_bars_at_the_null_rate():
    """Every null and test day drawn from one distribution (allc-pre-like, 4 null days): test units
    above tau_0 must stay ~0 / day (exchangeable expectation n_test / (N_null + 1) ~ 0.25), above
    tau_1 ~1 / day. Per-day N_min used to give test units a higher z ceiling than any null unit, which
    put ~1.1 / day above tau_0 and ~2.1 / day above tau_1."""
    def day(rng, n):
        x = np.where(rng.random(n) < 0.6, 0.0, 0.5 * rng.lognormal(0.0, 0.6, n))
        return list(zip(x.tolist(), rng.beta(2, 6, n).tolist()))
    rng = np.random.default_rng(0)
    days = [f"2000-01-{i + 1:02d}" for i in range(4)]
    null = {d: day(rng, 5000) for d in days}
    fn = SU.FamilyNull.fit(T.NullUnit(d, "process", F.P2_modify, True, x) for d, r in null.items() for x, _b in r)
    rows = [(d, "process", fn.z("process", F.P2_modify, True, x, exclude_day=d), b) for d, r in null.items()
            for x, b in r]
    em = N2.EMap.fit(rows)
    tau = N2.bars([max(b, em.s_hat("process", e, exclude_day=d)) for d, _t, e, b in rows], len(days))
    c0, c1 = [], []
    for _ in range(30):
        s = np.asarray([max(b, em.s_hat("process", fn.z("process", F.P2_modify, True, x))) for x, b in day(rng, 5000)])
        c0.append(int(np.sum(s >= tau[0])))
        c1.append(int(np.sum(s >= tau[1])))
    assert np.mean(c0) < 0.5 and np.mean(c1) < 1.6


def test_combine_and_habit():
    assert SU.combine({}) == (0.0, 0.0)
    e, em = SU.combine({F.P1_parent: 5.0, F.P2_modify: 2.0, F.P3_peers: None})
    assert (e, em) == (5.0, 2.0)                                      # P1 is not edge-monotone
    units = [T.NullUnit("d1", "process", f, False, x) for f in (F.P1_parent, F.P2_modify)
             for x in (0.0,) * 8 + (1.0, 2.0)]
    null = SU.FamilyNull.fit(units)
    obs = T.NodeObs(9, "s", "process", "k", present={"P2": {"dir:x"}, "P3": set(), "P4": set(), "P5": set()},
                    choice={"P1": "usual"})
    h = SU.habit(obs, {F.P1_parent: 3.0, F.P2_modify: 1.5, F.P3_peers: 0.0}, null, False)
    assert h.node == 9 and not h.attested and h.s_hat is None
    assert h.fams[F.P2_modify].p == pytest.approx(2 / 11) and h.fams[F.P2_modify].z == pytest.approx(-math.log(2 / 11))
    assert h.fams[F.P3_peers].z is None and h.fams[F.P3_peers].stratum is None          # not admitted
    assert h.fams[F.P1_parent].stratum == ("process", "P1", None)
    assert h.E == pytest.approx(math.log(11)) and h.E_mono == pytest.approx(-math.log(2 / 11))
    nu = SU.null_units("d1", "process", True, {F.P2_modify: 1.0, "P3": 0.0, F.P4_children: None})
    assert [(u.family, u.x) for u in nu] == [(F.P2_modify, 1.0), (F.P3_peers, 0.0)]


# --------------------------------------------------------------------------- #
# 3. null2: k plans
# --------------------------------------------------------------------------- #
def _host(ds, clean, test):
    return FakeHabitHost({}, {}, ds=ds, art=f"/nonexistent/habitus_fake_{ds}/", clean=clean, test=test)


def test_plans_and_test_refs_on_the_fake_host(host):
    c1, c2, c3 = CLEAN
    t1 = TEST[1]
    k2 = N2.plan(host, "k2")
    assert [(r.day, r.ref_days, r.k_mode) for r in k2] == [(c1, (c2,), "k2"), (c2, (c1,), "k2")]   # lodo_plan
    assert N2.ref_for(host, t1, "k2") == T.Ref(host.ds, t1, (c1, c2), "k2")
    for kp in ("k2+1cal", "k2-recent", "allc"):                       # ablation plans are not in this repo
        with pytest.raises(ValueError):
            N2.plan(host, kp)
    pre = N2.plan(host, "allc-pre")
    assert [(r.day, r.ref_days, r.k_mode) for r in pre] == [(c1, (c2, c3), "allc-pre"), (c2, (c1, c3), "allc-pre"),
                                                            (c3, (c1, c2), "allc-pre")]
    assert N2.ref_for(host, t1, "allc-pre") == T.Ref(host.ds, t1, CLEAN, "allc-pre")
    k0 = N2.plan(host, "k0")                                         # the host's own pre-deployment days
    assert [(r.ds, r.day, r.k_mode) for r in k0] == [(host.ds, c, "k0") for c in CLEAN]
    assert N2.ref_for(host, t1, "k0") == T.Ref(host.ds, t1, (), "k0")
    assert N2.ref_for(host, c1, "k2") == k2[0]                       # a null day returns its plan Ref
    with pytest.raises(ValueError):
        N2.ref_for(host, c3, "k2")                                  # neither test nor k2 null day
    with pytest.raises(ValueError):
        N2.plan(host, "k3")
    assert N2.null_days(host, "allc-pre") == list(CLEAN)
    assert N2.plan_refs(host, "k2") == [(r, "null") for r in k2] + [(N2.ref_for(host, d, "k2"), "test") for d in TEST]


def test_plans_use_only_the_pre_deployment_window():
    # clean days after the first test day (CADETS 04-07..04-10 pattern) never enter a plan
    h = _host("CADETS_E3", ("04-02", "04-03", "04-04", "04-05", "04-07", "04-08"), ("04-06", "04-11"))
    assert [r.day for r in N2.plan(h, "allc-pre")] == ["04-02", "04-03", "04-04", "04-05"]
    assert N2.ref_for(h, "04-11", "allc-pre").ref_days == ("04-02", "04-03", "04-04", "04-05")
    assert [r.day for r in N2.plan(h, "k0")] == ["04-02", "04-03", "04-04", "04-05"]
    bad = _host("XHOST", ("04-02", "04-07"), ("04-06",))              # shots(2) crosses the first test day
    with pytest.raises(ValueError):
        N2.plan(bad, "k2")
    with pytest.raises(ValueError):
        N2.ref_for(bad, "04-06", "k2")
    with pytest.raises(ValueError):
        N2.plan(bad, "allc-pre")                                    # 1 pre-deployment day: no LODO


# --------------------------------------------------------------------------- #
# 4. null2: s_hat map, bars, X3c, persistence
# --------------------------------------------------------------------------- #
def _emap_rows(rng, days=("d1", "d2", "d3"), n=400):
    rows = []
    for d in days:
        e = rng.exponential(1.0, n) * (rng.random(n) > 0.5)
        b = rng.beta(2, 5, n)
        rows += [(d, "process", float(x), float(y)) for x, y in zip(e, b)]
        rows += [(d, "file", float(x), float(y)) for x, y in zip(e[: n // 4] * 2, b[: n // 4])]
    return rows


def test_emap_is_a_monotone_quantile_quantile_map():
    rng = np.random.default_rng(5)
    rows = _emap_rows(rng)
    m = N2.EMap.fit(rows)
    es = np.sort([r[2] for r in rows if r[1] == "process"])
    bs = np.sort([r[3] for r in rows if r[1] == "process"])
    assert m.s_hat("process", 0.0) == pytest.approx(bs[0])            # E = 0 ties -> the lowest base0
    for j in (5, 700, len(es) - 1):                                   # unique E_(j) -> b_(rank of E_(j))
        v = es[j]
        if np.sum(es == v) == 1:
            assert m.s_hat("process", v) == pytest.approx(bs[int(np.searchsorted(es, v))])
    e_max, b_max = es[-1], bs[-1]
    assert m.s_hat("process", e_max) == pytest.approx(b_max)          # continuous at the unique maximum
    for d in (0.1, 1.0, 5.0):
        assert m.s_hat("process", e_max + d) == pytest.approx(b_max + (1 - b_max) * (1 - math.exp(-d)))
    grid = np.sort(np.concatenate([[0.0], rng.exponential(2.0, 400), [e_max, e_max + 1e-9, 40.0]]))
    for t in ("process", "file"):
        for ex in (None, "d1", "d2", "d3"):
            s = [m.s_hat(t, v, ex) for v in grid]
            assert all(a <= b for a, b in zip(s, s[1:])) and 0.0 <= s[0] and s[-1] < 1.0
    assert m.cdf("process", 0.0) == 0.0 and m.quantile("process", 1.0) == pytest.approx(b_max)
    assert m.s_hat("netflow", 3.0) is None and m.n("netflow") == 0
    with pytest.raises(ValueError):
        N2.EMap.fit([("d1", "process", 1.0, 1.5)])
    with pytest.raises(ValueError):
        N2.EMap.fit([("d1", "process", float("inf"), 0.5)])


def test_emap_cross_fitting_permutation_and_json():
    rng = np.random.default_rng(6)
    rows = _emap_rows(rng)
    m = N2.EMap.fit(rows)
    shuffled = list(rows)
    random.Random(1).shuffle(shuffled)
    m2 = N2.EMap.fit(shuffled)
    m3 = N2.EMap.from_json(json.loads(json.dumps(m.to_json())))
    assert json.dumps(m.to_json(), sort_keys=True) == json.dumps(m2.to_json(), sort_keys=True)
    for d in ("d1", "d2", "d3"):
        ref = N2.EMap.fit(r for r in rows if r[0] != d)
        for v in (0.0, 0.2, 1.0, 3.0, 9.0):
            assert m.s_hat("process", v, d) == ref.s_hat("process", v) == m2.s_hat("process", v, d) \
                == m3.s_hat("process", v, d)
    assert m.report()["ntypes"]["process"]["n"] == 1200
    h = T.NodeHabit(1, "process", True, {}, 2.0, 2.0)
    assert N2.with_s_hat(h, m, "d2").s_hat == m.s_hat("process", 2.0, "d2")


def test_bars_match_the_label_free_bar_copy():
    rng = np.random.default_rng(7)
    v = rng.random(1000)
    b = N2.bars(v, 4)
    assert set(b) == set(H.BARS)
    for B, tau in b.items():
        assert tau == label_free_bar(v, 4, B)
        assert np.sum(v >= tau) / 4 <= B
    assert N2.bars(v, 4, (500,))[500] == -math.inf and N2.bars([], 4)[0] is None
    assert b[0] > b[1] > b[20]


def test_save_load_roundtrip_and_byte_determinism(hb_root):
    rng = np.random.default_rng(8)
    null = SU.FamilyNull.fit(_units(rng, [("d1", "file", "F1", True, 30, 1.0), ("d2", "file", "F1", False, 30, 1.0)]))
    emap = N2.EMap.fit(_emap_rows(rng, n=40))
    p1 = N2.save(null, "HBFAKE", "fnull_HB-Z_k2", hb_root)
    with open(p1, "rb") as f:
        b1 = f.read()
    p2 = N2.save(SU.FamilyNull.fit(reversed(list(_units(np.random.default_rng(8), [
        ("d1", "file", "F1", True, 30, 1.0), ("d2", "file", "F1", False, 30, 1.0)])))), "HBFAKE", "fnull_HB-Z_k2",
        hb_root)
    with open(p2, "rb") as f:
        assert f.read() == b1                                        # same units, any order: same bytes
    assert p1 == os.path.join(hb_root, "nulls", "HBFAKE", "fnull_HB-Z_k2.json.gz")
    back = N2.load("HBFAKE", "fnull_HB-Z_k2", hb_root)
    assert isinstance(back, SU.FamilyNull) and back.p("file", "F1", True, 0.7) == null.p("file", "F1", True, 0.7)
    N2.save(emap, "HBFAKE", "emap_HB-Z_k2.json.gz", hb_root)
    assert N2.load("HBFAKE", "emap_HB-Z_k2", hb_root).s_hat("process", 1.0) == emap.s_hat("process", 1.0)
    N2.save({"bars": N2.bars([0.1, 0.2], 1)}, "HBFAKE", "bars", hb_root)
    assert N2.load("HBFAKE", "bars", hb_root)["bars"]["0"] == pytest.approx(np.nextafter(0.2, 1))
    with gzip.open(p1, "rt") as f:
        assert json.load(f)["kind"] == "FamilyNull"




def test_h501_parity_split_and_selection_days():
    h501 = _host("optc_h501", ("2019-09-19", "2019-09-20", "2019-09-21", "2019-09-22"),
                 ("2019-09-23", "2019-09-24", "2019-09-25"))
    assert N2.selection_days(h501) == ["2019-09-19", "2019-09-21"] == list(H.H501_SELECTION_DAYS)
    theia = _host("THEIA_E3", ("04-03", "04-04", "04-05", "04-09", "04-11"), ("04-10", "04-12"))
    assert N2.selection_days(theia) == ["04-03", "04-04", "04-05", "04-09"]
    assert [r.day for r in N2.plan(theia, "k0")] == ["04-03", "04-04", "04-05", "04-09"]


def test_null_files_are_refused_under_the_release_dir(tmp_path, monkeypatch):
    from habitus.config import load_settings
    from habitus.decide import scorefile
    rel = tmp_path / "release"
    s = load_settings(overrides={"paths": {"data": str(rel), "runs": str(tmp_path / "runs")}})
    monkeypatch.setattr(scorefile, "_settings", lambda settings=None: s)
    with pytest.raises(PermissionError):
        N2.save({"x": 1}, "HBFAKE", "x", root=str(rel / "habit"))
    assert N2.save({"x": 1}, "HBFAKE", "x", root=str(tmp_path / "runs" / "habit")).endswith("x.json.gz")


def test_surprise_and_null_modules_pass_the_hygiene_grep_and_import_no_label_code():
    hb = os.path.join(REPO, "src", "habitus", "decide", "habit")
    for m in ("surprise", "null2"):
        text = open(os.path.join(hb, f"{m}.py"), encoding="utf-8").read()
        assert [ln for ln in text.splitlines() if HYGIENE.search(ln)] == [], m
    code = ("import sys; import habitus.decide.habit.surprise, habitus.decide.habit.null2; "
            "from habitus import guard; print([m for m in sys.modules if guard.blocked_import(m)])")
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "src"))
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "[]"
