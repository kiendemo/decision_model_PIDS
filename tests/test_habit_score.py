"""Habit composite, score files, bars and HB0 (CPU, the shared fake host and synthetic arrays):
score (the lineage-split composite: bounds, E2 special case, monotone in the tier; rows per member;
byte-identical score files under the habit root, refused under the release dir; node / entity
scores and null pools), bars (the label-free bar, the two-tier cap by construction, alert rows) and
retrofit (HB0 on E2-format files: plan, meta / reference / head checks, reproduces E2 with every node
LINEAGE, k0 has no acquittal, allc-pre refused where E2_kall is not the window, deterministic run),
plus the real attestation under added links, the label guard and the real E2 files (asset)."""

import glob
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import REPO, missing_asset
from habit_fakes import ATTACK, BENIGN, CLEAN, ID, TEST, FakeHabitHost, build_day, make_habit_host

from habitus.decide.composite import LAMBDA, compose_e2
from habitus.decide.habit import bars as decide
from habitus.decide.habit import null2, retrofit, score
from habitus.decide.null import bar as label_free_bar
from habitus.decide.scorefile import write_score
from habitus.questions.habit import constants as H
from habitus.questions.habit import head
from habitus.questions.habit import types as T
from habitus.state.select import CAPS, TYPES
from habitus.state.select import head as state_head

HYGIENE = __import__("re").compile(r"gt\(|ground_truth|host_labels|Ground_Truth|gt_")
MINE = ("score.py", "bars.py", "retrofit.py", "channel.py")
TIERS = tuple(T.Attest)


@pytest.fixture
def hb_root(tmp_path):
    return str(tmp_path / "hb")


@pytest.fixture(scope="module")
def host():
    return make_habit_host()


def _u(key, i=0):
    h = hashlib.sha1(f"{key}#{i}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


def _route(sig, k0=False):
    """A deterministic E2-like route record of one signature."""
    qv, qt = round(_u(sig, 1), 6), round(_u(sig, 2), 6)
    qa = None if k0 else round(_u(sig, 3), 6)
    rec = {"qv": qv, "qt": qt, "qa": qa, "hard": _u(sig, 4) < 0.15, "cov_min": 0.99}
    rec["base"] = max(qv, qt)
    rec["composite"] = compose_e2(rec)
    return rec


# --------------------------------------------------------------------------- #
# 1. the composite - property tests
# --------------------------------------------------------------------------- #
def _grid(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        base = float(rng.uniform(0, 1)) if rng.uniform() > 0.05 else float(rng.choice([0.0, 1.0]))
        qa = None if rng.uniform() < 0.1 else (float(rng.uniform(0, 1)) if rng.uniform() > 0.05 else 1.0)
        yield base, qa, bool(rng.uniform() < 0.2), TIERS[int(rng.integers(len(TIERS)))], \
            T.NODE_TYPES[int(rng.integers(3))]


def test_composite_bounds_hold_for_every_input():
    for base, qa, hard, tier, ntype in _grid():
        s, a = score.composite(base, qa, hard, tier, ntype)
        assert 0.0 <= a <= 1.0
        assert s <= base + 1e-15                                        # never above base
        assert s >= 0.5 * base - 1e-15                                  # s >= 0.5 base always
        assert math.isclose(s, base * (1 - LAMBDA * a), rel_tol=0, abs_tol=1e-15)
        if hard or qa is None:
            assert s == base and a == 0.0                               # hard tier / QA not asked
        if tier != T.Attest.LINEAGE:
            assert s >= 0.75 * base - 1e-15                             # s >= 0.75 base unless LINEAGE


def test_composite_lineage_split_and_e2_special_case():
    for base, qa, hard, tier, ntype in _grid(1000, seed=1):
        rec = {"qv": base, "qt": base * 0.5, "qa": qa, "hard": hard}
        s_lin, _ = score.composite(base, qa, hard, T.Attest.LINEAGE, ntype)
        assert math.isclose(s_lin, compose_e2(rec), abs_tol=1e-15)           # every node LINEAGE == E2
        s, a = score.composite(base, qa, hard, T.Attest.IDENTITY, ntype)
        if not hard and qa is not None:
            assert math.isclose(a, H.C_K * qa)


def test_composite_is_non_increasing_in_the_tier():
    """A higher tier can only lower s: revoking LINEAGE (added links) never lowers a score."""
    for base, qa, hard, _tier, ntype in _grid(1000, seed=2):
        s = [score.composite(base, qa, hard, t, ntype)[0] for t in TIERS]
        assert all(x >= y - 1e-15 for x, y in zip(s, s[1:]))


def test_composite_rejects_bad_inputs():
    for bad in ((math.nan, 0.5), (math.inf, 0.5), (1.2, 0.5), (-0.1, 0.5), (0.5, math.nan), (0.5, 1.5)):
        with pytest.raises(ValueError):
            score.composite(bad[0], bad[1], False, T.Attest.NONE)
    with pytest.raises(ValueError):
        score.composite(0.5, 0.5, False, T.Attest.NONE, ntype="socket")
    assert score.composite(0.5, 0.5, False, int(T.Attest.LINEAGE)) == score.composite(0.5, 0.5, False, T.Attest.LINEAGE)


def test_base_of():
    assert score.base_of(0.2, 0.3) == 0.3
    assert score.base_of(0.2, 0.3, None) == 0.3
    assert score.base_of(0.2, 0.3, 0.9) == 0.9
    assert score.base_of(0.2, 0.3, 0.1) == 0.3                               # accuse-only
    with pytest.raises(ValueError):
        score.base_of(0.2, math.nan)
    with pytest.raises(ValueError):
        score.base_of(0.2, 0.3, 1.5)


# --------------------------------------------------------------------------- #
# 2. rows_for_day on the fake host
# --------------------------------------------------------------------------- #
def _day_parts(host, day, ref_days, caps=CAPS, k0=False):
    kmode = "k0" if not ref_days else "k2"
    ref = T.Ref(host.ds, day, tuple(ref_days), kmode)
    h = head.q0(host, ref, caps)
    routes = {s: {k: _route(s, k0)[k] for k in score.ROUTE_KEYS} for s in head.head_sigs(h)}
    mem = head.members(host, day, h)
    return ref, h, routes, mem


def test_q0_is_the_routed_head(host):
    ref = T.Ref(host.ds, TEST[1], CLEAN[:2], "k2")
    assert head.q0(host, ref) == state_head(host, TEST[1], list(CLEAN[:2]), CAPS)
    assert head.q0(host, T.Ref(host.ds, TEST[1], (), "k0")) == state_head(host, TEST[1], None, CAPS)
    mem = head.members(host, TEST[1], head.q0(host, ref))
    assert sum(len(v) for v in mem.values()) == len(host.nodes(TEST[1]))       # tiny host: everything in head


def test_rows_cover_every_member_and_follow_the_tiers(host):
    d = TEST[1]
    ref, h, routes, mem = _day_parts(host, d, CLEAN[:2])
    nodes = sorted(int(n) for v in mem.values() for n in v.tolist())
    tiers = {n: TIERS[n % 4] for n in nodes}
    rows = score.rows_for_day(host, ref, h, routes, tiers)
    assert [r.node for r in rows] == nodes                                   # every member, node order
    twin = host.nodes(d)[ID["sh"]]
    assert {r.node for r in rows if r.sig == twin} == {ID["sh"], ID["sh_twin"]}
    for r in rows:
        assert r.ntype == host.card(r.sig)["type"] and r.tier == tiers[r.node]
        s, a = score.composite(r.base, r.qa, r.hard, r.tier, r.ntype)
        assert (r.s, r.a) == (s, a) and r.base == max(r.qv, r.qt) and r.s_hat is None
    # an s_hat only raises base (accuse-only)
    habit = {n: T.NodeHabit(n, "process", True, {}, 3.0, 3.0, s_hat=0.97) for n in nodes[:3]}
    rows2 = {r.node: r for r in score.rows_for_day(host, ref, h, routes, tiers, habit=habit)}
    for n in nodes[:3]:
        assert rows2[n].base == max(0.97, rows2[n].qv, rows2[n].qt) and rows2[n].s_hat == 0.97
    with pytest.raises(KeyError):
        score.rows_for_day(host, ref, h, routes, {n: t for n, t in tiers.items() if n != nodes[0]})
    with pytest.raises(KeyError):
        score.rows_for_day(host, ref, h, {s: r for s, r in routes.items() if s != twin}, tiers)
    census = score.tier_census(rows)
    assert sum(sum(v.values()) for v in census.values()) == len(rows)


def test_fail_safe_downgrading_tiers_never_lowers_a_score(host):
    """Links can only revoke attestation (attest contract): any tier map that is pointwise lower
    gives pointwise higher-or-equal composites."""
    ref, h, routes, mem = _day_parts(host, TEST[1], CLEAN[:2])
    nodes = sorted(int(n) for v in mem.values() for n in v.tolist())
    rng = np.random.default_rng(3)
    for _ in range(20):
        before = {n: TIERS[int(rng.integers(4))] for n in nodes}
        after = {n: T.Attest(int(rng.integers(0, int(t) + 1))) for n, t in before.items()}
        sb = {r.node: r.s for r in score.rows_for_day(host, ref, h, routes, before)}
        sa = {r.node: r.s for r in score.rows_for_day(host, ref, h, routes, after)}
        assert all(sa[n] >= sb[n] for n in nodes)


# --------------------------------------------------------------------------- #
# 3. score files
# --------------------------------------------------------------------------- #
def _scoreday(host, day, ref_days, role, level="HB0", arm=None, ds=None, tiers_of=lambda n: T.Attest.NONE):
    ref, h, routes, mem = _day_parts(host, day, ref_days)
    nodes = sorted(int(n) for v in mem.values() for n in v.tolist())
    rows = score.rows_for_day(host, ref, h, routes, {n: tiers_of(n) for n in nodes})
    meta = T.base_meta(T.Ref(ds or host.ds, day, ref.ref_days, ref.k_mode), level, "k2", role, CAPS, arm=arm)
    return T.ScoreDay.from_rows(meta, h, routes, rows, score.head_fallback(host, day, h))


def test_write_read_roundtrip_is_byte_identical(hb_root, host):
    sd = _scoreday(host, TEST[0], CLEAN[:2], "test")
    p = score.write(sd, hb_root)
    assert p == score.day_path(host.ds, "HB0_k2", TEST[0], hb_root)
    assert p == os.path.join(hb_root, "scores", host.ds, "HB0_k2", f"{TEST[0]}.json.gz")
    b1 = open(p, "rb").read()
    assert score.write(sd, hb_root) == p and open(p, "rb").read() == b1      # same inputs -> same bytes
    assert b1[4:8] == b"\x00\x00\x00\x00"                                    # gzip mtime 0
    back = score.read(p)
    assert back.to_json() == json.loads(json.dumps(sd.to_json()))
    got, missing = score.load_days(host.ds, "HB0_k2", [TEST[0], TEST[1]], hb_root)
    assert list(got) == [TEST[0]] and missing == [TEST[1]]
    sd2 = _scoreday(host, TEST[0], CLEAN[:2], "test", level="HB-L", arm="mq")
    assert score.write(sd2, hb_root).endswith(os.path.join("HB-L.mq_k2", f"{TEST[0]}.json.gz"))
    assert score.ldir_of(sd2.meta) == T.level_dir("HB-L.mq", "k2") == "HB-L.mq_k2"


def test_score_files_are_refused_under_the_release_dir(tmp_path, host, monkeypatch):
    from habitus.config import load_settings
    from habitus.decide import scorefile
    rel = tmp_path / "release"
    s = load_settings(overrides={"paths": {"data": str(rel)}})
    monkeypatch.setattr(scorefile, "_settings", lambda settings=None: s)
    sd = _scoreday(host, TEST[0], CLEAN[:2], "test")
    with pytest.raises(PermissionError):
        score.write(sd, str(rel / "habit"))
    assert os.path.exists(score.write(sd, str(tmp_path / "ok")))


def test_level_dirs_and_split_level():
    assert T.level_dir("HB0", "k2") == "HB0_k2" and T.level_dir("HB-Z", "allc-pre") == "HB-Z_allc-pre"
    assert T.split_level("HB.head") == ("HB", "head") and T.split_level("HB-L") == ("HB-L", None)
    assert [T.level_dir(a, "k0") for a in H.ARM_LEVELS] == ["HB-L_k0", "HB-L.s0_k0", "HB-L.mq_k0", "HB-L.zs_k0",
                                                            "HB.head_k0", "HB.08b_k0"]
    for bad in (("HB-C", "k2"), ("HB0", "k2+1cal"), ("HB-S", "k2"), ("HB0", "allc")):
        with pytest.raises(ValueError):
            T.level_dir(*bad)
    assert set(H.ARM_DM) == set(H.HABIT_LEVELS) and "HB-L.full" not in H.ARM_DM


def test_node_scores_entity_scores_and_null_pool(host):
    by_test = {d: _scoreday(host, d, CLEAN[:2], "test") for d in TEST}
    s = score.node_scores(host, by_test)
    assert s.shape == host.universe.shape and np.isfinite(s).all()
    ref = np.full(len(host.universe), -np.inf)
    for d in TEST:
        ids, v = by_test[d].node_values(host)
        for n, x in zip(ids.tolist(), v.tolist()):
            i = int(np.searchsorted(host.universe, n))
            ref[i] = max(ref[i], x)
    assert np.array_equal(s, ref)
    with pytest.raises(KeyError):
        score.node_scores(host, {TEST[0]: by_test[TEST[0]]})
    with pytest.raises(ValueError):
        score.node_scores(host, {**by_test, TEST[1]: _scoreday(host, TEST[1], CLEAN[:2], "null")})
    with pytest.raises(ValueError):                                          # two levels never mix
        score.node_scores(host, {**by_test, TEST[1]: _scoreday(host, TEST[1], CLEAN[:2], "test", level="HB-L",
                                                               arm="mq")})
    with pytest.raises(ValueError):                                          # s_hat is None at HB0
        score.node_scores(host, by_test, column="s_hat")
    codes, es = score.entity_scores(host, s)
    for c, x in zip(codes.tolist(), es.tolist()):
        assert x == s[host.ukey_codes == c].max()
    nulls = {CLEAN[0]: _scoreday(host, CLEAN[0], CLEAN[1:2], "null"),
             CLEAN[1]: _scoreday(host, CLEAN[1], CLEAN[0:1], "null")}
    v, n = score.null_pool(host, nulls)
    assert n == 2 and len(v) == sum(len(host.nodes(d)) for d in CLEAN[:2])
    with pytest.raises(ValueError):
        score.null_pool(host, by_test)


# --------------------------------------------------------------------------- #
# 4. bars: the label-free bar, alerts, rows
# --------------------------------------------------------------------------- #
def _days(n_days=6, n=400, seed=0, ids_str=False):
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n_days):
        ids = np.arange(1000 * i, 1000 * i + n, dtype=np.int64)
        s = np.round(rng.beta(1.2, 4, n), 3)                                  # rounded: many ties
        out[f"2020-02-{i + 1:02d}"] = (ids.astype(str) if ids_str else ids, s)
    return out


def test_bars_are_the_label_free_bar():
    rng = np.random.default_rng(0)
    null = rng.uniform(size=3000)
    b = decide.bars(null, 7)
    assert set(b) == set(H.BARS)
    for B in H.BARS:
        assert b[B] == label_free_bar(null, 7, B) == null2.bars(null, 7)[B]
        assert (null >= b[B]).sum() / 7 <= B
    assert b[0] > null.max() and b[0] >= b[1] >= b[20]
    assert decide.bars(null[:10], 2)[20] == -math.inf                       # budget covers the whole null
    with pytest.raises(ValueError):
        decide.bars([], 3)
    with pytest.raises(ValueError):
        decide.bars([0.1, math.nan], 3)


def test_two_tier_cap_bounds_the_weak_tier_by_construction():
    for seed in range(12):
        dv = _days(seed=seed, n=int(50 + 100 * seed))
        null = np.random.default_rng(100 + seed).beta(1.2, 6, 3000)
        bars = decide.bars(null, 3)
        for cap in (None, 0, 1, 5, 20):
            al = decide.alerts(dv, bars, 20, weak_cap=cap)
            for d, a in al.items():
                ids, s = dv[d]
                want = 20 if cap is None else cap
                assert len(a.weak) <= want                                   # weak <= cap per day
                assert set(a.strong) == set(ids[s >= bars[1]].tolist())    # strong uncapped
                band = {i: x for i, x in zip(ids.tolist(), s.tolist()) if bars[20] <= x < bars[1]}
                assert set(a.weak) <= set(band) and len(a.weak) == min(want, len(band))
                if a.weak:                                                   # largest margin first, ties by id
                    worst = min(band[i] for i in a.weak)
                    assert all(band[i] <= worst for i in set(band) - set(a.weak))
                    tied_out = [i for i in set(band) - set(a.weak) if band[i] == worst]
                    tied_in = [i for i in a.weak if band[i] == worst]
                    assert not tied_out or max(tied_in) < min(tied_out)
                assert not set(a.strong) & set(a.weak) and a.alerts == tuple(sorted(set(a.strong) | set(a.weak)))
        b0 = decide.alerts(dv, bars, 0)
        unc = decide.alerts(dv, bars, 20, capped=False)
        for d, (ids, s) in dv.items():
            assert set(b0[d].alerts) == set(ids[s >= bars[0]].tolist()) and not b0[d].weak
            assert set(unc[d].alerts) == set(ids[s >= bars[20]].tolist())
            assert set(decide.alerts(dv, bars, 20)[d].alerts) <= set(unc[d].alerts)


def test_alerts_validate_bars_and_scores():
    dv = _days()
    with pytest.raises(KeyError):
        decide.alerts(dv, {0: 0.9, 20: 0.1}, 20)                             # no tau_1
    with pytest.raises(KeyError):
        decide.alerts(dv, {0: 0.9, 1: 0.5}, 20)
    with pytest.raises(ValueError):
        decide.alerts(dv, {0: 0.9, 1: 0.2, 20: 0.5}, 20)                     # tau_20 above tau_1
    with pytest.raises(ValueError):
        decide.alerts({"d": (np.arange(3), np.array([0.1, math.nan, 0.2]))}, {0: 1, 1: 0.5, 20: 0.1}, 20)


def test_fixed_rate_ranking_and_string_ids():
    dv = {"d1": (np.array([5, 3, 9, 1]), np.array([0.5, 0.5, 0.9, 0.1])),
          "d2": (np.array(["b", "a", "c"]), np.array([0.2, 0.2, 0.2]))}
    assert decide.fixed_rate(dv, 2) == {"d1": (9, 3), "d2": ("a", "b")}
    r = decide.ranking(dv, max_per_day=3)
    assert r["d1"].tolist() == [9, 3, 5] and r["d2"].tolist() == ["a", "b", "c"]
    al = decide.alerts(dv, {0: 0.95, 1: 0.6, 20: 0.15}, 20, weak_cap=1)
    assert al["d1"].strong == (9,) and al["d1"].weak == (3,) and al["d2"].weak == ("a",)


def test_alert_rows_counts_and_cap_loss():
    dv = _days(n=600)
    bars = decide.bars(np.random.default_rng(5).beta(1.2, 8, 2000), 2)
    rows = decide.alert_rows(dv, bars)
    assert set(rows["budgets"]) == set(H.BARS) and set(rows["fixed"]) == set(H.FIXED_ALERT_ROWS)
    r = rows["budgets"][20]
    for d in dv:
        assert r["capped"][d] == r["strong"][d] + r["weak"][d] <= r["uncapped"][d]
        assert r["cap_loss"][d] == r["uncapped"][d] - r["capped"][d] >= 0 and r["weak"][d] <= H.WEAK_CAP
        assert len(r["alerts"][d]) == r["capped"][d]
        assert len(rows["fixed"][5][d]) == 5
    assert rows["budgets"][0]["cap_loss_per_day"] == 0                      # B = 0 is never capped


# --------------------------------------------------------------------------- #
# 5. HB0 retrofit on E2-format files
# --------------------------------------------------------------------------- #
def _write_e2(root, host, kplan="k2"):
    """E2-format files (the shipped routed score-file layout) of `host` for a plan, routes from _route."""
    k0 = kplan == "k0"
    for ref, role in retrofit.plan(host, kplan):
        h = state_head(host, ref.day, list(ref.ref_days) or None, CAPS)
        scores = {s: _route(s, k0) for t in TYPES for s in h[t]}
        meta = {"ds": host.ds, "day": ref.day, "level": "E2", "k": retrofit.E2_KLABEL[kplan],
                "ref": list(ref.ref_days), "role": role, "caps": dict(CAPS), "engine": "fake",
                "written": "2026-10-20 00:00:00"}
        write_score({"meta": meta, "head": h, "scores": scores, "fallback": {}},
                    retrofit.e2_path(host.ds, kplan, ref.day, root))


@pytest.fixture
def e2root(tmp_path, host):
    root = str(tmp_path / "e2")
    for kp in ("k2", "allc-pre", "k0"):
        _write_e2(root, host, kp)
    return root


class _Prof:
    """Stand-in reference profile (retrofit only checks prof is not None; tiers are patched)."""
    def __init__(self, ref):
        self.ref = tuple(ref)


def _rule(n):
    return (T.Attest.LINEAGE, T.Attest.IDENTITY, T.Attest.NONE)[n % 3]


@pytest.fixture
def fake_tiers(monkeypatch):
    """Patch the real lineage_tiers / profile with a deterministic rule: LINEAGE for node ids divisible
    by 3, IDENTITY for 1 mod 3, NONE otherwise."""
    monkeypatch.setattr(retrofit, "lineage_tiers", lambda host, day, prof, nodes=None: {n: _rule(n) for n in nodes})
    monkeypatch.setattr(retrofit, "profile", lambda host, ref_days: _Prof(ref_days) if ref_days else None)
    return _rule


def test_plan_follows_lodo_and_predeploy(host):
    p = retrofit.plan(host, "k2")
    assert [(r.day, r.ref_days, role) for r, role in p] == \
        [(CLEAN[0], (CLEAN[1],), "null"), (CLEAN[1], (CLEAN[0],), "null")] + \
        [(d, CLEAN[:2], "test") for d in TEST]
    a = retrofit.plan(host, "allc-pre")
    assert [(r.day, r.ref_days) for r, role in a if role == "null"] == \
        [(c, tuple(x for x in CLEAN if x != c)) for c in CLEAN]
    assert all(r.ref_days == CLEAN for r, role in a if role == "test")
    assert all(r.k_mode == "k0" and not r.ref_days for r, _ in retrofit.plan(host, "k0"))
    with pytest.raises(ValueError):
        retrofit.plan(host, "k2+1cal")
    assert retrofit.ldir("k2") == "HB0_k2"


def test_e2_day_checks_meta_and_reference(e2root, host):
    ref = T.Ref(host.ds, TEST[0], CLEAN[:2], "k2")
    raw = retrofit.e2_day(host.ds, "k2", TEST[0], e2root, ref=ref, role="test")
    routes = retrofit.e2_routes(raw)
    assert set(routes) == {s for t in TYPES for s in raw["head"][t]}
    assert all(set(r) == set(score.ROUTE_KEYS) for r in routes.values())
    with pytest.raises(ValueError):
        retrofit.e2_day(host.ds, "k2", TEST[0], e2root, ref=T.Ref(host.ds, TEST[0], CLEAN[1:], "k2"))
    with pytest.raises(ValueError):
        retrofit.e2_day(host.ds, "k2", TEST[0], e2root, role="null")
    with pytest.raises(FileNotFoundError):
        retrofit.e2_day(host.ds, "k2", "2020-03-01", e2root)
    with pytest.raises(ValueError):
        retrofit.e2_path(host.ds, "allc", TEST[0], e2root)
    bad = dict(raw, scores={s: r for s, r in list(raw["scores"].items())[1:]})
    with pytest.raises(KeyError):
        retrofit.e2_routes(bad)


def test_retrofit_day_reproduces_e2_and_applies_the_split(e2root, host, fake_tiers):
    for d in TEST:
        ref = T.Ref(host.ds, d, CLEAN[:2], "k2")
        raw = retrofit.e2_day(host.ds, "k2", d, e2root)
        all_lin = {n: T.Attest.LINEAGE for n in host.nodes(d)}
        sd = retrofit.retrofit_day(host, ref, "k2", _Prof(ref.ref_days), "test", e2=e2root, tiers=all_lin)
        rows = sd.nodes
        for sig, s in zip(rows["sig"], rows["s"]):
            assert math.isclose(s, raw["scores"][sig]["composite"], abs_tol=1e-15)   # all LINEAGE == E2
        sd = retrofit.retrofit_day(host, ref, "k2", _Prof(ref.ref_days), "test", e2=e2root)
        assert sd.meta["level"] == "HB0" and sd.meta["arm"] is None and sd.meta["head_rule"] == "Q0"
        assert sd.meta["ref"] == list(CLEAN[:2]) and sd.meta["role"] == "test"
        assert sd.meta["attest"] == "strict" and sd.meta["a_net"] == "split" and sd.meta["c_k"] == H.C_K
        assert sd.meta["e2"] == {"dir": "E2_k2", "engine": "fake", "written": "2026-10-20 00:00:00"}
        assert sorted(sd.nodes["id"]) == sd.nodes["id"] and len(sd.nodes["id"]) == len(host.nodes(d))
        for n, sig, s, base, a, tier in zip(*(sd.nodes[c] for c in ("id", "sig", "s", "base", "a", "tier"))):
            r = raw["scores"][sig]
            assert tier == int(fake_tiers(n))
            assert base == max(r["qv"], r["qt"]) and s >= 0.5 * base
            if tier != T.Attest.LINEAGE:
                assert s >= 0.75 * base
            if r["hard"] or r["qa"] is None:
                assert s == base
        ids, v = sd.node_values(host)
        assert len(ids) == len(host.nodes(d))
    with pytest.raises(ValueError):                                          # wrong k mode for the plan
        retrofit.retrofit_day(host, T.Ref(host.ds, TEST[0], CLEAN, "allc-pre"), "k2", None, "test", e2=e2root)
    with pytest.raises(ValueError):                                          # the plan's role is checked
        retrofit.retrofit_day(host, T.Ref(host.ds, TEST[0], CLEAN[:2], "k2"), "k2", None, "null", e2=e2root)


def test_retrofit_refuses_a_head_that_is_not_q0(e2root, host, fake_tiers):
    d = TEST[1]
    p = retrofit.e2_path(host.ds, "k2", d, e2root)
    raw = json.load(gzip.open(p, "rt"))
    raw["head"]["process"] = raw["head"]["process"][:-1]
    write_score(raw, p)
    with pytest.raises(ValueError):
        retrofit.retrofit_day(host, T.Ref(host.ds, d, CLEAN[:2], "k2"), "k2", _Prof(CLEAN[:2]), "test", e2=e2root)


def test_retrofit_k0_has_no_acquittal(e2root, host, fake_tiers):
    ref = T.Ref(host.ds, TEST[1], (), "k0")
    sd = retrofit.retrofit_day(host, ref, "k0", None, "test", e2=e2root)
    assert sd.nodes["s"] == sd.nodes["base"] and set(sd.nodes["tier"]) == {int(T.Attest.NONE)}


def test_allc_pre_refused_where_e2_kall_is_not_the_window(e2root, hb_root, fake_tiers):
    days, links = {}, {}
    late = "2020-01-07"                                                       # a clean day after the first test day
    for d in CLEAN + TEST + (late,):
        days[d], links[d] = build_day(d, list(BENIGN) + (list(ATTACK) if d == TEST[1] else []))
    h2 = FakeHabitHost(days, links, clean=CLEAN + (late,), test=TEST)
    assert not retrofit.kall_valid(h2)
    with pytest.raises(ValueError):
        retrofit.retrofit_day(h2, T.Ref(h2.ds, TEST[0], CLEAN, "allc-pre"), "allc-pre", _Prof(CLEAN), "test",
                              e2=e2root)
    assert retrofit.run(h2.ds, kplans=("allc-pre",), root=hb_root, e2=e2root, host=h2) == []   # refused, logged
    with pytest.raises(ValueError):
        retrofit.run("NOT_A_HOST", host=h2)                                    # the host must be the dataset's


def test_run_writes_test_and_null_days_deterministically(hb_root, e2root, host, fake_tiers):
    out = retrofit.run(host.ds, kplans="2,allc-pre", root=hb_root, e2=e2root, host=host)
    n_days = {kp: len(retrofit.plan(host, kp)) for kp in ("k2", "allc-pre")}
    assert len(out) == sum(n_days.values())
    assert all(p.startswith(os.path.join(hb_root, "scores", host.ds) + os.sep) for p in out)
    for kp in ("k2", "allc-pre"):
        got, missing = score.load_days(host.ds, retrofit.ldir(kp), [r.day for r, _ in retrofit.plan(host, kp)],
                                       hb_root)
        assert not missing
        for d, sd in got.items():
            assert sd.meta["role"] == ("test" if d in TEST else "null")
    before = {p: open(p, "rb").read() for p in out}
    assert retrofit.run(host.ds, kplans=("k2", "allc-pre"), root=hb_root, e2=e2root, host=host) == out
    assert all(open(p, "rb").read() == b for p, b in before.items())
    nb, _ = score.load_days(host.ds, "HB0_k2", list(CLEAN[:2]), hb_root)
    v, n = score.null_pool(host, nb)
    tau = decide.bars(v, n)
    tb, _ = score.load_days(host.ds, "HB0_k2", list(TEST), hb_root)
    rows = decide.alert_rows(score.day_values(host, tb, role="test"), tau)
    assert all(rows["budgets"][20]["weak"][d] <= H.WEAK_CAP for d in TEST)


# --------------------------------------------------------------------------- #
# 6. real attestation
# --------------------------------------------------------------------------- #
def test_added_links_never_upgrade_a_tier(host):
    from habitus.questions.habit import attest, profile2
    d = TEST[1]
    prof = profile2.get_profile2(host, list(CLEAN[:2]))
    before = attest.lineage_tiers(host, d, prof)
    nodes = sorted(before)
    procs = [n for n in nodes if host.card(host.nodes(d)[n])["type"] == "process"]
    rng = np.random.default_rng(11)
    base_links = host.links(d)
    for trial in range(10):
        extra = []
        for _ in range(int(rng.integers(1, 6))):
            c = int(rng.choice(procs))
            e = int(rng.choice(nodes))
            code = {"process": 1, "file": 2, "netflow": 4}[host.card(host.nodes(d)[e])["type"]]
            if c != e:
                extra.append([c, e, code])
        rows = sorted({tuple(r) for r in base_links.tolist()} | {tuple(r) for r in extra})
        days = {x: host.day(x) for x in CLEAN + TEST}
        links = {x: host.links(x).tolist() for x in CLEAN + TEST}
        links[d] = [list(r) for r in rows]
        h2 = FakeHabitHost(days, links, art=f"/nonexistent/habitus_fake_links{trial}/")
        prof2 = profile2.get_profile2(h2, list(CLEAN[:2]))
        after = attest.lineage_tiers(h2, d, prof2)
        assert all(int(after[n]) <= int(before[n]) for n in nodes), extra       # never upgraded
        for n in nodes:                                                         # tier0 is link-free:
            if before[n] != T.Attest.LINEAGE:                                   # only LINEAGE can move,
                assert after[n] == before[n], (n, extra)
            else:                                                               # and only to IDENTITY
                assert after[n] in (T.Attest.LINEAGE, T.Attest.IDENTITY), (n, extra)
        ref, h, routes, _mem = _day_parts(host, d, CLEAN[:2])
        sb = {r.node: r.s for r in score.rows_for_day(host, ref, h, routes, before)}
        sa = {r.node: r.s for r in score.rows_for_day(host, ref, h, routes, after)}
        assert all(sa[n] >= sb[n] for n in sb)


# --------------------------------------------------------------------------- #
# 7. label hygiene and the guard
# --------------------------------------------------------------------------- #
def test_hygiene_grep_on_the_decision_modules():
    hb = os.path.join(REPO, "src", "habitus", "decide", "habit")
    for f in MINE:
        lines = [ln for ln in open(os.path.join(hb, f), encoding="utf-8").read().splitlines() if HYGIENE.search(ln)]
        assert lines == [], f


def test_decision_modules_import_under_the_guard():
    code = (
        "import sys\n"
        "from habitus import guard\n"
        "guard.install_guard('decide')\n"
        "from habitus.decide.habit import score, bars, retrofit, channel, null2, surprise\n"
        "bad = [m for m in sys.modules if guard.blocked_import(m)]\n"
        "print('OK' if not bad else bad)\n")
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "src"))
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and r.stdout.strip().endswith("OK"), r.stderr[-2000:]


# --------------------------------------------------------------------------- #
# 8. real E2 files (asset: the release scores, or this server's PIDSMaker score dir)
# --------------------------------------------------------------------------- #
def real_e2_root():
    from habitus.config import settings
    s = settings()
    for d in (s.paths.scores, os.path.join(s.paths.pidsmaker_artifacts, "reflex", "kcs1", "scores")
              if s.paths.pidsmaker_artifacts else None):
        if d and glob.glob(os.path.join(d, "CADETS_E3", "E2_k2", "*.json.gz")):
            return d
    return None


def test_real_e2_records_are_the_lineage_special_case():
    """On the frozen E2 files, the HB0 composite with every node LINEAGE reproduces E2's stored
    composite exactly (no host / card needed), and the C_K split stays within [0.75 base, base]."""
    root = real_e2_root()
    if root is None:
        missing_asset("routed E2 score files (release scores)")
    for ds in ("CADETS_E3", "THEIA_E3", "ATLASV2_EDR"):
        for p in sorted(glob.glob(os.path.join(root, ds, "E2_k2", "*.json.gz")))[:2]:
            day = os.path.basename(p)[:-len(".json.gz")]
            raw = retrofit.e2_day(ds, "k2", day, root)
            routes = retrofit.e2_routes(raw)
            for s, r in raw["scores"].items():
                base = score.base_of(r["qv"], r["qt"])
                assert math.isclose(score.composite(base, r["qa"], r["hard"], T.Attest.LINEAGE)[0],
                                    r["composite"], abs_tol=1e-12)
                s2, _ = score.composite(base, routes[s]["qa"], routes[s]["hard"], T.Attest.IDENTITY)
                assert 0.75 * base - 1e-12 <= s2 <= base
