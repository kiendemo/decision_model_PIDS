"""decide.routed: per-day E2 / E1 score files from DecisionModel answers.

Ports test_kcs1.test_score_config_e2_end_to_end (E2 on the synthetic host with a fake DM),
test_score_config_end_to_end (E1), test_score_meta_engine_path_and_skip_ignores_it, and adds the
refusal / cache-miss rules and the asset replay: CLEARSCOPE_E3 E2 k=all and L4 k=2 on 2018-04-11
from the shipped answer cache (cache-only) equal the shipped score files exactly."""

import hashlib
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest

from habitus.decide import inventory as I
from habitus.decide import routed as R
from habitus.decide import scorefile as SF
from habitus.decide.composite import FIELDS_E2, compose_e2
from habitus.dm.base import CacheMiss, Decision, Refused
from habitus.state.select import CAPS, flat, limit_caps

REQUIRE = os.environ.get("HABITUS_REQUIRE_ASSETS", "") not in ("", "0")
E1_LEVELS = {"k0": ("L0", "L1", "L3"), "k2": ("L0", "L1", "L2", "L3", "L4"), "kall": ("L0", "L1", "L2", "L3", "L4")}


def _skip_or_fail(msg):
    if REQUIRE:
        pytest.fail(f"HABITUS_REQUIRE_ASSETS: {msg}")
    pytest.skip(msg)


class FakeDM:
    """A DecisionModel whose canonical answer to a question is a hash of (family, state): the
    canonical option 0 gets x, the others share 1 - x."""
    family, layout, tag = "jev", "letter", "fake"

    def __init__(self, refuse=(), miss=False):
        self.calls, self.refuse, self.miss = [], set(refuse), miss

    def spec(self):
        return {"engine": self.tag, "family": self.family, "layout": self.layout}

    def refused(self, qs):
        return {q.qkey: "over budget" for q in qs if q.family in self.refuse}

    def decide(self, qs):
        if self.miss:
            raise CacheMiss([q.qkey for q in qs])
        self.calls.append([q.family for q in qs])
        out = {}
        for q in qs:
            assert q.qkey not in out
            x = int(hashlib.sha1(f"{q.family}|{q.forms[0].state}".encode()).hexdigest()[:8], 16) / 16 ** 8
            lp, cp, cov = {}, {}, {}
            for f in q.forms:
                n = len(f.options)
                canon = (x,) + ((1.0 - x) / (n - 1),) * (n - 1)
                lp[f.name] = (tuple(canon[f.canon[i]] for i in range(n)),)
                cp[f.name] = (canon,)
                cov[f.name] = 0.99
            out[q.qkey] = Decision(q.qkey, self.layout, tuple(lp), lp, cp, cov, 0.99)
        return out

    def close(self):
        pass


class PSDM(FakeDM):
    def spec(self):
        return {**super().spec(), "prefix_share": True, "batch_size": 8, "grouping": "direct"}


def _settings(tmp_path):
    return SimpleNamespace(paths=SimpleNamespace(data=str(tmp_path / "release"), scores=str(tmp_path / "release" / "scores"),
                                                 runs=str(tmp_path / "runs")))


@pytest.fixture(scope="module")
def host():
    from fakes import make_kcs1_host
    from state_helpers import clear_caches
    clear_caches()
    return make_kcs1_host()


def _spec(k0_nulls=False):
    return SimpleNamespace(k0_nulls=k0_nulls, e1_levels=E1_LEVELS)


def _job(host, level, k, day, ref=None, role="test"):
    return I.ScoreJob(host.ds, level, k, day, tuple(R.ref_for(host, k) if ref is None else ref), role)


def test_ref_for_and_check_ks(host):
    assert R.ref_for(host, 0) == [] and R.ref_for(host, 2) == ["c1", "c2"] and R.ref_for(host, None) == ["c1", "c2"]
    assert R.check_ks("0,2,all") == [0, 2, None] and R.check_ks("all") == [None]
    for bad in ("1", "0,3", "x", ""):
        with pytest.raises(ValueError):
            R.check_ks(bad)


def test_score_e2_end_to_end(host, tmp_path):
    st, root, dm = _settings(tmp_path), str(tmp_path / "runs" / "scores"), FakeDM()
    js = [_job(host, "E2", k, d) for d in host.test for k in (0, 2)]
    assert R.run(host.ds, dm=dm, root=root, jobs=js, host=host, settings=st) == {"done": 4, "skip": 0}
    assert R.run(host.ds, dm=dm, root=root, jobs=js[:1], host=host, settings=st) == {"done": 0, "skip": 1}
    with pytest.raises(ValueError):                                       # same file, another reference
        R.run(host.ds, dm=dm, root=root, jobs=[_job(host, "E2", 2, host.test[0], ref=["c1"])], host=host, settings=st)
    for k in (0, 2):
        raw = SF.read_score(SF.score_path(host.ds, "E2", k, "t2", root))
        m = raw["meta"]
        assert m["level"] == "E2" and m["e2"]["lambda"] == 0.5 and m["e2"]["spec"] == "docs/DESIGN.md"
        assert list(m) == ["ds", "day", "level", "k", "ref", "role", "caps", "n_head", "n_states", "n_unique_states",
                           "n_qa", "n_hard", "e2", "seconds", "engine", "prefix_share", "batch_size", "grouping",
                           "low_cov", "written"]
        assert m["engine"] == "fake" and (m["prefix_share"], m["batch_size"], m["grouping"]) == (None, None, None)
        assert m["ref"] == ([] if k == 0 else ["c1", "c2"]) and m["caps"] == CAPS and m["low_cov"] == 0
        assert set(raw["scores"]) == set(flat(raw["head"])) and not set(raw["scores"]) & set(raw["fallback"])
        assert m["n_states"] == len(raw["scores"]) >= m["n_unique_states"]
        for s, r in raw["scores"].items():
            assert list(r) == ["qv", "qt", "qa", "hard", "cov_min", "base", "composite"]
            assert r["base"] == max(r["qv"], r["qt"]) and r["composite"] == pytest.approx(compose_e2(r))
            assert r["cov_min"] == 0.99
            if k == 0:
                assert r["qa"] is None and r["composite"] == r["base"]
            if r["hard"]:
                assert r["composite"] == r["base"]
        assert m["n_hard"] == (2 if k == 0 else 0)          # x9q + payload.sh (exec n/a at k = 2)
        if k == 2:
            assert m["n_qa"] > 0
        by_day, missing = SF.load_by_day(host.ds, "E2", k, host.test, root)
        assert not missing
        for f in FIELDS_E2:
            s = SF.node_scores(host, by_day, f)
            assert np.isfinite(s).all() and s.shape == host.universe.shape
    # E2 jobs: k = 0 nulls only where k0_nulls; every job is an E2 file
    assert {j.k for j in I.jobs(host, levels=("E2",), ks=(0, 2), spec=_spec(False)) if j.role == "null"} == {2}
    assert {j.k for j in I.jobs(host, levels=("E2",), ks=(0, 2), spec=_spec(True)) if j.role == "null"} == {0, 2}


def test_score_e2_dedupes_by_the_join_key(host):
    dm = FakeDM()
    obj = R.score_job(host, dm, _job(host, "E2", 2, "t2"))
    fams = dm.calls[-1]
    m = obj["meta"]
    assert len(dm.calls) == 1 and fams.count("qv") == fams.count("qt") == m["n_unique_states"]
    assert 0 < fams.count("qa") <= m["n_unique_states"]
    from habitus.state.builder import StateBuilderE2, serialize, state_hash
    states = StateBuilderE2(host, ["c1", "c2"], 2).build("t2", flat(obj["head"]))
    keys = {state_hash("\x1e".join(f"{q}\x1f{serialize(x[q])}" for q in sorted(x))) for x in states.values()}
    assert len(keys) == m["n_unique_states"]
    # signatures sharing a routed state share their record
    by_key = {}
    for s, x in states.items():
        hk = state_hash("\x1e".join(f"{q}\x1f{serialize(x[q])}" for q in sorted(x)))
        by_key.setdefault(hk, []).append(obj["scores"][s])
    assert all(all(r == rs[0] for r in rs) for rs in by_key.values())


def test_score_e1_end_to_end(host, tmp_path):
    st, root, dm = _settings(tmp_path), str(tmp_path / "runs" / "scores"), FakeDM()
    js = I.jobs(host, levels=I.LEVELS[1:], ks=(0, 2), nulls=False, spec=_spec())
    assert {(j.level, j.k) for j in js} == {(lv, 0) for lv in ("L0", "L1", "L3")} | {(lv, 2) for lv in R.LEVELS}
    assert R.run(host.ds, dm=dm, root=root, jobs=js, host=host, settings=st)["done"] == len(js)
    assert R.run(host.ds, dm=dm, root=root, jobs=[_job(host, "L4", 2, host.test[0])], host=host,
                 settings=st) == {"done": 0, "skip": 1}
    by_day, missing = SF.load_by_day(host.ds, "L4", 2, host.test, root)
    assert not missing
    raw = SF.read_score(SF.score_path(host.ds, "L4", 2, "t2", root))
    assert set(raw) == {"meta", "head", "scores", "fallback"} and raw["meta"]["role"] == "test"
    assert set(raw["scores"]) == set(flat(raw["head"])) and not set(raw["scores"]) & set(raw["fallback"])
    assert all(list(r) == ["qv", "qt", "cov_min"] for r in raw["scores"].values())       # E1: no extra keys
    assert not {"n_qa", "n_hard", "e2"} & set(raw["meta"])
    s = SF.node_scores(host, by_day)
    assert np.isfinite(s).all() and s.shape == host.universe.shape
    assert not os.path.exists(SF.score_path(host.ds, "L2", 0, "t1", root))      # k = 0: L0, L1, L3 only
    path = SF.score_path(host.ds, "L4", 2, host.test[0], root)
    assert SF.read_meta(path) == SF.read_score(path)["meta"]
    for caps, ref in ((limit_caps(5), ["c1", "c2"]), (CAPS, ["c1"])):            # resumable only over the same
        with pytest.raises(ValueError):                                          # head rule and reference
            R.run(host.ds, dm=dm, root=root, jobs=[_job(host, "L4", 2, host.test[0], ref=ref)], host=host,
                  caps=caps, settings=st)
    # E1 dedupe: one qv + one qt question per distinct serialised state
    from habitus.state.builder import serialize, state_hash
    from habitus.state.ladder import StateBuilder
    dm2 = FakeDM()
    obj = R.score_job(host, dm2, _job(host, "L4", 2, "t2"))
    states = StateBuilder(host, ["c1", "c2"], "L4", 2).build("t2", flat(obj["head"]))
    n_u = len({state_hash(serialize(x)) for x in states.values()})
    assert obj["meta"]["n_unique_states"] == n_u and sorted(dm2.calls[0]) == ["qt"] * n_u + ["qv"] * n_u


def test_null_jobs_and_lodo_scores_round_trip(host, tmp_path):
    from habitus.decide import null as N
    st, root, dm = _settings(tmp_path), str(tmp_path / "runs" / "scores"), FakeDM()
    js = [j for j in I.jobs(host, levels=("L4", "E2"), ks=(2,), spec=_spec()) if j.role == "null"]
    assert [(j.day, j.ref) for j in js if j.level == "L4"] == [("c1", ("c2",)), ("c2", ("c1",))]
    assert R.run(host.ds, dm=dm, root=root, jobs=js, host=host, settings=st)["done"] == 4
    for lv in ("L4", "E2"):
        n = N.lodo_scores(host, 2, lv, "composite", root)
        assert n["n_days"] == 2 and n["caps"] == [CAPS, CAPS] and len(n["values"]) > 0
        assert np.isfinite(n["values"]).all()


def test_score_meta_engine_path_and_skip_ignores_it(host, tmp_path):
    st, root = _settings(tmp_path), str(tmp_path / "runs" / "scores")
    assert R.run(host.ds, dm=PSDM(), root=root, jobs=[_job(host, "L4", 2, "t2"), _job(host, "E2", 2, "t2")],
                 host=host, settings=st)["done"] == 2
    assert R.run(host.ds, dm=FakeDM(), root=root, jobs=[_job(host, "L4", 2, "t1")], host=host, settings=st)["done"] == 1
    m1, m2 = (SF.read_meta(SF.score_path(host.ds, lv, 2, "t2", root)) for lv in ("L4", "E2"))
    assert (m1["prefix_share"], m1["batch_size"], m1["grouping"]) == (True, 8, "state")
    assert (m2["prefix_share"], m2["batch_size"], m2["grouping"]) == (True, 8, "qvqt")
    m0 = SF.read_meta(SF.score_path(host.ds, "L4", 2, "t1", root))
    assert (m0["prefix_share"], m0["batch_size"], m0["grouping"]) == (None, None, None)
    p = SF.score_path(host.ds, "L4", 2, "t2", root)                  # an old file without the keys
    raw = SF.read_score(p)
    for key in ("prefix_share", "batch_size", "grouping"):
        raw["meta"].pop(key)
    SF.write_score(raw, p, st)
    assert R.run(host.ds, dm=FakeDM(), root=root, jobs=[_job(host, "L4", 2, "t2")], host=host,
                 settings=st) == {"done": 0, "skip": 1}
    # a chunked grouping is recorded as such, and no prefix share means no grouping
    dm = PSDM()
    dm.spec = lambda: {"prefix_share": True, "batch_size": 32, "grouping": "chunked"}
    assert R.engine_path(dm, "E2") == {"prefix_share": True, "batch_size": 32, "grouping": "chunked"}
    dm.spec = lambda: {"prefix_share": False, "batch_size": 32, "grouping": "direct"}
    assert R.engine_path(dm, "L4") == {"prefix_share": False, "batch_size": 32, "grouping": None}


def test_refusal_and_cache_miss_raise_and_write_nothing(host, tmp_path):
    st, root = _settings(tmp_path), str(tmp_path / "runs" / "scores")
    with pytest.raises(Refused) as e:
        R.run(host.ds, dm=FakeDM(refuse=("qa",)), root=root, jobs=[_job(host, "E2", 2, "t2")], host=host, settings=st)
    assert e.value.reasons and all(k.endswith("|qa") for k in e.value.reasons)
    with pytest.raises(CacheMiss):
        R.run(host.ds, dm=FakeDM(miss=True), root=root, jobs=[_job(host, "L3", 0, "t2")], host=host, settings=st)
    assert not os.path.exists(os.path.join(root, host.ds))
    # k = 0 asks no QA, so a QA-refusing model is fine there
    assert R.run(host.ds, dm=FakeDM(refuse=("qa",)), root=root, jobs=[_job(host, "E2", 0, "t2")], host=host,
                 settings=st)["done"] == 1


def test_run_refuses_the_release_dir_and_bad_jobs(host, tmp_path):
    st = _settings(tmp_path)
    with pytest.raises(PermissionError):
        R.run(host.ds, dm=FakeDM(), root=st.paths.scores, jobs=[_job(host, "E2", 2, "t2")], host=host, settings=st)
    assert not os.path.exists(st.paths.scores)
    with pytest.raises(ValueError):
        R.score_job(host, FakeDM(), I.ScoreJob("OTHER", "E2", 2, "t2", ("c1", "c2"), "test"))
    with pytest.raises(ValueError):
        R.score_job(host, FakeDM(), _job(host, "L5", 2, "t2"))
    with pytest.raises(ValueError):
        R.score_job(host, FakeDM(), _job(host, "E2", 1, "t2", ref=["c1"]))


# --------------------------------------------------------------------------- #
# Asset replay: the shipped answer cache reproduces the shipped score files exactly
# --------------------------------------------------------------------------- #
IGNORED_META = {"seconds", "written"}
E2_FIELDS = ("qv", "qt", "qa", "hard", "cov_min", "base", "composite")
E1_FIELDS = ("qv", "qt", "cov_min")


def compare_score_files(new, old):
    """(max |delta| over the record fields, list of problems) of a recomputed vs a shipped file."""
    bad, worst = [], 0.0
    if new["head"] != old["head"]:
        bad.append("head differs")
    if new["fallback"] != old["fallback"]:
        bad.append("fallback differs")
    e2 = old["meta"].get("level") == "E2"
    fields = E2_FIELDS if e2 else E1_FIELDS
    if set(new["scores"]) != set(old["scores"]):
        bad.append("scored signatures differ")
    for s, r in old["scores"].items():
        n = new["scores"].get(s)
        if n is None:
            continue
        if tuple(n) != fields:
            bad.append(f"{s}: keys {list(n)}")
        for f in fields:
            a, b = n.get(f), r.get(f)
            if isinstance(b, bool) or b is None or isinstance(a, bool) or a is None:
                if a != b:
                    bad.append(f"{s}.{f}: {a!r} != {b!r}")
            else:
                d = abs(float(a) - float(b))
                worst = max(worst, d) if math.isfinite(d) else math.inf
    for key, v in old["meta"].items():
        if key in IGNORED_META:
            continue
        w = new["meta"].get(key)
        if key == "e2":
            v, w = {x: y for x, y in v.items() if x != "spec"}, {x: y for x, y in (w or {}).items() if x != "spec"}
        if w != v:
            bad.append(f"meta {key}: {w!r} != {v!r}")
    return worst, bad


@pytest.fixture(scope="module")
def clearscope():
    from habitus.config import settings
    s = settings()
    ds = "CLEARSCOPE_E3"
    need = {"scores": os.path.join(s.paths.scores, ds), "cards": os.path.join(s.paths.cards, ds),
            "answers": s.paths.answers, "precedent": s.paths.precedent}
    missing = [k for k, p in need.items() if not os.path.exists(p)]
    if missing:
        _skip_or_fail(f"release assets missing: {missing}")
    from habitus.data.host import Host
    return Host(ds)


@pytest.mark.parametrize("level,k", [("E2", None), ("L4", 2)])
def test_asset_replay_clearscope_matches_shipped(clearscope, level, k, tmp_path):
    from habitus.dm import get_dm
    ds, day = "CLEARSCOPE_E3", "2018-04-11"
    job = next(j for j in I.inventory(ds) if (j.level, j.k, j.day) == (level, k, day))
    assert job.role == "test" and list(job.ref) == R.ref_for(clearscope, k)
    dm = get_dm("jev-qwen3.5-4b", ds=ds, cache_only=True)
    try:
        root = str(tmp_path / "scores")
        assert R.run(ds, dm=dm, root=root, jobs=[job], host=clearscope) == {"done": 1, "skip": 0}
    finally:
        dm.close()
    new = SF.read_score(SF.score_path(ds, level, k, day, root))
    old = SF.read_score(SF.score_path(ds, level, k, day))
    worst, bad = compare_score_files(new, old)
    assert bad == [] and worst == 0.0
