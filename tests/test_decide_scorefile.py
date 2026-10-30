"""decide.scorefile: score-file paths / IO / meta, and the head + fallback + node-score rule
(port of test_kcs1.test_head_caps_fallback_and_node_scores)."""

import gzip
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from habitus.decide import scorefile as SF
from habitus.decide.composite import compose

FIELDS = ("composite", "qv", "qt")                        # the E1 fields (kcs1.score.FIELDS)
CAPS = {"process": 4000, "file": 2000, "netflow": 6000}


def _settings(tmp_path):
    return SimpleNamespace(paths=SimpleNamespace(data=str(tmp_path / "release"), scores=str(tmp_path / "release" / "scores"),
                                                 runs=str(tmp_path / "runs")))


def _obj(meta_extra=None):
    meta = {"ds": "KCS1T", "day": "t1", "level": "E2", "k": "2", "ref": ["c1", "c2"], "role": "test", "caps": CAPS}
    meta.update(meta_extra or {})
    return {"meta": meta, "head": {"process": ["a" * 16], "file": [], "netflow": []},
            "scores": {"a" * 16: {"qv": 0.25, "qt": 0.5, "qa": None, "hard": False, "cov_min": 0.9,
                                  "base": 0.5, "composite": 0.5}},
            "fallback": {"b" * 16: -0.75}}


def test_paths_and_k_labels(tmp_path):
    assert SF.klabel(None) == "all" and SF.klabel(0) == "0" and SF.klabel(2) == "2"
    assert SF.parse_k("all,2,0") == [None, 2, 0] and SF.parse_k("") == []
    with pytest.raises(ValueError):
        SF.parse_k("x")
    assert SF.score_path("CADETS_E3", "E2", None, "2018-04-06", "R") == os.path.join("R", "CADETS_E3", "E2_kall",
                                                                                      "2018-04-06.json.gz")
    assert SF.score_path("CADETS_E3", "L3", 0, "d", tmp_path).endswith(os.path.join("L3_k0", "d.json.gz"))


def test_score_file_round_trip_meta_first_and_deterministic(tmp_path):
    st = _settings(tmp_path)
    p = SF.score_path("KCS1T", "E2", 2, "t1", str(tmp_path / "runs" / "scores"))
    obj = _obj()
    assert SF.write_score(obj, p, st) == p
    assert SF.read_score(p) == obj and SF.read_meta(p) == obj["meta"]
    with gzip.open(p, "rt", encoding="utf-8") as f:
        assert f.read(8) == '{"meta":'                                   # read_meta relies on it
    b1 = open(p, "rb").read()
    SF.write_score(obj, p, st)
    assert open(p, "rb").read() == b1                                    # gzip mtime 0: same bytes
    assert not [x for x in os.listdir(os.path.dirname(p)) if ".tmp" in x]
    assert SF.read_json_gz is SF.read_score and SF.write_json_gz is SF.write_score


def test_read_meta_large_and_other_layouts(tmp_path):
    st = _settings(tmp_path)
    root = str(tmp_path / "runs" / "scores")
    big = _obj({"pad": "x" * (SF.META_BYTES + 10)})                      # meta beyond the first 64 KiB
    p = SF.write_score(big, SF.score_path("KCS1T", "E2", 2, "t1", root), st)
    assert SF.read_meta(p) == big["meta"]
    q = SF.score_path("KCS1T", "E2", 2, "t2", root)                      # meta not first: full read
    with gzip.open(q, "wt", encoding="utf-8") as f:
        json.dump({"scores": {}, "meta": {"k": "2"}}, f)
    assert SF.read_meta(q) == {"k": "2"}


def test_write_refuses_release_dir(tmp_path):
    st = _settings(tmp_path)
    for root in (st.paths.scores, st.paths.data, os.path.join(st.paths.data, "elsewhere")):
        with pytest.raises(PermissionError):
            SF.write_score(_obj(), SF.score_path("KCS1T", "E2", 2, "t1", root), st)
    assert not os.path.exists(st.paths.data)
    SF.check_writable(os.path.join(str(tmp_path), "release2", "x"), st)   # a sibling name is fine


def test_load_by_day_missing_and_metas(tmp_path):
    st = _settings(tmp_path)
    root = str(tmp_path / "runs" / "scores")
    SF.write_score(_obj(), SF.score_path("KCS1T", "E2", 2, "t1", root), st)
    metas = {}
    by_day, missing = SF.load_by_day("KCS1T", "E2", 2, ["t1", "t2"], root, metas)
    assert set(by_day) == {"t1"} and missing == ["t2"] and metas["t1"]["caps"] == CAPS
    assert by_day["t1"] == _obj()["scores"]


class _Host:
    """Two test days over a 6-node universe; node 5 appears on t2 only."""
    ds, test = "KCS1T", ["t1", "t2"]
    universe = np.array([1, 2, 3, 4, 5, 7], dtype=np.int64)
    _n = {"t1": {1: "a" * 16, 2: "a" * 16, 3: "b" * 16, 4: "c" * 16, 7: "c" * 16},
          "t2": {1: "b" * 16, 3: "b" * 16, 5: "d" * 16, 7: "a" * 16, 9: "a" * 16}}    # 9: outside the universe

    def nodes(self, d):
        return self._n[d]


def test_node_scores_hand_computed():
    host = _Host()
    a, b, c, d = ("a" * 16, "b" * 16, "c" * 16, "d" * 16)
    by_day = {"t1": {a: {"qv": 0.2, "qt": 0.6, "cov_min": 1.0}},
              "t2": {b: {"qv": 0.1, "qt": 0.3, "qa": 0.5, "hard": False, "cov_min": 1.0}, d: 0.05}}
    s = SF.node_scores(host, by_day)
    fb = lambda n: -1.0 + 0.5 * (1.0 / (1.0 + n))                       # noqa: E731
    e2 = 0.3 * (1.0 - 0.5 * 0.5)                                        # max(qv, qt) * (1 - LAMBDA * qa)
    # node 1: t1 a (0.6), t2 b (e2); node 2: t1 a; node 3: t1 b fallback(1), t2 b;
    # node 4: t1 c fallback(2); node 5: t2 d = 0.05; node 7: t1 c fallback(2), t2 a fallback(2)
    assert s.tolist() == [0.6, 0.6, e2, fb(2), 0.05, fb(2)]
    assert SF.node_scores(host, by_day, "qv").tolist() == [0.2, 0.2, 0.1, fb(2), 0.05, fb(2)]
    with pytest.raises(ValueError):                                     # node 5 is not on t1
        SF.node_scores(host, by_day, "composite", days=["t1"])
    assert SF.head_mask(host, by_day).tolist() == [True, True, True, False, True, False]
    ids, v = SF.day_values(host, "t2", by_day["t2"])
    assert ids.tolist() == [1, 3, 5, 7, 9] and v.tolist() == [e2, e2, 0.05, fb(2), fb(2)]
    assert SF.upos(host.universe, np.array([7, 9, 1, 0])).tolist() == [5, -1, 0, -1]
    with pytest.raises(KeyError):                                       # a test day without a file
        SF.node_scores(host, {"t1": by_day["t1"]})
    with pytest.raises(ValueError):                                     # out of [0, 1]
        SF.node_scores(host, {"t1": {a: 1.5}, "t2": {}})
    with pytest.raises(ValueError):
        SF.node_scores(host, {"t1": {a: {"qv": float("nan"), "qt": 0.1}}, "t2": {}}, "qv")


# --------------------------------------------------------------------------- #
# Port of test_kcs1.py:402 on the synthetic KC-S1 host
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def host():
    from fakes import make_kcs1_host
    from state_helpers import clear_caches
    clear_caches()
    return make_kcs1_host()


def test_head_caps_fallback_and_node_scores(host):
    from habitus.state import profile as P
    from habitus.state import select as SEL
    caps = {"process": 3, "file": 2, "netflow": 4}
    prof = P.get_profile(host, ["c1", "c2"])
    for ref in (None, ["c1", "c2"], prof):
        for d in host.test:
            h = SEL.head(host, d, ref, caps)
            assert all(len(h[t]) <= caps[t] for t in SEL.TYPES)
            assert all(host.ntype(s) == t for t in SEL.TYPES for s in h[t])
            n = SEL.day_counts(host, d)
            full = SEL.head(host, d, ref)
            assert full["process"] == sorted(full["process"], key=lambda s: (n[s], s))
            assert h["process"] == full["process"][:3] and h["file"] == full["file"][:2]
            if ref is not None:                                   # unseen kappa first
                seen = SEL.ref_kappas(host, ref)
                for t in ("file", "netflow"):
                    flags = [host.inv_key(s) in seen for s in full[t]]
                    assert flags == sorted(flags)
    h = SEL.head(host, "t2", prof)
    pay = next(s for s in h["file"] if host.card(s)["invariant"]["path"].endswith("/payload.sh"))
    assert h["file"][0] == pay                                     # unseen + changed on disk
    assert next(s for s in h["netflow"] if host.card(s)["invariant"]["remote_ip"] == "6.6.6.6") == h["netflow"][0]
    # fallback strictly below every engine score
    rng = np.random.default_rng(1)
    by_day, heads = {}, {}
    for d in host.test:
        heads[d] = SEL.flat(SEL.head(host, d, prof, caps))
        by_day[d] = {s: {"qv": float(rng.random()), "qt": float(rng.random()), "cov_min": 1.0} for s in heads[d]}
        fb = SEL.fallback_rank(host, d)
        assert all(-1.0 < v < -0.5 + 1e-12 for v in fb.values())
        assert fb == {s: -1 + 0.5 / (1 + c) for s, c in SEL.day_counts(host, d).items()}
    for f in FIELDS:
        s = SF.node_scores(host, by_day, f)
        assert s.shape == host.universe.shape and np.isfinite(s).all()
        m = SF.head_mask(host, by_day)
        assert m.any() and (~m).any()
        assert (s[m] >= 0).all() and (s[~m] < 0).all() and s[~m].max() < s[m].min()
    # node score = max over the test days of its node-day scores
    s = SF.node_scores(host, by_day, "composite")
    for i, nid in enumerate(host.universe.tolist()):
        vals = []
        for d in host.test:
            sig = host.nodes(d).get(nid)
            if sig is not None:
                r = by_day[d].get(sig)
                vals.append(compose(r) if r else SEL.fallback_value(SEL.day_counts(host, d)[sig]))
        assert s[i] == pytest.approx(max(vals))
    with pytest.raises(KeyError):
        SF.node_scores(host, {host.test[0]: by_day[host.test[0]]})
    bad = {d: dict(r) for d, r in by_day.items()}
    bad[host.test[0]][heads[host.test[0]][0]] = {"qv": float("nan"), "qt": 0.1}
    with pytest.raises(ValueError):
        SF.node_scores(host, bad, "qv")
