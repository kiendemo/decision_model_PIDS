"""decide.inventory (G4a): the null rules + HostSpec.e1_levels reproduce the shipped score-file
inventory, and inventory() checks every shipped meta's reference against lodo_plan / ref_for.

tests/golden/inventory.json was taken once from the metas (first 64 KiB) of the shipped KC-S1
score dirs of the 5 hosts: (dir, day, role, ref, caps) per file, no score and no label. Its layout
is {"version": 1, "source": ..., "hosts": {ds: inventory.scan(ds, <shipped scores root>)}}
(json.dump indent=1)."""

import json
import os
from collections import Counter
from types import SimpleNamespace

import pytest

from habitus.decide import inventory as I
from habitus.decide import scorefile as SF

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "golden", "inventory.json")
HOSTS = ("CADETS_E3", "THEIA_E3", "CLEARSCOPE_E3", "optc_h501", "ATLASV2_EDR")
CAPS = {"process": 4000, "file": 2000, "netflow": 6000}
REQUIRE = os.environ.get("HABITUS_REQUIRE_ASSETS", "") not in ("", "0")

# G4a file counts per dir: (files, of which null)
COUNTS = {
    "CADETS_E3": {"E2_k0": (12, 8), "E2_k2": (6, 2), "E2_kall": (12, 8), "L2_k2": (6, 2), "L2_kall": (12, 8),
                  "L3_k0": (12, 8), "L4_k2": (6, 2), "L4_kall": (12, 8), "L0_k0": (4, 0), "L0_k2": (4, 0),
                  "L0_kall": (4, 0), "L1_k0": (4, 0), "L1_k2": (4, 0), "L1_kall": (4, 0), "L3_k2": (4, 0),
                  "L3_kall": (4, 0)},
    "CLEARSCOPE_E3": {"E2_k0": (2, 0), "E2_k2": (4, 2), "E2_kall": (9, 7), "L2_k2": (4, 2), "L2_kall": (9, 7),
                      "L3_k0": (2, 0), "L4_k2": (4, 2), "L4_kall": (9, 7)},
}
for _ds, _k0, _k2, _ka, _t in (("THEIA_E3", (7, 4), (5, 2), (7, 4), 3), ("optc_h501", (7, 4), (5, 2), (7, 4), 3),
                               ("ATLASV2_EDR", (4, 3), (3, 2), (4, 3), 1)):
    COUNTS[_ds] = {"E2_k0": _k0, "E2_k2": _k2, "E2_kall": _ka, "L2_k2": _k2, "L2_kall": _ka, "L3_k0": _k0,
                   "L4_k2": _k2, "L4_kall": _ka, **{d: (_t, 0) for d in ("L0_k0", "L0_k2", "L0_kall", "L1_k0",
                                                                           "L1_k2", "L1_kall", "L3_k2", "L3_kall")}}


def _skip_or_fail(msg):
    if REQUIRE:
        pytest.fail(f"HABITUS_REQUIRE_ASSETS: {msg}")
    pytest.skip(msg)


def golden():
    with open(GOLDEN, encoding="utf-8") as f:
        return json.load(f)


def _no_caps(rows):
    return [{k: v for k, v in r.items() if k != "caps"} for r in rows]


def test_golden_counts_and_caps():
    g = golden()
    assert tuple(g["hosts"]) == HOSTS
    for ds in HOSTS:
        rows = g["hosts"][ds]
        files = Counter(r["dir"] for r in rows)
        nulls = Counter(r["dir"] for r in rows if r["role"] == "null")
        assert {d: (files[d], nulls[d]) for d in files} == COUNTS[ds], ds
        assert all(r["caps"] == CAPS for r in rows)
        assert sum(files.values()) == {"CADETS_E3": 110, "THEIA_E3": 74, "CLEARSCOPE_E3": 43, "optc_h501": 74,
                                       "ATLASV2_EDR": 37}[ds]


@pytest.mark.parametrize("ds", HOSTS)
def test_jobs_reproduce_the_shipped_inventory(ds):
    """G4a, rule side: jobs() from NULL_RULES + HostSpec over the frozen day lists == the golden."""
    js = I.jobs(ds)
    assert len(set(js)) == len(js)
    assert I.rows(js) == _no_caps(golden()["hosts"][ds])
    host = I.spec_host(ds)
    assert {j for j in js} == set(I.jobs(host, levels=I.LEVELS, ks=I.KS, nulls=True))
    tests_only = I.jobs(host, nulls=False)
    assert tests_only and all(j.role == "test" for j in tests_only)
    assert set(tests_only) == {j for j in js if j.role == "test"}
    e2 = I.jobs(host, levels=("E2",), ks=(None, 2, 0))
    assert e2 == [j for j in js if j.level == "E2"]


@pytest.mark.parametrize("ds", HOSTS)
def test_inventory_of_the_release_equals_the_golden(ds):
    """G4a, file side: inventory(ds, release scores) == jobs() == the golden, with every null ref
    == lodo_plan and every test ref == ref_for (inventory asserts both)."""
    root = SF.default_root()
    if not os.path.isdir(os.path.join(root, ds)):
        _skip_or_fail(f"no release score dir for {ds} under {root}")
    rows = I.scan(ds, root)
    assert rows == golden()["hosts"][ds]
    inv = I.inventory(ds, root, caps=CAPS)
    assert set(inv) == set(I.jobs(ds)) and inv == I.jobs(ds)


def test_null_rules_constant():
    assert I.NULL_RULES == {"E2": {"k2": "all_hosts", "kall": "all_hosts", "k0": "k0_nulls"},
                            "L2": {"k2": "all_hosts", "kall": "all_hosts"},
                            "L4": {"k2": "all_hosts", "kall": "all_hosts"},
                            "L3": {"k0": "k0_nulls"}}


def _find(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            hit = _find(v, key)
            if hit is not None:
                return hit
    return None


def _norm_rules(r):
    if isinstance(r, list):                       # a list of {level: {k: rule}} blocks
        out = {}
        for block in r:
            out.update(block)
        r = out
    return {str(lv): {str(k): str(v) for k, v in ks.items()} for lv, ks in r.items()}


def test_null_rules_mirror_in_pipeline_yaml():
    """configs/pipeline.yaml mirrors NULL_RULES (the code is the source of truth)."""
    path = os.path.join(os.path.dirname(HERE), "configs", "pipeline.yaml")
    if not os.path.exists(path):
        pytest.skip("configs/pipeline.yaml not written")
    import yaml
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    r = _find(cfg, "NULL_RULES") or _find(cfg, "null_rules")
    if r is None:
        pytest.skip("configs/pipeline.yaml has no NULL_RULES block")
    assert _norm_rules(r) == _norm_rules(I.NULL_RULES)


def test_jobs_rules_on_a_synthetic_spec():
    host = SimpleNamespace(ds="KCS1T", clean=["c1", "c2", "c3"], test=["t1", "t2"],
                           shots=lambda k: ["c1", "c2", "c3"] if k is None else ["c1", "c2", "c3"][:k])
    lv = {"k0": ("L0", "L1", "L3"), "k2": ("L0", "L1", "L2", "L3", "L4"), "kall": ("L0", "L1", "L2", "L3", "L4")}
    for k0_nulls in (True, False):
        spec = SimpleNamespace(k0_nulls=k0_nulls, e1_levels=lv)
        js = I.jobs(host, spec=spec)
        nul = Counter((j.level, j.k) for j in js if j.role == "null")
        want = {("E2", 2): 2, ("E2", None): 3, ("L2", 2): 2, ("L2", None): 3, ("L4", 2): 2, ("L4", None): 3}
        if k0_nulls:
            want.update({("E2", 0): 3, ("L3", 0): 3})
        assert dict(nul) == want
        tst = Counter((j.level, j.k) for j in js if j.role == "test")
        assert set(tst) == {("E2", k) for k in I.KS} | {(x, k) for k in I.KS for x in lv[I.kkey(k)]}
        assert all(n == 2 for n in tst.values())
        for j in js:
            if j.role == "test":
                assert j.ref == (() if j.k == 0 else tuple(host.shots(j.k)))
        assert {j.ref for j in js if j.role == "null" and j.k == 2} == {("c2",), ("c1",)}
        assert js == I.sort_jobs(js)
    # k = 0: E1 has no L2 / L4; a request for them gives no job
    assert I.jobs(host, levels=("L2", "L4"), ks=(0,), spec=SimpleNamespace(k0_nulls=True, e1_levels=lv)) == []
    with pytest.raises(ValueError):
        I.jobs(host, levels=("L5",), spec=SimpleNamespace(k0_nulls=True, e1_levels=lv))
    with pytest.raises(ValueError):
        I.jobs(host, ks=(1,), spec=SimpleNamespace(k0_nulls=True, e1_levels=lv))


def test_score_job_is_hashable_and_frozen():
    j = I.ScoreJob("CADETS_E3", "E2", None, "2018-04-06", ("2018-04-02",), "test")
    assert j.dir == "E2_kall" and j.row() == {"dir": "E2_kall", "day": "2018-04-06", "role": "test",
                                               "ref": ["2018-04-02"]}
    assert {j, I.ScoreJob("CADETS_E3", "E2", None, "2018-04-06", ("2018-04-02",), "test")} == {j}
    with pytest.raises(Exception):
        j.day = "x"


# --------------------------------------------------------------------------- #
# inventory() checks on synthetic metas (a tmp root; real host name, frozen day lists)
# --------------------------------------------------------------------------- #
def _settings(tmp_path):
    return SimpleNamespace(paths=SimpleNamespace(data=str(tmp_path / "release"), scores=str(tmp_path / "release" / "x"),
                                                 runs=str(tmp_path)))


def _put(root, st, ds, level, k, day, ref, role, caps=CAPS):
    meta = {"ds": ds, "day": day, "level": level, "k": SF.klabel(k), "ref": list(ref), "role": role, "caps": caps}
    SF.write_score({"meta": meta, "head": {}, "scores": {}, "fallback": {}},
                   SF.score_path(ds, level, k, day, root), st)


def test_inventory_asserts_references(tmp_path):
    ds = "ATLASV2_EDR"
    st, root = _settings(tmp_path), str(tmp_path / "scores")
    host = I.spec_host(ds)
    for j in I.jobs(host):
        _put(root, st, ds, j.level, j.k, j.day, j.ref, j.role)
    assert I.inventory(ds, root, caps=CAPS) == I.jobs(host)
    assert I.rows(I.inventory(ds, root)) == _no_caps(golden()["hosts"][ds])
    with pytest.raises(ValueError):                                        # caps differ
        I.inventory(ds, root, caps={"process": 1, "file": 1, "netflow": 1})
    null2 = next(j for j in I.jobs(host) if j.role == "null" and j.k == 2)
    _put(root, st, ds, null2.level, 2, null2.day, list(host.clean[:2]), "null")   # not the LODO ref
    with pytest.raises(ValueError, match="ref"):
        I.inventory(ds, root)
    _put(root, st, ds, null2.level, 2, null2.day, null2.ref, "null")
    t = next(j for j in I.jobs(host) if j.role == "test" and j.k is None)
    _put(root, st, ds, t.level, None, t.day, host.clean[:2], "test")              # not ref_for
    with pytest.raises(ValueError, match="ref"):
        I.inventory(ds, root)
    _put(root, st, ds, t.level, None, t.day, t.ref, "test")
    I.inventory(ds, root)
    os.makedirs(os.path.join(root, ds, "L5_k1"))                                    # unknown dir
    with pytest.raises(ValueError, match="unexpected"):
        I.scan(ds, root)
    os.rmdir(os.path.join(root, ds, "L5_k1"))
    p = SF.score_path(ds, "E2", 2, "2022-07-16", root)                              # meta / path mismatch
    raw = SF.read_score(p)
    raw["meta"]["day"] = "2022-07-17"
    SF.write_score(raw, p, st)
    with pytest.raises(ValueError, match="does not match"):
        I.scan(ds, root)


def test_inventory_refuses_out_of_scope_names(tmp_path):
    with pytest.raises(Exception):
        I.scan("NOT_A_HOST", str(tmp_path))
    with pytest.raises(Exception):
        I.jobs("NOT_A_HOST")
