"""Paired stratified entity bootstrap (habitus.eval.bootstrap): strata, chunk-free draws, the weighted
oracle MCC equal to metrics.best_mcc at unit weights, merge / CSV; the shipped draws reproduce
bootstrap_mcc.csv; asset test (release bundle): ATLASV2_EDR points and draws equal the shipped ones."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from habitus.eval import bootstrap as B
from habitus.eval import metrics as M

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results" / "habitus"


def _need(*paths):
    missing = [str(p) for p in paths if p is None or not Path(p).exists()]
    if missing:
        if os.environ.get("HABITUS_REQUIRE_ASSETS", "") not in ("", "0"):
            pytest.fail(f"assets missing: {missing}")
        pytest.skip(f"assets missing: {missing}")


def test_strata_columns_and_sizes():
    pos = np.array([1, 0, 0, 1, 0, 0], dtype=bool)
    keys = np.array([0, 0, 1, 2, 3, 3])
    st = B.Strata(pos, keys, n_total=3)                      # one positive without a card
    # P entities {0, 2} + 1 cardless singleton, N entities {1, 3}
    assert (st.n_pos_ent, st.n_p, st.n_b, st.n_cols, st.n_miss) == (2, 3, 2, 5, 1)
    assert st.node_col.tolist() == [0, 0, 3, 1, 4, 4]
    assert st.node_pos.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]
    assert st.sizes["nodes_in_pos_entities"] == 3 and st.sizes["largest_entity_nodes"] == 2
    with pytest.raises(ValueError):
        B.Strata(pos, keys, n_total=1)


def test_draws_are_chunk_free_and_stratified():
    rng = np.random.default_rng(3)
    pos = rng.random(200) < 0.1
    keys = rng.integers(0, 60, 200)
    st = B.Strata(pos, keys, n_total=int(pos.sum()) + 2)
    a = st.draws(0, 10, 101)
    assert np.array_equal(a, np.concatenate([st.draws(0, 4, 101), st.draws(4, 10, 101)], axis=1))
    assert not np.array_equal(a, st.draws(0, 10, 102))
    assert (a[:st.n_p].sum(0) == st.n_p).all() and (a[st.n_p:].sum(0) == st.n_b).all()


def test_weighted_best_mcc_at_unit_weights_equals_best_mcc():
    rng = np.random.default_rng(7)
    for _ in range(25):
        n = int(rng.integers(20, 120))
        s = np.round(rng.random(n), 1)
        y = rng.random(n) < 0.2
        if not y.any():
            continue
        miss = int(rng.integers(0, 3))
        keys = rng.integers(0, n, n)
        st = B.Strata(y, keys, n_total=int(y.sum()) + miss)
        negcol = np.bincount(st.node_col[~st.pos], minlength=st.n_cols).astype(np.float64)
        rk = B.Ranked(s, st.pos, st.node_col, st.n_cols)
        got = float(B._best_mcc_weighted(rk, st, negcol, st.ones())[0])
        ref = M.best_mcc(s, y, n_missing_pos=miss)["mcc"] or 0.0
        assert got == pytest.approx(ref, abs=1e-12)


def test_ranked_ap_at_unit_weights_equals_ap():
    rng = np.random.default_rng(11)
    s = np.round(rng.random(80), 1)
    y = rng.random(80) < 0.25
    st = B.Strata(y, np.arange(80), n_total=int(y.sum()) + 1)
    rk = B.Ranked(s, st.pos, st.node_col, st.n_cols)
    ones = st.ones()
    assert rk.ap(ones, B.totals(st, ones)[0])[0] == pytest.approx(M.ap(s, y, int(y.sum()) + 1))


def _boot(ds, point, draws):
    return {"ds": ds, "point": point, "info": {"best baseline": {"system": "X"}}, "boot": draws}


def test_merge_deltas_points_and_csv(tmp_path):
    boots = {"CADETS_E3": _boot("CADETS_E3", {"a": 0.5, "b": 0.25}, {"a": [0.5, 0.6, 0.4, 0.5], "b": [0.3, 0.2, 0.5, 0.1]}),
             "THEIA_E3": _boot("THEIA_E3", {"a": 0.2, "b": 0.3}, {"a": [0.2, 0.2, 0.2, 0.2], "b": [0.3, 0.1, 0.3, 0.3]})}
    s = B.merge(boots, pairs=[("a", "b")])
    per = s["boot_deltas"]["a - b"]
    assert per["CADETS_E3"]["point"] == 0.25 and per["CADETS_E3"]["p_le_0"] == 0.25       # deltas .2 .4 -.1 .4
    assert per["mean5"]["point"] == pytest.approx((0.25 - 0.1) / 2)
    assert s["boot_points"]["THEIA_E3"]["_info"] == {"best baseline": {"system": "X"}}
    rows = B.csv_rows(s)
    assert [(r["kind"], r["ds"]) for r in rows][:3] == [("delta", "CADETS_E3"), ("delta", "THEIA_E3"), ("delta", "mean5")]
    p = B.write_csv(s, tmp_path / "b.csv")
    assert p.read_bytes().startswith(b"kind,name,ds,point,ci95_lo,ci95_hi,p_le_0,p_ge_0\r\n")


def test_strip_systems_keeps_the_other_draws():
    b = _boot("CADETS_E3", {"a": 0.5, "rule-lex": 0.4}, {"a": [0.1, 0.2], "rule-lex": [0.3, 0.4]})
    s = B.strip_systems(b, ("rule-lex",))
    assert s["point"] == {"a": 0.5} and s["boot"] == {"a": [0.1, 0.2]} and "rule-lex" in b["point"]


def test_shipped_draws_reproduce_bootstrap_csv():
    files = [RES / "bootstrap_draws" / f"boot_{ds}.json" for ds in B.HOSTS]
    _need(*files, RES / "bootstrap_mcc.csv")
    boots = B.load_boots(files)
    assert all(not ({"rule-lex", "rule-count"} & set(b["boot"])) for b in boots.values())
    assert all(len(b["boot"]["HABITUS main"]) == B.N_BOOT and b["seed"] == B.SEED[ds] for ds, b in boots.items())
    s = B.merge(boots)
    from habitus.eval.derived import csv_text
    assert csv_text(B.csv_rows(s), B.CSV_COLS).encode("utf-8") == (RES / "bootstrap_mcc.csv").read_bytes()
    d = s["boot_deltas"]["HABITUS main - best baseline"]["mean5"]
    assert (round(d["point"], 3), round(d["ci95"][0], 3), round(d["ci95"][1], 3), round(d["p_le_0"], 3)) == \
        (0.163, 0.049, 0.273, 0.003)


def test_atlas_boot_equals_the_shipped_draws(tmp_path):
    from habitus.config import settings
    from habitus.data.host import Host
    st = settings()
    ds = "ATLASV2_EDR"
    npz = B.baseline_npz_path(ds, st)
    _need(Path(st.paths.scores) / ds, st.paths.ground_truth, npz, RES / "bootstrap_draws" / f"boot_{ds}.json")
    if (Path(st.paths.nodeindex) / ds).is_dir():
        host = Host(ds, source="nodeindex")
    else:
        _need(Path(st.paths.cards) / ds)
        host = Host(ds, source="cards")
    b = B.boot(ds, root=st.paths.scores, operating_dir=tmp_path, baseline=npz, host=host)
    ref = json.loads((RES / "bootstrap_draws" / f"boot_{ds}.json").read_text())
    b = json.loads(json.dumps(b))
    for n, v in b["point"].items():                          # form1 / form2 need operating.forms first
        assert v == ref["point"][n] and b["boot"][n] == ref["boot"][n], n
    assert b["info"] == ref["info"] and b["sizes"] == ref["sizes"]
