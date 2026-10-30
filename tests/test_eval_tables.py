"""The ablation table writer (habitus.eval.tables) and the derived means (habitus.eval.derived).

Synthetic: the original writer (csv.DictWriter default dialect = CRLF, round(x, 6), '' for None, HOSTS
order) and run_host over a tiny stand-in host with score files in tmp. Shipped results/: the CSV round
trips byte for byte, main_config.csv is its main rows, the markdown renders from the rows, the derived
means equal summary_means.json and the paper. Asset test (release bundle): ATLASV2_EDR's rows are
recomputed byte-equal (a G1 subset).
"""

import gzip
import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

from habitus.eval import derived as D
from habitus.eval import tables as T

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"


def _need(*paths):
    missing = [str(p) for p in paths if p is None or not Path(p).exists()]
    if missing:
        if os.environ.get("HABITUS_REQUIRE_ASSETS", "") not in ("", "0"):
            pytest.fail(f"assets missing: {missing}")
        pytest.skip(f"assets missing: {missing}")


def _row(ds, **kw):
    r = {c: None for c in T.COLS}
    r.update({"ds": ds, "system": "E2 (routed)", "level": "E2", "k": "k2", "column": "composite", "main_config": "",
              "flags": "", "note": ""})
    r.update(kw)
    return r


# --------------------------------------------------------------------------- #
# Writer
# --------------------------------------------------------------------------- #
def test_writer_crlf_round6_none_and_host_order():
    rows = [_row("ATLASV2_EDR", tp=3, mcc=0.123456789, precision=None, note="a, b"),
            _row("CADETS_E3", tp=1, mcc=1 / 3, threshold=-0.5000004)]
    b = T.write_csv(T.ordered(rows))
    assert T.write_csv(rows) != b                             # the writer keeps the given order; merge orders
    lines = b.split(b"\r\n")
    assert lines[0] == ",".join(T.COLS).encode() and lines[-1] == b"" and b"\n" not in b.replace(b"\r\n", b"")
    cad, atl = lines[1].decode().split(","), lines[2].decode()
    assert cad[0] == "CADETS_E3" and cad[T.COLS.index("mcc")] == "0.333333"       # HOSTS order, round 6
    assert cad[T.COLS.index("threshold")] == "-0.5" and cad[T.COLS.index("precision")] == ""
    assert atl.endswith(',"a, b"') and ",0.123457," in atl                         # quoting: minimal
    with pytest.raises(ValueError):
        T.ordered([_row("NOT_A_HOST")])


def test_parse_k():
    assert [T.parse_k(x) for x in ("2", "all", 2, None, "k0", "kall", "0")] == [2, None, 2, None, 0, None, 0]


# --------------------------------------------------------------------------- #
# run_host over a tiny stand-in host (no release assets)
# --------------------------------------------------------------------------- #
class _Spec:
    main_k = "2"


class _Host:
    ds = "TINY"
    spec = _Spec()
    test = ["t1", "t2"]
    _nodes = {"t1": {1: "a" * 16, 2: "b" * 16, 3: "c" * 16}, "t2": {2: "b" * 16, 4: "d" * 16, 5: "c" * 16}}
    universe = np.array([1, 2, 3, 4, 5], dtype=np.int64)

    def nodes(self, d):
        return self._nodes[d]

    def upos(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        pos = np.searchsorted(self.universe, ids)
        ok = pos < len(self.universe)
        ok[ok] = self.universe[pos[ok]] == ids[ok]
        return np.where(ok, pos, -1)

    def gt(self):
        return {2, 4}, 3                                      # one GT node without a card


def _score_file(root, level, k, day, scores):
    p = Path(root) / "TINY" / f"{level}_k{k}" / f"{day}.json.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wt", encoding="utf-8") as f:
        json.dump({"meta": {"ds": "TINY", "day": day, "level": level, "k": k}, "head": {}, "scores": scores,
                   "fallback": {}}, f)


def test_run_host_over_a_tiny_host(tmp_path):
    a, b, c, d = ("a" * 16, "b" * 16, "c" * 16, "d" * 16)
    e2 = {"t1": {b: {"qv": 0.9, "qt": 0.2, "qa": 0.5, "hard": False}, a: {"qv": 0.1, "qt": 0.3, "qa": None, "hard": False}},
          "t2": {d: {"qv": 0.4, "qt": 0.8, "qa": 0.0, "hard": True}, b: {"qv": 0.2, "qt": 0.1, "qa": 1.0, "hard": False}}}
    for day, sc in e2.items():
        _score_file(tmp_path, "E2", "2", day, sc)
        _score_file(tmp_path, "L4", "2", day, {s: {"qv": r["qv"], "qt": r["qt"]} for s, r in sc.items()})
    rows = T.run_host("TINY", root=tmp_path, host=_Host())
    keys = {(r["level"], r["k"], r["column"]) for r in rows}
    assert keys == {("E2", "k2", f) for f in T.E2_FIELDS} | {("L4", "k2", f) for f in T.E1_FIELDS}
    main = [r for r in rows if r["main_config"] == "yes"]
    assert [(r["level"], r["k"], r["column"]) for r in main] == [("E2", "k2", "composite")]
    m = main[0]
    # composite: b t1 = 0.9 * (1 - 0.25) = 0.675, d t2 hard = 0.8 -> node 4: 0.8, node 2: 0.675; a 0.3
    # GT {2, 4} + 1 cardless: cut >= 0.675 -> tp 2 fp 0 fn 1
    assert (m["tp"], m["fp"], m["fn"], m["gt_without_card"], m["n_universe"], m["gt_in_head"]) == (2, 0, 1, 1, 6, 2)
    st = {(s["level"], s["k"]): s["status"] for s in rows.status}
    assert st[("L2", "k0")] == "by-design" and st[("L0", "k2")] == "missing" and st[("E2", "k2")] == "ok"
    b_ = T.write_csv(rows)
    assert b_.count(b"\r\n") == len(rows) + 1


# --------------------------------------------------------------------------- #
# Shipped results/habitus
# --------------------------------------------------------------------------- #
def _shipped():
    p = RES / "habitus" / "ablation_e1_e2.csv"
    _need(p)
    raw = p.read_bytes()
    return raw, T.rows_from_csv(raw)


def test_shipped_csv_round_trips_byte_for_byte():
    raw, rows = _shipped()
    assert len(rows) == 251
    assert T.write_csv(rows) == raw
    e2, e1 = T.ref_check(rows)
    assert all(e2.values()) and all(e1)


def test_shipped_main_config_is_the_main_rows():
    raw, rows = _shipped()
    main = T.main_rows(rows)
    assert [r["ds"] for r in main] == list(T.HOSTS)
    assert all(r["k"] == T.KNAME[T.main_k(r["ds"])] for r in main)
    assert (RES / "habitus" / "main_config.csv").read_bytes() == T.write_csv(main)
    assert [round(r["mcc"], 6) for r in main] == [0.405488, 0.595552, 0.112087, 0.392856, 0.405215]


_NUM = re.compile(r"[-+]?\d+\.\d+")


def test_markdown_renders_from_the_rows():
    """The shipped CSV holds 6-dp values, so a re-render can differ from the shipped md by one unit in a
    3-dp cell; every other character must be equal (the full-precision render is the asset test below)."""
    _raw, rows = _shipped()
    md = (RES / "habitus" / "ablation_e1_e2.md").read_text(encoding="utf-8")
    _c, mine = T.merge(rows, generated=T.REF_GENERATED)
    a, b = md.split("\n"), mine.split("\n")
    assert len(a) == len(b)
    for x, y in zip(a, b):
        if x == y:
            continue
        assert _NUM.sub("#", x) == _NUM.sub("#", y), (x, y)
        assert all(abs(float(p) - float(q)) <= 1.0001e-3 for p, q in zip(_NUM.findall(x), _NUM.findall(y))), (x, y)
    assert T.md_body(mine) == T.md_body(T.merge(rows)[1])
    assert "<time>" in T.md_body(mine) and T.REF_GENERATED not in T.md_body(mine)


def test_summary_means_equal_the_shipped_json_and_paper():
    _raw, rows = _shipped()
    s = D.summary_means(rows)
    assert json.loads(json.dumps(s)) == json.loads((RES / "habitus" / "summary_means.json").read_text())
    assert s["main_mean_mcc"] == 0.3822396 and s["main_mean_mcc_ties_neg"] == 0.327636
    assert round(s["habitus_k2"]["mean_mcc"], 3) == 0.374
    assert s["ladder"]["E2|k2"]["mean5"] == s["habitus_k2"]["mean_mcc"]


def test_paper_means_from_the_results_csvs():
    _need(RES / "baselines" / "pidsmaker_rows.csv", RES / "extension_habit" / "decision_models_best_mcc.csv",
          RES / "extension_habit" / "hb0_rows.csv", RES / "habitus" / "route_cost.csv")
    p = D.paper_means(RES)
    b = p["baselines"]
    assert round(b["best_adp_any"]["mean"], 6) == 0.221626 and b["best_adp_any"]["hosts_habitus_above"] == 4
    assert [v[0] for v in b["best_adp_any"]["best"].values()] == [
        "Velox [tuned*]", "Velox [default]", "ThreaTrace [default]", "Velox [tuned*]", "Velox [default]"]
    assert round(b["epoch11"]["mean"], 4) == 0.2190 and round(b["better_of_two_epochs"]["mean"], 4) == 0.2431
    assert round(b["best_adp_ties_neg_mean"], 3) == 0.183
    e2 = p["habit"]["E2"]
    d = e2["delta_composite_vs_HB0"]
    assert (d["mean"], d["n_better"], d["n_worse"], d["n_equal"]) == (0.0045, 5, 4, 0)
    assert e2["mean_composite_9cells"] == 0.3753
    assert round(p["habit_main_config"]["HB-L.mq|composite"], 4) == 0.3942
    assert round(p["route_engine"]["CADETS_E3"]["test_k2_s_per_day"], 1) == 506.8
    assert not any("HB-C" in k for k in p["habit_main_config"]) and "HB-C" not in p["habit"]


# --------------------------------------------------------------------------- #
# Asset test: a G1 subset over the release bundle
# --------------------------------------------------------------------------- #
def _release_host(ds):
    from habitus.config import settings
    from habitus.data.host import Host
    st = settings()
    _need(Path(st.paths.scores) / ds, st.paths.ground_truth)
    if (Path(st.paths.nodeindex) / ds).is_dir():
        return Host(ds, source="nodeindex")
    _need(Path(st.paths.cards) / ds)
    return Host(ds, source="cards")


def test_g1_subset_atlas_rows_byte_equal():
    from habitus.config import settings
    raw, _rows = _shipped()
    host = _release_host("ATLASV2_EDR")
    rows = T.run_host("ATLASV2_EDR", root=settings().paths.scores, host=host)
    lines = raw.split(b"\r\n")
    want = [lines[0]] + [x for x in lines[1:] if x.startswith(b"ATLASV2_EDR,")]
    assert T.write_csv(rows).split(b"\r\n")[:-1] == want
