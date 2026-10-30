"""Operating tables (habitus.eval.operating): composite variants, repr-float CSVs, the label-free forms
replay child; asset tests (release bundle): ATLASV2_EDR budget rows / per-attack recall / lambda-combiner
sensitivity equal the shipped CSV rows, and the forms replay reproduces the shipped per-form rows."""

import ast
import csv
import io
import os
from pathlib import Path

import pytest

from habitus.eval import operating as O

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results" / "habitus"


def _need(*paths):
    missing = [str(p) for p in paths if p is None or not Path(p).exists()]
    if missing:
        if os.environ.get("HABITUS_REQUIRE_ASSETS", "") not in ("", "0"):
            pytest.fail(f"assets missing: {missing}")
        pytest.skip(f"assets missing: {missing}")


def test_composite_variants():
    mx = lambda r: max(r["qv"], r["qt"])                     # noqa: E731
    f = O._composite(0.5, mx)
    assert f({"qv": 0.8, "qt": 0.2, "qa": 0.5}) == pytest.approx(0.8 * 0.75)
    assert f({"qv": 0.8, "qt": 0.2, "qa": 0.5, "hard": True}) == 0.8          # hard tier: no acquittal
    assert f({"qv": 0.1, "qt": 0.6, "qa": None}) == 0.6                        # QA not asked
    assert O._composite(1.0, mx)({"qv": 0.4, "qt": 0.0, "qa": 1.0}) == 0.0
    bd = O._transform({"d": {"s": {"qv": 0.3, "qt": 0.7, "qa": 0.0}}}, f)
    assert bd == {"d": {"s": 0.7}}


def _budget_dict():
    row = {"tp": 1, "fp": 2, "tn": 3, "fn": 4, "precision": 0.1 + 0.2, "recall": 0.25, "mcc": 1 / 3, "ap": 0.5,
           "auc": 0.75, "threshold": 0.6, "alerts": 3, "mcc_ties_neg": None}
    return {"ds": "ATLASV2_EDR", "plans": {"k2": {
        "main": False, "oracle": row,
        "budget": {"null_days": ["c1", "c2"], "null_units": 10, "n_null_days": 2, "same_caps": True,
                   "rows": {"0": {"bar": 0.9, "alerts": 1, "tp": 0, "fp": 1, "alerts_per_day": 1.0, "fp_per_day": 1.0,
                                  "recall": 0.0, "alerts_by_day": {"t1": 1}},
                            "20": {"bar": None, "alerts": 9, "tp": 2, "fp": 7, "alerts_per_day": 9.0,
                                   "fp_per_day": 7.0, "recall": 0.5, "alerts_by_day": {"t1": 9}}}},
        "oracle_by_attack": {"g1": {"gt_nodes": 2, "in_universe": 2, "alerted": 1}},
        "budget_by_attack": {"0": {"g1": {"gt_nodes": 2, "in_universe": 2, "alerted": 0}}},
        "sensitivity": {"max, lambda=0": row}}}}


def test_csvs_write_repr_floats(tmp_path):
    w = O.write_csvs({"ATLASV2_EDR": _budget_dict()}, None, tmp_path)
    assert set(w) == {"budget_rows.csv", "per_attack_recall.csv", "sensitivity_composition.csv"}
    rows = list(csv.DictReader(io.StringIO(w["budget_rows.csv"].read_text(), newline="")))
    assert [r["B"] for r in rows] == ["0", "20"] and rows[1]["bar"] == "" and rows[0]["alerts_by_day"] == "t1=1"
    assert rows[0]["oracle_mcc"] == repr(1 / 3) and rows[0]["null_days"] == "c1;c2"
    sens = list(csv.DictReader(io.StringIO(w["sensitivity_composition.csv"].read_text(), newline="")))
    assert sens[0]["precision"] == "0.30000000000000004" and sens[0]["mcc_ties_neg"] == ""
    att = list(csv.DictReader(io.StringIO(w["per_attack_recall.csv"].read_text(), newline="")))
    assert [(r["at"], r["group"], r["alerted"]) for r in att] == [("oracle", "g1", "1"), ("B=0", "g1", "0")]
    assert w["budget_rows.csv"].read_bytes().count(b"\r\n") == 3                  # csv default dialect


def test_forms_child_is_label_free_and_guarded_first():
    tree = ast.parse(O._FORMS_CHILD)
    mods = []
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            mods.append(n.module)
        elif isinstance(n, ast.Import):
            mods += [a.name for a in n.names]
    assert not any(m and m.startswith("habitus.eval") for m in mods)
    habitus_mods = [m for m in mods if m and m.startswith("habitus")]
    assert habitus_mods[0] == "habitus.guard"
    src = O._FORMS_CHILD
    assert src.index("install_guard(") < src.index("from habitus.data.host")
    assert "cache_only=True" in src and "get_dm(" in src


def test_forms_keys_and_vectors():
    assert O.FORM_KEYS[0] == "both" and set(O.VEC_KEYS) <= set(O.FORM_KEYS)


# --------------------------------------------------------------------------- #
# Asset tests (release bundle: scores, node index or cards, GT; forms: cards, answers, tokenizer, precedent)
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


def _shipped_lines(name, ds):
    p = RES / name
    _need(p)
    lines = p.read_bytes().split(b"\r\n")
    return [lines[0]] + [x for x in lines[1:] if x.startswith(ds.encode() + b",")]


def test_atlas_budget_rows_equal_the_shipped_csv(tmp_path):
    from habitus.config import settings
    ds = "ATLASV2_EDR"
    host = _release_host(ds)
    b = O.budget(ds, root=settings().paths.scores, host=host)
    w = O.write_csvs({ds: b}, None, tmp_path)
    for name in ("budget_rows.csv", "per_attack_recall.csv", "sensitivity_composition.csv"):
        assert w[name].read_bytes().split(b"\r\n")[:-1] == _shipped_lines(name, ds), name


def test_atlas_forms_replay_equals_the_shipped_rows(tmp_path):
    from habitus.config import settings
    st = settings()
    ds = "ATLASV2_EDR"
    _need(Path(st.paths.cards) / ds, Path(st.paths.answers) / "qwen3.5-4b", Path(st.paths.precedent) / ds)
    host = _release_host(ds)
    s = O.forms(ds, root=st.paths.scores, out_dir=tmp_path, host=host)
    assert s["recon_max_abs"] == 0.0 and s["cache_misses"] == 0
    w = O.write_csvs(None, {ds: s}, tmp_path)
    for name in ("sensitivity_forms.csv", "form_gaps.csv"):
        assert w[name].read_bytes().split(b"\r\n")[:-1] == _shipped_lines(name, ds), name
    assert (tmp_path / f"vec_{ds}.npz").exists()
