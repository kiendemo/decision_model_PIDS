"""Metric core (habitus.eval.metrics; CPU, synthetic).

Every expected number is worked out by hand in the comments: the confusion at the MCC-maximising
threshold, the MCC formula, ties at the threshold counted positive (the whole block), the ties-negative
counts at the same threshold, ground truth without a card as FN, AP / AUC with missing GT, the clean-day
bar and the budget rows.
"""

import ast
import inspect
import math
import os
from pathlib import Path

import numpy as np
import pytest

from habitus.eval import metrics as M


def _mcc(tp, fp, tn, fn):
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / den if den else 0.0


# --------------------------------------------------------------------------- #
# mcc / confusion_counts
# --------------------------------------------------------------------------- #
def test_mcc_hand_values():
    assert M.mcc(2, 0, 3, 1) == pytest.approx(6 / math.sqrt(2 * 3 * 3 * 4))          # 0.7071
    assert M.mcc(5, 0, 5, 0) == pytest.approx(1.0)
    assert M.mcc(0, 5, 0, 5) == pytest.approx(-1.0)
    assert M.mcc(0, 0, 7, 3) == 0.0                                                  # empty alert set
    v = M.mcc(np.array([1, 2]), np.array([0, 1]), np.array([4, 3]), np.array([2, 1]))
    assert v == pytest.approx([_mcc(1, 0, 4, 2), _mcc(2, 1, 3, 1)])


def test_confusion_counts_rates():
    c = M.confusion_counts(2, 1, 3, 5)                       # tp 2 fp 1 -> fn 1 tn 4
    assert (c["tp"], c["fp"], c["tn"], c["fn"]) == (2, 1, 4, 1)
    assert c["precision"] == pytest.approx(2 / 3) and c["recall"] == pytest.approx(2 / 3)
    assert c["f1"] == pytest.approx(4 / 6) and c["fpr"] == pytest.approx(1 / 5)
    assert c["mcc"] == pytest.approx((2 * 4 - 1 * 1) / math.sqrt(3 * 3 * 5 * 5))      # 7/15
    assert M.confusion_counts(0, 0, 3, 5)["precision"] is None
    with pytest.raises(ValueError):
        M.confusion_counts(4, 0, 3, 5)


# --------------------------------------------------------------------------- #
# best_mcc
# --------------------------------------------------------------------------- #
def test_best_mcc_distinct_scores():
    # sorted: 0.9 P, 0.8 P, 0.7 N, 0.6 P, 0.5 N, 0.4 N   (P = 3, N = 3)
    s = np.array([0.5, 0.9, 0.7, 0.8, 0.4, 0.6])
    y = np.array([0, 1, 0, 1, 0, 1], dtype=bool)
    # cuts: top1 (1,0) mcc .4472; top2 (2,0) .7071; top3 (2,1) .3333; top4 (3,1) .7071; top5 .4472; top6 0
    r = M.best_mcc(s, y)
    assert (r["tp"], r["fp"], r["tn"], r["fn"]) == (2, 0, 3, 1)                       # highest of the tied MCC
    assert r["threshold"] == 0.8 and r["alerts"] == 2
    assert r["mcc"] == pytest.approx(math.sqrt(0.5))
    assert r["precision"] == 1.0 and r["recall"] == pytest.approx(2 / 3)
    assert r["tie_block"] == {"n": 1, "gt": 1}
    tn = r["ties_negative"]                                  # s > 0.8: only the 0.9 node
    assert (tn["tp"], tn["fp"], tn["tn"], tn["fn"], tn["alerts"]) == (1, 0, 3, 2, 1)
    assert tn["mcc"] == pytest.approx(3 / math.sqrt(45))                            # 0.4472
    assert r["n_thresholds"] == 6 and r["n"] == 6 and r["n_pos"] == 3


def test_best_mcc_tie_block_counted_positive_never_split():
    # a tie block at 0.9 holding 1 GT + 1 benign, then 0.5: 1 GT, 0.1: 2 benign  (P = 2, N = 3)
    s = np.array([0.9, 0.9, 0.5, 0.1, 0.1])
    y = np.array([1, 0, 1, 0, 0], dtype=bool)
    # cut >= 0.9: (1,1) tn 2 fn 1 -> mcc (2-1)/sqrt(2*2*3*3) = 1/6
    # cut >= 0.5: (2,1) tn 2 fn 0 -> (4-0)/sqrt(3*2*3*2) = 4/6
    # cut >= 0.1: all -> mcc 0
    r = M.best_mcc(s, y)
    assert r["threshold"] == 0.5 and (r["tp"], r["fp"], r["tn"], r["fn"]) == (2, 1, 2, 0)
    assert r["mcc"] == pytest.approx(4 / 6)
    # the split "GT of the 0.9 block first" (tp 1 fp 0, mcc 0.6124) is never a candidate
    r2 = M.best_mcc(np.array([0.9, 0.9, 0.1]), np.array([1, 0, 0], dtype=bool))
    assert (r2["tp"], r2["fp"]) == (1, 1) and r2["tie_block"] == {"n": 2, "gt": 1}
    assert r2["mcc"] == pytest.approx(_mcc(1, 1, 1, 0))                               # 0.5
    assert r2["ties_negative"]["alerts"] == 0 and r2["ties_negative"]["mcc"] == 0.0


def test_best_mcc_missing_gt_and_minus_inf_are_fn():
    # universe: 0.9 P, 0.2 N, -inf P (no score), -inf N; plus 2 GT nodes without a card
    s = np.array([0.9, 0.2, -np.inf, -np.inf])
    y = np.array([1, 0, 1, 0], dtype=bool)
    r = M.best_mcc(s, y, n_missing_pos=2)
    # P = 2 + 2 = 4, N = 2; -inf never a threshold -> candidates 0.9 (1,0) and 0.2 (1,1)
    assert r["n_pos"] == 4 and r["n_neg"] == 2 and r["n_thresholds"] == 2
    assert (r["tp"], r["fp"], r["tn"], r["fn"]) == (1, 0, 2, 3)
    assert r["mcc"] == pytest.approx(_mcc(1, 0, 2, 3))                                # 2/sqrt(1*4*2*5)
    assert r["recall"] == pytest.approx(0.25)
    assert r["n_alertable"] == 2 and r["gt_alertable"] == 1


def test_best_mcc_edge_cases():
    assert M.best_mcc(np.array([0.1, 0.2]), np.array([0, 0], dtype=bool))["mcc"] is None
    r = M.best_mcc(np.array([-np.inf, -np.inf]), np.array([1, 0], dtype=bool))
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (0, 0, 1, 1) and r["threshold"] is None and r["mcc"] == 0.0
    with pytest.raises(ValueError):
        M.best_mcc(np.array([np.nan, 0.1]), np.array([1, 0], dtype=bool))
    with pytest.raises(ValueError):
        M.best_mcc(np.array([0.1]), np.array([1, 0], dtype=bool))
    # an inverted scorer: every cut has MCC <= 0 (0.9: -1, all: 0) -> the empty alert set, MCC 0
    r = M.best_mcc(np.array([0.9, 0.1]), np.array([0, 1], dtype=bool))
    assert r["mcc"] == 0.0 and r["alerts"] == 0 and r["threshold"] is None and (r["tp"], r["fn"]) == (0, 1)
    assert "no threshold has MCC > 0" in r["why"]


def test_best_mcc_matches_a_brute_force_sweep():
    rng = np.random.default_rng(0)
    for _ in range(30):
        n = int(rng.integers(5, 40))
        s = np.round(rng.random(n), 1)                       # many ties
        y = rng.random(n) < 0.3
        if not y.any() or y.all():
            continue
        r = M.best_mcc(s, y)
        best = max(_mcc(int((y & (s >= t)).sum()), int((~y & (s >= t)).sum()),
                        int((~y & (s < t)).sum()), int((y & (s < t)).sum())) for t in np.unique(s))
        assert r["mcc"] == pytest.approx(max(best, 0.0))
        assert r["tp"] + r["fn"] == int(y.sum()) and r["fp"] + r["tn"] == int((~y).sum())


# --------------------------------------------------------------------------- #
# confusion(host, ...): universe alignment, labels, GT without a card
# --------------------------------------------------------------------------- #
class TinyHost:
    """A labelled stand-in: universe ids 10..15, GT {11, 13} + one GT node without a test-day card."""
    ds = "TINY"
    universe = np.array([10, 11, 12, 13, 14, 15], dtype=np.int64)
    test = ("2020-01-02",)

    def upos(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        pos = np.searchsorted(self.universe, ids)
        ok = pos < len(self.universe)
        ok[ok] = self.universe[pos[ok]] == ids[ok]
        return np.where(ok, pos, -1)

    def gt(self):
        return {11, 13}, 3


def test_confusion_counts_cardless_gt_as_fn():
    h = TinyHost()
    r = M.confusion(h, {11: 0.9, 13: 0.8, 12: 0.7, 10: 0.1})          # 14, 15 unscored -> -inf
    # P = 3 (one cardless), N = 4; cut >= 0.8: tp 2 fp 0 tn 4 fn 1
    assert (r["tp"], r["fp"], r["tn"], r["fn"]) == (2, 0, 4, 1)
    assert r["mcc"] == pytest.approx(8 / math.sqrt(120)) and r["threshold"] == 0.8
    assert r["universe"]["gt_without_card"] == 1 and r["universe"]["nodes"] == 7
    assert r["ap"] == pytest.approx(2 / 3)                   # AP: (1 + 1) / |GT_total| = 3
    aligned = np.array([0.1, 0.9, 0.7, 0.8, -np.inf, -np.inf])
    assert M.confusion(h, aligned, universe_n=6)["mcc"] == pytest.approx(r["mcc"])
    with pytest.raises(ValueError):
        M.confusion(h, aligned, universe_n=5)
    with pytest.raises(ValueError):
        M.confusion(h, aligned[:4])


def test_own_confusion_uses_the_systems_node_set():
    r = M.own_confusion([0.9, 0.8, 0.1, 0.05], [1, 0, 1, 0])
    # cuts: 0.9 (1,0) mcc 0.577; 0.8 (1,1) 0; 0.1 (2,1) 0.577 -> highest threshold 0.9
    assert (r["tp"], r["fp"], r["tn"], r["fn"]) == (1, 0, 2, 1) and r["threshold"] == 0.9
    assert r["universe"] == {"name": "own", "nodes": 4, "gt_total": 2, "gt_in_universe": 2, "gt_without_card": 0}


# --------------------------------------------------------------------------- #
# AP / AUC
# --------------------------------------------------------------------------- #
def test_ap_auc_equal_sklearn_without_missing_gt():
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(1)
    for _ in range(20):
        n = int(rng.integers(8, 60))
        s = np.round(rng.random(n), 1)
        y = rng.random(n) < 0.3
        if not y.any() or y.all():
            continue
        assert M.ap(s, y) == pytest.approx(average_precision_score(y, s))
        assert M.auc(s, y) == pytest.approx(roc_auc_score(y, s))


def test_ap_auc_missing_gt_hand_values():
    s, y = np.array([0.9, 0.5, 0.1]), np.array([1, 0, 1], dtype=bool)
    # retrieved GT: rank 1 (precision 1), rank 3 (precision 2/3); |GT_total| = 3 -> (1 + 2/3) / 3
    assert M.ap(s, y, 3) == pytest.approx((1 + 2 / 3) / 3)
    # the missing GT sits strictly below 0.1: pairs (P, N) = 3 P x 1 N; P above N: 0.9 > 0.5 only -> 1/3
    assert M.auc(s, y, 3) == pytest.approx(1 / 3)
    assert M.ap(np.zeros(0), np.zeros(0, dtype=bool), 0) is None
    with pytest.raises(ValueError):
        M.ap(s, y, 1)                                        # n_total below the GT inside the universe


def test_pool_max_score_any_gt():
    ps, py = M.pool(np.array([0.1, 0.7, 0.3, 0.2]), np.array([0, 0, 1, 0], dtype=bool), np.array([2, 0, 2, 5]))
    # keys 0 -> 0.7 (N); 2 -> max(0.1, 0.3) (P); 5 -> 0.2 (N); used codes ascending
    assert ps.tolist() == [0.7, 0.3, 0.2] and py.tolist() == [False, True, False]


# --------------------------------------------------------------------------- #
# bar (re-exported) and budget rows
# --------------------------------------------------------------------------- #
def test_bar_is_the_label_free_copy():
    from habitus.decide import null
    assert M.bar is null.bar


def test_bar_hand_values():
    null = [0.9, 0.5, 0.5, 0.1]
    # 2 days, B = 1: m = 2 null units may reach the bar; sorted desc [0.9, 0.5, 0.5, 0.1] -> just above 0.5
    t = M.bar(null, 2, 1)
    assert t == np.nextafter(0.5, np.inf) and sum(x >= t for x in null) / 2 <= 1
    assert M.bar(null, 2, 0) == np.nextafter(0.9, np.inf)    # B = 0: above the highest null unit
    assert M.bar(null, 2, 2) == -math.inf                     # the budget covers the whole null
    assert M.bar([], 2, 1) is None and M.bar(null, 0, 1) is None


def test_budget_rows_hand_values():
    s = np.array([0.95, 0.6, 0.55, 0.2])
    y = np.array([1, 1, 0, 0], dtype=bool)
    null = [0.9, 0.5, 0.5, 0.1]
    rows = M.budget_rows(s, y, null, 2, 2, budgets=(0, 1, 2), n_total=3,
                         day_index={"d1": np.array([0, 1]), "d2": np.array([2, 3])})
    # B = 0: bar just above 0.9 -> alerts {0.95}: tp 1 fp 0; B = 1: above 0.5 -> {0.95, 0.6, 0.55}: tp 2 fp 1
    assert (rows["0"]["tp"], rows["0"]["fp"], rows["0"]["alerts"]) == (1, 0, 1)
    assert (rows["1"]["tp"], rows["1"]["fp"], rows["1"]["alerts"]) == (2, 1, 3)
    assert rows["1"]["fp_per_day"] == 0.5 and rows["1"]["recall"] == pytest.approx(2 / 3)
    assert rows["1"]["alerts_by_day"] == {"d1": 2, "d2": 1}
    assert rows["2"]["bar"] is None and rows["2"]["alerts"] == 4          # -inf bar: every node


# --------------------------------------------------------------------------- #
# Provenance: bar equals the original evalkit.bar (needs the original source tree)
# --------------------------------------------------------------------------- #
def _original(relpath):
    if os.environ.get("HABITUS_PROVENANCE") != "1":
        pytest.skip("provenance test: set HABITUS_PROVENANCE=1")
    from habitus.config import settings
    src = settings().paths.pidsmaker_src
    p = Path(str(src or "")) / relpath
    if not src or not p.is_file():
        pytest.skip(f"original source missing: {relpath}")
    return p.read_text(encoding="utf-8")


def _function(text, name):
    tree = ast.parse(text)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    body = [b for b in fn.body if not (isinstance(b, ast.Expr) and isinstance(getattr(b, "value", None), ast.Constant))]
    return fn, ast.dump(ast.Module(body=body, type_ignores=[]))


@pytest.mark.provenance
def test_bar_equals_original_evalkit_bar():
    """The original is read as text and its `bar` compiled alone: no original module is imported."""
    text = _original("pidsmaker/reflex/anchor/evalkit.py")
    fn, body = _function(text, "bar")
    _f2, mine = _function(inspect.getsource(M.bar), "bar")
    assert body == mine
    ns = {"np": np, "math": math}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "evalkit_bar", "exec"), ns)
    rng = np.random.default_rng(5)
    for _ in range(50):
        v, nd, b = rng.random(int(rng.integers(1, 300))), int(rng.integers(1, 6)), int(rng.integers(0, 30))
        assert M.bar(v, nd, b) == ns["bar"](v, nd, b)


@pytest.mark.provenance
def test_metric_core_equals_original():
    text = _original("pidsmaker/reflex/habitus/evaluate.py")
    for name in ("mcc", "confusion_counts", "best_mcc", "_universe_vector"):
        assert _function(text, name)[1] == _function(inspect.getsource(getattr(M, name)), name)[1], name
    text = _original("pidsmaker/reflex/anchor/evalkit.py")
    for name in ("_check", "_finite", "_total", "_blocks", "ap", "pool", "budget_rows"):
        assert _function(text, name)[1] == _function(inspect.getsource(getattr(M, name)), name)[1], name
