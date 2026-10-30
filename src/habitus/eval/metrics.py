"""Metric core: oracle best-MCC confusion, AP / AUC over a universe, FP-budget rows.

Oracle threshold: the threshold that maximises MCC on the TEST labels (label-selected for every system
alike: an upper bound, never a deployable operating point). Node score = max over the test days; the
universe is every test-day node of the host plus the ground-truth nodes without a test-day card (always
negative, i.e. FN). The sweep visits every distinct score value (s >= t, the whole tie block at t counted
positive), so a tie block is never split: `ties_negative` gives the counts at the SAME threshold with its
tie block excluded, `tie_block` its size and ground-truth count.

Ranking metrics over the universe U, with the GT nodes that have no card kept in the denominators:
  ap   (1/|GT_total|) * sum over GT nodes in U of precision at the node's rank; a tie block counts as one
       threshold, every GT node in it gets the precision at the end of the block (sklearn's AP when no
       GT is missing).
  auc  sklearn's ROC AUC with every missing GT node appended strictly below the minimum score of U.

FP budget: bar(B) = the smallest score whose count of null units per day is <= B (label-free;
re-exported from habitus.decide.null, the single copy); alerts are the nodes with score >= bar.

Pure arithmetic except confusion(), which reads the labels through habitus.eval.labels.
"""

from __future__ import annotations

import math

import numpy as np

from habitus.decide.null import bar  # noqa: F401  (re-exported: the one label-free copy)

BUDGETS = (1, 5, 20, 40, 100)


# --------------------------------------------------------------------------- #
# Oracle best-MCC confusion
# --------------------------------------------------------------------------- #
def mcc(tp, fp, tn, fn):
    """Matthews correlation coefficient (0 where a margin is empty); scalars or equal-shape arrays."""
    tp, fp, tn, fn = (np.asarray(x, dtype=np.float64) for x in (tp, fp, tn, fn))
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    num = tp * tn - fp * fn
    out = np.divide(num, den, out=np.zeros(np.broadcast(num, den).shape), where=den > 0)
    return float(out) if out.ndim == 0 else out


def confusion_counts(tp, fp, n_pos, n_neg):
    """Confusion and the derived rates of one alert set: tp / fp alerted, n_pos / n_neg the class sizes."""
    tp, fp, n_pos, n_neg = int(tp), int(fp), int(n_pos), int(n_neg)
    fn, tn = n_pos - tp, n_neg - fp
    if min(tp, fp, fn, tn) < 0:
        raise ValueError(f"inconsistent counts tp={tp} fp={fp} n_pos={n_pos} n_neg={n_neg}")
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / n_pos if n_pos else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "fpr": fp / n_neg if n_neg else None,
            "mcc": mcc(tp, fp, tn, fn)}


def best_mcc(s, y, n_missing_pos=0, n_missing_neg=0):
    """Pure arithmetic: the MCC-maximising threshold of scores `s` (higher = more suspicious, -inf =
    never alerted) against labels `y`, with n_missing_pos / n_missing_neg extra nodes outside `s` that are
    never alerted (ground truth without a card: FN). Every distinct finite value t is a candidate
    (alert = s >= t); the highest threshold wins among equal MCC; when no cut has MCC > 0 the empty alert
    set (MCC 0, threshold None) is reported. Returns the confusion at t*, t*, the alert count, the tie
    block at t* and the counts with that block excluded (ties negative)."""
    s, y = np.asarray(s, dtype=np.float64), np.asarray(y, dtype=bool)
    if s.shape != y.shape or s.ndim != 1:
        raise ValueError(f"scores {s.shape} and labels {y.shape} must be equal 1-d arrays")
    if np.isnan(s).any():
        raise ValueError("NaN score")
    n_pos, n_neg = int(y.sum()) + int(n_missing_pos), int((~y).sum()) + int(n_missing_neg)
    alertable = s > -np.inf
    base = {"n": n_pos + n_neg, "n_pos": n_pos, "n_neg": n_neg, "n_missing_pos": int(n_missing_pos),
            "n_alertable": int(alertable.sum()), "gt_alertable": int((alertable & y).sum())}
    if n_pos == 0 or n_neg == 0:
        return {**base, "mcc": None, "why": "one class is empty"}
    sf, yf = s[alertable], y[alertable]
    if not len(sf):
        return {**base, **confusion_counts(0, 0, n_pos, n_neg), "threshold": None, "alerts": 0, "n_thresholds": 0,
                "tie_block": {"n": 0, "gt": 0}, "ties_negative": {**confusion_counts(0, 0, n_pos, n_neg), "alerts": 0}}
    order = np.argsort(-sf, kind="mergesort")
    ss, ys = sf[order], yf[order]
    end = np.r_[ss[1:] != ss[:-1], True]
    tp = np.cumsum(ys)[end]
    n = np.flatnonzero(end) + 1
    fp = n - tp
    m = mcc(tp, fp, n_neg - fp, n_pos - tp)
    i = int(np.argmax(m))
    if m[i] <= 0.0:                         # no cut beats alerting nothing (MCC 0): report the empty set
        return {**base, **confusion_counts(0, 0, n_pos, n_neg), "threshold": None, "alerts": 0,
                "n_thresholds": int(len(n)), "tie_block": {"n": 0, "gt": 0},
                "ties_negative": {**confusion_counts(0, 0, n_pos, n_neg), "alerts": 0},
                "why": f"no threshold has MCC > 0 (best cut {float(m[i]):+.4f} at {float(ss[end][i]):.4g})"}
    prev_n, prev_tp = (int(n[i - 1]), int(tp[i - 1])) if i else (0, 0)
    return {**base, **confusion_counts(tp[i], fp[i], n_pos, n_neg), "threshold": float(ss[end][i]),
            "alerts": int(n[i]), "n_thresholds": int(len(n)),
            "tie_block": {"n": int(n[i]) - prev_n, "gt": int(tp[i]) - prev_tp},
            "ties_negative": {**confusion_counts(prev_tp, prev_n - prev_tp, n_pos, n_neg), "alerts": prev_n}}


def _universe_vector(host, scores_by_node):
    """Universe-aligned float vector from an aligned array or {node id: score}; unscored nodes -inf."""
    if isinstance(scores_by_node, dict):
        s = np.full(len(host.universe), -np.inf)
        if scores_by_node:
            ids = np.fromiter(scores_by_node.keys(), dtype=np.int64, count=len(scores_by_node))
            v = np.fromiter(scores_by_node.values(), dtype=np.float64, count=len(scores_by_node))
            pos = host.upos(ids)
            ok = pos >= 0
            np.maximum.at(s, pos[ok], v[ok])
        return s
    s = np.asarray(scores_by_node, dtype=np.float64)
    if s.shape != (len(host.universe),):
        raise ValueError(f"{host.ds}: score vector {s.shape} is not aligned with the universe ({len(host.universe)})")
    return s


def _check_universe(host, nodes):
    """An allow-listed host's evaluation universe must have HostSpec.n_universe nodes."""
    from habitus.hosts import DATASETS, spec
    if getattr(host, "ds", None) not in DATASETS or callable(getattr(host, "gt", None)):
        return
    want = getattr(spec(host.ds), "n_universe", None)
    if want is not None and int(want) != int(nodes):
        raise ValueError(f"{host.ds}: evaluation universe {nodes} != configs/datasets.yaml n_universe {want}")


def confusion(host, scores_by_node, universe_n=None):
    """Best-MCC confusion of one system over the WHOLE test universe of `host` (labels read here):
    ground-truth nodes without a test-day card (or without a score) are FN. universe_n: the expected
    universe size (a sanity check). AP / AUC (same universe) are attached for context."""
    from habitus.eval.labels import host_labels
    s = _universe_vector(host, scores_by_node)
    if universe_n is not None and int(universe_n) != len(host.universe):
        raise ValueError(f"{host.ds}: universe {len(host.universe)} != expected {universe_n}")
    y, n_total = host_labels(host)
    miss = int(n_total) - int(y.sum())
    row = best_mcc(s, y, n_missing_pos=miss)
    row["ap"] = ap(s, y, n_total)
    row["auc"] = auc(s, y, n_total)
    row["universe"] = {"name": "habitus", "nodes": int(len(host.universe)) + miss, "test_day_nodes": int(len(host.universe)),
                       "gt_total": int(n_total), "gt_in_universe": int(y.sum()), "gt_without_card": miss}
    _check_universe(host, row["universe"]["nodes"])
    return row


def own_confusion(s, y):
    """Best-MCC confusion on a system's OWN node set and labels (the PIDSMaker evaluation space)."""
    s, y = np.asarray(s, dtype=np.float64), np.asarray(y, dtype=bool)
    row = best_mcc(s, y)
    row["ap"] = ap(s, y)
    row["auc"] = auc(s, y)
    row["universe"] = {"name": "own", "nodes": int(len(s)), "gt_total": int(y.sum()), "gt_in_universe": int(y.sum()),
                       "gt_without_card": 0}
    return row


# --------------------------------------------------------------------------- #
# Ranking metrics over the universe (missing GT counted as never retrieved)
# --------------------------------------------------------------------------- #
def _check(s, y):
    s, y = np.asarray(s, dtype=np.float64), np.asarray(y, dtype=bool)
    if s.shape != y.shape or s.ndim != 1:
        raise ValueError(f"scores {s.shape} and labels {y.shape} must be equal 1-d arrays")
    if np.isnan(s).any():
        raise ValueError("NaN score")
    return s, y


def _finite(s):
    """Order-preserving map of +-inf to finite values (sklearn rejects infinities)."""
    fin = np.isfinite(s)
    if fin.all():
        return s
    lo, hi = (s[fin].min(), s[fin].max()) if fin.any() else (0.0, 0.0)
    return np.where(s == -np.inf, lo - 1.0, np.where(s == np.inf, hi + 1.0, s))


def _total(y, n_total):
    n_in = int(y.sum())
    n_total = n_in if n_total is None else int(n_total)
    if n_total < n_in:
        raise ValueError(f"n_total={n_total} < {n_in} GT nodes in the universe")
    return n_total


def _blocks(s, y):
    """Tie blocks in descending score order: the sort order, the block-end mask over sorted
    positions, and cumulative GT and cumulative count at each block end."""
    order = np.argsort(-s, kind="mergesort")
    ss, ys = s[order], y[order]
    end = np.r_[ss[1:] != ss[:-1], True]
    tp = np.cumsum(ys)[end]
    n = np.flatnonzero(end) + 1
    return order, end, tp, n


def ap(s, y, n_total=None):
    """AP with GT outside the universe counted as never retrieved (n_total = |GT_total|)."""
    s, y = _check(s, y)
    n_total = _total(y, n_total)
    if n_total == 0:
        return None
    if not len(s):
        return 0.0
    _o, _e, tp, n = _blocks(s, y)
    g = np.diff(tp, prepend=0)
    return float((g * tp / n).sum() / n_total)


def auc(s, y, n_total=None, tie=False):
    """ROC AUC with every missing GT node appended strictly below the minimum score of the
    universe (`tie`: at the minimum, the partial-credit form)."""
    from sklearn.metrics import roc_auc_score
    s, y = _check(s, y)
    miss = _total(y, n_total) - int(y.sum())
    if miss:
        lo = float(_finite(s).min()) if len(s) else 0.0
        s = np.r_[_finite(s), np.full(miss, lo if tie else lo - 1.0)]
        y = np.r_[y, np.ones(miss, dtype=bool)]
    if y.all() or not y.any():
        return None
    return float(roc_auc_score(y, _finite(s)))


def pool(s, y, keys):
    """Entity pools: (max score, any GT) per distinct key. `keys` may be integer codes (used as
    pool ids directly, e.g. Host.ukey_codes) or any array np.unique accepts."""
    s, y = _check(s, y)
    keys = np.asarray(keys)
    if keys.shape != s.shape:
        raise ValueError("one key per universe node")
    dense = (np.issubdtype(keys.dtype, np.integer) and len(keys)
             and keys.min() >= 0 and keys.max() <= 4 * len(keys))
    codes = keys.astype(np.int64) if dense else np.unique(keys, return_inverse=True)[1]
    m = int(codes.max()) + 1 if len(codes) else 0
    ps = np.full(m, -np.inf)
    np.maximum.at(ps, codes, s)
    py = np.zeros(m, dtype=bool)
    py[codes[y]] = True
    used = np.zeros(m, dtype=bool)
    used[codes] = True
    return ps[used], py[used]


# --------------------------------------------------------------------------- #
# FP-budget rows
# --------------------------------------------------------------------------- #
def budget_rows(s, y, null, n_null_days, n_test_days, budgets=BUDGETS, n_total=None, day_index=None):
    """TP / FP / alerts per test day at each budget's bar. `day_index`: {day: universe positions present
    that day} (Host.day_index) for alerts per individual test day."""
    s, y = _check(s, y)
    n_total = _total(y, n_total)
    rows = {}
    for b in budgets:
        t = bar(null, n_null_days, b)
        if t is None:
            continue
        al = s >= t
        tp = int((al & y).sum())
        n_al = int(al.sum())
        row = {"bar": t if math.isfinite(t) else None, "alerts": n_al, "tp": tp, "fp": n_al - tp,
               "alerts_per_day": n_al / max(1, n_test_days),
               "fp_per_day": (n_al - tp) / max(1, n_test_days),
               "recall": tp / n_total if n_total else None}
        if day_index is not None:
            row["alerts_by_day"] = {d: int(al[idx].sum()) for d, idx in day_index.items()}
        rows[f"{b:g}"] = row
    return rows
