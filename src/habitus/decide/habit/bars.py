"""Habit-channel decision layer: clean-day bars, the two-tier per-day cap and alert / ranking rows.

    tau = bars(null_values, n_null_days)                  {0: tau_0, 1: tau_1, 20: tau_20}
    al = alerts(day_values, tau, 20)                       {day: AlertDay(strong, weak)}
    alert_rows(day_values, tau)                            counts per budget / tier / day + cap loss
    fixed_rate(day_values, 5), ranking(day_values)        top-n per day, TP-curve order

`day_values` = {test day: (unit ids, scores)} (score.day_values: every node of the day, head rows and
the rarity fallback).

Bars: tau_B = bar(null composite pool, n_null_days, B) = the smallest t with #{null units >= t} /
n_days <= B (decide.null.bar, the label-free bar; -inf when the budget covers the whole null). B = 0:
alerts {s >= tau_0}, one tier, uncapped. B = 1: {s >= tau_1}. B > 1 (B = 20): strong = {s >= tau_1}
(uncapped) + weak = at most `weak_cap` (default B; B = 20 -> WEAK_CAP = 20) nodes per test day from
[tau_B, tau_1), largest margin s - tau_B first (= largest s), ties by unit id. The weak bound holds by
construction (a code check, not a result); the cap is a contract mechanism, never a detection gain.
Uncapped rows ({s >= tau_B}) are always reported next to it, with the cap loss (uncapped alerts the
cap drops).

The best-MCC confusion tables do not use this module (oracle threshold, no budget, no cap); it serves
the FP-budget rows only.

Label-free: only clean-day bars and test-day score order; no label is read here.
"""

import math

import numpy as np

from habitus.decide.null import bar
from habitus.questions.habit import types as T
from habitus.questions.habit.constants import BARS, FIXED_ALERT_ROWS, TP_CURVE_MAX, WEAK_CAP

assert WEAK_CAP == max(BARS) == 20, "the B = 20 weak tier holds WEAK_CAP alerts per day"


# --------------------------------------------------------------------------- #
# Bars and alerts
# --------------------------------------------------------------------------- #
def bars(null_values, n_days, budgets=BARS):
    """{B: tau_B} from the null composite pool (node-day units) and its number of days. A missing
    null (no unit or no day) raises: no decision is made without clean-day bars."""
    v = np.asarray(null_values, dtype=np.float64)
    if not np.isfinite(v).all():
        raise ValueError(f"{int((~np.isfinite(v)).sum())} non-finite null values")
    out = {}
    for b in budgets:
        t = bar(v, n_days, b)
        if t is None:
            raise ValueError(f"no null ({len(v)} units over {n_days} days): no bar")
        out[int(b)] = float(t)
    return out


def _arrays(ids, s):
    ids, s = np.asarray(ids), np.asarray(s, dtype=np.float64)
    if ids.shape != s.shape or ids.ndim != 1:
        raise ValueError("one score per unit id")
    if not np.isfinite(s).all():
        raise ValueError(f"{int((~np.isfinite(s)).sum())} non-finite scores")
    return ids, s


def _order(ids, s):
    """Positions by descending score, ties by ascending id (ints or strings)."""
    return np.lexsort((ids, -s))


def _ids(a):
    return tuple(sorted(x for x in a.tolist()))


def alerts(day_values, bars, budget, weak_cap=None, capped=True):
    """{day: AlertDay} at budget B (module docstring). capped=False gives the uncapped single-tier
    row {s >= tau_B}. weak_cap None -> B (B = 20: WEAK_CAP)."""
    budget = int(budget)
    if budget not in bars:
        raise KeyError(f"no bar for budget {budget}: {sorted(bars)}")
    tb = bars[budget]
    two = capped and budget > 1
    if two:
        if 1 not in bars:
            raise KeyError("the two-tier rule needs tau_1")
        t1 = bars[1]
        if tb > t1:
            raise ValueError(f"tau_{budget} {tb} above tau_1 {t1}: bars of another null?")
        cap = budget if weak_cap is None else int(weak_cap)
        if cap < 0:
            raise ValueError(f"weak cap {cap}")
    out = {}
    for d in sorted(day_values):
        ids, s = _arrays(*day_values[d])
        if not two:
            out[d] = T.AlertDay(d, budget, _ids(ids[s >= tb]), (), dict(bars))
            continue
        band = (s >= tb) & (s < t1)
        bi, bs = ids[band], s[band]
        weak = bi[_order(bi, bs)[:cap]]
        out[d] = T.AlertDay(d, budget, _ids(ids[s >= t1]), _ids(weak), dict(bars))
    return out


def fixed_rate(day_values, per_day):
    """{day: the top `per_day` unit ids by descending score, ties by id} (ranked order)."""
    out = {}
    for d in sorted(day_values):
        ids, s = _arrays(*day_values[d])
        out[d] = tuple(ids[_order(ids, s)[:int(per_day)]].tolist())
    return out


def ranking(day_values, max_per_day=TP_CURVE_MAX):
    """{day: unit ids by descending score, ties by id, the first max_per_day} - the order the
    TP-vs-alerts/day curve 0..max_per_day reads (alerts/day n = the first n of every day)."""
    out = {}
    for d in sorted(day_values):
        ids, s = _arrays(*day_values[d])
        out[d] = ids[_order(ids, s)[:int(max_per_day)]]
    return out


def alert_rows(day_values, bars, budgets=BARS, fixed=FIXED_ALERT_ROWS, weak_cap=None):
    """The label-free half of the budget report (habitus.eval adds TP / FP from the ids):
    {"bars", "budgets": {B: {"uncapped": {day: n}, "strong": {day: n}, "weak": {day: n},
    "capped": {day: n}, "cap_loss": {day: n}, per-day means, "alerts": {day: ids}, "uncapped_alerts":
    {day: ids}}}, "fixed": {n: {day: ids}}}."""
    rows = {"bars": {int(b): (t if math.isfinite(t) else None) for b, t in bars.items()}, "budgets": {},
            "fixed": {int(n): fixed_rate(day_values, n) for n in fixed}}
    days = sorted(day_values)
    nd = max(1, len(days))
    for b in budgets:
        cap = alerts(day_values, bars, b, weak_cap=weak_cap, capped=True)
        unc = alerts(day_values, bars, b, capped=False)
        r = {"uncapped": {d: len(unc[d].strong) for d in days}, "strong": {d: len(cap[d].strong) for d in days},
             "weak": {d: len(cap[d].weak) for d in days}, "capped": {d: len(cap[d].alerts) for d in days},
             "alerts": {d: cap[d].alerts for d in days}, "uncapped_alerts": {d: unc[d].strong for d in days}}
        r["cap_loss"] = {d: r["uncapped"][d] - r["capped"][d] for d in days}
        for k in ("uncapped", "strong", "weak", "capped", "cap_loss"):
            r[f"{k}_per_day"] = sum(r[k].values()) / nd
        rows["budgets"][int(b)] = r
    return rows


__all__ = ["bars", "alerts", "fixed_rate", "ranking", "alert_rows"]
