"""Label-free clean-day nulls for FP-budget bars.

A null is a pool of NODE-DAY scores on attack-free days, scored exactly as a test day is (same head
rule, same level, same composite), plus the number of days it spans; `bar(null, n_days, B)` then
gives the score above which at most B null units per day fall.

  k = all   LODO: each clean day c is scored with reference = shots(None) minus c
  k = 2     each of the two shot days is scored with the OTHER shot day as reference (a 1-day
            reference for a 2-day test reference: declared mismatch, the null holds 2 days)
  k = 0     the host's own clean days, scored with no reference

Clean-day score files sit next to the test-day ones (`<root>/<ds>/<level>_k<k>/<clean day>.json.gz`):
the reference of a clean day is fully fixed by k, so the directory is unambiguous. A null is only
built when every clean day has a file whose meta reference is lodo_plan's (a null day uses the
reference construction of the test day it stands in for); otherwise None and a log line saying
which days are missing or mis-referenced. The files' head caps travel with the null ("caps", one
per day) so that the caller can require them equal to the test days' caps (same head rule).

bar() lives here because it is clean-day null calibration: the habit channel and habitus.eval
both use this one copy, so a guarded (label-free) command never needs to import habitus.eval.
Nothing here reads labels.
"""

import math

import numpy as np

from habitus.decide.scorefile import day_values, klabel, load_by_day
from habitus.log import log


def lodo_plan(host, k):
    """[(clean day, its reference days)] for the null at k (2 or None = all)."""
    if k == 0:
        return [(c, []) for c in host.shots(None)]
    shots = host.shots(k)
    if len(shots) < 2:
        raise ValueError(f"{host.ds}: LODO needs >= 2 shot days, have {shots}")
    return [(c, [s for s in shots if s != c]) for c in shots]


def null_days(host, k):
    return [c for c, _ref in lodo_plan(host, k)]


def pool(host, by_day, field="composite"):
    """(node-day null values, number of days) from {clean day: {sig: rec}}."""
    vals = [day_values(host, d, recs, field)[1] for d, recs in sorted(by_day.items())]
    return (np.concatenate(vals) if vals else np.zeros(0)), len(vals)


def lodo_scores(host, k, level, field="composite", root=None):
    """The host's own null at k: {"values", "n_days", "days", "caps": [head caps per day]}. None
    (logged) when a clean day's score file is missing or was scored with another reference than
    lodo_plan's. root None = settings.paths.scores."""
    plan = lodo_plan(host, k)
    days, metas = [c for c, _ref in plan], {}
    by_day, missing = load_by_day(host.ds, level, k, days, root, metas)
    what = f"[decide.null] {host.ds} {level} k={klabel(k)} under {root if root is not None else 'release'}"
    if missing:
        log(f"{what}: clean days {missing} have no score file, no null")
        return None
    bad = {c: metas[c].get("ref") for c, ref in plan if sorted(metas[c].get("ref") or []) != sorted(ref)}
    if bad:
        log(f"{what}: clean days scored with another reference than LODO's {dict(plan)}: {bad}, null dropped")
        return None
    v, n = pool(host, by_day, field)
    return {"values": v, "n_days": n, "days": days, "caps": [metas[c].get("caps") for c in days]}


def bar(null, n_days, budget):
    """Smallest t with #{null units >= t} / n_days <= budget; -inf when the budget covers the
    whole null; None without a null."""
    u = np.sort(np.asarray(null, dtype=np.float64))[::-1]
    if not len(u) or n_days <= 0:
        return None
    m = int(math.floor(budget * n_days + 1e-9))
    if m >= len(u):
        return -math.inf
    return float(np.nextafter(u[m], np.inf))
