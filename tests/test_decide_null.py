"""decide.null: LODO plan, null pools, lodo_scores and the label-free bar() (port of
test_kcs1.test_null_requires_lodo_reference_and_carries_caps minus the label-side caps_match)."""

import math
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from habitus.decide import null as N
from habitus.decide import scorefile as SF

CAPS = {"process": 4000, "file": 2000, "netflow": 6000}


class StubHost:
    """Day lists + node maps only (what lodo_plan / day_values read)."""

    def __init__(self, clean=("c1", "c2", "c3"), test=("t1",), ds="KCS1T"):
        self.ds, self.clean, self.test = ds, list(clean), list(test)
        rng = np.random.default_rng(7)
        self._nodes = {}
        for i, d in enumerate(self.clean + self.test):
            sigs = [f"{x:016x}" for x in rng.integers(0, 2 ** 40, size=6)]
            self._nodes[d] = {100 * i + j: sigs[j % len(sigs)] for j in range(9)}
        self.universe = np.array(sorted({n for d in self.test for n in self._nodes[d]}), dtype=np.int64)

    def shots(self, k):
        return list(self.clean) if k is None else self.clean[:k]

    def nodes(self, d):
        return self._nodes[d]


def _settings(tmp_path):
    return SimpleNamespace(paths=SimpleNamespace(data=str(tmp_path / "release"), scores=str(tmp_path / "release" / "scores"),
                                                 runs=str(tmp_path / "runs")))


def _write_day(host, root, level, k, day, ref, settings, value=0.5):
    sigs = sorted(set(host.nodes(day).values()))
    head_sigs = sigs[:3]
    obj = {"meta": {"ds": host.ds, "day": day, "level": level, "k": SF.klabel(k), "ref": list(ref), "role": "null",
                    "caps": CAPS},
           "head": {"process": head_sigs, "file": [], "netflow": []},
           "scores": {s: {"qv": value, "qt": value / 2, "cov_min": 1.0} for s in head_sigs},
           "fallback": {}}
    return SF.write_score(obj, SF.score_path(host.ds, level, k, day, root), settings)


def test_lodo_plan_and_null_days():
    host = StubHost()
    assert N.lodo_plan(host, 0) == [("c1", []), ("c2", []), ("c3", [])]
    assert N.lodo_plan(host, 2) == [("c1", ["c2"]), ("c2", ["c1"])]
    assert N.lodo_plan(host, None) == [("c1", ["c2", "c3"]), ("c2", ["c1", "c3"]), ("c3", ["c1", "c2"])]
    assert N.null_days(host, None) == ["c1", "c2", "c3"]
    with pytest.raises(ValueError):
        N.lodo_plan(StubHost(clean=("c1",)), 2)


def test_lodo_scores_requires_lodo_reference_and_carries_caps(tmp_path):
    host, root, st = StubHost(), str(tmp_path / "runs" / "scores"), _settings(tmp_path)
    plan = N.lodo_plan(host, 2)
    assert N.lodo_scores(host, 2, "L4", "composite", root) is None                # missing days: no null
    for c, ref in plan:
        _write_day(host, root, "L4", 2, c, ref, st)
    n = N.lodo_scores(host, 2, "L4", "composite", root)
    assert n["n_days"] == 2 and n["days"] == ["c1", "c2"] and n["caps"] == [CAPS, CAPS]
    v, nd = N.pool(host, SF.load_by_day(host.ds, "L4", 2, ["c1", "c2"], root)[0])
    assert nd == 2 and np.array_equal(v, n["values"]) and len(v) == 18
    assert (v[v >= 0] == 0.5).all() and (v[v < 0] > -1).all()                      # head value or fallback
    p = SF.score_path(host.ds, "L4", 2, "c1", root)                                # a null day with another
    raw = SF.read_score(p)                                                         # reference is dropped
    raw["meta"]["ref"] = ["c1", "c2"]
    SF.write_score(raw, p, st)
    assert N.lodo_scores(host, 2, "L4", "composite", root) is None
    assert N.pool(host, {})[1] == 0


def test_bar_hand_values():
    u = [0.5, 0.9, 0.6, 0.8, 0.7]
    assert N.bar(u, 1, 2) == float(np.nextafter(0.7, np.inf))                     # 0.9, 0.8 above the bar
    assert sum(x >= N.bar(u, 1, 2) for x in u) == 2
    assert N.bar(u, 1, 0) == float(np.nextafter(0.9, np.inf))                     # budget 0: nothing above
    assert N.bar(u, 2, 1) == float(np.nextafter(0.7, np.inf))                     # 1/day over 2 days = 2 units
    assert N.bar(u, 1, 5) == -math.inf and N.bar(u, 1, 7) == -math.inf            # the budget covers it all
    assert N.bar(u, 4, 0.5) == float(np.nextafter(0.7, np.inf))                   # floor(0.5 * 4) = 2
    assert N.bar(u, 3, 0.34) == float(np.nextafter(0.8, np.inf))                  # floor(1.02) = 1
    assert N.bar([], 3, 1) is None and N.bar(u, 0, 1) is None
    ties = [0.5, 0.5, 0.5, 0.1]
    t = N.bar(ties, 1, 1)                                                          # a tie block is never split
    assert t == float(np.nextafter(0.5, np.inf)) and sum(x >= t for x in ties) == 0
    assert N.bar(np.array([0.3]), 1, 0) == float(np.nextafter(0.3, np.inf))


def test_decide_never_imports_eval_or_labels():
    """The decide stage is label-free: importing it pulls in no habitus.eval module."""
    code = ("import sys; import habitus.decide.null, habitus.decide.scorefile, habitus.decide.composite, "
            "habitus.decide.inventory, habitus.decide.routed; "
            "print([m for m in sys.modules if m.startswith('habitus.eval')])")
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    env = dict(os.environ, PYTHONPATH=src + os.pathsep + os.environ.get("PYTHONPATH", ""))
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and r.stdout.strip().splitlines()[-1] == "[]", (r.stdout[-500:], r.stderr[-2000:])
