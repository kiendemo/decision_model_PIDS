"""E1 state ladder L0..L4: ONE cumulative state per signature (state-ablation table only).

ABLATION ONLY. The main system is E2 (habitus.state.builder.StateBuilderE2). This builder exists to
replay the shipped E1 score files behind the state-ablation table (results/habitus/ablation_e1_e2.csv)
and their E1 answer-cache keys; it is asked with SYSTEM (not SYSTEM_E2) and qv / qt only.

    sb = StateBuilder(host, host.shots(2), "L4", 2)
    states = sb.build(day, sigs)                     # {sig: state dict}, sanitised

Levels are cumulative (LEVELS); k = 0 runs K0_LEVELS only, and PRIMARY names the E1 primary level per k.
"""

import time

from habitus.state.builder import builder_obs, builder_refindex
from habitus.state.profile import get_profile
from habitus.state.sanitize import sanitize
from habitus.state.sections import (SECTIONS, section_deviation, section_entity, section_evidence,  # noqa: F401
                                    section_hard_bits, section_precedent, section_reference)

LEVELS = {
    "L0": ["entity"],
    "L1": ["entity", "hard_bits"],
    "L2": ["entity", "hard_bits", "reference", "deviation"],
    "L3": ["entity", "hard_bits", "reference", "deviation", "evidence"],
    "L4": ["entity", "hard_bits", "reference", "deviation", "evidence", "precedent"],
}
K0_LEVELS = ("L0", "L1", "L3")
PRIMARY = {0: "L3", 2: "L4", None: "L4"}


def levels_for(k, levels):
    return [lv for lv in levels if k != 0 or lv in K0_LEVELS]


class StateBuilder:
    """States of one (host, reference days, level). k = 0 <=> ref is empty; `k` is the nominal
    shot count (0, 2, "all"/None) and only checked against `ref`; the reference size is len(ref)."""

    def __init__(self, host, ref, level, k, profile=None, refindex=None):
        if level not in LEVELS:
            raise ValueError(f"level must be one of {list(LEVELS)}, not {level!r}")
        self.host, self.level, self.k = host, level, k
        self.ref = tuple(sorted(set(ref or ())))
        if (str(k) == "0") != (not self.ref):
            raise ValueError(f"k={k!r} does not match reference days {list(self.ref)}")
        self.secs = LEVELS[level]
        if profile is not None and tuple(profile.ref) != self.ref:
            raise ValueError(f"profile reference {profile.ref} != {self.ref}")
        needs = self.ref and any(s in self.secs for s in ("reference", "deviation", "evidence", "precedent"))
        self.profile = profile if profile is not None else (get_profile(host, self.ref) if needs else None)
        self._refindex = refindex

    @property
    def refindex(self):
        return builder_refindex(self)

    def obs(self, day):
        return builder_obs(self, day)

    def build(self, day, sigs):
        if day in self.ref:
            raise ValueError(f"{self.host.ds}: day {day} is one of the reference days {list(self.ref)}")
        if self.profile is not None:
            self.profile.check(day)
        t0 = time.time()
        host, secs, prof = self.host, self.secs, self.profile
        sigs = list(dict.fromkeys(sigs))
        obs = self.obs(day)
        parts = {name: {} for name in secs}
        for s in sigs:
            card = host.card(s)
            parts["entity"][s] = section_entity(card, host.os)
            if "hard_bits" in secs:
                parts["hard_bits"][s] = section_hard_bits(card, obs)
            if "reference" in secs:
                parts["reference"][s] = section_reference(prof, card, s, host.inv_key(s))
            if "deviation" in secs:
                dv = section_deviation(prof, card)
                if dv is not None:
                    parts["deviation"][s] = dv
        t1 = time.time()
        if "evidence" in secs:
            parts["evidence"] = section_evidence(host, day, sigs, prof)
        t2 = time.time()
        if "precedent" in secs and prof is not None and prof.k:
            parts["precedent"] = section_precedent(host, prof, self._refindex, sigs)
        t3 = time.time()
        out = {}
        for s in sigs:
            out[s] = sanitize({name: parts[name][s] for name in secs if s in parts[name]})
        self.timing = {"cards": t1 - t0, "evidence": t2 - t1, "precedent": t3 - t2,
                       "total": time.time() - t0, "n": len(sigs)}
        return out
