"""E2 state builder: one compiled host-reference state PER QUESTION (the main HABITUS system).

    from habitus.state.builder import StateBuilderE2, serialize, state_hash
    sb = StateBuilderE2(host, host.shots(2), 2)       # profile built once, cached
    states = sb.build("2018-04-12", sigs)             # {sig: {"qv": state, "qt": state[, "qa": state]}}
    text = serialize(states[sig]["qv"])               # compact JSON, schema key order

Routes (E2_ROUTES): QV reads entity, hard_bits, reference, deviation2; QT reads entity, hard_bits,
deviation2, evidence_nf; QA reads entity, reference, precedent2. Each state keeps SECTIONS order and
every string leaf goes through sanitize(). k = 0: QV = entity, hard_bits, reference stub {"days": 0};
QT = entity, hard_bits, evidence_nf (under "all"); no QA. QA is also absent when precedent2 is
"none" (then qa = 0). hard_tier(state) reads the QV state's HARD bits.

Reference and observability: a profile of the reference days (k >= 1) or, at k = 0, the
observability of the scored day itself. The builder never scores one of its own reference days, and
the same (host, ref, day, sigs) always give byte-identical serialisations.
"""

import hashlib
import json
import time

from habitus.state.precedent import get_refindex
from habitus.state.profile import get_profile
from habitus.state.sanitize import sanitize
from habitus.state.sections import (E2_ROUTES, HARD, SECTIONS, _observability, hard_tier,  # noqa: F401
                                    section_deviation2, section_entity, section_evidence_nf,
                                    section_hard_bits, section_precedent2, section_reference)


def builder_refindex(builder):
    """The builder's explicit RefIndex, else get_refindex's (global LRU of MAX_REFINDEX: a builder never
    keeps one alive)."""
    return builder._refindex if builder._refindex is not None else get_refindex(builder.host, builder.ref)


def builder_obs(builder, day):
    """Observability: of the reference days (k >= 1), of the scored day itself (k = 0)."""
    if builder.profile is not None:
        return builder.profile.obs
    return _observability(builder.host, builder.ref or (day,))


def serialize(state):
    """Compact JSON in schema key order (no sort_keys), sanitised, non-ASCII kept."""
    return json.dumps(sanitize(state), separators=(",", ":"), ensure_ascii=False)


def state_hash(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class StateBuilderE2:
    """E2 states of one (host, reference days): `.build(day, sigs) -> {sig: {"qv": state,
    "qt": state[, "qa": state]}}`, each state holding only the sections of E2_ROUTES[question].
    k = 0 <=> ref is empty; `k` is the nominal shot count (0, 2, "all"/None), only checked against
    `ref`; the reference size is len(ref)."""

    def __init__(self, host, ref, k, profile=None, refindex=None):
        self.host, self.k = host, k
        self.ref = tuple(sorted(set(ref or ())))
        if (str(k) == "0") != (not self.ref):
            raise ValueError(f"k={k!r} does not match reference days {list(self.ref)}")
        if profile is not None and tuple(profile.ref) != self.ref:
            raise ValueError(f"profile reference {profile.ref} != {self.ref}")
        self.profile = profile if profile is not None else (get_profile(host, self.ref) if self.ref else None)
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
        host, prof = self.host, self.profile
        sigs = list(dict.fromkeys(sigs))
        obs = self.obs(day)
        parts = {name: {} for name in SECTIONS}
        for s in sigs:
            card, kap = host.card(s), host.inv_key(s)
            parts["entity"][s] = section_entity(card, host.os)
            parts["hard_bits"][s] = section_hard_bits(card, obs)
            parts["reference"][s] = section_reference(prof, card, s, kap)
            dv = section_deviation2(prof, card, kap)
            if dv is not None:
                parts["deviation"][s] = dv
        t1 = time.time()
        parts["evidence"] = section_evidence_nf(host, day, sigs, prof)
        t2 = time.time()
        if prof is not None and prof.k:
            parts["precedent"] = section_precedent2(host, prof, self._refindex, sigs)
        t3 = time.time()
        out = {}
        for s in sigs:
            st = {}
            for q, secs in E2_ROUTES.items():
                if q == "qa" and parts["precedent"].get(s, {}).get("nearest_reference", "none") == "none":
                    continue
                st[q] = sanitize({name: parts[name][s] for name in SECTIONS if name in secs and s in parts[name]})
            out[s] = st
        self.timing = {"cards": t1 - t0, "evidence": t2 - t1, "precedent": t3 - t2,
                       "total": time.time() - t0, "n": len(sigs)}
        return out
