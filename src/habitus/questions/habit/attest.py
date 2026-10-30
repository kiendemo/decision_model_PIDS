"""Attestation tiers: how much of a node's identity the reference attests.

    from habitus.questions.habit import attest
    attest.tier0(card, prof)                          # NONE | PATTERN | IDENTITY (link-free)
    tiers = attest.lineage_tiers(host, day, prof)     # {node id: Attest}, LINEAGE where earned
    attest.census(host, day, prof)                    # {ntype: {tier: {"sigs", "nodes"}}}

Tiers (ordered; a higher tier implies every lower one):
  NONE      the ident key (kappa' for a process = its invariant minus parent_name; the full kappa
            for a file / netflow; types.ident_key) was never seen on a reference day, or k = 0.
  PATTERN   seen, but the card carries a cards.abstract_text placeholder (<hex:n>, <rand:n>,
            <guid>, <n>) in an invariant field, or every identity field is empty (types.uninformative:
            a path-less file, an address-less netflow, a nameless process). abstract_text keeps only
            a placeholder's length, so "seen" is evidence of a pattern, not of an identity.
  IDENTITY  seen and placeholder-free: raw strings may be rendered (vocab.entity_words), identity
            habits apply, the R0 null stratum flag (tier0 only; never reads a link).
  LINEAGE   IDENTITY, the FULL kappa (types.full_kappa = Host.inv_key, parent_name included) seen
            on a reference day, AND every causal parent present today attested in that role:
              process  spawn: the code-1 causes whose card file_name equals this card's parent_name
                       (the recorded first-spawn parent; links carry no time order): none -> vacuous,
                       one distinct ident -> (parent kappa', child kappa) must be a reference spawn
                       pair, more than one distinct ident -> ambiguous, not attested;
                       exec: every code-3 cause (file kappa, process kappa) must be a reference exec pair
              file     write: every code-2 cause (writer kappa', path kappa) must be a reference pair
              netflow  net: every code-4 cause (owner kappa', connection kappa) must be a reference
                       pair (CADETS process -> file "send" links end at a file and never count)
            and a causer counts only when it is itself IDENTITY-attested (CAUSER_MIN_TIER): a
            placeholder-bearing or empty parent / writer / owner matches a pattern, not an identity.
            A cause node without a card today, or of the wrong type, revokes (fail-safe).
            Only LINEAGE earns the full acquittal (decide.habit.score.composite); everything else
            gets C_K * qa.

Monotonicity (tested on fakes): with cards fixed, adding links or nodes to a day can only REVOKE
LINEAGE - an added link adds a cause, never removes one, and tier0 does not read links. Where lineage
is unobservable (CADETS, CLEARSCOPE: no spawn link) the process condition reduces to the full kappa
plus the placeholder rule; files and netflows still need their writer / owner pairs.

Label-free: card invariants, the day's links and the Profile2 reference indices (attack-free
pre-deployment days) only.
"""

import weakref

import numpy as np

from habitus.questions.habit import types as T
from habitus.questions.habit.types import PLACEHOLDER_RE, Attest, has_placeholder, uninformative
from habitus.state.profile import EXEC, NET, SPAWN, WRITE

PLACEHOLDER_KINDS = ("guid", "hex", "rand", "n")      # cards.abstract_text / identity.normalize_cmd tokens
TIER_NAMES = tuple(a.name for a in Attest)
# (effect node type, link code) -> the causal role a cause plays; other pairs are not causal parents
ROLE_OF = {("process", SPAWN): "spawn", ("process", EXEC): "exec", ("file", WRITE): "write",
           ("netflow", NET): "net"}
CAUSE_TYPE = {"spawn": "process", "exec": "file", "write": "process", "net": "process"}
LINK_KINDS = tuple(sorted(CAUSE_TYPE))
CAUSER_MIN_TIER = Attest.IDENTITY      # a causer below this tier never attests a lineage


# --------------------------------------------------------------------------- #
# Placeholder detector
# --------------------------------------------------------------------------- #
def placeholder_kinds(text):
    """Placeholder kinds present in `text`, in PLACEHOLDER_KINDS order (() when none)."""
    found = {m.group(1).split(":", 1)[0] for m in PLACEHOLDER_RE.finditer(str(text or ""))}
    return tuple(k for k in PLACEHOLDER_KINDS if k in found)


# --------------------------------------------------------------------------- #
# tier0 (link-free)
# --------------------------------------------------------------------------- #
def tier0(card, prof):
    """NONE | PATTERN | IDENTITY of one card against a reference profile (T.Profile2Like); NONE
    when prof is None (k = 0). Reads the invariant and the reference only."""
    if prof is None or prof.seen_days(T.ident_key(card)) <= 0:
        return Attest.NONE
    if has_placeholder(card) or uninformative(card):
        return Attest.PATTERN
    return Attest.IDENTITY


_KEYS = weakref.WeakKeyDictionary()     # host -> {sig: (ident key, full kappa)}


def sig_keys(host, sig):
    """(ident key, full kappa) of a card signature, memoised per host object (kappa via the host's
    own inv_key cache; ident = kappa unless a process card carries parent_name)."""
    memo = _KEYS.get(host)
    if memo is None:
        memo = _KEYS[host] = {}
    hit = memo.get(sig)
    if hit is None:
        card = host.card(sig)
        kap = host.inv_key(sig)
        idt = T.ident_key(card) if card["type"] == "process" and any(
            f in card["invariant"] for f in T.PARENT_FIELDS) else kap
        hit = memo[sig] = (idt, kap)
    return hit


class _SigTiers:
    """Per-call memo of tier0 / full-kappa-seen per signature (one host, one profile)."""

    def __init__(self, host, prof):
        self.host, self.prof = host, prof
        self._t0, self._ks = {}, {}

    def card(self, sig):
        return self.host.card(sig)

    def ident(self, sig):
        return sig_keys(self.host, sig)[0]

    def kappa(self, sig):
        return sig_keys(self.host, sig)[1]

    def tier0(self, sig):
        t = self._t0.get(sig)
        if t is None:
            card = self.host.card(sig)
            if self.prof.seen_days(self.ident(sig)) <= 0:
                t = Attest.NONE
            elif has_placeholder(card) or uninformative(card):
                t = Attest.PATTERN
            else:
                t = Attest.IDENTITY
            self._t0[sig] = t
        return t

    def kappa_seen(self, sig):
        v = self._ks.get(sig)
        if v is None:
            v = self._ks[sig] = self.prof.kappa_seen_days(self.kappa(sig)) > 0
        return v


# --------------------------------------------------------------------------- #
# LINEAGE
# --------------------------------------------------------------------------- #
def _in_role(st, kind, cause_sig, effect_kappa):
    if st.tier0(cause_sig) < CAUSER_MIN_TIER:
        return False
    return bool(st.prof.lineage_seen(kind, st.ident(cause_sig), effect_kappa))


def _revoked(host, day, prof, nd, cand, st):
    """Candidate node ids (IDENTITY + full kappa seen) whose causal parents today are not all
    attested in their role (module docstring). `nd` = host.nodes(day)."""
    L = host.links(day)
    if not len(L) or not len(cand):
        return set()
    rows = L[np.isin(L[:, 1], np.asarray(sorted(cand), dtype=np.int64))]
    bad, parents = set(), {}
    for c, e, code in rows.tolist():
        se = nd[e]
        ce = st.card(se)
        kind = ROLE_OF.get((ce["type"], code))
        if kind is None:
            continue
        sc = nd.get(c)
        if sc is None or st.card(sc)["type"] != CAUSE_TYPE[kind]:
            bad.add(e)                                    # an unknown or ill-typed causer: fail-safe
            continue
        if kind == "spawn":
            pn = str(ce["invariant"].get("parent_name") or "")
            if str(st.card(sc)["invariant"].get("file_name") or "") == pn:
                parents.setdefault(e, set()).add(sc)      # the recorded parent's candidates
            continue
        if e not in bad and not _in_role(st, kind, sc, st.kappa(se)):
            bad.add(e)
    for e in sorted(parents):
        if e in bad:
            continue
        sigs = parents[e]
        if len({st.ident(s) for s in sigs}) > 1:
            bad.add(e)                                    # ambiguous parent: not attested
        elif not all(_in_role(st, "spawn", s, st.kappa(nd[e])) for s in sorted(sigs)):
            bad.add(e)
    return bad


def lineage_tiers(host, day, prof, nodes=None):
    """{node id: Attest} for `nodes` of `day` (every node of the day when None), ascending ids.
    prof None (k = 0) -> all NONE. Refuses a reference day of `prof` (prof.check). Raises KeyError
    for an id that is not a node of the day."""
    nd = host.nodes(day)
    ids = sorted(nd) if nodes is None else sorted({int(n) for n in nodes})
    missing = [n for n in ids if n not in nd]
    if missing:
        raise KeyError(f"{host.ds} {day}: {len(missing)} ids are not nodes of the day, e.g. {missing[:3]}")
    if prof is None:
        return {n: Attest.NONE for n in ids}
    prof.check(day)
    st = _SigTiers(host, prof)
    out, cand = {}, []
    for n in ids:
        s = nd[n]
        t0 = st.tier0(s)
        if t0 == Attest.IDENTITY and st.kappa_seen(s):
            out[n] = Attest.LINEAGE
            cand.append(n)
        else:
            out[n] = t0
    for n in _revoked(host, day, prof, nd, cand, st):
        out[n] = Attest.IDENTITY
    return out


def census(host, day, prof):
    """Per-host tier census of one day: {ntype: {tier name: {"sigs": distinct signatures with >= 1
    node in the tier, "nodes": node count}}} over every node of the day (counts only)."""
    tiers = lineage_tiers(host, day, prof)
    nd = host.nodes(day)
    nodes = {t: {a: 0 for a in TIER_NAMES} for t in T.NODE_TYPES}
    sigs = {t: {a: set() for a in TIER_NAMES} for t in T.NODE_TYPES}
    for n, a in tiers.items():
        s = nd[n]
        t = host.card(s)["type"]
        nodes[t][a.name] += 1
        sigs[t][a.name].add(s)
    return {t: {a: {"sigs": len(sigs[t][a]), "nodes": nodes[t][a]} for a in TIER_NAMES} for t in T.NODE_TYPES}


__all__ = ["PLACEHOLDER_RE", "PLACEHOLDER_KINDS", "has_placeholder", "uninformative", "placeholder_kinds", "tier0",
           "sig_keys", "lineage_tiers", "census", "TIER_NAMES", "ROLE_OF", "CAUSE_TYPE",
           "LINK_KINDS", "CAUSER_MIN_TIER"]
