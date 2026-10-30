"""Per-node observed sets, buckets, candidates and the family-availability gate.

    from habitus.questions.habit import targets
    ok, why = targets.gate(ds, "P4", "k2", card)                    # family applicable on this host / card
    cands = {(s, f): targets.candidates(f, views[s][kind], prof, "k2") ...}   # f(view, reference) only
    obs = targets.observe(host, day, prof, sigs, "k2", cands)     # {node id: NodeObs}, every member
    obs[n].present[Family.P4_children]                            # frozenset of bucket keys, e.g.
                                                                   # {"id:<child kappa'>", "gen:never-seen"}

What a habit question asks about (S_f(v)) is read from ONE node's own complete op keys
(the card's "ops": every (verb, object class) of the node, never the TOP = 5 lists) and its own
links of the day (cards.py [cause, effect, code]: 1 spawn, 2 write, 3 exec, 4 net). Never from the
signature's representatives (lower-id decoys could blind them) and never from another member:
two nodes with one card share views and candidates, not observations.

    P1  choice    the ident (kappa') of the code-1 cause whose card file_name equals the node's
                  parent_name: usual (one of <= 4 listed) | attested-other | never-seen; no
                  parent_name and no code-1 cause -> none; anything else (0 or > 1 distinct
                  matching idents, causes without a parent_name) -> n/a. Links carry no time
                  order, so "the first spawn" cannot be recovered: ambiguity is n/a, never a guess.
    P2  dir:<w>   dir classes of the op keys write | create | rename | delete:<dir class>
    P3  ep:<w>    endpoint classes of the NETFLOW effects of the node's code-4 links, and of its op
                  keys <verb>:net-<endpoint> (CADETS process -> file "send" links never count)
    P4  id: / gen:  idents of the process effects of the node's code-1 links
    P5  dir:<w>   dir classes of the file causes of code-3 links whose effect is the node
    F1  id: / gen:  idents of the process causes of code-2 links whose effect is the file
    F2  id: / gen:  idents of the process effects of code-3 links whose cause is the file
    N1  id: / gen:  idents of the process causes of code-4 links whose effect is the netflow
    N2  per owner (the N1 causes): bucket id:<owner> (IDENTITY-attested) | gen:typed, observed option
                  same-address | same-network | other-programs | no-program (n2_option)

Element -> bucket (by EXACT identity, k >= 1): a listed identity of the (view, family)'s
candidates -> "id:<ident>"; else IDENTITY-attested on the reference (prof.ident_inv(e) is not None)
-> gen:attested-other; else (unseen, placeholder-bearing or empty identity) -> gen:never-seen, so a
masquerading child / writer / owner is never-seen whatever its name. k = 0: "cls:<image_dir class of
the other end>". Class families map to "dir:<w>" / "ep:<w>" (closed words, anything else clamped).

Candidates C_f are a function of (view, reference) ONLY: candidates() takes no host, day
or link, only view.meta and the profile's habit indices. Listed identities: <= MAX_LISTED
identity-attested elements by (-reference days, key) of P4 habit(kappa', P4), F1 / F2
class_habit((dir, ext class)), N1 class_habit((endpoint, port class, prefix level)); plus
gen:attested-other and gen:never-seen (<= 8 buckets). P1: <= MAX_P1_NAMES usual parents for option
A, alphabetical by name. N2: owner buckets are per node (NodeObs.owners). So every present bucket of
every member is a candidate key, and the union over a signature's members is <= 8 per family.

Gate (`gate`): the frozen host table (types.available: spawn absent on CADETS_E3 and CLEARSCOPE_E3;
exec only on CADETS_E3 and THEIA_E3), k (P1, N2 n/a at k = 0) and invariant fields (N2: n/a without a
prefix chain). Applicability never depends on test-day counts, so an attacker cannot switch a family
off. A host outside the table (test doubles) is measured on its pre-deployment link files
(`link_availability`: a code is present when some day has more than LINK_MIN_PER_DAY links of it).

Monotonicity (E_mono prerequisite, tested): for a fixed card, signature and reference, adding links
or op keys can only ADD present buckets of P2-P5, F1, F2, N1 and owners of N2; P1 (ambiguity is
link-dependent) is not monotone and is excluded from E_mono.

Label-free: reads the day's cards and links and the reference profile (pre-deployment clean days)
only; nothing here touches a label, a score or a test-day count beyond the node's own record.
"""

from collections import OrderedDict

import numpy as np

from habitus.questions.habit import types as T
from habitus.questions.habit import views as V
from habitus.questions.habit.constants import MAX_BUCKETS, MAX_LISTED, MAX_P1_NAMES
from habitus.questions.habit.types import (DIR_ORDER, EP_ORDER, GEN_ATTESTED_OTHER, GEN_NEVER_SEEN, GEN_TYPED,
                                           Bucket, Candidates, Family, OwnerObs)
from habitus.state.profile import EXEC, NET, SPAWN, WRITE, chain_of, host_key

MODIFY_VERBS = frozenset(("write", "create", "rename", "delete"))   # P2 op-key verbs
NO_PARENT = ""                      # p1_parent: no parent recorded (option "none"); None = n/a
LINK_MIN_PER_DAY = 10               # a link code with <= 10 links on every pre-deployment day is absent
LINK_NAMES = {SPAWN: "spawn", WRITE: "write", EXEC: "exec", NET: "net"}

# family -> (link code, the node's side: 0 = cause / 1 = effect, the other end's node type)
RELATION = {
    Family.P1_parent: (SPAWN, 1, "process"),
    Family.P3_peers: (NET, 0, "netflow"),
    Family.P4_children: (SPAWN, 0, "process"),
    Family.P5_exec_from: (EXEC, 1, "file"),
    Family.F1_writer: (WRITE, 1, "process"),
    Family.F2_executor: (EXEC, 0, "process"),
    Family.N1_owner: (NET, 1, "process"),
    Family.N2_address: (NET, 1, "process"),
}
IDENT_FAMILIES = (Family.P4_children, Family.F1_writer, Family.F2_executor, Family.N1_owner)

# --------------------------------------------------------------------------- #
# Bucket phrases (closed vocabulary; id: buckets name an identity-attested program)
# --------------------------------------------------------------------------- #
DIR_PHRASES = {"system": "system directories", "application": "installed application directories",
               "home": "user home directories", "staging": "staging or temporary directories",
               "cache": "cache directories", "config": "configuration or system state directories",
               "other": "other directories", "none": "locations with no recorded path"}
EP_PHRASES = {"loopback": "the loopback interface (this computer itself)", "private": "private network addresses",
              "enclave": "this computer's own /24 network", "internal": "the organisation's network (the same /16)",
              "external": "the external internet", "unknown": "endpoints of unknown class"}
GEN_PHRASES = {
    GEN_ATTESTED_OTHER: "a program whose exact identity was seen on this computer's reference days, "
                        "but not one of the usual ones",
    GEN_NEVER_SEEN: "a program whose exact identity was never seen on this computer's reference days",
    GEN_TYPED: "a program whose exact identity is not attested on this computer's reference days",
}
assert set(DIR_PHRASES) == set(DIR_ORDER) and set(EP_PHRASES) == set(EP_ORDER)
BUCKET_PHRASES = {**{f"dir:{w}": p for w, p in DIR_PHRASES.items()},
                  **{f"ep:{w}": p for w, p in EP_PHRASES.items()},
                  **{f"cls:{w}": (f"a program whose image is in {p}" if w != "none" else
                                  "a program with no recorded image location") for w, p in DIR_PHRASES.items()},
                  **GEN_PHRASES}


def id_phrase(name):
    """Bucket text of a listed identity-attested program (its attested short name)."""
    return f"the program {name}"


# --------------------------------------------------------------------------- #
# Gate
# --------------------------------------------------------------------------- #
def gate(ds, family, k_mode, card, obs=None):
    """(applicable, reason) of `family` for `card` on host `ds` under `k_mode` (reason "" when True):
    the node type, k (P1 / N2 need a reference), the host table (types.available; `obs` =
    {"exec", "lineage"} for a host outside it) and, for N2, the card's prefix chain."""
    f = Family.of(family)
    sp = f.spec
    if card["type"] != sp.node_type:
        return False, f"{f.value} is asked about {sp.node_type} nodes"
    if k_mode == "k0" and not sp.at_k0:
        return False, "no reference (k0)"
    if f not in T.available(ds, k_mode, obs):
        return False, f"host records no {sp.needs} links"
    if f is Family.N2_address and chain_of(card) is None:
        return False, "no prefix chain (private / loopback / unusable address)"
    return True, ""


_LINKS = OrderedDict()


def link_availability(host, days=None):
    """Link codes counted on `days` (default: the pre-deployment window): {"days", "per_day": {name:
    [count per day]}, "obs": {"lineage", "exec"}} with a code present when some day has more than
    LINK_MIN_PER_DAY links of it. Label-free (link files only); cached per (host, days)."""
    days = list(T.predeploy(host) if days is None else days)
    key = (*host_key(host), tuple(days))
    if key in _LINKS:
        return _LINKS[key]
    per = {name: [] for name in LINK_NAMES.values()}
    for d in days:
        c = np.bincount(host.links(d)[:, 2], minlength=max(LINK_NAMES) + 1)
        for code, name in LINK_NAMES.items():
            per[name].append(int(c[code]))
    obs = {"lineage": any(x > LINK_MIN_PER_DAY for x in per["spawn"]),
           "exec": any(x > LINK_MIN_PER_DAY for x in per["exec"])}
    out = _LINKS[key] = {"days": days, "per_day": per, "obs": obs}
    while len(_LINKS) > 32:
        _LINKS.popitem(last=False)
    return out


def host_obs(host):
    """The `obs` gate argument of a host: None for a host of the frozen table, the test double's
    declared `obs`, else the measured link_availability of its pre-deployment window."""
    if host.ds in T.MEASURED_HOSTS:
        return None
    o = getattr(host, "obs", None)
    return dict(o) if o is not None else link_availability(host)["obs"]


def family_table(host, k_mode="k2"):
    """{family value: applicable on this host under k_mode} (the frozen row, or the measured one)."""
    av = T.available(host.ds, k_mode, host_obs(host))
    return {f.value: f in av for f in T.FAMILIES}


# --------------------------------------------------------------------------- #
# Candidates: f(view, reference) only
# --------------------------------------------------------------------------- #
def _class_buckets(fam, prefix, words):
    return tuple(Bucket(fam, f"{prefix}{w}", BUCKET_PHRASES[f"{prefix}{w}"]) for w in words)


def _generic(fam):
    return tuple(Bucket(fam, g, GEN_PHRASES[g], True) for g in (GEN_ATTESTED_OTHER, GEN_NEVER_SEEN))


def candidates(family, view, prof, k_mode):
    """What (view, family) asks about (types.Candidates), from view.meta and the profile ONLY: never a
    link, an op key or anything of the scored day."""
    f = Family.of(family)
    sp = f.spec
    if view.kind is not sp.view:
        raise ValueError(f"{f.value} is asked on {sp.view.value}, not {view.kind.value}")
    if (k_mode == "k0") != (prof is None) or (k_mode == "k0") != (view.meta["k_mode"] == "k0"):
        raise ValueError(f"k_mode {k_mode} / view k_mode {view.meta['k_mode']} / profile {prof is not None}")
    if k_mode == "k0" and not sp.at_k0:
        return Candidates(f, applicable=False, reason="no reference (k0)")
    if f in (Family.P2_modify, Family.P5_exec_from):
        return Candidates(f, _class_buckets(f, "dir:", DIR_ORDER))
    if f is Family.P3_peers:
        return Candidates(f, _class_buckets(f, "ep:", EP_ORDER))
    if f is Family.N2_address:
        return Candidates(f)                                    # owner buckets are per node
    if k_mode == "k0":                                          # P4, F1, F2, N1 without a reference
        return Candidates(f, _class_buckets(f, "cls:", DIR_ORDER))
    meta = view.meta
    if f is Family.P1_parent:
        listed, _ = V.listed_idents(prof.habit(meta["ident"], f), prof, MAX_P1_NAMES)
        names = V.display_names([e for e, _ in listed], prof)
        order = sorted((names[e], e) for e, _ in listed)
        return Candidates(f, listed=tuple(Bucket(f, f"id:{e}", n) for n, e in order))
    if f is Family.P4_children:
        el = prof.habit(meta["ident"], f)
    elif f is Family.N1_owner:
        el = prof.class_habit(f, V.n1_class(meta))
    else:                                                       # F1, F2
        el = prof.class_habit(f, V.file_class(meta))
    listed, _ = V.listed_idents(el, prof, MAX_LISTED)
    names = V.display_names([e for e, _ in listed], prof)
    b = tuple(Bucket(f, f"id:{e}", id_phrase(names[e])) for e, _ in listed) + _generic(f)
    assert len(b) <= MAX_BUCKETS
    return Candidates(f, b)


def candidates_for(view, prof, k_mode, families):
    """{family: Candidates} of every family of `families` asked on `view`."""
    return {Family.of(f): candidates(f, view, prof, k_mode) for f in families
            if T.FAMILY_SPEC[Family.of(f)].view is view.kind}


# --------------------------------------------------------------------------- #
# Per-node observations
# --------------------------------------------------------------------------- #
def neighbours(links, ids, code, side):
    """{node id: sorted distinct other-end ids} over the links of `code` whose `side` (0 cause,
    1 effect) is one of `ids`."""
    if not len(links) or not len(ids):
        return {}
    L = links[links[:, 2] == code]
    L = L[np.isin(L[:, side], ids)]
    if not len(L):
        return {}
    a, b = L[:, side], L[:, 1 - side]
    o = np.lexsort((b, a))
    a, b = a[o], b[o]
    keep = np.ones(len(a), dtype=bool)
    keep[1:] = (a[1:] != a[:-1]) | (b[1:] != b[:-1])
    a, b = a[keep], b[keep]
    starts = np.flatnonzero(np.r_[True, a[1:] != a[:-1]])
    ends = np.r_[starts[1:], len(a)]
    return {int(a[s]): b[s:e].tolist() for s, e in zip(starts, ends)}


def n2_option(flow_card, owner_ident, prof):
    """N2 observed option of one (flow, owner): same-address if the owner's reference
    chains hold the flow's /32 (IPv6 /128) or ip|port element; same-network if they hold its /16 or
    /24 (IPv6 /48 or /64); other-programs if the host's chains hold its /16 (/48) or deeper;
    no-program otherwise. Owner chains are looked up by the owner's ident whatever its tier."""
    ch = chain_of(flow_card)
    if ch is None:
        raise ValueError("N2 is not applicable to a flow without a prefix chain")
    own, host = prof.chains(owner_ident), prof.chains(None)
    if ch[3] in own or ch[4] in own:
        return "same-address"
    if ch[1] in own or ch[2] in own:
        return "same-network"
    if any(x in host for x in ch[1:]):
        return "other-programs"
    return "no-program"


class _Day:
    """One scored day's lookups shared by observe / p1_parent: node -> sig, card, ident key."""

    def __init__(self, host, day):
        self.host, self.nodes = host, host.nodes(day)
        self._ident = {}

    def card(self, node):
        s = self.nodes.get(int(node))
        return None if s is None else self.host.card(s)

    def ident(self, node):
        s = self.nodes.get(int(node))
        if s is None:
            return None
        if s not in self._ident:
            self._ident[s] = T.ident_key(self.host.card(s))
        return self._ident[s]


def _p1_parent(card, causes, dx):
    """P1 rule over the node's code-1 causes (see the module docstring)."""
    pn = card["invariant"].get("parent_name")
    if not pn:
        return NO_PARENT if not causes else None
    match = set()
    for o in causes:
        c = dx.card(o)
        if c is not None and c["type"] == "process" and c["invariant"].get("file_name") == pn:
            match.add(dx.ident(o))
    return next(iter(match)) if len(match) == 1 else None


def p1_parent(host, day, node, card):
    """The parent ident (kappa') of `node` on `day` by the P1 rule; NO_PARENT ("") when no parent is
    recorded at all (option "none"); None when n/a (ambiguous or unmatched)."""
    L = host.links(day)
    causes = L[(L[:, 2] == SPAWN) & (L[:, 1] == int(node)), 0].tolist() if len(L) else []
    return _p1_parent(card, sorted(set(causes)), _Day(host, day))


def _ops_buckets(card, family):
    """Class buckets of the card's complete op keys: P2 dir:<w> of modify verbs, P3 ep:<w> of net-."""
    out = set()
    for op in (card.get("behaviour") or {}).get("ops") or {}:
        verb, _, cls = str(op).partition(":")
        if family is Family.P2_modify:
            if verb in MODIFY_VERBS and cls != "process" and not cls.startswith("net-"):
                out.add(f"dir:{V.dir_word(cls)}")
        elif cls.startswith("net-"):
            out.add(f"ep:{V.ep_word(cls[4:])}")
    return out


def _check_ref(prof, k_mode, day):
    if k_mode not in T.KMODES:
        raise ValueError(f"k_mode {k_mode!r} not in {T.KMODES}")
    if (k_mode == "k0") != (prof is None):
        raise ValueError(f"k_mode {k_mode} with {'no' if prof is None else 'a'} reference profile")
    if prof is not None:
        prof.check(day)


def sig_candidates(host, prof, k_mode, sig, families):
    """{family: Candidates} of one signature computed from its views (the observe default)."""
    card = host.card(sig)
    out = {}
    for kind, view in V.views_for(card, prof, k_mode, families).items():
        out.update(candidates_for(view, prof, k_mode, families))
    return out


def observe(host, day, prof, sigs, k_mode, cands=None):
    """{node id: NodeObs} for EVERY node of `day` whose signature is in `sigs`. `cands` =
    {(sig, family): Candidates} (computed from the views when None); with `cands` given, a family
    whose (sig, family) is missing or not applicable is n/a, except N2, whose owner buckets are per
    node (it needs no candidates; N2 views are checked by surprise.node_x). Families failing
    the gate are n/a. Every mapped bucket is a candidate key by construction (asserted, never
    filtered)."""
    _check_ref(prof, k_mode, day)
    dx = _Day(host, day)
    want = set(sigs)
    ids = np.asarray(sorted(n for n, s in dx.nodes.items() if s in want), dtype=np.int64)
    links = host.links(day)
    rel = {}
    for code, side, _t in RELATION.values():
        if (code, side) not in rel:
            rel[(code, side)] = neighbours(links, ids, code, side)
    obs = host_obs(host)
    k0 = k_mode == "k0"
    inv_cache, ops_cache = {}, {}

    def attested(e):
        if e not in inv_cache:
            inv_cache[e] = prof.ident_inv(e) is not None
        return inv_cache[e]

    per_sig = {}
    for s in sorted(want & set(dx.nodes.values())):
        card = host.card(s)
        fams = T.FAMILIES_OF[card["type"]]
        ok = [f for f in fams if gate(host.ds, f, k_mode, card, obs)[0]]
        if cands is None:
            cs = sig_candidates(host, prof, k_mode, s, ok)
        else:
            cs = {f: cands[(s, f)] for f in ok if (s, f) in cands}
        if Family.N2_address in ok and Family.N2_address not in cs:
            cs[Family.N2_address] = Candidates(Family.N2_address)  # owner buckets are per node
        cs = {f: c for f, c in cs.items() if c.applicable}
        per_sig[s] = (card, cs, frozenset(f for f in fams if f not in cs), {f: frozenset(c.keys()) for f, c in cs.items()})

    out = {}
    for n in ids.tolist():
        s = dx.nodes[n]
        card, cs, na, ckeys = per_sig[s]
        present, choice, owners, na = {}, {}, (), set(na)
        for f, c in cs.items():
            code, side, otype = RELATION.get(f, (None, None, None))
            others = [o for o in rel.get((code, side), {}).get(n, ()) if (dx.card(o) or {}).get("type") == otype] \
                if code is not None else []
            if f is Family.P1_parent:
                par = _p1_parent(card, rel[(SPAWN, 1)].get(n, []), dx)
                if par is None:
                    na.add(f)
                elif par == NO_PARENT:
                    choice[f] = "none"
                elif any(b.key == f"id:{par}" for b in c.listed):
                    choice[f] = "usual"
                else:
                    choice[f] = "attested-other" if attested(par) else "never-seen"
                continue
            if f is Family.N2_address:
                ow = set()
                for o in others:
                    e = dx.ident(o)
                    ow.add(OwnerObs(f"id:{e}" if attested(e) else GEN_TYPED, e, n2_option(card, e, prof)))
                owners = tuple(sorted(ow, key=lambda x: (x.bucket, x.owner, x.observed)))
                continue
            if f in (Family.P2_modify, Family.P3_peers):
                if (s, f) not in ops_cache:
                    ops_cache[(s, f)] = frozenset(_ops_buckets(card, f))
                got = set(ops_cache[(s, f)])
                if f is Family.P3_peers:
                    got |= {f"ep:{V.ep_word(dx.card(o)['invariant'].get('endpoint'))}" for o in others}
            elif f is Family.P5_exec_from:
                got = {f"dir:{V.dir_word(dx.card(o)['invariant'].get('dir'))}" for o in others}
            elif k0:                                                # P4, F1, F2, N1 at k = 0
                got = {f"cls:{V.dir_word(dx.card(o)['invariant'].get('image_dir'))}" for o in others}
            else:
                got = set()
                for o in others:
                    e = dx.ident(o)
                    got.add(f"id:{e}" if f"id:{e}" in ckeys[f] else GEN_ATTESTED_OTHER if attested(e) else GEN_NEVER_SEEN)
            bad = got - ckeys[f]
            if bad:
                raise AssertionError(f"node {n} {f.value}: buckets {sorted(bad)} are not candidates")
            present[f] = frozenset(got)
        out[n] = T.NodeObs(n, s, card["type"], dx.ident(n), present, choice, owners, frozenset(na))
    return out


def union_present(obs, sig):
    """{family: union of present buckets over the members of `sig`} (<= 8 per family by construction)."""
    out = {}
    for o in obs.values():
        if o.sig == sig:
            for f, b in o.present.items():
                out.setdefault(f, set()).update(b)
    return {f: frozenset(b) for f, b in out.items()}


def owner_buckets(obs, sig):
    """Sorted N2 owner bucket keys present on any member of `sig` (the V_own views to build)."""
    return sorted({ow.bucket for o in obs.values() if o.sig == sig for ow in o.owners})


__all__ = ["BUCKET_PHRASES", "DIR_PHRASES", "EP_PHRASES", "GEN_PHRASES", "NO_PARENT", "RELATION",
           "LINK_MIN_PER_DAY", "id_phrase", "gate", "link_availability", "host_obs", "family_table",
           "candidates", "candidates_for", "sig_candidates", "neighbours", "n2_option", "p1_parent",
           "observe", "union_present", "owner_buckets"]
