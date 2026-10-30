"""Profile2: exact-identity habit indices over the reference days.

    from habitus.questions.habit.profile2 import get_profile2, prefix_level, class_key
    prof = get_profile2(host, ["2018-04-02", "2018-04-03"])   # cached per (host, sorted ref)
    prof.habit(ident, Family.P4_children)    # {child ident key: reference days}
    prof.class_habit(Family.F1_writer, ("home", "document"))   # {writer ident key: days}
    prof.lineage_seen("spawn", parent_ident, child_kappa)       # lineage pair sets
    prof.check(day)                          # raises if day is one of the reference days

The routed state's HostProfile records habits under NAME keys (a process's lower-cased file_name), so a
masquerader picks which name's habits it is compared with. Profile2 keys every habit by the EXACT
identity instead (types.ident_key: kappa' = the process invariant minus parent_name; the path kappa
of a file; the connection kappa of a netflow), and turns names into evidence only through the
attestation tiers (attest.py). Everything is computed in ONE pass per reference day over that day's
cards and links; no label, score or tuned knob is read.

Indices (every value is a bitmask of reference days, exposed as a day count n, rendered "n/k"):
  identities   ident -> days seen, node count (frequency level per type, the state.profile rule
               HostProfile._levels reused as is), invariant (process: minus parent_name); full
               kappa -> days seen (LINEAGE needs the full kappa).
  habits       ident x family -> element -> days, elements exactly as targets.py observes them on a
               test day, all from a node's OWN complete op keys and links:
                 P1  parent idents: code-1 causes whose file_name equals the child's parent_name
                 P2  "dir:<class>"  op keys write / create / rename / delete:<dir class>
                 P3  "ep:<class>"   endpoint classes of code-4 netflow effects + op keys <verb>:net-<ep>
                 P4  child idents:  code-1 process effects
                 P5  "dir:<class>"  dir classes of code-3 file causes
                 F1  writer idents of a file: code-2 process causes
                 F2  executor idents of a file: code-3 process effects
                 N1  owner idents of a netflow: code-4 process causes (CADETS process -> file code-4
                     links end at a file and never count)
                 N2  habit(owner, N2) = the owner's reference chain elements (state.profile.ip_levels)
               Placeholder-bearing idents have habits too (the PATTERN-level habits): they are
               exact keys of the abstracted identity; the tier decides how a consumer may use them.
  lineage      (parent ident, child full kappa) spawn pairs [P1 links], (writer ident, path kappa)
               [= F1], (file ident, process full kappa) exec pairs, (owner ident, connection kappa)
               [= N1]: lineage_seen(kind, cause ident, effect kappa) for attest.lineage_tiers.
  chains       chain elements (/8, /16, /24, /32, ip|port; IPv6 /32 /48 /64 /128, ip|port) of every
               reference netflow card with an applicable chain (state.profile.chain_of), host-wide
               and per owner ident.
  class habits class_habit(family, cls) = the habits of every ident of the class pooled (OR of day
               masks); class_days(family, cls) = days on which an ident of the class occurred:
                 P1 / P2 / P3 / P5  cls = (role class, image_dir class) of the process
                 P4                 the same cls -> "cls:<child image_dir class>"
                 F1 / F2            cls = (dir class, ext class) of the file -> writer / executor idents
                 N1                 cls = (endpoint, port class, prefix level) -> owner idents
               The role class is vocab.role_class under the ident's own tier (IDENTITY unless
               placeholder-bearing or empty, which pool as role "unknown"). cls = () pools the whole
               host (host-common habits). A reference flow's prefix level is cross-fitted inside R:
               the leading chain elements also seen on ANOTHER reference day (a flow seen on >= 2
               days keeps its full chain), the analogue of prefix_level() of a test-day flow against
               all of R; with k = 1 every reference flow has level 0. Use class_key() for the class
               of a scored card, so that producer and consumer share one definition.

Fold assertions: a profile is built only from pre-deployment clean days (types.assert_predeploy:
usable clean days strictly before the host's first test day; never a test day, never a day >=
min(test)). role="train" (training-item export) refuses any host outside the training hosts, so no
other host can ever shape a model.

Label-free: cards and links of attack-free pre-deployment reference days only.
"""

import time
from collections import Counter, OrderedDict
from types import MappingProxyType, SimpleNamespace

from habitus.log import log
from habitus.questions.habit import attest, vocab
from habitus.questions.habit import types as T
from habitus.questions.habit.constants import TRAIN_HOSTS
from habitus.questions.habit.types import Family
from habitus.state.profile import EXEC, NET, SPAWN, WRITE, HostProfile, chain_of, host_key
from habitus.state.sections import DIR_CLASSES, basename

MAX_PROFILES = 8                         # LRU of built profiles
MODIFY_VERBS = frozenset(("write", "create", "rename", "delete"))     # P2 op-key verbs
ROLES = ("score", "train")
NO_LEVEL = "none"                        # N1 class level of a flow without an applicable chain
HOST_COMMON = ()                         # class key of the host-common habits
P1, P2, P3, P4, P5 = (Family.P1_parent, Family.P2_modify, Family.P3_peers, Family.P4_children,
                      Family.P5_exec_from)
F1, F2, N1, N2 = Family.F1_writer, Family.F2_executor, Family.N1_owner, Family.N2_address
IDENT_FAMILIES = frozenset((P1, P4, F1, F2, N1))      # habit() elements are ident keys
HABIT_FAMILIES = (P1, P2, P3, P4, P5, F1, F2, N1)     # N2 lives in the owner chains


def _or(store, key, elem, bit):
    d = store.get(key)
    if d is None:
        d = store[key] = {}
    d[elem] = d.get(elem, 0) | bit


def _counts(masks):
    return {e: m.bit_count() for e, m in sorted(masks.items())}


# --------------------------------------------------------------------------- #
# Fold assertions
# --------------------------------------------------------------------------- #
def check_ref(host, ref_days, role="score"):
    """Raise unless `ref_days` may serve as a reference of `host` (module docstring)."""
    ref = list(ref_days or ())
    if not ref:
        raise ValueError(f"{host.ds}: empty reference (k = 0 has no profile: pass prof=None)")
    if role not in ROLES:
        raise ValueError(f"role {role!r} not in {ROLES}")
    if role == "train":
        if host.ds not in TRAIN_HOSTS:
            raise ValueError(f"{host.ds} is not a training host {TRAIN_HOSTS}: no other host enters a "
                             f"training pool")
    tests = sorted(set(ref) & set(host.test))
    if tests:
        raise ValueError(f"{host.ds}: test days {tests} can never be reference days")
    T.assert_predeploy(host, ref)


# --------------------------------------------------------------------------- #
# Profile2
# --------------------------------------------------------------------------- #
class Profile2:
    """T.Profile2Like over the reference days `ref` of `host` (see the module docstring)."""

    def __init__(self, host, ref, role="score"):
        check_ref(host, ref, role)
        t0 = time.time()
        self.ds, self.os = host.ds, host.os
        self.ref = tuple(sorted(set(ref)))
        self.k = len(self.ref)
        self.obs = {"exec": False, "lineage": False, "parent_name": False}
        self._days, self._kdays, self._nodes = {}, {}, Counter()   # ident / kappa -> mask; ident -> nodes
        self._type, self._inv, self._cls = {}, {}, {}              # ident -> type / invariant / base class
        self._h = {f: {} for f in HABIT_FAMILIES}                  # family -> ident -> {element: mask}
        self._spawn, self._exec = {}, {}                           # effect kappa -> {cause ident: mask}
        self._chain, self._own, self._flow_chain = {}, {}, {}      # element -> mask; owner -> {element: mask}
        for i, d in enumerate(self.ref):
            self._absorb_day(host, d, 1 << i)
        self._lv = self._att = self._index = self._names = None
        self._class_cache, self._chain_cache = {}, {}
        self.seconds = time.time() - t0
        s = self.summary(attestable=False)
        log(f"{self.ds} ref={list(self.ref)}: idents {s['idents']}, kappas {s['kappas']}, "
            f"spawn/exec pairs {s['spawn_pairs']}/{s['exec_pairs']}, chain elements {s['chain_elements']}, "
            f"obs {self.obs} ({self.seconds:.1f}s)", "habit.profile2")

    # --- the one pass per reference day ------------------------------------------------------ #
    def _absorb_day(self, host, d, bit):
        nodes, cards = host.nodes(d), host.day(d)["cards"]
        L = host.links(d)
        if len(L):
            codes = L[:, 2]
            self.obs["exec"] = self.obs["exec"] or bool((codes == EXEC).any())
            self.obs["lineage"] = self.obs["lineage"] or bool((codes == SPAWN).any())
        info = {}                                                   # sig -> (type, ident, kappa)
        for s, n in sorted(Counter(nodes.values()).items()):
            card = cards[s]
            t = card["type"]
            idt, kap = attest.sig_keys(host, s)
            info[s] = (t, idt, kap)
            self._days[idt] = self._days.get(idt, 0) | bit
            self._kdays[kap] = self._kdays.get(kap, 0) | bit
            self._nodes[idt] += n
            inv = card["invariant"]
            if idt not in self._type:
                self._type[idt] = t
                self._inv[idt] = {k: v for k, v in inv.items() if k not in T.PARENT_FIELDS} \
                    if t == "process" and any(f in inv for f in T.PARENT_FIELDS) else inv
                self._cls[idt] = self._base_class(card)
            if t == "process":
                if inv.get("parent_name"):
                    self.obs["parent_name"] = True
                for op in (card.get("behaviour") or {}).get("ops") or {}:
                    verb, _, cls = op.partition(":")
                    if verb in MODIFY_VERBS and cls in DIR_CLASSES:
                        _or(self._h[P2], idt, "dir:" + cls, bit)
                    elif cls.startswith("net-"):
                        _or(self._h[P3], idt, "ep:" + vocab.canon_endpoint(cls[4:]), bit)
            elif t == "netflow":
                ch = chain_of(card)
                if ch is not None:
                    self._flow_chain.setdefault(idt, tuple(ch))
                    for e in ch:
                        self._chain[e] = self._chain.get(e, 0) | bit
        if not len(L):
            return
        for a, b, code in L.tolist():
            sa, sb = nodes.get(a), nodes.get(b)
            if sa is None or sb is None:
                continue
            ta, ia, _ka = info[sa]
            tb, ib, kb = info[sb]
            if code == SPAWN:
                if ta != "process" or tb != "process":
                    continue
                _or(self._h[P4], ia, ib, bit)
                if str(cards[sa]["invariant"].get("file_name") or "") == \
                        str(cards[sb]["invariant"].get("parent_name") or ""):
                    _or(self._h[P1], ib, ia, bit)                  # the recorded (first-spawn) parent
                    _or(self._spawn, kb, ia, bit)
            elif code == WRITE:
                if ta == "process" and tb == "file":
                    _or(self._h[F1], ib, ia, bit)
            elif code == EXEC:
                if ta == "file" and tb == "process":
                    _or(self._h[F2], ia, ib, bit)
                    _or(self._h[P5], ib, "dir:" + vocab.canon_dir(cards[sa]["invariant"].get("dir")), bit)
                    _or(self._exec, kb, ia, bit)
            elif code == NET:
                if ta == "process" and tb == "netflow":
                    _or(self._h[N1], ib, ia, bit)
                    _or(self._h[P3], ia, "ep:" + vocab.canon_endpoint(cards[sb]["invariant"].get("endpoint")), bit)
                    for e in self._flow_chain.get(ib, ()):
                        _or(self._own, ia, e, bit)

    def _base_class(self, card):
        """Class key of a reference card (netflows without their cross-fitted level, added later).
        The card's ident is seen by construction, so its tier is IDENTITY unless placeholder-bearing
        or empty (PATTERN)."""
        t, inv = card["type"], card["invariant"]
        if t == "process":
            tier = T.Attest.PATTERN if T.has_placeholder(card) or T.uninformative(card) else T.Attest.IDENTITY
            return vocab.role_class(card, tier), vocab.canon_dir(inv.get("image_dir"))
        if t == "file":
            return vocab.canon_dir(inv.get("dir")), vocab.ext_class(card)
        return vocab.canon_endpoint(inv.get("endpoint")), vocab.port_class(inv.get("remote_port"))

    # --- identities -------------------------------------------------------------------------- #
    def check(self, day):
        if day in self.ref:
            raise ValueError(f"{self.ds}: day {day} is a reference day of this profile {self.ref}")

    def seen_days(self, ident):
        return self._days.get(ident, 0).bit_count()

    def kappa_seen_days(self, kappa):
        return self._kdays.get(kappa, 0).bit_count()

    def tier0(self, card):
        return attest.tier0(card, self)

    def frequency(self, ident):
        """Frequency level of an ident by its node count over R among the idents of its type
        (state.profile.HostProfile._levels, reused unchanged): high | medium | low; unseen -> none."""
        if self._lv is None:
            self._lv = HostProfile._levels(SimpleNamespace(kappa_nodes=self._nodes, kappa_type=self._type))
        return self._lv.get(ident, "none")

    def attestable(self, ident):
        """True when `ident` is IDENTITY-attested by this reference (seen, placeholder-free and
        informative); the ident-level twin of tier0 (targets' gen:attested-other rule)."""
        if self._att is None:
            self._att = {}
        hit = self._att.get(ident)
        if hit is None:
            inv = self._inv.get(ident)
            pseudo = {"type": self._type.get(ident), "invariant": inv}
            hit = self._att[ident] = inv is not None and not T.has_placeholder(pseudo) and not T.uninformative(pseudo)
        return hit

    def ident_inv(self, ident):
        """Read-only invariant (process: minus parent_name) of an IDENTITY-attested ident, else None."""
        return MappingProxyType(self._inv[ident]) if self.attestable(ident) else None

    def ident_type(self, ident):
        return self._type.get(ident)

    def display(self, ident):
        """Short name of an IDENTITY-attested ident: a process's file_name (else its image
        basename), a file's basename of the abstracted path; None for netflows and every other ident."""
        inv = self.ident_inv(ident)
        if inv is None:
            return None
        return self._short_name(self._type[ident], inv)

    @staticmethod
    def _short_name(t, inv):
        if t == "process":
            return str(inv.get("file_name") or basename(inv.get("executable") or "") or "") or None
        if t == "file":
            return basename(inv.get("path") or "") or None
        return None

    def name_group(self, ident):
        """Sorted IDENTITY-attested idents of the WHOLE reference with the type and display name
        (None -> "?") of `ident`: views.display_names tells equal names apart against this group,
        not just inside one rendered list, so that a listed name always determines its ident (two
        signatures can render one view only when their listed idents agree; check f). A name held
        by one ident only is returned without the attestation scan (a non-attested ident is never
        listed, so it never needs telling apart). () for an unseen ident or a netflow."""
        if self._names is None:
            groups = {}
            for idt in sorted(self._type):
                t = self._type[idt]
                if t in ("process", "file"):
                    groups.setdefault((t, str(self._short_name(t, self._inv[idt]) or "?")), []).append(idt)
            self._names = {key: tuple(g) if len(g) == 1 else tuple(e for e in g if self.attestable(e))
                           for key, g in groups.items()}
        t = self._type.get(ident)
        if t not in ("process", "file"):
            return ()
        return self._names.get((t, str(self._short_name(t, self._inv[ident]) or "?")), ())

    # --- habits ------------------------------------------------------------------------------ #
    def habit(self, ident, family):
        """{element: reference days} of an exact ident (module docstring); N2: the owner's chain
        elements. {} for an unseen ident."""
        f = Family.of(family)
        if f is N2:
            return _counts(self._own.get(ident, {}))
        return _counts(self._h[f].get(ident, {}))

    def _class_index(self):
        """{node type: {class key: [idents]}}, with HOST_COMMON = every ident of the type."""
        if self._index is None:
            idx = {t: {HOST_COMMON: []} for t in T.NODE_TYPES}
            for idt in sorted(self._type):
                t = self._type[idt]
                cls = self._cls[idt]
                if t == "netflow":
                    cls = cls + (self._lodo_level(idt),)
                idx[t].setdefault(cls, []).append(idt)
                idx[t][HOST_COMMON].append(idt)
            self._index = idx
        return self._index

    def _lodo_level(self, idt):
        """Cross-fitted prefix level of a reference flow: leading chain elements seen on >= 2
        reference days, i.e. on a day other than the flow's own (module docstring)."""
        ch = self._flow_chain.get(idt)
        if ch is None:
            return NO_LEVEL
        n = 0
        for e in ch:
            if self._chain.get(e, 0).bit_count() < 2:
                break
            n += 1
        return str(n)

    def _class_masks(self, family, cls):
        f = Family.of(family)
        key = (f, tuple(cls))
        hit = self._class_cache.get(key)
        if hit is not None:
            return hit
        if f is N2:
            raise ValueError("N2 has no class habit (owner chains: chains(owner))")
        idents = self._class_index()[T.FAMILY_SPEC[f].node_type].get(tuple(cls), ())
        out, days = {}, 0
        store = self._h[f]
        for i in idents:
            days |= self._days[i]
            for e, m in store.get(i, {}).items():
                if f is P4:
                    e = "cls:" + vocab.canon_dir(self._inv[e].get("image_dir")) if e in self._inv else "cls:none"
                out[e] = out.get(e, 0) | m
        hit = self._class_cache[key] = (out, days)
        return hit

    def class_habit(self, family, cls):
        """{element: reference days} pooled over the idents of class `cls` (module docstring)."""
        return _counts(self._class_masks(family, cls)[0])

    def class_days(self, family, cls):
        """Reference days on which some ident of class `cls` (of the family's node type) occurred."""
        return self._class_masks(family, cls)[1].bit_count()

    def usual(self, key, family, n, cls=False):
        """Top-n (element, days) of habit(key, family) (cls=True: class_habit(family, key)) by
        (-days, element); ident-valued elements are restricted to identity-attested ones."""
        f = Family.of(family)
        src = self.class_habit(f, key) if cls else self.habit(key, f)
        items = list(src.items())
        if f in IDENT_FAMILIES and not (cls and f is P4):
            items = [(e, d) for e, d in items if self.attestable(e)]
        return sorted(items, key=lambda x: (-x[1], x[0]))[:max(0, int(n))]

    # --- lineage pairs, chains --------------------------------------------------------------- #
    def lineage_seen(self, kind, cause_ident, effect_kappa):
        """True when (cause ident, effect FULL kappa) occurred in role `kind` on a reference day:
        spawn (parent ident, child kappa), write (writer ident, path kappa), exec (file ident,
        process kappa), net (owner ident, connection kappa)."""
        if kind == "spawn":
            d = self._spawn.get(effect_kappa)
        elif kind == "exec":
            d = self._exec.get(effect_kappa)
        elif kind == "write":
            d = self._h[F1].get(effect_kappa)
        elif kind == "net":
            d = self._h[N1].get(effect_kappa)
        else:
            raise ValueError(f"lineage kind {kind!r} not in {attest.LINK_KINDS}")
        return bool(d) and cause_ident in d

    def chains(self, owner=None):
        """frozenset of reference chain elements: host-wide (owner None) or of one owner ident
        (whatever its tier)."""
        hit = self._chain_cache.get(owner)
        if hit is None:
            src = self._chain if owner is None else self._own.get(owner, {})
            hit = self._chain_cache[owner] = frozenset(src)
        return hit

    # --- reporting --------------------------------------------------------------------------- #
    def summary(self, attestable=True):
        """Sizes of every index (counts only), for logs and the probe report (attestable=False skips
        the per-ident placeholder scan)."""
        types = Counter(self._type.values())
        att = Counter(self._type[i] for i in self._type if self.attestable(i)) if attestable else {}
        return {"ds": self.ds, "ref": list(self.ref), "k": self.k,
                "idents": {t: types.get(t, 0) for t in T.NODE_TYPES}, "kappas": len(self._kdays),
                "attestable": {t: att.get(t, 0) for t in T.NODE_TYPES} if attestable else None,
                "habits": {f.value: {"idents": len(self._h[f]), "entries": sum(len(v) for v in self._h[f].values())}
                           for f in HABIT_FAMILIES},
                "spawn_pairs": sum(len(v) for v in self._spawn.values()),
                "exec_pairs": sum(len(v) for v in self._exec.values()),
                "chain_elements": len(self._chain), "owners_with_chains": len(self._own),
                "obs": dict(self.obs), "seconds": round(getattr(self, "seconds", 0.0), 2)}

    def __repr__(self):
        return f"Profile2({self.ds!r}, ref={list(self.ref)})"


# --------------------------------------------------------------------------- #
# Cache, class keys
# --------------------------------------------------------------------------- #
_CACHE = OrderedDict()


def get_profile2(host, ref_days, role="score"):
    """Profile2 cached per (state.profile.host_key(host), sorted ref) (LRU of MAX_PROFILES); every call
    re-checks the fold assertions (check_ref) before the cache is consulted."""
    check_ref(host, ref_days, role)
    ref = tuple(sorted(set(ref_days)))
    key = (*host_key(host), ref)
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    prof = _CACHE[key] = Profile2(host, ref, role=role)
    while len(_CACHE) > MAX_PROFILES:
        _CACHE.popitem(last=False)
    return prof


def clear_cache():
    _CACHE.clear()


def prefix_level(card, prof):
    """Leading elements of the netflow card's chain (state.profile.chain_of) found in the host's
    reference chains, 0..5; None without a reference or when the chain is not applicable
    (loopback / private endpoint, unusable address) or the card is not a netflow."""
    if prof is None or card["type"] != "netflow":
        return None
    ch = chain_of(card)
    if ch is None:
        return None
    seen = prof.chains(None)
    n = 0
    for e in ch:
        if e not in seen:
            break
        n += 1
    return n


def class_key(family, card, prof):
    """The class_habit key of a scored card for `family` (the one definition consumers use):
    process (role class under tier0, image_dir); file (dir class, ext class); netflow (endpoint,
    port class, str(prefix level) or NO_LEVEL)."""
    f = Family.of(family)
    t, inv = card["type"], card["invariant"]
    if T.FAMILY_SPEC[f].node_type != t:
        raise ValueError(f"{f.value} is a {T.FAMILY_SPEC[f].node_type} family, not {t}")
    if t == "process":
        return vocab.role_class(card, attest.tier0(card, prof)), vocab.canon_dir(inv.get("image_dir"))
    if t == "file":
        return vocab.canon_dir(inv.get("dir")), vocab.ext_class(card)
    lv = prefix_level(card, prof)
    return (vocab.canon_endpoint(inv.get("endpoint")), vocab.port_class(inv.get("remote_port")),
            NO_LEVEL if lv is None else str(lv))


__all__ = ["Profile2", "get_profile2", "clear_cache", "check_ref", "prefix_level", "class_key",
           "NO_LEVEL", "HOST_COMMON", "IDENT_FAMILIES", "MODIFY_VERBS", "MAX_PROFILES"]
