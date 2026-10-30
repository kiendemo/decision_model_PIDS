"""Host profile: what ONE host did on its attack-free reference days.

    from habitus.state.profile import HostProfile, get_profile, observability
    obs  = observability(host, days)        # {"exec", "lineage", "parent_name"}: bools
    prof = get_profile(host, host.shots(2)) # cached per (host, ref)
    prof.check(day)                         # raises if day is one of the reference days

The profile is the host-specific benign knowledge that LLM-PIDS papers paste into a generative
prompt (frequency levels, a baseline record, "usual interactions"), compiled once into plain sets
and counts so that the state builder can state it as short facts. Everything is computed from the
reference days' cards and links only: no label, no score, no tuned knob. A reference day is a shot
(Host.shots(k)); k = 0 has no profile at all.

  kappa_days   kappa (identity key, Host.inv_key) -> number of reference days it appears on
  kappa_level  "high" | "medium" | "low" by rank of the kappa's total node count over the
               reference days among the kappas of the same type (top 20% high, to 50% medium,
               rest low). Ties take the LOWEST position of their tie block, so a level never
               depends on hash order and frequency is never overstated (n = 1 -> high).
               Unseen kappa -> "none".
  sig_seen     full card signatures seen (the identity AND its day behaviour were seen)
  name_days    (type, name) -> reference days. name = process file_name.lower(); file (dir, ext)
               class pair; netflow (endpoint, port class).
  ops_max      (type, name) -> {op key: max bin ordinal} over the reference cards of that name;
  children / write_dirs / exec_dirs / peers   process (type, name) -> set union.
  pair_count   Counter[(parent file_name, child file_name)] over spawn links (code 1) of the
               reference days, both ends read from their own cards (lower-cased), and pair_total
               N. None when lineage is unobservable on the reference.
  ip_chains    every element of ip_levels(remote_ip, remote_port) (/8, /16, /24, /32, ip|port) of
               the reference days' netflow cards. Not applicable (None, rendered "n/a"): loopback /
               private endpoint classes and the addresses ip_levels rejects.

Observability is a host fact, not a node fact: a code that never occurs (no exec on Windows
collectors, no spawn on CADETS) makes the related state fields "n/a", never "no".

E2: common_prefix (precedent2 prefix_match). E2 keeps E1's netflow name key (endpoint, port class)
for every name_* statistic; a netflow's address novelty reaches E2 through chain_unseen
(deviation2's remote_prefix_unseen_levels) and common_prefix.
"""

import ipaddress
import time
from collections import Counter, OrderedDict

from habitus.log import log

BINS = ("0", "1", "2-7", "8-31", "32-127", "128+")
BIN_ORD = {b: i for i, b in enumerate(BINS)}
SPAWN, WRITE, EXEC, NET = 1, 2, 3, 4          # cards.LINK_CODES
LEVEL_CUTS = ((0.2, "high"), (0.5, "medium"))  # frequency levels, else "low"
MAX_PROFILES = 8                               # cached profiles (LODO references included)
PRIVATE_ENDPOINTS = ("loopback", "private")
IPV6_LEVELS = (32, 48, 64, 128)


def host_key(host):
    """(ds, data location) that keys the per-host LRUs (profiles, day indices, RefIndex, observability)."""
    return (host.ds, getattr(host, "art", None))


def ip_levels(remote_ip, remote_port):
    """KAIROS prefix chain of a public address, None when the address is not applicable."""
    try:
        ip = ipaddress.ip_address(str(remote_ip).strip())
    except ValueError:
        return None
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast \
            or ip.is_unspecified or ip.is_reserved:
        return None
    if ip.version == 4:
        o = str(ip).split(".")
        chain = [".".join(o[:i]) for i in range(1, 5)]
    else:
        chain = [str(ipaddress.ip_network(f"{ip}/{p}", strict=False)) for p in IPV6_LEVELS]
    return chain + [f"{ip}|{remote_port}"]


def observability(host, days):
    """{"exec", "lineage", "parent_name"} over `days`: any exec link, any spawn link, any process
    card carrying parent_name. Label-free; for k = 0 pass the test day itself."""
    ex = lin = par = False
    for d in days:
        codes = host.links(d)[:, 2]
        ex = ex or bool((codes == EXEC).any())
        lin = lin or bool((codes == SPAWN).any())
        if not par:
            par = any(c["type"] == "process" and c["invariant"].get("parent_name")
                      for c in host.day(d)["cards"].values())
    return {"exec": ex, "lineage": lin, "parent_name": par}


def port_class(port):
    try:
        p = int(str(port).strip())
    except (TypeError, ValueError):
        return "none"
    if p < 0:
        return "none"
    return "well-known" if p < 1024 else "registered" if p < 49152 else "ephemeral"


def name_key(card):
    """(type, name): the coarse name the host's habits are recorded under."""
    t, inv = card["type"], card["invariant"]
    if t == "process":
        return t, str(inv.get("file_name") or "").lower()
    if t == "file":
        return t, (inv.get("dir", ""), inv.get("ext", ""))
    return t, (inv.get("endpoint", ""), port_class(inv.get("remote_port")))


def chain_of(card):
    """KAIROS prefix chain of a netflow card, None when not applicable (private / unusable)."""
    inv = card["invariant"]
    if inv.get("endpoint") in PRIVATE_ENDPOINTS:
        return None
    return ip_levels(inv.get("remote_ip"), inv.get("remote_port"))


class HostProfile:
    def __init__(self, host, ref):
        t0 = time.time()
        self.ds, self.ref = host.ds, tuple(sorted(set(ref or ())))
        self.k = len(self.ref)
        self.obs = observability(host, self.ref) if self.ref else None
        self.kappa_days, self.kappa_nodes, self.kappa_type = Counter(), Counter(), {}
        self.sig_seen, self.name_days = set(), Counter()
        self.ops_max, self.children, self.write_dirs, self.exec_dirs, self.peers = {}, {}, {}, {}, {}
        self.ip_chains = set()
        lineage = bool(self.obs and self.obs["lineage"])
        self.pair_count, self.pair_total = (Counter(), 0) if lineage else (None, None)
        fname = {}                                       # sig -> lower-cased process file_name
        for d in self.ref:
            nodes, cards = host.nodes(d), host.day(d)["cards"]
            kd, nd = set(), set()
            for s, n in Counter(nodes.values()).items():
                card = cards[s]
                kap = host.inv_key(s)
                self.kappa_nodes[kap] += n
                self.kappa_type[kap] = card["type"]
                kd.add(kap)
                nk = name_key(card)
                nd.add(nk)
                if s not in self.sig_seen:
                    self.sig_seen.add(s)
                    self._absorb(nk, card)
            self.kappa_days.update(kd)
            self.name_days.update(nd)
            if lineage:
                L = host.links(d)
                for a, b in L[L[:, 2] == SPAWN, :2].tolist():
                    sa, sb = nodes.get(a), nodes.get(b)
                    if sa is None or sb is None or cards[sa]["type"] != "process" \
                            or cards[sb]["type"] != "process":
                        continue
                    for s in (sa, sb):
                        if s not in fname:
                            fname[s] = str(cards[s]["invariant"].get("file_name") or "").lower()
                    self.pair_count[(fname[sa], fname[sb])] += 1
                    self.pair_total += 1
        self.kappa_level = self._levels()
        self.seconds = time.time() - t0
        log(f"{self.ds} ref={list(self.ref)}: {len(self.kappa_days)} kappas, "
            f"{len(self.sig_seen)} sigs, {len(self.name_days)} names, {len(self.ip_chains)} chain "
            f"elements, spawn pairs {self.pair_total}, obs {self.obs} ({self.seconds:.1f}s)", "state.profile")

    def _absorb(self, nk, card):
        beh = card.get("behaviour") or {}
        om = self.ops_max.setdefault(nk, {})
        for op, b in (beh.get("ops") or {}).items():
            o = BIN_ORD.get(b, 0)
            if o > om.get(op, -1):
                om[op] = o
        t = card["type"]
        if t == "process":
            for field, store in (("children", self.children), ("write_dirs", self.write_dirs),
                                 ("exec_dirs", self.exec_dirs)):
                store.setdefault(nk, set()).update(beh.get(field) or ())
            self.peers.setdefault(nk, set()).update((beh.get("peers") or {}).keys())
        elif t == "netflow":
            lv = chain_of(card)
            if lv is not None:
                self.ip_chains.update(lv)

    def _levels(self):
        """Rank rule per type; a tie block takes the level of its last position."""
        by_type = {}
        for kap, n in self.kappa_nodes.items():
            by_type.setdefault(self.kappa_type[kap], []).append(n)
        level = {}
        for t, counts in by_type.items():
            m = len(counts)
            ge = Counter(counts)                         # count -> multiplicity
            at_or_above, lv_of = 0, {}
            for c in sorted(ge, reverse=True):
                at_or_above += ge[c]
                pos = (at_or_above - 1) / m              # 0-based last position of the tie block
                lv_of[c] = next((name for cut, name in LEVEL_CUTS if pos < cut), "low")
            for kap, n in self.kappa_nodes.items():
                if self.kappa_type[kap] == t:
                    level[kap] = lv_of[n]
        return level

    # --- queries ---------------------------------------------------------------------------- #
    def check(self, day):
        if day in self.ref:
            raise ValueError(f"{self.ds}: day {day} is a reference day of this profile {self.ref}")

    def level(self, kappa):
        return self.kappa_level.get(kappa, "none")

    def chain_unseen(self, card):
        """(#unseen chain levels, #levels) of a netflow card, None when not applicable."""
        lv = chain_of(card)
        if lv is None:
            return None
        return sum(1 for x in lv if x not in self.ip_chains), len(lv)

    def __repr__(self):
        return f"HostProfile({self.ds!r}, ref={list(self.ref)})"


_CACHE = OrderedDict()


def get_profile(host, ref):
    """HostProfile cached per (host, sorted ref) in memory (LRU of MAX_PROFILES)."""
    key = (*host_key(host), tuple(sorted(set(ref or ()))))
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    prof = _CACHE[key] = HostProfile(host, key[2])
    while len(_CACHE) > MAX_PROFILES:
        _CACHE.popitem(last=False)
    return prof


# --------------------------------------------------------------------------- #
# E2: how deep two netflow addresses agree
# --------------------------------------------------------------------------- #
def common_prefix(card_a, card_b):
    """Deepest common chain level (0-5) of two netflow cards (precedent2 prefix_match); None when
    either chain is not applicable. Each level contains the one above it, so this is the count of
    leading equal elements."""
    a, b = chain_of(card_a), chain_of(card_b)
    if a is None or b is None:
        return None
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n
