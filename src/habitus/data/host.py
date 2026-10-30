"""Host: one host's evidence cards, causal links and days, read lazily (the data contract of the
state builders, the decision layer and eval).

    from habitus.data.host import Host
    h = Host("CADETS_E3")                  # cards from settings.paths.cards
    h.clean, h.test, h.shots(2)            # frozen day lists (habitus.hosts), date order
    h.nodes("2018-04-12")[node_id]         # -> card signature
    h.card(sig), h.inv_key(sig)            # card, identity key kappa
    h.universe, h.utype, h.day_index(d)    # test-node universe (sorted ids), type, per-day members
    Host("CADETS_E3", source="nodeindex")  # the string-free node index: no card(), day() or links()

Days come from the frozen HostSpec (clean, test). A listed day whose file is missing, or holds no
node, raises MissingDay: the frozen lists and the files must agree (no silent skip).

Universe: every node id that has a card on >= 1 test day, sorted (the ids scores are evaluated
over). Its type is the type of its card on its first test day; `day_index(d)` gives the universe
positions present on test day d, from which `node_days` (node -> its test days) is derived.

Identity key: inv_key(sig) = kappa = signature({type, invariant}), the part of a card an
edge-adding attacker cannot change; `ukey` is kappa per universe node (first test day's card).

Memory: day files are parsed on demand and kept in an LRU of `max_days` (THEIA's 2018-04-12 is
~0.5 GB of Python objects); links in their own LRU. Cards are interned in one host-wide store as days
are read, so `card(sig)` works after the day that holds it is evicted and the same card is never
held twice. Nothing here reads ground truth.

Vendored from the original reflex/anchor/host.py (HostData): the same API and algorithms, without
the PIDSMaker config, the graph-size day enumeration and the ground-truth accessor.
"""

import os
from collections import OrderedDict

import numpy as np

from habitus import hosts
from habitus.data.cards import card_path, link_path, signature
from habitus.io import load_json
from habitus.log import log

SMALL_FILE = 4096       # bytes: a card file this small may be an empty day (checked by parsing it)
EMPTY_LINKS = np.zeros((0, 3), dtype=np.int64)
SOURCES = ("cards", "nodeindex")


class MissingDay(FileNotFoundError):
    """A day listed in the HostSpec has no file, or its file holds no node."""


class _LRU:
    def __init__(self, size):
        self.size, self.data = max(1, int(size)), OrderedDict()

    def get(self, key):
        if key in self.data:
            self.data.move_to_end(key)
            return self.data[key]
        return None

    def put(self, key, value):
        self.data[key] = value
        self.data.move_to_end(key)
        while len(self.data) > self.size:
            self.data.popitem(last=False)


def nodeindex_path(nodeindex_dir, ds, day):
    return os.path.join(os.fspath(nodeindex_dir), ds, f"{day}.json.gz")


class Host:
    def __init__(self, ds, *, source="cards", cards_dir=None, nodeindex_dir=None, spec=None, max_days=4,
                 max_links=8, settings=None):
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
        if spec is None:
            spec = hosts.spec(ds)                     # ForbiddenDataset outside the allow-list
        elif spec.ds != ds:
            raise ValueError(f"spec is for {spec.ds!r}, not {ds!r}")
        self.spec, self.ds, self.os, self.source = spec, ds, spec.os, source
        if source == "cards" and cards_dir is None or source == "nodeindex" and nodeindex_dir is None:
            if settings is None:
                from habitus.config import settings as _settings
                settings = _settings()
            cards_dir = cards_dir if cards_dir is not None else settings.paths.cards
            nodeindex_dir = nodeindex_dir if nodeindex_dir is not None else settings.paths.nodeindex
        self.cards_dir = os.fspath(cards_dir) if cards_dir is not None else None
        self.nodeindex_dir = os.fspath(nodeindex_dir) if nodeindex_dir is not None else None
        # cache-key identity of the data this host reads (state caches key on (host.ds, host.art))
        self.art = self.cards_dir if source == "cards" else self.nodeindex_dir
        self._init_stores(max_days, max_links)
        self.clean = self._existing(spec.clean)
        self.test = self._existing(spec.test)
        log(f"{ds} ({source}): {len(self.clean)} clean days {self.clean[:1]}..{self.clean[-1:]}, "
            f"test days {self.test}", "data.host")

    def _init_stores(self, max_days=4, max_links=8):
        self._days, self._links = _LRU(max_days), _LRU(max_links)
        self._cards, self._merged, self._kappa, self._types = {}, set(), {}, {}
        self._u = None

    # --- files ----------------------------------------------------------------------------- #
    def _path(self, d):
        if self.source == "cards":
            return card_path(self.cards_dir, self.ds, d)
        return nodeindex_path(self.nodeindex_dir, self.ds, d)

    def _existing(self, days):
        """The listed days, checked: MissingDay when a file is absent or holds no node."""
        out = []
        for d in days:
            p = self._path(d)
            if not os.path.exists(p):
                raise MissingDay(f"{self.ds} {d}: listed day has no {self.source} file {p}")
            if os.path.getsize(p) < SMALL_FILE and not load_json(p)["nodes"]:
                raise MissingDay(f"{self.ds} {d}: listed day's {self.source} file {p} has no node")
            out.append(d)
        return out

    def _need_cards(self, what):
        if self.source != "cards":
            raise NotImplementedError(f"Host({self.ds!r}, source={self.source!r}) has no {what}: "
                                      f"the node index carries no card text and no links")

    # --- days ------------------------------------------------------------------------------ #
    def _load(self, d):
        hit = self._days.get(d)
        if hit is not None:
            return hit
        raw = load_json(self._path(d))
        if self.source == "cards":
            cards = raw["cards"]
            for sig, c in cards.items():
                cards[sig] = self._cards.setdefault(sig, c)     # one object per signature, host-wide
        else:
            for sig, t in raw["types"].items():
                self._types.setdefault(sig, t)
            for sig, k in raw["kappa"].items():
                self._kappa.setdefault(sig, k)
        self._merged.add(d)
        entry = (raw, {int(n): s for n, s in raw["nodes"].items()})
        self._days.put(d, entry)
        return entry

    def day(self, d):
        """The day's card file as stored: {"day", "dataset", "version", "cards", "nodes"}."""
        self._need_cards("day()")
        return self._load(d)[0]

    def nodes(self, d):
        """node id (int) -> card signature on day d (card-file order)."""
        return self._load(d)[1]

    def _find(self, sig, store):
        v = store.get(sig)
        if v is not None:
            return v
        for d in self.clean + self.test:            # a signature from a day not read yet
            if d not in self._merged:
                self._load(d)
                if sig in store:
                    return store[sig]
        raise KeyError(f"{self.ds}: no card with signature {sig}")

    def card(self, sig):
        self._need_cards("card()")
        return self._find(sig, self._cards)

    def ntype(self, sig):
        if self.source == "nodeindex":
            return self._find(sig, self._types)
        return self.card(sig)["type"]

    def inv_key(self, sig):
        k = self._kappa.get(sig)
        if k is None:
            if self.source == "nodeindex":
                return self._find(sig, self._kappa)
            c = self.card(sig)
            k = self._kappa[sig] = signature({"type": c["type"], "invariant": c["invariant"]})
        return k

    def links(self, d):
        """(m, 3) int64 [cause, effect, code]; codes 1 spawn, 2 write, 3 exec, 4 net (cards.LINK_CODES)."""
        self._need_cards("links()")
        hit = self._links.get(d)
        if hit is not None:
            return hit
        p = link_path(self.cards_dir, self.ds, d)
        if not os.path.exists(p):
            raise MissingDay(f"{self.ds} {d}: no link file {p}")
        rows = load_json(p)["links"]
        arr = np.asarray(rows, dtype=np.int64).reshape(-1, 3) if rows else EMPTY_LINKS
        self._links.put(d, arr)
        return arr

    def shots(self, k):
        return list(self.clean) if k is None else self.clean[:k]

    def keep_clean_days(self):
        """Size the day LRU to hold every clean day plus one, so LODO references (k = all: all clean
        days but one, rotated) re-parse no day file. A pure speedup: the LRU only decides what is
        re-read from disk, never what is read. Returns the host."""
        lru = getattr(self, "_days", None)
        if lru is not None and hasattr(lru, "size"):
            lru.size = max(lru.size, len(self.clean) + 1)
        return self

    # --- the test-node universe ------------------------------------------------------------ #
    def _universe(self):
        if self._u is not None:
            return self._u
        ids, types = [], []
        for d in self.test:
            nd = self.nodes(d)
            ids.append(np.fromiter(nd.keys(), dtype=np.int64, count=len(nd)))
            types.append(np.array([self.ntype(s) for s in nd.values()], dtype="<U8"))
        all_ids = np.concatenate(ids) if ids else np.zeros(0, np.int64)
        universe, first = np.unique(all_ids, return_index=True)   # first occurrence = first test day
        utype = (np.concatenate(types) if types else np.zeros(0, "<U8"))[first]
        day_index = {d: np.sort(np.searchsorted(universe, i)) for d, i in zip(self.test, ids)}
        self._u = {"universe": universe, "utype": utype, "day_index": day_index}
        counts = {str(t): int(c) for t, c in zip(*np.unique(utype, return_counts=True))}
        log(f"{self.ds}: universe {len(universe)} nodes {counts} over {len(all_ids)} test node-days",
            "data.host")
        return self._u

    @property
    def universe(self):
        return self._universe()["universe"]

    @property
    def utype(self):
        return self._universe()["utype"]

    def day_index(self, d):
        """Sorted universe positions of the nodes that have a card on test day d."""
        return self._universe()["day_index"][d]

    def upos(self, node_ids):
        """Universe position of each node id, -1 for an id outside the universe."""
        u = self.universe
        ids = np.asarray(node_ids, dtype=np.int64)
        pos = np.searchsorted(u, ids)
        ok = pos < len(u)
        ok[ok] = u[pos[ok]] == ids[ok]
        return np.where(ok, pos, -1)

    @property
    def node_days(self):
        """node id -> test days it has a card on (date order). Built once, on first use."""
        u = self._universe()
        if "node_days" not in u:
            nd = {}
            uni = u["universe"].tolist()
            for d in self.test:
                for i in u["day_index"][d].tolist():
                    nd.setdefault(uni[i], []).append(d)
            u["node_days"] = nd
        return u["node_days"]

    @property
    def ukey(self):
        """kappa (inv_key) per universe node, from its card on its first test day."""
        u = self._universe()
        if "ukey" not in u:
            key = np.empty(len(u["universe"]), dtype=object)
            todo = np.ones(len(key), dtype=bool)
            for d in self.test:
                idx = u["day_index"][d]
                idx = idx[todo[idx]]
                if not len(idx):
                    continue
                nd = self.nodes(d)
                key[idx] = [self.inv_key(nd[n]) for n in u["universe"][idx].tolist()]
                todo[idx] = False
            u["ukey"] = key.astype("<U16")
        return u["ukey"]

    @property
    def ukey_codes(self):
        """Integer code per universe node, equal codes = equal kappa (entity pools)."""
        u = self._universe()
        if "ukey_codes" not in u:
            u["ukey_codes"] = np.unique(self.ukey, return_inverse=True)[1].astype(np.int64)
        return u["ukey_codes"]

    def __getstate__(self):
        # a worker process re-reads days on demand; never pickle GBs of parsed cards
        state = dict(self.__dict__)
        state["_days"], state["_links"] = _LRU(self._days.size), _LRU(self._links.size)
        state["_cards"], state["_merged"] = {}, set()
        if self.source == "nodeindex":
            state["_kappa"], state["_types"] = {}, {}
        return state

    def __repr__(self):
        return f"Host({self.ds!r}, source={self.source!r}, clean={len(self.clean)}, test={len(self.test)})"
