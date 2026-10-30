"""Evidence cards: one bounded, deterministic summary per node and day, for every node type.

A card has two sections:
  invariant  identity fields an edge-adding attacker cannot change: a process's image, name,
             normalised command line and creating parent (its one creation record); a file's
             path; a netflow's endpoint.
  behaviour  everything derived from the node's incident edges in the day: operations by object
             class, children, directories written, executed files, peers, and order-based flags
             (wrote a file that was then executed, ...).

Cards are abstracted before they are stored, so that one behaviour pattern is one card: per-instance
noise in paths and command lines (GUIDs, long numbers, hex/hash runs) becomes typed placeholders that
keep the fact "a random-looking name" (<hex:32>), and counts become log-scaled bins. The card's
signature (sha1 of its canonical JSON, 16 hex) is the key of every downstream cache, and render()
is the text the reference index reads: both, and every table and regex below, are byte-identical to
the original reflex/cards.py (only the build section is rewired).

Storage: <cards_dir>/<dataset>/<day>.json.gz =
    {"day", "dataset", "version", "cards": {sig: card}, "nodes": {node_id: sig}}
and <cards_dir>/<dataset>/links/<day>.json.gz = {"verbs": LINK_CODES, "links": [[cause, effect, code]]}.
No statistics file is written (the original stats.json held label-derived counts).
"""

import hashlib
import ipaddress
import json
import os
import re
import time
from collections import Counter

from habitus.data.graphs import CREATION_OPS, Window, graph_paths, windows
from habitus.data.identity import _GUID, MAX_CMD, _clean_path, process_identity
from habitus.data.profiles import WINDOWS_STAGING, in_staging_dir, normalize_path
from habitus.data.resolver import basename
from habitus.hosts import OS_FAMILY
from habitus.io import dump_json
from habitus.log import log

TOP = 5                 # how many names / classes a list field keeps
# causal links kept in the link files: verb -> code
LINK_VERBS = {"spawn": 1, "write": 2, "create": 2, "rename": 2, "exec": 3, "connect": 4, "send": 4}
LINK_CODES = {"1": "spawn", "2": "write", "3": "exec", "4": "net"}
CARD_VERSION = 2       # v2: Carbon Black deletes/disconnects/packets, image_deleted, decimal runs
                       # are numbers not hex, "internal" endpoints (same /16 as the host)

# --------------------------------------------------------------------------- #
# Vocabulary: operations and object classes, the same on every collector
# --------------------------------------------------------------------------- #
_VERB_RULES = (           # (substring of the collector's op label, verb); first match wins
    ("REGISTRY", "registry"), ("MODULE_LOAD", "load"), ("MMAP", "load"), ("LOADLIBRARY", "load"),
    ("UNLINK", "delete"), ("DELETE", "delete"), ("RENAME", "rename"), ("TRUNCATE", "write"),
    ("SENDTO", "send"), ("SENDMSG", "send"), ("RECVFROM", "recv"), ("RECVMSG", "recv"),
    ("PACKET_SENT", "send"), ("PACKET_RECEIVED", "recv"), ("DISCONNECT", "disconnect"),
    ("CONNECT", "connect"), ("ACCEPT", "connect"), ("NETFLOW_CREATED", "connect"),
    ("EXECUTE", "exec"), ("WRITE", "write"), ("MODIFY", "write"), ("COPIED", "write"),
    ("READ", "read"), ("OPEN", "open"), ("CREATE", "create"), ("TERMINATE", "terminate"),
)


def verb(label, src_type, dst_type, profile):
    """Coarse verb of one event; process creation and network direction need the endpoint types."""
    lab = str(label or "").upper()
    if src_type == "subject" and dst_type == "subject":
        return "spawn" if lab in CREATION_OPS or "EXEC" in lab else "ipc"
    if lab in profile.execute:
        return "connect" if "netflow" in (src_type, dst_type) else "exec"
    if lab == "MESSAGE":                                # OpTC: direction = edge direction
        return "send" if src_type == "subject" else "recv"
    if "DELET" in lab or "UNLINK" in lab:              # Carbon Black lists FILE_DELETED as a write
        return "delete"
    v = "write" if lab in profile.write else next((v for sub, v in _VERB_RULES if sub in lab), "other")
    if "netflow" in (src_type, dst_type) and v in ("write", "read"):
        v = "send" if v == "write" else "recv"      # DARPA TC writes/reads sockets
    return v


_WIN_SYSTEM = ("/windows/system32/", "/windows/syswow64/", "/windows/winsxs/", "/windows/")
_WIN_APP = ("/program files/", "/program files (x86)/")
_WIN_HOME = ("/users/",)
_POSIX_SYSTEM = ("/bin/", "/sbin/", "/usr/bin/", "/usr/sbin/", "/lib/", "/lib64/", "/usr/lib/",
                 "/usr/libexec/", "/libexec/", "/system/", "/apex/", "/vendor/", "/boot/")
_POSIX_APP = ("/usr/local/", "/opt/", "/usr/share/", "/data/app/", "/snap/", "/usr/games/")
_POSIX_HOME = ("/home/", "/root/", "/users/", "/sdcard/", "/storage/", "/data/media/")
_POSIX_CONFIG = ("/etc/", "/var/", "/proc/", "/sys/", "/dev/", "/run/", "/data/system/",
                 "/data/misc/")


def dir_class(path, profile):
    """system | application | home | staging | cache | config | other | none (no path)."""
    p = normalize_path(_clean_path(path) or "")
    if not p:
        return "none"
    if profile is not None and in_staging_dir(p, profile):
        return "staging"
    if "/cache/" in p or "/cache2/" in p or "/.cache/" in p or "/inetcache/" in p:
        return "cache"
    if profile is not None and profile.staging_dirs == WINDOWS_STAGING:
        if any(d in p for d in _WIN_APP):
            return "application"
        if any(d in p for d in _WIN_SYSTEM):
            return "system"
        if any(d in p for d in _WIN_HOME):
            return "home"
        return "other"
    p = p if p.startswith("/") else "/" + p
    for cls, dirs in (("system", _POSIX_SYSTEM), ("application", _POSIX_APP),
                      ("home", _POSIX_HOME), ("config", _POSIX_CONFIG)):
        if p.startswith(dirs):
            return cls
    if p.startswith("/data/data/"):
        return "application"                        # Android app-private data
    return "other"


# a hex/hash run has both digits and a-f letters; a run of digits alone is a number (<n>), not a hash
_HEX = re.compile(r"(?<![0-9a-zA-Z])(?=[0-9a-fA-F]*\d)(?=[0-9a-fA-F]*[a-fA-F])[0-9a-fA-F]{8,}(?![0-9a-zA-Z])")
_MIXED = re.compile(r"(?<![0-9a-zA-Z])(?=[0-9a-zA-Z]*\d)(?=[0-9a-zA-Z]*[a-zA-Z])[0-9a-zA-Z]{16,}(?![0-9a-zA-Z])")
_DIGITS = re.compile(r"\d{4,}")


def abstract_text(s):
    """Per-instance noise -> typed placeholders that keep the length (a random name stays visibly
    random): GUIDs, hex/hash runs, long mixed alphanumeric tokens, long numbers."""
    if not s:
        return s
    s = _GUID.sub("<guid>", s.strip())      # before the hex rule: a GUID is one token, not five
    s = _HEX.sub(lambda m: f"<hex:{len(m.group())}>", s)
    s = _MIXED.sub(lambda m: f"<rand:{len(m.group())}>", s)
    return _DIGITS.sub("<n>", s)[:MAX_CMD]


def ext_of(path):
    b = basename(_clean_path(path) or "") or ""
    return b.rsplit(".", 1)[1].lower()[:8] if "." in b.strip(".") else ""


def bin_count(n):
    """Log-scaled count bin: the card says '4-15 writes', not '11 writes'."""
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    for lo, hi in ((2, 7), (8, 31), (32, 127)):
        if n <= hi:
            return f"{lo}-{hi}"
    return "128+"


def _top(counter, n=TOP):
    return [k for k, _ in sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))[:n]]


# --------------------------------------------------------------------------- #
# One day of one host -> cards
# --------------------------------------------------------------------------- #
class _Node:
    __slots__ = ("type", "ent", "parent", "ops", "names_a", "names_b", "dirs", "exts", "execs",
                 "peers", "ports", "flags", "readers", "first_write", "exec_t", "writer_ids", "src")

    def __init__(self, ntype, ent):
        self.type, self.ent, self.parent = ntype, ent, None
        self.ops = Counter()          # (verb, object class)
        self.names_a = Counter()      # process: children | file: writers | netflow: processes
        self.names_b = Counter()      # file: executors
        self.dirs, self.exts = Counter(), Counter()
        self.execs = set()            # process: file nodes it executed
        self.peers, self.ports = set(), Counter()
        self.flags = set()
        self.readers = set()
        self.first_write = None       # file: first write time of the day
        self.exec_t = []              # file: (time, process) of each execution
        self.writer_ids = set()       # file: processes that wrote it
        self.src = None               # process without a recorded image: the file it was exec'ed from


class DayCards:
    def __init__(self, dataset, resolver, profile):
        self.ds, self.os, self.resolver, self.profile = dataset, OS_FAMILY[dataset], resolver, profile
        self.nodes = {}
        self._pname = {}
        self.links = set()       # (cause, effect, verb): the day's causal record (link file)
        host_ip = getattr(resolver, "host_ip", None)
        try:
            self._net16 = ipaddress.ip_network(f"{host_ip}/16", strict=False) if host_ip and ":" not in host_ip else None
        except ValueError:
            self._net16 = None

    def _node(self, wg, n):
        rec = self.nodes.get(n)
        if rec is None:
            rec = self.nodes[n] = _Node(wg.ntype.get(n), wg.ent.get(n))
        return rec

    def pname(self, wg, n):
        """Process file name (the identity's `name`) of a subject node."""
        if n not in self._pname:
            e = wg.ent.get(n)
            self._pname[n] = process_identity(self.os, e.path, e.cmd).name or "?" if e else "?"
        return self._pname[n]

    def endpoint(self, e):
        """The resolver's endpoint class, except that an 'external' address in the monitored host's
        own /16 is 'internal': the resolver's enclave is the host's /24, so OpTC's enterprise network
        (142.20.0.0/16) and DARPA's 128.55.0.0/16 were called external. Label-free (host IP only)."""
        cls = (e.endpoint_class if e else None) or "unknown"
        if cls == "external" and self._net16 is not None and e.remote_ip:
            try:
                if ipaddress.ip_address(e.remote_ip) in self._net16:
                    return "internal"
            except ValueError:
                pass
        return cls

    def obj_class(self, wg, n):
        t = wg.ntype.get(n)
        e = wg.ent.get(n)
        if t == "subject":
            return "process"
        if t == "netflow":
            return f"net-{self.endpoint(e)}"
        return dir_class(e.path if e else None, self.profile)

    def add_window(self, wg):
        G = wg.G
        for n in G.nodes():
            rec = self._node(wg, n)
            if rec.type == "subject" and rec.src is None and not (rec.ent and rec.ent.path):
                rec.src = wg.exec_source(n)
        for u, v, d in G.edges(data=True):
            tu, tv = wg.ntype.get(u), wg.ntype.get(v)
            vb = verb(d.get("label"), tu, tv, self.profile)
            t = d.get("time", 0)
            if tu == "subject" and tv == "subject":
                a, b = self.nodes[u], self.nodes[v]
                if vb == "spawn" and u != v:
                    self.links.add((u, v, "spawn"))
                    a.names_a[self.pname(wg, v)] += 1
                    a.ops[("spawn", "process")] += 1
                    if b.parent is None:
                        b.parent = u
                else:
                    a.ops[(vb, "process")] += 1
                continue
            if "subject" not in (tu, tv):
                continue
            actor, obj = (u, v) if tu == "subject" else (v, u)
            pa, po = self.nodes[actor], self.nodes[obj]
            ocls = self.obj_class(wg, obj)
            pa.ops[(vb, ocls)] += 1
            if vb in LINK_VERBS:      # direction of causality: actor -> what it made / used to run
                self.links.add((obj, actor, vb) if vb == "exec" else (actor, obj, vb))
            po.ops[(vb, "process")] += 1
            name = self.pname(wg, actor)
            if po.type == "file":
                e = po.ent
                if vb in ("write", "create", "rename"):
                    po.names_a[name] += 1
                    po.writer_ids.add(actor)
                    pa.dirs[ocls] += 1
                    x = ext_of(e.path if e else None)
                    if x:
                        pa.exts[x] += 1
                    if po.first_write is None or t < po.first_write:
                        po.first_write = t
                elif vb == "exec":
                    po.names_b[name] += 1
                    po.exec_t.append((t, actor))
                    pa.execs.add(obj)
                elif vb == "delete":
                    po.flags.add("deleted")
                elif vb in ("read", "open", "load"):
                    po.readers.add(actor)
            elif po.type == "netflow":
                po.names_a[name] += 1
                e = po.ent
                if e is not None and e.remote_ip:
                    pa.peers.add((e.remote_ip, self.endpoint(e)))
                    if e.remote_port:
                        pa.ports[str(e.remote_port)] += 1

    def finalize(self):
        """Order-based flags (need the whole day), then abstract + dedupe."""
        for f in self.nodes.values():
            if f.type != "file" or not f.exec_t or f.first_write is None:
                continue
            later = [p for t, p in f.exec_t if t > f.first_write]
            if later:
                f.flags.add("executed_after_write")
                for p in later:
                    self.nodes[p].flags.add("executed_file_written_today")
                for w in f.writer_ids:
                    self.nodes[w].flags.add("wrote_file_executed_later")
        cards, nodes = {}, {}
        self.link_list = sorted([int(a), int(b), LINK_VERBS[v] if v in LINK_VERBS else 0]
                                for a, b, v in self.links if a != b)
        for nid, rec in self.nodes.items():
            card = self.card(nid, rec)
            if card is None:
                continue
            sig = signature(card)
            cards.setdefault(sig, card)
            nodes[str(nid)] = sig
        return cards, nodes

    # --- the card ---------------------------------------------------------- #
    def card(self, nid, rec):
        e = rec.ent
        ops = {f"{v}:{c}": bin_count(n) for (v, c), n in sorted(rec.ops.items())}
        if rec.type == "subject":
            par = self.nodes.get(rec.parent).ent if rec.parent in self.nodes else None
            ident = process_identity(self.os, e.path if e else None, e.cmd if e else None,
                                     parent=(par.path, par.cmd) if par is not None else None,
                                     exec_path=rec.src)
            inv = {k: abstract_text(v) if k in ("executable", "command_line") else v
                   for k, v in ident.state().items() if k != "os"}
            inv["image_dir"] = dir_class(ident.image, self.profile) if ident.image else "none"
            if e is not None and str(e.path or "").strip().strip('"').endswith("(deleted)"):
                inv["image_deleted"] = "yes"          # only present when true: other cards keep their text
            beh = {"ops": ops}
            if rec.names_a:
                beh["children"] = _top(rec.names_a)
            if rec.dirs:
                beh["write_dirs"] = _top(rec.dirs)
            if rec.exts:
                beh["write_exts"] = _top(rec.exts, 3)
            if rec.execs:
                ex = [self.nodes[f].ent for f in rec.execs if self.nodes.get(f) and self.nodes[f].ent]
                beh["exec_dirs"] = _top(Counter(dir_class(x.path, self.profile) for x in ex))
            if rec.peers:
                beh["peers"] = {cls or "unknown": bin_count(n) for cls, n in
                                sorted(Counter(c for _ip, c in rec.peers).items(), key=str)}
                beh["ports"] = _top(rec.ports, 3)
            if rec.flags:
                beh["flags"] = sorted(rec.flags)
            return {"v": CARD_VERSION, "type": "process", "os": self.os, "invariant": inv,
                    "behaviour": beh}
        if rec.type == "file":
            path = _clean_path(e.path) if e else None
            inv = {"path": abstract_text(path) if path else "", "dir": dir_class(path, self.profile),
                   "ext": ext_of(path)}
            beh = {"ops": ops}
            if rec.names_a:
                beh["writers"] = _top(rec.names_a)
            if rec.names_b:
                beh["executors"] = _top(rec.names_b)
            if rec.readers:
                beh["readers"] = bin_count(len(rec.readers))
            if rec.flags:
                beh["flags"] = sorted(rec.flags)
            return {"v": CARD_VERSION, "type": "file", "os": self.os, "invariant": inv,
                    "behaviour": beh}
        if rec.type == "netflow":
            inv = {"remote_ip": (e.remote_ip if e else "") or "", "remote_port": str((e.remote_port if e else "") or ""),
                   "endpoint": self.endpoint(e)}
            beh = {"ops": ops}
            if rec.names_a:
                beh["processes"] = _top(rec.names_a)
            return {"v": CARD_VERSION, "type": "netflow", "os": self.os, "invariant": inv,
                    "behaviour": beh}
        return None


def signature(card):
    return hashlib.sha1(json.dumps(card, sort_keys=True).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Rendering: the card as text (the reference index reads it)
# --------------------------------------------------------------------------- #
_LABEL = {"executable": "Executable", "file_name": "File name", "command_line": "Command line",
          "parent_name": "Started by", "image_dir": "Executable directory class",
          "path": "Path", "dir": "Directory class", "ext": "Extension",
          "remote_ip": "Remote address", "remote_port": "Remote port", "endpoint": "Endpoint class",
          "image_deleted": "Executable deleted from disk"}


def render(card, sections=("invariant", "behaviour")):
    """Card -> text. `sections` restricts what a question may read (invariant-only questions
    never see edge-derived fields)."""
    lines = [f"Node type: {card['type']} on a {card['os']} host"]
    if "invariant" in sections:
        for k, v in card["invariant"].items():
            if v not in ("", None):
                lines.append(f"{_LABEL.get(k, k)}: {v}")
    if "behaviour" in sections:
        b = card.get("behaviour", {})
        if b.get("ops"):
            lines.append("Activity during the day (operation:object class = count): "
                         + ", ".join(f"{k}={v}" for k, v in b["ops"].items()))
        for k in ("children", "writers", "executors", "processes", "write_dirs", "write_exts",
                  "exec_dirs", "ports"):
            if b.get(k):
                lines.append(f"{k.replace('_', ' ').capitalize()}: {', '.join(map(str, b[k]))}")
        if b.get("peers"):
            lines.append("Network peers by class: " + ", ".join(f"{k}={v}" for k, v in b["peers"].items()))
        if b.get("readers"):
            lines.append(f"Distinct reading processes: {b['readers']}")
        if b.get("flags"):
            lines.append("Observed: " + ", ".join(f.replace("_", " ") for f in b["flags"]))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Build (paths are computed, never created; the writers create the files' parents)
# --------------------------------------------------------------------------- #
def card_path(cards_dir, ds, day):
    return os.path.join(os.fspath(cards_dir), ds, f"{day}.json.gz")


def link_path(cards_dir, ds, day):
    return os.path.join(os.fspath(cards_dir), ds, "links", f"{day}.json.gz")


def build_day(ds, graphs_dir, resolver, day, out):
    """Cards and links of one day from the PIDSMaker window graphs -> card_path / link_path under `out`."""
    t = time.time()
    dc = DayCards(ds, resolver, None)
    n_win = 0
    for _wname, G, window in windows(graph_paths(graphs_dir, day)):
        wg = Window(window, resolver, G, ds)
        dc.profile = wg.profile
        dc.add_window(wg)
        n_win += 1
        del wg, G
    cards, nodes = dc.finalize()
    by_type = Counter(c["type"] for c in cards.values())
    dump_json({"day": day, "dataset": ds, "version": CARD_VERSION, "cards": cards, "nodes": nodes},
              card_path(out, ds, day))
    dump_json({"verbs": LINK_CODES, "links": dc.link_list}, link_path(out, ds, day))
    log(f"{ds} {day}: {n_win} windows, {len(nodes)} nodes -> {len(cards)} cards {dict(by_type)} "
        f"({time.time() - t:.0f}s)", "data.cards")
    return card_path(out, ds, day), link_path(out, ds, day)


def days_of(spec):
    """(attack-free days, configured test days), each sorted: the days that get a card file."""
    from habitus.data.days import _benign_dates
    return sorted(_benign_dates(spec)), sorted(spec.configured_test)


def build(ds, out=None, *, days=None, force=False, graphs_dir=None, resolver=None, settings=None):
    """Build the card days of `ds` (HostSpec.card_days, or `days`) into `out` (default runs/cards).
    A day whose link file exists is skipped unless `force` (the link file is written last).
    graphs_dir defaults to data.upstream.graphs_dir(ds), resolver to data.resolver.get_resolver(ds)
    (a cache; ResolverUnavailable without one). Returns the days built."""
    from habitus.config import check_writable, settings as _settings
    from habitus.hosts import spec
    s = settings or _settings()
    out = check_writable(out or os.path.join(s.paths.runs, "cards"), s)
    benign, test = days_of(spec(ds))
    todo = list(days) if days is not None else benign + test
    bad = [d for d in todo if d not in set(benign + test)]
    if bad:
        raise ValueError(f"{ds}: {bad} are not card days of this host ({benign + test})")
    todo = [d for d in todo if force or not os.path.exists(link_path(out, ds, d))]
    if not todo:
        return []
    if graphs_dir is None:
        from habitus.data.upstream import graphs_dir as _graphs_dir
        graphs_dir = _graphs_dir(ds, settings=s)
    if resolver is None:
        from habitus.data.resolver import get_resolver
        resolver = get_resolver(ds, settings=s)
    for day in todo:
        build_day(ds, graphs_dir, resolver, day, out)
    return todo
