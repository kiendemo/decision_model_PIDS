"""Semantic resolver: node identity (path, command line, remote endpoint) from the DATABASE node tables.

The graph's node "label" is produced by the detector's label config; the card builder instead
resolves `index_id -> (path, cmd, remote endpoint)` from the PIDSMaker Postgres node tables, so card
contents do not depend on the detector configuration. The graph still owns `node_type`.

Network endpoints are classified against the monitored host's own address, auto-detected as the
dominant routable src_addr of the netflow table (logged).

Cache: <dir>/<ds>.pkl = {"subject": {index_id: (path, cmd)}, "file": {index_id: path},
"netflow": {index_id: (remote_ip, remote_port, local_port)}, "host_ip"}. A trusted local pickle.
Lookup order: runs/defense/semantics/<ds>.pkl, then <pidsmaker_artifacts>/defense/semantics/<ds>.pkl;
a DB build writes runs/defense/semantics/<ds>.pkl.

Unlike the original, construction FAILS HARD (ResolverUnavailable) when neither a cache nor a
database is available: the original's silent fallback to graph labels produced different cards.
Per-node label parsing (_from_label) is kept for subject nodes that have no node-table row, exactly
as in the original.

Vendored from the original defense/semantics.py: constants, Entity, classify_endpoint and the
resolution methods are verbatim; construction takes explicit cache / database arguments.
"""

import ipaddress
import os
import pickle
from collections import Counter
from dataclasses import dataclass

from habitus.log import log

# Endpoint classes, ordered from least to most suspicious.
EP_LOOPBACK = "loopback"
EP_PRIVATE = "private"
EP_ENCLAVE = "enclave"
EP_EXTERNAL = "external"
EP_UNKNOWN = "unknown"

ENCLAVE_PREFIX_LEN = 24  # remote addresses in the host's own /24 are the monitored enclave
UNKNOWN_FILE_KEY = "<file-path-not-recorded>"
_NULLISH = {"", "null", "None", "none", "NULL"}
CACHE_SUBDIR = os.path.join("defense", "semantics")


class ResolverUnavailable(RuntimeError):
    """Neither a resolver cache nor a reachable database: cards cannot be built faithfully."""


def _clean(value):
    """Normalize a DB string field: '', 'null', 'None' all mean *absent*."""
    if value is None:
        return None
    s = str(value).strip()
    return None if s in _NULLISH else s


def basename(path):
    """Last path component. Windows telemetry uses `\\`, POSIX `/`; both are split here so
    one identity rule covers every host family."""
    if not path:
        return None
    return path.replace("\\", "/").rsplit("/", 1)[-1] or path


def dirname(path):
    if not path:
        return None
    p = path.replace("\\", "/")
    if "/" not in p:
        return None
    return p.rsplit("/", 1)[0] or "/"


@dataclass(frozen=True)
class Entity:
    """Resolved identity of one graph node."""

    node_id: str
    node_type: str
    path: str = None
    cmd: str = None
    remote_ip: str = None
    remote_port: str = None
    local_port: str = None
    endpoint_class: str = None
    resolved: bool = False  # True when the attributes came from the DB, not from the label

    @property
    def key(self):
        """Canonical identity used as the knowledge-base lookup key."""
        if self.node_type == "subject":
            return basename(self.path) or self.cmd or "?"
        if self.node_type == "file":
            return self.path or UNKNOWN_FILE_KEY
        if self.node_type == "netflow":
            return self.remote_ip or "?"
        return "?"

    @property
    def display(self):
        """Human-readable identity. Never emits the string 'None'."""
        if self.node_type == "subject":
            if self.path and self.cmd and basename(self.path) != self.cmd:
                return f"{self.cmd} ({self.path})"
            return self.path or self.cmd or f"process #{self.node_id} (name not recorded)"
        if self.node_type == "file":
            return self.path or f"file #{self.node_id} (path not recorded in the audit log)"
        if self.node_type == "netflow":
            if not self.remote_ip:
                return f"netflow #{self.node_id}"
            port = f":{self.remote_port}" if self.remote_port else ""
            return f"{self.remote_ip}{port} [{self.endpoint_class or EP_UNKNOWN}]"
        return f"#{self.node_id}"

    @property
    def is_external(self):
        return self.endpoint_class == EP_EXTERNAL

    @property
    def has_identity(self):
        """False when we know nothing beyond the node type (most CADETS_E3 file nodes)."""
        if self.node_type == "subject":
            return bool(self.path or self.cmd)
        if self.node_type == "file":
            return bool(self.path)
        return bool(self.remote_ip)


def classify_endpoint(remote_ip, host_ip):
    if not remote_ip:
        return EP_UNKNOWN
    try:
        ip = ipaddress.ip_address(remote_ip)
    except ValueError:
        return EP_UNKNOWN
    if ip.is_loopback:
        return EP_LOOPBACK
    if ip.is_private or ip.is_link_local or ip.is_multicast or ip.is_unspecified:
        return EP_PRIVATE
    if host_ip and ip.version == 4:
        try:
            net = ipaddress.ip_network(f"{host_ip}/{ENCLAVE_PREFIX_LEN}", strict=False)
            if ip in net:
                return EP_ENCLAVE
        except ValueError:
            pass
    return EP_EXTERNAL


def connect_db(db):
    """(cursor, connection) for `db` = {database, host, user, password, port} (host None: local socket)."""
    import psycopg2
    kw = {k: db.get(k) for k in ("database", "user", "password", "port")}
    if db.get("host") is not None:
        kw["host"] = db["host"]
    connect = psycopg2.connect(**kw)
    return connect.cursor(), connect


class SemanticResolver:
    """`index_id -> Entity`, backed by the dataset's node tables (cached on disk).

    Only rows that actually carry information are kept, so the cache stays small: on
    CADETS_E3 that is 224k subjects + 42k files (of 2.3M — the rest have a NULL path) +
    155k netflows, instead of 2.68M dense entries.

        SemanticResolver(ds, cache_path)                  # load the cache (ResolverUnavailable if absent)
        SemanticResolver(ds, cache_path, db=db_params)    # cache, else build from the DB and write it
        SemanticResolver.from_tables(...)                 # in-memory tables (tests)
    """

    def __init__(self, ds, cache_path, *, db=None, force=False, write_path=None, host_ip=None):
        self._subject = {}   # index_id -> (path, cmd)
        self._file = {}      # index_id -> path        (absent => path not recorded)
        self._netflow = {}   # index_id -> (remote_ip, remote_port, local_port)
        self.host_ip = host_ip
        self.available = False
        self._entity_cache = {}
        self.dataset = ds
        self._cache_path = os.fspath(cache_path) if cache_path is not None else None
        self._write_path = os.fspath(write_path) if write_path is not None else self._cache_path
        self._load(db, force=force)

    @classmethod
    def from_tables(cls, ds, subject=None, file=None, netflow=None, host_ip=None):
        """A resolver over in-memory tables (no cache file, no database)."""
        self = cls.__new__(cls)
        self._subject = {int(k): tuple(v) for k, v in (subject or {}).items()}
        self._file = {int(k): v for k, v in (file or {}).items()}
        self._netflow = {int(k): tuple(v) for k, v in (netflow or {}).items()}
        self.host_ip = host_ip
        self.available = True
        self._entity_cache = {}
        self.dataset = ds
        self._cache_path = self._write_path = None
        return self

    # --- construction ----------------------------------------------------- #
    def _load(self, db, force):
        if not force and self._cache_path and os.path.exists(self._cache_path):
            try:
                with open(self._cache_path, "rb") as f:
                    blob = pickle.load(f)
                self._subject = blob["subject"]
                self._file = blob["file"]
                self._netflow = blob["netflow"]
                self.host_ip = blob["host_ip"]
                self.available = True
                log(f"loaded <- {self._cache_path} (subjects={len(self._subject)}, "
                    f"files={len(self._file)}, netflows={len(self._netflow)}, host_ip={self.host_ip})",
                    "data.resolver")
                return
            except Exception as e:  # noqa: BLE001 - any unreadable cache is rebuilt or refused below
                if db is None:
                    raise ResolverUnavailable(f"{self.dataset}: resolver cache {self._cache_path} is "
                                              f"unreadable ({e}) and no database was given") from e
                log(f"cache unreadable ({e}) -> rebuilding from the database", "data.resolver")
        if db is None:
            raise ResolverUnavailable(
                f"{self.dataset}: no resolver cache at {self._cache_path} and no database: run "
                f"`habitus data resolver --ds {self.dataset}` against the restored PIDSMaker database")
        try:
            self._build_from_db(db)
        except Exception as e:
            raise ResolverUnavailable(f"{self.dataset}: building the resolver from the database "
                                      f"failed ({e}); refusing to fall back to graph labels") from e

    def _build_from_db(self, db):
        cur, connect = connect_db(db)
        try:
            cur.execute("SELECT index_id, path, cmd FROM subject_node_table;")
            for index_id, path, cmd in cur.fetchall():
                p, c = _clean(path), _clean(cmd)
                if p or c:
                    self._subject[int(index_id)] = (p, c)

            cur.execute("SELECT index_id, path FROM file_node_table;")
            for index_id, path in cur.fetchall():
                p = _clean(path)
                if p:
                    self._file[int(index_id)] = p

            cur.execute(
                "SELECT index_id, src_addr, src_port, dst_addr, dst_port FROM netflow_node_table;"
            )
            rows = cur.fetchall()
        finally:
            connect.close()

        # The preprocessing stores the CDM `localAddress` in src_addr, so the monitored
        # host is simply the dominant routable src_addr. Detect it rather than hard-code it.
        src_counts = Counter()
        for _i, src_addr, _sp, _da, _dp in rows:
            s = _clean(src_addr)
            if s:
                src_counts[s] += 1
        self.host_ip = self._pick_host_ip(src_counts)

        for index_id, src_addr, src_port, dst_addr, dst_port in rows:
            sa, sp = _clean(src_addr), _clean(src_port)
            da, dp = _clean(dst_addr), _clean(dst_port)
            # the remote peer is whichever endpoint is not the host
            if sa and sa == self.host_ip:
                remote_ip, remote_port, local_port = da, dp, sp
            elif da and da == self.host_ip:
                remote_ip, remote_port, local_port = sa, sp, dp
            else:
                remote_ip, remote_port, local_port = da or sa, dp, sp
            self._netflow[int(index_id)] = (remote_ip, remote_port, local_port)

        self.available = True
        if self._write_path:
            from habitus.config import check_writable
            check_writable(self._write_path)          # never into a read-only source tree
            os.makedirs(os.path.dirname(os.path.abspath(self._write_path)), exist_ok=True)
            tmp = f"{self._write_path}.tmp{os.getpid()}"
            with open(tmp, "wb") as f:
                pickle.dump(
                    {"subject": self._subject, "file": self._file, "netflow": self._netflow,
                     "host_ip": self.host_ip}, f, protocol=4,
                )
            os.replace(tmp, self._write_path)
        log(f"built from the database -> {self._write_path} (subjects={len(self._subject)}, "
            f"files_with_path={len(self._file)}, netflows={len(self._netflow)}, "
            f"detected host_ip={self.host_ip})", "data.resolver")

    @staticmethod
    def _pick_host_ip(src_counts):
        for addr, _n in src_counts.most_common():
            try:
                ip = ipaddress.ip_address(addr)
            except ValueError:
                continue  # 'localhost', 'NETLINK', unix socket paths, ...
            if not (ip.is_loopback or ip.is_unspecified):
                return addr
        return None

    # --- resolution ------------------------------------------------------- #
    def resolve(self, node_id, node_type, label=None):
        cached = self._entity_cache.get(node_id)
        if cached is not None:
            return cached
        entity = self._resolve_uncached(node_id, node_type, label)
        self._entity_cache[node_id] = entity
        return entity

    def _resolve_uncached(self, node_id, node_type, label):
        nid = str(node_id)
        try:
            iid = int(nid)
        except (TypeError, ValueError):
            iid = None

        if self.available and iid is not None:
            if node_type == "subject":
                path, cmd = self._subject.get(iid, (None, None))
                if path or cmd:
                    return Entity(nid, node_type, path=path, cmd=cmd, resolved=True)
            elif node_type == "file":
                path = self._file.get(iid)
                # a miss is informative here: the audit log genuinely has no path
                return Entity(nid, node_type, path=path, resolved=True)
            elif node_type == "netflow":
                nf = self._netflow.get(iid)
                if nf is not None:
                    remote_ip, remote_port, local_port = nf
                    return Entity(
                        nid, node_type, remote_ip=remote_ip, remote_port=remote_port,
                        local_port=local_port,
                        endpoint_class=classify_endpoint(remote_ip, self.host_ip),
                        resolved=True,
                    )
        return self._from_label(nid, node_type, label)

    def _from_label(self, nid, node_type, label):
        """Recover what we can from the graph label (a node without a node-table row)."""
        toks = [t for t in (label or "").split() if t not in _NULLISH]
        if node_type == "subject":
            path = next((t for t in toks if t.startswith("/")), None)
            cmd = next((t for t in reversed(toks) if not t.startswith("/")), None)
            return Entity(nid, node_type, path=path, cmd=cmd)
        if node_type == "file":
            return Entity(nid, node_type, path=next((t for t in toks if t.startswith("/")), None))
        if node_type == "netflow":
            ip = port = None
            for tok in toks:
                head = tok.split("->")[-1]
                if ":" in head:
                    head, _, port = head.rpartition(":")
                try:
                    ipaddress.ip_address(head)
                    ip = head
                    break
                except ValueError:
                    continue
            return Entity(nid, node_type, remote_ip=ip, remote_port=port,
                          endpoint_class=classify_endpoint(ip, self.host_ip))
        return Entity(nid, node_type)


# --------------------------------------------------------------------------- #
# Locations
# --------------------------------------------------------------------------- #
def cache_candidates(ds, s=None):
    """Resolver cache files to try, in order: runs/ (a local DB build), then pidsmaker_artifacts."""
    from habitus.config import settings
    s = s or settings()
    out = [os.path.join(s.paths.runs, CACHE_SUBDIR, f"{ds}.pkl")]
    if s.paths.pidsmaker_artifacts:
        out.append(os.path.join(s.paths.pidsmaker_artifacts, CACHE_SUBDIR, f"{ds}.pkl"))
    return out


def build_target(ds, s=None):
    """Where a database build writes the cache (always under runs/)."""
    from habitus.config import settings
    s = s or settings()
    return os.path.join(s.paths.runs, CACHE_SUBDIR, f"{ds}.pkl")


_RESOLVERS = {}


def get_resolver(ds, *, db=None, cache_path=None, force=False, settings=None):
    """A shared SemanticResolver for `ds` (built at most once per process and cache path).
    The cache is the first existing candidate (cache_candidates) unless `cache_path` is given."""
    from habitus.hosts import assert_allowed
    assert_allowed(ds)
    if cache_path is None:
        cands = cache_candidates(ds, settings)
        cache_path = next((c for c in cands if os.path.exists(c)), cands[0])
    key = (ds, os.path.abspath(os.fspath(cache_path)))
    if force or key not in _RESOLVERS:
        _RESOLVERS[key] = SemanticResolver(ds, cache_path, db=db, force=force,
                                           write_path=build_target(ds, settings) if (force or db) else None)
    return _RESOLVERS[key]


def build(ds, *, db=None, settings=None):
    """Build the resolver cache of `ds` from its restored PIDSMaker database (one pass over the three
    node tables) -> runs/defense/semantics/<ds>.pkl. `db`: data.upstream.db_params(ds, ...) (default:
    the upstream defaults host/user/password postgres, port 5432). Returns the cache path."""
    from habitus.data.upstream import db_params
    from habitus.hosts import assert_allowed
    assert_allowed(ds)
    target = build_target(ds, settings)
    r = SemanticResolver(ds, target, db=db or db_params(ds), force=True, write_path=target)
    _RESOLVERS[(ds, os.path.abspath(target))] = r
    return target
