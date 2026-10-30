"""Shared habit-channel test doubles: a synthetic host whose cards and links follow data.cards
exactly, and a deterministic AnswerSource. Habit tests import these; extend in your own test
module so that every habit test runs against the same data.

    from habit_fakes import make_habit_host, FakeAnswerSource, CLEAN, TEST, ID
    host = make_habit_host()                 # 3 clean days, 2 test days, ds "HBFAKE", os "linux"
    host.nodes(TEST[1])[ID["payload"]]       # -> the payload's card signature on the attack day

Content (stable node ids across days, ID below; one causal link per event, cards.LINK_VERBS codes):
  clean days CLEAN and test day TEST[0]: init spawns sshd and cron; cron spawns sh, sh spawns curl
    (/usr/bin/curl); sshd spawns bash; bash spawns ls,
    python3 and firefox; ls is exec'ed from /bin/ls, python3 from /usr/bin/python3 (exec links);
    python3 writes /home/u/out.txt and talks to 10.0.0.5:53 (private); cron writes /var/log/cron.log;
    firefox writes a cache file whose path holds a placeholder (<hex:16>), talks to 93.184.216.34:443
    (external) and 128.55.12.10:53 (enclave); every process reads a path-less file (uninformative).
  attack day TEST[1] adds: firefox spawns a payload /tmp/<hex:16> (staging, placeholder identity) that
    writes /tmp/.x/drop.sh, runs it, and talks to 6.6.6.6:4444; python3 (an attested identity) also
    talks to 6.6.6.7:443 (external: a new peer class for it); a renamed copy of python3's name
    ("python3" from /tmp/python3) writes /home/u/out.txt (masquerading writer); a twin sh node (id 12,
    card identical to the usual sh 11, so ONE signature with n_s = 2) spawns a renamed curl from /tmp -
    the representative-blinding case: same card, different per-node P4 child set.
Observability: spawn, write, exec, net all present (FakeHabitHost.obs). Nothing here is a label.

FakeHabitHost is tests/fakes.FakeHost (a data.host.Host over in-memory days): Host's own algorithms
(card store, kappa, universe, upos, ukey) run unchanged on top of its _load / links hooks.
"""

import hashlib
import os
from collections import Counter

import numpy as np

from fakes import FakeHost, make_spec
from habitus.data.cards import LINK_VERBS, _top, bin_count, signature


def _real_role_table():
    """True when settings.paths.catalogs holds the role-table catalogs (the frozen table is then
    built and checked); otherwise an empty explicit table is installed (every role "unknown") so that
    the fake-host tests still run. Catalog-specific tests check HAVE_CATALOGS."""
    from habitus.questions.habit import vocab
    try:
        root = vocab.catalogs_root()
    except Exception:                       # noqa: BLE001 - no settings: no catalogs
        root = None
    if root and os.path.isfile(os.path.join(root, "Contents-amd64.gz")):
        return True
    vocab.ROLE_TABLE.use_table({"posix": {}, "windows": {}})
    return False


def _sanitizer():
    """The sanitizer needs the pinned tokenizer's vocabulary; without it (no tokenizer on disk) the
    frozen snapshot is allowed for the fake-host unit tests (state.sanitize.allow_frozen_snapshot)."""
    from habitus.state import sanitize
    if sanitize.pattern_status() == "snapshot":
        sanitize.allow_frozen_snapshot(True)
        return False
    return True


HAVE_CATALOGS = _real_role_table()
LIVE_SANITIZER = _sanitizer()

CLEAN = ("2020-01-01", "2020-01-02", "2020-01-03")
TEST = ("2020-01-05", "2020-01-06")
DS = "HBFAKE"
OBS = {"exec": True, "lineage": True, "parent_name": True}

# entity id -> (type, invariant without parent_name); processes also carry their image dir class
ENTITIES = {
    1: ("process", {"file_name": "init", "executable": "/sbin/init", "image_dir": "system"}),
    2: ("process", {"file_name": "sshd", "executable": "/usr/sbin/sshd", "image_dir": "system"}),
    3: ("process", {"file_name": "bash", "executable": "/bin/bash", "command_line": "bash -l",
                    "image_dir": "system"}),
    4: ("process", {"file_name": "ls", "executable": "/bin/ls", "command_line": "ls -la",
                    "image_dir": "system"}),
    5: ("process", {"file_name": "python3", "executable": "/usr/bin/python3",
                    "command_line": "python3 /home/u/job.py", "image_dir": "system"}),
    6: ("process", {"file_name": "firefox", "executable": "/usr/lib/firefox/firefox", "image_dir": "system"}),
    7: ("process", {"file_name": "cron", "executable": "/usr/sbin/cron", "image_dir": "system"}),
    8: ("process", {"file_name": "<hex:16>", "executable": "/tmp/<hex:16>",
                    "command_line": "/tmp/<hex:16> -c <n>", "image_dir": "staging"}),
    9: ("process", {"file_name": "python3", "executable": "/tmp/python3", "command_line": "python3 /home/u/job.py",
                    "image_dir": "staging"}),
    11: ("process", {"file_name": "sh", "executable": "/bin/sh", "image_dir": "system"}),
    12: ("process", {"file_name": "sh", "executable": "/bin/sh", "image_dir": "system"}),
    13: ("process", {"file_name": "curl", "executable": "/usr/bin/curl", "image_dir": "system"}),
    14: ("process", {"file_name": "curl", "executable": "/tmp/curl", "image_dir": "staging"}),
    101: ("file", {"path": "/home/u/out.txt", "dir": "home", "ext": "txt"}),
    102: ("file", {"path": "/var/log/cron.log", "dir": "config", "ext": "log"}),
    103: ("file", {"path": "/bin/ls", "dir": "system", "ext": ""}),
    104: ("file", {"path": "/usr/bin/python3", "dir": "system", "ext": ""}),
    105: ("file", {"path": "/home/u/.cache/ff/<hex:16>.tmp", "dir": "cache", "ext": "tmp"}),
    106: ("file", {"path": "", "dir": "none", "ext": ""}),
    107: ("file", {"path": "/tmp/.x/drop.sh", "dir": "staging", "ext": "sh"}),
    201: ("netflow", {"remote_ip": "93.184.216.34", "remote_port": "443", "endpoint": "external"}),
    202: ("netflow", {"remote_ip": "10.0.0.5", "remote_port": "53", "endpoint": "private"}),
    203: ("netflow", {"remote_ip": "128.55.12.10", "remote_port": "53", "endpoint": "enclave"}),
    204: ("netflow", {"remote_ip": "6.6.6.6", "remote_port": "4444", "endpoint": "external"}),
    205: ("netflow", {"remote_ip": "6.6.6.7", "remote_port": "443", "endpoint": "external"}),
}
ID = {"init": 1, "sshd": 2, "bash": 3, "ls": 4, "python3": 5, "firefox": 6, "cron": 7, "payload": 8,
      "fake_python3": 9, "sh": 11, "sh_twin": 12, "curl": 13, "fake_curl": 14, "out_txt": 101, "cron_log": 102,
      "bin_ls": 103, "bin_python3": 104, "ff_cache": 105, "pathless": 106, "drop_sh": 107, "web": 201,
      "dns_private": 202, "dns_enclave": 203, "c2": 204, "lotl_peer": 205}

BENIGN = [("spawn", 1, 2), ("spawn", 1, 7), ("spawn", 2, 3), ("spawn", 3, 4), ("spawn", 3, 5), ("spawn", 3, 6),
          ("exec", 4, 103), ("exec", 5, 104), ("write", 5, 101), ("connect", 5, 202), ("write", 7, 102),
          ("create", 6, 105), ("connect", 6, 201), ("send", 6, 201), ("connect", 6, 203),
          ("spawn", 7, 11), ("spawn", 11, 13)] + \
         [("read", p, 106) for p in (2, 3, 4, 5, 6, 7)]
ATTACK = [("spawn", 6, 8), ("create", 8, 107), ("exec", 8, 107), ("connect", 8, 204), ("connect", 5, 205),
          ("spawn", 3, 9), ("write", 9, 101), ("spawn", 7, 12), ("spawn", 12, 14)]


def _endpoint_class(nid):
    return ENTITIES[nid][1]["endpoint"]


def _obj_class(nid):
    t, inv = ENTITIES[nid]
    if t == "process":
        return "process"
    if t == "netflow":
        return f"net-{inv['endpoint']}"
    return inv["dir"]


def build_day(day, events):
    """(day dict as stored by data.cards, link rows) for one list of (verb, actor, object) events.
    Event order is the time order (the first spawn of a child sets its parent_name)."""
    nodes = sorted({n for _v, a, o in events for n in (a, o)})
    ops = {n: Counter() for n in nodes}
    children, writers, executors, procs = ({n: Counter() for n in nodes} for _ in range(4))
    wdirs, wexts, edirs, ports = ({n: Counter() for n in nodes} for _ in range(4))
    peers = {n: set() for n in nodes}
    readers = {n: set() for n in nodes}
    parent, written, executed = {}, {}, {}
    links = set()
    for t, (vb, a, o) in enumerate(events):
        name = ENTITIES[a][1]["file_name"]
        ot = ENTITIES[o][0]
        if ot == "process":
            links.add((a, o, LINK_VERBS["spawn"]))
            ops[a][("spawn", "process")] += 1
            children[a][ENTITIES[o][1]["file_name"]] += 1
            parent.setdefault(o, a)
            continue
        ops[a][(vb, _obj_class(o))] += 1
        ops[o][(vb, "process")] += 1
        if vb in LINK_VERBS:
            links.add((o, a, LINK_VERBS[vb]) if vb == "exec" else (a, o, LINK_VERBS[vb]))
        if ot == "file":
            if vb in ("write", "create", "rename"):
                writers[o][name] += 1
                wdirs[a][ENTITIES[o][1]["dir"]] += 1
                if ENTITIES[o][1]["ext"]:
                    wexts[a][ENTITIES[o][1]["ext"]] += 1
                written.setdefault(o, (t, set()))[1].add(a)
            elif vb == "exec":
                executors[o][name] += 1
                edirs[a][ENTITIES[o][1]["dir"]] += 1
                executed.setdefault(o, []).append((t, a))
            elif vb == "read":
                readers[o].add(a)
        else:
            procs[o][name] += 1
            peers[a].add(_endpoint_class(o))
            ports[a][ENTITIES[o][1]["remote_port"]] += 1
    flags = {n: set() for n in nodes}
    for f, runs in executed.items():
        if f in written:
            t0, ws = written[f]
            later = [p for t, p in runs if t > t0]
            if later:
                flags[f].add("executed_after_write")
                for p in later:
                    flags[p].add("executed_file_written_today")
                for w in ws:
                    flags[w].add("wrote_file_executed_later")
    cards, out = {}, {}
    for n in nodes:
        t, inv = ENTITIES[n]
        inv = dict(inv)
        o = {f"{v}:{c}": bin_count(k) for (v, c), k in sorted(ops[n].items())}
        beh = {"ops": o}
        if t == "process":
            if n in parent:
                inv["parent_name"] = ENTITIES[parent[n]][1]["file_name"]
            inv = {k: inv[k] for k in ("file_name", "executable", "command_line", "parent_name", "image_dir")
                   if k in inv}
            if children[n]:
                beh["children"] = _top(children[n])
            if wdirs[n]:
                beh["write_dirs"] = _top(wdirs[n])
            if wexts[n]:
                beh["write_exts"] = _top(wexts[n], 3)
            if edirs[n]:
                beh["exec_dirs"] = _top(edirs[n])
            if peers[n]:
                beh["peers"] = {c: "1" for c in sorted(peers[n])}
                beh["ports"] = _top(ports[n], 3)
        elif t == "file":
            if writers[n]:
                beh["writers"] = _top(writers[n])
            if executors[n]:
                beh["executors"] = _top(executors[n])
            if readers[n]:
                beh["readers"] = bin_count(len(readers[n]))
        elif procs[n]:
            beh["processes"] = _top(procs[n])
        if flags[n]:
            beh["flags"] = sorted(flags[n])
        card = {"v": 2, "type": t, "os": "linux", "invariant": inv, "behaviour": beh}
        s = signature(card)
        cards[s] = card
        out[str(n)] = s
    rows = sorted([a, b, c] for a, b, c in links if a != b)
    return {"day": day, "dataset": DS, "version": 2, "cards": cards, "nodes": out}, rows


class FakeHabitHost(FakeHost):
    """tests/fakes.FakeHost (data.host.Host over in-memory days) with a FAKE HostSpec (configured_test =
    test), interning cards, and the original habit double's leniency: listed days are not checked
    against the in-memory dict (plan tests use day-less hosts). ds / art are unique per instance
    family, so caches keyed by state.profile.host_key never mix with a real host's."""

    def __init__(self, days, links, ds=DS, art="/nonexistent/habitus_fake/", clean=CLEAN, test=TEST):
        super().__init__(make_spec(ds=ds, os="linux", clean=clean, test=test), days, links, art=art)
        self.obs = dict(OBS)

    def _existing(self, days):
        return list(days)

    def gt(self):
        raise AssertionError("habit-channel code must never read attack labels")


def make_habit_host(attack=True):
    days, links = {}, {}
    for d in CLEAN + TEST:
        ev = list(BENIGN) + (list(ATTACK) if attack and d == TEST[1] else [])
        days[d], links[d] = build_day(d, ev)
    return FakeHabitHost(days, links)


class FakeAnswerSource:
    """Deterministic AnswerSource: p_yes / option probabilities from sha1(qkey); counts calls.
    Inclusion buckets "gen:never-seen" get a small p_yes, so surprise tests see a tail."""

    tag = "fake"

    def __init__(self):
        self.asked = Counter()

    def _u(self, key, i=0):
        h = hashlib.sha1(f"{key}#{i}".encode()).digest()
        return (int.from_bytes(h[:8], "big") + 1) / (2 ** 64 + 2)

    def score(self, questions, orders=("fwd", "rev")):
        from habitus.questions.habit.types import GEN_NEVER_SEEN, Answer
        out = {}
        for q in questions:
            k = q.qkey
            self.asked[k] += 1
            if len(q.options) == 2:
                p = 0.02 + 0.96 * self._u(k)
                if q.bucket is not None and q.bucket.key == GEN_NEVER_SEEN:
                    p = 0.001 + 0.01 * self._u(k)
                probs = (p, 1.0 - p)
            else:
                w = np.array([self._u(k, i) for i in range(len(q.options))])
                probs = tuple(float(x) for x in w / w.sum())
            out[k] = Answer(probs, {"fwd": probs, "rev": probs}, 1.0)
        return out

    def close(self):
        pass
