"""State sections: the compact facts one decision-model call reads about ONE node signature on ONE day.

Sections, always in SECTIONS order; a section that does not apply is omitted, never null:
  entity     type, os and the card's invariant (command_line <= 200 chars, executable <= 160,
             file path <= 160 with "" -> "unknown"; netflow remote_ip, remote_port, endpoint).
             The only section an attacker chooses the text of.
  hard_bits  kernel-event facts, no names: image location / deletion, write-then-exec flags,
             writes to writable dirs, executable file type, endpoint and port class, plus the
             host's observability. A fact the host's collector cannot record is "n/a" (exec for
             Windows collectors, lineage for CADETS), never "no".
  reference  the reference days (k shots): how often this identity (kappa) and its name were seen,
             frequency level, exact card seen, the name's usual children / write dirs / peers,
             parent-child pair seen, endpoint prefix-chain levels unseen. k = 0: {"days": 0}.
  deviation  set-membership novelty against the name's reference habits (monotone in added edges):
             new op keys, ops above their usual bin, new children / write dirs / exec dirs / peer
             classes; an unseen name says so and lists its ops instead. k >= 1.
  evidence   E1 (ablation ladder): one-hop provenance neighbours on the day, aggregated over ALL
             nodes carrying the signature (the 200 smallest ids): lines "<relation> <type>:<label>
             (xN)", <= 10 lines, unseen-kappa neighbours first (k >= 1), then rarer, then text.
  precedent  E1: the 2 nearest reference cards of the same type (RefIndex), as short identity +
             seen days + top ops. k >= 1.

E2 (the main system) keeps the section names and order and replaces three of them:
  deviation2   "identity_unseen" first, for netflows "remote_prefix_unseen_levels", then E1's
               deviation (E1's name keys).
  evidence_nf  name-free classes from closed word lists over <= 8 representative nodes, novelty =
               kappa unseen in the reference; `connected` only towards a netflow.
  precedent2   E1's nearest answers; a netflow precedent is rendered as class + prefix_match.
hard_tier(state) reads the HARD bits (bounded-acquittal exemption).

Empty lists are written [] (not "none": "none" is also a directory class). The host-level "n/a"
rule covers every field fed by an unobservable link code. Evidence is computed for a whole day's
head at once (searchsorted / lexsort over the day's link array).
"""

import re
from collections import OrderedDict

import numpy as np

from habitus.state.precedent import PREC_E, PREC_OPS, nearest_sigs  # noqa: F401  (PREC_E re-exported)
from habitus.state.profile import (BIN_ORD, EXEC, NET, SPAWN, WRITE, common_prefix, host_key, name_key,
                                   observability, port_class)

SECTIONS = ("entity", "hard_bits", "reference", "deviation", "evidence", "precedent")
NA = "n/a"
WRITABLE = frozenset(("staging", "cache", "home", "other"))
NOT_WRITABLE = frozenset(("system", "application", "config"))
EXEC_EXTS = frozenset(("exe", "dll", "sys", "scr", "ps1", "bat", "cmd", "vbs", "js", "hta", "sh", "py",
                       "pl", "so", "elf", "bin", "run", "jar", "msi", "lnk"))
CMD_CHARS, EXE_CHARS, PATH_CHARS, BASE_CHARS = 200, 160, 160, 40
LIST_CAP = 5                 # reference / deviation lists
EV_NODES, EV_LINES = 200, 10
NUM_WORDS = ("zero", "one", "two", "three", "four", "five or more")   # writers / processes keep top 5

TYPES = ("process", "file", "netflow")
TCODE = {t: i for i, t in enumerate(TYPES)}
REL_NAMES = ("spawned_by", "spawned", "wrote", "ran_file", "connected", "written_by", "run_by",
             "opened_by")
# REL[type code, link code, d]: d = 0 the node is the link's cause, 1 its effect; -1 = not shown
REL = np.full((3, 5, 2), -1, dtype=np.int64)
for (_t, _c, _d), _r in {("process", SPAWN, 1): 0, ("process", SPAWN, 0): 1, ("process", WRITE, 0): 2,
                         ("process", EXEC, 1): 3, ("process", NET, 0): 4, ("file", WRITE, 1): 5,
                         ("file", EXEC, 0): 6, ("netflow", NET, 1): 7}.items():
    REL[TCODE[_t], _c, _d] = _r
_SEP = re.compile(r"[\\/]")


def yn(b):
    return "yes" if b else "no"


def count_word(n):
    return NUM_WORDS[min(int(n), len(NUM_WORDS) - 1)]


def n_word(n):
    """Distinct-neighbour count of an evidence line: exact up to 7, then bin words."""
    return str(n) if n <= 7 else "8+" if n <= 31 else "32+" if n <= 127 else "128+"


def top_ops(ops, n):
    """Op keys by bin descending, then key."""
    return sorted(ops, key=lambda o: (-BIN_ORD.get(ops[o], 0), o))[:n]


def basename(path):
    return _SEP.split(str(path or "").rstrip("/\\"))[-1]


def label(card, port="raw"):
    """Short identity of a neighbour (evidence) or a reference card (precedent, port="class")."""
    t, inv = card["type"], card["invariant"]
    if t == "process":
        return f"{inv.get('file_name', '')} [{inv.get('image_dir', 'none')}]"
    if t == "file":
        base = basename(inv.get("path"))[:BASE_CHARS] or "unknown"
        return f"{inv.get('dir', 'none')}/{inv.get('ext', '')} {base}"
    p = inv.get("remote_port", "") if port == "raw" else port_class(inv.get("remote_port"))
    return f"{inv.get('endpoint', 'unknown')}:{p}"


# --------------------------------------------------------------------------- #
# Sections (one card)
# --------------------------------------------------------------------------- #
def section_entity(card, host_os):
    t, inv = card["type"], dict(card["invariant"])
    if t == "process":
        if "command_line" in inv:
            inv["command_line"] = inv["command_line"][:CMD_CHARS]
        if "executable" in inv:
            inv["executable"] = inv["executable"][:EXE_CHARS]
    elif t == "file":
        inv["path"] = (inv.get("path") or "unknown")[:PATH_CHARS]
    else:
        inv = {k: inv.get(k, "") for k in ("remote_ip", "remote_port", "endpoint")}
    return {"type": t, "os": card.get("os") or host_os, **inv}


def section_hard_bits(card, obs):
    t, inv = card["type"], card["invariant"]
    beh = card.get("behaviour") or {}
    flags = set(beh.get("flags") or ())
    ex = obs["exec"]
    if t == "process":
        idir = inv.get("image_dir", "none")
        out = {"image_location": idir,
               "image_deleted": yn(inv.get("image_deleted") == "yes"),
               "runs_from_writable": "yes" if idir in WRITABLE else "no" if idir in NOT_WRITABLE else NA,
               "ran_file_written_today": yn("executed_file_written_today" in flags) if ex else NA,
               "wrote_file_run_later": yn("wrote_file_executed_later" in flags) if ex else NA,
               "writes_to_writable_dirs": yn(bool(WRITABLE & set(beh.get("write_dirs") or ()))),
               "runs_from_dirs": sorted(beh.get("exec_dirs") or ()) if ex else NA,
               "external_peers": yn("external" in (beh.get("peers") or {})),
               "has_parent": yn(bool(inv.get("parent_name"))) if obs["parent_name"] else NA}
    elif t == "file":
        ext, executed = inv.get("ext", ""), bool(beh.get("executors"))
        out = {"location": inv.get("dir", "none"),
               "executable_type": yn(ext in EXEC_EXTS or (not ext and executed)),
               "run_after_written_today": yn("executed_after_write" in flags) if ex else NA,
               "deleted_today": yn("deleted" in flags),
               "n_writers": count_word(len(beh.get("writers") or ())),
               "executed": yn(executed) if ex else NA}
    else:
        out = {"endpoint": inv.get("endpoint", "unknown"),
               "port_class": port_class(inv.get("remote_port")),
               "n_processes": count_word(len(beh.get("processes") or ()))}
    out["observable"] = {"exec": yn(ex), "lineage": yn(obs["lineage"])}
    return out


def section_reference(prof, card, sig, kappa):
    """{"days": 0} without a reference (k = 0)."""
    if prof is None or not prof.k:
        return {"days": 0}
    k, t, inv = prof.k, card["type"], card["invariant"]
    nk = name_key(card)
    out = {"days": k,
           "identity_seen_days": f"{prof.kappa_days.get(kappa, 0)}/{k}",
           "identity_frequency": prof.level(kappa),
           "name_seen_days": f"{prof.name_days.get(nk, 0)}/{k}",
           "behaviour_seen_before": yn(sig in prof.sig_seen)}
    if t == "process":
        pn, lin = inv.get("parent_name"), prof.obs["lineage"]
        out["parent_child_seen"] = NA if prof.pair_count is None or not pn else \
            yn(prof.pair_count.get((pn.lower(), nk[1]), 0) > 0)
        out["name_usual_children"] = sorted(prof.children.get(nk, ()))[:LIST_CAP] if lin else NA
        out["name_usual_write_dirs"] = sorted(prof.write_dirs.get(nk, ()))[:LIST_CAP]
        out["name_usual_peers"] = sorted(prof.peers.get(nk, ()))[:LIST_CAP]
    elif t == "file":
        out["class_seen_days"] = f"{prof.name_days.get(nk, 0)}/{k}"
    else:
        cu = prof.chain_unseen(card)
        out["endpoint_chain_unseen_levels"] = NA if cu is None else f"{cu[0]}/{cu[1]}"
    return out


def section_deviation(prof, card):
    """None without a reference (k = 0): the section is then omitted."""
    if prof is None or not prof.k:
        return None
    beh = card.get("behaviour") or {}
    ops = beh.get("ops") or {}
    nk = name_key(card)
    if nk not in prof.name_days:
        return {"name_unseen": "yes", "ops_today": top_ops(ops, LIST_CAP)}
    om = prof.ops_max.get(nk, {})
    out = {"new_ops": sorted(o for o in ops if o not in om)[:LIST_CAP],
           "ops_above_usual": sorted(o for o in ops if o in om and BIN_ORD.get(ops[o], 0) > om[o])[:LIST_CAP]}
    if card["type"] == "process":
        def new(field, store):
            return sorted(set(beh.get(field) or ()) - store.get(nk, set()))[:LIST_CAP]
        out["new_children"] = new("children", prof.children) if prof.obs["lineage"] else NA
        out["new_write_dirs"] = new("write_dirs", prof.write_dirs)
        out["new_exec_dirs"] = new("exec_dirs", prof.exec_dirs) if prof.obs["exec"] else NA
        out["new_peer_classes"] = sorted(set((beh.get("peers") or {}).keys())
                                         - prof.peers.get(nk, set()))[:LIST_CAP]
    return out


# --------------------------------------------------------------------------- #
# evidence: one day's head at once
# --------------------------------------------------------------------------- #
class DayIndex:
    """One day's nodes grouped by signature (ids ascending) and its link array."""

    def __init__(self, host, day):
        nodes = host.nodes(day)
        self.cards = host.day(day)["cards"]
        ids = np.fromiter(nodes.keys(), dtype=np.int64, count=len(nodes))
        self.code_of = {}
        codes = np.fromiter((self.code_of.setdefault(s, len(self.code_of)) for s in nodes.values()),
                            dtype=np.int64, count=len(nodes))
        self.sigs = list(self.code_of)
        o = np.argsort(ids, kind="stable")
        self.ids, self.codes = ids[o], codes[o]                   # by node id: neighbour lookup
        g = np.lexsort((ids, codes))
        self.g_ids = ids[g]                                       # by (sig code, id): members
        self.g_start = np.searchsorted(codes[g], np.arange(len(self.sigs) + 1))
        self.links = host.links(day)

    def members(self, sig, cap=EV_NODES):
        c = self.code_of.get(sig)
        if c is None:
            return self.g_ids[:0]
        a = self.g_start[c]
        return self.g_ids[a:min(self.g_start[c + 1], a + cap)]


_DAYIDX = OrderedDict()
MAX_DAYIDX = 2


def day_index(host, day):
    """DayIndex cached per (host, day): builders of different levels / k share it."""
    key = (*host_key(host), day)
    if key in _DAYIDX:
        _DAYIDX.move_to_end(key)
        return _DAYIDX[key]
    ix = _DAYIDX[key] = DayIndex(host, day)
    while len(_DAYIDX) > MAX_DAYIDX:
        _DAYIDX.popitem(last=False)
    return ix


def section_evidence(host, day, sigs, prof=None, index=None):
    """E1 evidence: {sig: {"lines": [...], "more": M}} for every sig of `sigs` (one vectorised pass)."""
    ix = index or day_index(host, day)
    sigs = list(sigs)
    out = {s: {"lines": [], "more": 0} for s in sigs}
    if not sigs or not len(ix.links):
        return out
    mem = [ix.members(s) for s in sigs]
    sel = np.concatenate(mem) if mem else np.zeros(0, np.int64)
    if not len(sel):
        return out
    selq = np.repeat(np.arange(len(sigs)), [len(m) for m in mem])
    qtype = np.array([TCODE[host.card(s)["type"]] for s in sigs], dtype=np.int64)
    o = np.argsort(sel, kind="stable")
    sid, sq = sel[o], selq[o]
    L = ix.links
    q_all, nb_all, rel_all = [], [], []
    for d, col, other in ((0, 0, 1), (1, 1, 0)):                 # node is cause / effect
        x = L[:, col]
        pos = np.minimum(np.searchsorted(sid, x), len(sid) - 1)
        hit = sid[pos] == x
        q = sq[pos[hit]]
        code = L[hit, 2]
        ok = (code >= 1) & (code <= 4)
        q, code, nb = q[ok], code[ok], L[hit, other][ok]
        rel = REL[qtype[q], code, d]
        keep = rel >= 0
        q_all.append(q[keep])
        nb_all.append(nb[keep])
        rel_all.append(rel[keep])
    q, nb, rel = np.concatenate(q_all), np.concatenate(nb_all), np.concatenate(rel_all)
    pos = np.minimum(np.searchsorted(ix.ids, nb), len(ix.ids) - 1)
    has = ix.ids[pos] == nb                                       # neighbours without a card: dropped
    q, nb, rel, nbc = q[has], nb[has], rel[has], ix.codes[pos[has]]
    if not len(q):
        return out
    # neighbour signature -> line label id and "kappa unseen in the reference"
    ucodes, inv = np.unique(nbc, return_inverse=True)
    lab_id, labels, unseen = {}, [], np.zeros(len(ucodes), dtype=np.int64)
    lab_of = np.empty(len(ucodes), dtype=np.int64)
    use_ref = prof is not None and prof.k > 0
    for i, c in enumerate(ucodes.tolist()):
        s = ix.sigs[c]
        card = ix.cards[s]
        text = f"{card['type']}:{label(card)}"
        lab_of[i] = lab_id.setdefault(text, len(lab_id))
        if lab_of[i] == len(labels):
            labels.append(text)
        if use_ref:
            unseen[i] = host.inv_key(s) not in prof.kappa_days
    lab, nun = lab_of[inv], unseen[inv]
    # distinct (q, rel, label, neighbour node), then count per (q, rel, label)
    o = np.lexsort((nb, lab, rel, q))
    q, rel, lab, nb, nun = q[o], rel[o], lab[o], nb[o], nun[o]
    first = np.ones(len(q), dtype=bool)
    first[1:] = (q[1:] != q[:-1]) | (rel[1:] != rel[:-1]) | (lab[1:] != lab[:-1]) | (nb[1:] != nb[:-1])
    q, rel, lab, nun = q[first], rel[first], lab[first], nun[first]
    start = np.flatnonzero(np.r_[True, (q[1:] != q[:-1]) | (rel[1:] != rel[:-1]) | (lab[1:] != lab[:-1])])
    gq, grel, glab = q[start], rel[start], lab[start]
    gn = np.diff(np.r_[start, len(q)])
    gun = np.maximum.reduceat(nun, start)
    # line text rank (relation + label), shared by all queries
    pair = grel * len(labels) + glab
    upair, pinv = np.unique(pair, return_inverse=True)
    texts = [f"{REL_NAMES[p // len(labels)]} {labels[p % len(labels)]}" for p in upair.tolist()]
    trank = np.empty(len(texts), dtype=np.int64)
    trank[np.argsort(np.array(texts, dtype=object), kind="stable")] = np.arange(len(texts))
    order = np.lexsort((trank[pinv], gn, 1 - gun, gq))          # per q: unseen first, N asc, text
    gq, gn, pinv = gq[order], gn[order], pinv[order]
    bounds = np.flatnonzero(np.r_[True, gq[1:] != gq[:-1], True])
    for a, b in zip(bounds[:-1].tolist(), bounds[1:].tolist()):
        s = sigs[int(gq[a])]
        keep = range(a, min(b, a + EV_LINES))
        out[s] = {"lines": [f"{texts[pinv[i]]} (x{n_word(int(gn[i]))})" for i in keep],
                  "more": int(b - a - len(keep))}
    return out


# --------------------------------------------------------------------------- #
# precedent (E1)
# --------------------------------------------------------------------------- #
def section_precedent(host, prof, refindex, sigs):
    """{sig: {"nearest_reference": [...] | "none"}} (k >= 1) from nearest_sigs (persistent cache,
    RefIndex on a miss). `refindex`: None (get_refindex on a miss) or an explicit RefIndex."""
    out = {}
    for s, ex in nearest_sigs(host, prof.ref, sigs, refindex).items():
        items = []
        for r in ex:
            c = host.card(r)
            items.append({"identity": label(c, port="class"),
                          "seen_days": f"{prof.kappa_days.get(host.inv_key(r), 0)}/{prof.k}",
                          "ops": top_ops((c.get("behaviour") or {}).get("ops") or {}, PREC_OPS)})
        out[s] = {"nearest_reference": items or "none"}
    return out


_OBS = OrderedDict()


def _observability(host, days):
    key = (*host_key(host), tuple(sorted(days)))
    if key not in _OBS:
        _OBS[key] = observability(host, key[2])
        while len(_OBS) > 64:
            _OBS.popitem(last=False)
    return _OBS[key]


# =========================================================================== #
# E2: routed, name-free sections. Section keys keep E1's names (deviation = deviation2,
# evidence = evidence_nf, precedent = precedent2), because SYSTEM_E2 describes the sections by those
# names; the order inside a state is SECTIONS order.
# =========================================================================== #
E2_ROUTES = {"qv": ("entity", "hard_bits", "reference", "deviation"),
             "qt": ("entity", "hard_bits", "deviation", "evidence"),
             "qa": ("entity", "reference", "precedent")}
# a "yes" on any of these hard_bits puts the node in the hard tier, which QA never lowers
HARD = ("ran_file_written_today", "wrote_file_run_later", "image_deleted", "run_after_written_today")
NF_REPS, NF_ENTRIES = 8, 8
REL_PRIORITY = ("ran_file", "wrote", "spawned", "spawned_by", "connected", "run_by", "written_by", "opened_by")
REL_RANK = tuple(REL_PRIORITY.index(r) for r in REL_NAMES)
REL_CONNECTED = REL_NAMES.index("connected")    # evidence_nf: only towards a netflow
EXT_CLASSES = (
    ("executable", frozenset(("exe", "dll", "sys", "scr", "msi", "com", "elf", "bin", "run", "so", "dylib", "ko"))),
    ("script", frozenset(("ps1", "bat", "cmd", "vbs", "js", "hta", "sh", "py", "pl", "rb", "php", "jar", "lnk"))),
    ("library", frozenset(("lib", "a", "o", "pyc", "class"))),
    ("archive", frozenset(("zip", "rar", "7z", "gz", "tgz", "tar", "bz2", "xz", "cab", "iso", "img"))),
    ("document", frozenset(("doc", "docx", "xls", "xlsx", "ppt", "pptx", "pdf", "rtf", "txt", "csv", "html", "htm",
                            "xml", "json", "eml", "msg", "jpg", "jpeg", "png", "gif"))),
)
# closed vocabularies of the class words evidence_nf may print (cards' dir classes, the resolver's
# endpoint classes + cards' "internal"); anything else is clamped, so no card text can leak through
DIR_CLASSES = frozenset(("system", "application", "home", "staging", "cache", "config", "other", "none"))
ENDPOINTS = frozenset(("loopback", "private", "enclave", "external", "internal", "unknown"))
EXT_WORDS = tuple(n for n, _ in EXT_CLASSES) + ("data", "none")
PORT_CLASSES = ("well-known", "registered", "ephemeral", "none")


def bin_word(n):
    """Card count bins ("0", "1", "2-7", "8-31", "32-127", "128+")."""
    n = int(n)
    return "0" if n <= 0 else "1" if n == 1 else "2-7" if n <= 7 else "8-31" if n <= 31 else \
        "32-127" if n <= 127 else "128+"


def _word(v, allowed, missing, other):
    return missing if not v else v if v in allowed else other


def ext_class(card):
    """executable | script | library | archive | document | data | none from the file's extension;
    a file without extension that was executed is executable."""
    ext = str(card["invariant"].get("ext") or "").lower()
    for name, exts in EXT_CLASSES:
        if ext in exts:
            return name
    if ext:
        return "data"
    return "executable" if (card.get("behaviour") or {}).get("executors") else "none"


def nf_class(card):
    """Name-free class of a neighbour: process:<image_dir>, file:<dir>/<extclass>,
    net:<endpoint>/<port class>. Never a file name, process name, path, IP or command line."""
    t, inv = card["type"], card["invariant"]
    if t == "process":
        return f"process:{_word(inv.get('image_dir'), DIR_CLASSES, 'none', 'other')}"
    if t == "file":
        return f"file:{_word(inv.get('dir'), DIR_CLASSES, 'none', 'other')}/{ext_class(card)}"
    return f"net:{_word(inv.get('endpoint'), ENDPOINTS, 'unknown', 'unknown')}/{port_class(inv.get('remote_port'))}"


def _nf_render(entries, known, b, novelty):
    if novelty:
        return {"new": entries, "known": bin_word(known), "representatives": b}
    return {"all": entries, "known": NA, "representatives": b}


def section_evidence_nf(host, day, sigs, prof=None, index=None):
    """{sig: evidence_nf} for every sig of `sigs`, one vectorised pass as E1 evidence.

    Representatives = the NF_REPS smallest node ids carrying the sig on `day`, so a 126-node and a
    1-node signature with the same neighbourhood read the same. Over the representatives' one-hop
    links (E1's relation table), each neighbour becomes its name-free class; with a reference
    (k >= 1) a neighbour is "new" when its kappa is unseen there. `connected` is kept only when the
    neighbour is a netflow (a code-4 link that ends at a file is dropped, also from "known"). An
    entry "<relation> <class>": "in a/b nodes" counts the representatives (a of b) with at least one
    new neighbour of that relation and class; <= NF_ENTRIES entries by relation priority, then a
    desc, then text. "known" = bin of the known neighbours summed over representatives (distinct
    (representative, neighbour) pairs). k = 0: no novelty, every neighbour under "all", known "n/a"."""
    ix = index or day_index(host, day)
    sigs = list(sigs)
    novelty = prof is not None and prof.k > 0
    mem = [ix.members(s, NF_REPS) for s in sigs]
    nrep = [len(m) for m in mem]
    out = {s: _nf_render({}, 0, nrep[i], novelty) for i, s in enumerate(sigs)}
    if not sigs or not len(ix.links) or not sum(nrep):
        return out
    sel = np.concatenate(mem)
    selq = np.repeat(np.arange(len(sigs)), nrep)
    qtype = np.array([TCODE[host.card(s)["type"]] for s in sigs], dtype=np.int64)
    o = np.argsort(sel, kind="stable")
    sid, sq = sel[o], selq[o]
    L = ix.links
    cols = []
    for d, col, other in ((0, 0, 1), (1, 1, 0)):                 # representative is cause / effect
        x = L[:, col]
        pos = np.minimum(np.searchsorted(sid, x), len(sid) - 1)
        hit = sid[pos] == x
        q, code, nb, rep = sq[pos[hit]], L[hit, 2], L[hit, other], x[hit]
        ok = (code >= 1) & (code <= 4)
        q, code, nb, rep = q[ok], code[ok], nb[ok], rep[ok]
        rel = REL[qtype[q], code, d]
        keep = rel >= 0
        cols.append((q[keep], rep[keep], rel[keep], nb[keep]))
    q, rep, rel, nb = (np.concatenate(c) for c in zip(*cols))
    pos = np.minimum(np.searchsorted(ix.ids, nb), len(ix.ids) - 1)
    has = ix.ids[pos] == nb                                       # neighbours without a card: dropped
    q, rep, rel, nb, nbc = q[has], rep[has], rel[has], nb[has], ix.codes[pos[has]]
    if not len(q):
        return out
    ucodes, inv = np.unique(nbc, return_inverse=True)
    is_net = np.array([ix.cards[ix.sigs[c]]["type"] == "netflow" for c in ucodes.tolist()], dtype=bool)
    keep = (rel != REL_CONNECTED) | is_net[inv]                   # `connected` only to a netflow
    if not keep.all():
        q, rep, rel, nb, nbc = q[keep], rep[keep], rel[keep], nb[keep], nbc[keep]
        if not len(q):
            return out
        ucodes, inv = np.unique(nbc, return_inverse=True)
    cls_id, classes = {}, []
    cls_of = np.empty(len(ucodes), dtype=np.int64)
    new_of = np.ones(len(ucodes), dtype=bool)
    for i, c in enumerate(ucodes.tolist()):
        s = ix.sigs[c]
        text = nf_class(ix.cards[s])
        cls_of[i] = cls_id.setdefault(text, len(cls_id))
        if cls_of[i] == len(classes):
            classes.append(text)
        if novelty:
            new_of[i] = host.inv_key(s) not in prof.kappa_days
    cls, new = cls_of[inv], new_of[inv]
    known = np.zeros(len(sigs), dtype=np.int64)
    if (~new).any():                                              # distinct (q, representative, neighbour)
        kn = np.unique(np.stack([q[~new], rep[~new], nb[~new]], axis=1), axis=0)
        known = np.bincount(kn[:, 0], minlength=len(sigs))
    entries = {}
    if new.any():                                                 # distinct (q, rel, class, rep) -> a
        e = np.unique(np.stack([q[new], rel[new], cls[new], rep[new]], axis=1), axis=0)
        g, a = np.unique(e[:, :3], axis=0, return_counts=True)
        for (qi, r, c), n in zip(g.tolist(), a.tolist()):
            entries.setdefault(qi, []).append((REL_RANK[r], -n, f"{REL_NAMES[r]} {classes[c]}", n))
    for qi, s in enumerate(sigs):
        ents = sorted(entries.get(qi, ()))[:NF_ENTRIES]
        out[s] = _nf_render({t: f"in {n}/{nrep[qi]} nodes" for _r, _n, t, n in ents}, int(known[qi]),
                            nrep[qi], novelty)
    return out


def section_deviation2(prof, card, kappa):
    """deviation2: "identity_unseen": "yes" first when the kappa is unseen in the reference, for
    netflows "remote_prefix_unseen_levels": "x/5" | "n/a" (the profile's prefix chain), then E1's
    deviation unchanged (E1's name keys and lists; a netflow's name stays (endpoint, port class)).
    None without a reference (k = 0)."""
    if prof is None or not prof.k:
        return None
    out = {}
    if kappa not in prof.kappa_days:
        out["identity_unseen"] = "yes"
    if card["type"] == "netflow":
        cu = prof.chain_unseen(card)
        out["remote_prefix_unseen_levels"] = NA if cu is None else f"{cu[0]}/{cu[1]}"
    out.update(section_deviation(prof, card))
    return out


def net_precedent(node, ref_card):
    """precedent2 identity of a netflow precedent: class words and the prefix depth it shares with
    the node, never its address."""
    inv = ref_card["invariant"]
    pm = common_prefix(node, ref_card)
    return (f"net:{_word(inv.get('endpoint'), ENDPOINTS, 'unknown', 'unknown')}/"
            f"{port_class(inv.get('remote_port'))} prefix_match:{NA if pm is None else pm}")


def section_precedent2(host, prof, refindex, sigs):
    """E1 precedent (the same nearest_sigs answers, cache included; same seen_days / ops); netflow
    identities rendered by net_precedent. {sig: {"nearest_reference": [...] | "none"}} (k >= 1)."""
    out = {}
    for s, ex in nearest_sigs(host, prof.ref, sigs, refindex).items():
        node, items = host.card(s), []
        net = node["type"] == "netflow"
        for r in ex:
            c = host.card(r)
            items.append({"identity": net_precedent(node, c) if net else label(c, port="class"),
                          "seen_days": f"{prof.kappa_days.get(host.inv_key(r), 0)}/{prof.k}",
                          "ops": top_ops((c.get("behaviour") or {}).get("ops") or {}, PREC_OPS)})
        out[s] = {"nearest_reference": items or "none"}
    return out


def hard_tier(state):
    """True when any HARD bit of the state's hard_bits is "yes" (QV / QT states carry hard_bits)."""
    hb = state.get("hard_bits") or {}
    return any(hb.get(h) == "yes" for h in HARD)
