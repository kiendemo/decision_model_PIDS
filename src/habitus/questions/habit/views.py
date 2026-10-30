"""Habit views: what the habit decision model reads about ONE signature, as a pure function of
(the card's type and invariant fields, the reference Profile2) - never of the day.

    from habitus.questions.habit import views
    vs = views.views_for(card, prof, "k2", T.available(ds, "k2"))   # {ViewKind: View}, no V_own
    own = views.own_views(flow_card, ["id:<owner kappa'>", "gen:typed"], prof, "k2")
    v = views.build(card, T.ViewKind.V_beh, prof, "k2")              # one view; ifc.assert_view'ed
    v.text                                                           # compact JSON, sanitised

Five kinds (types.VIEW_SECTIONS gives the section order):
  V_beh   process, P2-P5   entity, hard_bits, reference (kappa'), class_habits (role class, image_dir)
  V_par   process, P1      entity, hard_bits, reference (kappa': usual parents)
  V_file  file, F1-F2      entity, hard_bits, reference (path kappa + (dir, ext class) usual writers /
                           executors)
  V_net   netflow, N1      entity, hard_bits, reference (connection kappa, host prefix level, usual
                           owners of (endpoint, port class, prefix level))
  V_own   netflow x owner bucket, N2   entity (the OWNER), connection (endpoint, port class),
                           reference (the owner's reference network summary). An "id:<owner>"
                           view renders an IDENTITY-attested owner (meta tier0 IDENTITY, owner =
                           its ident); the ONE "gen:typed" view of a flow signature is shared by
                           every other owner and renders no owner fact (meta tier0 NONE, owner "").
                           meta ident is the flow's ident in both.

Removal table (REMOVAL): every behaviour, deviation, evidence and precedent field is
absent from every view (they are never computed: the builder reads only card["type"], card["os"]
and card["invariant"]); in addition the fields the table names are never emitted. In particular:
  - process views drop parent_name and every parent-derived field; their identity key is kappa'
    (types.ident_key: the invariant minus parent_name), because the full kappa embeds parent_name
    and would hand P1 its answer; their reference facts are keyed by kappa' too (seen_days,
    frequency), never by the full kappa (E2's identity_seen_days / identity_frequency);
  - file views give executable_type from the extension only (E2 also reads "executed");
  - netflow views never render the remote address or port (only endpoint / port class); V_own
    also drops every chain element of the connection and its prefix level (N2's answer is a
    function of them).
E2's name-keyed habits (name_seen_days, name_usual_*) never enter: a masquerader chooses the name
whose habits would be compared. Every usual element is keyed by EXACT identity (Profile2), only
identity-attested identities are listed by name (<= MAX_LISTED, ordered by (-reference days, key),
the same order targets.candidates lists its id: buckets in), with day bins "n/k".

Attested-name rule: a raw name / path / command line appears only when the view's identity
is IDENTITY-attested by tier0 (prof.tier0, link-free); otherwise only typed words. The entity words,
class words and role class come from habitus.vocab, the prefix level and the N1 class level from
habitus.profile2 (one definition each, shared with Profile2's class keys: tested equal).

Length: every section is bounded by the reference (<= MAX_LISTED names per list) and the invariant
(state truncation 200 / 160 chars), so a view cannot grow with test-day activity, and the 1,024-token
refusal (text.fits) cannot be triggered by an attacker.

Serialisation: sections are sanitised (state.sanitize) and types.View serialises them with
state.builder.serialize (compact JSON, schema key order, lists in a fixed order): the same (card
invariant, profile) give byte-identical text. ifc.assert_view runs on every view before it is
returned.

Label-free: reads card invariants and the reference profile (pre-deployment clean days) only; no
label, no score, no link and no behaviour field of the scored day.
"""

from habitus.questions.habit import ifc, profile2, vocab
from habitus.questions.habit import types as T
from habitus.questions.habit.constants import MAX_LISTED, MAX_P1_NAMES
from habitus.questions.habit.types import Attest, Family, ViewKind
from habitus.state.profile import chain_of
from habitus.state.sanitize import sanitize
from habitus.state.sections import EXE_CHARS, EXEC_EXTS, NA, NOT_WRITABLE, WRITABLE, bin_word, yn

VIEW_NODE_TYPE = {ViewKind.V_beh: "process", ViewKind.V_par: "process", ViewKind.V_file: "file",
                  ViewKind.V_net: "netflow", ViewKind.V_own: "netflow"}
TYPED_OWNER = "not identity-attested"       # V_own reference "owner" of the shared gen:typed view
NO_LEVEL = profile2.NO_LEVEL                # N1 class level of a flow without a prefix chain ("none")

# --------------------------------------------------------------------------- #
# Removal table (dotted names, a bare name is a whole section)
# --------------------------------------------------------------------------- #
_ALL_BEHAVIOUR = ("behaviour", "deviation", "evidence", "precedent", "reference.behaviour_seen_before")
_NAME_KEYED = ("reference.name_seen_days", "reference.name_usual_children", "reference.name_usual_write_dirs",
               "reference.name_usual_peers")
_PARENT = ("entity.parent_name", "hard_bits.has_parent", "reference.parent_child_seen",
           "reference.identity_seen_days", "reference.identity_frequency")
_PROCESS_BEH_BITS = ("hard_bits.ran_file_written_today", "hard_bits.wrote_file_run_later",
                     "hard_bits.writes_to_writable_dirs", "hard_bits.runs_from_dirs", "hard_bits.external_peers")
_ADDRESS = ("entity.remote_ip", "entity.remote_port")
REMOVAL = {
    ViewKind.V_beh: _ALL_BEHAVIOUR + _PROCESS_BEH_BITS + _PARENT + _NAME_KEYED,
    ViewKind.V_par: _ALL_BEHAVIOUR + _PROCESS_BEH_BITS + _PARENT + _NAME_KEYED,
    ViewKind.V_file: _ALL_BEHAVIOUR + ("hard_bits.n_writers", "hard_bits.executed", "hard_bits.run_after_written_today",
                                       "hard_bits.deleted_today") + _NAME_KEYED,
    ViewKind.V_net: _ALL_BEHAVIOUR + ("hard_bits.n_processes", "entity.processes") + _ADDRESS + _NAME_KEYED,
    ViewKind.V_own: _ALL_BEHAVIOUR + ("hard_bits.n_processes", "entity.processes") + _ADDRESS
    + ("connection.remote_ip", "connection.remote_port", "connection.prefix_level", "reference.prefix_level",
       "reference.endpoint_chain_unseen_levels", "deviation.remote_prefix_unseen_levels") + _NAME_KEYED,
}
# every answer field of a family asked on a kind is in that kind's removal list (and ifc labels none of them)
for _k, _rm in REMOVAL.items():
    for _f in T.VIEW_FAMILIES[_k]:
        _miss = set(ifc.ANSWER_FIELDS[_f]) - set(_rm)
        assert not _miss, f"REMOVAL[{_k.value}] lacks answer fields {sorted(_miss)} of {_f.value}"


# --------------------------------------------------------------------------- #
# Typed words (habitus.vocab) and the prefix level (habitus.profile2): one definition each
# --------------------------------------------------------------------------- #
def dir_word(v):
    """A dir class word of the closed vocabulary (vocab.canon_dir: "other" for unknown text, "none"
    when empty)."""
    return vocab.canon_dir(v)


def ep_word(v):
    """An endpoint class word of the closed vocabulary (vocab.canon_endpoint: "unknown" otherwise)."""
    return vocab.canon_endpoint(v)


def entity_words(c0, tier):
    """vocab.entity_words: typed words, raw strings only for an IDENTITY-attested tier."""
    return dict(vocab.entity_words(c0, tier))


def role_of(c0, tier):
    """vocab.role_class: "unknown" unless the identity is IDENTITY-attested."""
    return vocab.role_class(c0, tier)


def prefix_level(card, prof):
    """profile2.prefix_level: leading elements of state.profile chain_of(card) found in the host's reference
    chains (None: no reference or no applicable chain)."""
    return profile2.prefix_level(card, prof)


def chain_level(element):
    """Chain index (0 = /8 or IPv6 /32 ... 3 = /32 or /128, 4 = ip|port) of one ip_levels element."""
    e = str(element)
    if "|" in e:
        return 4
    if "/" in e:
        return {32: 0, 48: 1, 64: 2, 128: 3}.get(int(e.rsplit("/", 1)[1]), 3)
    return min(e.count("."), 3)


# --------------------------------------------------------------------------- #
# Class keys shared with targets.candidates and Profile2.class_habit
# --------------------------------------------------------------------------- #
def beh_class(meta):
    """(role class, image_dir): P2 / P3 / P4 / P5 class habits."""
    return meta["role"], meta["image_dir"]


def file_class(meta):
    """(dir, ext class): F1 / F2 class habits."""
    return meta["dir"], meta["ext_class"]


def n1_class(meta):
    """(endpoint, port class, str(prefix level)): N1 class habits; a netflow without a prefix chain
    has level NO_LEVEL (= profile2.class_key, the producer's definition; tested equal)."""
    pl = meta["prefix_level"]
    return meta["endpoint"], meta["port_class"], NO_LEVEL if pl == "" else str(pl)


def invariant_card(card):
    """The only part of a card a view may read: type, os and the invariant, with the parent fields
    removed for processes (the view identity is kappa')."""
    t = card["type"]
    inv = {k: v for k, v in card["invariant"].items() if not (t == "process" and k in T.PARENT_FIELDS)}
    return {"type": t, "os": card.get("os") or "", "invariant": inv}


# --------------------------------------------------------------------------- #
# Reference renderings (R0)
# --------------------------------------------------------------------------- #
def _name_group(prof, local):
    """Every IDENTITY-attested ident of the reference sharing the type and display name of the
    equally named idents `local` of one list (Profile2.name_group: profile-wide), else - a protocol
    double without name_group - just `local`."""
    fn = getattr(prof, "name_group", None)
    group = set(local)
    if fn is not None:
        for e in local:
            group.update(fn(e))
    return sorted(group)


def display_names(idents, prof):
    """{ident: short name} for IDENTITY-attested idents (prof.display); equal names are told apart
    by the attested image path, then by a rank among the equal ones (sorted ident order). Equality
    is decided against EVERY identity-attested ident of the reference with that name (Profile2.
    name_group), not only inside this list: a rendered name then names one ident of the profile,
    so two views that print the same list have the same listed idents (check f; export groups
    signatures by view text)."""
    idents = sorted(set(idents))
    base = {e: str(prof.display(e) or "?") for e in idents}
    out = {}
    for name in sorted(set(base.values())):
        mine = [e for e in idents if base[e] == name]
        group = _name_group(prof, mine)
        if len(group) == 1:
            out[group[0]] = name
            continue
        longer = {}
        for e in group:
            inv = prof.ident_inv(e) or {}
            exe = str(inv.get("executable") or inv.get("path") or "")
            longer[e] = f"{name} ({exe[:EXE_CHARS]})" if exe and exe != name else name
        for e in mine:
            same = sorted(x for x in group if longer[x] == longer[e])
            out[e] = longer[e] if len(same) == 1 else f"{longer[e]} #{same.index(e) + 1}"
    return out


def listed_idents(elements, prof, n=MAX_LISTED):
    """Up to n IDENTITY-attested elements of an ident-valued habit {ident: reference days}, by
    (-days, key), and the number of the other elements (attested beyond n, pattern or unseen)."""
    att = sorted(((e, int(d)) for e, d in elements.items() if prof.ident_inv(e) is not None),
                 key=lambda x: (-x[1], x[0]))
    return att[:n], len(elements) - len(att[:n])


def _ids(elements, prof, k, n=MAX_LISTED):
    listed, other = listed_idents(elements, prof, n)
    names = display_names([e for e, _ in listed], prof)
    return [f"{names[e]} {d}/{k}" for e, d in listed], bin_word(other)


def _classes(elements, prefix, k, n=MAX_LISTED):
    """{class word: "n/k"} of a class-valued habit restricted to keys with `prefix`, by (-days, word)."""
    items = sorted(((e[len(prefix):], int(d)) for e, d in elements.items() if str(e).startswith(prefix)),
                   key=lambda x: (-x[1], x[0]))
    return {w: f"{d}/{k}" for w, d in items[:n]}


def _observable(prof):
    o = getattr(prof, "obs", None) if prof is not None else None
    return None if o is None else {"exec": bool(o.get("exec")), "lineage": bool(o.get("lineage"))}


def _hard_bits(c0, words, prof):
    t, inv = c0["type"], c0["invariant"]
    if t == "process":
        idir = words["image_dir"]
        out = {"image_location": idir, "image_deleted": yn(inv.get("image_deleted") == "yes"),
               "runs_from_writable": "yes" if idir in WRITABLE else "no" if idir in NOT_WRITABLE else NA}
    elif t == "file":
        out = {"location": words["dir"], "executable_type": yn(str(inv.get("ext") or "").lower() in EXEC_EXTS)}
    else:
        out = {"endpoint": words["endpoint"], "port_class": words["port_class"]}
    ob = _observable(prof)
    if ob is not None:
        out["observable"] = {"exec": yn(ob["exec"]), "lineage": yn(ob["lineage"])}
    return out


def _common_ref(prof, ident):
    k = prof.k
    return {"days": k, "seen_days": f"{prof.seen_days(ident)}/{k}", "frequency": prof.frequency(ident)}


def _ref_beh(prof, ident):
    k, ob = prof.k, _observable(prof) or {"exec": True, "lineage": True}
    out = _common_ref(prof, ident)
    out["usual_modify_dirs"] = _classes(prof.habit(ident, Family.P2_modify), "dir:", k)
    out["usual_peers"] = _classes(prof.habit(ident, Family.P3_peers), "ep:", k)
    if ob["lineage"]:
        out["usual_children"], out["other_children"] = _ids(prof.habit(ident, Family.P4_children), prof, k)
    else:
        out["usual_children"] = NA
    out["usual_exec_dirs"] = _classes(prof.habit(ident, Family.P5_exec_from), "dir:", k) if ob["exec"] else NA
    return out


def _class_habits(prof, meta):
    k, ob = prof.k, _observable(prof) or {"exec": True, "lineage": True}
    cls = beh_class(meta)
    return {"role": meta["role"], "image_dir": meta["image_dir"],
            "modify_dirs": _classes(prof.class_habit(Family.P2_modify, cls), "dir:", k),
            "peers": _classes(prof.class_habit(Family.P3_peers, cls), "ep:", k),
            "children": _classes(prof.class_habit(Family.P4_children, cls), "cls:", k) if ob["lineage"] else NA,
            "exec_dirs": _classes(prof.class_habit(Family.P5_exec_from, cls), "dir:", k) if ob["exec"] else NA}


def _ref_par(prof, ident):
    k, ob = prof.k, _observable(prof) or {"exec": True, "lineage": True}
    out = _common_ref(prof, ident)
    if ob["lineage"]:
        out["usual_parents"], out["other_parents"] = _ids(prof.habit(ident, Family.P1_parent), prof, k, MAX_P1_NAMES)
    else:
        out["usual_parents"] = NA
    return out


def _ref_file(prof, ident, meta):
    k, ob = prof.k, _observable(prof) or {"exec": True, "lineage": True}
    cls = file_class(meta)
    out = _common_ref(prof, ident)
    out["class_seen_days"] = f"{prof.class_days(Family.F1_writer, cls)}/{k}"
    out["usual_writers"] = _ids(prof.habit(ident, Family.F1_writer), prof, k)[0]
    out["class_usual_writers"], out["class_other_writers"] = _ids(prof.class_habit(Family.F1_writer, cls), prof, k)
    if ob["exec"]:
        out["usual_executors"] = _ids(prof.habit(ident, Family.F2_executor), prof, k)[0]
        out["class_usual_executors"], out["class_other_executors"] = \
            _ids(prof.class_habit(Family.F2_executor, cls), prof, k)
    else:
        out["usual_executors"] = NA
    return out


def _ref_net(prof, ident, meta, card):
    k = prof.k
    out = _common_ref(prof, ident)
    ch = chain_of(card)
    pl = meta["prefix_level"]
    out["prefix_level"] = NA if ch is None or pl == "" else f"{pl}/{len(ch)}"
    out["usual_owners"] = _ids(prof.habit(ident, Family.N1_owner), prof, k)[0]
    out["class_usual_owners"], out["class_other_owners"] = _ids(prof.class_habit(Family.N1_owner, n1_class(meta)),
                                                                prof, k)
    return out


def _networks(chains, level):
    return bin_word(sum(1 for e in chains if chain_level(e) == level))


def _ref_own(prof, owner):
    k = prof.k
    if owner is None:
        out = {"days": k, "owner": TYPED_OWNER}
    else:
        own = prof.chains(owner)
        out = {"days": k, "owner_seen_days": f"{prof.seen_days(owner)}/{k}", "owner_frequency": prof.frequency(owner),
               "owner_networks": _networks(own, 1), "owner_addresses": _networks(own, 3)}
    out["host_networks"] = _networks(prof.chains(None), 1)
    return out


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _meta(ident, tier, k_mode):
    m = {key: "" for key in T.VIEW_META_KEYS}
    m.update(ident=ident, tier0=Attest(tier).name, k_mode=k_mode)
    return m


def build(card, kind, prof, k_mode, owner=None):
    """One habit view of `card` (a stored card; only type / os / invariant are read). V_own: `card` is
    the netflow card and `owner` the owner's ident key (an IDENTITY-attested owner, bucket "id:<owner>")
    or None for the shared gen:typed view. k0 <=> prof is None (reference = {"days": 0}, no
    class_habits; V_own does not exist at k = 0). Raises ifc.IFCError on any IFC violation."""
    kind = ViewKind(kind)
    t = card["type"]
    if t != VIEW_NODE_TYPE[kind]:
        raise ValueError(f"{kind.value} is a {VIEW_NODE_TYPE[kind]} view, not a {t} view")
    if k_mode not in T.KMODES:
        raise ValueError(f"k_mode {k_mode!r} not in {T.KMODES}")
    if (k_mode == "k0") != (prof is None):
        raise ValueError(f"k_mode {k_mode} with {'no' if prof is None else 'a'} reference profile")
    if owner is not None and kind is not ViewKind.V_own:
        raise ValueError("owner is a V_own argument")
    if kind is ViewKind.V_own and prof is None:
        raise ValueError("V_own (N2) does not exist at k = 0")
    c0 = invariant_card(card)
    ident = T.ident_key(card)
    tier = Attest.NONE if prof is None else min(Attest(prof.tier0(c0)), Attest.IDENTITY)

    if kind is ViewKind.V_own:
        if owner is not None:
            inv = prof.ident_inv(owner)
            if inv is None:
                raise ValueError(f"owner {owner} is not IDENTITY-attested: it belongs to the gen:typed view")
            oc = invariant_card({"type": "process", "os": c0["os"], "invariant": dict(inv)})
            otier = Attest.IDENTITY
            ent = entity_words(oc, otier)
            orole = role_of(oc, otier)
        else:
            otier, orole = Attest.NONE, ""
            ent = {"type": "process", "os": c0["os"]}
        conn = {"endpoint": ep_word(c0["invariant"].get("endpoint")),
                "port_class": vocab.port_class(c0["invariant"].get("remote_port"))}
        meta = _meta(ident, otier, k_mode)
        pl = prefix_level(card, prof)
        meta.update(image_dir=ent.get("image_dir", ""), ext_class=ent.get("ext_class", ""), role=orole,
                    endpoint=conn["endpoint"], port_class=conn["port_class"], prefix_level="" if pl is None else pl,
                    owner=owner or "")
        sections = {"entity": ent, "connection": conn, "reference": _ref_own(prof, owner)}
    else:
        ent = entity_words(c0, tier)
        meta = _meta(ident, tier, k_mode)
        if t == "process":
            meta.update(image_dir=ent["image_dir"], ext_class=ent["ext_class"], role=role_of(c0, tier))
        elif t == "file":
            meta.update(dir=ent["dir"], ext_class=ent["ext_class"])
        else:
            pl = prefix_level(card, prof) if prof is not None else None
            meta.update(endpoint=ent["endpoint"], port_class=ent["port_class"], prefix_level="" if pl is None else pl)
        sections = {"entity": ent, "hard_bits": _hard_bits(c0, ent, prof)}
        if prof is None:
            sections["reference"] = {"days": 0}
        elif kind is ViewKind.V_beh:
            sections["reference"] = _ref_beh(prof, ident)
            sections["class_habits"] = _class_habits(prof, meta)
        elif kind is ViewKind.V_par:
            sections["reference"] = _ref_par(prof, ident)
        elif kind is ViewKind.V_file:
            sections["reference"] = _ref_file(prof, ident, meta)
        else:
            sections["reference"] = _ref_net(prof, ident, meta, c0)
    view = T.View(kind, t, sanitize(sections), REMOVAL[kind], meta)
    return ifc.assert_view(view)


def kinds_for(ntype, families, k_mode):
    """View kinds (no V_own) the given families of node type `ntype` need, in ViewKind order. At
    k = 0 the families that do not exist without a reference (P1, N2) are skipped."""
    want = set()
    for f in families:
        sp = T.FAMILY_SPEC[Family.of(f)]
        if sp.node_type != ntype or sp.view is ViewKind.V_own or (k_mode == "k0" and not sp.at_k0):
            continue
        want.add(sp.view)
    return [k for k in ViewKind if k in want]


def views_for(card, prof, k_mode, families):
    """{ViewKind: View} of every non-V_own view the card's applicable `families` need (e.g.
    T.available(ds, k_mode)); families of another node type are ignored."""
    return {kd: build(card, kd, prof, k_mode) for kd in kinds_for(card["type"], families, k_mode)}


def own_views(flow_card, owner_buckets, prof, k_mode):
    """{owner bucket key: V_own View} for N2: "id:<owner ident>" (an IDENTITY-attested owner, its own
    view) or "gen:typed" (one typed view shared by every other owner of the flow)."""
    out = {}
    for b in sorted(set(owner_buckets)):
        if b == T.GEN_TYPED:
            out[b] = build(flow_card, ViewKind.V_own, prof, k_mode, owner=None)
        elif b.startswith("id:"):
            out[b] = build(flow_card, ViewKind.V_own, prof, k_mode, owner=b[3:])
        else:
            raise ValueError(f"N2 owner bucket {b!r}")
    return out


__all__ = ["REMOVAL", "VIEW_NODE_TYPE", "TYPED_OWNER", "NO_LEVEL", "build", "views_for", "own_views", "kinds_for",
           "invariant_card", "entity_words", "role_of", "dir_word", "ep_word", "prefix_level", "chain_level",
           "display_names", "listed_idents", "beh_class", "file_class", "n1_class"]
