"""IFC router: one information-flow label for every field a habit view can carry, the per-family
read sets, and the build-time assertion that a view never carries its own answer.

    from habitus.questions.habit import ifc
    ifc.label_view(view)        # {"entity.image_dir": Label.ATAU, "reference.seen_days": Label.R0, ...}
    ifc.violations(view)        # [] for a clean view, else one line per broken rule
    ifc.assert_view(view)       # raises IFCError (views.build calls it before returning)
    ifc.read_set(Family.P2_modify)   # every "section.field" the P2 question may read

Labels (types.Label):
  K0    invariant field or host constant (incl. the raw strings of an IDENTITY-attested identity)
  K+    test-day behaviour: forbidden in habit views
  R0    a fact of the reference days only (seen days, usual elements, prefix level, ...)
  Atau  a typed word of a closed vocabulary derived from an attacker-choosable string
  A     a raw unattested string: forbidden in habit views
  K-    anything padding can move toward benign (capped lists, name-keyed habits, retrieval): forbidden
        in every new view

A habit view may carry only K0, R0 and Atau (types.HABIT_LABELS). The raw identity strings
(RAW_FIELDS: file_name, executable, command_line, path) are labelled per view: K0 when the view's
meta tier0 is IDENTITY (the attested-name rule), A otherwise, so the same field is legal in an
attested view and a violation in any other.

Answer fields: ANSWER_FIELDS[f] lists every dotted name whose presence would hand family f its own
answer (today's behaviour of the queried attribute and everything derived from it, including the
leaks that are easy to miss: kappa embeds parent_name, so P1 must not see the full-kappa reference
facts; behaviour_seen_before hashes the whole card). Two assertions keep them out:
  - at import: no field of FIELD_LABELS for a view kind is an answer field of a family asked on it;
  - at build time (assert_view): no answer field of the view's families is present in the view.
views.REMOVAL is asserted (in views.py) to contain every answer field of its kind's families.

Label-free: pure schema code over field names; it reads no data, no label, no reference.
"""

from habitus.questions.habit import types as T
from habitus.questions.habit.types import Family, Label, ViewKind

K0, KP, R0, AT, A, KM = Label.K0, Label.KPLUS, Label.R0, Label.ATAU, Label.A, Label.KMINUS

RAW_FIELDS = frozenset(("file_name", "executable", "command_line", "path"))   # entity raw strings
ATTESTED_TIERS = ("IDENTITY", "LINEAGE")                                      # meta tier0 names
REMOVED_SECTIONS = ("behaviour", "deviation", "evidence", "precedent")       # never in a habit view


class IFCError(ValueError):
    """A habit view that breaks the IFC rules (a forbidden label, an unattested raw string, a removed
    or answer field present, an unlabelled field)."""


# --------------------------------------------------------------------------- #
# Field labels of the habit views (every field views.py can emit)
# --------------------------------------------------------------------------- #
# entity words (vocab.ENTITY_KEYS; the raw strings only appear for IDENTITY-attested identities)
_ENTITY = {
    "process": {"type": K0, "os": K0, "image_dir": AT, "ext_class": AT, "role": AT, "cmd_shape": AT,
                "file_name": K0, "executable": K0, "command_line": K0},
    "file": {"type": K0, "os": K0, "dir": AT, "ext_class": AT, "path_shape": AT, "path": K0},
    "netflow": {"type": K0, "os": K0, "endpoint": AT, "port_class": AT},
}
# invariant hard bits only (state section_hard_bits minus every behaviour-derived bit)
_HARD = {
    "process": {"image_location": AT, "image_deleted": K0, "runs_from_writable": AT, "observable": K0},
    "file": {"location": AT, "executable_type": AT, "observable": K0},      # executable_type: extension only
    "netflow": {"endpoint": AT, "port_class": AT, "observable": K0},
}
_REF_COMMON = {"days": R0, "seen_days": R0, "frequency": R0}
_SECTIONS = {
    ViewKind.V_beh: {
        "entity": _ENTITY["process"], "hard_bits": _HARD["process"],
        "reference": {**_REF_COMMON, "usual_modify_dirs": R0, "usual_peers": R0, "usual_children": R0,
                      "other_children": R0, "usual_exec_dirs": R0},
        "class_habits": {"role": AT, "image_dir": AT, "modify_dirs": R0, "peers": R0, "children": R0,
                         "exec_dirs": R0}},
    ViewKind.V_par: {
        "entity": _ENTITY["process"], "hard_bits": _HARD["process"],
        "reference": {**_REF_COMMON, "usual_parents": R0, "other_parents": R0}},
    ViewKind.V_file: {
        "entity": _ENTITY["file"], "hard_bits": _HARD["file"],
        "reference": {**_REF_COMMON, "class_seen_days": R0, "usual_writers": R0, "class_usual_writers": R0,
                      "class_other_writers": R0, "usual_executors": R0, "class_usual_executors": R0,
                      "class_other_executors": R0}},
    ViewKind.V_net: {
        "entity": _ENTITY["netflow"], "hard_bits": _HARD["netflow"],
        "reference": {**_REF_COMMON, "prefix_level": R0, "usual_owners": R0, "class_usual_owners": R0,
                      "class_other_owners": R0}},
    ViewKind.V_own: {
        "entity": _ENTITY["process"],                     # the OWNER (typed-only for the gen:typed view)
        "connection": {"endpoint": AT, "port_class": AT},
        "reference": {"days": R0, "owner": R0, "owner_seen_days": R0, "owner_frequency": R0,
                      "owner_networks": R0, "owner_addresses": R0, "host_networks": R0}},
}
FIELD_LABELS = {(k, s, f): lab for k, secs in _SECTIONS.items() for s, fields in secs.items()
                for f, lab in fields.items()}
for (_k, _s, _f) in FIELD_LABELS:
    assert _s in T.VIEW_SECTIONS[_k], (_k, _s)

# --------------------------------------------------------------------------- #
# Answer fields per family (dotted names; a bare name is a whole section)
# --------------------------------------------------------------------------- #
_BEHAVIOUR_ALL = REMOVED_SECTIONS + ("reference.behaviour_seen_before",)
ANSWER_FIELDS = {
    Family.P1_parent: _BEHAVIOUR_ALL + ("entity.parent_name", "hard_bits.has_parent", "reference.parent_child_seen",
                                        "reference.identity_seen_days", "reference.identity_frequency"),
    Family.P2_modify: _BEHAVIOUR_ALL + ("hard_bits.writes_to_writable_dirs", "hard_bits.wrote_file_run_later",
                                        "reference.name_usual_write_dirs"),
    Family.P3_peers: _BEHAVIOUR_ALL + ("hard_bits.external_peers", "reference.name_usual_peers"),
    Family.P4_children: _BEHAVIOUR_ALL + ("reference.name_usual_children",),
    Family.P5_exec_from: _BEHAVIOUR_ALL + ("hard_bits.runs_from_dirs", "hard_bits.ran_file_written_today"),
    Family.F1_writer: _BEHAVIOUR_ALL + ("hard_bits.n_writers", "hard_bits.run_after_written_today",
                                        "hard_bits.deleted_today"),
    Family.F2_executor: _BEHAVIOUR_ALL + ("hard_bits.executed", "hard_bits.run_after_written_today"),
    Family.N1_owner: _BEHAVIOUR_ALL + ("hard_bits.n_processes", "entity.processes"),
    Family.N2_address: _BEHAVIOUR_ALL + ("entity.remote_ip", "entity.remote_port", "connection.remote_ip",
                                         "connection.remote_port", "connection.prefix_level",
                                         "reference.prefix_level", "reference.endpoint_chain_unseen_levels",
                                         "deviation.remote_prefix_unseen_levels"),
}
assert set(ANSWER_FIELDS) == set(T.FAMILIES)


def read_set(family):
    """Every "section.field" the question of `family` may read (the labelled fields of its view)."""
    kind = T.FAMILY_SPEC[Family.of(family)].view
    return frozenset(f"{s}.{f}" for (k, s, f) in FIELD_LABELS if k == kind)


READ_SETS = {f: read_set(f) for f in T.FAMILIES}

# build-time (import) assertion: a view kind never labels an answer field of a family asked on it
for _f, _ans in ANSWER_FIELDS.items():
    _bad = READ_SETS[_f] & set(_ans)
    _bad |= {a for a in _ans if "." not in a and a in T.VIEW_SECTIONS[T.FAMILY_SPEC[_f].view]}
    assert not _bad, f"{_f.value}: its view {T.FAMILY_SPEC[_f].view.value} labels its answer fields {sorted(_bad)}"


# --------------------------------------------------------------------------- #
# Labelling and checks of one view
# --------------------------------------------------------------------------- #
def attested(view):
    """True when the view's rendered identity is IDENTITY-attested (raw strings allowed)."""
    return str(view.meta.get("tier0", "")) in ATTESTED_TIERS


def label_view(view):
    """{"section.field": Label} of every top-level field of every section of `view`. KeyError on a
    field FIELD_LABELS does not know. Raw identity strings: K0 on an attested view, A otherwise."""
    att = attested(view)
    out = {}
    for s, body in view.sections.items():
        for f in body:
            lab = FIELD_LABELS[(view.kind, s, f)]
            if s == "entity" and f in RAW_FIELDS and not att:
                lab = A
            out[f"{s}.{f}"] = lab
    return out


def _present(view, dotted):
    s, _, f = dotted.partition(".")
    if s not in view.sections:
        return False
    return not f or f in view.sections[s]


def violations(view):
    """Every IFC rule `view` breaks, one line each ([] = clean): a field without a label, a label
    outside HABIT_LABELS, a raw string on a view that is not IDENTITY-attested, a removed field or an
    answer field of one of the view's families present."""
    out = []
    att = attested(view)
    for s, body in view.sections.items():
        for f in body:
            key = (view.kind, s, f)
            if key not in FIELD_LABELS:
                out.append(f"{view.kind.value}: unlabelled field {s}.{f}")
                continue
            lab = FIELD_LABELS[key]
            if s == "entity" and f in RAW_FIELDS:
                if not att:
                    out.append(f"{view.kind.value}: raw string {s}.{f} on a {view.meta.get('tier0')} view")
                    lab = A
            if lab not in T.HABIT_LABELS:
                out.append(f"{view.kind.value}: field {s}.{f} labelled {lab.value}")
    for r in view.removed_fields:
        if _present(view, r):
            out.append(f"{view.kind.value}: removed field {r} present")
    for fam in T.VIEW_FAMILIES[view.kind]:
        for a in ANSWER_FIELDS[fam]:
            if _present(view, a):
                out.append(f"{view.kind.value}: answer field {a} of {fam.value} present")
    return out


def assert_view(view):
    """Raise IFCError listing every violation of `view` (views.build calls this before returning)."""
    bad = violations(view)
    if bad:
        raise IFCError("; ".join(bad))
    return view


__all__ = ["RAW_FIELDS", "FIELD_LABELS", "ANSWER_FIELDS", "READ_SETS", "IFCError", "read_set", "label_view",
           "violations", "assert_view", "attested"]
