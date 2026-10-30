"""Habit-channel data contract: every object that crosses a habit-module boundary, with its
invariants checked at construction.

    Ref            one scored (host, day) with its reference days and k mode
    Attest         attestation tiers NONE < PATTERN < IDENTITY < LINEAGE
    Family         the 9 habit families (FAMILY_SPEC: node type, readout kind, view, link needs, ...)
    ViewKind       V_beh / V_par / V_file / V_net / V_own
    Label          IFC field labels K0 / K+ / R0 / Atau / A / K-
    Bucket         one element bucket of a family (closed-vocabulary or identity-attested phrase)
    Candidates     the buckets (inclusion) or listed names (P1) of one (view, family): f(view, reference)
    View           one habit view: sections in a fixed order, its serialised text and sha1
    HabitQuestion  one (view, family, bucket) question; qkey = the answer key
    Answer         p_bar over canonical options (T = 1) + per-order raw readouts
    AnswerSource   protocol of anything that answers HabitQuestions (DMAnswerSource, test fakes)
    NodeObs        per-node observed buckets / options from the node's OWN complete op keys and links
    FamilyScore, NodeHabit, NullUnit    x_f / z_f / E / E_mono / s_hat per node; null units
    CompositeRow   one node-day of the composite
    ScoreDay       one score file (columnar node rows + routes + fallback), JSON round trip
    AlertDay       one test day's alerts at one budget (strong / weak tier)
    Item           one exported training item (rendered prompt, offered letters, soft target)
    Profile2Like   protocol of the reference-side habit index (profile2.Profile2 implements it)

The decision-model bridge also lives here: to_typed (HabitQuestion -> dm.base.TypedQuestion),
answer_of (dm.base.Decision -> Answer) and DMAnswerSource (any DecisionModel as an AnswerSource).

Nothing here reads data or labels.
"""

import enum
import json
import math
import re
from dataclasses import dataclass, field
from typing import Mapping, NamedTuple, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from habitus.data.cards import signature
from habitus.dm.base import LETTERS
from habitus.questions.habit.constants import (CLONE_ARMS, DEV_HOSTS, KPLANS, LEVELS, MAX_BUCKETS, MAX_OPTIONS,
                                               MAX_P1_NAMES, SPEC_VERSION, TRAIN_ONLY)
from habitus.state.builder import serialize, state_hash
from habitus.state.sections import DIR_CLASSES, ENDPOINTS, PORT_CLASSES

NODE_TYPES = ("process", "file", "netflow")
KMODES = ("k0", "k1", "k2", "allc-pre")             # Ref.k_mode
ORDERS = ("fwd", "rev")                             # option orders; fwd = canonical, rev = reversed
ROLES = ("test", "null")                            # score-file roles
# inner = the temperature day's items; select = the held-out dev host's selection-day items (arm
# comparison: never trained)
ITEM_ROLES = ("train", "inner", "select")
TRAIN_ROLES = ("train", "inner")                    # the only roles a trainer / temperature fit may read
ITEM_SOURCES = ("code",)                            # every target is computed by code

# closed vocabularies in a fixed order (sets equal state.sections'; asserted below)
DIR_ORDER = ("system", "application", "home", "staging", "cache", "config", "other", "none")
EP_ORDER = ("loopback", "private", "enclave", "internal", "external", "unknown")
assert set(DIR_ORDER) == set(DIR_CLASSES) and set(EP_ORDER) == set(ENDPOINTS)
PORT_ORDER = tuple(PORT_CLASSES)

# canonical option keys of the readouts (fwd order = this order, rev = reversed)
INCLUSION_OPTIONS = ("yes", "no")
P1_OPTIONS = ("usual", "attested-other", "never-seen", "none")
N2_OPTIONS = ("same-address", "same-network", "other-programs", "no-program")

# bucket keys: "dir:<DIR_ORDER word>" (P2, P5) | "ep:<EP_ORDER word>" (P3) | "id:<ident key>" (a listed
# identity-attested element: P4, F1, F2, N1; N2 owners) | "cls:<DIR_ORDER word>" (k = 0 class buckets
# of P4, F1, F2, N1: the image_dir class of the other end) | the generic ones below
GEN_ATTESTED_OTHER = "gen:attested-other"   # identity-attested on a reference day, not a listed one
GEN_NEVER_SEEN = "gen:never-seen"           # not identity-attested (unseen, placeholder or empty identity)
GEN_TYPED = "gen:typed"                     # N2: every owner that is not identity-attested (one typed view)
GENERIC_BUCKETS = (GEN_ATTESTED_OTHER, GEN_NEVER_SEEN, GEN_TYPED)
BUCKET_PREFIXES = ("dir:", "ep:", "id:", "cls:", "gen:")
NO_BUCKET = "-"                             # the bucket field of a choice-family question / item


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class Attest(enum.IntEnum):
    """Attestation tier of a node (ordered: a higher tier implies every lower one)."""
    NONE = 0          # identity never seen on a reference day
    PATTERN = 1       # seen, but placeholder-bearing or uninformative: typed-only rendering, c_K cap
    IDENTITY = 2      # exact identity seen, placeholder-free: raw strings, identity habits, R0 stratum
    LINEAGE = 3       # IDENTITY + full kappa seen + every causal parent present today attested in role

    @property
    def identity_attested(self):
        return self >= Attest.IDENTITY


class Family(str, enum.Enum):
    P1_parent = "P1"
    P2_modify = "P2"
    P3_peers = "P3"
    P4_children = "P4"
    P5_exec_from = "P5"
    F1_writer = "F1"
    F2_executor = "F2"
    N1_owner = "N1"
    N2_address = "N2"

    @property
    def spec(self):
        return FAMILY_SPEC[self]

    @classmethod
    def of(cls, value):
        """Family from its value ("P1"), its name ("P1_parent") or itself."""
        if isinstance(value, Family):
            return value
        try:
            return cls(value)
        except ValueError:
            return cls[value]


class ViewKind(str, enum.Enum):
    V_beh = "V_beh"       # process, shared by P2-P5
    V_par = "V_par"       # process, P1
    V_file = "V_file"     # file, F1-F2
    V_net = "V_net"       # netflow, N1
    V_own = "V_own"       # netflow x owner bucket, N2


class Label(str, enum.Enum):
    """IFC labels of view fields."""
    K0 = "K0"             # invariant field or host constant
    KPLUS = "K+"          # test-day behaviour: forbidden in habit views
    R0 = "R0"             # reference-only fact
    ATAU = "Atau"         # typed word derived from an attacker-choosable string (closed vocabulary)
    A = "A"               # raw unattested string: forbidden in habit views
    KMINUS = "K-"         # anything padding can move toward benign: forbidden in every new view


HABIT_LABELS = frozenset((Label.K0, Label.R0, Label.ATAU))     # the only labels a habit view may carry
# A raw string of an IDENTITY-attested identity is labelled K0 (it is an invariant the reference
# attests); the same field of any other identity is A and must not appear (attested-name rule).


# --------------------------------------------------------------------------- #
# Families (question bank, availability)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FamilySpec:
    family: Family
    qid: str                       # question-bank id
    node_type: str                 # the node type the family is asked about
    kind: str                      # "inclusion" (A / B per bucket) | "choice" (4 options per view)
    view: ViewKind
    needs: Optional[str]           # link code the family needs on the host: "spawn" | "exec" | None
    monotone: bool                 # enters E_mono (edge-monotone per node)
    at_k0: bool                    # applicable without a reference (k = 0)
    options: tuple                 # canonical option keys
    source: str                    # where S_f(v) / the observed option comes from (code)


FAMILY_SPEC = {s.family: s for s in (
    FamilySpec(Family.P1_parent, "H-P1-parent", "process", "choice", ViewKind.V_par, "spawn", False, False,
               P1_OPTIONS, "kappa' of the code-1 link cause whose card file_name equals this card's "
                           "parent_name; several distinct such kappa' or none with a parent_name -> n/a"),
    FamilySpec(Family.P2_modify, "H-P2-modify", "process", "inclusion", ViewKind.V_beh, None, True, True,
               INCLUSION_OPTIONS, "dir classes of the card's op keys write|create|rename|delete:<dir class>"),
    FamilySpec(Family.P3_peers, "H-P3-peers", "process", "inclusion", ViewKind.V_beh, None, True, True,
               INCLUSION_OPTIONS, "endpoint classes of the netflow effects of the node's code-4 links, and "
                                  "of the card's op keys <verb>:net-<endpoint>"),
    FamilySpec(Family.P4_children, "H-P4-children", "process", "inclusion", ViewKind.V_beh, "spawn", True, True,
               INCLUSION_OPTIONS, "ident keys of the process effects of the node's code-1 links"),
    FamilySpec(Family.P5_exec_from, "H-P5-exec-from", "process", "inclusion", ViewKind.V_beh, "exec", True, True,
               INCLUSION_OPTIONS, "dir classes of the file causes of code-3 links whose effect is the node"),
    FamilySpec(Family.F1_writer, "H-F1-writer", "file", "inclusion", ViewKind.V_file, None, True, True,
               INCLUSION_OPTIONS, "ident keys of the process causes of code-2 links whose effect is the file"),
    FamilySpec(Family.F2_executor, "H-F2-executor", "file", "inclusion", ViewKind.V_file, "exec", True, True,
               INCLUSION_OPTIONS, "ident keys of the process effects of code-3 links whose cause is the file"),
    FamilySpec(Family.N1_owner, "H-N1-owner", "netflow", "inclusion", ViewKind.V_net, None, True, True,
               INCLUSION_OPTIONS, "ident keys of the process causes of code-4 links whose effect is the "
                                  "netflow (process -> file 'send' links of CADETS never count)"),
    FamilySpec(Family.N2_address, "H-N2-address", "netflow", "choice", ViewKind.V_own, None, True, False,
               N2_OPTIONS, "per owner bucket: deepest level of the flow's ip_levels chain found in the "
                           "owner's reference chains, then in the host's"),
)}
FAMILIES = tuple(FAMILY_SPEC)
FAMILIES_OF = {t: tuple(f for f in FAMILIES if FAMILY_SPEC[f].node_type == t) for t in NODE_TYPES}
MONOTONE = frozenset(f for f in FAMILIES if FAMILY_SPEC[f].monotone)
VIEW_FAMILIES = {k: tuple(f for f in FAMILIES if FAMILY_SPEC[f].view == k) for k in ViewKind}

# Link availability measured 2026-10-21 by counting codes in cards/<ds>/links/*.json.gz (label-free):
# spawn (code 1) absent on CADETS_E3 and CLEARSCOPE_E3; exec (code 3) present only on CADETS_E3 and
# THEIA_E3. Frozen.
SPAWN_HOSTS = ("THEIA_E3", "ATLASV2_EDR", "optc_h501")
EXEC_HOSTS = ("CADETS_E3", "THEIA_E3")
MEASURED_HOSTS = DEV_HOSTS + TRAIN_ONLY


def available(ds, k_mode="k2", obs=None):
    """Families answerable on host `ds` under `k_mode`: the frozen table for the 5 measured hosts;
    any other host (test doubles) needs `obs` = {"exec", "lineage"} bools measured label-free on its
    pre-deployment link files (state.profile.observability)."""
    if ds in MEASURED_HOSTS:
        spawn, exe = ds in SPAWN_HOSTS, ds in EXEC_HOSTS
    elif obs is not None:
        spawn, exe = bool(obs["lineage"]), bool(obs["exec"])
    else:
        raise KeyError(f"{ds}: no frozen family-availability row; pass its measured observability")
    if k_mode not in KMODES:
        raise ValueError(f"k_mode {k_mode!r} not in {KMODES}")
    return frozenset(f for f, s in FAMILY_SPEC.items()
                     if (s.needs != "spawn" or spawn) and (s.needs != "exec" or exe)
                     and (k_mode != "k0" or s.at_k0))


# --------------------------------------------------------------------------- #
# Identities
# --------------------------------------------------------------------------- #
PARENT_FIELDS = ("parent_name",)
PLACEHOLDER_RE = re.compile(r"<(hex:\d+|rand:\d+|guid|n)>")       # cards.abstract_text / normalize_cmd
IDENTITY_FIELDS = {"process": ("file_name", "executable", "command_line"), "file": ("path",),
                   "netflow": ("remote_ip",)}


def ident_key(card):
    """The EXACT identity a habit is keyed by: kappa' = signature({type, invariant minus parent
    fields}) for a process (kappa embeds parent_name, which P1 asks about); for a file (path kappa)
    and a netflow (connection kappa) it is the full kappa."""
    inv = card["invariant"]
    if card["type"] == "process":
        inv = {k: v for k, v in inv.items() if k not in PARENT_FIELDS}
    return signature({"type": card["type"], "invariant": inv})


def full_kappa(card):
    """kappa = signature({type, invariant}), equal to Host.inv_key(sig) of the card."""
    return signature({"type": card["type"], "invariant": card["invariant"]})


def has_placeholder(card):
    """True when any invariant field carries a cards.abstract_text placeholder (<hex:n>, <rand:n>,
    <guid>, <n>): such an identity is a pattern, so it is at most PATTERN-attested."""
    return any(PLACEHOLDER_RE.search(str(v)) for v in card["invariant"].values())


def uninformative(card):
    """True when every identity field of the card is empty (a file without a path, a netflow
    without an address, a process with neither name, image nor command line). An empty identity
    cannot be attested: it is at most PATTERN-attested."""
    inv = card["invariant"]
    return all(not str(inv.get(f) or "").strip() for f in IDENTITY_FIELDS[card["type"]])


# --------------------------------------------------------------------------- #
# Days (the pre-deployment window)
# --------------------------------------------------------------------------- #
def first_test_day(host):
    """The host's first CONFIGURED test day: the frozen HostSpec.configured_test of the host (before
    any card-file filter: a test day whose card file is missing must still close the window); a test
    double without a spec gives its declared host.test. Raises when the host has no test day at all:
    the window is then undefined, never "every clean day"."""
    days = None
    spec = getattr(host, "spec", None)
    if spec is not None and not callable(spec):
        days = getattr(spec, "configured_test", None)
        days = list(days) if days else None
    if days is None:
        days = list(host.test)
    if not days:
        raise ValueError(f"{host.ds}: no configured test day: the pre-deployment window is undefined")
    return min(days)


def predeploy(host):
    """Usable clean days strictly before the host's first configured test day (first_test_day), date
    order. Every reference, null and training day of the habit channel is one of them (CADETS_E3:
    04-02..04-05)."""
    t0 = first_test_day(host)
    return [d for d in host.clean if d < t0]


def inner_day(host):
    """The temperature-fit day of a training host: its last pre-deployment day (never a training
    day c, never in any reference set)."""
    pre = predeploy(host)
    if not pre:
        raise ValueError(f"{host.ds}: no pre-deployment day")
    return pre[-1]


def assert_predeploy(host, days):
    """Raise unless every day is a pre-deployment clean day."""
    ok = set(predeploy(host))
    bad = [d for d in days if d not in ok]
    if bad:
        raise ValueError(f"{host.ds}: days {bad} are not pre-deployment clean days")


@dataclass(frozen=True)
class Ref:
    """One scored (host, day) and its reference. ref_days are normalised to sorted unique.
    k0: no reference; k1: 1 day (training augmentation only); k2: 2 days (test) or 1 day (a k = 2
    null day); allc-pre: the pre-deployment window (minus the day itself on a null / training day)."""
    ds: str
    day: str
    ref_days: tuple = ()
    k_mode: str = "k2"

    def __post_init__(self):
        object.__setattr__(self, "ref_days", tuple(sorted(set(self.ref_days))))
        n = len(self.ref_days)
        if self.k_mode not in KMODES:
            raise ValueError(f"k_mode {self.k_mode!r} not in {KMODES}")
        if self.day in self.ref_days:
            raise ValueError(f"{self.ds}: day {self.day} is one of its own reference days")
        if (self.k_mode == "k0") != (n == 0):
            raise ValueError(f"k_mode {self.k_mode} with {n} reference days")
        if (self.k_mode == "k1" and n != 1) or (self.k_mode == "k2" and n not in (1, 2)):
            raise ValueError(f"k_mode {self.k_mode} with {n} reference days")

    @property
    def k(self):
        return len(self.ref_days)

    def to_json(self):
        return {"ds": self.ds, "day": self.day, "ref_days": list(self.ref_days), "k_mode": self.k_mode}

    @classmethod
    def from_json(cls, d):
        return cls(d["ds"], d["day"], tuple(d.get("ref_days") or ()), d["k_mode"])


# --------------------------------------------------------------------------- #
# Buckets, candidates, views, questions, answers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Bucket:
    family: Family
    # text: the phrase text.py inserts verbatim - closed vocabulary or an identity-attested name,
    # never unattested card text
    key: str            # see BUCKET_PREFIXES
    text: str
    is_generic: bool = False

    def __post_init__(self):
        object.__setattr__(self, "family", Family.of(self.family))
        if not self.key.startswith(BUCKET_PREFIXES):
            raise ValueError(f"bucket key {self.key!r} has no known prefix {BUCKET_PREFIXES}")
        if self.key.startswith("gen:") != bool(self.is_generic):
            raise ValueError(f"bucket {self.key!r}: is_generic must be {self.key.startswith('gen:')}")
        if self.key.startswith("gen:") and self.key not in GENERIC_BUCKETS:
            raise ValueError(f"unknown generic bucket {self.key!r}")


@dataclass(frozen=True)
class Candidates:
    """What one (view, family) asks about; a function of (view, reference) only.
    inclusion: buckets = <= MAX_BUCKETS in a fixed order (listed identities by (-reference days,
    key), then the generic ones; or the class words in DIR_ORDER / EP_ORDER); listed = ().
    P1: buckets = (); listed = <= MAX_P1_NAMES usual-parent identities (id: buckets) for option A,
    alphabetical by text. N2: buckets = the owner buckets are per node (NodeObs.owners); () here."""
    family: Family
    buckets: tuple = ()
    listed: tuple = ()
    applicable: bool = True
    reason: str = ""

    def __post_init__(self):
        object.__setattr__(self, "family", Family.of(self.family))
        if len(self.buckets) > MAX_BUCKETS:
            raise ValueError(f"{self.family.value}: {len(self.buckets)} buckets > {MAX_BUCKETS}")
        if len(self.listed) > MAX_P1_NAMES:
            raise ValueError(f"{self.family.value}: {len(self.listed)} listed names > {MAX_P1_NAMES}")
        if len({b.key for b in self.buckets}) != len(self.buckets):
            raise ValueError(f"{self.family.value}: duplicate bucket keys")

    def keys(self):
        return tuple(b.key for b in self.buckets)


VIEW_SECTIONS = {
    ViewKind.V_beh: ("entity", "hard_bits", "reference", "class_habits"),
    ViewKind.V_par: ("entity", "hard_bits", "reference"),
    ViewKind.V_file: ("entity", "hard_bits", "reference"),
    ViewKind.V_net: ("entity", "hard_bits", "reference"),
    ViewKind.V_own: ("entity", "connection", "reference"),
}
# View.meta: NOT rendered, not hashed; what non-text consumers (surprise strata, candidates) need.
# Required keys first; the rest are "" when they do not apply.
VIEW_META_REQUIRED = ("ident", "tier0", "k_mode")
VIEW_META_KEYS = VIEW_META_REQUIRED + ("image_dir", "ext_class", "role", "dir", "endpoint", "port_class",
                                       "prefix_level", "owner")


@dataclass(frozen=True, eq=False)
class View:
    """One habit view. `sections` (plain dicts) must be in VIEW_SECTIONS[kind] order (a section may
    be absent, e.g. no reference at k = 0); `text` = state.builder.serialize (sanitised compact JSON),
    `sha1` = state_hash(text). Equality / hash = (kind, sha1): two identities that render the same
    bytes ARE the same view (byte invariance)."""
    kind: ViewKind
    node_type: str
    sections: Mapping
    removed_fields: tuple = ()
    meta: Mapping = field(default_factory=dict)
    text: str = field(init=False)
    sha1: str = field(init=False)

    def __post_init__(self):
        kind = ViewKind(self.kind)
        object.__setattr__(self, "kind", kind)
        order = VIEW_SECTIONS[kind]
        names = list(self.sections)
        if any(n not in order for n in names) or names != [n for n in order if n in self.sections]:
            raise ValueError(f"{kind.value}: sections {names} not in the order {order}")
        if self.node_type not in NODE_TYPES:
            raise ValueError(f"node type {self.node_type!r}")
        missing = [k for k in VIEW_META_REQUIRED if k not in self.meta]
        if missing:
            raise ValueError(f"{kind.value}: meta lacks {missing}")
        text = serialize(dict(self.sections))
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "sha1", state_hash(text))

    def __eq__(self, other):
        return isinstance(other, View) and (self.kind, self.sha1) == (other.kind, other.sha1)

    def __hash__(self):
        return hash((self.kind, self.sha1))


def qkey(view_sha1, family, bucket_key=None):
    """Answer key of one question: '<view sha1>|<family value>|<bucket key or ->'."""
    return f"{view_sha1}|{Family.of(family).value}|{bucket_key or NO_BUCKET}"


@dataclass(frozen=True, eq=False)
class HabitQuestion:
    """One habit question (text.build). Inclusion: bucket set, options = INCLUSION_OPTIONS.
    Choice (P1, N2): bucket None (N2: the owner bucket is in the view's meta "owner"), options =
    P1_OPTIONS / N2_OPTIONS. option_text is in CANONICAL order; text.render lays the letters
    out per order (fwd = canonical, rev = reversed)."""
    family: Family
    view: View
    bucket: Optional[Bucket]
    options: tuple
    option_text: tuple
    text: str

    def __post_init__(self):
        fam = Family.of(self.family)
        object.__setattr__(self, "family", fam)
        sp = fam.spec
        if tuple(self.options) != sp.options or len(self.option_text) != len(self.options):
            raise ValueError(f"{fam.value}: options {self.options} / texts {len(self.option_text)}")
        if len(self.options) > MAX_OPTIONS:
            raise ValueError(f"{fam.value}: {len(self.options)} options > {MAX_OPTIONS}")
        if (sp.kind == "inclusion") != (self.bucket is not None):
            raise ValueError(f"{fam.value}: an inclusion question needs a bucket, a choice question none")
        if self.view.kind != sp.view:
            raise ValueError(f"{fam.value} is asked on {sp.view.value}, not {self.view.kind.value}")

    @property
    def qkey(self):
        return qkey(self.view.sha1, self.family, self.bucket.key if self.bucket else None)

    @property
    def letters(self):
        return LETTERS[:len(self.options)]


def order_keys(options, order):
    """Canonical option keys in the letter order of `order` (A first)."""
    if order not in ORDERS:
        raise ValueError(f"order {order!r} not in {ORDERS}")
    return tuple(options) if order == "fwd" else tuple(reversed(options))


def to_canonical(probs_by_letter, order):
    """Letter-order probabilities of one order -> canonical option order."""
    p = tuple(float(x) for x in probs_by_letter)
    return p if order == "fwd" else tuple(reversed(p))


def pbar(dists, eps=1e-300):
    """The readout: geometric mean of the per-order distributions (each in CANONICAL order, T = 1),
    renormalised over the offered options. One distribution is returned renormalised."""
    a = np.asarray(dists, dtype=np.float64)
    if a.ndim != 2 or not len(a):
        raise ValueError("pbar needs a non-empty list of distributions")
    g = np.exp(np.log(np.maximum(a, eps)).mean(axis=0))
    return tuple(float(x) for x in g / g.sum())


@dataclass(frozen=True)
class Answer:
    probs: tuple                    # p_bar over canonical options, T = 1, sums to 1
    raw: Mapping = field(default_factory=dict)     # order -> canonical-order probabilities (T = 1)
    cov_min: float = 1.0            # min letter coverage over the orders (encoders: 1.0)

    def __post_init__(self):
        p = tuple(float(x) for x in self.probs)
        if not p or not all(math.isfinite(x) and x >= 0 for x in p) or abs(sum(p) - 1.0) > 1e-6:
            raise ValueError(f"answer probabilities {p} are not a distribution")
        object.__setattr__(self, "probs", p)

    @property
    def p_yes(self):
        return self.probs[0]


@runtime_checkable
class AnswerSource(Protocol):
    """Anything that answers habit questions: DMAnswerSource over a DecisionModel, test fakes.
    Keys of the result are HabitQuestion.qkey; every question gets an answer."""
    tag: str

    def score(self, questions: Sequence[HabitQuestion]) -> dict: ...

    def close(self) -> None: ...


# --------------------------------------------------------------------------- #
# Decision-model bridge
# --------------------------------------------------------------------------- #
def _orders(orders):
    """A non-empty subset of ORDERS, in ORDERS order."""
    want = set(orders)
    if not want or not want <= set(ORDERS):
        raise ValueError(f"orders {orders!r} must be a non-empty subset of {ORDERS}")
    return tuple(o for o in ORDERS if o in want)


def to_typed(q, orders=ORDERS):
    """The dm.base.TypedQuestion of one HabitQuestion: one Form per order (ORDERS order; forms[0]
    is the first, canonical when fwd is asked). Form.system = SYSTEM_H, Form.state = the view text,
    Form.question = the question text, Form.options = the option texts in that order's letter
    order, Form.canon = the canonical option index of each letter. qtype 'choice'; group
    "<view kind>|<view sha1>" (one shared prefix / encode-once group per view)."""
    from habitus.dm.base import Form, TypedQuestion
    from habitus.questions.habit.text import SYSTEM_H
    n = len(q.options)
    forms = []
    for o in _orders(orders):
        canon = tuple(range(n)) if o == "fwd" else tuple(reversed(range(n)))
        texts = tuple(q.option_text) if o == "fwd" else tuple(reversed(q.option_text))
        forms.append(Form(o, SYSTEM_H, q.view.text, q.text, texts, canon))
    return TypedQuestion(q.qkey, q.family.value, "choice", tuple(forms), f"{q.view.kind.value}|{q.view.sha1}")


def answer_of(d, orders=ORDERS):
    """The Answer of one dm.base.Decision.
    layout 'mq': probs = pbar over the members of 'fwd'; raw = {'fwd': that pbar} (renormalised
    even with one member); cov_min = 1.0.
    Otherwise: raw[o] = the canonical-order member distribution when there is one member, else the
    pbar over the members; probs = pbar over orders x members (orders outer, members inner);
    cov_min = d.cov_min."""
    if d.layout == "mq":
        p = pbar(list(d.canon_probs["fwd"]))
        return Answer(p, {"fwd": p}, 1.0)
    raw, every = {}, []
    for o in _orders(orders):
        members = [tuple(float(x) for x in m) for m in d.canon_probs[o]]
        if not members:
            raise ValueError(f"{d.qkey}: no member distribution for order {o}")
        raw[o] = members[0] if len(members) == 1 else pbar(members)
        every += members
    return Answer(pbar(every), raw, float(d.cov_min))


class DMAnswerSource:
    """Any dm.base.DecisionModel as an AnswerSource: score(questions, orders) -> {qkey: Answer},
    refused(questions) -> {qkey: reason}. Duplicate qkeys are answered once (two different
    questions under one qkey raise). A layout-'mq' model is always asked the canonical form."""

    def __init__(self, dm):
        self.dm = dm
        self.tag = dm.tag

    def _typed(self, questions, orders):
        orders = ORDERS if self.dm.layout == "mq" else _orders(orders)
        out = {}
        for q in questions:
            tq = to_typed(q, orders)
            prev = out.get(tq.qkey)
            if prev is None:
                out[tq.qkey] = tq
            elif prev.forms != tq.forms:
                raise ValueError(f"two different questions share the answer key {tq.qkey}")
        return out

    def refused(self, questions):
        typed = self._typed(questions, ORDERS)
        return dict(self.dm.refused(list(typed.values())) or {})

    def score(self, questions, orders=ORDERS):
        typed = self._typed(questions, orders)
        if not typed:
            return {}
        dec = self.dm.decide(list(typed.values()))
        miss = [k for k in typed if k not in dec]
        if miss:
            raise KeyError(f"{self.tag}: the decision model answered no decision for {len(miss)} questions, "
                           f"e.g. {miss[:3]}")
        orders = ORDERS if self.dm.layout == "mq" else _orders(orders)
        return {k: answer_of(dec[k], orders) for k in typed}

    def spec(self):
        return self.dm.spec()

    def close(self):
        self.dm.close()


# --------------------------------------------------------------------------- #
# Per-node observations and habit statistics
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OwnerObs:
    """One N2 owner bucket of one netflow node."""
    bucket: str          # "id:<owner ident>" (identity-attested owner) | GEN_TYPED
    owner: str           # the owner's ident key (also for GEN_TYPED)
    observed: str        # N2 option key (N2_OPTIONS)

    def __post_init__(self):
        if self.observed not in N2_OPTIONS:
            raise ValueError(f"N2 observed option {self.observed!r}")
        if not (self.bucket == GEN_TYPED or self.bucket.startswith("id:")):
            raise ValueError(f"N2 owner bucket {self.bucket!r}")


@dataclass(frozen=True)
class NodeObs:
    """targets.observe: what ONE node did on the day, from its own complete op keys and links,
    mapped onto its signature's candidates. For every applicable inclusion family f, present[f]
    exists (frozenset of candidate keys, possibly empty -> x_f = 0); P1 is in choice when applicable
    and unambiguous; N2 applicable <=> N2 not in na (owners may be empty -> x_N2 = 0). Every family
    of the node's type is either applicable or in na."""
    node: int
    sig: str
    ntype: str
    ident: str
    present: Mapping = field(default_factory=dict)
    choice: Mapping = field(default_factory=dict)
    owners: tuple = ()
    na: frozenset = frozenset()

    def __post_init__(self):
        fams = set(FAMILIES_OF[self.ntype])
        pres = {Family.of(f): frozenset(v) for f, v in self.present.items()}
        cho = {Family.of(f): v for f, v in self.choice.items()}
        na = frozenset(Family.of(f) for f in self.na)
        object.__setattr__(self, "present", pres)
        object.__setattr__(self, "choice", cho)
        object.__setattr__(self, "na", na)
        object.__setattr__(self, "owners", tuple(self.owners))
        for f in set(pres) | set(cho) | na:
            if f not in fams:
                raise ValueError(f"{f.value} is not a {self.ntype} family")
        for f in pres:
            if FAMILY_SPEC[f].kind != "inclusion" or f in na:
                raise ValueError(f"present[{f.value}] for a non-inclusion or n/a family")
        for f, v in cho.items():
            if f is not Family.P1_parent or v not in P1_OPTIONS or f in na:
                raise ValueError(f"choice[{f.value}] = {v!r}")
        undecided = [f for f in fams if f not in na and f not in pres and f not in cho and f is not Family.N2_address]
        if undecided:
            raise ValueError(f"node {self.node}: families {[f.value for f in undecided]} neither observed nor n/a")
        if self.owners and Family.N2_address in na:
            raise ValueError(f"node {self.node}: N2 owners on an n/a family")

    def applicable(self, family):
        f = Family.of(family)
        return f in FAMILIES_OF[self.ntype] and f not in self.na


@dataclass(frozen=True)
class FamilyScore:
    family: Family
    x: Optional[float]              # raw surprise (None = n/a); >= 0
    p: Optional[float] = None       # null upper-tail p (None before the null transform / n/a)
    z: Optional[float] = None       # -ln p
    stratum: Optional[tuple] = None  # (node type, family value, identity-attested flag or None = merged)


class NullUnit(NamedTuple):
    """One null node-day of one family (surprise.FamilyNull input)."""
    day: str
    ntype: str
    family: Family
    attested: bool                  # identity-attested (the R0 stratum flag)
    x: float


@dataclass(frozen=True)
class NodeHabit:
    """surprise / null2 output for one node-day."""
    node: int
    ntype: str
    attested: bool
    fams: Mapping                   # Family -> FamilyScore (applicable families only)
    E: float                        # max over admitted families of z_f (0.0 when none)
    E_mono: float                   # the same over MONOTONE families
    s_hat: Optional[float] = None   # null2 EMap(E); None when s_hat is not admitted / not fitted


@dataclass(frozen=True)
class CompositeRow:
    """score.composite of one node-day."""
    node: int
    sig: str
    ntype: str
    qv: float
    qt: float
    qa: Optional[float]             # None = QA not asked
    hard: bool
    tier: Attest
    s_hat: Optional[float]          # None at HB0
    base: float                     # max(qv, qt, s_hat)
    a: float                        # acquittal weight actually applied
    s: float                        # base * (1 - LAMBDA * a)


ROW_FIELDS = ("id", "sig", "s", "base", "a", "tier", "qv", "qt", "s_hat", "E", "E_mono")
REQUIRED_META = ("ds", "day", "level", "kplan", "role", "ref", "caps", "head_rule", "spec")


@dataclass
class ScoreDay:
    """One habit-channel score file: scores/<ds>/<level dir>/<day>.json.gz =
    {"meta", "head", "routes": {sig: {"qv", "qt", "qa", "hard", "cov_min"}},
     "nodes": {column: [value per row]}, "fallback": {non-head sig: float}}.
    Rows = EVERY node of the day whose signature is in the head (all members, not representatives);
    columns ROW_FIELDS plus optional "z.<family value>" / "x.<family value>" (None = n/a).
    Non-head nodes score state.select.fallback_value(n_s) (in (-1, -0.5], below every row)."""
    meta: dict
    head: dict
    routes: dict
    nodes: dict
    fallback: dict

    def __post_init__(self):
        missing = [k for k in REQUIRED_META if k not in self.meta]
        if missing:
            raise ValueError(f"score meta lacks {missing}")
        if self.meta["role"] not in ROLES or self.meta["kplan"] not in KPLANS:
            raise ValueError(f"role {self.meta['role']!r} / kplan {self.meta['kplan']!r}")
        n = {len(v) for v in self.nodes.values()}
        if len(n) > 1:
            raise ValueError(f"node columns of unequal length {sorted(n)}")
        for c in ("id", "sig", "s"):
            if c not in self.nodes:
                raise ValueError(f"node column {c!r} missing")

    @classmethod
    def from_rows(cls, meta, head, routes, rows, fallback, habit=None):
        """Columnar ScoreDay from CompositeRows (+ optional {node: NodeHabit} for E / z columns)."""
        cols = {c: [] for c in ROW_FIELDS}
        zc = {}
        for r in rows:
            h = habit.get(r.node) if habit else None
            vals = {"id": int(r.node), "sig": r.sig, "s": float(r.s), "base": float(r.base), "a": float(r.a),
                    "tier": int(r.tier), "qv": float(r.qv), "qt": float(r.qt), "s_hat": r.s_hat,
                    "E": None if h is None else float(h.E), "E_mono": None if h is None else float(h.E_mono)}
            for c in ROW_FIELDS:
                cols[c].append(vals[c])
            if habit:
                for f in FAMILIES:
                    fs = h.fams.get(f) if h is not None else None
                    zc.setdefault(f"z.{f.value}", []).append(None if fs is None else fs.z)
                    zc.setdefault(f"x.{f.value}", []).append(None if fs is None else fs.x)
        cols.update(zc)
        return cls(dict(meta), dict(head), dict(routes), cols, dict(fallback))

    def to_json(self):
        return {"meta": self.meta, "head": self.head, "routes": self.routes, "nodes": self.nodes,
                "fallback": self.fallback}

    @classmethod
    def from_json(cls, d):
        return cls(d["meta"], d["head"], d["routes"], d["nodes"], d["fallback"])

    def node_values(self, host, column="s"):
        """(node ids on the day, value per node): the row value for head nodes, the fallback for
        every other node (decide.scorefile.day_arrays order). Raises if a row is not a node of the day."""
        from habitus.decide.scorefile import day_arrays
        ids, inv, _usigs, fb = day_arrays(host, self.meta["day"])
        v = np.asarray(fb, dtype=np.float64)[inv].copy()
        rid = np.asarray(self.nodes["id"], dtype=np.int64)
        if not len(rid):
            return ids, v
        rv = np.asarray([np.nan if x is None else x for x in self.nodes[column]], dtype=np.float64)
        o = np.argsort(ids, kind="stable")
        pos = np.searchsorted(ids[o], rid)
        ok = pos < len(ids)
        ok[ok] = ids[o][pos[ok]] == rid[ok]
        if not ok.all():
            raise ValueError(f"{(~ok).sum()} score rows are not nodes of {self.meta['day']}")
        v[o[pos]] = rv
        return ids, v


@dataclass(frozen=True)
class AlertDay:
    """bars.alerts: one test day at budget B. B = 0: strong = {s >= tau_0}, weak = ().
    B = 20: strong = {s >= tau_1}, weak = the <= WEAK_CAP nodes of [tau_20, tau_1) with the largest
    margin s - tau_20 (ties by node id). Node ids ascending inside each tier."""
    day: str
    budget: int
    strong: tuple
    weak: tuple
    bars: Mapping

    @property
    def alerts(self):
        return tuple(sorted(set(self.strong) | set(self.weak)))


# --------------------------------------------------------------------------- #
# Training items
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Item:
    """One exported training item: a fully rendered prompt (text.render, byte-identical to the jev
    decision model's), the letters it offers and a soft target ALIGNED WITH those letters (the
    variant's letter order). Inclusion target = share of the signature's member nodes with the
    bucket present (yes first in fwd); choice target = mean one-hot over the applicable members."""
    ds: str
    day: str
    ref_days: tuple
    k_mode: str
    sig: str
    family: Family
    bucket: str                     # bucket key, NO_BUCKET for a choice family
    view_sha1: str
    qkey: str
    variant: str                    # the order the prompt was rendered in ("fwd" | "rev")
    prompt_text: str
    letters: tuple
    target: tuple
    weight: float = 1.0
    n_members: int = 1
    role: str = "train"
    fold: str = ""
    src: str = "code"

    def __post_init__(self):
        object.__setattr__(self, "family", Family.of(self.family))
        object.__setattr__(self, "ref_days", tuple(self.ref_days))
        object.__setattr__(self, "letters", tuple(self.letters))
        object.__setattr__(self, "target", tuple(float(x) for x in self.target))
        Ref(self.ds, self.day, self.ref_days, self.k_mode)          # validates the reference
        if self.variant not in ORDERS or self.role not in ITEM_ROLES or self.src not in ITEM_SOURCES:
            raise ValueError(f"variant {self.variant!r} / role {self.role!r} / src {self.src!r}")
        if not 2 <= len(self.letters) <= MAX_OPTIONS or self.letters != LETTERS[:len(self.letters)]:
            raise ValueError(f"letters {self.letters}")
        if len(self.target) != len(self.letters) or abs(sum(self.target) - 1.0) > 1e-6 \
                or min(self.target) < 0:
            raise ValueError(f"target {self.target} for letters {self.letters}")
        if not (self.weight > 0 and math.isfinite(self.weight)):
            raise ValueError(f"weight {self.weight}")
        if self.qkey != qkey(self.view_sha1, self.family, None if self.bucket == NO_BUCKET else self.bucket):
            raise ValueError(f"qkey {self.qkey} does not match view / family / bucket")

    def to_json(self):
        return {"ds": self.ds, "day": self.day, "ref_days": list(self.ref_days), "k_mode": self.k_mode,
                "sig": self.sig, "family": self.family.value, "bucket": self.bucket, "view_sha1": self.view_sha1,
                "qkey": self.qkey, "variant": self.variant, "prompt_text": self.prompt_text,
                "letters": list(self.letters), "target": list(self.target), "weight": self.weight,
                "n_members": self.n_members, "role": self.role, "fold": self.fold, "src": self.src}

    @classmethod
    def from_json(cls, d):
        return cls(**{**d, "ref_days": tuple(d["ref_days"]), "letters": tuple(d["letters"]),
                      "target": tuple(d["target"])})

    def dumps(self):
        return json.dumps(self.to_json(), separators=(",", ":"), ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Reference-side habit index (profile2.Profile2 implements this protocol)
# --------------------------------------------------------------------------- #
@runtime_checkable
class Profile2Like(Protocol):
    """Exact-identity habit indices over the reference days R only. Element keys of habit():
    P1 parent ident keys; P2 / P5 "dir:<class>"; P3 "ep:<class>"; P4 child ident keys; F1 writer /
    F2 executor ident keys (for a FILE ident: of that exact path); N1 owner ident keys (of that
    exact connection). class_habit() keys: P2 / P3 / P5 (role class, image_dir) -> the same element
    keys; P4 (role class, image_dir) -> "cls:<child image_dir>"; F1 / F2 (dir, ext class) -> writer /
    executor ident keys; N1 (endpoint, port class, str(prefix level)) -> owner ident keys. Values =
    number of reference days the element occurred (render as f"{n}/{k}")."""
    ds: str
    ref: tuple
    k: int
    os: str
    obs: Optional[Mapping]

    def check(self, day: str) -> None: ...                              # ValueError if day in ref

    def seen_days(self, ident: str) -> int: ...                         # by ident_key

    def kappa_seen_days(self, kappa: str) -> int: ...                   # by full kappa

    def tier0(self, card: Mapping) -> Attest: ...                       # NONE | PATTERN | IDENTITY

    def frequency(self, ident: str) -> str: ...                         # high | medium | low | none

    def ident_inv(self, ident: str) -> Optional[Mapping]: ...           # invariant of an IDENTITY ident

    def display(self, ident: str) -> Optional[str]: ...                 # short name of an IDENTITY ident

    def habit(self, ident: str, family: Family) -> Mapping: ...

    def class_habit(self, family: Family, cls: tuple) -> Mapping: ...

    def class_days(self, family: Family, cls: tuple) -> int: ...

    def lineage_seen(self, kind: str, cause_ident: str, effect_kappa: str) -> bool: ...

    def chains(self, owner: Optional[str] = None) -> frozenset: ...


# --------------------------------------------------------------------------- #
# Score directories
# --------------------------------------------------------------------------- #
def split_level(level, arm=None):
    """(level, arm) of a level name that may be a clone arm (constants.CLONE_ARMS: "HB.head" -> ("HB",
    "head"), "HB-L" -> ("HB-L", None)); an extra `arm` joins the arm suffix with "-". Plain levels
    pass through."""
    if level in CLONE_ARMS:
        base, sub = CLONE_ARMS[level]
    else:
        base, sub = level, None
    if base not in LEVELS:
        raise ValueError(f"level {level!r}")
    if arm is not None and not re.fullmatch(r"[A-Za-z0-9-]+", str(arm)):
        raise ValueError(f"arm {arm!r}")
    joined = "-".join(str(x) for x in (sub, arm) if x)
    return base, joined or None


def level_dir(level, kplan, arm=None):
    """Score directory name "<level>[.<arm>]_<kplan>" (e.g. HB0_k2, HB-L.mq_allc-pre, HB.head_k2: a
    clone arm's suffix comes first, split_level)."""
    if level not in LEVELS or kplan not in KPLANS:
        raise ValueError(f"level {level!r} / kplan {kplan!r}")
    base, arm = split_level(level, arm)
    return f"{base}{'.' + arm if arm else ''}_{kplan}"


def base_meta(ref, level, kplan, role, caps, head_rule="Q0", arm=None, **extra):
    """The REQUIRED_META block of a score file for one Ref."""
    return {"ds": ref.ds, "day": ref.day, "level": level, "kplan": kplan, "role": role,
            "ref": list(ref.ref_days), "k_mode": ref.k_mode, "caps": dict(caps), "head_rule": head_rule,
            "arm": arm, "spec": SPEC_VERSION, **extra}
