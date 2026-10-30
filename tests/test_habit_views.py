"""Habit-channel views, targets and IFC (CPU), on the shared fake host (tests/habit_fakes.py) with an
exact-identity reference double (FakeProfile2 below, the types.Profile2Like protocol):
  (e)  answer independence: every view of every node is byte-identical whether or not any single
       attack event (hence any present bucket) happened; views never move with behaviour, links or
       parent_name;
  (f)  candidates are a function of (view, reference) only: they never touch the host, a link or an
       op key, are identical across days and twins, and every present bucket is a candidate;
  (g') the per-node E-monotonicity prerequisite: adding links or op keys can only add present
       buckets (monotone families), owners only grow, and n/a never changes for them;
  IFC: every emitted field is labelled, raw strings only for IDENTITY-attested identities, no
       removed or answer field anywhere, the build-time assertion fires on hand-made bad views;
  plus the P1 pname rule, the CADETS process -> file exclusion, the availability gate and the frozen
  table, k = 0 class buckets, N2 options, determinism, the label-hygiene grep, and a real-host check of
  the measured link availability (skips without card files).
"""

import copy
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
import pytest

from conftest import REPO, missing_asset
from habit_fakes import ATTACK, BENIGN, CLEAN, ID, TEST, FakeHabitHost, build_day, make_habit_host

from habitus.questions.habit import ifc, targets
from habitus.questions.habit import types as T
from habitus.questions.habit import views as V
from habitus.questions.habit.constants import MAX_BUCKETS, MAX_LISTED, MAX_P1_NAMES
from habitus.questions.habit.types import Attest, Family, ViewKind
from habitus.state.profile import chain_of

HB = os.path.join(REPO, "src", "habitus", "questions", "habit")
HYGIENE = re.compile(r"gt\(|ground_truth|host_labels|Ground_Truth|gt_")     # the label-token pattern
MINE = ("targets.py", "views.py", "ifc.py")
D = TEST[1]                                                                 # the attack day
CLOSED_WORDS = frozenset(T.DIR_ORDER + T.EP_ORDER + T.PORT_ORDER + ("none", "unknown"))


# --------------------------------------------------------------------------- #
# An exact-identity reference double (types.Profile2Like), independent of targets.py
# --------------------------------------------------------------------------- #
class FakeProfile2:
    """Profile2Like over a host's reference days: habits keyed by EXACT ident (kappa' for processes),
    values = reference days the element occurred. Class keys use views' own words (role, image_dir;
    dir, ext class; endpoint, port class, str(prefix level)) so they agree with what views put in
    View.meta. Elements: P1 parent idents (code-1 causes whose file_name = parent_name), P2 / P5
    "dir:<w>", P3 "ep:<w>", P4 child idents, F1 / F2 writer / executor idents, N1 owner idents."""

    def __init__(self, host, ref, obs=None):
        self.ds, self.ref, self.k, self.os = host.ds, tuple(sorted(ref)), len(ref), host.os
        self.obs = dict(obs or getattr(host, "obs", None) or {"exec": True, "lineage": True})
        self._seen, self._kseen = Counter(), Counter()
        self._card = {}
        self._habit, self._cls, self._cdays = defaultdict(Counter), defaultdict(Counter), Counter()
        self._chains, self._lineage = defaultdict(set), set()
        per_day = []
        for d in self.ref:
            nodes, L = host.nodes(d), host.links(d)
            idn = {n: T.ident_key(host.card(s)) for n, s in nodes.items()}
            card = {n: host.card(s) for n, s in nodes.items()}
            for n, c in card.items():
                self._card.setdefault(idn[n], c)
            self._seen.update(set(idn.values()))
            self._kseen.update({T.full_kappa(c) for c in card.values()})
            per_day.append((nodes, L, idn, card))
        for nodes, L, idn, card in per_day:
            hab, cls, cdays = set(), set(), set()
            for n, c in card.items():
                e = idn[n]
                t = c["type"]
                if t == "process":
                    bc = self._beh_cls(c)
                    cdays.add((Family.P2_modify, bc))
                    for op in c["behaviour"].get("ops", {}):
                        v, _, oc = op.partition(":")
                        if v in targets.MODIFY_VERBS and oc != "process" and not oc.startswith("net-"):
                            hab.add((e, Family.P2_modify, f"dir:{V.dir_word(oc)}"))
                            cls.add((Family.P2_modify, bc, f"dir:{V.dir_word(oc)}"))
                        if oc.startswith("net-"):
                            hab.add((e, Family.P3_peers, f"ep:{V.ep_word(oc[4:])}"))
                            cls.add((Family.P3_peers, bc, f"ep:{V.ep_word(oc[4:])}"))
                elif t == "file":
                    cdays.add((Family.F1_writer, self._file_cls(c)))
                elif t == "netflow":
                    for x in chain_of(c) or ():
                        self._chains[None].add(x)
            for a, b, code in L.tolist():
                ca, cb = card.get(a), card.get(b)
                if ca is None or cb is None:
                    continue
                ea, eb = idn[a], idn[b]
                if code == 1:
                    hab.add((ea, Family.P4_children, eb))
                    cls.add((Family.P4_children, self._beh_cls(ca), f"cls:{V.dir_word(cb['invariant'].get('image_dir'))}"))
                    if ca["invariant"].get("file_name") == cb["invariant"].get("parent_name"):
                        hab.add((eb, Family.P1_parent, ea))
                    self._lineage.add(("spawn", ea, T.full_kappa(cb)))
                elif code == 2 and cb["type"] == "file":
                    hab.add((eb, Family.F1_writer, ea))
                    cls.add((Family.F1_writer, self._file_cls(cb), ea))
                    self._lineage.add(("write", ea, T.full_kappa(cb)))
                elif code == 3:
                    hab.add((eb, Family.P5_exec_from, f"dir:{V.dir_word(ca['invariant'].get('dir'))}"))
                    cls.add((Family.P5_exec_from, self._beh_cls(cb), f"dir:{V.dir_word(ca['invariant'].get('dir'))}"))
                    hab.add((ea, Family.F2_executor, eb))
                    cls.add((Family.F2_executor, self._file_cls(ca), eb))
                elif code == 4 and cb["type"] == "netflow":
                    ep = f"ep:{V.ep_word(cb['invariant'].get('endpoint'))}"
                    hab.add((ea, Family.P3_peers, ep))
                    cls.add((Family.P3_peers, self._beh_cls(ca), ep))
                    hab.add((eb, Family.N1_owner, ea))
                    cls.add((Family.N1_owner, self._net_cls(cb), ea))
                    self._chains[ea].update(chain_of(cb) or ())
                    self._lineage.add(("net", ea, T.full_kappa(cb)))
            for e, f, x in hab:
                self._habit[(e, f)][x] += 1
            for f, c, x in cls:
                self._cls[(f, c)][x] += 1
            for fc in cdays:
                self._cdays[fc] += 1

    # class keys (what views put in View.meta)
    def _words(self, card):
        c0 = V.invariant_card(card)
        return V.entity_words(c0, Attest.NONE), c0

    def _beh_cls(self, card):
        w, c0 = self._words(card)
        return V.role_of(c0, self.tier0(c0)), w["image_dir"]

    def _file_cls(self, card):
        w, _ = self._words(card)
        return w["dir"], w["ext_class"]

    def _net_cls(self, card):
        w, _ = self._words(card)
        ch = chain_of(card)
        return w["endpoint"], w["port_class"], V.NO_LEVEL if ch is None else str(len(ch))   # a reference flow: all seen

    # protocol
    def check(self, day):
        if day in self.ref:
            raise ValueError(f"{day} is a reference day")

    def seen_days(self, ident):
        return self._seen.get(ident, 0)

    def kappa_seen_days(self, kappa):
        return self._kseen.get(kappa, 0)

    def tier0(self, card):
        if self.seen_days(T.ident_key(card)) <= 0:
            return Attest.NONE
        if T.has_placeholder(card) or T.uninformative(card):
            return Attest.PATTERN
        return Attest.IDENTITY

    def frequency(self, ident):
        d = self.seen_days(ident)
        return "none" if not d else "high" if d == self.k else "medium"

    def ident_inv(self, ident):
        c = self._card.get(ident)
        if c is None or self.tier0(c) is not Attest.IDENTITY:
            return None
        return {k: v for k, v in c["invariant"].items() if k not in T.PARENT_FIELDS}

    def display(self, ident):
        inv = self.ident_inv(ident)
        if inv is None:
            return None
        t = self._card[ident]["type"]
        return inv.get("file_name") if t == "process" else os.path.basename(inv.get("path", "")) if t == "file" else None

    def habit(self, ident, family):
        return dict(self._habit.get((ident, Family.of(family)), {}))

    def class_habit(self, family, cls):
        return dict(self._cls.get((Family.of(family), tuple(cls)), {}))

    def class_days(self, family, cls):
        return self._cdays.get((Family.of(family), tuple(cls)), 0)

    def lineage_seen(self, kind, cause_ident, effect_kappa):
        return (kind, cause_ident, effect_kappa) in self._lineage

    def chains(self, owner=None):
        return frozenset(self._chains.get(owner, ()))


class Recorder:
    """Wraps a profile and records every method called (check f)."""

    def __init__(self, prof):
        self._p, self.calls = prof, Counter()

    def __getattr__(self, name):
        a = getattr(self._p, name)
        if callable(a):
            def wrapped(*args, **kw):
                self.calls[name] += 1
                return a(*args, **kw)
            return wrapped
        return a


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def host():
    return make_habit_host()


@pytest.fixture(scope="module")
def prof(host):
    return FakeProfile2(host, CLEAN[:2])


def host_without(event):
    """The fake host whose attack day lacks one ATTACK event (everything else identical)."""
    days, links = {}, {}
    for d in CLEAN + TEST:
        ev = list(BENIGN) + ([e for e in ATTACK if e != event] if d == D else [])
        days[d], links[d] = build_day(d, ev)
    return FakeHabitHost(days, links)


def host_with_links(base, day, extra):
    """A copy of `base` whose `day` carries extra [cause, effect, code] links (cards unchanged)."""
    days = {d: copy.deepcopy(base._raw[d]) for d in base._raw}
    links = {d: [list(r) for r in base._rawlinks[d]] for d in base._rawlinks}
    links[day] = sorted({tuple(r) for r in links[day]} | {tuple(r) for r in extra if r[0] != r[1]})
    links[day] = [list(r) for r in links[day]]
    return FakeHabitHost(days, links)


def all_views(host, day, prof, k_mode="k2"):
    """{node: {ViewKind: View}} for every node of the day (every family of the THEIA-like fake)."""
    out = {}
    for n, s in sorted(host.nodes(day).items()):
        card = host.card(s)
        out[n] = V.views_for(card, prof, k_mode, T.FAMILIES)
    return out


def node_card(host, day, name):
    return host.card(host.nodes(day)[ID[name]])


def observe_all(host, day, prof, k_mode="k2"):
    return targets.observe(host, day, prof, sorted(set(host.nodes(day).values())), k_mode)


# --------------------------------------------------------------------------- #
# 1. hygiene, IFC schema
# --------------------------------------------------------------------------- #
def test_ifc_schema_and_answer_fields_disjoint():
    assert set(ifc.ANSWER_FIELDS) == set(T.FAMILIES) and set(ifc.READ_SETS) == set(T.FAMILIES)
    for (k, s, f), lab in ifc.FIELD_LABELS.items():
        assert s in T.VIEW_SECTIONS[k] and lab in T.HABIT_LABELS, (k, s, f)
    for f in T.FAMILIES:
        assert not ifc.read_set(f) & set(ifc.ANSWER_FIELDS[f]), f
        kind = T.FAMILY_SPEC[f].view
        assert set(ifc.ANSWER_FIELDS[f]) <= set(V.REMOVAL[kind]), f
    for kind, rm in V.REMOVAL.items():
        assert {"behaviour", "deviation", "evidence", "precedent"} <= set(rm)
        for r in rm:                                          # nothing removed is ever labelled
            s, _, f = r.partition(".")
            assert (kind, s, f) not in ifc.FIELD_LABELS, (kind, r)
    assert "reference.identity_seen_days" in V.REMOVAL[ViewKind.V_par]       # full kappa embeds parent_name
    assert "entity.remote_ip" in V.REMOVAL[ViewKind.V_net] and "reference.prefix_level" in V.REMOVAL[ViewKind.V_own]


def _hand_view(kind, sections, tier="NONE"):
    meta = {k: "" for k in T.VIEW_META_KEYS}
    meta.update(ident="x", tier0=tier, k_mode="k2")
    return T.View(kind, V.VIEW_NODE_TYPE[kind], sections, V.REMOVAL[kind], meta)


def test_ifc_assertion_fires_on_bad_views():
    ok = {"entity": {"type": "process", "os": "linux", "image_dir": "system", "file_name": "bash"},
          "hard_bits": {"image_location": "system"}, "reference": {"days": 2}}
    good = _hand_view(ViewKind.V_par, ok, "IDENTITY")
    assert ifc.violations(good) == [] and ifc.assert_view(good) is good
    assert ifc.label_view(good)["entity.file_name"] is T.Label.K0
    bad = _hand_view(ViewKind.V_par, ok, "PATTERN")                       # raw name, not attested
    assert ifc.label_view(bad)["entity.file_name"] is T.Label.A
    assert any("raw string entity.file_name" in v for v in ifc.violations(bad))
    with pytest.raises(ifc.IFCError):
        ifc.assert_view(bad)
    leak = _hand_view(ViewKind.V_par, {**ok, "reference": {"days": 2, "identity_seen_days": "2/2"}}, "IDENTITY")
    vs = ifc.violations(leak)                                             # P1's own answer (full kappa)
    assert any("answer field reference.identity_seen_days of P1" in v for v in vs)
    assert any("removed field reference.identity_seen_days" in v for v in vs)
    with pytest.raises(KeyError):
        ifc.label_view(leak)
    peers = _hand_view(ViewKind.V_beh, {**ok, "hard_bits": {"image_location": "system", "external_peers": "yes"}},
                       "IDENTITY")
    assert any("answer field hard_bits.external_peers of P3" in v for v in ifc.violations(peers))
    own = _hand_view(ViewKind.V_own, {"entity": {"type": "process", "os": "linux"},
                                      "connection": {"endpoint": "external", "port_class": "well-known",
                                                     "prefix_level": "3/5"}, "reference": {"days": 2}})
    assert any("connection.prefix_level of N2" in v for v in ifc.violations(own))


# --------------------------------------------------------------------------- #
# 2. views: purity, attested-name rule, removal
# --------------------------------------------------------------------------- #
def test_every_view_is_ifc_clean_and_compact(host, prof):
    seen_kinds = Counter()
    for d in (CLEAN[2], D):
        for n, vs in all_views(host, d, prof).items():
            for kind, v in vs.items():
                assert ifc.violations(v) == [], (n, kind)
                lab = ifc.label_view(v)
                assert set(lab.values()) <= T.HABIT_LABELS
                assert v.text == json.dumps(dict(v.sections), separators=(",", ":"), ensure_ascii=False)
                assert list(v.sections) == [s for s in T.VIEW_SECTIONS[kind] if s in v.sections]
                assert set(v.meta) == set(T.VIEW_META_KEYS) and v.removed_fields == V.REMOVAL[kind]
                for bad in ("behaviour", "parent_name", "ops", "external_peers", "n_writers", "processes",
                            "has_parent", "behaviour_seen_before", "identity_seen_days", "name_usual"):
                    assert f'"{bad}"' not in v.text, (n, kind, bad)
                seen_kinds[kind] += 1
    assert set(seen_kinds) == {ViewKind.V_beh, ViewKind.V_par, ViewKind.V_file, ViewKind.V_net}


def _mutated(card, seed):
    """Same type / os / invariant, arbitrary behaviour (and a different parent for processes)."""
    rng = np.random.default_rng(seed)
    c = copy.deepcopy(card)
    c["behaviour"] = {"ops": {f"write:{w}": "128+" for w in rng.choice(T.DIR_ORDER, 3)} | {"connect:net-external": "1"},
                      "children": ["mimikatz"], "flags": ["executed_file_written_today", "deleted"],
                      "writers": ["x"], "executors": ["y"], "processes": ["z"], "peers": {"external": "1"}}
    if c["type"] == "process":
        c["invariant"]["parent_name"] = "evil_parent_" + str(seed)
    return c


def test_views_are_pure_functions_of_type_invariant_and_reference(host, prof):
    for n, s in sorted(host.nodes(D).items()):
        card = host.card(s)
        base = V.views_for(card, prof, "k2", T.FAMILIES)
        for seed in range(3):
            mut = V.views_for(_mutated(card, seed), prof, "k2", T.FAMILIES)
            assert {k: v.text for k, v in mut.items()} == {k: v.text for k, v in base.items()}, n
        no_parent = copy.deepcopy(card)
        no_parent["invariant"].pop("parent_name", None)
        assert {k: v.text for k, v in V.views_for(no_parent, prof, "k2", T.FAMILIES).items()} == \
            {k: v.text for k, v in base.items()}
    # the same invariant renders the same bytes on a clean day and on the attack day
    py_clean, py_att = node_card(host, CLEAN[2], "python3"), node_card(host, D, "python3")
    assert host.nodes(CLEAN[2])[ID["python3"]] != host.nodes(D)[ID["python3"]]       # behaviour differs
    a, b = V.views_for(py_clean, prof, "k2", T.FAMILIES), V.views_for(py_att, prof, "k2", T.FAMILIES)
    assert {k: v.text for k, v in a.items()} == {k: v.text for k, v in b.items()}


@pytest.mark.parametrize("event", ATTACK)
def test_x0e_views_identical_whether_or_not_any_attack_event_happened(host, prof, event):
    """Answer independence (check e): removing one attack event removes (or changes) present buckets of
    the nodes it touches, while every view of every node stays byte-identical."""
    other = host_without(event)
    va, vb = all_views(host, D, prof), all_views(other, D, prof)
    common = set(va) & set(vb)
    assert common
    for n in common:
        assert {k: v.text for k, v in va[n].items()} == {k: v.text for k, v in vb[n].items()}, (event, n)
    oa, ob = observe_all(host, D, prof), observe_all(other, D, prof)
    changed = [n for n in set(oa) & set(ob)
               if (oa[n].present, oa[n].choice, oa[n].owners) != (ob[n].present, ob[n].choice, ob[n].owners)]
    assert changed, event                                               # the answer did change


def test_attested_name_rule_on_views(host, prof):
    tiers = Counter()
    for n, s in sorted(host.nodes(D).items()):
        card = host.card(s)
        tier = prof.tier0(V.invariant_card(card))
        tiers[tier] += 1
        raws = [str(card["invariant"].get(f) or "") for f in T.IDENTITY_FIELDS[card["type"]]]
        for kind, v in V.views_for(card, prof, "k2", T.FAMILIES).items():
            assert v.meta["tier0"] == tier.name
            ent = v.sections["entity"]
            if card["type"] == "netflow":                                # an address is never rendered
                assert card["invariant"]["remote_ip"] not in v.text
                continue
            if tier.identity_attested:
                continue
            assert not set(ent) & ifc.RAW_FIELDS, (n, kind)
            for r in raws:
                for part in re.split(r"[\s/]+", r):
                    if len(part) >= 4 and part not in CLOSED_WORDS:
                        assert part not in v.text, (n, kind, part)
    assert tiers[Attest.NONE] and tiers[Attest.PATTERN] and tiers[Attest.IDENTITY]
    py = V.build(node_card(host, D, "python3"), ViewKind.V_beh, prof, "k2")
    assert py.sections["entity"]["file_name"] == "python3" and py.meta["tier0"] == "IDENTITY"
    fake = V.build(node_card(host, D, "fake_python3"), ViewKind.V_beh, prof, "k2")
    assert fake.meta["tier0"] == "NONE" and "python3" not in fake.text            # a masquerade borrows nothing
    pay = V.build(node_card(host, D, "payload"), ViewKind.V_par, prof, "k2")
    assert "<hex" not in pay.text and "file_name" not in pay.sections["entity"]


def test_build_time_ifc_assertion_inside_build(host, prof, monkeypatch):
    """views.build refuses (ifc.IFCError) a view whose entity would carry an unattested raw string."""
    leaky = V.entity_words

    def inject(c0, tier):
        w = leaky(c0, tier)
        if c0["type"] == "process":
            w["file_name"] = c0["invariant"].get("file_name") or "x"
        return w
    monkeypatch.setattr(V, "entity_words", inject)
    V.build(node_card(host, D, "python3"), ViewKind.V_beh, prof, "k2")            # IDENTITY: allowed
    with pytest.raises(ifc.IFCError):
        V.build(node_card(host, D, "fake_python3"), ViewKind.V_beh, prof, "k2")


def test_view_reference_content_and_k0(host, prof):
    py = V.build(node_card(host, D, "python3"), ViewKind.V_beh, prof, "k2")
    ref = py.sections["reference"]
    assert ref["days"] == 2 and ref["seen_days"] == "2/2" and ref["usual_peers"] == {"private": "2/2"}
    assert ref["usual_modify_dirs"] == {"home": "2/2"} and ref["usual_exec_dirs"] == {"system": "2/2"}
    assert ref["usual_children"] == [] and py.sections["class_habits"]["image_dir"] == "system"
    bash = V.build(node_card(host, D, "bash"), ViewKind.V_beh, prof, "k2")
    assert sorted(x.split()[0] for x in bash.sections["reference"]["usual_children"]) == ["firefox", "ls", "python3"]
    par = V.build(node_card(host, D, "python3"), ViewKind.V_par, prof, "k2")
    assert par.sections["reference"]["usual_parents"] == ["bash 2/2"]
    out = V.build(node_card(host, D, "out_txt"), ViewKind.V_file, prof, "k2")
    assert out.sections["reference"]["class_usual_writers"] == ["python3 2/2"]
    assert out.sections["hard_bits"]["executable_type"] == "no" and out.meta["dir"] == "home"
    web = V.build(node_card(host, D, "web"), ViewKind.V_net, prof, "k2")
    assert web.sections["reference"]["prefix_level"] == "5/5" and web.meta["prefix_level"] == 5
    assert web.sections["reference"]["class_usual_owners"] == ["firefox 2/2"]
    assert "93.184" not in web.text and "443" not in web.text
    for name, kind in (("python3", ViewKind.V_beh), ("out_txt", ViewKind.V_file), ("web", ViewKind.V_net)):
        v0 = V.build(node_card(host, D, name), kind, None, "k0")
        assert v0.sections["reference"] == {"days": 0} and "class_habits" not in v0.sections
        assert "observable" not in v0.sections["hard_bits"] and v0.meta["tier0"] == "NONE"
        assert not set(v0.sections["entity"]) & ifc.RAW_FIELDS
    with pytest.raises(ValueError):
        V.build(node_card(host, D, "web"), ViewKind.V_own, None, "k0")
    with pytest.raises(ValueError):
        V.build(node_card(host, D, "web"), ViewKind.V_beh, prof, "k2")
    with pytest.raises(ValueError):
        V.build(node_card(host, D, "python3"), ViewKind.V_beh, prof, "k0")
    assert set(V.views_for(node_card(host, D, "python3"), None, "k0", T.FAMILIES)) == {ViewKind.V_beh}


def test_own_views(host, prof):
    web = node_card(host, D, "web")
    ff = T.ident_key(node_card(host, D, "firefox"))
    own = V.own_views(web, [f"id:{ff}", T.GEN_TYPED, f"id:{ff}"], prof, "k2")
    assert list(own) == sorted({f"id:{ff}", T.GEN_TYPED})
    idv, typed = own[f"id:{ff}"], own[T.GEN_TYPED]
    assert idv.sections["entity"]["file_name"] == "firefox" and idv.meta["owner"] == ff and idv.meta["tier0"] == "IDENTITY"
    assert typed.sections["entity"] == {"type": "process", "os": "linux"} and typed.meta["tier0"] == "NONE"
    assert typed.sections["reference"]["owner"] == V.TYPED_OWNER
    for v in (idv, typed):
        assert ifc.violations(v) == []
        assert list(v.sections) == ["entity", "connection", "reference"]
        assert "prefix_level" not in v.text and "93.184" not in v.text and "443" not in v.text
        for el in chain_of(web):
            assert el not in v.text or len(el) <= 3, el                 # no chain element of the flow
    assert idv.sections["reference"]["owner_addresses"] != "0"
    # the gen:typed view is shared: another flow with the same classes renders the same bytes
    other = V.own_views(node_card(host, D, "lotl_peer"), [T.GEN_TYPED], prof, "k2")[T.GEN_TYPED]
    assert other == typed
    with pytest.raises(ValueError):                                      # an unattested owner has no id: view
        V.own_views(web, [f"id:{T.ident_key(node_card(host, D, 'payload'))}"], prof, "k2")
    with pytest.raises(ValueError):
        V.own_views(web, ["dir:home"], prof, "k2")


# --------------------------------------------------------------------------- #
# 3. candidates (check f) and buckets
# --------------------------------------------------------------------------- #
def _cands_of(card, prof, k_mode="k2"):
    out = {}
    for kind, v in V.views_for(card, prof, k_mode, T.FAMILIES).items():
        for f, c in targets.candidates_for(v, prof, k_mode, T.FAMILIES).items():
            out[f] = c
    return out


def test_candidates_read_only_view_meta_and_reference(host, prof):
    rec = Recorder(prof)
    for n, s in sorted(host.nodes(D).items()):
        card = host.card(s)
        vs = V.views_for(card, prof, "k2", T.FAMILIES)
        for kind, v in vs.items():
            for f in T.VIEW_FAMILIES[kind]:
                a = targets.candidates(f, v, rec, "k2")
                b = targets.candidates(f, v, prof, "k2")
                assert a == b
                # the same view from a behaviour-mutated card gives the same candidates
                mv = V.build(_mutated(card, 1), kind, prof, "k2")
                assert targets.candidates(f, mv, prof, "k2") == b
                assert len(b.buckets) <= MAX_BUCKETS and len(b.listed) <= MAX_P1_NAMES
    assert set(rec.calls) <= {"habit", "class_habit", "ident_inv", "display"}
    assert rec.calls["habit"] and rec.calls["class_habit"]


def test_candidate_buckets_order_and_phrases(host, prof):
    bash = _cands_of(node_card(host, D, "bash"), prof)
    p4 = bash[Family.P4_children]
    keys = p4.keys()
    assert keys[-2:] == (T.GEN_ATTESTED_OTHER, T.GEN_NEVER_SEEN) and len(keys) == 5
    listed = [b for b in p4.buckets if b.key.startswith("id:")]
    assert [b.key[3:] for b in listed] == sorted(b.key[3:] for b in listed)        # equal days: key order
    assert {b.text for b in listed} == {"the program firefox", "the program ls", "the program python3"}
    assert bash[Family.P2_modify].keys() == tuple(f"dir:{w}" for w in T.DIR_ORDER)
    assert bash[Family.P3_peers].keys() == tuple(f"ep:{w}" for w in T.EP_ORDER)
    py = _cands_of(node_card(host, D, "python3"), prof)
    assert [b.text for b in py[Family.P1_parent].listed] == ["bash"] and py[Family.P1_parent].buckets == ()
    for k in [f"dir:{w}" for w in T.DIR_ORDER] + [f"ep:{w}" for w in T.EP_ORDER] + \
             [f"cls:{w}" for w in T.DIR_ORDER] + list(T.GENERIC_BUCKETS):
        assert targets.BUCKET_PHRASES[k] and k in targets.BUCKET_PHRASES
    for k, phrase in targets.BUCKET_PHRASES.items():
        assert "{" not in phrase and len(phrase) < 120, k
    # MAX_LISTED: a class habit with 9 attested writers lists 6, by (-days, key)
    hab = {f"w{i}": (i % 3) + 1 for i in range(9)}

    class P(FakeProfile2):
        def __init__(self):
            self.k, self.obs = 3, {"exec": True, "lineage": True}

        def class_habit(self, family, cls):
            return dict(hab)

        def ident_inv(self, e):
            return {"file_name": "same"}

        def display(self, e):
            return "same"
    v = V.build(node_card(host, D, "out_txt"), ViewKind.V_file, prof, "k2")
    c = targets.candidates(Family.F1_writer, v, P(), "k2")
    ids = [b.key[3:] for b in c.buckets if b.key.startswith("id:")]
    assert len(ids) == MAX_LISTED == 6 and ids == sorted(hab, key=lambda e: (-hab[e], e))[:6]
    assert len({b.text for b in c.buckets}) == len(c.buckets)            # equal names are told apart


def test_display_names_disambiguate(host, prof):
    a, b = "a" * 16, "b" * 16

    class P:
        def display(self, e):
            return "python3"

        def ident_inv(self, e):
            return {"file_name": "python3", "executable": "/usr/bin/python3" if e == a else "/opt/py/python3"}
    n = V.display_names([b, a], P())
    assert n == {a: "python3 (/usr/bin/python3)", b: "python3 (/opt/py/python3)"}

    class Q(P):
        def ident_inv(self, e):
            return {"file_name": "python3"}
    assert V.display_names([b, a], Q()) == {a: "python3 #1", b: "python3 #2"}

    # profile-wide: a list holding ONE of two equally named idents still names it apart
    class R(Q):
        def name_group(self, e):
            return (a, b)
    assert V.display_names([b], R()) == {b: "python3 #2"} and V.display_names([a], R()) == {a: "python3 #1"}


def test_equal_views_have_equal_candidates_when_names_collide_across_lists(monkeypatch):
    """Review finding (optc_h501 09-19, export crash): two PATTERN children render the same V_par
    while their single listed parents are two DIFFERENT attested idents with the same short name
    (python3 with and without a command line). Names are told apart profile-wide, so equal view
    text implies equal candidates (check f), for every view of the day (real Profile2)."""
    import habit_fakes as HF
    from habitus.questions.habit import profile2 as p2
    extra = {21: ("process", {"file_name": "python3", "executable": "/usr/bin/python3", "image_dir": "system"}),
             22: ("process", {"file_name": "worker", "executable": "/usr/bin/worker",
                              "command_line": "worker --port <n>", "image_dir": "system"}),
             23: ("process", {"file_name": "pinger", "executable": "/usr/bin/pinger",
                              "command_line": "pinger -w <n>", "image_dir": "system"})}
    for k, v in extra.items():
        monkeypatch.setitem(HF.ENTITIES, k, v)
    days, links = {}, {}
    for d in CLEAN + TEST:
        days[d], links[d] = build_day(d, list(BENIGN) + [("spawn", 3, 21), ("spawn", 5, 22), ("spawn", 21, 23)])
    h = FakeHabitHost(days, links, ds="HBFAKE_names", art="/nonexistent/habitus_fake_names/")
    prof = p2.get_profile2(h, CLEAN[:2])
    nodes = h.nodes(CLEAN[2])
    w, pg = h.card(nodes[22]), h.card(nodes[23])
    assert prof.tier0(V.invariant_card(w)) == Attest.PATTERN == prof.tier0(V.invariant_card(pg))
    vw, vp = V.build(w, ViewKind.V_par, prof, "k2"), V.build(pg, ViewKind.V_par, prof, "k2")
    cw = targets.candidates(Family.P1_parent, vw, prof, "k2")
    cp = targets.candidates(Family.P1_parent, vp, prof, "k2")
    assert [b.key for b in cw.listed] != [b.key for b in cp.listed]     # different parent idents ...
    assert vw.text != vp.text                                          # ... so different views
    assert {b.text for b in cw.listed} | {b.text for b in cp.listed} == \
        {"python3 (/usr/bin/python3) #1", "python3 (/usr/bin/python3) #2"}
    groups = defaultdict(set)                                           # (family, kind, view) -> candidates
    for s in sorted(set(nodes.values())):
        for kind, v in V.views_for(h.card(s), prof, "k2", T.FAMILIES).items():
            for f, c in targets.candidates_for(v, prof, "k2", [f for f in T.FAMILIES
                                                                if f is not Family.N2_address]).items():
                groups[(f, kind, v.sha1)].add((c.keys(), tuple(b.key for b in c.listed)))
    assert all(len(v) == 1 for v in groups.values())


# --------------------------------------------------------------------------- #
# 4. observe: per-node present sets
# --------------------------------------------------------------------------- #
def test_observe_expected_sets_on_the_attack_day(host, prof):
    obs = observe_all(host, D, prof)
    assert set(obs) == set(host.nodes(D))
    idk = {name: T.ident_key(node_card(host, D, name)) for name in ID}
    py = obs[ID["python3"]]
    assert py.present[Family.P2_modify] == {"dir:home"}
    assert py.present[Family.P3_peers] == {"ep:private", "ep:external"}           # the LOTL lever
    assert py.present[Family.P4_children] == frozenset() and py.present[Family.P5_exec_from] == {"dir:system"}
    assert py.choice[Family.P1_parent] == "usual"
    bash = obs[ID["bash"]]
    assert bash.present[Family.P4_children] == {f"id:{idk['ls']}", f"id:{idk['python3']}", f"id:{idk['firefox']}",
                                                T.GEN_NEVER_SEEN}                  # + the renamed python3 copy
    pay = obs[ID["payload"]]
    assert pay.choice[Family.P1_parent] == "attested-other"                       # firefox: attested, not usual
    assert pay.present[Family.P2_modify] == {"dir:staging"} and pay.present[Family.P5_exec_from] == {"dir:staging"}
    assert pay.present[Family.P3_peers] == {"ep:external"}
    assert obs[ID["fake_python3"]].choice[Family.P1_parent] == "attested-other"   # bash: not ITS usual parent
    out = obs[ID["out_txt"]]
    assert out.present[Family.F1_writer] == {f"id:{idk['python3']}", T.GEN_NEVER_SEEN}   # masquerading writer
    assert out.present[Family.F2_executor] == frozenset()
    drop = obs[ID["drop_sh"]]
    assert drop.present[Family.F1_writer] == {T.GEN_NEVER_SEEN} == drop.present[Family.F2_executor]
    assert obs[ID["pathless"]].present[Family.F1_writer] == frozenset()
    c2 = obs[ID["c2"]]
    assert c2.present[Family.N1_owner] == {T.GEN_NEVER_SEEN}
    assert c2.owners == (T.OwnerObs(T.GEN_TYPED, idk["payload"], "no-program"),)
    lotl = obs[ID["lotl_peer"]]
    assert lotl.owners == (T.OwnerObs(f"id:{idk['python3']}", idk["python3"], "no-program"),)
    assert lotl.present[Family.N1_owner] == {T.GEN_ATTESTED_OTHER}               # attested, not a usual owner
    web = obs[ID["web"]]
    assert web.owners == (T.OwnerObs(f"id:{idk['firefox']}", idk["firefox"], "same-address"),)
    assert web.present[Family.N1_owner] == {f"id:{idk['firefox']}"}
    assert not obs[ID["dns_private"]].applicable(Family.N2_address)              # no prefix chain
    assert obs[ID["dns_enclave"]].applicable(Family.N2_address)
    assert obs[ID["init"]].choice[Family.P1_parent] == "none"
    for o in obs.values():                                                       # every present bucket is a candidate
        cs = _cands_of(host.card(o.sig), prof)
        for f, b in o.present.items():
            assert b <= set(cs[f].keys()), (o.node, f)
    assert targets.owner_buckets(obs, host.nodes(D)[ID["web"]]) == [f"id:{idk['firefox']}"]


def test_observe_is_per_node_not_per_representative(host, prof):
    nd = host.nodes(D)
    assert nd[ID["sh"]] == nd[ID["sh_twin"]]                                      # one signature, n_s = 2
    obs = observe_all(host, D, prof)
    curl = T.ident_key(node_card(host, D, "curl"))
    assert obs[ID["sh"]].present[Family.P4_children] == {f"id:{curl}"}
    assert obs[ID["sh_twin"]].present[Family.P4_children] == {T.GEN_NEVER_SEEN}   # the renamed curl from /tmp
    assert targets.union_present(obs, nd[ID["sh"]])[Family.P4_children] == {f"id:{curl}", T.GEN_NEVER_SEEN}
    # the twin's view and candidates are its signature's (shared), its observation is its own
    assert _cands_of(node_card(host, D, "sh_twin"), prof) == _cands_of(node_card(host, D, "sh"), prof)


def test_observe_with_given_candidates_and_refused_views(host, prof):
    sigs = sorted(set(host.nodes(D).values()))
    cands = {}
    for s in sigs:
        for f, c in _cands_of(host.card(s), prof).items():
            if f is not Family.N2_address:                   # export never computes N2 candidates
                cands[(s, f)] = c
    full = targets.observe(host, D, prof, sigs, "k2", cands)
    assert full == observe_all(host, D, prof)
    py_sig = host.nodes(D)[ID["python3"]]
    drop = {k: v for k, v in cands.items() if k != (py_sig, Family.P3_peers)}     # a refused view's family
    part = targets.observe(host, D, prof, [py_sig], "k2", drop)
    assert Family.P3_peers in part[ID["python3"]].na and Family.P2_modify in part[ID["python3"]].present
    web_sig = host.nodes(D)[ID["web"]]
    assert targets.observe(host, D, prof, [web_sig], "k2", cands)[ID["web"]].applicable(Family.N2_address)
    only = targets.observe(host, D, prof, [py_sig], "k2")
    assert set(only) == {ID["python3"]}
    with pytest.raises(ValueError):
        targets.observe(host, CLEAN[0], prof, [py_sig], "k2")                     # a reference day
    with pytest.raises(ValueError):
        targets.observe(host, D, None, [py_sig], "k2")


def _random_links(host, day, rng, n):
    """n random well-typed links among the day's nodes (spawn, write, exec, net incl. CADETS-like
    process -> file code 4)."""
    nd = host.nodes(day)
    by = defaultdict(list)
    for x, s in sorted(nd.items()):
        by[host.card(s)["type"]].append(x)
    out = []
    for _ in range(n):
        code = int(rng.integers(1, 5))
        p = int(rng.choice(by["process"]))
        if code == 1:
            out.append([p, int(rng.choice(by["process"])), 1])
        elif code == 2:
            out.append([p, int(rng.choice(by["file"])), 2])
        elif code == 3:
            out.append([int(rng.choice(by["file"])), p, 3])
        else:
            out.append([p, int(rng.choice(by["netflow"] + by["file"])), 4])
    return out


@pytest.mark.parametrize("seed", range(6))
def test_monotone_present_sets_under_added_links(host, prof, seed):
    """E_mono prerequisite: added links only ADD present buckets / owners of monotone families; their
    n/a status never changes (gates are link-free)."""
    rng = np.random.default_rng(seed)
    before = observe_all(host, D, prof)
    more = host_with_links(host, D, _random_links(host, D, rng, 12))
    after = observe_all(more, D, prof)
    assert set(before) == set(after)
    for n, a in before.items():
        b = after[n]
        for f in T.FAMILIES_OF[a.ntype]:
            if f not in T.MONOTONE:
                continue
            assert (f in a.na) == (f in b.na), (n, f)
            if f in a.present:
                assert a.present[f] <= b.present[f], (seed, n, f)
        assert set(a.owners) <= set(b.owners), (seed, n)


def test_monotone_present_sets_under_added_op_keys(host, prof):
    sig = host.nodes(D)[ID["python3"]]
    before = observe_all(host, D, prof)[ID["python3"]]
    more = host_with_links(host, D, [])
    card = more._raw[D]["cards"][sig]
    card["behaviour"]["ops"]["delete:staging"] = "1"
    card["behaviour"]["ops"]["recv:net-loopback"] = "2-7"
    after = targets.observe(more, D, prof, [sig], "k2")[ID["python3"]]
    assert before.present[Family.P2_modify] < after.present[Family.P2_modify] == {"dir:home", "dir:staging"}
    assert before.present[Family.P3_peers] < after.present[Family.P3_peers]
    assert "ep:loopback" in after.present[Family.P3_peers]


def test_cadets_process_to_file_send_never_feeds_p3_or_n1(host, prof):
    before = observe_all(host, D, prof)
    more = host_with_links(host, D, [[ID["python3"], ID["out_txt"], 4], [ID["payload"], ID["pathless"], 4]])
    after = observe_all(more, D, prof)
    for n in (ID["python3"], ID["payload"], ID["out_txt"], ID["pathless"]):
        assert after[n] == before[n], n


def test_p1_parent_rules(host, prof):
    nd = host.nodes(D)
    py = node_card(host, D, "python3")
    assert targets.p1_parent(host, D, ID["python3"], py) == T.ident_key(node_card(host, D, "bash"))
    assert targets.p1_parent(host, D, ID["init"], node_card(host, D, "init")) == targets.NO_PARENT
    # parent_name without a matching cause (its spawn link dropped) -> n/a
    L = [list(r) for r in host._rawlinks[D] if not (r[1] == ID["python3"] and r[2] == 1)]
    nolink = FakeHabitHost(copy.deepcopy(host._raw), {**host._rawlinks, D: L})
    assert targets.p1_parent(nolink, D, ID["python3"], py) is None
    assert Family.P1_parent in targets.observe(nolink, D, prof, [nd[ID["python3"]]], "k2")[ID["python3"]].na
    # a cause without a parent_name -> n/a (init gets a spawn link from sshd)
    more = host_with_links(host, D, [[ID["sshd"], ID["init"], 1]])
    assert targets.p1_parent(more, D, ID["init"], node_card(host, D, "init")) is None
    # two distinct identities with the recorded parent name spawn it -> ambiguous -> n/a
    amb = host_with_links(host, D, [[ID["fake_python3"], ID["ls"], 1], [ID["python3"], ID["ls"], 1],
                                    [ID["bash"], ID["ls"], 1]])
    ls = copy.deepcopy(node_card(host, D, "ls"))
    ls["invariant"]["parent_name"] = "python3"
    assert targets.p1_parent(amb, D, ID["ls"], ls) is None
    one = host_with_links(host, D, [[ID["python3"], ID["ls"], 1]])
    assert targets.p1_parent(one, D, ID["ls"], ls) == T.ident_key(py)
    assert Family.P1_parent not in T.MONOTONE


# --------------------------------------------------------------------------- #
# 5. gates, availability, k = 0, N2 options
# --------------------------------------------------------------------------- #
def test_gate_and_frozen_availability(host):
    proc, f, nf = node_card(host, D, "python3"), node_card(host, D, "out_txt"), node_card(host, D, "web")
    priv = node_card(host, D, "dns_private")
    assert targets.gate("THEIA_E3", "P1", "k2", proc) == (True, "")
    assert targets.gate("CADETS_E3", "P1", "k2", proc)[0] is False
    assert targets.gate("CADETS_E3", "P4", "k2", proc)[0] is False and targets.gate("CADETS_E3", "P5", "k2", proc)[0]
    assert targets.gate("ATLASV2_EDR", "P5", "k2", proc)[0] is False and targets.gate("ATLASV2_EDR", "P4", "k2", proc)[0]
    assert targets.gate("CLEARSCOPE_E3", "F2", "k2", f)[0] is False and targets.gate("THEIA_E3", "F2", "k2", f)[0]
    assert targets.gate("THEIA_E3", "N2", "k0", nf)[0] is False and targets.gate("THEIA_E3", "N1", "k0", nf)[0]
    assert targets.gate("THEIA_E3", "N2", "k2", priv) == (False, "no prefix chain (private / loopback / unusable address)")
    assert targets.gate("THEIA_E3", "N2", "k2", nf) == (True, "")
    assert targets.gate("THEIA_E3", "P2", "k2", f)[0] is False                    # wrong node type
    with pytest.raises(KeyError):
        targets.gate("NOT_A_HOST", "P2", "k2", proc)                              # never guessed
    assert targets.gate("NOT_A_HOST", "P5", "k2", proc, {"exec": True, "lineage": False})[0]
    la = targets.link_availability(host)
    assert la["days"] == list(CLEAN) and la["obs"] == {"lineage": False, "exec": False}   # <= 10 links a day
    assert targets.host_obs(host) == host.obs and all(targets.family_table(host).values())
    for ds in T.MEASURED_HOSTS:
        av = T.available(ds)
        assert (Family.P4_children in av) == (ds in T.SPAWN_HOSTS) == (Family.P1_parent in av)
        assert (Family.P5_exec_from in av) == (ds in T.EXEC_HOSTS) == (Family.F2_executor in av)


def test_k0_class_buckets(host, prof):
    obs = observe_all(host, D, None, "k0")
    for o in obs.values():
        assert Family.P1_parent in o.na or o.ntype != "process"
        assert Family.N2_address in o.na or o.ntype != "netflow"
        for f, b in o.present.items():
            assert all(k.startswith(("dir:", "ep:", "cls:")) for k in b), (o.node, f)
    assert obs[ID["out_txt"]].present[Family.F1_writer] == {"cls:system", "cls:staging"}
    assert obs[ID["sh_twin"]].present[Family.P4_children] == {"cls:staging"}
    assert obs[ID["c2"]].present[Family.N1_owner] == {"cls:staging"}
    c = _cands_of(node_card(host, D, "bash"), None, "k0")
    assert set(c) == {Family.P2_modify, Family.P3_peers, Family.P4_children, Family.P5_exec_from}
    assert c[Family.P4_children].keys() == tuple(f"cls:{w}" for w in T.DIR_ORDER)
    v = V.build(node_card(host, D, "python3"), ViewKind.V_par, None, "k0")
    assert targets.candidates(Family.P1_parent, v, None, "k0").applicable is False


def test_n2_options_every_level(host):
    class P:
        def __init__(self, own, hostc):
            self.own, self.hostc = own, hostc

        def chains(self, owner=None):
            return frozenset(self.hostc if owner is None else self.own)
    flow = {"type": "netflow", "invariant": {"remote_ip": "93.184.216.34", "remote_port": "443", "endpoint": "external"}}
    ch = chain_of(flow)
    assert targets.n2_option(flow, "o", P({ch[4]}, ())) == "same-address"
    assert targets.n2_option(flow, "o", P({ch[3]}, ())) == "same-address"
    assert targets.n2_option(flow, "o", P({ch[2]}, ())) == "same-network"
    assert targets.n2_option(flow, "o", P({ch[1]}, ())) == "same-network"
    assert targets.n2_option(flow, "o", P({ch[0]}, {ch[1]})) == "other-programs"
    assert targets.n2_option(flow, "o", P({ch[0]}, {ch[0]})) == "no-program"
    v6 = {"type": "netflow", "invariant": {"remote_ip": "2606:4700::1", "remote_port": "443", "endpoint": "external"}}
    c6 = chain_of(v6)
    assert [V.chain_level(x) for x in c6] == [0, 1, 2, 3, 4] == [V.chain_level(x) for x in ch]
    assert targets.n2_option(v6, "o", P({c6[3]}, ())) == "same-address"
    assert targets.n2_option(v6, "o", P(set(), {c6[2]})) == "other-programs"
    with pytest.raises(ValueError):
        targets.n2_option({"type": "netflow", "invariant": {"remote_ip": "10.0.0.5", "endpoint": "private"}}, "o", P((), ()))


def test_determinism(host, prof):
    a = observe_all(host, D, prof)
    b = observe_all(make_habit_host(), D, FakeProfile2(make_habit_host(), CLEAN[:2]))
    assert a == b and list(a) == sorted(a)
    va = {n: {k: v.text for k, v in vs.items()} for n, vs in all_views(host, D, prof).items()}
    vb = {n: {k: v.text for k, v in vs.items()} for n, vs in all_views(make_habit_host(), D, prof).items()}
    assert va == vb


def test_real_profile2_views_when_present(host):
    """With the real Profile2: every view is IFC-clean, behaviour-free and every observed bucket is a
    candidate."""
    from habitus.questions.habit import profile2 as p2
    prof = p2.get_profile2(host, CLEAN[:2])
    for n, s in sorted(host.nodes(D).items()):
        card = host.card(s)
        base = V.views_for(card, prof, "k2", T.FAMILIES)
        for v in base.values():
            assert ifc.violations(v) == []
        mut = V.views_for(_mutated(card, 0), prof, "k2", T.FAMILIES)
        assert {k: v.text for k, v in mut.items()} == {k: v.text for k, v in base.items()}
    obs = observe_all(host, D, prof)
    for o in obs.values():
        cs = _cands_of(host.card(o.sig), prof)
        for f, b in o.present.items():
            assert b <= set(cs[f].keys())
    # one class-key definition: what views put in View.meta is what Profile2 indexes under
    assert V.NO_LEVEL == p2.NO_LEVEL
    keyfn = {Family.P2_modify: V.beh_class, Family.F1_writer: V.file_class, Family.N1_owner: V.n1_class}
    for d in (CLEAN[2], D):
        for n, s in sorted(host.nodes(d).items()):
            card = host.card(s)
            for kind, v in V.views_for(card, prof, "k2", T.FAMILIES).items():
                for f, fn in keyfn.items():
                    if T.FAMILY_SPEC[f].view is kind:
                        assert fn(v.meta) == p2.class_key(f, card, prof), (n, f)
    for name in ("dns_private", "web", "lotl_peer"):
        v = V.build(node_card(host, D, name), ViewKind.V_net, prof, "k2")
        assert V.n1_class(v.meta) == p2.class_key(Family.N1_owner, node_card(host, D, name), prof)
        assert v.meta["prefix_level"] in ("", p2.prefix_level(node_card(host, D, name), prof))


# --------------------------------------------------------------------------- #
# 6. real host: measured link availability equals the frozen table (label-free, link files only)
# --------------------------------------------------------------------------- #


def test_hygiene_grep_on_the_view_modules():
    for f in MINE:
        lines = [ln for ln in open(os.path.join(HB, f), encoding="utf-8").read().splitlines() if HYGIENE.search(ln)]
        assert lines == [], f


# --------------------------------------------------------------------------- #
# 6. real host: measured link availability equals the frozen table (label-free, link files only)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ds", ("CADETS_E3", "ATLASV2_EDR"))
def test_real_host_link_availability_matches_frozen_table(ds):
    from test_habit_reference import real_cards_dir
    d = real_cards_dir(ds)
    if d is None:
        missing_asset(f"{ds} card files")
    from habitus.data.host import Host
    h = Host(ds, cards_dir=d, max_days=2)
    la = targets.link_availability(h)
    assert la["days"] == T.predeploy(h) and la["days"]
    assert la["obs"]["lineage"] == (ds in T.SPAWN_HOSTS) and la["obs"]["exec"] == (ds in T.EXEC_HOSTS)
    assert T.available(ds, "k2", la["obs"]) == T.available(ds)
