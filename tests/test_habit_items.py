"""Habit questions, training-item export and sampling (CPU): question tables, the deny-list, the
SYSTEM_H budget, the ONE renderer (== the Qwen chat template), provenance asserts (never a test day,
never a late clean day, never the inner day, never the held-out host), references, member-share
targets, absent items, P1 / N2 items, view pooling, orders, A4 twins, the byte-identical rebuild,
dedupe / stratify / waterfill / balance / sample_fold, and the label-hygiene grep. Builder doubles
(FakeProf & co.) exercise export alone; one test runs it with the real profile2 / views / targets."""

import dataclasses
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict

import numpy as np
import pytest

from conftest import REPO, missing_asset
from habit_fakes import BENIGN, TEST, FakeHabitHost, build_day

from habitus.questions.habit import constants as H
from habitus.questions.habit import text as Q
from habitus.questions.habit import types as T
from habitus.train import items as X

S = X                                   # sampling lives in train.items too
HYGIENE = re.compile(r"gt\(|ground_truth|host_labels|Ground_Truth|gt_")     # the label-token pattern
# sha256 of the sorted JSON of {fixed question texts, deny-list, bucket phrases, A4 injections}, computed
# from the ORIGINAL question bank (habitus/questions.py, targets.py, export.py) on 2026-10-30
FIXED_TEXT_SHA256 = "d4176c5b9235b1ffea2c8b47e9446b14b7cf101106a38401762b868fdaafaf45"
FOLD = "loho-HBHELD"

# a 5-clean-day fake host: pre-deployment 01-01..01-04 (inner 01-04), 01-08 is clean but after the
# first test day (the CADETS 04-07..04-10 case); 01-02 adds a twin sh (one signature, two members
# with different children) and a new external peer of python3
CLEAN5 = ("2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-08")
EXTRA = {"2020-01-02": [("spawn", 7, 12), ("spawn", 12, 14), ("connect", 5, 205)]}


def make_host(ds="HBFAKE"):
    days, links = {}, {}
    for d in CLEAN5 + TEST:
        days[d], links[d] = build_day(d, list(BENIGN) + EXTRA.get(d, []))
    for d in days:
        days[d]["dataset"] = ds
    return FakeHabitHost(days, links, ds=ds, clean=CLEAN5, test=TEST)


def qwen_tok():
    try:
        from habitus.dm.jev import load_tokenizer
        return load_tokenizer()
    except (OSError, ValueError, ImportError):
        return None


@pytest.fixture(scope="module")
def qtok():
    t = qwen_tok()
    if t is None:
        missing_asset("the pinned Qwen3.5-4B tokenizer")
    return t


@pytest.fixture
def hb_root(tmp_path):
    return str(tmp_path / "hb")


class FakeTok:
    """Qwen-format chat template + whitespace "tokens" (fast; render goes through text.render)."""

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=None):
        assert tokenize is False and add_generation_prompt is True and enable_thinking is False
        s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
        return s + "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __call__(self, text, add_special_tokens=False):
        if isinstance(text, str):
            return {"input_ids": text.split()}
        return {"input_ids": [t.split() for t in text]}


@pytest.fixture
def fold(monkeypatch):
    monkeypatch.setitem(H.FOLDS, FOLD, ("HBFAKE",))
    monkeypatch.setitem(H.FOLD_HELDOUT, FOLD, "HBHELD")
    return FOLD


@pytest.fixture(scope="module")
def host():
    return make_host()


# --------------------------------------------------------------------------- #
# Reference-side / view-side doubles (Deps signatures)
# --------------------------------------------------------------------------- #
class FakeProf:
    """Exact-identity reference index over the reference days: seen days per ident key, spawn /
    write / net pairs by ident (the parts the doubles below read)."""

    def __init__(self, host, ref_days):
        T.assert_predeploy(host, ref_days)
        self.ds, self.ref, self.k = host.ds, tuple(sorted(ref_days)), len(ref_days)
        self.days, self.cards = Counter(), {}
        self.children, self.parents, self.writers, self.owners, self.execs = (defaultdict(Counter) for _ in range(5))
        for d in self.ref:
            nodes = host.nodes(d)
            for i in {T.ident_key(host.card(s)) for s in set(nodes.values())}:
                self.days[i] += 1
            for s in set(nodes.values()):
                self.cards[T.ident_key(host.card(s))] = host.card(s)
            pairs = {(c, T.ident_key(host.card(nodes[a])), T.ident_key(host.card(nodes[b])))
                     for a, b, c in host.links(d).tolist()}
            for c, ia, ib in sorted(pairs):
                if c == 1:                                  # [parent, child]
                    self.children[ia][ib] += 1
                    self.parents[ib][ia] += 1
                elif c == 2:                                # [process, file]: file -> writer
                    self.writers[ib][ia] += 1
                elif c == 3:                                # [file, process]: file -> executor
                    self.execs[ia][ib] += 1
                else:                                       # [process, netflow]: flow -> owner
                    self.owners[ib][ia] += 1

    def tier0(self, card):
        i = T.ident_key(card)
        if not self.days.get(i):
            return T.Attest.NONE
        if T.has_placeholder(card) or T.uninformative(card):
            return T.Attest.PATTERN
        return T.Attest.IDENTITY

    def display(self, ident):
        c = self.cards.get(ident)
        if c is None or self.tier0(c) < T.Attest.IDENTITY:
            return None
        return c["invariant"].get("file_name") or os.path.basename(c["invariant"].get("path", "")) or None


def fake_get_profile2(host, ref_days):
    return FakeProf(host, ref_days)


def fake_build(card, kind, prof, k_mode, owner=None):
    kind = T.ViewKind(kind)
    inv = card["invariant"]
    tier = prof.tier0(card) if prof else T.Attest.NONE
    if kind is T.ViewKind.V_own:
        ent = {"type": "process", "owner": "typed" if owner is None else prof.display(owner) or "typed"}
        secs = {"entity": ent, "connection": {"endpoint": inv["endpoint"]},
                "reference": {"days": prof.k if prof else 0}}
    else:
        ent = {"type": card["type"], "os": card["os"]}
        ent.update({"process": {"image_dir": inv.get("image_dir", "none")}, "file": {"dir": inv.get("dir", "none")},
                    "netflow": {"endpoint": inv.get("endpoint", "unknown")}}[card["type"]])
        if tier >= T.Attest.IDENTITY:
            for k in X.RAW_FIELDS:
                if inv.get(k):
                    ent[k] = inv[k]
        secs = {"entity": ent, "reference": {"days": prof.k if prof else 0,
                                             "seen": f"{prof.days.get(T.ident_key(card), 0)}/{prof.k}" if prof else "0/0"}}
    meta = {"ident": T.ident_key(card), "tier0": tier.name, "k_mode": k_mode, "owner": owner or ""}
    return T.View(kind, card["type"], secs, (), meta)


def fake_views_for(card, prof, k_mode, families):
    out = {}
    for f in families:
        kind = T.FAMILY_SPEC[T.Family.of(f)].view
        if kind is not T.ViewKind.V_own and kind not in out:
            out[kind] = fake_build(card, kind, prof, k_mode)
    return out


IDENT_HABIT = {"P4": "children", "F1": "writers", "F2": "execs", "N1": "owners", "P1": "parents"}


def fake_candidates(family, view, prof, k_mode):
    f = T.Family.of(family)
    if f in (T.Family.P2_modify, T.Family.P5_exec_from):
        return T.Candidates(f, tuple(T.Bucket(f, f"dir:{w}", f"{w} directories") for w in T.DIR_ORDER))
    if f is T.Family.P3_peers:
        return T.Candidates(f, tuple(T.Bucket(f, f"ep:{w}", f"{w} endpoints") for w in T.EP_ORDER))
    if prof is None:
        return T.Candidates(f, tuple(T.Bucket(f, f"cls:{w}", f"a program in {w} directories") for w in T.DIR_ORDER))
    hab = getattr(prof, IDENT_HABIT[f.value])[view.meta["ident"]]
    usual = [i for i, _ in sorted(hab.items(), key=lambda kv: (-kv[1], kv[0])) if prof.display(i)]
    if f is T.Family.P1_parent:
        return T.Candidates(f, (), tuple(T.Bucket(f, f"id:{i}", f"the program {prof.display(i)}")
                                         for i in usual[:H.MAX_P1_NAMES]))
    listed = tuple(T.Bucket(f, f"id:{i}", f"the program {prof.display(i)}") for i in usual[:H.MAX_LISTED])
    return T.Candidates(f, listed + (T.Bucket(f, T.GEN_ATTESTED_OTHER, "a program seen on the reference days", True),
                                     T.Bucket(f, T.GEN_NEVER_SEEN, "a program never seen on the reference days", True)))


def fake_observe(host, day, prof, sigs, k_mode, cands=None):
    nodes = host.nodes(day)
    want = set(sigs)
    links = host.links(day).tolist()
    avail = T.available(host.ds, k_mode, host.obs)

    def bucket(f, other_card):
        if prof is None:
            return "cls:" + other_card["invariant"].get("image_dir", "none")
        i = T.ident_key(other_card)
        if f"id:{i}" in cands[key].keys():
            return f"id:{i}"
        return T.GEN_ATTESTED_OTHER if prof.tier0(other_card) >= T.Attest.IDENTITY else T.GEN_NEVER_SEEN

    out = {}
    for n, s in sorted(nodes.items()):
        if s not in want:
            continue
        card = host.card(s)
        present, choice, na, owners = {}, {}, set(), []
        ops = card["behaviour"].get("ops", {})
        for f in T.FAMILIES_OF[card["type"]]:
            key = (s, f)
            if f not in avail:
                na.add(f)
                continue
            if f is T.Family.N2_address:
                own = sorted({a for a, b, c in links if b == n and c == 4})
                for a in own:
                    oc = host.card(nodes[a])
                    oi = T.ident_key(oc)
                    b = f"id:{oi}" if prof.tier0(oc) >= T.Attest.IDENTITY else T.GEN_TYPED
                    seen = oi in prof.owners[T.ident_key(card)]
                    owners.append(T.OwnerObs(b, oi, "same-address" if seen else "no-program"))
                continue
            if key not in cands:
                na.add(f)
                continue
            if f is T.Family.P1_parent:
                pname = card["invariant"].get("parent_name")
                causes = {T.ident_key(host.card(nodes[a])) for a, b, c in links
                          if b == n and c == 1 and host.card(nodes[a])["invariant"].get("file_name") == pname}
                if not pname and not causes:
                    choice[f] = "none"
                elif len(causes) != 1:
                    na.add(f)
                else:
                    pi = next(iter(causes))
                    pc = prof.cards.get(pi) or next(host.card(nodes[a]) for a, b, c in links if b == n and c == 1)
                    choice[f] = ("usual" if f"id:{pi}" in {b.key for b in cands[key].listed} else
                                 "attested-other" if prof.tier0(pc) >= T.Attest.IDENTITY else "never-seen")
                continue
            if f is T.Family.P2_modify:
                el = {f"dir:{k.split(':', 1)[1]}" for k in ops
                      if k.split(":", 1)[0] in ("write", "create", "rename", "delete") and k.split(":", 1)[1] in T.DIR_ORDER}
            elif f is T.Family.P3_peers:
                el = {f"ep:{k.split(':net-', 1)[1]}" for k in ops if ":net-" in k}
            elif f is T.Family.P4_children:
                el = {bucket(f, host.card(nodes[b])) for a, b, c in links if a == n and c == 1}
            elif f is T.Family.P5_exec_from:
                el = {f"dir:{host.card(nodes[a])['invariant']['dir']}" for a, b, c in links if b == n and c == 3}
            elif f is T.Family.F1_writer:
                el = {bucket(f, host.card(nodes[a])) for a, b, c in links if b == n and c == 2}
            elif f is T.Family.F2_executor:
                el = {bucket(f, host.card(nodes[b])) for a, b, c in links if a == n and c == 3}
            else:
                el = {bucket(f, host.card(nodes[a])) for a, b, c in links if b == n and c == 4}
            present[f] = frozenset(el)
        out[n] = T.NodeObs(n, s, card["type"], T.ident_key(card), present, choice, tuple(owners), frozenset(na))
    return out


def fake_deps():
    return X.Deps(fake_get_profile2, fake_views_for, fake_build, fake_candidates, fake_observe)


def a_view(ident="i1", tier="IDENTITY", entity=None, kind=T.ViewKind.V_beh, ntype="process"):
    ent = entity or {"type": ntype, "os": "linux", "image_dir": "system", "file_name": "bash"}
    secs = {"entity": ent, "reference": {"days": 2}}
    if kind is T.ViewKind.V_own:
        secs = {"entity": ent, "connection": {"endpoint": "external"}, "reference": {"days": 2}}
    return T.View(kind, ntype, secs, (), {"ident": ident, "tier0": tier, "k_mode": "k2"})


# --------------------------------------------------------------------------- #
# 1. questions
# --------------------------------------------------------------------------- #
def test_tables_cover_every_family_and_match_the_spec():
    assert set(Q.TEMPLATES) == set(T.FAMILIES) == set(Q.OPTION_TEXT)
    for f, sp in T.FAMILY_SPEC.items():
        assert len(Q.OPTION_TEXT[f]) == len(sp.options) <= H.MAX_OPTIONS
        assert ("{bucket}" in Q.TEMPLATES[f]) == (sp.kind == "inclusion")
    assert "{names}" in Q.OPTION_TEXT[T.Family.P1_parent][0]
    assert Q.THINK_SUFFIX == "</think>\n\n" and "{" not in Q.SYSTEM_H


def test_deny_list_audit_is_clean_and_catches_products():
    assert Q.audit() == {}
    assert Q.deny_hits("Runs EQNEDT32.EXE, then Plink and an Equation-Editor macro") == \
        ["eqnedt", "equation editor", "plink"]
    assert Q.deny_hits("the DARPA engagement") == ["darpa", "engagement"]
    assert Q.deny_hits("ec2 instance, trace-free, e35") == ["trace"]        # short entries are whole words
    assert Q.audit({"x": "a malicious payload"}) == {"x": ["malicious", "payload"]}
    for txt in Q.fixed_texts().values():
        for w in ("plink", "equation editor", "darpa", "attack", "carbanak", "mimikatz"):
            assert w not in txt.lower()


def test_system_h_token_budget(qtok):
    n = len(qtok(Q.SYSTEM_H, add_special_tokens=False)["input_ids"])
    assert n <= H.SYSTEM_H_MAX_TOKENS, n
    for section in {s for secs in T.VIEW_SECTIONS.values() for s in secs}:
        assert section + " = " in Q.SYSTEM_H, section            # the glossary names every section


def test_build_inclusion_and_choice_questions():
    v = a_view()
    b = T.Bucket("P2", "dir:staging", "staging or temporary directories")
    q = Q.build(v, "P2", b)
    assert q.text.endswith("Does it create, write, rename or delete files in staging or temporary directories today?")
    assert q.option_text == Q.OPTION_TEXT[T.Family.P2_modify] and q.options == T.INCLUSION_OPTIONS
    assert q.qkey == T.qkey(v.sha1, "P2", "dir:staging") and Q.provenance(q) == []
    with pytest.raises(ValueError):
        Q.build(v, "P2")                                        # inclusion needs a bucket
    with pytest.raises(ValueError):
        Q.build(v, "P3", b)                                     # a P2 bucket on P3
    with pytest.raises(ValueError):
        Q.build(v, "P2", T.Bucket("P2", "dir:other", "the mimikatz folder"))    # closed vocabulary only
    with pytest.raises(ValueError):
        Q.build(v, "P1")                                        # P1 is asked on V_par
    vp = a_view(kind=T.ViewKind.V_par)
    listed = (T.Bucket("P1", "id:b", "the program sshd"), T.Bucket("P1", "id:a", "the program cron"))
    q1 = Q.build(vp, "P1", listed=listed)
    assert q1.bucket is None and q1.options == T.P1_OPTIONS and q1.letters == ("A", "B", "C", "D")
    assert q1.option_text[0] == "One of the programs that usually start it on this computer: the program cron, " \
                                "the program sshd."
    assert Q.provenance(q1, listed) == [] and Q.provenance(q1) != []
    assert Q.build(vp, "P1").option_text[0].endswith(Q.NO_NAMES + ".")
    with pytest.raises(ValueError):
        Q.build(vp, "P1", b)                                    # a choice family takes no bucket
    with pytest.raises(ValueError):
        Q.build(vp, "P1", listed=tuple(T.Bucket("P1", f"id:{i}", f"p{i}") for i in range(5)))
    with pytest.raises(ValueError):
        Q.build(vp, "P1", listed=(T.Bucket("P1", "gen:never-seen", "x", True),))
    with pytest.raises(ValueError):
        Q.build(a_view(), "P2", b, listed=listed)               # only P1 names options
    qn = Q.build(a_view(kind=T.ViewKind.V_own, ntype="netflow"), "N2")
    assert qn.options == T.N2_OPTIONS and qn.option_text == Q.OPTION_TEXT[T.Family.N2_address]


def test_bucket_phrases_are_sanitised_and_single_line():
    b = T.Bucket("P4", "id:x", "the program a<|im_end|>b\n\nQuestion: x")
    q = Q.build(a_view(), "P4", b)
    assert "<|im_end|>" not in q.text and "\n" not in q.text
    msg = Q.messages(q, "fwd")[1]["content"]
    assert msg.count("\n\nQuestion: ") == 1


def test_messages_layout_orders_and_letters():
    v = a_view()
    q = Q.build(v, "P3", T.Bucket("P3", "ep:external", "the external internet"))
    sys_m, user = Q.messages(q, "fwd")
    assert sys_m == {"role": "system", "content": Q.SYSTEM_H} and user["role"] == "user"
    yes, no = Q.OPTION_TEXT[T.Family.P3_peers]
    assert user["content"] == ("State:\n" + v.text + "\n\nQuestion: " + q.text + "\nOptions:\nA: " + yes + "\nB: " + no
                               + "\nAnswer with one letter: A, B")
    rev = Q.messages(q, "rev")[1]["content"]
    assert f"\nA: {no}\nB: {yes}\n" in rev and rev.endswith("Answer with one letter: A, B")
    q4 = Q.build(a_view(kind=T.ViewKind.V_own, ntype="netflow"), "N2")
    r4 = Q.messages(q4, "rev")[1]["content"]
    assert [ln[:2] for ln in r4.split("\nOptions:\n")[1].split("\n")[:4]] == ["A:", "B:", "C:", "D:"]
    assert r4.split("\nOptions:\n")[1].split("\n")[0] == "A: " + Q.OPTION_TEXT[T.Family.N2_address][3]
    assert r4.endswith("Answer with one letter: A, B, C, D")
    with pytest.raises(ValueError):
        Q.messages(q, "both")


def test_render_prefix_and_fits_on_the_fake_tokenizer():
    tok = FakeTok()
    v = a_view()
    qs = [Q.build(v, f, T.Bucket(f, "dir:home", "home directories")) for f in ("P2", "P5")]
    qs.append(Q.build(v, "P3", T.Bucket("P3", "ep:private", "private network endpoints")))
    pre = {Q.prefix_text(tok, q) for q in qs}
    assert len(pre) == 1
    p = pre.pop()
    assert p.endswith(v.text + "\n\nQuestion: ")
    for q in qs:
        for o in T.ORDERS:
            r = Q.render(tok, q, o)
            assert r.startswith(p) and r.endswith(Q.THINK_SUFFIX)
            assert r == Q.render(tok, Q.build(a_view(), q.family, q.bucket), o)    # rebuilt: byte-identical
    assert Q.fits(tok, qs[0]) and Q.fits(tok, v)
    long_v = a_view(entity={"type": "process", "os": "linux", "command_line": " ".join(["x"] * 2000)})
    assert not Q.fits(tok, long_v) and Q.fits(tok, long_v, limit=10 ** 6)

    class NoThink(FakeTok):
        def apply_chat_template(self, msgs, **kw):
            return "x"
    with pytest.raises(ValueError):
        Q.render(NoThink(), qs[0], "fwd")


def test_render_is_the_qwen_chat_template(qtok):
    v = a_view(kind=T.ViewKind.V_file, ntype="file", entity={"type": "file", "os": "linux", "dir": "home"})
    q = Q.build(v, "F1", T.Bucket("F1", T.GEN_NEVER_SEEN, "a program never seen on this computer's reference days",
                                  True))
    for o in T.ORDERS:
        r = Q.render(qtok, q, o)
        direct = qtok.apply_chat_template(Q.messages(q, o), tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False)
        assert r == direct and r.endswith("<think>\n\n</think>\n\n")
        assert r.startswith("<|im_start|>system\n" + Q.SYSTEM_H + "<|im_end|>\n<|im_start|>user\nState:\n")
        assert r.startswith(Q.prefix_text(qtok, q))
    assert Q.fits(qtok, q)


# --------------------------------------------------------------------------- #
# 2. export: provenance and references
# --------------------------------------------------------------------------- #
def test_assert_trainable_refuses_everything_outside_the_window(host, fold):
    assert T.predeploy(host) == list(CLEAN5[:4]) and T.inner_day(host) == "2020-01-04"
    for d in CLEAN5[:3]:
        X.assert_trainable(host, d, fold)
    X.assert_trainable(host, "2020-01-04", fold, "inner")
    bad = [("2020-01-04", "train"), ("2020-01-01", "inner"), ("2020-01-08", "train"), (TEST[0], "train"),
           (TEST[1], "train"), ("2020-01-07", "train"), ("2020-01-01", "test")]
    for d, role in bad:
        with pytest.raises(X.ProvenanceError):
            X.assert_trainable(host, d, fold, role)
    with pytest.raises(X.ProvenanceError):
        X.assert_trainable(host, CLEAN5[0], "no-such-fold")


def test_window_closes_at_the_configured_first_test_day_even_without_test_cards(host, fold):
    """Review finding: predeploy / assert_trainable used the card-filtered host.test and failed open
    (every clean day) when the test-day card files were missing; the window now comes from the
    configured test dates (HostSpec.configured_test), and a host without any test day is refused."""
    from types import SimpleNamespace
    h = FakeHabitHost(host._raw, host._rawlinks, ds=host.ds, art="/nonexistent/habitus_export_t0/",
                      clean=CLEAN5, test=())
    with pytest.raises(ValueError, match="no configured test day"):
        T.predeploy(h)                                                    # undefined, never "all clean"
    h.spec = SimpleNamespace(configured_test=list(TEST))
    assert T.first_test_day(h) == TEST[0] and T.predeploy(h) == T.predeploy(host) == list(CLEAN5[:4])
    for d in ("2020-01-07", "2020-01-08"):                                # clean, but after the first test day
        with pytest.raises(X.ProvenanceError):
            X.assert_trainable(h, d, fold)
    X.assert_trainable(h, CLEAN5[0], fold)


def test_refs_never_contain_the_day_or_the_inner_day(host, fold):
    inner = T.inner_day(host)
    seen = set()
    for seed in range(30):
        for d in CLEAN5[:3]:
            refs = X.refs_for_day(host, d, fold, np.random.default_rng(seed))
            for r in refs:
                assert d not in r.ref_days and inner not in r.ref_days and r.day == d
                assert set(r.ref_days) <= set(T.predeploy(host))
                seen.add(r.k_mode)
                if r.k_mode == "allc-pre" and r.k == 2:
                    assert set(r.ref_days) == set(CLEAN5[:3]) - {d}
            assert [r.k_mode for r in refs] == ["k0", "k1", "k2", "allc-pre"]
    assert seen == {"k0", "k1", "k2", "allc-pre"}
    a = X.refs_for_day(host, CLEAN5[0], fold, np.random.default_rng(3))
    b = X.refs_for_day(host, CLEAN5[0], fold, np.random.default_rng(3))
    assert a == b
    inn = X.refs_for_day(host, inner, fold, np.random.default_rng(0), role="inner")
    assert [r.k_mode for r in inn] == ["k0", "k2", "allc-pre"]
    assert all(inner not in r.ref_days for r in inn) and inn[-1].ref_days == CLEAN5[:3]
    # A3: a replaced reference is a proper, non-empty subset with its parent's k mode
    sub = [r for s in range(40) for r in X.refs_for_day(host, CLEAN5[0], fold, np.random.default_rng(s), a3=1.0)
           if r.k_mode in ("k2", "allc-pre")]
    assert sub and all(r.k == 1 for r in sub)
    with pytest.raises(X.ProvenanceError):
        X.refs_for_day(host, TEST[0], fold, np.random.default_rng(0))


def test_check_ref_refuses_inner_test_and_late_reference_days(host, fold):
    for days in (("2020-01-04",), (TEST[0],), ("2020-01-08",), ("2020-01-01", "2020-01-08")):
        with pytest.raises(X.ProvenanceError):
            X.check_ref(host, T.Ref(host.ds, "2020-01-02", days, "k2" if len(days) == 2 else "k1"), fold)
    with pytest.raises(X.ProvenanceError):                       # raised at call time, before iteration
        X.items_for_ref(host, T.Ref(host.ds, "2020-01-02", ("2020-01-04",), "k1"), fold, FakeTok(),
                        np.random.default_rng(0), deps=fake_deps())
    with pytest.raises(X.ProvenanceError):
        X.items_for_ref(host, T.Ref(host.ds, TEST[0], ("2020-01-01",), "k1"), fold, FakeTok(),
                        np.random.default_rng(0), deps=fake_deps())


# --------------------------------------------------------------------------- #
# 3. export: items of one reference
# --------------------------------------------------------------------------- #
def _items(host, fold, day="2020-01-02", ref_days=("2020-01-01",), k_mode="k1", **kw):
    ref = T.Ref(host.ds, day, ref_days, k_mode)
    return list(X.items_for_ref(host, ref, fold, FakeTok(), np.random.default_rng(0), deps=fake_deps(), **kw))


def test_inclusion_targets_are_member_shares(host, fold):
    items = _items(host, fold, twin_rate=0.0)
    for it in items:
        assert isinstance(it, T.Item) and it.role == "train" and it.fold == fold and it.src == "code"
    sh = host.nodes("2020-01-02")[11]
    assert host.nodes("2020-01-02")[12] == sh                     # the twin sh: one signature, two members
    p4 = {it.bucket: S.canonical_target(it) for it in items if it.sig == sh and it.family is T.Family.P4_children}
    curl = T.ident_key(host.card(host.nodes("2020-01-02")[13]))
    assert p4[f"id:{curl}"] == (0.5, 0.5)                         # one of the two members starts curl
    assert p4[T.GEN_NEVER_SEEN] == (0.5, 0.5)                     # the other a never-seen /tmp curl
    absent = [b for b, t in p4.items() if t[0] == 0]
    assert len(absent) == 1 and absent == [T.GEN_ATTESTED_OTHER]  # = min(#absent cands, #present)
    assert all(it.n_members == 2 for it in items if it.sig == sh and it.family is T.Family.P4_children)


def test_absent_items_match_present_items_per_signature_and_family(host, fold):
    items = _items(host, fold, twin_rate=0.0)
    by = defaultdict(Counter)
    cand_n = {T.Family.P2_modify: 8, T.Family.P3_peers: 6, T.Family.P5_exec_from: 8}
    for it in items:
        if T.FAMILY_SPEC[it.family].kind == "inclusion":
            by[(it.sig, it.family)][S.polarity(it)] += 1
    assert by
    for (sig, f), c in by.items():
        n_cand = cand_n.get(f)
        if n_cand is not None:
            assert c["absent"] == min(c["present"], n_cand - c["present"]), (f, c)
        else:
            assert c["absent"] <= c["present"]
        assert c["present"] >= 1


def test_choice_items_p1_and_n2(host, fold):
    items = _items(host, fold, twin_rate=0.0)
    p1 = [it for it in items if it.family is T.Family.P1_parent]
    n2 = [it for it in items if it.family is T.Family.N2_address]
    assert p1 and n2
    for it in p1 + n2:
        t = S.canonical_target(it)
        assert it.bucket == T.NO_BUCKET and it.letters == ("A", "B", "C", "D") and abs(sum(t) - 1) < 1e-9
    nodes = host.nodes("2020-01-02")
    init = [it for it in p1 if it.sig == nodes[1]]
    assert S.canonical_target(init[0]) == (0.0, 0.0, 0.0, 1.0)    # init: no parent recorded -> "none"
    bash = [it for it in p1 if it.sig == nodes[3]][0]
    assert S.canonical_target(bash)[0] == 1.0 and "the program sshd" in bash.prompt_text
    lotl = [it for it in n2 if it.sig == nodes[205]]
    assert len(lotl) == 1 and S.canonical_target(lotl[0]) == (0.0, 0.0, 0.0, 1.0)


def test_signatures_sharing_a_view_are_pooled_into_one_question(host, fold):
    """At k0 the typed V_net view of every external flow is the same bytes (web 201, the new python3
    peer 205): one N1 question per bucket whose target pools both flows' owners."""
    st = Counter()
    items = _items(host, fold, ref_days=(), k_mode="k0", twin_rate=0.0, stats=st)
    keys = [(it.view_sha1, it.family, it.bucket) for it in items]
    assert len(keys) == len(set(keys))                                  # one item per (view, family, bucket)
    nodes = host.nodes("2020-01-02")
    ext = [it for it in items if it.family is T.Family.N1_owner and '"endpoint":"external"' in it.prompt_text]
    assert {it.sig for it in ext} <= {nodes[201], nodes[205]} and ext
    present = [it for it in ext if S.polarity(it) == "present"]
    assert [it.bucket for it in present] == ["cls:system"] and present[0].n_members == 2
    assert S.canonical_target(present[0]) == (1.0, 0.0)
    assert st["groups_shared"] >= 1


def test_orders_letters_and_prompts_are_consistent(host, fold):
    items = _items(host, fold, twin_rate=0.0)
    tok = FakeTok()
    orders = Counter(it.variant for it in items)
    assert orders["fwd"] and orders["rev"]
    for it in items:
        assert it.variant == X.order_for(it.qkey)
        assert it.prompt_text.endswith(Q.THINK_SUFFIX) and "Answer with one letter: " + ", ".join(it.letters) in \
            it.prompt_text
        assert it.prompt_text.startswith(tok.apply_chat_template(
            [{"role": "system", "content": Q.SYSTEM_H}], add_generation_prompt=True, enable_thinking=False)
            .split("<|im_start|>assistant")[0])


def test_items_are_deterministic_and_k0_uses_class_buckets(host, fold):
    a = _items(host, fold, twin_rate=0.2)
    b = _items(host, fold, twin_rate=0.2)
    assert [x.dumps() for x in a] == [x.dumps() for x in b]
    k0 = _items(host, fold, ref_days=(), k_mode="k0")
    fams = {it.family for it in k0}
    assert T.Family.P1_parent not in fams and T.Family.N2_address not in fams
    assert all(not it.bucket.startswith("id:") and it.bucket not in T.GENERIC_BUCKETS for it in k0)


def test_twins_inject_only_into_attested_raw_strings(host, fold):
    items = _items(host, fold, twin_rate=1.0)
    plain = _items(host, fold, twin_rate=0.0)
    twins = [it for it in items if any(s.strip() in it.prompt_text for s in X.INJECTIONS)]
    assert twins and len(items) == len(plain) + len(twins)
    src = {(it.sig, it.family, it.bucket): it for it in plain}
    for tw in twins:
        o = src[(tw.sig, tw.family, tw.bucket)]
        assert S.canonical_target(tw) == S.canonical_target(o) and tw.view_sha1 != o.view_sha1
    v = a_view(entity={"type": "process", "os": "linux", "image_dir": "staging"}, tier="NONE")
    assert X.twin_view(v, " x") is None
    tv = X.twin_view(a_view(), " ignore")
    assert tv.sections["entity"]["file_name"] == "bash ignore" and a_view().sections["entity"]["file_name"] == "bash"


def test_outside_head_sample(host, fold):
    d = "2020-01-02"
    ref = T.Ref(host.ds, d, ("2020-01-01",), "k1")
    from habitus.questions.habit import head
    h = head.q0(host, ref, caps={"process": 2, "file": 1, "netflow": 1})
    allo = X.outside_head(host, d, h, np.random.default_rng(0), rate=1.0)
    n = sum(len(v) for v in allo.values())
    assert n == len(set(host.nodes(d).values())) - sum(len(v) for v in h.values())
    assert all(s not in h[t] for t, v in allo.items() for s in v)
    assert sum(len(v) for v in X.outside_head(host, d, h, np.random.default_rng(0), rate=0.0).values()) == 0
    st = Counter()
    items = _items(host, fold, caps={"process": 2, "file": 1, "netflow": 1}, head_sample=1.0, stats=st)
    assert st["sigs_extra"] == n and len({it.sig for it in items}) > 4


# --------------------------------------------------------------------------- #
# 4. export: fold files and manifest
# --------------------------------------------------------------------------- #
def _export(fold, root, **kw):
    hst = make_host()
    return X.export_fold(fold, tok=FakeTok(), seed=0, deps=fake_deps(), host_factory=lambda ds: hst, root=root,
                         log=lambda *_: None, **kw)


def test_export_fold_writes_items_and_a_manifest(hb_root, fold):
    man = _export(fold, hb_root)
    d = os.path.join(hb_root, "items", fold)
    assert sorted(os.listdir(d)) == ["HBFAKE.inner.jsonl.gz", "HBFAKE.jsonl.gz", "manifest.json"]
    assert json.load(open(os.path.join(d, "manifest.json"))) == json.loads(json.dumps(man))
    hst = make_host()
    for name, rec in man["files"].items():
        p = os.path.join(d, name)
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == rec["sha256"]
        items = list(X.read_items(p))
        assert len(items) == rec["n"] > 0
        assert len({it.prompt_text for it in items}) == len(items)               # duplicates merged
        for it in items:
            X.check_item(it, hst, fold)
            assert it.day not in TEST and it.day != "2020-01-08"
            if name.endswith(".inner.jsonl.gz"):
                assert it.role == "inner" and it.day == "2020-01-04" and it.k_mode != "k1"
            else:
                assert it.role == "train" and it.day in CLEAN5[:3]
            assert "2020-01-04" not in it.ref_days
    tot = sum(r["n"] for r in man["files"].values())
    assert sum(man["per_family"].values()) == tot == sum(man["per_kmode"].values())
    assert sum(n for f in man["counts"]["HBFAKE"].values() for n in f.values()) == tot
    assert set(man["per_family"]) == {f.value for f in T.FAMILIES}
    assert {k.split(":")[1] for k in man["per_kmode"]} == {"k0", "k1", "k2", "allc-pre"}
    assert man["tokens"]["all"]["n"] == tot and man["tokens"]["all"]["p50"] > 0
    assert man["code_sha256"]["train/items.py"] and man["spec"]["version"] == H.SPEC_VERSION
    assert man["per_host"]["HBFAKE"]["inner_day"] == "2020-01-04"


def test_export_rebuild_is_byte_identical(hb_root, fold, tmp_path):
    m1 = _export(fold, hb_root)
    m2 = _export(fold, str(tmp_path / "hb2"))
    assert m1["files"] == m2["files"] and m1 == m2
    for name in m1["files"]:
        a = open(os.path.join(hb_root, "items", fold, name), "rb").read()
        assert a == open(os.path.join(tmp_path, "hb2", "items", fold, name), "rb").read()
    m3 = X.export_fold(fold, tok=FakeTok(), seed=1, deps=fake_deps(), host_factory=lambda ds: make_host(),
                       root=str(tmp_path / "hb3"), log=lambda *_: None)
    assert m3["files"] != m1["files"]


def test_export_restrictions_for_probes(hb_root, fold):
    man = _export(fold, hb_root, days={"HBFAKE": ["2020-01-02"]}, kmodes=("allc-pre",), inner=False, tag="probe",
                  a3=0.0)
    d = os.path.join(hb_root, "items", f"{fold}.probe")
    assert sorted(os.listdir(d)) == ["HBFAKE.jsonl.gz", "manifest.json"]
    items = list(X.read_items(os.path.join(d, "HBFAKE.jsonl.gz")))
    assert items and {(it.day, it.k_mode, it.ref_days) for it in items} == \
        {("2020-01-02", "allc-pre", ("2020-01-01", "2020-01-03"))}
    assert man["restrict"]["kmodes"] == ["allc-pre"] and man["restrict"]["inner"] is False
    with pytest.raises(X.ProvenanceError):
        _export(fold, hb_root, hosts=("THEIA_E3",))


def test_bucket_phrases_and_raw_keys_agree_with_the_question_bank():
    from habitus.questions.habit import targets, vocab
    fam = {"dir": "P2", "ep": "P3", "cls": "P4", "gen": "P4"}
    for k, ph in sorted(targets.BUCKET_PHRASES.items()):
        assert Q.deny_hits(ph) == [], (k, ph)
        f = fam[k.split(":")[0]]
        q = Q.build(a_view(), f, T.Bucket(f, k, ph, k.startswith("gen:")))
        assert Q.provenance(q) == [] and ph in q.text
    assert set(X.RAW_FIELDS) == {k for ks in vocab.RAW_KEYS.values() for k in ks}


def test_items_with_the_real_builder_modules(host, fold):
    """Integration with the real profile2 / views / targets on the fake host: member-share targets, k0
    rules, and no unattested card string in any prompt."""
    from habitus.questions.habit import profile2, targets, views
    deps = X.Deps(profile2.get_profile2, views.views_for, views.build, targets.candidates, targets.observe)
    d = "2020-01-02"
    out = {}
    for km, rd in (("k0", ()), ("k1", ("2020-01-01",)), ("allc-pre", ("2020-01-01", "2020-01-03"))):
        ref = T.Ref(host.ds, d, rd, km)
        out[km] = list(X.items_for_ref(host, ref, fold, FakeTok(), np.random.default_rng(0), deps=deps))
    assert {it.family for it in out["k1"]} == set(T.FAMILIES)
    assert not {T.Family.P1_parent, T.Family.N2_address} & {it.family for it in out["k0"]}
    sh = host.nodes(d)[11]
    p4 = {it.bucket: S.canonical_target(it) for it in out["k1"] if it.sig == sh and it.family is T.Family.P4_children
          and not any(x.strip() in it.prompt_text for x in X.INJECTIONS)}
    curl = T.ident_key(host.card(host.nodes(d)[13]))
    assert p4[f"id:{curl}"] == (0.5, 0.5) and p4[T.GEN_NEVER_SEEN] == (0.5, 0.5)
    for km, items in out.items():
        for it in items:
            assert "/tmp/curl" not in it.prompt_text                        # an unattested executable
            assert '"executable":"/usr/bin/curl"' not in it.prompt_text or km != "k0"


# --------------------------------------------------------------------------- #
# 5. sample
# --------------------------------------------------------------------------- #
def _item(ds="A", family="P2", bucket="dir:home", k_mode="k2", ref=("d0",), p=1.0, variant="fwd", w=1.0, role="train",
          prompt=None, day="d9", sig="s"):
    f = T.Family.of(family)
    sp = T.FAMILY_SPEC[f]
    sha = hashlib.sha1(f"{ds}{sig}{family}{prompt}".encode()).hexdigest()
    if sp.kind == "inclusion":
        canon = (p, 1 - p)
    else:
        canon = tuple(1.0 if i == int(p) else 0.0 for i in range(len(sp.options)))
        bucket = T.NO_BUCKET
    t = canon if variant == "fwd" else tuple(reversed(canon))
    if k_mode == "k0":
        ref = ()
    return T.Item(ds, day, ref, k_mode, sig, f, bucket, sha, T.qkey(sha, f, None if bucket == T.NO_BUCKET else bucket),
                  variant, prompt or f"{ds}|{sig}|{family}|{bucket}|{k_mode}|{variant}", T.LETTERS[:len(canon)], t, w,
                  1, role, "f", "code")


def test_dedupe_merges_exact_duplicate_prompts():
    a = _item(p=1.0, prompt="P", day="d1")
    b = _item(p=0.0, prompt="P", day="d2", w=3.0)
    c = _item(p=0.5, prompt="Q")
    d = _item(p=1.0, prompt="P", role="inner", day="d3")
    out = S.dedupe([c, b, a, d])
    assert len(out) == 3
    m = [x for x in out if x.prompt_text == "P" and x.role == "train"][0]
    assert m.target == pytest.approx((0.25, 0.75)) and m.weight == 4.0 and m.n_members == 2 and m.day == "d1"
    assert S.dedupe([a, b, c, d]) == out
    with pytest.raises(ValueError):
        S.dedupe([_item(prompt="P"), dataclasses.replace(_item(prompt="P"), fold="g")])


def test_waterfill_quotas():
    q = S.waterfill({"a": 0.5, "b": 0.3, "c": 0.2}, {"a": 1000, "b": 1000, "c": 1000}, 100)
    assert q == {"a": 50, "b": 30, "c": 20}
    q = S.waterfill({"a": 0.5, "b": 0.3, "c": 0.2}, {"a": 10, "b": 1000, "c": 1000}, 100)
    assert q["a"] == 10 and sum(q.values()) == 100 and abs(q["b"] / q["c"] - 1.5) < 0.1
    q = S.waterfill({"a": 1.0, "b": 1.0}, {"a": 3, "b": 4}, 100)
    assert q == {"a": 3, "b": 4}
    assert S.waterfill({"a": 0.0, "b": 1.0}, {"a": 5, "b": 5}, 3) == {"b": 3}


def _population(seed=0):
    rng = np.random.default_rng(seed)
    out = []
    fams = {"process": ("P2", "P3", "P1"), "file": ("F1",), "netflow": ("N1", "N2")}
    for ds, mult in (("A", 3), ("B", 1)):
        for nt, fs in fams.items():
            for f in fs:
                for k in ("k0", "k1", "k2", "allc-pre"):
                    if f in ("P1", "N2") and k == "k0":
                        continue
                    for i in range(40 * mult):
                        sig = f"{nt}{f}{k}{i}"
                        if T.FAMILY_SPEC[T.Family.of(f)].kind == "choice":
                            out.append(_item(ds, f, k_mode=k, p=int(rng.integers(4)), sig=sig))
                            continue
                        rare = i % 10 == 0
                        b = {"P2": "dir:staging" if rare else "dir:system", "P3": "ep:external" if rare else "ep:private",
                             "F1": T.GEN_NEVER_SEEN if rare else "id:x", "N1": T.GEN_NEVER_SEEN if rare else "id:y"}[f]
                        out.append(_item(ds, f, b, k_mode=k, p=1.0 if i % 3 else 0.5, sig=sig,
                                         variant="fwd" if i % 2 else "rev"))
                        out.append(_item(ds, f, "dir:cache" if f == "P2" else "ep:loopback" if f == "P3" else
                                         T.GEN_ATTESTED_OTHER, k_mode=k, p=0.0, sig=sig + "x"))
    return out


def test_stratify_composition_hosts_rarity_and_weights():
    pop = _population()
    n = 800
    s = S.stratify(pop, n, seed=0)
    assert len(s) == n and all(x.role == "train" for x in s)
    sm = S.summary(s)
    shares = {k: v["n"] / n for k, v in sm["ntype"].items()}
    for nt, want in S.COMPOSITION["ntype"].items():
        assert abs(shares[nt] - want) < 0.03, (nt, shares)
    ks = {k: v["n"] / n for k, v in sm["kmode"].items()}
    for k, want in S.COMPOSITION["kmode"].items():
        assert abs(ks[k] - want) < 0.03, (k, ks)
    pol = sm["polarity"]
    assert abs(pol["present"]["n"] - pol["absent"]["n"]) <= 0.05 * n
    assert abs(sm["host"]["A"]["n"] - sm["host"]["B"]["n"]) <= 0.1 * n      # host-balanced quotas
    # rare present buckets are oversampled (<= 4x) and the weights restore their population share
    rare = {"dir:staging", "ep:external", T.GEN_NEVER_SEEN}
    present = [x for x in s if S.polarity(x) == "present"]
    rare_n = sum(x.bucket in rare for x in present) / len(present)
    rare_w = sum(x.weight for x in present if x.bucket in rare) / sum(x.weight for x in present)
    pp = [x for x in pop if S.polarity(x) == "present"]
    pop_rare = sum(x.bucket in rare for x in pp) / len(pp)
    assert 2.0 * pop_rare < rare_n <= 4.0 * pop_rare / (1 + 3 * pop_rare) * 1.25
    assert abs(rare_w - pop_rare) < 0.3 * pop_rare
    fac = S.rarity(pop)
    assert max(fac.values()) <= S.COMPOSITION["rare_max"] and min(fac.values()) >= 1.0
    assert all(x.weight > 0 for x in s)
    assert [x.dumps() for x in S.stratify(list(reversed(pop)), n, seed=0)] == [x.dumps() for x in s]
    assert [x.dumps() for x in S.stratify(pop, n, seed=1)] != [x.dumps() for x in s]
    with pytest.raises(ValueError):
        S.stratify(pop + [_item(role="inner", sig="zz")], 10)
    assert len(S.stratify(pop, 10 ** 6)) == len(pop)                          # n above the population: all


def test_dedupe_by_members_pools_member_shares():
    a = dataclasses.replace(_item(p=1.0, prompt="P", sig="s1"), n_members=1)
    b = dataclasses.replace(_item(p=0.0, prompt="P", sig="s2"), n_members=3)
    m = S.dedupe([a, b], by="members")
    assert len(m) == 1 and m[0].target == pytest.approx((0.25, 0.75)) and m[0].weight == 1.0 and m[0].n_members == 4
    with pytest.raises(ValueError):
        S.dedupe([a], by="nodes")


def test_sample_fold_writes_a_stratified_sample(hb_root, fold):
    man = _export(fold, hb_root)
    sm = S.sample_fold(fold, n=40, seed=0, root=hb_root)
    d = os.path.join(hb_root, "items", fold, "sample")
    name = "train_n40_s0.jsonl.gz"
    assert sorted(os.listdir(d)) == [name, "train_n40_s0.manifest.json"]
    items = list(X.read_items(os.path.join(d, name)))
    assert len(items) == sm["file"][name]["n"] == 40 and all(it.role == "train" for it in items)
    assert sm["sources"] == {"HBFAKE.jsonl.gz": man["files"]["HBFAKE.jsonl.gz"]["sha256"]}
    assert sum(it.weight for it in items) == pytest.approx(40.0)
    assert S.sample_fold(fold, n=40, seed=0, root=hb_root)["file"] == sm["file"]   # deterministic
    with open(os.path.join(hb_root, "items", fold, "HBFAKE.jsonl.gz"), "ab") as f:
        f.write(b"x")
    with pytest.raises(ValueError):
        S.sample_fold(fold, n=40, root=hb_root)                                     # sha256 checked


def test_balance_hosts_and_n_fold():
    items = [_item("A", sig=f"a{i}") for i in range(30)] + [_item("B", sig=f"b{i}", w=2.0) for i in range(5)]
    out = S.balance_hosts(items)
    tot = Counter()
    for it in out:
        tot[it.ds] += it.weight
    assert tot["A"] == pytest.approx(tot["B"]) and sum(tot.values()) == pytest.approx(40.0)
    assert S.n_fold(1.0) == 60_000 and S.n_fold(2.5) == 90_000 and S.n_fold(10.0) == 120_000
    with pytest.raises(ValueError):
        S.n_fold(0)


# --------------------------------------------------------------------------- #
# 6. hygiene
# --------------------------------------------------------------------------- #
def test_export_items_gz_is_plain_jsonl(hb_root, fold):
    man = _export(fold, hb_root, days={"HBFAKE": ["2020-01-01"]}, kmodes=("k0",), inner=False)
    p = os.path.join(hb_root, "items", fold, "HBFAKE.jsonl.gz")
    raw = gzip.open(p, "rt").read().splitlines()
    assert len(raw) == man["files"]["HBFAKE.jsonl.gz"]["n"]
    rec = json.loads(raw[0])
    assert set(rec) == set(T.Item.__dataclass_fields__) and rec["fold"] == fold
    assert open(p, "rb").read()[4:8] == b"\x00\x00\x00\x00"                  # gzip mtime 0


def test_assert_host_refuses_the_held_out_host(monkeypatch):
    monkeypatch.setitem(H.FOLDS, "loho-x", ("HBFAKE", "HBHELD"))
    monkeypatch.setitem(H.FOLD_HELDOUT, "loho-x", "HBHELD")
    X.assert_host("HBFAKE", "loho-x")
    for ds in ("HBHELD", "THEIA_E3"):
        with pytest.raises(X.ProvenanceError):
            X.assert_host(ds, "loho-x")
    for f in H.FOLDS:
        if f == "loho-x":
            continue
        held = H.FOLD_HELDOUT[f]
        if held:
            with pytest.raises(X.ProvenanceError):
                X.assert_host(held, f)
        for ds in H.FOLDS[f]:
            X.assert_host(ds, f)
    assert set(H.FOLDS["all"]) == set(H.TRAIN_HOSTS) and H.FOLD_HELDOUT["all"] is None


def test_fixed_texts_are_the_original_question_bank():
    from habitus.questions.habit import targets
    d = {"fixed": Q.fixed_texts(), "deny": list(Q.DENY_LIST), "phrases": targets.BUCKET_PHRASES,
         "injections": list(X.INJECTIONS)}
    assert hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False).encode()).hexdigest() == FIXED_TEXT_SHA256


def test_glue_strings_and_form_user_equal_the_habit_user_prompt():
    from habitus.dm import base
    assert (Q.STATE_HEAD, Q.QUESTION_HEAD, Q.OPTIONS_HEAD, Q.ANSWER_HEAD, Q.THINK_SUFFIX) == \
        (base.STATE_HEAD, base.QUESTION_HEAD, base.OPTIONS_HEAD, base.ANSWER_HEAD, base.THINK_SUFFIX)
    assert Q.LETTERS == base.LETTERS == ("A", "B", "C", "D")
    qs = [Q.build(a_view(), "P3", T.Bucket("P3", "ep:external", "the external internet")),
          Q.build(a_view(kind=T.ViewKind.V_own, ntype="netflow"), "N2"),
          Q.build(a_view(kind=T.ViewKind.V_par), "P1", listed=(T.Bucket("P1", "id:a", "the program cron"),))]
    for q in qs:
        tq = T.to_typed(q)
        assert [f.name for f in tq.forms] == list(T.ORDERS) and tq.qkey == q.qkey and tq.qtype == "choice"
        assert tq.group == f"{q.view.kind.value}|{q.view.sha1}" and tq.family == q.family.value
        for f in tq.forms:
            assert f.system == Q.SYSTEM_H and f.state == q.view.text and f.question == q.text
            assert f.user() == Q.user_prompt(q, f.name) and Q.messages(q, f.name)[1]["content"] == f.user()
        assert T.to_typed(q, ("rev",)).forms[0].canon == tuple(reversed(range(len(q.options))))


def test_hygiene_grep_on_the_question_and_item_modules():
    for p in (os.path.join("questions", "habit", "text.py"), os.path.join("train", "items.py")):
        lines = open(os.path.join(REPO, "src", "habitus", p), encoding="utf-8").read().splitlines()
        assert [ln for ln in lines if HYGIENE.search(ln)] == [], p


def test_importing_the_item_export_pulls_no_label_code():
    code = ("import sys; import habitus.train.items, habitus.questions.habit.text; "
            "bad = [m for m in sys.modules if m.startswith('habitus.eval')]; print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, env={**os.environ, "PYTHONPATH": os.path.join(REPO, "src")},
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr


def test_deps_real_uses_the_habit_modules():
    d = X.Deps.real()
    assert callable(d.observe) and d.get_profile2.keywords == {"role": "train"}
    assert X.Deps.real("select").get_profile2.keywords == {"role": "score"}
