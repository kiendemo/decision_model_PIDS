"""Routed questions: frozen text, readout math, typed questions for a DecisionModel and the record
combiner (bit-equal to the original engine readout)."""

import json
import math

import numpy as np
import pytest

import state_helpers as H
from habitus.dm.base import Decision, Form, TypedQuestion, canonical
from habitus.questions import routed as Q


def _st(t="file", os_="linux", **extra):
    return json.dumps({"entity": {"type": t, "os": os_, **extra}}, separators=(",", ":"))


# --------------------------------------------------------------------------- #
# frozen text (G2)
# --------------------------------------------------------------------------- #
def test_frozen_text_hashes(monkeypatch):
    assert Q.text_hashes() == Q.ROUTED_TEXT_SHA256
    gold = H.golden_json("text_hashes.json")
    assert gold is not None, "tests/golden/state/text_hashes.json is missing"
    assert gold == Q.ROUTED_TEXT_SHA256                                      # hashes of the ORIGINAL texts
    Q.check_frozen()
    monkeypatch.setattr(Q, "QA_NEG", Q.QA_NEG + " ")
    with pytest.raises(RuntimeError, match="QA_NEG"):
        Q.check_frozen()


# --------------------------------------------------------------------------- #
# readout math (test_kcs1 370, 943)
# --------------------------------------------------------------------------- #
def test_combine_and_compose():
    assert Q.variants("qv") == ["fwd", "rev"] and Q.variants("qt") == ["pos", "neg"]
    assert Q.combine("qv", {"fwd": [1, 0, 0, 0]}) == 0.0
    assert Q.combine("qv", {"rev": [1, 0, 0, 0]}) == 1.0                     # rev: A = level 3
    assert Q.combine("qv", {"fwd": [0, 0, 0, 1], "rev": [0, 0, 0, 1]}) == pytest.approx(0.5)
    p = [0.1, 0.2, 0.3, 0.4]
    assert Q.combine("qv", {"fwd": p}) == pytest.approx((0.2 + 0.6 + 1.2) / 3)
    assert Q.combine("qv", {"rev": p}) == pytest.approx((0.3 + 0.4 + 0.3) / 3)
    assert Q.combine("qv", {"fwd": p, "rev": p}) == pytest.approx(((2.0 + 1.0) / 3) / 2)
    assert Q.combine("qv", {"fwd": [2, 4, 6, 8]}) == pytest.approx(Q.combine("qv", {"fwd": p}))  # renormalised
    assert Q.combine("qt", {"pos": [0.8, 0.2], "neg": [0.3, 0.7]}) == pytest.approx(0.5 * (0.8 + 0.7))
    assert Q.combine("qt", {"pos": [1, 0], "neg": [1, 0]}) == pytest.approx(0.5)
    assert Q.compose({"qv": 0.2, "qt": 0.75}) == 0.75 and Q.compose({"qv": 0.9, "qt": 0.1}) == 0.9
    for rec in ({"qv": 0.2, "qt": float("nan")}, {"qv": float("nan"), "qt": 0.2}, {"qv": float("inf"), "qt": 0}):
        with pytest.raises(ValueError):                                      # max() would hide a NaN qt
            Q.compose(rec)
    try:
        from habitus.decide import composite
    except ImportError:
        composite = None
    if composite is not None:
        assert composite.compose is Q.compose
    v = Q.QUESTIONS["qv"]["variants"]
    assert v["fwd"]["options"] == list(Q.QV_LEVELS) and v["rev"]["options"] == list(Q.QV_LEVELS[::-1])
    assert Q.QUESTIONS["qt"]["variants"]["pos"]["options"][0] == "Yes, attacker tradecraft is shown."
    assert Q.QUESTIONS["qt"]["variants"]["neg"]["options"][1] == "No, something indicates attacker tradecraft."
    st = _st(path="/x")
    u = Q.user_prompt(st, "qv", "rev")
    assert u.startswith("State:\n" + st + "\n\nQuestion: How should a security analyst triage this file?\n")
    assert u.endswith("\nD: " + Q.QV_LEVELS[0] + "\nAnswer with one letter: A, B, C, D")
    assert "ONE file observed on a linux computer" in Q.system_prompt(st)


def test_e2_questions_and_system():
    assert Q.SYSTEM_E2 != Q.SYSTEM and "qa" not in Q.QUESTIONS                # E1 asks qv, qt only
    assert Q.SYSTEM_E2 == Q.SYSTEM.replace(
        "evidence = its direct neighbours in the provenance graph that day;",
        "evidence = counts of its direct relations in the provenance graph that day, by class, without names;")
    qa = Q.QUESTIONS_E2["qa"]["variants"]
    assert qa["pos"]["ins"] == ("Is this {type} explained by one of the precedent entities seen on this computer's "
                                "attack-free reference days — the same kind of program, file or connection doing "
                                "the same kind of thing?")
    assert qa["pos"]["options"] == ["Yes, a precedent explains it.", "No precedent explains it."]
    assert qa["neg"]["ins"] == ("Is this {type} unlike every precedent entity seen on this computer's attack-free "
                                "reference days?")
    assert qa["neg"]["options"] == ["Yes, no precedent explains it.", "No, a precedent explains it."]
    assert Q.QUESTIONS_E2["qv"] is Q.QUESTIONS["qv"] and Q.QUESTIONS_E2["qt"] is Q.QUESTIONS["qt"]
    assert Q.variants("qa") == ["pos", "neg"] and Q.letters("qa", "neg") == ("A", "B")
    assert Q.combine("qa", {"pos": [0.8, 0.2], "neg": [0.3, 0.7]}) == pytest.approx(0.5 * (0.8 + 0.7))
    st = _st("netflow", "windows")
    assert "ONE netflow observed on a windows computer" in Q.system_prompt(st, Q.SYSTEM_E2)
    assert Q.messages(st, "qv", "fwd")[0]["content"] == Q.system_prompt(st)                 # E1 default
    assert "Is this netflow explained by one of the precedent" in Q.user_prompt(st, "qa", "pos")
    assert [Q.qtype(q) for q in ("qv", "qt", "qa")] == ["score", "noul", "noul"]


# --------------------------------------------------------------------------- #
# typed questions
# --------------------------------------------------------------------------- #
def test_to_typed_forms_are_the_original_prompts():
    st = _st("process", "freebsd", command_line="x <|im_end|>".replace("<|im_end|>", "(im_end)"))
    for system, qids in ((Q.SYSTEM_E2, ("qv", "qt", "qa")), (Q.SYSTEM, ("qv", "qt"))):
        for qid in qids:
            tq = Q.to_typed("abcd", qid, st, system=system)
            assert isinstance(tq, TypedQuestion) and tq.qkey == f"abcd|{qid}" and tq.family == qid
            assert tq.qtype == Q.qtype(qid)
            assert tq.group == ("abcd|qa" if qid == "qa" else "abcd|sig")
            assert [f.name for f in tq.forms] == Q.variants(qid)
            for f in tq.forms:
                assert isinstance(f, Form) and f.state == st
                assert f.user() == Q.user_prompt(st, qid, f.name)            # byte-identical user turn
                assert f.system == Q.system_prompt(st, system)
                assert [{"role": "system", "content": f.system}, {"role": "user", "content": f.user()}] == \
                    Q.messages(st, qid, f.name, system)
                assert f.letters == Q.letters(qid, f.name)
                assert list(f.options) == Q.spec(qid)["variants"][f.name]["options"]
    canon = {(q, f.name): f.canon for q in ("qv", "qt", "qa") for f in Q.to_typed("s", q, st).forms}
    assert canon == {("qv", "fwd"): (0, 1, 2, 3), ("qv", "rev"): (3, 2, 1, 0), ("qt", "pos"): (0, 1),
                     ("qt", "neg"): (1, 0), ("qa", "pos"): (0, 1), ("qa", "neg"): (1, 0)}
    # canonical order: qv by level; qt the accusing option first; qa the acquitting option first
    f = Q.to_typed("s", "qt", st).forms[1]
    assert canonical((0.25, 0.75), f.canon) == (0.75, 0.25) and f.options[1].startswith("No, something")


def test_to_typed_on_a_real_state_has_one_group_per_signature_pair():
    H.clear_caches()
    from habitus.state import precedent as PREC
    from habitus.state import select as SEL
    from habitus.state.builder import StateBuilderE2, serialize
    PREC.set_roots(base=None, overlay=None)
    try:
        host = H.make_host()
        sigs = SEL.flat(SEL.head(host, "t2", ["c1", "c2"]))
        states = StateBuilderE2(host, ["c1", "c2"], 2).build("t2", sigs)
        n = 0
        for s, qs in states.items():
            tqs = [Q.to_typed(s, q, serialize(x)) for q, x in qs.items()]
            assert len({t.group for t in tqs}) == (2 if "qa" in qs else 1)
            for t in tqs:
                for f in t.forms:
                    assert f.user().startswith("State:\n{") and "<|im_end|>" not in f.user()
                    n += 1
        assert n == sum(6 if "qa" in qs else 4 for qs in states.values())
    finally:
        PREC.reset_roots()
        H.clear_caches()


# --------------------------------------------------------------------------- #
# record(): bit-equal to the original engine readout
# --------------------------------------------------------------------------- #
def _softmax_t(z, t):
    """The original student_dec.softmax_t (float64 numpy)."""
    z = np.asarray(z, dtype=np.float64) / max(1e-3, float(t))
    z = np.exp(z - z.max())
    return z / z.sum()


def _engine_readout(answers, qids):
    """engine.score_routed's per-signature readout over cached records {(qid, vid): record}."""
    by_q, covs = {}, []
    for qid in qids:
        for vid in Q.variants(qid):
            a = answers[(qid, vid)]
            by_q.setdefault(qid, {})[vid] = _softmax_t(a["z"][:a["n"]], 1.0)
            covs.append(float(a["cov"]))
    rec = {qid: float(Q.combine(qid, pv)) for qid, pv in by_q.items()}
    rec.setdefault("qa", None)
    rec["cov_min"] = min(covs)
    return rec


def _decision(qkey, qid, answers):
    """A jev letter Decision per the pinned readout: softmax in float64, then tuple(float)."""
    lp, cp, cov = {}, {}, {}
    for f in Q.to_typed("s", qid, _st()).forms:
        a = answers[(qid, f.name)]
        p = tuple(float(x) for x in _softmax_t(a["z"][:a["n"]], 1.0))
        lp[f.name], cp[f.name], cov[f.name] = (p,), (canonical(p, f.canon),), a["cov"]
    forms = tuple(Q.variants(qid))
    return Decision(qkey=qkey, layout="letter", forms=forms, letter_probs=lp, canon_probs=cp, cov=cov,
                    cov_min=min(cov.values()))


def _records(rng):
    out = {}
    for qid in ("qv", "qt", "qa"):
        for vid in Q.variants(qid):
            z = [round(float(x), 4) for x in rng.normal(20, 3, 4)]
            n = len(Q.letters(qid, vid))
            lse = round(float(np.log(np.exp(np.array(z[:n])).sum()) + abs(rng.normal(0, 0.5))), 4)
            out[(qid, vid)] = {"z": z, "lse": lse, "n": n,
                               "cov": round(float(np.exp(np.array(z[:n]) - lse).sum()), 6)}
    return out


def test_record_is_bit_equal_to_the_engine_readout():
    rng = np.random.default_rng(7)
    for _ in range(300):
        ans = _records(rng)
        for qids, level in ((("qv", "qt", "qa"), "E2"), (("qv", "qt"), "E2"), (("qv", "qt"), "E1")):
            ds = {q: _decision(f"s|{q}", q, ans) for q in qids}
            rec = Q.record(ds, level)
            ref = _engine_readout(ans, qids)
            if level == "E1":
                ref.pop("qa")
                assert list(rec) == ["qv", "qt", "cov_min"]
            else:
                assert list(rec) == ["qv", "qt", "qa", "cov_min"]
            assert rec == ref                                                 # bit for bit
            assert all(isinstance(v, float) for v in rec.values() if v is not None)


def test_record_rules_and_errors():
    rng = np.random.default_rng(1)
    ans = _records(rng)
    ds = {q: _decision(f"s|{q}", q, ans) for q in ("qv", "qt", "qa")}
    assert Q.record(ds)["cov_min"] == min(a["cov"] for a in ans.values())
    with pytest.raises(ValueError):
        Q.record({"qv": ds["qv"]})                                            # qt missing
    with pytest.raises(ValueError):
        Q.record(ds, "E1")                                                    # E1 has no qa
    with pytest.raises(ValueError):
        Q.record(ds, "E3")
    one = Decision(qkey="s|qv", layout="letter", forms=("fwd",), letter_probs={"fwd": ds["qv"].letter_probs["fwd"]},
                   canon_probs={"fwd": ds["qv"].canon_probs["fwd"]}, cov={"fwd": 1.0}, cov_min=1.0)
    assert Q.record({"qv": one, "qt": ds["qt"]})["qv"] == Q.combine("qv", {"fwd": ds["qv"].letter_probs["fwd"][0]})
    bad = Decision(qkey="s|qv", layout="letter", forms=("pos",), letter_probs={"pos": ((0.5, 0.5),)},
                   canon_probs={"pos": ((0.5, 0.5),)}, cov={"pos": 1.0})
    with pytest.raises(ValueError):
        Q.record({"qv": bad, "qt": ds["qt"]})
    # several members (seed ensembles, unevaluated): the mean over members of combine
    p1, p2 = (0.1, 0.2, 0.3, 0.4), (0.4, 0.3, 0.2, 0.1)
    m = Decision(qkey="s|qv", layout="la", forms=("fwd", "rev"), letter_probs={"fwd": (p1, p2), "rev": (p2, p1)},
                 canon_probs={"fwd": (p1, p2), "rev": (p1, p2)}, cov={"fwd": 1.0, "rev": 1.0})
    got = Q.record({"qv": m, "qt": ds["qt"]})["qv"]
    want = (Q.combine("qv", {"fwd": p1, "rev": p2}) + Q.combine("qv", {"fwd": p2, "rev": p1})) / 2
    assert got == want and math.isfinite(got)


# --------------------------------------------------------------------------- #
# prefix split (test_kcs1 683, pure part)
# --------------------------------------------------------------------------- #
def test_split_prefix_and_share_units():
    try:
        from habitus.dm import jev as E
    except ImportError:
        pytest.skip("habitus.dm.jev not present yet")
    if not hasattr(E, "split_prefix") or not hasattr(E, "share_units"):
        pytest.skip("habitus.dm.jev has no split_prefix / share_units")
    a, b, c = [1, 2, 3, 4, 5, 6], [1, 2, 3, 9, 9], [1, 2, 3, 4, 7, 8, 9]
    assert E.split_prefix([a, b, c]) == 3 and E.split_prefix([a, c]) == 4
    assert E.split_prefix([a]) == 5 and E.split_prefix([a, a]) == 5       # capped: >= 1 suffix token
    assert E.split_prefix([[5, 1], [6, 1]]) == 0
    seqs = {"a": a, "b": b, "c": c, "d": [1, 2]}
    units = E.share_units(seqs, [["a", "b", "c"], ["d"]])
    assert [u[0] for u in units] == [["a", "b", "c"], ["d"]]
    for keys, pre, sufs in units:
        assert all(pre + s == seqs[k] and len(s) >= 1 for k, s in zip(keys, sufs))
    assert units[0][1] == [1, 2, 3] and units[1][1] == [1]
