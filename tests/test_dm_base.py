"""The pinned decision-model contract (habitus.dm.base)."""

import importlib.util
import os
import sys

import pytest

import dm_helpers as H
from habitus.dm import base as B


def _form(name="fwd", options=("Yes.", "No."), canon=(0, 1), state='{"entity":{"type":"process","os":"linux"}}'):
    return B.Form(name, "SYS", state, "Is it?", options, canon)


def test_form_user_layout_and_letters():
    f = _form(options=("one", "two", "three"), canon=(2, 0, 1))
    assert f.letters == ("A", "B", "C")
    assert f.user() == ('State:\n{"entity":{"type":"process","os":"linux"}}\n\nQuestion: Is it?\nOptions:\n'
                        "A: one\nB: two\nC: three\nAnswer with one letter: A, B, C")
    assert (B.STATE_HEAD, B.QUESTION_HEAD, B.OPTIONS_HEAD, B.ANSWER_HEAD) == (
        "State:\n", "\n\nQuestion: ", "\nOptions:\n", "\nAnswer with one letter: ")
    assert B.THINK_SUFFIX == "</think>\n\n" and B.LETTERS == ("A", "B", "C", "D")


def test_form_validation():
    with pytest.raises(ValueError):
        _form(options=("only",), canon=(0,))
    with pytest.raises(ValueError):
        _form(options=tuple("abcde"), canon=tuple(range(5)))
    with pytest.raises(ValueError):
        _form(canon=(0, 0))
    with pytest.raises(ValueError):
        _form(canon=(0,))
    f = _form(options=["x", "y"], canon=[1, 0])
    assert f.options == ("x", "y") and f.canon == (1, 0) and hash(f) == hash(_form(options=("x", "y"), canon=(1, 0)))


def test_canonical_maps_letters_to_canonical_order():
    assert B.canonical((0.1, 0.2, 0.3, 0.4), (3, 2, 1, 0)) == (0.4, 0.3, 0.2, 0.1)
    assert B.canonical((0.7, 0.3), (1, 0)) == (0.3, 0.7)
    assert B.canonical((0.25, 0.75), (0, 1)) == (0.25, 0.75)
    with pytest.raises(ValueError):
        B.canonical((0.5, 0.5), (0, 1, 2))


def test_typed_question_and_decision_validation():
    q = B.TypedQuestion("s|qt", "qt", "noul", [_form("pos"), _form("neg", canon=(1, 0))], "s|sig")
    assert isinstance(q.forms, tuple) and q.n_options == 2
    with pytest.raises(ValueError):
        B.TypedQuestion("s|qt", "qt", "noul", (), "s|sig")
    with pytest.raises(ValueError):
        B.TypedQuestion("s|qt", "qt", "noul", (_form("pos"), _form("pos")), "s|sig")
    with pytest.raises(ValueError):
        B.TypedQuestion("s|qt", "qt", "noul", (_form("pos"), _form("neg", ("a", "b", "c"), (0, 1, 2))), "s|sig")
    d = B.Decision("s|qt", "letter", ("pos",), {"pos": ((0.4, 0.6),)}, {"pos": ((0.4, 0.6),)}, {"pos": 0.9}, 0.9)
    assert d.members == 1
    with pytest.raises(ValueError):
        B.Decision("s|qt", "nope", ("pos",), {"pos": ((1.0,),)}, {"pos": ((1.0,),)}, {"pos": 1.0}, 1.0)
    with pytest.raises(ValueError):
        B.Decision("s|qt", "la", ("pos", "neg"), {"pos": ((1.0,),)}, {"pos": ((1.0,),)}, {"pos": 1.0}, 1.0)


def test_unique_questions_and_errors():
    a = B.TypedQuestion("k", "qt", "noul", (_form("pos"),), "g")
    b = B.TypedQuestion("k", "qt", "noul", (_form("pos", options=("Y", "N")),), "g")
    assert list(B.unique_questions([a, a])) == ["k"]
    with pytest.raises(ValueError, match="share the answer key"):
        B.unique_questions([a, b])
    e = B.CacheMiss(["k1", "k2"])
    assert isinstance(e, RuntimeError) and e.keys == ["k1", "k2"]
    r = B.Refused({"q": "too long"})
    assert isinstance(r, ValueError) and r.reasons == {"q": "too long"}


def test_jev_satisfies_the_protocol():
    dm = H.make_dm(cache_only=True)
    assert isinstance(dm, B.DecisionModel) and dm.family == "jev" and dm.layout == "letter"
    assert dm.refused([]) == {}


def test_form_user_equals_routed_and_habit_glue():
    """Form.user() == questions.routed.user_prompt for every routed question and form, and uses the
    habit renderer's glue strings."""
    from habitus.questions import routed as R
    st = H.state(3)
    for qid in ("qv", "qt", "qa"):
        q = R.to_typed("sig0", qid, st)
        assert [f.name for f in q.forms] == R.variants(qid)
        for f, vid in zip(q.forms, R.variants(qid)):
            assert f.user() == R.user_prompt(st, qid, vid)
            assert f.system == R.system_prompt(st, R.SYSTEM_E2)
    from habitus.questions.habit import text as HT
    assert (HT.STATE_HEAD, HT.QUESTION_HEAD, HT.OPTIONS_HEAD, HT.ANSWER_HEAD) == (
        B.STATE_HEAD, B.QUESTION_HEAD, B.OPTIONS_HEAD, B.ANSWER_HEAD)


def _original_kcs1_questions():
    from habitus.config import settings
    src = settings().paths.pidsmaker_src
    if not src:
        H.missing("paths.pidsmaker_src not configured")
    path = os.path.join(src, "pidsmaker", "reflex", "kcs1", "questions.py")
    if not os.path.isfile(path):
        H.missing(f"{path} missing")
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("_orig_kcs1_questions", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)            # json + math only; reads the file, writes nothing
    return mod


@pytest.mark.provenance
def test_form_user_equals_original_kcs1_user_prompt():
    orig = _original_kcs1_questions()
    from habitus.questions import routed as R
    for i in range(4):
        st = H.state(i, kind=("process", "file", "netflow", "process")[i])
        for qid in ("qv", "qt", "qa"):
            for f, vid in zip(R.to_typed(f"s{i}", qid, st).forms, orig.variants(qid)):
                assert f.user() == orig.user_prompt(st, qid, vid)
                assert f.system == orig.system_prompt(st, orig.SYSTEM_E2)
        for qid in ("qv", "qt"):
            for f, vid in zip(R.to_typed(f"s{i}", qid, st, R.SYSTEM).forms, orig.variants(qid)):
                assert f.system == orig.system_prompt(st, orig.SYSTEM)
