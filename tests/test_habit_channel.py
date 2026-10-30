"""The habit channel over ANY DecisionModel (CPU): the decision-model bridge (to_typed / answer_of /
DMAnswerSource, the MQ raw-readout rule), the jev decision model's habit prompts == the training
items' prompts and its readout == the habit engine's (forward seam, no weights), score_level end to
end on the fake host (files, nulls, s_hat, determinism, shat off, test roles reusing the saved null,
clone-arm directories, refused questions n/a, fold checks), habit_vectors, and the whole chain
export -> answers -> surprise -> null -> composite -> bars inside a guarded `decide` process. The
HB0 asset check on the real CADETS_E3 files runs only with HABITUS_HB0_ASSET=1 (reported, not gated)."""

import hashlib
import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from conftest import REPO, missing_asset
from habit_fakes import ATTACK, BENIGN, CLEAN, ID, TEST, FakeAnswerSource, FakeHabitHost, build_day

from habitus.decide.habit import bars as BR
from habitus.decide.habit import channel as C
from habitus.decide.habit import null2 as N2
from habitus.decide.habit import score as SC
from habitus.dm.base import Decision
from habitus.questions.habit import constants as H
from habitus.questions.habit import text as Q
from habitus.questions.habit import types as T
from habitus.state.select import flat

GT_NODES = (ID["payload"], ID["drop_sh"], ID["c2"], ID["fake_python3"], ID["fake_curl"])


def make_host(ds="HBFAKE", art=None):
    days, links = {}, {}
    for d in CLEAN + TEST:
        days[d], links[d] = build_day(d, list(BENIGN) + (list(ATTACK) if d == TEST[1] else []))
    return FakeHabitHost(days, links, ds=ds, art=art or f"/nonexistent/habitus_channel/{ds}/")


def _u(*parts):
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8], 16) / 16 ** 8


def synthetic_routes(host, ref, kplan, h, role, salt="r"):
    """Deterministic E2-like routes (a routes_fn); the attack-day nodes' signatures score high on qv."""
    nd = host.nodes(ref.day)
    hot = {nd[n] for n in GT_NODES if n in nd}
    out = {}
    for s in flat(h):
        qa = _u(salt, s, "a")
        out[s] = {"qv": 0.9 + 0.1 * _u(salt, s, "v") if s in hot else 0.6 * _u(salt, s, "v"),
                  "qt": 0.5 * _u(salt, s, "t"), "qa": None if qa < 0.2 or kplan == "k0" else qa,
                  "hard": _u(salt, s, "h") < 0.1, "cov_min": 1.0}
    return out


@pytest.fixture
def root(tmp_path):
    return str(tmp_path / "habit")


def qwen_tok():
    try:
        from habitus.dm.jev import load_tokenizer
        return load_tokenizer()
    except (OSError, ValueError, ImportError):
        return None


@pytest.fixture(scope="module")
def tok():
    t = qwen_tok()
    if t is None:
        missing_asset("the pinned Qwen3.5-4B tokenizer")
    return t


def a_view(kind=T.ViewKind.V_beh, ntype="process", tag="x"):
    ent = {"type": ntype, "os": "linux", "tag": tag}
    secs = {"entity": ent, "reference": {"days": 2}}
    if kind is T.ViewKind.V_own:
        secs = {"entity": ent, "connection": {"endpoint": "external"}, "reference": {"days": 2}}
    return T.View(kind, ntype, secs, (), {"ident": "i", "tier0": "NONE", "k_mode": "k2"})


def _dec(layout, forms, members, cov=None):
    """A Decision with `members` = {form: [letter-order probs per member]} (canon = fwd identity, rev reversed)."""
    lp = {f: tuple(tuple(m) for m in members[f]) for f in forms}
    cp = {f: tuple(tuple(m) if f == "fwd" else tuple(reversed(m)) for m in members[f]) for f in forms}
    cov = cov or {f: 1.0 for f in forms}
    return Decision("q", layout, tuple(forms), lp, cp, cov, min(cov.values()))


# --------------------------------------------------------------------------- #
# 1. the decision-model bridge
# --------------------------------------------------------------------------- #
def test_answer_of_the_mq_layout_renormalises_even_one_member():
    d = _dec("mq", ("fwd",), {"fwd": [(0.5, 0.3, 0.1, 0.1)]})
    a = T.answer_of(d)
    p = T.pbar([(0.5, 0.3, 0.1, 0.1)])
    assert a.probs == p and a.raw == {"fwd": p} and a.cov_min == 1.0
    d3 = _dec("mq", ("fwd",), {"fwd": [(0.6, 0.4), (0.5, 0.5), (0.9, 0.1)]}, cov={"fwd": 0.3})
    a3 = T.answer_of(d3, ("fwd", "rev"))                                  # orders are ignored: one pass
    assert a3.probs == T.pbar([(0.6, 0.4), (0.5, 0.5), (0.9, 0.1)]) and a3.raw == {"fwd": a3.probs}
    assert a3.cov_min == 1.0


def test_answer_of_letter_and_la_layouts():
    fwd, rev = (0.7, 0.3), (0.2, 0.8)                                     # rev letter order: A = no
    d = _dec("letter", ("fwd", "rev"), {"fwd": [fwd], "rev": [rev]}, cov={"fwd": 0.999, "rev": 0.97})
    a = T.answer_of(d)
    assert a.raw == {"fwd": fwd, "rev": (0.8, 0.2)} and a.cov_min == 0.97
    assert a.probs == T.pbar([fwd, (0.8, 0.2)])                           # the habit engine readout
    one = T.answer_of(d, ("rev",))
    assert one.raw == {"rev": (0.8, 0.2)} and one.probs == T.pbar([(0.8, 0.2)])
    m = {"fwd": [(0.6, 0.4), (0.7, 0.3), (0.9, 0.1)], "rev": [(0.3, 0.7), (0.4, 0.6), (0.2, 0.8)]}
    la = T.answer_of(_dec("la", ("fwd", "rev"), m))
    canon = {"fwd": m["fwd"], "rev": [tuple(reversed(x)) for x in m["rev"]]}
    assert la.raw == {o: T.pbar(canon[o]) for o in T.ORDERS}
    assert la.probs == T.pbar(canon["fwd"] + canon["rev"])                # orders outer, members inner
    with pytest.raises(ValueError):
        T.answer_of(d, ())


class FakeDM:
    """A DecisionModel double: records the typed questions; letter probs from sha1(form user turn)."""
    family, tag = "fake", "fake-dm"

    def __init__(self, layout="letter", refuse=()):
        self.layout, self.refuse, self.seen, self.closed = layout, set(refuse), [], False

    def spec(self):
        return {"engine": self.tag, "layout": self.layout}

    def refused(self, qs):
        return {q.qkey: "over budget" for q in qs if q.qkey in self.refuse}

    def decide(self, qs):
        self.seen.append(list(qs))
        out = {}
        for q in qs:
            forms = q.forms[:1] if self.layout == "mq" else q.forms
            lp, cp = {}, {}
            for f in forms:
                w = np.array([_u(f.user(), i) + 0.01 for i in range(len(f.options))])
                p = tuple(float(x) for x in w / w.sum())
                lp[f.name] = (p,)
                c = [0.0] * len(p)
                for i, k in enumerate(f.canon):
                    c[k] = p[i]
                cp[f.name] = (tuple(c),)
            out[q.qkey] = Decision(q.qkey, self.layout, tuple(lp), lp, cp, {f: 1.0 for f in lp}, 1.0)
        return out

    def close(self):
        self.closed = True


def test_dm_answer_source_contract():
    v = a_view()
    qs = [Q.build(v, "P2", T.Bucket("P2", f"dir:{w}", f"{w} directories")) for w in ("home", "staging")]
    src = T.DMAnswerSource(FakeDM())
    assert isinstance(src, T.AnswerSource) and src.tag == "fake-dm" and src.spec()["layout"] == "letter"
    ans = src.score(qs + [qs[0]])                                         # duplicates answered once
    assert set(ans) == {q.qkey for q in qs} and len(src.dm.seen[-1]) == 2
    for q in qs:
        assert set(ans[q.qkey].raw) == set(T.ORDERS) and abs(sum(ans[q.qkey].probs) - 1) < 1e-12
    src.score(qs, orders=("rev",))
    assert all([f.name for f in tq.forms] == ["rev"] for tq in src.dm.seen[-1])
    other = Q.build(a_view(tag="y"), "P2", T.Bucket("P2", "dir:home", "home directories"))
    clash = T.HabitQuestion(other.family, other.view, other.bucket, other.options, other.option_text, "changed?")
    object.__setattr__(clash, "view", v)                                  # same qkey, other text
    with pytest.raises(ValueError, match="share the answer key"):
        src.score([qs[0], clash])
    mq = T.DMAnswerSource(FakeDM("mq"))
    a = mq.score(qs, orders=("rev",))                                     # MQ: always the canonical form
    assert all([f.name for f in tq.forms] == ["fwd", "rev"] for tq in mq.dm.seen[-1])
    assert all(set(x.raw) == {"fwd"} for x in a.values())
    ref = T.DMAnswerSource(FakeDM(refuse={qs[1].qkey}))
    assert ref.refused(qs) == {qs[1].qkey: "over budget"}
    assert ref.score([]) == {}
    ref.close()
    assert ref.dm.closed


def test_jev_habit_prompts_are_the_item_prompts_and_its_readout_the_engine_readout(tok):
    """JevDecisionModel (forward seam: no weights) through DMAnswerSource: every form's prompt is
    text.render's (the training items' prompt, byte for byte), and the Answer is the habit engine's
    readout: pbar over orders of the canonical T = 1 softmax of the cached letter logits."""
    from dm_helpers import fake_logits

    from habitus.dm.jev import JevDecisionModel, prompt_key, softmax_t
    host = make_host("HBFAKE_jev")
    ref = T.Ref(host.ds, TEST[1], CLEAN[:2], "k2")
    from habitus.questions.habit import head as HD
    from habitus.questions.habit.profile2 import get_profile2
    prof = get_profile2(host, list(ref.ref_days))
    hd = C.habit_day(host, ref, prof, HD.head_sigs(HD.q0(host, ref)), tok)
    qs = hd["questions"]
    assert qs
    dm = JevDecisionModel(device="cpu", forward=fake_logits, ds=None, cache_tag="hb-test", verbose=False)
    src = T.DMAnswerSource(dm)
    ans = src.score(qs)
    keys = dm.keys([T.to_typed(q) for q in qs])
    for q in qs:
        assert keys[q.qkey] == tuple(prompt_key(Q.render(tok, q, o)) for o in T.ORDERS)
        raw = {}
        for o, k in zip(T.ORDERS, keys[q.qkey]):
            rec = dm.cache.get(k)
            raw[o] = T.to_canonical(softmax_t(rec["z"][:len(q.options)], 1.0), o)
        assert ans[q.qkey].raw == raw and ans[q.qkey].probs == T.pbar([raw[o] for o in T.ORDERS])
        assert ans[q.qkey].cov_min == min(float(dm.cache.get(k)["cov"]) for k in keys[q.qkey])
    assert src.refused(qs) == {}


# --------------------------------------------------------------------------- #
# 2. score_level on the fake host
# --------------------------------------------------------------------------- #
def test_parse_kplans_and_fold_for():
    assert C.parse_kplans("0,2,allc-pre,2") == ["k0", "k2", "allc-pre"] == C.parse_kplans(["k0", "k2", "allc-pre"])
    for bad in ("all", "k2+1cal", "allc", ""):
        with pytest.raises(ValueError):
            C.parse_kplans(bad)
    assert C.fold_for("THEIA_E3") == "loho-THEIA_E3" and C.fold_for("CLEARSCOPE_E3") == "all"
    assert C.fold_for("CLEARSCOPE_E3", "loho-optc_h501") == "loho-optc_h501"          # host-seen row
    for ds, fold in (("THEIA_E3", "all"), ("THEIA_E3", "loho-CADETS_E3"), ("optc_h501", "loho-THEIA_E3")):
        with pytest.raises(ValueError, match="trains on"):
            C.fold_for(ds, fold)
    with pytest.raises(ValueError):
        C.fold_for("THEIA_E3", "loho-nowhere")
    with pytest.raises(ValueError, match="trains on"):                    # refused before any model is built
        C.source_for("HB", "all", "THEIA_E3")
    with pytest.raises(ValueError):
        C.source_for("HB-C", "loho-THEIA_E3", "THEIA_E3")                 # no rule comparator arm


def test_score_level_end_to_end_with_the_fake_answer_source(root):
    host = make_host()
    out = C.score_level(host, "HB-Z", "k2", FakeAnswerSource(), root=root, tok=None, routes_fn=synthetic_routes)
    assert out["ldir"] == "HB-Z_k2" and len(out["written"]) == len(CLEAN[:2]) + len(TEST) and out["questions"] > 0
    first = {}
    for d in TEST:
        sd = SC.read(SC.day_path(host.ds, "HB-Z_k2", d, root))
        first[d] = sd.to_json()
        assert sd.meta["level"] == "HB-Z" and sd.meta["role"] == "test" and sd.meta["source"] == "fake"
        assert sd.meta["shat"] is True and sd.meta["spec"] == H.SPEC_VERSION
        n_mem = sum(1 for _n, s in host.nodes(d).items() if s in set(flat(sd.head)))
        assert len(sd.nodes["id"]) == n_mem
        assert all(x is not None for x in sd.nodes["E"]) and any(x is not None for x in sd.nodes["s_hat"])
        for s_, b in zip(sd.nodes["s"], sd.nodes["base"]):
            assert 0.5 * b - 1e-12 <= s_ <= b + 1e-12
        assert {f"z.{f.value}" for f in T.FAMILIES} <= set(sd.nodes)
    for name in ("familynull", "emap", "fitinfo", f"x_{TEST[1]}"):
        assert os.path.exists(N2.null_path(host.ds, f"HB-Z_k2/{name}", root))
    # deterministic: the same inputs give the same files
    C.score_level(host, "HB-Z", "k2", FakeAnswerSource(), root=root, tok=None, routes_fn=synthetic_routes)
    assert {d: SC.read(SC.day_path(host.ds, "HB-Z_k2", d, root)).to_json() for d in TEST} == first
    # decide on the null pool: the weak tier holds <= WEAK_CAP per test day
    nb, _ = SC.load_days(host.ds, "HB-Z_k2", list(CLEAN[:2]), root)
    v, n = SC.null_pool(host, nb)
    tau = BR.bars(v, n)
    tb, _ = SC.load_days(host.ds, "HB-Z_k2", list(TEST), root)
    al = BR.alerts(SC.day_values(host, tb, role="test"), tau, 20)
    assert set(al) == set(TEST) and all(len(a.weak) <= H.WEAK_CAP for a in al.values())
    vec, why = C.habit_vectors(host, "HB-Z_k2", root)
    assert why is None and vec["source"] == "habitus" and vec["habit"][0] == "s_hat"
    assert np.isfinite(vec["s"]).all() and np.array_equal(vec["s"], SC.node_scores(host, tb))
    assert C.habit_vectors(host, "HB.head_k2", root)[0] is None


def test_score_level_without_s_hat_and_refusals(root):
    host = make_host("HBFAKE_noshat")
    out = C.score_level(host, "HB-Z", "k2", FakeAnswerSource(), root=root, tok=None, routes_fn=synthetic_routes,
                        shat=False)
    for d in TEST:
        sd = SC.read(SC.day_path(host.ds, "HB-Z_k2", d, root))
        assert all(x is None for x in sd.nodes["s_hat"]) and sd.meta["shat"] is False
        assert all(abs(b - max(qv, qt)) < 1e-12 for b, qv, qt in zip(sd.nodes["base"], sd.nodes["qv"], sd.nodes["qt"]))
    assert out["refused"] == 0
    with pytest.raises(ValueError):
        C.score_level(host, "HB0", "k2", FakeAnswerSource(), root=root, tok=None)
    with pytest.raises(ValueError):
        C.score_level(host, "HB-C", "k2", FakeAnswerSource(), root=root, tok=None)
    with pytest.raises(RuntimeError, match="fresh routes"):                 # no frozen E2 file for the fake host
        C.score_level(make_host("HBFAKE2"), "HB-Z", "k2", FakeAnswerSource(), root=root, tok=None,
                      e2=os.path.join(root, "no-e2"))


def test_score_level_test_roles_only_reuse_the_saved_null(root):
    host = make_host("HBFAKE_roles")
    kw = dict(root=root, tok=None, routes_fn=synthetic_routes)
    with pytest.raises(FileNotFoundError, match="no saved null fit"):
        C.score_level(host, "HB-Z", "k2", FakeAnswerSource(), roles=("test",), **kw)
    C.score_level(host, "HB-Z", "k2", FakeAnswerSource(), **kw)
    full = {d: SC.read(SC.day_path(host.ds, "HB-Z_k2", d, root)).to_json() for d in TEST}
    fit = open(N2.null_path(host.ds, "HB-Z_k2/familynull", root), "rb").read()
    C.score_level(host, "HB-Z", "k2", FakeAnswerSource(), roles=("test",), **kw)
    again = {d: SC.read(SC.day_path(host.ds, "HB-Z_k2", d, root)).to_json() for d in TEST}
    assert again == full and any(x is not None for d in TEST for x in again[d]["nodes"]["s_hat"])
    assert open(N2.null_path(host.ds, "HB-Z_k2/familynull", root), "rb").read() == fit
    info = N2.load(host.ds, "HB-Z_k2/fitinfo", root)
    N2.save({**info, "null_refs": {CLEAN[0]: []}}, host.ds, "HB-Z_k2/fitinfo", root)
    with pytest.raises(ValueError, match="not this plan"):
        C.score_level(host, "HB-Z", "k2", FakeAnswerSource(), roles=("test",), **kw)


def test_score_level_writes_clone_arm_directories(root):
    host = make_host("HBFAKE_arms")
    out = C.score_level(host, "HB-L.mq", "k2", FakeAnswerSource(), root=root, tok=None, routes_fn=synthetic_routes)
    assert out["ldir"] == "HB-L.mq_k2"
    sd = SC.read(SC.day_path(host.ds, "HB-L.mq_k2", TEST[1], root))
    assert sd.meta["level"] == "HB-L" and sd.meta["arm"] == "mq" and sd.meta["clone_arm"] == "HB-L.mq"
    assert SC.ldir_of(sd.meta) == "HB-L.mq_k2"
    out = C.score_level(host, "HB.head", "allc-pre", FakeAnswerSource(), root=root, tok=None,
                        routes_fn=synthetic_routes)
    assert out["ldir"] == "HB.head_allc-pre" and len(out["written"]) == len(CLEAN) + len(TEST)
    assert os.path.exists(N2.null_path(host.ds, "HB-L.mq_k2/familynull", root))
    assert C.parse_ldir("HB-L.mq_k2") == ("HB-L", "mq", "k2") and C.parse_ldir("HB0_allc-pre") == ("HB0", None,
                                                                                                    "allc-pre")
    with pytest.raises(ValueError):
        C.parse_ldir("HB-C_k2")
    k0 = C.score_level(host, "HB-Z", "k0", FakeAnswerSource(), root=root, tok=None, routes_fn=synthetic_routes)
    sd0 = SC.read(SC.day_path(host.ds, "HB-Z_k0", TEST[1], root))
    assert k0["ldir"] == "HB-Z_k0" and sd0.nodes["s"] == sd0.nodes["base"]            # no QA at k0


class RefusingSource(FakeAnswerSource):
    """A source with a length budget: declines every question whose sha1(qkey) is 0 mod 3 (refused()
    lists them, score() must never be handed one)."""

    tag = "fake-refusing"

    @staticmethod
    def _no(k):
        return int(hashlib.sha1(k.encode()).hexdigest()[:4], 16) % 3 == 0

    def refused(self, questions):
        return {q.qkey: "head of 300 tokens > head_max_len 256" for q in questions if self._no(q.qkey)}

    def score(self, questions, orders=T.ORDERS):
        questions = list(questions)
        assert not any(self._no(q.qkey) for q in questions), "a refused question reached score()"
        return super().score(questions)


def test_score_level_makes_refused_questions_na_instead_of_failing_the_day(root):
    host = make_host("HBFAKE_refuse")
    out = C.score_level(host, "HB-L", "k2", RefusingSource(), root=root, tok=None, routes_fn=synthetic_routes)
    assert out["ldir"] == "HB-L_k2" and out["written"]
    metas = [SC.read(SC.day_path(host.ds, "HB-L_k2", d, root)).meta for d in out["days"]]
    assert sum(int(m.get("refused_questions") or 0) for m in metas) > 0


def test_score_level_with_a_decision_model_source(root):
    """DMAnswerSource over a DecisionModel double: the same pipeline, source tag and spec in the meta."""
    host = make_host("HBFAKE_dm")
    src = T.DMAnswerSource(FakeDM())
    out = C.score_level(host, "HB-Z", "k2", src, root=root, tok=None, routes_fn=synthetic_routes)
    sd = SC.read(SC.day_path(host.ds, out["ldir"], TEST[1], root))
    assert sd.meta["source"] == "fake-dm" and sd.meta["source_spec"] == {"engine": "fake-dm", "layout": "letter"}
    assert any(x is not None for x in sd.nodes["s_hat"])


# --------------------------------------------------------------------------- #
# 3. the whole chain inside a guarded `decide` process
# --------------------------------------------------------------------------- #
def test_export_answers_surprise_null_composite_bars_under_the_guard(tmp_path):
    code = textwrap.dedent("""
        import json, sys
        from habitus import guard
        guard.install_guard("decide")
        sys.path.insert(0, "tests")
        from test_habit_channel import make_host, synthetic_routes, qwen_tok
        from test_habit_items import FakeTok
        from habit_fakes import CLEAN, TEST, FakeAnswerSource
        from habitus.decide.habit import bars as BR, channel as C, score as SC
        from habitus.questions.habit import constants as H, types as T
        from habitus.train import items as X
        root = sys.argv[1]
        host = make_host("THEIA_E3")              # a training host of loho-CADETS_E3 (fake days)
        man = X.export_fold("loho-CADETS_E3", tok=FakeTok(), hosts=["THEIA_E3"], host_factory=lambda ds: host,
                            root=root, lengths=False, log=lambda *a: None)
        res = {"items": sum(v["n"] for v in man["files"].values())}
        sm = X.sample_fold("loho-CADETS_E3", n=50, root=root)
        res["sample"] = sum(v["n"] for v in sm["file"].values())
        tok = qwen_tok()
        sources = {"HB-Z": FakeAnswerSource()}
        if tok is not None:
            from dm_helpers import fake_logits
            from habitus.dm.jev import JevDecisionModel
            sources["HB"] = T.DMAnswerSource(JevDecisionModel(forward=fake_logits, ds=None, cache_tag="hb-guard",
                                                              verbose=False))
        for level, src in sources.items():
            out = C.score_level(host, level, "k2", src, root=root, tok=tok, routes_fn=synthetic_routes)
            nb, _ = SC.load_days(host.ds, out["ldir"], list(CLEAN[:2]), root)
            tb, _ = SC.load_days(host.ds, out["ldir"], list(TEST), root)
            v, n = SC.null_pool(host, nb)
            rows = BR.alert_rows(SC.day_values(host, tb, role="test"), BR.bars(v, n))
            vec, why = C.habit_vectors(host, out["ldir"], root)
            res[level] = {"files": len(out["written"]), "questions": out["questions"],
                          "weak_max": max(rows["budgets"][20]["weak"].values()), "habit": vec["habit"][0],
                          "s_hat": sum(x is not None for d in TEST for x in tb[d].nodes["s_hat"])}
        res["guard"] = guard.guard_active()
        res["eval_loaded"] = [m for m in sys.modules if m.startswith("habitus.eval")]
        res["jev"] = tok is not None
        print("RESULT " + json.dumps(res))
    """)
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "src"))
    r = subprocess.run([sys.executable, "-c", code, str(tmp_path / "habit")], env=env, cwd=REPO, capture_output=True,
                       text=True, timeout=900)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    assert r.returncode == 0 and line, r.stdout[-2000:] + r.stderr[-4000:]
    res = json.loads(line[-1][len("RESULT "):])
    assert res["guard"] == "decide" and res["eval_loaded"] == [] and res["items"] > 0 and res["sample"] > 0
    for level in ("HB-Z",) + (("HB",) if res["jev"] else ()):
        got = res[level]
        assert got["files"] == len(CLEAN[:2]) + len(TEST) and got["questions"] > 0 and got["s_hat"] > 0
        assert got["weak_max"] <= H.WEAK_CAP and got["habit"] == "s_hat"


# --------------------------------------------------------------------------- #
# 4. HB0 on the real CADETS_E3 files (asset; reported, not gated)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(os.environ.get("HABITUS_HB0_ASSET") != "1", reason="set HABITUS_HB0_ASSET=1 (minutes of CPU)")
def test_hb0_on_the_real_cadets_files(tmp_path, monkeypatch, record_property):
    """HB0 k2 on CADETS_E3 from the real cards + routed E2 files, then the oracle best-MCC (habitus.eval;
    a test may read labels, a guarded command never). Expected from results/extension_habit/hb0_rows.csv:
    MCC .392183 (TP 12 / FP 1). Reported (record_property + a warning on a mismatch), not gated."""
    import csv
    import warnings

    from test_habit_reference import real_cards_dir
    from test_habit_score import real_e2_root

    from habitus import config
    from habitus.data.host import Host
    from habitus.decide.habit import retrofit
    cards, e2 = real_cards_dir("CADETS_E3"), real_e2_root()
    if cards is None or e2 is None:
        missing_asset("CADETS_E3 cards and routed E2 score files")
    host = Host("CADETS_E3", cards_dir=cards).keep_clean_days()
    root = str(tmp_path / "habit")
    out = retrofit.run("CADETS_E3", kplans=("k2", "allc-pre"), root=root, e2=e2, host=host)
    assert len(out) == len(retrofit.plan(host, "k2"))                      # allc-pre refused on CADETS_E3
    vec, why = C.habit_vectors(host, "HB0_k2", root)
    assert why is None and np.isfinite(vec["s"]).all()
    s = config.settings()
    if not os.path.isdir(s.paths.ground_truth) and s.paths.upstream_gt:
        monkeypatch.setenv("HABITUS_GT", str(s.paths.upstream_gt))
        config.reset_settings()
        monkeypatch.setattr(config, "_CACHE", {})
    from habitus.eval import metrics
    row = metrics.confusion(host, dict(zip(host.universe.tolist(), vec["s"].tolist())))
    config.reset_settings()
    want = {r["k_plan"]: r for r in csv.DictReader(open(os.path.join(REPO, "results", "extension_habit",
                                                                      "hb0_rows.csv"))) if r["ds"] == "CADETS_E3"}
    got = {"tp": row["tp"], "fp": row["fp"], "mcc": round(row["mcc"], 6)}
    record_property("hb0_cadets_k2", got)
    print("HB0 CADETS_E3 k2:", got, "frozen:", {k: want["k2"][k] for k in ("tp", "fp", "mcc_recomputed")})
    if (got["tp"], got["fp"], got["mcc"]) != (int(want["k2"]["tp"]), int(want["k2"]["fp"]),
                                              float(want["k2"]["mcc_recomputed"])):
        warnings.warn(f"HB0 CADETS_E3 k2 recompute {got} != frozen {want['k2']}")


# --------------------------------------------------------------------------- #
# 5. provenance: the ORIGINAL habit pipeline on its own fake host == this one (HABITUS_PROVENANCE=1)
# --------------------------------------------------------------------------- #
ORIGINAL_SIDE = r'''
import json, os, sys, hashlib
out_dir, src = sys.argv[1], sys.argv[2]
def audit(event, args):                      # write audit: nothing outside out_dir (the source tree is read only)
    if event == "open" and len(args) > 1 and isinstance(args[0], (str, bytes, os.PathLike)):
        mode = args[1]
        w = any(c in mode for c in "wax+") if isinstance(mode, str) else bool(
            isinstance(mode, int) and mode & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
        p = os.path.abspath(os.fsdecode(args[0]))
        if w and not (p.startswith(out_dir) or p.startswith("/dev/")):
            raise PermissionError(f"write audit: {p}")
    elif event in ("os.mkdir", "os.rename", "os.replace", "os.remove"):
        p = os.path.abspath(os.fsdecode(args[0]))
        if not p.startswith(out_dir):
            raise PermissionError(f"write audit: {event} {p}")
sys.addaudithook(audit)
os.environ["HABITUS_ROOT"] = os.path.join(out_dir, "hb")
os.environ["HABITUS_ALLOW_ROOTFS"] = "1"
sys.path[:0] = [src, os.path.join(src, "tests")]
from habitus_fakes import make_habit_host, FakeAnswerSource
from pidsmaker.reflex.habitus import types as T, questions as Q, head as HD, run as R, surprise as SU, null2 as N2
from pidsmaker.reflex.habitus import attest as A, profile2
from pidsmaker.reflex.kcs1 import audit as KA
from pidsmaker.reflex.kcs1.select import TYPES
tok = KA.qwen_tokenizer()
def _u(key, i=0):
    h = hashlib.sha1(f"{key}#{i}".encode()).digest()
    return (int.from_bytes(h[:8], "big") + 1) / (2 ** 64 + 2)
def routes_fn(host, ref, kplan, h, role):
    return {s: {"qv": round(_u(s, 1), 6), "qt": round(_u(s, 2), 6), "qa": None if kplan == "k0" else round(_u(s, 3), 6),
                "hard": _u(s, 4) < 0.1, "cov_min": 0.99} for t in TYPES for s in h.get(t, [])}
host = make_habit_host()
dump = {}
for kplan in ("k0", "k2", "allc-pre"):
    refs = [(r, "null") for r in N2.plan(host, kplan)] + [(N2.ref_for(host, d, kplan), "test") for d in host.test]
    for ref, role in refs:
        prof = profile2.get_profile2(host, list(ref.ref_days)) if ref.k else None
        hd = R.habit_day(host, ref, prof, HD.head_sigs(HD.q0(host, ref)), tok)
        ans = FakeAnswerSource().score(hd["questions"])
        dump[f"{kplan}|{ref.day}|{role}"] = {
            "views": {s: {k.value: v.text for k, v in vs.items()} for s, vs in hd["views"].items()},
            "own": {s: {b: v.text for b, v in vs.items()} for s, vs in hd["own"].items()},
            "prompts": {q.qkey: [Q.render(tok, q, o) for o in T.ORDERS] for q in hd["questions"]},
            "x": {str(n): {f.value: v for f, v in SU.node_x(o, hd["views"].get(o.sig, {}), hd["own"].get(o.sig, {}),
                                                           ans).items()} for n, o in sorted(hd["obs"].items())},
            "tiers": {str(n): int(t) for n, t in A.lineage_tiers(host, ref.day, prof).items()}}
    R.score_level(host, "HB-Z", kplan, source_factory=lambda ref, prof: FakeAnswerSource(), routes_fn=routes_fn, tok=tok)
json.dump(dump, open(os.path.join(out_dir, "dump.json"), "w"), sort_keys=True)
print("ORIGINAL OK")
'''


@pytest.mark.provenance
def test_the_original_habit_pipeline_gives_the_same_views_prompts_statistics_and_files(tmp_path, tok):
    """The original habit channel (paths.pidsmaker_src, read-only under a write audit) and this one, on
    their fake hosts (identical cards and links): views, rendered prompts (real Qwen chat template), x
    per node, lineage tiers, and every HB-Z score / null file of k0, k2 and allc-pre are identical."""
    import glob
    import gzip

    from habitus.config import settings
    from habitus.questions.habit import attest as A
    from habitus.questions.habit import head as HD
    from habitus.questions.habit.profile2 import get_profile2
    s = settings()
    if not s.paths.pidsmaker_src:
        missing_asset("paths.pidsmaker_src (configs/local.yaml)")
    out = str(tmp_path / "orig")
    os.makedirs(out)
    env = dict(os.environ, HF_HOME=str(s.paths.models), PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    r = subprocess.run([sys.executable, "-B", "-c", ORIGINAL_SIDE, out, str(s.paths.pidsmaker_src)], env=env,
                       cwd=out, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0 and "ORIGINAL OK" in r.stdout, r.stdout[-2000:] + r.stderr[-4000:]
    orig = json.load(open(os.path.join(out, "dump.json")))

    def routes_fn(host, ref, kplan, h, role):
        def u(key, i):
            hh = hashlib.sha1(f"{key}#{i}".encode()).digest()
            return (int.from_bytes(hh[:8], "big") + 1) / (2 ** 64 + 2)
        return {sg: {"qv": round(u(sg, 1), 6), "qt": round(u(sg, 2), 6),
                     "qa": None if kplan == "k0" else round(u(sg, 3), 6), "hard": u(sg, 4) < 0.1, "cov_min": 0.99}
                for sg in flat(h)}

    from habitus.decide.habit import surprise as SU
    host = make_host()
    new_root = str(tmp_path / "new")
    for kplan in ("k0", "k2", "allc-pre"):
        for ref, role in N2.plan_refs(host, kplan):
            prof = get_profile2(host, list(ref.ref_days)) if ref.k else None
            hd = C.habit_day(host, ref, prof, HD.head_sigs(HD.q0(host, ref)), tok)
            ans = FakeAnswerSource().score(hd["questions"])
            o = orig[f"{kplan}|{ref.day}|{role}"]
            assert o["views"] == {sg: {k.value: v.text for k, v in vs.items()} for sg, vs in hd["views"].items()}
            assert o["own"] == {sg: {b: v.text for b, v in vs.items()} for sg, vs in hd["own"].items()}
            assert o["prompts"] == {q.qkey: [Q.render(tok, q, x) for x in T.ORDERS] for q in hd["questions"]}
            assert o["x"] == {str(n): {f.value: v for f, v in SU.node_x(ob, hd["views"].get(ob.sig, {}),
                                                                         hd["own"].get(ob.sig, {}), ans).items()}
                              for n, ob in sorted(hd["obs"].items())}
            assert o["tiers"] == {str(n): int(t) for n, t in A.lineage_tiers(host, ref.day, prof).items()}
        C.score_level(host, "HB-Z", kplan, FakeAnswerSource(), root=new_root, tok=tok, routes_fn=routes_fn)
    ob = os.path.join(out, "hb")
    fo = sorted(os.path.relpath(p, ob) for p in glob.glob(ob + "/*/**/*.json.gz", recursive=True))
    fn = sorted(os.path.relpath(p, new_root) for p in glob.glob(new_root + "/*/**/*.json.gz", recursive=True))
    assert fo == fn and fo
    for rel in fo:
        a = json.load(gzip.open(os.path.join(ob, rel), "rt"))
        b = json.load(gzip.open(os.path.join(new_root, rel), "rt"))
        assert a == b, rel
