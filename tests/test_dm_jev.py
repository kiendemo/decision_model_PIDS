"""JevDecisionModel: prompts and keys, the letter readout from cache records, cache-only mode before
any model import, the forward seam, OOM / non-finite rules, prefix sharing on a tiny random Qwen3.5,
the routed keys, and the GPU guard."""

import hashlib
import json
import sys
import types

import numpy as np
import pytest

import dm_helpers as H
from habitus.dm import base as B
from habitus.dm import gpu as G
from habitus.dm import jev as J
from habitus.dm.cache import AnswerCache


@pytest.fixture(scope="module")
def tok():
    return H.qwen_tokenizer()


@pytest.fixture(scope="module")
def R():
    from habitus.questions import routed
    return routed


def _e1(R, n=4, extra=0):
    """E1-style questions (SYSTEM; qv + qt about one state) for n synthetic states."""
    qs = []
    for i in range(n):
        st = H.state(i, extra=extra * i)
        qs += [R.to_typed(f"s{i}", q, st, R.SYSTEM) for q in ("qv", "qt")]
    return qs


def _e2(R, n=3, k0=2):
    """E2 questions: n signatures with qv / qt / qa states, k0 more with qv / qt only."""
    qs, with_qa, without = [], [], []
    for i in range(n + k0):
        sig = f"sig{i}"
        sts = {"qv": H.state(i, extra=1), "qt": H.state(i, extra=2)}
        if i < n:
            sts["qa"] = H.state(i, kind="process", extra=3)
            with_qa.append(sig)
        else:
            without.append(sig)
        qs += H.routed_questions(sig, sts)
    return qs, with_qa, without


# --------------------------------------------------------------------------- #
# tags and adapters
# --------------------------------------------------------------------------- #
def test_engine_and_habit_tags(tmp_path):
    assert J.engine_tag("Qwen/Qwen3.5-4B") == "qwen3.5-4b"
    assert J.engine_tag("Qwen/Qwen3.5-0.8B", str(tmp_path / "runs" / "HB" / "adapter")) == "qwen3.5-0.8b+HB"
    assert J.engine_tag("Qwen/Qwen3.5-4B", "/x/my run") == "qwen3.5-4b+my_run"
    assert J.habit_tag() == "hb-qwen3.5-4b"
    ad = tmp_path / "HB" / "adapter"
    ad.mkdir(parents=True)
    (ad / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-4B"}))
    (ad / "adapter_model.safetensors").write_bytes(b"weights")
    (ad / "README.md").write_text("ignored")
    h = J.adapter_hash(str(ad))
    want = hashlib.sha256()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        want.update(f"{name}\x00{hashlib.sha256((ad / name).read_bytes()).hexdigest()}\n".encode())
    assert h == want.hexdigest()
    assert J.habit_tag("Qwen/Qwen3.5-4B", str(ad)) == f"hb-qwen3.5-4b+HB-{h[:12]}"
    assert J.resolve_adapter(str(tmp_path / "HB")) == str(ad)
    (tmp_path / "HB" / "habitus.json").write_text(json.dumps({"backbone": "Qwen/Qwen3.5-0.8B"}))
    with pytest.raises(ValueError, match="trained on"):
        J.resolve_adapter(str(tmp_path / "HB"))
    (tmp_path / "HB" / "habitus.json").write_text(json.dumps({"backbone": "Qwen/Qwen3.5-4B", "variant": "head"}))
    with pytest.raises(ValueError, match="variant 'head'"):
        H.make_dm(adapter=str(tmp_path / "HB"), cache_only=True)
    with pytest.raises(FileNotFoundError):
        J.resolve_adapter(str(tmp_path))


# --------------------------------------------------------------------------- #
# prompts, letters, keys
# --------------------------------------------------------------------------- #
def test_prompt_suffix_and_letter_tokens(tok, R):
    ids = J.letter_ids(tok)
    assert len(set(ids)) == 4 and all(tok.convert_ids_to_tokens(i) == L for i, L in zip(ids, B.LETTERS))
    dm = H.make_dm(tok, cache_only=True)
    st = H.state(2)
    for qid in ("qv", "qt", "qa"):
        for system in (R.SYSTEM, R.SYSTEM_E2):
            for f in R.to_typed("s", qid, st, system).forms:
                text = dm.render(f)
                assert text.endswith("</think>\n\n") and st in text and "<think>\n\n</think>\n\n" in text
                base = tok(text, add_special_tokens=False)["input_ids"]
                for L, i in zip(f.letters, ids):
                    assert tok(text + L, add_special_tokens=False)["input_ids"] == base + [i]


def test_render_is_the_chat_template_and_hostile_text_stays_data(tok, R):
    dm = H.make_dm(tok, cache_only=True)
    f = B.Form("fwd", "SYS " + H.HOSTILE, '{"x":"' + H.HOSTILE + '"}', "Q?", ("a", "b"), (0, 1))
    want = tok.apply_chat_template([{"role": "system", "content": f.system}, {"role": "user", "content": f.user()}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert dm.render(f) == want and dm.key(f) == hashlib.sha1(want.encode("utf-8")).hexdigest()
    st = H.state(1)
    for qid in ("qv", "qt"):
        for f, vid in zip(R.to_typed("s", qid, st, R.SYSTEM_E2).forms, R.variants(qid)):
            msgs = R.messages(st, qid, vid, R.SYSTEM_E2)
            assert dm.render(f) == tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                           enable_thinking=False)
    e1 = R.to_typed("s", "qv", st, R.SYSTEM).forms[0]
    assert dm.render(e1) != dm.render(R.to_typed("s", "qv", st, R.SYSTEM_E2).forms[0])


def test_pinned_tokenizer_dir_renders_like_the_snapshot(tok, R, tmp_path):
    """The release layout <paths.tokenizers>/Qwen3.5-4B@851bf6e8 (exactly the 6 tokenizer files) is
    preferred over the HF snapshot and renders and tokenizes identically."""
    import shutil
    from habitus.config import load_settings
    src = J.tokenizer_source()
    d = tmp_path / "tokenizer" / "Qwen3.5-4B@851bf6e8"
    d.mkdir(parents=True)
    for name in J.TOKENIZER_FILES:
        shutil.copy(src / name, d / name)
    s = load_settings(overrides={"paths": {"tokenizers": str(tmp_path / "tokenizer")}})
    assert J.tokenizer_dir(settings=s) == d and J.tokenizer_source(settings=s) == d
    t2 = J.load_tokenizer(settings=s)
    assert t2 is not tok and t2 is J.load_tokenizer(settings=s)               # one load per directory
    dm1, dm2 = H.make_dm(tok, cache_only=True), H.make_dm(t2, cache_only=True)
    for q in _e2(R, 2, 1)[0]:
        for f in q.forms:
            assert dm1.render(f) == dm2.render(f) and dm1.key(f) == dm2.key(f)
            assert tok(dm1.render(f), add_special_tokens=False)["input_ids"] == \
                t2(dm2.render(f), add_special_tokens=False)["input_ids"]
    assert J.letter_ids(t2) == J.letter_ids(tok)
    with pytest.raises(FileNotFoundError):
        J.tokenizer_source("Qwen/Qwen3.5-4B", "0" * 40, settings=s)


def test_keys_need_no_cache_and_no_model(tok, R):
    class NoCache:
        def get(self, key):
            raise AssertionError("keys() must not read the cache")

    def no_forward(seqs):
        raise AssertionError("keys() must not forward")

    dm = H.make_dm(tok, cache=NoCache(), forward=no_forward)
    qs, _, _ = _e2(R)
    keys = dm.keys(qs + qs[:2])
    assert list(keys) == [q.qkey for q in qs]
    for q in qs:
        assert keys[q.qkey] == tuple(hashlib.sha1(dm.render(f).encode("utf-8")).hexdigest() for f in q.forms)
    bad = B.TypedQuestion(qs[0].qkey, "qv", "score", qs[1].forms, "g")
    with pytest.raises(ValueError, match="share the answer key"):
        dm.keys([qs[0], bad])


# --------------------------------------------------------------------------- #
# readout through the forward seam and the cache
# --------------------------------------------------------------------------- #
def test_forward_seam_records_then_reads_back(tok, R, tmp_path):
    calls = []

    def fwd(seqs):
        calls.append(len(seqs))
        return H.fake_logits(seqs)

    over = tmp_path / "answers" / "X.jsonl"
    cache = AnswerCache("test", "X", overlay=over)
    dm = H.make_dm(tok, cache=cache, forward=fwd)
    qs = _e1(R, 3)
    out = dm.decide(qs)
    assert set(out) == {q.qkey for q in qs} and sum(calls) == 12
    recs = {r["k"]: r for r in map(json.loads, over.read_text(encoding="utf-8").splitlines())}
    assert len(recs) == 12
    for q in qs:
        d = out[q.qkey]
        assert d.layout == "letter" and d.forms == tuple(f.name for f in q.forms)
        for f in q.forms:
            r = recs[dm.key(f)]
            assert set(r) == {"k", "z", "lse", "n", "cov"} and r["n"] == len(f.options) and len(r["z"]) == 4
            n = len(f.options)
            ids = tok(dm.render(f), add_special_tokens=False)["input_ids"]
            z = [float(len(ids) % 7), 1.0, 0.5, -1.0]
            assert r["z"] == [round(x, 4) for x in z] and r["lse"] == 9.0
            assert r["cov"] == round(float(np.exp(np.asarray(z[:n]) - 9.0).sum()), 6) == d.cov[f.name]
            p = tuple(float(x) for x in J.softmax_t(r["z"][:n], 1.0))
            assert d.letter_probs[f.name] == (p,) and d.canon_probs[f.name] == (B.canonical(p, f.canon),)
        assert d.cov_min == min(d.cov.values())
    rec = R.record({q.family: out[q.qkey] for q in qs[:2]}, level="E1")
    assert set(rec) == {"qv", "qt", "cov_min"}
    dm.close()
    again = H.make_dm(tok, cache=AnswerCache("test", "X", overlay=over, read_only=True), cache_only=True)
    out2 = again.decide(qs)
    assert sum(calls) == 12 and all(out2[k] == out[k] for k in out)              # cached: bit-identical


def test_e1_fake_engine_values(tok, R):
    """The fake-engine readout the E1 snapshot freezes: qv / qt / cov_min of 4 states."""
    dm = H.make_dm(tok, forward=H.fake_logits)
    qs = _e1(R, 4, extra=1)
    out = dm.decide(qs)
    for i in range(4):
        rec = R.record({"qv": out[f"s{i}|qv"], "qt": out[f"s{i}|qt"]}, level="E1")
        man = {}
        for q in (q for q in qs if q.qkey.startswith(f"s{i}|")):
            pv = {}
            for f in q.forms:
                n = len(tok(dm.render(f), add_special_tokens=False)["input_ids"])
                z = np.array([n % 7, 1.0, 0.5, -1.0])[:len(f.options)]
                pv[f.name] = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
            man[q.family] = R.combine(q.family, pv)
        assert abs(rec["qv"] - man["qv"]) < 1e-12 and abs(rec["qt"] - man["qt"]) < 1e-12


def test_habit_forms_canonical_order(tok):
    dm = H.make_dm(tok, forward=H.fake_logits)
    q = H.habit_question("v1|P1|-", '{"entity":{"type":"process"}}', "Who is the parent?",
                         ("usual", "attested other", "never seen", "none"))
    d = dm.decide([q])["v1|P1|-"]
    assert d.forms == ("fwd", "rev")
    assert d.canon_probs["fwd"] == d.letter_probs["fwd"]
    assert d.canon_probs["rev"][0] == tuple(reversed(d.letter_probs["rev"][0]))
    assert all(abs(sum(p[0]) - 1) < 1e-12 for p in d.letter_probs.values())


def test_cache_only_raises_with_every_missing_key(tok, R):
    def no_forward(seqs):
        raise AssertionError("cache-only must not forward")

    qs = _e1(R, 2)
    cache = AnswerCache("test", None)
    dm = H.make_dm(tok, cache=cache, forward=H.fake_logits)
    dm.decide(qs[:1])
    ro = H.make_dm(tok, cache=cache, cache_only=True, forward=no_forward)
    assert set(ro.decide(qs[:1])) == {qs[0].qkey}
    with pytest.raises(B.CacheMiss) as e:
        ro.decide(qs)
    want = [k for q in qs[1:] for k in ro.keys([q])[q.qkey]]
    assert e.value.keys == want and len(want) == 6
    assert ro.model is None


def test_cache_only_never_imports_model_code_or_torch(tok, R, tmp_path):
    """A fresh interpreter with torch blocked (as in the torch-free [replay] install): a cached
    question is answered, a missing one raises CacheMiss, and no torch / peft / modeling code loads.
    The rendered prompt (hence every key) is the same with and without torch."""
    q = _e1(R, 1)[0]
    dm = H.make_dm(tok, forward=H.fake_logits)
    want = dm.decide([q])[q.qkey]
    key = dm.key(q.forms[0])
    base = tmp_path / "X.jsonl"
    base.write_text("".join(json.dumps({"k": k, **dm.cache.get(k)}) + "\n" for k in dm.keys([q])[q.qkey]))
    code = r'''
import json, sys
BLOCK = sys.argv[2] == "1"
if BLOCK:
    sys.modules["torch"] = None
from habitus.dm.base import CacheMiss
from habitus.dm.cache import AnswerCache
from habitus.dm.jev import JevDecisionModel
from habitus.questions import routed as R
sys.path.insert(0, "tests")
import dm_helpers as H
q = R.to_typed("s0", "qv", H.state(0), R.SYSTEM)
miss = R.to_typed("s9", "qv", H.state(9), R.SYSTEM)
dm = JevDecisionModel(cache_only=True, cache=AnswerCache("t", "X", base=sys.argv[1], read_only=True), verbose=False)
d = dm.decide([q])[q.qkey]
try:
    dm.decide([miss])
    raised = False
except CacheMiss as e:
    raised = len(e.keys) == 2
mods = [m for m, v in sys.modules.items() if v is not None and (m == "torch" or m.startswith(("torch.", "peft")) or
        "modeling_qwen" in m or m.endswith(".modeling_utils"))]
print(json.dumps({"raised": raised, "mods": mods, "key": dm.key(q.forms[0]), "p": d.letter_probs["fwd"][0],
                  "cov": d.cov}))
'''
    for block in ("1", "0"):
        r = H.run_python(code, base, block)
        assert r.returncode == 0, r.stderr[-3000:]
        res = json.loads(r.stdout.strip().splitlines()[-1])
        assert res["raised"] and res["key"] == key, block
        assert tuple(res["p"]) == want.letter_probs["fwd"][0] and res["cov"] == dict(want.cov)
        if block == "1":
            assert res["mods"] == [], res["mods"]
        else:
            assert not [m for m in res["mods"] if "modeling_qwen" in m or m.startswith("peft")
                        or m.endswith(".modeling_utils")], res["mods"]


# --------------------------------------------------------------------------- #
# engine bookkeeping (forward patched): OOM halving, non-finite answers
# --------------------------------------------------------------------------- #
def test_oom_halving_and_non_finite_rule(tok, R, tmp_path, monkeypatch):
    import torch
    over = tmp_path / "answers" / "X.jsonl"
    dm = H.make_dm(tok, cache=AnswerCache("test", "X", overlay=over), model=object())
    calls = []

    def oom_above_2(seqs):
        calls.append(len(seqs))
        assert sys.exc_info()[0] is None, "retry ran inside the OOM handler"
        if len(seqs) > 2:
            raise torch.cuda.OutOfMemoryError("fake OOM")
        return H.fake_logits(seqs)

    monkeypatch.setattr(dm, "_forward", oom_above_2)
    qs = _e1(R, 4)
    out = dm.decide(qs[:6])
    assert calls[0] == 12 and max(calls[1:]) <= 6 and sum(c for c in calls if c <= 2) == 12
    assert len(out) == 6
    n_ok = len(dm.cache)

    def nan_first_row(seqs):
        z, lse = H.fake_logits(seqs)
        z[0, 0] = np.nan
        return z, lse

    monkeypatch.setattr(dm, "_forward", nan_first_row)
    with pytest.raises(RuntimeError, match="non-finite"):
        dm.decide(qs[6:])
    assert dm.stats["non_finite"] == 2 and len(dm.cache) == n_ok + 3               # 3 good prompts kept
    lines = [json.loads(x) for x in over.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == n_ok + 3 and all(np.isfinite(r["z"]).all() for r in lines)
    dm.close()


# --------------------------------------------------------------------------- #
# prefix sharing on a tiny random Qwen3.5 (fp32, CPU)
# --------------------------------------------------------------------------- #
def test_split_prefix_and_share_units():
    a, b, c = [1, 2, 3, 4, 5, 6], [1, 2, 3, 9, 9], [1, 2, 3, 4, 7, 8, 9]
    assert J.split_prefix([a, b, c]) == 3 and J.split_prefix([a, c]) == 4
    assert J.split_prefix([a]) == 5 and J.split_prefix([a, a]) == 5       # capped: >= 1 suffix token
    assert J.split_prefix([[5, 1], [6, 1]]) == 0
    seqs = {"a": a, "b": b, "c": c, "d": [1, 2]}
    units = J.share_units(seqs, [["a", "b", "c"], ["d"]])
    assert [u[0] for u in units] == [["a", "b", "c"], ["d"]]
    for keys, pre, sufs in units:
        assert all(pre + s == seqs[k] and len(s) >= 1 for k, s in zip(keys, sufs))
    assert units[0][1] == [1, 2, 3] and units[1][1] == [1]


@pytest.fixture(scope="module")
def tiny():
    mp = pytest.MonkeyPatch()
    try:
        yield H.tiny_qwen(mp)
    finally:
        mp.undo()


def test_prefix_share_matches_full_forward_tiny_model(tok, R, tiny):
    dm = H.make_dm(tok, model=tiny, batch_size=8, prefix_share=True)
    qs = _e1(R, 5, extra=2)                                 # distinct lengths: padded prefixes
    seqs, groups = {}, []
    for i in range(5):
        g = []
        for q in qs[2 * i:2 * i + 2]:
            for f in q.forms:
                k = (i, q.family, f.name)
                seqs[k] = tok(dm.render(f), add_special_tokens=False)["input_ids"]
                g.append(k)
        groups.append(g)
    units = J.share_units(seqs, groups)
    for keys, pre, sufs in units:          # the state and system prompt are shared, the questions are not
        assert all(pre + x == seqs[k] for k, x in zip(keys, sufs))
        assert len(pre) > max(map(len, sufs)) and H.state(keys[0][0], extra=2 * keys[0][0]) in tok.decode(pre)
    assert len({len(u[1]) for u in units}) >= 3

    def full(us):
        return dm._forward([u[1] + x for u in us for x in u[2]])

    singles = J.share_units(seqs, [[k] for k in list(seqs)[::3]])     # 1-token suffixes: the decode path
    qv_pair = J.share_units(seqs, [[(0, "qv", "fwd"), (0, "qv", "rev")]])  # no padding at all
    assert all(len(u[2][0]) == 1 for u in singles) and len(qv_pair[0][1]) > len(units[0][1])
    for us in (units, units[:1], singles, qv_pair):
        (zs, ls), (zf, lf) = dm._forward_shared(us), full(us)
        assert zs.shape == zf.shape == (sum(len(u[0]) for u in us), 4)
        assert np.abs(zf).max() > 0.05 and np.ptp(zf) > 0.1                       # not a degenerate model
        assert np.abs(zs - zf).max() < 2e-4 and np.abs(ls - lf).max() < 2e-4, np.abs(zs - zf).max()

    out, recs = {}, {}
    for share in (False, True):
        c = AnswerCache("test", None)
        e = H.make_dm(tok, model=tiny, batch_size=8, prefix_share=share, cache=c)
        out[share] = e.decide(qs)
        recs[share] = dict(c._load())
    assert recs[True].keys() == recs[False].keys() and len(recs[True]) == 4 * 5
    for k, r in recs[True].items():
        assert set(r) == {"z", "lse", "n", "cov"} and r["n"] == recs[False][k]["n"]
        assert np.abs(np.subtract(r["z"], recs[False][k]["z"])).max() < 5e-4
    for i in range(5):
        a = R.record({"qv": out[True][f"s{i}|qv"], "qt": out[True][f"s{i}|qt"]}, level="E1")
        b = R.record({"qv": out[False][f"s{i}|qv"], "qt": out[False][f"s{i}|qt"]}, level="E1")
        assert all(abs(a[f] - b[f]) < 1e-4 for f in ("qv", "qt", "cov_min"))
    c = AnswerCache("test", None)                            # partly cached groups still share their prefix
    for n, (k, r) in enumerate(recs[False].items()):
        if n % 3 == 0:
            c.put(k, r)
    part = H.make_dm(tok, model=tiny, batch_size=8, prefix_share=True, cache=c).decide(qs)
    for k in out[False]:
        for f in out[False][k].forms:
            assert np.abs(np.subtract(part[k].letter_probs[f][0], out[False][k].letter_probs[f][0])).max() < 1e-4


def test_routed_keys_and_prefix_share_exact(tok, R, tiny):
    """decide on routed E2 questions (tiny fp32 Qwen3.5): the same cache keys (sha1 of the prompt
    rendered with SYSTEM_E2) and the same answers with prefix sharing off and on, for both groupings;
    qa is None exactly for the signatures without a QA state."""
    qs, with_qa, without = _e2(R)
    out, recs = {}, {}
    for share, grouping in ((False, "direct"), (True, "direct"), (True, "chunked")):
        c = AnswerCache("test", None)
        e = H.make_dm(tok, model=tiny, batch_size=8, prefix_share=share, grouping=grouping, cache=c)
        out[(share, grouping)] = e.decide(qs)
        recs[(share, grouping)] = dict(c._load())
    ref = recs[(False, "direct")]
    dm = H.make_dm(tok, cache_only=True)
    want = {dm.key(f) for q in qs for f in q.forms}
    assert set(ref) == want and len(want) == sum(len(q.forms) for q in qs) == 2 * len(qs)
    by_sig = lambda o, s: {q.family: o[q.qkey] for q in qs if q.qkey.startswith(s + "|")}  # noqa: E731
    r0 = {s: R.record(by_sig(out[(False, "direct")], s)) for s in with_qa + without}
    for key, rr in recs.items():
        assert rr.keys() == ref.keys(), key
        assert max(np.abs(np.subtract(r["z"], ref[k]["z"])).max() for k, r in rr.items()) < 5e-4, key
        for s in with_qa + without:
            rec = R.record(by_sig(out[key], s))
            for f in ("qv", "qt", "cov_min"):
                assert abs(rec[f] - r0[s][f]) < 1e-4, (key, s, f)
            assert (rec["qa"] is None) == (s in without)
            if s in with_qa:
                assert abs(rec["qa"] - r0[s]["qa"]) < 1e-4


def test_chunked_groups_respect_batch_limits(tok):
    dm = H.make_dm(tok, batch_size=4, prefix_share=True, grouping="chunked")
    jobs = {f"k{i}": (f"text {chr(97 + (i * 7) % 26)}", 2) for i in range(10)}
    groups = {"g1": [f"k{i}" for i in range(7)], "g2": ["k7", "k8", "k9"]}
    chunks = dm._chunks(groups, jobs)
    assert [len(c) for c in chunks] == [4, 3, 3]
    assert chunks[0] + chunks[1] == sorted(groups["g1"], key=lambda k: jobs[k][0])
    long = {f"k{i}": ("word " * 1000 + str(i), 2) for i in range(10)}           # > LONG tokens each
    assert [len(c) for c in dm._chunks({"g1": groups["g1"]}, long)] == [4, 3]
    dm32 = H.make_dm(tok, batch_size=32, grouping="chunked")
    assert (J.LONG, J.LONG_BS) == (900, 8)
    assert [len(c) for c in dm32._chunks({"g": list(long)}, long)] == [8, 2]
    assert [len(c) for c in dm32._chunks({"g": list(jobs)}, jobs)] == [10]


def test_spec_and_tag(tok):
    dm = H.make_dm(tok, cache_only=True, prefix_share=True)
    s = dm.spec()
    assert s["engine"] == dm.tag == "qwen3.5-4b" and s["family"] == "jev" and s["layout"] == "letter"
    assert s["revision"] == J.DEFAULT_REVISION and s["batch_size"] == 32 and s["prefix_share"] is True
    assert s["grouping"] == "direct" and s["adapter_hash"] is None and s["members"] == 1
    assert H.make_dm(tok, cache_only=True, cache_tag="hb-qwen3.5-4b", batch_size=99).spec()["batch_size"] == 32
    with pytest.raises(ValueError):
        H.make_dm(tok, cache_only=True, grouping="qvqt")


# --------------------------------------------------------------------------- #
# GPU guard
# --------------------------------------------------------------------------- #
def test_physical_gpu_and_check_device(monkeypatch):
    s = types.SimpleNamespace(forbidden_gpus=(2,))
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    for cvd, gpu, phys in ((None, 1, 1), (None, 2, 2), ("1,2", 1, 2), ("2", 0, 2), ("0,1", 1, 1),
                           ("1, 0", 0, 1), ("GPU-5f3c", 0, None), ("1", 3, None), ("", 0, None)):
        if cvd is None:
            monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        else:
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cvd)
        assert G.physical_gpu(gpu) == phys == G.physical_gpu(f"cuda:{gpu}"), (cvd, gpu)
        if phys is None or phys == 2:
            with pytest.raises(G.ForbiddenDevice):
                G.check_device(f"cuda:{gpu}", s)
            with pytest.raises(G.ForbiddenDevice):
                J.JevDecisionModel(device=f"cuda:{gpu}", settings=s, cache=AnswerCache("t", None), verbose=False)
        else:
            assert G.check_device(f"cuda:{gpu}", s) == f"cuda:{gpu}"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert G.check_device("cpu", s) == "cpu" and G.physical_gpu("cpu") is None
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(G.ForbiddenDevice):
        G.check_device("cuda:0", s)
    with pytest.raises(ValueError):
        G.check_device("tpu:0", s)
    assert G.forbidden_gpus(types.SimpleNamespace(forbidden_gpus=[])) == set()
