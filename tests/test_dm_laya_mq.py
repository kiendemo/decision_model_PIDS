"""habitus.dm.laya_mq (layout MQ) on CPU: a tiny randomly initialised ModernBERT (fp32, sdpa, 4 layers =
full / local / local / full, local window +-64) under the vendored DecisionModelHBL, with the Laya
checkpoint's ModernBERT tokenizer. Covers the layout and mask rules, (a) question isolation, (b)
option-order invariance, (c, d) the state's and a block's hidden states at the encoder and after the
head, the equality with DecisionModelHBL's own MQ hooks, (e) negative controls that must break, the
never-cut length rules (= L-a's bounds), packing exactness, and LayaMQDecisionModel (Decision 'mq' with
the single form 'fwd', canonical layout from any form, cache, cache-only, member order, GPU guard)."""

import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from habitus.dm import laya as LA  # noqa: E402
from habitus.dm import laya_mq as MQ  # noqa: E402
from habitus.dm.base import CacheMiss  # noqa: E402
from test_dm_laya import habit, laya_tokenizer, make_settings, routed_question, save_ckpt, tiny_config  # noqa: E402

TOL = 1e-5                       # |dp| of the exact tests; also hidden states
TOL_Z = 1e-4                     # |d logit|: fp32 noise x the scorer gain of the tiny model
BREAK_Z = 1e-2                   # a negative control must move a logit by >= 100 x TOL_Z


# --------------------------------------------------------------------------- #
# Fixtures: tokenizer, a tiny Laya-style checkpoint, habit questions
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tok():
    return laya_tokenizer()


MQ_CFG = {"hidden": 128, "layers": 4, "local_attention": 128, "global_every": 3, "inter": 256, "init_range": 0.1}


def mq_model(tok, seed):
    from transformers import AutoModel
    torch.manual_seed(seed)
    enc = AutoModel.from_config(tiny_config(tok, **MQ_CFG), attn_implementation="sdpa")
    model = LA.DecisionModelHBL(enc, 2, 2).eval()
    with torch.no_grad():                    # spread the random logits so 1e-5 is a strict tolerance
        model.scorer[3].weight.mul_(50.0)
    return model


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory, tok):
    return save_ckpt(tmp_path_factory.mktemp("mq") / "tiny-laya", tok, model=mq_model(tok, 0), **MQ_CFG)


@pytest.fixture(scope="module")
def ckpt2(tmp_path_factory, tok):
    return save_ckpt(tmp_path_factory.mktemp("mq2") / "tiny-laya-b", tok, model=mq_model(tok, 1), **MQ_CFG)


@pytest.fixture(scope="module")
def model(ckpt):
    return LA.load_model(ckpt, "cpu")


@pytest.fixture
def st(tmp_path):
    return make_settings(tmp_path / "runs")


def _view(kind, i, n_names=24, ntype="process"):
    """A habit view long enough (> 64 ModernBERT tokens) for the local window to matter."""
    T, _X = habit()
    names = [f"prog{i}_{j}.exe {1 + j % 3}/3" for j in range(n_names)]
    sec = {"entity": {"type": ntype, "os": "windows", "image_dir": ["system", "home", "staging"][i % 3]},
           "hard_bits": {"observable": "yes"},
           "reference": {"days": 3, "seen_days": f"{i % 3}/3", "usual": names}}
    if kind is T.ViewKind.V_beh:
        sec["class_habits"] = {"modify": ["system", "cache"], "peers": ["internal"]}
    return T.View(kind, ntype, sec, meta={"ident": f"id{i}", "tier0": "IDENTITY", "k_mode": "k2"})


DIRS = ("system", "application", "home", "staging", "cache", "config", "other", "none")
EPS = ("loopback", "private", "internal", "external")


def beh_questions(i, n=10):
    """n inclusion questions (P2 dir classes, then P3 endpoint classes) on one V_beh view."""
    T, X = habit()
    v = _view(T.ViewKind.V_beh, i)
    qs = [X.build(v, T.Family.P2_modify, T.Bucket(T.Family.P2_modify, f"dir:{d}", f"{d} directories")) for d in DIRS]
    qs += [X.build(v, T.Family.P3_peers, T.Bucket(T.Family.P3_peers, f"ep:{e}", f"{e} addresses")) for e in EPS]
    return qs[:n]


def p1_question(i, long_names=True):
    T, X = habit()
    v = _view(T.ViewKind.V_par, i)
    names = ([f"C:/Program Files/Vendor Suite {k}/bin/updater_service_host_{k}.exe" for k in range(4)]
             if long_names else ["services.exe"])
    listed = tuple(T.Bucket(T.Family.P1_parent, f"id:{n}", n) for n in names)
    return X.build(v, T.Family.P1_parent, listed=listed)


def tq(qs, orders=None):
    T, _X = habit()
    return [T.to_typed(q, orders or T.ORDERS) for q in qs]


def _enc(tok, qs):
    return [MQ.encode_parts(tok, MQ.canonical_parts(q.forms[0]), q.qtype) for q in tq(qs)]


def _z(model, layouts, tok, **kw):
    with torch.no_grad():
        return MQ.forward(model, MQ.collate(layouts, tok.pad_token_id), **kw)


def _gap(za, zb):
    """(max |softmax(za) - softmax(zb)|, max |za - zb|) of two logit vectors."""
    za = torch.as_tensor(np.asarray(za, dtype=np.float64))
    zb = torch.as_tensor(np.asarray(zb, dtype=np.float64))
    return float((torch.softmax(za, -1) - torch.softmax(zb, -1)).abs().max()), float((za - zb).abs().max())


def _same(za, zb):
    dp, dz = _gap(za, zb)
    return dp <= TOL and dz <= TOL_Z


def _max_gap(pairs):
    g = [_gap(a, b) for a, b in pairs]
    return max(x for x, _ in g), max(y for _, y in g)


def _isolation_gap(model, tok, state, blocks, **kw):
    """(max |dp|, max |d logit|) between block j in the full row and block j alone, over all blocks."""
    zf = _z(model, [MQ.row_layout(state, blocks)], tok, **kw)
    return _max_gap((zf[j, :b.n], _z(model, [MQ.row_layout(state, [b])], tok, **kw)[0, :b.n])
                    for j, b in enumerate(blocks))


def _perm_gap(model, tok, state, blocks, **kw):
    """(max |dp|, max |d logit|) of option k before / after reversing every block's options (mapped back)."""
    rev = [MQ.Block(b.q_ids, tuple(reversed(b.opt_ids)), b.qtype) for b in blocks]
    za = _z(model, [MQ.row_layout(state, blocks)], tok, **kw)
    zb = _z(model, [MQ.row_layout(state, rev)], tok, **kw)
    return _max_gap((za[j, :b.n], zb[j, :b.n].flip(0)) for j, b in enumerate(blocks))


# --------------------------------------------------------------------------- #
# 1. layout, positions and the mask rules
# --------------------------------------------------------------------------- #
def test_layout_positions_and_mask_rules(tok):
    enc = _enc(tok, beh_questions(0, 3))
    state = enc[0][0]
    assert all(s == state for s, _ in enc)                          # one state per view
    blocks = [b for _, b in enc]
    lay = MQ.row_layout(state, blocks)
    S = len(state)
    assert lay["ids"][0] == tok.cls_token_id and lay["ids"][S - 1] == tok.sep_token_id
    assert lay["pos"][:S] == list(range(S)) and len(lay["ids"]) == S + sum(b.length for b in blocks)
    for j, (b, mk) in enumerate(zip(blocks, lay["markers"]), 1):
        a = len(b.q_ids)
        q0 = lay["seg"].index(j)
        assert lay["pos"][q0:q0 + a] == list(range(S, S + a))       # every question part starts at S
        assert lay["ids"][q0 + a - 1] == tok.sep_token_id
        for k, m in enumerate(mk):
            assert lay["ids"][m] == tok.mask_token_id and lay["opt"][m] == k + 1
            assert lay["pos"][m] == S + a                            # options share their start position
    batch = MQ.collate([lay, MQ.row_layout(state, blocks[:1])], tok.pad_token_id)
    full, slide = MQ.block_masks(batch["seg"], batch["opt"], batch["pos"], batch["valid"], 64)
    seg, opt, pos = (batch[k][0] for k in ("seg", "opt", "pos"))
    f = full[0]
    stt, q1 = seg == 0, (seg == 1) & (opt == 0)
    o11, o12, b2 = (seg == 1) & (opt == 1), (seg == 1) & (opt == 2), seg == 2
    assert f[stt][:, stt].all() and not f[stt][:, ~stt].any()                   # state -> state only
    assert f[q1][:, seg == 1].all() and f[q1][:, stt].all() and not f[q1][:, b2].any()
    assert f[o11][:, o11].all() and f[o11][:, q1].all() and not f[o11][:, o12].any() and not f[o11][:, b2].any()
    near = (pos[:, None] - pos[None, :]).abs() <= 64
    assert torch.equal(slide[0], f & near)                                       # window on logical positions
    pad = ~batch["valid"][1]
    assert pad.any() and full[1][pad][:, batch["seg"][1] == 0].any() and not full[1][:, pad].any()
    assert batch["marker_pos"].shape[0] == 3 + 1 and batch["blk_row"].tolist() == [0, 0, 0, 1]


def test_lengths_never_cut_and_equal_to_la_bounds(tok):
    q = tq([p1_question(1)])[0]
    parts = MQ.canonical_parts(q.forms[0])
    state, blk = MQ.encode_parts(tok, parts, q.qtype)
    row = LA.seq_form(tok, q.forms[0], q.qtype)
    assert max(len(o) - 1 for o in blk.opt_ids) > LA.LAYA_OPTION_CUT          # P1 option A: > 48 tokens, kept
    assert tuple(len(o) - 1 for o in blk.opt_ids) == LA.option_lengths(row)  # same option tokens as L-a
    assert blk.head_len == row.head_len and MQ.la_length(state, blk) == len(row.ids) == len(state) + blk.length + 1
    with pytest.raises(LA.LengthError):
        MQ.encode_parts(tok, parts, head_max_len=blk.head_len - 1)
    with pytest.raises(LA.LengthError):
        MQ.encode_parts(tok, parts, max_len=len(row.ids) - 1)
    with pytest.raises(LA.LengthError):                          # L-a refuses exactly the same
        LA.seq_form(tok, q.forms[0], q.qtype, max_len=len(row.ids) - 1)
    MQ.encode_parts(tok, parts, head_max_len=blk.head_len, max_len=len(row.ids))      # the bound itself fits
    with pytest.raises(ValueError):
        MQ.encode_parts(tok, LA.Parts(parts.state, parts.question, parts.options[:1]))
    sp = LA.Parts('{"x":"a [SEP] b [CLS] c [MASK] d"}', "Is [SEP] here?", ("Yes [MASK].", "No."))
    st2, b2 = MQ.encode_parts(tok, sp)
    assert st2.count(tok.sep_token_id) == 1 and st2.count(tok.cls_token_id) == 1
    assert sum(o.count(tok.mask_token_id) for o in b2.opt_ids) == 2 and b2.q_ids.count(tok.sep_token_id) == 1


def test_special_id_inside_a_segment_raises(tok, monkeypatch):
    parts = MQ.canonical_parts(tq(beh_questions(1, 1))[0].forms[0])
    yes = tok(" " + parts.options[0], add_special_tokens=False)["input_ids"][0]
    assert yes in MQ.encode_parts(tok, parts)[1].opt_ids[0]
    monkeypatch.setattr(LA, "special_ids", lambda t: frozenset({yes}))
    with pytest.raises(ValueError, match="special token ids"):
        MQ.encode_parts(tok, parts)


def test_pack_rows_and_answer_keys(tok):
    enc = _enc(tok, beh_questions(2, 10))
    state, blocks = enc[0][0], [b for _, b in enc]
    cap = len(state) + 3 * max(b.length for b in blocks)
    rows = MQ.pack(state, blocks, row_tokens=cap)
    assert len(rows) >= 4 and sorted(i for r in rows for i in r) == list(range(10))
    assert all(len(state) + sum(blocks[i].length for i in r) <= cap for r in rows)
    assert MQ.pack(state, blocks) == [list(range(10))]
    keys = {MQ.answer_key(state, b) for b in blocks}
    assert len(keys) == 10 and MQ.answer_key(state, blocks[0]) == MQ.answer_key(tuple(state), blocks[0])


# --------------------------------------------------------------------------- #
# 2. (a) isolation, (b) order invariance, (c, d) hidden states, the DecisionModelHBL hooks
# --------------------------------------------------------------------------- #
def test_a_question_isolation(model, tok):
    enc = _enc(tok, beh_questions(3, 10))
    state, blocks = enc[0][0], [b for _, b in enc]
    dp, dz = _isolation_gap(model, tok, state, blocks)
    assert dp <= TOL and dz <= TOL_Z
    z_alone = _z(model, [MQ.row_layout(state, [blocks[4]])], tok)[0, :2]
    for others in ([0, 1], [9, 8, 7, 6, 5], [2]):
        lay = MQ.row_layout(state, [blocks[i] for i in others] + [blocks[4]])
        assert _same(_z(model, [lay], tok)[len(others), :2], z_alone)
    # blocks of different Laya types in one row (routed qv 'score' / qt 'noul' next to habit 'choice')
    mixed = [MQ.Block(b.q_ids, b.opt_ids, LA.QTYPES["noul"]) for b in blocks[:5]] + blocks[5:]
    dp, dz = _isolation_gap(model, tok, state, mixed)
    assert dp <= TOL and dz <= TOL_Z
    z_mixed = _z(model, [MQ.row_layout(state, mixed)], tok)
    assert _gap(z_mixed[0, :2], _z(model, [MQ.row_layout(state, blocks[:1])], tok)[0, :2])[1] > TOL_Z


def test_b_option_order_invariance(model, tok):
    T, _X = habit()
    for q in tq(beh_questions(4, 4) + [p1_question(4, long_names=False)]):
        state, blk = MQ.encode_parts(tok, MQ.canonical_parts(q.forms[0]), q.qtype)
        base = _z(model, [MQ.row_layout(state, [blk])], tok)[0, :blk.n]
        rng = np.random.default_rng(0)
        for _ in range(3):
            perm = rng.permutation(blk.n)
            pb = MQ.Block(blk.q_ids, tuple(blk.opt_ids[i] for i in perm), blk.qtype)
            assert _same(_z(model, [MQ.row_layout(state, [pb])], tok)[0, :blk.n], base[torch.as_tensor(perm)])
        rev = LA.form_parts(q.forms[1])                             # the rev form laid out as given
        st_r, blk_r = MQ.encode_parts(tok, rev, q.qtype)
        assert _same(_z(model, [MQ.row_layout(st_r, [blk_r])], tok)[0, :blk.n].flip(0), base)
    enc = _enc(tok, beh_questions(5, 6))
    dp, dz = _perm_gap(model, tok, enc[0][0], [b for _, b in enc])
    assert dp <= TOL and dz <= TOL_Z


def test_c_d_state_and_block_hidden_states(model, tok):
    enc = _enc(tok, beh_questions(6, 8))
    state, blocks = enc[0][0], [b for _, b in enc]
    S = len(state)
    lay_full = MQ.row_layout(state, blocks)
    with torch.no_grad():
        _zf, hf = MQ.forward(model, MQ.collate([lay_full], tok.pad_token_id), return_hidden=True)
        _zs, hs = MQ.forward(model, MQ.collate([MQ.row_layout(state, blocks[:1])], tok.pad_token_id),
                             return_hidden=True)
        for level in ("encoder", "head"):
            assert float((hf[level][0, :S] - hs[level][0, :S]).abs().max()) <= TOL
        seg = torch.tensor(lay_full["seg"])
        for j in (2, 7):
            _z1, h1 = MQ.forward(model, MQ.collate([MQ.row_layout(state, [blocks[j - 1]])], tok.pad_token_id),
                                 return_hidden=True)
            sel = seg == j
            n = int(sel.sum())
            for level in ("encoder", "head"):
                assert float((hf[level][0][sel] - h1[level][0, S:S + n]).abs().max()) <= TOL


def test_forward_equals_the_vendored_model_hooks(model, tok):
    """One block per row: MQ.forward == DecisionModelHBL.forward through its MQ hooks (enc_masks,
    position_ids, the 3-D head src_mask, per-position qtype)."""
    e1, e2 = _enc(tok, beh_questions(7, 5)), _enc(tok, beh_questions(8, 3))
    single = [MQ.row_layout(e1[0][0], [e1[0][1]]), MQ.row_layout(e2[0][0], [e2[2][1]])]
    batch = MQ.collate(single, tok.pad_token_id)
    full, slide = MQ.block_masks(batch["seg"], batch["opt"], batch["pos"], batch["valid"],
                                 model.encoder.config.sliding_window)
    with torch.no_grad():
        zh = model(batch["ids"], batch["valid"].long(), batch["marker_pos"], batch["marker_mask"], batch["qt"],
                   position_ids=batch["pos"], enc_masks=MQ.encoder_masks(full, slide),
                   head_mask=(~full).repeat_interleave(MQ.head_heads(model), dim=0))
        zm = MQ.forward(model, batch)
    assert all(_same(zh[i, :n], zm[i, :n]) for i, n in enumerate(batch["marker_mask"].sum(1).tolist()))


# --------------------------------------------------------------------------- #
# 3. (e) negative controls: each must break the guarantees visibly
# --------------------------------------------------------------------------- #
def test_e_negative_controls_break(model, tok, monkeypatch):
    enc = _enc(tok, beh_questions(9, 8))
    state, blocks = enc[0][0], [b for _, b in enc]
    S = len(state)
    assert S > 2 * model.encoder.config.sliding_window                 # the window cuts into the state
    assert _isolation_gap(model, tok, state, blocks, isolate=False)[1] > BREAK_Z     # no block mask
    assert _isolation_gap(model, tok, state, blocks, window="index")[1] > BREAK_Z    # window on the index
    assert _perm_gap(model, tok, state, blocks, window="index")[1] > BREAK_Z
    assert _perm_gap(model, tok, state, blocks, isolate=False)[1] <= TOL_Z  # shared positions alone keep (I2)

    def head_padding_only(m, h, full, lengths):
        pad = torch.zeros(h.shape[:2], dtype=torch.bool)
        for b, n in enumerate(lengths):
            pad[b, n:] = True
        for layer in m.head.layers:
            h = layer(h, src_key_padding_mask=pad)
        return h
    monkeypatch.setattr(MQ, "_head", head_padding_only)
    assert _isolation_gap(model, tok, state, blocks)[1] > BREAK_Z
    with torch.no_grad():
        _a, hf = MQ.forward(model, MQ.collate([MQ.row_layout(state, blocks)], tok.pad_token_id), return_hidden=True)
        _b, h1 = MQ.forward(model, MQ.collate([MQ.row_layout(state, blocks[:1])], tok.pad_token_id),
                            return_hidden=True)
    assert float((hf["encoder"][0, :S] - h1["encoder"][0, :S]).abs().max()) <= TOL
    assert float((hf["head"][0, :S] - h1["head"][0, :S]).abs().max()) > 10 * BREAK_Z


# --------------------------------------------------------------------------- #
# 4. batching / packing is exact
# --------------------------------------------------------------------------- #
def test_run_rows_batching_and_packing_exact(model, tok):
    views = [_enc(tok, beh_questions(10 + i, n)) for i, n in enumerate((10, 3, 6))]
    one = [MQ.row_layout(e[0][0], [b for _, b in e]) for e in views]
    ref = MQ.run_rows(model, one, tok.pad_token_id, "cpu", batch_rows=8)
    lays, tags = [], []
    for v, e in enumerate(views):
        state, blocks = e[0][0], [b for _, b in e]
        for row in MQ.pack(state, blocks, row_tokens=len(state) + 2 * max(b.length for b in blocks)):
            lays.append(MQ.row_layout(state, [blocks[i] for i in row]))
            tags.append([(v, i) for i in row])
    assert len(lays) > len(views)
    got = MQ.run_rows(model, lays, tok.pad_token_id, "cpu", batch_rows=2, batch_tokens=4000)
    seen = set()
    for tg, zs in zip(tags, got):
        for (v, i), z in zip(tg, zs):
            assert _same(z, ref[v][i])
            seen.add((v, i))
    assert len(seen) == 19


# --------------------------------------------------------------------------- #
# 5. LayaMQDecisionModel
# --------------------------------------------------------------------------- #
def test_decide_contract_isolation_and_cache(model, tok, ckpt, st, monkeypatch):
    T, _X = habit()
    qs = tq(beh_questions(20, 10) + beh_questions(21, 4) + [p1_question(22)])
    dm = MQ.LayaMQDecisionModel(ckpt, ds="HBFAKE", settings=st, preload=[model])
    assert dm.family == "laya" and dm.layout == "mq" and dm.tag.startswith("hbl-mq-tiny-laya-")
    assert dm.tag != LA.hbl_tag("tiny-laya", LA.model_sha256(ckpt))
    out = dm.decide(qs + qs[:2])                                  # duplicates answered once
    assert set(out) == {q.qkey for q in qs} and dm.stats["forwarded"] == len(qs)
    for q in qs:
        d = out[q.qkey]
        assert d.layout == "mq" and d.forms == ("fwd",) and d.cov == {"fwd": 1.0} and d.members == 1
        assert d.letter_probs == d.canon_probs and abs(sum(d.canon_probs["fwd"][0]) - 1) < 1e-9
        state, blk = MQ.encode_parts(tok, MQ.canonical_parts(q.forms[0]), q.qtype)
        z = _z(model, [MQ.row_layout(state, [blk])], tok)[0, :blk.n].double().numpy()
        assert np.abs(np.asarray(d.canon_probs["fwd"][0]) - LA.softmax(z)).max() <= TOL
        a = T.answer_of(d)
        assert a.raw == {"fwd": a.probs} and a.cov_min == 1.0
    fresh = MQ.LayaMQDecisionModel(ckpt, settings=st, preload=[model])
    for q in (qs[3], qs[12], qs[-1]):                             # isolation through the API
        alone = fresh.decide([q])[q.qkey].canon_probs["fwd"][0]
        fresh2 = MQ.LayaMQDecisionModel(ckpt, settings=st, preload=[model])
        assert np.abs(np.asarray(alone) - np.asarray(out[q.qkey].canon_probs["fwd"][0])).max() <= TOL
        mixed = fresh2.decide(qs[::-1][:7] + [q])[q.qkey].canon_probs["fwd"][0]
        assert np.abs(np.asarray(mixed) - np.asarray(alone)).max() <= TOL
    rev_only = tq(beh_questions(20, 1) + [p1_question(22)], ("rev",))  # canonical layout from any form
    o2 = dm.decide(rev_only)
    assert all(o2[q.qkey] == out[q.qkey] for q in rev_only)
    dm.close()
    cache = Path(str(st.paths.runs)) / "answers" / dm.tag / "HBFAKE.jsonl"
    lines = cache.read_text().splitlines()
    assert len(lines) == len(qs) and all({"k", "z", "lse", "n", "cov"} == set(json.loads(x)) for x in lines)
    monkeypatch.setattr(LA, "load_model", lambda *a, **k: (_ for _ in ()).throw(AssertionError("loaded")))
    again = MQ.LayaMQDecisionModel(ckpt, ds="HBFAKE", settings=st, cache_only=True)
    o3 = again.decide(qs)
    assert again.stats["forwarded"] == 0 and all(
        np.abs(np.asarray(o3[k].canon_probs["fwd"][0]) - np.asarray(out[k].canon_probs["fwd"][0])).max() < 1e-4
        for k in out)
    with pytest.raises(CacheMiss):
        again.decide(tq(beh_questions(23, 2)))
    assert set(again.keys(qs)) == {q.qkey for q in qs}
    again.close()


def test_routed_questions_encode_forms0(model, tok, ckpt, st):
    state = '{"entity":{"type":"process","os":"linux","cmd":"/bin/sh -c id"}}'
    qv, qt = routed_question(state, "qv"), routed_question(state, "qt")
    dm = MQ.LayaMQDecisionModel(ckpt, settings=st, preload=[model])
    out = dm.decide([qv, qt])
    for q in (qv, qt):
        state_ids, blk = MQ.encode_parts(tok, LA.form_parts(q.forms[0]), q.qtype)     # canon is the identity
        z = _z(model, [MQ.row_layout(state_ids, [blk])], tok)[0, :blk.n].double().numpy()
        assert out[q.qkey].forms == ("fwd",) and blk.qtype == LA.QTYPES[q.qtype]
        assert np.abs(np.asarray(out[q.qkey].canon_probs["fwd"][0]) - LA.softmax(z)).max() <= TOL
    dm.close()


def test_member_order_ensemble_and_gpu_rule(model, tok, ckpt, ckpt2, st, monkeypatch):
    T, _X = habit()
    m2 = LA.load_model(ckpt2, "cpu")
    qs = tq(beh_questions(30, 4))
    ens = MQ.LayaMQDecisionModel([ckpt, ckpt2], settings=st, preload=[model, m2])
    out = ens.decide(qs)
    for q in qs:
        state, blk = MQ.encode_parts(tok, MQ.canonical_parts(q.forms[0]), q.qtype)
        lay = [MQ.row_layout(state, [blk])]
        per = [tuple(LA.softmax(_z(m, lay, tok)[0, :blk.n].double().numpy())) for m in (model, m2)]
        d = out[q.qkey]
        assert d.members == 2
        for s in range(2):
            assert np.abs(np.asarray(d.canon_probs["fwd"][s]) - np.asarray(per[s])).max() <= TOL
        assert np.abs(np.asarray(T.answer_of(d).probs) - np.asarray(T.pbar(per))).max() <= TOL
    assert ens.tag.startswith("hbl-mq-mix-") and len(ens.spec()["members"]) == 2
    ens.close()
    from habitus.dm.gpu import ForbiddenDevice
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    import dataclasses
    bad = dataclasses.replace(st, forbidden_gpus=(2,))
    with pytest.raises(ForbiddenDevice):
        MQ.LayaMQDecisionModel(ckpt, device="cuda:2", settings=bad, preload=[model])
    with pytest.raises(ValueError, match="layout_version"):
        MQ.LayaMQDecisionModel(ckpt, settings=st, layout_version="mq0")


def test_encoder_must_be_sdpa(model, tok):
    enc = _enc(tok, beh_questions(31, 2))
    batch = MQ.collate([MQ.row_layout(enc[0][0], [b for _, b in enc])], tok.pad_token_id)
    old = model.encoder.config._attn_implementation
    try:
        model.encoder.config._attn_implementation = "eager"
        with pytest.raises(ValueError):
            MQ.forward(model, batch)
    finally:
        model.encoder.config._attn_implementation = old


def test_resolve_mq_runs_and_inits(ckpt, st, tmp_path):
    import shutil
    d = tmp_path / "mq-run"
    shutil.copytree(ckpt, d)
    (d / LA.RUN_CONFIG).write_text(json.dumps({"layout": "mq", "init": "tiny-laya", "fold": "all"}))
    assert MQ.resolve_mq(str(d), st) == (str(d), "tiny-laya")
    with pytest.raises(ValueError):
        LA.resolve_init(str(d), st)                         # an MQ run is never read as L-a
    assert MQ.resolve_mq(ckpt, st) == LA.resolve_init(ckpt, st)
