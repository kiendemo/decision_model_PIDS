"""habitus.dm.laya (layout L-a): prompt parsing (split_prompt round trip with the ONE habit renderer),
rows (layout, never-cut refusal, special tokens, Laya types), train row == serve row, the vendored
decision model (strict state-dict loads, the dict-mask / position-id hooks), and LayaDecisionModel
(Decision layout / member order, answer cache, cache-only, refusal, seed ensembles, GPU guard).

CPU only. The tokenizer is the Laya checkpoint's ModernBERT tokenizer under settings.paths.laya_hub
(read-only); models are a tiny randomly initialised ModernBertConfig saved in-test as a Laya-format
checkpoint. Tests needing a local asset skip when it is missing, unless HABITUS_REQUIRE_ASSETS is set.
Provenance tests (laya.common of Laya 0.3.5 under settings.paths.laya_src) run only with
HABITUS_PROVENANCE=1.
"""

import importlib.util
import json
import os
import re
import shutil
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from habitus import config  # noqa: E402
from habitus.dm import laya as LA  # noqa: E402
from habitus.dm.base import CacheMiss, Form, TypedQuestion  # noqa: E402

SRC = Path(LA.__file__).resolve().parent
DS, DAY, REF = "CADETS_E3", "2018-04-02", ("2018-04-03", "2018-04-04")
FOLD = "loho-optc_h501"
PROVENANCE = pytest.mark.skipif(os.environ.get("HABITUS_PROVENANCE") != "1",
                                reason="provenance test: set HABITUS_PROVENANCE=1 (needs paths.laya_src)")


# --------------------------------------------------------------------------- #
# Shared helpers (also imported by test_dm_laya_mq.py and test_train_laya.py)
# --------------------------------------------------------------------------- #
def need(ok, why):
    """Skip when a local asset is missing; fail instead under HABITUS_REQUIRE_ASSETS."""
    if not ok:
        if os.environ.get("HABITUS_REQUIRE_ASSETS", "") not in ("", "0"):
            pytest.fail(f"HABITUS_REQUIRE_ASSETS: {why}")
        pytest.skip(why)


def need_weights(ok, why):
    """The Laya weights are not in the release bundle (tests marked local_assets): skip without them,
    fail only under HABITUS_REQUIRE_LOCAL_ASSETS."""
    if not ok:
        from conftest import missing_local_asset
        missing_local_asset(why)


def make_settings(tmp, **over):
    """Settings whose runs directory is `tmp` (the session's HABITUS_RUNS is ignored here)."""
    env = {k: v for k, v in os.environ.items() if k != "HABITUS_RUNS"}
    return config.load_settings(overrides={"paths": {"runs": str(tmp)}, **over}, env=env)


def typed_decisions_dir():
    try:
        return LA.inits(config.settings())["typed-decisions"]
    except Exception:              # no config: no checkpoint
        return None


def laya_tokenizer_dir():
    """The Laya ModernBERT tokenizer: <laya_hub>/typed-decisions/tokenizer, else its release copy
    <tokenizers>/laya-typed-decisions; None when neither exists."""
    cands = []
    td = typed_decisions_dir()
    if td:
        cands.append(os.path.join(td, "tokenizer"))
    try:
        tk = config.settings().paths.tokenizers
    except Exception:
        tk = None
    if tk:
        cands.append(os.path.join(str(tk), LA.RELEASE_TOKENIZER_DIR))
    return next((c for c in cands if os.path.isfile(os.path.join(c, "tokenizer.json"))), None)


def laya_tokenizer():
    d = laya_tokenizer_dir()
    need(d is not None, "Laya tokenizer (paths.laya_hub/typed-decisions/tokenizer or its release copy) not present")
    return LA.load_tokenizer_dir(d)


def tiny_config(tok, hidden=64, layers=2, local_attention=8, global_every=2, inter=96, init_range=0.2):
    from transformers import ModernBertConfig
    return ModernBertConfig(vocab_size=len(tok), hidden_size=hidden, intermediate_size=inter,
                            num_hidden_layers=layers, num_attention_heads=4, local_attention=local_attention,
                            global_attn_every_n_layers=global_every, max_position_embeddings=4096,
                            pad_token_id=tok.pad_token_id, bos_token_id=tok.cls_token_id,
                            eos_token_id=tok.sep_token_id, cls_token_id=tok.cls_token_id,
                            sep_token_id=tok.sep_token_id, initializer_range=init_range)


def tiny_model(tok, seed=0, **cfg):
    """A randomly initialised DecisionModelHBL on a tiny encoder (parameters perturbed so logits differ)."""
    from transformers import AutoModel
    torch.manual_seed(seed)
    m = LA.DecisionModelHBL(AutoModel.from_config(tiny_config(tok, **cfg), attn_implementation="sdpa"))
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.05)
        m.temperature.fill_(1.0)
    return m.eval()


def save_ckpt(d, tok, seed=0, model=None, **cfg):
    """A Laya-format checkpoint directory of a tiny model (encoder/, tokenizer/, config, weights)."""
    from safetensors.torch import save_file
    d = str(d)
    os.makedirs(os.path.join(d, "encoder"), exist_ok=True)
    tiny_config(tok, **cfg).to_json_file(os.path.join(d, "encoder", "config.json"))
    shutil.copytree(laya_tokenizer_dir(), os.path.join(d, "tokenizer"), dirs_exist_ok=True)
    with open(os.path.join(d, LA.LAYA_CONFIG), "w", encoding="utf-8") as f:
        json.dump({"encoder": "tiny-modernbert", "head_layers": 2, "act_costs": {"escalate": 0.5},
                   "max_len": 1024, "head_max_len": 256}, f)
    m = model if model is not None else tiny_model(tok, seed, **cfg)
    save_file({k: v.contiguous() for k, v in m.state_dict().items()}, os.path.join(d, LA.WEIGHTS))
    return d


class ChatTok:
    """The Qwen3.5 chat template on (system, user) messages with thinking disabled."""

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True):
        s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
        if add_generation_prompt:
            s += "<|im_start|>assistant\n" + ("" if enable_thinking else "<think>\n\n</think>\n\n")
        return s


CHAT = ChatTok()


def habit():
    """(types, text) of the habit question layer."""
    from habitus.questions.habit import text as X
    from habitus.questions.habit import types as T
    return T, X


def view(kind, i, ntype="process", extra=None):
    T, _X = habit()
    sec = {"entity": {"type": ntype, "os": "linux", "image_dir": ["system", "home", "staging"][i % 3],
                      "file_name": f"prog{i}", **(extra or {})},
           "hard_bits": {"observable": {"exec": "yes", "lineage": "yes"}},
           "reference": {"days": 2, "seen_days": f"{i % 3}/2", "frequency": "high"}}
    if kind is T.ViewKind.V_beh:
        sec["class_habits"] = {"role": "unknown", "modify_dirs": {"home": "2/2"}}
    if kind is T.ViewKind.V_own:
        sec = {"entity": sec["entity"], "connection": {"port_class": "web"}, "reference": sec["reference"]}
    return T.View(kind, ntype, sec, meta={"ident": f"id{i}", "tier0": "IDENTITY", "k_mode": "k2"})


LONG_NAMES = ("/usr/lib/x86_64-linux-gnu/libexec/session-manager-helper-daemon",
              "/opt/vendor/agent/bin/background-update-scheduler-service",
              "/usr/local/sbin/periodic-maintenance-wrapper-runner", "/usr/libexec/accounts-daemon-helper")


def make_questions(n_views=3):
    """Habit questions of every kind (inclusion, P1 with long names, N2) built by text.build."""
    T, X = habit()
    qs = []
    for i in range(n_views):
        v = view(T.ViewKind.V_beh, i)
        for w, txt in (("staging", "staging directories"), ("home", "home directories")):
            qs.append(X.build(v, T.Family.P2_modify, T.Bucket(T.Family.P2_modify, f"dir:{w}", txt)))
        qs.append(X.build(v, T.Family.P3_peers, T.Bucket(T.Family.P3_peers, "ep:external", "external addresses")))
        qs.append(X.build(view(T.ViewKind.V_file, i, "file", {"path": f"/home/u/f{i}.txt"}), T.Family.F1_writer,
                          T.Bucket(T.Family.F1_writer, f"id:w{i}", f"/usr/bin/editor{i}")))
        listed = tuple(T.Bucket(T.Family.P1_parent, f"id:p{i}{j}", n) for j, n in enumerate(LONG_NAMES[:1 + i % 4]))
        qs.append(X.build(view(T.ViewKind.V_par, i), T.Family.P1_parent, listed=listed))
        qs.append(X.build(view(T.ViewKind.V_own, i, "netflow"), T.Family.N2_address))
    return qs


def typed(qs, orders=None):
    T, _X = habit()
    return [T.to_typed(q, orders or T.ORDERS) for q in qs]


def make_items(questions, role="train", day=DAY, ref=REF, ds=DS, fold=FOLD, seed=0, both=False, chat=CHAT):
    """Items rendered by text.render: inclusion yes-share 0.8 for staging / external, 0.1 otherwise;
    choice (0.7, 0.1, 0.1, 0.1). both=True: each question in both orders, else one random order."""
    T, X = habit()
    rng = np.random.default_rng(seed)
    out = []
    for q in questions:
        orders = T.ORDERS if both else (T.ORDERS[int(rng.integers(2))],)
        for order in orders:
            if q.bucket is not None:
                y = 0.8 if q.bucket.key in ("dir:staging", "ep:external") else 0.1
                canon = (y, 1 - y)
            else:
                canon = (0.7, 0.1, 0.1, 0.1)
            out.append(T.Item(ds, day, ref, "k2", "sig", q.family, q.bucket.key if q.bucket else T.NO_BUCKET,
                              q.view.sha1, q.qkey, order, X.render(chat, q, order), q.letters,
                              T.to_canonical(canon, order), weight=float(rng.uniform(0.5, 2.0)), role=role,
                              fold=fold))
    return out


def laya_common():
    """laya.common of the local Laya 0.3.5 (settings.paths.laya_src), loaded under a private name."""
    src = config.settings().paths.laya_src
    path = os.path.join(src, "laya", "common.py") if src else None
    if path is None or not os.path.isfile(path):             # provenance tests are outside REQUIRE_ASSETS
        pytest.skip("paths.laya_src (Laya 0.3.5) not configured / present")
    spec = importlib.util.spec_from_file_location("_laya035_common", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def routed_question(state, qid="qv"):
    """A routed-style TypedQuestion built directly from Forms (qv: 'score' with fwd / rev; qt: 'noul')."""
    if qid == "qv":
        opts = ("Routine.", "New but plausibly legitimate.", "Suspicious.", "Likely malicious.")
        forms = (Form("fwd", "S", state, "How should a security analyst triage this process?", opts, (0, 1, 2, 3)),
                 Form("rev", "S", state, "How should a security analyst triage this process?", opts[::-1],
                      (3, 2, 1, 0)))
        return TypedQuestion("sig|qv", "qv", "score", forms, "sig|sig")
    forms = (Form("pos", "S", state, "Does the evidence show attacker tradecraft?",
                  ("Yes, attacker tradecraft is shown.", "No attacker tradecraft is shown."), (0, 1)),
             Form("neg", "S", state, "Is everything consistent with ordinary activity?",
                  ("Yes, consistent with ordinary legitimate activity.", "No, something indicates tradecraft."),
                  (1, 0)))
    return TypedQuestion("sig|qt", "qt", "noul", forms, "sig|sig")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tok():
    return laya_tokenizer()


@pytest.fixture(scope="module")
def ckpts(tok, tmp_path_factory):
    base = tmp_path_factory.mktemp("laya")
    return [save_ckpt(base / f"tiny-{s}", tok, s) for s in (0, 1)]


@pytest.fixture
def st(tmp_path):
    return make_settings(tmp_path / "runs")


# --------------------------------------------------------------------------- #
# split_prompt / parts_from_prompt
# --------------------------------------------------------------------------- #
def test_split_prompt_round_trips_the_one_renderer():
    T, X = habit()
    for q in make_questions(4):
        for o in T.ORDERS:
            text = X.render(CHAT, q, o)
            p = LA.split_prompt(text, tok=CHAT)
            f = T.to_typed(q, (o,)).forms[0]
            assert p == LA.form_parts(f) == LA.parts_from_prompt(text)
            assert p.state == q.view.text and p.question == q.text
            want = q.option_text if o == "fwd" else tuple(reversed(q.option_text))
            assert p.options == want
            assert LA.user_text(p) == X.user_prompt(q, o) == f.user()


def test_split_prompt_is_strict():
    T, X = habit()
    q = make_questions(1)[0]
    text = X.render(CHAT, q, "fwd")
    bad = {
        "other system": text.replace("You predict", "You guess", 1),
        "no think tail": text[:-len(LA.THINK_SUFFIX)],
        "letter": text.replace("\nB: ", "\nC: ", 1),
        "answer line": text.replace("Answer with one letter: A, B", "Answer with one letter: A, B, C"),
        "newline in question": text.replace(q.text, q.text[:5] + "\n" + q.text[5:], 1),
        "empty option": text.replace("\nB: " + q.option_text[1], "\nB: ", 1),
        "no options": text.replace(LA.OPTIONS_HEAD, "\nChoices:\n", 1),
    }
    for name, t in bad.items():
        assert t != text, name
        with pytest.raises(LA.PromptError):
            LA.split_prompt(t)
    other = ChatTok()
    other.apply_chat_template = lambda *a, **k: text + " "
    with pytest.raises(LA.PromptError, match="re-render"):
        LA.split_prompt(text, tok=other)


def test_split_prompt_with_the_real_qwen_template():
    from transformers import AutoTokenizer
    from habitus.state import sanitize as SAN
    s = config.settings()
    cands = [Path(str(s.paths.tokenizers)) / SAN.TOKENIZER_DIR,
             Path(str(s.paths.models)) / "hub" / "models--Qwen--Qwen3.5-4B" / "snapshots" / SAN.QWEN_REVISION]
    snap = next((c for c in cands if (c / "tokenizer_config.json").is_file()), None)
    need(snap is not None, "Qwen3.5-4B tokenizer (release copy or HF snapshot) not present")
    qt = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
    T, X = habit()
    for q in make_questions(2):
        for o in T.ORDERS:
            assert LA.split_prompt(X.render(qt, q, o), tok=qt) == LA.form_parts(T.to_typed(q, (o,)).forms[0])


# --------------------------------------------------------------------------- #
# Rows
# --------------------------------------------------------------------------- #
def test_row_layout_and_definition_only_options(tok):
    qs = typed(make_questions(3))
    for q in qs:
        for f in q.forms:
            r = LA.seq_form(tok, f, q.qtype)
            assert tok.convert_ids_to_tokens([r.ids[0], r.ids[r.sep], r.ids[-1]]) == ["[CLS]", "[SEP]", "[SEP]"]
            assert r.ids[r.markers[0] - 1] == tok.sep_token_id
            assert all(r.ids[m] == tok.mask_token_id for m in r.markers) and len(r.markers) == len(f.options)
            assert r.qtype == LA.QTYPES["choice"]
            assert tok.decode(r.ids[1:r.markers[0] - 1]).startswith("choice question: ")
            for k, o in enumerate(f.options):           # each option whole behind its marker
                ids = tok(" " + o, add_special_tokens=False)["input_ids"]
                assert list(r.ids[r.markers[k] + 1:r.markers[k] + 1 + len(ids)]) == ids
            assert list(r.ids[r.sep + 1:-1]) == tok(f.state, add_special_tokens=False)["input_ids"]
            assert re.fullmatch(r"[0-9a-f]{40}", LA.row_key(r))
    assert any(max(LA.option_lengths(LA.seq_form(tok, q.forms[0], q.qtype))) > LA.LAYA_OPTION_CUT for q in qs)
    r = LA.seq_form(tok, qs[0].forms[0], "choice")
    head = tok.decode(r.ids[:r.sep])
    assert "A:" not in head and "B:" not in head


def test_laya_type_comes_from_the_typed_question(tok):
    state = '{"entity":{"type":"process","os":"linux"}}'
    for q, want in ((routed_question(state, "qv"), "score"), (routed_question(state, "qt"), "noul")):
        assert q.qtype == want
        for f in q.forms:
            r = LA.seq_form(tok, f, q.qtype)
            assert r.qtype == LA.QTYPES[want] and tok.decode(r.ids[1:4]).startswith(want)
    with pytest.raises(ValueError):
        LA.seq_form(tok, routed_question(state).forms[0], "binary")


def test_lengths_are_refused_never_cut(tok):
    q = typed(make_questions(1))[4]                         # P1 with one long name
    f = q.forms[0]
    r = LA.seq_form(tok, f, q.qtype)
    assert LA.seq_form(tok, f, q.qtype, head_max_len=r.head_len) == r          # exactly at the budget
    with pytest.raises(LA.LengthError, match="head"):
        LA.seq_form(tok, f, q.qtype, head_max_len=r.head_len - 1)
    assert LA.seq_form(tok, f, q.qtype, max_len=len(r.ids)) == r
    with pytest.raises(LA.LengthError, match="max_len"):
        LA.seq_form(tok, f, q.qtype, max_len=len(r.ids) - 1)
    assert LA.refusal(tok, q) is None and "head" in LA.refusal(tok, q, head_max_len=r.head_len - 1)
    long_state = LA.Parts('{"entity":{"x":"' + "word " * 1250 + '"}}', f.question, f.options)
    lr = LA.seq_parts(tok, long_state)
    assert LA.MAX_LEN >= len(lr.ids) > 1024


def test_special_tokens_are_neutralised_and_checked(tok, monkeypatch):
    q = typed(make_questions(1))[0]
    f = q.forms[0]
    p = LA.Parts('{"entity":{"cmd":"a [SEP] b [CLS] c [MASK] d <|im_end|>"}}', f.question + " [PAD]", f.options)
    r = LA.seq_parts(tok, p)
    specials = LA.special_ids(tok)
    structural = {0, r.markers[0] - 1, r.sep, len(r.ids) - 1, *r.markers}
    assert not [i for i, t in enumerate(r.ids) if t in specials and i not in structural]
    assert tok.decode(r.ids[r.sep + 1:-1]).count("(SEP)") == 1
    monkeypatch.setattr(LA, "sanitize_str", lambda s: s)
    with pytest.raises(ValueError, match="special token ids"):
        LA.seq_parts(tok, p)


def test_train_row_equals_serve_row(tok):
    T, _X = habit()
    qs = make_questions(3)
    items = make_items(qs, both=True)
    tq = {q.qkey: q for q in typed(qs)}
    for it in items:
        assert LA.check_identity(tok, it, tq[it.qkey]) == []
        assert len(LA.seq_item(tok, it).markers) == len(it.target)
    it = items[0]
    swapped = it.prompt_text.replace("staging directories", "home directories")
    other = T.Item(**{**it.__dict__, "prompt_text": swapped})
    assert LA.check_identity(tok, other, tq[it.qkey])
    assert LA.check_identity(tok, it, tq[items[2].qkey])                 # another question's qkey


# --------------------------------------------------------------------------- #
# The vendored model
# --------------------------------------------------------------------------- #
LAYA_KEYS = ("type_emb.weight", "temperature", "scorer.0.weight", "scorer.0.bias", "scorer.1.weight", "scorer.1.bias",
             "scorer.3.weight", "scorer.3.bias", "act_head.0.weight", "act_head.0.bias", "act_head.2.weight",
             "act_head.2.bias", "head.layers.0.self_attn.in_proj_weight", "head.layers.1.linear2.bias")


def test_vendored_state_dict_keys_load_strict(tok, tmp_path):
    m = tiny_model(tok, 2)
    keys = list(m.state_dict())
    assert set(LAYA_KEYS) <= set(keys) and all(k.startswith(("encoder.", "head.", "type_emb", "scorer", "act_head",
                                                            "temperature")) for k in keys)
    d = save_ckpt(tmp_path / "ck", tok, model=m)
    m2 = LA.load_model(d)                                    # strict=True inside
    assert all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), m2.state_dict().values()))
    from safetensors.torch import load_file, save_file
    sd = load_file(os.path.join(d, LA.WEIGHTS))
    sd["extra.weight"] = torch.zeros(1)
    save_file(sd, os.path.join(d, LA.WEIGHTS))
    with pytest.raises(RuntimeError, match="Unexpected"):
        LA.load_model(d)


@pytest.mark.local_assets
def test_real_typed_decisions_weights_load_strict():
    td = typed_decisions_dir()
    need_weights(td is not None and os.path.isfile(os.path.join(td, LA.WEIGHTS)), "laya_hub/typed-decisions weights")
    from safetensors import safe_open
    with safe_open(os.path.join(td, LA.WEIGHTS), "pt") as f:
        names = set(f.keys())
    assert set(LAYA_KEYS) <= names
    hm = LA.load_model(td)
    assert set(hm.state_dict()) == names
    assert torch.equal(hm.temperature, torch.ones(3)) and hm.act_head[0].weight.shape[1] == 1024 + 4
    assert LA.resolve_init("typed-decisions") == (td, "typed-decisions")


def test_dict_masks_position_ids_and_the_local_window(tok):
    """A {"full_attention", "sliding_attention"} dict replaces the padding mask, position_ids pass
    through, local layers attend over |delta position| <= sliding_window; head_mask reaches the head."""
    m = tiny_model(tok, 5)
    assert m.encoder.config.sliding_window == 4 and "sliding_attention" in m.encoder.config.layer_types
    f = typed(make_questions(1))[0].forms[0]
    rows = [LA.seq_parts(tok, LA.Parts(f.state + " x" * k, f.question, f.options)) for k in (0, 7, 30)]
    b = LA.collate(rows, tok.pad_token_id)
    L = b["input_ids"].shape[1]
    pos = torch.arange(L)[None].expand(len(rows), -1)
    w = m.encoder.config.sliding_window
    with torch.no_grad():
        z0 = m(**b)
        z1 = m(**b, position_ids=pos, enc_masks=LA.padding_masks(b["attention_mask"], pos, w))
        z2 = m(**b, position_ids=pos, enc_masks=LA.padding_masks(b["attention_mask"], pos, w + 1))   # control
        z3 = m(**b, head_mask=torch.zeros(L, L, dtype=torch.bool))
        blk = torch.zeros(L, L, dtype=torch.bool)
        blk[:, 1:3] = True
        z4 = m(**b, head_mask=blk)
    assert torch.allclose(z1, z0, atol=1e-5, rtol=0) and torch.allclose(z3, z0, atol=1e-6, rtol=0)
    assert (z2 - z0).abs().max() > 1e-4 and (z4 - z0).abs().max() > 1e-4
    qt = torch.zeros(len(rows), L, dtype=torch.long)
    with torch.no_grad():
        assert torch.equal(m(**{**b, "qtype": qt}), z0)


# --------------------------------------------------------------------------- #
# LayaDecisionModel
# --------------------------------------------------------------------------- #
def _disamb(j):
    exe = (f"C:\\Program Files (x86)\\Common Files\\Vendor{j} Shared\\Installer Cache\\"
           f"{{7C3B2A1F-90AB-4CDE-8F12-3456789ABC0{j}}}\\Package Cache\\x64\\Release\\Bootstrap\\setup.exe")
    return f"setup.exe ({exe[:160]}) #{j + 1}"


def long_p1(i=7):
    """A P1 question whose option A names 4 path-disambiguated parents: its head exceeds 256 tokens."""
    T, X = habit()
    listed = tuple(T.Bucket(T.Family.P1_parent, f"id:s{i}{j}", _disamb(j)) for j in range(4))
    return X.build(view(T.ViewKind.V_par, i), T.Family.P1_parent, listed=listed)


def test_long_head_is_refused_and_left_out(tok, ckpts, st):
    from habitus.dm import laya_mq as MQ
    q, qs = typed([long_p1()])[0], typed(make_questions(2))
    head = LA.seq_form(tok, q.forms[0], q.qtype, head_max_len=10 ** 6).head_len
    assert head > LA.HEAD_MAX_LEN and LA.seq_form(tok, q.forms[1], q.qtype, head_max_len=10 ** 6).head_len == head
    assert "head" in LA.refusal(tok, q) and all(LA.refusal(tok, x) is None for x in qs)
    for cls in (LA.LayaDecisionModel, MQ.LayaMQDecisionModel):
        dm = cls(ckpts[0], settings=st)
        assert set(dm.refused(qs + [q])) == {q.qkey}
        out = dm.decide(qs + [q])
        assert set(out) == {x.qkey for x in qs}                     # refused: never answered, never cut
        dm.close()


def _direct(model, tok, q):
    """Per form: softmax of the row's marker logits (letter order)."""
    return {f.name: LA.softmax(LA.row_logits(model, [LA.seq_form(tok, f, q.qtype)], tok.pad_token_id)[0])
            for f in q.forms}


def test_decide_layout_tag_cache_and_cache_only(tok, ckpts, st, monkeypatch):
    T, _X = habit()
    qs = typed(make_questions(3))
    dm = LA.LayaDecisionModel(ckpts[0], ds=DS, settings=st, batch_rows=5)
    assert dm.family == "laya" and dm.layout == "la"
    digest = LA.model_sha256(ckpts[0])
    assert dm.tag == f"hbl-tiny-0-{digest[:12]}" and re.fullmatch(r"hbl-[A-Za-z0-9_.+-]+-[0-9a-f]{12}", dm.tag)
    out = dm.decide(qs + qs[:3])                                     # duplicates answered once
    assert set(out) == {q.qkey for q in qs} and dm.stats["questions"] == len(qs)
    model = dm.members[0].model
    for q in qs:
        d, raw = out[q.qkey], _direct(model, tok, q)
        assert d.layout == "la" and d.forms == ("fwd", "rev") and d.cov == {"fwd": 1.0, "rev": 1.0}
        assert d.cov_min == 1.0 and d.members == 1
        for f in q.forms:
            assert np.allclose(d.letter_probs[f.name][0], raw[f.name], atol=1e-4)
            assert d.canon_probs[f.name][0] == LA.canonical(d.letter_probs[f.name][0], f.canon)
        a = T.answer_of(d)
        assert np.allclose(a.probs, T.pbar([d.canon_probs["fwd"][0], d.canon_probs["rev"][0]]), atol=1e-12)
    keys = dm.keys(qs)
    cache = Path(str(st.paths.runs)) / "answers" / dm.tag / f"{DS}.jsonl"
    lines = [json.loads(x) for x in cache.read_text().splitlines()]
    assert len(lines) == 2 * len(qs) and {r["k"] for r in lines} == {k for v in keys.values() for k in v}
    assert all(set(r) == {"k", "z", "lse", "n", "cov"} and len(r["z"]) == r["n"] for r in lines)
    with pytest.raises(RuntimeError, match="held by another"):
        other = LA.LayaDecisionModel(ckpts[0], ds=DS, settings=st)
        other.decide(typed(make_questions(4))[-6:])                  # a new answer needs the writer lock
    dm.close()
    again = LA.LayaDecisionModel(ckpts[0], ds=DS, settings=st, cache_only=True)
    monkeypatch.setattr(LA.Member, "load", lambda self: (_ for _ in ()).throw(AssertionError("model loaded")))
    o2 = again.decide(qs)                                            # fully cached: no model, same answers
    assert all(o2[k].letter_probs == out[k].letter_probs for k in out) and again.stats["forwarded"] == 0
    with pytest.raises(CacheMiss) as e:
        again.decide(typed(make_questions(4))[-6:])
    assert e.value.keys and all(re.fullmatch(r"[0-9a-f]{40}", k) for k in e.value.keys)
    one = again.decide(typed(make_questions(1), ("fwd",)))           # a subset of the forms: its own rows
    assert all(d.forms == ("fwd",) for d in one.values())
    sp = again.spec()
    assert sp["layout"] == "la" and sp["family"] == "laya" and sp["members"][0]["model_sha256"] == digest
    assert {"engine", "backbone", "revision", "adapter_hash", "prefix_share", "batch_size", "grouping"} <= set(sp)
    again.close()


def test_explicit_cache_tag_is_a_fresh_namespace(ckpts, st):
    qs = typed(make_questions(1))
    one = LA.LayaDecisionModel(ckpts[0], ds=DS, settings=st, cache_tag="smoke-1")
    assert one.tag == "smoke-1" and one.members[0].tag == "smoke-1"
    one.decide(qs)
    assert one.stats["forwarded"] == 2 * len(qs)
    one.close()
    ens = LA.LayaDecisionModel(ckpts, ds=DS, settings=st, cache_tag="smoke-2")
    assert ens.tag == "smoke-2" and [m.tag for m in ens.members] == ["smoke-2.m0", "smoke-2.m1"]
    ens.close()
    runs = Path(str(st.paths.runs)) / "answers"
    assert (runs / "smoke-1" / f"{DS}.jsonl").is_file()
    assert not (runs / LA.hbl_tag("tiny-0", LA.model_sha256(ckpts[0]))).exists()


def test_routed_questions_are_supported(tok, ckpts, st):
    state = '{"entity":{"type":"process","os":"linux","cmd":"/bin/sh -c id"}}'
    dm = LA.LayaDecisionModel(ckpts[0], settings=st)
    qv, qt = routed_question(state, "qv"), routed_question(state, "qt")
    out = dm.decide([qv, qt])
    assert out[qv.qkey].forms == ("fwd", "rev") and out[qt.qkey].forms == ("pos", "neg")
    for q in (qv, qt):
        raw = _direct(dm.members[0].model, tok, q)
        for f in q.forms:
            assert np.allclose(out[q.qkey].letter_probs[f.name][0], raw[f.name], atol=1e-4)
    assert out[qv.qkey].canon_probs["rev"][0] == tuple(reversed(out[qv.qkey].letter_probs["rev"][0]))
    dm.close()


def test_seed_ensemble_members_in_order_and_pbar(tok, ckpts, st):
    T, _X = habit()
    qs = typed(make_questions(2))
    ens = LA.LayaDecisionModel(ckpts, settings=st)
    out = ens.decide(qs)
    singles = [LA.LayaDecisionModel(c, settings=st) for c in ckpts]
    per = [s.decide(qs) for s in singles]
    assert ens.tag not in {s.tag for s in singles} and ens.tag.startswith("hbl-mix-")        # two init labels
    for q in qs:
        d = out[q.qkey]
        assert d.members == 2
        for f in q.forms:
            for s in range(2):
                assert d.canon_probs[f.name][s] == per[s][q.qkey].canon_probs[f.name][0]
        every = [per[s][q.qkey].canon_probs[o][0] for o in T.ORDERS for s in range(2)]      # orders outer
        a = T.answer_of(d)
        assert np.allclose(a.probs, T.pbar(every), atol=1e-12)
        for o in T.ORDERS:
            assert np.allclose(a.raw[o], T.pbar([per[s][q.qkey].canon_probs[o][0] for s in range(2)]), atol=1e-12)
    for s in singles + [ens]:
        s.close()


def test_forbidden_gpu_and_bad_inits(ckpts, tmp_path, monkeypatch):
    import dataclasses
    from habitus.dm.gpu import ForbiddenDevice
    st = dataclasses.replace(make_settings(tmp_path), forbidden_gpus=(2,))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    with pytest.raises(ForbiddenDevice):
        LA.LayaDecisionModel(ckpts[0], device="cuda:2", settings=st)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(ForbiddenDevice):
        LA.LayaDecisionModel(ckpts[0], device="cuda:0", settings=st)
    with pytest.raises(FileNotFoundError):
        LA.resolve_init(str(tmp_path / "nonexistent"), st)
    rec_dir = tmp_path / "mqrun"
    shutil.copytree(ckpts[0], rec_dir)
    (rec_dir / LA.RUN_CONFIG).write_text(json.dumps({"layout": "mq", "fold": FOLD}))
    with pytest.raises(ValueError, match="mq"):
        LA.resolve_init(str(rec_dir), st)


@pytest.mark.local_assets
def test_registry_builds_the_zero_shot_arm(tmp_path):
    td = typed_decisions_dir()
    need_weights(td is not None and os.path.isfile(os.path.join(td, LA.WEIGHTS)), "laya_hub/typed-decisions weights")
    from habitus.dm import get_dm
    st = make_settings(tmp_path)
    dm = get_dm("laya-typed-zs", settings=st)
    assert isinstance(dm, LA.LayaDecisionModel) and dm.cache_only and dm.device == "cpu"
    assert dm.tag == LA.hbl_tag("typed-decisions", LA.model_sha256(td)) and dm.members[0].dir == td
    assert dm.kw == {"max_len": LA.MAX_LEN, "head_max_len": LA.HEAD_MAX_LEN}
    dm.close()


def test_hygiene_of_the_laya_modules():
    pat = re.compile(r"(?i)g[t]\(|ground[_]truth|host[_]labels|g[t]_")         # char-classed: no self-match
    train = SRC.parent / "train"
    for p in (SRC / "laya.py", SRC / "laya_mq.py", train / "laya.py", train / "laya_mq.py"):
        src = p.read_text(encoding="utf-8")
        assert not [ln for ln in src.splitlines() if pat.search(ln)], p
        assert "Label-free" in src, p
        assert not re.search(r"sys\.path\.(insert|append)|^\s*(from|import) laya\b", src, flags=re.M), p
        assert not re.search(r"^\s*(from|import) habitus\.eval", src, flags=re.M), p
        assert not re.search(r"/h[o]me/|/D[A]TA|/t[m]p/", src), p
    head = (SRC / "laya.py").read_text(encoding="utf-8").split('"""')[0]
    assert "Apache License, Version 2.0" in head and "Convai Innovations" in head and "Laya 0.3.5" in head
    assert "nothing is ever truncated" in head and "temperature buffer is reset to 1" in head


# --------------------------------------------------------------------------- #
# Provenance (Laya 0.3.5 source under paths.laya_src; HABITUS_PROVENANCE=1)
# --------------------------------------------------------------------------- #
def _laya_batch(LC, tok, parts, qtype="choice"):
    items = []
    for p in parts:
        ids, mk = LC.build_sequence(tok, p.state, {"t": qtype, "ins": p.question, "crit": {x: None for x in p.options}},
                                    max_len=1 << 16, head_max_len=1 << 16)
        items.append({"ids": ids, "markers": mk, "qtype": LC.QTYPES[qtype]})
    return LC.collate_items([items], tok.pad_token_id)


@pytest.mark.provenance
@PROVENANCE
def test_la_row_is_laya_build_sequence_without_cuts(tok):
    LC = laya_common()
    assert LC.QTYPES == LA.QTYPES
    qs = typed(make_questions(4))
    for q in qs:
        for f in q.forms:
            p, r = LA.form_parts(f), LA.seq_form(tok, f, q.qtype)
            ids, mk = LC.build_sequence(tok, p.state, {"t": "choice", "ins": p.question,
                                                       "crit": {x: None for x in p.options}},
                                        max_len=1 << 16, head_max_len=1 << 16)
            if max(LA.option_lengths(r)) <= LA.LAYA_OPTION_CUT:
                assert list(r.ids) == ids and list(r.markers) == mk          # Laya's layout, token for token
            else:                                                            # Laya cuts the option at 48
                assert len(ids) < len(r.ids)
    assert any(max(LA.option_lengths(LA.seq_form(tok, q.forms[0], q.qtype))) > LA.LAYA_OPTION_CUT for q in qs)


@pytest.mark.provenance
@PROVENANCE
def test_vendored_model_is_laya_common_bit_for_bit(tok):
    from transformers import AutoModel
    LC = laya_common()
    torch.manual_seed(3)
    lm = LC.DecisionModel(AutoModel.from_config(tiny_config(tok), attn_implementation="sdpa")).eval()
    with torch.no_grad():
        for p in lm.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    hm = LA.DecisionModelHBL(AutoModel.from_config(tiny_config(tok), attn_implementation="sdpa")).eval()
    assert list(hm.state_dict()) == list(lm.state_dict())
    hm.load_state_dict(lm.state_dict(), strict=True)
    qs = [q for q in typed(make_questions(3))
          if max(LA.option_lengths(LA.seq_form(tok, q.forms[0], q.qtype))) <= LA.LAYA_OPTION_CUT]
    parts = [LA.form_parts(f) for q in qs for f in q.forms]
    b = _laya_batch(LC, tok, parts)
    with torch.no_grad():
        z1, _act = lm(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
        z2 = hm(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
        z3 = hm(**LA.collate([LA.seq_parts(tok, p) for p in parts], tok.pad_token_id))
    assert torch.equal(z1, z2) and torch.equal(z2, z3)
    assert (z1 == LA.MASK_FILL).any() and torch.isfinite(z1).all()


@pytest.mark.provenance
@PROVENANCE
def test_real_laya_weights_reproduce_laya_logits(tok):
    td = typed_decisions_dir()
    need(td is not None and os.path.isfile(os.path.join(td, LA.WEIGHTS)), "laya_hub/typed-decisions weights")
    from safetensors.torch import load_file
    LC = laya_common()
    hm = LA.load_model(td)
    cfg = LA.laya_config(td)
    lm = LC.DecisionModel(LA.build_model(td, empty=True).encoder, cfg.get("head_layers", 2),
                          len(cfg.get("act_costs", {})) + 1)
    lm.load_state_dict(load_file(os.path.join(td, LA.WEIGHTS)), strict=True)
    lm.eval()
    qs = [q for q in typed(make_questions(2))
          if max(LA.option_lengths(LA.seq_form(tok, q.forms[0], q.qtype))) <= LA.LAYA_OPTION_CUT]
    parts = [LA.form_parts(f) for q in qs[:4] for f in q.forms]
    b = _laya_batch(LC, tok, parts)
    with torch.no_grad():
        z1, _ = lm(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
        z2 = hm(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
    assert torch.equal(z1, z2)
