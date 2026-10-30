# DecisionModelHBL below is a modified copy of laya.common.DecisionModel from Laya 0.3.5
# (laya/common.py, lines 89-136; Convai Innovations; licensed under the Apache License, Version 2.0,
# http://www.apache.org/licenses/LICENSE-2.0). The changes, as Apache-2.0 section 4(b) asks:
#   1. forward returns the masked marker logits only; act_head stays in the state dict (strict
#      checkpoint loads) but is never run;
#   2. optional position_ids / enc_masks / head_mask inputs (the MQ layout); None is Laya's own call;
#   3. qtype may be per position ([B, L]) as well as per row ([B]);
#   4. the temperature buffer is reset to 1 on load; the checkpoint's temperature tables are unused
#      (every answer is read at T = 1);
#   5. nothing is ever truncated: build_sequence's silent option / question / state cuts are replaced by
#      a refusal (LengthError) of the whole row.
"""The 'laya' decision models, layout L-a: an encoder with one [MASK] scorer per option.

One row per (question, form):

    [CLS] tok("<qtype> question: " + question) [SEP] ( [MASK] tok(" " + option_i) )_i [SEP] tok(state) [SEP]

the layout of laya.common.build_sequence (Laya 0.3.5), with the options in the form's LETTER order and
qtype = TypedQuestion.qtype ('choice' for habit questions, 'score' for qv, 'noul' for qt / qa). Every
text segment goes through habitus.state.sanitize.sanitize_str (a no-op on states and questions that
were already sanitised); a special id of the Laya tokenizer outside the structural positions ([CLS],
the two head [SEP]s, the [MASK] markers, the closing [SEP]) raises ValueError.

Nothing is ever cut. A row whose head (question + options) exceeds HEAD_MAX_LEN = 256 tokens or whose
length exceeds MAX_LEN = 2048 tokens is REFUSED: `refused()` names it, `decide()` leaves it out, and
training raises on it.

Readout: softmax over the row's marker logits at T = 1 (no off-option mass, so cov = 1.0). A model is
one checkpoint (a Laya init, or a habitus.train.laya run directory) or a seed ensemble of run
directories; a Decision carries one distribution per member, in member order. Each member keeps its
own answer cache, tag 'hbl-<init label>-<12 hex of sha256(model.safetensors)>', key = sha1 of the row's
int32 little-endian ids, record {"k", "z" (marker logits, 5 dp), "lse", "n", "cov" = 1.0} (the
dm.cache.AnswerCache fields; only z is read back); the first record of a key wins. Models load on the
first cache miss only; cache_only=True raises CacheMiss before any model is loaded.

Label-free: this module renders rows from the forms handed to it, loads weights and reads / writes
its own answer caches; it opens no card, link, score or label file and imports no metric code.
"""

import hashlib
import json
import os
import re
import time
from contextlib import nullcontext
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn

from habitus.dm.base import (ANSWER_HEAD, LETTERS, OPTIONS_HEAD, QUESTION_HEAD, STATE_HEAD, THINK_SUFFIX,
                             CacheMiss, Decision, canonical, unique_questions)
from habitus.state.sanitize import sanitize_str, special_ids_laya

FAMILY = "laya"
LAYOUT = "la"                         # one row per (question, form); "mq" = habitus.dm.laya_mq
DEFAULT_INIT = "typed-decisions"      # settings.paths.laya_hub/typed-decisions (convaiinnovations/laya@1c5edc17)
INIT_NAMES = ("typed-decisions", "base")
MAX_LEN = 2048                        # row budget (Laya typed-decisions: 1024)
HEAD_MAX_LEN = 256                    # question + [MASK]-led options (Laya's head_max_len budget)
LAYA_OPTION_CUT = 48                  # build_sequence's silent per-option cut: never applied, only reported
MAX_OPTIONS = 4
QTYPES = {"choice": 0, "score": 1, "noul": 2}      # laya.common.QTYPES
TRAIN_QTYPE = "choice"                # the Laya type of every habit training item (recipe key "qtype")
MASK_FILL = -1e4                      # laya.common: logits.masked_fill(~marker_mask, -1e4)
TAG_PREFIX = "hbl-"
HASH_CHARS = 12
RELEASE_TOKENIZER_DIR = "laya-typed-decisions"   # the Laya tokenizer's copy under settings.paths.tokenizers
RUN_CONFIG = "habitus.json"           # habitus.train.laya's run record
LAYA_CONFIG = "rl_agent_config.json"  # Laya's checkpoint config
WEIGHTS = "model.safetensors"
BATCH_ROWS = 64                       # rows per inference forward
MAX_TOKENS = 32768                    # padded tokens per inference forward
SEGMENT_CACHE_MAX = 200_000
FLUSH = 500                           # pending answer records appended to the overlay at a time

# the Qwen3.5 chat-template frame of a rendered habit prompt (enable_thinking=False)
IM_START, IM_END = "<|im_start|>", "<|im_end|>"
USER_OPEN = IM_START + "user\n"
ASSISTANT_TAIL = IM_END + "\n" + IM_START + "assistant\n<think>\n\n" + THINK_SUFFIX


def _log(msg, tag="habitus.dm.laya"):
    from habitus.log import log
    log(msg, tag)


def _settings(settings=None):
    if settings is not None:
        return settings
    from habitus import config
    s = config.settings
    return s() if callable(s) and not hasattr(s, "paths") else s


class PromptError(ValueError):
    """A text that is not a rendered habit prompt (or does not round-trip)."""


class LengthError(ValueError):
    """A row whose head or total length exceeds the budget: refused, never truncated."""


# --------------------------------------------------------------------------- #
# Row parts: the same three pieces the jev backbone reads
# --------------------------------------------------------------------------- #
class Parts(NamedTuple):
    state: str
    question: str
    options: tuple       # option texts in LETTER order (A first)


def form_parts(form):
    """Parts of one dm.base.Form."""
    return Parts(form.state, form.question, tuple(form.options))


def user_text(parts):
    """The user turn of these parts (byte-identical to Form.user())."""
    letters = LETTERS[:len(parts.options)]
    return (STATE_HEAD + parts.state + QUESTION_HEAD + parts.question + OPTIONS_HEAD
            + "\n".join(f"{L}: {t}" for L, t in zip(letters, parts.options)) + ANSWER_HEAD + ", ".join(letters))


def _system_h():
    from habitus.questions.habit.text import SYSTEM_H
    return SYSTEM_H


def split_prompt(text, system=None, tok=None):
    """Parts(state, question, options in letter order) of one rendered prompt (a training item's
    prompt_text). Strict: the system turn must be `system` (default: the habit system prompt SYSTEM_H),
    the frame the Qwen chat template's with an empty think block, the option lines "A: ..", "B: .."
    (2..4), the answer line their letters; no piece may hold a newline; and the parts must re-assemble
    to the user turn exactly. With `tok` (a Qwen tokenizer) the prompt is also re-rendered through its
    chat template and must equal `text`. Raises PromptError otherwise."""
    system = _system_h() if system is None else system
    head = IM_START + "system\n" + system + IM_END + "\n" + USER_OPEN
    if not text.startswith(head):
        raise PromptError("not a habit prompt: the system turn / user-turn opening differ")
    if not text.endswith(ASSISTANT_TAIL) or len(text) < len(head) + len(ASSISTANT_TAIL):
        raise PromptError("not a habit prompt: no empty-think assistant tail")
    user = text[len(head):len(text) - len(ASSISTANT_TAIL)]
    if not user.startswith(STATE_HEAD):
        raise PromptError("user turn does not start with the state header")
    i = user.find(QUESTION_HEAD, len(STATE_HEAD))
    j = user.find(OPTIONS_HEAD, i + len(QUESTION_HEAD)) if i >= 0 else -1
    k = user.rfind(ANSWER_HEAD)
    if i < 0 or j < 0 or k < j + len(OPTIONS_HEAD):
        raise PromptError("user turn lacks the question / options / answer markers")
    state, question = user[len(STATE_HEAD):i], user[i + len(QUESTION_HEAD):j]
    lines = user[j + len(OPTIONS_HEAD):k].split("\n")
    if not 2 <= len(lines) <= MAX_OPTIONS:
        raise PromptError(f"{len(lines)} option lines (2..{MAX_OPTIONS})")
    letters = LETTERS[:len(lines)]
    opts = []
    for L, line in zip(letters, lines):
        if not line.startswith(L + ": ") or len(line) == len(L) + 2:
            raise PromptError(f"option line {line[:40]!r} is not '{L}: <text>'")
        opts.append(line[len(L) + 2:])
    if user[k + len(ANSWER_HEAD):] != ", ".join(letters):
        raise PromptError("answer line does not list the offered letters")
    if "\n" in state or "\n" in question or not state or not question:
        raise PromptError("state / question text is empty or spans lines")
    parts = Parts(state, question, tuple(opts))
    if user_text(parts) != user:
        raise PromptError("parsed parts do not re-assemble to the user turn")
    if tok is not None:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        again = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if again != text:
            raise PromptError("parsed parts do not re-render to the prompt through the chat template")
    return parts


def parts_from_prompt(text, system=None, tok=None):
    """split_prompt: the row parts of a training item's rendered prompt."""
    return split_prompt(text, system, tok)


# --------------------------------------------------------------------------- #
# Rows (layout L-a)
# --------------------------------------------------------------------------- #
class Row(NamedTuple):
    ids: tuple           # token ids of the row
    markers: tuple       # position of each option's [MASK] (letter order)
    qtype: int
    head_len: int        # question + options (Laya's head_max_len budget)
    sep: int             # position of the [SEP] that closes the options (the state starts after it)


def option_lengths(row):
    """Tokens of each option (its [MASK] excluded), letter order."""
    ends = list(row.markers[1:]) + [row.sep]
    return tuple(e - m - 1 for m, e in zip(row.markers, ends))


def row_key(row):
    """Answer-cache key of a row: sha1 of its int32 little-endian ids (the markers follow from the ids)."""
    return hashlib.sha1(np.asarray(row.ids, dtype="<i4").tobytes()).hexdigest()


_SPECIAL = {}


def special_ids(tok):
    """habitus.state.sanitize.special_ids_laya(tok), memoised per tokenizer."""
    s = _SPECIAL.get(id(tok))
    if s is None:
        s = _SPECIAL[id(tok)] = special_ids_laya(tok)
    return s


class SegmentTokens:
    """Memoised tokenisation of text segments (add_special_tokens=False): a state is tokenised once per
    text, a question / option text once, whatever the number of rows that carry it. Tokenising the
    segments separately is exactly what laya build_sequence does."""

    def __init__(self, tok, maxsize=SEGMENT_CACHE_MAX):
        self.tok, self.maxsize, self.memo = tok, int(maxsize), {}

    def warm(self, texts):
        todo = sorted({t for t in texts if t not in self.memo})
        if len(self.memo) + len(todo) > self.maxsize:
            self.memo.clear()
        for s in range(0, len(todo), 1024):
            chunk = todo[s:s + 1024]
            for t, ids in zip(chunk, self.tok(chunk, add_special_tokens=False)["input_ids"]):
                self.memo[t] = list(ids)

    def __call__(self, text):
        ids = self.memo.get(text)
        if ids is None:
            if len(self.memo) >= self.maxsize:
                self.memo.clear()
            ids = self.memo[text] = list(self.tok(text, add_special_tokens=False)["input_ids"])
        return ids


def check_qtype(qtype):
    if qtype not in QTYPES:
        raise ValueError(f"qtype {qtype!r} not in {sorted(QTYPES)}")
    return qtype


def segments(parts, qtype=TRAIN_QTYPE):
    """The sanitised text segments of a row: (question segment, option segments, state segment)."""
    return (f"{qtype} question: " + sanitize_str(parts.question),
            tuple(" " + sanitize_str(o) for o in parts.options), sanitize_str(parts.state))


def seq_parts(tok, parts, qtype=TRAIN_QTYPE, max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN, seg=None):
    """The L-a row of `parts`: [CLS] q [SEP] ([MASK] option)* [SEP] state [SEP]. Raises LengthError when
    question + options > head_max_len or the row > max_len (never cuts), ValueError when a special id
    of the Laya tokenizer survives outside the structural positions."""
    check_qtype(qtype)
    if not 2 <= len(parts.options) <= MAX_OPTIONS:
        raise ValueError(f"{len(parts.options)} options (2..{MAX_OPTIONS})")
    enc = seg if seg is not None else (lambda s: tok(s, add_special_tokens=False)["input_ids"])
    q_seg, o_segs, v_seg = segments(parts, qtype)
    q_ids = list(enc(q_seg))
    opt_ids = [[tok.mask_token_id] + list(enc(o)) for o in o_segs]
    head_len = len(q_ids) + sum(len(o) for o in opt_ids)
    if head_len > head_max_len:
        raise LengthError(f"head of {head_len} tokens > head_max_len {head_max_len} (option tokens "
                          f"{[len(o) - 1 for o in opt_ids]}): refused, never cut")
    ids = [tok.cls_token_id] + q_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    sep = len(ids)
    ids.append(tok.sep_token_id)
    ids.extend(enc(v_seg))
    ids.append(tok.sep_token_id)
    if len(ids) > max_len:
        raise LengthError(f"row of {len(ids)} tokens > max_len {max_len} (head {head_len}, state "
                          f"{len(ids) - sep - 2}): refused, never cut")
    structural = {0, len(q_ids) + 1, sep, len(ids) - 1, *markers}
    bad = [(i, t) for i, t in enumerate(ids) if t in special_ids(tok) and i not in structural]
    if bad:
        raise ValueError(f"special token ids {bad[:5]} outside the structural positions (sanitiser gap)")
    return Row(tuple(ids), tuple(markers), QTYPES[qtype], head_len, sep)


def seq_form(tok, form, qtype, **kw):
    """The serve row of one Form asked as Laya type `qtype`."""
    return seq_parts(tok, form_parts(form), qtype=qtype, **kw)


def seq_item(tok, item, system=None, qtype=TRAIN_QTYPE, **kw):
    """The training row of one habit Item (its prompt parsed by split_prompt; option order = the
    item's variant, so Item.target is aligned with the markers)."""
    parts = split_prompt(item.prompt_text, system)
    if len(parts.options) != len(item.letters):
        raise PromptError(f"item {item.qkey}: {len(parts.options)} options for letters {item.letters}")
    return seq_parts(tok, parts, qtype=qtype, **kw)


def refusal(tok, q, seg=None, max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN):
    """The LengthError message of TypedQuestion `q` (first over-budget form), or None when every form
    fits. A question is refused, never cut."""
    for f in q.forms:
        try:
            seq_form(tok, f, q.qtype, seg=seg, max_len=max_len, head_max_len=head_max_len)
        except LengthError as e:
            return str(e)
    return None


def check_identity(tok, item, q, **kw):
    """Problems (empty when clean) of the train / serve identity of one (item, question) pair: the row
    from the item's prompt must be the row of the question's form named by the item's variant."""
    out = []
    if item.qkey != q.qkey:
        out.append(f"item {item.qkey} is not question {q.qkey}")
    forms = [f for f in q.forms if f.name == item.variant]
    if not forms:
        return out + [f"{q.qkey}: no form {item.variant!r}"]
    try:
        a, b = seq_item(tok, item, qtype=q.qtype, **kw), seq_form(tok, forms[0], q.qtype, **kw)
    except ValueError as e:
        return out + [f"{item.qkey} ({item.variant}): {e}"]
    if a != b:
        out.append(f"{item.qkey} ({item.variant}): the item's row differs from the question's row")
    return out


# --------------------------------------------------------------------------- #
# The vendored Laya decision model (see the header for the Apache-2.0 change list)
# --------------------------------------------------------------------------- #
class DecisionModelHBL(nn.Module):
    """laya.common.DecisionModel (Laya 0.3.5): bidirectional encoder + typed decision head, same
    parameter names (strict checkpoint loads). forward returns the masked marker logits only.
    position_ids / enc_masks / head_mask: the MQ hooks (None = Laya's own call, bit for bit)."""

    def __init__(self, encoder, head_layers=2, n_act=2, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        nhead = max(1, d // 64)
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False) if head_layers > 0 else None
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
        self.register_buffer("temperature", torch.ones(3))

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, position_ids=None, enc_masks=None,
                head_mask=None, detach_encoder=False):
        am = attention_mask if enc_masks is None else enc_masks
        if position_ids is None:
            h = self.encoder(input_ids=input_ids, attention_mask=am).last_hidden_state
        else:
            h = self.encoder(input_ids=input_ids, attention_mask=am, position_ids=position_ids).last_hidden_state
        if detach_encoder:
            h = h.detach()
        te = self.type_emb(qtype)
        h = h + (te[:, None, :] if qtype.dim() == 1 else te)      # [B] per row; [B, L] per position (MQ blocks)
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                if head_mask is None:
                    h = layer(h, src_key_padding_mask=pad)
                else:
                    h = layer(h, src_mask=head_mask, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        return logits.masked_fill(~marker_mask, MASK_FILL)


def padding_masks(attention_mask, position_ids, sliding_window):
    """The {"full_attention", "sliding_attention"} dict (bool [B, 1, L, L], True = attend) that is
    equivalent to a padding mask: every query attends to every real key; local layers also need
    |position_q - position_k| <= sliding_window, measured on POSITION IDS. The base of a block mask (MQ)."""
    keys = attention_mask.bool()[:, None, :]
    full = keys.expand(-1, attention_mask.shape[1], -1)
    masks = {"full_attention": full[:, None]}
    if sliding_window is not None:
        near = (position_ids[:, :, None] - position_ids[:, None, :]).abs() <= int(sliding_window)
        masks["sliding_attention"] = (full & near)[:, None]
    return masks


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def inits(settings=None):
    """{init name: checkpoint directory}: "typed-decisions" = <laya_hub>/typed-decisions, "base" =
    <laya_hub> (settings.paths.laya_hub)."""
    hub = Path(_settings(settings).paths.laya_hub)
    return {"typed-decisions": str(hub / "typed-decisions"), "base": str(hub)}


def _label(text):
    return re.sub(r"[^A-Za-z0-9_.+-]", "_", str(text))


def read_run_record(d):
    """habitus.json of a checkpoint directory, or {}."""
    rc = os.path.join(d, RUN_CONFIG)
    if not os.path.isfile(rc):
        return {}
    with open(rc, encoding="utf-8") as f:
        return json.load(f)


def check_checkpoint(d, what="Laya checkpoint"):
    for need in (LAYA_CONFIG, WEIGHTS, os.path.join("encoder", "config.json"), "tokenizer"):
        if not os.path.exists(os.path.join(d, need)):
            raise FileNotFoundError(f"{d}: not a {what} ({need} missing)")


def resolve_init(init, settings=None):
    """(checkpoint directory, init label) of `init`: "typed-decisions" | "base" (laya_hub), a Laya
    checkpoint directory (rl_agent_config.json + model.safetensors + encoder/ + tokenizer/), or a
    habitus.train.laya run directory (the same files + habitus.json; label = its init; layout must be
    L-a)."""
    if init in INIT_NAMES:
        d, label = inits(settings)[init], init
    else:
        d = os.path.abspath(str(init))
        label = os.path.basename(d.rstrip(os.sep))
        rec = read_run_record(d)
        if rec:
            if rec.get("layout", LAYOUT) != LAYOUT:
                raise ValueError(f"{d}: a {rec.get('layout')!r} run, not layout {LAYOUT!r}")
            label = rec.get("init") or label
    check_checkpoint(d)
    return d, _label(label)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def model_sha256(model_dir):
    """sha256 of the checkpoint's weights file (a run directory records it; checked when present)."""
    return file_sha256(os.path.join(model_dir, WEIGHTS))


def hbl_tag(label, digest):
    return f"{TAG_PREFIX}{label}-{digest[:HASH_CHARS]}"


def member_cache_tags(cache_tag, n):
    """Per-member answer-cache namespaces under an explicit cache_tag: the tag itself for one member,
    '<cache_tag>.m<i>' for member i of several (None: each member's own weight-derived tag)."""
    if not cache_tag:
        return [None] * n
    return [cache_tag] if n == 1 else [f"{cache_tag}.m{i}" for i in range(n)]


def ensemble_tag(prefix, members):
    """The tag of a model of several members: '<prefix><label or mix>-<12 hex of sha256 of the members'
    weight hashes, newline-joined, in member order>'."""
    labels = sorted({m.label for m in members})
    joined = "\n".join(m.sha256 for m in members)
    return f"{prefix}{labels[0] if len(labels) == 1 else 'mix'}-" \
           f"{hashlib.sha256(joined.encode()).hexdigest()[:HASH_CHARS]}"


def laya_config(model_dir):
    with open(os.path.join(model_dir, LAYA_CONFIG), encoding="utf-8") as f:
        return json.load(f)


def load_tokenizer(model_dir):
    """The checkpoint's ModernBERT tokenizer (<model_dir>/tokenizer)."""
    return load_tokenizer_dir(os.path.join(model_dir, "tokenizer"))


def load_tokenizer_dir(d):
    """A Laya ModernBERT tokenizer directory (tokenizer.json + tokenizer_config.json), read-only: Laya's
    _fix_tokenizer_config would rewrite the file, so a config that needs that fix is refused instead."""
    from transformers import AutoTokenizer
    with open(os.path.join(d, "tokenizer_config.json"), encoding="utf-8") as f:
        tc = json.load(f)
    if tc.get("tokenizer_class") in (None, "TokenizersBackend") or isinstance(tc.get("extra_special_tokens"), list):
        raise ValueError(f"{d}: tokenizer_config.json needs Laya's _fix_tokenizer_config (not applied in place)")
    tok = AutoTokenizer.from_pretrained(d, local_files_only=True)
    for name in ("cls_token_id", "sep_token_id", "mask_token_id", "pad_token_id"):
        if getattr(tok, name) is None:
            raise ValueError(f"{d}: tokenizer has no {name}")
    return tok


def build_model(model_dir, empty=False):
    """A DecisionModelHBL with the checkpoint's encoder config (sdpa) and head size
    (laya.common.build_model): encoder AutoModel.from_config, n_act = len(act_costs) + 1. empty=True
    skips the random initialisation (transformers no_init_weights), for a caller that loads EVERY
    parameter next (load_model: strict=True)."""
    from transformers import AutoConfig, AutoModel
    from transformers.initialization import no_init_weights
    cfg = laya_config(model_dir)
    ecfg = AutoConfig.from_pretrained(os.path.join(model_dir, "encoder"), local_files_only=True)
    with no_init_weights() if empty else nullcontext():
        enc = AutoModel.from_config(ecfg, attn_implementation="sdpa")
        return DecisionModelHBL(enc, cfg.get("head_layers", 2), len(cfg.get("act_costs", {})) + 1)


def load_model(model_dir, device="cpu"):
    """DecisionModelHBL with the checkpoint's weights (strict=True), temperature buffer reset to 1, eval
    mode, fp32 weights (Laya trains and serves fp32 weights under bf16 autocast)."""
    from safetensors.torch import load_file
    model = build_model(model_dir, empty=True)
    model.load_state_dict(load_file(os.path.join(model_dir, WEIGHTS)), strict=True)
    with torch.no_grad():
        model.temperature.fill_(1.0)
    return model.to(device).eval()


# --------------------------------------------------------------------------- #
# Batches and forwards
# --------------------------------------------------------------------------- #
def collate(rows, pad_id, device="cpu"):
    """laya.common.collate_items for Rows: right-padded ids, attention mask, marker positions / mask,
    qtype."""
    n, L = len(rows), max(len(r.ids) for r in rows)
    kmax = max(len(r.markers) for r in rows)
    ids = torch.full((n, L), int(pad_id), dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    for i, r in enumerate(rows):
        ids[i, :len(r.ids)] = torch.tensor(r.ids, dtype=torch.long)
        att[i, :len(r.ids)] = 1
        mpos[i, :len(r.markers)] = torch.tensor(r.markers, dtype=torch.long)
        mmask[i, :len(r.markers)] = True
    qtype = torch.tensor([r.qtype for r in rows], dtype=torch.long)
    return {k: v.to(device) for k, v in (("input_ids", ids), ("attention_mask", att), ("marker_pos", mpos),
                                         ("marker_mask", mmask), ("qtype", qtype))}


def length_batches(lengths, batch_rows=BATCH_ROWS, max_tokens=MAX_TOKENS):
    """Index batches of length-sorted rows: <= batch_rows rows and <= max_tokens padded tokens each (a
    single row longer than max_tokens forms its own batch). Every index appears exactly once."""
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    out, cur = [], []
    for i in order:
        if cur and (len(cur) >= batch_rows or (len(cur) + 1) * lengths[i] > max_tokens):
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


def _autocast(device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda"))


@torch.no_grad()
def _forward_rows(model, rows, pad_id, device):
    b = collate(rows, pad_id, device)
    with _autocast(device):
        z = model(**b)
    return z.float().cpu().numpy()


def _forward_safe(model, rows, pad_id, device):
    """_forward_rows, halving the batch on CUDA OOM (the halves run outside the except block so the
    failed forward's activations are freed first)."""
    try:
        return [np.asarray(z[:len(r.markers)], dtype=np.float64)
                for z, r in zip(_forward_rows(model, rows, pad_id, device), rows)]
    except torch.cuda.OutOfMemoryError:
        if len(rows) == 1:
            raise
    torch.cuda.empty_cache()
    h = len(rows) // 2
    return _forward_safe(model, rows[:h], pad_id, device) + _forward_safe(model, rows[h:], pad_id, device)


def row_logits(model, rows, pad_id, device="cpu", batch_rows=BATCH_ROWS, max_tokens=MAX_TOKENS):
    """Marker logits (1-D float64, letter order) of every row, in input order; length-sorted batches."""
    model.eval()
    out = [None] * len(rows)
    for b in length_batches([len(r.ids) for r in rows], batch_rows, max_tokens):
        for i, z in zip(b, _forward_safe(model, [rows[i] for i in b], pad_id, device)):
            out[i] = z
    return out


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


def cache_record(z):
    """The answer-cache record of one forward: {"z": marker logits rounded to 5 dp, "lse": their
    logsumexp (5 dp), "n": number of markers, "cov": 1.0} (the AnswerCache record fields; the readout
    uses z only)."""
    zr = [round(float(x), 5) for x in z]
    a = np.asarray(zr, dtype=np.float64)
    lse = float(a.max() + np.log(np.exp(a - a.max()).sum()))
    return {"z": zr, "lse": round(lse, 5), "n": len(zr), "cov": 1.0}


def valid_record(rec):
    try:
        return len(rec["z"]) == int(rec["n"]) and all(np.isfinite(float(x)) for x in rec["z"])
    except (KeyError, TypeError, ValueError):
        return False


# --------------------------------------------------------------------------- #
# Members (one checkpoint each) and their answer caches
# --------------------------------------------------------------------------- #
def open_cache(tag, ds, settings=None, read_only=False):
    """The AnswerCache of one member tag on host `ds` (dm.cache.cache_paths: the shipped base, when
    present, and the overlay <runs>/answers/<tag>/<ds>.jsonl)."""
    from habitus.dm.cache import AnswerCache, cache_paths
    base, overlay = cache_paths(tag, ds, _settings(settings))
    return AnswerCache(tag, ds, base=base, overlay=overlay, read_only=read_only)


class Member:
    """One checkpoint of a Laya model: directory, label, weight hash, run record, tag, answer cache and
    lazily loaded model. `resolve` maps an init / run argument to (directory, label); `tag_of` maps
    (label, weight sha256) to the tag; `cache_tag` replaces that tag (a scratch answer-cache namespace)."""

    def __init__(self, init, device, ds=None, settings=None, read_only=False, model=None, resolve=None,
                 tag_of=hbl_tag, verbose=False, cache_tag=None):
        resolve = resolve or (lambda x: resolve_init(x, settings))
        self.dir, self.label = resolve(init)
        self.sha256 = model_sha256(self.dir)
        self.run_record = read_run_record(self.dir)
        rec_sha = self.run_record.get("model_sha256")
        if rec_sha and rec_sha != self.sha256:
            raise ValueError(f"{self.dir}: weights sha256 {self.sha256[:12]} != recorded {rec_sha[:12]}")
        self.tag = cache_tag or tag_of(self.label, self.sha256)
        self.device, self.model, self.verbose = device, model, verbose
        self.answers = {}
        self.cache = open_cache(self.tag, ds, settings, read_only) if ds else None
        self.stats = {"forwarded": 0, "seconds": 0.0}

    def lookup(self, keys):
        """Fill self.answers from the cache for `keys`; returns the keys still missing."""
        missing = []
        for k in keys:
            if k in self.answers:
                continue
            rec = self.cache.get(k) if self.cache is not None else None
            if rec is not None and valid_record(rec):
                self.answers[k] = {"z": [float(x) for x in rec["z"]], "n": int(rec["n"])}
            else:
                missing.append(k)
        return missing

    def store(self, key, rec):
        self.answers[key] = rec
        if self.cache is not None:
            self.cache.put(key, rec)
            if self.cache.pending >= FLUSH:
                self.cache.flush()

    def load(self):
        if self.model is None:
            t0 = time.time()
            self.model = load_model(self.dir, self.device)
            if self.verbose:
                _log(f"loaded {self.dir} on {self.device} in {time.time() - t0:.1f}s", f"habitus.dm.laya {self.tag}")
        return self.model

    def flush(self):
        if self.cache is not None:
            self.cache.flush()

    def close(self):
        if self.cache is not None:
            self.cache.flush()
            self.cache.close()
            self.cache = None


def check_member_tokenizers(members):
    toks = {file_sha256(os.path.join(m.dir, "tokenizer", "tokenizer.json")) for m in members}
    if len(toks) != 1:
        raise ValueError("the members of one model must share their tokenizer")


def check_member_budgets(members, max_len, head_max_len):
    """A trained member's recipe fixes its budgets and its Laya type rule: a different budget, or a
    recipe qtype other than 'choice' (the dropped qt2 ablation), is refused."""
    for m in members:
        rec = m.run_record.get("recipe") or {}
        if rec.get("qtype", TRAIN_QTYPE) != TRAIN_QTYPE:
            raise ValueError(f"{m.dir}: recipe qtype {rec.get('qtype')!r} is not supported (only {TRAIN_QTYPE!r})")
        for name, want in (("max_len", max_len), ("head_max_len", head_max_len)):
            if int(rec.get(name, want)) != int(want):
                raise ValueError(f"{m.dir}: trained with {name} {rec.get(name)}, model asked for {want}")


def check_fold(members, fold):
    """Fold safety: every trained member's run record must name `fold` (a model trained on the scored
    host's own days must never score it). Laya inits (no run record) pass."""
    for m in members:
        if m.run_record and m.run_record.get("fold") != fold:
            raise ValueError(f"{m.dir} was trained for fold {m.run_record.get('fold')!r}, not {fold!r}")


def dedupe_questions(qs):
    """{qkey: question} in first-appearance order (dm.base.unique_questions: a repeated qkey must carry
    the same forms), every qtype a Laya type."""
    out = unique_questions(qs)
    for q in out.values():
        check_qtype(q.qtype)
    return out


# --------------------------------------------------------------------------- #
# The decision model
# --------------------------------------------------------------------------- #
class LayaDecisionModel:
    """dm.base.DecisionModel, family 'laya', layout 'la': one row per (question, form), softmax over the
    row's [MASK] scorer logits at T = 1, one distribution per member.

    runs: None (zero-shot from `init`), one run directory, or several (a seed ensemble; Decision lists
    the members in this order). init: "typed-decisions" | "base" | a checkpoint directory (zero-shot
    only). ds: the host whose answer caches are used (None = in memory only). cache_only: never load a
    model; a missing answer raises CacheMiss. fold: when given, every trained member must have been
    trained for it. ensemble: the <base>.ensemble.json record of the seed runs (report temperatures
    only; serving is at T = 1). cache_tag: an explicit answer-cache namespace instead of the
    weight-derived tags (e.g. a scratch smoke run; member_cache_tags). preload: built models aligned
    with the members (tests)."""

    family = FAMILY
    layout = LAYOUT

    def __init__(self, runs=None, init=DEFAULT_INIT, device="cpu", batch_rows=BATCH_ROWS, max_tokens=MAX_TOKENS,
                 max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN, cache_only=False, ds=None, fold=None, ensemble=None,
                 cache_tag=None, settings=None, verbose=False, preload=None):
        from habitus.dm.gpu import check_device
        self.settings = _settings(settings)
        check_device(device, self.settings)
        models = [init] if runs is None else ([runs] if isinstance(runs, (str, os.PathLike)) else list(runs))
        if not models:
            raise ValueError("LayaDecisionModel needs at least one model")
        preload = list(preload) if preload is not None else [None] * len(models)
        if len(preload) != len(models):
            raise ValueError("preload must align with the members")
        self.device, self.ds, self.cache_only, self.verbose = device, ds, bool(cache_only), verbose
        self.batch_rows, self.max_tokens = int(batch_rows), int(max_tokens)
        self.kw = {"max_len": int(max_len), "head_max_len": int(head_max_len)}
        self.members = []
        try:
            for m, pre, ct in zip(models, preload, member_cache_tags(cache_tag, len(models))):
                self.members.append(Member(m, device, ds, self.settings, read_only=self.cache_only, model=pre,
                                           verbose=verbose, cache_tag=ct))
            check_member_tokenizers(self.members)
            check_member_budgets(self.members, **self.kw)
            if fold is not None:
                check_fold(self.members, fold)
            self.tok = load_tokenizer(self.members[0].dir)
        except Exception:
            self.close()
            raise
        self.ensemble = str(ensemble) if ensemble else None
        self.seg = SegmentTokens(self.tok)
        self.tag = cache_tag or (self.members[0].tag if len(self.members) == 1 else
                                 ensemble_tag(TAG_PREFIX, self.members))
        self.stats = {"questions": 0, "rows": 0, "cached": 0, "forwarded": 0}

    # -- contract -------------------------------------------------------------
    def spec(self):
        """Score-file meta: engine, family, layout, backbone, revision, adapter_hash, prefix_share,
        batch_size, grouping, members (+ the Laya budgets and the readout)."""
        cfg = laya_config(self.members[0].dir)
        return {"engine": "laya", "family": FAMILY, "layout": LAYOUT, "tag": self.tag,
                "backbone": cfg.get("encoder"), "revision": None, "adapter_hash": None, "prefix_share": False,
                "batch_size": self.batch_rows, "max_tokens": self.max_tokens, "grouping": "length", **self.kw,
                "members": [{"dir": m.dir, "init": m.label, "model_sha256": m.sha256, "tag": m.tag,
                             "seed": m.run_record.get("seed"), "fold": m.run_record.get("fold")}
                            for m in self.members],
                "ensemble": self.ensemble,
                "readout": "Laya [MASK] scorer, softmax over the offered markers at T=1, one distribution per member"}

    def row(self, form, qtype):
        return seq_form(self.tok, form, qtype, seg=self.seg, **self.kw)

    def _warm(self, qs):
        texts = set()
        for q in qs:
            for f in q.forms:
                qs_, os_, vs = segments(form_parts(f), q.qtype)
                texts.update((qs_, vs, *os_))
        self.seg.warm(texts)

    def refused(self, qs):
        """{qkey: reason} of the questions with an over-budget form (never cut)."""
        out = {}
        for q in qs:
            if q.qkey not in out:
                why = refusal(self.tok, q, seg=self.seg, **self.kw)
                if why is not None:
                    out[q.qkey] = why
        return out

    def _plan(self, qs):
        """({qkey: (question, [(form, row key)])}, {row key: Row}, {qkey: refusal reason})."""
        uniq = dedupe_questions(qs)
        self._warm(uniq.values())
        plan, rows, refused = {}, {}, {}
        for qk, q in uniq.items():
            try:
                lst = []
                for f in q.forms:
                    r = self.row(f, q.qtype)
                    k = row_key(r)
                    lst.append((f, k, r))
            except LengthError as e:
                refused[qk] = str(e)
                continue
            for _f, k, r in lst:
                rows.setdefault(k, r)
            plan[qk] = (q, [(f, k) for f, k, _r in lst])
        return plan, rows, refused

    def keys(self, qs):
        """{qkey: (row key per form)} of every non-refused question: no cache lookup, no model."""
        plan, _rows, _ref = self._plan(qs)
        return {qk: tuple(k for _f, k in lst) for qk, (_q, lst) in plan.items()}

    def _run(self, m, rows, todo):
        model = m.load()
        t0 = time.time()
        rr = [rows[k] for k in todo]
        for b in length_batches([len(r.ids) for r in rr], self.batch_rows, self.max_tokens):
            zs = _forward_safe(model, [rr[i] for i in b], self.tok.pad_token_id, self.device)
            for i, z in zip(b, zs):
                if not np.isfinite(z).all():
                    z = _forward_safe(model, [rr[i]], self.tok.pad_token_id, self.device)[0]   # once more alone
                    if not np.isfinite(z).all():
                        m.flush()
                        raise RuntimeError(f"[{m.tag}] non-finite logits for row {todo[i]} ({len(rr[i].ids)} tokens)")
                m.store(todo[i], cache_record(z))
        m.flush()
        dt = time.time() - t0
        m.stats["forwarded"] += len(todo)
        m.stats["seconds"] += dt
        if self.verbose:
            lens = np.array([len(r.ids) for r in rr])
            _log(f"forwarded {len(todo)} rows in {dt:.1f}s = {len(todo) / max(dt, 1e-9):.1f} rows/s "
                 f"(tokens p50 {int(np.median(lens))} max {int(lens.max())})", f"habitus.dm.laya {m.tag}")
        return len(todo)

    def decide(self, qs):
        """{qkey: Decision} for every non-refused question (a repeated qkey is answered once; two
        different questions under one qkey raise). Decision.letter_probs[form] / canon_probs[form]:
        one T = 1 distribution per member, member order; cov = 1.0 per form."""
        plan, rows, _refused = self._plan(qs)
        missing = [m.lookup(list(rows)) for m in self.members]
        if self.cache_only and any(missing):
            raise CacheMiss(sorted({k for v in missing for k in v}))
        fwd = 0
        for m, todo in zip(self.members, missing):
            if todo:
                fwd += self._run(m, rows, todo)
        out = {}
        for qk, (q, lst) in plan.items():
            lp, cp = {}, {}
            for f, k in lst:
                n = len(f.options)
                per_l, per_c = [], []
                for m in self.members:
                    a = m.answers[k]
                    if int(a["n"]) != n:
                        raise ValueError(f"cached answer {k} has {a['n']} markers, form {qk}:{f.name} {n}")
                    p = tuple(float(x) for x in softmax(a["z"]))
                    per_l.append(p)
                    per_c.append(canonical(p, f.canon))
                lp[f.name], cp[f.name] = tuple(per_l), tuple(per_c)
            names = tuple(f.name for f, _k in lst)
            out[qk] = Decision(qkey=qk, layout=LAYOUT, forms=names, letter_probs=lp, canon_probs=cp,
                               cov={f: 1.0 for f in names}, cov_min=1.0)
        n_rows = sum(len(v[1]) for v in plan.values())
        self.stats["questions"] += len(plan)
        self.stats["rows"] += n_rows
        self.stats["forwarded"] += fwd
        self.stats["cached"] += len(self.members) * len(rows) - fwd
        if self.verbose:
            _log(f"{len(plan)} questions = {n_rows} rows ({len(rows)} distinct) x {len(self.members)} model(s); "
                 f"{fwd} forwarded", f"habitus.dm.laya {self.tag}")
        return out

    def load(self):
        """Load every member's model now."""
        for m in self.members:
            m.load()

    def flush(self):
        for m in self.members:
            m.flush()

    def close(self):
        for m in getattr(self, "members", ()):
            m.close()


__all__ = ["LayaDecisionModel", "parts_from_prompt", "split_prompt", "user_text", "Parts", "form_parts", "Row",
           "PromptError", "LengthError", "seq_parts", "seq_form", "seq_item", "check_identity", "refusal",
           "option_lengths", "row_key", "SegmentTokens", "segments", "canonical", "DecisionModelHBL",
           "padding_masks", "inits", "resolve_init", "load_tokenizer", "load_tokenizer_dir", "build_model",
           "load_model", "model_sha256", "file_sha256", "hbl_tag", "collate", "length_batches", "row_logits", "softmax",
           "MAX_LEN", "HEAD_MAX_LEN", "QTYPES", "LAYOUT", "DEFAULT_INIT"]
