"""The 'laya' decision model, layout MQ: encode the state once, decide many questions.

The sibling of habitus.dm.laya (layout L-a, one row per (question, form)): the same vendored Laya
decision model (DecisionModelHBL), the same sanitised text segments, the same length budgets and the
same checkpoints (a Laya init or an L-a run is a valid init); only the row layout and the masks differ.

Layout (one row per distinct state; more rows when its blocks exceed row_tokens - exact by (I1)):

    [CLS] tok(state) [SEP] | B_1 | B_2 | ... | B_Q
    B_j = tok("<qtype> question: " + q_j) [SEP] ( [MASK] tok(" " + option_jk) )_k

    position ids    state 0..S-1; the question part of EVERY block starts at S; every option of a block
                    starts right after its block's question part (the options of a block share positions)
    attention       state token     -> state only (the state's encoding is the same whatever is asked)
    (key allowed)   question token  -> state + its whole block (question part + every option)
                    option-k token  -> state + its block's question part + option k itself
                    padding         -> state (no empty query row); no real token ever sees padding
    local layers    the same mask AND |pos_q - pos_k| <= encoder.config.sliding_window, measured on the
                    LOGICAL position ids, never on the sequence index

The masks reach ModernBertModel.forward as the transformers 5.14.1 dict {"full_attention": [B,1,L,L],
"sliding_attention": [B,1,L,L]} (bool, True = may attend) with explicit position_ids, so the encoder
must run attn_implementation "sdpa" (eager adds a mask to the scores and flash ignores it: both are
refused). The 2 Laya head layers get the SAME block mask as src_mask (True = blocked), one row at a
time with a 2-D [L_b, L_b] mask and no padding. type_emb is added per token (a block's tokens its
qtype, the shared state tokens STATE_QTYPE = choice).

Guarantees (tests/test_dm_laya_mq.py, tiny random ModernBERT, fp32, CPU):
  (I1) question isolation: no token of B_j attends to another block, the state attends only to itself
       and B_j's position ids / windows are fixed by (S, B_j) alone; so the answer to q_j does not
       depend on the other questions in the row (|dp| <= 1e-5). Packing a state's blocks into several
       rows is therefore exact, and an answer is cached under sha1(LAYOUT_VERSION, state ids, block).
  (I2) option-order invariance: a block's options share position ids and never see each other, its
       question tokens see them as a set, the head has no positions: permuting the options permutes
       the marker logits (|dp| <= 1e-5), so ONE pass replaces L-a's two forms.
  NOT claimed: equality with L-a's single-question output (MQ is state-first, L-a head-first). In bf16
  the option-order gap is about 2.3e-2 (fp32: 2.5e-5).

Serving: questions are grouped by TypedQuestion.group, and inside a group by their state; each
question is ONE block of forms[0] laid out in CANONICAL option order (Form.canon). The Decision has
the single form 'fwd' (layout 'mq'), one T = 1 softmax per member; habitus.questions.habit.types
.answer_of reads it as pbar over the members (raw = {'fwd': pbar}, renormalised even for 1 member).
Lengths (never cut): a block is refused exactly when the L-a row of the same form is
(head <= HEAD_MAX_LEN, L-a row length <= MAX_LEN).

Label-free: this module tokenises the forms handed to it, loads model weights and reads / writes its
own answer caches; it opens no card, link, score or label file and imports no metric code.
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from habitus.dm import laya as LA
from habitus.dm.base import CacheMiss, Decision

FAMILY = LA.FAMILY
LAYOUT = "mq"
LAYOUT_VERSION = "mq1"                 # part of every answer-cache key: a layout change never reuses answers
FORM = "fwd"                           # the single answered form (canonical option order)
MAX_LEN = LA.MAX_LEN                   # 2048: a block's L-a row bound
HEAD_MAX_LEN = LA.HEAD_MAX_LEN         # 256: question + options of one block (= the L-a head budget)
ROW_TOKENS = MAX_LEN                   # physical tokens of one packed row (state + its blocks)
BATCH_ROWS = 8                         # rows per inference forward
BATCH_TOKENS = 16384                   # padded tokens (rows x longest row) per inference forward
STATE_QTYPE = LA.QTYPES["choice"]      # the type embedding the shared state tokens carry
MASK_FILL = LA.MASK_FILL               # -1e4, Laya's logit for an absent marker slot
TAG_PREFIX = LA.TAG_PREFIX + "mq-"     # "hbl-mq-<init>-<12 hex>": never an L-a tag, even on equal weights
HEAD = "row"


def _log(msg, tag="habitus.dm.laya_mq"):
    from habitus.log import log
    log(msg, tag)


# --------------------------------------------------------------------------- #
# Tokens: the state and one block per question (laya.segments; never cut)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Block:
    """One question block: q_ids = tok("<qtype> question: " + q) + [SEP]; opt_ids = per option [MASK] +
    tok(" " + option), in LAYOUT order (canonical when serving, the item's letter order when training)."""
    q_ids: tuple
    opt_ids: tuple
    qtype: int = STATE_QTYPE

    @property
    def n(self):
        return len(self.opt_ids)

    @property
    def length(self):
        """Physical tokens of the block in a row (question part with its [SEP] + every option)."""
        return len(self.q_ids) + sum(len(o) for o in self.opt_ids)

    @property
    def head_len(self):
        """laya.Row.head_len of the same question: question + options, no [SEP]."""
        return self.length - 1

    def key_bytes(self):
        return json.dumps([self.qtype, list(self.q_ids), [list(o) for o in self.opt_ids]],
                          separators=(",", ":")).encode()


def _check_special(tok, ids, what):
    bad = sorted(set(ids) & LA.special_ids(tok))
    if bad:
        raise ValueError(f"special token ids {bad[:5]} inside the {what} segment (sanitiser gap)")


def encode_parts(tok, parts, qtype=LA.TRAIN_QTYPE, head_max_len=HEAD_MAX_LEN, max_len=MAX_LEN, seg=None):
    """(state ids, Block) of one laya.Parts (state text, question text, options in layout order).
    Raises laya.LengthError when head > head_max_len or the L-a row of the same question > max_len,
    ValueError for a special id inside a segment or a bad option count."""
    LA.check_qtype(qtype)
    if not 2 <= len(parts.options) <= LA.MAX_OPTIONS:
        raise ValueError(f"{len(parts.options)} options (2..{LA.MAX_OPTIONS})")
    enc = seg if seg is not None else (lambda s: tok(s, add_special_tokens=False)["input_ids"])
    q_seg, o_segs, v_seg = LA.segments(parts, qtype)
    v, q, opts = list(enc(v_seg)), list(enc(q_seg)), [list(enc(o)) for o in o_segs]
    _check_special(tok, v, "state")
    _check_special(tok, q, "question")
    for o in opts:
        _check_special(tok, o, "option")
    blk = Block(tuple(q) + (tok.sep_token_id,), tuple((tok.mask_token_id,) + tuple(o) for o in opts),
                LA.QTYPES[qtype])
    if blk.head_len > head_max_len:
        raise LA.LengthError(f"head of {blk.head_len} tokens > head_max_len {head_max_len} (option tokens "
                             f"{[len(o) for o in opts]}): refused, never cut")
    state = (tok.cls_token_id,) + tuple(v) + (tok.sep_token_id,)
    check_fit(state, blk, max_len)
    return state, blk


def la_length(state, blk):
    """Tokens of the L-a row of the same question: [CLS] q [SEP] options [SEP] state [SEP] = the MQ row
    [CLS] state [SEP] q [SEP] options + 1."""
    return len(state) + blk.length + 1


def check_fit(state, blk, max_len=MAX_LEN):
    """The question's L-a row length (la_length) <= max_len, else LengthError: MQ refuses exactly the
    questions L-a refuses (its own single-block row is one token shorter)."""
    if la_length(state, blk) > max_len:
        raise LA.LengthError(f"row of {la_length(state, blk)} tokens (L-a layout) > max_len {max_len} (head "
                             f"{blk.head_len}, state {len(state) - 2}): refused, never cut")


def answer_key(state, blk):
    """Answer-cache key: by (I1) an answer is a function of (state ids, block ids) alone."""
    h = hashlib.sha1(LAYOUT_VERSION.encode())
    h.update(np.asarray(state, dtype="<i4").tobytes())
    h.update(b"|")
    h.update(blk.key_bytes())
    return h.hexdigest()


def pack(state, blocks, row_tokens=ROW_TOKENS):
    """Greedy packing of one state's blocks into rows of <= row_tokens physical tokens: [[block index]]
    (a block whose row alone exceeds row_tokens still gets its own row; check_fit bounds that row)."""
    rows, cur, used = [], [], len(state)
    for i, b in enumerate(blocks):
        if cur and used + b.length > row_tokens:
            rows.append(cur)
            cur, used = [], len(state)
        cur.append(i)
        used += b.length
    if cur:
        rows.append(cur)
    return rows


def row_layout(state, blocks):
    """One row: {"ids", "pos" (logical position ids), "seg" (0 state, j for block j), "opt" (0 state /
    question part, k for option k of its block), "qt" (type id per token), "markers" ([[sequence index of
    each option's [MASK]]] per block, layout order)}."""
    S = len(state)
    ids, pos = list(state), list(range(S))
    seg, opt, qt = [0] * S, [0] * S, [STATE_QTYPE] * S
    markers = []
    for j, b in enumerate(blocks, 1):
        a = len(b.q_ids)
        ids += b.q_ids
        pos += range(S, S + a)
        seg += [j] * a
        opt += [0] * a
        qt += [b.qtype] * a
        mk = []
        for k, o in enumerate(b.opt_ids, 1):
            mk.append(len(ids))
            ids += o
            pos += range(S + a, S + a + len(o))
            seg += [j] * len(o)
            opt += [k] * len(o)
            qt += [b.qtype] * len(o)
        markers.append(mk)
    return {"ids": ids, "pos": pos, "seg": seg, "opt": opt, "qt": qt, "markers": markers}


def collate(layouts, pad_id, device="cpu"):
    """Right-padded [B, L] tensors of row layouts (ids, pos, seg, opt, qt, valid) + every block's marker
    positions in row-major block order (blk_row [N], marker_pos / marker_mask [N, K]) + row lengths."""
    B, L = len(layouts), max(len(r["ids"]) for r in layouts)
    out = {k: torch.zeros((B, L), dtype=torch.long) for k in ("ids", "pos", "seg", "opt", "qt")}
    out["ids"].fill_(int(pad_id))
    valid = torch.zeros((B, L), dtype=torch.bool)
    blk_row, mk = [], []
    for b, r in enumerate(layouts):
        n = len(r["ids"])
        for k in ("ids", "pos", "seg", "opt", "qt"):
            out[k][b, :n] = torch.tensor(r[k], dtype=torch.long)
        valid[b, :n] = True
        for m in r["markers"]:
            blk_row.append(b)
            mk.append(m)
    K = max(len(m) for m in mk)
    marker_pos = torch.zeros((len(mk), K), dtype=torch.long)
    marker_mask = torch.zeros((len(mk), K), dtype=torch.bool)
    for i, m in enumerate(mk):
        marker_pos[i, :len(m)] = torch.tensor(m, dtype=torch.long)
        marker_mask[i, :len(m)] = True
    out.update({"valid": valid, "blk_row": torch.tensor(blk_row, dtype=torch.long), "marker_pos": marker_pos,
                "marker_mask": marker_mask})
    out = {k: v.to(device) for k, v in out.items()}
    out["lengths"] = [len(r["ids"]) for r in layouts]
    return out


# --------------------------------------------------------------------------- #
# Masks and the forward pass
# --------------------------------------------------------------------------- #
def block_masks(seg, opt, pos, valid, sliding_window, isolate=True, window="pos"):
    """(full, sliding or None): bool [B, L, L], True = query (dim 1) may attend to key (dim 2).
    isolate=False (every real token sees every real token) and window="index" (the local window on the
    sequence index) are NEGATIVE CONTROLS for the tests - never a serving or training mode."""
    vk = valid[:, None, :]
    if isolate:
        sq, sk = seg[:, :, None], seg[:, None, :]
        oq, ok = opt[:, :, None], opt[:, None, :]
        same = (sq == sk) & (sq > 0) & ~((oq > 0) & (ok > 0) & (oq != ok))
        full = vk & ((sk == 0) | same)
    else:
        full = vk.expand(-1, seg.shape[1], -1)
    if sliding_window is None:
        return full, None
    if window == "pos":
        p = pos
    elif window == "index":
        p = torch.arange(seg.shape[1], device=seg.device)[None, :].expand_as(pos)
    else:
        raise ValueError(f"window {window!r}")
    near = (p[:, :, None] - p[:, None, :]).abs() <= int(sliding_window)
    return full, full & near


def encoder_masks(full, slide):
    """The transformers 5.14.1 ModernBertModel mask dict (bool [B, 1, L, L], True = attend)."""
    masks = {"full_attention": full[:, None]}
    if slide is not None:
        masks["sliding_attention"] = slide[:, None]
    return masks


def check_encoder(enc):
    impl = getattr(enc.config, "_attn_implementation", None)
    if impl != "sdpa":
        raise ValueError(f"the MQ layout needs the encoder's sdpa attention (boolean 4-D masks), got {impl!r}")


def head_heads(model):
    return model.head.layers[0].self_attn.num_heads


def _head(model, h, full, lengths):
    """The Laya head layers under the block mask (src_mask, True = blocked), one row at a time with a
    2-D [L_b, L_b] mask and no padding."""
    if getattr(model, "head", None) is None:
        return h
    ckpt = getattr(model, "head_checkpointing", False) and model.training and torch.is_grad_enabled()

    def run(x, mask):
        for layer in model.head.layers:
            x = checkpoint(layer, x, src_mask=mask, use_reentrant=False) if ckpt else layer(x, src_mask=mask)
        return x
    outs = [run(h[b:b + 1, :n], ~full[b, :n, :n])[0] for b, n in enumerate(lengths)]
    return torch.nn.utils.rnn.pad_sequence(outs, batch_first=True)


def forward(model, batch, isolate=True, window="pos", return_hidden=False):
    """Marker logits [N blocks, K] (float32; absent slots MASK_FILL) of a collated batch, through
    model.encoder (dict masks + position ids), type_emb per token, the head layers under the block mask
    and the scorer - DecisionModelHBL's own modules. return_hidden adds {"encoder": [B, L, d] encoder
    output, "head": [B, L, d] head output, "full": the block mask}."""
    enc = model.encoder
    check_encoder(enc)
    sw = getattr(enc.config, "sliding_window", None)
    full, slide = block_masks(batch["seg"], batch["opt"], batch["pos"], batch["valid"], sw, isolate, window)
    he = enc(input_ids=batch["ids"], attention_mask=encoder_masks(full, slide),
             position_ids=batch["pos"]).last_hidden_state
    h = he + model.type_emb(batch["qt"])
    hh = _head(model, h, full, batch["lengths"])
    m = hh[batch["blk_row"][:, None], batch["marker_pos"]]
    z = model.scorer(m).squeeze(-1).float().masked_fill(~batch["marker_mask"], MASK_FILL)
    if return_hidden:
        return z, {"encoder": he, "head": hh, "full": full}
    return z


def _amp(device, amp=True):
    on = bool(amp) and str(device).startswith("cuda")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=on)


def chunks(lengths, batch_rows, batch_tokens):
    return LA.length_batches(lengths, batch_rows, batch_tokens)


def run_rows(model, layouts, pad_id, device="cpu", batch_rows=BATCH_ROWS, batch_tokens=BATCH_TOKENS, amp=True):
    """Inference over row layouts: per row, [np.float64 logits of each block (its n options)]; batches of
    length-sorted rows (<= batch_rows rows, <= batch_tokens padded tokens), halved on CUDA OOM."""
    res = [None] * len(layouts)
    was = model.training
    model.eval()

    def go(ch):
        try:
            with torch.no_grad(), _amp(device, amp):
                z = forward(model, collate([layouts[i] for i in ch], pad_id, device))
            return z.double().cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            if len(ch) == 1:
                raise
        torch.cuda.empty_cache()
        h = len(ch) // 2
        return np.concatenate([go(ch[:h]), go(ch[h:])])
    try:
        for ch in chunks([len(r["ids"]) for r in layouts], batch_rows, batch_tokens):
            z, j = go(ch), 0
            for i in ch:
                res[i] = [z[j + t, :len(m)].copy() for t, m in enumerate(layouts[i]["markers"])]
                j += len(layouts[i]["markers"])
    finally:
        model.train(was)
    return res


# --------------------------------------------------------------------------- #
# Checkpoints: Laya inits, L-a runs and MQ runs
# --------------------------------------------------------------------------- #
def resolve_mq(init, settings=None):
    """(checkpoint directory, init label) of an MQ model: an MQ run directory (habitus.json layout "mq";
    label = its init) or anything laya.resolve_init accepts (a Laya init or an L-a run: zero-shot MQ /
    the MQ init)."""
    if init not in LA.INIT_NAMES:
        d = os.path.abspath(str(init))
        rec = LA.read_run_record(d)
        if rec.get("layout") == LAYOUT:
            LA.check_checkpoint(d, "MQ checkpoint")
            return d, LA._label(rec.get("init") or os.path.basename(d))
    return LA.resolve_init(init, settings)


def mq_tag(label, digest):
    """'hbl-mq-<init label>-<12 hex of sha256(model.safetensors)>'."""
    return f"{TAG_PREFIX}{label}-{digest[:LA.HASH_CHARS]}"


def canonical_parts(form):
    """laya.Parts of a form with its options in CANONICAL order (Form.canon)."""
    opts = [None] * len(form.options)
    for i, c in enumerate(form.canon):
        opts[c] = form.options[i]
    return LA.Parts(form.state, form.question, tuple(opts))


# --------------------------------------------------------------------------- #
# The decision model
# --------------------------------------------------------------------------- #
class LayaMQDecisionModel:
    """dm.base.DecisionModel, family 'laya', layout 'mq': questions grouped by TypedQuestion.group (and
    state), each state encoded once per packed row, every question one block of forms[0] in canonical
    option order; one pass per question (order-invariant by construction).

    runs: one checkpoint (an MQ run directory, or any laya.resolve_init input: zero-shot MQ on a Laya
    / L-a checkpoint), a list (seeds; Decision lists the members in this order), or None (zero-shot
    from `init`). row_tokens: physical tokens of one packed row; batch_rows / batch_tokens: rows and
    padded tokens per forward. The other arguments are those of laya.LayaDecisionModel."""

    family = FAMILY
    layout = LAYOUT

    def __init__(self, runs=None, device="cpu", batch_rows=BATCH_ROWS, batch_tokens=BATCH_TOKENS,
                 row_tokens=ROW_TOKENS, init=LA.DEFAULT_INIT, max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN,
                 cache_only=False, ds=None, fold=None, ensemble=None, layout_version=LAYOUT_VERSION, cache_tag=None,
                 settings=None, verbose=False, preload=None):
        from habitus.dm.gpu import check_device
        if layout_version != LAYOUT_VERSION:
            raise ValueError(f"layout_version {layout_version!r}: this code is {LAYOUT_VERSION!r}")
        self.settings = LA._settings(settings)
        check_device(device, self.settings)
        models = [init] if runs is None else ([runs] if isinstance(runs, (str, os.PathLike)) else list(runs))
        if not models:
            raise ValueError("LayaMQDecisionModel needs at least one model")
        preload = list(preload) if preload is not None else [None] * len(models)
        if len(preload) != len(models):
            raise ValueError("preload must align with the members")
        self.device, self.ds, self.cache_only, self.verbose = device, ds, bool(cache_only), verbose
        self.batch_rows, self.batch_tokens, self.row_tokens = int(batch_rows), int(batch_tokens), int(row_tokens)
        self.kw = {"max_len": int(max_len), "head_max_len": int(head_max_len)}
        self.members = []
        try:
            for m, pre, ct in zip(models, preload, LA.member_cache_tags(cache_tag, len(models))):
                self.members.append(LA.Member(m, device, ds, self.settings, read_only=self.cache_only, model=pre,
                                              resolve=lambda x: resolve_mq(x, self.settings), tag_of=mq_tag,
                                              verbose=verbose, cache_tag=ct))
            LA.check_member_tokenizers(self.members)
            LA.check_member_budgets(self.members, **self.kw)
            if fold is not None:
                LA.check_fold(self.members, fold)
            self.tok = LA.load_tokenizer(self.members[0].dir)
        except Exception:
            self.close()
            raise
        self.ensemble = str(ensemble) if ensemble else None
        self.seg = LA.SegmentTokens(self.tok)
        self.tag = cache_tag or (self.members[0].tag if len(self.members) == 1 else
                                 LA.ensemble_tag(TAG_PREFIX, self.members))
        self.stats = {"questions": 0, "states": 0, "cached": 0, "forwarded": 0}

    def spec(self):
        """Score-file meta (dm.base contract keys + the MQ layout, budgets and readout)."""
        cfg = LA.laya_config(self.members[0].dir)
        return {"engine": "laya", "family": FAMILY, "layout": LAYOUT, "layout_version": LAYOUT_VERSION,
                "tag": self.tag, "backbone": cfg.get("encoder"), "revision": None, "adapter_hash": None,
                "prefix_share": False, "batch_size": self.batch_rows, "batch_tokens": self.batch_tokens,
                "row_tokens": self.row_tokens, "grouping": "group", "head": HEAD, **self.kw,
                "members": [{"dir": m.dir, "init": m.label, "model_sha256": m.sha256, "tag": m.tag,
                             "seed": m.run_record.get("seed"), "fold": m.run_record.get("fold")}
                            for m in self.members],
                "ensemble": self.ensemble,
                "readout": "Laya [MASK] scorer, softmax over a block's markers at T=1, one pass (order-invariant), "
                           "one distribution per member"}

    def encode(self, q):
        """(state ids, Block) of one TypedQuestion: forms[0] in canonical option order."""
        return encode_parts(self.tok, canonical_parts(q.forms[0]), q.qtype, seg=self.seg, **self.kw)

    def _warm(self, qs):
        texts = set()
        for q in qs:
            qs_, os_, vs = LA.segments(canonical_parts(q.forms[0]), q.qtype)
            texts.update((qs_, vs, *os_))
        self.seg.warm(texts)

    def refused(self, qs):
        """{qkey: reason} of the questions whose block is over budget (= the L-a row of forms[0])."""
        out = {}
        for q in qs:
            if q.qkey not in out:
                try:
                    self.encode(q)
                except LA.LengthError as e:
                    out[q.qkey] = str(e)
        return out

    def _plan(self, qs):
        """({qkey: (n options, answer key)}, {(group, state ids): (state ids, {answer key: Block})},
        {qkey: refusal reason})."""
        uniq = LA.dedupe_questions(qs)
        self._warm(uniq.values())
        plan, states, refused = {}, {}, {}
        for qk, q in uniq.items():
            try:
                st, blk = self.encode(q)
            except LA.LengthError as e:
                refused[qk] = str(e)
                continue
            k = answer_key(st, blk)
            plan[qk] = (blk.n, k)
            states.setdefault((q.group, st), (st, {}))[1].setdefault(k, blk)
        return plan, states, refused

    def keys(self, qs):
        """{qkey: (answer key,)} of every non-refused question: no cache lookup, no model."""
        plan, _states, _ref = self._plan(qs)
        return {qk: (k,) for qk, (_n, k) in plan.items()}

    def _run(self, m, states, todo):
        """Forward the blocks `todo` (a set of answer keys) of member m, packed per state."""
        layouts, keys = [], []
        for st, blocks in states.values():
            ks = [k for k in blocks if k in todo]
            for row in pack(st, [blocks[k] for k in ks], self.row_tokens):
                layouts.append(row_layout(st, [blocks[ks[i]] for i in row]))
                keys.append([ks[i] for i in row])
        if not layouts:
            return 0
        model = m.load()
        check_encoder(model.encoder)
        t0 = time.time()
        zs = run_rows(model, layouts, self.tok.pad_token_id, self.device, self.batch_rows, self.batch_tokens)
        n = 0
        for ks, z in zip(keys, zs):
            for k, zz in zip(ks, z):
                if not np.isfinite(zz).all():
                    m.flush()
                    raise RuntimeError(f"[{m.tag}] non-finite logits for answer {k}")
                m.store(k, LA.cache_record(zz))
                n += 1
        m.flush()
        dt = time.time() - t0
        m.stats["forwarded"] += n
        m.stats["seconds"] += dt
        if self.verbose:
            lens = np.array([len(r["ids"]) for r in layouts])
            _log(f"forwarded {n} questions in {len(layouts)} rows in {dt:.1f}s = {n / max(dt, 1e-9):.1f} q/s "
                 f"(row tokens p50 {int(np.median(lens))} max {int(lens.max())})", f"habitus.dm.laya_mq {m.tag}")
        return n

    def decide(self, qs):
        """{qkey: Decision} for every non-refused question: layout 'mq', the single form 'fwd' in canonical
        option order, one T = 1 distribution per member; cov = 1.0."""
        plan, states, _refused = self._plan(qs)
        all_keys = [k for _st, blocks in states.values() for k in blocks]
        missing = [m.lookup(all_keys) for m in self.members]
        if self.cache_only and any(missing):
            raise CacheMiss(sorted({k for v in missing for k in v}))
        fwd = 0
        for m, todo in zip(self.members, missing):
            if todo:
                fwd += self._run(m, states, set(todo))
        out = {}
        for qk, (n, k) in plan.items():
            dists = []
            for m in self.members:
                a = m.answers[k]
                if int(a["n"]) != n:
                    raise ValueError(f"cached answer {k} has {a['n']} markers, question {qk} {n}")
                dists.append(tuple(float(x) for x in LA.softmax(a["z"])))
            probs = tuple(dists)
            out[qk] = Decision(qkey=qk, layout=LAYOUT, forms=(FORM,), letter_probs={FORM: probs},
                               canon_probs={FORM: probs}, cov={FORM: 1.0}, cov_min=1.0)
        self.stats["questions"] += len(plan)
        self.stats["states"] += len(states)
        self.stats["forwarded"] += fwd
        self.stats["cached"] += len(self.members) * len(all_keys) - fwd
        if self.verbose:
            _log(f"{len(plan)} questions on {len(states)} states ({len(all_keys)} distinct blocks) x "
                 f"{len(self.members)} model(s); {fwd} forwarded", f"habitus.dm.laya_mq {self.tag}")
        return out

    def load(self):
        for m in self.members:
            m.load()

    def flush(self):
        for m in self.members:
            m.flush()

    def close(self):
        for m in getattr(self, "members", ()):
            m.close()


__all__ = ["LayaMQDecisionModel", "Block", "encode_parts", "la_length", "check_fit", "answer_key", "pack",
           "row_layout", "collate", "block_masks", "encoder_masks", "check_encoder", "forward", "run_rows",
           "resolve_mq", "mq_tag", "canonical_parts", "LAYOUT", "LAYOUT_VERSION", "ROW_TOKENS", "BATCH_ROWS",
           "BATCH_TOKENS", "MAX_LEN", "HEAD_MAX_LEN"]
