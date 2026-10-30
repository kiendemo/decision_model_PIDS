"""jev-head (HB.head): the jev prompt read out by an fp32 letter head on the final hidden state.

    dm = JevHeadDecisionModel(run_dir="runs/adapters/<fold>/HB.head", ds=ds)    # trained LoRA + head
    dm = JevHeadDecisionModel(None, ds=ds)                                       # stock head

z = W0 h + A (h - mu) / sigma + b over the n_slots = 4 letters, in fp32, where h is the decoder's final
hidden state (after its final norm: what lm_head reads) at the answer position. W0 [4, d] and b are
initialised from the lm_head rows (bias) of the letters A..D, A [4, d] = 0, and mu / sigma are
per-dimension statistics fitted once on the training items' step-0 hidden states and frozen. At
step 0 the head IS the letter readout in fp32 (|d p| < 1e-5); the jev letter readout reads the same
rows off the bf16 lm_head, so an untrained head differs from HB-Z by the bf16 rounding of the logits
(up to ~0.07 nats on the 4B). The offered letters are the first n slots; the record's lse is the
logsumexp over those n slots, so the coverage is 1 by construction. Trained together with the LoRA
(LoRA lr 1e-4, head lr 2e-4: train.jev with variant 'head').

The prompt, batching, prefix sharing (the shared path gathers the last hidden state from the suffix
forward), answer cache and non-finite rules are JevDecisionModel's. Tag: 'hb.head-<engine tag>' and,
with a trained run, '-<12 hex of sha256(adapter hash, head hash)>', so its answers never mix with
the letter readout's.

Files of a trained run: <run>/adapter/ (peft), <run>/head/{head.safetensors, head.json},
<run>/habitus.json with "variant": "head" and "head_hash".
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os

import numpy as np

from habitus.dm.base import MAX_OPTIONS, THINK_SUFFIX
from habitus.dm.jev import (DEFAULT_BACKBONE, DEFAULT_REVISION, JevDecisionModel, adapter_hash, collate,
                            engine_tag, letter_ids, resolve_adapter, run_record)

VARIANT = "head"
VARIANT_TEXT = "HB.head: fp32 letter head z = W0 h + A norm(h) + b (W0, b from lm_head rows of A..D, A = 0)"
TAG_PREFIX = "hb."
HEAD_DIR, HEAD_WEIGHTS, HEAD_CONFIG = "head", "head.safetensors", "head.json"
RECIPE_HEAD = {"head_lr": 2e-4, "norm_fit_items": 256}   # head 2e-4 next to the LoRA's 1e-4
PAD_LOGIT = -1e4                                         # unused slots (as the letter option logits)
READOUT_DTYPE_HEAD = "fp32 head"


# --------------------------------------------------------------------------- #
# Hidden states of the decoder
# --------------------------------------------------------------------------- #
def causal_lm(model):
    """The causal LM (decoder `.model` + `.lm_head`) of a peft-wrapped model (its LoRA layers are
    injected in place, so the decoder's forward is the adapted one) or the model itself."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def final_hidden(model, ids, att):
    """[B, L, d]: the decoder's final hidden states (after its final norm: what lm_head reads)."""
    return causal_lm(model).model(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state


def shared_hidden(model, units, pad, dev, positions):
    """Final hidden states of the rows of `units` ((keys, prefix ids, [suffix ids])) at positions[r]
    (indices into row r's full prompt = prefix + suffix_r), each prefix forwarded ONCE as in
    JevDecisionModel._forward_shared; the prefix forward's hidden states are kept as well, since a
    position may lie in the shared prefix. -> [sum_r len(positions[r]), d]."""
    import torch
    dec = causal_lm(model).model
    pre_ids, pre_att = collate([{"ids": u[1]} for u in units], pad, dev)
    out = dec(input_ids=pre_ids, attention_mask=pre_att, use_cache=True)
    pre_h, cache = out.last_hidden_state, out.past_key_values
    rows = torch.tensor([i for i, u in enumerate(units) for _ in u[2]], device=dev)
    cache.reorder_cache(rows)
    sufs = [s for u in units for s in u[2]]
    R, S = len(sufs), max(len(s) for s in sufs)
    ids = torch.full((R, S), pad, dtype=torch.long)
    att = torch.zeros((R, S), dtype=torch.long)
    for r, s in enumerate(sufs):
        ids[r, :len(s)] = torch.tensor(s)
        att[r, :len(s)] = 1
    att = torch.cat([pre_att.index_select(0, rows), att.to(dev)], dim=1)
    h = dec(input_ids=ids.to(dev), attention_mask=att, past_key_values=cache, use_cache=True).last_hidden_state
    del cache
    P = pre_ids.shape[1]
    sel, r = [], 0
    for ui, u in enumerate(units):
        n = len(u[1])
        for s in u[2]:
            for p in positions[r]:
                if not 0 <= p < n + len(s):
                    raise IndexError(f"position {p} outside row {r} of {n + len(s)} tokens")
                sel.append(pre_h[ui, P - n + p] if p < n else h[r, p - n])
            r += 1
    return torch.stack(sel)


def _slot_lse(z, n):
    zz = np.asarray(z[:n], dtype=np.float64)
    m = float(zz.max())
    return m + math.log(float(np.exp(zz - m).sum()))


def _file_sha(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def head_hash(head_dir):
    """sha256 over (name, content sha256) of the head files: equal heads hash equal, wherever they are."""
    h = hashlib.sha256()
    for name in (HEAD_CONFIG, HEAD_WEIGHTS):
        p = os.path.join(head_dir, name)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{head_dir}: no {name} (not a trained HB.head)")
        h.update(f"{name}\x00{_file_sha(p)}\n".encode())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# The head (an fp32 torch module, built lazily so that importing this module needs no torch)
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def _letter_head_class():
    import torch
    from torch import nn

    class LetterHead(nn.Module):
        """z = W0 h + A (h - mu) / sigma + b over n_slots letters, fp32 (autocast off). mu / sigma are
        frozen buffers (fit_norm, once, before training); W0, A, b are trained."""

        def __init__(self, hidden, n_slots=MAX_OPTIONS):
            super().__init__()
            self.W0 = nn.Parameter(torch.zeros(n_slots, hidden, dtype=torch.float32))
            self.A = nn.Parameter(torch.zeros(n_slots, hidden, dtype=torch.float32))
            self.b = nn.Parameter(torch.zeros(n_slots, dtype=torch.float32))
            self.register_buffer("mu", torch.zeros(hidden, dtype=torch.float32))
            self.register_buffer("sigma", torch.ones(hidden, dtype=torch.float32))

        @classmethod
        def from_lm_head(cls, lm_head, letter_ids):
            """W0 = the lm_head rows of `letter_ids` (fp32 copies, so a tied embedding is not shared),
            b = their bias (0 without one), A = 0: at step 0 the head IS the letter readout in fp32."""
            w = lm_head.weight.detach()
            head = cls(w.shape[1], len(letter_ids))
            idx = torch.tensor(list(letter_ids), dtype=torch.long, device=w.device)
            with torch.no_grad():
                head.W0.copy_(w.index_select(0, idx).float().cpu())
                if getattr(lm_head, "bias", None) is not None:
                    head.b.copy_(lm_head.bias.detach().index_select(0, idx).float().cpu())
            return head

        def fit_norm(self, hs, eps=1e-6):
            """Freeze mu / sigma = the per-dimension mean / std of hidden states hs [N, d] (N >= 2)."""
            with torch.no_grad():
                x = hs.detach().float()
                if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] != self.mu.shape[0]:
                    raise ValueError(f"fit_norm needs [N >= 2, {self.mu.shape[0]}] hidden states, "
                                     f"got {tuple(x.shape)}")
                self.mu.copy_(x.mean(0).to(self.mu.device))
                self.sigma.copy_(x.std(0).clamp_min(eps).to(self.sigma.device))

        def forward(self, h):
            with torch.autocast(device_type=h.device.type, enabled=False):
                x = h.float()
                return x @ self.W0.t() + ((x - self.mu) / self.sigma) @ self.A.t() + self.b

    return LetterHead


def __getattr__(name):
    if name == "LetterHead":
        return _letter_head_class()
    raise AttributeError(name)


# --------------------------------------------------------------------------- #
# The readout (training hooks + serving forwards)
# --------------------------------------------------------------------------- #
class HeadReadout:
    """HB.head's readout: training (prepare / option_logits / logits_for / parameters) and serving
    (attach / engine_full / engine_shared); records (describe / save / load)."""
    variant = VARIANT
    slot = True                      # own slot logits: coverage 1 by construction
    prefix_share = True
    prompt_suffix = THINK_SUFFIX

    def __init__(self, module=None, digest=None):
        self.module, self.digest = module, digest

    # --- training ---
    def parameters(self):
        return list(self.module.parameters()) if self.module is not None else []

    @property
    def n_params(self):
        return int(sum(p.numel() for p in self.parameters()))

    def to(self, device):
        if self.module is not None:
            self.module.to(device)
        return self

    def train(self, mode=True):
        if self.module is not None:
            self.module.train(mode)
        return self

    def prepare(self, model, enc, device, pad_id, rec):
        """Fit the frozen norm statistics on the first norm_fit_items training rows (step 0)."""
        import torch
        n = int(rec.get("norm_fit_items", RECIPE_HEAD["norm_fit_items"]))
        rows = enc[:max(2, n)]
        if len(rows) < 2:
            raise ValueError("HB.head: fitting the frozen norm statistics needs >= 2 training items")
        was = model.training
        model.eval()
        hs = []
        with torch.no_grad():
            for s in range(0, len(rows), 16):
                ids, att = collate(rows[s:s + 16], pad_id, device)
                hs.append(final_hidden(model, ids, att)[:, -1, :].float())
        self.module.fit_norm(torch.cat(hs))
        model.train(was)

    def option_logits(self, model, ids, att, o_ids, mask, rows=None, train=True):
        z = self.module(final_hidden(model, ids, att)[:, -1, :])[:, :mask.shape[1]]
        return z.masked_fill(~mask, PAD_LOGIT)

    def logits_for(self, model, tok, enc, opt_table, device, batch_size=32):
        """Raw slot logits per row (eval mode), in the row's letter order."""
        import torch
        from habitus.train.jev import batch_options
        model.eval()
        self.train(False)
        order = sorted(range(len(enc)), key=lambda i: len(enc[i]["ids"]))
        res = [None] * len(enc)
        with torch.no_grad():
            for s in range(0, len(order), batch_size):
                idx = order[s:s + batch_size]
                b = [enc[i] for i in idx]
                ids, att = collate(b, tok.pad_token_id, device)
                opt, mask = batch_options(b, opt_table, device)
                z = self.option_logits(model, ids, att, opt, mask, b, train=False).float().cpu().numpy()
                for r, i in enumerate(idx):
                    res[i] = z[r, :len(opt_table[b[r]["q"]])]
        return res

    # --- serving ---
    def attach(self, model, letter_ids, device):
        if self.module is None:                          # stock head: the lm_head rows (HB-Z's readout in fp32)
            self.module = _letter_head_class().from_lm_head(causal_lm(model).lm_head, letter_ids)
        return self.to(device).train(False)

    def _z(self, h):
        import torch
        z = self.module(h)
        return z.cpu().numpy(), torch.logsumexp(z, dim=-1).cpu().numpy()

    def engine_full(self, model, seqs, pad, dev, letter_ids):
        ids, att = collate([{"ids": s} for s in seqs], pad, dev)
        return self._z(final_hidden(model, ids, att)[:, -1, :])

    def engine_shared(self, model, units, pad, dev, letter_ids):
        pos = [[len(u[1]) + len(s) - 1] for u in units for s in u[2]]
        return self._z(shared_hidden(model, units, pad, dev, pos))

    # --- records ---
    def describe(self):
        return {"variant": self.variant, "readout": VARIANT_TEXT, "prefix_share": self.prefix_share,
                "params": self.n_params, "head_hash": self.digest, "readout_dtype": READOUT_DTYPE_HEAD}

    def save(self, head_dir):
        from safetensors.torch import save_file
        m = self.module
        save_file({k: v.detach().cpu().contiguous() for k, v in m.state_dict().items()},
                  os.path.join(head_dir, HEAD_WEIGHTS))
        with open(os.path.join(head_dir, HEAD_CONFIG), "w", encoding="utf-8") as f:
            json.dump({"variant": VARIANT, "hidden": int(m.W0.shape[1]), "n_slots": int(m.W0.shape[0]),
                       "init": "lm_head rows of the letters", "readout": VARIANT_TEXT}, f, indent=1, sort_keys=True)
        self.digest = head_hash(head_dir)
        return self.digest

    @classmethod
    def load(cls, head_dir):
        from safetensors.torch import load_file
        digest = head_hash(head_dir)
        with open(os.path.join(head_dir, HEAD_CONFIG), encoding="utf-8") as f:
            cfg = json.load(f)
        if cfg.get("variant") != VARIANT:
            raise ValueError(f"{head_dir}: a {cfg.get('variant')} head, not an HB.head")
        m = _letter_head_class()(cfg["hidden"], cfg["n_slots"])
        m.load_state_dict(load_file(os.path.join(head_dir, HEAD_WEIGHTS)), strict=True)
        return cls(m, digest)


def variant_tag(variant, backbone, adapter_dir=None, adapter_digest=None, head_digest=None):
    """'hb.<variant>-<engine_tag(backbone, adapter)>' + with an adapter '-<12 hex>' of
    sha256(adapter hash, head hash) (the adapter hash alone without a head)."""
    if variant != VARIANT:
        raise ValueError(f"variant {variant!r}: only {VARIANT!r} is served")
    t = f"{TAG_PREFIX}{variant}-{engine_tag(backbone, adapter_dir)}"
    if adapter_dir is None:
        return t
    d = adapter_digest if head_digest is None else hashlib.sha256(
        f"{adapter_digest}\x00{head_digest}".encode()).hexdigest()
    return f"{t}-{d[:12]}"


# --------------------------------------------------------------------------- #
# Training hooks (train.jev)
# --------------------------------------------------------------------------- #
def recipe():
    return dict(RECIPE_HEAD)


def train_readout(model, tok, device):
    """The fresh trainable head on the (LoRA-wrapped) `model`: the lm_head rows of the letters."""
    lm = causal_lm(model)
    return HeadReadout(_letter_head_class().from_lm_head(lm.lm_head, letter_ids(tok))).to(device)


def save_run(readout, run_dir):
    """Write the head files under <run_dir>/head and return the habitus.json fields of the run:
    variant, head_hash, readout record."""
    hd = os.path.join(run_dir, HEAD_DIR)
    os.makedirs(hd, exist_ok=True)
    readout.save(hd)
    return {"variant": VARIANT, "head_hash": readout.digest, "readout": readout.describe()}


# --------------------------------------------------------------------------- #
# The decision model
# --------------------------------------------------------------------------- #
class JevHeadDecisionModel(JevDecisionModel):
    """DecisionModel 'jev' / layout 'head' (module docstring). run_dir: a train.jev run of variant
    'head' (adapter/ + head/ + habitus.json), or None for the stock head on the stock backbone."""
    layout = "head"
    variant = VARIANT

    def __init__(self, run_dir=None, backbone=DEFAULT_BACKBONE, revision=DEFAULT_REVISION, device="cpu",
                 batch_size=32, prefix_share=True, grouping="chunked", cache_tag=None, ds=None, cache_only=False,
                 forward=None, settings=None, *, cache=None, model=None, dtype="bfloat16", verbose=True,
                 readout=None):
        adapter = None
        if run_dir is not None:
            rec = run_record(run_dir)
            if rec.get("variant") != VARIANT:
                raise ValueError(f"{run_dir}: a run of variant {rec.get('variant')!r}, not an HB.head run")
            adapter = resolve_adapter(run_dir, str(backbone))
            readout = HeadReadout.load(os.path.join(os.path.dirname(adapter), HEAD_DIR))
        self.readout = readout if readout is not None else HeadReadout()
        if cache_tag is None:
            cache_tag = variant_tag(VARIANT, str(backbone), adapter, adapter_hash(adapter) if adapter else None,
                                    self.readout.digest)
        super().__init__(backbone, revision, adapter, device, batch_size, prefix_share and self.readout.prefix_share,
                         grouping, cache_tag, ds, cache_only, forward, settings, cache=cache, model=model,
                         dtype=dtype, verbose=verbose)
        self.run_dir = run_dir

    def _readout_dtype(self):
        return READOUT_DTYPE_HEAD

    def spec(self):
        return {**super().spec(), "variant": VARIANT, "readout": self.readout.describe(),
                "readout_dtype": READOUT_DTYPE_HEAD}

    def _attach(self, model):
        if model is not None:
            self.readout.attach(model, self.letter_ids, self.device)

    def _forward(self, seqs):
        import torch
        with torch.no_grad():
            return self.readout.engine_full(self._model, seqs, self.tok.pad_token_id, self.device, self.letter_ids)

    def _forward_shared(self, units):
        import torch
        with torch.no_grad():
            return self.readout.engine_shared(self._model, units, self.tok.pad_token_id, self.device,
                                              self.letter_ids)

    def _record(self, key, n, z, lse):
        return super()._record(key, n, z, _slot_lse(z, n))
