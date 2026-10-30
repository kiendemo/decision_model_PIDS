"""jev: a decoder LLM read as a decision model through the logits of its option letters.

    dm = JevDecisionModel(ds="CADETS_E3", cache_only=True)       # MAIN: stock Qwen3.5-4B @851bf6e8
    decisions = dm.decide(typed_questions)                         # {qkey: Decision}
    keys = dm.keys(typed_questions)                                # {qkey: (sha1 per form)}, no model

One forward pass per presented form, no generation. The prompt is the backbone's chat template over
[system = form.system, user = form.user()] with thinking disabled, which ends with the empty think
block "<think>\\n\\n</think>\\n\\n" (asserted). The answer is the next-token distribution over the
OFFERED letters: z = the logits of A..D at the answer position (logits_to_keep=1), p = softmax(z[:n])
at T = 1 in float64. The full-vocabulary logsumexp at the same position gives the coverage
cov = sum exp(z[:n] - lse) (the softmax mass the model put on the letters at all); it is recorded,
logged and returned (Decision.cov), never used to drop or reweight an answer.

Answer cache: sha1 of the full rendered prompt -> {"z": A..D logits (4 dp), "lse" (4 dp), "n",
"cov" (6 dp)} in an AnswerCache (dm/cache.py: the shipped read-only base + an append-only overlay
under runs/answers/<tag>/<ds>.jsonl). A freshly forwarded answer is recorded FIRST and read back from
its record, so a fresh answer and a cached one give the same floats (the readout always sees the 4-dp
z and the 6-dp cov). Every Decision comes from a cache record.

cache_only=True: a question whose prompt is not cached raises CacheMiss (with every missing key)
before any model code is imported; the tokenizer alone (a 6-file directory, no weights) renders and
keys the prompts. Otherwise the model loads on the first miss: bf16 weights at the pinned revision,
a LoRA adapter merged with merge_and_unload, device cpu by default (dm.gpu.check_device refuses a
forbidden physical GPU).

Batching: <= batch_size (<= 32) prompts per forward, <= 8 once a prompt is longer than 900 tokens,
length-bucketed; on CUDA OOM a batch is halved, with the retry run outside the exception handler so
the failed forward's activations are freed first. A non-finite logit / logsumexp / coverage is never
cached: the prompt is retried alone and, if still non-finite, the run stops with its key.

Prefix sharing (prefix_share=True). Prompts of one group share their text up to the question (chat
template + system turn + "State:\\n" + state + "\\n\\nQuestion: "). `split_prefix` takes the longest
common prefix of the group's TOKEN IDS (capped so every suffix keeps >= 1 token; nothing is
re-tokenised). `_forward_shared` forwards the left-padded prefixes once with use_cache (the hybrid
cache of Qwen3.5: K/V of the full-attention layers, conv window and fp32 recurrent state of the Gated
DeltaNet layers), replicates each prefix's cache once per suffix (reorder_cache), forwards the
suffixes RIGHT-padded with that cache and the mask [prefix mask, suffix mask], and applies lm_head to
each row's last real token only. The model is causal, so this is the same function as the full
forward; only the floating-point evaluation differs (kernel shapes, chunk boundaries), at the level of
a different batch makeup (bf16: |d letter logit| p99 0.25, |d qv| p99 ~0.02; fp32: <= 2.7e-5).
GPU re-inference is therefore not bit-exact; the answer cache is the canonical record.

Groups (grouping): 'direct' = one group per TypedQuestion.group (routed: the 4 QV + QT prompts of a
signature, its 2 QA prompts apart; E1: the 4 prompts of a state); 'chunked' = the prompts of one group
sorted by text and cut into chunks of <= batch_size rows (<= 8 once a prompt of the group is > 900
tokens), for groups larger than a batch (the habit views). Any grouping gives the same function.

forward= (test seam): a callable(list of token-id lists) -> (z [N, 4] float32, lse [N]) that
replaces the model's forward and shared forward; with it no model is ever loaded.

Tags: engine_tag(backbone, adapter) ('qwen3.5-4b'; '<name>+<adapter run>' with an adapter) unless
cache_tag is given; the habit arms use habit_tag ('hb-qwen3.5-4b', 'hb-<engine tag>-<12 hex of the
adapter content hash>').
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from habitus.dm.base import LETTERS, THINK_SUFFIX, CacheMiss, Decision, canonical, unique_questions
from habitus.dm.cache import AnswerCache, cache_paths

__all__ = ["JevDecisionModel", "engine_tag", "habit_tag", "THINK_SUFFIX", "load_tokenizer", "letter_ids",
           "pinned_revision", "split_prefix", "share_units", "softmax_t", "resolve_adapter", "run_record",
           "adapter_hash", "tokenizer_dir", "snapshot_dir", "DEFAULT_BACKBONE", "DEFAULT_REVISION", "REVISIONS"]

DEFAULT_BACKBONE = "Qwen/Qwen3.5-4B"
DEFAULT_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
REVISIONS = {"Qwen/Qwen3.5-4B": DEFAULT_REVISION,
             "Qwen/Qwen3.5-0.8B": "2fc06364715b967f1860aea9cf38778875588b17"}
TOKENIZER_FILES = ("chat_template.jinja", "config.json", "merges.txt", "tokenizer.json",
                   "tokenizer_config.json", "vocab.json")
MAX_BS, LONG, LONG_BS = 32, 900, 8     # never batch > 32; prompts > 900 tokens go 8 at a time
FLUSH = 500
LOW_COV = 0.5                          # flagged in the log, never dropped
HABIT_TAG_PREFIX = "hb-"
HASH_CHARS = 12                        # adapter content-hash digits in a habit tag
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")
RUN_CONFIG = "habitus.json"            # train.jev's run record next to adapter/
READOUT_DTYPE_LM = "bf16 lm_head"
GROUPINGS = ("direct", "chunked")
DTYPES = ("bfloat16", "float32")


def _log(msg, tag="dm.jev"):
    from habitus.log import log
    log(msg, tag)


def _settings(settings=None):
    if settings is not None:
        return settings
    from habitus.config import settings as _s
    return _s()


# --------------------------------------------------------------------------- #
# Tags, adapters, run records
# --------------------------------------------------------------------------- #
def engine_tag(backbone, adapter=None):
    """Directory-safe engine name: 'qwen3.5-4b', or 'qwen3.5-4b+<adapter run dir>' with an adapter."""
    t = str(backbone).rstrip("/").split("/")[-1].lower()
    if adapter:
        a = os.path.normpath(str(adapter))
        name = os.path.basename(os.path.dirname(a)) if os.path.basename(a) == "adapter" else os.path.basename(a)
        t += "+" + name
    return re.sub(r"[^A-Za-z0-9_.+-]", "_", t)


def adapter_hash(adapter_dir):
    """sha256 over the adapter's config and weight files (name + content sha256, sorted by name):
    two adapters with equal weights and config hash equal, whatever their directory."""
    h = hashlib.sha256()
    found = 0
    for name in sorted(os.listdir(adapter_dir)):
        if name not in ADAPTER_FILES:
            continue
        with open(os.path.join(adapter_dir, name), "rb") as f:
            d = hashlib.sha256()
            for b in iter(lambda: f.read(1 << 20), b""):
                d.update(b)
        h.update(f"{name}\x00{d.hexdigest()}\n".encode())
        found += 1
    if not found:
        raise FileNotFoundError(f"{adapter_dir}: no adapter files {ADAPTER_FILES}")
    return h.hexdigest()


def habit_tag(backbone=DEFAULT_BACKBONE, adapter_dir=None, digest=None):
    """'hb-qwen3.5-4b' (stock, HB-Z) or 'hb-qwen3.5-4b+<run name>-<12 hex of the adapter hash>'."""
    if adapter_dir is None:
        return HABIT_TAG_PREFIX + engine_tag(backbone)
    digest = digest or adapter_hash(adapter_dir)
    return f"{HABIT_TAG_PREFIX}{engine_tag(backbone, adapter_dir)}-{digest[:HASH_CHARS]}"


def resolve_adapter(adapter, backbone=DEFAULT_BACKBONE):
    """The peft adapter directory of `adapter` (a train.jev run directory holding adapter/ and
    habitus.json, or the adapter directory itself). FileNotFoundError without an adapter_config.json;
    ValueError when the run record or the adapter config names another backbone (a LoRA applied to
    the wrong base would load silently and answer garbage)."""
    a = os.path.abspath(str(adapter))
    run_dir = None
    if os.path.isdir(os.path.join(a, "adapter")):
        run_dir, a = a, os.path.join(a, "adapter")
    elif os.path.isfile(os.path.join(os.path.dirname(a), RUN_CONFIG)):
        run_dir = os.path.dirname(a)
    if not os.path.isfile(os.path.join(a, "adapter_config.json")):
        raise FileNotFoundError(f"{adapter}: no peft adapter (adapter_config.json) found")
    if run_dir is not None and os.path.isfile(os.path.join(run_dir, RUN_CONFIG)):
        with open(os.path.join(run_dir, RUN_CONFIG), encoding="utf-8") as f:
            rb = json.load(f).get("backbone")
        if rb is not None and rb != backbone:
            raise ValueError(f"adapter {a} was trained on {rb}, not {backbone}")
    with open(os.path.join(a, "adapter_config.json"), encoding="utf-8") as f:
        base = json.load(f).get("base_model_name_or_path")
    if base and base != backbone:
        raise ValueError(f"adapter {a} is a LoRA of {base}, not {backbone}")
    return a


def run_record(path):
    """habitus.json of a run directory, of the parent of its adapter/ directory, or {}."""
    d = os.path.abspath(str(path))
    for cand in (os.path.join(d, RUN_CONFIG), os.path.join(os.path.dirname(d), RUN_CONFIG)):
        if os.path.isfile(cand):
            with open(cand, encoding="utf-8") as f:
                return json.load(f)
    return {}


# --------------------------------------------------------------------------- #
# Tokenizer, letters, prefix split (no torch)
# --------------------------------------------------------------------------- #
def _hub_name(backbone):
    return str(backbone).rstrip("/").split("/")[-1]


def pinned_revision(backbone, revision=None):
    """`revision`, else the pinned revision of a known hub backbone (REVISIONS), else None."""
    return revision if revision is not None else REVISIONS.get(str(backbone))


def tokenizer_dir(backbone=DEFAULT_BACKBONE, revision=None, settings=None):
    """The pinned tokenizer directory <paths.tokenizers>/<name>@<rev[:8]> (the 6 tokenizer files),
    or None (revision None: the pinned one of `backbone`)."""
    revision = pinned_revision(backbone, revision)
    if revision is None:
        return None
    tk = getattr(_settings(settings).paths, "tokenizers", None)
    if tk is None:
        return None
    d = Path(str(tk)) / f"{_hub_name(backbone)}@{revision[:8]}"
    return d if (d / "tokenizer_config.json").is_file() else None


def snapshot_dir(backbone=DEFAULT_BACKBONE, revision=None, settings=None):
    """The HF hub snapshot <paths.models>/hub/models--<org>--<name>/snapshots/<revision>, or None."""
    revision = pinned_revision(backbone, revision)
    if revision is None or "/" not in str(backbone) or os.path.isdir(str(backbone)):
        return None
    models = getattr(_settings(settings).paths, "models", None)
    if models is None:
        return None
    repo = "models--" + str(backbone).strip("/").replace("/", "--")
    d = Path(str(models)) / "hub" / repo / "snapshots" / revision
    return d if d.is_dir() else None


def tokenizer_source(backbone=DEFAULT_BACKBONE, revision=None, settings=None):
    """Where the tokenizer of (backbone, revision) loads from: the pinned tokenizer dir, else the HF
    snapshot of that revision, else a local backbone directory. FileNotFoundError otherwise (the
    tokenizer is never downloaded)."""
    if os.path.isdir(str(backbone)):
        return Path(str(backbone))
    for d in (tokenizer_dir(backbone, revision, settings), snapshot_dir(backbone, revision, settings)):
        if d is not None:
            return d
    raise FileNotFoundError(f"no local tokenizer for {backbone}@{revision}: expected "
                            f"<paths.tokenizers>/{_hub_name(backbone)}@{str(revision)[:8]} or the HF snapshot "
                            "under <paths.models>/hub")


_TOKENIZERS = {}                       # resolved source directory -> tokenizer (one load per process)


def load_tokenizer(backbone=DEFAULT_BACKBONE, revision=None, settings=None):
    """AutoTokenizer from local files only (tokenizer_source), left padding, pad = eos if missing.
    Loaded once per source directory and process."""
    src = str(tokenizer_source(backbone, revision, settings))
    tok = _TOKENIZERS.get(src)
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(src, local_files_only=True)
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        _TOKENIZERS[src] = tok
    return tok


def render_messages(tok, system, user):
    """The chat prompt of one (system, user) pair, ending with the empty think block (checked)."""
    text = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if not text.endswith(THINK_SUFFIX):
        raise ValueError(f"chat template did not end with the empty think block: {text[-40:]!r}")
    return text


def prompt_key(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def letter_ids(tok):
    """Token ids of A..D: each a single token on its own and all distinct."""
    ids = [tok.convert_tokens_to_ids(L) for L in LETTERS]
    for L, i in zip(LETTERS, ids):
        if not isinstance(i, int) or i == tok.unk_token_id:
            raise ValueError(f"letter {L} is not a vocabulary token ({i})")
        if tok(L, add_special_tokens=False)["input_ids"] != [i]:
            raise ValueError(f"letter {L} does not encode as the single token {i}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"letter ids {ids} are not distinct")
    return ids


def split_prefix(seqs):
    """Length of the longest common token prefix of a group's prompts, capped at (shortest - 1) so
    that every suffix keeps at least its answer-position token."""
    n = min(len(s) for s in seqs) - 1
    first = seqs[0]
    p = 0
    while p < n and all(s[p] == first[p] for s in seqs[1:]):
        p += 1
    return p


def share_units(seqs, groups):
    """[(group keys, prefix ids, [suffix ids per key])] for groups of prompt keys; `seqs` maps key ->
    token ids."""
    units = []
    for g in groups:
        ss = [seqs[k] for k in g]
        p = split_prefix(ss)
        units.append((list(g), ss[0][:p], [s[p:] for s in ss]))
    return units


def softmax_t(z, t):
    z = np.asarray(z, dtype=np.float64) / max(1e-3, float(t))
    z = np.exp(z - z.max())
    return z / z.sum()


# --------------------------------------------------------------------------- #
# Model (torch, imported lazily)
# --------------------------------------------------------------------------- #
def collate(items, pad_id, device):
    """Left-padded batch of {"ids": [...]} rows: the answer position is the last one of every row."""
    import torch
    L = max(len(it["ids"]) for it in items)
    ids = torch.full((len(items), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(items), L), dtype=torch.long)
    for i, it in enumerate(items):
        n = len(it["ids"])
        ids[i, L - n:] = torch.tensor(it["ids"])
        att[i, L - n:] = 1
    return ids.to(device), att.to(device)


def model_source(backbone=DEFAULT_BACKBONE, revision=None, settings=None):
    """(path or repo id, from_pretrained kwargs) of the backbone weights: a local directory, the HF
    snapshot of the pinned revision, else the hub id at that revision from <paths.models>/hub (local
    files only when settings.offline)."""
    if os.path.isdir(str(backbone)):
        return str(backbone), {}
    snap = snapshot_dir(backbone, revision, settings)
    if snap is not None:
        return str(snap), {}
    s = _settings(settings)
    kw = {"revision": pinned_revision(backbone, revision), "local_files_only": bool(getattr(s, "offline", True))}
    models = getattr(s.paths, "models", None)
    if models is not None:
        kw["cache_dir"] = str(Path(str(models)) / "hub")
    return str(backbone), kw


def load_model(backbone=DEFAULT_BACKBONE, revision=None, adapter_dir=None, device="cpu",
               dtype="bfloat16", settings=None):
    """The backbone (+ a peft adapter, not merged) in `dtype` on `device`."""
    import torch
    from transformers import AutoModelForCausalLM
    if dtype not in DTYPES:
        raise ValueError(f"dtype {dtype!r} not in {DTYPES}")
    src, kw = model_source(backbone, revision, settings)
    model = AutoModelForCausalLM.from_pretrained(src, dtype=getattr(torch, dtype), **kw)
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    return model.to(device)


# --------------------------------------------------------------------------- #
# The decision model
# --------------------------------------------------------------------------- #
class JevDecisionModel:
    """DecisionModel 'jev' / layout 'letter' (module docstring).

    Beyond the pinned arguments: cache (an AnswerCache to use instead of the (tag, ds) one), model (a
    preloaded causal LM, tests), dtype ('bfloat16' | 'float32'), verbose."""
    family = "jev"
    layout = "letter"
    variant = None                     # the run-record variant this class serves (None = HB's letters)

    def __init__(self, backbone=DEFAULT_BACKBONE, revision=DEFAULT_REVISION, adapter=None, device="cpu",
                 batch_size=32, prefix_share=False, grouping="direct", cache_tag=None, ds=None,
                 cache_only=False, forward=None, settings=None, *, cache=None, model=None, dtype="bfloat16",
                 verbose=True):
        if grouping not in GROUPINGS:
            raise ValueError(f"grouping {grouping!r} not in {GROUPINGS}")
        if dtype not in DTYPES:
            raise ValueError(f"dtype {dtype!r} not in {DTYPES}")
        self.settings = _settings(settings)
        self.backbone, self.device = str(backbone), str(device)
        self.revision = pinned_revision(self.backbone, revision)
        self.batch_size = max(1, min(int(batch_size), MAX_BS))
        self.prefix_share, self.grouping = bool(prefix_share), grouping
        self.cache_only, self.dtype, self.verbose, self.ds = bool(cache_only), dtype, verbose, ds
        self._seam = forward
        self._model = model
        if not self.cache_only and forward is None and model is None:
            from habitus.dm.gpu import check_device
            check_device(self.device, self.settings)
        self.adapter = resolve_adapter(adapter, self.backbone) if adapter else None
        self.adapter_hash = adapter_hash(self.adapter) if self.adapter else None
        if self.adapter:
            rv = run_record(self.adapter).get("variant")
            if rv not in (None, "lora") and rv != self.variant:
                raise ValueError(f"adapter {self.adapter} was trained as variant {rv!r}; serve it with the "
                                 f"matching decision model, not {type(self).__name__}")
        self.tag = cache_tag or engine_tag(self.backbone, self.adapter)
        if cache is None:
            base = overlay = None
            if ds is not None:
                base, overlay = cache_paths(self.tag, ds, self.settings)
            cache = AnswerCache(self.tag, ds, base=base, overlay=overlay, read_only=self.cache_only)
        self.cache = cache
        self._tok = self._letter_ids = None
        self.stats = {"questions": 0, "asked": 0, "cached": 0, "forwarded": 0, "seconds": 0.0, "low_cov": 0,
                      "non_finite": 0}

    # --- prompts --------------------------------------------------------------------------------- #
    @property
    def tok(self):
        if self._tok is None:
            self._tok = load_tokenizer(self.backbone, self.revision, self.settings)
        return self._tok

    @property
    def letter_ids(self):
        if self._letter_ids is None:
            self._letter_ids = letter_ids(self.tok)
        return self._letter_ids

    @property
    def model(self):
        return self._model

    def render(self, form):
        """The full chat prompt of one form; the answer letter comes next."""
        return render_messages(self.tok, form.system, form.user())

    def key(self, form):
        return prompt_key(self.render(form))

    def keys(self, qs):
        """{qkey: (prompt sha1 per form)}: no cache lookup, no model."""
        out = {}
        for qk, q in unique_questions(qs).items():
            out[qk] = tuple(self.key(f) for f in q.forms)
        return out

    def refused(self, qs):
        return {}

    def spec(self):
        return {"engine": self.tag, "family": self.family, "layout": self.layout, "backbone": self.backbone,
                "revision": self.revision, "adapter": self.adapter, "adapter_hash": self.adapter_hash,
                "prefix_share": self.prefix_share, "batch_size": self.batch_size, "grouping": self.grouping,
                "members": 1, "readout": "letter softmax T=1", "readout_dtype": self._readout_dtype()}

    def _readout_dtype(self):
        return READOUT_DTYPE_LM if self.dtype == "bfloat16" else f"{self.dtype} lm_head"

    # --- deciding -------------------------------------------------------------------------------- #
    def decide(self, qs):
        """{qkey: Decision} for every question (duplicate qkeys answered once)."""
        uq = unique_questions(qs)
        plan, jobs, groups = {}, {}, {}
        for qk, q in uq.items():
            n = q.n_options
            lst = []
            for f in q.forms:
                text = self.render(f)
                k = prompt_key(text)
                if k not in jobs and self.cache.get(k) is None:
                    jobs[k] = (text, n)
                    groups.setdefault(q.group, []).append(k)
                lst.append((f, k))
            plan[qk] = (n, lst)
        n_prompts = sum(len(v[1]) for v in plan.values())
        if jobs and self.cache_only:
            raise CacheMiss(list(jobs), f"[{self.tag}] {len(jobs)} of {n_prompts} prompts are not in the answer "
                                        f"cache (cache-only), first {list(jobs)[:3]}")
        if jobs:
            if not self.prefix_share:
                g = None
            elif self.grouping == "direct":
                g = list(groups.values())
            else:
                g = self._chunks(groups, jobs)
            self._run(jobs, g)
        out, low = {}, 0
        for qk, (n, lst) in plan.items():
            lp, cp, cov = {}, {}, {}
            for f, k in lst:
                a = self.cache.get(k)
                if int(a["n"]) != n:
                    raise ValueError(f"cached answer {k} offers {a['n']} letters, question {qk} {n}")
                p = tuple(float(x) for x in softmax_t(a["z"][:n], 1.0))
                lp[f.name] = (p,)
                cp[f.name] = (canonical(p, f.canon),)
                cov[f.name] = float(a["cov"])
                low += a["cov"] < LOW_COV
            out[qk] = Decision(qk, self.layout, tuple(lp), lp, cp, cov, min(cov.values()))
        self.stats["questions"] += len(plan)
        self.stats["asked"] += n_prompts
        self.stats["cached"] += n_prompts - len(jobs)
        if self.verbose and jobs:
            self._log(f"decided {len(plan)} questions = {n_prompts} prompts ({n_prompts - len(jobs)} cached/dup, "
                      f"{len(jobs)} forwarded); coverage < {LOW_COV}: {low / max(1, n_prompts):.4f} of prompts")
        return out

    def _chunks(self, groups, jobs):
        """Keys of each group sorted by prompt text, cut into chunks of <= batch_size rows (<= LONG_BS
        once a prompt of the group is > LONG tokens)."""
        keys = [k for g in groups.values() for k in g]
        if not keys:
            return []
        lens = dict(zip(keys, map(len, self.tok([jobs[k][0] for k in keys], add_special_tokens=False)["input_ids"])))
        out = []
        for g in groups.values():
            g = sorted(g, key=lambda k: jobs[k][0])
            cap = min(self.batch_size, LONG_BS) if max(lens[k] for k in g) > LONG else self.batch_size
            out += [g[i:i + cap] for i in range(0, len(g), cap)]
        return out

    # --- model ----------------------------------------------------------------------------------- #
    def _load_model(self):
        if self._seam is not None:
            return None
        if self._model is None:
            from habitus.dm.gpu import check_device
            check_device(self.device, self.settings)
            t0 = time.time()
            m = load_model(self.backbone, self.revision, self.adapter, self.device, self.dtype, self.settings)
            if self.adapter:
                m = m.merge_and_unload()
            self._model = m.eval()
            self._log(f"loaded {self.backbone}@{self.revision} adapter={self.adapter} on {self.device} in "
                      f"{time.time() - t0:.1f}s")
        self._attach(self._model)
        return self._model

    def _attach(self, model):
        """Hook on every model access before a forward (a head readout attaches here; idempotent)."""

    def _forward(self, seqs):
        """Letter logits [B, 4] and full-vocab logsumexp [B] at the answer position."""
        import torch
        with torch.no_grad():
            ids, att = collate([{"ids": s} for s in seqs], self.tok.pad_token_id, self.device)
            last = self._model(input_ids=ids, attention_mask=att, logits_to_keep=1,
                               use_cache=False).logits[:, -1, :].float()
            lse = torch.logsumexp(last, dim=-1)
            return last[:, self.letter_ids].cpu().numpy(), lse.cpu().numpy()

    def _forward_safe(self, seqs):
        """_forward, halving the batch on CUDA OOM (the halves run OUTSIDE the except block: inside it
        the traceback keeps the failed forward's activations alive)."""
        if self._seam is not None:
            return self._seam(seqs)
        import torch
        try:
            return self._forward(seqs)
        except torch.cuda.OutOfMemoryError:
            if len(seqs) == 1:
                raise
        torch.cuda.empty_cache()
        h = len(seqs) // 2
        self._log(f"CUDA OOM on a batch of {len(seqs)}: retrying as {h} + {len(seqs) - h}")
        (z1, l1), (z2, l2) = self._forward_safe(seqs[:h]), self._forward_safe(seqs[h:])
        return np.concatenate([z1, z2]), np.concatenate([l1, l2])

    def _forward_shared(self, units):
        """_forward for the prompts of `units` (share_units), each prefix forwarded once (module
        docstring). Returns letter logits [R, 4] and full-vocab logsumexp [R], rows in unit order."""
        import torch
        with torch.no_grad():
            m = self._model
            dec, head = m.model, m.lm_head
            pad, dev = self.tok.pad_token_id, self.device
            pre_ids, pre_att = collate([{"ids": u[1]} for u in units], pad, dev)
            cache = dec(input_ids=pre_ids, attention_mask=pre_att, use_cache=True).past_key_values
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
            last_pos = torch.tensor([len(s) - 1 for s in sufs], device=dev)
            h = h[torch.arange(R, device=dev), last_pos][:, None, :]          # [R, 1, H], as logits_to_keep=1
            last = head(h)[:, -1, :].float()
            lse = torch.logsumexp(last, dim=-1)
            return last[:, self.letter_ids].cpu().numpy(), lse.cpu().numpy()

    def _forward_shared_safe(self, units):
        """_forward_shared, halving the units (never splitting one) on CUDA OOM, as _forward_safe."""
        if self._seam is not None:
            return self._seam([u[1] + s for u in units for s in u[2]])
        import torch
        try:
            return self._forward_shared(units)
        except torch.cuda.OutOfMemoryError:
            if len(units) == 1:
                raise
        torch.cuda.empty_cache()
        h = len(units) // 2
        self._log(f"CUDA OOM on a shared batch of {len(units)} groups: retrying as {h} + {len(units) - h}")
        (z1, l1), (z2, l2) = self._forward_shared_safe(units[:h]), self._forward_shared_safe(units[h:])
        return np.concatenate([z1, z2]), np.concatenate([l1, l2])

    def _record(self, key, n, z, lse):
        """Cache one answer; False (nothing cached) when its logits / logsumexp / coverage are not finite."""
        cov = float(np.exp(np.asarray(z[:n], dtype=np.float64) - float(lse)).sum())
        rec = {"z": [round(float(x), 4) for x in z], "lse": round(float(lse), 4), "n": n, "cov": round(cov, 6)}
        from habitus.dm.cache import finite_record
        if not finite_record(rec):
            self.stats["non_finite"] += 1
            self._log(f"non-finite answer for prompt {key}: z={rec['z']} lse={rec['lse']} (not cached)")
            return False
        self.cache.put(key, rec)
        self.stats["low_cov"] += cov < LOW_COV
        return True

    def _fail_non_finite(self, key, n_tokens):
        self.cache.flush()
        raise RuntimeError(f"[dm.jev {self.tag}] non-finite logits for prompt {key} ({n_tokens} tokens), "
                           "also when forwarded alone")

    def _run(self, jobs, groups=None):
        """jobs: {key: (text, n_letters)} not yet answered -> records in the cache. groups (prefix_share
        only): lists of job keys whose prompts share a prefix."""
        if not jobs:
            return
        if self.prefix_share and groups:
            return self._run_shared(jobs, groups)
        self._load_model()
        keys = list(jobs)
        seqs = self.tok([jobs[k][0] for k in keys], add_special_tokens=False)["input_ids"]
        order = sorted(range(len(keys)), key=lambda i: (len(seqs[i]), i))   # length buckets
        t0, i = time.time(), 0
        while i < len(order):
            bs = self.batch_size
            if len(seqs[order[min(i + bs, len(order)) - 1]]) > LONG:
                bs = min(bs, LONG_BS)
            idx = order[i:i + bs]
            i += len(idx)
            z, lse = self._forward_safe([seqs[j] for j in idx])
            bad = [j for r, j in enumerate(idx) if not self._record(keys[j], jobs[keys[j]][1], z[r], lse[r])]
            for j in bad:                       # once more alone (no padding), then fail loudly
                z1, l1 = self._forward_safe([seqs[j]])
                if not self._record(keys[j], jobs[keys[j]][1], z1[0], l1[0]):
                    self._fail_non_finite(keys[j], len(seqs[j]))
            if self.cache.pending >= FLUSH:
                self.cache.flush()
        self.cache.flush()
        dt = time.time() - t0
        self.stats["forwarded"] += len(keys)
        self.stats["seconds"] += dt
        lens = np.array([len(s) for s in seqs])
        self._log(f"forwarded {len(keys)} prompts in {dt:.1f}s = {len(keys) / max(dt, 1e-9):.1f} prompts/s "
                  f"(tokens p50 {int(np.median(lens))} max {lens.max()})")

    def _run_shared(self, jobs, groups):
        """_run with one prefix forward per group. Same records, keys, flushing and non-finite rule (a
        non-finite answer is retried ALONE with the full forward)."""
        keys = list(jobs)
        seqs = dict(zip(keys, self.tok([jobs[k][0] for k in keys], add_special_tokens=False)["input_ids"]))
        seen, gs = set(), []
        for g in groups:
            g = [k for k in dict.fromkeys(g) if k in jobs and k not in seen]
            seen.update(g)
            if g:
                gs.append(g)
        gs += [[k] for k in keys if k not in seen]
        units = share_units(seqs, gs)
        plain = {k: jobs[k] for u in units if not u[1] for k in u[0]}   # no common prefix: full path
        units = [u for u in units if u[1]]
        if plain:
            self._run(plain)
        if not units:
            return
        self._load_model()
        order = sorted(range(len(units)), key=lambda i: (len(units[i][1]), max(map(len, units[i][2])), i))
        longest = [len(units[j][1]) + max(map(len, units[j][2])) for j in range(len(units))]
        t0, i, n_fwd = time.time(), 0, 0
        while i < len(order):
            # rows per forward as in _run: <= batch_size, <= LONG_BS once a prompt is > LONG; a
            # group's prompts are never split across forwards
            batch, rows, top = [], 0, 0
            while i < len(order):
                j = order[i]
                cap = min(self.batch_size, LONG_BS) if max(top, longest[j]) > LONG else self.batch_size
                if batch and rows + len(units[j][0]) > cap:
                    break
                batch.append(units[j])
                rows += len(units[j][0])
                top = max(top, longest[j])
                i += 1
            z, lse = self._forward_shared_safe(batch)
            n_fwd += sum(len(u[1]) + sum(map(len, u[2])) for u in batch)
            bad, r = [], 0
            for u in batch:
                for k in u[0]:
                    if not self._record(k, jobs[k][1], z[r], lse[r]):
                        bad.append(k)
                    r += 1
            for k in bad:                       # once more alone, full forward (no padding), then fail loudly
                z1, l1 = self._forward_safe([seqs[k]])
                if not self._record(k, jobs[k][1], z1[0], l1[0]):
                    self._fail_non_finite(k, len(seqs[k]))
            if self.cache.pending >= FLUSH:
                self.cache.flush()
        self.cache.flush()
        dt = time.time() - t0
        n = sum(len(u[0]) for u in units)
        self.stats["forwarded"] += n
        self.stats["seconds"] += dt
        lens = np.array([len(seqs[k]) for u in units for k in u[0]])
        self._log(f"forwarded {n} prompts of {len(units)} groups (prefix shared) in {dt:.1f}s = "
                  f"{n / max(dt, 1e-9):.1f} prompts/s (tokens p50 {int(np.median(lens))} max {lens.max()}; "
                  f"{n_fwd} of {int(lens.sum())} prompt tokens forwarded)")

    # --- lifetime -------------------------------------------------------------------------------- #
    def flush(self):
        self.cache.flush()

    def close(self):
        self.cache.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _log(self, msg):
        if self.verbose:
            _log(f"[{self.tag}] {msg}")
