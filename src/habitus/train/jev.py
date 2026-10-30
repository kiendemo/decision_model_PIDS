"""Training a jev decision model on habit items: a LoRA (variant 'lora': HB, HB.08b) or a LoRA + fp32
letter head (variant 'head': HB.head).

    rec = train_jev(items, inner, "runs/adapters/loho-THEIA_E3/HB", backbone="Qwen/Qwen3.5-4B",
                    revision=DEFAULT_REVISION, fold="loho-THEIA_E3", device="cuda:0")

The model learns to answer exported habit items with the SAME readout it is served with: the
next-token logits of the offered letters (<= 4) at the answer position after the chat template's
empty think block (variant 'lora'), or the fp32 letter head on the final hidden state there (variant
'head', dm/jev_head.py). No generation.

Loss: calibrate.soft_ce over the offered letters, the item's soft target in its letter order,
weighted by the item weight, normalised by (rows of the optimizer batch) x (mean weight over the
training set), so that a batch of light items moves the model less than a batch of heavy ones and
gradient accumulation over micro-batches sums to the batch loss.

Recipe (RECIPE, fixed a priori; configs/train/jev-lora.yaml mirrors it): 1 epoch; AdamW lr 1e-4,
weight decay 0; cosine decay to 0 after a 3 % linear warmup; effective batch 16 (micro-batches of 8
rows, gradients accumulated); length-bucketed batches (seeded shuffle, sort by token length inside
windows of 64 batches, shuffled batch order); grad clip 1.0; gradient checkpointing; bf16 weights
with CUDA autocast. LoRA r 16, alpha 32, dropout 0.05 on q / k / v / o (the full-attention layers)
and gate / up / down (every MLP); `verify_targets` refuses a target that matches no Linear module.
Variant 'head' adds RECIPE_HEAD (configs/train/jev-head.yaml): head lr 2e-4 as a second AdamW group,
norm statistics fitted on 256 training items at step 0.

Temperatures (calibrate.fit_temperatures) are fitted on the INNER-day items only and recorded; serving
is at T = 1.

Outputs: <out_dir>/adapter/ (peft save_pretrained), <out_dir>/head/ (variant 'head') and
<out_dir>/habitus.json {fold, backbone, revision, recipe, targets, temperatures, items_sha256,
code_sha256, adapter_hash, inner_report, ...}; a run directory that already holds habitus.json is
refused.

Label-free: items carry targets computed by code from pre-deployment clean days of the fold's
training hosts; each item's dataset is re-checked against the fold here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np

from habitus.dm.jev import (DEFAULT_BACKBONE, DEFAULT_REVISION, THINK_SUFFIX, adapter_hash, collate, engine_tag,
                            letter_ids, load_tokenizer, model_source)
from habitus.train.calibrate import FORMS, calibration_report, fit_temperatures, form_of, soft_ce

BACKBONE = DEFAULT_BACKBONE
FALLBACK_BACKBONE = "Qwen/Qwen3.5-0.8B"
SMOKE_MIN_RATE = 2.0                  # items/s; below it the 0.8B fallback applies
SMOKE_N = 2000
SPEC_VERSION = "HB-v0-draft"
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
RECIPE = {
    "epochs": 1, "lr": 1e-4, "weight_decay": 0.0, "schedule": "cosine", "warmup": 0.03,
    "batch_size": 16,                 # effective batch (one optimizer step)
    "micro_batch": 8,                 # rows per forward; batch_size / micro_batch accumulation steps
    "megabatch": 64,                  # length_buckets: batches sorted by length inside 64-batch windows
    "grad_clip": 1.0, "grad_ckpt": True, "bf16": True,
    "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "seed": 0,
}
VARIANTS = ("lora", "head")
TRAIN_ROLES = ("train", "inner")
MAX_OPTIONS = 4
LOG_EVERY = 50                        # optimizer steps between progress lines
RUN_CONFIG = "habitus.json"


def _log(msg):
    from habitus.log import log
    log(msg, "train.jev")


# --------------------------------------------------------------------------- #
# The LoRA wrapper
# --------------------------------------------------------------------------- #
def verify_targets(model, targets=LORA_TARGETS):
    """{target name: number of Linear modules whose name ends with it}; ValueError for a name that
    matches none (that LoRA target would silently be a no-op)."""
    import torch
    counts = {t: 0 for t in targets}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            leaf = name.rsplit(".", 1)[-1]
            if leaf in counts:
                counts[leaf] += 1
    missing = [t for t, n in counts.items() if not n]
    if missing:
        raise ValueError(f"LoRA targets {missing} match no Linear module of {type(model).__name__}")
    return counts


def wrap_lora(model, targets=LORA_TARGETS, recipe=RECIPE):
    """A fresh trainable LoRA (r / alpha / dropout of `recipe`) on exactly `targets`."""
    from peft import LoraConfig, get_peft_model
    verify_targets(model, targets)
    cfg = LoraConfig(r=recipe["lora_r"], lora_alpha=recipe["lora_alpha"], lora_dropout=recipe["lora_dropout"],
                     target_modules=list(targets), task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


def load_backbone(backbone, revision=DEFAULT_REVISION, bf16=True, settings=None):
    import torch
    from transformers import AutoModelForCausalLM
    src, kw = model_source(backbone, revision, settings)
    return AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16 if bf16 else torch.float32, **kw)


# --------------------------------------------------------------------------- #
# Items
# --------------------------------------------------------------------------- #
def _folds():
    from habitus.dm import folds
    return folds()


def check_fold_items(items, fold):
    """Every item's dataset is a training host of `fold` (never its held-out host), its fold field is
    empty or `fold`, and its role is a training role."""
    fs = _folds()
    if fold not in fs:
        raise ValueError(f"fold {fold!r} not in {sorted(fs)}")
    held = fold[len("loho-"):] if fold.startswith("loho-") else None
    allowed = set(fs[fold]) - {held}
    bad = sorted({(it.ds, it.fold) for it in items if it.ds not in allowed or it.fold not in ("", fold)})
    if bad:
        raise ValueError(f"fold {fold}: items from {bad[:5]} are not trainable")
    roles = sorted({it.role for it in items if it.role not in TRAIN_ROLES})
    if roles:
        raise ValueError(f"fold {fold}: items of role {roles} are not trainable (only {TRAIN_ROLES})")


def items_sha256(items):
    """sha256 of the items' canonical JSON lines, in order."""
    h = hashlib.sha256()
    for it in items:
        h.update(it.dumps().encode("utf-8") + b"\n")
    return h.hexdigest()


def code_sha256():
    """sha256 over (relative path, sha256) of every .py file of the habitus package, sorted."""
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for f in sorted(root.rglob("*.py")):
        rel = f.relative_to(root).as_posix()
        h.update(f"{rel}\x00{hashlib.sha256(f.read_bytes()).hexdigest()}\n".encode())
    return h.hexdigest()


def encode(tok, items):
    """[{"ids", "q": offered letters, "target", "w"}] per item; the prompt must end with the empty
    think block (the answer position is the last token, as when served)."""
    texts = [it.prompt_text for it in items]
    for it, t in zip(items, texts):
        if not t.endswith(THINK_SUFFIX):
            raise ValueError(f"item {it.qkey} ({it.variant}): prompt does not end with the empty think block")
    ids = tok(texts, add_special_tokens=False)["input_ids"] if texts else []
    return [{"ids": s, "q": len(it.letters), "target": list(it.target), "w": float(it.weight)}
            for it, s in zip(items, ids)]


def option_table(tok):
    """{n offered letters: token ids of A.. (first n)}."""
    lid = letter_ids(tok)
    return {n: lid[:n] for n in range(2, MAX_OPTIONS + 1)}


def length_buckets(items, batch_size, seed, megabatch=RECIPE["megabatch"]):
    """Batches of item indices (each <= batch_size) of similar token length: seeded shuffle, then
    inside each window of `megabatch` batches a stable sort by length, cut into batches, and a seeded
    shuffle of the batch order. Items are ints (lengths), {"ids": ...} dicts or Items (prompt character
    length as the proxy). Every index appears exactly once; same seed -> same batches."""
    def length(it):
        if isinstance(it, (int, np.integer)):
            return int(it)
        if isinstance(it, dict):
            return len(it["ids"])
        return len(it.prompt_text)
    lens = [length(it) for it in items]
    rng = random.Random(seed)
    idx = list(range(len(lens)))
    rng.shuffle(idx)
    win = max(1, int(batch_size) * int(megabatch))
    batches = []
    for s in range(0, len(idx), win):
        w = sorted(idx[s:s + win], key=lambda i: lens[i])
        batches += [w[j:j + batch_size] for j in range(0, len(w), batch_size)]
    rng.shuffle(batches)
    return batches


# --------------------------------------------------------------------------- #
# Letter logits (the readout as a differentiable function)
# --------------------------------------------------------------------------- #
def option_logits(model, ids, att, opt_ids, mask):
    """[B, Kmax] logits of the option tokens at the answer position (masked -> -1e4)."""
    import torch
    out = model(input_ids=ids, attention_mask=att, logits_to_keep=1, use_cache=False)
    last = out.logits[:, -1, :].float()
    z = torch.gather(last, 1, opt_ids.clamp(min=0))
    return z.masked_fill(~mask, -1e4)


def batch_options(items, opt_table, device):
    """Padded option-id tensor and mask for a batch of encoded rows."""
    import torch
    K = max(len(opt_table[it["q"]]) for it in items)
    opt = torch.full((len(items), K), -1, dtype=torch.long)
    mask = torch.zeros((len(items), K), dtype=torch.bool)
    for i, it in enumerate(items):
        o = opt_table[it["q"]]
        opt[i, :len(o)] = torch.tensor(o)
        mask[i, :len(o)] = True
    return opt.to(device), mask.to(device)


def logits_for(model, tok, items, opt_table, device="cpu", batch_size=32):
    """Raw option logits per encoded row (1-D numpy arrays, one entry per offered letter)."""
    import torch
    model.eval()
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))   # length buckets
    res = [None] * len(items)
    with torch.no_grad():
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            b = [items[i] for i in idx]
            ids, att = collate(b, tok.pad_token_id, device)
            opt, mask = batch_options(b, opt_table, device)
            z = option_logits(model, ids, att, opt, mask).cpu().numpy()
            for r, i in enumerate(idx):
                res[i] = z[r, :len(opt_table[b[r]["q"]])]
    return res


def letter_logits(model, tok, items, device, batch_size=32, readout=None):
    """Offered-letter logits per item (1-D float64 arrays, the item's letter order)."""
    enc = encode(tok, items)
    if not enc:
        return []
    if readout is None:
        z = logits_for(model, tok, enc, option_table(tok), device, batch_size)
    else:
        z = readout.logits_for(model, tok, enc, option_table(tok), device, batch_size)
    return [np.asarray(x, dtype=np.float64) for x in z]


# --------------------------------------------------------------------------- #
# The training loop
# --------------------------------------------------------------------------- #
def _schedule(steps, warmup):
    warm = max(1, int(round(warmup * steps)))

    def f(s):
        if s < warm:
            return (s + 1) / warm
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, (s - warm) / max(1, steps - warm))))
    return f


def _fit(model, enc, batches, rec, device, opt_table, pad_id, max_steps=None, t_skip=0, readout=None):
    """The training loop over `batches` (lists of indices into enc): returns stats. t_skip: the first
    t_skip optimizer steps are excluded from the timing (smoke). readout: the head variant's readout;
    its parameters are one more AdamW group at rec["head_lr"]."""
    import torch
    params = [p for p in model.parameters() if p.requires_grad]
    hp = readout.parameters() if readout is not None else []
    groups = params if not hp else [{"params": params}, {"params": hp, "lr": rec["head_lr"]}]
    opt = torch.optim.AdamW(groups, lr=rec["lr"], weight_decay=rec["weight_decay"])
    steps = rec["epochs"] * len(batches) if max_steps is None else min(max_steps, rec["epochs"] * len(batches))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _schedule(steps, rec["warmup"]))
    w_scale = float(np.mean([e["w"] for e in enc])) or 1.0
    amp = bool(rec["bf16"]) and str(device).startswith("cuda")
    mb = max(1, int(rec["micro_batch"]))
    model.train()
    if readout is not None:
        readout.train(True)
    st = {"steps": 0, "items": 0, "padded_tokens": 0, "losses": [], "seconds": 0.0, "timed_items": 0,
          "timed_tokens": 0}
    t0 = tt = time.time()
    for ep in range(rec["epochs"]):
        order = batches if ep == 0 else length_buckets(enc, rec["batch_size"], rec["seed"] + ep, rec["megabatch"])
        for b in order:
            if st["steps"] >= steps:
                break
            tot = 0.0
            for s in range(0, len(b), mb):
                rows = [enc[i] for i in b[s:s + mb]]
                ids, att = collate(rows, pad_id, device)
                o_ids, mask = batch_options(rows, opt_table, device)
                tgt = torch.zeros(mask.shape, dtype=torch.float32, device=device)
                for r, e in enumerate(rows):
                    tgt[r, :len(e["target"])] = torch.tensor(e["target"], dtype=torch.float32)
                w = torch.tensor([e["w"] for e in rows], dtype=torch.float32, device=device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                    if readout is None:
                        z = option_logits(model, ids, att, o_ids, mask)
                    else:
                        z = readout.option_logits(model, ids, att, o_ids, mask, rows)
                loss = soft_ce(z, tgt, mask, w, scale=len(b) * w_scale)
                if not torch.isfinite(loss):                 # fail loudly, never step on a NaN
                    raise RuntimeError(f"non-finite training loss at step {st['steps']} (batch of {len(b)})")
                loss.backward()
                tot += float(loss.detach())
                st["padded_tokens"] += int(ids.shape[0] * ids.shape[1])
                if st["steps"] >= t_skip:
                    st["timed_tokens"] += int(ids.shape[0] * ids.shape[1])
            torch.nn.utils.clip_grad_norm_(params + hp, rec["grad_clip"])
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            st["steps"] += 1
            st["items"] += len(b)
            st["losses"].append(tot)
            if st["steps"] == t_skip:
                tt = time.time()
            elif st["steps"] > t_skip:
                st["timed_items"] += len(b)
            if st["steps"] % LOG_EVERY == 0:
                el = time.time() - t0
                _log(f"step {st['steps']}/{steps}: loss {np.mean(st['losses'][-LOG_EVERY:]):.4f}, "
                     f"{st['items'] / el:.2f} items/s, {st['padded_tokens'] / el:.0f} padded tokens/s, "
                     f"lr {sched.get_last_lr()[0]:.2e}")
    st["seconds"] = time.time() - t0
    st["timed_seconds"] = time.time() - tt
    model.eval()
    if readout is not None:
        readout.train(False)
    return st


def _loss_summary(losses):
    n = len(losses)
    if not n:
        return {}
    k = max(1, n // 10)
    return {"first10pct": float(np.mean(losses[:k])), "last10pct": float(np.mean(losses[-k:])),
            "mean": float(np.mean(losses)), "steps": n}


def _count(items, key):
    out = {}
    for it in items:
        k = key(it)
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def _seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def variant_recipe(variant, recipe=None):
    """RECIPE, then the variant's defaults (RECIPE_HEAD for 'head'), then the caller's recipe: None, a
    flat override mapping, or a configs/train/*.yaml mapping (its "recipe" entry is used, as
    train.laya.resolve_recipe does). Unknown keys raise."""
    if variant not in VARIANTS:
        raise ValueError(f"variant {variant!r} not in {VARIANTS}")
    if variant == "lora":
        base = dict(RECIPE)
    else:
        from habitus.dm.jev_head import recipe as head_recipe
        base = {**RECIPE, **head_recipe()}
    if not recipe:
        return base
    r = dict(recipe)
    if isinstance(r.get("recipe"), dict):
        r = dict(r["recipe"])
    unknown = sorted(set(r) - set(base))
    if unknown:
        raise ValueError(f"unknown {variant} recipe keys {unknown} (known {sorted(base)})")
    return {**base, **r}


def _prepare(backbone, revision, device, rec, model, tok, settings):
    tok = tok if tok is not None else load_tokenizer(backbone, revision, settings)
    if model is None:
        model = load_backbone(backbone, revision, rec["bf16"], settings)
    counts = verify_targets(model, LORA_TARGETS)
    model = wrap_lora(model, LORA_TARGETS, rec).to(device)
    if rec["grad_ckpt"]:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    return model, tok, counts


def _readout(variant, model, tok, device):
    if variant == "lora":
        return None
    from habitus.dm.jev_head import train_readout
    return train_readout(model, tok, device)


def _split(items, inner):
    items = list(items)
    if inner is None:
        return [it for it in items if it.role == "train"], [it for it in items if it.role == "inner"]
    inner = list(inner)
    bad = sorted({it.role for it in items if it.role != "train"} | {it.role for it in inner if it.role != "inner"})
    if bad:
        raise ValueError(f"train items must have role 'train' and inner items role 'inner', got {bad}")
    return items, inner


def train_jev(items, inner, out_dir, *, backbone=BACKBONE, revision=DEFAULT_REVISION, fold, device="cpu",
              recipe=None, variant="lora", model=None, tok=None, ref_p=None, settings=None):
    """Train one fold's LoRA (+ head) on the role-'train' items, fit temperatures on the role-'inner'
    items (inner=None: both taken from `items` by role), write <out_dir>/adapter (+ head) and
    habitus.json. model / tok: a preloaded backbone and tokenizer (tests). ref_p: comparator p_yes per
    qkey for the inner-day report. Returns the run record."""
    from habitus.dm.gpu import check_device
    rec = variant_recipe(variant, recipe)
    check_device(device, settings)
    tr, inner = _split(items, inner)
    check_fold_items(tr + inner, fold)
    if not tr:
        raise ValueError(f"fold {fold}: no training items")
    out = Path(out_dir)
    cfg = out / RUN_CONFIG
    if cfg.exists():
        raise FileExistsError(f"{out} already holds a trained run")
    out.mkdir(parents=True, exist_ok=True)
    _seed(rec["seed"])
    t0 = time.time()
    model, tok, counts = _prepare(backbone, revision, device, rec, model, tok, settings)
    readout = _readout(variant, model, tok, device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) + (readout.n_params if readout else 0)
    enc = encode(tok, tr)
    if readout is not None:
        readout.prepare(model, enc, device, tok.pad_token_id, rec)
    batches = length_buckets(enc, rec["batch_size"], rec["seed"], rec["megabatch"])
    lens = np.array([len(e["ids"]) for e in enc])
    _log(f"fold {fold}: {len(tr)} training items ({len(batches)} steps of {rec['batch_size']}), {len(inner)} "
         f"inner items; {backbone} LoRA r={rec['lora_r']} on {LORA_TARGETS} ({variant}; "
         f"{n_train / 1e6:.2f}M trainable); tokens p50 {int(np.median(lens))} max {int(lens.max())}; device {device}")
    st = _fit(model, enc, batches, rec, device, option_table(tok), tok.pad_token_id, readout=readout)
    _log(f"fold {fold}: {st['steps']} steps in {st['seconds']:.0f}s = {st['items'] / max(st['seconds'], 1e-9):.2f} "
         f"items/s; loss {_loss_summary(st['losses'])}")
    temps, report = {f: 1.0 for f in FORMS}, None
    if inner:
        zl = letter_logits(model, tok, inner, device, batch_size=2 * rec["batch_size"], readout=readout)
        temps = fit_temperatures(zl, inner)
        report = calibration_report(zl, inner, temps, ref_p=ref_p)
        _log(f"fold {fold}: temperatures {temps}; inner-day nll {report['nll']}, ece15 {report['ece15']}, "
             f"tail log score {report['tail_log_score']}, rare AUC {report['rare_auc']}")
    adir = out / "adapter"
    model.save_pretrained(str(adir))
    extra = {}
    if readout is not None:
        from habitus.dm.jev_head import save_run
        extra = save_run(readout, str(out))
    rec_out = {
        "fold": fold, "backbone": str(backbone), "revision": revision, "recipe": rec, "targets": list(LORA_TARGETS),
        "target_modules": counts, "temperatures": temps,
        "items_sha256": items_sha256(tr), "inner_sha256": items_sha256(inner), "code_sha256": code_sha256(),
        "spec": SPEC_VERSION, "adapter_hash": adapter_hash(str(adir)), "n_items": len(tr), "n_inner": len(inner),
        "per_ds": _count(tr, lambda it: it.ds), "per_form": _count(tr, form_of),
        "per_src": _count(tr, lambda it: it.src),
        "per_kmode": _count(tr, lambda it: it.k_mode), "trainable_params": int(n_train),
        "loss": _loss_summary(st["losses"]), "steps": st["steps"],
        "seconds": round(time.time() - t0, 1), "items_per_s": st["items"] / max(st["seconds"], 1e-9),
        "padded_tokens_per_s": st["padded_tokens"] / max(st["seconds"], 1e-9), "inner_report": report,
        "device": str(device), "written": time.strftime("%Y-%m-%d %H:%M:%S"), **extra,
    }
    tmp = out / (RUN_CONFIG + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec_out, f, indent=1, sort_keys=True)
    os.replace(tmp, cfg)
    _log(f"fold {fold}: adapter -> {adir} (hash {rec_out['adapter_hash'][:12]})")
    return rec_out


def smoke(items, backbone=BACKBONE, revision=DEFAULT_REVISION, device="cpu", n=SMOKE_N, recipe=None, variant="lora",
          model=None, tok=None, warmup_steps=2, infer=True, settings=None):
    """Training throughput on the first n role-'train' items (the recipe's loop, warm-up steps
    excluded from the timing) and, with infer, forward-only readout throughput on the same items. No
    adapter is written. Fallback rule: train items/s < SMOKE_MIN_RATE -> FALLBACK_BACKBONE with the
    identical recipe."""
    import torch
    from habitus.dm.gpu import check_device
    rec = variant_recipe(variant, recipe)
    check_device(device, settings)
    tr = [it for it in items if it.role == "train"][:int(n)]
    if not tr:
        raise ValueError("smoke: no training items")
    _seed(rec["seed"])
    model, tok, _counts = _prepare(backbone, revision, device, rec, model, tok, settings)
    readout = _readout(variant, model, tok, device)
    enc = encode(tok, tr)
    if readout is not None:
        readout.prepare(model, enc, device, tok.pad_token_id, rec)
    batches = length_buckets(enc, rec["batch_size"], rec["seed"], rec["megabatch"])
    cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    skip = min(warmup_steps, max(0, len(batches) - 1))
    st = _fit(model, enc, batches, rec, device, option_table(tok), tok.pad_token_id, t_skip=skip, readout=readout)
    rate = st["timed_items"] / max(st["timed_seconds"], 1e-9)
    lens = np.array([len(e["ids"]) for e in enc])
    out = {"backbone": str(backbone), "revision": revision, "variant": variant, "targets": list(LORA_TARGETS),
           "device": str(device), "n": len(tr), "steps": st["steps"], "warmup_steps_excluded": skip,
           "train_items_per_s": rate, "padded_tokens_per_s": st["timed_tokens"] / max(st["timed_seconds"], 1e-9),
           "tokens_p50": int(np.median(lens)), "tokens_max": int(lens.max()), "recipe": rec,
           "fallback": bool(rate < SMOKE_MIN_RATE), "smoke_min_rate": SMOKE_MIN_RATE,
           "fallback_backbone": FALLBACK_BACKBONE if rate < SMOKE_MIN_RATE and backbone != FALLBACK_BACKBONE else None,
           "per_form": _count(tr, form_of), "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    if infer:
        t1 = time.time()
        letter_logits(model, tok, tr, device, batch_size=2 * rec["batch_size"], readout=readout)
        out["infer_items_per_s"] = len(tr) / max(time.time() - t1, 1e-9)
    if cuda:
        out["peak_mem_gb"] = round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 2)
    _log(f"smoke {engine_tag(backbone)} ({variant}): {rate:.2f} train items/s "
         f"({out['padded_tokens_per_s']:.0f} padded tokens/s), infer {out.get('infer_items_per_s', 0):.2f} items/s; "
         f"below {SMOKE_MIN_RATE} items/s: {'yes' if out['fallback'] else 'no'}"
         f"{' -> ' + out['fallback_backbone'] if out['fallback_backbone'] else ''}")
    return out
