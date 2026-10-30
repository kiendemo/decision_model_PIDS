"""Trainer of the 'laya' decision model in layout MQ (encode once, decide many; habitus.dm.laya_mq).

`train_mq` has train.laya.train_laya's signature and return value (train_laya(layout="mq") dispatches
here) and reuses its optimiser groups, schedule, calibration, run directory and ensemble record. Only
the rows differ: `item_rows` packs the items by view_sha1 (each item one block in its own letter
order, so its target is aligned with the markers; by (I1) the grouping changes no logit),
`train_batches` cuts optimizer batches of >= batch_size ITEMS from length-bucketed rows, and `mq_loss`
is train.calibrate.soft_ce over the blocks (normaliser: items of the optimizer batch x mean weight).

RECIPE_MQ = train.laya.RECIPE_L with gradient checkpointing off (it changes no number) plus the MQ keys
(layout, row_tokens, head). A run directory is a Laya checkpoint whose habitus.json says layout "mq":
laya.resolve_init refuses it (an MQ model is never read as L-a), laya_mq.resolve_mq accepts it.

Label-free: this module tokenises the habit items handed to it, loads model weights and writes its own
run directories; it opens no card, link, score or label file and imports no metric code.
"""

import hashlib
import random
import time

import numpy as np
import torch

from habitus.dm import laya as LA
from habitus.dm import laya_mq as MQ
from habitus.questions.habit.constants import SPEC_VERSION
from habitus.train import calibrate as CAL
from habitus.train import laya as TL

SEEDS = TL.SEEDS
ARM = "HB-L/MQ"                        # the run record's arm field
RUN_BASE = "HB-L.mq"                   # the default run base name: <out_dir>/HB-L.mq-s<seed>
LOG_EVERY = 50
# train.laya.RECIPE_L's vocabulary and values (parity tested), plus the MQ keys; grad_ckpt off.
RECIPE_MQ = {
    "epochs": 1, "encoder_lr": 2.5e-5, "head_lr": 1e-4, "min_lr": 1e-6, "weight_decay": 0.01,
    "schedule": "cosine", "warmup": 0.03,
    "batch_size": 64,                  # items (= blocks) per optimizer step
    "micro_batch": 16,                 # rows per forward (gradients accumulated)
    "max_tokens": 16384,               # padded tokens (rows x longest row) per training forward
    "megabatch": 64,                   # length bucketing: rows sorted by length inside 64-batch windows
    "grad_clip": 1.0, "grad_ckpt": False, "bf16": True, "seed": 0, "loss": "soft-ce",
    "max_len": MQ.MAX_LEN, "head_max_len": MQ.HEAD_MAX_LEN, "qtype": LA.TRAIN_QTYPE,
    "layout": MQ.LAYOUT, "row_tokens": MQ.ROW_TOKENS, "head": MQ.HEAD,
}
MQ_KEYS = ("layout", "row_tokens", "head")      # the keys RECIPE_L does not have


def _log(msg, tag="habitus.train.laya_mq"):
    from habitus.log import log
    log(msg, tag)


# --------------------------------------------------------------------------- #
# Items -> packed rows, batches, loss, letter-order logits
# --------------------------------------------------------------------------- #
def item_targets(items):
    """Per item, its letter-order training target."""
    return [list(it.target) for it in items]


def item_rows(tok, items, row_tokens=MQ.ROW_TOKENS, max_len=MQ.MAX_LEN, head_max_len=MQ.HEAD_MAX_LEN,
              qtype=LA.TRAIN_QTYPE, seg=None):
    """[(row layout, [item index per block])]: each item one block (laya.split_prompt; options in the
    ITEM's letter order), items grouped by view_sha1 (the parsed state text must hash to it) and packed
    into rows in order of first appearance."""
    seg = seg or LA.SegmentTokens(tok)
    parts, texts = [], set()
    for it in items:
        p = LA.split_prompt(it.prompt_text)
        if len(p.options) != len(it.letters):
            raise LA.PromptError(f"item {it.qkey}: {len(p.options)} options for letters {it.letters}")
        if hashlib.sha1(p.state.encode("utf-8")).hexdigest() != it.view_sha1:
            raise ValueError(f"item {it.qkey}: its prompt's view does not hash to view_sha1 {it.view_sha1}")
        parts.append(p)
        qs, os_, vs = LA.segments(p, qtype)
        texts.update((qs, vs, *os_))
    seg.warm(texts)
    groups, order = {}, []
    for i, (it, p) in enumerate(zip(items, parts)):
        try:
            st, blk = MQ.encode_parts(tok, p, qtype, head_max_len, max_len, seg)
        except LA.LengthError as e:
            raise LA.LengthError(f"item {it.qkey} ({it.variant}): {e}") from None
        g = groups.get(it.view_sha1)
        if g is None:
            g = groups[it.view_sha1] = {"state": st, "blocks": [], "idx": []}
            order.append(it.view_sha1)
        g["blocks"].append(blk)
        g["idx"].append(i)
    out = []
    for v in order:
        g = groups[v]
        for row in MQ.pack(g["state"], g["blocks"], row_tokens):
            out.append((MQ.row_layout(g["state"], [g["blocks"][j] for j in row]), [g["idx"][j] for j in row]))
    return out


def train_batches(rows, batch_size, seed, megabatch=RECIPE_MQ["megabatch"]):
    """Optimizer batches of row indices holding >= batch_size items each (a window's last batch may hold
    fewer), rows of similar length together: seeded shuffle, stable sort by row length inside windows of
    `megabatch` batches, cut, seeded shuffle of the batch order. Every row appears exactly once."""
    if not rows:
        return []
    n_items = [len(r[1]) for r in rows]
    lens = [len(r[0]["ids"]) for r in rows]
    per = max(1, int(round(batch_size / max(1e-9, float(np.mean(n_items))))))
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    win = max(1, per * int(megabatch))
    batches = []
    for s in range(0, len(idx), win):
        cur, k = [], 0
        for i in sorted(idx[s:s + win], key=lambda i: lens[i]):
            cur.append(i)
            k += n_items[i]
            if k >= batch_size:
                batches.append(cur)
                cur, k = [], 0
        if cur:
            batches.append(cur)
    rng.shuffle(batches)
    return batches


def micro_batches(rows, b, max_tokens, micro_batch):
    """One optimizer batch's rows, length-sorted, cut into forwards of <= micro_batch rows and
    <= max_tokens padded tokens."""
    return [[b[j] for j in ch] for ch in MQ.chunks([len(rows[i][0]["ids"]) for i in b], micro_batch, max_tokens)]


def mq_loss(model, rows, targets, weights, mb, pad_id, device, scale, amp=True):
    """soft_ce over the blocks of rows[mb]: target = targets[item] (its letter order), weight =
    weights[item], normaliser `scale`. Returns (loss, padded tokens of the forward)."""
    batch = MQ.collate([rows[i][0] for i in mb], pad_id, device)
    idx = [j for i in mb for j in rows[i][1]]
    K = batch["marker_mask"].shape[1]
    target = torch.zeros((len(idx), K), dtype=torch.float32)
    for r, i in enumerate(idx):
        target[r, :len(targets[i])] = torch.tensor(targets[i], dtype=torch.float32)
    w = torch.tensor([weights[i] for i in idx], dtype=torch.float32, device=device)
    with MQ._amp(device, amp):
        z = MQ.forward(model, batch)
    loss = CAL.soft_ce(z, target.to(device), batch["marker_mask"], w, scale=scale)
    return loss, int(batch["ids"].shape[0] * batch["ids"].shape[1])


def letter_logits(model, tok, items, device="cpu", batch_rows=MQ.BATCH_ROWS, batch_tokens=MQ.BATCH_TOKENS, amp=True,
                  row_tokens=MQ.ROW_TOKENS, max_len=MQ.MAX_LEN, head_max_len=MQ.HEAD_MAX_LEN, qtype=LA.TRAIN_QTYPE):
    """Marker logits per item (1-D float64 arrays in the item's letter order, its offered options): the
    input of fit_temperatures / calibration_report / train.laya.calibrate."""
    if not items:
        return []
    rows = item_rows(tok, items, row_tokens, max_len, head_max_len, qtype)
    zs = MQ.run_rows(model, [r[0] for r in rows], tok.pad_token_id, device, batch_rows, batch_tokens, amp)
    out = [None] * len(items)
    for (_lay, idx), z in zip(rows, zs):
        for i, zz in zip(idx, z):
            out[i] = np.asarray(zz, dtype=np.float64)
    return out


def rec_logits(model, tok, items, device, rec):
    """letter_logits with a recipe's budgets / Laya type."""
    return letter_logits(model, tok, items, device, amp=rec["bf16"], row_tokens=rec["row_tokens"],
                         max_len=rec["max_len"], head_max_len=rec["head_max_len"], qtype=rec["qtype"])


# --------------------------------------------------------------------------- #
# The training loop (train.laya's optimiser, schedule, calibration and run directory)
# --------------------------------------------------------------------------- #
def fit_mq(model, tok, items, rec, device, targets=None, max_steps=None):
    """The MQ loop over `items` (fold-checked by the caller): stats {steps, items, rows, padded_tokens,
    items_per_s, padded_tokens_per_s, peak_gib (CUDA), loss summary, census}."""
    TJ = TL._jev()
    rows = item_rows(tok, items, rec["row_tokens"], rec["max_len"], rec["head_max_len"], rec["qtype"])
    targets = targets if targets is not None else item_targets(items)
    weights = [float(it.weight) for it in items]
    batches = train_batches(rows, rec["batch_size"], rec["seed"], rec["megabatch"])
    groups = TL.param_groups(model, rec)
    params = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=rec["weight_decay"])
    steps = rec["epochs"] * len(batches) if max_steps is None else min(max_steps, rec["epochs"] * len(batches))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, [TL.lr_factor(steps, rec, g["lr"]) for g in groups])
    w_scale = float(np.mean(weights)) if weights else 1.0
    cuda = str(device).startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    lens = np.array([len(r[0]["ids"]) for r in rows]) if rows else np.zeros(1)
    st = {"steps": 0, "items": 0, "rows": 0, "padded_tokens": 0,
          "n_rows": len(rows), "n_batches": len(batches), "items_per_row": len(items) / max(1, len(rows)),
          "census": {"row_p50": int(np.percentile(lens, 50)), "row_max": int(lens.max()),
                     "row_budget": rec["row_tokens"]}}
    losses = []
    t0 = time.time()
    for ep in range(rec["epochs"]):
        order = batches if ep == 0 else train_batches(rows, rec["batch_size"], rec["seed"] + ep, rec["megabatch"])
        for b in order:
            if st["steps"] >= steps:
                break
            n_b = sum(len(rows[i][1]) for i in b)
            tot = 0.0
            for mb in micro_batches(rows, b, rec["max_tokens"], rec["micro_batch"]):
                loss, ptok = mq_loss(model, rows, targets, weights, mb, tok.pad_token_id, device, n_b * w_scale,
                                     rec["bf16"])
                if not torch.isfinite(loss):               # fail loudly, never step on a NaN
                    raise RuntimeError(f"non-finite training loss at step {st['steps']} ({n_b} items)")
                loss.backward()
                tot += float(loss.detach())
                st["padded_tokens"] += ptok
            torch.nn.utils.clip_grad_norm_(params, rec["grad_clip"])
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            st["steps"] += 1
            st["items"] += n_b
            st["rows"] += len(b)
            losses.append(tot)
            if st["steps"] % LOG_EVERY == 0:
                el = time.time() - t0
                _log(f"step {st['steps']}/{steps}: loss {np.mean(losses[-LOG_EVERY:]):.4f}, "
                     f"{st['items'] / el:.2f} items/s, {st['padded_tokens'] / el:.0f} padded tokens/s, "
                     f"lr {sched.get_last_lr()[0]:.2e}")
    if cuda:
        torch.cuda.synchronize(device)
    st["seconds"] = time.time() - t0
    st["items_per_s"] = st["items"] / max(st["seconds"], 1e-9)
    st["padded_tokens_per_s"] = st["padded_tokens"] / max(st["seconds"], 1e-9)
    st["peak_gib"] = torch.cuda.max_memory_allocated(device) / 2 ** 30 if cuda else None
    st["loss"] = TJ._loss_summary(losses)
    model.eval()
    return st


def train_one(tr, inner, out, init, fold, device, seed, recipe=None, ref_p=None, max_steps=None, settings=None):
    """train.laya.train_one for layout "mq": ONE seed on the role-"train" items `tr`, calibrated on
    `inner`, written to the run directory `out` (a Laya checkpoint + habitus.json with layout "mq").
    Returns (record, inner logits)."""
    TJ = TL._jev()
    rec = {**TL.resolve_recipe(recipe, RECIPE_MQ), "seed": int(seed)}
    TL._fresh(out)
    TL.seed_all(rec["seed"])
    t0 = time.time()
    model, tok, init_dir, label = TL.prepare(init, device, rec, settings)
    MQ.check_encoder(model.encoder)
    n_par = sum(p.numel() for p in model.parameters())
    _log(f"fold {fold} seed {seed}: {len(tr)} training items, {len(inner)} inner items; init {label} "
         f"({n_par / 1e6:.1f}M parameters, full fine-tune, layout mq); device {device}")
    st = fit_mq(model, tok, tr, rec, device, max_steps=max_steps)
    temps, raw, clamped, report, zl = {f: 1.0 for f in CAL.FORMS}, {f: None for f in CAL.FORMS}, {}, None, []
    if inner:
        zl = rec_logits(model, tok, inner, device, rec)
        temps, raw, clamped, report = TL.calibrate(zl, inner, ref_p)
        _log(f"fold {fold} seed {seed}: temperatures {temps} (raw {raw}, clamped {clamped}); inner nll "
             f"{report['nll']}, ece15 {report['ece15']}, tail log score {report['tail_log_score']}")
    rec_out = {
        "fold": fold, "init": label, "init_dir": init_dir, "init_sha256": LA.model_sha256(init_dir),
        "layout": MQ.LAYOUT, "layout_version": MQ.LAYOUT_VERSION, "arm": ARM,
        "recipe": rec, "seed": int(seed), "full": False, "kd": None,
        "temperatures": temps, "raw_T": raw, "clamped": clamped, "laya_buckets": TL.laya_buckets(temps),
        "items_sha256": TJ.items_sha256(tr), "inner_sha256": TJ.items_sha256(inner), "code_sha256": TJ.code_sha256(),
        "spec": SPEC_VERSION, "n_items": len(tr), "n_inner": len(inner),
        "per_ds": TL.count(tr, lambda it: it.ds), "per_form": TL.count(tr, CAL.form_of),
        "parameters": int(n_par), "census": st["census"], "loss": st["loss"], "steps": st["steps"],
        "rows": st["n_rows"], "items_per_row": st["items_per_row"], "seconds": round(time.time() - t0, 1),
        "items_per_s": st["items_per_s"], "padded_tokens_per_s": st["padded_tokens_per_s"],
        "peak_gib": st["peak_gib"], "inner_report": report, "device": str(device),
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    TL.save_run(model, init_dir, out, rec_out, rec)
    _log(f"fold {fold} seed {seed}: Laya MQ model -> {out} (sha256 {rec_out['model_sha256'][:12]})")
    del model
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rec_out, zl


def train_mq(items, inner, out_dir, *, init=LA.DEFAULT_INIT, fold, device="cpu", seeds=SEEDS, recipe=None,
             name=None, ref_p=None, max_steps=None, settings=None):
    """train.laya.train_laya for layout "mq" (same signature and return value): `items` (role "train")
    train, `inner` (role "inner") calibrate; one run directory per seed, <out_dir>/<name>-s<seed>/ (name
    default "HB-L.mq"); with >= 2 seeds and inner items the geometric-mean ensemble is calibrated too
    (<out_dir>/<name>.ensemble.json, layout "mq"). init: laya.resolve_init input (a Laya init or an L-a
    run). recipe: overrides of RECIPE_MQ, flat or as a configs/train/laya-mq.yaml mapping. Returns
    {"runs": [habitus.json records], "ensemble": record or None}."""
    TL.check_device(device, settings)
    items, inner = list(items), list(inner)
    TL.check_items(items, inner, fold)
    if not items:
        raise ValueError(f"fold {fold}: no training items")
    seeds = tuple(int(s) for s in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds {seeds}")
    TL.resolve_recipe(recipe, RECIPE_MQ)
    LA.resolve_init(init, settings)
    name = name or RUN_BASE
    dirs = [TL.writable(TL.run_dir(out_dir, name, s), settings) for s in seeds]
    ens_path = TL.writable(TL.ensemble_path(out_dir, name), settings)
    for d in dirs:
        TL._fresh(d)
    runs, zs = [], []
    for s, d in zip(seeds, dirs):
        r, z = train_one(items, inner, d, init, fold, device, s, recipe, ref_p, max_steps, settings)
        runs.append(r)
        zs.append(z)
    ens = None
    if len(seeds) > 1 and inner:
        ens = {**TL.ensemble_record(fold, dirs, runs, TL.geo_logits(zs), inner, ref_p), "layout": MQ.LAYOUT}
        TL.write_json(ens_path, ens)
    return {"runs": runs, "ensemble": ens}


__all__ = ["train_mq", "train_one", "fit_mq", "item_rows", "item_targets", "train_batches", "micro_batches",
           "mq_loss", "letter_logits", "rec_logits", "RECIPE_MQ", "MQ_KEYS", "SEEDS"]
