"""Trainer of the 'laya' decision models: full fine-tune of the vendored Laya decision model on the
label-free habit items (layout L-a here; layout MQ in habitus.train.laya_mq).

    rec = train_laya(items, inner, "runs/adapters/loho-optc_h501/hbl", fold="loho-optc_h501",
                     device="cuda:1")          # -> hbl/HB-L-s0/, HB-L-s1/, HB-L-s2/, HB-L.ensemble.json

Items (habitus.questions.habit.types.Item): role "train" items train, role "inner" items (the last
pre-deployment day of each training host) calibrate; every item must come from a training host of the
fold (check_fold_items). Each item becomes the L-a row of its own rendered prompt (laya.seq_item:
split_prompt -> the row in the item's variant order, Laya type "choice"), so Item.target is already in
marker order; a row over the budgets raises (never cut).

Loss: the weighted soft cross-entropy over the markers (train.calibrate.soft_ce), normaliser = rows of
the optimizer batch x mean weight. Soft-CE only.

Recipe (RECIPE_L): 1 epoch; AdamW, encoder lr 2.5e-5, head lr 1e-4 (every non-encoder parameter: head
layers, type_emb, scorer; act_head gets no gradient and stays as loaded), weight decay 0.01; 3 % linear
warm-up, then cosine to min_lr 1e-6; effective batch 64 (length-bucketed batches, micro-batches of <= 16
rows and <= max_tokens padded tokens, gradients accumulated); grad clip 1.0; bf16 autocast on CUDA with
fp32 weights; gradient checkpointing on the encoder (non-reentrant); seeds 0, 1, 2.

Temperatures: per readout form ("inclusion-2" = Laya bucket "choice:2", "choice-4" = "choice:3-5"),
fitted on the INNER items with train.calibrate.fit_temperatures (clamped to T_CLAMP = [0.5, 3]);
`raw_temperatures` also records the unclamped optimum and a `clamped` flag per form. Report only: every
served answer is at T = 1. A seed ensemble is the geometric mean of the seeds' distributions
(`geo_logits`); its temperatures go to <base>.ensemble.json.

Outputs: one run directory per seed, <out_dir>/<name>-s<seed>/ (name: "HB-L"; "HB-L.mq" for layout
MQ): model.safetensors (fp32 state dict),
encoder/config.json and tokenizer/ (copied from the init), rl_agent_config.json (the init's config with
max_len 2048 / head_max_len 256, temperature [1, 1, 1] and temperature_by_options = the refitted buckets)
- so the directory is itself a Laya checkpoint that LayaDecisionModel loads - and habitus.json (the run
record: fold, init, layout, recipe, seed, temperatures, raw_T, clamped, laya_buckets, items_sha256,
inner_sha256, code_sha256, spec, model_sha256, inner_report, ...). With >= 2 seeds and inner items the
ensemble record goes to <out_dir>/<name>.ensemble.json.

Label-free: items carry targets computed by code from pre-deployment clean days of the fold's training
hosts; this module opens the init checkpoint and writes its own outputs, never a card, link, score or
label file.
"""

import json
import math
import os
import random
import re
import shutil
import time

import numpy as np
import torch

from habitus.dm import laya as LA
from habitus.questions.habit.constants import SPEC_VERSION, T_CLAMP
from habitus.train import calibrate as CAL

RECIPE_L = {
    "epochs": 1, "encoder_lr": 2.5e-5, "head_lr": 1e-4, "min_lr": 1e-6, "weight_decay": 0.01,
    "schedule": "cosine", "warmup": 0.03,
    "batch_size": 64,                 # effective batch (one optimizer step)
    "micro_batch": 16,                # rows per forward (gradients accumulated)
    "max_tokens": 16384,              # padded tokens per training forward (a micro-batch is split further)
    "megabatch": 64,                  # length-bucketing window (batches)
    "grad_clip": 1.0, "grad_ckpt": True, "bf16": True, "seed": 0, "loss": "soft-ce",
    "max_len": LA.MAX_LEN, "head_max_len": LA.HEAD_MAX_LEN, "qtype": LA.TRAIN_QTYPE,
}
SEEDS = (0, 1, 2)
ARM = "HB-L"
RAW_BETA = (1 / 20.0, 10.0)           # unclamped temperature search: T in [0.1, 20]
LOG_EVERY = 50
INFER_ROWS = 64
INFER_TOKENS = 32768


def _log(msg, tag="habitus.train.laya"):
    from habitus.log import log
    log(msg, tag)


def _jev():
    """habitus.train.jev: the trainer helpers shared by both families (fold check, item hashes,
    length buckets, loss summary)."""
    from habitus.train import jev
    return jev


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def check_device(device, settings=None):
    from habitus.dm.gpu import check_device as _check
    return _check(device, LA._settings(settings))


def writable(path, settings=None):
    """`path` (absolute) unless it lies under a configured read-only input (config.check_writable)."""
    from habitus import config
    return config.check_writable(str(path), LA._settings(settings))


def check_items(items, inner, fold):
    """Training items are role "train", calibration items role "inner", all from the fold's training
    hosts (raises otherwise)."""
    bad = sorted({it.role for it in items if it.role != "train"} | {it.role for it in inner if it.role != "inner"})
    if bad:
        raise ValueError(f"fold {fold}: items of unexpected roles {bad} (train items 'train', inner items 'inner')")
    _jev().check_fold_items(list(items) + list(inner), fold)


def count(items, key):
    out = {}
    for it in items:
        k = key(it)
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(path + ".tmp", path)


# --------------------------------------------------------------------------- #
# Items -> rows
# --------------------------------------------------------------------------- #
def encode_items(tok, items, rec=RECIPE_L):
    """[{"row": laya.Row, "ids", "target": letter-order target, "w": weight}] per item (the row of the
    item's own prompt; LengthError names the item)."""
    seg = LA.SegmentTokens(tok)
    texts, parts = set(), []
    for it in items:
        p = LA.split_prompt(it.prompt_text)
        if len(p.options) != len(it.letters):
            raise LA.PromptError(f"item {it.qkey}: {len(p.options)} options for letters {it.letters}")
        parts.append(p)
        qs, os_, vs = LA.segments(p, rec["qtype"])
        texts.update((qs, vs, *os_))
    seg.warm(texts)
    out = []
    for it, p in zip(items, parts):
        try:
            row = LA.seq_parts(tok, p, qtype=rec["qtype"], max_len=rec["max_len"], head_max_len=rec["head_max_len"],
                               seg=seg)
        except LA.LengthError as e:
            raise LA.LengthError(f"item {it.qkey} ({it.variant}): {e}") from None
        out.append({"row": row, "ids": row.ids, "target": list(it.target), "w": float(it.weight)})
    return out


def length_census(rows):
    """Token-length census of encoded rows: head / row quantiles, longest option, rows with an option
    longer than Laya's silent 48-token cut (never applied here)."""
    if not rows:
        return {"n": 0}
    heads = np.array([r.head_len for r in rows])
    lens = np.array([len(r.ids) for r in rows])
    opt = [LA.option_lengths(r) for r in rows]

    def q(a):
        return {"p50": int(np.percentile(a, 50)), "p95": int(np.percentile(a, 95)), "max": int(a.max())}
    return {"n": len(rows), "head": q(heads), "row": q(lens), "option_max": int(max(max(o) for o in opt)),
            "rows_option_over_48": int(sum(any(x > LA.LAYA_OPTION_CUT for x in o) for o in opt)),
            "head_budget": LA.HEAD_MAX_LEN, "row_budget": LA.MAX_LEN}


# --------------------------------------------------------------------------- #
# Optimisation
# --------------------------------------------------------------------------- #
def param_groups(model, rec):
    """Laya's two groups: encoder.* at encoder_lr, every other trainable parameter at head_lr."""
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    return [{"params": enc, "lr": rec["encoder_lr"]}, {"params": head, "lr": rec["head_lr"]}]


def lr_factor(steps, rec, base):
    """Multiplier of `base` at optimizer step s: linear warm-up over max(1, round(warmup x steps)) steps,
    then cosine from base down to rec["min_lr"] at the last step."""
    warm = max(1, int(round(rec["warmup"] * steps)))
    floor = min(1.0, float(rec["min_lr"]) / float(base))

    def f(s):
        if s < warm:
            return (s + 1) / warm
        c = 0.5 * (1.0 + math.cos(math.pi * min(1.0, (s - warm) / max(1, steps - warm))))
        return floor + (1.0 - floor) * c
    return f


def micro_batches(idx, enc, rec):
    """Split one optimizer batch (indices into enc) into forwards of <= micro_batch rows and <=
    max_tokens padded tokens, by length."""
    idx = sorted(idx, key=lambda i: (len(enc[i]["ids"]), i))
    out, cur = [], []
    for i in idx:
        L = len(enc[i]["ids"])
        if cur and (len(cur) >= rec["micro_batch"] or (len(cur) + 1) * L > rec["max_tokens"]):
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


def batches_of(enc, rec, seed):
    """The optimizer batches of one epoch (train.jev.length_buckets over the row lengths)."""
    return _jev().length_buckets([len(e["ids"]) for e in enc], rec["batch_size"], seed, rec["megabatch"])


def fit(model, enc, batches, rec, device, pad_id, max_steps=None):
    """One pass of the recipe over `batches` (index lists into enc): returns stats."""
    groups = param_groups(model, rec)
    params = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=rec["weight_decay"])
    steps = rec["epochs"] * len(batches) if max_steps is None else min(max_steps, rec["epochs"] * len(batches))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, [lr_factor(steps, rec, g["lr"]) for g in groups])
    w_scale = float(np.mean([e["w"] for e in enc])) or 1.0
    amp = bool(rec["bf16"]) and str(device).startswith("cuda")
    cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    model.train()
    st = {"steps": 0, "items": 0, "padded_tokens": 0, "forwards": 0, "losses": [], "seconds": 0.0}
    t0 = time.time()
    for ep in range(rec["epochs"]):
        order = batches if ep == 0 else batches_of(enc, rec, rec["seed"] + ep)
        for b in order:
            if st["steps"] >= steps:
                break
            tot = 0.0
            for mb in micro_batches(b, enc, rec):
                bt = LA.collate([enc[i]["row"] for i in mb], pad_id, device)
                mask = bt["marker_mask"]
                tgt = torch.zeros(mask.shape, dtype=torch.float32, device=device)
                for r, i in enumerate(mb):
                    tgt[r, :len(enc[i]["target"])] = torch.tensor(enc[i]["target"], dtype=torch.float32)
                w = torch.tensor([enc[i]["w"] for i in mb], dtype=torch.float32, device=device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                    z = model(**bt)
                loss = CAL.soft_ce(z, tgt, mask, w, scale=len(b) * w_scale)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite training loss at step {st['steps']} (batch of {len(b)})")
                loss.backward()
                tot += float(loss.detach())
                st["padded_tokens"] += int(bt["input_ids"].shape[0] * bt["input_ids"].shape[1])
                st["forwards"] += 1
            torch.nn.utils.clip_grad_norm_(params, rec["grad_clip"])
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            st["steps"] += 1
            st["items"] += len(b)
            st["losses"].append(tot)
            if st["steps"] % LOG_EVERY == 0:
                el = time.time() - t0
                _log(f"step {st['steps']}/{steps}: loss {np.mean(st['losses'][-LOG_EVERY:]):.4f}, "
                     f"{st['items'] / el:.2f} items/s, {st['padded_tokens'] / el:.0f} padded tokens/s, "
                     f"lr enc {sched.get_last_lr()[0]:.2e} head {sched.get_last_lr()[1]:.2e}")
    if cuda:
        torch.cuda.synchronize(device)
    st["seconds"] = time.time() - t0
    model.eval()
    return st


def prepare(init, device, rec, settings=None):
    """(model on device, tokenizer, init dir, init label) with the recipe's gradient checkpointing."""
    d, label = LA.resolve_init(init, settings)
    tok = LA.load_tokenizer(d)
    model = LA.load_model(d, device)
    if rec["grad_ckpt"]:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, tok, d, label


# --------------------------------------------------------------------------- #
# Inference logits, temperatures, ensembles
# --------------------------------------------------------------------------- #
def item_logits(model, tok, items, device, rec=RECIPE_L):
    """Marker logits per item (1-D float64 arrays in the item's letter order)."""
    if not items:
        return []
    rows = [e["row"] for e in encode_items(tok, items, rec)]
    return LA.row_logits(model, rows, tok.pad_token_id, device, INFER_ROWS, INFER_TOKENS)


def geo_logits(per_seed):
    """Per item, the mean over seeds of log softmax(z_seed): logits of the seeds' geometric-mean
    distribution (pbar over seeds, T = 1). per_seed: [seed][item] -> 1-D logits."""
    if not per_seed:
        raise ValueError("geo_logits needs >= 1 seed")
    n = {len(s) for s in per_seed}
    if len(n) != 1:
        raise ValueError(f"seeds score different item counts {sorted(n)}")
    out = []
    for zs in zip(*per_seed):
        lp = [np.log(np.maximum(LA.softmax(np.asarray(z, dtype=np.float64)), 1e-300)) for z in zs]
        out.append(np.mean(lp, axis=0))
    return out


def raw_temperatures(logits, items, beta=RAW_BETA):
    """{form: unclamped NLL-optimal T} over the inner items (golden-section on beta = 1 / T within
    `beta`; None for a form without items). Report only: the clamped value is fit_temperatures'."""
    bad = sorted({it.role for it in items if it.role != "inner"})
    if bad:
        raise ValueError(f"temperatures are fitted on inner-day items only, got roles {bad} (L6)")
    out = {f: None for f in CAL.FORMS}
    for form, rows in CAL._by_form(logits, items).items():
        Z = np.stack([z for z, _ in rows])
        Tg = np.array([it.target for _, it in rows], dtype=np.float64)
        W = np.array([it.weight for _, it in rows], dtype=np.float64)
        a, b = beta
        g = (math.sqrt(5) - 1) / 2
        c, d = b - g * (b - a), a + g * (b - a)
        fc, fd = CAL._nll(Z, Tg, W, 1 / c), CAL._nll(Z, Tg, W, 1 / d)
        for _ in range(100):
            if fc <= fd:
                b, d, fd = d, c, fc
                c = b - g * (b - a)
                fc = CAL._nll(Z, Tg, W, 1 / c)
            else:
                a, c, fc = c, d, fd
                d = a + g * (b - a)
                fd = CAL._nll(Z, Tg, W, 1 / d)
        out[form] = round(float(2.0 / (a + b)), 6)
    return out


def laya_buckets(temps):
    """The refitted Laya temperature_by_options: inclusion-2 -> "choice:2", choice-4 -> "choice:3-5"."""
    return {f"{LA.TRAIN_QTYPE}:2": float(temps["inclusion-2"]), f"{LA.TRAIN_QTYPE}:3-5": float(temps["choice-4"])}


def calibrate(logits, inner, ref_p=None):
    """(temperatures, raw_T, clamped, report) on the inner items: fit_temperatures (clamped), the
    unclamped optimum per form and whether the clamp bound it, calibration_report."""
    temps = CAL.fit_temperatures(logits, inner)
    raw = raw_temperatures(logits, inner)
    lo, hi = T_CLAMP
    clamped = {f: (raw[f] is not None and not (lo <= raw[f] <= hi)) for f in CAL.FORMS}
    return temps, raw, clamped, CAL.calibration_report(logits, inner, temps, ref_p=ref_p)


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #
def _check_name(name):
    if not name or os.sep in str(name) or str(name) in (".", ".."):
        raise ValueError(f"run base name {name!r}")
    return str(name)


def run_dir(out_dir, name, seed):
    """<out_dir>/<name>-s<seed>: the run directory of one seed (out_dir: the directory holding the
    runs, e.g. runs/adapters/<fold>/hbl; name: the run base, e.g. HB-L)."""
    return os.path.join(os.path.abspath(str(out_dir)), f"{_check_name(name)}-s{int(seed)}")


def ensemble_path(out_dir, name):
    """<out_dir>/<name>.ensemble.json: the ensemble record of the seed runs <name>-s<seed>."""
    return os.path.join(os.path.abspath(str(out_dir)), f"{_check_name(name)}.ensemble.json")


def resolve_recipe(recipe, base):
    """The full recipe: `base` updated by `recipe`, which is None, a flat override mapping, or a
    configs/train/*.yaml mapping (its "recipe" entry is used). Unknown keys raise."""
    if recipe is None:
        return dict(base)
    r = dict(recipe)
    if isinstance(r.get("recipe"), dict):
        r = dict(r["recipe"])
    unknown = sorted(set(r) - set(base))
    if unknown:
        raise ValueError(f"unknown recipe keys {unknown} (known {sorted(base)})")
    return {**base, **r}


def save_run(model, init_dir, out, rec_out, rec):
    """Write the run directory `out`: model.safetensors (+ its sha256 into rec_out), encoder/config.json
    and tokenizer/ from the init, rl_agent_config.json (budgets, T = 1, refitted buckets only),
    habitus.json."""
    from safetensors.torch import save_file
    os.makedirs(os.path.join(out, "encoder"), exist_ok=True)
    os.makedirs(os.path.join(out, "tokenizer"), exist_ok=True)
    wpath = os.path.join(out, LA.WEIGHTS)
    save_file({k: v.detach().to("cpu").contiguous() for k, v in model.state_dict().items()}, wpath + ".tmp")
    os.replace(wpath + ".tmp", wpath)
    shutil.copyfile(os.path.join(init_dir, "encoder", "config.json"), os.path.join(out, "encoder", "config.json"))
    tdir = os.path.join(init_dir, "tokenizer")
    for name in sorted(os.listdir(tdir)):
        if os.path.isfile(os.path.join(tdir, name)):
            shutil.copyfile(os.path.join(tdir, name), os.path.join(out, "tokenizer", name))
    cfg = LA.laya_config(init_dir)
    cfg.update({"max_len": rec["max_len"], "head_max_len": rec["head_max_len"], "temperature": [1.0, 1.0, 1.0],
                "temperature_by_options": rec_out["laya_buckets"], "model_name": f"hbl-{rec_out['init']}",
                "fine_tuned": True, "habitus_run": os.path.basename(out),
                "gradient_checkpointing": bool(rec["grad_ckpt"]),
                "training": {"updates": rec_out["steps"], "epochs_completed": rec["epochs"],
                             "hours": round(rec_out["seconds"] / 3600.0, 3), "world_size": 1}})
    write_json(os.path.join(out, LA.LAYA_CONFIG), cfg)
    rec_out["model_sha256"] = LA.file_sha256(wpath)
    write_json(os.path.join(out, LA.RUN_CONFIG), rec_out)
    return out


def _fresh(out):
    if os.path.exists(os.path.join(out, LA.RUN_CONFIG)):
        raise FileExistsError(f"{out} already holds a trained Laya model")


def train_one(tr, inner, out, init, fold, device, seed, recipe=None, ref_p=None, settings=None):
    """Train ONE seed on the role-"train" items `tr`, calibrate on `inner`, write the run directory
    `out`. Returns (habitus.json record, inner logits)."""
    TJ = _jev()
    rec = {**resolve_recipe(recipe, RECIPE_L), "seed": int(seed)}
    _fresh(out)
    seed_all(rec["seed"])
    t0 = time.time()
    model, tok, init_dir, label = prepare(init, device, rec, settings)
    enc = encode_items(tok, tr, rec)
    batches = batches_of(enc, rec, rec["seed"])
    census = length_census([e["row"] for e in enc])
    n_par = sum(p.numel() for p in model.parameters())
    _log(f"fold {fold} seed {seed}: {len(tr)} training items ({len(batches)} steps of {rec['batch_size']}), "
         f"{len(inner)} inner items; init {label} ({n_par / 1e6:.1f}M parameters, full fine-tune); "
         f"rows {census['row']} head {census['head']}; device {device}")
    st = fit(model, enc, batches, rec, device, tok.pad_token_id)
    _log(f"fold {fold} seed {seed}: {st['steps']} steps in {st['seconds']:.0f}s = "
         f"{st['items'] / max(st['seconds'], 1e-9):.2f} items/s; loss {TJ._loss_summary(st['losses'])}")
    temps, raw, clamped, report, zl = {f: 1.0 for f in CAL.FORMS}, {f: None for f in CAL.FORMS}, {}, None, []
    if inner:
        zl = item_logits(model, tok, inner, device, rec)
        temps, raw, clamped, report = calibrate(zl, inner, ref_p)
        _log(f"fold {fold} seed {seed}: temperatures {temps} (raw {raw}, clamped {clamped}); inner nll "
             f"{report['nll']}, ece15 {report['ece15']}, tail log score {report['tail_log_score']}, "
             f"rare AUC {report['rare_auc']}")
    rec_out = {
        "fold": fold, "init": label, "init_dir": init_dir, "init_sha256": LA.model_sha256(init_dir),
        "layout": LA.LAYOUT, "arm": ARM, "recipe": rec, "seed": int(seed), "full": False, "kd": None,
        "temperatures": temps, "raw_T": raw, "clamped": clamped, "laya_buckets": laya_buckets(temps),
        "items_sha256": TJ.items_sha256(tr), "inner_sha256": TJ.items_sha256(inner), "code_sha256": TJ.code_sha256(),
        "spec": SPEC_VERSION, "n_items": len(tr), "n_inner": len(inner), "per_ds": count(tr, lambda it: it.ds),
        "per_form": count(tr, CAL.form_of), "per_kmode": count(tr, lambda it: it.k_mode),
        "per_src": count(tr, lambda it: it.src), "parameters": int(n_par), "census": census,
        "loss": TJ._loss_summary(st["losses"]), "steps": st["steps"], "seconds": round(time.time() - t0, 1),
        "items_per_s": st["items"] / max(st["seconds"], 1e-9),
        "padded_tokens_per_s": st["padded_tokens"] / max(st["seconds"], 1e-9), "inner_report": report,
        "device": str(device), "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_run(model, init_dir, out, rec_out, rec)
    _log(f"fold {fold} seed {seed}: Laya model -> {out} (sha256 {rec_out['model_sha256'][:12]})")
    del model
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rec_out, zl


def train_laya(items, inner, out_dir, *, init=LA.DEFAULT_INIT, fold, device="cpu", layout=LA.LAYOUT, seeds=SEEDS,
               recipe=None, name=None, ref_p=None, settings=None):
    """Train a Laya decision model on one fold: `items` (role "train") train, `inner` (role "inner")
    calibrate. out_dir: the directory holding the runs (runs/adapters/<fold>/hbl); name: the run base
    (default "HB-L"; "HB-L.mq" for layout mq). One run directory per seed, <out_dir>/<name>-s<seed>/;
    with >= 2 seeds and inner items the geometric-mean ensemble is calibrated too
    (<out_dir>/<name>.ensemble.json). recipe: overrides of RECIPE_L, flat or as a configs/train/laya.yaml
    mapping. layout "mq" dispatches to habitus.train.laya_mq.train_mq. Returns {"runs": [habitus.json
    records], "ensemble": record or None}."""
    if layout == "mq":
        from habitus.train.laya_mq import train_mq
        return train_mq(items, inner, out_dir, init=init, fold=fold, device=device, seeds=seeds, recipe=recipe,
                        name=name, ref_p=ref_p, settings=settings)
    if layout != LA.LAYOUT:
        raise ValueError(f"layout {layout!r}: {LA.LAYOUT!r} (one row per question form) or 'mq' (laya_mq)")
    check_device(device, settings)
    items, inner = list(items), list(inner)
    check_items(items, inner, fold)
    if not items:
        raise ValueError(f"fold {fold}: no training items")
    seeds = tuple(int(s) for s in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds {seeds}")
    resolve_recipe(recipe, RECIPE_L)
    LA.resolve_init(init, settings)
    name = name or ARM
    dirs = [writable(run_dir(out_dir, name, s), settings) for s in seeds]
    ens_path = writable(ensemble_path(out_dir, name), settings)
    for d in dirs:
        _fresh(d)
    runs, zs = [], []
    for s, d in zip(seeds, dirs):
        r, z = train_one(items, inner, d, init, fold, device, s, recipe, ref_p, settings)
        runs.append(r)
        zs.append(z)
    ens = None
    if len(seeds) > 1 and inner:
        ens = ensemble_record(fold, dirs, runs, geo_logits(zs), inner, ref_p)
        write_json(ens_path, ens)
    return {"runs": runs, "ensemble": ens}


def ensemble_record(fold, dirs, runs, z, inner, ref_p=None):
    """The <base>.ensemble.json record of seed runs (`z`: the geometric-mean logits on the inner items)."""
    temps, raw, clamped, report = calibrate(z, inner, ref_p)
    return {"fold": fold, "members": [{"dir": d, "seed": r.get("seed"), "model_sha256": r.get("model_sha256")}
                                      for d, r in zip(dirs, runs)],
            "combine": "geometric mean of the seeds' distributions (types.pbar), T = 1",
            "temperatures": temps, "raw_T": raw, "clamped": clamped, "inner_sha256": _jev().items_sha256(inner),
            "inner_report": report, "spec": SPEC_VERSION, "written": time.strftime("%Y-%m-%d %H:%M:%S")}


def ensemble_report(run_dirs, inner, fold, device="cpu", ref_p=None, write=True, settings=None):
    """The geometric-mean ensemble of trained seed runs (layout "la" or "mq", all one layout), calibrated
    and reported on the inner items with each layout's own letter logits; written to
    <dir of the runs>/<common run base>.ensemble.json when `write`. Seeds trained in separate processes
    get their record this way."""
    check_device(device, settings)
    _jev().check_fold_items(list(inner), fold)
    recs = []
    for d in run_dirs:
        rec = LA.read_run_record(os.path.abspath(str(d)))
        if not rec:
            raise FileNotFoundError(f"{d}: no {LA.RUN_CONFIG} (not a trained Laya run)")
        if rec.get("fold") != fold:
            raise ValueError(f"{d} is a run of fold {rec.get('fold')!r}, not {fold!r}")
        recs.append(rec)
    layouts = {r.get("layout", LA.LAYOUT) for r in recs}
    if len(layouts) != 1:
        raise ValueError(f"the runs mix layouts {sorted(layouts)}: one ensemble is one layout")
    zs, dirs = [], []
    for d, rec in zip(run_dirs, recs):
        if rec.get("layout", LA.LAYOUT) == "mq":
            from habitus.dm import laya_mq as MQ
            from habitus.train import laya_mq as TMQ
            dd, _label = MQ.resolve_mq(d, settings)
            model, tok = LA.load_model(dd, device), LA.load_tokenizer(dd)
            zs.append(TMQ.rec_logits(model, tok, inner, device, {**TMQ.RECIPE_MQ, **rec.get("recipe", {})}))
        else:
            dd, _label = LA.resolve_init(d, settings)
            model, tok = LA.load_model(dd, device), LA.load_tokenizer(dd)
            zs.append(item_logits(model, tok, inner, device, {**RECIPE_L, **rec.get("recipe", {})}))
        dirs.append(dd)
        del model
    ens = ensemble_record(fold, dirs, recs, geo_logits(zs), inner, ref_p)
    if layouts == {"mq"}:
        ens["layout"] = "mq"
    if write:
        base = re.sub(r"-s?$", "", os.path.commonprefix([os.path.basename(d) for d in dirs])) or "hbl"
        write_json(writable(os.path.join(os.path.dirname(dirs[0]), f"{base}.ensemble.json"), settings), ens)
    return ens


def infer_rate(model, tok, items, device, rec=RECIPE_L):
    """Inference throughput of the L-a serve path on `items`: both option orders of each item's parsed
    prompt, the serving batching. Returns {"questions", "rows", "seconds", "questions_per_s", "rows_per_s"}."""
    rows = []
    seg = LA.SegmentTokens(tok)
    for it in items:
        p = LA.split_prompt(it.prompt_text)
        for opts in (p.options, tuple(reversed(p.options))):
            rows.append(LA.seq_parts(tok, LA.Parts(p.state, p.question, opts), qtype=rec["qtype"],
                                     max_len=rec["max_len"], head_max_len=rec["head_max_len"], seg=seg))
    cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    LA.row_logits(model, rows[:min(len(rows), INFER_ROWS)], tok.pad_token_id, device, INFER_ROWS, INFER_TOKENS)
    if cuda:
        torch.cuda.synchronize(device)
    t0 = time.time()
    LA.row_logits(model, rows, tok.pad_token_id, device, INFER_ROWS, INFER_TOKENS)
    if cuda:
        torch.cuda.synchronize(device)
    dt = max(time.time() - t0, 1e-9)
    return {"questions": len(items), "rows": len(rows), "seconds": round(dt, 3), "questions_per_s": len(items) / dt,
            "rows_per_s": len(rows) / dt}


__all__ = ["train_laya", "train_one", "ensemble_report", "ensemble_record", "encode_items", "length_census",
           "param_groups", "lr_factor", "micro_batches", "fit", "prepare", "item_logits", "geo_logits",
           "raw_temperatures", "laya_buckets", "calibrate", "save_run", "run_dir", "ensemble_path", "resolve_recipe",
           "infer_rate",
           "RECIPE_L", "SEEDS"]
