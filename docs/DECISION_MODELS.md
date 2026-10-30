# Decision models

HABITUS is a decision-model pipeline. A DecisionModel answers a TYPED question (<= 4 options) about
a COMPILED host-reference state. It makes one forward pass per presented form and returns a T=1
distribution over the offered options. A question family then combines the forms.

The 'jev' backbone of the main system is a stock, instruction-tuned decoder LLM (Qwen3.5-4B
@851bf6e8), used as follows:

- It is given a chat-formatted prompt: the system turn is SYSTEM_E2, and the user turn is the state
  plus one closed question.
- It is read out by the logits of the letter tokens at the answer position.
- Nothing is generated and no free text is parsed.

The 'laya' family is an encoder with a [MASK] scorer per option (see `docs/dm_laya.md`).

The pipeline also contains fixed, a-priori gating and fallback rules (the head selection rule, the
rarity fallback for non-head signatures, the hard-tier acquittal exemption, and in the habit channel
the HB0 lineage-split acquittal). None of them is learned or label-tuned, and none of them is a
decision model. Rule comparators are not part of this repository (`results/not_included.md`).

## The contract (`habitus.dm.base`)

| Object | Fields | Notes |
|---|---|---|
| `Form` | `name, system, state, question, options, canon` | One presentation (one option order). `options` are in LETTER order (2..4); `canon[i]` is the canonical option index of letter `i`. `Form.user()` is the user turn. |
| `TypedQuestion` | `qkey, family, qtype, forms, group` | `forms[0]` is canonical. `group` is the prefix-share / encode-once key. |
| `Decision` | `qkey, layout, forms, letter_probs, canon_probs, cov, cov_min` | Per form, one probability tuple per member (letter and canonical order), T = 1. |
| `DecisionModel` | `family, layout, tag, spec(), refused(qs), decide(qs), close()` | `decide` answers every non-refused qkey once; two questions with one qkey and different forms raise. |
| `CacheMiss` | `keys` | Cache-only mode met uncached prompts; raised before any model code is imported. |
| `Refused` | `reasons` | Questions a backend refuses (laya: over-budget rows, never cut). |

`Form.user()` is

    "State:\n" + state + "\n\nQuestion: " + question + "\nOptions:\n"
    + "\n".join(f"{L}: {text}") + "\nAnswer with one letter: " + ", ".join(letters)

byte for byte the user turn of the routed questions (`questions.routed`) and of the habit questions
(`questions.habit`). The question families build the typed questions:

- routed: `questions.routed.to_typed(sig, qid, state, system=SYSTEM_E2)`; forms in `variants(qid)`
  order (qv: fwd, rev; qt and qa: pos, neg); `group` is `"<sig>|sig"` for qv and qt and
  `"<sig>|qa"` for qa. `questions.routed.record(decisions, level)` combines them: each value is
  `combine(qid, {form: letter_probs[form][0]})`, and `cov_min` is the minimum `Decision.cov` over
  all forms of all questions of the signature.
- habit: `questions.habit.types.to_typed(q, orders)`; one form per order (fwd = canonical, rev =
  reversed), system turn SYSTEM_H, `group` `"<view kind>|<view sha1>"`;
  `questions.habit.types.answer_of(decision, orders)` gives the habit `Answer`.

## Backends

| Key | Class | Layout | Readout |
|---|---|---|---|
| `jev` | `habitus.dm.jev.JevDecisionModel` | `letter` | Logits of the letter tokens A..D at the answer position (one forward pass, `logits_to_keep=1`). |
| `jev-head` | `habitus.dm.jev_head.JevHeadDecisionModel` | `head` | An fp32 letter head on the final hidden state at the answer position (HB.head). |
| `laya` | `habitus.dm.laya.LayaDecisionModel` | `la` | One encoder row per (question, form) with one [MASK] per option. |
| `laya-mq` | `habitus.dm.laya_mq.LayaMQDecisionModel` | `mq` | The state encoded once per group, one block per question. |

### jev letter readout (exact)

1. The prompt is `tok.apply_chat_template([system turn, user turn], tokenize=False,
   add_generation_prompt=True, enable_thinking=False)`. It must end with the empty think block
   `"<think>\n\n</think>\n\n"` (checked); the answer letter is the next token.
2. One forward pass gives `z` = the logits of A..D (always all four) and `lse` = the
   full-vocabulary logsumexp at the answer position.
3. The answer is recorded first: `{"z": z rounded to 4 dp, "lse": 4 dp, "n": offered letters,
   "cov": sum(exp(z[:n] - lse)) rounded to 6 dp}`. A non-finite record is never cached: the prompt
   is retried alone, and a second non-finite answer stops the run.
4. The decision is read back from the record: `p = softmax(z[:n])` at T = 1 in float64, then
   `tuple(float(x))`; `Decision.cov[form]` is the record's `cov`. A fresh answer and a cached one
   therefore give the same floats.

Coverage is the softmax mass the model put on the offered letters at all. It is recorded and
returned, never used to drop or reweight an answer (answers with `cov < 0.5` are only counted).

### jev head readout (HB.head)

`z = W0 h + A (h - mu) / sigma + b` over 4 slots in fp32, where `h` is the final hidden state at the
answer position. `W0` and `b` start as the lm_head rows (bias) of the letters, `A = 0`, and
`mu / sigma` are fitted once on the training items' step-0 hidden states and frozen. At step 0 the
head equals the fp32 letter readout (|dp| < 1e-5, tested); the stock letter readout reads the bf16
lm_head instead. The record's `lse` is the logsumexp over the offered slots, so `cov = 1`.

## Answer cache (`habitus.dm.cache.AnswerCache`)

- Key: sha1 of the full rendered prompt (UTF-8). Any change to a question text, the system prompt,
  the state serialization, the sanitizer, the chat template, the tokenizer snapshot, transformers or
  jinja2 changes every key, which is why those are pinned.
- Record (JSON line): `{"k", "z", "lse", "n", "cov"}`.
- Base: the shipped `<paths.answers>/<tag>/<ds>.jsonl.gz` (or `.jsonl`), opened read-only; no lock
  file is ever created next to it.
- Overlay: `<paths.runs>/answers/<tag>/<ds>.jsonl`, append-only. The first write takes a
  non-blocking flock on `<overlay>.lock` for the cache's lifetime (one writer per (tag, ds)); every
  flush appends under an flock on the file itself and repairs a partial last line.
- The FIRST record of a key wins (base before overlay, line order inside a file). Unparsable lines
  and records with a non-finite `z`, `lse` or `cov` are skipped.
- `cache_only=True` (the default of every config) raises `CacheMiss` listing every missing key and
  imports no model code; only the tokenizer (6 files, no weights) is needed to render and key the
  prompts. In an environment without torch, the tokenizer and the chat template still load.
- `JevDecisionModel.keys(qs)` returns `{qkey: (sha1 per form)}` with no cache lookup and no model;
  `scripts/make_release.py` prunes the shipped cache with it and gate G2 checks it.

### Tags (answer-cache namespaces)

| Arm | Tag |
|---|---|
| MAIN | `qwen3.5-4b` (`engine_tag(backbone)`) |
| HB-Z | `hb-qwen3.5-4b` (`habit_tag(backbone)`) |
| HB, HB.08b | `hb-<engine_tag(backbone, adapter)>-<12 hex of the adapter content hash>`, e.g. `hb-qwen3.5-4b+HB-0123456789ab` |
| HB.head | `hb.head-<engine_tag(backbone, adapter)>-<12 hex of sha256(adapter hash, head hash)>` (stock head: `hb.head-qwen3.5-4b`) |
| laya | see `docs/dm_laya.md` (`hbl-<init>-<12 hex>`, `hbl-mq-...`) |

The adapter content hash is the sha256 over (file name, file sha256) of `adapter_config.json` and the
adapter weights, so a retrained adapter in the same run directory never reuses stale answers.

## Batching, prefix sharing and exactness

- At most `batch_size` (<= 32) prompts per forward, at most 8 once a prompt is longer than 900
  tokens, length-bucketed. On CUDA OOM a batch is halved, with the retry run outside the exception
  handler.
- `prefix_share=True`: the prompts of one group are the same text up to the question. The longest
  common prefix of their token ids is forwarded once with the hybrid Qwen3.5 cache (K/V of the
  full-attention layers, conv window and fp32 recurrent state of the Gated DeltaNet layers), the
  cache is replicated per suffix, and the right-padded suffixes continue it. The model is causal, so
  this is the same function as the full forward: on a tiny random fp32 Qwen3.5 the two agree to
  2e-4 in the letter logits (tested).
- `grouping='direct'` uses one group per `TypedQuestion.group` (the routed questions: the 4 QV + QT
  prompts of a signature, its 2 QA prompts apart). `grouping='chunked'` sorts each group's prompts
  by text and cuts them into chunks of <= `batch_size` rows (<= 8 when a prompt of the group is
  longer than 900 tokens), for the habit views whose groups exceed a batch.
- GPU re-inference with another batch makeup is not bit-exact: in bf16 it moves a letter logit by a
  few ulps. Measured on an RTX A5000 (CLEARSCOPE_E3 E2 k=all 2018-04-11, 256 questions / 512 forms):
  |d letter logit| p99 0.20-0.25, |d letter prob| p99 0.042-0.062 (fresh vs fresh 0.060-0.062,
  fresh vs cache 0.061 on the 64-question smoke), |d qv / qt / qa| p99 0.029-0.035, argmax agreement
  0.965-0.99. `habitus dm smoke` therefore passes the subset smoke at p99 |d letter prob| <= 0.08
  (liveness), and `dm smoke --full-day` replays a whole cached job in its original batch makeup with
  a tight bound (p99 <= 1e-3). The shipped answer cache is the canonical record of the main numbers;
  replaying it on CPU is exact.
- `spec()['grouping']` is the decision model's grouping (`direct` | `chunked`). The score-file meta
  word (`qvqt` for E2, `state` for E1) is written by `decide.routed`.

## Registry and arms (`habitus.dm`)

`get_dm(name_or_cfg, *, ds=None, device=None, cache_only=None, settings=None, fold=None)` builds a
decision model from `configs/dm/<name>.yaml`; `get_arm(name, ...)` does it for a registered arm. The
backend is `cfg['dm']` plus a suffix from `cfg['layout']`. Every config key except the registry keys
(`dm`, `layout`, `questions`) is a constructor argument. Path values may hold `<fold>`; a relative
path under `runs/` resolves under `settings.paths.runs`. `cache_tag: auto` is the habit tag of the
backbone and the adapter. `cache_only` defaults to true and `device` to `settings.device` (cpu).

| Arm | Config | Role |
|---|---|---|
| MAIN | `jev-qwen3.5-4b` | routed QV/QT/QA |
| HB-Z | `jev-qwen3.5-4b-habit` | habit |
| HB | `jev-qwen3.5-4b-lora` | habit |
| HB.head | `jev-qwen3.5-4b-head` | habit |
| HB.08b | `jev-qwen3.5-0.8b-lora` | habit |
| HB-L.s0 | `laya-typed-s0` | habit |
| HB-L | `laya-typed-ens3` | habit |
| HB-L.mq | `laya-typed-mq` | habit |
| HB-L.zs | `laya-typed-zs` | habit |

Fold safety: an arm backed by trained runs reads each run's `habitus.json` and refuses a run trained
for another fold, a run without a record, and a fold that trains on the scored dev host
(`loho-<host>` scores `<host>`; the train-only host CLEARSCOPE_E3 is scored by `all`, host-seen).
An adapter whose run record names the `head` variant is refused by `JevDecisionModel` and served by
`JevHeadDecisionModel`.

GPU guard (`habitus.dm.gpu`): `check_device(device, settings)` resolves a logical `cuda:i` to its
physical index through `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES`, and refuses an
index listed in `settings.forbidden_gpus` (set in the gitignored `configs/local.yaml`) or one that
cannot be resolved. `cpu` is always allowed and needs no resolution. The model is loaded only on the
first cache miss of a non-cache-only model, after this check.

## Training and calibration (`habitus.train`)

- `train.jev.train_jev(items, inner, out_dir, *, backbone, revision, fold, device, recipe,
  variant='lora'|'head')` writes `<out_dir>/adapter/`, `<out_dir>/head/` (variant head) and
  `<out_dir>/habitus.json`. Items are `questions.habit.types.Item`: the full rendered chat prompt, the
  offered letters, a soft target in letter order, a weight and the role `train` | `inner`. Every
  item's dataset must be a training host of the fold.
- The recipe is fixed a priori (`configs/train/jev-lora.yaml`, `configs/train/jev-head.yaml` mirror
  the code): 1 epoch, AdamW lr 1e-4 (head 2e-4), cosine after 3 % warmup, effective batch 16 in
  micro-batches of 8, length-bucketed, grad clip 1.0, gradient checkpointing, bf16; LoRA r 16 /
  alpha 32 / dropout 0.05 on q, k, v, o, gate, up, down.
- `recipe` overrides the fixed recipe: a flat mapping, or a `configs/train/*.yaml` mapping whose
  `recipe` entry is used (as `train.laya.resolve_recipe` does); unknown keys raise. `habitus train jev`
  passes that entry and writes, by default, to the run directory its arm's config reads:
  `runs/adapters/<fold>/HB` (4B LoRA), `HB.head` (4B head), `HB.08b` (`--backbone Qwen/Qwen3.5-0.8B`).
  `habitus train laya` takes its seeds from the YAML (`laya`: 0, 1, 2; `laya-mq`: 0) unless `--seeds`
  is given.
- The loss is `train.calibrate.soft_ce`, the weighted log score over the offered letters, shared by
  every trainer (jev and laya).
- Temperatures (`train.calibrate.fit_temperatures`) are fitted per readout form on role-`inner`
  items only, clamped to [0.5, 3], and used for `calibration_report` only. Serving is at T = 1.

## Status

<!-- STATUS:BEGIN -->
| Question set | Arm | Config | Status | Evidence | Note |
|---|---|---|---|---|---|
| Routed QV/QT/QA (HABITUS main) | MAIN (jev stock Qwen3.5-4B) | `jev-qwen3.5-4b` | REPORTED | `results/habitus/main_config.csv` | Reproducible on CPU from the shipped answer cache: the main table, E1/E2 ablations, budget, sensitivity, forms and bootstrap. |
| Routed QV/QT/QA (HABITUS main) | jev 4B+LoRA | `jev-qwen3.5-4b-lora` | SUPPORTED, NOT EVALUATED | - | No published number exists. |
| Routed QV/QT/QA (HABITUS main) | jev 4B+head | `jev-qwen3.5-4b-head` | SUPPORTED, NOT EVALUATED | - | No published number exists. |
| Routed QV/QT/QA (HABITUS main) | jev 0.8B+LoRA | `jev-qwen3.5-0.8b-lora` | SUPPORTED, NOT EVALUATED | - | No published number exists. |
| Routed QV/QT/QA (HABITUS main) | laya s0 | `laya-typed-s0` | SUPPORTED, NOT EVALUATED | - | No published number exists. laya refuses over-budget rows (never cuts) and decide raises on any refusal. |
| Routed QV/QT/QA (HABITUS main) | laya ens3 | `laya-typed-ens3` | SUPPORTED, NOT EVALUATED | - | No published number exists. laya refuses over-budget rows (never cuts) and decide raises on any refusal. |
| Routed QV/QT/QA (HABITUS main) | laya mq | `laya-typed-mq` | SUPPORTED, NOT EVALUATED | - | No published number exists. laya refuses over-budget rows (never cuts) and decide raises on any refusal. |
| Routed QV/QT/QA (HABITUS main) | laya zs | `laya-typed-zs` | SUPPORTED, NOT EVALUATED | - | No published number exists. laya refuses over-budget rows (never cuts) and decide raises on any refusal. |
| Habit questions (HABITUS+H) | HB-Z | `jev-qwen3.5-4b-habit` | REPORTED (FROZEN CSV) | `results/extension_habit/decision_models_best_mcc.csv` | Stock backbone; its habit answer cache and nulls were lost with the former data disk. Retrained numbers will differ. |
| Habit questions (HABITUS+H) | HB | `jev-qwen3.5-4b-lora` | REPORTED (FROZEN CSV) | `results/extension_habit/decision_models_best_mcc.csv` | Adapters, run dirs and caches were lost with the former data disk and are being retrained. Retrained numbers will differ. |
| Habit questions (HABITUS+H) | HB.head | `jev-qwen3.5-4b-head` | REPORTED (FROZEN CSV) | `results/extension_habit/decision_models_best_mcc.csv` | Adapters, run dirs and caches were lost with the former data disk and are being retrained. Retrained numbers will differ. |
| Habit questions (HABITUS+H) | HB.08b | `jev-qwen3.5-0.8b-lora` | REPORTED (FROZEN CSV) | `results/extension_habit/decision_models_best_mcc.csv` | Adapters, run dirs and caches were lost with the former data disk and are being retrained. Retrained numbers will differ. |
| Habit questions (HABITUS+H) | HB-L.s0 | `laya-typed-s0` | REPORTED (FROZEN CSV) | `results/extension_habit/decision_models_best_mcc.csv` | Adapters, run dirs and caches were lost with the former data disk and are being retrained. Retrained numbers will differ. |
| Habit questions (HABITUS+H) | HB-L | `laya-typed-ens3` | REPORTED (FROZEN CSV) | `results/extension_habit/decision_models_best_mcc.csv` | Adapters, run dirs and caches were lost with the former data disk and are being retrained. Retrained numbers will differ. |
| Habit questions (HABITUS+H) | HB-L.mq | `laya-typed-mq` | REPORTED (FROZEN CSV) | `results/extension_habit/decision_models_best_mcc.csv` | Adapters, run dirs and caches were lost with the former data disk and are being retrained. Retrained numbers will differ. |
| Habit questions (HABITUS+H) | HB-L.zs | `laya-typed-zs` | FROZEN SELECTION METRICS ONLY | `results/extension_habit/selection_metrics_3folds.csv` | The config is supported. |
| Habit questions (HABITUS+H) | HB-L.full | - | UNSUPPORTED | `results/extension_habit/selection_metrics_3folds.csv` | Frozen selection rows only; there is no arm or config. |
| HB0 decision layer (no DM call) | HB0 | - | REPRODUCED (G11) | `results/extension_habit/hb0_rows.csv` | Lineage-split decision layer over the frozen E2 routes, on CPU via `habitus decide hb0` + `habitus eval habit --levels HB0`: gate G11 recomputes all 9 rows exactly. Needs the role-table catalogs (scripts/fetch_catalogs.sh; their table must match ROLE_TABLE_SHA256). |
<!-- STATUS:END -->
