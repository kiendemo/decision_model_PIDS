# HABITUS

HABITUS is a **decision-model framework** for label-free host provenance intrusion detection. It
compiles what a host normally does into a bounded host-reference state, asks a decision model
closed, TYPED questions about each node signature (at most 4 options), and turns the answers into
node scores with a fixed, a-priori decision layer. The decision model is pluggable:

- **`jev`**: a decoder (causal language model) read out by the logits of the option-letter tokens at the answer position
  (one forward pass per presented form, T = 1; nothing is generated and no free text is parsed). The
  main system uses the stock, instruction-tuned Qwen3.5-4B at revision 851bf6e8.
- **`laya`**: an encoder (ModernBERT-large, initialised from Laya typed-decisions) with one `[MASK]`
  scorer per option, in layout L-a (one row per question form) or MQ (the state encoded once).

No label is used before evaluation: the reference is built from the host's own attack-free days, and
`habitus.eval` is the only package that may open ground truth (a label guard enforces it).

Read this before the numbers:

- Every MCC below is at the **oracle best-MCC threshold** (chosen on the test labels, for every
  system alike): an upper bound, not an operating point.
- All 5 hosts are **design-informed development hosts**: the E2 routing was chosen after a
  label-reading diagnosis on THEIA_E3 ([docs/DESIGN.md](docs/DESIGN.md)). There is no held-out host.
- Further limitations are listed under [Limitations](#limitations).

## Pipeline

```
 DARPA TC E3 / OpTC / ATLASv2 Postgres dumps        (downloaded by you: docs/DATA.md)
     |  habitus data graphs                          upstream PIDSMaker 2289cd9, as a subprocess
     v
 PIDSMaker provenance graphs                         15-minute windows, per day
     |  habitus data resolver + habitus data cards   CPU, label-free
     v
 evidence cards + link files                         one bounded summary per node signature and day
     |  habitus state build                          habitus.state (byte-identical to the original)
     v
 compiled host-reference state per head signature
     entity | hard bits | reference (k clean days) | deviation | evidence | precedent
     |  habitus questions render                     habitus.questions
     v
 typed questions, 2 forms each                       routed QV (verdict), QT (accuse), QA (acquit)
     |  DecisionModel.decide()                       habitus.dm: configs/dm/<name>.yaml, chosen with --dm
     v      jev  = decoder letter logits             laya = encoder, one [MASK] per option
 T = 1 option distributions                          answer cache keyed by sha1(prompt): CPU replay is exact
     |  habitus decide routed                        habitus.decide
     v      composite = max(qv, qt) * (1 - 0.5 qa)   bounded acquittal; the hard tier is never acquitted
 score files per (host, level, k, day)  + clean-day nulls  ->  label-free FP-budget bars
     |  habitus eval / reproduce / gates             habitus.eval: the ONLY label reader
     v
 best-MCC node confusion, ablations, operating points, paired bootstrap
```

| stage | package | what it does |
|---|---|---|
| data | `habitus.data` | PIDSMaker graphs -> evidence cards and links -> node index ([docs/DATA.md](docs/DATA.md)) |
| state | `habitus.state` | compiled host-reference states: entity, hard bits, reference, deviation, evidence, precedent |
| questions | `habitus.questions` | typed questions: routed QV / QT / QA (main) and the habit questions (HABITUS+H extension) |
| dm | `habitus.dm` | decision models `jev` (letter logits, optional fp32 head) and `laya` (L-a, MQ) ([docs/DECISION_MODELS.md](docs/DECISION_MODELS.md)) |
| decide | `habitus.decide` | answers -> score files: E2 composite with bounded acquittal, rarity fallback outside the head, clean-day nulls |
| eval | `habitus.eval` | the only label reader: best-MCC confusion, tables, operating points, bootstrap |

The fixed rules of the pipeline (which signatures are asked at all, the rarity fallback
-1 + 0.5 / (1 + n_s) for the others, the hard-tier exemption, and in the habit channel the HB0
lineage split) are set a priori: none is learned or label-tuned, and none is a decision model.
Every command except `habitus eval`, `habitus reproduce` and `habitus gates` runs under the label
guard; `reproduce` and `gates` run each inference and state step as a guarded child process.

## Choosing the decision model

A decision model is a YAML file in `configs/dm/`, selected by name with `--dm` (an arm name such as
`HB` or `HB-L.mq` works too):

| config | backend | layout | question set it is used for | weights |
|---|---|---|---|---|
| `jev-qwen3.5-4b` | jev | letter | routed (MAIN) | stock Qwen3.5-4B |
| `jev-qwen3.5-4b-habit` | jev | letter | habit (HB-Z) | stock Qwen3.5-4B |
| `jev-qwen3.5-4b-lora` | jev | letter | habit (HB) | Qwen3.5-4B + the fold's LoRA |
| `jev-qwen3.5-4b-head` | jev | head | habit (HB.head) | Qwen3.5-4B + LoRA + fp32 letter head |
| `jev-qwen3.5-0.8b-lora` | jev | letter | habit (HB.08b) | Qwen3.5-0.8B + the fold's LoRA |
| `laya-typed-zs` | laya | la | habit (HB-L.zs) | Laya typed-decisions, zero-shot |
| `laya-typed-s0` | laya | la | habit (HB-L.s0) | one fine-tuned seed |
| `laya-typed-ens3` | laya | la | habit (HB-L) | 3 fine-tuned seeds, geometric mean |
| `laya-typed-mq` | laya | mq | habit (HB-L.mq) | one fine-tuned seed, encode-once layout |

```bash
habitus dm info --dm jev-qwen3.5-4b                     # resolved config, cache tag, revision, cache status
habitus decide routed --ds CADETS_E3 --level E2 --ks 2 --dm jev-qwen3.5-4b               # cache-only (default)
habitus decide routed --ds CLEARSCOPE_E3 --level E2 --ks all --dm laya-typed-zs --gpu 1 --root runs/scores-laya
habitus decide habit  --ds optc_h501 --kplans 2,allc-pre --arm HB-L.mq --fold loho-optc_h501 --gpu 1   # after (c)
```

Every config is cache-only by default: answers come from the answer cache, and a miss raises
`CacheMiss` before any model code is imported. `--gpu N` forwards the misses (a physical GPU listed
in `forbidden_gpus` is refused). The shipped configs mirror code constants (tests assert YAML ==
code); put site paths in the gitignored `configs/local.yaml` (template `configs/local.example.yaml`),
and use `habitus.dm.get_dm(mapping)` from Python for experiments. A new backend implements the
`DecisionModel` contract (`spec`, `refused`, `decide`, `close`) of
[docs/DECISION_MODELS.md](docs/DECISION_MODELS.md) and is registered in `habitus.dm.BACKEND_PATHS`.

## Quickstart

Python 3.13 with the pins of [requirements-lock.txt](requirements-lock.txt). Only an editable
install or a checkout on `PYTHONPATH=src` is supported (the package reads `configs/` from the source
tree); `make <target>` runs the CLI from `src/` with the threads capped.

### (a) Reproduce the numbers on a CPU from the release bundle

```bash
pip install -e .[replay]               # numpy, scipy, scikit-learn, PyYAML, transformers, jinja2; no torch
# the bundle goes to release/habitus-cache-v1/ (or set HABITUS_DATA / paths.data)
habitus reproduce --level tables       # tables from the shipped score files                 34 s
habitus reproduce --level replay       # every score file recomputed from the answer cache   10 min 23 s
habitus reproduce --level operating    # budget, sensitivity, forms, bootstrap               4 min 23 s
habitus gates --skip G7                # all CPU gates (G11 also needs the role-table catalogs)  about 25 min
```

Each level prints one PASS / FAIL line per number and exits non-zero on any mismatch. The wall times
were measured on an Intel Xeon Platinum 8180, with 8 threads per child process and 5 children.

**The bundle is not published yet** (431 MB: answer caches, score files, cards, node index,
precedent cache, tokenizer, ground truth, baseline vectors). Publishing it waits on a check that the
dataset terms allow redistributing the derived evidence cards, which hold paths, command lines and
addresses from the DARPA / ATLASv2 logs (see NOTICE). No download URL or DOI exists; this README and
[docs/REPRODUCE.md](docs/REPRODUCE.md) will name it once it is published. Without the bundle you can
run `make test` (on a fresh clone: 499 passed, 111 skipped) and `habitus gates --only G5,G8`. Model
weights and the role-table catalogs are never in the bundle; where to get them is in
[docs/REPRODUCE.md](docs/REPRODUCE.md).

### (b) Full pipeline from the raw dumps

You download the DARPA TC E3, OpTC and ATLASv2 Postgres dumps yourself and restore them with upstream
PIDSMaker 2289cd9 + [docs/upstream.patch](docs/upstream.patch); the prerequisites, download ids,
sizes and checks are in [docs/DATA.md](docs/DATA.md). Then, per host:

```bash
DS=CADETS_E3                           # also THEIA_E3, CLEARSCOPE_E3, optc_h501, ATLASV2_EDR
habitus data graphs   --ds $DS [--db-host localhost]   # runs upstream PIDSMaker (paths.pidsmaker_python)
habitus data resolver --ds $DS                         # DB node tables -> runs/defense/semantics/$DS.pkl
habitus data cards    --ds $DS                         # -> runs/cards/$DS/<day>.json.gz + links/
habitus data manifest --ds $DS                         # usable clean days must equal configs/datasets.yaml
habitus data verify   --ds $DS --cards runs/cards      # card / link sha256 vs the bundle's MANIFEST.json
habitus data index    --ds $DS --cards runs/cards      # -> runs/nodeindex/$DS/
```

Point a config overlay at your cards (`paths.cards: runs/cards`, `paths.nodeindex: runs/nodeindex`,
and, when the bundle is absent, `paths.ground_truth` at the PIDSMaker `Ground_Truth/orthrus`
directory), then score and evaluate:

```bash
habitus --config my.yaml state build --ds $DS --level E2 --k 2 --day 2018-04-06 --text   # inspect one day's states
habitus --config my.yaml decide routed --ds $DS --level E2 --ks all,2,0 --nulls --dm jev-qwen3.5-4b --gpu 0
habitus --config my.yaml eval tables --root runs/scores
habitus --config my.yaml eval confusion --ds $DS --level E2 --ks 2 --root runs/scores
```

Cards identical to the shipped ones hit the shipped answer cache; anything else is forwarded on the
GPU, which is not bit-exact against the cache (bf16 noise; see the limitations).

### (c) Training the habit decision models

Training is label-free: the items are computed by code from the pre-deployment clean days of the
fold's training hosts (folds `loho-<host>` for the 4 development hosts, and `all`; CLEARSCOPE_E3 is
train-only). It needs the cards, the role-table catalogs (`scripts/fetch_catalogs.sh
models/catalogs2`), the backbone weights ([docs/REPRODUCE.md](docs/REPRODUCE.md)) and a GPU. The
recipes are fixed a priori in `configs/train/` ([docs/HABIT_CHANNEL.md](docs/HABIT_CHANNEL.md),
section 10).

```bash
F=loho-optc_h501
habitus train items --fold $F                                                  # CPU -> runs/items/
habitus train jev  --fold $F --variant lora --gpu 0 --n 60000                  # HB      -> runs/adapters/$F/HB
habitus train jev  --fold $F --variant head --gpu 0 --n 60000                  # HB.head -> runs/adapters/$F/HB.head
habitus train jev  --fold $F --variant lora --backbone Qwen/Qwen3.5-0.8B --gpu 0 --n 60000   # HB.08b
habitus train laya --fold $F --layout la --gpu 1 --n 60000     # seeds 0,1,2 -> hbl/HB-L-s{0,1,2} + HB-L.ensemble.json
habitus train laya --fold $F --layout mq --gpu 1 --n 60000     # seed 0 -> hbl/HB-L.mq-s0
habitus decide habit --ds optc_h501 --kplans 2,allc-pre --arm HB --fold $F --gpu 0
habitus eval habit   --ds optc_h501 --levels HB --kplans 2,allc-pre
```

`laya-typed-s0` reads seed 0 of the L-a run and `laya-typed-ens3` all three. `--n 60000` is the floor
of the item budget; the exact size per fold of the retraining is not fixed yet. A trained arm reads
its run record and refuses a run trained for another fold.

## Main results

HABITUS main configuration: E2 composite, reference k = all pre-deployment clean days (k = 2 on
CADETS_E3, whose k = all reference would include clean days after its first test day). Node-level
confusion at the oracle best-MCC threshold. The best PIDSMaker baseline of a host is the run with the
highest own-node-set MCC at PIDSMaker's own `best_adp` epoch, over the 104 evaluated rows of 9
systems (`results/baselines/pidsmaker_rows.csv`); it is shown on the HABITUS node universe, the one
the paired bootstrap compares on.

| host | system | TP | FP | TN | FN | precision | MCC |
|---|---|---:|---:|---:|---:|---:|---:|
| CADETS_E3 | **HABITUS** (E2, k2) | 15 | 4 | 281,513 | 57 | 0.789 | **0.405** |
| | best baseline: Velox [tuned*], epoch 9 | 12 | 1 | 281,516 | 60 | 0.923 | 0.392 |
| THEIA_E3 | **HABITUS** (E2, k all) | 60 | 26 | 700,758 | 58 | 0.698 | **0.596** |
| | best baseline: Velox [default], epoch 0 | 17 | 16 | 700,768 | 101 | 0.515 | 0.272 |
| CLEARSCOPE_E3 | **HABITUS** (E2, k all) | 39 | 2,831 | 108,531 | 2 | 0.014 | 0.112 |
| | best baseline: ThreaTrace [default] (flag D), epoch 7 | 32 | 1,501 | 109,861 | 9 | 0.021 | **0.126** |
| optc_h501 | **HABITUS** (E2, k all) | 714 | 3,684 | 1,484,765 | 35 | 0.162 | **0.393** |
| | best baseline: Velox [tuned*], epoch 5 | 162 | 961 | 1,487,488 | 587 | 0.144 | 0.176 |
| ATLASV2_EDR | **HABITUS** (E2, k all) | 24 | 49 | 152,666 | 24 | 0.329 | **0.405** |
| | best baseline: Velox [default], epoch 1 | 22 | 569 | 152,146 | 26 | 0.037 | 0.130 |
| **mean** | **HABITUS** | | | | | | **0.382** |
| | best baseline | | | | | | 0.219 |

Paired stratified entity bootstrap (B = 2000), HABITUS minus the best baseline: mean +0.163
[+0.049, +0.273], p(delta <= 0) = 0.0025. Per host: CADETS_E3 +0.013 [-0.179, +0.243], THEIA_E3
+0.323 [-0.212, +0.665], CLEARSCOPE_E3 -0.014 [-0.051, +0.071], optc_h501 +0.217 [+0.023, +0.300],
ATLASV2_EDR +0.275 [+0.116, +0.516]; the interval excludes 0 on optc_h501 and ATLASV2_EDR only.

Notes: on its own node set (which lacks the 4 cardless ground-truth nodes of CADETS_E3) Velox scores
0.404 on CADETS_E3. `best_adp` is PIDSMaker's own epoch selection, made on the test labels; `tuned*`
marks hyperparameters of unclear provenance (possibly chosen on test data); flag D marks a degenerate
port score (the best unflagged CLEARSCOPE_E3 run is NodLink [default], 0.125). Every number, its
file, the command that recomputes it and its status: [results/README.md](results/README.md).

## Status of the decision models

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

REPORTED = a number this repository ships and recomputes on a CPU; SUPPORTED, NOT EVALUATED = the
code runs this combination, but no number exists; REPORTED (FROZEN CSV) = a record of a run whose
inputs are gone. The block is generated from `habitus.dm.status` (`make status`).

## Gate results

Last full run on 2026-10-30: `habitus gates --skip G7` on the release bundle (default config, CPU,
8 threads per child, at most 5 children). Every CPU gate passes, in 25 min wall. The report is
written to `runs/gates/report.{json,md}`.

| gate | what it proves | status | evidence of this run |
|---|---|---|---|
| G1 | eval over the shipped score files reproduces the tables | PASS, 20/20 checks, 66 s | `ablation_e1_e2.csv` byte-identical (43,288 B, 251 rows), `main_config.csv` byte-identical (1,052 B), `summary_means.json` equal, `ablation_e1_e2.md` 144 lines equal (time stamp exempt); the 5 main MCCs 0.405488 / 0.595552 / 0.112087 / 0.392856 / 0.405215 and the mean 0.3822396 exact; node-index source == cards source, byte for byte |
| G2 | the rendered questions hit the answer cache; the frozen texts are unchanged | PASS, 27/27, 54 s | 12 sampled host-days (test, LODO null, k0 and E1 L4 days; 265,450 keys): 0 misses against the pruned release answers; sanitizer live == snapshot on all 12; the 10 frozen text constants equal the original sha256 |
| G2u | the same against the unpruned original answer files (server only) | PASS, 24/24, 49 s | 0 misses on all 12 samples (run at 02:51 the same day) |
| G3 | the state compiler is byte-identical to the original | PASS, 38/38, 57 s | 12 golden samples (51,822 head signatures): head, per-signature sha1, digest and state count equal; E1 snapshot test 2 passed |
| G4 | every shipped score file is recomputed exactly from the answer cache | PASS, 98/98, 606 s | job rules == shipped inventory == golden on all 5 hosts; 5 guarded `decide routed --from-inventory --cache-only` children: 338 / 338 files (110 / 74 / 43 / 74 / 37), 72 (host, level, k) groups with max abs delta 0.0, 0 extra files, 0 cache misses; the tables over the replay root are byte-identical |
| G5 | import isolation, label isolation, host allow-list | PASS, 8/8, 25 s | import graph clean; label reads only under `habitus/eval` plus the allow-list; the 47 open logs of guarded children show no ground-truth open; the guard blocks a ground-truth open and `import habitus.eval`; out-of-scope host scan of 241 files clean |
| G6 | the full CPU test suite with every bundle asset required | PASS, 293 s | 609 passed, 1 skipped (the opt-in HB0 asset test), 15 deselected (provenance), 0 failed |
| G7 | optional GPU smoke of the decision models (GPU 0) | PASS, 4/4, 439 s | full-day replay of CLEARSCOPE_E3 E2 k=all 2018-04-11 in the original batch makeup: 5,037 questions / 10,074 forms, max abs delta 0.0 vs the answer cache (bit-exact), argmax agreement 1.0, 314 s; jev subset p99 0.0608 <= 0.08; laya-typed-zs 64 routed questions finite, 0 refused; see below |
| G8 | no server path, secret or e-mail address is shipped | PASS, 7/7, 3 s | 237 tracked + 4 release text files clean; LICENSE Apache-2.0 and NOTICE present; `configs/local.yaml` gitignored; no stats file among the cards |
| G9 | the operating tables reproduce | PASS, 24/24, 269 s | budget_rows (30 rows), per_attack_recall (120), sensitivity_composition (80), sensitivity_forms (45), form_gaps (15), bootstrap_mcc (148): integers exact, floats max abs delta 0; forms replay reconstruction error 0.0 with 0 cache misses on all 5 hosts |
| G10 | the manifests, the results table and the STATUS blocks are consistent | PASS, 10/10, 4.5 s | `results/MANIFEST.sha256` verifies 41 files and the release SHA256SUMS 559; the 19 `results/README.md` rows have their files and every REPRODUCED row names a passing gate; the 3 STATUS blocks equal `render_markdown()`; PROVENANCE lists 130 sources with sha256; config mirror tests 40 passed |
| G11 | the HB0 decision layer reproduces | PASS, 14/14, 126 s | guarded `decide hb0` on the 5 hosts, then `eval habit --levels HB0`: all 9 rows of `hb0_rows.csv` equal in every column (e.g. CADETS_E3 k2 12 / 1 / 281,516 / 60, MCC 0.392183; THEIA_E3 allc-pre MCC 0.596117) |

G7 (GPU; physical GPUs 0 and 1 only) last ran on 2026-10-30 under its earlier rule. The jev subset
smoke failed that rule's bound, p99 |delta letter prob| <= 0.03, with 0.0608 (the same value on
re-run). Two fresh bf16 runs with a different batch makeup differ by as much (0.0603-0.0613), while
two runs with the same batch makeup are bit-identical. The laya-typed-zs smoke (64 routed questions:
finite, max sum error 2.2e-16, 0 refused), laya-mq in fp32 (question isolation max |dp| 1.41e-6,
option permutation 9.98e-7, bound 1e-5), the stock jev 0.8B decide and the GPU guard passed. The rule
was then re-pinned to p99 <= 0.08, and a full-day replay in the original batch makeup with p99 <= 1e-3
was added. Both parts were then run on GPU 0 (`habitus gates --only G7 --gpu 0`, 2026-10-30,
`runs/gates/g7_final/`): the full-day replay of CLEARSCOPE_E3 E2 k=all 2018-04-11 (10,074 forms)
reproduced the cached answers bit for bit (max abs delta 0.0, argmax agreement 1.0), the subset smokes
passed (jev p99 0.0608 <= 0.08; laya-typed-zs finite, 0 refused): 4 PASS, 0 FAIL.

## Limitations

- **Oracle threshold.** All MCCs use the threshold that maximises MCC on the test labels. The
  label-free clean-day bars miss their FP budget (realised FP per test day > B) in 12 of 15
  (host, B) cells ([results/README.md](results/README.md), tab:budget). Ties at the threshold count
  positive; with ties negative the HABITUS mean drops to 0.328 (optc_h501 to 0.209, through a
  348-node tie block).
- **Design-informed development hosts.** The E2 routing and the name-free evidence were chosen after a
  label-reading diagnosis on THEIA_E3, and every host was seen during design. No number here is
  held out.
- **No advantage over a no-LLM rule.** rule-lex over the same compiled state ties HABITUS on the mean
  and beats it on CADETS_E3, so the gain over PIDSMaker is not shown to come from the decision model
  rather than from the compiled state ([results/not_included.md](results/not_included.md)).
- **Frozen habit-channel CSVs.** The HABITUS+H arms (HB-Z, HB, HB.head, HB.08b, HB-L.s0, HB-L,
  HB-L.mq) ship as frozen CSVs only: their adapters, run directories, habit nulls and answer caches
  were on a former data disk that is no longer readable. The code is here; retrained numbers will
  differ ([results/extension_habit/FROZEN.md](results/extension_habit/FROZEN.md)). The selection
  metrics M1 / M2 used the excluded HB-C selector and cannot be recomputed. Of the habit channel only
  HB0 (no decision-model call) is recomputed (G11).
- **laya on the routed questions is not evaluated.** The backend answers them, but no number exists.
  E2 rows over the 2048-token budget are refused, never cut, and `decide routed` raises on a refusal;
  a tokenizer-only scan of the first test day of each host refused 0 of 82,005 routed questions
  (longest row 516 tokens). The trained jev variants are not evaluated on the routed questions either.
- **GPU re-inference is not bit-exact.** A bf16 re-forward with another batch makeup moves the letter
  probabilities by p99 about 0.06 and qv / qt / qa by p99 0.029-0.035 (argmax agreement 0.965-0.99,
  RTX A5000). In the original batch makeup the GPU path is bit-exact (G7 full-day replay, max abs
  delta 0.0); the shipped answer cache stays the canonical record.
- **Baselines.** The PIDSMaker rows are frozen for outside users (they need the upstream run
  artifacts); the best baseline uses PIDSMaker's label-selected `best_adp` epoch, and two of the five
  best rows use `tuned*` arguments.
- **The rebuild from the dumps is partly verified.** The card builder rebuilt all 45 card and link
  files JSON-equal from the stored graphs. That pristine PIDSMaker 2289cd9 + `docs/upstream.patch`
  gives the same graph bytes was not checked, and the CLEARSCOPE_E3 graph bytes were never compared
  with an upstream run ([docs/DATA.md](docs/DATA.md)).
- **The bundle is unpublished**, so outside users cannot run the data gates yet.

## Repository layout

```
configs/    default.yaml, datasets.yaml, pipeline.yaml, habit.yaml, prerelease.yaml, local.example.yaml,
            dm/ (9 decision-model configs), train/ (jev-lora, jev-head, laya, laya-mq recipes)
docs/       DATA, DESIGN, DECISION_MODELS, dm_laya, HABIT_CHANNEL, REPRODUCE, PROVENANCE, upstream.patch
results/    habitus/ (main system), baselines/ (PIDSMaker rows), extension_habit/ (HABITUS+H, frozen),
            protocol/, not_included.md, README.md (paper number -> file -> command -> status), MANIFEST.sha256
scripts/    make_release.py, build_graphs.py, export_baselines.py, fetch_catalogs.sh, results/, dev/
src/habitus data/ state/ questions/ dm/ decide/ train/ eval/ repro/, cli.py, config.py, guard.py, hosts.py
tests/      625 tests (15 provenance tests need the original source trees); golden files in tests/golden/
```

Generated and never tracked: `runs/` (every file HABITUS writes), `release/` (the bundle), `models/`
(weights and catalogs) and `configs/local.yaml` (site paths).

## Licence

Apache-2.0 ([LICENSE](LICENSE)); third-party attributions, the origin of the ground truth and the
change list of the vendored Laya code: [NOTICE](NOTICE). Provenance of every ported source file:
[docs/PROVENANCE.md](docs/PROVENANCE.md).
