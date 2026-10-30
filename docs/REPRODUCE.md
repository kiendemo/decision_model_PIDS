# Reproducing the HABITUS numbers

Everything below runs on a CPU from the result bundle `release/habitus-cache-v1/` (point
`paths.data` or `HABITUS_DATA` elsewhere if it lives on another disk). No GPU and no model weights
are needed: the main decision model's answers are replayed from its answer cache, keyed by the sha1
of the full rendered prompt.

**Where to get the bundle.** It is not published yet: no download URL or DOI exists. It lives on the
build server only, until a check that the dataset terms allow redistributing the derived evidence
cards (its PUBLISH.md) is done; this section will name the location once it is published. Without
the bundle an outside user can run:

| what | needs |
|---|---|
| `make test` | this repository (every test that needs the bundle, a tokenizer, model weights or catalogs skips) |
| `habitus gates --only G5,G8` | this repository |
| `habitus gates --only G10` | this repository; its `release SHA256SUMS` check fails without the bundle, and its REPRODUCED-row check needs the data gates to have passed |
| every `habitus reproduce` level, gates G1-G4, G6, G9, G11 | the bundle |

## Assets outside the bundle

The bundle holds no model weights and no catalogs. None of them is needed for `habitus reproduce`;
they are needed for:

| asset | needed by | where to get it | config key |
|---|---|---|---|
| role-table catalogs (Debian Contents, xcyclopedia, GTFOBins, LOLBAS) | HB0 (`habitus decide hb0`, gate G11), the habit questions, the tests marked `local_assets` | `scripts/fetch_catalogs.sh models/catalogs2` (pinned sha256s / commits; the built role table must match `vocab.ROLE_TABLE_SHA256`) | `paths.catalogs` (default `models/catalogs2`) |
| Laya checkpoint convaiinnovations/laya @ 1c5edc17a7acd8701df6fc341c0d179f1c62c982 | the laya backends, the tests marked `local_assets` | `hf download convaiinnovations/laya --revision 1c5edc17a7acd8701df6fc341c0d179f1c62c982 --local-dir models/laya_hub` | `paths.laya_hub` (default `models/laya_hub`) |
| Qwen/Qwen3.5-4B @ 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, Qwen/Qwen3.5-0.8B @ 2fc06364715b967f1860aea9cf38778875588b17 | GPU forwards only (`decide --gpu`, `dm smoke`, `train jev`) | `HF_HOME=models/hf hf download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (same for 0.8B) | `paths.models` (an HF_HOME layout, default `models/hf`) |

The downloads need network access: run them with `HF_HUB_OFFLINE=0`; every habitus command stays
offline. `scripts/fetch_catalogs.sh` writes only inside this repository and never under a read-only
input; a catalog directory elsewhere is verified, never updated. Under `HABITUS_REQUIRE_ASSETS=1`
(G6) a missing bundle asset fails a test; the `local_assets` tests (Laya weights, catalogs) fail only
under `HABITUS_REQUIRE_LOCAL_ASSETS=1`, which G6 sets on the build server (`paths.pidsmaker_src`
configured) and leaves off elsewhere, where they skip.

## Environment

The published numbers were produced with Python 3.13.9 and the pins of `requirements-lock.txt`.
For a replay, these pins decide every answer-cache key and must not change:

| package / file | pin | why |
|---|---|---|
| transformers | 5.14.1 | renders the chat template (`apply_chat_template(..., enable_thinking=False)`) |
| tokenizers | 0.22.2 | the tokenizer behind it |
| jinja2 / MarkupSafe | 3.1.6 / 3.0.2 | the template engine of the chat template |
| Qwen3.5-4B tokenizer | revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a | `tokenizer/Qwen3.5-4B@851bf6e8/` in the bundle: exactly chat_template.jinja, config.json, merges.txt, tokenizer.json, tokenizer_config.json, vocab.json |
| Laya tokenizer (laya CPU tests only) | the typed-decisions checkpoint's ModernBERT tokenizer | `tokenizer/laya-typed-decisions/` in the bundle: tokenizer.json, tokenizer_config.json (copied when `paths.laya_hub` is set; no replayed number uses it) |

The `[replay]` extra (`pip install -e .[replay]`) has no torch: AutoTokenizer and the chat template
work without it. Only an editable install or a checkout on `PYTHONPATH=src` is supported (the package
reads `configs/` from the source tree). `requirements-lock.txt` pins the CUDA 13.2 torch wheel
(`torch==2.12.0+cu132`): install it with `--extra-index-url https://download.pytorch.org/whl/cu132`. Set `HF_HUB_OFFLINE=1` (the default in `configs/default.yaml`: nothing is ever
downloaded). Cap threads on a shared machine (`threads` in the config, or OMP_NUM_THREADS).

## Commands, outputs and runtimes

| command | needs | output | gate | wall time (measured) |
|---|---|---|---|---|
| `habitus reproduce --level tables` | bundle: scores, nodeindex, ground_truth | `runs/reproduce/tables/` | G1 | 34 s |
| `habitus reproduce --level replay` | + cards (with links), precedent_cache, answers, tokenizer | `runs/reproduce/replay/` | G4 | 623 s (5 hosts in parallel) |
| `habitus reproduce --level operating` | + baselines (the forms replay reads the cards too) | `runs/reproduce/operating/` | G9 | 263 s |
| `habitus reproduce --level all` | all of it | all three | G1, G4, G9 | about 15 min |
| `habitus gates` (G7 skipped) | the bundle + this repository; G11 also the role-table catalogs; G10's salvage check and the `local_assets` tests of G6 only on the build server | `runs/gates/report.{json,md}` | G1-G6, G8-G11 | about 25 min (G4 600 s, G6 290 s, G9 268 s, G11 128 s) |

The wall times were measured on the build server: an Intel Xeon Platinum 8180 (112 hardware threads),
each child process capped at 8 threads (OMP / MKL), at most 5 child processes at once (`--jobs 5`).
G7 needs a GPU and is not in these times.

Every comparison prints one PASS / FAIL line; the exit status is non-zero on any mismatch. A gate
whose optional inputs are absent (G11 without the catalogs) reports SKIP with the reason and does not
fail the run, but a REPRODUCED row of `results/README.md` that names it then fails G10.
`habitus gates --only G1,G5` runs a subset (the report is merged with the previous one). G7, a GPU
smoke of the decision models, is optional and runs alone: `habitus gates --only G7 --gpu N`. It has
two parts: a 64-question liveness subset per backend (another batch makeup than the cached run, so the
jev bound is p99 |delta letter prob| <= 0.08 against a measured bf16 noise floor of about 0.06) and
`habitus dm smoke --dm jev-qwen3.5-4b --full-day`, which replays the whole CLEARSCOPE_E3 E2 k=all
2018-04-11 job (about 8.7k prompts, the first E2 file written for that host, forwarded in one uncached
call) with its original batch makeup and must match the cache to p99 <= 1e-3. Last run 2026-10-30 on
GPU 0: 4 PASS, 0 FAIL; the full-day replay matched the cache bit for bit (10,074 forms, max abs delta
0.0) in 314 s; the whole G7 run took 439 s.

Before the bundle exists (server only): `habitus --config configs/prerelease.yaml gates --only
G2,G3,G5,G6,G8` reads the original research caches in place, read only, through
`paths.pidsmaker_artifacts` / `paths.upstream_gt` of `configs/local.yaml` (cards, the UNPRUNED answer
jsonl, the shipped score files, the precedent cache, the GT CSVs; see the header of
`configs/prerelease.yaml`). G2 then checks the keys against the unpruned answers, which is what
`--answers unpruned` (G2u) checks after the release is built. Without the bundle and without this
view, G1/G2/G3/G4/G6/G9 fail at once on a single `<gate> inputs` check that names the missing
`paths` keys (no child process is started); `runs/gates/report.json` records the config each gate
ran under.

What is compared:

- **tables (G1)**: `ablation_e1_e2.csv` byte for byte with `results/habitus/` (CRLF line endings,
  csv.DictWriter quoting, floats as round(x, 6), '' for None, host order), except the declared `note`
  column; `ablation_e1_e2.md` except its "Generated <time>" stamp; `main_config.csv`;
  `summary_means.json`; the 251 rows, the 5 main rows (0.405488 / 0.595552 / 0.112087 / 0.392856 /
  0.405215) and their mean 0.3822396. The same bytes from the cards as from the node index.
- **replay (G4)**: each host is a GUARDED child `habitus decide routed --from-inventory --cache-only`
  (label guard, write audit limited to `runs/`, open log). Every shipped score file is recomputed and
  compared per level, over the keys present in the shipped file: head and fallback equal; E2 records
  (qv, qt, qa, hard, cov_min, base, composite) and E1 records (exactly qv, qt, cov_min) with max
  |delta| == 0.0; every shipped meta key equal except `seconds`, `written` and `e2.spec`. A cache
  miss fails the run. Then the tables over the replay root must give the same CSV bytes.
- **operating (G9)**: budget_rows, per_attack_recall, sensitivity_composition, sensitivity_forms,
  form_gaps and bootstrap_mcc: integers exact, floats written with repr() and |delta| <= 1e-12; the
  forms replay (a guarded child) with recon_max_abs 0.0 and 0 cache misses on all 5 hosts; the
  bootstrap (B = 2000, seeds 101..105) draws do not depend on the system list.
- **HB0 (G11)**: each host is a GUARDED child `habitus decide hb0` over the release E2 score files
  (k2 everywhere, allc-pre except on CADETS_E3), then `habitus eval habit --levels HB0`; every column
  of `results/extension_habit/hb0_rows.csv` must match: TP / FP / TN / FN and the tie block exactly,
  the floats after rounding to the frozen column's decimals (MCC 6 dp, threshold 4 dp, the rest 3 dp).
  HB0 needs the role-table catalogs (above); without them `decide hb0` stops with `RoleTableError` and
  G11 reports SKIP.

## Exactness rules

The state compiler, the question texts, the system turns, the serialization and the control-token
neutraliser are byte-identical to the original code (G2 / G3 check them against golden hashes of the
original): any change to them changes every cache key and turns the replay into misses. The answer
readout takes the letter probabilities softmax(z[:n]) at T = 1 in float64 and the CACHED 6-dp
coverage; the question family combines the forms in their fixed order. E1 states are deduplicated by
the hash of the serialized state, E2 states by the join key of their routed states.

## GPU re-inference is not bit-exact

Forwarding the decision model again with another batch makeup (`habitus decide routed --gpu N` on
a partly cached day, `habitus dm smoke`) gives answers within bf16 noise of the cache, not the same
bits: batch makeup, kernel shapes and prefix-sharing chunk boundaries change the floating-point
evaluation. Measured on an RTX A5000: letter logits |delta| p99 0.20-0.25, letter probabilities
|delta| p99 about 0.06 (0.042-0.062 over the measured pairs), |delta qv / qt / qa| p99 0.029-0.035,
argmax agreement 0.965-0.99. The shipped answer cache is the canonical record. The cached answers came
from the flash-linear-attention path of Qwen3.5 (fla / fla-core 0.5.2, installed) with causal_conv1d
NOT installed (its convolution then runs on the torch fallback); pin the same for any GPU run.
A GPU device is refused when it is listed in `forbidden_gpus` (configs/local.yaml) or cannot be
resolved to a physical index (CUDA_DEVICE_ORDER=PCI_BUS_ID).

## Rebuilding the bundle (server only)

`python scripts/make_release.py --hosts CADETS_E3,THEIA_E3,CLEARSCOPE_E3,optc_h501,ATLASV2_EDR` is the
only writer of `release/habitus-cache-v1/`. It reads the source roots from the gitignored
`configs/local.yaml`, copies file by file from an explicit 5-host allow-list (never a stats.json or a
lock file), builds the node index, prunes the answer caches to the keys of every shipped score file
(a guarded CPU child per host, after G3 passed), gzips them deterministically, and writes
MANIFEST.json (source paths relative to the research tree), SHA256SUMS, README.md and PUBLISH.md.
Rebuilding the cards themselves from the raw dumps is described in [DATA.md](DATA.md).
