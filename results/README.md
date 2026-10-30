# HABITUS results

Every number of the HABITUS paper that this repository ships, the file it lives in, the command that recomputes it and its status. Paths below are relative to this directory.

**Metric.** Node-level confusion (TP / FP / TN / FN, precision, recall, F1) and the Matthews correlation coefficient MCC = (TP·TN − FP·FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)) (0 when a margin is empty). Universe = every test-day node plus the ground-truth nodes without a test-day card (always FN); node score = max over the test days; signatures outside the decision head get the rarity fallback (< 0, below every decision score). A tie block at the threshold counts positive; the ties-negative MCC is reported alongside.

**Oracle threshold.** Every MCC here is at the threshold that maximises MCC on the TEST labels, for every system alike: an upper bound, not an operating point. The label-free clean-day bars miss their FP budget (realised FP per test day > B) in 12 of 15 cells (tab:budget, habitus/budget_rows.csv).

**Main configuration.** E2 composite with all pre-deployment clean days as the reference (k = all) on THEIA_E3, CLEARSCOPE_E3, optc_h501 and ATLASV2_EDR, and k2 on CADETS_E3: its k = all reference includes clean days after its first test day (04-07..04-10 vs 04-06), so that row is flagged oracle-k and never used as the main one.

**Development hosts.** All 5 hosts are design-informed development hosts: E2's routing and name-free evidence were chosen after a label-reading diagnosis on THEIA_E3 (docs/DESIGN.md). These are not held-out numbers.

## Paper number -> file -> command -> status

| Paper item | Numbers | File | Command | Status |
|---|---|---|---|---|
| tab:summary / tab:main HABITUS main | CADETS 0.405 (15/4), THEIA 0.596 (60/26), CLEARSCOPE 0.112 (39/2831), optc 0.393 (714/3684), ATLASV2 0.405 (24/49); mean 0.382 | `habitus/main_config.csv`, `habitus/summary_means.json` | `habitus reproduce --level tables` (exact replay: `--level replay`) | REPRODUCED, G1 + G4 |
| tab:summary k2 on every host | 0.405/0.557/0.086/0.419/0.405 (0.374) | `habitus/ablation_e1_e2.csv` (E2 k2 composite) | `habitus reproduce --level tables` | REPRODUCED, G1 |
| tab:summary ties negative | 0.353/0.589/0.095/0.209/0.391 (0.328) | `habitus/ablation_e1_e2.csv` (mcc_ties_neg) | `habitus reproduce --level tables` | REPRODUCED, G1 |
| tab:ablation-state | L0..L4 and E2 at k0 / k2 / kall; mean4 / mean5 | `habitus/ablation_e1_e2.csv` (E1 rows), `habitus/summary_means.json` | `habitus reproduce --level tables` (+ replay of every shipped E1 dir) | REPRODUCED, G1 / G4; CLEARSCOPE_E3 L0 / L1 / L3 at k>=1 and L0 / L1 at k0 never run |
| tab:ablation-routes | qv / qt / base / composite / 1-qa / acq_all | `habitus/ablation_e1_e2.csv` | `habitus reproduce --level tables` | REPRODUCED, G1 |
| tab:ties | GT in head 68/72, 110/118, 41/41, 729/749, 48/48; tie blocks 3 (3), 1 (1), 55 (6), 348 (348), 1 (1) | `habitus/ablation_e1_e2.csv` (gt_in_head, tie_n, tie_gt) | `habitus reproduce --level tables` | REPRODUCED, G1 |
| tab:budget | B = 0 / 1 / 20 TP:FP: CADETS 12:1/12:4/19:262 (2 null days); THEIA 23:23/42:23/67:136 (4 null days); CLEARSCOPE 0:0/0:3/0:176 (7 null days); optc 5:2/5:5/6:27 (4 null days); ATLASV2 0:17/2:23/23:49 (3 null days) | `habitus/budget_rows.csv`, `habitus/per_attack_recall.csv` | `habitus reproduce --level operating` | REPRODUCED, G9 |
| tab:sensitivity lambda / combiner | 8 rows x 5 hosts | `habitus/sensitivity_composition.csv` | `habitus reproduce --level operating` | REPRODUCED, G9 |
| tab:sensitivity forms | form1, form2, single forms (45 rows); recon_max_abs 0, cache misses 0 | `habitus/sensitivity_forms.csv`, `habitus/form_gaps.csv` | `habitus reproduce --level operating` (cards + tokenizer + precedent cache) | REPRODUCED, G9 |
| tab:uncertainty HABITUS vs best baseline | mean delta +0.163 [+0.049, +0.273] p=0.003 | `habitus/bootstrap_mcc.csv`, `habitus/bootstrap_draws/` | `habitus reproduce --level operating` (release baselines npz) | REPRODUCED, G9; rule columns not included |
| tab:main / summary PIDSMaker rows | 104 rows; best_adp / epoch-11 / better-of-two | `baselines/pidsmaker_rows.csv` | `python scripts/export_baselines.py` (server only) | FROZEN for external users; regenerable on this server |
| tab:efficiency route engine | CADETS 506.8, THEIA 599.2, CLEARSCOPE 252.5, optc 1447.9, ATLASV2 1231.0 s per test day; 26.8-29.2 prompts/s | `habitus/route_cost.csv` | none (derived from the route-engine queue logs) | FROZEN |
| tab:datasets | GT / universe: CADETS 72 GT / 281,589, THEIA 118 GT / 700,902, CLEARSCOPE 41 GT / 111,403, optc 749 GT / 1,489,198, ATLASV2 48 GT / 152,763 | `habitus/main_config.csv`, `habitus/route_stats.json` | `habitus reproduce --level tables` | REPRODUCED (sizes), G1 |
| tab:habit HABITUS rows | k2 0.405/0.557/0.086/0.419/0.405 | `habitus/ablation_e1_e2.csv` | `habitus reproduce --level tables` | REPRODUCED, G1 |
| tab:habit HB0 | e.g. k2 CADETS 0.392 (12/1) | `extension_habit/hb0_rows.csv` | `habitus decide hb0` + `habitus eval habit --levels HB0` (`habitus gates --only G11`; needs the role-table catalogs, `scripts/fetch_catalogs.sh`) | REPRODUCED, G11 |
| tab:habit HB-Z / HB / HB.head / HB.08b / HB-L.s0 / HB-L / HB-L.mq | k2 means 0.3758/0.3481/0.3173/0.3631/0.3248/0.358/0.3518; allc-pre 0.3742/0.3744/0.3758/0.3642/0.3789/0.376/0.375 | `extension_habit/decision_models_best_mcc.csv`, `extension_habit/decision_models_mcc_matrix_k2.csv`, `extension_habit/decision_models_mcc_matrix_allc-pre.csv` | none (frozen) | FROZEN, retraining |
| tab:selection | M1 / M2 / M7 / M3 over 3 folds (27 rows) | `extension_habit/selection_metrics_3folds.csv` | none | FROZEN (ATLASV2_EDR fold missing; M1 / M2 use the HB-C selector; HB-L.full unsupported) |
| tab:efficiency habit rows | GPU-h, q/s, p50 / p95, GiB | `extension_habit/decision_models_speed_per_fold.csv`, `extension_habit/decision_models_speed_mean.csv` | none | FROZEN |
| tab:rules, rule-lex / rule-count, probes, HB-C | – | `not_included.md` | – | EXCLUDED by framing |

REPRODUCED = recomputed from the release cache bundle by the named gate (G1: tables, G4: exact replay of every score file from the answer cache, G9: operating tables, G11: the HB0 rows). FROZEN = a record of a run whose inputs are gone or server-only. Full precision lives in the CSVs; the paper rounds to 3 dp.

## Decision-model status

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

## Layout

- `habitus/`: the main system (jev stock Qwen3.5-4B on the routed QV / QT / QA questions).
- `baselines/`: PIDSMaker 2289cd9 baseline rows (server export) and their provenance.
- `extension_habit/`: HABITUS+H, frozen (see `extension_habit/FROZEN.md`).
- `protocol/datasets.md`: the 5 hosts (rendered from configs/datasets.yaml).
- `not_included.md`: comparators excluded by the decision-model framing, with their paper numbers.
- `MANIFEST.sha256`: sha256 of every file here (`sha256sum -c MANIFEST.sha256` from this directory), with the source of each file as a comment.
