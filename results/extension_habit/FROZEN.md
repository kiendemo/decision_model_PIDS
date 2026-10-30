# HABITUS+H extension: FROZEN results

Everything in this directory except `hb0_rows.csv` (reproduced by gate G11) is a frozen record. The habit-channel adapters, Laya run directories, ensemble temperatures, habit nulls, habit score files and habit answer caches lived on the former data disk and are unavailable. They are being retrained; retrained numbers will differ (seeds, bf16).

- `decision_models_best_mcc.csv` (126 rows) is the authoritative source of the 7 detection-scored arms (HB-Z, HB, HB.head, HB.08b, HB-L.s0, HB-L, HB-L.mq; composite and habit-only, k2 and allc-pre). It supersedes the partial arm rows of the stale confusion_reference.md (2026-10-22).
- `decision_models_mcc_matrix_k2.csv`, `decision_models_mcc_matrix_allc-pre.csv`: 4-dp MCC matrices.
- `decision_models_speed_per_fold.csv`, `decision_models_speed_mean.csv`: training / serving speed (RTX A5000, bf16); the GiB columns are constants measured once in the GPU smoke run.
- `selection_metrics_3folds.csv`: label-free selection metrics M1 / M2 / M7 / M3 over 3 LOHO folds (the ATLASV2_EDR fold is missing). The HB-C rows (3) are removed. The 3 HB-L.full rows are kept and flagged FROZEN/unsupported: this repository has no HB-L.full arm or config. M1 / M2 used the HB-C count back-off as the tail selector (arms_pilot_loho-optc_h501.md), so they cannot be recomputed here.
- `hb0_rows.csv`: HB0 (lineage-split decision layer over the frozen E2 routes) rows parsed from confusion_reference.md. REPRODUCED on CPU: `habitus decide hb0` + `habitus eval habit --levels HB0` recompute all 9 rows exactly, every column (gate G11; needs the role-table catalogs of `scripts/fetch_catalogs.sh`, whose table must match ROLE_TABLE_SHA256).
- `run_facts_20261029.md`, `arms_pilot_loho-optc_h501.md`, `arms_smoke.md`: ARCHIVAL session records of the research server, kept as provenance (HB-C table rows removed, the former data-disk paths replaced, an archival banner added). They may cite files and hardware that are not part of this repository; nothing here reads them.
- `provenance/make_csv.py`: the generator of the decision-model CSVs, kept as provenance only (it read the reports on the former data disk and cannot run).
