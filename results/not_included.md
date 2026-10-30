# Not included in this repository

This repository frames HABITUS as a decision-model pipeline. Rule COMPARATORS are therefore not part of it: no code, no rows, no bootstrap systems. The paper's own numbers for them are cited below, because one of them is the paper's key negative finding. (The pipeline itself does contain fixed, a-priori gating and fallback rules - the head selection rule, the rarity fallback, the hard-tier acquittal exemption, the HB0 lineage split - none learned or label-tuned; see the repository README.)

## rule-lex: ties HABITUS

A no-LLM rule over the same compiled state, written after the HABITUS results were known and not tuned: rule-lex = (4·[hard tier] + 2·[identity_unseen] + n_dev/(1+n_dev) + n_ev/(1+n_ev)) / 8.

- Oracle node MCC: CADETS_E3 0.601, THEIA_E3 0.505, CLEARSCOPE_E3 0.106, optc_h501 0.332, ATLASV2_EDR 0.371; mean 0.383. HABITUS main: mean 0.382.
- **rule-lex ties HABITUS on the 5-host mean**: delta HABITUS − rule-lex = -0.001 [-0.049, +0.050] (paired stratified entity bootstrap, B = 2000), and it beats HABITUS on CADETS_E3 (0.601 vs 0.405).
- Source: `paper/habitus/derive/editor/state_<DS>.json` (rules), `paper/habitus/derive/editor/editor_cpu.json` (summary.boot_deltas); generator `paper/habitus/derive/editor_cpu.py state / boot / merge`.

## rule-count

rule-count = (n_dev + n_ev) / (1 + n_dev + n_ev) over the same state.

- Oracle node MCC: CADETS_E3 0.219, THEIA_E3 0.293, CLEARSCOPE_E3 0.096, optc_h501 0.465, ATLASV2_EDR 0.425; mean 0.300; delta HABITUS − rule-count = +0.082 [-0.020, +0.162].

## Label-free probes (allowlist, rarity)

- allowlist (name / kappa unseen on the reference days), main configuration: CADETS_E3 0.092, THEIA_E3 0.015, CLEARSCOPE_E3 0.116, optc_h501 0.156, ATLASV2_EDR 0.220; mean 0.120.
- rarity probe (signature rarity vs the reference days): CADETS_E3 0.045, THEIA_E3 0.015, CLEARSCOPE_E3 0.073, optc_h501 0.116, ATLASV2_EDR 0.048; mean 0.060.
- Source: `paper/habitus/derive/zeroshot_derived.json` (probe_*_main), from the probe rows of `research/results/reflex/habitus/confusion_reference.md`.

## HB-C (count back-off) and tab:rules

- HB-C = HB0 + a habit channel from a Laplace count back-off over the fold's training items (no model). Main-configuration mean MCC: composite 0.3949, habit-only 0.1095 (`paper/habitus/derive/zeroshot_derived.json` habit_main_config).
- Its selection-metric rows are removed from `extension_habit/selection_metrics_3folds.csv`. Because HB-C's count back-off is the tail selector of M1 / M2, those two metrics cannot be recomputed here.
- tab:rules (X2, R-L2) is defined against HB-C and is omitted.
