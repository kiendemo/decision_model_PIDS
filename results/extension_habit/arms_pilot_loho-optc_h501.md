> ARCHIVAL session record of the research server, kept as provenance only (paths scrubbed). It may cite files, hardware and runs that are not part of this repository; nothing here reads it.

> Redacted copy for the habitus-dm repository: the HB-C (count back-off) table rows are removed; HB-C is not part of this repository. The tail selector described below (count back-off p) is the one M1 / M2 used, which is why M1 / M2 cannot be recomputed here.

# HABITUS arm comparison - loho-optc_h501 (held-out optc_h501)

Selection items: 9363 items / 8795 questions on ['2019-09-19', '2019-09-21'] (k plans ['k0', 'k2', 'allc-pre'], 5 contexts, 1506664 head nodes); tail items 409 (selector p <= 0.07988; pooled-only tail 9, no rule); questions over HB-L's length budget (n/a for every arm): 0.

Tail selector: tail = present inclusion items of the ELIGIBLE questions whose pooled count back-off p_yes (CountBackoff(target_prof=None, pool): the fold's training hosts only) is in the lowest 0.1 quantile of that p over the eligible present items (ties kept), >= 30 items or the fold is refused; eligible = HB-C draws on the held-out host's own reference (ident@R / class@R) in every context of the question, so HB-C's p is never the selector's (k0 and no-reference items: reported apart as pooled_only_tail, no rule); residual: HB-C's Witten-Bell interpolation keeps the pooled levels as its prior (deferred X2 finding: HB-C's own p < 0.05 selected HB-C's own misses; a fixed 0.05 cut was below the pooled floor).

Definition sha256 914319014071.

SPEC_HB HB-v0-draft sha256 9c9104d59c51, arms section 18: present; code 1f55ca1b7eb8; device cuda:0 NVIDIA RTX A5000.

| arm | kind | M1 tail | M1 -ln p | M2 rare AUC | M3 NLL T1 / Tfit | M4 ECE | M5 cov 1e-2 | M6 |dp| / flip | M7 within-view | S1 items/s | S2 GPU-h | S3 q/s | S4 ms/node | S5 p50 / p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HB-Z | hb-z | 0.8922 | 1.157 | 0.8382 | 0.7926 / 0.7926 | 0.1123 | - | 0.1929 / 0.191 | 0.5845 | - | - | 22.8 | 0.2561 | 133.1 / 218.2 |
| HB | hb | 0.9313 | 1.437 | 0.8026 | 0.7241 / 0.6601 | 0.02241 | - | 0.03538 / 0.02456 | 0.7553 | 5.018 | 1.121 | 22.78 | 0.2562 | 131.4 / 218.6 |
| HB.head | hb-head | 0.8185 | 1.237 | 0.8222 | 0.7222 / 0.6616 | 0.02746 | - | 0.06353 / 0.05514 | 0.7236 | 5.16 | 1.095 | 22.87 | 0.2552 | 132.2 / 219.5 |
| HB.08b | hb | 0.7638 | 1.022 | 0.8566 | 0.7585 / 0.6775 | 0.03024 | - | 0.02949 / 0.03297 | 0.7459 | 17.8 | 0.3166 | 76.15 | 0.07666 | 124 / 154.4 |
| HB-L.s0 | hb-l | 0.7764 | 1.085 | 0.8376 | 0.727 / 0.6875 | 0.04362 | 55.1 | 0.02957 / 0.02365 | 0.7197 | 40.67 | 0.1384 | 65.16 | 0.08959 | 26.87 / 86.99 |
| HB-L | hb-l | 0.7631 | 1.026 | 0.8562 | 0.7276 / 0.6788 | 0.03427 | 59.29 | 0.02232 / 0.02956 | 0.7291 | 40.69 | 0.4129 | 22.14 | 0.2637 | 79.33 / 254.2 |
| HB-L.full | hb-l | 0.7048 | 0.7776 | 0.8507 | 0.7416 / 0.7021 | 0.03591 | 33.71 | 0.03906 / 0.04105 | 0.6706 | 40.86 | 0.1378 | 65.43 | 0.08922 | 25.58 / 96.26 |
| HB-L.mq | hb-l-mq | 0.7471 | 0.8592 | 0.8518 | 0.7769 / 0.7096 | 0.04599 | 14.95 | 0.001278 / 0.005 (MQ check) | 0.6686 | 92.74 | 0.06325 | 249.1 | 0.02344 | 27.92 / 31.3 |
| HB-L.zs | hb-l | 0.7578 | 0.6247 | 0.5844 | 0.8696 / 0.8696 | 0.1357 | - | 0.05211 / 0.1003 | 0.5275 | - | - | 65.39 | 0.08927 | 27.88 / 86.85 |

M8 KL(target || p) per family:

| arm | P1 | P2 | P3 | P4 | P5 | F1 | F2 | N1 | N2 |
|---|---|---|---|---|---|---|---|---|---|
| HB-Z | 1.284 | 0.3702 | 0.3493 | 0.4368 | - | 0.6405 | - | 0.2704 | 1.397 |
| HB | 1.096 | 0.2536 | 0.5392 | 0.3959 | - | 0.539 | - | 0.4119 | 0.5701 |
| HB.head | 1.168 | 0.3058 | 0.4368 | 0.368 | - | 0.4384 | - | 0.3783 | 0.5743 |
| HB.08b | 1.246 | 0.2642 | 0.4391 | 0.4127 | - | 0.5838 | - | 0.3849 | 0.8151 |
| HB-L.s0 | 1.089 | 0.2977 | 0.5593 | 0.3752 | - | 0.5903 | - | 0.3441 | 0.5895 |
| HB-L | 1.182 | 0.269 | 0.4903 | 0.3778 | - | 0.5469 | - | 0.3089 | 0.6172 |
| HB-L.full | 1.141 | 0.301 | 0.5042 | 0.4236 | - | 0.4858 | - | 0.3608 | 0.649 |
| HB-L.mq | 1.28 | 0.3019 | 0.4512 | 0.4283 | - | 0.6087 | - | 0.3375 | 0.6814 |
| HB-L.zs | 1.016 | 0.4584 | 0.8274 | 0.5903 | - | 0.5657 | - | 0.7158 | 1.086 |

Rule inputs (05 6.6; None = an input is missing, X3b / A1 are score-file gates):

- HB: {"X2": {"conditions": {"m1": false, "m2": false}, "fold_pass": false}}
- HB.head: {"R-J": {"a1": null, "fold_pass": null, "m1_gain": true, "selection_fold": true}}
- HB.08b: {"R-J08": {"conditions": {"m1": true, "m2": true, "s4": true}, "fold_pass": true, "s4_speedup": 3.3425114088456804}}
- HB-L.s0: {"R-L1": {"conditions": {"m1": true, "m2": true, "m6_flip": true, "m7": false, "s4": false, "x3b": null}, "fold_pass": null, "s4_speedup": 2.8599947177977256}, "R-L2": {"m2_over_hbc": true}}
- HB-L: {"R-L1": {"conditions": {"m1": true, "m2": true, "m6_flip": true, "m7": false, "s4": false, "x3b": null}, "fold_pass": null, "s4_speedup": 0.9717590665937951}, "R-L2": {"m2_over_hbc": true}}
- HB-L.full: {"R-L1": {"conditions": {"m1": true, "m2": true, "m6_flip": true, "m7": false, "s4": false, "x3b": null}, "fold_pass": null, "s4_speedup": 2.8719265425337634}, "R-L2": {"m2_over_hbc": true}}
- HB-L.mq: {"R-L1": {"conditions": {"m1": true, "m2": true, "m6_flip": true, "m7": false, "s4": true, "x3b": null}, "fold_pass": null, "s4_speedup": 10.93197770802565}, "R-L2": {"m2_over_hbc": true}}
- HB-L.zs: {"R-L1": {"conditions": {"m1": true, "m2": false, "m6_flip": false, "m7": false, "s4": false, "x3b": null}, "fold_pass": null, "s4_speedup": 2.8704160363594835}, "R-L2": {"m2_over_hbc": false}}

Not computed here: {"M9": "X3 / X3b drift gates: score files (checks / decide), not the selection items", "A1": "score-file gate"}
