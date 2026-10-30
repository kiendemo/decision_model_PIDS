# Zero-shot HABITUS ablations: node-level MCC at the oracle best-MCC threshold

Generated 2026-10-29 18:45:14 by `habitus eval tables` (CPU only) from the frozen score files `scores/<ds>/<level>_k<k>/` of the result bundle, metric code `habitus.eval.metrics.confusion` / `best_mcc`. Full rows (TP, FP, TN, FN, Precision, Recall, F1, MCC, AUC, AP, threshold, tie block, ties-negative MCC, GT in head): `ablation_e1_e2.csv`.

**Protocol.** Universe = every test-day node + ground-truth nodes without a test-day card (always FN); node score = max over the test days; signatures outside the engine head get the KC-S1 rarity fallback (< 0, below every engine score). The threshold MAXIMISES MCC on the test labels for every row (oracle upper bound, no FP budget); AUC / AP are threshold-free context. Cells: MCC (TP/FP). `‡` = CADETS_E3 k=all, whose reference includes clean days after its first test day (oracle-ish). `n/a` = not defined by design (k=0 has no reference: L2 = L1 + stub, L4 = L3 + stub, SPEC 4). `·` = never run on that host. All 5 hosts are design-informed development hosts (E2's routing and name-free evidence were chosen after a label-reading diagnosis on THEIA_E3); these are not held-out numbers.

Score columns. E1 (single unrouted state, every question reads every section): composite = max(qv, qt). E2 (routed): qv = QV alone (verdict, 4 levels), qt = QT alone (accuse-only tradecraft), base = max(qv, qt) (routing, no acquittal), composite = base * (1 - 0.5 qa) outside the hard tier (the HABITUS main detector), 1-qa = the acquittal question alone read as suspicion (1 when QA is not asked, i.e. no precedent), acq_all = bounded acquittal applied to every node (removes the hard-tier exemption; extra ablation). At k=0 QA is never asked, so composite = base and 1-qa is constant.

## 1. State ladder (E1, unrouted, composite = max(qv, qt)) vs routed E2

Levels (cumulative, SPEC 4): L0 entity; L1 + hard bits; L2 + reference profile + deviation; L3 + evidence (neighbour NAMES); L4 + precedent. E2 = routed questions over entity / hard bits / reference / deviation2 / name-free evidence / precedent2 with bounded acquittal.

### k0

| state | CADETS_E3 | THEIA_E3 | CLEARSCOPE_E3 | optc_h501 | ATLASV2_EDR | mean (n) |
|---|---|---|---|---|---|---|
| L0 | 0.289 (6/0) | 0.065 (1/1) | · | 0.005 (11/3871) | 0.354 (6/0) | 0.178 (4) |
| L1 | 0.378 (12/2) | 0.307 (25/31) | · | 0.014 (5/161) | 0.279 (33/258) | 0.245 (4) |
| L2 | n/a | n/a | n/a | n/a | n/a | – (0) |
| L3 | 0.312 (7/0) | 0.117 (31/559) | 0.008 (1/290) | 0.284 (355/1730) | 0.343 (35/182) | 0.213 (5) |
| L4 | n/a | n/a | n/a | n/a | n/a | – (0) |
| E2 composite | 0.354 (9/0) | 0.166 (18/81) | 0.009 (1/247) | 0.155 (719/27275) | 0.219 (36/522) | 0.181 (5) |

### k2

| state | CADETS_E3 | THEIA_E3 | CLEARSCOPE_E3 | optc_h501 | ATLASV2_EDR | mean (n) |
|---|---|---|---|---|---|---|
| L0 | 0.289 (6/0) | 0.093 (23/488) | · | 0.005 (11/4019) | 0.289 (6/3) | 0.169 (4) |
| L1 | 0.378 (12/2) | 0.302 (25/33) | · | 0.014 (5/167) | 0.275 (33/266) | 0.242 (4) |
| L2 | 0.374 (11/1) | 0.579 (58/27) | 0.019 (21/17052) | 0.263 (721/9219) | 0.353 (22/59) | 0.318 (5) |
| L3 | 0.236 (4/0) | 0.281 (61/338) | · | 0.271 (721/8685) | 0.450 (34/85) | 0.309 (4) |
| L4 | 0.263 (5/0) | 0.287 (61/322) | 0.009 (41/90996) | 0.180 (724/20493) | 0.479 (34/71) | 0.244 (5) |
| E2 composite | 0.405 (15/4) | 0.557 (59/36) | 0.086 (37/4313) | 0.419 (713/3141) | 0.405 (21/35) | 0.374 (5) |

### kall

| state | CADETS_E3 | THEIA_E3 | CLEARSCOPE_E3 | optc_h501 | ATLASV2_EDR | mean (n) |
|---|---|---|---|---|---|---|
| L0 | 0.289 (6/0)‡ | 0.095 (23/475) | · | 0.005 (11/4027) | 0.289 (6/3) | 0.169 (4) |
| L1 | 0.378 (12/2)‡ | 0.302 (25/33) | · | 0.014 (5/165) | 0.272 (33/272) | 0.242 (4) |
| L2 | 0.392 (12/1)‡ | 0.593 (58/23) | 0.012 (41/79209) | 0.262 (721/9321) | 0.355 (22/58) | 0.323 (5) |
| L3 | 0.365 (12/3)‡ | 0.296 (60/288) | · | 0.278 (721/8229) | 0.453 (34/83) | 0.348 (4) |
| L4 | 0.391 (11/0)‡ | 0.300 (62/300) | 0.013 (1/134) | 0.194 (724/17628) | 0.483 (34/69) | 0.276 (5) |
| E2 composite | 0.392 (12/1)‡ | 0.596 (60/26) | 0.112 (39/2831) | 0.393 (714/3684) | 0.405 (24/49) | 0.380 (5) |

## 2. E2 question ablation (routed)

### k0

| score | CADETS_E3 | THEIA_E3 | CLEARSCOPE_E3 | optc_h501 | ATLASV2_EDR | mean (n) |
|---|---|---|---|---|---|---|
| QV only | 0.312 (7/0) | 0.166 (18/81) | 0.011 (41/82136) | 0.155 (716/27175) | 0.220 (36/518) | 0.173 (5) |
| QT only | 0.391 (11/0) | 0.283 (25/41) | 0.001 (41/110869) | 0.026 (1/1) | 0.266 (43/501) | 0.193 (5) |
| base = max(QV, QT), no acquittal | 0.354 (9/0) | 0.166 (18/81) | 0.009 (1/247) | 0.155 (719/27275) | 0.219 (36/522) | 0.181 (5) |
| composite (bounded acquittal, QA) | 0.354 (9/0) | 0.166 (18/81) | 0.009 (1/247) | 0.155 (719/27275) | 0.219 (36/522) | 0.181 (5) |

At k=0 QA is not asked (no reference, no precedent): composite = base; 1-qa / acq_all are undefined.

### k2

| score | CADETS_E3 | THEIA_E3 | CLEARSCOPE_E3 | optc_h501 | ATLASV2_EDR | mean (n) |
|---|---|---|---|---|---|---|
| QV only | 0.314 (8/1) | 0.569 (56/26) | 0.077 (41/6489) | 0.430 (714/2954) | 0.319 (20/62) | 0.342 (5) |
| QT only | 0.391 (11/0) | 0.496 (57/55) | 0.005 (41/103839) | 0.223 (722/13121) | 0.315 (29/147) | 0.286 (5) |
| base = max(QV, QT), no acquittal | 0.392 (12/1) | 0.509 (59/55) | 0.077 (41/6522) | 0.430 (714/2956) | 0.360 (23/62) | 0.353 (5) |
| composite (bounded acquittal, QA) | 0.405 (15/4) | 0.557 (59/36) | 0.086 (37/4313) | 0.419 (713/3141) | 0.405 (21/35) | 0.374 (5) |
| QA only, 1 - qa | 0.119 (13/151) | 0.230 (58/482) | 0.041 (37/16328) | 0.336 (711/5257) | 0.277 (15/46) | 0.200 (5) |
| acquittal on every node (no hard tier) | 0.374 (11/1) | 0.560 (59/35) | 0.086 (37/4313) | 0.419 (713/3141) | 0.405 (21/35) | 0.369 (5) |

### kall

| score | CADETS_E3 | THEIA_E3 | CLEARSCOPE_E3 | optc_h501 | ATLASV2_EDR | mean (n) |
|---|---|---|---|---|---|---|
| QV only | 0.314 (8/1)‡ | 0.590 (57/22) | 0.106 (41/3515) | 0.375 (713/4087) | 0.335 (20/54) | 0.344 (5) |
| QT only | 0.391 (11/0)‡ | 0.563 (57/30) | 0.005 (41/103011) | 0.228 (722/12537) | 0.387 (30/95) | 0.315 (5) |
| base = max(QV, QT), no acquittal | 0.392 (12/1)‡ | 0.589 (59/26) | 0.106 (41/3523) | 0.376 (714/4091) | 0.371 (24/63) | 0.367 (5) |
| composite (bounded acquittal, QA) | 0.392 (12/1)‡ | 0.596 (60/26) | 0.112 (39/2831) | 0.393 (714/3684) | 0.405 (24/49) | 0.380 (5) |
| QA only, 1 - qa | 0.217 (13/37)‡ | 0.236 (57/438) | 0.070 (1/4) | 0.347 (713/4912) | 0.504 (16/5) | 0.274 (5) |
| acquittal on every node (no hard tier) | 0.374 (11/1)‡ | 0.596 (59/24) | 0.112 (39/2831) | 0.393 (714/3684) | 0.405 (24/49) | 0.376 (5) |

## 3. Reference size (E2 composite)

| k | CADETS_E3 | THEIA_E3 | CLEARSCOPE_E3 | optc_h501 | ATLASV2_EDR | mean (n) |
|---|---|---|---|---|---|---|
| k0 | 0.354 (9/0) | 0.166 (18/81) | 0.009 (1/247) | 0.155 (719/27275) | 0.219 (36/522) | 0.181 (5) |
| k2 | 0.405 (15/4) | 0.557 (59/36) | 0.086 (37/4313) | 0.419 (713/3141) | 0.405 (21/35) | 0.374 (5) |
| kall | 0.392 (12/1)‡ | 0.596 (60/26) | 0.112 (39/2831) | 0.393 (714/3684) | 0.405 (24/49) | 0.380 (5) |
| **main config** (kall; CADETS k2) | 0.405 (15/4) | 0.596 (60/26) | 0.112 (39/2831) | 0.393 (714/3684) | 0.405 (24/49) | **0.382** (5) |

AP / AUC of the same rows (threshold-free context):

| k | CADETS_E3 | THEIA_E3 | CLEARSCOPE_E3 | optc_h501 | ATLASV2_EDR |
|---|---|---|---|---|---|
| k0 | 0.164 / 0.574 | 0.033 / 0.907 | 0.000 / 0.116 | 0.025 / 0.946 | 0.046 / 0.984 |
| k2 | 0.214 / 0.767 | 0.333 / 0.945 | 0.007 / 0.967 | 0.143 / 0.970 | 0.128 / 0.985 |
| kall | 0.217 / 0.763 | 0.361 / 0.949 | 0.010 / 0.977 | 0.137 / 0.970 | 0.139 / 0.989 |

## 4. Sanity cross-check vs the published E2 rows

Reference = `paper/habitus/facts.md` routes-only (E2) rows = `confusion_reference.md` E2 (KC-S1) rows. `loader identical` = the node vector computed here equals `habitus.evaluate.e2_vector` element-wise.

| host | k | reference MCC (TP/FP) | here MCC (TP/FP) | match | loader identical |
|---|---|---|---|---|---|
| CADETS_E3 | k2 | 0.4055 (15/4) | 0.4055 (15/4) | yes | True |
| CADETS_E3 | kall | 0.3922 (12/1) | 0.3922 (12/1) | yes | True |
| THEIA_E3 | k2 | 0.5572 (59/36) | 0.5572 (59/36) | yes | True |
| THEIA_E3 | kall | 0.5956 (60/26) | 0.5956 (60/26) | yes | True |
| CLEARSCOPE_E3 | k2 | 0.0855 (37/4313) | 0.0855 (37/4313) | yes | True |
| CLEARSCOPE_E3 | kall | 0.1121 (39/2831) | 0.1121 (39/2831) | yes | True |
| optc_h501 | k2 | 0.4192 (713/3141) | 0.4192 (713/3141) | yes | True |
| optc_h501 | kall | 0.3929 (714/3684) | 0.3929 (714/3684) | yes | True |
| ATLASV2_EDR | k2 | 0.4048 (21/35) | 0.4048 (21/35) | yes | True |
| ATLASV2_EDR | kall | 0.4052 (24/49) | 0.4052 (24/49) | yes | True |

All E2 composite rows reproduce the published numbers: **yes**. Main-config mean MCC here = 0.3822 (facts.md: 0.3822).

E1 ladder: AP of 7/7 spot-checked cells equals the frozen `research/results/reflex/kcs1/e1_ladder.md` AP to 3 decimals (CADETS_E3 L1 k0 0.172, CADETS_E3 L2 kall 0.199, THEIA_E3 L2 k2 0.355, THEIA_E3 L4 kall 0.081, ATLASV2_EDR L3 k0 0.129, ATLASV2_EDR L4 k2 0.214, optc_h501 L3 kall 0.109).

## 5. Derived deltas (mean MCC difference over hosts where both rows exist)

| comparison | k0 | k2 | kall |
|---|---|---|---|
| acquittal: E2 composite - E2 base | +0.000 (0/0/5) | +0.021 (4/1/5) | +0.013 (4/0/5) |
| routed + name-free state, no acquittal: E2 base - E1 L4 | – | +0.110 (4/1/5) | +0.091 (4/1/5) |
| full E2 - E1 L4 (E1 primary) | – | +0.131 (4/1/5) | +0.103 (4/1/5) |
| full E2 - E1 L2 | – | +0.057 (4/1/5) | +0.057 (4/0/5) |
| full E2 - E1 L3 (E1 primary at k0) | -0.032 (3/2/5) | +0.137 (3/1/4) | +0.098 (3/1/4) |
| hard tier: E2 composite - acq_all | – | +0.006 (1/1/5) | +0.003 (1/1/5) |
| E2 composite - QV only | +0.008 (2/2/5) | +0.033 (3/2/5) | +0.035 (5/0/5) |
| E2 composite - QT only | -0.013 (2/3/5) | +0.088 (5/0/5) | +0.065 (5/0/5) |

Cells: mean delta (hosts better / worse / n). CADETS_E3 kall is included in the kall column (‡ caveat).

## 6. Availability of score directories

| host | missing (never run) | by design (k=0) |
|---|---|---|
| CADETS_E3 | none | L2_k0, L4_k0 |
| THEIA_E3 | none | L2_k0, L4_k0 |
| CLEARSCOPE_E3 | L0_k0, L0_k2, L0_kall, L1_k0, L1_k2, L1_kall, L3_k2, L3_kall | L2_k0, L4_k0 |
| optc_h501 | none | L2_k0, L4_k0 |
| ATLASV2_EDR | none | L2_k0, L4_k0 |

Test days: CADETS_E3 2018-04-06, 2018-04-11, 2018-04-12, 2018-04-13; THEIA_E3 2018-04-10, 2018-04-12, 2018-04-13; CLEARSCOPE_E3 2018-04-11, 2018-04-12; optc_h501 2019-09-23, 2019-09-24, 2019-09-25; ATLASV2_EDR 2022-07-19.
