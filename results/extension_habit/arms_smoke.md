> ARCHIVAL session record of the research server, kept as provenance only (paths scrubbed). It may cite files, hardware and runs that are not part of this repository; nothing here reads it.

# HABITUS clone arms: speed smoke on real habit items (loho-optc_h501, 2026-10-21 23:47:59)

Speed only; no accuracy number is read here. Physical GPU 0 / 1 (A5000 24 GB), never GPU 2. Training: 960 items of one stratified sample of the 18,526 distinct exported training items; serving: 504 real selection questions (h501 09-19 / 09-21) on whole views, median of 3 warm passes. Source: `reports/arms_smoke.json`.

## Findings

- Training (S1, A5000, real habit items, p50 336 Qwen tokens): HB 4B LoRA 5.51 items/s at micro-batch 8 (10.3 GiB; micro-batch 16 is slower, 4.74), HB.head 5.48, HB.08b 19.1 (mb 8) / 20.4 (mb 16), HB-L 46.6 with gradient checkpointing (8.7 GiB; 58.4 without, 18.0 GiB), HB-L.mq 59.8 (15.8 GiB, 1.43 items per packed row).
- Serving (S3, 504 selection questions, both orders, prefix share for HB-J): HB 4B 22.5 questions/s, HB.head 22.8, HB.warm 23.7, HB.08b 75.0 (3.3x), HB-L L-a 65.6 (2.9x), HB-L.mq 247 (11x). Streaming bs 1: HB 4B 163 ms/view p50, HB-L 25, MQ 29.
- HB-L L-a serves at 2.9x HB 4B, just under R-L1's S4 >= 3x bar: HB-J's prefix share forwards 140k tokens for the 504 questions, L-a 361k (the view twice per question); per token HB-L is ~7.5x cheaper. MQ (76k tokens, one pass) clears it at 11x. The serve paths are GPU-bound (CPU encoding <= 4 % of a pass).
- MQ exactness holds to fp32 rounding only: under the default bf16 autocast the in-row order gap is up to 2.3e-2 (1.5 % argmax flips) and adding a block of the same view moves p by up to 1.0e-2; TF32 cuts the gap to 2.4e-3 at 6.5x HB, fp32 to 2.5e-5 (isolation 2.4e-6) at 3.4x HB. M6 = 0 'by construction' needs fp32 serving.
- Every Qwen3.5 arm runs the Gated DeltaNet torch fallback (causal-conv1d not installed): HB-J rates are what this server gives today, not the fast-path ceiling.
- The uncapped fold export holds 18,526 distinct training items; every HB-J arm trains on all of them inside ~2 h, so no cap binds and every sample-based arm shares one sample.

## Training (S1)

| arm | setting | items/s | padded tok/s | peak GiB | items in 2 h | pilot n | pilot train min |
|---|---|---|---|---|---|---|---|
| HB | micro_batch_16 | 4.74 | 1,507 | 12.23 | 34,155 | 18,526 | 65.1 |
| HB | micro_batch_8 | 5.51 | 1,742 | 10.28 | 39,682 | 18,526 | 56.0 |
| HB.head | micro_batch_8 | 5.48 | 1,732 | 10.28 | 39,452 | 18,526 | 56.3 |
| HB.08b | micro_batch_16 | 20.44 | 6,496 | 3.06 | 147,183 | 18,526 | 15.1 |
| HB.08b | micro_batch_8 | 19.12 | 6,042 | 2.65 | 137,645 | 18,526 | 16.2 |
| HB-L | grad_ckpt | 46.57 | 11,777 | 8.68 | 335,288 | 18,526 | 6.6 |
| HB-L | no_grad_ckpt | 58.38 | 14,763 | 17.98 | 420,301 | 18,526 | 5.3 |
| HB-L.mq | no_grad_ckpt | 59.80 | 12,034 | 15.83 | 430,586 | 18,526 | 5.2 |

## Serving (S3 / S5)

| arm | questions/s | x HB 4B | forwarded tok/s | stream p50 / p95 ms per view | peak GiB |
|---|---|---|---|---|---|
| HB | 22.5 | 1.0 | 6,259 | 163.1 / 233.1 | 10.55 |
| HB.head | 22.8 | 1.01 | 6,333 | 134.2 / 219.3 | 10.56 |
| HB.08b | 75.0 | 3.33 | 20,809 | 105.3 / 114.2 | 2.54 |
| HB-L | 65.6 | 2.91 | 47,053 | 25.0 / 79.2 | 3.44 |
| HB-L.mq | 247.2 | 10.97 | 37,352 | 28.7 / 30.4 | 2.61 |
| HB.warm | 23.7 | 1.05 | 6,570 | 151.1 / 235.1 | 10.54 |

Projected on the full selection set (8,795 questions, 3,095 views, 1,506,664 head nodes; subset 3.02 vs full 2.84 questions/view) at the subset rates:

| arm | s per pass | S4 ms/node | x HB | batch ms/view | compare-arms min |
|---|---|---|---|---|---|
| HB | 390.9 | 0.2594 | 1.0 | 126.3 | 22.0 |
| HB.head | 385.7 | 0.256 | 1.01 | 124.6 | 21.7 |
| HB.08b | 117.3 | 0.0778 | 3.33 | 37.9 | 6.9 |
| HB-L | 134.1 | 0.089 | 2.91 | 43.3 | 7.9 |
| HB-L.mq | 35.6 | 0.0236 | 10.98 | 11.5 | 2.5 |
| HB.warm | 371.1 | 0.2463 | 1.05 | 119.9 | 20.9 |

Serve pass split (GPU 0, subset): L-a wall 7.54 s of which forward 7.30 s; MQ wall 2.01 s, forward 1.92 s: the serve paths are GPU-bound (CPU encoding <= 4 %).

MQ order check on the GPU (bf16, zero-shot init, 200 questions): max |p_fwd - p_rev| 0.0226, mean 0.0036, argmax flips 0.015 (exact only in fp32: the CPU tests' 1e-5 bound).

Encoder precision on the GPU (zero-shot init, subset; x HB = vs HB 4B bf16 serving):

| precision | HB-L q/s (x HB) | HB-L.mq q/s (x HB) | MQ order gap max / mean | MQ argmax flips |
|---|---|---|---|---|
| bf16 | 66.3 (2.95) | 249.1 (11.07) | 2.26e-02 / 3.61e-03 | 0.015 |
| tf32 | 39.5 (1.76) | 146.4 (6.51) | 2.42e-03 / 2.19e-04 | 0.000 |
| fp32 | 18.9 (0.84) | 77.4 (3.44) | 2.50e-05 / 7.89e-07 | 0.000 |

MQ question isolation (a block alone vs with a second block of its view in the row, 26 pairs): max |dp| bf16 1.00e-02, fp32 2.41e-06. Exact (I1) / order invariance hold to fp32 rounding only; under the default bf16 autocast both move p by up to ~1e-2.

## HB.warm checkpoint

OneJev1.6-4B (Apache-2.0, 9.10 GB, pinned revision 7d55a87ce87f) downloaded to former data disk in 95 min (~1.6 MB/s over xet, connection-limited; plain HTTPS ~0.6 MB/s); sha256 of the weights equals the LFS blob id, refs/main is the pinned revision, no remote code. Tokenizer vs Qwen3.5-4B: letter ids equal, 1008 / 1008 real habit prompts render and tokenize identically (prefix share and HB's letter readout carry over). Stock serving 23.7 questions/s, i.e. the 4B cost. pplx-decider-flash-v1.2 was not downloaded: no LM head, so no letter logits (heads.WARM: incompatible).

## Pilot items and plan

- Pilot export `items/loho-optc_h501`: uncapped (full fold export took ~3 min), 18,526 distinct training items + 5,236 inner items of CADETS_E3, THEIA_E3, ATLASV2_EDR, CLEARSCOPE_E3; prompt tokens p50 344 / max 609; 0 views refused.
- Selection items `items/loho-optc_h501.select`: 9,363 items / 8,795 questions (re-export byte-identical).
- Every distinct training item (18,526) fits the ~2 h budget of the 4B arms, so every sample-based arm trains on ONE sample (sample_fold n = 18,526 keeps every item, host-balanced) and compare-arms' same-data check holds; HB-L.full (every exported item, its own weights) is the 'HB-L 1 seed full pilot', HB-L.s0 its same-sample twin.
- Scripts (not started): `logs/run_pilot_gpu0.sh` (HB-L.full, HB-L.s0, HB-L.mq, HB.head) and `logs/run_pilot_gpu1.sh` (HB, HB.08b, then compare-arms with HB-C / HB-Z / HB-L.zs once SPEC_HB 18's sha256 is logged in research/29, or PILOT_ALLOW_UNREGISTERED=1).
