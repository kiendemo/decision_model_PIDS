# The habit channel (HABITUS+H)

Status: **FROZEN.** The habit-channel numbers in `results/extension_habit/` are frozen CSVs. The
adapters, Laya runs, ensemble temperatures, habit nulls, score files and answer caches of the habit
arms were stored on the former data disk and are gone. They are being retrained (GPU, later phase).
Retrained numbers will differ from the frozen ones (seeds, bf16 batch noise), so the code here is
the supported path, not a reproduction of the frozen rows. HB0 (no model call) is reproduced on CPU:
`habitus decide hb0` + `habitus eval habit --levels HB0` recompute all 9 rows of
`results/extension_habit/hb0_rows.csv` exactly, every column (gate G11, about 2 minutes). HB0 needs the
role-table catalogs (`scripts/fetch_catalogs.sh`; their table must match `vocab.ROLE_TABLE_SHA256`),
which the result bundle does not hold: without them it stops with `RoleTableError` and G11 is SKIP.

The selection metrics M1 / M2 in `results/extension_habit/selection_metrics_3folds.csv` used the
HB-C count back-off as their tail selector (section 18). HB-C is a rule comparator and is not part of
this repository, so M1 / M2 ship as frozen rows and cannot be recomputed here.

This document is the habit part of the original specification, trimmed to what the repository
implements (sections 2, 6-13 and 18). Code map:

| piece | module |
|---|---|
| constants (mirrored by `configs/habit.yaml`) | `habitus.questions.habit.constants` |
| data contract, `to_typed`, `answer_of`, `DMAnswerSource` | `habitus.questions.habit.types` |
| vocabularies, role table | `habitus.questions.habit.vocab` |
| attestation tiers | `habitus.questions.habit.attest` |
| reference habit indices | `habitus.questions.habit.profile2` |
| views, IFC | `habitus.questions.habit.views`, `habitus.questions.habit.ifc` |
| buckets, candidates, observations | `habitus.questions.habit.targets` |
| question text, renderer | `habitus.questions.habit.text` |
| Q0 head | `habitus.questions.habit.head` |
| training items | `habitus.train.items` |
| surprise, null, s_hat, bars, score files | `habitus.decide.habit.{surprise, null2, bars, score}` |
| HB0 | `habitus.decide.habit.retrofit` |
| driver over any decision model | `habitus.decide.habit.channel` |

Every habit question is answered by a DecisionModel (`habitus.dm`) through
`types.DMAnswerSource`: `to_typed(q)` turns a habit question into a `dm.base.TypedQuestion` (one
`Form` per option order, system SYSTEM_H, state = the view text), and `answer_of(decision)` turns
the model's `Decision` back into an `Answer`.

## 2. Hosts, days, folds, k modes

Host roles (`constants`): DEV = CADETS_E3, THEIA_E3, ATLASV2_EDR, optc_h501; TRAIN_ONLY =
CLEARSCOPE_E3 (train-only in every fold; host-seen row, never a claim). All five are design-informed
development hosts.

**Pre-deployment window** `types.predeploy(host)` = usable clean days (with a card file) strictly
before the first configured test day. Every reference, null and training day is in it. Frozen day
lists (`configs/datasets.yaml`):

| host | pre-deployment days | test days | inner (T) day |
|---|---|---|---|
| CADETS_E3 | 04-02, 04-03, 04-04, 04-05 (04-07..04-10 are after the first test day) | 04-06, 04-11, 04-12, 04-13 | 04-05 |
| THEIA_E3 | 04-03, 04-04, 04-05, 04-09 | 04-10, 04-12, 04-13 | 04-09 |
| ATLASV2_EDR | 07-16, 07-17, 07-18 | 07-19 | 07-18 |
| optc_h501 | 09-19, 09-20, 09-21, 09-22 | 09-23, 09-24, 09-25 | 09-22 |
| CLEARSCOPE_E3 | 04-03..04-05, 04-07..04-10 (7) | 04-11, 04-12 | 04-10 |

**Folds** (`FOLDS`): `loho-<dev host>` trains on the other 3 dev hosts + CLEARSCOPE_E3 and scores the
held-out dev host's test and null days; `all` (M_all) trains on all 5 hosts. A fold never reads its
held-out host's days in training, and a dev host is never scored by a fold that trains on it.
**Inner day** `types.inner_day(host)` = last pre-deployment day of a training host: never a training
day, never in any reference; its items (role "inner") fit temperatures only.
**h501 parity split**: the arm-selection metrics score only 09-19 and 09-21 of optc_h501.

**k modes** (`Ref.k_mode`, KMODES): `k0` no reference; `k1` one day (training augmentation only);
`k2` = shots(2) for a test day (verified pre-deployment, asserted), one day on a k = 2 null day;
`allc-pre` = the whole window (minus the day itself on null / training days). **k plans** (score
directories, KPLANS): `k0` (the host's own pre-deployment days as null; exploratory on dev hosts),
`k2` (decide.null.lodo_plan: each shot day scored with the other as reference; the disclosed 1-day
vs 2-day null mismatch), `allc-pre` (LODO inside the window; equal to the routed k = all on every
host except CADETS_E3, so the E2_kall files are reused there; CADETS_E3 allc-pre needs fresh routes
and is refused here). k = 1 result rows do not exist.

## 6. Families and availability

| fam | qid | node | readout | view | S_f(v) / observed option (from v's OWN complete op keys and links) | needs | E_mono | k=0 |
|---|---|---|---|---|---|---|---|---|
| P1 | H-P1-parent | process | choice-4 | V_par | kappa' of the code-1 cause whose file_name = parent_name -> usual / attested-other / never-seen; no parent_name and no code-1 cause -> none; else n/a | spawn | no | n/a |
| P2 | H-P2-modify | process | A/B per bucket | V_beh | dir classes of op keys write/create/rename/delete:<dir> | - | yes | classes |
| P3 | H-P3-peers | process | A/B | V_beh | endpoint classes of code-4 netflow effects + op keys <verb>:net-<ep> | - | yes | classes |
| P4 | H-P4-children | process | A/B | V_beh | ident keys of code-1 process effects | spawn | yes | cls:<child image_dir> |
| P5 | H-P5-exec-from | process | A/B | V_beh | dir classes of code-3 file causes (link [file, process, 3]) | exec | yes | classes |
| F1 | H-F1-writer | file | A/B | V_file | ident keys of code-2 process causes | - | yes | cls:<writer image_dir> |
| F2 | H-F2-executor | file | A/B | V_file | ident keys of code-3 process effects | exec | yes | cls:<executor image_dir> |
| N1 | H-N1-owner | netflow | A/B | V_net | ident keys of code-4 process causes (CADETS process->file send links never count) | - | yes | cls:<owner image_dir> |
| N2 | H-N2-address | netflow | choice-4 per owner bucket | V_own | deepest match of the flow's ip_levels chain in the owner's / host's reference chains (section 7) | - | yes | n/a |

Links (`data.cards`): `[cause, effect, code]`, 1 spawn [parent, child], 2 write [process, file],
3 exec [file, process], 4 net [process, netflow] (CADETS also [process, file]); a sorted set with
no time order. Availability (measured 2026-10-21 by counting codes, label-free; `types.available`):

| host | P1 | P2 | P3 | P4 | P5 | F1 | F2 | N1 | N2 |
|---|---|---|---|---|---|---|---|---|---|
| CADETS_E3 | - | x | x | - | x | x | x | x | x |
| THEIA_E3 | x | x | x | x | x | x | x | x | x |
| ATLASV2_EDR, optc_h501 | x | x | x | x | - | x | - | x | x |
| CLEARSCOPE_E3 | - | x | x | - | - | x | - | x | x |
| any other host (test doubles) | measured on its pre-deployment link files (`available(ds, k, obs)`) |

At k = 0, P1 and N2 are n/a. Applicability never depends on test-day counts: host table, k, and
invariant fields only (N2: n/a when `state.profile.chain_of(card)` is None, i.e. private / loopback /
unusable address). P1 ambiguity is link-dependent, which is why P1 is excluded from E_mono.

## 7. Buckets and candidates

- Candidates `C_f` are a function of (view, reference) ONLY: never of today's links.
- Class families: P2 / P5 buckets `dir:<w>` for the 8 dir words, P3 `ep:<w>` for the 6 endpoint words
  (fixed DIR_ORDER / EP_ORDER).
- Identity families (P4, F1, F2, N1), k >= 1: up to MAX_LISTED = 6 listed usual identities `id:<ident>`
  (only identity-attested elements; ordered by (-reference days, key)) from: P4 the node's own
  habit(kappa', P4); F1 / F2 class_habit((dir, ext class)); N1 class_habit((endpoint, port class,
  prefix level)); plus `gen:attested-other` (identity-attested on R, not listed) and `gen:never-seen`
  (anything else: unseen, PATTERN, empty). <= 8 buckets. A masquerading element is never-seen
  whatever its name. At k = 0: `cls:<image_dir class of the other end>` (<= 8).
- P1 options (canonical) `usual` (one of <= 4 listed identity-attested usual parents, alphabetical),
  `attested-other`, `never-seen`, `none`.
- N2 owner buckets per netflow node: `id:<owner kappa'>` for each identity-attested owner (own V_own
  view), `gen:typed` for all others (one typed V_own view per flow signature). Options (canonical):
  `same-address` (owner's reference chains contain the flow's /32 or ip|port element; IPv6: /128),
  `same-network` (else they contain its /16 or /24; IPv6 /48 or /64), `other-programs` (else the
  host's chains contain its /16 or deeper), `no-program` (else). Owner chains are looked up by the
  owner's ident key whatever its tier; x_N2 = max over all owner buckets.
- Observed sets are PER NODE (never the signature's representatives, never the card's TOP = 5 lists);
  the union of present buckets over a signature's members is <= 8 per family, so prompts are scored
  once per (view, bucket).

## 8. Views, removal table and IFC

Views are functions of (invariant card fields, reference) ONLY: length cannot grow with test-day
activity; > MAX_VIEW_TOKENS = 1024 tokens -> refused, logged, family n/a (never truncated).
Sections (`types.VIEW_SECTIONS`, fixed order, compact JSON via `state.builder.serialize` after
`state.sanitize`):

| view | families | sections | removed (in addition to ALL behaviour, deviation, evidence, precedent) |
|---|---|---|---|
| V_beh | P2-P5 | entity, hard_bits, reference, class_habits | behaviour-derived hard bits (ran_file_written_today, wrote_file_run_later, writes_to_writable_dirs, runs_from_dirs, external_peers), parent fields (parent_name, has_parent, parent_child_seen), behaviour_seen_before |
| V_par | P1 | entity, hard_bits, reference | parent_name, has_parent, parent_child_seen, the full-kappa identity_seen_days (kappa' is used), behaviour_seen_before |
| V_file | F1, F2 | entity, hard_bits, reference | n_writers, executed, run_after_written_today, deleted_today, behaviour_seen_before; executable_type from the extension only |
| V_net | N1 | entity, hard_bits, reference | n_processes, processes, behaviour_seen_before, the raw remote address |
| V_own | N2 | entity (owner), connection, reference | the remote address and every chain element of this connection |

- entity = `vocab.entity_words(card, tier0)`: typed words; raw strings only for IDENTITY-attested
  identities. hard_bits = invariant subset only (process image_location, image_deleted,
  runs_from_writable, observable; file location, executable_type (extension); netflow endpoint,
  port_class). reference = R0 facts keyed by EXACT identity (days, seen_days "n/k", frequency level,
  usual elements with day bins, <= 6 listed). class_habits = R0 class-level habits of (role class or
  "unknown", image_dir). k = 0: reference = {"days": 0}, no class_habits. E2's name-keyed habits never
  enter a habit view.
- The role class comes from a table built by code from public catalogs (Debian bookworm
  Contents-amd64, GTFOBins, LOLBAS, xcyclopedia), fetched at pinned versions by
  `scripts/fetch_catalogs.sh` (never vendored) and checked against `vocab.ROLE_TABLE_SHA256`; missing
  catalogs or a mismatch raise. The script writes only inside this repository (e.g.
  `scripts/fetch_catalogs.sh models/catalogs2`); a catalog directory elsewhere, or under a read-only
  input, is verified (sha256s, pinned commits, role table) and never updated.
- IFC labels (`types.Label`): K0 invariant / host constant (incl. raw strings of identity-attested
  identities); K+ test-day behaviour; R0 reference-only; Atau typed word of an attacker-choosable
  string; A raw unattested string; K- anything padding can move toward benign. Habit views may carry
  only K0, R0, Atau; `ifc.assert_view` raises at build time.

## 9. Questions and readout

- One prompt = Qwen chat template (`enable_thinking=False`, `</think>\n\n` tail asserted) with system
  SYSTEM_H (<= 60 tokens: data-guard sentence + glossary of the section names) and user
  `"State:\n" + view.text + "\n\nQuestion: " + question + "\nOptions:\n" + "A: ...\nB: ..." +
  "\nAnswer with one letter: A, B"`. Everything up to and including `"\n\nQuestion: "` is the
  per-view prefix (prefix sharing). `text.render` (training items) and `dm.base.Form.user()` of
  `types.to_typed(q)` (the jev decision model) are byte-identical (tested).
- Inclusion: options (canonical) yes / no; fwd = A Yes, B No; rev = A No, B Yes. Choice: 4 options;
  fwd = canonical order, rev = reversed. Neutral letters, the full definition in the option text,
  only fixed vocabulary or identity-attested names in options. A deny-list of tool, dataset and
  verdict words is audited on every fixed text.
- Readout (`types.answer_of`): per order, the decision model's T = 1 distribution over the OFFERED
  options in canonical order; `p_bar = types.pbar` = geometric mean over orders x members,
  renormalised. The MQ layout answers one form (`fwd`) and its raw readout is the pbar over members.
- Decision models (`habitus.dm`): HB-Z the stock Qwen3.5-4B; HB one LoRA (r 16, alpha 32, dropout .05
  on q/k/v/o/gate/up/down) on Qwen3.5-4B; HB.head an fp32 letter head; HB.08b Qwen3.5-0.8B with HB's
  recipe; HB-L / HB-L.s0 / HB-L.mq / HB-L.zs the Laya family (section 18). QV / QT / QA stay the stock
  model's E2 values.

## 10. Training

- Items (`train.items.export_fold`): per fold, training host, pre-deployment day c (not the inner
  day), reference R in {k0, one other day, two other days, allc-pre minus c} (never containing c or
  the inner day); select.head(c, R) plus a 5 % sample outside the head; per (view, bucket) inclusion
  items for present buckets plus an equal number of absent candidates; choice items per view.
  Targets: inclusion = share of the signature's member nodes with the bucket present; choice = mean
  one-hot over applicable members. Exact-duplicate prompts merged (targets weight-averaged). One
  order (fwd / rev) drawn per item.
- Composition (`train.items.sample_fold`): process 55 % / file 15 % / netflow 30 % (within available
  families); references allc-pre 30 / k2 30 / k1 20 / k0 20; present:absent ~1:1; rare buckets
  oversampled <= 4x with importance weights restoring the population; hosts balanced. Augmentations:
  A1 k-aug (above), A2 order, A3 20 % of |R| >= 2 items use a random subset of R, A4 5 % attested-text
  injection twins (target unchanged), no rename augmentation.
- Loss: `L = sum_i w_i * (-sum_o t_io log softmax(z_i)_o) / sum_i w_i` over the offered letters,
  w = importance x host balance (`train.calibrate.soft_ce`).
- Recipe (`configs/train/jev-lora.yaml`): 1 epoch, AdamW lr 1e-4, cosine, 3 % warmup, effective batch
  16, length-bucketed batches, grad clip 1.0, gradient checkpointing, bf16. N_fold = min(1.2e5, 10 h x
  measured rate), floor 6e4. The realised sizes of the frozen runs (run_facts): optc_h501 fold 18,526
  train + 5,236 inner items; CADETS 35,525; THEIA 35,458; ATLAS 39,532. The exact N per fold of the
  retraining is TBD.
- Temperatures: one T per readout form (inclusion-2, choice-4), NLL on inner-day items, clamped
  [0.5, 3]; used only for calibration reports. Every served answer is at T = 1.

## 11. Statistic, null and s_hat

- Inclusion family f, node v: `x_f(v) = max_{e in S_f(v)} -ln p_bar(yes | view(sig v), e)`; 0 if S_f(v)
  is empty; n/a if the gate fails. P1: `x = -ln PIT`, PIT = sum of p_bar(o) over options with
  p_bar(o) <= p_bar(observed). N2: the same PIT per owner bucket, x = max over owner buckets. A
  question the decision model refuses (laya over its length budget) is n/a, never cut.
- Strata s = (node type, family, identity-attested flag). The attested split is merged when either
  side has < NULL_MIN_STRATUM = 500 null node-days. `p_f(v) = (1 + #{u in Null_s: x_u >= x_v}) /
  (1 + |Null_s|)` up to the 95th null percentile; beyond it an exponential (GPD shape 0) fitted to the
  top 5 % when |Null_s| >= TAIL_FIT_MIN = 200. Floor p >= 1 / (1 + N_min), N_min = smallest admitted
  stratum of that node type. `z_f = -ln p_f`. Null_s = the host's pre-deployment null node-days at the
  same k plan, EXCLUDING day c for a unit on day c (cross-fitting). The merge decisions and N_min are
  those of the WHOLE null, shared by null and test units. Strata never use today's |S|.
- `E(v) = max_f z_f(v)` over admitted families; `E_mono(v)` = the same over MONOTONE families (P1 out).
- `s_hat(v) = Q_base,t(F_E,t(E(v)))`: the cross-fitted null CDF of E for node type t composed with the
  quantile function of base0 = max(qv, qt) on the same null nodes. Above the null maximum:
  `s_hat = b_max + (1 - b_max)(1 - exp(-(E - E_max)))`. The map is fitted on E itself, never assumed.

## 12. Composite and decision layer

- `base(v) = max(qv(sig), qt(sig), s_hat(v))` (HB0: max(qv, qt)). qv / qt / qa are E2's stock values
  from the shipped score files (E2_k2; E2_kall as allc-pre on every host but CADETS_E3).
- `a = 0` in the hard tier (HARD bits: ran_file_written_today, wrote_file_run_later, image_deleted,
  run_after_written_today) or when QA was not asked; else `a = qa` if LINEAGE, else `a = C_K * qa`
  (C_K = 0.5). Netflows follow the same rule. `s = base * (1 - LAMBDA * a)`, LAMBDA = 0.5. Hence
  s >= 0.75 base unless LINEAGE, s >= 0.5 base always. This is the HB0 lineage-split acquittal; it is
  fixed a priori, never learned or label-tuned.
- Node score = max over test days (universe-aligned; GT without a card is a miss); non-head nodes =
  the rarity fallback `-1 + 0.5 / (1 + n_s)`. Entity score = max over the kappa pool.
- Bars `tau_B = bar(null composite pool, n_null_days, B)` for B in BARS = {0, 1, 20}
  (`decide.null.bar`, the label-free copy). B = 0: alerts {s >= tau_0}. B = 20: strong {s >= tau_1}
  (uncapped) + weak = top WEAK_CAP = 20 per test day by margin (s - tau_20) from [tau_20, tau_1) (ties
  by node id). The weak bound is a code check, not a result; the cap is a contract mechanism, not a
  detection gain.
- Reported with the bars: uncapped rows, cap loss, TP-vs-alerts/day 0..100, fixed alerts/day rows
  1 / 5 / 20 / 50 (`decide.habit.bars`). The best-MCC confusion tables use the oracle threshold and
  no bar (an upper bound, not an operating point).

## 13. Score files

`<runs>/habit/scores/<ds>/<level>[.<arm>]_<kplan>/<day>.json.gz` (`types.level_dir`), `types.ScoreDay`:
`{"meta", "head", "routes": {sig: {qv, qt, qa, hard, cov_min}}, "nodes": {column: [...]},
"fallback": {sig: float}}`. Rows = EVERY node whose signature is in the head (all members).
Columns: id, sig, s, base, a, tier, qv, qt, s_hat, E, E_mono (+ z.<F>, x.<F>; None = n/a). Meta
(REQUIRED_META): ds, day, level, kplan, role (test | null), ref, caps, head_rule (Q0), spec (+ k_mode,
arm, source tag and spec, written). Test and null roles of one (level, kplan) live side by side; a
null file's ref must be the plan's. Nulls (FamilyNull, EMap, fit record, per-day x) live under
`<runs>/habit/nulls/<ds>/<level dir>/`.

## 18. Arms and the arm comparison

Arms (`constants.CLONE_ARMS`, `constants.ARM_DM`; score directories `<level>.<arm>_<kplan>`, own
nulls; decision-model config in brackets):
- HB-Z [jev-qwen3.5-4b-habit] the stock Qwen3.5-4B; HB [jev-qwen3.5-4b-lora] the primary trained
  decider.
- HB-L [laya-typed-ens3]: the Laya family (vendored Laya 0.3.5 DecisionModel, ModernBERT-large,
  init laya typed-decisions, full fine-tune, soft-CE only), layout L-a (one row per question and
  order, definition-only options, qtype choice), trained on HB's very sample, 3 seeds; the
  pre-declared output is the geometric mean over seeds x orders. HB-L.s0 [laya-typed-s0] (seed 0
  alone), HB-L.zs [laya-typed-zs] (the Laya init zero-shot, layout L-a).
- HB-L.mq [laya-typed-mq]: encode the view once, one block per question, exact question isolation and
  option-order invariance (one pass).
- HB.head [jev-qwen3.5-4b-head] an fp32 letter head; HB.08b [jev-qwen3.5-0.8b-lora] Qwen3.5-0.8B.
- Readout precision: HB / HB-Z read the letter logits off the bf16 lm_head; HB.head computes them in
  fp32 (its step 0 equals the fp32 letter readout, |dp| < 1e-5).
- Length budget: HB-L max_len 2048 / head budget 256 ModernBERT tokens. A question whose question +
  options exceed the head budget is refused, never cut, and is n/a in that arm's score files.
- HB-L.full (every exported item) appears only as frozen selection rows: no arm or config exists for
  it here (unsupported).

Selection items (role "select", `train.items.export_select`, items/<fold>.select/): code-computed habit
items of the fold's HELD-OUT dev host on its selection days only (`null2.selection_days`: optc_h501
09-19 / 09-21, the parity half; CADETS_E3 / THEIA_E3 / ATLASV2_EDR their pre-deployment window), under
the null references of the k plans k0 / k2 / allc-pre on those days, head only, present buckets +
absent candidates 1:1, one order per item, merged per (day, reference) only; the HabitQuestion of
every item is stored and must re-render to the item's prompt. Never a training host, test day or
fold "all" (`assert_selectable`); never trained on.

Tail selector (frozen; NOT recomputable here): the tail of M1 / M2 = present inclusion items of the
eligible questions whose pooled HB-C count back-off p_yes (fold training hosts only) is in the lowest
10 % of that p over the eligible present inclusion items. HB-C is a rule comparator and is excluded
from this repository, so M1 / M2 are frozen rows (`results/not_included.md`).

Metrics (label-free, frozen rows): M1 tail log score at T = 1, M2 rare-present AUC (tail vs absent),
M3 NLL at T = 1 and at T_fit, M4 ECE15 at T_fit, M5 tail coverage, M6 fwd / rev order sensitivity,
M7 within-view AUC (present vs absent buckets of one view and family), M8 KL(target || p) per family.
Speed: S1 train items/s, S2 training GPU-h per fold, S3 questions/s of the selection questions, S4 ms
per head node, S5 streaming latency p50 / p95 per view; physical GPU 0 or 1 only.

The pre-registered selection rules (R-L1 / R-L2 / R-L3, R-J, R-J08, X2) compared every arm with HB and
HB-C; their frozen outcomes are reported in the paper. None of them is re-run here.
