# The laya decision models

The `laya` family is the encoder backend of the HABITUS decision-model framework. It answers the same
typed questions as the `jev` backbone (`habitus.dm.base.TypedQuestion`, at most 4 options). The model
is a bidirectional encoder (ModernBERT-large) with one `[MASK]` scorer per option. It makes one forward
pass per row and returns a T = 1 softmax over the offered options. Nothing is generated.

| Module | Class | Layout | Arms |
|---|---|---|---|
| `habitus.dm.laya` | `LayaDecisionModel` | `la`: one row per (question, form) | HB-L.zs, HB-L.s0, HB-L |
| `habitus.dm.laya_mq` | `LayaMQDecisionModel` | `mq`: the state encoded once, one block per question | HB-L.mq |
| `habitus.train.laya` | `train_laya(layout="la" \| "mq")` | trainer | HB-L.s0, HB-L, HB-L.mq |
| `habitus.train.laya_mq` | `train_mq` | MQ trainer (`train_laya` dispatches to it) | HB-L.mq |

Configs: `configs/dm/laya-typed-{zs,s0,ens3,mq}.yaml` and `configs/train/laya{,-mq}.yaml`.
`tests/test_dm_mirror_laya.py` asserts that the YAML equals the code constants.

## Status

- **Habit questions (HABITUS+H).** HB-L.s0, HB-L and HB-L.mq are reported as FROZEN numbers only.
  Their run directories, ensemble records and answer caches were lost with the former data disk.
  Retraining (GPU, a later phase) gives different numbers. HB-L.zs (the zero-shot Laya init) has
  frozen selection metrics only; its config is supported.
- **Routed QV / QT / QA questions (HABITUS main).** laya on routed questions is **SUPPORTED, NOT
  EVALUATED**. No published number comes from it. The questions are typed (`qv` as Laya type `score`,
  `qt` / `qa` as `noul`) and the backend answers them. E2 states may exceed the 2048-token row budget,
  however. Such rows are refused, never cut, and `decide.routed` raises on any refusal. Any number
  from it would be new and unvalidated, and would need a GPU.

## The vendored model

`DecisionModelHBL` is a modified copy of `laya.common.DecisionModel` from Laya 0.3.5 (Convai
Innovations, Apache License 2.0). The change list sits in the header of `habitus/dm/laya.py`:

1. `forward` returns the masked marker logits only. `act_head` stays in the state dict, so checkpoint
   loads remain strict, but it is never run.
2. `forward` takes optional `position_ids`, `enc_masks` and `head_mask` inputs for the MQ layout.
   `None` reproduces Laya's own call bit for bit.
3. `qtype` may be given per position (`[B, L]`) as well as per row (`[B]`).
4. The temperature buffer is reset to 1 on load, and the checkpoint's temperature tables are unused.
   Every answer is read at T = 1.
5. Nothing is ever truncated. A row over budget is refused (see below).

The parameter names are Laya's, so `<laya_hub>/typed-decisions` (convaiinnovations/laya@1c5edc17) and
every trained run load with `strict=True`. The provenance tests (`HABITUS_PROVENANCE=1`, with
`paths.laya_src` pointing at the Laya 0.3.5 source) check two things. The vendored model equals
`laya.common.DecisionModel` bit for bit, on a tiny random model and on the real weights. The L-a row
equals `laya.common.build_sequence` token for token whenever Laya's 48-token option cut does not
apply.

The checkpoint location is `settings.paths.laya_hub` (`typed-decisions/` for the init, the hub root
for `base`): a plain snapshot of the Hugging Face repository convaiinnovations/laya at revision
1c5edc17a7acd8701df6fc341c0d179f1c62c982, for example
`hf download convaiinnovations/laya --revision 1c5edc17a7acd8701df6fc341c0d179f1c62c982 --local-dir
models/laya_hub` (network access; `HF_HUB_OFFLINE=0` for that one command). A Hugging Face cache entry
that holds refs only (no snapshot) is not a weight source.

## Layout L-a

Each form of each question becomes one row:

    [CLS] tok("<qtype> question: " + question) [SEP] ( [MASK] tok(" " + option_i) )_i [SEP] tok(state) [SEP]

- The pieces are exactly `(form.state, form.question, form.options)`. `form.options` is in the form's
  LETTER order. These are the same three pieces the jev backbone reads in `Form.user()`.
- `qtype` is `TypedQuestion.qtype`: `choice` for every habit family, `score` for `qv`, `noul` for
  `qt` and `qa`.
- The options are definition-only texts, with no `A:` or `label:` prefix.
- Every segment goes through `habitus.state.sanitize.sanitize_str`. A special id of the Laya tokenizer
  outside the structural positions raises `ValueError`. That would be a sanitiser gap, not a refusal.
- A training item becomes the row of its own rendered prompt. `parts_from_prompt` (= `split_prompt`)
  parses the item back into `(state, question, options)`. The parse is strict: the pieces must
  re-assemble to the user turn byte for byte. So a train row and a serve row of the same (question,
  form) carry the same token ids (`check_identity`, tested).

### Refusal, never truncation

Laya's `build_sequence` silently cuts every option to 48 tokens, cuts the options again when they
exceed the head budget, cuts the question, and cuts the state to the room left. HABITUS applies none
of these cuts. A P1 option A that names four usual parents reaches over 100 ModernBERT tokens, and a cut
would drop the very names the question asks about.

Instead, a row is refused when either bound is exceeded:

- the head (question + `[MASK]`-led options) is over `HEAD_MAX_LEN = 256` tokens, or
- the whole row is over `MAX_LEN = 2048` tokens.

`refused(qs)` returns `{qkey: reason}`. `decide(qs)` answers every non-refused question and leaves the
refused ones out. The habit channel turns them into n/a; the routed pipeline raises. Training raises
on an over-budget item.

### Readout and Decision

For a row, `p = softmax(z)` over its markers, at T = 1 in float64. The readout is a softmax over the
offered markers only, so there is no off-option mass and `cov = 1.0` for every form.

`Decision.layout = "la"`. `Decision.forms` lists the question's forms in order.
`letter_probs[form]` and `canon_probs[form]` hold one distribution per member, in member order,
mapped to canonical order through `Form.canon`. `questions.habit.types.answer_of` combines them as
follows:

- `raw[order]` is the member's distribution when there is one member, else the pbar over the members;
- `probs` is the pbar over orders x members, with orders outer and members inner.

### Answer cache

- Each member has its own tag: `hbl-<init label>-<12 hex of sha256(model.safetensors)>`.
- A model of several members carries the tag `hbl-<label | mix>-<12 hex>`, where the hex is the
  sha256 of the members' weight hashes, newline-joined.
- The key is the sha1 of the row's int32 little-endian token ids.
- Records sit in `habitus.dm.cache.AnswerCache` with the fields `{k, z, lse, n, cov}`: `z` is the
  marker logits at 5 dp, `lse` their logsumexp, `n` the number of markers, and `cov` is 1.0. Only `z`
  is read back.
- The first record of a key wins.
- An explicit `cache_tag` (for example a scratch smoke namespace) replaces the weight-derived tags:
  the tag itself for one member, `<cache_tag>.m<i>` for member i of several.
- Models load on the first cache miss only. `cache_only=True` (the config default) raises `CacheMiss`
  before any model is loaded.

## Seed ensembles

A model is either one checkpoint or a list of trained run directories. HB-L is the pre-declared
3-seed ensemble (seeds 0, 1, 2); HB-L.s0 is seed 0 alone. All members must share the tokenizer and the
recipe budgets. A trained member's recipe must use Laya type `choice`: the `qt2` ablation was
dropped. With `fold=` given, every run record must name that fold. The registry checks the fold for
the configured arms.

The trainer fits per-form temperatures on the inner items, for the seeds and for the ensemble (the
geometric mean of the seeds' distributions, `geo_logits`). The fit is clamped to [0.5, 3], and the
unclamped optimum is recorded as `raw_T`. The temperatures are stored in `habitus.json` and in
`<base>.ensemble.json`, and they serve calibration reports only. Serving is at T = 1.

## Layout MQ

The state is encoded once. Every question is one block appended to it:

    [CLS] tok(state) [SEP] | B_1 | ... | B_Q      B_j = tok("<qtype> question: " + q_j) [SEP] ( [MASK] tok(" " + o_jk) )_k

The rows follow four rules:

- **Positions.** The question part of every block starts at the state length S. Every option of a
  block starts right after the question part, so the options of a block share their positions.
- **Attention.** A state token sees only the state. A question token sees the state and its whole
  block. An option token sees the state, its block's question part and itself.
- **Local layers.** They also need `|pos_q - pos_k| <= sliding_window`, measured on the logical
  position ids.
- **Encoder and head.** The masks reach ModernBERT as the transformers 5.14.1 dict of boolean 4-D
  masks, so the encoder must run sdpa. The two Laya head layers get the same block mask, one row at a
  time (`head = "row"` only).

These rules give two guarantees in exact arithmetic, tested in fp32:

- **(I1) question isolation.** An answer does not depend on the other questions in the row. So
  packing a state's blocks into several rows is exact, and the answer key is
  `sha1("mq1", state ids, block)`.
- **(I2) option-order invariance.** Permuting a block's options permutes its logits. One pass
  replaces L-a's two forms.

The tests measure `|dp| <= 1e-5` in fp32. In bf16, the option-order gap is about **2.3e-2**, with 1.5%
argmax flips, against **2.5e-5** in fp32. The reported MQ rows were served in bf16. MQ is not equal to
L-a's single-question output: MQ puts the state first, L-a puts the head first.

Serving groups the questions by `TypedQuestion.group`, and within a group by their state. Each
question is one block of `forms[0]`, laid out in CANONICAL option order through `Form.canon`. The
Decision has the single form `fwd` (`layout = "mq"`, `cov = 1.0`) and one distribution per member.

The MQ raw readout follows `encoder_mq.py:724-726`: `answer_of` gives
`probs = raw["fwd"] = pbar over the members`. The pbar is renormalised even when there is a single
member.

A block is refused exactly when the L-a row of the same form is refused. The MQ tag is
`hbl-mq-<label>-<12 hex>`, which is never an L-a tag. The layout version `mq1` is part of every key.

## Training

`train_laya(items, inner, out_dir, *, init="typed-decisions", fold, device="cpu", layout="la" | "mq",
seeds=(0, 1, 2), recipe=None, name=None)` trains on the label-free habit items:

- `items` must be role `train` and `inner` must be role `inner`.
- Every item must come from a training host of the fold.
- The loss is the weighted soft cross-entropy over the markers (`train.calibrate.soft_ce`).

`RECIPE_L` is a full fine-tune for 1 epoch:

- AdamW, encoder lr 2.5e-5, head lr 1e-4, weight decay 0.01;
- 3% warm-up, then cosine to 1e-6;
- effective batch 64, as micro-batches of 16 rows within 16384 padded tokens;
- gradient clip 1.0, bf16 autocast on CUDA, gradient checkpointing.

`RECIPE_MQ` is the same recipe with gradient checkpointing off (it changes no number), plus
`layout`, `row_tokens` and `head`. MQ optimiser batches count items (blocks), not rows.

`out_dir` is the directory of the fold's runs (`runs/adapters/<fold>/hbl`). `name` is the run base;
it defaults to `HB-L` (`HB-L.mq` for layout MQ), matching the paths in `configs/dm/laya-typed-*.yaml`.
The outputs keep the original formats:

- one run directory per seed, `<out_dir>/<name>-s<seed>/`, holding:
  - `model.safetensors` (fp32);
  - `encoder/config.json` and `tokenizer/`, copied from the init;
  - `rl_agent_config.json`, with the budgets, temperature `[1, 1, 1]` and the refitted
    `temperature_by_options` buckets `choice:2` / `choice:3-5`;
  - `habitus.json`, the run record (fold, init, layout, recipe, seed, temperatures, `raw_T`,
    `clamped`, `laya_buckets`, item and code hashes, spec, `model_sha256`, inner report, and more);
- with 2 or more seeds and inner items, the ensemble record `<out_dir>/<name>.ensemble.json`.

`recipe` overrides the recipe keys. It is either a flat mapping or a whole `configs/train/*.yaml`
mapping, whose `recipe` entry is used; an unknown key raises.

A run directory is itself a Laya checkpoint. `laya.resolve_init` refuses an MQ run, so an MQ model is
never read as L-a. `laya_mq.resolve_mq` accepts both kinds. Training writes only outside the
configured read-only inputs (`config.check_writable`).
