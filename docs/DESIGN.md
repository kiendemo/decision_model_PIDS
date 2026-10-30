# HABITUS design: compiled host-reference states and routed typed questions

This document describes the main HABITUS system (the "E2" routed configuration) and the E1 state
ladder used only for the state-ablation table. Code: `habitus.state.*`, `habitus.questions.routed`,
`habitus.decide.routed`. Every rule below is implemented byte for byte as in the code that produced
the shipped answer cache; the frozen texts are hashed in `habitus.questions.routed.ROUTED_TEXT_SHA256`.

## 0. What the system is

HABITUS is a decision-model pipeline. A DecisionModel answers a TYPED question (at most 4 options)
about a COMPILED host-reference state. It makes one forward pass per presented form and returns a
T = 1 distribution over the offered options. A question family then combines the forms.

The `jev` backbone of the main system is a stock, instruction-tuned decoder LLM (Qwen3.5-4B at
revision 851bf6e8), used as follows:

- It is given a chat-formatted prompt: the system turn is `SYSTEM_E2`, and the user turn is the
  state plus one closed question.
- It is read out by the logits of the letter tokens at the answer position.
- Nothing is generated and no free text is parsed.

The `laya` family is an encoder with a [MASK] scorer per option (supported, not evaluated on the
routed questions).

The pipeline also contains fixed, a-priori gating and fallback rules. None of them is learned or
label-tuned:

- the head selection rule (section 6);
- the rarity fallback for non-head signatures, `-1 + 0.5 / (1 + n_s)` (section 6);
- the hard-tier acquittal exemption (section 5);
- in the habit channel, the HB0 lineage-split acquittal over attestation tiers (see
  `docs/HABIT_CHANNEL.md`).

What LLM-based provenance detectors hand a generative model at alert time (the host's benign
profile, the deviation from it, bounded graph evidence, benign precedent) is compiled into the
*state* instead, so the decision model never has to fetch, reason or write.

**Design disclosure.** The E2 routing below was designed from an exploratory, label-reading
diagnosis on THEIA_E3 (section 1). All five hosts of this repository (CADETS_E3, THEIA_E3,
CLEARSCOPE_E3, optc_h501, ATLASV2_EDR) are therefore design-informed development hosts; no result
on them is held-out evidence for the design.

## 1. What the diagnosis showed (mechanisms, not numbers to fit)

The E1 state (section 8) gave every question every section and printed neighbour names. A
label-reading diagnosis on THEIA_E3 found four mechanisms. Every E2 choice below is fixed by one of
these mechanisms, never by a development-host metric.

- **M1. Evidence lines carry neighbour NAMES.** Names with alarm words (browser Safe-Browsing lists
  "goog-phish", "test-malware", user files "Missile-Defense.jpg", "eraseme") pushed benign processes
  to the top: every benign node that crossed 0.62 at L3 had such a word, and 84% had no neighbour new
  to the reference. This is the name-driven judgement failure documented for generative LLM judges,
  and an attacker lever.
- **M2.** Line counts scale with the number of nodes sharing a signature ("x32+", "more: 48"), so
  one 126-node browser signature dominates.
- **M3.** Both questions read every section, so a section meant to accuse (behaviour) also moves the
  verdict question, and a section meant to acquit (precedent) also moves the tradecraft question.
- **M4.** A netflow's name key is (endpoint class, port class); (external, well-known) is always
  seen, so a C2 flow to a never-seen external IP renders as "name seen, nothing new"; empty deviation
  lists read as "routine".

## 2. Routing: each question reads only the sections it names (fixes M3)

| question | role | reads | at k = 0 |
|---|---|---|---|
| QV (E1 text, forms fwd / rev) | verdict | entity, hard_bits, reference, deviation2 | entity, hard_bits, reference stub |
| QT (E1 text, forms pos / neg) | ACCUSE only | entity, hard_bits, deviation2, evidence_nf | entity, hard_bits, evidence_nf |
| QA (forms pos / neg) | ACQUIT only | entity, reference, precedent2 | not asked |

`SYSTEM_E2` is E1's `SYSTEM` verbatim except that its section list describes `evidence` as "counts
of its direct relations in the provenance graph that day, by class, without names". Sections a
question does not read are absent from that question's state, so the state string differs per
question. Section order inside a state is E1's order: entity, hard_bits, reference, deviation,
evidence, precedent (E2 keeps E1's section names: deviation = deviation2, evidence = evidence_nf,
precedent = precedent2).

QA texts (verbatim):

- pos: "Is this {type} explained by one of the precedent entities seen on this computer's
  attack-free reference days — the same kind of program, file or connection doing the same kind of
  thing?" Options: A "Yes, a precedent explains it." B "No precedent explains it."
- neg: "Is this {type} unlike every precedent entity seen on this computer's attack-free reference
  days?" Options: A "Yes, no precedent explains it." B "No, a precedent explains it."

`qa = 0.5 * (P(A | pos) + P(B | neg))`. If precedent2 is "none", QA is not asked and qa = 0 in the
composite. At k = 0 there is no reference, so a signature has 4 forms (QV, QT); at k >= 1 it has 6.

## 3. evidence_nf: name-free, novelty-first, node-count-independent (fixes M1, M2)

Built from the same links as E1 evidence, but:

- **Representative nodes:** the 8 smallest node ids carrying the signature on the day (all if
  fewer). Counts are per representative node and reported as "in a/b nodes", where b is the number
  of representatives (fixes M2).
- **Neighbour class, never a name:** process -> `process:<image_dir>`; file ->
  `file:<dir>/<extclass>`; netflow -> `net:<endpoint>/<port class>`. Class words come from closed
  vocabularies; anything else is clamped (`other` / `unknown`). No file names, process names, paths,
  IPs or command lines.
- **Novelty:** at k >= 1 a neighbour is "new" when its identity key (kappa) is unseen in the
  reference days. At k = 0 there is no novelty.
- **Rendering:** `{"new": {"<relation> <class>": "in a/b nodes", ...}, "known": "<bin>",
  "representatives": b}` with at most 8 entries, ordered by relation priority [ran_file, wrote,
  spawned, spawned_by, connected, run_by, written_by, opened_by], then a descending, then text.
  "known" is the count bin of the known neighbours summed over representatives (distinct
  (representative, neighbour) pairs). At k = 0 the same structure is rendered under `"all"`, with
  known "n/a".
- **`connected` only towards a netflow (D2):** a link-code-4 link from a process to a FILE (CADETS
  records SENDTO / SENDMSG on file objects with that code) is dropped before anything is counted, so
  it is neither an entry, nor a known neighbour. E1 evidence still shows it.
- **Extension classes:** executable {exe, dll, sys, scr, msi, com, elf, bin, run, so, dylib, ko} and
  files without extension that were executed; script {ps1, bat, cmd, vbs, js, hta, sh, py, pl, rb,
  php, jar, lnk}; library {lib, a, o, pyc, class}; archive {zip, rar, 7z, gz, tgz, tar, bz2, xz, cab,
  iso, img}; document {doc, docx, xls, xlsx, ppt, pptx, pdf, rtf, txt, csv, html, htm, xml, json,
  eml, msg, jpg, jpeg, png, gif}; data = any other extension; none = no extension and not executed.

## 4. deviation2 and precedent2 (fixes M4)

- **deviation2** = `"identity_unseen": "yes"` (first) when the kappa is unseen in the reference;
  for netflows `"remote_prefix_unseen_levels": "x/5"` from the profile's prefix chain (/8, /16, /24,
  /32, ip|port; "n/a" for private or unusable addresses); then E1's deviation unchanged.
- **D1.** E2 keeps E1's netflow name key (endpoint, port class) for every name_* statistic and E1's
  deviation lists. (A name key extended by the longest seen prefix level was dropped before any E2
  metric was read: every reference flow sits at level 5, so "name seen" would degenerate to
  "identity seen".) Netflow novelty reaches E2 through `identity_unseen`,
  `remote_prefix_unseen_levels` and precedent2's `prefix_match`. An E2 state's `reference` section is
  E1's.
- **precedent2** = E1's precedent (the same nearest reference cards, seen days and top ops), except
  that a netflow precedent identity is rendered as `net:<endpoint>/<port class> prefix_match:<0-5>`
  (the deepest common prefix level with the node), never its address.

## 5. Bounded-acquittal composite (pre-registered)

`HARD = {ran_file_written_today, wrote_file_run_later, image_deleted, run_after_written_today}`
(hard_bits keys; any "yes" in the QV state puts the node in the hard tier). `base = max(qv, qt)`.

- `s = base` when the node is in the hard tier or qa is not available;
- otherwise `s = base * (1 - 0.5 * qa)`.

λ = 0.5 is fixed a priori: an acquittal can at most halve a score and can never touch the hard tier.
Score records carry `qv, qt, qa, hard, cov_min, base, composite`.

## 6. Candidate head and fallback (fixed a priori, label-free)

Per day d, only a **head** of the day's card signatures is asked; every other signature gets a
fallback rank strictly below every decision score, so the whole test universe is still ranked.

- `n_s` = number of nodes carrying signature s on d (day-local, label-free).
- **process:** every signature (cap 4000; beyond the cap, n_s ascending, then signature).
- **netflow:** cap 6000; (k >= 1: kappa unseen in the reference first), then n_s ascending, then
  signature.
- **file:** cap 2000; (k >= 1: kappa unseen first), then "changed on disk" first (an op key starting
  with write/create/rename/delete/exec, or any flag), then n_s ascending, then signature.
- k = 0: no reference, so the unseen-first key is dropped; the order is otherwise the same.
- **fallback:** `-1 + 0.5 * (1 / (1 + n_s))` for non-head signatures: rarer is higher, always in
  (-1, -0.5], below every decision score in [0, 1].
- **node score:** the max over the test days a node appears on of its signature's value that day
  (head record or fallback).

The same head rule picks the head of a clean day when its null scores are computed, with the
reference of the test day it stands in for. A head is never drawn on one of its own reference days.

## 7. Prompt, readout and answer cache (decision-model contract)

- **User turn** (`Form.user()`, identical to `routed.user_prompt`): `"State:\n" + state + "\n\nQuestion: "
  + question + "\nOptions:\n" + "\n".join(f"{L}: {text}") + "\nAnswer with one letter: " +
  ", ".join(letters)`.
- **System turn:** `SYSTEM_E2.format(type, os)` (E1 ladder: `SYSTEM`), with type and os read from
  the state's entity.
- **Rendering (jev):** `apply_chat_template([system, user], tokenize=False,
  add_generation_prompt=True, enable_thinking=False)`; the text must end with `"</think>\n\n"`.
- **Answer key:** sha1 of the full rendered prompt (UTF-8). One cache record per key holds the
  logits of tokens A..D at the answer position (always 4 values), the full-vocabulary logsumexp, the
  number of offered letters and the coverage `cov = sum exp(z[:n] - lse)` (6 dp). The first record of
  a key wins.
- **Readout:** per form, `softmax(z[:n])` at T = 1 over the offered letters, in float64.
  - QV: `qv` = mean over the forms of E[level] / 3 (fwd: A = level 0; rev: A = level 3);
  - QT: `qt = 0.5 * (P(A | pos) + P(B | neg))`;
  - QA: `qa = 0.5 * (P(A | pos) + P(B | neg))`.
  Probabilities are renormalised per form; forms are combined in their fixed order (qv: fwd, rev;
  qt, qa: pos, neg). `cov_min` is the minimum cached coverage over every form of the signature;
  answers with coverage < 0.5 are counted in the score-file meta, never dropped.
- **Grouping:** the QV and QT forms of a signature form one prefix group, the QA forms another
  ("qvqt"). Prefix sharing changes only speed, never a cache key.
- **Exactness:** a GPU re-run is not bit-exact (bf16 batch-makeup noise); the shipped answer cache is
  the canonical record, and every number in `results/` is recomputed from it on CPU.

## 8. States: sections, profile, sanitisation, precedent cache

Sections (one state per signature and question; a section that does not apply is omitted):

- **entity:** type, os and the card's invariant (command_line <= 200 characters, executable <= 160,
  file path <= 160 with "" -> "unknown"; netflow remote_ip, remote_port, endpoint). The only section
  an attacker chooses the text of.
- **hard_bits:** kernel-event facts, no names: image location / deletion, write-then-exec flags,
  writes to writable dirs, executable file type, endpoint and port class, plus the host's
  observability. A fact the host's collector cannot record is "n/a" (exec on the Windows
  collectors, lineage on CADETS_E3), never "no".
- **reference** (k >= 1; k = 0 -> `{"days": 0}`): how often this identity and its name were seen on
  the reference days, the identity's frequency level (rank of its node count among the identities of
  its type: top 20% high, to 50% medium, rest low; a tie block takes its lowest position), exact card
  seen, the name's usual children / write dirs / peers, parent-child pair seen, endpoint prefix-chain
  levels unseen.
- **deviation(2)**, **evidence(_nf)**, **precedent(2)**: sections 3 and 4.

The **host profile** is built from the k reference days' cards and links only (no label, no score):
k = 2 is the first two usable clean days, k = all every usable clean day, k = 0 has no profile and
reads the observability of the scored day itself.

**Sanitisation:** every string leaf and key of a state is rewritten before serialisation so that no
control-token text survives (`[SEP]` -> `(SEP)`, `<|im_end|>` -> `(im_end)`, `</think>` ->
`(/think)`), to a fixed point. The regex covers the BERT-style specials, any `<|...|>`, think /
tool-call tags, Laya's added tokens and the frozen Qwen3.5 added vocabulary. The pinned tokenizer's
added vocabulary must equal that snapshot (`pattern_status() == "live==snapshot"`); otherwise state
construction refuses to run, because the state bytes (and every cache key) would change.

**Serialisation:** compact JSON (`separators=(",", ":")`, `ensure_ascii=False`), schema key order,
no sort_keys. The builder is deterministic: the same (host, reference, day, signatures) give
byte-identical states in any process.

**Precedent cache:** the 2 nearest reference cards of a signature come from a TF-IDF index over the
reference days' distinct cards (hashed invariant and behaviour tokens, sublinear TF, L2 norm,
similarities rounded to 1e-9, so an answer depends on the query card alone). Answers are stored per
(host, sorted reference days) in `<sha1(','.join(ref))>_e2.json.gz`. The shipped cache is read-only;
answers computed on a miss go to an overlay under `runs/`. States are byte-identical with or without
the cache.

## 9. The E1 ladder (state-ablation table only)

E1 builds ONE cumulative state per signature and asks QV and QT with `SYSTEM`:

| level | sections |
|---|---|
| L0 | entity |
| L1 | + hard_bits |
| L2 | + reference, deviation |
| L3 | + evidence (named neighbour lines, <= 10 lines over the signature's <= 200 smallest node ids) |
| L4 | + precedent |

At k = 0 only L0, L1 and L3 are run (`K0_LEVELS`); the E1 primary levels are L3 at k = 0 and L4 at
k >= 1. E1 records carry `qv, qt, cov_min`; their composite is `max(qv, qt)`. E1 states are
deduplicated by the hash of the serialised state, E2 signatures by the joined per-question states.
