"""Habit questions (HABITUS+H): self-supervised questions about ONE habit view whose answers the
host's own attack-free pre-deployment days give by code.

    constants  fixed a-priori constants (mirrored by configs/habit.yaml)
    types      the data contract + the decision-model bridge (to_typed, answer_of, DMAnswerSource)
    vocab      closed vocabularies, shapes, the catalog role table, the attested-name rule
    attest     attestation tiers NONE < PATTERN < IDENTITY < LINEAGE
    ifc        information-flow labels of view fields; a view never carries its own answer
    profile2   exact-identity habit indices over the reference days
    views      V_beh / V_par / V_file / V_net / V_own, pure functions of (card invariant, reference)
    targets    per-node observed buckets, candidates, the family-availability gate
    text       question bank, prompt text and the one renderer
    head       Q0 = the routed E2 head

Any DecisionModel (habitus.dm) answers these questions through types.DMAnswerSource; the decision
stage is habitus.decide.habit. Nothing in this package reads a label.
"""
