"""Compiled host-reference states (no imports here; import the submodules).

    select     fixed a-priori head rule per day, rarity fallback value, limit_caps
    sanitize   control-token neutralisation of every string leaf (pinned tokenizer check)
    profile    k-day host reference profile (observability, kappa / name days, ops maxima, lineage)
    refindex   TF-IDF nearest reference cards (built only on a precedent-cache miss)
    precedent  persistent nearest() answers: read-only shipped base + writable runs/ overlay
    sections   entity, hard_bits, reference, deviation(2), evidence(_nf), precedent(2)
    builder    StateBuilderE2 (MAIN), serialize, state_hash, hard_tier
    ladder     E1 StateBuilder L0..L4 (state-ablation table only)
"""
