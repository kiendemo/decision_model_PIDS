"""The habit channel's decision stage (HABITUS+H) over any DecisionModel.

    surprise   x_f per node and the cross-fitted per-family null (FamilyNull)
    null2      k plans, the s_hat map (EMap), bars, null persistence
    score      the lineage-split composite and the habit score files
    retrofit   HB0: the decision layer over the frozen routed E2 score files (no model call)
    bars       clean-day bars and the two-tier per-day cap (FP-budget rows)
    channel    score_level: head -> views -> questions -> DecisionModel answers -> score files

Every output goes under runs/habit. Nothing here reads a label or imports habitus.eval.
"""
