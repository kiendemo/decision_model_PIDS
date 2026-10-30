"""Decision stage: DecisionModel answers -> per-day score files -> node scores.

    scorefile   score-file paths / IO / meta, load_by_day, day_arrays, day_values, node_scores
    composite   compose (max(qv, qt)), compose_e2 (bounded acquittal), value / value_e2, LAMBDA
    inventory   ScoreJob, the shipped score-file inventory, and jobs() from the null rules
    routed      per-day E2 / E1 scoring: head -> states -> typed questions -> DM -> records -> file
    null        clean-day LODO plan, null pools, lodo_scores and the label-free bar()
    habit/      the habit channel

Nothing in this package reads labels. This file imports nothing, so importing a submodule only
pulls in what that submodule needs.
"""
