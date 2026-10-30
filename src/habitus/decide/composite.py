"""Per-signature composites of a score record (fixed a priori, label-free).

    compose(rec)      max(qv, qt): the E1 composite and the E2 base (re-exported from
                      habitus.questions.routed; raises on a non-finite input)
    compose_e2(rec)   the E2 bounded-acquittal composite
                          base = max(qv, qt)
                          s = base                       if rec["hard"] or qa is None
                          s = base * (1 - LAMBDA * qa)   otherwise,  LAMBDA = 0.5
                      so an acquittal can at most halve a score and never touches the hard tier
                      (write->exec, deleted image).
    value(rec, field) the value node_scores reads. A record is E2's iff it has a "qa" key (E1
                      records never do), so every E1 value is max(qv, qt) computed at read time;
                      E1 records carry no base / composite / hard.

E2 fields (FIELDS_E2): composite, base, qv, qt and "1-qa" (= 1 - qa, 0 when QA was not asked: the
acquittal question alone, read in the suspicion direction so that "higher = more suspicious" and
node_scores' [0, 1] rule hold).
"""

import math

from habitus.questions.routed import compose  # noqa: F401  (re-exported: max(qv, qt))

FIELDS_E2 = ("composite", "base", "qv", "qt", "1-qa")
LAMBDA = 0.5                             # E2 acquittal weight, fixed a priori


def compose_e2(rec):
    """max(qv, qt), times (1 - LAMBDA * qa) unless the node is in the hard tier or QA was not
    asked (qa None). Non-finite inputs raise (as compose)."""
    base = compose(rec)
    qa = rec.get("qa")
    if rec.get("hard") or qa is None:
        return base
    qa = float(qa)
    if not (math.isfinite(qa) and 0.0 <= qa <= 1.0):
        raise ValueError(f"qa {qa!r} outside [0, 1]")
    return base * (1.0 - LAMBDA * qa)


def value_e2(rec, field="composite"):
    if field == "composite":
        return float(compose_e2(rec))
    if field == "base":
        return float(compose(rec))
    if field == "1-qa":
        return 1.0 - float(rec.get("qa") or 0.0)
    return float(rec[field])


def value(rec, field="composite"):
    if not isinstance(rec, dict):
        return float(rec)
    if "qa" in rec:                         # an E2 record (E1 records have no "qa")
        return value_e2(rec, field)
    return float(compose(rec)) if field == "composite" else float(rec[field])
