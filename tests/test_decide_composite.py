"""decide.composite: the E1 composite max(qv, qt) and the E2 bounded acquittal
(port of test_kcs1.test_e2_composite_bounded_acquittal and the compose part of test_combine_and_compose)."""

import numpy as np
import pytest

from habitus.decide import composite as C


def test_compose_is_the_routed_compose_and_raises_on_nan():
    from habitus.questions import routed as Q
    assert C.compose is Q.compose
    assert C.compose({"qv": 0.2, "qt": 0.75}) == 0.75 and C.compose({"qv": 0.9, "qt": 0.1}) == 0.9
    for rec in ({"qv": 0.2, "qt": float("nan")}, {"qv": float("nan"), "qt": 0.2}, {"qv": float("inf"), "qt": 0}):
        with pytest.raises(ValueError):                                      # max() would hide a NaN qt
            C.compose(rec)
    assert C.value({"qv": 0.3, "qt": 0.6}, "qv") == 0.3
    assert C.LAMBDA == 0.5 and C.FIELDS_E2 == ("composite", "base", "qv", "qt", "1-qa")


def test_e2_composite_bounded_acquittal():
    rng = np.random.default_rng(3)
    for _ in range(2000):
        qv, qt, qa = (float(x) for x in rng.random(3))
        hard = bool(rng.random() < 0.3)
        rec = {"qv": qv, "qt": qt, "qa": qa if rng.random() < 0.8 else None, "hard": hard}
        base, s = max(qv, qt), C.compose_e2(rec)
        assert s <= base + 1e-15 and s >= (1 - C.LAMBDA) * base - 1e-15                  # QA only lowers
        if hard or rec["qa"] is None:
            assert s == base                                                             # never the hard tier
        else:
            assert s == pytest.approx(base * (1 - 0.5 * qa))
            assert C.compose_e2({**rec, "qa": min(1.0, qa + 0.1)}) <= s                   # more acquittal, lower
        assert C.value(rec, "composite") == s and C.value(rec, "base") == base
        assert C.value(rec, "1-qa") == 1 - (rec["qa"] or 0.0)
    assert C.value({"qv": 0.3, "qt": 0.6}) == 0.6                                        # E1 records unchanged
    with pytest.raises(ValueError):
        C.compose_e2({"qv": 0.2, "qt": 0.1, "qa": 1.5, "hard": False})


def test_value_rules():
    assert C.value(0.25) == 0.25 and C.value(-0.75, "qv") == -0.75                       # plain floats
    e1 = {"qv": 0.1, "qt": 0.4, "cov_min": 1.0}
    assert C.value(e1) == 0.4 and C.value(e1, "qt") == 0.4
    with pytest.raises(KeyError):                                                        # E1 has no base
        C.value(e1, "base")
    e2 = {"qv": 0.8, "qt": 0.2, "qa": None, "hard": False, "cov_min": 1.0, "base": 0.8, "composite": 0.8}
    assert C.value(e2) == 0.8 and C.value(e2, "1-qa") == 1.0                             # QA not asked
    e2q = dict(e2, qa=0.5)
    assert C.value(e2q) == 0.8 * 0.75 and C.value(e2q, "1-qa") == 0.5
    assert C.value(dict(e2q, hard=True)) == 0.8                                          # hard tier exempt
    # the stored composite is never read back: value recomputes it from the record
    assert C.value(dict(e2q, composite=0.0)) == 0.8 * 0.75
