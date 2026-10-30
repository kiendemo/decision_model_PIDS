"""The shared calibration layer: soft_ce, inner-day temperatures, weighted AUC / ECE, reports."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from habitus.train import calibrate as C


def _item(family="P2", letters=("A", "B"), target=(1.0, 0.0), weight=1.0, role="inner", variant="fwd", qkey="q"):
    return SimpleNamespace(family=family, letters=letters, target=target, weight=weight, role=role, variant=variant,
                           qkey=qkey)


def test_soft_ce_hand_values():
    import torch
    z = torch.tensor([[0.0, 0.0, 5.0, 5.0], [math.log(3.0), 0.0, -2.0, 7.0]])
    mask = torch.tensor([[True, True, False, False], [True, True, False, False]])
    t = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]])
    w = torch.tensor([1.0, 2.0])
    row1, row2 = math.log(2.0), 0.5 * (math.log(4.0 / 3.0) + math.log(4.0))
    assert float(C.soft_ce(z, t, mask, w)) == pytest.approx((row1 + 2 * row2) / 3, abs=1e-6)
    assert float(C.soft_ce(z, t, mask, w, scale=10.0)) == pytest.approx((row1 + 2 * row2) / 10, abs=1e-6)
    z4 = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    p = torch.softmax(z4, -1)[0]
    t4 = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    want = -float((t4[0] * torch.log(p)).sum())
    assert float(C.soft_ce(z4, t4, torch.ones(1, 4, dtype=torch.bool), torch.ones(1))) == pytest.approx(want, abs=1e-6)


def test_fit_temperatures_recovers_and_clamps():
    rng = np.random.default_rng(0)
    for t0, want in ((2.0, 2.0), (0.8, 0.8), (5.0, 3.0), (0.2, 0.5)):
        Z = rng.normal(0, 3, size=(400, 2))
        P = C._softmax(Z, t0)
        items = [_item(target=tuple(p), weight=float(1 + i % 3)) for i, p in enumerate(P)]
        Zc = rng.normal(0, 3, size=(200, 4))
        Pc = C._softmax(Zc, t0)
        items += [_item("P1", ("A", "B", "C", "D"), tuple(p)) for p in Pc]
        temps = C.fit_temperatures(list(Z) + list(Zc), items)
        assert set(temps) == set(C.FORMS)
        assert temps["inclusion-2"] == pytest.approx(want, abs=2e-3)
        assert temps["choice-4"] == pytest.approx(want, abs=2e-3)
    assert C.fit_temperatures([], []) == {"inclusion-2": 1.0, "choice-4": 1.0}
    with pytest.raises(ValueError, match="inner-day"):
        C.fit_temperatures([np.zeros(2)], [_item(role="train")])
    with pytest.raises(ValueError, match="logits for"):
        C.fit_temperatures([np.zeros(3)], [_item()])
    assert C.T_CLAMP == (0.5, 3.0)


def test_t_clamp_matches_habit_constants():
    from habitus.questions.habit import constants as K
    assert tuple(K.T_CLAMP) == C.T_CLAMP


def test_weighted_auc_and_ece_hand_values():
    assert C.weighted_auc([0.9, 0.8], [1, 1], [0.1, 0.8], [1, 1]) == pytest.approx(3.5 / 4)
    assert C.weighted_auc([0.9, 0.8], [2, 1], [0.1, 0.8], [1, 1]) == pytest.approx(5.5 / 6)
    assert C.weighted_auc([1.0], [1], [0.0], [1]) == 1.0 and C.weighted_auc([0.0], [1], [1.0], [1]) == 0.0
    assert C.weighted_auc([], [], [0.1], [1]) is None
    conf, acc, w = np.array([0.95, 0.55]), np.array([1.0, 0.0]), np.array([1.0, 1.0])
    assert C._ece(conf, acc, w) == pytest.approx((0.05 + 0.55) / 2)
    assert C._ece(np.array([]), np.array([]), np.array([])) is None


def test_calibration_report_fields_and_tail():
    z = [np.array([2.0, 0.0]), np.array([0.0, 2.0]), np.array([-3.0, 3.0]), np.array([1.0, 0.0, -1.0, 0.5])]
    items = [_item(target=(0.6, 0.4), qkey="a"),
             _item(target=(0.0, 1.0), qkey="b"),                       # absent
             _item(target=(0.2, 0.8), variant="rev", qkey="c"),       # rev: yes is the LAST letter
             _item("P1", ("A", "B", "C", "D"), (1.0, 0.0, 0.0, 0.0), qkey="d")]
    rep = C.calibration_report(z, items, {"inclusion-2": 1.0})
    assert rep["n"] == 4 and set(rep["by_form"]) == {"inclusion-2", "choice-4"}
    assert rep["n_tail"] == 2 and rep["n_absent"] == 1 and rep["tail_no_ref"] == 0
    p_a = 1 / (1 + math.exp(-2.0))
    p_c = 1 / (1 + math.exp(-6.0))                                   # rev: p_yes = P(B)
    want = -(0.6 * math.log(p_a) + 0.4 * math.log(1 - p_a) + 0.8 * math.log(p_c) + 0.2 * math.log(1 - p_c)) / 2
    assert rep["tail_log_score"] == pytest.approx(want, abs=1e-9)
    p_b = 1 / (1 + math.exp(2.0))
    assert rep["rare_auc"] == pytest.approx(1.0 if min(p_a, p_c) > p_b else 0.5)
    nll1 = -(0.6 * math.log(p_a) + 0.4 * math.log(1 - p_a))
    assert rep["by_form"]["inclusion-2"]["nll_t1"] == pytest.approx(
        (nll1 - math.log(1 - p_b) - (0.2 * math.log(1 - p_c) + 0.8 * math.log(p_c))) / 3, abs=1e-9)
    ref = C.calibration_report(z, items, None, ref_p={"a": 0.01, "c": 0.5})
    assert ref["n_tail"] == 1 and ref["tail_no_ref"] == 0
    noref = C.calibration_report(z, items, None, ref_p={"a": 0.01})
    assert noref["n_tail"] == 1 and noref["tail_no_ref"] == 1
    assert set(rep["tail_coverage"]) == {"0.01", "0.001"}
    assert C.form_of(items[0]) == "inclusion-2" and C.form_of(items[3]) == "choice-4"
