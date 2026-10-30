"""configs/pipeline.yaml mirrors the code constants (the code is the source of truth)."""

import os

import pytest
import yaml

from habitus.questions import routed as R
from habitus.state import ladder as LD
from habitus.state import precedent as PREC
from habitus.state import sections as S
from habitus.state import select as SEL

CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "pipeline.yaml")


@pytest.fixture(scope="module")
def cfg():
    with open(CFG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_e2_block_equals_code(cfg):
    e2 = cfg["e2"]
    assert e2["caps"] == SEL.CAPS
    assert {q: tuple(v) for q, v in e2["routes"].items()} == S.E2_ROUTES
    assert list(e2["routes"]) == list(S.E2_ROUTES)
    assert tuple(e2["hard_bits"]) == S.HARD
    assert e2["precedent_e"] == PREC.PREC_E and e2["precedent_ops"] == PREC.PREC_OPS
    assert e2["nf_reps"] == S.NF_REPS and e2["nf_entries"] == S.NF_ENTRIES
    assert e2["forms"] == {q: R.variants(q) for q in ("qv", "qt", "qa")}
    assert e2["low_cov"] == R.LOW_COV
    assert e2["ks"] == ["all", "2", "0"]
    assert e2["fallback"] == "-1+0.5/(1+n_s)"
    assert all(SEL.fallback_value(n) == -1 + 0.5 / (1 + n) for n in range(50))
    assert e2["node_score"] == "max_over_test_days"
    assert e2["record"] == ["qv", "qt", "qa", "hard", "cov_min", "base", "composite"]
    # grouping qvqt: QV and QT forms of a signature share a group, QA has its own
    st = '{"entity":{"type":"process","os":"linux"}}'
    g = {q: R.to_typed("s", q, st).group for q in ("qv", "qt", "qa")}
    assert e2["grouping"] == "qvqt" and g["qv"] == g["qt"] != g["qa"]


def test_e2_lambda_equals_composite(cfg):
    try:
        from habitus.decide.composite import LAMBDA
    except ImportError:
        pytest.skip("habitus.decide.composite not present yet")
    assert cfg["e2"]["lambda"] == LAMBDA


def test_ladder_block_equals_code(cfg):
    lad = cfg["ladder"]
    assert lad["levels"] == list(LD.LEVELS)
    assert tuple(lad["k0_levels"]) == LD.K0_LEVELS
    prim = {(0 if k == "0" else 2 if k == "2" else None): v for k, v in lad["primary"].items()}
    assert prim == LD.PRIMARY
    assert lad["e1_record"] == ["qv", "qt", "cov_min"] and lad["e1_questions"] == ["qv", "qt"]
    assert lad["e1_dedupe"] == "state_hash(serialized)"


def test_null_rules_equal_inventory(cfg):
    try:
        from habitus.decide.inventory import NULL_RULES
    except ImportError:
        pytest.skip("habitus.decide.inventory not present yet")
    assert cfg["null_rules"] == NULL_RULES
    assert set(cfg["null_rules"]) == {"E2", "L2", "L3", "L4"}                 # no L0 / L1 nulls
    assert set(cfg["null_rules"]["L3"]) == {"k0"}                             # no L3 nulls at k >= 1


def test_eval_block(cfg):
    assert cfg["eval"] == {"threshold": "oracle_best_mcc", "ties_negative": "reported", "gt_without_card": "FN"}
