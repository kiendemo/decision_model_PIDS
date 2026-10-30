"""configs/habit.yaml mirrors the habit-channel code constants (the code is the source of truth)."""

import os

import yaml

from conftest import REPO

from habitus.questions.habit import constants as C
from habitus.questions.habit import vocab
from habitus.train import items as X

HABIT_YAML = os.path.join(REPO, "configs", "habit.yaml")


def _cfg():
    with open(HABIT_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_habit_yaml_equals_the_constants():
    y = _cfg()
    assert y["composite"] == {"C_K": C.C_K, "BARS": list(C.BARS), "WEAK_CAP": C.WEAK_CAP,
                              "FIXED_ALERT_ROWS": list(C.FIXED_ALERT_ROWS), "TP_CURVE_MAX": C.TP_CURVE_MAX}
    assert y["questions"] == {"MAX_OPTIONS": C.MAX_OPTIONS, "MAX_BUCKETS": C.MAX_BUCKETS, "MAX_LISTED": C.MAX_LISTED,
                              "MAX_P1_NAMES": C.MAX_P1_NAMES, "MAX_VIEW_TOKENS": C.MAX_VIEW_TOKENS,
                              "SYSTEM_H_MAX_TOKENS": C.SYSTEM_H_MAX_TOKENS, "T_CLAMP": list(C.T_CLAMP)}
    assert y["nulls"] == {"NULL_MIN_STRATUM": C.NULL_MIN_STRATUM, "TAIL_FIT_MIN": C.TAIL_FIT_MIN, "TAIL_Q": C.TAIL_Q}
    h = y["hosts"]
    assert h["DEV_HOSTS"] == list(C.DEV_HOSTS) and h["TRAIN_ONLY"] == list(C.TRAIN_ONLY)
    assert {k: tuple(v) for k, v in h["folds"].items()} == C.FOLDS
    assert h["H501_SELECTION_DAYS"] == list(C.H501_SELECTION_DAYS)
    assert y["kplans"] == list(C.KPLANS) and y["select_kplans"] == list(C.SELECT_KPLANS)
    assert y["arms"] == C.ARM_DM


def test_habit_yaml_equals_the_item_export_constants():
    y = _cfg()
    assert y["export"] == {"HEAD_SAMPLE": X.HEAD_SAMPLE, "ABSENT_RATIO": X.ABSENT_RATIO, "A3_FRAC": X.A3_FRAC,
                           "TWIN_RATE": X.TWIN_RATE, "seed": X.SEED}
    s = y["sample"]
    assert s["composition"] == X.COMPOSITION
    assert s["n_fold"] == {"max": X.N_FOLD_MAX, "min": X.N_FOLD_MIN, "hours": X.N_FOLD_HOURS}
    assert s["n_per_fold"] is None                                  # TBD: the retraining's N per fold


def test_habit_yaml_pins_the_role_table_and_the_fetch_script():
    rt = _cfg()["role_table"]
    assert rt["sha256"] == vocab.ROLE_TABLE_SHA256
    script = open(os.path.join(REPO, "scripts", "fetch_catalogs.sh"), encoding="utf-8").read()
    for k in ("gtfobins_commit", "lolbas_commit", "contents_amd64_sha256", "xcyclopedia_sha256"):
        assert rt[k] in script, k


def test_no_out_of_scope_host_in_the_habit_constants():
    hosts = set(C.TRAIN_HOSTS)
    assert hosts == {"CADETS_E3", "THEIA_E3", "CLEARSCOPE_E3", "optc_h501", "ATLASV2_EDR"}
    for fold, members in C.FOLDS.items():
        assert set(members) <= hosts and (C.FOLD_HELDOUT[fold] is None or C.FOLD_HELDOUT[fold] not in members)
