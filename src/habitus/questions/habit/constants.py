"""Fixed a-priori constants of the habit channel (HABITUS+H), for the 5 in-scope hosts.

Every value here was fixed before any metric was computed and is never tuned. configs/habit.yaml
mirrors them (tests/test_habit_mirror.py asserts YAML == code); the code is the source of truth.
"""

SPEC_VERSION = "HB-v0-draft"            # written into every habit score file's meta "spec"

# --------------------------------------------------------------------------- #
# Composite and decision layer
# --------------------------------------------------------------------------- #
C_K = 0.5                   # acquittal factor for a node that is not lineage-attested: a = C_K * qa
BARS = (0, 1, 20)           # tau_B = bar(null composite pool, n_null_days, B) for B in BARS
WEAK_CAP = 20               # B = 20: the weak tier [tau_20, tau_1) holds at most 20 alerts per test day
FIXED_ALERT_ROWS = (1, 5, 20, 50)   # fixed alerts/day rows reported for every system
TP_CURVE_MAX = 100          # TP-vs-alerts/day curves over 0..100 alerts/day

# --------------------------------------------------------------------------- #
# Questions and readout
# --------------------------------------------------------------------------- #
MAX_OPTIONS = 4             # every habit question offers <= 4 letters (A..D)
MAX_BUCKETS = 8             # <= 8 buckets per (view, family): <= 6 listed identities + 2 generic
MAX_LISTED = 6              # listed usual identities of an inclusion family (P4, F1, F2, N1)
MAX_P1_NAMES = 4            # identity-attested usual parents named in P1 option A
MAX_VIEW_TOKENS = 1024      # a longer view is refused (logged) and its family becomes n/a
SYSTEM_H_MAX_TOKENS = 60    # SYSTEM_H (data-guard sentence + habit-section glossary)
T_CLAMP = (0.5, 3.0)        # temperature clamp; T is used ONLY for calibration reports (serving is T = 1)

# --------------------------------------------------------------------------- #
# Statistic and nulls
# --------------------------------------------------------------------------- #
NULL_MIN_STRATUM = 500      # the identity-attested split of a stratum is merged below 500 null node-days
TAIL_FIT_MIN = 200          # exponential tail fitted on the top 5% when a stratum has >= 200 units
TAIL_Q = 0.95               # empirical upper tail up to the 95th null percentile, exponential beyond

# --------------------------------------------------------------------------- #
# Hosts, folds, days
# --------------------------------------------------------------------------- #
DEV_HOSTS = ("CADETS_E3", "THEIA_E3", "ATLASV2_EDR", "optc_h501")
TRAIN_ONLY = ("CLEARSCOPE_E3",)                 # train-only in every fold, host-seen row only
TRAIN_HOSTS = DEV_HOSTS + TRAIN_ONLY
FOLDS = {f"loho-{h}": tuple(x for x in DEV_HOSTS if x != h) + TRAIN_ONLY for h in DEV_HOSTS}
FOLDS["all"] = TRAIN_HOSTS                      # deploy model M_all
FOLD_HELDOUT = {f"loho-{h}": h for h in DEV_HOSTS}
FOLD_HELDOUT["all"] = None
H501_SELECTION_DAYS = ("2019-09-19", "2019-09-21")   # arm-selection metrics of optc_h501 only
# Score-directory k plans: the reference construction a score file stands for
KPLANS = ("k0", "k2", "allc-pre")

# --------------------------------------------------------------------------- #
# Levels and arms
# --------------------------------------------------------------------------- #
# An arm is a level plus a score-directory arm suffix, "<level>.<arm>" (types.level_dir /
# split_level). HB-L is a level of its own (the Laya family, layout L-a, 3-seed geometric mean);
# the others are readout / backbone variants of HB or the MQ layout of HB-L.
# Value: (level, arm suffix).
CLONE_ARMS = {
    "HB-L": ("HB-L", None),             # Laya, L-a rows, fwd + rev, pbar over seeds x orders
    "HB-L.s0": ("HB-L", "s0"),          # seed 0 alone (reported next to the 3-seed output)
    "HB-L.mq": ("HB-L", "mq"),          # encode once / decide many (MQ masks), one pass
    "HB-L.zs": ("HB-L", "zs"),          # the Laya init zero-shot (L-a, never MQ)
    "HB.head": ("HB", "head"),          # fp32 letter head over the last hidden state
    "HB.08b": ("HB", "08b"),            # HB on Qwen3.5-0.8B (speed-accuracy point)
}
ARM_LEVELS = tuple(CLONE_ARMS)          # "HB-L" first: the clone level itself
LEVELS = ("E2", "HB0", "HB-Z", "HB") + tuple(a for a in ARM_LEVELS if "." not in a) + \
    tuple(a for a in ARM_LEVELS if "." in a)
HABIT_LEVELS = ("HB-Z", "HB") + ARM_LEVELS      # levels carrying s_hat
SELECT_KPLANS = ("k0", "k2", "allc-pre")        # reference plans of the label-free arm-selection items

# Every habit arm and the decision-model config that answers it (configs/dm/<name>.yaml).
ARM_DM = {
    "HB-Z": "jev-qwen3.5-4b-habit",
    "HB": "jev-qwen3.5-4b-lora",
    "HB.head": "jev-qwen3.5-4b-head",
    "HB.08b": "jev-qwen3.5-0.8b-lora",
    "HB-L.s0": "laya-typed-s0",
    "HB-L": "laya-typed-ens3",
    "HB-L.mq": "laya-typed-mq",
    "HB-L.zs": "laya-typed-zs",
}
TRAINED_ARMS = ("HB", "HB.head", "HB.08b", "HB-L.s0", "HB-L", "HB-L.mq")   # answered by a fold run

__all__ = ["SPEC_VERSION", "C_K", "BARS", "WEAK_CAP", "FIXED_ALERT_ROWS", "TP_CURVE_MAX", "MAX_OPTIONS",
           "MAX_BUCKETS", "MAX_LISTED", "MAX_P1_NAMES", "MAX_VIEW_TOKENS", "SYSTEM_H_MAX_TOKENS", "T_CLAMP",
           "NULL_MIN_STRATUM", "TAIL_FIT_MIN", "TAIL_Q", "DEV_HOSTS", "TRAIN_ONLY", "TRAIN_HOSTS", "FOLDS",
           "FOLD_HELDOUT", "H501_SELECTION_DAYS", "KPLANS", "CLONE_ARMS", "ARM_LEVELS", "LEVELS", "HABIT_LEVELS",
           "SELECT_KPLANS", "ARM_DM", "TRAINED_ARMS"]
