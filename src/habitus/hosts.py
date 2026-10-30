"""The five in-scope hosts and their frozen day lists (the code is the source of truth; configs/
datasets.yaml mirrors it and tests/test_config.py asserts the mirror).

    from habitus import hosts
    s = hosts.spec("CADETS_E3")          # HostSpec; any other name raises ForbiddenDataset
    s.clean, s.test, s.k2, s.predeploy   # frozen day lists (dates as 'YYYY-MM-DD')

Day lists
  clean       usable attack-free days in date order: train + val dates minus attack dates, without
              days whose transformed graphs hold < 10% of the median clean-day bytes (data.days).
              They were computed once from the PIDSMaker graphs and are frozen here, so scoring needs
              the card files only (`habitus data manifest` re-checks them after a rebuild).
  configured_test  the PIDSMaker test dates; `test` keeps those with a non-empty card file.
  predeploy   clean days strictly before the first configured test day (C1 pre-deployment window).
  inner       the last pre-deployment day (temperature-fit day of a training host).
  k2          the first two clean days (the k = 2 reference).
  card_days   every day with a card file: attack-free days, then configured test days (sorted each).
Counts: test_day_nodes = nodes with a card on >= 1 test day (Host.universe); n_universe = the
evaluation universe, which adds the ground-truth nodes that have no test-day card (CADETS_E3: 4).
Only habitus.eval reads ground truth; gt_files only names the files, in PIDSMaker config order.
"""

import re
from dataclasses import dataclass, field, fields


class FrozenDict(dict):
    """A read-only dict (picklable and JSON-serialisable, unlike a mappingproxy)."""

    def _readonly(self, *_a, **_k):
        raise TypeError(f"{type(self).__name__} is read-only")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _readonly

    def __hash__(self):
        return hash(tuple(sorted(self.items())))

    def __reduce__(self):
        return (FrozenDict, (dict(self),))


DATASETS = ("CADETS_E3", "THEIA_E3", "CLEARSCOPE_E3", "optc_h501", "ATLASV2_EDR")

# What the decision model is told about the host: its operating-system family, never the dataset
# name. card["os"] is inside every card signature: these strings must not change.
OS_FAMILY = FrozenDict({
    "CADETS_E3": "freebsd", "THEIA_E3": "linux", "CLEARSCOPE_E3": "android",
    "optc_h501": "windows", "ATLASV2_EDR": "windows",
})

DEV_HOSTS = ("CADETS_E3", "THEIA_E3", "ATLASV2_EDR", "optc_h501")
TRAIN_ONLY = ("CLEARSCOPE_E3",)          # train-only in every fold (host-seen rows only)
TRAIN_HOSTS = DEV_HOSTS + TRAIN_ONLY
FOLDS = FrozenDict({
    **{f"loho-{h}": tuple(x for x in DEV_HOSTS if x != h) + TRAIN_ONLY for h in DEV_HOSTS},
    "all": TRAIN_HOSTS,
})
FOLD_HELDOUT = FrozenDict({**{f"loho-{h}": h for h in DEV_HOSTS}, "all": None})

LEVELS_E1 = ("L0", "L1", "L2", "L3", "L4")
_E1_ALL = FrozenDict({"k0": ("L0", "L1", "L3"), "k2": LEVELS_E1, "kall": LEVELS_E1})

# A name that looks like a benchmark host (an engagement suffix, an EDR collector, an OpTC host id)
# may not be used for a fake spec unless it is in DATASETS.
_REAL_LOOKING = re.compile(r"(_E[0-9]+|_EDR|_h[0-9]+)$", re.IGNORECASE)


class ForbiddenDataset(ValueError):
    """A dataset outside the five-host allow-list."""


def assert_allowed(ds):
    """Raise ForbiddenDataset unless `ds` is one of DATASETS. Returns ds."""
    if ds not in DATASETS:
        raise ForbiddenDataset(f"dataset {ds!r} is outside the allow-list {DATASETS}")
    return ds


@dataclass(frozen=True)
class HostSpec:
    ds: str
    os: str
    db_name: str
    configured_test: tuple
    test: tuple
    clean: tuple
    predeploy: tuple
    inner: str
    k2: tuple
    main_k: str                      # "2" or "all": the k of the host's main-table row
    k0_nulls: bool                   # k = 0 null (LODO clean-day) score files exist (E2 and E1-L3)
    e1_levels: FrozenDict            # {"k0"|"k2"|"kall": (E1 levels with score files)}
    gt_files: tuple                  # ground-truth CSVs, relative to Ground_Truth/orthrus, config order
    test_day_nodes: int
    n_universe: int
    gt_total: int
    gt_in_universe: int
    all_card_days: int
    card_days: tuple = ()
    train_dates: tuple = ()
    val_dates: tuple = ()
    attack_windows: tuple = ()       # ((gt file, start, end), ...) as in the PIDSMaker config
    fake: bool = field(default=False, compare=False)

    def __post_init__(self):
        if self.ds not in DATASETS:
            if not self.fake:
                raise ForbiddenDataset(f"HostSpec {self.ds!r}: not in the allow-list (fake specs: tests only)")
            if _REAL_LOOKING.search(self.ds):
                raise ForbiddenDataset(f"fake HostSpec name {self.ds!r} looks like a benchmark host")
        for f in ("configured_test", "test", "clean", "predeploy", "k2", "gt_files", "card_days",
                  "train_dates", "val_dates"):
            object.__setattr__(self, f, tuple(getattr(self, f)))
        object.__setattr__(self, "attack_windows", tuple(tuple(w) for w in self.attack_windows))
        object.__setattr__(self, "e1_levels", FrozenDict(
            {k: tuple(v) for k, v in dict(self.e1_levels).items()}))
        if self.main_k not in ("0", "2", "all"):
            raise ValueError(f"HostSpec {self.ds}: main_k must be '0', '2' or 'all', got {self.main_k!r}")
        if self.card_days and len(self.card_days) != self.all_card_days:
            raise ValueError(f"HostSpec {self.ds}: {len(self.card_days)} card days != all_card_days "
                             f"{self.all_card_days}")

    def __hash__(self):
        return hash((self.ds, self.clean, self.test, self.configured_test))

    def as_dict(self):
        """Plain dict (lists for tuples) in field order, without `fake`: the datasets.yaml layout."""
        out = {}
        for f in fields(self):
            if f.name in ("ds", "fake"):
                continue
            v = getattr(self, f.name)
            if f.name == "e1_levels":
                v = {k: list(x) for k, x in v.items()}
            elif f.name == "attack_windows":
                v = [list(w) for w in v]
            elif isinstance(v, tuple):
                v = list(v)
            out[f.name] = v
        return out


def _days(prefix, *dd):
    return tuple(f"{prefix}-{d}" for d in dd)


_SPECS = {
    "CADETS_E3": HostSpec(
        ds="CADETS_E3", os="freebsd", db_name="cadets_e3",
        configured_test=_days("2018-04", "06", "11", "12", "13"),
        test=_days("2018-04", "06", "11", "12", "13"),
        clean=_days("2018-04", "02", "03", "04", "05", "07", "08", "09", "10"),
        predeploy=_days("2018-04", "02", "03", "04", "05"),
        inner="2018-04-05",
        k2=_days("2018-04", "02", "03"),
        main_k="2",                  # k = all would include the post-test clean days 04-07..10
        k0_nulls=True, e1_levels=_E1_ALL,
        gt_files=("E3-CADETS/node_Nginx_Backdoor_06.csv", "E3-CADETS/node_Nginx_Backdoor_12.csv",
                  "E3-CADETS/node_Nginx_Backdoor_13.csv"),
        test_day_nodes=281585, n_universe=281589, gt_total=72, gt_in_universe=68, all_card_days=12,
        card_days=_days("2018-04", "02", "03", "04", "05", "07", "08", "09", "10", "06", "11", "12", "13"),
        train_dates=_days("2018-04", "02", "03", "04", "05", "07", "08", "09"),
        val_dates=_days("2018-04", "10"),
        attack_windows=(
            ("E3-CADETS/node_Nginx_Backdoor_06.csv", "2018-04-06 11:20:00", "2018-04-06 12:09:00"),
            ("E3-CADETS/node_Nginx_Backdoor_12.csv", "2018-04-12 13:59:00", "2018-04-12 14:39:00"),
            ("E3-CADETS/node_Nginx_Backdoor_13.csv", "2018-04-13 09:03:00", "2018-04-13 09:16:00"),
        ),
    ),
    "THEIA_E3": HostSpec(
        ds="THEIA_E3", os="linux", db_name="theia_e3",
        configured_test=_days("2018-04", "10", "12", "13"),
        test=_days("2018-04", "10", "12", "13"),
        clean=_days("2018-04", "03", "04", "05", "09"),       # 04-02 and 04-06..08 have no graph bytes
        predeploy=_days("2018-04", "03", "04", "05", "09"),
        inner="2018-04-09",
        k2=_days("2018-04", "03", "04"),
        main_k="all",
        k0_nulls=True, e1_levels=_E1_ALL,
        gt_files=("E3-THEIA/node_Browser_Extension_Drakon_Dropper.csv",
                  "E3-THEIA/node_Firefox_Backdoor_Drakon_In_Memory.csv"),
        test_day_nodes=700902, n_universe=700902, gt_total=118, gt_in_universe=118, all_card_days=11,
        card_days=_days("2018-04", "02", "03", "04", "05", "06", "07", "08", "09", "10", "12", "13"),
        train_dates=_days("2018-04", "02", "03", "04", "05", "06", "07", "08"),
        val_dates=_days("2018-04", "09"),
        attack_windows=(
            ("E3-THEIA/node_Browser_Extension_Drakon_Dropper.csv", "2018-04-12 12:40:00", "2018-04-12 13:30:00"),
            ("E3-THEIA/node_Firefox_Backdoor_Drakon_In_Memory.csv", "2018-04-10 14:30:00", "2018-04-10 15:00:00"),
        ),
    ),
    "CLEARSCOPE_E3": HostSpec(
        ds="CLEARSCOPE_E3", os="android", db_name="clearscope_e3",
        configured_test=_days("2018-04", "11", "12"),
        test=_days("2018-04", "11", "12"),
        clean=_days("2018-04", "03", "04", "05", "07", "08", "09", "10"),   # 04-02 dropped by day mass
        predeploy=_days("2018-04", "03", "04", "05", "07", "08", "09", "10"),
        inner="2018-04-10",
        k2=_days("2018-04", "03", "04"),
        main_k="all",
        k0_nulls=False,
        e1_levels={"k0": ("L3",), "k2": ("L2", "L4"), "kall": ("L2", "L4")},
        gt_files=("E3-CLEARSCOPE/node_clearscope_e3_firefox_0411.csv",),
        test_day_nodes=111403, n_universe=111403, gt_total=41, gt_in_universe=41, all_card_days=10,
        card_days=_days("2018-04", "02", "03", "04", "05", "07", "08", "09", "10", "11", "12"),
        train_dates=_days("2018-04", "03", "04", "05", "07", "08", "09", "10"),
        val_dates=_days("2018-04", "02"),
        attack_windows=(
            ("E3-CLEARSCOPE/node_clearscope_e3_firefox_0411.csv", "2018-04-11 13:54:00", "2018-04-11 14:48:00"),
        ),
    ),
    "optc_h501": HostSpec(
        ds="optc_h501", os="windows",
        db_name="optc_h501",         # upstream 2289cd9 says optc_501: overridden (data.upstream, docs/upstream.patch)
        configured_test=_days("2019-09", "23", "24", "25"),
        test=_days("2019-09", "23", "24", "25"),
        clean=_days("2019-09", "19", "20", "21", "22"),
        predeploy=_days("2019-09", "19", "20", "21", "22"),
        inner="2019-09-22",
        k2=_days("2019-09", "19", "20"),
        main_k="all",
        k0_nulls=True, e1_levels=_E1_ALL,
        gt_files=("h501/node_h501_0924.csv",),
        test_day_nodes=1489198, n_universe=1489198, gt_total=749, gt_in_universe=749, all_card_days=7,
        card_days=_days("2019-09", "19", "20", "21", "22", "23", "24", "25"),
        train_dates=_days("2019-09", "19", "20", "21"),
        val_dates=_days("2019-09", "22"),
        attack_windows=(("h501/node_h501_0924.csv", "2019-09-24 10:28:00", "2019-09-24 15:29:00"),),
    ),
    "ATLASV2_EDR": HostSpec(
        ds="ATLASV2_EDR", os="windows", db_name="atlasv2_edr",
        configured_test=_days("2022-07", "19", "20"),
        test=_days("2022-07", "19"),                            # 07-20 has no graph (empty card file)
        clean=_days("2022-07", "16", "17", "18"),
        predeploy=_days("2022-07", "16", "17", "18"),
        inner="2022-07-18",
        k2=_days("2022-07", "16", "17"),
        main_k="all",
        k0_nulls=True, e1_levels=_E1_ALL,
        gt_files=tuple(f"atlasv2_edr/atlasv2_edr_{x}.csv"
                       for x in ("s1", "s2", "s3", "s4", "m1", "m2", "m3", "m4", "m5", "m6")),
        test_day_nodes=152763, n_universe=152763, gt_total=48, gt_in_universe=48, all_card_days=5,
        card_days=_days("2022-07", "16", "17", "18", "19", "20"),
        train_dates=_days("2022-07", "16", "18"),
        val_dates=_days("2022-07", "17"),
        attack_windows=(
            ("atlasv2_edr/atlasv2_edr_s1.csv", "2022-07-19 13:00:00", "2022-07-19 13:40:00"),
            ("atlasv2_edr/atlasv2_edr_s2.csv", "2022-07-19 13:45:00", "2022-07-19 14:20:00"),
            ("atlasv2_edr/atlasv2_edr_s3.csv", "2022-07-19 14:20:00", "2022-07-19 15:05:00"),
            ("atlasv2_edr/atlasv2_edr_s4.csv", "2022-07-20 00:31:00", "2022-07-20 01:15:00"),
            ("atlasv2_edr/atlasv2_edr_m1.csv", "2022-07-19 16:00:00", "2022-07-19 17:50:00"),
            ("atlasv2_edr/atlasv2_edr_m2.csv", "2022-07-19 19:32:00", "2022-07-19 20:02:00"),
            ("atlasv2_edr/atlasv2_edr_m3.csv", "2022-07-19 20:06:00", "2022-07-19 20:40:00"),
            ("atlasv2_edr/atlasv2_edr_m4.csv", "2022-07-19 22:31:00", "2022-07-19 23:04:00"),
            ("atlasv2_edr/atlasv2_edr_m5.csv", "2022-07-19 23:16:00", "2022-07-19 23:46:00"),
            ("atlasv2_edr/atlasv2_edr_m6.csv", "2022-07-19 23:54:00", "2022-07-20 00:27:00"),
        ),
    ),
}


def spec(ds):
    """The frozen HostSpec of an allowed dataset (ForbiddenDataset otherwise)."""
    return _SPECS[assert_allowed(ds)]


def specs():
    return tuple(_SPECS[d] for d in DATASETS)


def os_family(ds):
    return OS_FAMILY[assert_allowed(ds)]


# --------------------------------------------------------------------------- #
# Day windows (accept a HostSpec, a Host, or a test double with .clean / .test)
# --------------------------------------------------------------------------- #
def _spec_of(x):
    if isinstance(x, HostSpec):
        return x
    s = getattr(x, "spec", None)
    return s if isinstance(s, HostSpec) else None


def first_test_day(x):
    """The first CONFIGURED test day (HostSpec.configured_test, before the card-file filter: a test
    day whose card file is empty still closes the pre-deployment window); a test double without a
    spec gives min(x.test). Raises when there is no test day: the window is then undefined."""
    s = _spec_of(x)
    days = list(s.configured_test) if s is not None else list(getattr(x, "test", None) or [])
    if not days:
        name = getattr(x, "ds", x)
        raise ValueError(f"{name}: no configured test day: the pre-deployment window is undefined")
    return min(days)


def predeploy(x):
    """Usable clean days strictly before the first configured test day, date order (CADETS_E3:
    04-02..04-05). Every reference, null and training day of the habit channel is one of them."""
    t0 = first_test_day(x)
    return [d for d in x.clean if d < t0]


def inner_day(x):
    """The temperature-fit day of a training host: its last pre-deployment day."""
    pre = predeploy(x)
    if not pre:
        raise ValueError(f"{getattr(x, 'ds', x)}: no pre-deployment day")
    return pre[-1]


def assert_predeploy(x, days, allow_oracle=False):
    """Raise unless every day is a pre-deployment clean day (allow_oracle: any clean day)."""
    ok = set(x.clean) if allow_oracle else set(predeploy(x))
    bad = [d for d in days if d not in ok]
    if bad:
        what = "clean" if allow_oracle else "pre-deployment clean"
        raise ValueError(f"{getattr(x, 'ds', x)}: days {bad} are not {what} days")
