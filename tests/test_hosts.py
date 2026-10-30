"""The five-host allow-list and the frozen day lists (habitus.hosts)."""

import ast
import os
import re

import pytest

from habitus import hosts as H

from conftest import asset

D = {ds: H.spec(ds) for ds in H.DATASETS}


def test_allow_list_and_os_family():
    assert H.DATASETS == ("CADETS_E3", "THEIA_E3", "CLEARSCOPE_E3", "optc_h501", "ATLASV2_EDR")
    assert dict(H.OS_FAMILY) == {"CADETS_E3": "freebsd", "THEIA_E3": "linux", "CLEARSCOPE_E3": "android",
                                 "optc_h501": "windows", "ATLASV2_EDR": "windows"}
    for ds in H.DATASETS:
        assert H.assert_allowed(ds) == ds and D[ds].ds == ds and D[ds].os == H.OS_FAMILY[ds]
    for bad in ("NOT_A_HOST", "cadets_e3", "", "FAKE"):
        with pytest.raises(H.ForbiddenDataset):
            H.spec(bad)
        with pytest.raises(H.ForbiddenDataset):
            H.assert_allowed(bad)
    assert issubclass(H.ForbiddenDataset, ValueError)


def test_specs_are_frozen():
    s = D["CADETS_E3"]
    with pytest.raises(Exception):
        s.clean = ()
    with pytest.raises(TypeError):
        s.e1_levels["k0"] = ("L0",)
    assert isinstance(s.clean, tuple) and isinstance(s.gt_files, tuple)
    assert hash(s) == hash(H.spec("CADETS_E3"))


def test_fake_spec_rules():
    from fakes import make_spec
    f = make_spec(ds="NOT_A_HOST", clean=("c1", "c2"), test=("t1",))
    assert f.fake and f.predeploy == ("c1", "c2") and f.inner == "c2" and f.k2 == ("c1", "c2")
    with pytest.raises(H.ForbiddenDataset):
        H.HostSpec(**{**D["CADETS_E3"].__dict__, "ds": "NOT_A_HOST"})      # a non-fake spec off the list
    for looks_real in ("SOMETHING_E5", "OTHER_EDR", "optc_h999"):
        with pytest.raises(H.ForbiddenDataset):
            make_spec(ds=looks_real)


def _d(prefix, *dd):
    return tuple(f"{prefix}-{x}" for x in dd)


FROZEN = {
    "CADETS_E3": dict(db_name="cadets_e3", configured_test=_d("2018-04", "06", "11", "12", "13"),
                      test=_d("2018-04", "06", "11", "12", "13"),
                      clean=_d("2018-04", "02", "03", "04", "05", "07", "08", "09", "10"),
                      predeploy=_d("2018-04", "02", "03", "04", "05"), inner="2018-04-05",
                      k2=_d("2018-04", "02", "03"), main_k="2", k0_nulls=True,
                      test_day_nodes=281585, n_universe=281589, gt_total=72, gt_in_universe=68, all_card_days=12),
    "THEIA_E3": dict(db_name="theia_e3", configured_test=_d("2018-04", "10", "12", "13"),
                     test=_d("2018-04", "10", "12", "13"), clean=_d("2018-04", "03", "04", "05", "09"),
                     predeploy=_d("2018-04", "03", "04", "05", "09"), inner="2018-04-09",
                     k2=_d("2018-04", "03", "04"), main_k="all", k0_nulls=True,
                     test_day_nodes=700902, n_universe=700902, gt_total=118, gt_in_universe=118, all_card_days=11),
    "CLEARSCOPE_E3": dict(db_name="clearscope_e3", configured_test=_d("2018-04", "11", "12"),
                          test=_d("2018-04", "11", "12"),
                          clean=_d("2018-04", "03", "04", "05", "07", "08", "09", "10"),
                          predeploy=_d("2018-04", "03", "04", "05", "07", "08", "09", "10"), inner="2018-04-10",
                          k2=_d("2018-04", "03", "04"), main_k="all", k0_nulls=False,
                          test_day_nodes=111403, n_universe=111403, gt_total=41, gt_in_universe=41,
                          all_card_days=10),
    "optc_h501": dict(db_name="optc_h501", configured_test=_d("2019-09", "23", "24", "25"),
                      test=_d("2019-09", "23", "24", "25"), clean=_d("2019-09", "19", "20", "21", "22"),
                      predeploy=_d("2019-09", "19", "20", "21", "22"), inner="2019-09-22",
                      k2=_d("2019-09", "19", "20"), main_k="all", k0_nulls=True,
                      test_day_nodes=1489198, n_universe=1489198, gt_total=749, gt_in_universe=749,
                      all_card_days=7),
    "ATLASV2_EDR": dict(db_name="atlasv2_edr", configured_test=_d("2022-07", "19", "20"),
                        test=_d("2022-07", "19"), clean=_d("2022-07", "16", "17", "18"),
                        predeploy=_d("2022-07", "16", "17", "18"), inner="2022-07-18",
                        k2=_d("2022-07", "16", "17"), main_k="all", k0_nulls=True,
                        test_day_nodes=152763, n_universe=152763, gt_total=48, gt_in_universe=48, all_card_days=5),
}


@pytest.mark.parametrize("ds", H.DATASETS)
def test_frozen_values(ds):
    s = D[ds]
    for k, v in FROZEN[ds].items():
        assert getattr(s, k) == v, (ds, k)
    assert s.k2 == s.clean[:2]
    assert len(s.card_days) == s.all_card_days == len(set(s.card_days))
    assert set(s.clean) <= set(s.card_days) and set(s.configured_test) <= set(s.card_days)
    assert set(s.test) <= set(s.configured_test)
    assert not set(s.clean) & set(s.configured_test)
    assert list(s.clean) == sorted(s.clean) and list(s.test) == sorted(s.test)


def test_card_days_total_45():
    assert sum(s.all_card_days for s in D.values()) == 45


def test_e1_levels_and_k0_nulls():
    full = {"k0": ("L0", "L1", "L3"), "k2": ("L0", "L1", "L2", "L3", "L4"), "kall": ("L0", "L1", "L2", "L3", "L4")}
    for ds in ("CADETS_E3", "THEIA_E3", "optc_h501", "ATLASV2_EDR"):
        assert dict(D[ds].e1_levels) == full and D[ds].k0_nulls
    cs = D["CLEARSCOPE_E3"]
    assert dict(cs.e1_levels) == {"k0": ("L3",), "k2": ("L2", "L4"), "kall": ("L2", "L4")} and not cs.k0_nulls


def test_gt_files_config_order():
    assert D["CADETS_E3"].gt_files == ("E3-CADETS/node_Nginx_Backdoor_06.csv", "E3-CADETS/node_Nginx_Backdoor_12.csv",
                                        "E3-CADETS/node_Nginx_Backdoor_13.csv")
    assert D["THEIA_E3"].gt_files == ("E3-THEIA/node_Browser_Extension_Drakon_Dropper.csv",
                                       "E3-THEIA/node_Firefox_Backdoor_Drakon_In_Memory.csv")
    assert D["ATLASV2_EDR"].gt_files == tuple(f"atlasv2_edr/atlasv2_edr_{x}.csv" for x in
                                              ("s1", "s2", "s3", "s4", "m1", "m2", "m3", "m4", "m5", "m6"))
    assert [w[0] for w in D["ATLASV2_EDR"].attack_windows] == list(D["ATLASV2_EDR"].gt_files)


def test_predeploy_inner_first_test_day():
    c = D["CADETS_E3"]
    assert H.first_test_day(c) == "2018-04-06"
    assert H.predeploy(c) == ["2018-04-02", "2018-04-03", "2018-04-04", "2018-04-05"]
    assert H.inner_day(c) == "2018-04-05"
    for ds, s in D.items():
        assert tuple(H.predeploy(s)) == s.predeploy, ds
        assert H.inner_day(s) == s.inner, ds
    # ATLAS: the empty configured test day 07-20 does not move the window; 07-19 closes it
    assert H.first_test_day(D["ATLASV2_EDR"]) == "2022-07-19"
    H.assert_predeploy(c, ["2018-04-02"])
    with pytest.raises(ValueError):
        H.assert_predeploy(c, ["2018-04-07"])                 # post-test clean day: oracle-ish
    H.assert_predeploy(c, ["2018-04-07"], allow_oracle=True)


def test_window_functions_on_hosts_and_doubles():
    from fakes import make_kcs1_host

    class Double:
        ds, clean, test = "X", ["a", "b", "z"], ["c"]
    assert H.first_test_day(Double()) == "c" and H.predeploy(Double()) == ["a", "b"]
    h = make_kcs1_host()
    assert H.predeploy(h) == ["c1", "c2"] and H.inner_day(h) == "c2"

    class NoTest:
        ds, clean, test = "Y", ["a"], []
    with pytest.raises(ValueError):
        H.first_test_day(NoTest())


def test_benign_dates_give_card_days_and_clean():
    from habitus.data.days import _benign_dates
    dropped = {"THEIA_E3": {"2018-04-02", "2018-04-06", "2018-04-07", "2018-04-08"},
               "CLEARSCOPE_E3": {"2018-04-02"}}
    for ds, s in D.items():
        benign = sorted(_benign_dates(s))
        assert tuple(benign + sorted(s.configured_test)) == s.card_days, ds
        assert set(benign) - set(s.clean) == dropped.get(ds, set()), ds


def test_folds():
    assert H.DEV_HOSTS == ("CADETS_E3", "THEIA_E3", "ATLASV2_EDR", "optc_h501")
    assert H.FOLDS["all"] == H.DEV_HOSTS + ("CLEARSCOPE_E3",)
    for h in H.DEV_HOSTS:
        f = H.FOLDS[f"loho-{h}"]
        assert h not in f and f[-1] == "CLEARSCOPE_E3" and len(f) == 4
        assert H.FOLD_HELDOUT[f"loho-{h}"] == h


def test_no_out_of_scope_host_names_in_my_files():
    # character classes: this file must not contain the names it looks for
    rx = re.compile(r"(CADETS|THEIA|CLEARSCOPE)_E[5]|(TRAC[E]|FIVEDIRECTION[S])_E[0-9]|optc_h(0[5]1|2[0]1)|"
                    r"CARB[A]NAK|atlasv2_h[0-9]", re.IGNORECASE)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mine = ["src/habitus/hosts.py", "src/habitus/config.py", "src/habitus/guard.py", "src/habitus/io.py",
            "src/habitus/log.py", "configs/datasets.yaml", "configs/default.yaml", "configs/local.example.yaml",
            "scripts/build_graphs.py", "docs/DATA.md", "docs/upstream.patch", "tests/conftest.py", "tests/fakes.py"]
    mine += [f"src/habitus/data/{f}" for f in ("__init__.py", "upstream.py", "graphs.py", "resolver.py", "profiles.py",
                                               "identity.py", "cards.py", "days.py", "nodeindex.py", "host.py")]
    mine += [f"tests/test_{f}.py" for f in ("config", "guard", "data_cards", "data_host", "data_nodeindex")]
    for rel in mine:
        p = os.path.join(repo, rel)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            hits = rx.findall(f.read())
        assert not hits, (rel, hits)


# --------------------------------------------------------------------------- #
# Provenance: the frozen values against the PIDSMaker config (AST, no import)
# --------------------------------------------------------------------------- #
def _config_datasets(path):
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "DATASET_DEFAULT_CONFIG" for t in node.targets):
            out = {}
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant) and k.value in H.DATASETS:     # never evaluate other hosts
                    out[k.value] = ast.literal_eval(v)
            return out
    raise AssertionError(f"no DATASET_DEFAULT_CONFIG in {path}")


@pytest.mark.provenance
@pytest.mark.parametrize("key", ["upstream_src", "pidsmaker_src"])
def test_config_order_matches_pidsmaker_config(settings, key):
    root = asset(getattr(settings.paths, key), f"paths.{key}")
    cfg = _config_datasets(asset(os.path.join(root, "pidsmaker", "config", "config.py"), "PIDSMaker config.py"))
    assert set(cfg) == set(H.DATASETS)
    for ds, s in D.items():
        c = cfg[ds]
        assert tuple(c["ground_truth_relative_path"]) == s.gt_files, ds
        assert tuple(c["train_dates"]) == s.train_dates and tuple(c["val_dates"]) == s.val_dates, ds
        assert tuple(c["test_dates"]) == s.configured_test, ds
        assert tuple(tuple(w) for w in c["attack_to_time_window"]) == s.attack_windows, ds
        if ds != "optc_h501":                    # both trees here carry the optc_h501 rename
            assert c["database"] == s.db_name, ds
        assert c["database"] == c["database_all_file"]
