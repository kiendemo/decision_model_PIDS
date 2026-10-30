"""Settings: defaults, merge order, env overrides, the former-data-disk refusal, and the YAML mirrors
of configs/default.yaml and configs/datasets.yaml (owner B1)."""

import json
import os
import re
import subprocess
import sys

import pytest
import yaml

from habitus import config as C
from habitus import hosts

REPO = C.REPO_ROOT
DISK = "/D" + "ATA"          # the former data disk, spelled so that repository scans do not match


def _iso(**kw):
    """Settings from default.yaml only (no local.yaml, empty environment) unless kw says otherwise."""
    kw.setdefault("local", False)
    kw.setdefault("env", {})
    return C.load_settings(**kw)


def test_defaults_are_repo_relative_and_complete():
    s = _iso()
    p = s.paths
    assert p.home == REPO
    assert p.data == os.path.join(REPO, "release", "habitus-cache-v1")
    assert p.cards == os.path.join(p.data, "cards")
    assert p.nodeindex == os.path.join(p.data, "nodeindex")
    assert p.ground_truth == os.path.join(p.data, "ground_truth", "orthrus")
    assert p.tokenizers == os.path.join(p.data, "tokenizer")
    assert p.runs == os.path.join(REPO, "runs")
    assert p.models == os.path.join(REPO, "models", "hf")
    for k in C.OPTIONAL_PATHS:
        assert getattr(p, k) is None, k
    assert (s.device, s.threads, s.forbidden_gpus, s.offline) == ("cpu", 8, (), True)
    assert s.sources == (C.DEFAULT_CONFIG,)


def test_no_default_on_former_data_disk():
    rx = re.compile(r"^/D[A]TA(/|$)")
    s = _iso()
    for k, v in s.paths.as_dict().items():
        assert v is None or not rx.match(v), (k, v)
    with open(C.DEFAULT_CONFIG, encoding="utf-8") as f:
        assert DISK not in f.read()
    with open(os.path.join(REPO, "configs", "local.example.yaml"), encoding="utf-8") as f:
        assert DISK not in f.read()


@pytest.mark.parametrize("how", ["overrides", "env"])
def test_former_data_disk_refused(how):
    target = DISK + "/someone/habitus"
    with pytest.raises(C.ConfigError, match="former data disk"):
        if how == "overrides":
            _iso(overrides={"paths": {"data": target}})
        else:
            _iso(env={"HABITUS_DATA": target})
    with pytest.raises(C.ConfigError):
        _iso(env={"HABITUS_RUNS": DISK})
    # the sibling disks belong to other users: refused as well, for any path and any output
    for sib in ("2/x", "3/y", "4/z"):
        with pytest.raises(C.ConfigError, match="another user's data disk"):
            _iso(env={"HABITUS_RUNS": DISK + sib})
        with pytest.raises(C.ConfigError):
            C.check_path("output", DISK + sib)
    # a longer name is a different path
    assert C.check_path("output", DISK + "SET/x") == DISK + "SET/x"


def test_env_overrides(tmp_path):
    env = {"HABITUS_DATA": str(tmp_path / "bundle"), "HABITUS_RUNS": str(tmp_path / "r"),
           "HABITUS_GT": str(tmp_path / "gt"), "HF_HOME": str(tmp_path / "hf"),
           "HABITUS_LAYA_HUB": str(tmp_path / "laya"), "HABITUS_CATALOGS": str(tmp_path / "cat"),
           "PIDSMAKER_ARTIFACTS": str(tmp_path / "art"), "PIDSMAKER_UPSTREAM_ARTIFACTS": str(tmp_path / "up"),
           "HABITUS_FORBIDDEN_GPUS": "2, 3"}
    s = _iso(env=env)
    p = s.paths
    assert p.data == str(tmp_path / "bundle")
    assert p.cards == str(tmp_path / "bundle" / "cards")           # ${data} follows the env override
    assert p.runs == str(tmp_path / "r") and p.ground_truth == str(tmp_path / "gt")
    assert p.models == str(tmp_path / "hf") and p.laya_hub == str(tmp_path / "laya")
    assert p.catalogs == str(tmp_path / "cat")
    assert p.pidsmaker_artifacts == str(tmp_path / "art") and p.upstream_artifacts == str(tmp_path / "up")
    assert s.forbidden_gpus == (2, 3)
    assert _iso(env={"HABITUS_FORBIDDEN_GPUS": ""}).forbidden_gpus == ()


def test_forbidden_gpus_union_env_only_adds(tmp_path):
    """forbidden_gpus is the union of every layer: the environment adds, never clears or replaces."""
    over = {"forbidden_gpus": [2]}
    assert _iso(overrides=over).forbidden_gpus == (2,)
    assert _iso(overrides=over, env={"HABITUS_FORBIDDEN_GPUS": ""}).forbidden_gpus == (2,)
    assert _iso(overrides=over, env={"HABITUS_FORBIDDEN_GPUS": "1"}).forbidden_gpus == (1, 2)
    main = tmp_path / "main.yaml"
    main.write_text("forbidden_gpus: [3]\n", encoding="utf-8")
    assert _iso(path=str(main), overrides={"forbidden_gpus": []}).forbidden_gpus == (3,)
    assert _iso(path=str(main), overrides=over).forbidden_gpus == (2, 3)


def test_model_and_catalog_stores_outside_home_are_read_only(tmp_path):
    """models / laya_hub / catalogs outside paths.home are read-only inputs (check_writable)."""
    home = tmp_path / "repo"
    ext = tmp_path / "elsewhere"
    s = _iso(overrides={"paths": {"home": str(home), "catalogs": str(ext / "catalogs2"),
                                  "models": str(ext / "hf"), "laya_hub": "models/laya_hub"}})
    roots = s.paths.read_only_roots()
    assert str(ext / "catalogs2") in roots and str(ext / "hf") in roots
    assert str(home / "models" / "laya_hub") not in roots           # inside home: writable (fetch target)
    for p in (ext / "catalogs2" / "gtfobins", ext / "hf" / "x"):
        with pytest.raises(C.ConfigError, match="read-only input"):
            C.check_writable(str(p), s)
    assert C.check_writable(str(home / "models" / "catalogs2"), s) == str(home / "models" / "catalogs2")


def test_merge_order_main_local_overrides_env(tmp_path):
    main = tmp_path / "main.yaml"
    main.write_text("threads: 4\npaths:\n  runs: main_runs\n  laya_src: /opt/laya\n", encoding="utf-8")
    s = _iso(path=str(main))
    assert s.threads == 4 and s.paths.runs == os.path.join(REPO, "main_runs")
    assert s.paths.laya_src == "/opt/laya"
    assert s.sources == (C.DEFAULT_CONFIG, str(main))
    s = _iso(path=str(main), overrides={"paths": {"runs": "/x/over"}, "device": "cuda:0"})
    assert s.paths.runs == "/x/over" and s.device == "cuda:0"
    s = _iso(path=str(main), overrides={"paths": {"runs": "/x/over"}}, env={"HABITUS_RUNS": "/x/env"})
    assert s.paths.runs == "/x/env"
    s = _iso(env={C.CONFIG_ENV: str(main)})                   # $HABITUS_CONFIG
    assert s.threads == 4


def test_relative_env_path_is_cwd_relative(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _iso(env={"HABITUS_RUNS": "rel"}).paths.runs == str(tmp_path / "rel")
    s = _iso(env={"HABITUS_DATA": "bundle"})              # ${data}/cards follows the cwd-relative data
    assert s.paths.data == str(tmp_path / "bundle") and s.paths.cards == str(tmp_path / "bundle" / "cards")


def test_interpolation_and_errors(tmp_path):
    s = _iso(overrides={"paths": {"laya_src": "${CYRAG_ROOT}/third_party/laya"}}, env={"CYRAG_ROOT": "/r"})
    assert s.paths.laya_src == "/r/third_party/laya"
    with pytest.raises(C.ConfigError, match="CYRAG_ROOT"):
        _iso(overrides={"paths": {"laya_src": "${CYRAG_ROOT}/x"}})
    with pytest.raises(C.ConfigError, match="cycle"):
        _iso(overrides={"paths": {"data": "${cards}/x", "cards": "${data}/cards"}})
    with pytest.raises(C.ConfigError, match="unknown path key"):
        _iso(overrides={"paths": {"no_such_key": "x"}})
    with pytest.raises(C.ConfigError, match="unknown key"):
        _iso(overrides={"gpus": [1]})
    with pytest.raises(C.ConfigError, match="null"):
        _iso(overrides={"paths": {"data": "${laya_src}/x"}})


def test_local_example_has_placeholders_only():
    p = os.path.join(REPO, "configs", "local.example.yaml")
    with open(p, encoding="utf-8") as f:
        text = f.read()
    raw = yaml.safe_load(text)
    assert raw["forbidden_gpus"] == [2]
    for k, v in raw["paths"].items():
        assert k in C.OPTIONAL_PATHS + C.CORE_PATHS, k
        assert v is None or v.startswith("${CYRAG_ROOT}/"), (k, v)
    assert not re.search(r"/h[o]me/|/t[m]p/", text)
    # with the placeholder defined, it loads
    s = C.load_settings(path=p, local=False, env={"CYRAG_ROOT": "/srv/cyrag"})
    assert s.paths.pidsmaker_src == "/srv/cyrag/PIDSMaker" and s.forbidden_gpus == (2,)


def test_local_yaml_is_gitignored():
    with open(os.path.join(REPO, ".gitignore"), encoding="utf-8") as f:
        assert "configs/local.yaml" in f.read().split()


def test_settings_cached_and_reset(monkeypatch, tmp_path):
    C.reset_settings()
    a = C.settings()
    assert C.settings() is a
    monkeypatch.setenv("HABITUS_RUNS", str(tmp_path / "other"))
    assert C.settings() is a                                  # cached per process
    C.reset_settings()
    assert C.settings().paths.runs == str(tmp_path / "other")
    C.reset_settings()


def test_check_writable_and_require(tmp_path):
    s = _iso(overrides={"paths": {"pidsmaker_artifacts": str(tmp_path / "art")}})
    with pytest.raises(C.ConfigError, match="read-only"):
        C.check_writable(str(tmp_path / "art" / "x"), s)
    assert C.check_writable(str(tmp_path / "ok"), s) == str(tmp_path / "ok")
    with pytest.raises(C.ConfigError, match="laya_src"):
        C.require("laya_src", s)
    assert C.require("pidsmaker_artifacts", s) == str(tmp_path / "art")


def test_import_mutates_no_environment(tmp_path):
    code = ("import os, json; before = dict(os.environ); import habitus.config as c; c.settings(); "
            "import habitus.hosts, habitus.guard, habitus.io, habitus.log, habitus.data.host; "
            "print(json.dumps(dict(os.environ) == before))")
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "src"))
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert json.loads(out.stdout.strip().splitlines()[-1]) is True


# --------------------------------------------------------------------------- #
# Mirrors: default.yaml and datasets.yaml == code
# --------------------------------------------------------------------------- #
def test_default_yaml_mirror():
    with open(C.DEFAULT_CONFIG, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    assert tuple(raw) == C.TOP_KEYS
    assert tuple(raw["paths"]) == C.CORE_PATHS + C.OPTIONAL_PATHS
    assert [f for f in C.Paths.__dataclass_fields__] == list(C.CORE_PATHS + C.OPTIONAL_PATHS)
    for k in C.OPTIONAL_PATHS:
        assert raw["paths"][k] is None, k
    assert raw["device"] == "cpu" and raw["threads"] == 8 and raw["forbidden_gpus"] == [] and raw["offline"] is True


def test_datasets_yaml_mirror():
    with open(os.path.join(REPO, "configs", "datasets.yaml"), encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    assert tuple(raw["datasets"]) == hosts.DATASETS
    assert raw["os_family"] == dict(hosts.OS_FAMILY)
    assert tuple(raw["dev_hosts"]) == hosts.DEV_HOSTS and tuple(raw["train_only"]) == hosts.TRAIN_ONLY
    assert {k: tuple(v) for k, v in raw["folds"].items()} == dict(hosts.FOLDS)
    assert tuple(raw["hosts"]) == hosts.DATASETS
    for ds in hosts.DATASETS:
        assert raw["hosts"][ds] == hosts.spec(ds).as_dict(), ds
