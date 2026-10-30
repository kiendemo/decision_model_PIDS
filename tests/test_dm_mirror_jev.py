"""configs/dm/jev*.yaml and configs/train/jev*.yaml mirror the code (YAML == code constants), and the
registry builds every jev config."""

import inspect

import pytest
import yaml

import dm_helpers as H
from habitus import dm as D
from habitus.dm import jev as J
from habitus.dm import jev_head as JH
from habitus.train import jev as TJ

CFG = H.REPO / "configs"
JEV = ("jev-qwen3.5-4b", "jev-qwen3.5-4b-habit", "jev-qwen3.5-4b-lora", "jev-qwen3.5-4b-head", "jev-qwen3.5-0.8b-lora")
COMMON = {"dm", "layout", "questions", "backbone", "revision", "dtype", "cache_tag", "cache_only", "device",
          "batch_size", "prefix_share", "grouping"}


def _load(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_train_yaml_mirror_recipes():
    lora = _load(CFG / "train" / "jev-lora.yaml")
    assert lora["variant"] == "lora" and lora["recipe"] == TJ.RECIPE
    assert tuple(lora["lora_targets"]) == TJ.LORA_TARGETS
    head = _load(CFG / "train" / "jev-head.yaml")
    assert head["variant"] == "head" and head["base"] == "jev-lora" and head["recipe"] == JH.RECIPE_HEAD
    assert TJ.variant_recipe("head") == {**lora["recipe"], **head["recipe"]}
    for v in lora["recipe"].values():
        assert not isinstance(v, str) or v == "cosine"          # 1e-4 must not parse as a string


def test_dm_yaml_mirror_code():
    files = sorted(p.stem for p in (CFG / "dm").glob("jev-*.yaml"))
    assert files == sorted(JEV)
    for name in JEV:
        cfg = _load(CFG / "dm" / f"{name}.yaml")
        assert COMMON <= set(cfg) and cfg["dm"] == "jev", name
        assert cfg["revision"] == J.REVISIONS[cfg["backbone"]], name
        assert 1 <= cfg["batch_size"] <= J.MAX_BS and cfg["grouping"] in J.GROUPINGS and cfg["dtype"] in J.DTYPES
        assert cfg["cache_only"] is True and cfg["device"] == "cpu" and cfg["prefix_share"] is True
        assert cfg["layout"] == ("head" if name.endswith("-head") else "letter")
        assert cfg["questions"] == ("routed" if name == "jev-qwen3.5-4b" else "habit")
        assert cfg["grouping"] == ("direct" if cfg["questions"] == "routed" else "chunked")
        if cfg["layout"] == "letter":
            assert set(cfg) == COMMON | {"adapter"}, name
        else:
            assert set(cfg) == COMMON | {"run"}, name
        cls = D.BACKENDS[D.backend_key(cfg)]
        params = inspect.signature(cls.__init__).parameters
        renamed = {D.RENAMES.get(D.backend_key(cfg), {}).get(k, k) for k in cfg} - set(D.REGISTRY_KEYS)
        assert renamed <= set(params), (name, renamed - set(params))
    main = _load(CFG / "dm" / "jev-qwen3.5-4b.yaml")
    assert main["cache_tag"] == J.engine_tag(main["backbone"]) == "qwen3.5-4b" and main["adapter"] is None
    hbz = _load(CFG / "dm" / "jev-qwen3.5-4b-habit.yaml")
    assert hbz["cache_tag"] == J.habit_tag(hbz["backbone"]) and hbz["adapter"] is None
    for name, run in (("jev-qwen3.5-4b-lora", "HB"), ("jev-qwen3.5-0.8b-lora", "HB.08b")):
        c = _load(CFG / "dm" / f"{name}.yaml")
        assert c["adapter"] == f"runs/adapters/<fold>/{run}" and c["cache_tag"] == "auto"
    head = _load(CFG / "dm" / "jev-qwen3.5-4b-head.yaml")
    assert head["run"] == "runs/adapters/<fold>/HB.head" and head["cache_tag"] == "auto"
    assert _load(CFG / "dm" / "jev-qwen3.5-0.8b-lora.yaml")["backbone"] == TJ.FALLBACK_BACKBONE


def test_registry_builds_the_stock_jev_arms():
    for arm, tag in (("MAIN", "qwen3.5-4b"), ("HB-Z", "hb-qwen3.5-4b")):
        dm = D.get_arm(arm, ds="CADETS_E3")
        assert isinstance(dm, J.JevDecisionModel) and dm.tag == tag and dm.cache_only and dm.device == "cpu"
        assert dm.cache.read_only and dm.cache.ds == "CADETS_E3"
        assert str(dm.cache.overlay).endswith(f"answers/{tag}/CADETS_E3.jsonl")
        assert dm.grouping == ("direct" if arm == "MAIN" else "chunked") and dm.prefix_share
        dm.close()
    assert set(D.ARMS) == {"MAIN", "HB-Z", "HB", "HB.head", "HB.08b", "HB-L.s0", "HB-L", "HB-L.mq", "HB-L.zs"}
    assert D.backend_key({"dm": "jev", "layout": "head"}) == "jev-head"
    assert D.backend_key({"dm": "laya", "layout": "mq"}) == "laya-mq"
    with pytest.raises(ValueError):
        D.backend_key({"dm": "jev", "layout": "la"})
    with pytest.raises(KeyError):
        D.BACKENDS["nope"]


def test_registry_fold_safety(tmp_path, monkeypatch):
    from habitus.config import load_settings
    monkeypatch.setenv("HABITUS_RUNS", str(tmp_path / "runs"))
    s = load_settings()
    with pytest.raises(ValueError, match="no habitus.json"):
        D.get_arm("HB", ds="THEIA_E3", settings=s)
    run = tmp_path / "runs" / "adapters" / "loho-THEIA_E3" / "HB"
    (run / "adapter").mkdir(parents=True)
    (run / "habitus.json").write_text('{"fold": "loho-CADETS_E3", "backbone": "Qwen/Qwen3.5-4B"}')
    with pytest.raises(ValueError, match="trained for fold"):
        D.get_arm("HB", ds="THEIA_E3", settings=s)
    with pytest.raises(ValueError, match="trains on THEIA_E3"):
        D.get_arm("HB", ds="THEIA_E3", fold="loho-CADETS_E3", settings=s)
    assert D.fold_for("THEIA_E3") == "loho-THEIA_E3" and D.fold_for("CLEARSCOPE_E3") == "all"
    D.check_fold_scores("all", "CLEARSCOPE_E3")
    (run / "habitus.json").write_text('{"fold": "loho-THEIA_E3", "backbone": "Qwen/Qwen3.5-4B"}')
    (run / "adapter" / "adapter_config.json").write_text('{"base_model_name_or_path": "Qwen/Qwen3.5-4B"}')
    (run / "adapter" / "adapter_model.safetensors").write_bytes(b"x")
    dm = D.get_arm("HB", ds="THEIA_E3", settings=s)
    h = J.adapter_hash(str(run / "adapter"))
    assert dm.tag == f"hb-qwen3.5-4b+HB-{h[:12]}" and dm.adapter == str(run / "adapter") and dm.cache_only
