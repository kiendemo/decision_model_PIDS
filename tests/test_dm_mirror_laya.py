"""configs/dm/laya-*.yaml and configs/train/laya*.yaml mirror the code (the code is the source of truth):
budgets, batching, layout version and run naming of the laya decision models, RECIPE_L / RECIPE_MQ
and the seed sets of the trainers. Every non-registry key of a laya config is a constructor argument
of its backend, so habitus.dm.get_dm can build it."""

import inspect
from pathlib import Path

import pytest
import yaml

from habitus.dm import laya as LA
from habitus.dm import laya_mq as MQ

REPO = Path(__file__).resolve().parents[1]
DM = REPO / "configs" / "dm"
TRAIN = REPO / "configs" / "train"
LA_CONFIGS = ("laya-typed-zs", "laya-typed-s0", "laya-typed-ens3")
ALL = LA_CONFIGS + ("laya-typed-mq",)
REGISTRY = ("dm", "layout", "questions", "role", "description", "layout_version")


def load(p):
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize("name", LA_CONFIGS)
def test_la_configs_mirror_the_code(name):
    c = load(DM / f"{name}.yaml")
    assert (c["dm"], c["layout"], c["questions"]) == ("laya", LA.LAYOUT, "habit")
    assert (c["max_len"], c["head_max_len"]) == (LA.MAX_LEN, LA.HEAD_MAX_LEN)
    assert (c["batch_rows"], c["max_tokens"]) == (LA.BATCH_ROWS, LA.MAX_TOKENS)
    assert c["init"] == LA.DEFAULT_INIT and c["cache_only"] is True and c["device"] == "cpu"


def test_run_naming_and_seed_sets():
    from habitus.train import laya as TL
    from habitus.train import laya_mq as TMQ
    assert load(DM / "laya-typed-zs.yaml")["runs"] is None
    assert load(DM / "laya-typed-s0.yaml")["runs"] == ["runs/adapters/<fold>/hbl/HB-L-s0"]
    ens = load(DM / "laya-typed-ens3.yaml")
    assert ens["runs"] == [f"runs/adapters/<fold>/hbl/HB-L-s{s}" for s in TL.SEEDS]
    assert ens["ensemble"] == "runs/adapters/<fold>/hbl/HB-L.ensemble.json"
    hbl = "runs/adapters/<fold>/hbl"
    assert [TL.run_dir(hbl, TL.ARM, s).endswith(f"{hbl}/HB-L-s{s}") for s in TL.SEEDS] == [True] * 3
    assert TL.ensemble_path(hbl, TL.ARM).endswith(f"{hbl}/HB-L.ensemble.json")
    assert TL.run_dir(hbl, TMQ.RUN_BASE, 0).endswith(f"{hbl}/HB-L.mq-s0")
    mq = load(DM / "laya-typed-mq.yaml")
    assert mq["runs"] == [f"runs/adapters/<fold>/hbl/HB-L.mq-s{s}" for s in load(TRAIN / "laya-mq.yaml")["seeds"]]
    assert set(load(TRAIN / "laya-mq.yaml")["seeds"]) <= set(TMQ.SEEDS)


def test_mq_config_mirrors_the_code():
    c = load(DM / "laya-typed-mq.yaml")
    assert (c["dm"], c["layout"], c["questions"]) == ("laya", MQ.LAYOUT, "habit")
    assert (c["max_len"], c["head_max_len"]) == (MQ.MAX_LEN, MQ.HEAD_MAX_LEN)
    assert (c["batch_rows"], c["batch_tokens"], c["row_tokens"]) == (MQ.BATCH_ROWS, MQ.BATCH_TOKENS, MQ.ROW_TOKENS)
    assert c["layout_version"] == MQ.LAYOUT_VERSION and c["cache_only"] is True and c["device"] == "cpu"


def test_train_configs_mirror_the_recipes():
    from habitus.train import laya as TL
    from habitus.train import laya_mq as TMQ
    la, mq = load(TRAIN / "laya.yaml"), load(TRAIN / "laya-mq.yaml")
    assert (la["trainer"], la["layout"], la["init"]) == ("laya", "la", LA.DEFAULT_INIT)
    assert (mq["trainer"], mq["layout"], mq["init"]) == ("laya", "mq", LA.DEFAULT_INIT)
    assert la["recipe"] == TL.RECIPE_L and list(la["recipe"]) == list(TL.RECIPE_L)
    assert mq["recipe"] == TMQ.RECIPE_MQ and list(mq["recipe"]) == list(TMQ.RECIPE_MQ)
    assert tuple(la["seeds"]) == TL.SEEDS


@pytest.mark.parametrize("name", ALL)
def test_every_config_key_is_a_constructor_argument(name):
    c = load(DM / f"{name}.yaml")
    cls = MQ.LayaMQDecisionModel if c["layout"] == "mq" else LA.LayaDecisionModel
    params = inspect.signature(cls.__init__).parameters
    unknown = sorted(k for k in c if k not in REGISTRY and k not in params)
    assert not unknown, unknown
    for k in ("ds", "settings", "cache_only", "device"):        # what habitus.dm.get_dm adds
        assert k in params
