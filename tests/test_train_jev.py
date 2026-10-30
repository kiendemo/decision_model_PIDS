"""train.jev on a tiny random Qwen3.5 (fp32, CPU): LoRA and LoRA + head round trips, after which the
decision models load the run; fold checks; batching."""

import json
import shutil

import numpy as np
import pytest

import dm_helpers as H
from habitus.dm import jev as J
from habitus.dm import jev_head as JH
from habitus.dm.cache import AnswerCache
from habitus.train import jev as TJ

FOLD = "loho-CADETS_E3"


@pytest.fixture(scope="module")
def mp():
    m = pytest.MonkeyPatch()
    yield m
    m.undo()


@pytest.fixture(scope="module")
def backbone(tmp_path_factory, mp):
    """A tiny random Qwen3.5 saved with the pinned tokenizer's files: a local backbone directory."""
    try:
        src = J.tokenizer_source()
    except FileNotFoundError as e:          # no bundle tokenizer and no local model snapshot
        H.missing(f"Qwen3.5-4B tokenizer not available: {e}")
    if not src:
        H.missing("Qwen3.5-4B tokenizer not available")
    d = tmp_path_factory.mktemp("tiny_backbone")
    H.tiny_qwen(mp).save_pretrained(str(d))
    for name in J.TOKENIZER_FILES:
        if name != "config.json":
            shutil.copy(src / name, d / name)
    return str(d)


@pytest.fixture(scope="module")
def tok(backbone):
    return J.load_tokenizer(backbone, None)


def _questions(n, prefix):
    qs = []
    for i in range(n):
        view = H.state(i, extra=i % 3)
        qs.append(H.habit_question(f"{prefix}{i}|P2|dir:staging", view, "Does it modify staging today?",
                                   ("Yes, it modifies staging.", "No, it does not.")))
    return qs


def _items(dm, qs, role, ds="THEIA_E3"):
    from habitus.questions.habit.types import Item
    out = []
    for i, q in enumerate(qs):
        view_sha1 = q.qkey.split("|")[0]
        y = 0.9 if i % 2 == 0 else 0.1
        for f in q.forms:
            tgt = (y, 1 - y) if f.name == "fwd" else (1 - y, y)
            out.append(Item(ds=ds, day="2018-04-05", ref_days=("2018-04-03", "2018-04-04"), k_mode="k2", sig=f"s{i}",
                            family="P2", bucket="dir:staging", view_sha1=view_sha1, qkey=q.qkey, variant=f.name,
                            prompt_text=dm.render(f), letters=("A", "B"), target=tgt, weight=1.0 + i % 2, role=role,
                            fold=FOLD))
    return out


RECIPE = {"bf16": False, "batch_size": 4, "micro_batch": 2, "lr": 5e-3, "warmup": 0.0}


@pytest.fixture(scope="module")
def data(backbone, tok):
    dm = J.JevDecisionModel(backbone, None, cache_only=True, cache=AnswerCache("t", None), verbose=False)
    tq, iq = _questions(6, "t"), _questions(3, "i")
    return tq, iq, _items(dm, tq, "train"), _items(dm, iq, "inner")


def _probs(z):
    return [tuple(float(x) for x in J.softmax_t(v, 1.0)) for v in z]


def test_lora_round_trip_engine_loads_adapter(backbone, tok, data, tmp_path):
    from peft import PeftModel
    tq, iq, tr, inner = data
    out = tmp_path / "runs" / "adapters" / FOLD / "HB"
    rec = TJ.train_jev(tr, inner, str(out), backbone=backbone, revision=None, fold=FOLD, device="cpu", recipe=RECIPE)
    assert rec["fold"] == FOLD and rec["backbone"] == backbone and "variant" not in rec
    assert rec["adapter_hash"] == J.adapter_hash(str(out / "adapter")) and rec["n_items"] == 12 and rec["n_inner"] == 6
    assert set(rec["temperatures"]) == {"inclusion-2", "choice-4"} and rec["inner_report"]["n"] == 6
    assert rec["items_sha256"] == TJ.items_sha256(tr) and rec["steps"] == 3 and rec["recipe"]["lr"] == 5e-3
    assert json.loads((out / "habitus.json").read_text()) == rec
    with pytest.raises(FileExistsError):
        TJ.train_jev(tr, inner, str(out), backbone=backbone, revision=None, fold=FOLD, device="cpu", recipe=RECIPE)
    # the decision model loads backbone + adapter (merged) and answers like the trained model
    dm = J.JevDecisionModel(backbone, None, adapter=str(out), dtype="float32", cache=AnswerCache("t", None),
                            verbose=False)
    assert dm.tag == f"{J.engine_tag(backbone)}+HB" and dm.adapter_hash == rec["adapter_hash"]
    assert J.habit_tag(backbone, dm.adapter) == f"hb-{J.engine_tag(backbone)}+HB-{rec['adapter_hash'][:12]}"
    got = dm.decide(iq)
    base = J.load_model(backbone, None, None, "cpu", "float32")
    peft = PeftModel.from_pretrained(J.load_model(backbone, None, None, "cpu", "float32"), str(out / "adapter"))
    want = _probs(TJ.letter_logits(peft, tok, inner, "cpu"))
    stock = _probs(TJ.letter_logits(base, tok, inner, "cpu"))
    worst, moved = 0.0, 0.0
    for it, p, p0 in zip(inner, want, stock):
        d = got[it.qkey].letter_probs[it.variant][0]
        worst = max(worst, float(np.abs(np.subtract(d, p)).max()))
        moved = max(moved, float(np.abs(np.subtract(p, p0)).max()))
    assert worst < 1e-4 and moved > 1e-3, (worst, moved)


def test_head_round_trip_engine_loads_head(backbone, tok, data, tmp_path):
    from peft import PeftModel
    tq, iq, tr, inner = data
    out = tmp_path / "HB.head"
    rec = TJ.train_jev(tr, inner, str(out), backbone=backbone, revision=None, fold=FOLD, device="cpu",
                       recipe=RECIPE, variant="head")
    assert rec["variant"] == "head" and rec["head_hash"] == JH.head_hash(str(out / "head"))
    assert rec["recipe"]["head_lr"] == JH.RECIPE_HEAD["head_lr"] and rec["readout"]["readout_dtype"] == "fp32 head"
    with pytest.raises(ValueError, match="variant 'head'"):
        J.JevDecisionModel(backbone, None, adapter=str(out), cache=AnswerCache("t", None), verbose=False)
    dm = JH.JevHeadDecisionModel(str(out), backbone, None, dtype="float32", cache=AnswerCache("t", None), verbose=False)
    d = JH.variant_tag("head", backbone, dm.adapter, rec["adapter_hash"], rec["head_hash"])
    assert dm.tag == d and dm.tag.startswith(f"hb.head-{J.engine_tag(backbone)}+HB.head-")
    got = dm.decide(iq)
    peft = PeftModel.from_pretrained(J.load_model(backbone, None, None, "cpu", "float32"), str(out / "adapter"))
    ro = JH.HeadReadout.load(str(out / "head"))
    want = _probs(TJ.letter_logits(peft, tok, inner, "cpu", readout=ro))
    worst = max(float(np.abs(np.subtract(got[it.qkey].letter_probs[it.variant][0], p)).max())
                for it, p in zip(inner, want))
    assert worst < 1e-4, worst
    assert all(c == 1.0 for q in iq for c in got[q.qkey].cov.values())


def test_fold_and_role_checks(data):
    tq, iq, tr, inner = data
    from dataclasses import replace
    TJ.check_fold_items(tr + inner, FOLD)
    with pytest.raises(ValueError, match="not trainable"):
        TJ.check_fold_items([replace(tr[0], ds="CADETS_E3")], FOLD)          # the held-out host
    with pytest.raises(ValueError, match="not trainable"):
        TJ.check_fold_items([replace(tr[0], fold="all")], FOLD)
    with pytest.raises(ValueError, match="role"):
        TJ.check_fold_items([replace(tr[0], role="select")], FOLD)
    with pytest.raises(ValueError):
        TJ.check_fold_items(tr, "loho-NOT_A_HOST")
    TJ.check_fold_items([replace(tr[0], ds="CLEARSCOPE_E3", fold="")], FOLD)
    with pytest.raises(ValueError, match="role 'train'"):
        TJ._split(inner, inner)
    a, b = TJ._split(tr + inner, None)
    assert a == tr and b == inner


def test_length_buckets_and_recipes():
    lens = [5, 1, 9, 3, 7, 2, 8, 4, 6, 0] * 7
    b1, b2 = TJ.length_buckets(lens, 4, seed=0, megabatch=2), TJ.length_buckets(lens, 4, seed=0, megabatch=2)
    assert b1 == b2 and sorted(i for b in b1 for i in b) == list(range(len(lens))) and max(map(len, b1)) <= 4
    assert TJ.length_buckets(lens, 4, seed=1, megabatch=2) != b1
    for b in b1:
        assert sorted(lens[i] for i in b) == [lens[i] for i in b]
    assert TJ.variant_recipe("lora") == TJ.RECIPE
    assert TJ.variant_recipe("head", {"lr": 1.0}) == {**TJ.RECIPE, **JH.RECIPE_HEAD, "lr": 1.0}
    # a configs/train mapping is unwrapped (its 'recipe' entry); unknown keys raise
    import yaml
    for name, variant in (("jev-lora", "lora"), ("jev-head", "head")):
        cfg = yaml.safe_load((H.REPO / "configs" / "train" / f"{name}.yaml").read_text(encoding="utf-8"))
        assert TJ.variant_recipe(variant, cfg) == TJ.variant_recipe(variant, cfg["recipe"])
        assert not {"variant", "recipe", "base", "lora_targets"} & set(TJ.variant_recipe(variant, cfg))
    with pytest.raises(ValueError, match="unknown lora recipe keys"):
        TJ.variant_recipe("lora", {"lora_targets": ["q_proj"]})
    with pytest.raises(ValueError):
        TJ.variant_recipe("ptr")


def test_smoke_writes_nothing(backbone, tok, data, tmp_path, monkeypatch):
    tq, iq, tr, inner = data
    monkeypatch.chdir(tmp_path)
    res = TJ.smoke(tr, backbone, None, "cpu", n=8, recipe=RECIPE)
    assert res["n"] == 8 and res["steps"] == 2 and res["train_items_per_s"] > 0 and res["infer_items_per_s"] > 0
    assert list(tmp_path.iterdir()) == []
