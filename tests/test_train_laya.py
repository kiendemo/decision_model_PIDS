"""habitus.train.laya / habitus.train.laya_mq on CPU with tiny Laya-format checkpoints: the recipe,
schedule and parameter groups, temperatures (clamped fit + the unclamped optimum), the geometric-mean
ensemble, item rows (train row == serve row, never cut), the L-a and MQ train round trips (run
directories, run records, ensemble records, served back by the decision models) and the MQ trainer
hook (letter-order logits, batches cover every row once)."""

import json
import os
import re
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from habitus.dm import laya as LA  # noqa: E402
from habitus.dm import laya_mq as MQ  # noqa: E402
from habitus.questions.habit.constants import T_CLAMP  # noqa: E402
from habitus.train import calibrate as CAL  # noqa: E402
from habitus.train import jev as TJ  # noqa: E402
from habitus.train import laya as TL  # noqa: E402
from habitus.train import laya_mq as TMQ  # noqa: E402
from test_dm_laya import (DS, FOLD, habit, laya_tokenizer, make_items, make_questions, make_settings,  # noqa: E402
                          save_ckpt, tiny_model, typed)


@pytest.fixture(scope="module")
def tok():
    return laya_tokenizer()


@pytest.fixture(scope="module")
def ckpt(tok, tmp_path_factory):
    return save_ckpt(tmp_path_factory.mktemp("tl") / "tiny-0", tok, 0)


@pytest.fixture
def st(tmp_path):
    return make_settings(tmp_path / "runs")


# --------------------------------------------------------------------------- #
# Recipe, schedule, groups
# --------------------------------------------------------------------------- #
def test_recipe_schedule_and_param_groups(tok):
    r = TL.RECIPE_L
    assert (r["encoder_lr"], r["head_lr"], r["weight_decay"], r["warmup"], r["batch_size"], r["epochs"]) == \
        (2.5e-5, 1e-4, 0.01, 0.03, 64, 1)
    assert r["bf16"] and r["grad_ckpt"] and r["loss"] == "soft-ce" and (r["max_len"], r["head_max_len"]) == (2048, 256)
    assert r["qtype"] == "choice" and TL.SEEDS == (0, 1, 2)
    f = TL.lr_factor(100, r, r["encoder_lr"])
    assert f(0) == pytest.approx(1 / 3) and f(2) == pytest.approx(1.0) and f(3) == pytest.approx(1.0)
    assert f(100) == pytest.approx(1e-6 / 2.5e-5) and f(50) < 1.0
    g = TL.param_groups(tiny_model(tok), r)
    assert [x["lr"] for x in g] == [2.5e-5, 1e-4] and len(g[0]["params"]) > 0 and len(g[1]["params"]) > 0
    enc = [{"ids": [0] * n} for n in (5, 50, 7, 300, 9)]
    mbs = TL.micro_batches(range(5), enc, {**r, "micro_batch": 2, "max_tokens": 120})
    assert sorted(i for m in mbs for i in m) == list(range(5)) and all(len(m) <= 2 for m in mbs)
    assert all(len(m) * max(len(enc[i]["ids"]) for i in m) <= 120 or len(m) == 1 for m in mbs)


def test_recipe_parity_mq_vs_la():
    """RECIPE_MQ = RECIPE_L except gradient checkpointing (off) plus the MQ keys; same seeds."""
    diff = {k for k in TL.RECIPE_L if TMQ.RECIPE_MQ.get(k) != TL.RECIPE_L[k]}
    assert diff == {"grad_ckpt"} and TMQ.RECIPE_MQ["grad_ckpt"] is False
    assert set(TMQ.RECIPE_MQ) - set(TL.RECIPE_L) == set(TMQ.MQ_KEYS) and TMQ.SEEDS == TL.SEEDS
    assert (TMQ.RECIPE_MQ["layout"], TMQ.RECIPE_MQ["row_tokens"], TMQ.RECIPE_MQ["head"]) == ("mq", 2048, "row")


# --------------------------------------------------------------------------- #
# Temperatures and the ensemble
# --------------------------------------------------------------------------- #
def _synthetic(form, n, t0, rng):
    T, _X = habit()
    k = 2 if form == "inclusion-2" else 4
    fam = T.Family.P2_modify if k == 2 else T.Family.P1_parent
    z = rng.normal(scale=3.0, size=(n, k))
    tg = CAL._softmax(z, t0)
    items = []
    for i in range(n):
        bucket = "dir:home" if k == 2 else T.NO_BUCKET
        sha = f"{i:040x}"
        items.append(T.Item(DS, "2018-04-05", ("2018-04-02",), "k2", "s", fam, bucket, sha,
                            T.qkey(sha, fam, None if k == 4 else bucket), "fwd", "p" + LA.THINK_SUFFIX,
                            LA.LETTERS[:k], tuple(tg[i] / tg[i].sum()), weight=1.0 + (i % 3), role="inner"))
    return list(z), items


def test_temperatures_keep_the_clamp_and_record_the_raw_optimum():
    T, _X = habit()
    rng = np.random.default_rng(0)
    z2, i2 = _synthetic("inclusion-2", 400, 0.4, rng)        # under-confident: T* below the clamp
    z4, i4 = _synthetic("choice-4", 400, 1.5, rng)
    temps, raw, clamped, rep = TL.calibrate(z2 + z4, i2 + i4)
    assert temps == CAL.fit_temperatures(z2 + z4, i2 + i4) and temps["inclusion-2"] == T_CLAMP[0]
    assert abs(raw["inclusion-2"] - 0.4) < 0.01 and abs(raw["choice-4"] - 1.5) < 0.02
    assert clamped == {"inclusion-2": True, "choice-4": False} and rep["n"] == 800
    assert TL.laya_buckets(temps) == {"choice:2": temps["inclusion-2"], "choice:3-5": temps["choice-4"]}
    assert TL.raw_temperatures(z2, i2)["choice-4"] is None
    with pytest.raises(ValueError, match="inner-day"):
        TL.raw_temperatures(z2[:2], [T.Item(**{**i2[0].__dict__, "role": "train"})] * 2)


def test_geo_logits_is_the_pbar_of_the_seeds():
    T, _X = habit()
    rng = np.random.default_rng(1)
    a, b = [rng.normal(size=4) for _ in range(5)], [rng.normal(size=4) for _ in range(5)]
    g = TL.geo_logits([a, b])
    for i in range(5):
        want = T.pbar([LA.softmax(a[i]), LA.softmax(b[i])])
        assert np.allclose(LA.softmax(g[i]), want, atol=1e-12)
    with pytest.raises(ValueError):
        TL.geo_logits([a, b[:4]])


# --------------------------------------------------------------------------- #
# Items -> rows
# --------------------------------------------------------------------------- #
def test_encode_items_and_length_errors(tok):
    qs = make_questions(2)
    items = make_items(qs)
    enc = TL.encode_items(tok, items)
    tq = {q.qkey: q for q in typed(qs)}
    for e, it in zip(enc, items):
        assert e["row"] == LA.seq_item(tok, it) and e["target"] == list(it.target) and e["w"] == it.weight
        f = [x for x in tq[it.qkey].forms if x.name == it.variant][0]
        assert e["row"] == LA.seq_form(tok, f, "choice")                 # train row == serve row
    with pytest.raises(LA.LengthError, match=re.escape(items[0].qkey)):
        TL.encode_items(tok, items[:1], {**TL.RECIPE_L, "head_max_len": 10})
    c = TL.length_census([e["row"] for e in enc])
    assert c["n"] == len(items) and c["rows_option_over_48"] >= 1 and c["option_max"] > 48


def test_mq_item_rows_letter_logits_and_temperatures(tok):
    T, _X = habit()
    model = tiny_model(tok, 4)
    qs = make_questions(3)
    items = make_items(qs, both=True)
    rows = TMQ.item_rows(tok, items)
    views = {it.view_sha1 for it in items}
    assert len(rows) == len(views) and sorted(i for _l, idx in rows for i in idx) == list(range(len(items)))
    z = TMQ.letter_logits(model, tok, items)
    by = {}
    for it, zz in zip(items, z):
        assert zz.shape == (len(it.letters),)
        by.setdefault(it.qkey, {})[it.variant] = zz
    for q in typed(qs):                       # rev item = fwd item reversed = the canonical serve logits
        state, blk = MQ.encode_parts(tok, MQ.canonical_parts(q.forms[0]), q.qtype)
        with torch.no_grad():
            serve = MQ.forward(model, MQ.collate([MQ.row_layout(state, [blk])], tok.pad_token_id))[0, :blk.n]
        serve = serve.double().numpy()
        assert np.allclose(by[q.qkey]["fwd"], serve, atol=1e-4)
        assert np.allclose(by[q.qkey]["rev"][::-1], serve, atol=1e-4)
    inner = make_items(qs, role="inner", day="2018-04-05", ref=("2018-04-02",), seed=1)
    zi = TMQ.letter_logits(model, tok, inner)
    temps, raw, clamped, rep = TL.calibrate(zi, inner)
    lo, hi = T_CLAMP
    for f in CAL.FORMS:
        if raw[f] is not None:
            assert clamped[f] == (not lo <= raw[f] <= hi)
    assert rep["n"] == len(inner)
    assert TMQ.rec_logits(model, tok, inner, "cpu", TMQ.RECIPE_MQ)[0].shape == zi[0].shape
    d = items[0].to_json()
    d["view_sha1"] = items[-1].view_sha1
    d["qkey"] = T.qkey(d["view_sha1"], items[0].family, items[0].bucket)
    with pytest.raises(ValueError):
        TMQ.item_rows(tok, [T.Item.from_json(d)])          # the prompt's view must hash to view_sha1


def test_mq_train_batches_cover_every_row_once(tok):
    items = make_items(make_questions(6))
    rows = TMQ.item_rows(tok, items)
    bs = TMQ.train_batches(rows, 8, seed=0, megabatch=2)
    assert sorted(i for b in bs for i in b) == list(range(len(rows)))
    assert bs == TMQ.train_batches(rows, 8, seed=0, megabatch=2)
    for b in bs:
        mbs = TMQ.micro_batches(rows, b, 1500, 2)
        assert sorted(i for mb in mbs for i in mb) == sorted(b) and all(len(mb) <= 2 for mb in mbs)


# --------------------------------------------------------------------------- #
# Train round trips
# --------------------------------------------------------------------------- #
def test_train_laya_round_trip_and_the_models_serve_the_runs(tok, ckpt, st, tmp_path):
    T, _X = habit()
    qs = make_questions(4)
    tr = make_items(qs, seed=0)
    inner = make_items(qs, role="inner", day="2018-04-05", ref=("2018-04-02",), seed=1)
    rec = {"batch_size": 4, "micro_batch": 2, "max_tokens": 4096, "grad_ckpt": False}
    hbl = Path(str(st.paths.runs)) / "adapters" / FOLD / "hbl"
    base = hbl / "HB-L"
    with pytest.raises(ValueError, match="layout"):
        TL.train_laya(tr, inner, hbl, init=ckpt, fold=FOLD, layout="xx", seeds=(0,), recipe=rec, settings=st)
    with pytest.raises(ValueError, match="roles"):
        TL.train_laya(inner, tr, hbl, init=ckpt, fold=FOLD, seeds=(0,), recipe=rec, settings=st)
    with pytest.raises(ValueError, match="unknown recipe keys"):
        TL.train_laya(tr, inner, hbl, init=ckpt, fold=FOLD, seeds=(0,), recipe={"lr": 1.0}, settings=st)
    held = make_items(qs[:2], ds="optc_h501", day="2019-09-20", ref=("2019-09-19",))
    with pytest.raises(ValueError, match="not trainable"):
        TL.train_laya(held, [], hbl, init=ckpt, fold=FOLD, seeds=(0,), recipe=rec, settings=st)
    from habitus.config import ConfigError
    ro = make_settings(Path(str(st.paths.runs)), paths={"runs": str(st.paths.runs), "laya_src": str(tmp_path / "ro")})
    with pytest.raises(ConfigError, match="read-only"):
        TL.train_laya(tr, inner, tmp_path / "ro", init=ckpt, fold=FOLD, seeds=(0,), recipe=rec, settings=ro)
    assert not (tmp_path / "ro").exists()
    yaml_rec = {"trainer": "laya", "layout": "la", "seeds": [0, 1], "recipe": rec}      # a train-config mapping
    res = TL.train_laya(tr, inner, hbl, init=ckpt, fold=FOLD, seeds=(0, 1), recipe=yaml_rec, settings=st)
    assert len(res["runs"]) == 2 and res["ensemble"] is not None
    dirs = [f"{base}-s{s}" for s in (0, 1)]
    from safetensors.torch import load_file
    init = load_file(os.path.join(ckpt, LA.WEIGHTS))
    for d, r in zip(dirs, res["runs"]):
        for f in (LA.WEIGHTS, LA.RUN_CONFIG, LA.LAYA_CONFIG, os.path.join("encoder", "config.json"),
                  os.path.join("tokenizer", "tokenizer.json")):
            assert os.path.isfile(os.path.join(d, f)), f
        rj = json.load(open(os.path.join(d, LA.RUN_CONFIG)))
        assert rj == json.loads(json.dumps(r))
        assert rj["model_sha256"] == LA.model_sha256(d) and rj["layout"] == "la" and rj["arm"] == "HB-L"
        assert set(rj["temperatures"]) == set(CAL.FORMS) and set(rj["raw_T"]) == set(CAL.FORMS)
        assert set(rj["laya_buckets"]) == {"choice:2", "choice:3-5"} and rj["init"] == "tiny-0"
        assert rj["items_sha256"] == TJ.items_sha256(tr) and rj["inner_sha256"] == TJ.items_sha256(inner)
        assert rj["recipe"]["loss"] == "soft-ce" and rj["fold"] == FOLD and rj["n_items"] == len(tr)
        assert rj["inner_report"]["n"] == len(inner) and np.isfinite(rj["loss"]["mean"])
        assert rj["full"] is False and rj["kd"] is None and rj["spec"] == "HB-v0-draft"
        lc = json.load(open(os.path.join(d, LA.LAYA_CONFIG)))
        assert lc["temperature_by_options"] == rj["laya_buckets"] and lc["temperature"] == [1.0, 1.0, 1.0]
        assert (lc["max_len"], lc["head_max_len"]) == (2048, 256) and lc["habitus_run"] == os.path.basename(d)
        w = load_file(os.path.join(d, LA.WEIGHTS))
        assert set(w) == set(init)
        assert torch.equal(w["act_head.0.weight"], init["act_head.0.weight"])          # never run, never trained
        assert not torch.equal(w["scorer.3.weight"], init["scorer.3.weight"])
    w0, w1 = (load_file(os.path.join(d, LA.WEIGHTS)) for d in dirs)
    assert not torch.equal(w0["scorer.3.weight"], w1["scorer.3.weight"])               # seeds differ
    ens = json.load(open(f"{base}.ensemble.json"))
    assert [m["seed"] for m in ens["members"]] == [0, 1] and set(ens["temperatures"]) == set(CAL.FORMS)
    assert ens["inner_sha256"] == TJ.items_sha256(inner)
    with pytest.raises(FileExistsError):
        TL.train_laya(tr, inner, hbl, init=ckpt, fold=FOLD, seeds=(0,), recipe=rec, settings=st)
    assert res["runs"][0]["recipe"] == {**TL.RECIPE_L, **rec, "seed": 0}
    dm = LA.LayaDecisionModel(dirs, settings=st, fold=FOLD)
    assert dm.tag.startswith("hbl-tiny-0-") and [m.label for m in dm.members] == ["tiny-0", "tiny-0"]
    out = dm.decide(typed(qs))
    assert set(out) == {q.qkey for q in qs} and all(d.members == 2 for d in out.values())
    dm.close()
    with pytest.raises(ValueError, match="fold"):
        LA.LayaDecisionModel(dirs, settings=st, fold="loho-CADETS_E3")
    with pytest.raises(ValueError, match="head_max_len"):
        LA.LayaDecisionModel(dirs, settings=st, head_max_len=200)
    rep = TL.ensemble_report(dirs, inner, FOLD, "cpu", settings=st)
    assert rep["inner_sha256"] == ens["inner_sha256"]
    assert json.load(open(f"{base}.ensemble.json"))["written"] == rep["written"]
    assert rep["temperatures"] == pytest.approx(ens["temperatures"], abs=1e-6)


def test_train_mq_round_trip(tok, ckpt, st):
    qs = make_questions(4)
    items = make_items(qs, both=True)
    inner = make_items(qs[:12], role="inner", day="2018-04-05", ref=("2018-04-02",), seed=2)
    recipe = {"epochs": 3, "batch_size": 8, "encoder_lr": 1e-3, "head_lr": 1e-3, "warmup": 0.0}
    hbl = Path(str(st.paths.runs)) / "adapters" / FOLD / "hbl"
    base = hbl / "HB-L.mq"
    out = TL.train_laya(items, inner, hbl, init=ckpt, fold=FOLD, layout="mq", seeds=(0, 1), recipe=recipe,
                        settings=st)
    assert len(out["runs"]) == 2 and out["ensemble"]["layout"] == "mq" and len(out["ensemble"]["members"]) == 2
    assert os.path.isfile(f"{base}.ensemble.json")
    rec, d = out["runs"][0], f"{base}-s0"
    for f in (LA.WEIGHTS, LA.RUN_CONFIG, LA.LAYA_CONFIG, os.path.join("encoder", "config.json"),
              os.path.join("tokenizer", "tokenizer.json")):
        assert os.path.exists(os.path.join(d, f)), f
    assert json.load(open(os.path.join(d, LA.RUN_CONFIG))) == json.loads(json.dumps(rec))
    assert rec["layout"] == "mq" and rec["layout_version"] == "mq1" and rec["arm"] == "HB-L/MQ"
    assert rec["fold"] == FOLD and rec["init"] == "tiny-0" and rec["recipe"]["grad_ckpt"] is False
    assert rec["model_sha256"] == LA.file_sha256(os.path.join(d, LA.WEIGHTS)) != LA.model_sha256(ckpt)
    assert out["runs"][1]["model_sha256"] != rec["model_sha256"]                  # seeds differ
    assert rec["items_sha256"] == TJ.items_sha256(items) and rec["inner_sha256"] == TJ.items_sha256(inner)
    assert rec["items_per_row"] > 1 and rec["loss"]["last10pct"] < rec["loss"]["first10pct"]
    with pytest.raises(ValueError):
        LA.resolve_init(d, st)                                   # an MQ run is never read as L-a
    assert MQ.resolve_mq(d, st) == (d, "tiny-0")
    dm = MQ.LayaMQDecisionModel(d, settings=st, fold=FOLD)
    a = dm.decide(typed(qs[:3]))
    assert dm.tag == MQ.mq_tag("tiny-0", rec["model_sha256"]) and len(a) == 3
    assert dm.spec()["layout"] == "mq" and dm.kw == {"max_len": MQ.MAX_LEN, "head_max_len": MQ.HEAD_MAX_LEN}
    dm.close()
    with pytest.raises(FileExistsError):
        TMQ.train_one(items, inner, d, ckpt, FOLD, "cpu", 0, recipe, max_steps=1, settings=st)
    rep = TL.ensemble_report([f"{base}-s0", f"{base}-s1"], inner, FOLD, "cpu", write=False, settings=st)
    assert rep["layout"] == "mq" and rep["temperatures"] == pytest.approx(out["ensemble"]["temperatures"], abs=1e-6)
