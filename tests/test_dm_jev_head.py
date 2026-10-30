"""JevHeadDecisionModel (HB.head): the fp32 letter head equals the fp32 letter readout at step 0,
shared == full forward, coverage 1, tags, head files."""

import hashlib
import json

import numpy as np
import pytest

import dm_helpers as H
from habitus.dm import jev_head as JH
from habitus.dm.cache import AnswerCache


@pytest.fixture(scope="module")
def tok():
    return H.qwen_tokenizer()


@pytest.fixture(scope="module")
def tiny():
    mp = pytest.MonkeyPatch()
    try:
        yield H.tiny_qwen(mp)
    finally:
        mp.undo()


def _questions():
    qs = []
    for i in range(4):
        view = H.state(i, extra=i)
        qs.append(H.habit_question(f"v{i}|P1|-", view, "Which parent is it?", ("usual", "other", "never", "none")))
        qs.append(H.habit_question(f"v{i}|P2|dir:staging", view, "Does it modify staging?", ("yes", "no")))
    return qs


def _head(tok, tiny, **kw):
    kw.setdefault("verbose", False)
    dm = JH.JevHeadDecisionModel(None, model=tiny, cache=AnswerCache("t", None), **kw)
    dm._tok = tok
    return dm


def test_head_at_step0_equals_fp32_letter_readout(tok, tiny):
    qs = _questions()
    head = _head(tok, tiny, prefix_share=False).decide(qs)
    letter = H.make_dm(tok, model=tiny, prefix_share=False).decide(qs)
    worst = 0.0
    for q in qs:
        d, e = head[q.qkey], letter[q.qkey]
        assert d.layout == "head" and e.layout == "letter"
        assert all(c == 1.0 for c in d.cov.values()) and d.cov_min == 1.0
        for f in q.forms:
            worst = max(worst, float(np.abs(np.subtract(d.letter_probs[f.name][0], e.letter_probs[f.name][0])).max()))
    assert worst < 1e-5, worst


def test_head_shared_equals_full(tok, tiny):
    qs = _questions()
    full = _head(tok, tiny, prefix_share=False).decide(qs)
    for grouping in ("chunked", "direct"):
        shared = _head(tok, tiny, prefix_share=True, grouping=grouping, batch_size=4).decide(qs)
        for q in qs:
            for f in q.forms:
                assert np.abs(np.subtract(shared[q.qkey].letter_probs[f.name][0],
                                          full[q.qkey].letter_probs[f.name][0])).max() < 1e-4


def test_letter_head_init_and_files(tok, tiny, tmp_path):
    import torch
    lid = [tok.convert_tokens_to_ids(L) for L in "ABCD"]
    head = JH.LetterHead.from_lm_head(tiny.lm_head, lid)
    assert torch.equal(head.W0, tiny.lm_head.weight.detach()[lid].float())
    assert float(head.A.detach().abs().sum()) == 0.0 and float(head.b.detach().abs().sum()) == 0.0
    head.fit_norm(torch.randn(8, head.W0.shape[1]))
    with pytest.raises(ValueError):
        head.fit_norm(torch.randn(1, head.W0.shape[1]))
    ro = JH.HeadReadout(head)
    hd = tmp_path / "head"
    hd.mkdir()
    digest = ro.save(str(hd))
    want = hashlib.sha256()
    for name in (JH.HEAD_CONFIG, JH.HEAD_WEIGHTS):
        want.update(f"{name}\x00{hashlib.sha256((hd / name).read_bytes()).hexdigest()}\n".encode())
    assert digest == want.hexdigest() == JH.head_hash(str(hd))
    back = JH.HeadReadout.load(str(hd))
    assert back.digest == digest
    assert all(torch.equal(a, b) for a, b in zip(back.module.state_dict().values(), head.state_dict().values()))
    assert json.loads((hd / JH.HEAD_CONFIG).read_text())["variant"] == "head"
    with pytest.raises(FileNotFoundError):
        JH.head_hash(str(tmp_path))


def test_variant_tag_and_run_checks(tmp_path):
    assert JH.variant_tag("head", "Qwen/Qwen3.5-4B") == "hb.head-qwen3.5-4b"
    a, h = "a" * 64, "b" * 64
    d = hashlib.sha256(f"{a}\x00{h}".encode()).hexdigest()
    tag = JH.variant_tag("head", "Qwen/Qwen3.5-4B", "/r/HB.head/adapter", a, h)
    assert tag == f"hb.head-qwen3.5-4b+HB.head-{d[:12]}"
    assert JH.variant_tag("head", "Qwen/Qwen3.5-4B", "/r/HB.head/adapter", a) == f"hb.head-qwen3.5-4b+HB.head-{a[:12]}"
    with pytest.raises(ValueError):
        JH.variant_tag("ptr", "Qwen/Qwen3.5-4B")
    run = tmp_path / "HB"
    run.mkdir()
    (run / "habitus.json").write_text(json.dumps({"fold": "all", "backbone": "Qwen/Qwen3.5-4B"}))
    with pytest.raises(ValueError, match="not an HB.head run"):
        JH.JevHeadDecisionModel(str(run), cache=AnswerCache("t", None), cache_only=True, verbose=False)
    stock = JH.JevHeadDecisionModel(None, cache=AnswerCache("t", None), cache_only=True, verbose=False)
    assert stock.tag == "hb.head-qwen3.5-4b" and stock.layout == "head" and stock.family == "jev"
    s = stock.spec()
    assert s["variant"] == "head" and s["readout_dtype"] == "fp32 head" and s["readout"]["variant"] == "head"


def test_dm_modules_import_without_torch():
    code = ("import sys\n"
            "import habitus.dm, habitus.dm.base, habitus.dm.cache, habitus.dm.gpu, habitus.dm.jev\n"
            "import habitus.dm.jev_head\n"
            "import habitus.dm.status, habitus.train, habitus.train.calibrate, habitus.train.jev\n"
            "print(sorted(m for m in sys.modules if m == 'torch' or m.startswith(('torch.', 'transformers', 'peft'))))")
    r = H.run_python(code)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip().splitlines()[-1] == "[]"
