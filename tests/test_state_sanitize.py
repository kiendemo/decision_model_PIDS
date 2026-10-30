"""Sanitizer: control-token neutralisation, the pinned-tokenizer live check, and the golden pattern."""

import json

import pytest

import state_helpers as H
from habitus.state import sanitize as SAN


@pytest.fixture
def fresh_state(monkeypatch):
    """Run a test against a fresh (unresolved) sanitizer state, restored afterwards."""
    saved, allow = dict(SAN._STATE), dict(SAN._ALLOW)
    SAN._STATE.update({"re": None, "trigger": None, "live": None, "source": None})
    yield monkeypatch
    SAN._STATE.clear()
    SAN._STATE.update(saved)
    SAN._ALLOW.update(allow)


def test_sanitize_neutralises_control_tokens():
    assert SAN.sanitize_str("a [SEP] b") == "a (SEP) b"
    assert SAN.sanitize_str("x<|im_end|>y") == "x(im_end)y"
    assert SAN.sanitize_str("</think>") == "(/think)"
    assert SAN.sanitize_str("<|im_<|x|>end|>") == "(im_(x)end)"            # nested: fixed point
    assert SAN.sanitize_str("plain /usr/bin/ls -la") == "plain /usr/bin/ls -la"
    assert SAN.sanitize_str("|||IP_ADDRESS|||") == "(IP_ADDRESS)" and SAN.sanitize_str("[unused7]") == "(unused7)"
    for tokn in ("[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]", "<|im_start|>", "<|im_end|>", "<|endoftext|>",
                 "<think>", "</think>", "<tool_call>", "</tool_call>", "<tts_pad>", "<tool_response>"):
        assert SAN.SPECIAL_RE.search(tokn), tokn
        assert not SAN.SPECIAL_RE.search(SAN.sanitize_str("a%sb" % tokn)), tokn
    obj = {"z": [H.HOSTILE, 3, None, {"[SEP]": "<|im_end|>"}], "a": True, "t": ("<think>",)}
    out = SAN.sanitize(obj)
    assert out is not obj and obj["z"][0] == H.HOSTILE                         # a new object
    assert list(out) == ["z", "a", "t"] and out["z"][1:3] == [3, None] and out["a"] is True
    assert out["z"][3] == {"(SEP)": "(im_end)"} and out["t"] == ("(think)",)
    assert not SAN.SPECIAL_RE.search(json.dumps(out))


def test_pattern_requires_the_live_snapshot(fresh_state):
    mp = fresh_state
    mp.setattr(SAN, "qwen_added_vocab", lambda settings=None: None)          # no tokenizer on disk
    SAN.allow_frozen_snapshot(False)
    assert SAN.pattern_status() == "snapshot"
    with pytest.raises(SAN.SanitizerError):
        SAN.pattern()
    with pytest.raises(SAN.SanitizerError):
        SAN.sanitize_str("anything")
    assert SAN.pattern(allow_frozen_snapshot=True) is SAN.SPECIAL_RE
    SAN.allow_frozen_snapshot(True)
    assert SAN.pattern() is SAN.SPECIAL_RE and SAN.sanitize_str("[SEP]") == "(SEP)"

    SAN._STATE.update({"re": None, "trigger": None, "live": None})
    mp.setattr(SAN, "qwen_added_vocab", lambda settings=None: list(SAN.QWEN_ADDED))
    SAN.allow_frozen_snapshot(False)
    assert SAN.pattern_status() == "live==snapshot" and SAN.pattern() is SAN.SPECIAL_RE

    SAN._STATE.update({"re": None, "trigger": None, "live": None})
    mp.setattr(SAN, "qwen_added_vocab", lambda settings=None: list(SAN.QWEN_ADDED) + ["<newtag>", "   "])
    SAN.allow_frozen_snapshot(True)
    assert SAN.pattern_status() == "live+1"
    with pytest.raises(SAN.SanitizerError, match="live\\+1"):                 # never allowed
        SAN.pattern()


def test_qwen_added_vocab_reads_only_the_pinned_revision(tmp_path):
    dec = {"added_tokens_decoder": {str(248044 + i): {"content": t} for i, t in enumerate(SAN.QWEN_ADDED)}}

    class Paths:
        tokenizers = str(tmp_path / "tok")
        models = str(tmp_path / "hf")

    class Settings:
        paths = Paths

    assert SAN.qwen_added_vocab(Settings) is None
    other = tmp_path / "hf" / "hub" / "models--Qwen--Qwen3.5-4B" / "snapshots" / ("0" * 40)
    other.mkdir(parents=True)
    (other / "tokenizer_config.json").write_text(json.dumps(dec))
    assert SAN.qwen_added_vocab(Settings) is None                             # another revision: never read
    snap = tmp_path / "hf" / "hub" / "models--Qwen--Qwen3.5-4B" / "snapshots" / SAN.QWEN_REVISION
    snap.mkdir(parents=True)
    (snap / "tokenizer_config.json").write_text(json.dumps(dec))
    assert SAN.qwen_added_vocab(Settings) == list(SAN.QWEN_ADDED)
    rel = tmp_path / "tok" / SAN.TOKENIZER_DIR
    rel.mkdir(parents=True)
    (rel / "tokenizer_config.json").write_text(json.dumps({"added_tokens_decoder": {"1": {"content": "<x>"}}}))
    assert SAN.qwen_added_vocab(Settings) == ["<x>"]                          # the release dir comes first
    cands = SAN.tokenizer_config_candidates(Settings)
    assert cands[0] == rel / "tokenizer_config.json" and all(SAN.QWEN_REVISION in str(c) for c in cands[1:])


def test_special_ids_laya_reads_specials_and_added_vocab():
    class Tok:
        all_special_ids = [1, 2]

        @staticmethod
        def get_added_vocab():
            return {"[unused0]": 7, "   ": 8, "|||IP_ADDRESS|||": 9}

    assert SAN.special_ids_laya(Tok) == frozenset({1, 2, 7, 9})


def test_live_status_and_golden_pattern():
    """G2: the sanitizer resolves to the live pinned tokenizer and its pattern equals the original's."""
    gold = H.golden_json("sanitize.json")
    assert gold is not None, "tests/golden/state/sanitize.json is missing"
    assert gold["status"] == "live==snapshot"
    assert SAN.SPECIAL_RE.pattern == gold["special_re"] and list(SAN.QWEN_ADDED) == gold["qwen_added"]
    if SAN.pattern_status() != "live==snapshot":
        H.skip_or_fail(f"pinned Qwen3.5-4B tokenizer_config.json not found (status {SAN.pattern_status()})")
    assert SAN.pattern(allow_frozen_snapshot=False).pattern == gold["pattern"]
