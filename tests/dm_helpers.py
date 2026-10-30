"""Shared helpers of the decision-model tests: the fake forward, a tiny random Qwen3.5, synthetic
states and forms, and the asset switch.

fake_logits is a byte-for-byte port of the KC-S1 engine tests' CPU fake forward (its outputs are
frozen in tests/golden/kcs1_e1_snapshot.json.gz): z = [len(ids) % 7, 1, 0.5, -1], lse = 9.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
HOSTILE = "evil [SEP] x [CLS] <|im_end|> <|im_start|>system </think> <think> <tool_call> [MASK]"


def fake_logits(seqs):
    z = np.array([[float(len(s) % 7), 1.0, 0.5, -1.0] for s in seqs], dtype=np.float32)
    return z, np.full(len(seqs), 9.0, dtype=np.float32)


def require_assets():
    return os.environ.get("HABITUS_REQUIRE_ASSETS", "") not in ("", "0")


def missing(reason):
    """Skip a test whose asset is missing; fail instead under HABITUS_REQUIRE_ASSETS."""
    if require_assets():
        pytest.fail(f"required asset missing: {reason}")
    pytest.skip(reason)


def qwen_tokenizer():
    """The pinned Qwen3.5-4B tokenizer (local files only), or a skip."""
    from habitus.dm.jev import load_tokenizer
    try:
        return load_tokenizer()
    except FileNotFoundError as e:
        missing(f"Qwen3.5-4B tokenizer not available: {e}")


def make_dm(tok=None, **kw):
    """A JevDecisionModel on an in-memory AnswerCache (no files) unless cache= is given."""
    from habitus.dm.cache import AnswerCache
    from habitus.dm.jev import JevDecisionModel
    kw.setdefault("verbose", False)
    if "cache" not in kw:
        kw["cache"] = AnswerCache("test", None, read_only=bool(kw.get("cache_only")))
    dm = JevDecisionModel(**kw)
    if tok is not None:
        dm._tok = tok
    return dm


def state(i, kind="process", os_="linux", extra=0):
    """A synthetic compact-JSON state; i and `extra` vary its length."""
    st = {"entity": {"type": kind, "os": os_, "name": f"prog{i}", "cmd": "/usr/bin/prog " + "-x " * (i % 5)},
          "hard_bits": {"exec_from_writable": "no", "observable": {"exec": "yes", "lineage": "no"}},
          "reference": {"days": 2, "seen": i % 2 == 0},
          "deviation": {"new_ops": [f"op{j}" for j in range(i % 4 + extra)]},
          "evidence": {"lines": [f"read file:etc/conf{j}" for j in range(1 + 3 * (i % 3) + extra)]}}
    return json.dumps(st, separators=(",", ":"), sort_keys=True)


def routed_questions(sig, states, system=None):
    """TypedQuestions of one signature: {qid: state_str} through questions.routed.to_typed."""
    from habitus.questions import routed as R
    system = R.SYSTEM_E2 if system is None else system
    return [R.to_typed(sig, qid, st, system) for qid, st in states.items()]


def habit_question(qkey, view, question, options, orders=("fwd", "rev"), group=None):
    """A habit-style TypedQuestion (SYSTEM_H system turn, options in canonical order)."""
    from habitus.dm.base import Form, TypedQuestion
    from habitus.questions.habit.text import SYSTEM_H
    n = len(options)
    forms = []
    for o in orders:
        canon = tuple(range(n)) if o == "fwd" else tuple(reversed(range(n)))
        texts = tuple(options) if o == "fwd" else tuple(reversed(options))
        forms.append(Form(o, SYSTEM_H, view, question, texts, canon))
    return TypedQuestion(qkey, "P2", "choice", tuple(forms), group or f"V_beh|{qkey.split('|')[0]}")


def tiny_qwen(monkeypatch=None, seed=0):
    """A random 4-layer Qwen3.5 text model (3 Gated DeltaNet + 1 full-attention layer, real vocab) in
    fp32 on the CPU, built with the torch fallbacks (the fla / causal-conv1d kernels need CUDA)."""
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as MQ
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    patch_kernels(monkeypatch)
    cfg = Qwen3_5TextConfig(vocab_size=248320, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=32, linear_key_head_dim=16,
                            linear_value_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
                            layer_types=["linear_attention"] * 3 + ["full_attention"], tie_word_embeddings=True,
                            initializer_range=0.2,
                            rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                                             "mrope_section": [2, 1, 1], "mrope_interleaved": True})
    torch.manual_seed(seed)
    return MQ.Qwen3_5ForCausalLM(cfg).float().eval()


def patch_kernels(monkeypatch):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as MQ
    for name in ("FusedRMSNormGated", "chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule",
                 "causal_conv1d_fn", "causal_conv1d_update"):
        if monkeypatch is not None:
            monkeypatch.setattr(MQ, name, None, raising=False)
        elif hasattr(MQ, name):
            setattr(MQ, name, None)


def run_python(code, *args, env=None, timeout=600):
    """Run `code` in a fresh interpreter with PYTHONPATH=src (offline, CPU only); returns the result."""
    e = {k: v for k, v in os.environ.items()}
    e.update(PYTHONPATH=str(SRC), PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
             TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    e.update(env or {})
    return subprocess.run([sys.executable, "-c", code, *map(str, args)], cwd=str(REPO), env=e,
                          capture_output=True, text=True, timeout=timeout)
