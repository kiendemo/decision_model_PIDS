"""The ported E1 snapshot (G3): every E1 state, rendered prompt / cache key, the fake-engine
answers, and the release E1 node scores / null pools equal the snapshot the original code froze
(tests/golden/kcs1_e1_snapshot.json.gz, a byte copy). Port of test_kcs1.test_e1_snapshot_unchanged.

Without HABITUS_REQUIRE_ASSETS a missing tokenizer drops the prompt / engine parts and missing
release cards or score files drop the real / extra parts; with it, every part is required."""

import hashlib
import json
import os

import pytest

import e1_snapshot as SNAP

REQUIRE = os.environ.get("HABITUS_REQUIRE_ASSETS", "") not in ("", "0")
GOLDEN_SHA256 = "27deef5e87a4b4067b3af92a9e2c78f267b4a503e4958c04f68b31fcd12549ed"


def test_snapshot_is_the_original_bytes():
    with open(SNAP.PATH, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == GOLDEN_SHA256


def test_e1_snapshot_unchanged():
    import state_helpers
    snap = SNAP.load()
    real = any(SNAP._has_cards(ds) for ds in SNAP.HOSTS)
    if REQUIRE:
        missing = [ds for ds in ("CADETS_E3", "optc_h501", "THEIA_E3", "ATLASV2_EDR") if not SNAP._has_cards(ds)]
        assert not missing, f"HABITUS_REQUIRE_ASSETS: no release cards for {missing}"
    now = json.loads(json.dumps(SNAP.compute(real=real)))
    if not now["tokenizer"]:
        assert not REQUIRE, "HABITUS_REQUIRE_ASSETS: the pinned tokenizer snapshot is missing"
        for part in ("fake", "real", "extra"):
            for key in ("prompts", "engine"):
                (snap.get(part) or {}).pop(key, None)
                (now.get(part) or {}).pop(key, None)
        snap.pop("tokenizer", None)             # the presence flag itself is not compared without it
        now.pop("tokenizer", None)
    if not real:
        snap.pop("real"), now.pop("real")
        snap.pop("extra", None), now.pop("extra", None)
    for key in ("node_scores", "null_pools"):     # release score files absent: not compared
        for name in set((snap.get("extra") or {}).get(key, {})) - set((now.get("extra") or {}).get(key, {})):
            assert not REQUIRE, f"HABITUS_REQUIRE_ASSETS: no release score files for {key} {name}"
            snap["extra"][key].pop(name)
    state_helpers.clear_caches()
    assert SNAP.diff(snap, now) == []
    assert len(snap["fake"]["states"]) > 500 and (not real or len(snap["real"]["states"]) >= 2000)
    if real and "extra" in snap:
        assert snap["extra"]["n"]["realcap|CADETS_E3|2018-04-12|L4|k2"] > 3000
    if not (real and now.get("tokenizer")):
        pytest.skip("compared without the release cards and/or the tokenizer (fake / text parts only)")
