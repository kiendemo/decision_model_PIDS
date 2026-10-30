"""Precedent cache: identical states with the cache off / cold / warm, read-only shipped base,
overlay-only writes, safe merging of concurrent writers."""

import glob
import gzip
import hashlib
import json
import os
import shutil

import pytest

import state_helpers as H
from habitus.state import precedent as PREC
from habitus.state.builder import StateBuilderE2, serialize
from habitus.state.ladder import StateBuilder
from habitus.state import select as SEL


@pytest.fixture(autouse=True)
def _restore_roots():
    yield
    PREC.reset_roots()
    H.clear_caches()


def _build(sigs_parts, e2=False, ds="KCS1PREC"):
    H.clear_caches()
    h = H.local_kcs1_host(ds=ds)
    out = {}
    for d in h.test:
        sigs = SEL.flat(SEL.head(h, d, ["c1", "c2"]))
        for part in sigs_parts(sigs):
            if e2:
                st = StateBuilderE2(h, ["c1", "c2"], 2).build(d, part)
                out.update({(d, s): {q: serialize(x) for q, x in qs.items()} for s, qs in st.items()})
            else:
                st = StateBuilder(h, ["c1", "c2"], "L4", 2).build(d, part)
                out.update({(d, s): serialize(x) for s, x in st.items()})
    return out


def _read(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def test_precedent_cache_identical_states_off_cold_warm(monkeypatch, tmp_path):
    whole = lambda sigs: [sigs]                          # noqa: E731
    halves = lambda sigs: [sigs[1::2], sigs]             # noqa: E731
    cache = str(tmp_path / "pc")
    for e2 in (False, True):
        PREC.set_roots(base=None, overlay=None)
        off = _build(whole, e2)
        assert not os.path.exists(cache) or e2
        PREC.set_roots(base=None, overlay=cache)
        cold = _build(halves, e2)
        files = glob.glob(os.path.join(cache, "KCS1PREC", "*_e2.json.gz"))
        assert len(files) == 1
        built = []
        real_get = PREC.get_refindex
        monkeypatch.setattr(PREC, "get_refindex", lambda *a: built.append(a) or real_get(*a))
        warm = _build(whole, e2)
        monkeypatch.setattr(PREC, "get_refindex", real_get)
        assert off == cold == warm and len(off) > 20 and built == []      # warm: no RefIndex at all
        texts = [t for v in off.values() for t in (v.values() if e2 else [v])]
        assert sum('"nearest_reference":[{' in t for t in texts) > 10
    raw = _read(files[0])
    assert raw["ds"] == "KCS1PREC" and raw["ref"] == ["c1", "c2"] and raw["e"] == PREC.PREC_E
    assert len(raw["nearest"]) > 10
    name = hashlib.sha1(",".join(["c1", "c2"]).encode("utf-8")).hexdigest() + "_e2.json.gz"
    assert os.path.basename(files[0]) == name                             # the shipped file naming


def test_shipped_base_is_read_only(monkeypatch, tmp_path):
    """A complete base answers everything: no RefIndex, nothing written anywhere, no lock file; a
    partial base sends only its misses to the overlay; the base file is never modified."""
    full = str(tmp_path / "full")
    PREC.set_roots(base=None, overlay=full)
    ref_states = _build(lambda sigs: [sigs], True)
    base = str(tmp_path / "base")
    shutil.copytree(full, base)
    for f in glob.glob(os.path.join(base, "*", "*.lock")):
        os.remove(f)
    before = {p: open(p, "rb").read() for p in glob.glob(os.path.join(base, "*", "*"))}
    os.chmod(os.path.join(base, "KCS1PREC"), 0o555)                     # a write into the base would fail
    try:
        overlay = str(tmp_path / "overlay")
        PREC.set_roots(base=base, overlay=overlay)
        built = []
        real_get = PREC.get_refindex
        monkeypatch.setattr(PREC, "get_refindex", lambda *a: built.append(a) or real_get(*a))
        assert _build(lambda sigs: [sigs], True) == ref_states and built == []
        assert not os.path.exists(overlay)
        assert {p: open(p, "rb").read() for p in glob.glob(os.path.join(base, "*", "*"))} == before
    finally:
        os.chmod(os.path.join(base, "KCS1PREC"), 0o755)
    # a partial base: drop half of its answers; the misses go to the overlay only
    (path,) = glob.glob(os.path.join(base, "KCS1PREC", "*_e2.json.gz"))
    obj = _read(path)
    keep = dict(sorted(obj["nearest"].items())[::2])
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump({**obj, "nearest": keep}, f, separators=(",", ":"), sort_keys=True)
    before = open(path, "rb").read()
    overlay = str(tmp_path / "overlay2")
    PREC.set_roots(base=base, overlay=overlay)
    assert _build(lambda sigs: [sigs], True) == ref_states
    assert open(path, "rb").read() == before
    (ov,) = glob.glob(os.path.join(overlay, "KCS1PREC", "*_e2.json.gz"))
    own = _read(ov)["nearest"]
    assert own and not set(own) & set(keep)                              # only the base's misses
    assert {**keep, **own} == {s: v for s, v in obj["nearest"].items() if s in {**keep, **own}}


def test_two_writers_merge_and_disagreement_raises(tmp_path):
    d = str(tmp_path / "two")
    a, b = PREC.PrecedentCache(d, "X", ("c1",), 2), PREC.PrecedentCache(d, "X", ("c1",), 2)
    a.add({"s1": ["r1", "r2"]})
    b.add({"s2": ["r3"]})
    c = PREC.PrecedentCache(d, "X", ("c1",), 2)
    assert c.map == {"s1": ["r1", "r2"], "s2": ["r3"]} and a.lookup(["s2"]) == []
    with pytest.raises(RuntimeError, match="disagree"):
        c.add({"s1": ["r9"]})
    with pytest.raises(ValueError, match="holds ds X"):                       # a file of another (ds, ref, e)
        PREC.PrecedentCache(d, "Y", ("c1",), 2)
    base = str(tmp_path / "two")                                              # base and overlay disagree
    other = str(tmp_path / "other")
    PREC.PrecedentCache(other, "X", ("c1",), 2).add({"s1": ["zz"]})
    with pytest.raises(RuntimeError, match="disagree"):
        PREC.PrecedentCache(other, "X", ("c1",), 2, base=base)
    mem = PREC.PrecedentCache(None, "X", ("c1",), 2, base=base)              # no overlay: memory only
    mem.add({"s9": ["r"]})
    assert mem.map["s9"] == ["r"] and mem.path is None


def test_roots_follow_settings_and_overrides(tmp_path):
    PREC.reset_roots()
    from habitus import config
    s = config.settings()
    base, overlay = PREC.precedent_roots()
    assert base == s.paths.precedent and overlay == os.path.join(s.paths.runs, "precedent_cache")
    PREC.set_roots(base=None, overlay=None)
    assert PREC.precedent_cache(H.local_kcs1_host(), ["c1"]) is None
    PREC.set_roots(overlay=str(tmp_path))
    pc = PREC.precedent_cache(H.local_kcs1_host(), ["c1"])
    assert pc.base_path is None and pc.path.startswith(str(tmp_path))
