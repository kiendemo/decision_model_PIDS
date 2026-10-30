"""Node index (habitus.data.nodeindex): every card file of every host, string free."""

import os

import pytest

from habitus import hosts as H
from habitus.data import nodeindex as N
from habitus.data.cards import card_path, signature
from habitus.data.host import MissingDay, nodeindex_path
from habitus.io import load_json

from fakes import day_of, make_spec, write_cards


def _cards():
    p = {"v": 2, "type": "process", "os": "linux", "invariant": {"file_name": "a"}, "behaviour": {"ops": {}}}
    f = {"v": 2, "type": "file", "os": "linux", "invariant": {"path": "/x", "dir": "other", "ext": ""},
         "behaviour": {"ops": {}}}
    return p, f


def test_index_day_record():
    p, f = _cards()
    raw = day_of({7: p, 3: f, 9: p})
    rec = N.index_day("FAKE", "t1", raw)
    assert rec["format"] == N.FORMAT and (rec["ds"], rec["day"]) == ("FAKE", "t1")
    assert list(rec["nodes"]) == ["7", "3", "9"]                     # card-file order
    assert rec["types"] == {signature(p): "process", signature(f): "file"}
    assert rec["kappa"][signature(p)] == signature({"type": "process", "invariant": p["invariant"]})
    with pytest.raises(ValueError):
        N.index_day("FAKE", "t1", {"cards": {}, "nodes": {"1": "deadbeefdeadbeef"}})


def test_build_fake_with_empty_day(tmp_path):
    p, f = _cards()
    days = {"c1": day_of({1: p}), "e": {"cards": {}, "nodes": {}}, "t1": day_of({1: p, 2: f})}
    spec = make_spec(ds="FAKE", clean=("c1",), test=("t1",), card_days=("c1", "e", "t1"), all_card_days=3)
    cdir = write_cards(tmp_path / "c", "FAKE", days)
    out = N.build("FAKE", cdir, str(tmp_path / "n"), spec=spec)
    assert out == [nodeindex_path(str(tmp_path / "n"), "FAKE", d) for d in ("c1", "e", "t1")]
    empty = load_json(out[1])
    assert empty["nodes"] == {} and empty["types"] == {} and empty["kappa"] == {}
    os.remove(card_path(cdir, "FAKE", "e"))
    with pytest.raises(MissingDay):
        N.build("FAKE", cdir, str(tmp_path / "n2"), spec=spec)
    with pytest.raises(H.ForbiddenDataset):
        N.build("NOT_A_HOST", cdir, str(tmp_path / "n3"))


def test_build_refuses_read_only_output(settings, tmp_path):
    if not settings.paths.pidsmaker_artifacts:
        pytest.skip("no paths.pidsmaker_artifacts configured")
    from habitus.config import ConfigError
    with pytest.raises(ConfigError):
        N.build("ATLASV2_EDR", str(tmp_path), os.path.join(settings.paths.pidsmaker_artifacts, "x"))


@pytest.mark.slow
def test_real_nodeindex_covers_all_45_card_files(nodeindex_real, cards_dir_real):
    out, files = nodeindex_real
    assert sum(len(v) for v in files.values()) == 45
    for ds in H.DATASETS:
        spec = H.spec(ds)
        assert [os.path.basename(p)[:10] for p in files[ds]] == list(spec.card_days), ds
        assert len(files[ds]) == spec.all_card_days
    for ds, day in (("THEIA_E3", "2018-04-02"), ("THEIA_E3", "2018-04-06"), ("ATLASV2_EDR", "2022-07-20")):
        rec = load_json(nodeindex_path(out, ds, day))                 # empty days are indexed too
        assert rec["nodes"] == {} and rec["day"] == day
    raw = load_json(card_path(cards_dir_real, "CLEARSCOPE_E3", "2018-04-02"))
    rec = load_json(nodeindex_path(out, "CLEARSCOPE_E3", "2018-04-02"))   # a dropped clean day: indexed
    assert list(rec["nodes"].items()) == list(raw["nodes"].items()) and rec["nodes"]
    for s, c in raw["cards"].items():
        assert rec["types"][s] == c["type"]
        assert rec["kappa"][s] == signature({"type": c["type"], "invariant": c["invariant"]})
