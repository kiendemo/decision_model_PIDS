"""Host (habitus.data.host): the HostData API over cards or the node index, with frozen day lists."""

import os
import pickle

import numpy as np
import pytest

from habitus import hosts as H
from habitus.data.cards import signature
from habitus.data.host import Host, MissingDay
from habitus.data import nodeindex

from fakes import FakeHost, day_of, make_kcs1_host, make_spec, write_cards


def _proc(name, d="system"):
    return {"v": 2, "type": "process", "os": "linux", "invariant": {"file_name": name, "image_dir": d},
            "behaviour": {"ops": {}}}


def _file(path):
    return {"v": 2, "type": "file", "os": "linux", "invariant": {"path": path, "dir": "home", "ext": ""},
            "behaviour": {"ops": {"read:process": "1"}}}


def _small_days():
    """c1, c2 clean; t1, t2 test; e empty. Node 5 changes card between t1 and t2 (same identity on t1
    as node 6, a behaviour-only difference on t2)."""
    a, b, f = _proc("a"), _proc("b"), _file("/home/u/x")
    b2 = {**_proc("b"), "behaviour": {"ops": {"write:home": "1"}}}
    return {
        "c1": day_of({1: a, 2: f}),
        "c2": day_of({1: a, 3: b}),
        "t1": day_of({5: b, 6: b, 9: f}),
        "t2": day_of({1: a, 5: b2, 7: f}),
        "e": {"cards": {}, "nodes": {}},
    }, {"c1": [[1, 2, 2]], "t1": [[5, 9, 2], [6, 9, 2]], "t2": [], "e": []}


def test_fake_host_universe_and_keys():
    days, links = _small_days()
    h = FakeHost(make_spec(clean=("c1", "c2"), test=("t1", "t2")), days, links)
    assert h.clean == ["c1", "c2"] and h.test == ["t1", "t2"] and h.os == "linux"
    assert h.shots(None) == ["c1", "c2"] and h.shots(1) == ["c1"] and h.shots(0) == []
    assert h.universe.tolist() == [1, 5, 6, 7, 9]
    assert h.utype.tolist() == ["process", "process", "process", "file", "file"]
    assert h.day_index("t1").tolist() == [1, 2, 4] and h.day_index("t2").tolist() == [0, 1, 3]
    assert h.upos([9, 1, 2, 100, -1]).tolist() == [4, 0, -1, -1, -1]
    assert h.node_days == {1: ["t2"], 5: ["t1", "t2"], 6: ["t1"], 7: ["t2"], 9: ["t1"]}
    b = signature(_proc("b"))
    kb = signature({"type": "process", "invariant": _proc("b")["invariant"]})
    assert h.inv_key(b) == kb
    assert h.ukey[1] == h.ukey[2] == kb                    # node 5's key comes from its first test day
    assert h.ukey.dtype == np.dtype("<U16")
    codes = h.ukey_codes
    assert codes[1] == codes[2] and codes[3] == codes[4] and len(set(codes.tolist())) == 3
    assert h.links("t1").tolist() == [[5, 9, 2], [6, 9, 2]] and h.links("t2").shape == (0, 3)
    assert h.ntype(signature(_file("/home/u/x"))) == "file"
    with pytest.raises(KeyError):
        h.card("0" * 16)
    assert h.keep_clean_days() is h and h._days.size >= 3


def test_fake_host_missing_and_empty_days():
    days, links = _small_days()
    with pytest.raises(MissingDay):
        FakeHost(make_spec(clean=("c1", "c9"), test=("t1",)), days, links)
    with pytest.raises(MissingDay):
        FakeHost(make_spec(clean=("c1",), test=("e",)), days, links)


def test_kcs1_host_shape():
    h = make_kcs1_host()
    assert (h.ds, h.os, h.clean, h.test) == ("KCS1T", "linux", ["c1", "c2"], ["t1", "t2"])
    assert len(h.nodes("t1")) == 30 and len(h.nodes("t2")) == 33
    assert h.universe.tolist() == list(range(33))
    assert h.links("t2").shape[1] == 3
    assert make_kcs1_host(n=4).universe.tolist() == list(range(15))


def _disk_host(tmp_path, **kw):
    days, links = _small_days()
    spec = make_spec(ds="FAKE", clean=("c1", "c2"), test=("t1", "t2"), card_days=("c1", "c2", "e", "t1", "t2"),
                     all_card_days=5, **kw)
    cdir = write_cards(tmp_path / "cards", "FAKE", days, links)
    return spec, cdir


def test_cards_and_nodeindex_sources_agree(tmp_path):
    spec, cdir = _disk_host(tmp_path)
    nidx = tmp_path / "nodeindex"
    paths = nodeindex.build("FAKE", cdir, str(nidx), spec=spec)
    assert len(paths) == 5 and all(os.path.exists(p) for p in paths)
    hc = Host("FAKE", spec=spec, cards_dir=cdir)
    hn = Host("FAKE", spec=spec, source="nodeindex", nodeindex_dir=str(nidx))
    assert hc.art == cdir and hn.art == str(nidx)
    for attr in ("universe", "utype", "ukey", "ukey_codes"):
        assert getattr(hc, attr).tolist() == getattr(hn, attr).tolist(), attr
    for d in hc.test:
        assert hc.day_index(d).tolist() == hn.day_index(d).tolist()
        assert list(hc.nodes(d).items()) == list(hn.nodes(d).items())        # same order
    assert hc.node_days == hn.node_days
    sig = hc.nodes("t1")[5]
    assert hc.ntype(sig) == hn.ntype(sig) and hc.inv_key(sig) == hn.inv_key(sig)
    assert hc.day("t1")["dataset"] == "FAKE" and hc.card(sig)["type"] == "process"
    assert hc.links("c1").tolist() == [[1, 2, 2]]
    for f in (lambda: hn.card(sig), lambda: hn.links("t1"), lambda: hn.day("t1")):
        with pytest.raises(NotImplementedError):
            f()


def test_disk_host_missing_files(tmp_path):
    spec, cdir = _disk_host(tmp_path)
    os.remove(os.path.join(cdir, "FAKE", "links", "t2.json.gz"))
    h = Host("FAKE", spec=spec, cards_dir=cdir)
    with pytest.raises(MissingDay):
        h.links("t2")
    os.remove(os.path.join(cdir, "FAKE", "c2.json.gz"))
    with pytest.raises(MissingDay):
        Host("FAKE", spec=spec, cards_dir=cdir)
    bad = make_spec(ds="FAKE", clean=("c1", "e"), test=("t1",))
    with pytest.raises(MissingDay, match="no node"):
        Host("FAKE", spec=bad, cards_dir=cdir)
    with pytest.raises(MissingDay):
        Host("FAKE", spec=spec, source="nodeindex", nodeindex_dir=str(tmp_path / "nowhere"))


def test_card_lookup_across_evicted_days(tmp_path):
    spec, cdir = _disk_host(tmp_path)
    h = Host("FAKE", spec=spec, cards_dir=cdir, max_days=1)
    sig_c2 = h.nodes("c2")[3]
    h.nodes("t1")                                           # evicts c2 from the day LRU
    assert h.card(sig_c2)["invariant"]["file_name"] == "b"  # interned host-wide
    fresh = Host("FAKE", spec=spec, cards_dir=cdir)
    assert fresh.card(signature(_proc("b")))["type"] == "process"    # found by reading days on demand


def test_pickle_drops_parsed_days(tmp_path):
    spec, cdir = _disk_host(tmp_path)
    h = Host("FAKE", spec=spec, cards_dir=cdir)
    u = h.universe.tolist()
    h2 = pickle.loads(pickle.dumps(h))
    assert h2._cards == {} and h2._merged == set() and h2.universe.tolist() == u
    assert h2.card(h.nodes("t1")[5])["type"] == "process"


def test_allow_list_and_spec_checks(tmp_path):
    with pytest.raises(H.ForbiddenDataset):
        Host("NOT_A_HOST", cards_dir=str(tmp_path))
    with pytest.raises(ValueError):
        Host("FAKE", spec=make_spec(ds="OTHER"), cards_dir=str(tmp_path))
    with pytest.raises(ValueError):
        Host("CADETS_E3", source="db", cards_dir=str(tmp_path))


def test_window_functions_on_host(tmp_path):
    spec, cdir = _disk_host(tmp_path)
    h = Host("FAKE", spec=spec, cards_dir=cdir)
    assert H.first_test_day(h) == "t1" and H.predeploy(h) == ["c1", "c2"] and H.inner_day(h) == "c2"


# --------------------------------------------------------------------------- #
# Real hosts (release cards, or the configured PIDSMaker artifacts' cards; read only)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.parametrize("ds", H.DATASETS)
def test_real_universe_sizes_and_source_parity(ds, cards_dir_real, nodeindex_real):
    spec = H.spec(ds)
    hc = Host(ds, cards_dir=cards_dir_real)
    assert hc.clean == list(spec.clean) and hc.test == list(spec.test)
    assert len(hc.universe) == spec.test_day_nodes
    want = {"CADETS_E3": 281585, "THEIA_E3": 700902, "CLEARSCOPE_E3": 111403, "optc_h501": 1489198,
            "ATLASV2_EDR": 152763}[ds]
    assert len(hc.universe) == want
    hn = Host(ds, source="nodeindex", nodeindex_dir=nodeindex_real[0])
    assert np.array_equal(hc.universe, hn.universe) and np.array_equal(hc.utype, hn.utype)
    for d in hc.test:
        assert np.array_equal(hc.day_index(d), hn.day_index(d)), d
    assert np.array_equal(hc.ukey, hn.ukey) and np.array_equal(hc.ukey_codes, hn.ukey_codes)


@pytest.mark.slow
def test_real_card_signatures_and_example(cards_dir_real):
    h = Host("THEIA_E3", cards_dir=cards_dir_real)
    c = h.card("f0d561f173b07c33")                          # THEIA 04-09: postgres
    assert c["type"] == "process" and c["invariant"]["file_name"] == "postgres"
    assert signature(c) == "f0d561f173b07c33"
    day = h.day("2018-04-09")
    assert all(signature(x) == s for s, x in day["cards"].items())
    assert set(h.nodes("2018-04-09").values()) == set(day["cards"])
