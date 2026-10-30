"""E1 ladder L0..L4 (state-ablation table only): cumulative levels, k = 0 rules, reference refusal."""

import pytest

import state_helpers as H
from habitus.state import ladder as LD
from habitus.state import precedent as PREC
from habitus.state import profile as P
from habitus.state import sections as S
from habitus.state import select as SEL


@pytest.fixture(autouse=True, scope="module")
def _tmp_precedent(tmp_path_factory):
    PREC.set_roots(base=None, overlay=str(tmp_path_factory.mktemp("prec")))
    H.clear_caches()
    yield
    PREC.reset_roots()
    H.clear_caches()


@pytest.fixture(scope="module")
def host():
    H.clear_caches()
    return H.make_host()


def test_ladder_constants():
    assert list(LD.LEVELS) == ["L0", "L1", "L2", "L3", "L4"]
    assert LD.K0_LEVELS == ("L0", "L1", "L3") and LD.PRIMARY == {0: "L3", 2: "L4", None: "L4"}
    assert LD.levels_for(0, list(LD.LEVELS)) == ["L0", "L1", "L3"]
    assert LD.levels_for(2, list(LD.LEVELS)) == list(LD.LEVELS) == LD.levels_for(None, list(LD.LEVELS))
    assert LD.LEVELS["L4"] == list(S.SECTIONS)


def test_levels_are_cumulative(host):
    names = list(LD.LEVELS)
    for a, b in zip(names, names[1:]):
        assert LD.LEVELS[a] == LD.LEVELS[b][:len(LD.LEVELS[a])]
    for ref, k in ((["c1", "c2"], 2), ([], 0)):
        sigs = SEL.flat(SEL.head(host, "t2", ref or None))
        built = {lv: LD.StateBuilder(host, ref, lv, k).build("t2", sigs) for lv in names}
        for a, b in zip(names, names[1:]):
            for s in sigs:
                sa, sb = built[a][s], built[b][s]
                assert set(sa) <= set(sb)
                assert all(sa[x] == sb[x] for x in sa)                     # shared sections are equal
                assert list(sb) == [x for x in S.SECTIONS if x in sb]
        if k == 0:
            for s in sigs:
                assert built["L2"][s] == {**built["L1"][s], "reference": {"days": 0}}
                assert built["L4"][s] == built["L3"][s]
                assert "deviation" not in built["L4"][s] and "precedent" not in built["L4"][s]
        else:
            assert all("precedent" in built["L4"][s] and "deviation" in built["L2"][s] for s in sigs)


def test_statebuilder_refuses_reference_day_and_bad_args(host):
    with pytest.raises(ValueError):
        LD.StateBuilder(host, ["c1", "c2"], "L5", 2)
    with pytest.raises(ValueError):
        LD.StateBuilder(host, [], "L2", 2)
    with pytest.raises(ValueError):
        LD.StateBuilder(host, ["c1"], "L2", 0)
    with pytest.raises(ValueError):
        LD.StateBuilder(host, ["c1", "c2"], "L4", 2).build("c2", [])
    with pytest.raises(ValueError):
        LD.StateBuilder(host, ["c1", "c2"], "L2", 2, profile=P.get_profile(host, ["c1"]))
    assert LD.StateBuilder(host, ["c1", "c2"], "L1", 2).profile is None      # L1 needs no profile
    with pytest.raises(ValueError):
        SEL.head(host, "c1", ["c1", "c2"])                                     # never a head on its own reference


def test_e1_evidence_lines_and_precedent(host):
    prof = P.get_profile(host, ["c1", "c2"])
    st = LD.StateBuilder(host, ["c1", "c2"], "L4", 2)
    sigs = SEL.flat(SEL.head(host, "t2", prof))
    out = st.build("t2", sigs)
    x9q = next(s for s in sigs if host.card(s)["invariant"].get("file_name") == "x9q")
    ev = out[x9q]["evidence"]
    assert ev["more"] == 0 and len(ev["lines"]) <= S.EV_LINES
    assert any(ln.startswith("wrote file:staging/sh payload.sh") for ln in ev["lines"])
    assert all(isinstance(p["nearest_reference"], list) for s, x in out.items() for p in [x["precedent"]])
    assert len(out[x9q]["precedent"]["nearest_reference"]) == PREC.PREC_E
    assert st.refindex is PREC.get_refindex(host, ["c1", "c2"])
    assert st.timing["n"] == len(sigs)
