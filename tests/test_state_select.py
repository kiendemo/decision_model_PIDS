"""Head rule, rarity fallback and limit_caps (fixed a priori, label-free)."""

import pytest

import state_helpers as H
from habitus.state import profile as P
from habitus.state import select as SEL


@pytest.fixture(scope="module")
def host():
    H.clear_caches()
    return H.make_host()


def test_head_caps_order_and_unseen_first(host):
    caps = {"process": 3, "file": 2, "netflow": 4}
    prof = P.get_profile(host, ["c1", "c2"])
    for ref in (None, ["c1", "c2"], prof):
        for d in host.test:
            h = SEL.head(host, d, ref, caps)
            assert list(h) == list(SEL.TYPES)
            assert all(len(h[t]) <= caps[t] for t in SEL.TYPES)
            assert all(host.ntype(s) == t for t in SEL.TYPES for s in h[t])
            n = SEL.day_counts(host, d)
            full = SEL.head(host, d, ref)
            assert full["process"] == sorted(full["process"], key=lambda s: (n[s], s))
            assert h["process"] == full["process"][:3] and h["file"] == full["file"][:2]
            if ref is not None:                                   # unseen kappa first
                seen = SEL.ref_kappas(host, ref)
                for t in ("file", "netflow"):
                    flags = [host.inv_key(s) in seen for s in full[t]]
                    assert flags == sorted(flags)
    h = SEL.head(host, "t2", prof)
    pay = next(s for s in h["file"] if host.card(s)["invariant"]["path"] == H.TMP + "/payload.sh")
    assert h["file"][0] == pay                                     # unseen + changed on disk
    assert next(s for s in h["netflow"] if host.card(s)["invariant"]["remote_ip"] == "6.6.6.6") == h["netflow"][0]
    assert SEL.head(host, "t2", ["c1", "c2"]) == SEL.head(host, "t2", prof)    # days or profile: same head
    assert SEL.flat(h) == h["process"] + h["file"] + h["netflow"]
    with pytest.raises(ValueError):
        SEL.head(host, "c2", ["c1", "c2"])


def test_fallback_below_every_decision_score(host):
    for d in host.test:
        fb = SEL.fallback_rank(host, d)
        assert all(-1.0 < v < -0.5 + 1e-12 for v in fb.values())
        assert fb == {s: -1 + 0.5 / (1 + c) for s, c in SEL.day_counts(host, d).items()}
    assert SEL.fallback_value(0) == -0.5 and SEL.fallback_value(1) == -0.75
    assert SEL.fallback_value(3) > SEL.fallback_value(4)                       # rarer is higher


def test_constants_and_limit_caps():
    assert SEL.CAPS == {"process": 4000, "file": 2000, "netflow": 6000}
    assert SEL.TYPES == ("process", "file", "netflow")
    assert SEL.limit_caps(None) == SEL.CAPS and SEL.limit_caps(None) is not SEL.CAPS
    assert SEL.limit_caps(5) == {"process": 5, "file": 5, "netflow": 5}
    assert SEL.limit_caps(3000) == {"process": 3000, "file": 2000, "netflow": 3000}
    assert SEL.changed_on_disk({"behaviour": {"ops": {"rename:x": "1"}}})
    assert SEL.changed_on_disk({"behaviour": {"flags": ["deleted"]}})
    assert not SEL.changed_on_disk({"behaviour": {"ops": {"read:x": "1"}}})
    assert SEL.ref_days_of(None) is None and SEL.ref_days_of(("c1",)) == ["c1"]
    assert SEL.ref_kappas(None, []) is None
