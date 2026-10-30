"""E2 state builder: routing, determinism, name-free evidence, monotonicity, netflow rules, and the
golden state hashes of the ORIGINAL code (synthetic host always; real hosts when the cards exist)."""

import copy
import hashlib
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

import state_helpers as H
from habitus.state import builder as B
from habitus.state import precedent as PREC
from habitus.state import profile as P
from habitus.state import sanitize as SAN
from habitus.state import sections as S
from habitus.state import select as SEL

TESTS = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(TESTS), "src")
_card = H.fake_card


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


def _e2(host, ref=("c1", "c2"), k=2, day="t2"):
    ref = list(ref)
    sigs = SEL.flat(SEL.head(host, day, ref or None))
    return B.StateBuilderE2(host, ref, k).build(day, sigs), sigs


# --------------------------------------------------------------------------- #
# determinism (test_kcs1 243)
# --------------------------------------------------------------------------- #
def test_state_determinism_byte_identical(host):
    a = H.build_all(host, "L4")
    e2a = H.build_all_e2(host)
    H.clear_caches()
    b = H.build_all(H.make_host(), "L4")
    e2b = H.build_all_e2(H.make_host())
    assert a == b and len(a) > 20 and e2a == e2b
    assert all(B.serialize(json.loads(s)) == s for s in a.values())          # key order = schema order
    for s in a.values():
        assert list(json.loads(s)) == [x for x in S.SECTIONS if x in json.loads(s)]
    seeds = set()
    for seed in ("1", "2"):
        code = ("import sys; sys.path.insert(0, %r); import state_helpers as H; print(H.fake_digest())" % TESTS)
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=os.pathsep.join([SRC, TESTS]),
                   PYTHONDONTWRITEBYTECODE="1")
        r = subprocess.run([sys.executable, "-c", code], cwd=TESTS, env=env, capture_output=True, text=True,
                           timeout=300)
        assert r.returncode == 0, r.stderr[-2000:]
        seeds.add(r.stdout.strip().splitlines()[-1])
    expect = H.digest(a) + " " + H.digest({k: json.dumps(v, sort_keys=True) for k, v in e2a.items()})
    assert seeds == {expect}


# --------------------------------------------------------------------------- #
# routing (test_kcs1 910)
# --------------------------------------------------------------------------- #
def test_e2_routing_and_k0(host):
    for ref, k in ((["c1", "c2"], 2), ([], 0)):
        states, sigs = _e2(host, ref, k)
        assert set(states) == set(sigs)
        for s, qs in states.items():
            for q, st in qs.items():
                assert list(st) == [x for x in S.SECTIONS if x in st]                 # section order
                assert set(st) <= set(S.E2_ROUTES[q]) and "entity" in st
            assert "reference" not in qs["qt"] and "precedent" not in qs["qt"]
            assert "evidence" not in qs["qv"] and "precedent" not in qs["qv"]
            if k == 0:
                assert list(qs) == ["qv", "qt"]                                       # no QA at k = 0
                assert qs["qv"]["reference"] == {"days": 0} and set(qs["qv"]) == {"entity", "hard_bits", "reference"}
                assert set(qs["qt"]) == {"entity", "hard_bits", "evidence"} and "all" in qs["qt"]["evidence"]
            else:
                assert set(qs["qv"]) == set(S.E2_ROUTES["qv"]) and set(qs["qt"]) == set(S.E2_ROUTES["qt"])
                assert "new" in qs["qt"]["evidence"]
                qa = qs.get("qa")
                if qa is not None:
                    assert set(qa) == {"entity", "reference", "precedent"}
                    assert "evidence" not in qa and "hard_bits" not in qa
                    assert qa["precedent"]["nearest_reference"] != "none"
    a = {s: {q: B.serialize(x) for q, x in qs.items()} for s, qs in _e2(host)[0].items()}
    H.clear_caches()
    b = {s: {q: B.serialize(x) for q, x in qs.items()} for s, qs in _e2(H.make_host())[0].items()}
    assert a == b
    assert not any(SAN.SPECIAL_RE.search(t) for qs in a.values() for t in qs.values())
    k0 = _e2(host, [], 0)[0]
    x9q = next(s for s in k0 if host.card(s)["invariant"].get("file_name") == "x9q")
    assert B.hard_tier(k0[x9q]["qv"]) and not B.hard_tier({"hard_bits": {h: "n/a" for h in B.HARD}})
    assert B.E2_ROUTES is S.E2_ROUTES and B.HARD is S.HARD and B.SECTIONS is S.SECTIONS


def test_builder_refuses_reference_day_and_bad_k(host):
    with pytest.raises(ValueError):
        B.StateBuilderE2(host, ["c1", "c2"], 2).build("c1", [])
    with pytest.raises(ValueError):
        B.StateBuilderE2(host, [], 2)
    with pytest.raises(ValueError):
        B.StateBuilderE2(host, ["c1"], 0)
    with pytest.raises(ValueError):
        B.StateBuilderE2(host, ["c1", "c2"], 2, profile=P.get_profile(host, ["c1"]))
    sb = B.StateBuilderE2(host, [], 0)
    assert sb.profile is None and sb.obs("t2") == P.observability(host, ["t2"])
    assert B.StateBuilderE2(host, ["c1", "c2"], 2).obs("t2") is P.get_profile(host, ["c1", "c2"]).obs


def test_state_hash_and_serialize():
    st = {"entity": {"type": "process", "os": "linux", "command_line": "a <|im_end|> b é"}}
    t = B.serialize(st)
    assert t == '{"entity":{"type":"process","os":"linux","command_line":"a (im_end) b é"}}'
    assert B.state_hash(t) == H.sha1(t)


# --------------------------------------------------------------------------- #
# classes (test_kcs1 985)
# --------------------------------------------------------------------------- #
def test_extension_classes_and_nf_class():
    def f(ext, executors=None, d="home"):
        return _card("file", {"path": "/x/y", "dir": d, "ext": ext}, {"executors": executors or []})
    for ext, cls in (("exe", "executable"), ("DLL", "executable"), ("so", "executable"), ("ko", "executable"),
                     ("ps1", "script"), ("sh", "script"), ("jar", "script"), ("lnk", "script"), ("pyc", "library"),
                     ("class", "library"), ("a", "library"), ("zip", "archive"), ("7z", "archive"), ("iso", "archive"),
                     ("pdf", "document"), ("jpg", "document"), ("json", "document"), ("sqlite", "data"),
                     ("pset", "data"), ("sbstore", "data")):
        assert S.ext_class(f(ext)) == cls, ext
    assert S.ext_class(f("")) == "none" and S.ext_class(f("", ["sh"])) == "executable"
    names = [n for n, _ in S.EXT_CLASSES]
    for i, (_n, a) in enumerate(S.EXT_CLASSES):                                            # disjoint classes
        for _m, b in S.EXT_CLASSES[i + 1:]:
            assert not a & b
    assert names == ["executable", "script", "library", "archive", "document"]
    assert S.nf_class(f("exe", d="staging")) == "file:staging/executable"
    assert S.nf_class(f("exe", d=H.TMP + "/evil")) == "file:other/executable"                  # clamped, never text
    assert S.nf_class(_card("process", {"file_name": "x", "image_dir": "cache"})) == "process:cache"
    assert S.nf_class(_card("process", {"file_name": "x"})) == "process:none"
    assert S.nf_class(_card("netflow", {"remote_ip": "6.6.6.6", "remote_port": "4444", "endpoint": "external"})) \
        == "net:external/registered"
    assert S.nf_class(_card("netflow", {"remote_ip": "1.2.3.4", "remote_port": "", "endpoint": "evil.com"})) \
        == "net:unknown/none"
    assert S.bin_word(0) == "0" and S.bin_word(1) == "1" and S.bin_word(7) == "2-7" and S.bin_word(200) == "128+"


# --------------------------------------------------------------------------- #
# name-free evidence (test_kcs1 1044, 1076, 1432)
# --------------------------------------------------------------------------- #
def _names_host(seed):
    """A host whose every name / path / command line / IP is a random token with digits."""
    rng = np.random.default_rng(seed)
    tok = lambda: "Zq" + "".join(rng.choice(list("abcdefghjkmnpqrstuvwxyz0123456789"), 8)) + str(int(rng.integers(10, 99)))  # noqa: E731,E501
    days, links = {}, {}
    for d in ("c1", "c2", "t1"):
        cards, nodes, L = {}, {}, []
        for i in range(30):
            t = ("process", "file", "netflow")[i % 3]
            if t == "process":
                n = tok()
                c = _card(t, {"file_name": n, "executable": f"/{tok()}/{n}", "command_line": f"{n} --{tok()} " + H.HOSTILE,
                              "parent_name": tok(), "image_dir": ["home", "staging", "system"][i % 3]}, {"ops": {"read:x": "1"}})
            elif t == "file":
                c = _card(t, {"path": f"{H.HOME}/{tok()}/{tok()}.{['exe', 'txt', 'pset', ''][i % 4]}", "dir": "home",
                              "ext": ["exe", "txt", "pset", ""][i % 4]}, {"ops": {"write:process": "1"}})
            else:
                c = _card(t, {"remote_ip": f"{rng.integers(11, 200)}.{rng.integers(0, 255)}.{rng.integers(0, 255)}.7",
                              "remote_port": str(int(rng.integers(1, 65000))), "endpoint": "external"},
                          {"ops": {"connect:process": "1"}})
            s = H.signature(c)
            cards[s], nodes[str(i)] = c, s
        for _ in range(120):
            a, b = int(rng.integers(0, 30)), int(rng.integers(0, 30))
            ta, tb = ("process", "file", "netflow")[a % 3], ("process", "file", "netflow")[b % 3]
            code = {("process", "process"): 1, ("process", "file"): 2, ("file", "process"): 3,
                    ("process", "netflow"): 4}.get((ta, tb))
            if code and a != b:
                L.append([a, b, code])
        days[d] = {"day": d, "cards": cards, "nodes": nodes}
        links[d] = L
    return H.LocalFakeHost(days, links, ds=f"NAMES{seed}")


def test_evidence_nf_contains_no_names():
    """Over random hosts, no neighbour name / path / IP / command line (or a 3+ char piece of one that
    is not a grammar word) appears in any evidence_nf, and every evidence_nf parses."""
    n_checked = 0
    for seed in range(6):
        h = _names_host(seed)
        for ref in ([], ["c1", "c2"]):
            prof = P.get_profile(h, ref) if ref else None
            sigs = SEL.flat(SEL.head(h, "t1", ref or None))
            nbrs = H.nf_neighbours(h, "t1", sigs)
            ev = S.section_evidence_nf(h, "t1", sigs, prof)
            for s in sigs:
                text = json.dumps(ev[s], ensure_ascii=False)
                assert H.nf_grammar(ev[s]) == [], ev[s]
                for nb in nbrs.get(s, ()):
                    for name in H.card_names(h.card(nb)):
                        pieces = {name} | set(re.split(r"[\s/\\.:|-]+", name))
                        for p in pieces:
                            if len(p) >= 3 and p.lower() not in H.VOCAB:
                                assert p not in text, (p, text)
                                n_checked += 1
    assert n_checked > 500
    hst = H.make_host()
    for ref in ([], ["c1", "c2"]):
        res = H.check_a_day(hst, "t2", ref)
        assert res["names"] == [] and res["grammar"] == []
    e1 = S.section_evidence(hst, "t2", SEL.flat(SEL.head(hst, "t2", None)))   # negative control: E1 names
    assert any("payload.sh" in ln for v in e1.values() for ln in v["lines"])


def test_evidence_nf_representatives_and_novelty(host):
    """<= 8 representatives, counts 'in a/b nodes', node-count independent; novelty = kappa unseen."""
    prof = P.get_profile(host, ["c1", "c2"])
    ix = S.day_index(host, "t2")
    sigs = SEL.flat(SEL.head(host, "t2", prof))
    ev = S.section_evidence_nf(host, "t2", sigs, prof)
    for s in sigs:
        b = len(ix.members(s, S.NF_REPS))
        assert ev[s]["representatives"] == b and 1 <= b <= S.NF_REPS
        for v in ev[s]["new"].values():
            a, bb = map(int, re.match(r"in (\d)/(\d) nodes", v).groups())
            assert 1 <= a <= bb == b
    x9q = next(s for s in sigs if host.card(s)["invariant"].get("file_name") == "x9q")
    assert ev[x9q]["new"] == {"ran_file file:staging/script": "in 1/1 nodes",
                              "wrote file:staging/script": "in 1/1 nodes",
                              "connected net:external/registered": "in 1/1 nodes"}      # relation priority order
    assert ev[x9q]["known"] == "1"                                                    # spawned_by a seen process
    raw, lk = H.kcs1_days()                                   # duplicate x9q's nodes: nothing changes
    t2 = raw["t2"]
    nid = max(map(int, t2["nodes"])) + 1
    extra = []
    for n, s in list(t2["nodes"].items()):
        if s == x9q:
            for j in range(5):
                t2["nodes"][str(nid + j)] = s
                extra += [[nid + j, b, c] for a, b, c in lk["t2"] if a == int(n)]
                extra += [[a, nid + j, c] for a, b, c in lk["t2"] if b == int(n)]
    lk["t2"] = lk["t2"] + extra
    big = H.LocalFakeHost(raw, lk, ds="KCS1BIG")
    pb = P.get_profile(big, ["c1", "c2"])
    evb = S.section_evidence_nf(big, "t2", [x9q], pb)[x9q]
    assert evb["representatives"] == 6 and {k: v.replace("/6", "/1").replace("in 6", "in 1")
                                             for k, v in evb["new"].items()} == ev[x9q]["new"]


def _host_with_links(extra, ds="KCS1LINK"):
    raw, lk = H.kcs1_days()
    lk["t2"] = lk["t2"] + [list(r) for r in extra]
    return H.LocalFakeHost(raw, lk, ds=ds)


def test_evidence_nf_connected_only_to_netflow():
    """A code-4 link from a process to a FILE is not rendered by evidence_nf (no entry, not known, not a
    neighbour), while E1 evidence still shows it; `connected` towards a netflow is rendered."""
    plain = H.local_kcs1_host()
    t2 = plain.nodes("t2")
    x9q = next(s for s in set(t2.values()) if plain.card(s)["invariant"].get("file_name") == "x9q")
    e = next(n for n, s in t2.items() if s == x9q)
    pay = next(n for n, s in t2.items() if plain.card(s)["invariant"].get("path") == H.TMP + "/payload.sh")
    known_file = next(n for n, s in t2.items() if plain.card(s)["type"] == "file" and n != pay)
    hosts = {"plain": plain, "send": _host_with_links([[e, pay, 4], [e, known_file, 4]])}
    ev, e1, nbrs = {}, {}, {}
    for name, h in hosts.items():
        for ref in ([], ["c1", "c2"]):
            prof = P.get_profile(h, ref) if ref else None
            ev[name, len(ref)] = S.section_evidence_nf(h, "t2", [x9q], prof)[x9q]
            nbrs[name, len(ref)] = H.nf_neighbours(h, "t2", [x9q]).get(x9q, set())
            e1[name, len(ref)] = S.section_evidence(h, "t2", [x9q], prof)[x9q]["lines"]
    for r in (0, 2):
        assert ev["send", r] == ev["plain", r]
        assert nbrs["send", r] == nbrs["plain", r]
        assert any(ln.startswith("connected file:") for ln in e1["send", r])
        assert not any(ln.startswith("connected file:") for ln in e1["plain", r])
    assert ev["plain", 2]["new"]["connected net:external/registered"] == "in 1/1 nodes"
    assert "connected net:external/registered" in ev["plain", 0]["all"]
    assert not any(k.startswith("connected file:") for v in ev.values() for k in v.get("new", v.get("all", {})))


# --------------------------------------------------------------------------- #
# monotonicity (test_kcs1 1113)
# --------------------------------------------------------------------------- #
def test_e2_monotone_under_added_edges(host):
    """deviation2 only gains under an augmented card; evidence_nf only gains under added links;
    removing links is caught (negative control)."""
    prof = P.get_profile(host, ["c1", "c2"])
    n = 0
    for d in host.test:
        ix = S.day_index(host, d)
        sigs = SEL.flat(SEL.head(host, d, prof))
        base = S.section_evidence_nf(host, d, sigs, prof, index=ix)
        for s in sigs:
            card = host.card(s)
            d1 = S.section_deviation2(prof, card, host.inv_key(s))
            d2 = S.section_deviation2(prof, H.augment(card), host.inv_key(s))
            assert H.dict_monotone(d1, d2), (d1, d2)
            assert list(d1)[:1] == list(d2)[:1]                                     # identity_unseen stays first
            extra = H.added_links(host, ix, s)
            after = S.section_evidence_nf(host, d, [s], prof, index=H.with_links(ix, extra))[s]
            assert H.nf_monotone(base[s], after), (base[s], after)
            n += 1
        lean = copy.copy(ix)
        lean.links = ix.links[::2]
        thin = S.section_evidence_nf(host, d, sigs, prof, index=lean)
        assert any(not H.nf_monotone(base[s], thin[s]) for s in sigs)                # the checker can fail
    assert n > 40
    for k in (0, 2):
        assert H.monotone_probes(host, k) == []


# --------------------------------------------------------------------------- #
# netflow prefix logic and deviation2 (test_kcs1 1142, 1172)
# --------------------------------------------------------------------------- #
def test_prefix_logic_synthetic_ips():
    def nf(ip, port="443", ep="external"):
        return _card("netflow", {"remote_ip": ip, "remote_port": port, "endpoint": ep})
    ref = [nf("93.184.216.34"), nf("8.8.8.8", "53")]
    chains = set()
    for c in ref:
        chains.update(P.chain_of(c))
    prof = P.HostProfile.__new__(P.HostProfile)                 # chain_unseen reads ip_chains only
    prof.ip_chains = chains
    for ip, port, unseen in (("93.184.216.34", "443", 0), ("93.184.216.34", "80", 1), ("93.184.216.99", "443", 2),
                             ("93.184.1.1", "443", 3), ("93.1.1.1", "443", 4), ("202.6.172.98", "443", 5),
                             ("8.8.8.8", "53", 0)):
        assert prof.chain_unseen(nf(ip, port)) == (unseen, 5), ip
    for ip, ep in (("10.0.0.5", "private"), ("127.0.0.1", "loopback"), ("NA", "external"), ("192.168.1.2", "external")):
        assert prof.chain_unseen(nf(ip, ep=ep)) is None
    assert P.common_prefix(nf("93.184.216.34"), nf("93.184.216.34")) == 5
    assert P.common_prefix(nf("93.184.216.34"), nf("93.184.216.34", "80")) == 4
    assert P.common_prefix(nf("93.184.216.34"), nf("93.184.7.1")) == 2
    assert P.common_prefix(nf("93.184.216.34"), nf("9.184.216.34")) == 0
    assert P.common_prefix(nf("93.184.216.34"), nf("10.0.0.1", ep="private")) is None
    assert prof.chain_unseen(nf("2001:db8::1")) is None                                   # reserved v6
    v6 = nf("2a00:1450:4001:80b::200e")
    prof.ip_chains = set(P.chain_of(v6)[:3])
    assert prof.chain_unseen(v6) == (2, 5) and len(P.chain_of(v6)) == 5
    assert S.net_precedent(nf("93.184.216.34"), nf("93.184.9.9", "8443")) == "net:external/registered prefix_match:2"
    assert S.net_precedent(nf("10.0.0.5", ep="private"), nf("93.184.9.9")).endswith("prefix_match:n/a")
    assert P.ip_levels("::ffff:93.184.216.34", 443) == ["93", "93.184", "93.184.216", "93.184.216.34",
                                                        "93.184.216.34|443"]


def test_deviation2_netflow_keeps_e1_name_key(host):
    """E2 keeps E1's netflow name key (endpoint, port class) and E1's deviation lists; a flow to a
    never-seen external address is flagged by identity_unseen (first) and remote_prefix_unseen_levels;
    reference is E1's."""
    prof = P.get_profile(host, ["c1", "c2"])
    seen_ip = next(c["invariant"]["remote_ip"] for d in ("c1", "c2") for c in host.day(d)["cards"].values()
                   if c["type"] == "netflow" and c["invariant"]["endpoint"] == "external")
    old = _card("netflow", {"remote_ip": seen_ip, "remote_port": "443", "endpoint": "external"},
                {"ops": {"connect:process": "1"}})
    new = _card("netflow", {"remote_ip": "202.6.172.98", "remote_port": "443", "endpoint": "external"},
                {"ops": {"connect:process": "1"}})
    odd = _card("netflow", {"remote_ip": "202.6.172.98", "remote_port": "4444", "endpoint": "external"},
                {"ops": {"connect:process": "1"}})
    kap = lambda c: H.signature({"type": c["type"], "invariant": c["invariant"]})  # noqa: E731
    e1 = S.section_deviation(prof, new)
    assert e1 == {"new_ops": [], "ops_above_usual": []}
    d_new = S.section_deviation2(prof, new, kap(new))
    assert list(d_new) == ["identity_unseen", "remote_prefix_unseen_levels", "new_ops", "ops_above_usual"]
    assert d_new["identity_unseen"] == "yes" and d_new["remote_prefix_unseen_levels"] == "5/5"
    assert {k: v for k, v in d_new.items() if k not in ("identity_unseen", "remote_prefix_unseen_levels")} == e1
    d_old = S.section_deviation2(prof, old, kap(old))
    assert d_old == {"remote_prefix_unseen_levels": "0/5", **S.section_deviation(prof, old)}
    d_odd = S.section_deviation2(prof, odd, kap(odd))
    assert list(d_odd) == ["identity_unseen", "remote_prefix_unseen_levels", "name_unseen", "ops_today"]
    proc = next(c for c in host.day("c1")["cards"].values() if c["type"] == "process")
    assert S.section_deviation2(prof, proc, kap(proc)) == S.section_deviation(prof, proc)
    states, sigs = _e2(host)
    for sg in sigs:
        card = host.card(sg)
        ref = SAN.sanitize(S.section_reference(prof, card, sg, host.inv_key(sg)))
        assert states[sg]["qv"]["reference"] == ref
        if "qa" in states[sg]:
            assert states[sg]["qa"]["reference"] == ref
        assert states[sg]["qv"]["deviation"] == SAN.sanitize(S.section_deviation2(prof, card, host.inv_key(sg)))


# --------------------------------------------------------------------------- #
# golden states of the ORIGINAL code
# --------------------------------------------------------------------------- #
def test_fake_host_states_equal_the_original():
    """Every E1 level and E2 state of the synthetic host (k = 2 and 0, both test days) hashes to the
    original code's state (tests/golden/state/fake_states.json, from scripts/dev/golden_states.py)."""
    gold = H.golden_json("fake_states.json")
    if gold is None:
        pytest.fail("tests/golden/state/fake_states.json is missing (scripts/dev/golden_states.py)")
    H.clear_caches()
    host = H.local_kcs1_host()
    from habitus.state.ladder import StateBuilder
    n = 0
    for ref, k in ((["c1", "c2"], 2), ([], 0)):
        for d in host.test:
            h = SEL.head(host, d, ref or None)
            assert h == gold[f"head|k{k}|{d}"]
            sigs = SEL.flat(h)
            for lv in ("L0", "L1", "L2", "L3", "L4"):
                st = StateBuilder(host, ref, lv, k).build(d, sigs)
                assert {s: H.sha1(B.serialize(x)) for s, x in st.items()} == gold[f"{lv}|k{k}|{d}"], (lv, k, d)
                n += len(st)
            st = B.StateBuilderE2(host, ref, k).build(d, sigs)
            assert {s: {q: H.sha1(B.serialize(x)) for q, x in qs.items()} for s, qs in st.items()} \
                == gold[f"E2|k{k}|{d}"], ("E2", k, d)
    x9q = next(s for s in host.nodes("t2").values() if host.card(s)["invariant"].get("file_name") == "x9q")
    st = B.StateBuilderE2(host, ["c1", "c2"], 2).build("t2", [x9q])[x9q]
    texts = {q: B.serialize(x) for q, x in st.items()}
    assert {q: {"sha1": H.sha1(t), "len": len(t), "neutralised": t.count("(im_end)"),
                "raw_control": t.count("<|im_end|>")} for q, t in texts.items()} == gold["x9q"]
    assert gold["x9q"]["qv"]["neutralised"] == 1 and gold["x9q"]["qv"]["raw_control"] == 0 and n > 100


def test_fakes_kcs1_host_matches_local_port():
    """tests/fakes.make_kcs1_host (B1) builds the same days as the local port (state hashes depend on it)."""
    try:
        from fakes import make_kcs1_host
    except ImportError:
        pytest.skip("tests/fakes.py not present yet")
    a, b = make_kcs1_host(), H.local_kcs1_host()
    assert a.ds == b.ds and a.os == b.os and list(a.clean) == b.clean and list(a.test) == b.test
    for d in b.clean + b.test:
        assert a.day(d)["cards"] == b.day(d)["cards"] and a.nodes(d) == b.nodes(d)
        assert a.links(d).tolist() == b.links(d).tolist()


def _samples():
    idx = H.golden_json("index.json")
    return [] if idx is None else idx["samples"]


@pytest.mark.parametrize("sample", _samples(), ids=lambda s: s["file"].replace(".json.gz", ""))
def test_golden_states_real_hosts(sample, tmp_path):
    """G3: per head signature, sha1(serialize(state[qid])) (E1: the single state) and the head lists
    equal the ORIGINAL code's (tests/golden/state/<ds>_<level>_k<k>_<day>.json.gz); G2: the sampled
    Form.user() / Form.system equal the original user / system prompts."""
    _check_golden(sample["file"], tmp_path, base=H.real_precedent_base(), prompts=True)


@pytest.mark.parametrize("name", ["CLEARSCOPE_E3_E2_k2_2018-04-03.json.gz", "CADETS_E3_E2_k2_2018-04-02.json.gz"])
def test_golden_states_with_a_cold_precedent_cache(name, tmp_path):
    """Without the shipped precedent cache every precedent comes from a fresh RefIndex over the
    reference cards: the states are still the original ones."""
    if H.golden_json(name) is None:
        pytest.fail(f"tests/golden/state/{name} is missing")
    built = []
    real_get = PREC.get_refindex
    PREC.get_refindex = lambda *a: built.append(a) or real_get(*a)
    try:
        _check_golden(name, tmp_path, base=None, prompts=False)
    finally:
        PREC.get_refindex = real_get
    assert built, "the RefIndex was never built"


def _check_golden(name, tmp_path, base, prompts):
    gold = H.golden_json(name)
    host = H.real_host(gold["ds"])
    if host is None:
        H.skip_or_fail(f"no card files for {gold['ds']}")
    PREC.set_roots(base=base, overlay=str(tmp_path / "prec"))
    H.clear_caches()
    try:
        ref, caps, k = gold["ref"], gold["caps"], {"0": 0, "2": 2, "all": None}[gold["k"]]
        h = SEL.head(host, gold["day"], ref or None, caps)
        assert h == gold["head"]
        sigs = SEL.flat(h)
        if gold["level"] == "E2":
            st = B.StateBuilderE2(host, ref, k).build(gold["day"], sigs)
            texts = {s: {q: B.serialize(x) for q, x in qs.items()} for s, qs in st.items()}
            got = {s: {q: H.sha1(t) for q, t in qs.items()} for s, qs in texts.items()}
        else:
            from habitus.state.ladder import StateBuilder
            st = StateBuilder(host, ref, gold["level"], k).build(gold["day"], sigs)
            texts = {s: {"state": B.serialize(x)} for s, x in st.items()}
            got = {s: H.sha1(t["state"]) for s, t in texts.items()}
        bad = [s for s in sigs if got[s] != gold["states"][s]]
        assert not bad, f"{len(bad)} of {len(sigs)} states differ, e.g. {bad[:3]}"
        dig = hashlib.sha1()
        for s in sigs:
            for q, t in texts[s].items():
                dig.update(f"{s}|{q}|{t}\n".encode("utf-8"))
        assert dig.hexdigest() == gold["digest"] and len(sigs) == gold["n_states"]
        if not prompts:
            return
        from habitus.questions import routed as R
        system = R.SYSTEM_E2 if gold["level"] == "E2" else R.SYSTEM
        sample = [p for p in H.golden_json("prompts.json")["prompts"] if p["file"] == name]
        assert sample
        for p in sample:
            tq = R.to_typed(p["sig"], p["qid"], texts[p["sig"]][p["state_key"]], system=system)
            (form,) = [f for f in tq.forms if f.name == p["variant"]]
            assert H.sha1(form.user()) == p["user_sha1"] and H.sha1(form.system) == p["system_sha1"], p
    finally:
        PREC.set_roots(base=None)
        H.clear_caches()


def test_golden_index_covers_the_g2_samples():
    idx = H.golden_json("index.json")
    assert idx is not None and set(idx["datasets"]) == {"CADETS_E3", "THEIA_E3", "CLEARSCOPE_E3", "optc_h501",
                                                        "ATLASV2_EDR"}
    have = {(x["ds"], x["level"], x["k"], x["day"], x["role"]) for x in idx["samples"]}
    assert ("CADETS_E3", "E2", "2", "2018-04-06", "test") in have and len(have) == 12
    assert {r for *_x, r in have} == {"test", "null"}
    assert H.golden_json("prompts.json")["n"] == 200
