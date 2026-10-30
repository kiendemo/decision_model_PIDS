"""Frozen E1 outputs of the original KC-S1 code (tests/golden/kcs1_e1_snapshot.json.gz, a byte copy
of the snapshot taken 2026-10-16 before any E2 code existed, extended the same day with the extra
part). test_e1_snapshot.test_e1_snapshot_unchanged recomputes every part with the habitus code and
requires equality: the E1 states, prompts, cache keys and node scores are byte-identical.

    python tests/e1_snapshot.py            # recompute and print the differences (none = UNCHANGED)

What is frozen:
  fake   the synthetic host (tests/fakes.make_kcs1_host): every head state at L0..L4 for k = 0 and
         k = 2, as sha1 of the serialisation, the digest of every E1 prompt's cache key per
         (level, k), and the answers on 4 L4 states through JevDecisionModel with the fake forward
         (tests/dm_helpers.fake_logits): render -> sha1 keys -> letter softmax -> record(level='E1').
  real   CADETS_E3 and optc_h501, first test day, head caps 40/30/30, k in {0, 2, all}, L0..L4:
         sha1 of every serialised state and, per (host, level, k), a digest of the sha1 of every
         rendered prompt (qv fwd/rev, qt pos/neg), i.e. of the answer cache keys.
  text   sha1 of SYSTEM and of the QUESTIONS table (E1 texts, letters, levels).
  extra  realcap      CADETS_E3 2018-04-12, L4 k = 2 with the head caps 4000/2000/6000: digest of
                      every serialised state and of every E1 prompt / cache key;
         lodo         THEIA_E3, first k = all LODO null day (reference = the other clean days),
                      caps 40/30/30, L0..L4: state and prompt digests;
         node_scores  sha1 of node_scores(composite) (float64 bytes, aligned to the universe) of the
                      release E1 score files, and of two LODO null pools (label-free).
Each real / extra part needs the release cards (and score files); the prompt parts need the
pinned Qwen3.5 tokenizer snapshot.
"""

import gzip
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "golden", "kcs1_e1_snapshot.json.gz")
HOSTS = ("CADETS_E3", "optc_h501")
CAPS = {"process": 40, "file": 30, "netflow": 30}
LEVELS = ("L0", "L1", "L2", "L3", "L4")
REALCAP = ("CADETS_E3", "2018-04-12", "L4", 2)            # the full head caps
LODO_HOST = "THEIA_E3"                                     # k = all LODO, first clean day
NODE_SCORES = (("CADETS_E3", "L4", None), ("CADETS_E3", "L4", 2), ("CADETS_E3", "L3", 0), ("CADETS_E3", "L2", None),
               ("THEIA_E3", "L4", None), ("THEIA_E3", "L3", 0), ("ATLASV2_EDR", "L4", None))
NULL_POOLS = (("CADETS_E3", "L4", None), ("THEIA_E3", "L4", 2))

if HERE not in sys.path:                                   # fakes / state_helpers / dm_helpers
    sys.path.insert(0, HERE)


def _sha(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _dm():
    """A tokenizer-only JevDecisionModel (renders and keys prompts; no cache, no model), or None
    when the pinned tokenizer snapshot is missing."""
    from habitus.dm.jev import JevDecisionModel
    try:
        dm = JevDecisionModel(device="cpu", cache_only=True, ds=None, verbose=False)
        dm.tok                                              # noqa: B018  (loads the tokenizer)
    except Exception as e:                                  # whatever a missing snapshot raises
        print(f"[e1_snapshot] no tokenizer: {type(e).__name__}: {e}", file=sys.stderr)
        return None
    return dm


def _prompt_digest(dm, states):
    """Digest over (key, qid, vid, sha1(rendered prompt)) for E1's 4 prompts of every state."""
    from habitus.questions.routed import QUESTIONS, SYSTEM, to_typed
    h = hashlib.sha1()
    for key in sorted(states):
        for qid in QUESTIONS:
            for f in to_typed(key, qid, states[key], system=SYSTEM).forms:
                h.update(f"{key}|{qid}|{f.name}|{dm.key(f)}\n".encode())
    return h.hexdigest()


def _fake_engine(host):
    """E1 answers on 4 L4 states of t2 through JevDecisionModel(forward=fake_logits)."""
    import dm_helpers
    import state_helpers
    from habitus.dm.jev import JevDecisionModel
    from habitus.questions.routed import QUESTIONS, SYSTEM, record, to_typed
    eng = JevDecisionModel(device="cpu", forward=dm_helpers.fake_logits, ds=None, verbose=False)
    states = sorted(set(state_helpers.build_all(host, "L4", days=["t2"]).values()))[:4]
    typed = {str(i): {q: to_typed(str(i), q, s, system=SYSTEM) for q in QUESTIONS} for i, s in enumerate(states)}
    dec = eng.decide([tq for per in typed.values() for tq in per.values()])
    out = {}
    for key, per in typed.items():
        r = record({q: dec[tq.qkey] for q, tq in per.items()}, level="E1")
        out[key] = ({f: round(float(r[f]), 9) for f in ("qv", "qt", "cov_min")}
                    | {"raw_keys": sorted(f for tq in per.values() for f in dec[tq.qkey].forms)})
    eng.close()
    return out


def _fake_part(dm):
    import fakes
    import state_helpers
    from habitus.state.builder import serialize
    from habitus.state.ladder import StateBuilder
    from habitus.state.select import flat, head
    state_helpers.clear_caches()
    host = fakes.make_kcs1_host()
    out = {"states": {}, "prompts": {}, "engine": None}
    for k, ref in ((0, []), (2, ["c1", "c2"])):
        for lv in LEVELS:
            sb = StateBuilder(host, ref, lv, k)
            st = {}
            for d in host.test:
                for s, x in sb.build(d, flat(head(host, d, ref or None))).items():
                    st[f"{d}|{s}"] = serialize(x)
            out["states"].update({f"{lv}|k{k}|{key}": _sha(t) for key, t in st.items()})
            if dm is not None:
                out["prompts"][f"{lv}|k{k}"] = _prompt_digest(dm, st)
    if dm is not None:
        out["engine"] = _fake_engine(host)
    return out


def _cards_dir():
    from habitus.config import settings
    return str(settings().paths.cards)


def _has_cards(ds):
    return os.path.isdir(os.path.join(_cards_dir(), ds))


def _host(ds):
    from habitus.data.host import Host
    return Host(ds)


def _kl(k):
    return "all" if k is None else str(k)


def _real_part(dm, hosts=HOSTS):
    from habitus.decide.routed import ref_for
    from habitus.state.builder import serialize
    from habitus.state.ladder import StateBuilder
    from habitus.state.select import flat, head
    out = {"states": {}, "prompts": {}, "days": {}}
    for ds in hosts:
        if not _has_cards(ds):
            continue
        host = _host(ds)
        day = host.test[0]
        out["days"][ds] = day
        for k in (0, 2, None):
            ref = ref_for(host, k)
            sigs = flat(head(host, day, ref or None, CAPS))
            kl = _kl(k)
            for lv in LEVELS:
                st = {s: serialize(x) for s, x in StateBuilder(host, ref, lv, k).build(day, sigs).items()}
                out["states"].update({f"{ds}|{day}|{lv}|k{kl}|{s}": _sha(t) for s, t in st.items()})
                if dm is not None:
                    out["prompts"][f"{ds}|{lv}|k{kl}"] = _prompt_digest(dm, st)
    return out


def _digest_states(st):
    h = hashlib.sha1()
    for s in sorted(st):
        h.update(f"{s}|{_sha(st[s])}\n".encode())
    return h.hexdigest()


def _digest_array(a):
    import numpy as np
    return hashlib.sha1(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


def _extra_part(dm):
    """realcap / lodo states and prompts, node_scores / null pool digests (module docstring)."""
    from habitus.decide import null as N
    from habitus.decide import scorefile as SF
    from habitus.state.builder import serialize
    from habitus.state.ladder import StateBuilder
    from habitus.state.select import CAPS as HEAD_CAPS
    from habitus.state.select import flat, head
    out = {"states": {}, "prompts": {}, "node_scores": {}, "null_pools": {}, "n": {}}
    hosts = {}

    def host_of(ds):
        if ds not in hosts:
            hosts[ds] = _host(ds)
        return hosts[ds]

    ds, day, lv, k = REALCAP
    if _has_cards(ds):
        host = host_of(ds)
        ref = host.shots(k)
        sigs = flat(head(host, day, ref, HEAD_CAPS))
        st = {s: serialize(x) for s, x in StateBuilder(host, ref, lv, k).build(day, sigs).items()}
        key = f"realcap|{ds}|{day}|{lv}|k{_kl(k)}"
        out["states"][key], out["n"][key] = _digest_states(st), len(st)
        if dm is not None:
            out["prompts"][key] = _prompt_digest(dm, st)
    if _has_cards(LODO_HOST):
        host = host_of(LODO_HOST)
        day, ref = N.lodo_plan(host, None)[0]
        sigs = flat(head(host, day, ref, CAPS))
        for lv in LEVELS:
            st = {s: serialize(x) for s, x in StateBuilder(host, ref, lv, None).build(day, sigs).items()}
            key = f"lodo|{LODO_HOST}|{day}|{lv}|kall|ref{len(ref)}"
            out["states"][key], out["n"][key] = _digest_states(st), len(st)
            if dm is not None:
                out["prompts"][key] = _prompt_digest(dm, st)
    for ds, lv, k in NODE_SCORES:
        if not _has_cards(ds):
            continue
        host = host_of(ds)
        by_day, missing = SF.load_by_day(ds, lv, k, host.test)
        if by_day and not missing:
            s = SF.node_scores(host, by_day, "composite")
            out["node_scores"][f"{ds}|{lv}|k{_kl(k)}"] = [len(s), _digest_array(s)]
    for ds, lv, k in NULL_POOLS:
        if not _has_cards(ds):
            continue
        n = N.lodo_scores(host_of(ds), k, lv, "composite")
        if n is not None:
            out["null_pools"][f"{ds}|{lv}|k{_kl(k)}"] = [len(n["values"]), n["n_days"], _digest_array(n["values"])]
    return out


def _text_part():
    from habitus.questions import routed as Q
    return {"system": _sha(Q.SYSTEM), "questions": _sha(json.dumps(Q.QUESTIONS, sort_keys=True)),
            "letters": list(Q.LETTERS)}


def compute(real=True, extra=True):
    dm = _dm()
    out = {"tokenizer": dm is not None, "text": _text_part(), "fake": _fake_part(dm),
           "real": _real_part(dm) if real else None}
    if real and extra:
        out["extra"] = _extra_part(dm)
    return out


def load():
    with gzip.open(PATH, "rt", encoding="utf-8") as f:
        return json.load(f)


def diff(a, b, path=""):
    """Human-readable list of differences (at most 20)."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'snapshot' if k in a else 'now'}")
            else:
                out += diff(a[k], b[k], f"{path}/{k}")
            if len(out) >= 20:
                break
    elif a != b:
        out.append(f"{path}: {str(a)[:80]} != {str(b)[:80]}")
    return out[:20]


if __name__ == "__main__":
    d = diff(load(), json.loads(json.dumps(compute())))
    print("E1 snapshot " + ("UNCHANGED" if not d else "DIFFERS:\n" + "\n".join(d)), flush=True)
    sys.exit(1 if d else 0)
