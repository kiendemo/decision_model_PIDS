"""Routed decision stage: one score file per (ds, level, k, day) from DecisionModel answers.

    score_job(host, dm, job)   head -> states -> typed questions -> dm.decide -> records -> file dict
    run(ds, dm=, root=, jobs=) score_job + write for every job; resumable

E2 (the main system), per job:
  head (select.head with the job's reference and caps), StateBuilderE2 routed states (one state per
  question: qv, qt and, at k >= 1, qa), dedupe by the join key
  state_hash('\\x1e'.join(q + '\\x1f' + serialize(state[q]) for q sorted)), questions.routed.to_typed
  (system SYSTEM_E2), dm.refused (any refusal raises Refused: a state is never cut), dm.decide,
  questions.routed.record(level='E2') -> per signature
      {"qv", "qt", "qa" (None = not asked), "hard" (hard tier of the QV state), "cov_min",
       "base" = compose, "composite" = compose_e2}.
E1 (the L0..L4 state ladder, ablation only):
  head, ladder.StateBuilder(level) states, dedupe by state_hash(serialize(state)), to_typed with the
  E1 system turn SYSTEM for qv and qt, record(level='E1') -> {"qv", "qt", "cov_min"} only (the E1
  composite max(qv, qt) is computed at read time, composite.value).
Every non-head signature of the day gets the rarity fallback -1 + 0.5 / (1 + n_s) ("fallback").

A score file is {"meta", "head", "scores": {sig: rec}, "fallback": {non-head sig: float}}. The meta
records the job (ds, day, level, k, ref, role, caps), head sizes, state counts, timings, the
DecisionModel's answer-cache tag ("engine") and its engine path: prefix_share, batch_size and the
prompt grouping ("qvqt" for E2 = the QV + QT prompts of a signature share a prefix, QA apart;
"state" for E1 = the 4 prompts of a state; None without prefix sharing). "low_cov" counts the
scored signatures whose cov_min < LOW_COV (flagged, never dropped).

run() never overwrites: an existing file is skipped when its meta caps and reference equal the
job's, and anything else raises (one head rule and one reference per file name and root), so a
recompute needs a fresh root. Writes go under runs/ only (scorefile.check_writable).
"""

import os
import time

from habitus.decide.composite import LAMBDA, compose, compose_e2
from habitus.decide.scorefile import check_writable, klabel, read_meta, runs_root, score_path, write_score
from habitus.log import log
from habitus.state.select import CAPS, TYPES, fallback_rank, flat, head

E2 = "E2"
LEVELS = ("L0", "L1", "L2", "L3", "L4")
KS = (0, 2, None)                               # k in {0, 2, all} only
E1_GROUPING, E2_GROUPING = "state", "qvqt"      # prompt grouping labels written into the meta
DESIGN_SPEC = "docs/DESIGN.md"


def check_ks(ks):
    """parse_k for the CLI ("all,2,0" or a list): k in {0, 2, all} only."""
    from habitus.decide.scorefile import parse_k
    try:
        out = parse_k(ks) if isinstance(ks, str) else list(ks)
    except ValueError:
        out = []
    if not out or any(k not in KS for k in out):
        raise ValueError(f"ks {ks!r}: k in {{0, 2, all}} only")
    return out


def ref_for(host, k):
    """The reference days of a TEST day at k: none at k = 0, else the first k clean days (all of
    them at k = all)."""
    return [] if k == 0 else host.shots(k)


def builder(builders, key, make):
    """The state builder of `key` = (level, k label, sorted ref), made by make() when missing.
    Builders of any other reference are dropped first (they would keep their profiles alive); the
    profile / RefIndex / day-index LRUs make a rebuilt builder cheap."""
    for other in [x for x in builders if x[2] != key[2]]:
        del builders[other]
    if key not in builders:
        builders[key] = make()
    return builders[key]


def _dm_spec(dm):
    fn = getattr(dm, "spec", None)
    try:
        return dict(fn()) if callable(fn) else {}
    except Exception:                       # a DM without a usable spec(): engine path unknown
        return {}


def engine_path(dm, level):
    """Score-file meta of the DecisionModel's engine path: prefix sharing, batch size and the prompt
    grouping. A 'direct' grouping (one prefix-share group per TypedQuestion.group) is recorded
    under the level's label (E2 'qvqt', E1 'state'), which is what those groups are."""
    spec = _dm_spec(dm)
    ps = spec.get("prefix_share", getattr(dm, "prefix_share", None))
    bs = spec.get("batch_size", getattr(dm, "batch_size", None))
    g = spec.get("grouping", getattr(dm, "grouping", None))
    label = E2_GROUPING if level == E2 else E1_GROUPING
    return {"prefix_share": ps, "batch_size": bs, "grouping": (label if g in (None, "direct") else g) if ps else None}


def _decide(dm, qs):
    """dm.decide(qs) after dm.refused(qs): any refusal raises Refused (a state is never cut)."""
    bad = dm.refused(qs)
    if bad:
        from habitus.dm.base import Refused
        raise Refused(bad, f"{getattr(dm, 'tag', dm)} refused {len(bad)} of {len(qs)} questions "
                           f"(states are never cut), e.g. {dict(list(bad.items())[:2])}")
    return dm.decide(qs)


def _meta(host, job, caps, h, sigs, uniq, scores, seconds, dm, extra=None):
    meta = {"ds": host.ds, "day": job.day, "level": job.level, "k": klabel(job.k), "ref": list(job.ref),
            "role": job.role, "caps": caps, "n_head": {t: len(h[t]) for t in TYPES}, "n_states": len(sigs),
            "n_unique_states": len(uniq)}
    meta.update(extra or {})
    from habitus.questions.routed import LOW_COV
    meta.update({"seconds": seconds, "engine": getattr(dm, "tag", None), **engine_path(dm, job.level),
                 "low_cov": sum(1 for r in scores.values() if r["cov_min"] < LOW_COV),
                 "written": time.strftime("%Y-%m-%d %H:%M:%S")})
    return meta


def _score_e2(host, dm, job, caps, builders):
    from habitus.questions.routed import SYSTEM_E2, record, to_typed
    from habitus.state.builder import E2_ROUTES, StateBuilderE2, hard_tier, serialize, state_hash
    ref = list(job.ref)
    t0 = time.time()
    h = head(host, job.day, ref or None, caps)
    sigs = flat(h)
    states = builder(builders, (E2, klabel(job.k), tuple(sorted(ref))),
                     lambda: StateBuilderE2(host, ref, job.k)).build(job.day, sigs)
    uniq, of = {}, {}
    for s in sigs:
        qs = {q: serialize(x) for q, x in states[s].items()}
        hk = state_hash("\x1e".join(f"{q}\x1f{qs[q]}" for q in sorted(qs)))
        if hk not in uniq:
            uniq[hk] = (s, qs)              # the first signature with this routed state stands for it
        of[s] = hk
    typed = {hk: {q: to_typed(rep, q, txt, system=SYSTEM_E2) for q, txt in qs.items()}
             for hk, (rep, qs) in uniq.items()}
    t1 = time.time()
    dec = _decide(dm, [tq for per in typed.values() for tq in per.values()])
    ans = {hk: record({q: dec[tq.qkey] for q, tq in per.items()}, level=E2) for hk, per in typed.items()}
    t2 = time.time()
    scores = {}
    for s in sigs:
        a = ans[of[s]]
        rec = {"qv": a["qv"], "qt": a["qt"], "qa": a.get("qa"), "hard": hard_tier(states[s]["qv"]),
               "cov_min": a["cov_min"]}
        rec["base"] = compose(rec)
        rec["composite"] = compose_e2(rec)
        scores[s] = rec
    extra = {"n_qa": sum(1 for r in scores.values() if r["qa"] is not None),
             "n_hard": sum(1 for r in scores.values() if r["hard"]),
             "e2": {"spec": DESIGN_SPEC, "lambda": LAMBDA, "routes": E2_ROUTES}}
    return h, sigs, uniq, scores, (t0, t1, t2), extra


def _score_e1(host, dm, job, caps, builders):
    from habitus.questions.routed import QUESTIONS, SYSTEM, record, to_typed
    from habitus.state.builder import serialize, state_hash
    from habitus.state.ladder import StateBuilder
    ref = list(job.ref)
    t0 = time.time()
    h = head(host, job.day, ref or None, caps)
    sigs = flat(h)
    states = builder(builders, (job.level, klabel(job.k), tuple(sorted(ref))),
                     lambda: StateBuilder(host, ref, job.level, job.k)).build(job.day, sigs)
    text = {s: serialize(st) for s, st in states.items()}
    uniq, of = {}, {}
    for s, t in text.items():
        hk = state_hash(t)
        if hk not in uniq:
            uniq[hk] = (s, t)
        of[s] = hk
    typed = {hk: {q: to_typed(rep, q, t, system=SYSTEM) for q in QUESTIONS} for hk, (rep, t) in uniq.items()}
    t1 = time.time()
    dec = _decide(dm, [tq for per in typed.values() for tq in per.values()])
    ans = {hk: record({q: dec[tq.qkey] for q, tq in per.items()}, level="E1") for hk, per in typed.items()}
    t2 = time.time()
    scores = {s: {"qv": ans[of[s]]["qv"], "qt": ans[of[s]]["qt"], "cov_min": ans[of[s]]["cov_min"]} for s in sigs}
    return h, sigs, uniq, scores, (t0, t1, t2), None


def score_job(host, dm, job, caps=CAPS, builders=None):
    """The score-file dict {"meta", "head", "scores", "fallback"} of one job (nothing written).
    `builders`: a dict reused across jobs (see builder()); a fresh one when None."""
    if job.ds != host.ds:
        raise ValueError(f"job {job} is not for host {host.ds}")
    if job.level != E2 and job.level not in LEVELS:
        raise ValueError(f"unknown level {job.level!r}")
    if job.k not in KS:
        raise ValueError(f"k {job.k!r}: k in {{0, 2, all}} only")
    builders = {} if builders is None else builders
    fn = _score_e2 if job.level == E2 else _score_e1
    h, sigs, uniq, scores, (t0, t1, t2), extra = fn(host, dm, job, caps, builders)
    inh = set(sigs)
    fb = {s: v for s, v in fallback_rank(host, job.day).items() if s not in inh}
    seconds = {"states": round(t1 - t0, 1), "engine": round(t2 - t1, 1)}
    meta = _meta(host, job, caps, h, sigs, uniq, scores, seconds, dm, extra)
    return {"meta": meta, "head": h, "scores": scores, "fallback": fb}


def _existing(path, job, caps):
    """'skip' when the file at `path` was scored with the job's caps and reference; raises else."""
    m = read_meta(path)
    if m.get("caps") != caps or sorted(m.get("ref") or []) != sorted(job.ref):
        raise ValueError(f"{path} exists with caps {m.get('caps')} ref {m.get('ref')}, but this run "
                         f"asks caps {caps} ref {sorted(job.ref)}: use another root")
    return "skip"


def run(ds, *, dm, root=None, jobs, host=None, caps=CAPS, settings=None):
    """Score and write every job of dataset `ds` under `root` (default <runs>/scores); returns
    {"done": n, "skip": n}. `host`: an already built Host (default Host(ds) over the cards)."""
    if host is None:
        from habitus.data.host import Host
        host = Host(ds)
    if host.ds != ds:
        raise ValueError(f"host {host.ds} given for ds {ds}")
    root = runs_root(settings) if root is None else str(root)
    check_writable(os.path.join(root, ds), settings)
    jobs = list(jobs)
    if any(j.role == "null" for j in jobs):            # LODO references: keep every clean day parsed
        keep = getattr(host, "keep_clean_days", None)
        if callable(keep):
            keep()
    builders, n = {}, {"done": 0, "skip": 0}
    t_all = time.time()
    for i, job in enumerate(jobs, 1):
        path = score_path(ds, job.level, job.k, job.day, root)
        if os.path.exists(path):
            n[_existing(path, job, caps)] += 1
            continue
        obj = score_job(host, dm, job, caps, builders)
        write_score(obj, path, settings)
        m, h = obj["meta"], obj["head"]
        log(f"{i}/{len(jobs)} +{time.time() - t_all:.0f}s {ds} {job.day} {job.level} k={klabel(job.k)} "
            f"({job.role}): head {'/'.join(str(len(h[t])) for t in TYPES)} (P/F/N), "
            f"{m['n_unique_states']} unique states, states {m['seconds']['states']}s, "
            f"engine {m['seconds']['engine']}s, cov<0.5 {m['low_cov']}", "decide.routed")
        n["done"] += 1
    log(f"{ds}: {n['done']} written, {n['skip']} skipped in {time.time() - t_all:.0f}s", "decide.routed")
    return n
