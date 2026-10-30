"""The habit channel driver over ANY DecisionModel (wrapped as types.DMAnswerSource).

    from habitus.decide.habit import channel
    src = channel.source_for("HB-L.mq", "loho-THEIA_E3", "THEIA_E3")      # DMAnswerSource of the arm
    channel.score_level("THEIA_E3", "HB-L.mq", "k2", src, fold="loho-THEIA_E3")
    vec, why = channel.habit_vectors(host, "HB-L.mq_k2")                   # node vectors of the test days

Pipeline of one (host, level, kplan) (score_level):
  1. per (Ref, role) of null2.plan_refs (null days first): prof = profile2.get_profile2 (None at k0),
     head = head.q0, routes = the frozen E2 file's (retrofit.e2_day / e2_routes; its head must be
     q0's), tiers = attest.lineage_tiers over every head member;
  2. habit_day (views_for + own_views, the text.fits gate, candidates, targets.observe, a question
     for every bucket present on >= 1 member, one per V_par / V_own), answers from the level's
     AnswerSource (a question the source refuses is n/a, never cut), x per node (surprise.node_x),
     stored as <root>/nulls/<ds>/<ldir>/x_<day>.json.gz;
  3. FamilyNull on the null days' units, z / E / E_mono per node (null days cross-fitted: exclude_day =
     own day), EMap on null (day, ntype, E, max(qv, qt)), s_hat per node (shat=False: E only);
  4. score.rows_for_day -> types.ScoreDay -> score.write under <root>/scores.
HB0 has no habit answers: retrofit.run writes it. Decisions and metrics are habitus.eval's.

Writes go under the habit root (runs/habit) only. Label-free: nothing here reads a label or imports
habitus.eval (the label guard blocks it).
"""

import glob
import gzip
import json
import os
import re
import time

import numpy as np

from habitus.decide.habit import null2, retrofit
from habitus.decide.habit import score as SC
from habitus.decide.habit import surprise as SU
from habitus.log import log
from habitus.questions.habit import attest as A
from habitus.questions.habit import head as HD
from habitus.questions.habit import targets as TG
from habitus.questions.habit import text as Q
from habitus.questions.habit import types as T
from habitus.questions.habit import views as V
from habitus.questions.habit.constants import (ARM_DM, CLONE_ARMS, DEV_HOSTS, FOLDS, HABIT_LEVELS, KPLANS,
                                               LEVELS)
from habitus.questions.habit.profile2 import get_profile2
from habitus.state.select import CAPS, TYPES

KS_ALIASES = {"0": "k0", "k0": "k0", "2": "k2", "k2": "k2", "allc-pre": "allc-pre"}
LEGACY_K = {"k0": 0, "k2": 2, "allc-pre": None}      # frozen routed E2 directories (E2_k0 / E2_k2 / E2_kall)
LEGACY_FIELD = {"s": "composite", "base": "base", "qv": "qv", "qt": "qt"}
HABIT_COLUMNS = ("s_hat", "E")                       # habit-only vectors: the first column with a value
ATTEST, A_NET = retrofit.ATTEST, retrofit.A_NET
AUTO = "auto"


def _log(msg):
    log(msg, "habit.channel")


# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #
def parse_kplans(spec):
    """'0,2,allc-pre' (or a sequence) -> ['k0', 'k2', 'allc-pre'] (KPLANS names, order kept,
    duplicates dropped). The routed 'all' is refused: on CADETS_E3 it reads clean days after the first
    test day; say allc-pre."""
    items = [x.strip() for x in spec.split(",")] if isinstance(spec, str) else [str(x).strip() for x in spec]
    out = []
    for x in items:
        if not x:
            continue
        if x not in KS_ALIASES:
            raise ValueError(f"k plan {x!r}: use one of {sorted(KS_ALIASES)} (KPLANS {KPLANS})")
        if KS_ALIASES[x] not in out:
            out.append(KS_ALIASES[x])
    if not out:
        raise ValueError(f"no k plan in {spec!r}")
    return out


def fold_for(ds, override=None):
    """The fold whose model scores `ds`: loho-<ds> for a dev host, else "all" (the train-only host's
    rows are host-seen). A dev host is never scored by a fold that trains on it (also for an
    explicit fold)."""
    fold = override or (f"loho-{ds}" if ds in DEV_HOSTS else "all")
    if fold not in FOLDS:
        raise ValueError(f"fold {fold!r} not in {sorted(FOLDS)}")
    if ds in DEV_HOSTS and ds in FOLDS[fold]:
        raise ValueError(f"{ds}: fold {fold} trains on {ds} (its own pre-deployment days); a dev host is scored "
                         f"by loho-{ds} only")
    return fold


def source_for(arm, fold, ds, *, device=None, cache_only=None, settings=None, **extra):
    """types.DMAnswerSource of a habit arm (constants.ARM_DM; habitus.dm.get_arm), fold-checked
    against the scored host (fold_for); trained arms also check their run records' fold."""
    if arm not in ARM_DM:
        raise ValueError(f"arm {arm!r} not in {sorted(ARM_DM)}")
    fold = fold_for(ds, fold)
    from habitus.dm import get_arm
    dm = get_arm(arm, ds=ds, fold=fold, device=device, cache_only=cache_only, settings=settings, **extra)
    return T.DMAnswerSource(dm)


def default_tokenizer(settings=None):
    """The pinned Qwen3.5-4B tokenizer of the MAX_VIEW_TOKENS gate (dm.jev.load_tokenizer)."""
    from habitus.dm.jev import load_tokenizer
    from habitus.state.sanitize import QWEN_BACKBONE, QWEN_REVISION
    return load_tokenizer(QWEN_BACKBONE, QWEN_REVISION, settings=settings)


def _host(ds_or_host, settings=None):
    if isinstance(ds_or_host, str):
        return retrofit.default_host(ds_or_host, settings)
    return ds_or_host


# --------------------------------------------------------------------------- #
# One (host, Ref): views, candidates, observations, questions
# --------------------------------------------------------------------------- #
def habit_day(host, ref, prof, sigs, tok=None, families=None, all_buckets=False):
    """Step 2 of the pipeline for one (host, Ref) and head signatures `sigs`:
    {"views": {sig: {ViewKind: View}}, "own": {sig: {owner bucket: V_own View}}, "cands": {(sig,
    family): Candidates}, "obs": {node: NodeObs}, "questions": [HabitQuestion] (unique qkeys, sorted),
    "q_sigs": {qkey: sorted sigs}, "refused": [(sig, view kind or owner bucket)]}. Questions: every
    inclusion bucket present on >= 1 member (all_buckets: every candidate bucket), one P1 per V_par
    with an applicable member, one N2 per V_own. With `tok`, a view whose text is longer than
    MAX_VIEW_TOKENS is dropped (its families n/a), never truncated."""
    fams = T.available(host.ds, ref.k_mode, getattr(host, "obs", None)) if families is None else frozenset(families)
    views, own, cands, refused = {}, {}, {}, []
    for s in sigs:
        card = host.card(s)
        vs = V.views_for(card, prof, ref.k_mode, fams)
        if tok is not None:
            for kind in sorted(vs, key=lambda k: k.value):
                if not Q.fits(tok, vs[kind]):
                    refused.append((s, kind.value))
                    del vs[kind]
        views[s] = vs
        for kind, v in vs.items():
            for f, c in TG.candidates_for(v, prof, ref.k_mode, [f for f in fams if f is not T.Family.N2_address]).items():
                cands[(s, f)] = c
    obs = TG.observe(host, ref.day, prof, sigs, ref.k_mode, cands)
    by_sig, cs_by_sig = {}, {}
    for n, o in obs.items():
        by_sig.setdefault(o.sig, {})[n] = o
    for (s, f), c in cands.items():
        cs_by_sig.setdefault(s, {})[f] = c
    for s in sigs:
        if T.Family.N2_address not in fams or prof is None or host.card(s)["type"] != "netflow":
            continue
        bks = TG.owner_buckets(by_sig.get(s, {}), s)
        if not bks:
            continue
        ov = V.own_views(host.card(s), bks, prof, ref.k_mode)
        if tok is not None:
            for b in sorted(ov):
                if not Q.fits(tok, ov[b]):
                    refused.append((s, b))
                    del ov[b]
        own[s] = ov
    qs, q_sigs = {}, {}

    def add(q, s):
        qs.setdefault(q.qkey, q)
        q_sigs.setdefault(q.qkey, set()).add(s)

    for s in sigs:
        mine = by_sig.get(s, {})
        pres = TG.union_present(mine, s)
        p1 = any(T.Family.P1_parent in o.choice for o in mine.values())
        for f, c in sorted(cs_by_sig.get(s, {}).items(), key=lambda kv: kv[0].value):
            if not c.applicable:
                continue
            view = views[s].get(f.spec.view)
            if view is None:
                continue
            if f.spec.kind == "inclusion":
                want = set(c.keys()) if all_buckets else set(pres.get(f, ()))
                for b in c.buckets:
                    if b.key in want:
                        add(Q.build(view, f, bucket=b), s)
            elif f is T.Family.P1_parent and (p1 or all_buckets):
                add(Q.build(view, f, listed=c.listed), s)
        for _b, v in sorted(own.get(s, {}).items()):
            add(Q.build(v, T.Family.N2_address), s)
    return {"views": views, "own": own, "cands": cands, "obs": obs,
            "questions": [qs[k] for k in sorted(qs)], "q_sigs": {k: sorted(v) for k, v in q_sigs.items()},
            "refused": refused}


def frozen_routes(e2=None):
    """routes_fn(host, ref, kplan, head, role): the frozen E2 file's routes (its head must equal
    `head`). Where no valid E2 file exists (CADETS_E3 allc-pre, a head of other caps) it raises:
    fresh routes need a GPU route engine, which this repo does not run."""
    def fn(host, ref, kplan, h, role):
        why = None
        if kplan in retrofit.E2_DIRS and (kplan != "allc-pre" or retrofit.kall_valid(host)):
            try:
                raw = retrofit.e2_day(host.ds, kplan, ref.day, e2, ref=ref, role=role)
                fh = {t: list(raw["head"].get(t, [])) for t in TYPES}
                if fh == {t: list(h.get(t, [])) for t in TYPES}:
                    r = retrofit.e2_routes(raw)
                    return {s: r[s] for t in fh for s in fh[t]}
                why = "the frozen E2 file's head differs from head.q0 (other caps?)"
            except (OSError, ValueError, KeyError) as e:
                why = repr(e)
        else:
            why = f"no frozen E2 directory stands for {kplan} on {host.ds}"
        raise RuntimeError(f"{host.ds} {ref.day} {kplan}: {why}; fresh routes need a GPU route engine")
    return fn


# --------------------------------------------------------------------------- #
# One (host, level, kplan)
# --------------------------------------------------------------------------- #
def score_level(ds, level, kplan, source, fold=None, root=None, *, tok=AUTO, e2=None, routes_fn=None, refs=None,
                shat=True, caps=None, roles=("null", "test"), head_rule="Q0", extra_meta=None, settings=None):
    """Write the score files of one (host, habit level, kplan) through the pipeline (module docstring);
    returns {"ldir", "written", "days", "questions", "refused"}. `ds`: a dataset name (Host over the
    release cards) or a Host; `source`: an AnswerSource (types.DMAnswerSource of the arm, or a test
    double); `fold`: checked against the host (fold_for) and recorded in the meta; `root`: the habit
    root (None = runs/habit); `tok`: the MAX_VIEW_TOKENS gate tokenizer (AUTO = the pinned Qwen
    tokenizer, None = no gate). roles=("test",) re-scores test days with the SAVED null of this plan."""
    host = _host(ds, settings)
    kplan = parse_kplans([kplan])[0]
    if level == "HB0":
        raise ValueError("HB0 has no habit answers: retrofit.run writes it")
    if level not in HABIT_LEVELS:
        raise ValueError(f"level {level!r} is not a habit level {HABIT_LEVELS}")
    if source is None:
        raise ValueError(f"level {level} needs an answer source")
    fold = fold_for(host.ds, fold) if host.ds in FOLDS["all"] else fold
    root = root if root is not None else null2.default_root(settings)
    tok = default_tokenizer(settings) if tok == AUTO else tok
    caps = dict(caps or CAPS)
    routes_fn = routes_fn or frozen_routes(e2)
    clone = level if level in CLONE_ARMS else None
    level, arm = T.split_level(level)                   # a clone arm: (its level, its arm suffix)
    ldir = T.level_dir(level, kplan, arm)
    extra_meta = {**(extra_meta or {}), **({"clone_arm": clone} if clone else {}), **({"fold": fold} if fold else {})}
    refs = null2.plan_refs(host, kplan) if refs is None else refs
    alien = sorted({r.ds for r, _role in refs if r.ds != host.ds})
    if alien:
        raise ValueError(f"{host.ds} {kplan}: null Refs of other hosts {alien}")
    plan_null = {r.day: list(r.ref_days) for r, role in refs if role == "null"}
    refs = [(r, role) for r, role in refs if role in roles]
    refs = [x for x in refs if x[1] == "null"] + [x for x in refs if x[1] == "test"]
    refit = "null" in roles
    if not refit:                           # test days only: the SAVED null fit of this plan, or nothing
        saved = load_fit(host.ds, ldir, plan_null, root)
    t0, ctx, units, nq, refused = time.time(), {}, [], 0, 0
    for ref, role in refs:
        prof = get_profile2(host, list(ref.ref_days)) if ref.k else None
        h = HD.q0(host, ref, caps)
        mem = HD.members(host, ref.day, h)
        routes = routes_fn(host, ref, kplan, h, role)
        nodes = sorted(int(n) for v in mem.values() for n in v.tolist())
        tiers = A.lineage_tiers(host, ref.day, prof, nodes=nodes)
        hd = habit_day(host, ref, prof, HD.head_sigs(h), tok)
        # a source with a length budget (laya) declines a question it cannot encode instead of
        # failing the day: that question is n/a (surprise.node_x)
        refused_q = source.refused(hd["questions"]) if hasattr(source, "refused") else {}
        ans = source.score([q for q in hd["questions"] if q.qkey not in refused_q])
        if refused_q:
            _log(f"{host.ds} {ldir} {ref.day}: {len(refused_q)} questions refused by {getattr(source, 'tag', source)} "
                 f"(n/a), e.g. {next(iter(refused_q.values()))}")
        x, att, att_sig = {}, {}, {}
        for n, o in sorted(hd["obs"].items()):
            x[n] = SU.node_x(o, hd["views"].get(o.sig, {}), hd["own"].get(o.sig, {}), ans, refused=refused_q)
            if o.sig not in att_sig:
                att_sig[o.sig] = A.tier0(host.card(o.sig), prof).identity_attested
            att[n] = att_sig[o.sig]
        if role == "null":
            for n in sorted(x):
                units += SU.null_units(ref.day, hd["obs"][n].ntype, att[n], x[n])
        save_x(host.ds, ldir, ref.day, role, x, root)
        ctx[ref.day] = {"ref": ref, "role": role, "head": h, "routes": routes, "tiers": tiers, "obs": hd["obs"],
                        "x": x, "att": att, "source": getattr(source, "tag", None),
                        "source_spec": source.spec() if hasattr(source, "spec") else None,
                        "refused_q": len(refused_q)}
        nq += len(hd["questions"])
        refused += len(hd["refused"])
        _log(f"{host.ds} {ldir} {ref.day} ({role}): {sum(len(v) for v in mem.values())} head members, "
             f"{len(hd['questions'])} questions (+{time.time() - t0:.0f}s)")
    if refit:
        null = SU.FamilyNull().fit(units)
    else:
        null, emap, info = saved
        spec = sorted({json.dumps([c.get("source"), c.get("source_spec")], sort_keys=True, default=str)
                       for c in ctx.values()})
        if spec and spec != [json.dumps([info["source"], info["source_spec"]], sort_keys=True, default=str)]:
            raise ValueError(f"{host.ds} {ldir}: the saved null was fitted with source {info['source']} "
                             f"{info['source_spec']}, not {spec}: re-score with roles null,test")
    erows = []
    for d, c in ctx.items():
        ex = d if c["role"] == "null" else None
        c["habit"] = {n: SU.habit(c["obs"][n], c["x"][n], null, c["att"][n], exclude_day=ex) for n in c["x"]}
        if c["role"] == "null":
            for n, hh in c["habit"].items():
                r = c["routes"][c["obs"][n].sig]
                erows.append((d, hh.ntype, hh.E, max(float(r["qv"]), float(r["qt"]))))
    if refit:
        emap = null2.EMap().fit(erows)
        nc = next((c for c in ctx.values() if c["role"] == "null"), {})
        null2.save(null, host.ds, f"{ldir}/familynull", root)
        null2.save(emap, host.ds, f"{ldir}/emap", root)
        null2.save({"kind": "fitinfo", "ds": host.ds, "ldir": ldir, "null_refs": plan_null,
                    "fitted_days": sorted(d for d, c in ctx.items() if c["role"] == "null"),
                    "source": nc.get("source"), "source_spec": nc.get("source_spec")}, host.ds, f"{ldir}/fitinfo", root)
    if shat:
        for d, c in ctx.items():
            ex = d if c["role"] == "null" else None
            c["habit"] = {n: null2.with_s_hat(hh, emap, exclude_day=ex) for n, hh in c["habit"].items()}
    written = []
    for d, c in ctx.items():
        rows = SC.rows_for_day(host, c["ref"], c["head"], c["routes"], c["tiers"], c.get("habit"))
        meta = T.base_meta(c["ref"], level, kplan, c["role"], caps, head_rule, arm, attest=ATTEST, a_net=A_NET,
                           shat=bool(shat), source=c.get("source"), source_spec=c.get("source_spec"),
                           **({"refused_questions": c["refused_q"]} if c.get("refused_q") else {}),
                           **extra_meta)
        sd = T.ScoreDay.from_rows(meta, c["head"], c["routes"], rows, SC.head_fallback(host, d, c["head"]),
                                  habit=c.get("habit"))
        written.append(SC.write(sd, root))
    _log(f"{host.ds} {ldir}: {len(written)} files in {time.time() - t0:.0f}s ({nq} questions, {refused} views refused)")
    return {"ldir": ldir, "written": written, "days": list(ctx), "questions": nq, "refused": refused}


def save_x(ds, ldir, day, role, x, root=None):
    """Store one day's per-node x under <root>/nulls/<ds>/<ldir>/x_<day>.json.gz."""
    rec = {"kind": "x", "ds": ds, "day": day, "role": role, "ldir": ldir,
           "x": {str(n): {f.value: v for f, v in sorted(xx.items(), key=lambda kv: kv[0].value)}
                 for n, xx in sorted(x.items())}}
    return null2.save(rec, ds, f"{ldir}/x_{day}", root)


def load_fit(ds, ldir, plan_null, root=None):
    """(FamilyNull, EMap, fit record) saved by the last full score_level of (ds, ldir); refuses when
    it is missing or was fitted on other null days / references than the plan's `plan_null` = {null
    day: reference days} (re-scoring test days alone must reuse the plan's null, never fit an empty
    one)."""
    if not plan_null:
        raise ValueError(f"{ds} {ldir}: the plan has no null day")
    try:
        info = null2.load(ds, f"{ldir}/fitinfo", root)
        null, emap = null2.load(ds, f"{ldir}/familynull", root), null2.load(ds, f"{ldir}/emap", root)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{ds} {ldir}: no saved null fit ({e}); score with roles null,test first") from None
    if info.get("null_refs") != {d: list(r) for d, r in plan_null.items()} or \
            set(null.days) - set(plan_null) or set(emap.days) - set(plan_null):
        raise ValueError(f"{ds} {ldir}: the saved null fit (null refs {info.get('null_refs')}) is not this plan's "
                         f"{plan_null}: re-score with roles null,test")
    return null, emap, info


# --------------------------------------------------------------------------- #
# Score directories: habit ScoreDay files and the frozen routed E2 files
# --------------------------------------------------------------------------- #
def parse_ldir(ldir):
    """'<level>[.<arm>]_<kplan>' -> (level, arm or None, kplan); ValueError for anything else."""
    head, sep, kplan = str(ldir).partition("_")
    level, _dot, arm = head.partition(".")
    if not sep or level not in LEVELS or kplan not in KPLANS or (arm and not re.fullmatch(r"[A-Za-z0-9-]+", arm)):
        raise ValueError(f"{ldir!r} is not a habit score directory <level>[.<arm>]_<kplan>")
    return level, arm or None, kplan


def scores_root(ds, root=None):
    return os.path.join(null2.default_root() if root is None else os.fspath(root), "scores", ds)


def legacy_valid(host, kplan):
    """(True, "") when the frozen E2_k* directory stands for E2 at `kplan` on this host; E2_kall is
    allc-pre only where the pre-deployment window is every clean day (not CADETS_E3)."""
    if kplan not in LEGACY_K:
        return False, f"no frozen E2 directory for k plan {kplan}"
    if kplan == "allc-pre" and list(T.predeploy(host)) != list(host.clean):
        return False, "E2_kall is not allc-pre here (clean days after the first test day): fresh routes needed"
    return True, ""


class ScoreDir:
    """One score directory of one host: day files by role and a reader for each.

    source "habitus": <root>/scores/<ds>/<ldir>/<day>.json.gz (types.ScoreDay); role = test for
    host.test days, null for every other (pre-deployment) day. source "legacy": E2 only, the frozen
    routed E2_k0 / E2_k2 / E2_kall files when no habit E2 directory exists; null days = the LODO plan's
    days inside the pre-deployment window. source None: nothing to read (`why` says why)."""

    def __init__(self, host, ldir, root=None, legacy_root=None):
        from habitus.decide.null import lodo_plan
        self.host, self.ldir = host, ldir
        self.level, self.arm, self.kplan = parse_ldir(ldir)
        self.dir = os.path.join(scores_root(host.ds, root), ldir)
        self.files, self.source, self.why = {}, None, ""
        if os.path.isdir(self.dir):
            self.source = "habitus"
            for p in sorted(glob.glob(os.path.join(self.dir, "*.json.gz"))):
                self.files[os.path.basename(p)[:-len(".json.gz")]] = p
        elif self.level == "E2" and self.arm is None:
            ok, why = legacy_valid(host, self.kplan)
            if ok:
                lroot = os.fspath(legacy_root) if legacy_root is not None else retrofit.e2_root()
                k = LEGACY_K[self.kplan]
                self.dir = os.path.join(lroot, host.ds, f"E2_k{'all' if k is None else k}")
                pre = set(T.predeploy(host))
                try:
                    nulls = [c for c, _r in lodo_plan(host, k)]
                except ValueError:
                    nulls = []
                for d in list(host.test) + [c for c in nulls if c in pre]:
                    p = os.path.join(self.dir, f"{d}.json.gz")
                    if os.path.exists(p):
                        self.files[d] = p
                self.source = "legacy" if self.files else None
                self.why = "" if self.files else f"no frozen E2 files under {self.dir}"
            else:
                self.why = why
        else:
            self.why = f"no directory {self.dir}"
        self.test = [d for d in host.test if d in self.files]
        self.null = [d for d in sorted(self.files) if d not in set(host.test)]
        self.missing_test = [d for d in host.test if d not in self.files]
        self.stray = [d for d in self.null if d not in set(host.clean)]

    @property
    def exists(self):
        return self.source is not None

    def read(self, day):
        """The day's file: a types.ScoreDay (habitus) or the raw routed dict (legacy)."""
        p = self.files[day]
        if self.source == "legacy":
            from habitus.io import read_json_gz
            return read_json_gz(p)
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return T.ScoreDay.from_json(json.load(f))

    def values(self, day, column="s", rec=None):
        """(node ids of the day, value per node): row value for head nodes, the rarity fallback otherwise."""
        rec = self.read(day) if rec is None else rec
        if self.source == "legacy":
            from habitus.decide.scorefile import day_values
            return day_values(self.host, day, rec["scores"], LEGACY_FIELD[column])
        return rec.node_values(self.host, column)

    def head_ids(self, day, rec=None):
        rec = self.read(day) if rec is None else rec
        if self.source == "legacy":
            recs = rec["scores"]
            return np.sort(np.asarray([n for n, s in self.host.nodes(day).items() if s in recs], dtype=np.int64))
        return np.sort(np.asarray(rec.nodes["id"], dtype=np.int64))


def _max_into(vec, host, ids, v):
    pos = host.upos(np.asarray(ids, dtype=np.int64))
    ok = pos >= 0
    np.maximum.at(vec, pos[ok], np.asarray(v, dtype=np.float64)[ok])


def habit_vectors(host, ldir, root=None):
    """({"s": composite node vector, "habit": (column, vector) or None, "meta": first test-day meta,
    "source"}, None) of one score directory's TEST days, or (None, why). Composite: head value, the
    rarity fallback for the rest; habit-only: the first HABIT_COLUMNS column with a finite value, max
    over test days of the head rows that carry it, -inf for every other node."""
    sd = ScoreDir(host, ldir, root)
    if not sd.exists:
        return None, sd.why or "no score directory"
    if sd.missing_test:
        return None, f"test days without a score file {sd.missing_test}"
    s = np.full(len(host.universe), -np.inf)
    hv = {c: np.full(len(host.universe), -np.inf) for c in HABIT_COLUMNS}
    meta = None
    for d in host.test:
        rec = sd.read(d)
        ids, v = sd.values(d, "s", rec)
        _max_into(s, host, ids, v)
        if sd.source != "habitus":
            continue
        meta = meta or dict(rec.meta)
        rid = np.asarray(rec.nodes["id"], dtype=np.int64)
        for c in HABIT_COLUMNS:
            col = rec.nodes.get(c)
            if col is None:
                continue
            xv = np.asarray([np.nan if t is None else t for t in col], dtype=np.float64)
            ok = np.isfinite(xv)
            if ok.any():
                _max_into(hv[c], host, rid[ok], xv[ok])
    if not np.isfinite(s).all():
        return None, f"{int((~np.isfinite(s)).sum())} universe nodes got no composite score"
    habit = next(((c, hv[c]) for c in HABIT_COLUMNS if np.isfinite(hv[c]).any()), None)
    return {"s": s, "habit": habit, "meta": meta, "source": sd.source}, None


__all__ = ["parse_kplans", "fold_for", "source_for", "default_tokenizer", "habit_day", "frozen_routes",
           "score_level", "save_x", "load_fit", "parse_ldir", "scores_root", "legacy_valid", "ScoreDir",
           "habit_vectors", "HABIT_COLUMNS"]
