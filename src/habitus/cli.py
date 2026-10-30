"""`habitus [--config PATH] [-v] <group> <command> [options]`: the single command-line entry point.

Groups: data, state, questions, dm, decide, train, eval, reproduce, gates (`habitus <group> -h`).

Label guard: for every group except `eval` (the only label reader), `reproduce` and `gates`, the
guard (habitus.guard.install_guard) is installed FIRST, before any other habitus module is imported:
inside a guarded command no ground-truth file can be opened and no habitus.eval module imported.
`reproduce` and `gates` run every state / questions / decide / dm step as a GUARDED `habitus ...`
child process and only compare outputs in the parent. A child started by them also gets
  HABITUS_WRITE_ROOTS   the write audit: writes / mkdirs outside these roots raise
  HABITUS_OPEN_LOG      a file that receives every path the child opened (the G5 runtime audit)

Devices: the default device is the CPU and --cache-only is the default: a cache-only command never
loads a model and runs with CUDA hidden. dm.gpu.check_device (forbidden_gpus from the settings, PCI
bus order) runs only when a model will be forwarded: `dm smoke`, `train jev|laya`, and `decide`
with --gpu.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
import time

EXEMPT = ("eval", "reproduce", "gates")
WRITE_ROOTS_ENV = "HABITUS_WRITE_ROOTS"
OPEN_LOG_ENV = "HABITUS_OPEN_LOG"
MAIN_DM = "jev-qwen3.5-4b"
KPLANS = ("k0", "k2", "allc-pre")
KPLAN_ALIASES = {"0": "k0", "k0": "k0", "2": "k2", "k2": "k2", "allc-pre": "allc-pre"}
# smoke thresholds (G7). The subset smoke is a liveness check: bf16 re-forwards with another batch
# makeup differ by p99 |d letter prob| ~0.06 (measured), so 0.08 is reachable. The full-day replay
# reproduces the original batch makeup of the job (one decide call per score_job call, original order)
# and must match the cache to p99 <= 1e-3 (bit-exact expected on the GPU architecture of the cache).
JEV_SMOKE_P99 = 0.08
JEV_FULL_DAY_P99 = 1e-3
DB_PASSWORD_ENV = "HABITUS_DB_PASSWORD"
E1_LEVELS = ("L0", "L1", "L2", "L3", "L4")
FIELDS = ("composite", "base", "qv", "qt", "1-qa", "acq_all")
ANSWER_FILE = {"release": "{answers}/{tag}/{ds}.jsonl.gz",
               "unpruned": "{pidsmaker_artifacts}/reflex/kcs1/answers/{tag}/{ds}.jsonl"}


def _log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [habitus] {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# Argument helpers
# --------------------------------------------------------------------------- #
def ds_list(spec):
    """'A,B' -> ['A', 'B'], each checked against the 5-host allow-list (raises for any other)."""
    from habitus import hosts
    out = [x.strip() for x in str(spec).split(",") if x.strip()]
    if not out:
        raise SystemExit(f"--ds {spec!r}: no dataset")
    for ds in out:
        hosts.assert_allowed(ds)
    return out


def parse_ks(spec):
    """'all,2,0' -> [None, 2, 0]: k in {0, 2, all} only (KC-S1 labels)."""
    out = []
    for x in str(spec).split(","):
        x = x.strip()
        if not x:
            continue
        if x not in ("0", "2", "all"):
            raise SystemExit(f"--ks/--k {spec!r}: k in {{0, 2, all}} only")
        k = None if x == "all" else int(x)
        if k not in out:
            out.append(k)
    if not out:
        raise SystemExit(f"--ks {spec!r}: no k")
    return out


def parse_kplans(spec, ds=None):
    out = []
    for x in str(spec).split(","):
        x = x.strip()
        if not x:
            continue
        if x not in KPLAN_ALIASES:
            raise SystemExit(f"--kplans {spec!r}: {x!r} is not one of 0/k0, 2/k2, allc-pre")
        if KPLAN_ALIASES[x] not in out:
            out.append(KPLAN_ALIASES[x])
    if not out:
        raise SystemExit(f"--kplans {spec!r}: no k plan")
    return out


def parse_levels(spec, allowed=("E2",) + E1_LEVELS):
    out = [x.strip() for x in str(spec).split(",") if x.strip()]
    bad = [x for x in out if x not in allowed]
    if bad or not out:
        raise SystemExit(f"--level {spec!r}: unknown {bad}; known {list(allowed)}")
    return out


def klabel(k):
    return "all" if k is None else str(k)


# The registries live in habitus.dm (ARMS) and habitus.hosts (FOLDS); they are imported lazily, after
# the guard is installed (module docstring).
def habit_arms():
    """{habit arm: configs/dm name} (habitus.dm.ARMS, habit questions)."""
    from habitus.dm import ARMS
    return {a.name: a.config for a in ARMS.values() if a.questions == "habit"}


def folds():
    """The training folds (habitus.hosts.FOLDS)."""
    from habitus.hosts import FOLDS
    return tuple(FOLDS)


def habit_arm(name):
    """A habit arm name, given the arm name or its configs/dm config name (one option for both)."""
    arms = habit_arms()
    if name in arms:
        return name
    by_cfg = {v: k for k, v in arms.items()}
    if name in by_cfg:
        return by_cfg[name]
    raise SystemExit(f"{name!r}: not a habit arm ({', '.join(arms)}) or an arm's config ({', '.join(by_cfg)})")


def dm_config(name):
    """A configs/dm name, given the config name or a registered arm name (habitus.dm.ARMS)."""
    from habitus.dm import ARMS
    return ARMS[name].config if name in ARMS else name


def _write_roots():
    """The write-audit roots of $HABITUS_WRITE_ROOTS (None: no audit), plus the null device: opening it
    for writing writes nothing (torch's import of dill does it)."""
    v = os.environ.get(WRITE_ROOTS_ENV)
    if not v:
        return None
    return [os.path.abspath(x) for x in v.split(os.pathsep) if x] + [os.devnull]


def _install_open_log():
    """Record every path this process opens (audit event 'open') into $HABITUS_OPEN_LOG at exit."""
    path = os.environ.get(OPEN_LOG_ENV)
    if not path:
        return
    path = os.path.abspath(path)
    seen, busy = set(), [False]

    def hook(event, args):
        if event != "open" or busy[0] or not args:
            return
        f = args[0]
        if isinstance(f, (str, bytes, os.PathLike)):
            try:
                seen.add(os.path.abspath(os.fsdecode(f)))
            except (TypeError, ValueError):
                pass

    def dump():
        busy[0] = True
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(sorted(seen)) + "\n")
        finally:
            busy[0] = False

    sys.addaudithook(hook)
    atexit.register(dump)


def _settings():
    from habitus import config
    return config.settings()


def _apply_env(st, a):
    """Thread caps and offline hub mode (process env, before numpy / torch / transformers import)."""
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, str(st.threads))
    if st.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if st.paths.models and forwards_model(a):       # model weights resolve through the hub cache layout
        os.environ.setdefault("HF_HOME", st.paths.models)


def forwards_model(a):
    """True when the command will forward a model (needs a checked GPU device)."""
    if a.group == "dm" and a.cmd == "smoke":
        return True
    if a.group == "train" and a.cmd in ("jev", "laya"):
        return True
    if a.group == "decide" and a.cmd in ("routed", "habit"):
        return getattr(a, "gpu", None) is not None
    if a.group == "gates":
        return getattr(a, "gpu", None) is not None
    return False


def device_of(a):
    gpu = getattr(a, "gpu", None)
    return "cpu" if gpu is None else f"cuda:{int(gpu)}"


def check_gpu(a, st):
    """dm.gpu.check_device for a forwarding command; SystemExit on a forbidden / unresolvable GPU."""
    dev = device_of(a)
    if dev == "cpu":
        return dev
    from habitus.dm.gpu import check_device
    try:
        check_device(dev, st)
    except Exception as e:                  # ForbiddenDevice / ValueError
        raise SystemExit(f"refused: {e}") from None
    return dev


def _runs(st, *parts):
    return os.path.join(st.paths.runs, *parts)


def _json_out(obj, out=None):
    text = json.dumps(obj, indent=1, sort_keys=True, default=str)
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)


# --------------------------------------------------------------------------- #
# Shared pieces: host, jobs, key collection, answer files
# --------------------------------------------------------------------------- #
def make_host(ds, source="cards"):
    from habitus.data.host import Host
    return Host(ds, source=source)


def job_for(host, level, k, day, role):
    """The ScoreJob of one (level, k, day, role): a test day's reference is ref_for(host, k), a
    clean day's is its lodo_plan(host, k) entry."""
    from habitus.decide.inventory import ScoreJob
    from habitus.decide.null import lodo_plan
    from habitus.decide.routed import ref_for
    if role == "test":
        if day not in host.test:
            raise SystemExit(f"{host.ds} {day} is not a test day ({list(host.test)})")
        ref = tuple(ref_for(host, k))
    else:
        plan = dict((c, tuple(r)) for c, r in lodo_plan(host, k))
        if day not in plan:
            raise SystemExit(f"{host.ds} {day} is not a null day at k={klabel(k)} ({sorted(plan)})")
        ref = plan[day]
    return ScoreJob(host.ds, level, k, day, ref, role)


class KeyCollector:
    """A DecisionModel stand-in for decide.routed.score_job that answers nothing: it records the
    answer-cache key of every presented form (JevDecisionModel.keys, tokenizer only) and returns
    uniform placeholder Decisions, so the exact question set of a job is enumerated without a
    model and without a cache. Its records are discarded (never written)."""

    family, layout = "jev", "letter"

    def __init__(self, dm):
        self.dm = dm
        self.tag = getattr(dm, "tag", None)
        self.keys = set()               # every form key seen
        self.per_question = []          # (qkey, family, qtype, group, form names, keys) of the last job

    def spec(self):
        return self.dm.spec()

    def refused(self, qs):
        return self.dm.refused(qs)

    def decide(self, qs):
        from habitus.dm.base import Decision
        ks = self.dm.keys(qs)
        out = {}
        for q in qs:
            fk = tuple(ks[q.qkey])
            self.keys.update(fk)
            self.per_question.append((q.qkey, q.family, q.qtype, q.group, tuple(f.name for f in q.forms), fk))
            probs = {f.name: (tuple(1.0 / len(f.options) for _ in f.options),) for f in q.forms}
            out[q.qkey] = Decision(qkey=q.qkey, layout="letter", forms=tuple(f.name for f in q.forms),
                                   letter_probs=probs, canon_probs=probs, cov={f.name: 1.0 for f in q.forms},
                                   cov_min=1.0)
        return out

    def close(self):
        pass


def answer_path(st, which, tag, ds):
    if which == "unpruned":
        from habitus.config import require
        require("pidsmaker_artifacts", st)
    p = ANSWER_FILE[which].format(answers=st.paths.answers, pidsmaker_artifacts=st.paths.pidsmaker_artifacts,
                                  tag=tag, ds=ds)
    if which == "release" and not os.path.exists(p) and os.path.exists(p[:-len(".gz")]):
        p = p[:-len(".gz")]                         # an uncompressed base (dm.cache accepts both)
    return p


def answer_keys(path):
    """The set of answer keys ("k") of an answer-cache jsonl(.gz)."""
    import gzip
    import re
    rx = re.compile(r'"k"\s*:\s*"([0-9a-f]{40})"')
    out = set()
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            m = rx.search(line[:80])
            if m:
                out.add(m.group(1))
            elif line.strip():
                try:
                    out.add(json.loads(line)["k"])
                except (ValueError, KeyError):
                    continue
    return out


def get_dm(name, *, ds=None, device="cpu", cache_only=True, fold=None, st=None):
    """habitus.dm.get_dm of a configs/dm name (a fold fills a trained arm's run directory)."""
    from habitus.dm import get_dm as _get
    return _get(name, ds=ds, device=device, cache_only=cache_only, fold=fold, settings=st)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def _db_password(a):
    """--db-password-file (first line), else $HABITUS_DB_PASSWORD, else --db-password (visible in `ps`:
    avoid it on a shared server), else None (the upstream default)."""
    f = getattr(a, "db_password_file", None)
    if f:
        with open(f, encoding="utf-8") as fh:
            return fh.readline().rstrip("\n")
    return os.environ.get(DB_PASSWORD_ENV) or getattr(a, "db_password", None)


def _db(a, ds):
    from habitus.data.upstream import db_params
    return db_params(ds, host=a.db_host, user=a.db_user, password=_db_password(a), port=a.db_port)


def _graphs_dir(a, ds, st):
    """--pidsmaker-artifacts DIR -> the dataset's graph directory under DIR (None: the default search of
    data.upstream.graphs_dir, runs/pidsmaker then paths.pidsmaker_artifacts)."""
    art = getattr(a, "pidsmaker_artifacts", None)
    if not art:
        return None
    from habitus.data.upstream import graphs_dir
    return graphs_dir(ds, artifacts=os.path.abspath(art), settings=st)


def cmd_data_graphs(a, st):
    """Upstream PIDSMaker construction + transformation as a subprocess under paths.pidsmaker_python."""
    from habitus.data.upstream import graphs_subprocess
    ds = ds_list(a.ds)[0]
    python = a.python or st.paths.pidsmaker_python
    if not python:
        raise SystemExit("--python or paths.pidsmaker_python (configs/local.yaml) is needed: the upstream "
                         "PIDSMaker graph build runs as a subprocess under that interpreter")
    r = graphs_subprocess(ds, python, _db(a, ds), artifacts=a.pidsmaker_artifacts, settings=st, check=False)
    return int(getattr(r, "returncode", r) or 0)


def cmd_data_resolver(a, st):
    from habitus.data import resolver
    ds = ds_list(a.ds)[0]
    out = resolver.build(ds, db=_db(a, ds), settings=st)
    _log(f"resolver {ds}: {out}")


def cmd_data_cards(a, st):
    from habitus.data import cards
    ds = ds_list(a.ds)[0]
    out = a.out or _runs(st, "cards")
    days = [d for d in a.days.split(",") if d] if a.days else None
    built = cards.build(ds, out, days=days, graphs_dir=_graphs_dir(a, ds, st), settings=st)
    _log(f"cards {ds}: {len(built or [])} day files built under {out}")


def cmd_data_manifest(a, st):
    from habitus.data import days
    ds = ds_list(a.ds)[0]
    m = days.manifest(ds, _graphs_dir(a, ds, st), settings=st)
    _json_out(m)
    if not m.get("ok"):
        raise SystemExit(f"{ds}: recomputed usable / card days differ from configs/datasets.yaml")


def cmd_data_index(a, st):
    from habitus.data import nodeindex
    ds = ds_list(a.ds)[0]
    out = nodeindex.build(ds, a.cards or st.paths.cards, a.out or _runs(st, "nodeindex"))
    _log(f"nodeindex {ds}: {len(out)} day files")


def cmd_data_verify(a, st):
    """JSON sha256 of every card / link file of DS under --cards vs the release MANIFEST.json."""
    from habitus.io import json_sha256
    ds = ds_list(a.ds)[0]
    man_p = os.path.join(st.paths.data, "MANIFEST.json")
    with open(man_p, encoding="utf-8") as fh:
        man = json.load(fh)
    cards_dir = a.cards or _runs(st, "cards")
    rows, bad = [], 0
    for rel, rec in sorted(man["files"].items()):
        if not rel.startswith(f"cards/{ds}/") or "json_sha256" not in rec:
            continue
        p = os.path.join(cards_dir, rel[len("cards/"):])
        got = json_sha256(p) if os.path.exists(p) else None
        ok = got == rec["json_sha256"]
        bad += not ok
        rows.append({"file": rel, "ok": ok, "rebuilt": got is not None})
    _json_out({"ds": ds, "files": len(rows), "mismatch": bad, "rows": rows})
    if bad or not rows:
        raise SystemExit(f"{ds}: {bad} of {len(rows)} card / link files differ from the release")


# --------------------------------------------------------------------------- #
# state / questions
# --------------------------------------------------------------------------- #
def _states(host, job):
    """(head, {sig: {qid: state}} for E2 or {sig: state} for E1) of one job."""
    from habitus.state.select import CAPS, flat, head
    ref = list(job.ref)
    h = head(host, job.day, ref or None, CAPS)
    sigs = flat(h)
    if job.level == "E2":
        from habitus.state.builder import StateBuilderE2
        return h, StateBuilderE2(host, ref, job.k).build(job.day, sigs)
    from habitus.state.ladder import StateBuilder
    return h, StateBuilder(host, ref, job.level, job.k).build(job.day, sigs)


def cmd_state_build(a, st):
    """One job's states -> jsonl: a header {"meta": {..., "n_states", "digest"}, "head": {...}}, then
    one line per (signature, question) {"sig", "qid", "sha1"} (E1: qid null) with sha1 of
    serialize(state). digest = sha1 over "<sig>|<qid or 'state'>|<text>\\n" in head order (the
    tests/golden/state digest)."""
    import hashlib

    from habitus.state.builder import serialize
    from habitus.state.select import flat
    ds = ds_list(a.ds)[0]
    level = parse_levels(a.level)[0]
    k = parse_ks(a.k)[0]
    host = make_host(ds)
    job = job_for(host, level, k, a.day, a.role)
    h, states = _states(host, job)
    sigs = flat(h)
    out = a.out or _runs(st, "states", f"{ds}_{level}_k{klabel(k)}_{a.day}_{a.role}.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    dig, lines = hashlib.sha1(), []
    for s in sigs:
        per = states[s].items() if level == "E2" else [(None, states[s])]
        for q, state in per:
            text = serialize(state)
            dig.update(f"{s}|{q or 'state'}|{text}\n".encode("utf-8"))
            rec = {"sig": s, "qid": q, "sha1": hashlib.sha1(text.encode("utf-8")).hexdigest()}
            if a.text:
                rec["text"] = text
            lines.append(json.dumps(rec))
    meta = {"ds": ds, "level": level, "k": klabel(k), "day": a.day, "role": a.role, "ref": list(job.ref),
            "n_states": len(sigs), "digest": dig.hexdigest()}
    with open(out + ".tmp", "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": meta, "head": h}) + "\n")
        fh.write("".join(x + "\n" for x in lines))
    os.replace(out + ".tmp", out)
    _log(f"state build {ds} {level} k={klabel(k)} {a.day} ({a.role}): {len(sigs)} signatures, {len(lines)} states "
         f"-> {out}")


def _sanitize_status():
    try:
        from habitus.state import sanitize
        return {"status": sanitize.pattern_status(), "pattern": sanitize.pattern().pattern}
    except Exception as e:                  # reported, judged by the caller (G2)
        return {"status": f"error: {type(e).__name__}: {e}", "pattern": None}


def cmd_questions_render(a, st):
    """Typed questions of one job: forms, cache keys and hit / miss counts against an answer file."""
    from habitus.decide.routed import score_job
    ds = ds_list(a.ds)[0]
    level = parse_levels(a.level)[0]
    k = parse_ks(a.k)[0]
    host = make_host(ds)
    job = job_for(host, level, k, a.day, a.role)
    dm = get_dm(dm_config(a.dm), ds=ds, device="cpu", cache_only=True)
    col = KeyCollector(dm)
    score_job(host, col, job)
    have = None
    if a.answers != "none":
        have = answer_keys(answer_path(st, a.answers, dm.tag, ds))
    out = a.out or _runs(st, "questions", f"{ds}_{level}_k{klabel(k)}_{a.day}_{a.role}.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    hits = misses = 0
    with open(out + ".tmp", "w", encoding="utf-8") as fh:
        for qkey, fam, qtype, group, forms, keys in col.per_question:
            hit = None if have is None else [kk in have for kk in keys]
            if hit is not None:
                hits += sum(hit)
                misses += len(hit) - sum(hit)
            fh.write(json.dumps({"qkey": qkey, "family": fam, "qtype": qtype, "group": group, "forms": list(forms),
                                 "keys": list(keys), "hit": hit}) + "\n")
    os.replace(out + ".tmp", out)
    summary = {"ds": ds, "level": level, "k": klabel(k), "day": a.day, "role": a.role, "ref": list(job.ref),
               "dm": a.dm, "tag": dm.tag, "answers": a.answers, "n_questions": len(col.per_question),
               "n_keys": len(col.keys), "hits": hits if have is not None else None,
               "misses": misses if have is not None else None, "sanitize": _sanitize_status(), "out": out}
    _json_out(summary, out[:-len(".jsonl")] + ".summary.json" if out.endswith(".jsonl") else out + ".summary.json")
    dm.close()
    if have is not None and misses:
        raise SystemExit(f"{misses} of {hits + misses} form keys are not in the {a.answers} answers")


def collect_keys(ds, jobs, dm, host=None, log_every=1):
    """The set of answer keys of every job (states rebuilt, no model, no cache)."""
    from habitus.decide.routed import score_job
    host = host or make_host(ds)
    if any(j.role == "null" for j in jobs):
        keep = getattr(host, "keep_clean_days", None)
        if callable(keep):
            keep()
    col, builders = KeyCollector(dm), {}
    t0 = time.time()
    for i, job in enumerate(jobs, 1):
        col.per_question = []
        score_job(host, col, job, builders=builders)
        if log_every and i % log_every == 0:
            _log(f"keys {ds} {i}/{len(jobs)} {job.level}_k{klabel(job.k)} {job.day} ({job.role}): "
                 f"{len(col.per_question)} questions, {len(col.keys)} keys so far, +{time.time() - t0:.0f}s")
    return col.keys


def _jobs_for(a, host):
    from habitus.decide import inventory as INV
    if a.from_inventory:
        return INV.inventory(host.ds, a.inventory_root or None)
    return INV.jobs(host, levels=tuple(parse_levels(a.level)), ks=tuple(parse_ks(a.ks)), nulls=a.nulls)


def cmd_questions_keys(a, st):
    """Every form key of the jobs (sorted, one per line) -> --out; with --answers, hit / miss counts."""
    ds = ds_list(a.ds)[0]
    host = make_host(ds)
    jobs = _jobs_for(a, host)
    dm = get_dm(dm_config(a.dm), ds=ds, device="cpu", cache_only=True)
    keys = collect_keys(ds, jobs, dm, host)
    out = a.out or _runs(st, "keys", f"{ds}.{dm.tag}.keys.txt")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out + ".tmp", "w", encoding="utf-8") as fh:
        fh.write("".join(k + "\n" for k in sorted(keys)))
    os.replace(out + ".tmp", out)
    summary = {"ds": ds, "tag": dm.tag, "jobs": len(jobs), "keys": len(keys), "out": out}
    if a.answers != "none":
        have = answer_keys(answer_path(st, a.answers, dm.tag, ds))
        summary.update(answers=a.answers, misses=len(keys - have), hits=len(keys & have))
    dm.close()
    _json_out(summary, out + ".summary.json")
    if summary.get("misses"):
        raise SystemExit(f"{summary['misses']} keys are not in the {a.answers} answers")


# --------------------------------------------------------------------------- #
# dm
# --------------------------------------------------------------------------- #
def cmd_dm_info(a, st):
    """Resolved config, tag and spec of a decision model (a config or arm name). A trained arm names
    fold run directories: give --fold, or --ds (its fold is dm.fold_for(ds))."""
    import yaml
    name = dm_config(a.dm)
    p = os.path.join(st.paths.home, "configs", "dm", f"{name}.yaml")
    with open(p, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    ds = ds_list(a.ds)[0] if a.ds else None
    if a.fold is not None and a.fold not in folds():
        raise SystemExit(f"--fold {a.fold!r}: one of {folds()}")
    info = {"dm": a.dm, "config_name": name, "config": cfg, "fold": a.fold, "ds": ds}
    try:
        dm = get_dm(name, ds=ds, fold=a.fold, device="cpu", cache_only=True)
        info.update(tag=getattr(dm, "tag", None), spec=dm.spec())
        dm.close()
    except Exception as e:                  # e.g. a trained arm whose run dir is not there yet
        info["error"] = f"{type(e).__name__}: {e}"
    from habitus.hosts import DATASETS
    tag = info.get("tag") or cfg.get("cache_tag")
    if tag:
        info["answer_cache"] = {ds: os.path.exists(answer_path(st, "release", tag, ds)) for ds in DATASETS}
    _json_out(info)


def smoke_stats(pairs, family):
    """{"finite", "max_sum_error", "p99_abs_delta", "max_abs_delta", "frac_exact", "argmax_agree", "n_forms"}
    of [(fresh Decision, cached Decision or None)]: every backend's fresh letter probabilities must be
    finite and sum to 1; for jev (letter logits of the same prompts) the first member's probabilities are
    compared with the cached answer per form."""
    import numpy as np
    d, finite, sums, agree = [], True, [], []
    for fresh, cached in pairs:
        for f in fresh.forms:
            for member in fresh.letter_probs[f]:
                p = np.asarray(member, dtype=float)
                finite &= bool(np.isfinite(p).all())
                sums.append(float(p.sum()))
            if family == "jev" and cached is not None:
                p = np.asarray(fresh.letter_probs[f][0], dtype=float)
                c = np.asarray(cached.letter_probs[f][0], dtype=float)
                d.append(float(np.abs(p - c).max()))
                agree.append(int(np.argmax(p)) == int(np.argmax(c)))
    return {"finite": finite, "n_forms": len(sums),
            "max_sum_error": max(abs(x - 1.0) for x in sums) if sums else None,
            "p99_abs_delta": float(np.quantile(d, 0.99)) if d else None,
            "max_abs_delta": max(d) if d else None,
            "frac_exact": (sum(x == 0.0 for x in d) / len(d)) if d else None,
            "argmax_agree": (sum(agree) / len(agree)) if agree else None}


def smoke_ok(res, p99_max):
    return bool(res["finite"] and res["max_sum_error"] is not None and res["max_sum_error"] < 1e-6
                and (res["p99_abs_delta"] is None or res["p99_abs_delta"] <= p99_max))


def cmd_dm_smoke(a, st):
    """G7 (optional, GPU): re-forward routed forms of one host-day with the answer cache bypassed (a
    scratch cache tag, overlay under runs/) and compare with the shipped answers. Every backend: finite,
    sums to 1.

    subset (default)  the first --n questions in ONE decide call (another batch makeup than the cached
                      run): a liveness check, jev p99 |d letter prob| <= JEV_SMOKE_P99 (0.08)
    --full-day        the whole job through decide.routed.score_job, each of its decide calls forwarded
                      as is (the original batch makeup and order; jev only): p99 <= JEV_FULL_DAY_P99
                      (1e-3). The default job, CLEARSCOPE_E3 E2 k=all 2018-04-11, was the first E2 file
                      written for that host, forwarded in one uncached call (about 8.7k prompts)."""
    import yaml

    from habitus.decide.routed import score_job
    from habitus.dm import get_dm as _get
    dev = device_of(a)
    ds = ds_list(a.ds)[0]
    name = dm_config(a.dm)
    host = make_host(ds)
    job = job_for(host, "E2", parse_ks(a.k)[0], a.day, "test")
    base = get_dm(MAIN_DM, ds=ds, device="cpu", cache_only=True)
    with open(os.path.join(st.paths.home, "configs", "dm", f"{name}.yaml"), encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    family = cfg.get("dm")
    if a.full_day and (family != "jev" or name != MAIN_DM):
        base.close()
        raise SystemExit(f"--full-day replays the cached main-config job: --dm {MAIN_DM} only")
    cfg["cache_tag"] = f"smoke-{time.strftime('%Y%m%d%H%M%S')}"      # a fresh namespace: every form is forwarded
    if a.full_day:
        dm = _get(cfg, ds=ds, device=dev, cache_only=False)
        pairs, calls = [], []

        class Tee:
            """score_job's decision model: every decide call goes to the fresh model unchanged (same
            questions, same order, same call boundaries) and is paired with the cached answers."""
            family, layout = getattr(dm, "family", "jev"), getattr(dm, "layout", "letter")
            tag = getattr(dm, "tag", None)

            def spec(self):
                return dm.spec()

            def refused(self, qs):
                return dm.refused(qs)

            def keys(self, qs):
                return dm.keys(qs)

            def decide(self, qs):
                fresh = dm.decide(qs)
                cached = base.decide(qs)
                calls.append(len(qs))
                pairs.extend((fresh[q.qkey], cached[q.qkey]) for q in qs)
                return fresh

            def close(self):
                pass
        t0 = time.time()
        score_job(host, Tee(), job)
        dt = time.time() - t0
        dm.close()
        base.close()
        res = {"dm": a.dm, "mode": "full-day", "device": dev, "job": f"{ds} E2 k={klabel(job.k)} {a.day} test",
               "n": len(pairs), "decide_calls": calls, "n_refused": 0, "seconds": round(dt, 1),
               **smoke_stats(pairs, family)}
        res["p99_max"] = JEV_FULL_DAY_P99
        res["ok"] = smoke_ok(res, JEV_FULL_DAY_P99)
        _json_out(res, a.out)
        if not res["ok"]:
            raise SystemExit("dm smoke --full-day failed")
        return
    picked = []

    class Pick(KeyCollector):
        def decide(self, qs):
            picked.extend(q for q in qs[: max(0, a.n - len(picked))])
            return super().decide(qs)

    score_job(host, Pick(base), job)
    cached = base.decide(picked)
    base.close()
    dm = _get(cfg, ds=ds, device=dev, cache_only=False)
    refused = dm.refused(picked) or {}              # laya: over-budget rows are refused (never cut), not answered
    asked = [q for q in picked if q.qkey not in refused]
    t0 = time.time()
    fresh = dm.decide(asked)
    dt = time.time() - t0
    dm.close()
    res = {"dm": a.dm, "mode": "subset", "device": dev, "n": len(asked), "n_refused": len(refused),
           "seconds": round(dt, 1), **smoke_stats([(fresh[q.qkey], cached[q.qkey]) for q in asked], family)}
    res["p99_max"] = JEV_SMOKE_P99 if family == "jev" else None
    res["ok"] = smoke_ok(res, JEV_SMOKE_P99)
    _json_out(res, a.out)
    if not res["ok"]:
        raise SystemExit("dm smoke failed")


# --------------------------------------------------------------------------- #
# decide
# --------------------------------------------------------------------------- #
def cmd_decide_routed(a, st):
    from habitus.decide import inventory as INV
    from habitus.decide import routed as R
    dev = device_of(a)
    cache_only = dev == "cpu"
    listing = {}
    for ds in ds_list(a.ds):
        if a.list_jobs:                             # metas and HostSpec day lists only, no card read
            host = INV.spec_host(ds)
            js = _jobs_for(a, host)
            listing[ds] = {"jobs": INV.rows(js)}
            if a.from_inventory:
                rule = INV.jobs(host, levels=INV.LEVELS, ks=INV.KS, nulls=True)
                listing[ds]["rules"] = INV.rows(rule)
                listing[ds]["equal"] = set(rule) == set(js)
            continue
        host = make_host(ds)
        js = _jobs_for(a, host)
        dm = get_dm(dm_config(a.dm), ds=ds, device=dev, cache_only=cache_only)
        try:
            n = R.run(ds, dm=dm, root=a.root or None, jobs=js, host=host)
        finally:
            dm.close()
        _log(f"decide routed {ds}: {n}")
    if a.list_jobs:
        _json_out(listing, a.list_jobs)


def cmd_decide_hb0(a, st):
    from habitus.decide.habit.retrofit import run
    for ds in ds_list(a.ds):
        kplans = parse_kplans(a.kplans)
        if ds == "CADETS_E3" and "allc-pre" in kplans:
            raise SystemExit("CADETS_E3: allc-pre is refused (its k=all reference includes clean days after "
                             "the first test day); use --kplans 2")
        run(ds, kplans=tuple(kplans), root=a.root or _runs(st, "habit"))


def cmd_decide_habit(a, st):
    """Habit-channel score files of one arm: channel.source_for (fold-checked DMAnswerSource) ->
    channel.score_level per k plan."""
    from habitus.decide.habit import channel
    arm = habit_arm(a.arm)
    if a.fold not in folds():
        raise SystemExit(f"--fold {a.fold!r}: one of {folds()}")
    dev = device_of(a)
    for ds in ds_list(a.ds):
        src = channel.source_for(arm, a.fold, ds, device=dev, cache_only=dev == "cpu", settings=st)
        try:
            for kp in parse_kplans(a.kplans):
                res = channel.score_level(ds, arm, kp, src, fold=a.fold, root=a.root or None, settings=st)
                _log(f"decide habit {ds} {arm} {kp}: {res.get('written')} written, {res.get('refused')} refused")
        finally:
            close = getattr(getattr(src, "dm", None), "close", None)
            if callable(close):
                close()


# --------------------------------------------------------------------------- #
# train (GPU except items; not run in this phase)
# --------------------------------------------------------------------------- #
def _train_config(st, name):
    """configs/train/<name>.yaml (the whole mapping: variant / base / recipe / seeds / ...)."""
    import yaml
    with open(os.path.join(st.paths.home, "configs", "train", f"{name}.yaml"), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _recipe(st, name):
    """The 'recipe' entry of configs/train/<name>.yaml: the hyper-parameters the trainers take."""
    cfg = _train_config(st, name)
    rec = cfg.get("recipe") if isinstance(cfg, dict) else None
    if not isinstance(rec, dict):
        raise SystemExit(f"configs/train/{name}.yaml has no 'recipe' mapping")
    return rec


# default run directory per (variant, backbone): the directories configs/dm/*.yaml read
JEV_RUN_DIRS = {("lora", "Qwen/Qwen3.5-4B"): "HB", ("head", "Qwen/Qwen3.5-4B"): "HB.head",
                ("lora", "Qwen/Qwen3.5-0.8B"): "HB.08b"}


def jev_run_dir(variant, backbone):
    """runs/adapters/<fold>/<this> of a jev training run (configs/dm jev-qwen3.5-4b-lora: HB,
    jev-qwen3.5-4b-head: HB.head, jev-qwen3.5-0.8b-lora: HB.08b)."""
    try:
        return JEV_RUN_DIRS[(variant, backbone)]
    except KeyError:
        raise SystemExit(f"train jev --variant {variant} --backbone {backbone}: no arm uses this combination "
                         f"({', '.join(f'{v} {b}' for v, b in JEV_RUN_DIRS)}); give --out") from None


def _check_fold(fold):
    if fold not in folds():
        raise SystemExit(f"--fold {fold!r}: one of {folds()}")
    return fold


def cmd_train_items(a, st):
    from habitus.train import items as IT
    fold = _check_fold(a.fold)
    if a.select:
        man = IT.export_select(fold, seed=a.seed, limit=a.limit, root=a.out, settings=st)
    else:
        man = IT.export_fold(fold, seed=a.seed, limit=a.limit, root=a.out, settings=st)
        if a.n:
            man = {"export": man, "sample": IT.sample_fold(fold, n=a.n, seed=a.seed, root=a.out, settings=st)}
    _json_out(man)


def _training_set(a, st):
    """(train + inner items) of a fold: train.items.sample_fold's sample plus the export's inner-day
    files; the trainers split them by role (inner=None)."""
    from habitus.train import items as IT
    sm = IT.sample_fold(a.fold, n=a.n, seed=a.seed, root=a.items, settings=st)
    d = IT.items_dir(a.fold, a.items, st)
    (name,) = sm["file"]
    items = list(IT.read_items(os.path.join(d, "sample", name)))
    with open(os.path.join(d, "manifest.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    for f in sorted(man["files"]):
        if f.endswith(".inner.jsonl.gz"):
            items += [it for it in IT.read_items(os.path.join(d, f)) if it.role == "inner"]
    return items


def cmd_train_jev(a, st):
    from habitus.dm.jev import REVISIONS
    from habitus.train.jev import train_jev
    fold = _check_fold(a.fold)
    out = a.out or _runs(st, "adapters", fold, jev_run_dir(a.variant, a.backbone))
    items = _training_set(a, st)
    rec = train_jev(items, None, out, backbone=a.backbone, revision=REVISIONS[a.backbone], fold=fold,
                    device=device_of(a), recipe=_recipe(st, "jev-lora" if a.variant == "lora" else "jev-head"),
                    variant=a.variant, settings=st)
    _json_out(rec)


def cmd_train_laya(a, st):
    from habitus.train.laya import train_laya
    fold = _check_fold(a.fold)
    name = "laya" if a.layout == "la" else "laya-mq"
    if a.seeds:
        seeds = tuple(int(x) for x in a.seeds.split(",") if x.strip())
    else:                                           # the YAML's seeds (laya: 0,1,2; laya-mq: 0)
        seeds = tuple(int(x) for x in (_train_config(st, name).get("seeds") or (0, 1, 2)))
    items = _training_set(a, st)
    out = a.out or _runs(st, "adapters", fold, "hbl")
    rec = train_laya(items, None, out, init=a.init, fold=fold, device=device_of(a), layout=a.layout, seeds=seeds,
                     recipe=_train_config(st, name), settings=st)
    _json_out(rec)


# --------------------------------------------------------------------------- #
# eval (the only label readers; exempt from the guard)
# --------------------------------------------------------------------------- #
def cmd_eval_confusion(a, st):
    from habitus.eval import tables as T
    rows = []
    for ds in ds_list(a.ds):
        for r in T.run_host(ds, root=a.root or None, source=a.source):
            if r["level"] == a.level and r["k"] in [T.KNAME[k] for k in parse_ks(a.ks)] and r["column"] == a.field:
                rows.append({c: r.get(c) for c in ("ds", "level", "k", "column", "tp", "fp", "tn", "fn", "precision",
                                                     "recall", "f1", "mcc", "threshold", "mcc_ties_neg")})
    _json_out(rows, a.out)


def cmd_eval_tables(a, st):
    from habitus.repro import reproduce
    out = a.out or _runs(st, "reproduce", "tables")
    paths = reproduce.tables(out, root=a.root or None, source=a.source, jobs=a.jobs)
    _json_out({k: str(v) for k, v in paths.items()})


def cmd_eval_operating(a, st):
    from habitus.eval import operating as OP
    whats = tuple(x for x in a.what.split(",") if x)
    out = a.out or _runs(st, "operating")
    paths = OP.run(out, what=whats, settings=st)
    _json_out({k: str(v) for k, v in (paths or {}).items()})


def cmd_eval_baselines(a, st):
    import subprocess
    cmd = [sys.executable, os.path.join(st.paths.home, "scripts", "export_baselines.py")]
    if a.upstream_artifacts:
        cmd += ["--upstream-artifacts", a.upstream_artifacts]
    return subprocess.call(cmd)


def cmd_eval_habit(a, st):
    """Best-MCC rows of habit-channel score directories (HB0 and the arms) under --root."""
    from habitus.decide.habit import channel
    from habitus.eval import metrics as M
    from habitus.questions.habit.types import level_dir
    rows = []
    for ds in ds_list(a.ds):
        host = make_host(ds, source="nodeindex")
        for lv in [x for x in a.levels.split(",") if x]:
            for kp in parse_kplans(a.kplans):
                ld = level_dir(lv, kp)
                vec, why = channel.habit_vectors(host, ld, root=a.root or None)
                if vec is None:
                    rows.append({"ds": ds, "level": lv, "kplan": kp, "error": why})
                    continue
                res = M.confusion(host, vec["s"])
                tie, neg = res.get("tie_block") or {}, res.get("ties_negative") or {}
                rows.append({"ds": ds, "level": lv, "kplan": kp, "ldir": ld,
                             **{k: res.get(k) for k in ("tp", "fp", "tn", "fn", "precision", "recall", "f1", "mcc",
                                                        "threshold")},
                             "tie_n": tie.get("n"), "tie_gt": tie.get("gt"), "mcc_ties_neg": neg.get("mcc"),
                             "ap": res.get("ap"), "auc": res.get("auc")})
    _json_out(rows, a.out)


# --------------------------------------------------------------------------- #
# reproduce / gates
# --------------------------------------------------------------------------- #
def cmd_reproduce(a, st):
    from habitus.repro import reproduce
    rep = reproduce.run(a.level, out=a.out, jobs=a.jobs)
    print(rep.table())
    return 0 if rep.ok else 1


def cmd_gates(a, st):
    from habitus.repro import gates
    only = [x.strip() for x in a.only.split(",") if x.strip()] if a.only else None
    skip = [x.strip() for x in a.skip.split(",") if x.strip()] if a.skip else []
    rep = gates.run(only=only, skip=skip, answers=a.answers, gpu=a.gpu, jobs=a.jobs, out=a.out)
    print(rep.table())
    return 0 if rep.ok else 1


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def _sub(sp, name, fn, help_):
    p = sp.add_parser(name, help=help_, description=help_)
    p.set_defaults(func=fn)
    return p


def _job_args(p, k_name="--k"):
    p.add_argument("--ds", required=True)
    p.add_argument("--level", default="E2", help="E2 | L0..L4")
    p.add_argument(k_name, required=True, help="0 | 2 | all")
    p.add_argument("--day", required=True)
    p.add_argument("--role", default="test", choices=("test", "null"),
                   help="test (reference ref_for) or null (the day's LODO reference)")


def _inventory_args(p):
    p.add_argument("--from-inventory", dest="from_inventory", action="store_true",
                   help="exactly the shipped score files (decide.inventory.inventory)")
    p.add_argument("--inventory-root", dest="inventory_root", default=None,
                   help="score root the inventory is read from (default paths.scores)")
    p.add_argument("--level", default="E2", help="rule jobs: E2 | L0..L4 (comma list)")
    p.add_argument("--ks", default="all,2,0", help="rule jobs: k in {0, 2, all}")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--nulls", dest="nulls", action="store_true", default=True)
    g.add_argument("--no-nulls", dest="nulls", action="store_false")


def _db_args(p):
    p.add_argument("--db-host", dest="db_host", default=None, help="postgres host (upstream default 'postgres')")
    p.add_argument("--db-user", dest="db_user", default=None)
    p.add_argument("--db-password-file", dest="db_password_file", default=None,
                   help="file whose first line is the password (mode 0600); or set $HABITUS_DB_PASSWORD")
    p.add_argument("--db-password", dest="db_password", default=None,
                   help="visible to other users in `ps`: prefer --db-password-file or $HABITUS_DB_PASSWORD")
    p.add_argument("--db-port", dest="db_port", type=int, default=None)


def _artifacts_arg(p):
    p.add_argument("--pidsmaker-artifacts", dest="pidsmaker_artifacts", default=None,
                   help="PIDSMaker artifact root holding the graphs (default: runs/pidsmaker, then "
                        "paths.pidsmaker_artifacts)")


def _device_args(p):
    g = p.add_mutually_exclusive_group()
    g.add_argument("--cache-only", dest="cache_only", action="store_true", default=True,
                   help="answers from the cache only; a miss raises (default)")
    g.add_argument("--gpu", type=int, default=None, help="forward misses on this GPU (never a forbidden one)")


def build_parser():
    p = argparse.ArgumentParser(prog="habitus", description="HABITUS decision-model pipeline: "
                                "data -> state -> questions -> dm -> decide -> eval")
    p.add_argument("--config", default=None, help="a YAML merged over configs/default.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    g = p.add_subparsers(dest="group", required=True, metavar="<group>")

    # data
    d = g.add_parser("data", help="PIDSMaker graphs -> cards -> node index (CPU; see docs/DATA.md)")
    ds_ = d.add_subparsers(dest="cmd", required=True, metavar="<command>")
    x = _sub(ds_, "graphs", cmd_data_graphs, "upstream PIDSMaker construction + transformation (subprocess)")
    x.add_argument("--ds", required=True)
    x.add_argument("--python", default=None, help="interpreter with upstream PIDSMaker (default paths.pidsmaker_python)")
    x.add_argument("--pidsmaker-artifacts", dest="pidsmaker_artifacts", default=None,
                   help="where the graphs are built (default runs/pidsmaker)")
    _db_args(x)
    x = _sub(ds_, "resolver", cmd_data_resolver, "DB node tables -> resolver cache (fails without a DB)")
    x.add_argument("--ds", required=True)
    _db_args(x)
    x = _sub(ds_, "cards", cmd_data_cards, "graphs + resolver -> cards/<ds>/<day>.json.gz + links/")
    x.add_argument("--ds", required=True)
    x.add_argument("--days", default=None)
    x.add_argument("--out", default=None, help="default runs/cards")
    _artifacts_arg(x)
    x = _sub(ds_, "manifest", cmd_data_manifest, "recompute usable days; FAIL if != configs/datasets.yaml")
    x.add_argument("--ds", required=True)
    _artifacts_arg(x)
    x = _sub(ds_, "index", cmd_data_index, "every card file -> nodeindex/<ds>/<day>.json.gz")
    x.add_argument("--ds", required=True)
    x.add_argument("--cards", default=None, help="default paths.cards")
    x.add_argument("--out", default=None, help="default runs/nodeindex")
    x = _sub(ds_, "verify", cmd_data_verify, "card / link JSON sha256 vs the release MANIFEST.json")
    x.add_argument("--ds", required=True)
    x.add_argument("--cards", default=None, help="rebuilt cards dir (default runs/cards)")

    # state
    s = g.add_parser("state", help="compiled host-reference states")
    ss = s.add_subparsers(dest="cmd", required=True, metavar="<command>")
    x = _sub(ss, "build", cmd_state_build, "one job's states -> jsonl of state sha1s")
    _job_args(x)
    x.add_argument("--text", action="store_true", help="also write the serialized state text")
    x.add_argument("--out", default=None)

    # questions
    q = g.add_parser("questions", help="typed questions and their answer-cache keys")
    qs = q.add_subparsers(dest="cmd", required=True, metavar="<command>")
    x = _sub(qs, "render", cmd_questions_render, "one job's routed typed questions: forms, keys, cache hits / "
             "misses (the habit questions are rendered inside `habitus decide habit`)")
    _job_args(x)
    x.add_argument("--dm", default=MAIN_DM, help="configs/dm name or arm name")
    x.add_argument("--answers", default="release", choices=("release", "unpruned", "none"))
    x.add_argument("--out", default=None, help="FILE.jsonl (a FILE.summary.json is written next to it)")
    x = _sub(qs, "keys", cmd_questions_keys, "every form key of a host's jobs (make_release, G2u)")
    x.add_argument("--ds", required=True)
    _inventory_args(x)
    x.add_argument("--dm", default=MAIN_DM)
    x.add_argument("--answers", default="none", choices=("release", "unpruned", "none"))
    x.add_argument("--out", default=None)

    # dm
    m = g.add_parser("dm", help="decision models")
    ms = m.add_subparsers(dest="cmd", required=True, metavar="<command>")
    x = _sub(ms, "info", cmd_dm_info, "resolved config, tag, revision, cache status")
    x.add_argument("--dm", required=True, help="configs/dm name or arm name (e.g. HB)")
    x.add_argument("--fold", default=None, help="a trained arm's fold (default: from --ds)")
    x.add_argument("--ds", default=None, help="the scored host (a trained arm's fold is dm.fold_for(ds))")
    x = _sub(ms, "smoke", cmd_dm_smoke, "G7 (optional): re-forward cached forms on a GPU")
    x.add_argument("--dm", required=True, help="configs/dm name or arm name")
    x.add_argument("--gpu", type=int, required=True)
    x.add_argument("--n", type=int, default=64, help="subset size (ignored with --full-day)")
    x.add_argument("--full-day", dest="full_day", action="store_true",
                   help="replay the whole job with its original batch makeup (main config only)")
    x.add_argument("--ds", default="CLEARSCOPE_E3")
    x.add_argument("--k", default="all")
    x.add_argument("--day", default="2018-04-11")
    x.add_argument("--out", default=None)

    # decide
    c = g.add_parser("decide", help="DecisionModel answers -> score files")
    cs = c.add_subparsers(dest="cmd", required=True, metavar="<command>")
    x = _sub(cs, "routed", cmd_decide_routed, "routed E2 / E1 score files (default: cache-only replay)")
    x.add_argument("--ds", required=True, help="DS[,DS..]")
    _inventory_args(x)
    x.add_argument("--dm", default=MAIN_DM, help="configs/dm name or arm name")
    _device_args(x)
    x.add_argument("--root", default=None, help="write root (default runs/scores)")
    x.add_argument("--list-jobs", dest="list_jobs", default=None,
                   help="write the job list (and, with --from-inventory, the rule jobs) as JSON; score nothing")
    x = _sub(cs, "hb0", cmd_decide_hb0, "HB0 lineage-split decision layer over the frozen E2 routes (CPU)")
    x.add_argument("--ds", required=True)
    x.add_argument("--kplans", default="2,allc-pre")
    x.add_argument("--root", default=None, help="default runs/habit")
    x = _sub(cs, "habit", cmd_decide_habit, "habit-channel score files of one arm")
    x.add_argument("--ds", required=True)
    x.add_argument("--kplans", default="2,allc-pre")
    x.add_argument("--arm", "--dm", dest="arm", required=True,
                   help="a habit arm (habitus.dm.ARMS, e.g. HB-L.zs) or its configs/dm name (e.g. laya-typed-zs)")
    x.add_argument("--fold", required=True)
    _device_args(x)
    x.add_argument("--root", default=None, help="default runs/habit")

    # train
    t = g.add_parser("train", help="label-free habit items and decision-model training (GPU except items)")
    ts = t.add_subparsers(dest="cmd", required=True, metavar="<command>")
    x = _sub(ts, "items", cmd_train_items, "export label-free habit items of a fold")
    x.add_argument("--fold", required=True)
    x.add_argument("--select", action="store_true")
    x.add_argument("--n", type=int, default=None)
    x.add_argument("--seed", type=int, default=0)
    x.add_argument("--limit", type=int, default=None)
    x.add_argument("--out", default=None, help="default runs/items")
    for name, fn in (("jev", cmd_train_jev), ("laya", cmd_train_laya)):
        x = _sub(ts, name, fn, f"train a {name} decision model on a fold's habit items")
        x.add_argument("--fold", required=True)
        x.add_argument("--gpu", type=int, required=True)
        x.add_argument("--n", type=int, required=True, help="items sampled from the fold")
        x.add_argument("--seed", type=int, default=0)
        x.add_argument("--items", default=None, help="items root (default runs/items)")
        x.add_argument("--out", default=None, help="default runs/adapters/<fold>/...")
        if name == "jev":
            x.add_argument("--variant", choices=("lora", "head"), default="lora")
            x.add_argument("--backbone", default="Qwen/Qwen3.5-4B", choices=("Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-0.8B"))
        else:
            x.add_argument("--layout", choices=("la", "mq"), default="la")
            x.add_argument("--seeds", default=None, help="comma list (default: the configs/train YAML's seeds)")
            x.add_argument("--init", default="typed-decisions")

    # eval
    e = g.add_parser("eval", help="the ONLY label readers: confusion, tables, operating, baselines")
    es = e.add_subparsers(dest="cmd", required=True, metavar="<command>")
    x = _sub(es, "confusion", cmd_eval_confusion, "best-MCC node confusion of one level / field")
    x.add_argument("--ds", required=True)
    x.add_argument("--level", default="E2")
    x.add_argument("--ks", default="2,all")
    x.add_argument("--field", default="composite", choices=FIELDS)
    x.add_argument("--root", default=None)
    x.add_argument("--source", default="nodeindex", choices=("nodeindex", "cards"))
    x.add_argument("--out", default=None)
    x = _sub(es, "tables", cmd_eval_tables, "ablation_e1_e2.csv/.md + main_config.csv + summary_means.json")
    x.add_argument("--root", default=None, help="score root (default paths.scores)")
    x.add_argument("--source", default="nodeindex", choices=("nodeindex", "cards"))
    x.add_argument("--out", default=None, help="default runs/reproduce/tables")
    x.add_argument("--jobs", type=int, default=1)
    x = _sub(es, "operating", cmd_eval_operating, "budget / sensitivity / forms / bootstrap tables")
    x.add_argument("--what", default="budget,sensitivity,forms,boot")
    x.add_argument("--out", default=None, help="default runs/operating")
    x = _sub(es, "baselines", cmd_eval_baselines, "server-only export of the PIDSMaker baseline rows")
    x.add_argument("--upstream-artifacts", dest="upstream_artifacts", default=None)
    x = _sub(es, "habit", cmd_eval_habit, "best-MCC rows of habit-channel levels")
    x.add_argument("--ds", required=True)
    x.add_argument("--levels", default="HB0")
    x.add_argument("--kplans", default="2,allc-pre")
    x.add_argument("--root", default=None, help="default runs/habit")
    x.add_argument("--out", default=None)

    # reproduce / gates
    r = g.add_parser("reproduce", help="recompute the published numbers and compare (PASS/FAIL per number)")
    r.set_defaults(func=cmd_reproduce, cmd=None)
    r.add_argument("--level", default="tables", choices=("tables", "replay", "operating", "all"))
    r.add_argument("--out", default=None, help="default runs/reproduce")
    r.add_argument("--jobs", type=int, default=5, help="parallel child processes (<= 5)")
    x = g.add_parser("gates", help="the release gates G1..G11 -> runs/gates/report.{json,md}")
    x.set_defaults(func=cmd_gates, cmd=None)
    x.add_argument("--only", default=None, help="comma list, e.g. G1,G5")
    x.add_argument("--skip", default=None)
    x.add_argument("--answers", default="release", choices=("release", "unpruned"))
    x.add_argument("--gpu", type=int, default=None, help="G7 only: a non-forbidden physical GPU")
    x.add_argument("--jobs", type=int, default=5)
    x.add_argument("--out", default=None, help="default runs/gates")
    return p


def _env_overrides(a):
    """Settings overrides given as options (applied to the environment before settings load)."""
    env = {}
    if getattr(a, "upstream_artifacts", None):
        env["PIDSMAKER_UPSTREAM_ARTIFACTS"] = os.path.abspath(a.upstream_artifacts)
    return env


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.config:
        os.environ["HABITUS_CONFIG"] = os.path.abspath(a.config)
    os.environ.update(_env_overrides(a))
    guarded = a.group not in EXEMPT
    roots = _write_roots()
    if roots:                                       # temp files must land inside the write audit's roots
        tmp = os.environ.get("TMPDIR")
        if not tmp or not any(os.path.abspath(tmp).startswith(r.rstrip(os.sep) + os.sep) for r in roots):
            tmp = os.path.join(roots[0], "tmp")
            os.environ["TMPDIR"] = tmp
        os.makedirs(tmp, exist_ok=True)
        os.environ.setdefault("HF_MODULES_CACHE", os.path.join(roots[0], "hf_modules"))
        # the Triton JIT cache (fla's Gated DeltaNet kernels on CUDA) writes on its first compile:
        # outside the roots (default ~/.triton/cache) a cold cache would fail every GPU forward
        tc = os.environ.get("TRITON_CACHE_DIR")
        if not tc or not any(os.path.abspath(tc).startswith(r.rstrip(os.sep) + os.sep) for r in roots):
            os.environ["TRITON_CACHE_DIR"] = os.path.join(roots[0], "triton")
        import tempfile
        tempfile.tempdir = None
    if guarded:                                     # FIRST: before any other habitus module is imported
        from habitus import guard
        guard.install_guard(a.group, write_roots=roots)
        _install_open_log()
    st = _settings()
    _apply_env(st, a)
    if forwards_model(a):
        check_gpu(a, st)
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""      # CPU commands never initialise CUDA
    if a.verbose:
        _log(f"{a.group} {a.cmd or ''}: guard {'installed' if guarded else 'exempt'}, runs {st.paths.runs}")
    from habitus.config import ConfigError
    from habitus.dm.base import CacheMiss, Refused
    try:
        rc = a.func(a, st)
    except CacheMiss as e:
        raise SystemExit(f"cache miss (cache-only): {e}") from None
    except Refused as e:
        raise SystemExit(f"refused: {e}") from None
    except ConfigError as e:
        raise SystemExit(f"configuration error: {e}") from None
    return int(rc or 0)


if __name__ == "__main__":
    sys.exit(main())
