"""`habitus reproduce`: recompute the published numbers from the release bundle and compare.

    habitus reproduce --level tables      release scores + nodeindex + GT -> eval.tables -> compare
                                          with results/habitus (minutes; G1)
    habitus reproduce --level replay      per host, the GUARDED child `habitus decide routed
                                          --from-inventory --cache-only` rebuilds every shipped score
                                          file from cards + precedent cache + answers + tokenizer;
                                          per-level compare with the shipped files (G4), then the
                                          tables again over the replay root (G1)
    habitus reproduce --level operating   budget / sensitivity / forms / bootstrap (G9)
    habitus reproduce --level all         the three

Outputs go under runs/reproduce/ (a directory this command owns: it is recreated on each run). The
parent process only compares outputs; every state / questions / decide step runs in a child process
with the label guard, a write audit limited to paths.runs and an open log (G5c). A PASS/FAIL line is
printed per compared quantity and the exit status is non-zero on any mismatch.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from habitus.repro.compare import (Check, Report, compare_csv_bytes, compare_csv_numeric, compare_json,
                                   compare_score_file, compare_text)

LEVELS = ("tables", "replay", "operating", "all")
MAIN_DM = "jev-qwen3.5-4b"
MAX_JOBS = 5
OWNED = ".habitus-out"                  # marker of a directory this package created and may recreate
# the published headline (results/habitus/main_config.csv; paper tab:summary)
N_ABLATION_ROWS = 251
MAIN_MEAN = 0.3822396
MAIN_MCC = {"CADETS_E3": 0.405488, "THEIA_E3": 0.595552, "CLEARSCOPE_E3": 0.112087, "optc_h501": 0.392856,
            "ATLASV2_EDR": 0.405215}
TABLE_FILES = ("ablation_e1_e2.csv", "ablation_e1_e2.md", "main_config.csv", "summary_means.json")
OPERATING_CSV = ("budget_rows.csv", "per_attack_recall.csv", "sensitivity_composition.csv", "sensitivity_forms.csv",
                 "form_gaps.csv", "bootstrap_mcc.csv")
FORMS_SUMMARY = "forms_summary.json"
FLOAT_TOL = 1e-12


# --------------------------------------------------------------------------- #
# Settings, directories, child processes
# --------------------------------------------------------------------------- #
def settings():
    from habitus import config
    return config.settings()


def hosts():
    from habitus.hosts import DATASETS
    return tuple(DATASETS)


def results_dir(st=None):
    return Path((st or settings()).paths.home) / "results" / "habitus"


def fresh_dir(path):
    """An empty directory at `path` that this package owns (marker file). An existing directory is
    recreated only when it carries the marker; a foreign non-empty one raises."""
    p = Path(path)
    if p.exists():
        if (p / OWNED).exists():
            shutil.rmtree(p)
        elif any(p.iterdir()):
            raise FileExistsError(f"{p} exists, is not empty and was not created by habitus: choose another --out")
    p.mkdir(parents=True, exist_ok=True)
    (p / OWNED).write_text("created by habitus; recreated on the next run\n", encoding="utf-8")
    return p


def src_dir():
    return str(Path(__file__).resolve().parents[2])


def habitus_argv(*args):
    """argv of a `habitus ...` child: this interpreter running habitus.cli (no console script needed)."""
    return [sys.executable, "-m", "habitus.cli", *[str(a) for a in args]]


def child_env(st=None, *, write_roots=None, open_log=None, gpu=False, extra=None):
    """Environment of a child: package on PYTHONPATH, thread caps, offline hub, CUDA hidden unless
    `gpu`, the write audit (HABITUS_WRITE_ROOTS) and the open log (HABITUS_OPEN_LOG)."""
    from habitus.cli import OPEN_LOG_ENV, WRITE_ROOTS_ENV
    st = st or settings()
    env = dict(os.environ)
    pp = env.get("PYTHONPATH", "")
    if src_dir() not in pp.split(os.pathsep):
        env["PYTHONPATH"] = src_dir() + (os.pathsep + pp if pp else "")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[v] = str(min(int(env.get(v) or st.threads), st.threads))
    env.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_HUB_DISABLE_TELEMETRY="1", HF_MODULES_CACHE=os.path.join(st.paths.runs, "hf_modules"))
    if not gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    if write_roots:
        env[WRITE_ROOTS_ENV] = os.pathsep.join(os.path.abspath(str(r)) for r in write_roots)
        env["TMPDIR"] = os.path.join(os.path.abspath(str(write_roots[0])), "tmp")   # temp files inside the audit
    else:
        env.pop(WRITE_ROOTS_ENV, None)
    if open_log:
        env[OPEN_LOG_ENV] = os.path.abspath(str(open_log))
    else:
        env.pop(OPEN_LOG_ENV, None)
    env.update(extra or {})
    return env


def run_children(specs, jobs=MAX_JOBS, timeout=None):
    """Run [(name, argv, env, log_path)] with at most `jobs` (<= 5) at a time.
    Returns {name: {"rc", "seconds", "log"}}; a child over `timeout` seconds is killed (rc -9)."""
    jobs = max(1, min(int(jobs or 1), MAX_JOBS))
    pending, running, done = list(specs), {}, {}
    while pending or running:
        while pending and len(running) < jobs:
            name, argv, env, log = pending.pop(0)
            argv = [str(x) for x in argv]
            Path(log).parent.mkdir(parents=True, exist_ok=True)
            fh = open(log, "w", encoding="utf-8")
            fh.write("$ " + " ".join(argv) + "\n")
            fh.flush()
            proc = subprocess.Popen(argv, env=env, stdout=fh, stderr=subprocess.STDOUT, cwd=settings().paths.home)
            running[name] = (proc, fh, time.time(), str(log))
        time.sleep(0.2)
        for name in list(running):
            proc, fh, t0, log = running[name]
            rc = proc.poll()
            if rc is None and timeout is not None and time.time() - t0 > timeout:
                proc.kill()
                rc = proc.wait()
            if rc is not None:
                fh.close()
                done[name] = {"rc": rc, "seconds": round(time.time() - t0, 1), "log": log}
                del running[name]
    return done


def log_tail(path, n=15):
    try:
        return "".join(Path(path).read_text(encoding="utf-8", errors="replace").splitlines(True)[-n:])
    except OSError:
        return ""


# --------------------------------------------------------------------------- #
# tables (G1)
# --------------------------------------------------------------------------- #
def _host_rows(args):
    ds, root, source = args
    from habitus.eval import tables as T
    return T.run_host(ds, root=root, source=source)


def tables(out, *, root=None, source="nodeindex", hosts_=None, jobs=1):
    """eval.tables over the score files under `root` (default paths.scores) -> out/{ablation_e1_e2.csv,
    ablation_e1_e2.md, main_config.csv, summary_means.json}; the md stamp is the shipped one."""
    from habitus.eval import tables as T
    hs = tuple(hosts_ or hosts())
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    args = [(ds, None if root is None else str(root), source) for ds in hs]
    if jobs and jobs > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=min(jobs, MAX_JOBS, len(hs)), mp_context=mp.get_context("spawn")) as ex:
            parts = list(ex.map(_host_rows, args))
    else:
        parts = [_host_rows(x) for x in args]
    rows, status, test_days = T.merge_parts(parts)
    return T.write_tables(rows, out, generated=T.REF_GENERATED, test_days=test_days, status=status)


def headline_checks(csv_bytes, name="tables"):
    """The published headline: 251 ablation rows, the 5 main rows and their mean."""
    import csv
    import io
    rows = list(csv.DictReader(io.StringIO(csv_bytes.decode("utf-8"), newline="")))
    out = [Check(f"{name}: ablation rows", len(rows) == N_ABLATION_ROWS, "", len(rows), N_ABLATION_ROWS)]
    main = [r for r in rows if r.get("main_config") == "yes" and r.get("column") == "composite"]
    got = {r["ds"]: float(r["mcc"]) for r in main}
    out.append(Check(f"{name}: main rows", sorted(got) == sorted(MAIN_MCC), "", sorted(got), sorted(MAIN_MCC)))
    for ds, want in MAIN_MCC.items():
        out.append(Check(f"{name}: main MCC {ds}", got.get(ds) == want, "", got.get(ds), want))
    if len(got) == len(MAIN_MCC):
        mean = round(sum(got.values()) / len(got), 7)
        out.append(Check(f"{name}: main mean MCC", mean == MAIN_MEAN, "", mean, MAIN_MEAN))
    return out


def compare_tables(new_dir, ref_dir=None, *, name="tables", headline=True):
    """Checks of out/{TABLE_FILES} against the shipped results/habitus copies."""
    from habitus.repro.compare import GENERATED_STAMP
    new_dir, ref_dir = Path(new_dir), Path(ref_dir or results_dir())
    checks = []
    for f in TABLE_FILES:
        a, b = new_dir / f, ref_dir / f
        if not a.exists() or not b.exists():
            checks.append(Check(f"{name}: {f}", False, f"missing {'new' if not a.exists() else 'reference'} file"))
            continue
        if f.endswith(".csv"):
            checks += compare_csv_bytes(a.read_bytes(), b.read_bytes(), ignore_cols=("note",), name=f"{name}: {f}")
        elif f.endswith(".md"):
            checks += compare_text(a.read_text(encoding="utf-8"), b.read_text(encoding="utf-8"),
                                   ignore=(GENERATED_STAMP,), name=f"{name}: {f}")
        else:
            checks += compare_json(json.loads(a.read_text(encoding="utf-8")), json.loads(b.read_text(encoding="utf-8")),
                                   tol=FLOAT_TOL, name=f"{name}: {f}")
    if headline and (new_dir / TABLE_FILES[0]).exists():
        checks += headline_checks((new_dir / TABLE_FILES[0]).read_bytes(), name)
    return checks


def same_files(dir_a, dir_b, files=TABLE_FILES, name="same"):
    """Byte equality of the same file names in two output directories."""
    out = []
    for f in files:
        a, b = Path(dir_a) / f, Path(dir_b) / f
        ok = a.exists() and b.exists() and a.read_bytes() == b.read_bytes()
        out.append(Check(f"{name}: {f}", ok, "byte-identical" if ok else "differ or missing"))
    return out


# --------------------------------------------------------------------------- #
# replay (G4b)
# --------------------------------------------------------------------------- #
def level_dirs():
    return [f"{lv}_k{k}" for lv in ("E2", "L0", "L1", "L2", "L3", "L4") for k in ("0", "2", "all")]


def score_files(root, ds):
    """{"<dir>/<day>.json.gz": Path} of the score files of one allowed host under `root`."""
    out = {}
    base = Path(root) / ds
    for d in level_dirs():
        if (base / d).is_dir():
            for p in sorted((base / d).iterdir()):
                if p.name.endswith(".json.gz"):
                    out[f"{d}/{p.name}"] = p
    return out


def replay_children(root, out, st=None, hosts_=None, inventory_root=None):
    """The guarded `decide routed --from-inventory --cache-only` child of every host."""
    st = st or settings()
    inv = inventory_root or st.paths.scores
    specs = []
    for ds in hosts_ or hosts():
        argv = habitus_argv("decide", "routed", "--ds", ds, "--from-inventory", "--inventory-root", inv,
                            "--dm", MAIN_DM, "--cache-only", "--root", root)
        env = child_env(st, write_roots=[st.paths.runs], open_log=Path(out) / f"open_{ds}.log")
        specs.append((f"replay {ds}", argv, env, Path(out) / "logs" / f"replay_{ds}.log"))
    return specs


def compare_replay(new_root, ref_root, hosts_=None):
    """Per (host, level dir): every shipped file recomputed equal (compare_score_file); no extra file."""
    from habitus.io import read_json_gz
    checks = []
    for ds in hosts_ or hosts():
        ref, new = score_files(ref_root, ds), score_files(new_root, ds)
        extra = sorted(set(new) - set(ref))
        checks.append(Check(f"replay {ds}: files", not extra and set(ref) <= set(new),
                            f"{len(ref)} shipped, {len(set(ref) & set(new))} recomputed, {len(extra)} extra"))
        by_dir = {}
        for rel, p in ref.items():
            by_dir.setdefault(rel.split("/")[0], []).append(rel)
        for d, rels in sorted(by_dir.items()):
            bad, worst, n = [], 0.0, 0
            for rel in rels:
                if rel not in new:
                    bad.append(f"{rel}: not recomputed")
                    continue
                cs, w = compare_score_file(read_json_gz(new[rel]), read_json_gz(ref[rel]), name=rel)
                worst, n = max(worst, w), n + 1
                bad += [f"{c.name}: {c.detail}" for c in cs if not c.ok]
            checks.append(Check(f"replay {ds}: {d}", not bad,
                                (f"{len(bad)} problems: " + "; ".join(bad[:5])) if bad else
                                f"{n} files equal, max |d| {worst!r}"))
    return checks


def replay(out, *, jobs=MAX_JOBS, st=None, hosts_=None, name="replay"):
    """Run the guarded replay children into out/root and compare (returns checks, root)."""
    st = st or settings()
    out = Path(out)
    root = fresh_dir(out / "root")
    res = run_children(replay_children(root, out, st, hosts_), jobs)
    checks = []
    for n, r in sorted(res.items()):
        checks.append(Check(f"{n}: exit", r["rc"] == 0, f"{r['seconds']}s, log {r['log']}"
                            + ("" if r["rc"] == 0 else f"\n{log_tail(r['log'])}")))
    checks += compare_replay(root, st.paths.scores, hosts_)
    return checks, root


# --------------------------------------------------------------------------- #
# operating (G9)
# --------------------------------------------------------------------------- #
def operating(out, *, st=None, what="budget,sensitivity,forms,boot"):
    """`habitus eval operating` into out/ (a child; its forms replay is a guarded grandchild), then
    the four G9 comparisons against results/habitus."""
    st = st or settings()
    out = fresh_dir(Path(out))
    argv = habitus_argv("eval", "operating", "--what", what, "--out", out)
    res = run_children([("operating", argv, child_env(st, write_roots=None), out / "logs" / "operating.log")], 1)
    r = res["operating"]
    checks = [Check("operating: exit", r["rc"] == 0, f"{r['seconds']}s, log {r['log']}"
                    + ("" if r["rc"] == 0 else f"\n{log_tail(r['log'])}"))]
    ref = results_dir(st)
    for f in OPERATING_CSV:
        a, b = out / f, ref / f
        if not a.exists() or not b.exists():
            checks.append(Check(f"operating: {f}", False, f"missing {'new' if not a.exists() else 'reference'} file"))
            continue
        checks += compare_csv_numeric(a.read_bytes(), b.read_bytes(), tol=FLOAT_TOL, name=f"operating: {f}")
    fs = out / FORMS_SUMMARY
    if not fs.exists():
        checks.append(Check("operating: forms replay", False, f"no {FORMS_SUMMARY}"))
    else:
        summary = json.loads(fs.read_text(encoding="utf-8"))
        for ds in hosts():
            s = summary.get(ds) or {}
            ok = s.get("recon_max_abs") == 0.0 and s.get("cache_misses") == 0
            checks.append(Check(f"operating: forms {ds}", ok, "", {k: s.get(k) for k in ("recon_max_abs", "cache_misses")},
                                {"recon_max_abs": 0.0, "cache_misses": 0}))
    return checks


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run(level="tables", *, out=None, jobs=MAX_JOBS):
    """The `habitus reproduce` command: returns a Report (also written to <out>/report.{json,md})."""
    if level not in LEVELS:
        raise ValueError(f"level {level!r} not in {LEVELS}")
    st = settings()
    out = Path(out or os.path.join(st.paths.runs, "reproduce"))
    out.mkdir(parents=True, exist_ok=True)
    rep = Report(f"habitus reproduce --level {level}")
    t0 = time.time()
    if level in ("tables", "all"):
        d = fresh_dir(out / "tables")
        tables(d, root=st.paths.scores, source="nodeindex", jobs=jobs)
        rep.extend(compare_tables(d, name="tables"))
    if level in ("replay", "all"):
        checks, root = replay(out / "replay", jobs=jobs, st=st)
        rep.extend(checks)
        d = fresh_dir(out / "replay" / "tables")
        tables(d, root=root, source="nodeindex", jobs=jobs)
        rep.extend(compare_tables(d, name="replay tables"))
    if level in ("operating", "all"):
        rep.extend(operating(out / "operating", st=st))
    rep.add("reproduce: seconds", True, f"{time.time() - t0:.0f}s")
    (out / "report.json").write_text(json.dumps(rep.as_dict(), indent=1, default=str) + "\n", encoding="utf-8")
    (out / "report.md").write_text("```\n" + rep.table() + "\n```\n", encoding="utf-8")
    return rep
