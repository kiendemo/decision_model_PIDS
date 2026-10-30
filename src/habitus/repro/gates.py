"""Release gates G1..G11: `habitus gates [--only G1,G2] [--skip G7] [--answers release|unpruned]`.

    G1  eval over the shipped score files reproduces ablation_e1_e2.csv / .md byte for byte (note
        column excepted), the 251 rows, the 5 main rows and their mean; cards == nodeindex source
    G2  the rendered questions of sampled host-days hash-match the release answers (0 misses), the
        frozen texts equal the original's hashes, the sanitizer resolves to the live snapshot
        (G2u with --answers unpruned: the same against the unpruned source answers; server only)
    G3  compiled states of the sampled days equal the ORIGINAL code's golden hashes; E1 snapshot test
    G4  G4a: the job rules reproduce the shipped inventory; G4b: the guarded cache-only replay
        recomputes every shipped score file exactly, then G1 over the replay root gives the same bytes
    G5  import isolation, label-operation scan, runtime guard checks, host allow-list
    G6  the unit tests pass on CPU with HABITUS_REQUIRE_ASSETS=1 (provenance tests excluded; tests marked
        local_assets, which need the Laya weights / role-table catalogs, are required on the build server only)
    G7  optional GPU smoke (only with --only G7 and --gpu N, never a forbidden GPU): a 64-question
        liveness subset per backend (jev p99 |d letter prob| <= 0.08, the bf16 batch-makeup noise floor
        being ~0.06) and the exactness replay of one whole cached job in its original batch makeup
        (`dm smoke --full-day`, p99 <= 1e-3)
    G8  no absolute server path, secret or e-mail address in the tracked files and the release text
    G9  `habitus reproduce --level operating` matches results/habitus
    G10 manifests verify, the results README table and STATUS blocks are consistent, provenance and
        salvage records hold (runs/salvage: build server only), the config mirror tests pass
    G11 HB0: guarded `habitus decide hb0` over the release E2 score files, then `habitus eval habit
        --levels HB0`, reproduces every column of results/extension_habit/hb0_rows.csv (SKIP, with the
        reason, when the role-table catalogs of paths.catalogs are absent: scripts/fetch_catalogs.sh)

Every state / questions / decide step runs as a GUARDED `habitus ...` child process (write audit
under paths.runs, open log); this process only compares. Results: runs/gates/report.{json,md}
(merged with the previous report, so a partial run updates its gates only).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

from habitus.repro import reproduce as RP
from habitus.repro import scan as SC
from habitus.repro.compare import Check, Report

GATES = ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10", "G11")
MAIN_DM = "jev-qwen3.5-4b"
# G2 / G3 samples: the main-config first test day, one LODO null day, (CLEARSCOPE_E3) the k0 first
# test day and one E1 L4 day. (ds, level, k, day, role)
G2_SAMPLES = (
    ("CADETS_E3", "E2", "2", "2018-04-06", "test"), ("CADETS_E3", "E2", "2", "2018-04-02", "null"),
    ("THEIA_E3", "E2", "all", "2018-04-10", "test"), ("THEIA_E3", "E2", "all", "2018-04-03", "null"),
    ("CLEARSCOPE_E3", "E2", "all", "2018-04-11", "test"), ("CLEARSCOPE_E3", "E2", "2", "2018-04-03", "null"),
    ("CLEARSCOPE_E3", "E2", "0", "2018-04-12", "test"), ("CLEARSCOPE_E3", "L4", "2", "2018-04-11", "test"),
    ("optc_h501", "E2", "all", "2019-09-23", "test"), ("optc_h501", "E2", "all", "2019-09-19", "null"),
    ("ATLASV2_EDR", "E2", "all", "2022-07-19", "test"), ("ATLASV2_EDR", "E2", "all", "2022-07-16", "null"),
)
G2_TESTS = ("tests/test_questions_routed.py", "tests/test_state_sanitize.py")
G3_TESTS = ("tests/test_e1_snapshot.py",)
G5_HABIT_TEST = "tests/test_habit_channel.py"
MIRROR_TESTS = ("tests/test_config.py", "tests/test_state_mirror.py", "tests/test_habit_mirror.py",
                "tests/test_dm_mirror_jev.py", "tests/test_dm_mirror_laya.py")


def _counts(test, clean, k0_nulls, e1):
    """{dir: (files, null files)} of a host under the null rules (pipeline.yaml NULL_RULES)."""
    out = {"E2_k0": (test + (clean if k0_nulls else 0), clean if k0_nulls else 0),
           "E2_k2": (test + 2, 2), "E2_kall": (test + clean, clean)}
    for kk, levels in e1.items():
        for lv in levels:
            n = {"L2": {"k2": 2, "kall": clean}, "L4": {"k2": 2, "kall": clean},
                 "L3": {"k0": clean if k0_nulls else 0}}.get(lv, {}).get(kk, 0)
            out[f"{lv}_{kk}"] = (test + n, n)
    return out


_E1_ALL = {"k0": ("L0", "L1", "L3"), "k2": ("L0", "L1", "L2", "L3", "L4"), "kall": ("L0", "L1", "L2", "L3", "L4")}
# G4a expected score-file counts per level dir (files, of which null), from the shipped inventory
EXPECTED_COUNTS = {
    "CADETS_E3": _counts(4, 8, True, _E1_ALL),
    "THEIA_E3": _counts(3, 4, True, _E1_ALL),
    "CLEARSCOPE_E3": _counts(2, 7, False, {"k0": ("L3",), "k2": ("L2", "L4"), "kall": ("L2", "L4")}),
    "optc_h501": _counts(3, 4, True, _E1_ALL),
    "ATLASV2_EDR": _counts(1, 3, True, _E1_ALL),
}
STATUS_FILES = ("README.md", "results/README.md", "docs/DECISION_MODELS.md")
STATUS_BEGIN, STATUS_END = "<!-- STATUS:BEGIN -->", "<!-- STATUS:END -->"
RELEASE_TEXT = ("MANIFEST.json", "SHA256SUMS", "README.md", "PUBLISH.md")
LAYA_CHANGES = "Changes to the vendored Laya code"


class Ctx:
    def __init__(self, out, jobs, answers, gpu, only):
        self.st = RP.settings()
        self.home = Path(self.st.paths.home)
        self.release = Path(self.st.paths.data)
        self.out = Path(out)
        self.jobs = jobs
        self.answers = answers
        self.gpu = gpu
        self.only = only
        self.passed = {}            # gate -> status of this run or the merged previous report

    def env(self, *, open_log=None, gpu=False, extra=None):
        return RP.child_env(self.st, write_roots=[self.st.paths.runs], open_log=open_log, gpu=gpu, extra=extra)


def _pytest(ctx, files, name, *, env_extra=None, guard=False, markers=None, log=None):
    """Run pytest on `files` in a child process; one Check (rc == 0) with the summary line."""
    have = [f for f in files if (ctx.home / f).exists()]
    missing = [f for f in files if f not in have]
    if not have:
        return [Check(name, False, f"test files missing: {missing}")]
    args = ["-q", "-p", "no:cacheprovider", *have] + (["-m", markers] if markers else [])
    if guard:
        code = ("import sys; from habitus import guard; guard.install_guard('decide'); import pytest; "
                f"sys.exit(pytest.main({args!r}))")
        argv = [sys.executable, "-c", code]
    else:
        argv = [sys.executable, "-m", "pytest", *args]
    env = RP.child_env(ctx.st, write_roots=None, extra=env_extra)
    log = log or ctx.out / "logs" / (re.sub(r"\W+", "_", name) + ".log")
    r = RP.run_children([(name, argv, env, log)], 1)[name]
    tail = RP.log_tail(r["log"], 3).strip().splitlines()
    summary = tail[-1] if tail else ""
    return [Check(name, r["rc"] == 0 and not missing,
                  f"{summary} ({r['seconds']}s)" + (f"; missing {missing}" if missing else "")
                  + ("" if r["rc"] == 0 else f"; log {r['log']}"))]


# Inputs each data gate reads (settings.paths keys). Checked up front: without them every child of
# the gate would fail on its own. Before scripts/make_release.py has run, G2 / G3 / G6 read the original
# research caches in place through `habitus --config configs/prerelease.yaml gates ...`.
NEEDS = {
    "G1": ("scores", "nodeindex", "cards", "ground_truth"),
    "G2": ("cards", "precedent", "answers"),
    "G2u": ("cards", "precedent"),               # the unpruned answers come from paths.pidsmaker_artifacts
    "G3": ("cards", "precedent"),
    "G4": ("scores", "nodeindex", "cards", "precedent", "answers", "ground_truth"),
    "G6": ("scores", "cards", "precedent", "answers", "ground_truth", "baselines"),
    "G9": ("scores", "nodeindex", "cards", "precedent", "answers", "ground_truth", "baselines"),
    "G11": ("scores", "nodeindex", "cards", "ground_truth"),
}
PRERELEASE_HINT = ("build the bundle with scripts/make_release.py, or (before the release exists; G2/G3/G6 only) "
                   "read the original caches in place: habitus --config configs/prerelease.yaml gates ...")


class Skip(Exception):
    """A gate that cannot apply on this machine (missing optional inputs): status SKIP with the reason."""


def preflight(ctx, gate):
    """[] when every input of `gate` exists, else one failing Check naming the missing paths keys."""
    missing = [k for k in NEEDS.get(gate, ()) if not getattr(ctx.st.paths, k, None)
               or not os.path.exists(getattr(ctx.st.paths, k))]
    if not missing:
        return []
    return [Check(f"{gate} inputs", False, f"paths {', '.join(missing)} missing ({PRERELEASE_HINT})")]


# --------------------------------------------------------------------------- #
# G1
# --------------------------------------------------------------------------- #
def g1(ctx):
    checks = preflight(ctx, "G1")
    if checks:
        return checks
    dirs = {}
    for source in ("nodeindex", "cards"):
        d = RP.fresh_dir(ctx.out / "g1" / source)
        RP.tables(d, root=ctx.st.paths.scores, source=source, jobs=ctx.jobs)
        checks += RP.compare_tables(d, name=f"G1 {source}", headline=source == "nodeindex")
        dirs[source] = d
    checks += RP.same_files(dirs["nodeindex"], dirs["cards"], name="G1 cards == nodeindex")
    return checks


# --------------------------------------------------------------------------- #
# G2
# --------------------------------------------------------------------------- #
def _sample_name(s):
    ds, level, k, day, role = s
    return f"{ds}_{level}_k{k}_{day}_{role}"


FROZEN_SCRIPT = r"""
import json, sys
from habitus import guard
guard.install_guard("questions")
from habitus.questions import routed as Q
from habitus.state import sanitize as S
gold = json.load(open(sys.argv[1], encoding="utf-8"))
san = json.load(open(sys.argv[2], encoding="utf-8"))
now = Q.text_hashes()
res = {"text": {k: now.get(k) == v for k, v in gold.items()}, "extra_texts": sorted(set(now) - set(gold)),
       "status": S.pattern_status(), "pattern_equal": S.pattern().pattern == san["pattern"]}
print("RESULT " + json.dumps(res))
"""


def _result_line(log):
    for line in reversed(Path(log).read_text(encoding="utf-8", errors="replace").splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    return None


def g2(ctx, answers=None):
    answers = answers or ctx.answers
    tag = "G2u" if answers == "unpruned" else "G2"
    missing = preflight(ctx, tag)
    if missing:
        return missing
    out = RP.fresh_dir(ctx.out / tag.lower())
    specs = []
    for s in G2_SAMPLES:
        ds, level, k, day, role = s
        o = out / f"{_sample_name(s)}.jsonl"
        argv = RP.habitus_argv("questions", "render", "--ds", ds, "--level", level, "--k", k, "--day", day,
                               "--role", role, "--dm", MAIN_DM, "--answers", answers, "--out", o)
        specs.append((_sample_name(s), argv, ctx.env(open_log=out / f"open_{_sample_name(s)}.log"),
                      out / "logs" / f"{_sample_name(s)}.log"))
    res = RP.run_children(specs, ctx.jobs)
    checks = []
    for s in G2_SAMPLES:
        n = _sample_name(s)
        sp = out / f"{n}.summary.json"
        summ = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
        ok = res[n]["rc"] == 0 and summ.get("misses") == 0 and summ.get("n_keys", 0) > 0
        checks.append(Check(f"{tag} keys {n}", ok, f"{summ.get('n_keys')} keys, {summ.get('misses')} misses vs "
                            f"{answers} ({res[n]['seconds']}s)" + ("" if res[n]["rc"] == 0 else f"; log {res[n]['log']}")))
        san = (summ.get("sanitize") or {}).get("status")
        checks.append(Check(f"{tag} sanitize {n}", san == "live==snapshot", "", san, "live==snapshot"))
    if tag == "G2u":
        return checks
    gold = ctx.home / "tests" / "golden" / "state"
    log = out / "logs" / "frozen.log"
    r = RP.run_children([("frozen", [sys.executable, "-c", FROZEN_SCRIPT, gold / "text_hashes.json",
                                     gold / "sanitize.json"], ctx.env(), log)], 1)["frozen"]
    got = _result_line(log) if r["rc"] == 0 else None
    if got is None:
        checks.append(Check("G2 frozen texts", False, f"child failed: {RP.log_tail(log)}"))
    else:
        bad = sorted(k for k, v in got["text"].items() if not v)
        checks.append(Check("G2 frozen texts", not bad and len(got["text"]) >= 10,
                            f"{len(got['text'])} text constants equal the original's sha256" if not bad else f"differ: {bad}"))
        checks.append(Check("G2 sanitize pattern", got["status"] == "live==snapshot" and got["pattern_equal"],
                            "", (got["status"], got["pattern_equal"]), ("live==snapshot", True)))
    checks += _pytest(ctx, G2_TESTS, "G2 prompt / sanitize golden tests")
    return checks


# --------------------------------------------------------------------------- #
# G3
# --------------------------------------------------------------------------- #
def _read_states(path):
    with open(path, encoding="utf-8") as fh:
        head = json.loads(fh.readline())
        rows = [json.loads(line) for line in fh if line.strip()]
    return head, rows


def compare_states(path, golden):
    """Checks of one `habitus state build` output against a tests/golden/state file."""
    head, rows = _read_states(path)
    name = Path(path).stem
    got = {}
    for r in rows:
        if r["qid"] is None:
            got[r["sig"]] = r["sha1"]
        else:
            got.setdefault(r["sig"], {})[r["qid"]] = r["sha1"]
    want = golden["states"]
    bad = sorted(s for s in set(want) | set(got) if got.get(s) != want.get(s))
    return [Check(f"G3 head {name}", head["head"] == golden["head"]),
            Check(f"G3 states {name}", not bad, f"{len(want)} signatures equal" if not bad else
                  f"{len(bad)} of {len(want)} differ, e.g. {bad[:3]}"),
            Check(f"G3 digest {name}", head["meta"].get("digest") == golden.get("digest")
                  and head["meta"].get("n_states") == golden.get("n_states"), "",
                  head["meta"].get("digest"), golden.get("digest"))]


def g3(ctx):
    import gzip
    missing = preflight(ctx, "G3")
    if missing:
        return missing
    out = RP.fresh_dir(ctx.out / "g3")
    gold_dir = ctx.home / "tests" / "golden" / "state"
    idx_p = gold_dir / "index.json"
    samples = json.loads(idx_p.read_text(encoding="utf-8"))["samples"] if idx_p.exists() else []
    want = {_sample_name(s) for s in G2_SAMPLES}
    have = {f"{s['ds']}_{s['level']}_k{s['k']}_{s['day']}_{s['role']}" for s in samples}
    checks = [Check("G3 golden samples", want <= have, f"{len(have)} golden samples cover the G2 sample"
                    if want <= have else f"missing golden: {sorted(want - have)}")]
    specs = []
    for s in samples:
        n = f"{s['ds']}_{s['level']}_k{s['k']}_{s['day']}_{s['role']}"
        argv = RP.habitus_argv("state", "build", "--ds", s["ds"], "--level", s["level"], "--k", s["k"],
                               "--day", s["day"], "--role", s["role"], "--out", out / f"{n}.jsonl")
        specs.append((n, argv, ctx.env(open_log=out / f"open_{n}.log"), out / "logs" / f"{n}.log"))
    res = RP.run_children(specs, ctx.jobs)
    for s in samples:
        n = f"{s['ds']}_{s['level']}_k{s['k']}_{s['day']}_{s['role']}"
        if res[n]["rc"] != 0:
            checks.append(Check(f"G3 build {n}", False, f"log {res[n]['log']}\n{RP.log_tail(res[n]['log'])}"))
            continue
        with gzip.open(gold_dir / s["file"], "rt", encoding="utf-8") as fh:
            golden = json.load(fh)
        checks += compare_states(out / f"{n}.jsonl", golden)
    checks += _pytest(ctx, G3_TESTS, "G3 E1 snapshot test", env_extra={"HABITUS_REQUIRE_ASSETS": "1"})
    return checks


# --------------------------------------------------------------------------- #
# G4
# --------------------------------------------------------------------------- #
def g4a(ctx):
    out = ctx.out / "g4"
    out.mkdir(parents=True, exist_ok=True)
    listing = out / "jobs.json"
    argv = RP.habitus_argv("decide", "routed", "--ds", ",".join(RP.hosts()), "--from-inventory",
                           "--inventory-root", ctx.st.paths.scores, "--list-jobs", listing)
    r = RP.run_children([("g4a", argv, ctx.env(), out / "logs" / "g4a.log")], 1)["g4a"]
    if r["rc"] != 0 or not listing.exists():
        return [Check("G4a job listing", False, f"log {r['log']}\n{RP.log_tail(r['log'])}")]
    got = json.loads(listing.read_text(encoding="utf-8"))
    gold = json.loads((ctx.home / "tests" / "golden" / "inventory.json").read_text(encoding="utf-8"))["hosts"]
    checks = []

    def key(row):
        return (row["dir"], row["day"], row["role"], tuple(row["ref"]))
    for ds in RP.hosts():
        g = got.get(ds) or {}
        inv, rules = {key(x) for x in g.get("jobs", [])}, {key(x) for x in g.get("rules", [])}
        golden = {key(x) for x in gold.get(ds, [])}
        checks.append(Check(f"G4a {ds} rules == inventory", bool(inv) and inv == rules,
                            f"{len(inv)} files" if inv == rules else f"{len(inv - rules)} only shipped, "
                            f"{len(rules - inv)} only by the rules"))
        checks.append(Check(f"G4a {ds} inventory == golden", inv == golden, "" if inv == golden else
                            f"{len(inv - golden)} not in golden, {len(golden - inv)} missing"))
        counts = {}
        for d, _day, role, _ref in inv:
            n, nn = counts.get(d, (0, 0))
            counts[d] = (n + 1, nn + (role == "null"))
        want = EXPECTED_COUNTS[ds]
        checks.append(Check(f"G4a {ds} file counts", counts == want, "" if counts == want else
                            f"differ: {sorted((d, counts.get(d), want.get(d)) for d in set(counts) | set(want) if counts.get(d) != want.get(d))}"))
    return checks


def g4b(ctx):
    checks, root = RP.replay(ctx.out / "g4" / "replay", jobs=ctx.jobs, st=ctx.st)
    d_new = RP.fresh_dir(ctx.out / "g4" / "tables_replay")
    RP.tables(d_new, root=root, source="nodeindex", jobs=ctx.jobs)
    d_ref = ctx.out / "g1" / "nodeindex"
    if not (d_ref / RP.TABLE_FILES[0]).exists():
        d_ref = RP.fresh_dir(ctx.out / "g4" / "tables_release")
        RP.tables(d_ref, root=ctx.st.paths.scores, source="nodeindex", jobs=ctx.jobs)
    checks += RP.same_files(d_new, d_ref, files=RP.TABLE_FILES[:1], name="G4 tables(replay) == tables(release)")
    return checks


def g4(ctx):
    return preflight(ctx, "G4") or [*g4a(ctx), *g4b(ctx)]


# --------------------------------------------------------------------------- #
# G5
# --------------------------------------------------------------------------- #
GUARD_SCRIPT = r"""
import json, sys
from habitus import guard
guard.install_guard("decide")
res = {"active": bool(guard.guard_active())}
def attempt(name, fn):
    try:
        fn()
        res[name] = "ok"
    except guard.LabelGuardError:
        res[name] = "blocked"
    except Exception as e:
        res[name] = "error " + type(e).__name__ + ": " + str(e)[:200]
attempt("gt_open", lambda: open(sys.argv[1], encoding="utf-8").read())
attempt("eval_import", lambda: __import__("habitus.eval.labels"))
print("RESULT " + json.dumps(res))
"""


def _open_logs(ctx):
    return sorted(ctx.out.rglob("open_*.log"))


def g5(ctx):
    from habitus import hosts as H
    src = ctx.home / "src" / "habitus"
    checks = []
    hits = SC.import_problems(SC.import_graph(src))
    checks.append(Check("G5a import graph", not hits, "; ".join(map(str, hits[:10])) or
                        "no pidsmaker.reflex/defense/attacks/detection; pidsmaker only in data.upstream (lazy); "
                        "torch only in dm/train/data.graphs/eval.baselines and never at import time of the CPU replay "
                        "entry points; eval imported by eval/repro/cli only"))
    tracked = SC.tracked_files(ctx.home)
    scripts = [f for f in SC.under(tracked, ctx.home, ("scripts/",)) if f.suffix == ".py"]
    hits = SC.scan_labels(src, extra_files=scripts, repo_root=ctx.home)
    checks.append(Check("G5b label operations", not hits, "; ".join(map(str, hits[:10])) or
                        "label reads only under habitus/eval (allow-list: " + ", ".join(SC.LABEL_ALLOWED) + ")"))
    # (c) runtime: open logs of the guarded children of this / the last run
    logs = _open_logs(ctx)
    if not logs:
        o = RP.fresh_dir(ctx.out / "g5" / "probe")
        argv = RP.habitus_argv("questions", "render", "--ds", "ATLASV2_EDR", "--level", "E2", "--k", "all",
                               "--day", "2022-07-19", "--answers", "none", "--out", o / "probe.jsonl")
        RP.run_children([("probe", argv, ctx.env(open_log=o / "open_probe.log"), o / "logs" / "probe.log")], 1)
        logs = _open_logs(ctx)
    gt_roots = [os.path.abspath(p) for p in (ctx.st.paths.ground_truth, ctx.st.paths.upstream_gt) if p]
    opened = []
    for lg in logs:
        for line in lg.read_text(encoding="utf-8").splitlines():
            if line and any(line == r or line.startswith(r.rstrip(os.sep) + os.sep) for r in gt_roots):
                opened.append(f"{lg.name}: {line}")
    checks.append(Check("G5c no GT open in guarded children", bool(logs) and not opened,
                        f"{len(logs)} open logs checked" if not opened else "; ".join(opened[:5])))
    # deliberate GT open and eval import under the guard (a fake GT tree: no real label file is read)
    fake = ctx.out / "g5" / "fake_gt"
    (fake / "E3-CADETS").mkdir(parents=True, exist_ok=True)
    (fake / "E3-CADETS" / "node_probe.csv").write_text("x\n", encoding="utf-8")
    log = ctx.out / "g5" / "logs" / "guard.log"
    env = ctx.env(extra={"HABITUS_GT": str(fake)})
    RP.run_children([("guard", [sys.executable, "-c", GUARD_SCRIPT, fake / "E3-CADETS" / "node_probe.csv"], env, log)], 1)
    got = _result_line(log) or {}
    checks.append(Check("G5c guard blocks a GT open", got.get("gt_open") == "blocked", "", got.get("gt_open"), "blocked"))
    checks.append(Check("G5c guard blocks habitus.eval", got.get("eval_import") == "blocked", "", got.get("eval_import"),
                        "blocked"))
    checks += _pytest(ctx, (G5_HABIT_TEST,), "G5c habit channel under the guard", guard=True)
    # (d) allow-list
    try:
        H.spec("NOT_A_HOST")
        refused = False
    except Exception:
        refused = True
    checks.append(Check("G5d hosts.spec refuses other names", refused))
    files = tracked + [ctx.release / f for f in RELEASE_TEXT if (ctx.release / f).exists()]
    hits = SC.scan_hosts(files, root=ctx.home)
    checks.append(Check("G5d out-of-scope host names", not hits, "; ".join(map(str, hits[:10])) or
                        f"{len(files)} files clean"))
    return checks


# --------------------------------------------------------------------------- #
# G6
# --------------------------------------------------------------------------- #
def g6(ctx):
    """The unit tests with HABITUS_REQUIRE_ASSETS=1 (a missing bundle asset fails). Tests marked
    local_assets (Laya weights, role-table catalogs: not in the bundle) are required as well only on the
    build server (paths.pidsmaker_src configured); elsewhere they skip without those files."""
    server = bool(ctx.st.paths.pidsmaker_src)
    env = {"HABITUS_REQUIRE_ASSETS": "1", "HABITUS_REQUIRE_LOCAL_ASSETS": "1" if server else "0"}
    name = ("G6 unit tests (HABITUS_REQUIRE_ASSETS=1" + (", HABITUS_REQUIRE_LOCAL_ASSETS=1" if server else "")
            + ", not provenance)")
    return preflight(ctx, "G6") or _pytest(ctx, ("tests/",), name, env_extra=env, markers="not provenance")


# --------------------------------------------------------------------------- #
# G7 (optional, GPU)
# --------------------------------------------------------------------------- #
def g7(ctx):
    from habitus.dm.gpu import check_device
    dev = f"cuda:{int(ctx.gpu)}"
    check_device(dev, ctx.st)                    # raises on a forbidden / unresolvable GPU
    out = RP.fresh_dir(ctx.out / "g7")
    specs = []
    for dm in (MAIN_DM, "laya-typed-zs"):           # liveness: a 64-question subset (another batch makeup)
        argv = RP.habitus_argv("dm", "smoke", "--dm", dm, "--gpu", ctx.gpu, "--n", 64, "--out", out / f"{dm}.json")
        specs.append((dm, argv, ctx.env(gpu=True), out / "logs" / f"{dm}.log"))
    res = RP.run_children(specs, 1, timeout=600)
    # exactness: the whole CLEARSCOPE_E3 E2 k=all 2018-04-11 job, original batch makeup (~8.7k prompts)
    name = f"{MAIN_DM} full-day"
    argv = RP.habitus_argv("dm", "smoke", "--dm", MAIN_DM, "--gpu", ctx.gpu, "--full-day",
                           "--out", out / f"{MAIN_DM}.full_day.json")
    res.update(RP.run_children([(name, argv, ctx.env(gpu=True), out / "logs" / f"{MAIN_DM}.full_day.log")], 1,
                               timeout=3600))
    checks = []
    for dm, r in res.items():
        p = out / (f"{MAIN_DM}.full_day.json" if dm == name else f"{dm}.json")
        got = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        checks.append(Check(f"G7 {dm}", r["rc"] == 0 and got.get("ok") is True,
                            json.dumps({k: v for k, v in got.items() if k != "decide_calls"})[:400]))
    return checks


# --------------------------------------------------------------------------- #
# G8
# --------------------------------------------------------------------------- #
def g8(ctx):
    import subprocess
    tracked = SC.tracked_files(ctx.home)
    rel_text = [ctx.release / f for f in RELEASE_TEXT if (ctx.release / f).exists()]
    files = tracked + rel_text
    checks = []
    for name, fn in (("paths", SC.scan_paths), ("secrets", SC.scan_secrets), ("e-mail", SC.scan_emails)):
        hits = fn(files, root=ctx.home)
        checks.append(Check(f"G8 {name}", not hits, "; ".join(map(str, hits[:10])) or
                            f"{len(tracked)} tracked + {len(rel_text)} release text files clean"))
    stats = [str(p) for ds in RP.hosts() for p in [ctx.release / "cards" / ds / "stats.json"] if p.exists()]
    checks.append(Check("G8 no stats.json in release cards", not stats, "; ".join(stats)))
    lic = ctx.home / "LICENSE"
    notice = ctx.home / "NOTICE"
    checks.append(Check("G8 LICENSE (Apache-2.0)", lic.exists() and "Apache License" in lic.read_text(encoding="utf-8")))
    checks.append(Check("G8 NOTICE + Laya change list", notice.exists() and
                        LAYA_CHANGES in notice.read_text(encoding="utf-8")))
    ign = subprocess.run(["git", "-C", str(ctx.home), "check-ignore", "-q", "configs/local.yaml"]).returncode == 0
    checks.append(Check("G8 configs/local.yaml is gitignored", ign))
    return checks


# --------------------------------------------------------------------------- #
# G9
# --------------------------------------------------------------------------- #
def g9(ctx):
    return preflight(ctx, "G9") or RP.operating(ctx.out / "g9", st=ctx.st)


# --------------------------------------------------------------------------- #
# G10
# --------------------------------------------------------------------------- #
def verify_sums(path, base):
    """Problems of a sha256sum-style file ('<sha256>  <relative path>' per line) under `base`."""
    from habitus.io import file_sha256
    bad, n = [], 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            bad.append(f"malformed: {line[:80]}")
            continue
        sha, rel = parts[0], parts[1].strip().lstrip("*").split("  ")[0].strip()
        n += 1
        p = Path(base) / rel
        if not p.exists():
            bad.append(f"missing {rel}")
        elif file_sha256(p) != sha:
            bad.append(f"sha256 differs: {rel}")
    return n, bad


def status_blocks(text):
    """The generated STATUS MATRIX blocks of a file (text between the markers, outer newlines stripped)."""
    out, i = [], 0
    while True:
        a = text.find(STATUS_BEGIN, i)
        if a < 0:
            return out
        b = text.find(STATUS_END, a)
        if b < 0:
            return out
        out.append(text[a + len(STATUS_BEGIN):b].strip("\n"))
        i = b + len(STATUS_END)


def readme_table(text):
    """[(files, status)] of the results/README paper-number table (columns 'File' and 'Status')."""
    rows, cols = [], None
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            cols = None if not line.strip() else cols
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        low = [c.lower() for c in cells]
        if "file" in low and "status" in low:
            cols = (low.index("file"), low.index("status"))
            continue
        if cols is None or set(line.replace("|", "").strip()) <= set("-: "):
            continue
        if max(cols) < len(cells):
            rows.append((cells[cols[0]], cells[cols[1]]))
    return rows


def _paths_of(cell):
    """results/-relative file names of a README 'File' cell ('`a.csv`, `b.json`', 'a.csv + b.csv',
    'a.csv (E2 rows)'): the first word of each comma / plus separated part that looks like a path."""
    names = re.findall(r"`([^`]+)`", cell) or re.split(r"[,+;]", cell)
    out = []
    for n in names:
        words = n.strip().split()
        if not words:
            continue
        w = words[0]
        if w in ("-", "–") or ("." not in w and "/" not in w):
            continue
        out.append(w)
    return out


def _exists(base_dirs, rel):
    """`rel` (a file, a directory or a glob) exists under one of `base_dirs`."""
    for b in base_dirs:
        if any(ch in rel for ch in "*?["):
            if list(Path(b).glob(rel)):
                return True
        elif (Path(b) / rel).exists():
            return True
    return False


def g10(ctx):
    checks = []
    res_dir = ctx.home / "results"
    man = res_dir / "MANIFEST.sha256"
    if man.exists():
        n, bad = verify_sums(man, res_dir)
        if bad and all(b.startswith("missing") for b in bad):
            n, bad = verify_sums(man, ctx.home)
        checks.append(Check("G10 results/MANIFEST.sha256", n > 0 and not bad, f"{n} files" if not bad else "; ".join(bad[:5])))
    else:
        checks.append(Check("G10 results/MANIFEST.sha256", False, "missing"))
    sums = ctx.release / "SHA256SUMS"
    if sums.exists():
        n, bad = verify_sums(sums, ctx.release)
        checks.append(Check("G10 release SHA256SUMS", n > 0 and not bad, f"{n} files" if not bad else "; ".join(bad[:5])))
    else:
        checks.append(Check("G10 release SHA256SUMS", False, "missing (scripts/make_release.py)"))
    rd = res_dir / "README.md"
    if rd.exists():
        rows = readme_table(rd.read_text(encoding="utf-8"))
        missing, unproven = [], []
        for files, status in rows:
            for f in _paths_of(files):
                if not _exists((res_dir, res_dir / "habitus", ctx.home), f):
                    missing.append(f)
            if status.upper().startswith("REPRODUCED"):
                gates = re.findall(r"G\d+", status)
                if not gates or any(ctx.passed.get(g) != "PASS" for g in gates):
                    unproven.append(f"{files} [{status}]")
        checks.append(Check("G10 results/README files exist", bool(rows) and not missing,
                            f"{len(rows)} rows" if not missing else f"missing {missing[:5]}"))
        checks.append(Check("G10 REPRODUCED rows name passing gates", bool(rows) and not unproven,
                            "" if not unproven else f"{len(unproven)} rows: " + "; ".join(unproven[:3])))
    else:
        checks.append(Check("G10 results/README", False, "missing"))
    try:
        from habitus.dm.status import render_markdown
        want = render_markdown().strip("\n")
        for f in STATUS_FILES:
            p = ctx.home / f
            blocks = status_blocks(p.read_text(encoding="utf-8")) if p.exists() else []
            checks.append(Check(f"G10 STATUS block {f}", len(blocks) == 1 and blocks[0] == want,
                                "equal to habitus.dm.status.render_markdown()" if blocks == [want] else
                                f"{len(blocks)} blocks, differ from render_markdown()"))
    except ImportError as e:
        checks.append(Check("G10 STATUS blocks", False, f"habitus.dm.status: {e}"))
    checks += provenance_checks(ctx.home, server=bool(ctx.st.paths.pidsmaker_src))
    checks += _pytest(ctx, MIRROR_TESTS, "G10 config mirror tests")
    return checks


def provenance_checks(home, server=True):
    """docs/PROVENANCE.md lists every snapshot source with a sha256; runs/salvage matches SALVAGE.json.

    runs/salvage is a gitignored, server-only record (scripts/dev/salvage.py): without it and with
    server=False (paths.pidsmaker_src unset: not the build server) its check is reported as skipped."""
    import importlib.util
    home = Path(home)
    checks = []
    spec = importlib.util.spec_from_file_location("snapshot_sources", home / "scripts" / "dev" / "snapshot_sources.py")
    snap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(snap)
    prov = home / "docs" / "PROVENANCE.md"
    table = snap.parse(prov.read_text(encoding="utf-8")) if prov.exists() else {}
    bad = [s for s in snap.SOURCES if not re.fullmatch(r"[0-9a-f]{64}", (table.get(s) or ("",))[0])]
    checks.append(Check("G10 docs/PROVENANCE.md", bool(table) and not bad,
                        f"{len(table)} sources with sha256" if not bad else f"{len(bad)} without: {bad[:3]}"))
    rec_p = home / "runs" / "salvage" / "SALVAGE.json"
    if rec_p.exists():
        from habitus.io import file_sha256
        rec = json.loads(rec_p.read_text(encoding="utf-8"))
        bad = [r for r, v in rec["files"].items() if not (rec_p.parent / r).is_file()
               or file_sha256(rec_p.parent / r) != v["sha256"]]
        checks.append(Check("G10 runs/salvage", bool(rec["files"]) and not bad,
                            f"{len(rec['files'])} files match SALVAGE.json" if not bad else f"differ: {bad}"))
    elif not server:
        checks.append(Check("G10 runs/salvage", True, "skipped: server-only record (paths.pidsmaker_src is unset, "
                            "so this is not the build server)"))
    else:
        checks.append(Check("G10 runs/salvage", False, "no SALVAGE.json (scripts/dev/salvage.py)"))
    return checks


# --------------------------------------------------------------------------- #
# G11 (HB0)
# --------------------------------------------------------------------------- #
HB0_ROWS = "results/extension_habit/hb0_rows.csv"
HB0_INTS = ("tp", "fp", "tn", "fn", "tie_n", "tie_gt")
# decimals of the frozen float columns (confusion_reference.md formatting; trailing zeros dropped)
HB0_DP = {"precision": 3, "recall": 3, "f1": 3, "mcc": 3, "mcc_recomputed": 6, "threshold": 4,
          "mcc_ties_neg": 3, "ap": 3, "auc": 3}
HB0_CATALOG_FILE = "Contents-amd64.gz"          # vocab._RoleTable's presence test


def hb0_expected(home):
    """{(ds, kplan): row} of the frozen HB0 rows."""
    import csv
    with open(Path(home) / HB0_ROWS, newline="", encoding="utf-8") as fh:
        return {(r["ds"], r["k_plan"]): r for r in csv.DictReader(fh)}


def compare_hb0(got, want):
    """Checks of `habitus eval habit` rows (eval_habit_row dicts) against the frozen hb0_rows.csv rows:
    integers exact, floats equal after rounding to the frozen column's decimals."""
    checks = []
    by = {(r["ds"], r["kplan"]): r for r in got}
    for key, w in sorted(want.items()):
        g = by.get(key)
        name = f"G11 {key[0]} {key[1]}"
        if g is None or g.get("error"):
            checks.append(Check(name, False, f"no recomputed row ({(g or {}).get('error', 'missing')})"))
            continue
        bad = []
        for c in HB0_INTS:
            if g.get(c) is None or int(g[c]) != int(w[c]):
                bad.append(f"{c} {g.get(c)} != {w[c]}")
        for c, dp in HB0_DP.items():
            gv = g.get("mcc" if c == "mcc_recomputed" else c)
            if gv is None or round(float(gv), dp) != float(w[c]):
                bad.append(f"{c} {gv} != {w[c]} ({dp} dp)")
        checks.append(Check(name, not bad, "; ".join(bad) if bad else
                            f"{w['tp']}/{w['fp']}/{w['tn']}/{w['fn']}, mcc {w['mcc_recomputed']}, thr {w['threshold']}, "
                            f"ap {w['ap']}, auc {w['auc']}, ties {w['tie_n']} ({w['tie_gt']}): every column equal"))
    extra = sorted(set(by) - set(want))
    if extra:
        checks.append(Check("G11 no extra rows", False, f"rows without a frozen counterpart: {extra}"))
    return checks


def g11(ctx):
    checks = preflight(ctx, "G11")
    if checks:
        return checks
    cat = ctx.st.paths.catalogs
    if not cat or not os.path.isfile(os.path.join(cat, HB0_CATALOG_FILE)):
        raise Skip(f"role-table catalogs missing under paths.catalogs ({'unset' if not cat else 'no ' + HB0_CATALOG_FILE}):"
                   " run scripts/fetch_catalogs.sh (HB0 needs the frozen ROLE_TABLE_SHA256 role table)")
    want = hb0_expected(ctx.home)
    plans = {}
    for ds, kp in want:
        plans.setdefault(ds, []).append(kp)
    out = RP.fresh_dir(ctx.out / "g11")
    root = out / "habit"
    specs = []
    for ds in RP.hosts():
        if ds not in plans:
            continue
        argv = RP.habitus_argv("decide", "hb0", "--ds", ds, "--kplans", ",".join(plans[ds]), "--root", root)
        specs.append((ds, argv, ctx.env(open_log=out / f"open_hb0_{ds}.log"), out / "logs" / f"hb0_{ds}.log"))
    res = RP.run_children(specs, ctx.jobs)
    for ds, r in res.items():
        checks.append(Check(f"G11 decide hb0 {ds}", r["rc"] == 0, f"{r['seconds']}s" + ("" if r["rc"] == 0 else
                            f"; log {r['log']}\n{RP.log_tail(r['log'])}")))
    ok_ds = [ds for ds, r in res.items() if r["rc"] == 0]
    specs = []
    for ds in ok_ds:                                # eval: the label reader (exempt from the guard)
        argv = RP.habitus_argv("eval", "habit", "--ds", ds, "--levels", "HB0", "--kplans", ",".join(plans[ds]),
                               "--root", root, "--out", out / f"rows_{ds}.json")
        specs.append((ds, argv, ctx.env(), out / "logs" / f"eval_{ds}.log"))
    res = RP.run_children(specs, ctx.jobs)
    got = []
    for ds, r in res.items():
        p = out / f"rows_{ds}.json"
        if r["rc"] != 0 or not p.exists():
            checks.append(Check(f"G11 eval habit {ds}", False, f"log {r['log']}\n{RP.log_tail(r['log'])}"))
            continue
        got += json.loads(p.read_text(encoding="utf-8"))
    checks += compare_hb0(got, {k: v for k, v in want.items() if k[0] in ok_ds} if ok_ds else want)
    return checks


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
FUNCS = {"G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5, "G6": g6, "G7": g7, "G8": g8, "G9": g9, "G10": g10,
         "G11": g11}


def _load_report(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"gates": {}}


# run order: GATES order with G10 last (its REPRODUCED-row check reads the status of the gates the
# results README names, G11 included, from this run)
RUN_ORDER = tuple(g for g in GATES if g != "G10") + ("G10",)


def select(only=None, skip=(), gpu=None):
    """The gates of a run: --only or all, in RUN_ORDER (G10 last); G7 only with --only G7 and --gpu."""
    want = [g for g in RUN_ORDER if (only is None or g in only) and g not in set(skip or ())]
    bad = sorted(set(only or ()) - set(GATES))
    if bad:
        raise SystemExit(f"--only: unknown gates {bad}; known {GATES}")
    skipped = {}
    if "G7" in want and not (only == ["G7"] and gpu is not None):
        want.remove("G7")
        skipped["G7"] = "optional GPU smoke: run alone with --only G7 --gpu N"
    return want, skipped


def run(only=None, skip=(), answers="release", gpu=None, jobs=RP.MAX_JOBS, out=None):
    st = RP.settings()
    out = Path(out or os.path.join(st.paths.runs, "gates"))
    out.mkdir(parents=True, exist_ok=True)
    want, skipped = select(only, skip, gpu)
    prev = _load_report(out / "report.json")
    ctx = Ctx(out, jobs, answers, gpu, only)
    ctx.passed = {g: v.get("status") for g, v in prev.get("gates", {}).items()}
    if answers == "unpruned" and "G2" in want:
        want.insert(want.index("G2") + 1, "G2u")
    rep = Report("habitus gates")
    gates = dict(prev.get("gates", {}))
    for g in want:
        t0 = time.time()
        why = None
        try:
            if g in ("G2", "G2u"):              # G2 is always the release answers; G2u the unpruned ones
                checks = g2(ctx, "unpruned" if g == "G2u" else "release")
            else:
                checks = FUNCS[g](ctx)
        except Skip as e:                       # not applicable here (missing optional inputs): the reason
            why = str(e)
            checks = [Check(f"{g} skipped", True, why)]
        except Exception as e:                  # a gate that cannot run fails, with the reason
            import traceback
            checks = [Check(f"{g} error", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")]
        status = "SKIP" if why is not None else "PASS" if checks and all(c.ok for c in checks) else "FAIL"
        ctx.passed[g] = status
        gates[g] = {"status": status, "seconds": round(time.time() - t0, 1), "when": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "checks": [c.as_dict() for c in checks]}
        if why is not None:
            gates[g]["why"] = why
        rep.extend(checks)
        rep.add(f"{g}", status != "FAIL", f"{status} in {gates[g]['seconds']}s" + (f": {why}" if why else ""))
    for g, why in skipped.items():
        gates.setdefault(g, {"status": "SKIP", "why": why})
    for g in want:                              # the config view each gate of this run read its inputs through
        gates[g]["config"] = os.path.basename(os.environ.get("HABITUS_CONFIG") or "") or None
    report = {"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "gates": dict(sorted(gates.items(),
                                                                                   key=lambda kv: _gate_order(kv[0])))}
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    (out / "report.md").write_text(render_md(report), encoding="utf-8")
    return rep


def _gate_order(g):
    return (GATES.index(g) if g in GATES else GATES.index("G2") + 0.5, g)


def render_md(report):
    lines = ["# HABITUS gates", "", f"Updated {report['updated']}.", "", "| gate | status | seconds | failing checks |",
             "|---|---|---|---|"]
    for g, v in report["gates"].items():
        fails = [c["name"] for c in v.get("checks", []) if not c.get("ok")]
        lines.append(f"| {g} | {v.get('status')} | {v.get('seconds', '-')} | "
                     f"{'; '.join(fails[:5]) if fails else (v.get('why') or '')} |")
    lines.append("")
    for g, v in report["gates"].items():
        if not v.get("checks"):
            continue
        lines += [f"## {g} ({v['status']})", ""]
        for c in v["checks"]:
            d = str(c.get("detail") or "").replace("\n", " ")[:300]
            lines.append(f"- {c['status']} {c['name']}" + (f": {d}" if d else ""))
        lines.append("")
    return "\n".join(lines)
