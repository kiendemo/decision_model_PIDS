"""Operating tables of the main configuration (CPU only): budget rows at the clean-day bars, lambda /
combiner sensitivity, per-attack recall, and the per-form ablation replayed from the answer cache.

    from habitus.eval import operating
    b = operating.budget("ATLASV2_EDR", root=scores_root)          # -> budget_<DS>.json content
    s = operating.forms("ATLASV2_EDR", root=scores_root, out_dir=runs_operating)
        # guarded subprocess replay -> forms_<DS>.json.gz (label-free), then confusion here
        # -> state_<DS>.json content; writes vec_<DS>.npz (form1, form2) for the bootstrap
    operating.write_csvs({ds: b}, {ds: s}, out_dir)                # budget_rows.csv, per_attack_recall.csv,
                                                                   # sensitivity_composition.csv,
                                                                   # sensitivity_forms.csv, form_gaps.csv

budget: single-tier bars tau_B (B = 0, 1, 20 alerts per clean day) from the LODO clean-day null of the
same E2 composite (decide.null.lodo_scores; metrics.budget_rows); a node alerts if s(v) >= tau_B. The
lambda / combiner grid (reported, never used for a choice): composite = combiner(qv, qt) *
(1 - lambda * qa) outside the hard tier; null-calibrated exceedance c_f(x) = 1 - (1 + #{null head
node-days of f >= x}) / (1 + N_f) over the same null.

forms: the E2 states of every head signature of the main configuration's test days are rebuilt
(state.builder.StateBuilderE2 over the score file's own reference and caps) and every form is answered by
get_dm('jev-qwen3.5-4b', cache_only=True) - a `python -c` child that imports no label code and runs under
the label guard plus a write audit limited to paths.runs and the output directory. Each
(question, form) value is combine() over that single form; 'both' recombines the two forms of every
question and must equal the score file (recon_max_abs 0.0, cache_misses 0). Form keys: form1 = QV fwd,
QT pos, QA pos; form2 = QV rev, QT neg, QA neg; qv_fwd / qv_rev / qt_pos / qt_neg = that single form (no
max, no acquittal); qa_pos-acq / qa_neg-acq = HABITUS with QA read in one polarity only.

Every main-configuration oracle MCC is asserted equal to the published value (|diff| < 6e-5) first.
CSV files converted from these dicts write floats with repr() (csv module default).
"""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HOSTS = ("CADETS_E3", "THEIA_E3", "CLEARSCOPE_E3", "optc_h501", "ATLASV2_EDR")
KN = {0: "k0", 2: "k2", None: "kall"}
REF_MCC = {("CADETS_E3", 2): 0.4055, ("THEIA_E3", 2): 0.5572, ("CLEARSCOPE_E3", 2): 0.0855, ("optc_h501", 2): 0.4192,
           ("ATLASV2_EDR", 2): 0.4048, ("CADETS_E3", None): 0.3922, ("THEIA_E3", None): 0.5956,
           ("CLEARSCOPE_E3", None): 0.1121, ("optc_h501", None): 0.3929, ("ATLASV2_EDR", None): 0.4052}
BUDGETS = (0, 1, 20)
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
FORM_KEYS = ("both", "form1", "form2", "qv_fwd", "qv_rev", "qt_pos", "qt_neg", "qa_pos-acq", "qa_neg-acq")
VEC_KEYS = ("form1", "form2")
MAIN_DM = "jev-qwen3.5-4b"
ROW_KEYS = ("tp", "fp", "tn", "fn", "precision", "recall", "mcc", "ap", "auc", "threshold", "alerts")


def _log(msg):
    try:
        from habitus.log import log
        log(msg, "eval.operating")
    except ImportError:
        print(f"[eval.operating] {msg}", file=sys.stderr, flush=True)


def _settings(settings=None):
    if settings is not None:
        return settings
    from habitus.config import settings as _s
    return _s()


def _scores_root(root, settings=None):
    return str(root) if root is not None else str(_settings(settings).paths.scores)


def runs_dir(settings=None) -> Path:
    return Path(str(_settings(settings).paths.runs)) / "operating"


def _host(ds, host=None, source="nodeindex", settings=None):
    from habitus.eval.tables import make_host
    from habitus.hosts import assert_allowed
    assert_allowed(ds)
    return host if host is not None else make_host(ds, source, settings)


def _row(r):
    tn = r.get("ties_negative") or {}
    return {k: r.get(k) for k in ROW_KEYS} | {"mcc_ties_neg": tn.get("mcc")}


def _transform(by_day, fn):
    return {d: {s: float(fn(r)) for s, r in recs.items()} for d, recs in by_day.items()}


def _composite(lam, base_fn):
    def f(r):
        b = base_fn(r)
        qa = r.get("qa")
        if r.get("hard") or qa is None:
            return b
        return b * (1.0 - lam * float(qa))
    return f


def _by_attack(host, groups, alert_mask):
    res = {}
    for g, ids in sorted(groups.items()):
        ids = np.fromiter(ids, dtype=np.int64, count=len(ids))
        pos = host.upos(ids)
        inu = pos >= 0
        hit = int(alert_mask[pos[inu]].sum()) if inu.any() else 0
        res[g] = {"gt_nodes": int(len(ids)), "in_universe": int(inu.sum()), "alerted": hit}
    return res


# --------------------------------------------------------------------------- #
# Budget rows, per-attack recall, lambda / combiner sensitivity
# --------------------------------------------------------------------------- #
def budget(ds, *, root=None, host=None, source="nodeindex", settings=None):
    """Budget rows at the clean-day bars, lambda / combiner sensitivity, per-attack recall (k2 and kall)."""
    from habitus.decide import null as N
    from habitus.decide import scorefile as S
    from habitus.eval.labels import attack_groups, host_labels
    from habitus.eval.metrics import budget_rows, confusion
    from habitus.eval.tables import main_k
    t0 = time.time()
    root = _scores_root(root, settings)
    host = _host(ds, host, source, settings)
    y, n_total = host_labels(host)
    groups = attack_groups(host)
    mk = main_k(ds)
    out = {"ds": ds, "test_days": list(host.test), "n_total_gt": int(n_total), "plans": {}}
    for k in (2, None):
        metas = {}
        by_day, missing = S.load_by_day(ds, "E2", k, host.test, root, metas)
        assert not missing, (ds, k, missing)
        s = S.node_scores(host, by_day, "composite")
        conf = confusion(host, s)
        assert abs(conf["mcc"] - REF_MCC[(ds, k)]) < 6e-5, (ds, k, conf["mcc"])
        plan = {"main": mk == k, "oracle": _row(conf)}
        # ---- clean-day null and budget rows (single-tier bars) ----
        nl = N.lodo_scores(host, k, "E2", "composite", root=root)
        if nl is None:
            plan["budget"] = None
        else:
            test_caps = [metas[d].get("caps") for d in host.test]
            same_caps = all(c == test_caps[0] for c in test_caps + nl["caps"])
            rows = budget_rows(s, y, nl["values"], nl["n_days"], len(host.test), budgets=BUDGETS,
                               n_total=n_total, day_index={d: host.day_index(d) for d in host.test})
            plan["budget"] = {"null_days": nl["days"], "null_units": int(len(nl["values"])), "n_null_days": nl["n_days"],
                              "same_caps": bool(same_caps), "rows": rows}
            if groups:
                plan["budget_by_attack"] = {}
                for b, r in rows.items():
                    t = r["bar"]
                    al = s >= (t if t is not None else -np.inf)
                    plan["budget_by_attack"][b] = _by_attack(host, groups, al)
        # ---- per-attack recall at the oracle threshold ----
        if groups and conf.get("threshold") is not None:
            plan["oracle_by_attack"] = _by_attack(host, groups, s >= conf["threshold"])
        # ---- lambda and combiner sensitivity (not used for any choice) ----
        sens = {}

        def mx(r):
            return max(float(r["qv"]), float(r["qt"]))

        def mean(r):
            return 0.5 * (float(r["qv"]) + float(r["qt"]))
        for lam in LAMBDAS:
            v = S.node_scores(host, _transform(by_day, _composite(lam, mx)))
            sens[f"max, lambda={lam:g}"] = _row(confusion(host, v))
        v = S.node_scores(host, _transform(by_day, _composite(0.5, mean)))
        sens["mean(qv,qt), lambda=0.5"] = _row(confusion(host, v))
        cal = {}
        for f in ("qv", "qt"):
            nf = N.lodo_scores(host, k, "E2", f, root=root)
            if nf is None:
                cal = None
                break
            u = np.sort(np.asarray(nf["values"], dtype=np.float64))
            u = u[u >= 0.0]
            cal[f] = u
        if cal is not None:
            def cfun(f, x):
                u = cal[f]
                ge = len(u) - np.searchsorted(u, x, side="left")
                return 1.0 - (1.0 + ge) / (1.0 + len(u))

            def calmax(r):
                return max(cfun("qv", float(r["qv"])), cfun("qt", float(r["qt"])))
            v = S.node_scores(host, _transform(by_day, _composite(0.5, calmax)))
            sens["max of null-calibrated exceedance (qv, qt), lambda=0.5"] = _row(confusion(host, v))
            v = S.node_scores(host, _transform(by_day, calmax))
            sens["max of null-calibrated exceedance (qv, qt), no acquittal"] = _row(confusion(host, v))
        plan["sensitivity"] = sens
        out["plans"][KN[k]] = plan
        _log(f"[{ds}] {KN[k]} done ({time.time() - t0:.0f}s)")
    out["groups"] = {g: len(v) for g, v in (groups or {}).items()}
    out["seconds"] = round(time.time() - t0, 1)
    return out


# --------------------------------------------------------------------------- #
# Per-form ablation: guarded replay child (label-free) + confusion here
# --------------------------------------------------------------------------- #
_FORMS_CHILD = r'''
import gzip, json, os, sys, time
args = json.loads(sys.argv[1])
from habitus.guard import install_guard
assert install_guard("questions", write_roots=args["write_roots"]), "label guard not active"
import numpy as np
from habitus.data.host import Host
from habitus.decide import scorefile as S
from habitus.dm import get_dm
from habitus.dm.base import CacheMiss
from habitus.questions.routed import SYSTEM_E2, combine, to_typed, variants
from habitus.state.builder import StateBuilderE2, hard_tier, serialize
from habitus.state.select import flat, head

ds, k, root = args["ds"], args["k"], args["root"]
t0 = time.time()
host = Host(ds)
dm = get_dm(args["dm"], ds=ds, cache_only=True)
forms = {n: {} for n in args["form_keys"]}
gaps = {"qv": [], "qt": [], "qa": []}
worst, miss = 0.0, 0
res = {"ds": ds, "k": args["kname"], "dm": args["dm"], "tag": getattr(dm, "tag", None)}
try:
    for d in host.test:
        sf = S.read_score(S.score_path(ds, "E2", k, d, root))
        ref, caps = sf["meta"]["ref"], sf["meta"]["caps"]
        sigs = flat(head(host, d, ref or None, caps))
        assert set(sigs) == set(sf["scores"]), (ds, d)
        st = StateBuilderE2(host, ref, k).build(d, sigs)
        typed = {s: {q: to_typed(s, q, serialize(x), system=SYSTEM_E2) for q, x in st[s].items()} for s in sigs}
        try:
            dec = dm.decide([tq for per in typed.values() for tq in per.values()])
        except CacheMiss as e:
            miss += len(getattr(e, "keys", None) or [1])
            raise
        for key in forms:
            forms[key][d] = {}
        for s in sigs:
            qs = st[s]
            hard = hard_tier(qs["qv"])
            pf = {}
            for q in qs:
                D = dec[typed[s][q].qkey]
                for v in variants(q):
                    pf[(q, v)] = combine(q, {v: D.letter_probs[v][0]})
            rec = sf["scores"][s]
            qv = 0.5 * (pf[("qv", "fwd")] + pf[("qv", "rev")])
            qt = 0.5 * (pf[("qt", "pos")] + pf[("qt", "neg")])
            qa = None if "qa" not in qs else 0.5 * (pf[("qa", "pos")] + pf[("qa", "neg")])
            worst = max(worst, abs(qv - rec["qv"]), abs(qt - rec["qt"]),
                        0.0 if qa is None else abs(qa - rec["qa"]))
            gaps["qv"].append(abs(pf[("qv", "fwd")] - pf[("qv", "rev")]))
            gaps["qt"].append(abs(pf[("qt", "pos")] - pf[("qt", "neg")]))
            if qa is not None:
                gaps["qa"].append(abs(pf[("qa", "pos")] - pf[("qa", "neg")]))
            hd = bool(rec.get("hard"))
            assert hd == hard, (ds, d, s)

            def comp(v_, t_, a_):
                b = max(v_, t_)
                return b if (hd or a_ is None) else b * (1.0 - 0.5 * a_)
            forms["both"][d][s] = comp(qv, qt, qa)
            forms["form1"][d][s] = comp(pf[("qv", "fwd")], pf[("qt", "pos")], None if qa is None else pf[("qa", "pos")])
            forms["form2"][d][s] = comp(pf[("qv", "rev")], pf[("qt", "neg")], None if qa is None else pf[("qa", "neg")])
            forms["qv_fwd"][d][s] = pf[("qv", "fwd")]
            forms["qv_rev"][d][s] = pf[("qv", "rev")]
            forms["qt_pos"][d][s] = pf[("qt", "pos")]
            forms["qt_neg"][d][s] = pf[("qt", "neg")]
            forms["qa_pos-acq"][d][s] = comp(qv, qt, None if qa is None else pf[("qa", "pos")])
            forms["qa_neg-acq"][d][s] = comp(qv, qt, None if qa is None else pf[("qa", "neg")])
        print(f"[{ds}] {d}: {len(sigs)} sigs ({time.time() - t0:.0f}s), worst recon {worst:.2e}, miss {miss}",
              file=sys.stderr, flush=True)
    res["forms"] = forms
    status = 0
except CacheMiss:
    status = 3
res.update({"recon_max_abs": worst, "cache_misses": miss,
            "gaps": {q: {"n": len(g), "mean": float(np.mean(g)) if g else None,
                         "p50": float(np.median(g)) if g else None,
                         "p99": float(np.quantile(g, 0.99)) if g else None} for q, g in gaps.items()},
            "seconds": round(time.time() - t0, 1)})
out = args["out"]
tmp = out + ".tmp"
with gzip.GzipFile(tmp, "wb", mtime=0) as fh:
    fh.write(json.dumps(res).encode("utf-8"))
os.replace(tmp, out)
dm.close()
sys.exit(status)
'''


def _child_env():
    import habitus
    src = str(Path(habitus.__file__).resolve().parent.parent)
    env = dict(os.environ)
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.setdefault(v, "8")
    return env


def forms_replay(ds, *, root=None, out_dir=None, k=None, dm=MAIN_DM, settings=None, python=None):
    """Run the guarded, label-free forms replay child; returns its output path (forms_<DS>.json.gz).
    Raises RuntimeError when the child fails or reports cache misses."""
    from habitus.eval.tables import main_k
    from habitus.hosts import assert_allowed
    assert_allowed(ds)
    k = main_k(ds) if k is None else k
    out_dir = Path(out_dir) if out_dir is not None else runs_dir(settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"forms_{ds}.json.gz"
    runs = Path(str(_settings(settings).paths.runs)).resolve()
    tmp = runs / "tmp"                                 # the child's TMPDIR: inside its write roots
    tmp.mkdir(parents=True, exist_ok=True)
    args = {"ds": ds, "k": k, "kname": KN[k], "root": _scores_root(root, settings), "dm": dm,
            "form_keys": list(FORM_KEYS), "out": str(out),
            "write_roots": sorted({str(runs), str(out_dir.resolve()), os.devnull})}
    env = _child_env()
    env["TMPDIR"] = str(tmp)
    t0 = time.time()
    p = subprocess.run([python or sys.executable, "-c", _FORMS_CHILD, json.dumps(args)], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        tail = "\n".join((p.stderr or "").splitlines()[-25:])
        raise RuntimeError(f"{ds}: forms replay child exited {p.returncode} after {time.time() - t0:.0f}s:\n{tail}")
    return out


def read_forms(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def forms(ds, *, root=None, out_dir=None, host=None, source="nodeindex", settings=None, replay=None, python=None):
    """Per-form ablation rows of the main configuration (the state_<DS>.json content without rules).
    Writes vec_<DS>.npz (form1, form2 node vectors) into out_dir for the bootstrap. `replay`: an
    existing forms_<DS>.json.gz to read instead of running the child."""
    from habitus.decide import scorefile as S
    from habitus.eval.metrics import confusion
    from habitus.eval.tables import main_k
    t0 = time.time()
    out_dir = Path(out_dir) if out_dir is not None else runs_dir(settings)
    k = main_k(ds)
    path = Path(replay) if replay is not None else forms_replay(ds, root=root, out_dir=out_dir, k=k,
                                                                 settings=settings, python=python)
    raw = read_forms(path)
    if raw.get("cache_misses"):
        raise RuntimeError(f"{ds}: forms replay had {raw['cache_misses']} cache misses")
    host = _host(ds, host, source, settings)
    res = {"ds": ds, "k": KN[k], "recon_max_abs": raw["recon_max_abs"], "cache_misses": raw["cache_misses"],
           "forms": {}, "gaps": raw["gaps"]}
    vecs = {}
    for key in FORM_KEYS:
        vecs[key] = S.node_scores(host, raw["forms"][key])
        res["forms"][key] = _row(confusion(host, vecs[key]))
    out_dir.mkdir(parents=True, exist_ok=True)
    vp = out_dir / f"vec_{ds}.npz"
    tmp = out_dir / f"vec_{ds}.tmp.npz"
    np.savez_compressed(tmp, **{k_: vecs[k_] for k_ in VEC_KEYS})
    os.replace(tmp, vp)
    assert abs(res["forms"]["both"]["mcc"] - REF_MCC[(ds, k)]) < 6e-5, res["forms"]["both"]
    res["seconds"] = round(time.time() - t0 + float(raw.get("seconds") or 0.0), 1)
    return res


# --------------------------------------------------------------------------- #
# JSON / CSV outputs
# --------------------------------------------------------------------------- #
def write_json(obj, path):
    """indent=1 JSON (default=float), atomic; the editor's format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)
    return path


BUDGET_COLS = ("ds", "plan", "main", "B", "bar", "alerts", "tp", "fp", "alerts_per_day", "fp_per_day", "recall",
               "alerts_by_day", "n_null_days", "null_days", "null_units", "same_caps",
               "oracle_tp", "oracle_fp", "oracle_mcc", "oracle_threshold")
ATTACK_COLS = ("ds", "plan", "main", "at", "group", "gt_nodes", "in_universe", "alerted")
SENS_COLS = ("ds", "plan", "main", "variant") + ROW_KEYS + ("mcc_ties_neg",)
FORM_COLS = ("ds", "plan", "form") + ROW_KEYS + ("mcc_ties_neg", "recon_max_abs", "cache_misses")
GAP_COLS = ("ds", "plan", "question", "n", "mean", "p50", "p99")


def budget_tables(budgets):
    """{ds: budget dict} -> (budget rows, per-attack rows, sensitivity rows) as dict rows (HOSTS order)."""
    brows, arows, srows = [], [], []
    for ds in [d for d in HOSTS if d in budgets]:
        b = budgets[ds]
        for plan, p in b["plans"].items():
            o = p["oracle"]
            bud = p.get("budget")
            if bud is not None:
                for B, r in bud["rows"].items():
                    brows.append({"ds": ds, "plan": plan, "main": p["main"], "B": B, **{k: r.get(k) for k in (
                        "bar", "alerts", "tp", "fp", "alerts_per_day", "fp_per_day", "recall")},
                        "alerts_by_day": ";".join(f"{d}={n}" for d, n in (r.get("alerts_by_day") or {}).items()),
                        "n_null_days": bud["n_null_days"], "null_days": ";".join(bud["null_days"]),
                        "null_units": bud["null_units"], "same_caps": bud["same_caps"],
                        "oracle_tp": o["tp"], "oracle_fp": o["fp"], "oracle_mcc": o["mcc"],
                        "oracle_threshold": o["threshold"]})
            for at, per in ([("oracle", p.get("oracle_by_attack"))] +
                            [(f"B={B}", v) for B, v in (p.get("budget_by_attack") or {}).items()]):
                for g, v in (per or {}).items():
                    arows.append({"ds": ds, "plan": plan, "main": p["main"], "at": at, "group": g, **v})
            for name, r in p["sensitivity"].items():
                srows.append({"ds": ds, "plan": plan, "main": p["main"], "variant": name, **r})
    return brows, arows, srows


def forms_tables(states):
    """{ds: forms() dict} -> (per-form rows, form-gap rows)."""
    frows, grows = [], []
    for ds in [d for d in HOSTS if d in states]:
        s = states[ds]
        for key, r in s["forms"].items():
            frows.append({"ds": ds, "plan": s["k"], "form": key, **r, "recon_max_abs": s["recon_max_abs"],
                          "cache_misses": s["cache_misses"]})
        for q, g in s["gaps"].items():
            grows.append({"ds": ds, "plan": s["k"], "question": q, **g})
    return frows, grows


def write_csvs(budgets, states, out_dir):
    """budget_rows.csv, per_attack_recall.csv, sensitivity_composition.csv (from budgets) and
    sensitivity_forms.csv, form_gaps.csv (from states) under out_dir; floats as repr."""
    from habitus.eval.derived import csv_text
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    if budgets:
        b, a, s = budget_tables(budgets)
        for name, rows, cols in (("budget_rows.csv", b, BUDGET_COLS), ("per_attack_recall.csv", a, ATTACK_COLS),
                                 ("sensitivity_composition.csv", s, SENS_COLS)):
            (out / name).write_bytes(csv_text(rows, cols).encode("utf-8"))
            written[name] = out / name
    if states:
        f, g = forms_tables(states)
        for name, rows, cols in (("sensitivity_forms.csv", f, FORM_COLS), ("form_gaps.csv", g, GAP_COLS)):
            (out / name).write_bytes(csv_text(rows, cols).encode("utf-8"))
            written[name] = out / name
    return written


WHAT = ("budget", "sensitivity", "forms", "boot")
FORMS_SUMMARY = "forms_summary.json"


def run(out_dir=None, what=WHAT, *, hosts=HOSTS, root=None, source="nodeindex", settings=None):
    """`habitus eval operating`: per host budget_<DS>.json (budget rows + lambda / combiner sensitivity:
    'budget' and 'sensitivity' are one computation), state_<DS>.json + vec_<DS>.npz ('forms', guarded
    replay child) and boot_<DS>.json ('boot', after forms: it reads vec_<DS>.npz); then the CSVs,
    forms_summary.json ({ds: recon_max_abs, cache_misses}) and bootstrap.json / bootstrap_mcc.csv.
    Returns {file name: path}."""
    from habitus.eval import bootstrap as BT
    what = tuple(x for x in what.split(",") if x) if isinstance(what, str) else tuple(what)
    bad = sorted(set(what) - set(WHAT))
    if bad:
        raise ValueError(f"unknown operating parts {bad}; choose from {WHAT}")
    out_dir = Path(out_dir) if out_dir is not None else runs_dir(settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    budgets, states, boots, paths = {}, {}, {}, {}
    for ds in hosts:
        host = _host(ds, None, source, settings)
        if "budget" in what or "sensitivity" in what:
            budgets[ds] = budget(ds, root=root, host=host, settings=settings)
            paths[f"budget_{ds}.json"] = write_json(budgets[ds], out_dir / f"budget_{ds}.json")
        if "forms" in what:
            states[ds] = forms(ds, root=root, out_dir=out_dir, host=host, settings=settings)
            paths[f"state_{ds}.json"] = write_json(states[ds], out_dir / f"state_{ds}.json")
        if "boot" in what:
            boots[ds] = BT.boot(ds, root=root, operating_dir=out_dir, host=host, settings=settings)
            paths[f"boot_{ds}.json"] = write_json(boots[ds], out_dir / f"boot_{ds}.json")
        del host
    paths.update(write_csvs(budgets, states, out_dir))
    if states:
        summ = {ds: {k: s[k] for k in ("k", "recon_max_abs", "cache_misses")} for ds, s in states.items()}
        paths[FORMS_SUMMARY] = write_json(summ, out_dir / FORMS_SUMMARY)
    if boots:
        bs = BT.merge(boots)
        paths["bootstrap.json"] = write_json({"summary": bs, "boot": {ds: {k: v for k, v in b.items() if k != "boot"}
                                                                     for ds, b in boots.items()}},
                                             out_dir / "bootstrap.json")
        paths["bootstrap_mcc.csv"] = BT.write_csv(bs, out_dir / "bootstrap_mcc.csv")
    return paths
