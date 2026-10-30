"""PIDSMaker baseline rows from the upstream run artifacts (server only: settings.paths.upstream_artifacts).

    from habitus.eval import baselines
    rows, why = baselines.upstream_rows(host, "velox")           # best_adp and last epoch of one run
    baselines.export(hosts, up=..., results_dir="results/baselines", runs_dir="runs/baselines")

The upstream runs (PIDSMaker 2289cd9) are found through their logs, never through the hash-level
run_config.yml (several runs share a hash): _logs/<run>__<ds>.log, line "Resutls saved to <dir>/results.pth"
(upstream's typo), then <dir>/../precision_recall_dir. Their scores_model_epoch_<e>.pkl and
stats_model_epoch_<e>.pth are torch pickles loaded with weights_only=False: TRUSTED LOCAL FILES ONLY.
They carry TEST labels (y_truth; test AP / AUC / ADP) and are read only here.

Rows per run: PIDSMaker's best_adp epoch (its best_model_selection on TEST labels: max (adp_score,
discrimination), the first epoch in numeric order on ties) and the last epoch. Each row holds the oracle
best-MCC confusion on the run's OWN node set and y_truth (primary), the same on the HABITUS universe with
the habitus labels (unscored universe nodes ranked last: min - 1), and PIDSMaker's own-threshold
confusion (stats file; recomputed from y_preds as a check).

export() writes results/baselines/{pidsmaker_rows.csv (full precision, repr floats), pidsmaker_missing.csv,
pidsmaker_args.sh, PROVENANCE.md} and runs/baselines/<ds>_best_baseline.npz (never release/: make_release
copies it). The best baseline of a host = the run with the highest own-node-set MCC at best_adp (BASELINE_RUNS
order on ties), aligned to the HABITUS universe.
"""

from __future__ import annotations

import glob
import hashlib
import os
import posixpath
import re
import shutil
import sys
from pathlib import Path

import numpy as np

HOSTS = ("CADETS_E3", "THEIA_E3", "CLEARSCOPE_E3", "optc_h501", "ATLASV2_EDR")
UPSTREAM_COMMIT = "2289cd9"
# (upstream log name, system, flags). Only <system>__<DS> and <system>_def__<DS> runs are clean: the
# attacked runs (*_masq, *_mim1000, *_mim5000) are never listed here.
BASELINE_RUNS = (
    ("orthrus", "ORTHRUS", ("snooped",)),
    ("orthrus_non_snooped", "ORTHRUS-ns", ()),
    ("orthrus_ns_def", "ORTHRUS-ns", ()),
    ("kairos", "KAIROS", ()),
    ("kairos_def", "KAIROS", ()),
    ("velox", "Velox", ()),
    ("velox_def", "Velox", ()),
    ("nodlink", "NodLink", ()),
    ("nodlink_def", "NodLink", ()),
    ("magic", "MAGIC", ("transductive",)),
    ("threatrace", "ThreaTrace", ("degenerate",)),
    ("flash", "Flash", ("degenerate",)),
    ("rcaid", "R-CAID", ("snooped",)),
)
# PIDSMaker docs/tuned_systems.md has args for these systems on these hosts only; elsewhere every run (also the
# one named orthrus_non_snooped / kairos / velox / nodlink) is the config default
TUNABLE = ("orthrus_non_snooped", "kairos", "velox", "nodlink")
TUNED_HOSTS = ("CADETS_E3", "THEIA_E3", "CLEARSCOPE_E3", "optc_h501")
NBR10 = ("orthrus", "orthrus_non_snooped", "orthrus_ns_def", "kairos", "kairos_def")   # optc_h501 only
FLAG_TEXT = {
    "snooped": "S = not label-free / transductive: its embedding (ORTHRUS word2vec, R-CAID doc2vec) is trained on "
               "every split including the test window",
    "transductive": "T = MAGIC port: the kNN reference of each test window is 5,000 random nodes of that same window "
                    "(attack nodes included), unseeded",
    "degenerate": "D = degenerate port score (a type-prediction confidence, not an anomaly score); PIDSMaker's "
                  "direction kept (higher = more anomalous), no sign flip: a direction is never chosen with labels",
    "tuned": "tuned* = docs/tuned_systems.md args (provenance unclear, possibly chosen on test data); the default "
             "rows are the fairer ones",
    "nbr10": "n10 = tgn_neighbor_size=10 (RAM limit) instead of the default 20",
}
FLAG_CODE = {"snooped": "S", "transductive": "T", "degenerate": "D", "tuned": "tuned*", "nbr10": "n10"}
# the run logs name the container path of the artifacts; it is rewritten to the local upstream root
_CONTAINER_ART = posixpath.join("/", "home", "artifacts")
ROW_COLS = ("ds", "system", "config", "log", "run_dir", "epoch", "policy", "best_adp_epoch", "last_epoch", "flags",
            "n", "n_pos", "tp", "fp", "tn", "fn", "precision", "recall", "f1", "mcc", "auc", "ap", "threshold",
            "alerts", "tie_n", "tie_gt", "mcc_ties_neg", "why",
            "n_H", "tp_H", "fp_H", "tn_H", "fn_H", "mcc_H", "mcc_ties_neg_H", "auc_H", "ap_H", "threshold_H",
            "nodes_scored_H", "outside_universe_H", "duplicates_H",
            "pm_tp", "pm_fp", "pm_tn", "pm_fn", "pm_precision", "pm_recall", "pm_mcc", "pm_recomputed_match")
MISSING_COLS = ("ds", "system", "config", "log", "epoch", "policy", "why")


def _log(msg):
    try:
        from habitus.log import log
        log(msg, "eval.baselines")
    except ImportError:
        print(f"[eval.baselines] {msg}", file=sys.stderr, flush=True)


def _f(x, nd=3):
    import math
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    return f"{x:.{nd + 1}f}" if 0 < abs(x) < 0.01 else f"{x:.{nd}f}"


def upstream_root(up=None, settings=None) -> str:
    if up is not None:
        return str(up)
    from habitus.config import settings as _s
    p = getattr((settings or _s()).paths, "upstream_artifacts", None)
    if not p:
        raise FileNotFoundError("settings.paths.upstream_artifacts is not set (configs/local.yaml, server only)")
    return str(p)


def _epoch_of(path):
    return int(re.search(r"(\d+)\.p[a-z]+$", os.path.basename(path)).group(1))


def upstream_run_dir(system, ds, up):
    """(precision_recall_dir of the clean upstream run `system` on `ds`, status). None with the reason when
    the run is missing, unfinished, or a cache hit of another run (no eval output)."""
    lg = os.path.join(up, "_logs", f"{system}__{ds}.log")
    if not os.path.exists(lg):
        return None, "no run"
    with open(lg, errors="ignore") as f:
        txt = f.read()
    if "Run finished" not in txt:
        return None, "unfinished"
    m = re.findall(r"Resutls saved to (\S+)/results\.pth", txt)
    if not m:
        return None, "finished without eval output (cache hit of a run with the same config)"
    d = os.path.join(os.path.dirname(m[-1].rstrip("/")).replace(_CONTAINER_ART, up), "precision_recall_dir")
    if not os.path.isdir(d):
        return None, f"no precision_recall_dir ({os.path.relpath(d, up)})"
    return d, "ok"


def upstream_epochs(run_dir):
    """({epoch: {adp_score, discrimination, ap, auc}} from every stats file, PIDSMaker's best_adp epoch,
    epochs that still have a score file). best_adp = max (adp_score, discrimination) on TEST labels, the
    first epoch in numeric order on ties."""
    import torch
    stats = {}
    for p in glob.glob(os.path.join(run_dir, "stats_model_epoch_*.pth")):
        st = torch.load(p, map_location="cpu", weights_only=False)          # trusted local file
        stats[_epoch_of(p)] = {k: (None if st.get(k) is None else float(st[k]))
                               for k in ("adp_score", "discrimination", "ap", "auc")}
    best, key = None, None
    for e in sorted(stats):
        kk = (stats[e]["adp_score"] or 0.0, stats[e]["discrimination"] or 0.0)
        if key is None or kk > key:
            best, key = e, kk
    have = sorted(_epoch_of(p) for p in glob.glob(os.path.join(run_dir, "scores_model_epoch_*.pkl")))
    return stats, best, have


def align_to_universe(universe, nodes, sc):
    """(scores aligned with `universe`, bool mask of universe nodes the baseline scored, info). A node
    scored twice keeps its max; a universe node the baseline never scored gets min - 1 (ranked strictly
    last; it can only help the baseline when such nodes are not GT)."""
    from habitus.decide.scorefile import upos
    pos = upos(universe, nodes)
    inside = pos >= 0
    s = np.full(len(universe), -np.inf)
    np.maximum.at(s, pos[inside], sc[inside])
    scored = np.isfinite(s)
    lo = float(sc[inside].min()) if inside.any() else 0.0
    s[~scored] = lo - 1.0
    return s, scored, {"n_scores": int(len(nodes)), "outside_universe": int((~inside).sum()),
                       "duplicates": int(len(nodes) - len(np.unique(nodes))),
                       "n_distinct_scores": int(len(np.unique(sc)))}


def _upstream_file(run_dir, epoch):
    """(nodes, pred_scores, y_truth, y_preds) of one upstream epoch (PIDSMaker's own node set)."""
    import torch
    S = torch.load(os.path.join(run_dir, f"scores_model_epoch_{epoch}.pkl"), map_location="cpu",
                   weights_only=False)                                   # trusted local file
    out = (np.asarray(S["nodes"], dtype=np.int64), np.asarray(S["pred_scores"], dtype=np.float64),
           np.asarray(S["y_truth"], dtype=np.int64) == 1,
           None if S.get("y_preds") is None else np.asarray(S["y_preds"], dtype=np.int64) == 1)
    del S
    if not (len(out[0]) == len(out[1]) == len(out[2])):
        raise ValueError(f"{run_dir} epoch {epoch}: unequal nodes / scores / y_truth")
    if not np.isfinite(out[1]).all():
        raise ValueError(f"{run_dir} epoch {epoch}: non-finite scores")
    return out


def _upstream_stats(run_dir, epoch):
    """PIDSMaker's own-threshold metrics of one epoch (stats_model_epoch_<e>.pth), or None."""
    import torch
    p = os.path.join(run_dir, f"stats_model_epoch_{epoch}.pth")
    if not os.path.exists(p):
        return None
    st = torch.load(p, map_location="cpu", weights_only=False)              # trusted local file

    def g(k):
        v = st.get(k)
        return None if v is None else float(v)
    out = {k: g(k) for k in ("tp", "fp", "tn", "fn", "precision", "recall", "fscore", "mcc", "fpr", "ap", "auc",
                             "adp_score", "discrimination")}
    for k in ("tp", "fp", "tn", "fn"):
        out[k] = None if out[k] is None else int(round(out[k]))
    return out


def upstream_label(logname, ds):
    """(system, config): tuned* = docs/tuned_systems.md args, original = ORTHRUS, default otherwise."""
    system = next(s for lg, s, _f in BASELINE_RUNS if lg == logname)
    tuned = logname in TUNABLE and ds in TUNED_HOSTS
    return system, "tuned*" if tuned else ("original" if logname == "orthrus" else "default")


def upstream_rows(host, logname, flags=(), up=None, *, keep_vectors=False):
    """Best-MCC rows of one clean PIDSMaker run on `host` at its best_adp and last epochs: on the run's OWN
    node set with its own y_truth (primary), on the HABITUS universe with the habitus labels, and
    PIDSMaker's own-threshold confusion. ([rows], None) or ([], why). keep_vectors: each evaluated row
    also carries "_s_u" (the universe-aligned scores) for the best-baseline export."""
    from habitus.eval.metrics import confusion, confusion_counts, own_confusion
    up = upstream_root(up)
    d, status = upstream_run_dir(logname, host.ds, up)
    if d is None:
        return [], status
    stats, best_adp, have = upstream_epochs(d)
    if not stats:
        return [], f"no stats file in {os.path.relpath(d, up)}"
    last = max(stats)
    system, config = upstream_label(logname, host.ds)
    fl = list(flags) + (["nbr10"] if host.ds == "optc_h501" and logname in NBR10 else [])
    rel = os.path.relpath(d, up)
    rows = []
    policies = [("best_adp", best_adp)] + ([("last", last)] if last != best_adp else [])
    for pol, e in policies:
        if e not in have:
            rows.append({"family": "upstream", "system": system, "config": config, "log": logname, "epoch": e,
                         "policy": pol, "missing": f"no score file for epoch {e} in {rel}"})
            continue
        nodes, sc, yt, yp = _upstream_file(d, e)
        own = own_confusion(sc, yt)
        s_u, scored, info = align_to_universe(host.universe, nodes, sc)
        hb = confusion(host, s_u)
        hb["coverage"] = {**info, "nodes_scored": int(scored.sum())}
        st = _upstream_stats(d, e) or {}
        pm = None
        if st.get("tp") is not None:
            pm = {**confusion_counts(st["tp"], st["fp"], st["tp"] + st["fn"], st["fp"] + st["tn"]),
                  "stats": st, "source": f"stats_model_epoch_{e}.pth"}
            if yp is not None:
                tp_r, fp_r = int((yp & yt).sum()), int((yp & ~yt).sum())
                pm["recomputed_from_y_preds"] = {"tp": tp_r, "fp": fp_r,
                                                 "match": tp_r == st["tp"] and fp_r == st["fp"]}
                pm["lowest_alerted_score"] = float(sc[yp].min()) if yp.any() else None
        row = {"family": "upstream", "system": system, "config": config, "log": logname, "epoch": e,
               "policy": "best_adp=last" if pol == "best_adp" and best_adp == last else pol,
               "best_adp_epoch": best_adp, "last_epoch": last, "run_dir": rel, "flags": fl,
               **own, "habitus_universe": hb, "pidsmaker_threshold": pm}
        if keep_vectors:
            row["_s_u"] = s_u
        rows.append(row)
        del nodes, sc, yt, yp, scored
        _log(f"{host.ds} {logname} ep{e} ({pol}): own MCC {_f(own.get('mcc'))} "
             f"TP {own.get('tp')} FP {own.get('fp')}; HB-universe MCC {_f(hb.get('mcc'))}; "
             f"PIDSMaker thr MCC {_f((pm or {}).get('mcc'))}")
    return rows, None


def flat_row(ds, r):
    """One pidsmaker_rows.csv row (full precision) of an upstream_rows row."""
    tb, tn = r.get("tie_block") or {}, r.get("ties_negative") or {}
    hb = r.get("habitus_universe") or {}
    hbt = hb.get("ties_negative") or {}
    cov = hb.get("coverage") or {}
    pm = r.get("pidsmaker_threshold") or {}
    rec = (pm.get("recomputed_from_y_preds") or {}).get("match")
    return {"ds": ds, "system": r["system"], "config": r["config"], "log": r["log"], "run_dir": r.get("run_dir"),
            "epoch": r["epoch"], "policy": r["policy"], "best_adp_epoch": r.get("best_adp_epoch"),
            "last_epoch": r.get("last_epoch"), "flags": ";".join(FLAG_CODE[f] for f in r.get("flags") or []),
            "n": r.get("n"), "n_pos": r.get("n_pos"), "tp": r.get("tp"), "fp": r.get("fp"), "tn": r.get("tn"),
            "fn": r.get("fn"), "precision": r.get("precision"), "recall": r.get("recall"), "f1": r.get("f1"),
            "mcc": r.get("mcc"), "auc": r.get("auc"), "ap": r.get("ap"), "threshold": r.get("threshold"),
            "alerts": r.get("alerts"), "tie_n": tb.get("n"), "tie_gt": tb.get("gt"), "mcc_ties_neg": tn.get("mcc"),
            "why": r.get("why"),
            "n_H": hb.get("n"), "tp_H": hb.get("tp"), "fp_H": hb.get("fp"), "tn_H": hb.get("tn"), "fn_H": hb.get("fn"),
            "mcc_H": hb.get("mcc"), "mcc_ties_neg_H": hbt.get("mcc"), "auc_H": hb.get("auc"), "ap_H": hb.get("ap"),
            "threshold_H": hb.get("threshold"), "nodes_scored_H": cov.get("nodes_scored"),
            "outside_universe_H": cov.get("outside_universe"), "duplicates_H": cov.get("duplicates"),
            "pm_tp": pm.get("tp"), "pm_fp": pm.get("fp"), "pm_tn": pm.get("tn"), "pm_fn": pm.get("fn"),
            "pm_precision": pm.get("precision"), "pm_recall": pm.get("recall"), "pm_mcc": pm.get("mcc"),
            "pm_recomputed_match": rec}


def universe_sha1(universe):
    return hashlib.sha1(np.ascontiguousarray(np.asarray(universe, dtype=np.int64)).tobytes()).hexdigest()


def write_best_npz(path, s_u, universe, info):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, s_u=np.asarray(s_u, dtype=np.float64), universe_n=np.int64(len(universe)),
                        universe_sha1=np.str_(universe_sha1(universe)), own_mcc=np.float64(info["own_mcc"]),
                        log=np.str_(info["log"]), system=np.str_(info["system"]), epoch=np.int64(info["epoch"]))
    os.replace(tmp, path)
    return path


def host_rows(host, up):
    """(evaluated rows, missing entries, best baseline (own_mcc, log, system, epoch, s_u) or None) of one host."""
    rows, missing, best = [], [], None
    for logname, system, flags in BASELINE_RUNS:
        rs, why = upstream_rows(host, logname, flags, up, keep_vectors=True)
        if not rs:
            sy, cfg = upstream_label(logname, host.ds)
            missing.append({"ds": host.ds, "system": sy, "config": cfg, "log": logname, "why": why})
            continue
        for r in rs:
            if "missing" in r:
                missing.append({"ds": host.ds, "system": r["system"], "config": r["config"], "log": logname,
                                "epoch": r["epoch"], "policy": r["policy"], "why": r["missing"]})
                continue
            s_u = r.pop("_s_u")
            if r["policy"] in ("best_adp", "best_adp=last"):
                own = r.get("mcc") or 0.0
                if best is None or own > best[0] + 1e-12:
                    best = (own, logname, system, r["epoch"], s_u)
            rows.append(r)
    return rows, missing, best


def provenance_md(missing):
    L = ["# PIDSMaker baselines: provenance", "",
         f"Upstream: PIDSMaker commit `{UPSTREAM_COMMIT}` (Apache-2.0), run in its own environment. The worktree "
         "the runs used carried local diffs relative to that commit: the optc_h501 database name (`optc_501` -> "
         "`optc_h501`, pidsmaker/config/config.py); a chunked eval-mode TGN memory flush (pidsmaker/tgn.py, same "
         "result); attack-injection hooks that are inert when unset (mimicry.py, a masquerade option in "
         "build_default_graphs.py, labelling.py, pipeline.py). Hyperparameters: `pidsmaker_args.sh` (a copy of the "
         "job script): docs/tuned_systems.md args where that file has them, config defaults elsewhere; "
         "tgn_neighbor_size=10 for the TGN systems on optc_h501 (RAM).", "",
         "Rows (`pidsmaker_rows.csv`, regenerated by `python scripts/export_baselines.py` on the server; full "
         "precision, floats written as repr): one row per clean run and epoch policy. `best_adp` = PIDSMaker's own "
         "best-model selection, max (adp_score, discrimination), computed on the TEST labels - a label-selected "
         "epoch; `last` = the final epoch; `best_adp=last` when they coincide. Primary columns: the oracle best-MCC "
         "confusion on the run's OWN node set and y_truth; `*_H`: the same on the HABITUS universe with the habitus "
         "labels, universe nodes the run never scored ranked last; `pm_*`: PIDSMaker's own-threshold confusion "
         "(stats file), `pm_recomputed_match` = recomputed from y_preds.", "",
         "Runs: " + ", ".join(f"`{lg}` ({sy})" for lg, sy, _f in BASELINE_RUNS) + ". Attacked runs (masquerade, "
         "mimicry) are never listed.", "",
         "Config labels: `tuned*` for " + ", ".join(f"`{x}`" for x in TUNABLE) + " on " + ", ".join(TUNED_HOSTS)
         + "; `original` for ORTHRUS; `default` otherwise.", "",
         "Flags:", ""]
    L += [f"- {t}" for t in FLAG_TEXT.values()]
    L += ["", "Not evaluated (see `pidsmaker_missing.csv`):", ""]
    L += [f"- {m['ds']} {m['system']} [{m['config']}] `{m['log']}`: {m['why']}" for m in missing] or ["- none"]
    L += ["", "The upstream score files (~8.5 GB of torch pickles, loaded with weights_only=False: trusted local "
          "files only) are not shipped; external users get these CSV rows and the best-baseline vectors in the cache "
          "bundle only.", ""]
    return "\n".join(L)


def export(hosts=HOSTS, *, up=None, results_dir="results/baselines", runs_dir="runs/baselines", source="nodeindex",
           settings=None, host_factory=None):
    """Write pidsmaker_rows.csv, pidsmaker_missing.csv, pidsmaker_args.sh, PROVENANCE.md under results_dir and
    <ds>_best_baseline.npz under runs_dir. Returns {ds: best-baseline info}."""
    from habitus.eval.derived import csv_text
    from habitus.eval.tables import make_host
    from habitus.hosts import assert_allowed
    up = upstream_root(up, settings)
    res, rd = Path(results_dir), Path(runs_dir)
    res.mkdir(parents=True, exist_ok=True)
    rd.mkdir(parents=True, exist_ok=True)
    flat, missing, best_info = [], [], {}
    for ds in hosts:
        assert_allowed(ds)
        host = host_factory(ds) if host_factory else make_host(ds, source, settings)
        rows, miss, best = host_rows(host, up)
        flat += [flat_row(ds, r) for r in rows]
        missing += miss
        if best is not None:
            info = {"own_mcc": best[0], "log": best[1], "system": best[2], "epoch": best[3]}
            write_best_npz(rd / f"{ds}_best_baseline.npz", best[4], host.universe, info)
            best_info[ds] = info
        del host
    (res / "pidsmaker_rows.csv").write_bytes(csv_text(flat, ROW_COLS).encode("utf-8"))
    (res / "pidsmaker_missing.csv").write_bytes(csv_text(missing, MISSING_COLS).encode("utf-8"))
    args = Path(up) / "_jobs" / "args.sh"
    if args.is_file():
        shutil.copyfile(args, res / "pidsmaker_args.sh")
    (res / "PROVENANCE.md").write_text(provenance_md(missing), encoding="utf-8")
    return best_info
