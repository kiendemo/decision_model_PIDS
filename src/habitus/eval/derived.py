"""Paper means and margins, derived from the shipped results/ CSVs only (no label is read here).

    from habitus.eval import derived
    s = derived.summary_means(tables.rows_from_csv(open(".../ablation_e1_e2.csv", "rb").read()))
    p = derived.paper_means("results")       # every derived number of the README table

Inputs (relative to the results directory):
  habitus/ablation_e1_e2.csv                    E1 / E2 cells (main config, k means, ladder, routes)
  baselines/pidsmaker_rows.csv                  PIDSMaker rows (best baseline per row choice, margins)
  extension_habit/decision_models_best_mcc.csv  frozen HABITUS+H arms (9-cell means, delta vs HB0)
  extension_habit/hb0_rows.csv                  HB0 rows
  extension_habit/decision_models_speed_mean.csv, habitus/route_cost.csv   cost rows

The arithmetic is that of the paper's derivation scripts (means over the 5 hosts in HOSTS order; the
complete-ladder means over the 4 hosts with every E1 level; 9 cells = 5 hosts at k2 + 4 at allc-pre,
CADETS_E3 having no allc-pre). MCC of a frozen row = round(MCC of its TP/FP/TN/FN, 6), as there.
"""

from __future__ import annotations

import csv
import io
import math
import statistics
from pathlib import Path

HOSTS = ("CADETS_E3", "THEIA_E3", "CLEARSCOPE_E3", "optc_h501", "ATLASV2_EDR")
D4 = ("CADETS_E3", "THEIA_E3", "optc_h501", "ATLASV2_EDR")          # hosts with a complete E1 ladder
ARMS = ("HB-L.mq", "HB-L.s0", "HB-L", "HB", "HB.head", "HB.08b", "HB-Z")
HABIT_PLAN = {d: ("k2" if d == "CADETS_E3" else "allc-pre") for d in HOSTS}
CELLS9 = [(ds, kp) for ds in HOSTS for kp in ("k2", "allc-pre") if not (ds == "CADETS_E3" and kp == "allc-pre")]
# test-day scoring at k2 of the habit channel: (questions over null + test days, days), run_facts_20261029.md
HABIT_K2_QUESTIONS = {"CADETS_E3": (2352, 6), "THEIA_E3": (1756, 5), "CLEARSCOPE_E3": (775, 4),
                      "optc_h501": (7027, 5), "ATLASV2_EDR": (1574, 3)}
FLAGGED = ("S", "T", "D")


def r4(x):
    return None if x is None else round(x, 4)


def mcc_counts(tp, fp, tn, fn):
    """MCC from integer counts (exact integer products; 0 when a margin is empty)."""
    d = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return (tp * tn - fp * fn) / math.sqrt(d) if d else 0.0


def read_csv(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _num(x):
    if x is None or x == "":
        return None
    try:
        return int(x)
    except ValueError:
        return float(x)


# --------------------------------------------------------------------------- #
# HABITUS (E2) and the E1 ladder: ablation_e1_e2.csv
# --------------------------------------------------------------------------- #
def _index(rows):
    return {(r["ds"], r["level"], r["k"], r["column"]): r for r in rows}


def _m(rows):
    v = [float(r["mcc"]) for r in rows if r is not None]
    return sum(v) / len(v) if v else None


MAIN_FIELDS = ("k", "tp", "fp", "tn", "fn", "precision", "recall", "mcc", "auc", "ap", "alerts", "tie_n", "tie_gt",
               "mcc_ties_neg", "gt_in_head", "gt_total")


def summary_means(rows):
    """Means of the ablation rows (typed rows: tables.rows_from_csv of the shipped CSV): main config,
    k means, ties negative, E1 ladder mean4 / mean5 and deltas, route means and deltas."""
    idx = _index(rows)

    def z(ds, lv, k, col):
        return idx.get((ds, lv, k, col))

    maink = {r["ds"]: r["k"] for r in rows if r.get("main_config") == "yes"}
    missing = [d for d in HOSTS if d not in maink]
    if missing:
        raise ValueError(f"no main_config row for {missing}")
    out = {}
    main = {d: z(d, "E2", maink[d], "composite") for d in HOSTS}
    out["main_k"] = {d: maink[d] for d in HOSTS}
    out["main_config"] = {d: {k: main[d][k] for k in MAIN_FIELDS} for d in HOSTS}
    out["main_mean_mcc"] = _m(main.values())
    out["main_mean_mcc_ties_neg"] = sum(float(main[d]["mcc_ties_neg"]) for d in HOSTS) / 5
    out["main_mean_ap"] = sum(float(main[d]["ap"]) for d in HOSTS) / 5
    out["main_mean_auc"] = sum(float(main[d]["auc"]) for d in HOSTS) / 5
    for k in ("k0", "k2", "kall"):
        rs = [z(d, "E2", k, "composite") for d in HOSTS]
        out[f"habitus_{k}"] = {"cells": {d: float(r["mcc"]) for d, r in zip(HOSTS, rs)}, "mean_mcc": _m(rs),
                               "mean_ap": sum(float(r["ap"]) for r in rs) / 5,
                               "mean_auc": sum(float(r["auc"]) for r in rs) / 5}
    out["main_ties_neg_cells"] = {d: float(main[d]["mcc_ties_neg"]) for d in HOSTS}
    out["ladder"] = {}
    for k in ("k0", "k2", "kall"):
        for lv in ("L0", "L1", "L2", "L3", "L4", "E2"):
            rs = {d: z(d, lv, k, "composite") for d in HOSTS}
            if all(v is None for v in rs.values()):
                continue
            out["ladder"][f"{lv}|{k}"] = {"mean4": _m([rs[d] for d in D4]) if all(rs[d] for d in D4) else None,
                                          "mean5": _m(rs.values()) if all(rs.values()) else None}
    for k in ("k2", "kall"):
        for a, b in (("L1", "L0"), ("L2", "L1"), ("L3", "L2"), ("L4", "L3")):
            dd = [float(z(x, a, k, "composite")["mcc"]) - float(z(x, b, k, "composite")["mcc"]) for x in D4]
            out["ladder"][f"delta4 {a}-{b}|{k}"] = {"mean": sum(dd) / 4, "up": sum(x > 0 for x in dd),
                                                    "down": sum(x < 0 for x in dd)}
    out["routes"] = {}
    for k in ("k0", "k2", "kall"):
        for col in ("qv", "qt", "base", "composite", "1-qa", "acq_all"):
            rs = [z(d, "E2", k, col) for d in HOSTS]
            if any(r is None for r in rs):
                continue
            out["routes"][f"{col}|{k}"] = _m(rs)
    for k in ("k0", "k2", "kall"):
        c = {d: float(z(d, "E2", k, "composite")["mcc"]) for d in HOSTS}
        for name, (lv, col) in {"acquittal (composite-base)": ("E2", "base"),
                                "hard tier (composite-acq_all)": ("E2", "acq_all"),
                                "composite-QV": ("E2", "qv"), "composite-QT": ("E2", "qt"),
                                "E2-E1 L4": ("L4", "composite"), "E2-E1 L2": ("L2", "composite"),
                                "E2-E1 L3": ("L3", "composite")}.items():
            o = {d: float(z(d, lv, k, col)["mcc"]) for d in HOSTS if z(d, lv, k, col) is not None}
            if not o:
                continue
            dd = [c[d] - o[d] for d in o]
            out["routes"][f"delta {name}|{k}"] = {"mean": sum(dd) / len(dd), "up": sum(x > 1e-9 for x in dd),
                                                  "down": sum(x < -1e-9 for x in dd), "n": len(dd)}
    return out


# --------------------------------------------------------------------------- #
# PIDSMaker baselines: pidsmaker_rows.csv
# --------------------------------------------------------------------------- #
def _flag_codes(r):
    return {x.strip() for x in (r.get("flags") or "").replace(",", ";").split(";") if x.strip()}


def baseline_rows(rows):
    """{ds: [row]} of pidsmaker_rows.csv rows that were evaluated (not 'missing'), with the derived keys
    best_adp / last (policy), mcc_exact (round(MCC of the counts, 6)), flagged (S / T / D)."""
    out = {d: [] for d in HOSTS}
    for r in rows:
        if r.get("ds") not in out or r.get("tp") in (None, ""):
            continue
        tp, fp, tn, fn = (int(r[k]) for k in ("tp", "fp", "tn", "fn"))
        pol = r["policy"]
        out[r["ds"]].append({**r, "best_adp": pol in ("best_adp", "best_adp=last"),
                             "last": pol in ("last", "best_adp=last"),
                             "mcc_exact": round(mcc_counts(tp, fp, tn, fn), 6),
                             "mcc_ties_neg_f": _num(r.get("mcc_ties_neg")),
                             "flagged": bool(set(FLAGGED) & _flag_codes(r)),
                             "label": f"{r['system']} [{r['config']}]"})
    return out


CHOICES = {
    "best_adp_any": lambda r: r["best_adp"],
    "best_adp_no_STD": lambda r: r["best_adp"] and not r["flagged"],
    "best_adp_untuned_no_STD": lambda r: r["best_adp"] and r["config"] != "tuned*" and not r["flagged"],
    "epoch11": lambda r: r["last"],
    "better_of_two_epochs": lambda r: r["best_adp"] or r["last"],
}


def baselines(base, main_mcc):
    """Best baseline per host under each row choice, its mean, and HABITUS's margins.
    base: baseline_rows(); main_mcc: {ds: HABITUS main-config MCC}."""
    out = {}
    for name, cond in CHOICES.items():
        best = {d: max((r for r in base[d] if cond(r)), key=lambda r: r["mcc_exact"]) for d in HOSTS}
        marg = {d: float(main_mcc[d]) - best[d]["mcc_exact"] for d in HOSTS}
        out[name] = {"best": {d: [best[d]["label"], best[d]["mcc_exact"]] for d in HOSTS},
                     "mean": sum(b["mcc_exact"] for b in best.values()) / 5,
                     "habitus_margin": marg, "habitus_margin_mean": sum(marg.values()) / 5,
                     "hosts_habitus_above": sum(1 for x in marg.values() if x > 0)}
    tn = {d: max(r["mcc_ties_neg_f"] for r in base[d] if r["best_adp"] and r["mcc_ties_neg_f"] is not None)
          for d in HOSTS}
    out["best_adp_ties_neg"] = tn
    out["best_adp_ties_neg_mean"] = sum(tn.values()) / 5
    return out


# --------------------------------------------------------------------------- #
# HABITUS+H (frozen arms) vs HB0 and E2: decision_models_best_mcc.csv, hb0_rows.csv
# --------------------------------------------------------------------------- #
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def habit_cells(dm_rows, hb0_rows, ablation_rows):
    """{(system, ds, k_plan, 'composite'|'habit-only'): MCC} for the arms, HB0 and E2 (HABITUS)."""
    cells = {}
    for r in dm_rows:
        sc = "composite" if r["score"].startswith("composite") else "habit-only"
        cells[(r["model"], r["dataset"], r["k_plan"], sc)] = round(
            mcc_counts(int(r["TP"]), int(r["FP"]), int(r["TN"]), int(r["FN"])), 6)
    for r in hb0_rows:
        cells[("HB0", r["ds"], r["k_plan"], "composite")] = round(
            mcc_counts(int(r["tp"]), int(r["fp"]), int(r["tn"]), int(r["fn"])), 6)
    idx = _index(ablation_rows)
    for ds in HOSTS:
        for kp, k in (("k2", "k2"), ("allc-pre", "kall")):
            if ds == "CADETS_E3" and kp == "allc-pre":
                continue                    # CADETS kall includes post-test clean days: never allc-pre
            r = idx.get((ds, "E2", k, "composite"))
            if r is not None:
                cells[("E2", ds, kp, "composite")] = round(
                    mcc_counts(int(r["tp"]), int(r["fp"]), int(r["tn"]), int(r["fn"])), 6)
    return cells


def habit_summary(cells):
    """Per system: 9-cell / k2 / allc-pre / best-k / main-config means, habit-only means for the arms,
    and the per-cell delta vs HB0 and vs E2 (HABITUS)."""
    summ = {}
    for v in ARMS + ("E2", "HB0"):
        def get(ds, kp, sc="composite", v=v):
            return cells.get((v, ds, kp, sc))
        s = {"mean_composite_9cells": r4(_mean([get(ds, kp) for ds, kp in CELLS9])),
             "mean_composite_k2_5hosts": r4(_mean([get(ds, "k2") for ds in HOSTS])),
             "mean_composite_allcpre_4hosts": r4(_mean([get(ds, "allc-pre") for ds in HOSTS if ds != "CADETS_E3"])),
             "mean_composite_bestk_per_dataset": r4(_mean([max(x for x in (get(ds, "k2"), get(ds, "allc-pre"))
                                                               if x is not None) for ds in HOSTS])),
             "mean_composite_main_config": r4(_mean([get(ds, HABIT_PLAN[ds]) for ds in HOSTS])),
             "per_cell_composite": {f"{ds}/{kp}": r4(get(ds, kp)) for ds, kp in CELLS9}}
        if v in ARMS:
            s["mean_habit_only_9cells"] = r4(_mean([get(ds, kp, "habit-only") for ds, kp in CELLS9]))
            s["mean_habit_only_k2_5hosts"] = r4(_mean([get(ds, "k2", "habit-only") for ds in HOSTS]))
            s["mean_habit_only_allcpre_4hosts"] = r4(_mean([get(ds, "allc-pre", "habit-only")
                                                            for ds in HOSTS if ds != "CADETS_E3"]))
            s["per_cell_habit_only"] = {f"{ds}/{kp}": r4(get(ds, kp, "habit-only")) for ds, kp in CELLS9}
        for base in ("HB0", "E2"):
            if v == base:
                continue
            d = {f"{ds}/{kp}": (None if get(ds, kp) is None or cells.get((base, ds, kp, "composite")) is None
                                else r4(get(ds, kp) - cells[(base, ds, kp, "composite")])) for ds, kp in CELLS9}
            vals = [x for x in d.values() if x is not None]
            s[f"delta_composite_vs_{base}"] = {
                "per_cell": d, "mean": r4(_mean(vals)), "n_cells": len(vals),
                "n_better": sum(x > 0 for x in vals), "n_worse": sum(x < 0 for x in vals),
                "n_equal": sum(x == 0 for x in vals),
                "mean_k2": r4(_mean([d[f"{ds}/k2"] for ds in HOSTS if d[f"{ds}/k2"] is not None])),
                "mean_allcpre": r4(_mean([d[f"{ds}/allc-pre"] for ds in HOSTS
                                          if ds != "CADETS_E3" and d.get(f"{ds}/allc-pre") is not None]))}
        summ[v] = s
    main = {}
    for model in ("HB0",) + tuple(m for m in ("HB-Z", "HB", "HB.head", "HB.08b", "HB-L.s0", "HB-L", "HB-L.mq")):
        for habit in (False, True):
            if model == "HB0" and habit:
                continue
            v = [cells.get((model, d, HABIT_PLAN[d], "habit-only" if habit else "composite")) for d in HOSTS]
            main[f"{model}|{'habit-only' if habit else 'composite'}"] = (sum(v) / 5) if None not in v else None
    return summ, main


# --------------------------------------------------------------------------- #
# Cost: route_cost.csv, decision_models_speed_mean.csv
# --------------------------------------------------------------------------- #
def route_engine(rc_rows):
    """Route-engine cost per host from route_cost.csv rows (keys '<ds>|<0|2|all>|<test|null>')."""
    rc = {r["key"]: r for r in rc_rows}

    def g(key, f):
        return _num(rc[key][f])
    mk = {d: ("2" if d == "CADETS_E3" else "all") for d in HOSTS}
    out = {d: {"test_kall_s_per_day": g(f"{d}|all|test", "engine_s_mean"),
               "test_kall_s_max_day": g(f"{d}|all|test", "engine_s_max"),
               "test_kall_prompts_per_s": g(f"{d}|all|test", "prompts_per_s_mean"),
               "routed_signatures_per_test_day": g(f"{d}|{mk[d]}|test", "routed_states_mean"),
               "test_k2_s_per_day": g(f"{d}|2|test", "engine_s_mean"),
               "test_k0_prompts_per_s": g(f"{d}|0|test", "prompts_per_s_mean"),
               "main_null_days": g(f"{d}|{mk[d]}|null", "n_days"),
               "main_null_s_per_day": g(f"{d}|{mk[d]}|null", "engine_s_mean"),
               "main_null_s_total": g(f"{d}|{mk[d]}|null", "engine_s_sum")} for d in HOSTS}
    total_h = sum(_num(r["engine_s_sum"]) for r in rc_rows if r["ds"] in HOSTS) / 3600
    return out, total_h


def habit_cost(speed_rows):
    """Habit-channel seconds per host-day at k2 from the frozen mean serving rate of each arm."""
    sp = {r["model"]: r for r in speed_rows}
    return {arm: {d: HABIT_K2_QUESTIONS[d][0] / HABIT_K2_QUESTIONS[d][1] / float(sp[arm]["infer_questions_per_s"])
                  for d in HOSTS} for arm in ("HB-L.mq", "HB-L.s0", "HB-L", "HB", "HB.head", "HB.08b", "HB-Z")
            if arm in sp}


# --------------------------------------------------------------------------- #
def paper_means(results_dir):
    """Every derived number of the README table from the results/ CSVs (missing inputs are skipped)."""
    from habitus.eval.tables import rows_from_csv
    R = Path(results_dir)
    abl = rows_from_csv((R / "habitus" / "ablation_e1_e2.csv").read_bytes())
    out = {"_doc": "derived by habitus.eval.derived.paper_means from the results/ CSVs (no label read)",
           "summary": summary_means(abl)}
    main_mcc = {d: out["summary"]["main_config"][d]["mcc"] for d in HOSTS}
    p = R / "baselines" / "pidsmaker_rows.csv"
    if p.exists():
        out["baselines"] = baselines(baseline_rows(read_csv(p)), main_mcc)
    dm_p, hb0_p = R / "extension_habit" / "decision_models_best_mcc.csv", R / "extension_habit" / "hb0_rows.csv"
    if dm_p.exists() and hb0_p.exists():
        summ, main = habit_summary(habit_cells(read_csv(dm_p), read_csv(hb0_p), abl))
        out["habit"] = summ
        out["habit_main_config"] = main
    rc_p = R / "habitus" / "route_cost.csv"
    if rc_p.exists():
        out["route_engine"], out["route_engine_5host_engine_h"] = route_engine(read_csv(rc_p))
    sp_p = R / "extension_habit" / "decision_models_speed_mean.csv"
    if sp_p.exists():
        out["habit_cost_s_per_host_day_k2"] = habit_cost(read_csv(sp_p))
    return out


def csv_text(rows, cols) -> str:
    """CSV text (csv module default dialect) of dict rows; floats are written as repr."""
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=list(cols), extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in cols})
    return buf.getvalue()
