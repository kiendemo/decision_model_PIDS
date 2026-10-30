#!/usr/bin/env python
"""One-off builder of results/ from the research and paper result files (read-only sources).

    python scripts/results/build_results.py [--research-results DIR] [--paper-derive DIR] [--out results]

Source roots default to settings.paths.research_results and settings.paths.paper_derive (the gitignored
configs/local.yaml); nothing is written under them. Every input is named file by file for the 5 hosts
(no glob over a parent directory). Transformations:
  - byte copies of the E1 / E2 ablation (CSV / md), the frozen decision-model CSVs and the label-free
    method-stats JSONs (each one checked unchanged by the scrubbers below);
  - CSVs converted from the editor JSONs (budget, per-attack recall, lambda / combiner and per-form
    sensitivity, form gaps, bootstrap), floats written as repr;
  - the rule comparators (rule-lex, rule-count) and the HB-C count back-off are stripped (rows, keys and
    bootstrap systems); the 3 HB-L.full selection rows are kept and flagged FROZEN / unsupported;
  - route_cost filtered to the 5 hosts; HB0 rows parsed from the stale confusion_reference.md (only the
    E2 / HB0 / baseline rows of that file are valid);
  - every written text is scrubbed: out-of-scope host names (repro.scan.redact_hosts), absolute server
    paths (made relative to the source tree root, or replaced), the former data-disk mount ('former data
    disk'), and then scanned (repro.scan.scan_paths / scan_hosts) - any remaining hit aborts;
  - README.md (paper-number table, the generated STATUS block, the oracle caveat), not_included.md,
    protocol/datasets.md, extension_habit/FROZEN.md and MANIFEST.sha256 (sha256 of every file, with its
    source path relative to the source tree root).
results/baselines/ is written by scripts/export_baselines.py (server only); this script indexes it.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
try:
    import habitus  # noqa: F401
except ImportError:                                    # run from a checkout without installing
    sys.path.insert(0, str(REPO / "src"))

HOSTS = ("CADETS_E3", "THEIA_E3", "CLEARSCOPE_E3", "optc_h501", "ATLASV2_EDR")
RULE_SYSTEMS = ("rule-lex", "rule-count")
HBC = "HB-C"
TOKEN_AUDIT = ("tok_ATLASV2_EDR_all_2022-07-19.json", "tok_CADETS_E3_2_2018-04-06.json",
               "tok_THEIA_E3_all_2018-04-10.json")
DM_CSVS = ("decision_models_best_mcc.csv", "decision_models_mcc_matrix_k2.csv",
           "decision_models_mcc_matrix_allc-pre.csv", "decision_models_speed_per_fold.csv",
           "decision_models_speed_mean.csv")
UNSUPPORTED_NOTE = "FROZEN/unsupported: no HB-L.full arm or config in this repo"
FORMER_DISK = re.compile(r"(?<![\w.~${}-])/D[A]TA[0-9]?(?:/[^\s`'\"(),;\]]*)?")
TMP_PATH = re.compile(r"(?<![\w.~${}-])/t[m]p/[^\s`'\"(),;\]]*")
HOME_PATH = re.compile(r"(?<![\w.~${}-])/h[o]me/[^\s`'\"(),;\]]*")
SCRATCH = re.compile(r"scratch[p]ad", re.I)


# --------------------------------------------------------------------------- #
# Writer with provenance
# --------------------------------------------------------------------------- #
class Out:
    def __init__(self, root, src_root):
        self.root, self.src_root, self.sources = Path(root), Path(src_root), {}

    def rel_src(self, p):
        p = Path(p)
        try:
            return p.resolve().relative_to(self.src_root.resolve()).as_posix()
        except ValueError:
            return p.name

    def put(self, rel, data, source):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, str):
            data = data.encode("utf-8")
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, p)
        self.sources[rel] = source
        return p


def scrubber(src_root):
    root = str(Path(src_root).resolve()).rstrip("/") + "/"
    from habitus.repro.scan import redact_hosts

    def scrub(text):
        text = text.replace(root, "")
        text = FORMER_DISK.sub("former data disk", text)
        text = TMP_PATH.sub("<temp path>", text)
        text = HOME_PATH.sub("<server path>", text)
        text = SCRATCH.sub("session scratch directory", text)
        return redact_hosts(text)
    return scrub


def csv_bytes(rows, cols, lineterminator="\r\n"):
    """CSV bytes, floats as repr; the default (CRLF) dialect is habitus.eval.derived.csv_text exactly."""
    if lineterminator == "\r\n":
        from habitus.eval.derived import csv_text
        return csv_text(rows, cols).encode("utf-8")
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=list(cols), lineterminator=lineterminator, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in cols})
    return buf.getvalue().encode("utf-8")


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        return rd.fieldnames, list(rd)


def load_json(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def json_bytes(obj, indent=1):
    return (json.dumps(obj, indent=indent, default=float) + "\n").encode("utf-8")


# --------------------------------------------------------------------------- #
# Pieces
# --------------------------------------------------------------------------- #
def copy_checked(out, scrub, src, rel, note="byte copy"):
    """Byte copy of a file that must already be clean (scrub leaves it unchanged)."""
    raw = Path(src).read_bytes()
    text = raw.decode("utf-8")
    if scrub(text) != text:
        raise SystemExit(f"{src}: needs scrubbing, refusing a byte copy for {rel}")
    out.put(rel, raw, f"{out.rel_src(src)} ({note})")


def ablation_md(src, scrub):
    """The research ablation md with its 'Generated <stamp> by ...' line replaced by
    habitus.eval.tables.provenance_line(<stamp>) (names `habitus eval tables` and ablation_e1_e2.csv);
    every other line must already be clean (scrub leaves it unchanged)."""
    from habitus.eval.tables import GENERATED_PREFIX, provenance_line
    text = Path(src).read_bytes().decode("utf-8")
    lines = text.split("\n")
    hits = [i for i, x in enumerate(lines) if x.startswith(GENERATED_PREFIX) and " by `" in x]
    if len(hits) != 1:
        raise SystemExit(f"{src}: expected one 'Generated <time> by' line, found {len(hits)}")
    i = hits[0]
    stamp = lines[i][len(GENERATED_PREFIX):lines[i].index(" by `")]
    rest = "\n".join(lines[:i] + lines[i + 1:])
    if scrub(rest) != rest:
        raise SystemExit(f"{src}: needs scrubbing, refusing a copy for habitus/ablation_e1_e2.md")
    lines[i] = provenance_line(stamp)
    return "\n".join(lines).encode("utf-8")


def hb0_rows(md_path):
    """HB0 rows of the per-host tables of confusion_reference.md (system | k | col | TP | FP | TN | FN | Prec |
    Rec | F1 | MCC | thr | tie block n (GT) | MCC ties- | AP | AUC | note)."""
    from habitus.eval.derived import mcc_counts

    def num(x):
        x = x.strip().replace(",", "").replace("**", "")
        if x in ("", "–", "-", "·"):
            return None
        try:
            return int(x)
        except ValueError:
            return float(x)
    rows, cur = [], None
    for line in Path(md_path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^## (\S+)$", line)
        if m:
            cur = m.group(1) if m.group(1) in HOSTS else None
            continue
        if cur is None or not line.startswith("| HB0 |"):
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        keys = ["system", "k", "col", "TP", "FP", "TN", "FN", "Prec", "Rec", "F1", "MCC", "thr", "tie", "ties", "AP",
                "AUC", "note"]
        r = dict(zip(keys, c))
        tie = re.match(r"(\d+) \((\d+)\)", r["tie"])
        tp, fp, tn, fn = (num(r[k]) for k in ("TP", "FP", "TN", "FN"))
        rows.append({"ds": cur, "system": "HB0", "k_plan": r["k"], "col": r["col"], "tp": tp, "fp": fp, "tn": tn,
                     "fn": fn, "precision": num(r["Prec"]), "recall": num(r["Rec"]), "f1": num(r["F1"]),
                     "mcc": num(r["MCC"]), "mcc_recomputed": round(mcc_counts(tp, fp, tn, fn), 6),
                     "threshold": num(r["thr"]), "tie_n": int(tie.group(1)) if tie else None,
                     "tie_gt": int(tie.group(2)) if tie else None, "mcc_ties_neg": num(r["ties"]),
                     "ap": num(r["AP"]), "auc": num(r["AUC"]), "note": r["note"]})
    return rows


HB0_COLS = ("ds", "system", "k_plan", "col", "tp", "fp", "tn", "fn", "precision", "recall", "f1", "mcc",
            "mcc_recomputed", "threshold", "tie_n", "tie_gt", "mcc_ties_neg", "ap", "auc", "note")
ROUTE_COLS = ("key", "ds", "k", "role", "n_days", "days", "engine_s_mean", "engine_s_sum", "engine_s_min",
              "engine_s_max", "state_s_mean", "routed_states_mean", "prompts_total_mean", "forwarded_mean",
              "prompts_per_s_mean")


def selection_rows(src):
    cols, rows = read_rows(src)
    kept = []
    for r in rows:
        if r["model"] == HBC:
            continue
        if r["model"] == "HB-L.full":
            r = dict(r, note=UNSUPPORTED_NOTE + ("; " + r["note"] if r["note"] else ""))
        kept.append(r)
    return cols, kept, len(rows) - len(kept)


ARCHIVAL = ("> ARCHIVAL session record of the research server, kept as provenance only (paths scrubbed). It may cite "
            "files, hardware and runs that are not part of this repository; nothing here reads it.\n\n")


def archival(text):
    """A session record with the archival banner first."""
    return ARCHIVAL + text


def redact_arms_pilot(text):
    lines = [ln for ln in text.split("\n") if not ln.startswith(f"| {HBC} |")]
    head = ("> Redacted copy for the habitus-dm repository: the HB-C (count back-off) table rows are removed; "
            "HB-C is not part of this repository. The tail selector described below (count back-off p) is "
            "the one M1 / M2 used, which is why M1 / M2 cannot be recomputed here.\n\n")
    return head + "\n".join(lines)


def frozen_md(n_sel_removed):
    return "\n".join([
        "# HABITUS+H extension: FROZEN results", "",
        "Everything in this directory except `hb0_rows.csv` (reproduced by gate G11) is a frozen record. The "
        "habit-channel adapters, Laya run directories, ensemble temperatures, habit nulls, habit score files and "
        "habit answer caches lived on the former data disk and are unavailable. They are being retrained; "
        "retrained numbers will differ (seeds, bf16).", "",
        "- `decision_models_best_mcc.csv` (126 rows) is the authoritative source of the 7 detection-scored arms "
        "(HB-Z, HB, HB.head, HB.08b, HB-L.s0, HB-L, HB-L.mq; composite and habit-only, k2 and allc-pre). It "
        "supersedes the partial arm rows of the stale confusion_reference.md (2026-10-22).",
        "- `decision_models_mcc_matrix_k2.csv`, `decision_models_mcc_matrix_allc-pre.csv`: 4-dp MCC matrices.",
        "- `decision_models_speed_per_fold.csv`, `decision_models_speed_mean.csv`: training / serving speed "
        "(RTX A5000, bf16); the GiB columns are constants measured once in the GPU smoke run.",
        f"- `selection_metrics_3folds.csv`: label-free selection metrics M1 / M2 / M7 / M3 over 3 LOHO folds "
        f"(the ATLASV2_EDR fold is missing). The HB-C rows ({n_sel_removed}) are removed. The 3 HB-L.full rows "
        "are kept and flagged FROZEN/unsupported: this repository has no HB-L.full arm or config. M1 / M2 used "
        "the HB-C count back-off as the tail selector (arms_pilot_loho-optc_h501.md), so they cannot be "
        "recomputed here.",
        "- `hb0_rows.csv`: HB0 (lineage-split decision layer over the frozen E2 routes) rows parsed from "
        "confusion_reference.md. REPRODUCED on CPU: `habitus decide hb0` + `habitus eval habit --levels HB0` "
        "recompute all 9 rows exactly, every column (gate G11; needs the role-table catalogs of "
        "`scripts/fetch_catalogs.sh`, whose table must match ROLE_TABLE_SHA256).",
        "- `run_facts_20261029.md`, `arms_pilot_loho-optc_h501.md`, `arms_smoke.md`: ARCHIVAL session records "
        "of the research server, kept as provenance (HB-C table rows removed, the former data-disk paths "
        "replaced, an archival banner added). They may cite files and hardware that are not part of this "
        "repository; nothing here reads them.",
        "- `provenance/make_csv.py`: the generator of the decision-model CSVs, kept as provenance only (it read "
        "the reports on the former data disk and cannot run).", ""])


def _fmt3(x):
    return "–" if x is None else f"{x:.3f}"


def readme(abl_rows, summ, budgets, states, boot_summary, base_rows, rc_rows, hb0, matrix, n_sel, status_block,
           have_baselines):
    from habitus.eval.derived import mcc_counts
    mk = summ["main_k"]
    main = summ["main_config"]
    main_cells = ", ".join(f"{d.split('_')[0]} {_fmt3(main[d]['mcc'])} ({main[d]['tp']}/{main[d]['fp']})" for d in HOSTS)
    k2 = summ["habitus_k2"]["cells"]
    tneg = summ["main_ties_neg_cells"]

    def bud(ds):
        p = budgets[ds]["plans"][mk[ds]]["budget"]
        return (f"{ds.split('_')[0]} " + "/".join(f"{p['rows'][b]['tp']}:{p['rows'][b]['fp']}" for b in ("0", "1", "20"))
                + f" ({p['n_null_days']} null days)")
    d = boot_summary["boot_deltas"]["HABITUS main - best baseline"]["mean5"]
    rc = {r["key"]: r for r in rc_rows}
    eng = ", ".join(f"{ds.split('_')[0]} {float(rc[f'{ds}|{2 if ds == 'CADETS_E3' else 'all'}|test']['engine_s_mean']):.1f}"
                    for ds in HOSTS)
    pps = [float(rc[f"{ds}|all|test"]["prompts_per_s_mean"]) for ds in HOSTS]
    hb0k2 = next(r for r in hb0 if r["ds"] == "CADETS_E3" and r["k_plan"] == "k2")
    means = {(r["model"], r["score"]): r["mean"] for r in matrix["k2"]}
    meansa = {(r["model"], r["score"]): r["mean"] for r in matrix["allc-pre"]}
    arms = ("HB-Z", "HB", "HB.head", "HB.08b", "HB-L.s0", "HB-L", "HB-L.mq")
    n_base = len(base_rows)
    sizes = ", ".join(f"{ds.split('_')[0]} {main[ds]['gt_total']} GT / {r['n_universe']:,}" for ds in HOSTS
                      for r in abl_rows if r["ds"] == ds and r["main_config"] == "yes")
    form_rows = sum(len(s["forms"]) for s in states.values())
    cells = [(ds, int(B), r) for ds in HOSTS for B, r in budgets[ds]["plans"][mk[ds]]["budget"]["rows"].items()]
    n_miss = sum(1 for _ds, B, r in cells if r["fp_per_day"] > B)
    sens_rows = len(next(iter(budgets.values()))["plans"]["k2"]["sensitivity"])
    T = [
        "| Paper item | Numbers | File | Command | Status |",
        "|---|---|---|---|---|",
        f"| tab:summary / tab:main HABITUS main | {main_cells}; mean {summ['main_mean_mcc']:.3f} | "
        "`habitus/main_config.csv`, `habitus/summary_means.json` | `habitus reproduce --level tables` (exact replay: "
        "`--level replay`) | REPRODUCED, G1 + G4 |",
        "| tab:summary k2 on every host | " + "/".join(_fmt3(k2[x]) for x in HOSTS)
        + f" ({summ['habitus_k2']['mean_mcc']:.3f}) | `habitus/ablation_e1_e2.csv` (E2 k2 composite) | "
        "`habitus reproduce --level tables` | REPRODUCED, G1 |",
        "| tab:summary ties negative | " + "/".join(_fmt3(tneg[x]) for x in HOSTS)
        + f" ({summ['main_mean_mcc_ties_neg']:.3f}) | `habitus/ablation_e1_e2.csv` (mcc_ties_neg) | "
        "`habitus reproduce --level tables` | REPRODUCED, G1 |",
        "| tab:ablation-state | L0..L4 and E2 at k0 / k2 / kall; mean4 / mean5 | `habitus/ablation_e1_e2.csv` (E1 rows), "
        "`habitus/summary_means.json` | `habitus reproduce --level tables` (+ replay of every shipped E1 dir) | "
        "REPRODUCED, G1 / G4; CLEARSCOPE_E3 L0 / L1 / L3 at k>=1 and L0 / L1 at k0 never run |",
        "| tab:ablation-routes | qv / qt / base / composite / 1-qa / acq_all | `habitus/ablation_e1_e2.csv` | "
        "`habitus reproduce --level tables` | REPRODUCED, G1 |",
        "| tab:ties | GT in head " + ", ".join(f"{main[x]['gt_in_head']}/{main[x]['gt_total']}" for x in HOSTS)
        + "; tie blocks " + ", ".join(f"{main[x]['tie_n']} ({main[x]['tie_gt']})" for x in HOSTS)
        + " | `habitus/ablation_e1_e2.csv` (gt_in_head, tie_n, tie_gt) | `habitus reproduce --level tables` | "
        "REPRODUCED, G1 |",
        "| tab:budget | B = 0 / 1 / 20 TP:FP: " + "; ".join(bud(x) for x in HOSTS)
        + " | `habitus/budget_rows.csv`, `habitus/per_attack_recall.csv` | `habitus reproduce --level operating` | "
        "REPRODUCED, G9 |",
        f"| tab:sensitivity lambda / combiner | {sens_rows} rows x 5 hosts | `habitus/sensitivity_composition.csv` | "
        "`habitus reproduce --level operating` | REPRODUCED, G9 |",
        f"| tab:sensitivity forms | form1, form2, single forms ({form_rows} rows); recon_max_abs 0, cache misses 0 | "
        "`habitus/sensitivity_forms.csv`, `habitus/form_gaps.csv` | `habitus reproduce --level operating` (cards + "
        "tokenizer + precedent cache) | REPRODUCED, G9 |",
        f"| tab:uncertainty HABITUS vs best baseline | mean delta {d['point']:+.3f} [{d['ci95'][0]:+.3f}, "
        f"{d['ci95'][1]:+.3f}] p={d['p_le_0']:.3f} | `habitus/bootstrap_mcc.csv`, `habitus/bootstrap_draws/` | "
        "`habitus reproduce --level operating` (release baselines npz) | REPRODUCED, G9; rule columns not included |",
        f"| tab:main / summary PIDSMaker rows | {n_base} rows; best_adp / epoch-11 / better-of-two | "
        "`baselines/pidsmaker_rows.csv` | `python scripts/export_baselines.py` (server only) | "
        + ("FROZEN for external users; regenerable on this server |" if have_baselines else "NOT YET EXPORTED |"),
        f"| tab:efficiency route engine | {eng} s per test day; {min(pps):.1f}-{max(pps):.1f} prompts/s | "
        "`habitus/route_cost.csv` | none (derived from the route-engine queue logs) | FROZEN |",
        f"| tab:datasets | GT / universe: {sizes} | `habitus/main_config.csv`, `habitus/route_stats.json` | "
        "`habitus reproduce --level tables` | REPRODUCED (sizes), G1 |",
        "| tab:habit HABITUS rows | k2 " + "/".join(_fmt3(k2[x]) for x in HOSTS) + " | `habitus/ablation_e1_e2.csv` | "
        "`habitus reproduce --level tables` | REPRODUCED, G1 |",
        f"| tab:habit HB0 | e.g. k2 CADETS {round(mcc_counts(hb0k2['tp'], hb0k2['fp'], hb0k2['tn'], hb0k2['fn']), 3):.3f} "
        f"({hb0k2['tp']}/{hb0k2['fp']}) | `extension_habit/hb0_rows.csv` | `habitus decide hb0` + "
        "`habitus eval habit --levels HB0` (`habitus gates --only G11`; needs the role-table catalogs, "
        "`scripts/fetch_catalogs.sh`) | REPRODUCED, G11 |",
        "| tab:habit HB-Z / HB / HB.head / HB.08b / HB-L.s0 / HB-L / HB-L.mq | k2 means "
        + "/".join(str(means[(a, "composite")]) for a in arms) + "; allc-pre "
        + "/".join(str(meansa[(a, "composite")]) for a in arms)
        + " | `extension_habit/decision_models_best_mcc.csv`, `extension_habit/decision_models_mcc_matrix_k2.csv`, "
        "`extension_habit/decision_models_mcc_matrix_allc-pre.csv` | none (frozen) | FROZEN, retraining |",
        f"| tab:selection | M1 / M2 / M7 / M3 over 3 folds ({n_sel} rows) | `extension_habit/selection_metrics_3folds.csv` | "
        "none | FROZEN (ATLASV2_EDR fold missing; M1 / M2 use the HB-C selector; HB-L.full unsupported) |",
        "| tab:efficiency habit rows | GPU-h, q/s, p50 / p95, GiB | `extension_habit/decision_models_speed_per_fold.csv`, "
        "`extension_habit/decision_models_speed_mean.csv` | none | "
        "FROZEN |",
        "| tab:rules, rule-lex / rule-count, probes, HB-C | – | `not_included.md` | – | EXCLUDED by framing |",
    ]
    L = ["# HABITUS results", "",
         "Every number of the HABITUS paper that this repository ships, the file it lives in, the command that "
         "recomputes it and its status. Paths below are relative to this directory.", "",
         "**Metric.** Node-level confusion (TP / FP / TN / FN, precision, recall, F1) and the Matthews correlation "
         "coefficient MCC = (TP·TN − FP·FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)) (0 when a margin is empty). "
         "Universe = every test-day node plus the ground-truth nodes without a test-day card (always FN); node "
         "score = max over the test days; signatures outside the decision head get the rarity fallback "
         "(< 0, below every decision score). A tie block at the threshold counts positive; the ties-negative "
         "MCC is reported alongside.", "",
         "**Oracle threshold.** Every MCC here is at the threshold that maximises MCC on the TEST labels, for "
         "every system alike: an upper bound, not an operating point. The label-free clean-day bars miss their "
         f"FP budget (realised FP per test day > B) in {n_miss} of {len(cells)} cells (tab:budget, "
         "habitus/budget_rows.csv).", "",
         "**Main configuration.** E2 composite with all pre-deployment clean days as the reference (k = all) on "
         "THEIA_E3, CLEARSCOPE_E3, optc_h501 and ATLASV2_EDR, and k2 on CADETS_E3: its k = all reference includes "
         "clean days after its first test day (04-07..04-10 vs 04-06), so that row is flagged oracle-k and never "
         "used as the main one.", "",
         "**Development hosts.** All 5 hosts are design-informed development hosts: E2's routing and name-free "
         "evidence were chosen after a label-reading diagnosis on THEIA_E3 (docs/DESIGN.md). These are not "
         "held-out numbers.", "",
         "## Paper number -> file -> command -> status", ""] + T + [
         "", "REPRODUCED = recomputed from the release cache bundle by the named gate (G1: tables, G4: exact replay "
         "of every score file from the answer cache, G9: operating tables, G11: the HB0 rows). FROZEN = a record of a run whose "
         "inputs are gone or server-only. Full precision lives in the CSVs; the paper rounds to 3 dp.", "",
         "## Decision-model status", "", status_block, "",
         "## Layout", "",
         "- `habitus/`: the main system (jev stock Qwen3.5-4B on the routed QV / QT / QA questions).",
         "- `baselines/`: PIDSMaker 2289cd9 baseline rows (server export) and their provenance.",
         "- `extension_habit/`: HABITUS+H, frozen (see `extension_habit/FROZEN.md`).",
         "- `protocol/datasets.md`: the 5 hosts (rendered from configs/datasets.yaml).",
         "- `not_included.md`: comparators excluded by the decision-model framing, with their paper numbers.",
         "- `MANIFEST.sha256`: sha256 of every file here (`sha256sum -c MANIFEST.sha256` from this directory), "
         "with the source of each file as a comment.", ""]
    return "\n".join(L)


def not_included(P, R, rel, main_mean):
    ed = load_json(P / "editor" / "editor_cpu.json")["summary"]["boot_deltas"]
    st = {ds: load_json(P / "editor" / f"state_{ds}.json") for ds in HOSTS}
    zd = load_json(P / "zeroshot_derived.json")

    def per(rule):
        return {ds: st[ds]["rules"][rule]["mcc"] for ds in HOSTS}
    lex, cnt = per("rule-lex"), per("rule-count")
    main_cadets = load_json(P / "editor" / "budget_CADETS_E3.json")["plans"]["k2"]["oracle"]["mcc"]
    dl, dc = ed["HABITUS main - rule-lex"]["mean5"], ed["HABITUS main - rule-count"]["mean5"]
    al, ra = zd["probe_allowlist_main"], zd["probe_rarity_probe_main"]
    hbc_c, hbc_h = zd["habit_main_config"]["HB-C|composite"], zd["habit_main_config"]["HB-C|habit-only"]

    def cells(d):
        return ", ".join(f"{ds} {d[ds]:.3f}" for ds in HOSTS)
    src_state = rel(P / "editor" / "state_CADETS_E3.json").replace("CADETS_E3", "<DS>")
    L = ["# Not included in this repository", "",
         "This repository frames HABITUS as a decision-model pipeline. Rule COMPARATORS are therefore not part of "
         "it: no code, no rows, no bootstrap systems. The paper's own numbers for them are cited below, because "
         "one of them is the paper's key negative finding. (The pipeline itself does contain fixed, a-priori "
         "gating and fallback rules - the head selection rule, the rarity fallback, the hard-tier acquittal "
         "exemption, the HB0 lineage split - none learned or label-tuned; see the repository README.)", "",
         "## rule-lex: ties HABITUS", "",
         "A no-LLM rule over the same compiled state, written after the HABITUS results were known and not "
         "tuned: rule-lex = (4·[hard tier] + 2·[identity_unseen] + n_dev/(1+n_dev) + n_ev/(1+n_ev)) / 8.", "",
         f"- Oracle node MCC: {cells(lex)}; mean {sum(lex.values()) / 5:.3f}. HABITUS main: mean {main_mean:.3f}.",
         f"- **rule-lex ties HABITUS on the 5-host mean**: delta HABITUS − rule-lex = {dl['point']:+.3f} "
         f"[{dl['ci95'][0]:+.3f}, {dl['ci95'][1]:+.3f}] (paired stratified entity bootstrap, B = 2000), and it "
         f"beats HABITUS on CADETS_E3 ({lex['CADETS_E3']:.3f} vs {main_cadets:.3f}).",
         f"- Source: `{src_state}` (rules), `{rel(P / 'editor' / 'editor_cpu.json')}` (summary.boot_deltas); "
         "generator `paper/habitus/derive/editor_cpu.py state / boot / merge`.", "",
         "## rule-count", "",
         "rule-count = (n_dev + n_ev) / (1 + n_dev + n_ev) over the same state.", "",
         f"- Oracle node MCC: {cells(cnt)}; mean {sum(cnt.values()) / 5:.3f}; delta HABITUS − rule-count = "
         f"{dc['point']:+.3f} [{dc['ci95'][0]:+.3f}, {dc['ci95'][1]:+.3f}].", "",
         "## Label-free probes (allowlist, rarity)", "",
         f"- allowlist (name / kappa unseen on the reference days), main configuration: {cells(al['cells'])}; mean "
         f"{al['mean']:.3f}.",
         f"- rarity probe (signature rarity vs the reference days): {cells(ra['cells'])}; mean {ra['mean']:.3f}.",
         f"- Source: `{rel(P / 'zeroshot_derived.json')}` (probe_*_main), from the probe rows of "
         f"`{rel(R / 'confusion_reference.md')}`.", "",
         "## HB-C (count back-off) and tab:rules", "",
         f"- HB-C = HB0 + a habit channel from a Laplace count back-off over the fold's training items (no "
         f"model). Main-configuration mean MCC: composite {hbc_c:.4f}, habit-only {hbc_h:.4f} "
         f"(`{rel(P / 'zeroshot_derived.json')}` habit_main_config).",
         "- Its selection-metric rows are removed from `extension_habit/selection_metrics_3folds.csv`. Because "
         "HB-C's count back-off is the tail selector of M1 / M2, those two metrics cannot be recomputed here.",
         "- tab:rules (X2, R-L2) is defined against HB-C and is omitted.", ""]
    return "\n".join(L)


def datasets_md():
    from habitus.hosts import spec
    L = ["# Datasets (rendered from configs/datasets.yaml)", "",
         "The 5 in-scope hosts. Day lists are frozen (usable clean days from the transformed-graph byte mass); GT "
         "files are listed in config order (attack groups follow it).", ""]
    for ds in HOSTS:
        s = spec(ds)
        L += [f"## {ds}", "",
              f"- os {s.os}; database {s.db_name}",
              f"- test days: {', '.join(s.test)} (configured: {', '.join(s.configured_test)})",
              f"- clean days: {', '.join(s.clean)}; pre-deployment: {', '.join(s.predeploy)}; inner day {s.inner}; "
              f"k2 reference {', '.join(s.k2)}",
              f"- main configuration k = {s.main_k}; k0 nulls: {'yes' if s.k0_nulls else 'no'}; E1 levels "
              + "; ".join(f"{k}: {', '.join(v)}" for k, v in s.e1_levels.items()),
              f"- GT files: {', '.join(s.gt_files)}",
              f"- test-day nodes {s.test_day_nodes:,}; evaluation universe {s.n_universe:,}; GT {s.gt_total} "
              f"({s.gt_in_universe} with a test-day card); card days {s.all_card_days}", ""]
    return "\n".join(L)


def manifest(out):
    lines = ["# sha256 of every file under results/ (verify: cd results && sha256sum -c MANIFEST.sha256).",
             "# Each hash line follows a comment naming its source (paths relative to the source tree root)."]
    for p in sorted(x for x in out.root.rglob("*") if x.is_file() and x.name != "MANIFEST.sha256"
                    and not x.name.endswith(".tmp")):
        rel = p.relative_to(out.root).as_posix()
        lines.append(f"# {rel} <- {out.sources.get(rel, 'pre-existing (not written by this run)')}")
        lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {rel}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
def build(R, P, out_dir, src_root=None):
    from habitus.dm.status import block
    from habitus.eval import bootstrap as BT
    from habitus.eval import operating as OP
    from habitus.eval.derived import read_csv, summary_means
    from habitus.eval.tables import main_rows, rows_from_csv, write_csv
    from habitus.repro.scan import scan_hosts, scan_paths
    R, P = Path(R), Path(P)
    src_root = Path(src_root) if src_root else Path(os.path.commonpath([str(R.resolve()), str(P.resolve())]))
    out = Out(out_dir, src_root)
    scrub = scrubber(src_root)
    rel = out.rel_src

    # ---- habitus/: the E1 / E2 ablation ----
    copy_checked(out, scrub, R / "zeroshot_ablation.csv", "habitus/ablation_e1_e2.csv")
    out.put("habitus/ablation_e1_e2.md", ablation_md(R / "zeroshot_ablation.md", scrub),
            f"{rel(R / 'zeroshot_ablation.md')} (provenance line rewritten by habitus.eval.tables.provenance_line; "
            "every other byte copied)")
    abl_raw = (R / "zeroshot_ablation.csv").read_bytes()
    abl = rows_from_csv(abl_raw)
    _cols, raw_rows = read_rows(R / "zeroshot_ablation.csv")
    mc = write_csv([r for r in raw_rows if r["main_config"] == "yes"])
    assert mc == write_csv(main_rows(abl)), "main_config rows differ between the string and typed writers"
    out.put("habitus/main_config.csv", mc, f"{rel(R / 'zeroshot_ablation.csv')} (rows main_config=yes, original writer)")
    summ = summary_means(abl)
    out.put("habitus/summary_means.json", json_bytes(summ), f"{rel(R / 'zeroshot_ablation.csv')} "
            "(habitus.eval.derived.summary_means)")

    # ---- habitus/: operating tables from the editor JSONs ----
    budgets = {ds: load_json(P / "editor" / f"budget_{ds}.json") for ds in HOSTS}
    states = {}
    for ds in HOSTS:
        s = load_json(P / "editor" / f"state_{ds}.json")
        states[ds] = {k: s[k] for k in ("ds", "k", "recon_max_abs", "cache_misses", "forms", "gaps")}
    b, a, sn = OP.budget_tables(budgets)
    f, g = OP.forms_tables(states)
    bsrc = rel(P / "editor" / "budget_CADETS_E3.json").replace("CADETS_E3", "<DS>")
    ssrc = rel(P / "editor" / "state_CADETS_E3.json").replace("CADETS_E3", "<DS>")
    out.put("habitus/budget_rows.csv", csv_bytes(b, OP.BUDGET_COLS), f"{bsrc} (plans.*.budget, repr floats)")
    out.put("habitus/per_attack_recall.csv", csv_bytes(a, OP.ATTACK_COLS),
            f"{bsrc} (oracle_by_attack, budget_by_attack, repr floats)")
    out.put("habitus/sensitivity_composition.csv", csv_bytes(sn, OP.SENS_COLS), f"{bsrc} (sensitivity, repr floats)")
    out.put("habitus/sensitivity_forms.csv", csv_bytes(f, OP.FORM_COLS), f"{ssrc} (forms; rules dropped, repr floats)")
    out.put("habitus/form_gaps.csv", csv_bytes(g, OP.GAP_COLS), f"{ssrc} (gaps, repr floats)")

    # ---- habitus/: bootstrap (rule systems stripped; draws of the others unchanged) ----
    boots = {}
    for ds in HOSTS:
        bo = BT.strip_systems(load_json(P / "editor" / f"boot_{ds}.json"), RULE_SYSTEMS)
        boots[ds] = bo
        out.put(f"habitus/bootstrap_draws/boot_{ds}.json", json_bytes(bo),
                f"{rel(P / 'editor' / f'boot_{ds}.json')} (rule-lex / rule-count systems stripped)")
    bsum = BT.merge(boots)
    ref = load_json(P / "editor" / "editor_cpu.json")["summary"]
    for name, per in bsum["boot_deltas"].items():
        if ref["boot_deltas"].get(name) != per:
            raise SystemExit(f"bootstrap delta {name!r} differs from editor_cpu.json")
    for ds in HOSTS:
        want = {k: v for k, v in ref["boot_points"][ds].items() if k not in RULE_SYSTEMS}
        if want != bsum["boot_points"][ds]:
            raise SystemExit(f"bootstrap points of {ds} differ from editor_cpu.json")
    out.put("habitus/bootstrap_mcc.csv", csv_bytes(BT.csv_rows(bsum), BT.CSV_COLS),
            f"{rel(P / 'editor' / 'editor_cpu.json')} (summary without the rule pairs; recomputed by "
            "habitus.eval.bootstrap.merge from the stripped draws and checked equal)")

    # ---- habitus/: route cost and label-free method stats ----
    rc = load_json(P / "route_cost.json")
    rc_rows = []
    for key in sorted(rc):
        ds, k, role = key.split("|")
        if ds not in HOSTS:
            continue
        v = rc[key]
        rc_rows.append({"key": key, "ds": ds, "k": k, "role": role, **{c: v.get(c) for c in ROUTE_COLS[4:]},
                        "days": ";".join(v["days"])})
    out.put("habitus/route_cost.csv", csv_bytes(rc_rows, ROUTE_COLS),
            f"{rel(P / 'route_cost.json')} (the 5 hosts' keys only, repr floats)")
    rs = load_json(R / "method_stats" / "routestats.json")
    rs = {k: v for k, v in rs.items() if k in HOSTS}
    out.put("habitus/route_stats.json", json_bytes(rs), f"{rel(R / 'method_stats' / 'routestats.json')} (5 hosts)")
    for name in TOKEN_AUDIT:
        copy_checked(out, scrub, R / "method_stats" / name, f"habitus/token_audit/{name}")
    copy_checked(out, scrub, R / "method_stats" / "checka_dev.json", "habitus/check_a_dev.json")
    copy_checked(out, scrub, R / "method_stats" / "example_state_ATLASV2_EDR_2022-07-19.json",
                 "habitus/example_state_ATLASV2_EDR_2022-07-19.json")

    # ---- extension_habit/ (frozen) ----
    dm_dir = R / "decision_models"
    for name in DM_CSVS:
        cols, rows = read_rows(dm_dir / name)
        if any(r.get("model") == HBC for r in rows):
            raise SystemExit(f"{name}: unexpected HB-C rows")
        copy_checked(out, scrub, dm_dir / name, f"extension_habit/{name}")
    cols, sel, n_removed = selection_rows(R / "selection_metrics_3folds.csv")
    out.put("extension_habit/selection_metrics_3folds.csv", csv_bytes(sel, cols, lineterminator="\n"),
            f"{rel(R / 'selection_metrics_3folds.csv')} (HB-C rows removed, HB-L.full flagged)")
    hb0 = hb0_rows(R / "confusion_reference.md")
    out.put("extension_habit/hb0_rows.csv", csv_bytes(hb0, HB0_COLS),
            f"{rel(R / 'confusion_reference.md')} (HB0 rows of the per-host tables)")
    for name in ("run_facts_20261029.md", "arms_smoke.md"):
        out.put(f"extension_habit/{name}", archival(scrub((R / name).read_text(encoding="utf-8"))),
                f"{rel(R / name)} (paths scrubbed, archival banner)")
    out.put("extension_habit/arms_pilot_loho-optc_h501.md",
            archival(scrub(redact_arms_pilot((R / "arms_pilot_loho-optc_h501.md").read_text(encoding="utf-8")))),
            f"{rel(R / 'arms_pilot_loho-optc_h501.md')} (HB-C rows removed, paths scrubbed, archival banner)")
    out.put("extension_habit/provenance/make_csv.py", scrub((dm_dir / "make_csv.py").read_text(encoding="utf-8")),
            f"{rel(dm_dir / 'make_csv.py')} (provenance only, paths scrubbed)")
    out.put("extension_habit/FROZEN.md", frozen_md(n_removed), "generated")

    # ---- top level ----
    base_p = out.root / "baselines" / "pidsmaker_rows.csv"
    base_rows = read_csv(base_p) if base_p.exists() else []
    matrix = {kp: read_csv(dm_dir / f"decision_models_mcc_matrix_{kp}.csv") for kp in ("k2", "allc-pre")}
    out.put("README.md", readme(abl, summ, budgets, states, bsum, base_rows, rc_rows, hb0, matrix, len(sel), block(),
                                base_p.exists()), "generated (numbers from the files above; STATUS block from "
            "habitus.dm.status)")
    out.put("not_included.md", not_included(P, R, rel, summ["main_mean_mcc"]), "generated (cites the paper's derivation files)")
    out.put("protocol/datasets.md", datasets_md(), "generated from habitus.hosts (configs/datasets.yaml)")
    for p in sorted((out.root / "baselines").glob("*")) if (out.root / "baselines").is_dir() else []:
        out.sources.setdefault(f"baselines/{p.name}", "artifacts_upstream (python scripts/export_baselines.py)")

    # ---- scan everything, then the manifest ----
    files = [p for p in out.root.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256"]
    hits = scan_paths(files, root=out.root) + scan_hosts(files, root=out.root)
    if hits:
        raise SystemExit("results/ scan hits:\n" + "\n".join(str(h) for h in hits[:40]))
    out.put("MANIFEST.sha256", manifest(out), "generated")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--research-results", default=None, help="default: settings.paths.research_results")
    ap.add_argument("--paper-derive", default=None, help="default: settings.paths.paper_derive")
    ap.add_argument("--out", default=str(REPO / "results"))
    a = ap.parse_args(argv)
    from habitus.config import settings
    st = settings()
    R = a.research_results or st.paths.research_results
    P = a.paper_derive or st.paths.paper_derive
    if not R or not P:
        raise SystemExit("research_results / paper_derive are not set (configs/local.yaml, server only)")
    out = build(R, P, a.out)
    print(f"wrote {len(out.sources)} files under {out.root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
