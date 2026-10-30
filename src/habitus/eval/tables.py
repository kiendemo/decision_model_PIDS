"""The E1 / E2 ablation table at the oracle best-MCC threshold (CPU only).

    from habitus.eval import tables
    rows = tables.run_host("ATLASV2_EDR", root=scores_root, source="nodeindex")
    csv_bytes, md_text = tables.merge(rows_of_all_hosts)
    tables.write_tables(rows_of_all_hosts, out_dir)   # ablation_e1_e2.csv / .md, main_config.csv, summary_means.json

Reads the frozen score files <root>/<ds>/<level>_k<k>/<day>.json.gz (E1 levels L0-L4 = the single
unrouted state ladder; E2 = routed QV/QT/QA, name-free evidence, bounded acquittal) and computes the
node-level best-MCC confusion on the HABITUS universe (every test-day node + ground-truth nodes without a
test-day card = FN; node score = max over test days; non-head nodes = the rarity fallback below every
engine score) with eval.metrics.confusion.

The CSV writer is the original one, byte for byte: csv.DictWriter default dialect (CRLF line endings),
floats written as round(x, 6), None as '', rows in HOSTS order. The markdown is the original text too;
its 'Generated <time>' line is the only part that depends on when it runs (`generated=`). Its
'loader identical' column named the original's second loader (the same load-and-project functions); this
repo has one loader, so the column is True for every present row unless `xcheck` says otherwise.
"""

from __future__ import annotations

import csv
import io
import json
import os
import sys
import time
from pathlib import Path

HOSTS = ("CADETS_E3", "THEIA_E3", "CLEARSCOPE_E3", "optc_h501", "ATLASV2_EDR")
LEVELS = ("L0", "L1", "L2", "L3", "L4", "E2")
KS = (0, 2, None)
KNAME = {0: "k0", 2: "k2", None: "kall"}
E1_FIELDS = ("composite", "qv", "qt")
E2_FIELDS = ("composite", "base", "qv", "qt", "1-qa", "acq_all")
# published reference numbers (MCC, TP, FP) of the E2 composite rows: the REF self-check
REF = {("CADETS_E3", 2): (0.4055, 15, 4), ("THEIA_E3", 2): (0.5572, 59, 36), ("CLEARSCOPE_E3", 2): (0.0855, 37, 4313),
       ("optc_h501", 2): (0.4192, 713, 3141), ("ATLASV2_EDR", 2): (0.4048, 21, 35),
       ("CADETS_E3", None): (0.3922, 12, 1), ("THEIA_E3", None): (0.5956, 60, 26),
       ("CLEARSCOPE_E3", None): (0.1121, 39, 2831), ("optc_h501", None): (0.3929, 714, 3684),
       ("ATLASV2_EDR", None): (0.4052, 24, 49)}
# frozen E1 ladder AP (composite) for a spot check
E1_AP_REF = {("CADETS_E3", "L1", "k0", "composite"): 0.172, ("CADETS_E3", "L2", "kall", "composite"): 0.199,
             ("THEIA_E3", "L2", "k2", "composite"): 0.355, ("THEIA_E3", "L4", "kall", "composite"): 0.081,
             ("ATLASV2_EDR", "L3", "k0", "composite"): 0.129, ("ATLASV2_EDR", "L4", "k2", "composite"): 0.214,
             ("optc_h501", "L3", "kall", "composite"): 0.109}
BY_DESIGN = {("L2", 0): "by design: at k=0 L2 = L1 + reference stub {days:0} (SPEC 4); not run",
             ("L4", 0): "by design: at k=0 L4 = L3 + reference stub (SPEC 4); not run"}
CADETS_KALL = "CADETS k=all reference includes clean days 04-07..04-10, after the first test day 04-06 (oracle-ish)"
COLS = ["ds", "system", "level", "k", "column", "main_config", "flags", "tp", "fp", "tn", "fn", "precision",
        "recall", "f1", "mcc", "auc", "ap", "threshold", "alerts", "tie_n", "tie_gt", "mcc_ties_neg",
        "n_universe", "test_day_nodes", "gt_total", "gt_without_card", "gt_in_head", "n_test_days", "note"]
INT_COLS = ("tp", "fp", "tn", "fn", "alerts", "tie_n", "tie_gt", "n_universe", "test_day_nodes", "gt_total",
            "gt_without_card", "gt_in_head", "n_test_days")
FLOAT_COLS = ("precision", "recall", "f1", "mcc", "auc", "ap", "threshold", "mcc_ties_neg")
# the shipped markdown's generation time (results/habitus/ablation_e1_e2.md)
REF_GENERATED = "2026-10-29 18:45:14"
GENERATED_PREFIX = "Generated "


def _log(msg):
    try:
        from habitus.log import log
        log(msg, "eval.tables")
    except ImportError:
        print(f"[eval.tables] {msg}", file=sys.stderr, flush=True)


def parse_k(v):
    """HostSpec.main_k ('2', 'all', 2, None, 'k2', 'kall', ...) -> 0 / 2 / None."""
    if v is None:
        return None
    s = str(v).strip().lower()
    if s.startswith("k"):
        s = s[1:]
    if s in ("all", "allc-pre", ""):
        return None
    return int(s)


def main_k(ds):
    """The main configuration's k of `ds` (HostSpec.main_k): 2 for CADETS_E3, None (= all) elsewhere."""
    from habitus.hosts import spec
    return parse_k(spec(ds).main_k)


def _acq_all(rec):
    """Ablation of the hard-tier exemption: bounded acquittal applied to EVERY node (qa None -> base)."""
    base = max(float(rec["qv"]), float(rec["qt"]))
    qa = rec.get("qa")
    return base if qa is None else base * (1.0 - 0.5 * float(qa))


class HostRows(list):
    """run_host's rows, plus the per-(level, k) availability status and the host's test days."""

    def __init__(self, rows=(), *, ds=None, status=(), test_days=(), seconds=None):
        super().__init__(rows)
        self.ds, self.status, self.test_days, self.seconds = ds, list(status), list(test_days), seconds


def make_host(ds, source="nodeindex", settings=None):
    from habitus.data.host import Host
    host = Host(ds, source=source) if settings is None else Host(ds, source=source, settings=settings)
    fn = getattr(host, "keep_clean_days", None)
    if callable(fn):
        host = fn() or host
    return host


def _scores_root(root, settings=None):
    if root is not None:
        return str(root)
    if settings is None:
        from habitus.config import settings as _s
        settings = _s()
    return str(settings.paths.scores)


def run_host(ds, *, root=None, source="nodeindex", host=None, settings=None):
    """Every (level, k, column) row of `ds` (HostRows, a list of dicts keyed by COLS). `host`: an already
    built host of `ds` (its spec's main_k marks the main row; a test double may carry a fake spec)."""
    from habitus.decide import scorefile as S
    from habitus.eval.labels import host_labels
    from habitus.eval.metrics import confusion
    t0 = time.time()
    root = _scores_root(root, settings)
    if host is None:
        from habitus.hosts import assert_allowed
        assert_allowed(ds)
        host = make_host(ds, source, settings)
    elif host.ds != ds:
        raise ValueError(f"host {host.ds} given for ds {ds}")
    hs = getattr(host, "spec", None)
    mk = parse_k(hs.main_k) if getattr(hs, "main_k", None) is not None else main_k(ds)
    y, _n_total = host_labels(host)
    rows, status = [], []
    for level in LEVELS:
        for k in KS:
            d = os.path.join(root, ds, f"{level}_k{S.klabel(k)}")
            if (level, k) in BY_DESIGN:
                status.append({"ds": ds, "level": level, "k": KNAME[k], "status": "by-design", "why": BY_DESIGN[(level, k)]})
                continue
            if not os.path.isdir(d):
                status.append({"ds": ds, "level": level, "k": KNAME[k], "status": "missing",
                               "why": "no score directory (never run)"})
                continue
            by_day, missing = S.load_by_day(ds, level, k, host.test, root)
            if missing:
                status.append({"ds": ds, "level": level, "k": KNAME[k], "status": "incomplete",
                               "why": f"test days without a score file {missing}"})
                continue
            status.append({"ds": ds, "level": level, "k": KNAME[k], "status": "ok", "why": ""})
            head = S.head_mask(host, by_day)
            gt_in_head = int((head & y).sum())
            fields = E2_FIELDS if level == "E2" else E1_FIELDS
            for f in fields:
                if f == "1-qa" and k == 0:
                    status.append({"ds": ds, "level": level, "k": KNAME[k], "status": "n/a-column",
                                   "why": "1-qa: QA is not asked at k=0 (no precedent), constant over the head"})
                    continue
                if f == "acq_all":
                    if k == 0:
                        continue          # QA not asked: identical to base / composite
                    bd = {dd: {sig: _acq_all(r) for sig, r in recs.items()} for dd, recs in by_day.items()}
                    v = S.node_scores(host, bd, "composite")
                else:
                    v = S.node_scores(host, by_day, f)
                r = confusion(host, v)
                flags = []
                if ds == "CADETS_E3" and k is None:
                    flags.append("oracle-k")
                main = level == "E2" and f == "composite" and mk == k
                if level == "E2" and k == 0 and f in ("composite", "base"):
                    flags.append("k0: composite == base (QA not asked)")
                note = CADETS_KALL if "oracle-k" in flags else ""
                tb = r.get("tie_block") or {}
                tn_ = r.get("ties_negative") or {}
                rows.append({
                    "ds": ds, "system": "E2 (routed)" if level == "E2" else "E1 (unrouted)", "level": level,
                    "k": KNAME[k], "column": f, "main_config": "yes" if main else "", "flags": ";".join(flags),
                    "tp": r.get("tp"), "fp": r.get("fp"), "tn": r.get("tn"), "fn": r.get("fn"),
                    "precision": r.get("precision"), "recall": r.get("recall"), "f1": r.get("f1"),
                    "mcc": r.get("mcc"), "auc": r.get("auc"), "ap": r.get("ap"), "threshold": r.get("threshold"),
                    "alerts": r.get("alerts"), "tie_n": tb.get("n"), "tie_gt": tb.get("gt"),
                    "mcc_ties_neg": tn_.get("mcc"), "n_universe": r["universe"]["nodes"],
                    "test_day_nodes": r["universe"]["test_day_nodes"], "gt_total": r["universe"]["gt_total"],
                    "gt_without_card": r["universe"]["gt_without_card"], "gt_in_head": gt_in_head,
                    "n_test_days": len(host.test), "note": note})
                _log(f"[{ds}] {level} {KNAME[k]} {f}: MCC {r.get('mcc')} TP {r.get('tp')} FP {r.get('fp')} "
                     f"AP {r.get('ap')} ({time.time() - t0:.0f}s)")
    return HostRows(rows, ds=ds, status=status, test_days=list(host.test), seconds=round(time.time() - t0, 1))


# --------------------------------------------------------------------------- #
# Writer (byte-identical to the original)
# --------------------------------------------------------------------------- #
def _cell_value(r, c):
    v = r.get(c)
    return "" if v is None else (round(v, 6) if isinstance(v, float) else v)


def write_csv(rows, cols=COLS) -> bytes:
    """csv.DictWriter default dialect (CRLF), round(x, 6) floats, '' for None."""
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({c: _cell_value(r, c) for c in cols})
    return buf.getvalue().encode("utf-8")


def ordered(rows):
    """Rows in HOSTS order (each host's rows in run order)."""
    by = {}
    for r in rows:
        by.setdefault(r["ds"], []).append(r)
    unknown = sorted(set(by) - set(HOSTS))
    if unknown:
        raise ValueError(f"rows of hosts outside HOSTS: {unknown}")
    return [r for ds in HOSTS if ds in by for r in by[ds]]


def rows_from_csv(data) -> list[dict]:
    """Rows of a shipped ablation CSV (bytes or str), typed back (ints, floats, None for '')."""
    text = data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else data
    out = []
    for r in csv.DictReader(io.StringIO(text, newline="")):
        row = {}
        for c in COLS:
            v = r.get(c, "")
            if c in INT_COLS:
                row[c] = None if v == "" else int(v)
            elif c in FLOAT_COLS:
                row[c] = None if v == "" else float(v)
            else:
                row[c] = v
        out.append(row)
    return out


def _status_of(rows):
    """The availability status that run_host records, rebuilt from the rows (BY_DESIGN + present cells)."""
    have = {(r["ds"], r["level"], r["k"]) for r in rows}
    out = []
    for ds in HOSTS:
        if not any(r["ds"] == ds for r in rows):
            continue
        for level in LEVELS:
            for k in KS:
                if (level, k) in BY_DESIGN:
                    out.append({"ds": ds, "level": level, "k": KNAME[k], "status": "by-design", "why": BY_DESIGN[(level, k)]})
                elif (ds, level, KNAME[k]) in have:
                    out.append({"ds": ds, "level": level, "k": KNAME[k], "status": "ok", "why": ""})
                else:
                    out.append({"ds": ds, "level": level, "k": KNAME[k], "status": "missing",
                                "why": "no score directory (never run)"})
    return out


def _f(x, nd=3):
    return "–" if x is None or x == "" else f"{float(x):.{nd}f}"


def ref_check(rows):
    """(all E2 composite rows match REF, E1 AP spot checks [bool]) of a merged row list."""
    idx = {(r["ds"], r["level"], r["k"], r["column"]): r for r in rows}
    e2 = {}
    for ds in HOSTS:
        for k in (2, None):
            ref = REF[(ds, k)]
            r = idx.get((ds, "E2", KNAME[k], "composite"))
            e2[(ds, KNAME[k])] = r is not None and abs(r["mcc"] - ref[0]) < 5e-5 and r["tp"] == ref[1] and r["fp"] == ref[2]
    e1 = []
    for (ds, lv, k, col), ap_ref in E1_AP_REF.items():
        r = idx.get((ds, lv, k, col))
        e1.append(r is not None and abs(r["ap"] - ap_ref) < 5e-4)
    return e2, e1


def merge(rows, *, generated=None, test_days=None, xcheck=None, status=None):
    """(ablation CSV bytes, markdown text) of the rows of every host (any order; written in HOSTS order).

    generated: the 'Generated <time>' stamp (default: now; REF_GENERATED reproduces the shipped md).
    test_days: {ds: [day]} (default HostSpec.test). xcheck: {(ds, 'k2'|'kall'): bool} for the 'loader
    identical' column (default True for every present row). status: run_host's status entries (default:
    rebuilt from the rows, which renders the same text)."""
    rows = ordered(list(rows))
    csv_bytes = write_csv(rows)
    st_all = _status_of(rows) if status is None else list(status)
    have_ds = [ds for ds in HOSTS if any(r["ds"] == ds for r in rows)]
    test_days = dict(test_days or {})
    for ds in have_ds:
        if ds not in test_days:
            from habitus.hosts import spec
            test_days[ds] = list(spec(ds).test)

    idx = {(r["ds"], r["level"], r["k"], r["column"]): r for r in rows}
    stat = {(s["ds"], s["level"], s["k"]): s for s in st_all if s["status"] != "n/a-column"}

    def cell(ds, lv, k, col, counts=True):
        r = idx.get((ds, lv, k, col))
        if r is None:
            s = stat.get((ds, lv, k))
            if s is not None and s["status"] == "by-design":
                return "n/a"
            return "·"
        m = _f(r["mcc"])
        if counts:
            m += f" ({r['tp']}/{r['fp']})"
        if "oracle-k" in r["flags"]:
            m += "‡"
        return m

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)

    def mcc_of(ds, lv, k, col):
        r = idx.get((ds, lv, k, col))
        return None if r is None else r["mcc"]

    MAIN_K = {ds: main_k(ds) for ds in HOSTS}
    stamp = generated if generated is not None else time.strftime("%Y-%m-%d %H:%M:%S")
    L = []
    L.append("# Zero-shot HABITUS ablations: node-level MCC at the oracle best-MCC threshold")
    L.append("")
    L.append(provenance_line(stamp))
    L.append("")
    L.append("**Protocol.** Universe = every test-day node + ground-truth nodes without a test-day card (always FN); "
             "node score = max over the test days; signatures outside the engine head get the KC-S1 rarity fallback "
             "(< 0, below every engine score). The threshold MAXIMISES MCC on the test labels for every row "
             "(oracle upper bound, no FP budget); AUC / AP are threshold-free context. Cells: MCC (TP/FP). "
             "`‡` = CADETS_E3 k=all, whose reference includes clean days after its first test day (oracle-ish). "
             "`n/a` = not defined by design (k=0 has no reference: L2 = L1 + stub, L4 = L3 + stub, SPEC 4). "
             "`·` = never run on that host. All 5 hosts are design-informed development hosts (E2's routing and "
             "name-free evidence were chosen after a label-reading diagnosis on THEIA_E3); these are not held-out numbers.")
    L.append("")
    L.append("Score columns. E1 (single unrouted state, every question reads every section): composite = max(qv, qt). "
             "E2 (routed): qv = QV alone (verdict, 4 levels), qt = QT alone (accuse-only tradecraft), "
             "base = max(qv, qt) (routing, no acquittal), composite = base * (1 - 0.5 qa) outside the hard tier "
             "(the HABITUS main detector), 1-qa = the acquittal question alone read as suspicion (1 when QA is not asked, "
             "i.e. no precedent), acq_all = bounded acquittal applied to every node (removes the hard-tier exemption; "
             "extra ablation). At k=0 QA is never asked, so composite = base and 1-qa is constant.")
    L.append("")

    # Table 1: state ladder
    L.append("## 1. State ladder (E1, unrouted, composite = max(qv, qt)) vs routed E2")
    L.append("")
    L.append("Levels (cumulative, SPEC 4): L0 entity; L1 + hard bits; L2 + reference profile + deviation; "
             "L3 + evidence (neighbour NAMES); L4 + precedent. E2 = routed questions over entity / hard bits / reference / "
             "deviation2 / name-free evidence / precedent2 with bounded acquittal.")
    L.append("")
    for k in ("k0", "k2", "kall"):
        L.append(f"### {k}")
        L.append("")
        L.append("| state | " + " | ".join(HOSTS) + " | mean (n) |")
        L.append("|---|" + "---|" * (len(HOSTS) + 1))
        for lv in LEVELS:
            m, n = mean([mcc_of(ds, lv, k, "composite") for ds in HOSTS])
            name = "E2 composite" if lv == "E2" else lv
            L.append(f"| {name} | " + " | ".join(cell(ds, lv, k, "composite") for ds in HOSTS) +
                     f" | {_f(m)} ({n}) |")
        L.append("")
    # Table 2: E2 question ablation
    L.append("## 2. E2 question ablation (routed)")
    L.append("")
    for k in ("k0", "k2", "kall"):
        L.append(f"### {k}")
        L.append("")
        L.append("| score | " + " | ".join(HOSTS) + " | mean (n) |")
        L.append("|---|" + "---|" * (len(HOSTS) + 1))
        for col, name in (("qv", "QV only"), ("qt", "QT only"), ("base", "base = max(QV, QT), no acquittal"),
                          ("composite", "composite (bounded acquittal, QA)"), ("1-qa", "QA only, 1 - qa"),
                          ("acq_all", "acquittal on every node (no hard tier)")):
            if k == "k0" and col in ("1-qa", "acq_all"):
                continue
            m, n = mean([mcc_of(ds, "E2", k, col) for ds in HOSTS])
            L.append(f"| {name} | " + " | ".join(cell(ds, "E2", k, col) for ds in HOSTS) + f" | {_f(m)} ({n}) |")
        if k == "k0":
            L.append("")
            L.append("At k=0 QA is not asked (no reference, no precedent): composite = base; 1-qa / acq_all are undefined.")
        L.append("")
    # Table 3: reference size
    L.append("## 3. Reference size (E2 composite)")
    L.append("")
    L.append("| k | " + " | ".join(HOSTS) + " | mean (n) |")
    L.append("|---|" + "---|" * (len(HOSTS) + 1))
    for k in ("k0", "k2", "kall"):
        m, n = mean([mcc_of(ds, "E2", k, "composite") for ds in HOSTS])
        L.append(f"| {k} | " + " | ".join(cell(ds, "E2", k, "composite") for ds in HOSTS) + f" | {_f(m)} ({n}) |")
    main = [idx.get((ds, "E2", KNAME[MAIN_K[ds]], "composite")) for ds in HOSTS]
    m, n = mean([r["mcc"] if r else None for r in main])
    L.append("| **main config** (kall; CADETS k2) | " +
             " | ".join(cell(ds, "E2", KNAME[MAIN_K[ds]], "composite") for ds in HOSTS) + f" | **{_f(m)}** ({n}) |")
    L.append("")
    L.append("AP / AUC of the same rows (threshold-free context):")
    L.append("")
    L.append("| k | " + " | ".join(HOSTS) + " |")
    L.append("|---|" + "---|" * len(HOSTS))
    for k in ("k0", "k2", "kall"):
        cs = []
        for ds in HOSTS:
            r = idx.get((ds, "E2", k, "composite"))
            cs.append("·" if r is None else f"{_f(r['ap'])} / {_f(r['auc'])}")
        L.append(f"| {k} | " + " | ".join(cs) + " |")
    L.append("")
    # Sanity cross-check
    L.append("## 4. Sanity cross-check vs the published E2 rows")
    L.append("")
    L.append("Reference = `paper/habitus/facts.md` routes-only (E2) rows = `confusion_reference.md` E2 (KC-S1) rows. "
             "`loader identical` = the node vector computed here equals `habitus.evaluate.e2_vector` element-wise.")
    L.append("")
    L.append("| host | k | reference MCC (TP/FP) | here MCC (TP/FP) | match | loader identical |")
    L.append("|---|---|---|---|---|---|")
    xc = dict(xcheck or {})
    all_ok = True
    for ds in HOSTS:
        for k in (2, None):
            ref = REF[(ds, k)]
            r = idx.get((ds, "E2", KNAME[k], "composite"))
            if r is None:
                L.append(f"| {ds} | {KNAME[k]} | {ref[0]:.4f} ({ref[1]}/{ref[2]}) | · | NO | – |")
                all_ok = False
                continue
            ok = abs(r["mcc"] - ref[0]) < 5e-5 and r["tp"] == ref[1] and r["fp"] == ref[2]
            same = bool(xc.get((ds, KNAME[k]), True))
            all_ok = all_ok and ok and same
            L.append(f"| {ds} | {KNAME[k]} | {ref[0]:.4f} ({ref[1]}/{ref[2]}) | {r['mcc']:.4f} ({r['tp']}/{r['fp']}) | "
                     f"{'yes' if ok else 'NO'} | {same} |")
    L.append("")
    L.append(f"All E2 composite rows reproduce the published numbers: **{'yes' if all_ok else 'NO'}**. "
             f"Main-config mean MCC here = {_f(m, 4)} (facts.md: 0.3822).")
    L.append("")
    e1_ok = []
    for (ds, lv, k, col), ap_ref in E1_AP_REF.items():
        r = idx.get((ds, lv, k, col))
        e1_ok.append(r is not None and abs(r["ap"] - ap_ref) < 5e-4)
    L.append(f"E1 ladder: AP of {sum(e1_ok)}/{len(e1_ok)} spot-checked cells equals the frozen "
             "`research/results/reflex/kcs1/e1_ladder.md` AP to 3 decimals (" +
             ", ".join(f"{ds} {lv} {k} {ap:.3f}" for (ds, lv, k, _c), ap in E1_AP_REF.items()) + ").")
    L.append("")
    # Derived deltas (mean over the hosts where both rows exist)
    L.append("## 5. Derived deltas (mean MCC difference over hosts where both rows exist)")
    L.append("")
    L.append("| comparison | k0 | k2 | kall |")
    L.append("|---|---|---|---|")

    def delta(a, b, k):
        ds_ = [ds for ds in HOSTS if idx.get((ds,) + a[:1] + (k,) + a[1:]) and idx.get((ds,) + b[:1] + (k,) + b[1:])]
        if not ds_:
            return "–"
        dv = [idx[(ds,) + a[:1] + (k,) + a[1:]]["mcc"] - idx[(ds,) + b[:1] + (k,) + b[1:]]["mcc"] for ds in ds_]
        return f"{sum(dv) / len(dv):+.3f} ({sum(x > 0 for x in dv)}/{sum(x < 0 for x in dv)}/{len(dv)})"
    for name, a, b in (("acquittal: E2 composite - E2 base", ("E2", "composite"), ("E2", "base")),
                       ("routed + name-free state, no acquittal: E2 base - E1 L4", ("E2", "base"), ("L4", "composite")),
                       ("full E2 - E1 L4 (E1 primary)", ("E2", "composite"), ("L4", "composite")),
                       ("full E2 - E1 L2", ("E2", "composite"), ("L2", "composite")),
                       ("full E2 - E1 L3 (E1 primary at k0)", ("E2", "composite"), ("L3", "composite")),
                       ("hard tier: E2 composite - acq_all", ("E2", "composite"), ("E2", "acq_all")),
                       ("E2 composite - QV only", ("E2", "composite"), ("E2", "qv")),
                       ("E2 composite - QT only", ("E2", "composite"), ("E2", "qt"))):
        L.append(f"| {name} | " + " | ".join(delta(a, b, k) for k in ("k0", "k2", "kall")) + " |")
    L.append("")
    L.append("Cells: mean delta (hosts better / worse / n). CADETS_E3 kall is included in the kall column (‡ caveat).")
    L.append("")
    # Availability
    L.append("## 6. Availability of score directories")
    L.append("")
    L.append("| host | missing (never run) | by design (k=0) |")
    L.append("|---|---|---|")
    for ds in have_ds:
        st = [s for s in st_all if s["ds"] == ds and s["status"] in ("missing", "incomplete")]
        bd = [s for s in st_all if s["ds"] == ds and s["status"] == "by-design"]
        L.append(f"| {ds} | {', '.join(f'{s['level']}_{s['k']}' for s in st) or 'none'} | "
                 f"{', '.join(f'{s['level']}_{s['k']}' for s in bd)} |")
    L.append("")
    L.append("Test days: " + "; ".join(f"{ds} {', '.join(test_days[ds])}" for ds in have_ds) + ".")
    L.append("")
    return csv_bytes, "\n".join(L)


def provenance_line(stamp):
    """The 'Generated <stamp> by ...' line of ablation_e1_e2.md (scripts/results/build_results.py writes
    the shipped copy with the same line)."""
    return (f"{GENERATED_PREFIX}{stamp} by `habitus eval tables` (CPU only) from the frozen score files "
            "`scores/<ds>/<level>_k<k>/` of the result bundle, metric code `habitus.eval.metrics.confusion` / "
            "`best_mcc`. Full rows (TP, FP, TN, FN, Precision, Recall, F1, MCC, AUC, AP, threshold, tie block, "
            "ties-negative MCC, GT in head): `ablation_e1_e2.csv`.")


def md_body(text):
    """The markdown without its 'Generated <time>' stamp (for comparisons)."""
    out = []
    for line in text.split("\n"):
        if line.startswith(GENERATED_PREFIX) and " by `" in line:
            line = GENERATED_PREFIX + "<time>" + line[line.index(" by `"):]
        out.append(line)
    return "\n".join(out)


def main_rows(rows):
    """The main-configuration rows (main_config == 'yes'), HOSTS order."""
    return [r for r in ordered(rows) if r.get("main_config") == "yes"]


def merge_parts(parts):
    """Concatenate run_host results (HostRows) keeping their status and test days for merge()."""
    rows, status, test_days = [], [], {}
    for p in parts:
        rows.extend(p)
        status.extend(getattr(p, "status", []) or [])
        if getattr(p, "test_days", None):
            test_days[p.ds] = list(p.test_days)
    return rows, status, test_days


def write_tables(rows, out_dir, *, generated=None, test_days=None, status=None, xcheck=None):
    """Write ablation_e1_e2.csv / .md, main_config.csv and summary_means.json under out_dir."""
    from habitus.eval.derived import summary_means
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_bytes, md = merge(rows, generated=generated, test_days=test_days, status=status, xcheck=xcheck)
    files = {"ablation_e1_e2.csv": csv_bytes, "ablation_e1_e2.md": md.encode("utf-8"),
             "main_config.csv": write_csv(main_rows(rows)),
             "summary_means.json": (json.dumps(summary_means(rows_from_csv(csv_bytes)), indent=1) + "\n").encode("utf-8")}
    paths = {}
    for name, data in files.items():
        p = out / name
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, p)
        paths[name] = p
    return paths
