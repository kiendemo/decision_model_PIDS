"""Build the decision-model CSVs (best-MCC detection metrics + speed) from the HABITUS reports.
Run: python3 make_csv.py   (reads former data disk, writes next to this file)."""
import csv
import glob
import json
import os

REP = "former data disk"
OUT = os.path.dirname(os.path.abspath(__file__))
HOSTS = ["CADETS_E3", "THEIA_E3", "CLEARSCOPE_E3", "optc_h501", "ATLASV2_EDR"]
ARCH = {   # arm -> (branch, architecture, backbone)
    "HB-Z": ("JEV-style", "decoder letter-logit, stock (no training)", "Qwen3.5-4B"),
    "HB": ("JEV-style", "decoder letter-logit + LoRA r16", "Qwen3.5-4B"),
    "HB.head": ("JEV-style", "decoder + LoRA + fp32 letter head (AutoTrust/Jeff)", "Qwen3.5-4B"),
    "HB.08b": ("JEV-style", "decoder letter-logit + LoRA r16", "Qwen3.5-0.8B"),
    "HB-L.s0": ("Laya-style", "encoder, one [MASK] per option (Laya clone), 1 seed", "ModernBERT-large 421M"),
    "HB-L": ("Laya-style", "encoder, one [MASK] per option (Laya clone), 3-seed ensemble", "ModernBERT-large 421M x3"),
    "HB-L.mq": ("Laya-style", "encoder, encode view once / many questions (Verdict-MQ)", "ModernBERT-large 421M"),
    "HB-L.full": ("Laya-style", "Laya clone, 1 seed, all exported items", "ModernBERT-large 421M"),
    "HB-L.zs": ("Laya-style", "Laya typed-decisions, zero-shot (no training)", "ModernBERT-large 421M"),
}
DET_ARMS = ["HB-Z", "HB", "HB.head", "HB.08b", "HB-L.s0", "HB-L", "HB-L.mq"]

rows = json.load(open(os.path.join(REP, "confusion_reference.json")))["rows"]
det = []
for r in rows.values():
    if r.get("family") != "habitus":
        continue
    base, _, col = r["system"].partition("/")
    if base not in DET_ARMS or r["ds"] not in HOSTS:
        continue
    br, arch, bb = ARCH[base]
    det.append({
        "dataset": r["ds"], "model": base, "branch": br, "architecture": arch, "backbone": bb,
        "k_plan": r.get("kplan"), "score": "habit-only (model alone)" if col else "composite (model + KC-S1 routes)",
        "TP": r["tp"], "FP": r["fp"], "TN": r["tn"], "FN": r["fn"],
        "Precision": round(r["precision"], 4), "Recall": round(r["recall"], 4), "F1": round(r["f1"], 4),
        "MCC": round(r["mcc"], 4), "AUC": round(r["auc"], 4) if r.get("auc") is not None else "",
        "AP": round(r["ap"], 4) if r.get("ap") is not None else "", "threshold": r.get("threshold"),
        "n_nodes": r["n"], "n_gt": r["n_pos"],
        "note": r.get("note", "") + ("; host-seen: clean days in training, test days never" if "host-seen" in str(r.get("flags", "")) + r.get("note", "") or r["ds"] == "CLEARSCOPE_E3" else ""),
    })
order = {h: i for i, h in enumerate(HOSTS)}
det.sort(key=lambda d: (order[d["dataset"]], d["k_plan"] != "k2", d["score"] != "composite (model + KC-S1 routes)", -d["MCC"]))
with open(os.path.join(OUT, "decision_models_best_mcc.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(det[0]))
    w.writeheader()
    w.writerows(det)

# MCC matrix per k plan: model x dataset, composite and habit-only (CADETS_E3 has no allc-pre)
for KP in ("k2", "allc-pre"):
  with open(os.path.join(OUT, f"decision_models_mcc_matrix_{KP}.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "score"] + HOSTS + ["mean"])
    for sc in ("composite (model + KC-S1 routes)", "habit-only (model alone)"):
        for a in DET_ARMS:
            v = [next((d["MCC"] for d in det if d["model"] == a and d["dataset"] == h and d["k_plan"] == KP and d["score"] == sc), "") for h in HOSTS]
            nums = [x for x in v if x != ""]
            w.writerow([a, sc.split(" ")[0]] + v + [round(sum(nums) / len(nums), 4) if nums else ""])

# speed from compare-arms (label-free selection items, same questions for every arm) per fold
sp = []
for p in sorted(glob.glob(os.path.join(REP, "arms_loho-*.json"))):
    rep = json.load(open(p))
    for a, v in rep["arms"].items():
        if a not in ARCH:
            continue
        s = v.get("speed", {})
        s5 = s.get("s5") or {}
        br, arch, bb = ARCH[a]
        sp.append({"fold": rep["fold"], "model": a, "branch": br, "architecture": arch, "backbone": bb,
                   "train_items_per_s": s.get("s1_items_per_s"),
                   "train_GPU_hours_per_fold": s.get("s2_gpu_hours"),
                   "infer_questions_per_s": s.get("s3_questions_per_s"), "infer_ms_per_node": s.get("s4_ms_per_node"),
                   "stream_p50_ms_per_view": s5.get("p50_ms") if isinstance(s5, dict) else "",
                   "stream_p95_ms_per_view": s5.get("p95_ms") if isinstance(s5, dict) else "",
                   "n_questions": rep.get("n_questions"), "device": "RTX A5000 (bf16)"})
with open(os.path.join(OUT, "decision_models_speed_per_fold.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(sp[0]))
    w.writeheader()
    w.writerows(sp)
agg = {}
for d in sp:
    agg.setdefault(d["model"], []).append(d)
with open(os.path.join(OUT, "decision_models_speed_mean.csv"), "w", newline="") as f:
    keys = ["train_items_per_s", "train_GPU_hours_per_fold", "infer_questions_per_s", "infer_ms_per_node",
            "stream_p50_ms_per_view", "stream_p95_ms_per_view"]
    w = csv.writer(f)
    # GPU memory measured once in the GPU smoke run (reports/arms_smoke.json, bf16, RTX A5000): train / serve GiB
    MEM = {"HB": (10.28, 10.55), "HB-Z": ("", 10.55), "HB.head": (10.28, 10.55), "HB.08b": (2.65, 2.54),
           "HB-L.s0": (8.68, 3.44), "HB-L": (8.68, 3.44), "HB-L.full": (8.68, 3.44), "HB-L.zs": ("", 3.44),
           "HB-L.mq": (15.83, 2.61)}
    w.writerow(["model", "branch", "backbone", "n_folds"] + keys + ["speedup_vs_HB_4B", "train_peak_GiB", "serve_peak_GiB"])
    hb = None
    means = {}
    for a, L in agg.items():
        m = {}
        for k in keys:
            xs = [x[k] for x in L if isinstance(x[k], (int, float))]
            m[k] = round(sum(xs) / len(xs), 4) if xs else ""
        means[a] = (L, m)
    hb = means.get("HB", (None, {}))[1].get("infer_questions_per_s")
    for a in ARCH:
        if a not in means:
            continue
        L, m = means[a]
        q = m["infer_questions_per_s"]
        w.writerow([a, ARCH[a][0], ARCH[a][2], len(L)] + [m[k] for k in keys] +
                   [round(q / hb, 2) if hb and isinstance(q, (int, float)) else ""] + list(MEM.get(a, ("", ""))))
print(len(det), "detection rows;", len(sp), "speed rows")
