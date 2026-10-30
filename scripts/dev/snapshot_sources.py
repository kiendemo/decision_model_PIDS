"""Source snapshot: sha256 / size / mtime of every original file this repository vendors or ports.

    python scripts/dev/snapshot_sources.py [--root CYRAG_ROOT] [--out docs/PROVENANCE.md] [--check]

The code behind the published numbers lives in an untracked part of the research tree (no commit
holds it), so this snapshot is the provenance record: it must be taken BEFORE anything changes in
that tree. CYRAG_ROOT defaults to the parent of paths.pidsmaker_src (configs/local.yaml). Every path
written is relative to that root; nothing is written into it. --check recomputes the table and fails
when docs/PROVENANCE.md differs (sha256 / size only; mtimes are informational).

SOURCES is an explicit list (no glob over a parent directory): data artifacts (answers, scores,
cards, precedent caches, ground truth, tokenizer files) are recorded by the release MANIFEST.json,
and the session derivation scripts by runs/salvage/SALVAGE.json (scripts/dev/salvage.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HOSTS = ("CADETS_E3", "THEIA_E3", "CLEARSCOPE_E3", "optc_h501", "ATLASV2_EDR")
R = "PIDSMaker/pidsmaker/reflex/"
SOURCES = (
    # KC-S1 E2 / E1 pipeline (state, questions, engine, scoring, nulls, checks, evaluation)
    *(R + "kcs1/" + f for f in ("state.py", "select.py", "questions.py", "engine.py", "sanitize.py", "profile.py",
                                "score.py", "null.py", "checks.py", "run.py", "audit.py", "evaluate.py", "paired.py",
                                "SPEC.md", "SPEC_E2.md")),
    # habit channel and decision-model arms
    *(R + "habitus/" + f for f in ("__init__.py", "paths.py", "guard.py", "types.py", "vocab.py", "attest.py", "ifc.py",
                                   "profile2.py", "views.py", "targets.py", "questions.py", "head.py", "export.py",
                                   "sample.py", "surprise.py", "null2.py", "score.py", "retrofit.py", "decide.py",
                                   "engine.py", "heads.py", "encoder.py", "encoder_mq.py", "arms.py", "train.py",
                                   "train_enc.py", "run.py", "checks.py", "evaluate.py", "SPEC_HB.md")),
    # shared reflex plumbing, host data, cards
    *(R + f for f in ("common.py", "identity.py", "cards.py", "ledger.py", "kshot.py", "s1/student_dec.py",
                      "anchor/host.py", "anchor/atoms_hist.py", "anchor/qc.py", "anchor/evalkit.py")),
    # PIDSMaker modules the data layer mirrors
    *("PIDSMaker/pidsmaker/" + f for f in ("utils/utils.py", "config/config.py", "attacks/evaluate.py",
                                            "defense/context.py", "defense/semantics.py", "defense/profiles.py",
                                            "defense/kb/build_kg.py", "defense/framework.py")),
    "PIDSMaker/build_graphs_only.py",
    "PIDSMaker/LICENSE",
    # upstream PIDSMaker 2289cd9 (graph construction config) and its licence
    "PIDSMaker_upstream/pidsmaker/config/pipeline.py",
    "PIDSMaker_upstream/pidsmaker/config/config.py",
    "PIDSMaker_upstream/pidsmaker/utils/utils.py",
    "PIDSMaker_upstream/LICENSE",
    # Laya 0.3.5 (vendored DecisionModel; Apache-2.0)
    "third_party/laya/laya/common.py",
    "third_party/laya/LICENSE",
    "third_party/laya/pyproject.toml",
    # original tests and fixtures that were ported
    "PIDSMaker/tests/test_kcs1.py",
    "PIDSMaker/tests/kcs1_e1_snapshot.py",
    "PIDSMaker/tests/kcs1_e1_snapshot.json.gz",
    "PIDSMaker/tests/habitus_fakes.py",
    *(f"PIDSMaker/tests/test_habitus_{n}.py" for n in ("infra", "reference", "views", "export", "engine", "heads",
                                                        "encoder", "encoder_mq", "surprise", "score", "cli",
                                                        "confusion", "arms")),
    # result generators and the result files results/ is built from
    *("research/results/reflex/habitus/" + f for f in (
        "zeroshot_ablation.py", "zeroshot_ablation.csv", "zeroshot_ablation.md", "confusion_reference.md",
        "selection_metrics_3folds.csv", "run_facts_20261029.md", "arms_pilot_loho-optc_h501.md", "arms_smoke.md",
        "decision_models/make_csv.py", "decision_models/decision_models_best_mcc.csv",
        "decision_models/decision_models_mcc_matrix_k2.csv", "decision_models/decision_models_mcc_matrix_allc-pre.csv",
        "decision_models/decision_models_speed_per_fold.csv", "decision_models/decision_models_speed_mean.csv",
        "method_stats/routestats.json", "method_stats/checka_dev.json",
        "method_stats/example_state_ATLASV2_EDR_2022-07-19.json",
        "method_stats/tok_ATLASV2_EDR_all_2022-07-19.json", "method_stats/tok_CADETS_E3_2_2018-04-06.json",
        "method_stats/tok_THEIA_E3_all_2018-04-10.json")),
    *("paper/habitus/derive/" + f for f in ("derive_zeroshot.py", "editor_cpu.py", "editor_facts.py", "route_cost.json",
                                            "editor/editor_cpu.json")),
    *(f"paper/habitus/derive/editor/{what}_{ds}.json" for what in ("budget", "boot", "state") for ds in HOSTS),
    "paper/habitus/experimental_setup.md",
)

BEGIN, END = "<!-- SNAPSHOT:BEGIN -->", "<!-- SNAPSHOT:END -->"
HEADER = """# Provenance

The numbers in `results/` were produced by research code that is not under version control (its
directory is untracked in the research repository), so the record of what was vendored is this
snapshot: the sha256, size and modification time of every original file the repository ports,
vendors or rebuilds its result files from, taken read-only by `scripts/dev/snapshot_sources.py`
before any porting began. Paths are relative to the root of the research tree. Data artifacts
(answer caches, score files, cards, precedent caches, ground truth, tokenizer files) are recorded
file by file, with their source sha256, in the release bundle's `MANIFEST.json`
(`scripts/make_release.py`). The short derivation scripts of the porting session are kept, with
their sha256, in the gitignored `runs/salvage/` (`scripts/dev/salvage.py`, `SALVAGE.json`).

Upstream PIDSMaker: commit 2289cd9 (graph construction and transformation); the research fork the
numbers come from matches upstream tag 2.1.1 for the files it did not change. Laya: release 0.3.5.
"""


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def default_root():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from habitus import config
    src = config.settings().paths.pidsmaker_src
    if not src:
        raise SystemExit("--root CYRAG_ROOT, or paths.pidsmaker_src in configs/local.yaml, is needed")
    return str(Path(src).parent)


def collect(root, sources=SOURCES):
    """[{path, sha256, bytes, mtime}] of every source (missing files are reported, not skipped)."""
    rows = []
    for rel in sources:
        p = os.path.join(root, rel)
        if not os.path.isfile(p):
            rows.append({"path": rel, "sha256": None, "bytes": None, "mtime": None})
            continue
        st = os.stat(p)
        rows.append({"path": rel, "sha256": sha256(p), "bytes": st.st_size,
                     "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))})
    return rows


def render(rows, taken=None):
    taken = taken or time.strftime("%Y-%m-%d")
    lines = [HEADER, f"Snapshot taken {taken}; {sum(1 for r in rows if r['sha256'])} of {len(rows)} files present.", "",
             BEGIN, "", "| file | sha256 | bytes | mtime |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| `{r['path']}` | {r['sha256'] or 'MISSING'} | {r['bytes'] if r['bytes'] is not None else '-'} "
                     f"| {r['mtime'] or '-'} |")
    lines += ["", END, ""]
    salvage = Path(__file__).resolve().parents[2] / "runs" / "salvage" / "SALVAGE.json"
    if salvage.is_file():
        rec = json.loads(salvage.read_text(encoding="utf-8"))
        lines += ["## Session derivation scripts (runs/salvage/)", "",
                  "Copied verbatim by `scripts/dev/salvage.py` (names relative to the session directory; "
                  "not published). `python scripts/dev/salvage.py --check` re-verifies them.", "",
                  "| file | sha256 | bytes |", "|---|---|---|"]
        lines += [f"| `{rel}` | {v['sha256']} | {v['bytes']} |" for rel, v in sorted(rec["files"].items())]
        lines.append("")
    return "\n".join(lines)


def parse(text):
    """{path: (sha256, bytes)} of a rendered PROVENANCE.md table."""
    out = {}
    if BEGIN not in text or END not in text:
        return out
    body = text[text.index(BEGIN):text.index(END)]
    for line in body.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 4 and cells[0].startswith("`"):
            out[cells[0].strip("`")] = (cells[1], cells[2])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=None, help="root of the research tree (read only)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "docs" / "PROVENANCE.md"))
    ap.add_argument("--check", action="store_true", help="compare with --out instead of writing it")
    ap.add_argument("--json", default=None, help="also write the rows as JSON here")
    a = ap.parse_args(argv)
    root = a.root or default_root()
    rows = collect(root)
    missing = [r["path"] for r in rows if r["sha256"] is None]
    if a.check:
        old = parse(Path(a.out).read_text(encoding="utf-8"))
        bad = [r["path"] for r in rows if old.get(r["path"]) != (r["sha256"] or "MISSING",
                                                                str(r["bytes"]) if r["bytes"] is not None else "-")]
        print(json.dumps({"files": len(rows), "differ": bad, "missing": missing}, indent=1))
        return 1 if bad else 0
    Path(a.out).write_text(render(rows), encoding="utf-8")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    print(f"{len(rows)} sources ({len(missing)} missing) -> {a.out}")
    if missing:
        print("missing: " + ", ".join(missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
