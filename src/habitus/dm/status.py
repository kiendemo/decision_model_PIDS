"""The STATUS MATRIX: which decision-model arm has which kind of number, and from where.

This constant is the single source of the generated copies in README.md, results/README.md and
docs/DECISION_MODELS.md (between the STATUS markers); G10 asserts every copy equals render_markdown().

    python -m habitus.dm.status --render              # print the block
    python -m habitus.dm.status --check FILE [...]    # exit 1 unless each FILE holds the current block
    python -m habitus.dm.status --write FILE [...]    # rewrite the block between the markers in place
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

BEGIN, END = "<!-- STATUS:BEGIN -->", "<!-- STATUS:END -->"
QUESTION_SETS = {"routed": "Routed QV/QT/QA (HABITUS main)", "habit": "Habit questions (HABITUS+H)",
                 "hb0": "HB0 decision layer (no DM call)"}
REPORTED, SUPPORTED_NE, FROZEN = "REPORTED", "SUPPORTED, NOT EVALUATED", "REPORTED (FROZEN CSV)"
SUPPORTED, UNSUPPORTED, FROZEN_SEL = "SUPPORTED", "UNSUPPORTED", "FROZEN SELECTION METRICS ONLY"
REPRODUCED = "REPRODUCED (G11)"

_NE = "No published number exists."
_NE_LAYA = ("No published number exists. laya refuses over-budget rows (never cuts) and decide raises on any "
            "refusal.")
_FROZEN = ("Adapters, run dirs and caches were lost with the former data disk and are being retrained. "
           "Retrained numbers will differ.")
_FROZEN_Z = ("Stock backbone; its habit answer cache and nulls were lost with the former data disk. "
             "Retrained numbers will differ.")
_HABIT_CSV = "results/extension_habit/decision_models_best_mcc.csv"
_SELECTION = "results/extension_habit/selection_metrics_3folds.csv"


@dataclass(frozen=True)
class StatusRow:
    questions: str        # key of QUESTION_SETS
    arm: str
    config: str           # configs/dm/<config>.yaml, or "-"
    status: str
    evidence: str         # repo-relative result file, or "-"
    note: str


STATUS_MATRIX = (
    StatusRow("routed", "MAIN (jev stock Qwen3.5-4B)", "jev-qwen3.5-4b", REPORTED, "results/habitus/main_config.csv",
              "Reproducible on CPU from the shipped answer cache: the main table, E1/E2 ablations, budget, "
              "sensitivity, forms and bootstrap."),
    StatusRow("routed", "jev 4B+LoRA", "jev-qwen3.5-4b-lora", SUPPORTED_NE, "-", _NE),
    StatusRow("routed", "jev 4B+head", "jev-qwen3.5-4b-head", SUPPORTED_NE, "-", _NE),
    StatusRow("routed", "jev 0.8B+LoRA", "jev-qwen3.5-0.8b-lora", SUPPORTED_NE, "-", _NE),
    StatusRow("routed", "laya s0", "laya-typed-s0", SUPPORTED_NE, "-", _NE_LAYA),
    StatusRow("routed", "laya ens3", "laya-typed-ens3", SUPPORTED_NE, "-", _NE_LAYA),
    StatusRow("routed", "laya mq", "laya-typed-mq", SUPPORTED_NE, "-", _NE_LAYA),
    StatusRow("routed", "laya zs", "laya-typed-zs", SUPPORTED_NE, "-", _NE_LAYA),
    StatusRow("habit", "HB-Z", "jev-qwen3.5-4b-habit", FROZEN, _HABIT_CSV, _FROZEN_Z),
    StatusRow("habit", "HB", "jev-qwen3.5-4b-lora", FROZEN, _HABIT_CSV, _FROZEN),
    StatusRow("habit", "HB.head", "jev-qwen3.5-4b-head", FROZEN, _HABIT_CSV, _FROZEN),
    StatusRow("habit", "HB.08b", "jev-qwen3.5-0.8b-lora", FROZEN, _HABIT_CSV, _FROZEN),
    StatusRow("habit", "HB-L.s0", "laya-typed-s0", FROZEN, _HABIT_CSV, _FROZEN),
    StatusRow("habit", "HB-L", "laya-typed-ens3", FROZEN, _HABIT_CSV, _FROZEN),
    StatusRow("habit", "HB-L.mq", "laya-typed-mq", FROZEN, _HABIT_CSV, _FROZEN),
    StatusRow("habit", "HB-L.zs", "laya-typed-zs", FROZEN_SEL, _SELECTION, "The config is supported."),
    StatusRow("habit", "HB-L.full", "-", UNSUPPORTED, _SELECTION,
              "Frozen selection rows only; there is no arm or config."),
    StatusRow("hb0", "HB0", "-", REPRODUCED, "results/extension_habit/hb0_rows.csv",
              "Lineage-split decision layer over the frozen E2 routes, on CPU via `habitus decide hb0` + `habitus eval "
              "habit --levels HB0`: gate G11 recomputes all 9 rows exactly. Needs the role-table catalogs "
              "(scripts/fetch_catalogs.sh; their table must match ROLE_TABLE_SHA256)."),
)


def _cell(text):
    return text.replace("|", "\\|")


def render_markdown(rows=STATUS_MATRIX):
    """The generated block (a markdown table, newline-terminated), without the markers."""
    out = ["| Question set | Arm | Config | Status | Evidence | Note |", "|---|---|---|---|---|---|"]
    for r in rows:
        cfg = f"`{r.config}`" if r.config != "-" else "-"
        ev = f"`{r.evidence}`" if r.evidence != "-" else "-"
        out.append(f"| {QUESTION_SETS[r.questions]} | {_cell(r.arm)} | {cfg} | {r.status} | {ev} | {_cell(r.note)} |")
    return "\n".join(out) + "\n"


def block():
    """The block with its markers, as embedded in a document."""
    return f"{BEGIN}\n{render_markdown()}{END}"


def extract(text):
    """The text between the markers of `text` (exactly one block); ValueError without one."""
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        raise ValueError("expected exactly one STATUS:BEGIN / STATUS:END block")
    a = text.index(BEGIN) + len(BEGIN)
    b = text.index(END)
    if b < a:
        raise ValueError("STATUS:END before STATUS:BEGIN")
    body = text[a:b]
    return body[1:] if body.startswith("\n") else body


def embed(text):
    """`text` with its STATUS block replaced by the current one."""
    extract(text)
    a = text.index(BEGIN)
    b = text.index(END) + len(END)
    return text[:a] + block() + text[b:]


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m habitus.dm.status")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--render", action="store_true", help="print the generated block")
    g.add_argument("--check", nargs="+", metavar="FILE", help="exit 1 unless every FILE holds the current block")
    g.add_argument("--write", nargs="+", metavar="FILE", help="rewrite the block of every FILE in place")
    a = p.parse_args(argv)
    if a.render:
        sys.stdout.write(render_markdown())
        return 0
    bad = 0
    for path in a.check or a.write:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        if a.check:
            ok = extract(text) == render_markdown()
            print(f"{'OK  ' if ok else 'DIFF'} {path}")
            bad += not ok
        else:
            new = embed(text)
            if new != text:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new)
            print(f"wrote {path}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
