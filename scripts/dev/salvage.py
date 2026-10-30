"""First build step: keep the porting session's derivation scripts before their directory goes away.

    python scripts/dev/salvage.py --src SESSION_DIR [--dest runs/salvage] [--check]

The verified prototypes of the porting session (the GPU-free replays cache_only_check*.py, the card
builder prototype minimal/hb_data/ with its harness verify_all.py, the paper-number derivations
parse_sources.py / build_facts.py / edit/*.py, env.sh) existed only in an ephemeral session
directory. They are copied VERBATIM into runs/salvage/ (gitignored, never published) and their
sha256 is recorded in runs/salvage/SALVAGE.json, which G10 verifies and docs/PROVENANCE.md cites.

A file already salvaged is never overwritten: an identical copy is recorded, a differing one is
reported (exit 1) and left alone. --check verifies SALVAGE.json against the copies only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FILES = ("parse_sources.py", "build_facts.py", "env.sh", "cache_only_check.py", "cache_only_check_e1.py",
         "verify_all.py")
DIRS = {"edit": ".py", "minimal/hb_data": ".py"}      # every *<suffix> file directly inside
RECORD = "SALVAGE.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def wanted(src):
    """Relative names of the files to salvage that exist under `src` (explicit names + 2 dirs)."""
    out = [f for f in FILES if (src / f).is_file()]
    for d, suffix in DIRS.items():
        if (src / d).is_dir():
            out += sorted(f"{d}/{p.name}" for p in (src / d).iterdir() if p.is_file() and p.name.endswith(suffix))
    return out


def check(dest):
    rec = json.loads((dest / RECORD).read_text(encoding="utf-8"))
    bad = [r for r, v in rec["files"].items() if not (dest / r).is_file() or sha256(dest / r) != v["sha256"]]
    return rec, bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default=os.environ.get("HABITUS_SALVAGE_SRC"),
                    help="the session directory (default $HABITUS_SALVAGE_SRC)")
    ap.add_argument("--dest", default=str(REPO / "runs" / "salvage"))
    ap.add_argument("--check", action="store_true", help="verify the recorded sha256s only")
    a = ap.parse_args(argv)
    dest = Path(a.dest)
    if a.check:
        rec, bad = check(dest)
        print(json.dumps({"files": len(rec["files"]), "bad": bad}, indent=1))
        return 1 if bad else 0
    if not a.src:
        raise SystemExit("--src SESSION_DIR (or $HABITUS_SALVAGE_SRC) is needed")
    src = Path(a.src)
    names = wanted(src)
    old = {}
    if (dest / RECORD).is_file():
        old = json.loads((dest / RECORD).read_text(encoding="utf-8")).get("files", {})
    files, differ, missing = dict(old), [], [f for f in FILES if f not in names]
    for rel in names:
        s, d = src / rel, dest / rel
        h = sha256(s)
        if d.exists():
            if sha256(d) != h:
                differ.append(rel)
                continue
            action = "kept"
        else:
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
            action = "copied"
        files[rel] = {"sha256": h, "bytes": s.stat().st_size,
                      "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(s.stat().st_mtime)),
                      "action": action}
    rec = {"recorded": time.strftime("%Y-%m-%d %H:%M:%S"), "files": dict(sorted(files.items())),
           "missing": missing, "differ": differ}
    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest / (RECORD + ".tmp")
    tmp.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, dest / RECORD)
    print(f"{len(names)} files ({sum(1 for v in files.values() if v.get('action') == 'copied')} copied); "
          f"missing {missing or 'none'}; differing (left alone) {differ or 'none'} -> {dest / RECORD}")
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
