"""The single writer of the result bundle release/habitus-cache-v1/ (local, gitignored).

    python scripts/make_release.py --hosts CADETS_E3,THEIA_E3,CLEARSCOPE_E3,optc_h501,ATLASV2_EDR \
        [--out release/habitus-cache-v1] [--full] [--tar] [--jobs 5] [--force] [--no-baselines]
    python scripts/make_release.py --refresh-meta [--out release/habitus-cache-v1]

Source roots come from configs/local.yaml (paths.pidsmaker_artifacts, upstream_gt, upstream_src,
models); everything there is only read. The copy is file by file from an explicit 5-host allow-list:
no glob over a parent directory, never a stats.json (it holds ground-truth counts) or a .lock file,
never another host's files.

Steps (into a staging directory, renamed into place at the end):
  1. scores/<ds>/<level>_k<k>/<day>.json.gz   byte copies of exactly tests/golden/inventory.json
  2. precedent_cache/<ds>/*.json.gz           byte copies (no .lock)
  3. cards/<ds>/<day>.json.gz + links/        byte copies of HostSpec.card_days (stats.json excluded)
  4. ground_truth/orthrus/<gt_files>          the 17 CSVs of configs/datasets.yaml (copied, not parsed)
     ground_truth/LICENSE                     the upstream PIDSMaker licence
  5. tokenizer/Qwen3.5-4B@851bf6e8/           exactly the 6 tokenizer files (no weights)
     tokenizer/laya-typed-decisions/          the Laya ModernBERT tokenizer (2 files, no weights; only
                                              when paths.laya_hub is set): the laya CPU tests then need
                                              no 842 MB checkpoint
  6. baselines/<ds>_best_baseline.npz         from runs/baselines/ (scripts/export_baselines.py)
  7. nodeindex/<ds>/<day>.json.gz             data.nodeindex.build over the staged cards (all 45 files)
  8. answers/qwen3.5-4b/<ds>.jsonl.gz         the source answer cache PRUNED to the keys of every
                                              inventory job (a GUARDED CPU child per host:
                                              `habitus questions keys --from-inventory`), first record
                                              per key, original line order, gzip -9 with mtime 0;
                                              --full keeps every line. Abort on any missing key.
  9. LICENSE, NOTICE                          the repository's Apache-2.0 licence and attribution
                                              notice (Apache-2.0 4(a) / 4(d) for the redistributed
                                              tokenizer files; sources repo:LICENSE, repo:NOTICE)
 10. MANIFEST.json (sha256, bytes, lines, source path relative to the research root, source sha256,
     card / link JSON sha256), SHA256SUMS, README.md, PUBLISH.md; --tar: an uncompressed tarball.
--refresh-meta updates an existing bundle written by this script without reading any source tree: it
re-copies LICENSE and NOTICE from the repository, adds their MANIFEST rows (every other row is kept)
and rewrites README.md, PUBLISH.md and SHA256SUMS.
Abort when / has under 5 GB free. Run it after gate G3 passed (the pruned key set is chosen by this
repo's code; G3 proves the states equal the original's).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from habitus import config, hosts  # noqa: E402

NAME = "habitus-cache-v1"
ALLOWED = ("CADETS_E3", "THEIA_E3", "CLEARSCOPE_E3", "optc_h501", "ATLASV2_EDR")
TAG = "qwen3.5-4b"
BACKBONE_REPO = "models--Qwen--Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
TOKENIZER_DIR = f"Qwen3.5-4B@{REVISION[:8]}"
TOKENIZER_FILES = ("chat_template.jinja", "config.json", "merges.txt", "tokenizer.json", "tokenizer_config.json",
                   "vocab.json")
LAYA_TOKENIZER_DIR = "laya-typed-decisions"     # habitus.dm.laya.RELEASE_TOKENIZER_DIR
LAYA_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
MIN_FREE = 5 << 30
OWNED = ".habitus-out"
KEY_RX = re.compile(rb'"k"\s*:\s*"([0-9a-f]{40})"')


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [make_release] {msg}", file=sys.stderr, flush=True)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def json_sha256(path):
    from habitus.io import json_sha256 as js
    return js(path)


class Bundle:
    """The staging directory plus the MANIFEST rows."""

    def __init__(self, stage, research_root):
        self.stage, self.files = Path(stage), {}
        self.root = Path(research_root) if research_root is not None else None

    def source_rel(self, src):
        src = os.path.abspath(src)
        if self.root is not None:
            try:
                return "research:" + Path(src).relative_to(self.root).as_posix()
            except ValueError:
                pass
        try:
            return "repo:" + Path(src).relative_to(REPO).as_posix()
        except ValueError:
            raise SystemExit(f"{src}: neither under the research root nor the repo (refusing an unrecorded source)")

    def copy(self, src, rel, *, json_hash=False):
        src = Path(src)
        if not src.is_file():
            raise SystemExit(f"missing source file {self.source_rel(src)}")
        if src.name == "stats.json" or src.name.endswith(".lock"):
            raise SystemExit(f"refusing to copy {src.name}")
        dst = self.stage / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        h = sha256_file(dst)
        row = {"sha256": h, "bytes": dst.stat().st_size, "source": self.source_rel(src), "source_sha256": sha256_file(src)}
        if row["source_sha256"] != h:
            raise SystemExit(f"copy of {rel} differs from its source")
        if json_hash:
            row["json_sha256"] = json_sha256(dst)
        self.files[rel] = row
        return dst

    def add(self, rel, **extra):
        p = self.stage / rel
        row = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        row.update(extra)
        self.files[rel] = row
        return row


def fresh(path):
    p = Path(path)
    if p.exists():
        if not (p / OWNED).exists():
            raise SystemExit(f"{p} exists and was not created by make_release; remove it yourself")
        shutil.rmtree(p)
    p.mkdir(parents=True)
    (p / OWNED).write_text("staging of scripts/make_release.py\n", encoding="utf-8")
    return p


def check_free(path):
    parent = Path(path).absolute().parent
    while not parent.exists():          # release/ does not exist before the first build
        parent = parent.parent
    for p in ("/", str(parent)):
        free = shutil.disk_usage(p).free
        if free < MIN_FREE:
            raise SystemExit(f"only {free / 2**30:.1f} GiB free on {p}; make_release needs >= {MIN_FREE / 2**30:.0f} GiB")


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
def copy_scores(b, art, ds, inventory):
    base = Path(art) / "reflex" / "kcs1" / "scores" / ds
    rows = inventory["hosts"][ds]
    want = {f"{r['dir']}/{r['day']}.json.gz" for r in rows}
    level_dirs = sorted({r["dir"] for r in rows})
    for d in [f"{lv}_k{k}" for lv in ("E2", "L0", "L1", "L2", "L3", "L4") for k in ("0", "2", "all")]:
        if (base / d).is_dir():
            have = {f"{d}/{n}" for n in os.listdir(base / d) if n.endswith(".json.gz")}
            extra = sorted(have - want)
            if extra:
                raise SystemExit(f"{ds}: shipped score files outside tests/golden/inventory.json: {extra[:5]}")
    for rel in sorted(want):
        b.copy(base / rel, f"scores/{ds}/{rel}")
    log(f"{ds}: {len(want)} score files in {len(level_dirs)} level dirs")


def copy_precedent(b, art, ds):
    base = Path(art) / "reflex" / "kcs1" / "precedent_cache" / ds
    names = sorted(n for n in os.listdir(base) if n.endswith(".json.gz"))
    for n in names:
        b.copy(base / n, f"precedent_cache/{ds}/{n}")
    log(f"{ds}: {len(names)} precedent-cache files")


def copy_cards(b, art, ds, spec):
    base = Path(art) / "reflex" / "cards" / ds
    for day in spec.card_days:
        b.copy(base / f"{day}.json.gz", f"cards/{ds}/{day}.json.gz", json_hash=True)
        b.copy(base / "links" / f"{day}.json.gz", f"cards/{ds}/links/{day}.json.gz", json_hash=True)
    if len(spec.card_days) != spec.all_card_days:
        raise SystemExit(f"{ds}: {len(spec.card_days)} card days != all_card_days {spec.all_card_days}")
    log(f"{ds}: {len(spec.card_days)} card days (+ links); stats.json excluded")


def copy_gt(b, gt_root, ds, spec):
    for rel in spec.gt_files:
        b.copy(Path(gt_root) / rel, f"ground_truth/orthrus/{rel}")
    log(f"{ds}: {len(spec.gt_files)} ground-truth files (copied, not read)")


def copy_tokenizer(b, models):
    snap = Path(models) / "hub" / BACKBONE_REPO / "snapshots" / REVISION
    for n in TOKENIZER_FILES:
        b.copy(snap / n, f"tokenizer/{TOKENIZER_DIR}/{n}")
    log(f"tokenizer {TOKENIZER_DIR}: {len(TOKENIZER_FILES)} files")


def copy_laya_tokenizer(b, laya_hub):
    """<laya_hub>/typed-decisions/tokenizer/{tokenizer.json, tokenizer_config.json} (Apache-2.0)."""
    src = Path(laya_hub) / "typed-decisions" / "tokenizer"
    for n in LAYA_TOKENIZER_FILES:
        b.copy(src / n, f"tokenizer/{LAYA_TOKENIZER_DIR}/{n}")
    log(f"tokenizer {LAYA_TOKENIZER_DIR}: {len(LAYA_TOKENIZER_FILES)} files")


def copy_baselines(b, ds, st):
    p = Path(st.paths.runs) / "baselines" / f"{ds}_best_baseline.npz"
    b.copy(p, f"baselines/{ds}_best_baseline.npz")


def build_nodeindex(b, ds, spec):
    from habitus.data import nodeindex
    out = nodeindex.build(ds, str(b.stage / "cards"), str(b.stage / "nodeindex"))
    want = [b.stage / "nodeindex" / ds / f"{d}.json.gz" for d in spec.card_days]
    missing = [p.name for p in want if not p.exists()]
    if missing:
        raise SystemExit(f"{ds}: data.nodeindex.build did not write {missing} under nodeindex/{ds}/ (got {out})")
    for p in want:
        b.add(p.relative_to(b.stage).as_posix(), source="built:data.nodeindex.build")
    log(f"{ds}: {len(want)} nodeindex files")


def collect_keys(b, st, hosts_, jobs):
    """The answer keys of every inventory job per host (guarded CPU children over the staged bundle)."""
    from habitus.repro import reproduce as RP
    out = Path(st.paths.runs) / "release" / "keys"
    out.mkdir(parents=True, exist_ok=True)
    specs = []
    for ds in hosts_:
        argv = RP.habitus_argv("questions", "keys", "--ds", ds, "--from-inventory", "--inventory-root",
                               b.stage / "scores", "--answers", "none", "--out", out / f"{ds}.keys.txt")
        env = RP.child_env(st, write_roots=[st.paths.runs], open_log=out / f"open_{ds}.log",
                           extra={"HABITUS_DATA": str(b.stage)})
        specs.append((ds, argv, env, out / f"keys_{ds}.log"))
    res = RP.run_children(specs, jobs)
    bad = {ds: r for ds, r in res.items() if r["rc"] != 0}
    if bad:
        raise SystemExit("key collection failed: " + "; ".join(f"{ds} (log {r['log']})" for ds, r in bad.items()))
    keys = {}
    for ds in hosts_:
        keys[ds] = {line.strip() for line in (out / f"{ds}.keys.txt").read_text(encoding="utf-8").splitlines() if line.strip()}
        log(f"{ds}: {len(keys[ds])} answer keys over the inventory jobs")
    return keys


def write_answers(b, art, ds, keys, full=False):
    src = Path(art) / "reflex" / "kcs1" / "answers" / TAG / f"{ds}.jsonl"
    rel = f"answers/{TAG}/{ds}.jsonl.gz"
    dst = b.stage / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    h, seen, total, kept, bad = hashlib.sha256(), set(), 0, 0, 0
    tmp = dst.with_name(dst.name + ".tmp")
    with open(src, "rb") as fin, open(tmp, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz:
            for line in fin:
                h.update(line)
                total += 1
                m = KEY_RX.search(line[:80])
                if m is None:
                    bad += 1
                    if full:
                        gz.write(line)
                        kept += 1
                    continue
                k = m.group(1).decode("ascii")
                if full or (k in keys and k not in seen):
                    gz.write(line)
                    kept += 1
                seen.add(k)
    os.replace(tmp, dst)
    missing = set() if keys is None else keys - seen
    if missing:
        raise SystemExit(f"{ds}: {len(missing)} needed answer keys are not in the source cache, e.g. {sorted(missing)[:3]}")
    b.add(rel, lines=kept, source=b.source_rel(src), source_sha256=h.hexdigest(), source_lines=total,
          unparsed_lines=bad, pruned=not full, keys=None if keys is None else len(keys))
    log(f"{ds}: answers {kept} of {total} lines kept ({'full' if full else 'pruned'})")


README = """# {name}

The result bundle of the HABITUS repository: everything `habitus reproduce` needs on a CPU. It is
written only by `scripts/make_release.py`; do not edit it (SHA256SUMS would no longer verify).

| directory | content |
|---|---|
| `scores/<ds>/<level>_k<k>/<day>.json.gz` | the shipped score files (E2 routed questions and the E1 L0-L4 ladder; test and LODO null days) |
| `nodeindex/<ds>/<day>.json.gz` | string-free node index of every card file (eval without card text) |
| `cards/<ds>/<day>.json.gz`, `cards/<ds>/links/` | evidence cards and links (state compile for the replay) |
| `precedent_cache/<ds>/` | precedent cache of the QA states |
| `answers/{tag}/<ds>.jsonl.gz` | the answer cache of the main decision model, {pruned} |
| `tokenizer/{tok}/` | the 6 tokenizer files of the pinned backbone revision (no weights) |
| `tokenizer/{laya_tok}/` | the Laya ModernBERT tokenizer (2 files, no weights), when it was available |
| `ground_truth/orthrus/` | the ground-truth CSVs of the 5 hosts (read only by `habitus.eval`) |
| `baselines/` | best PIDSMaker baseline score vectors (bootstrap comparison) |
| `LICENSE`, `NOTICE` | the repository's Apache-2.0 licence and attribution notice (they cover the tokenizer files) |

Hosts: {hosts}. `MANIFEST.json` records every file's sha256, size and source; `SHA256SUMS` verifies
the bundle (`sha256sum -c SHA256SUMS`). Point `paths.data` (or `HABITUS_DATA`) at this directory.
"""

PUBLISH = """# Before publishing {name}

This bundle is local. Decide each item before any public upload:

- [ ] Cards and links contain paths, command lines, IP addresses and e-mail-like strings from the
      DARPA TC (E3), DARPA OpTC and ATLASv2 audit logs. Check that their terms allow redistribution
      of derived data. What each level needs (docs/REPRODUCE.md):
        `habitus reproduce --level tables`     scores, nodeindex, ground_truth
        `habitus reproduce --level replay`     + cards (with links), precedent_cache, answers, tokenizer
        `habitus reproduce --level operating`  + baselines (its forms replay needs the cards as well)
      Without the cards only `--level tables` works, and only when ground_truth is published too
      (then: scores, nodeindex, ground_truth, LICENSE, NOTICE).
- [ ] The ground-truth CSVs come from the upstream PIDSMaker repository (Apache-2.0, LICENSE in
      ground_truth/); keep the licence file next to them.
- [ ] The tokenizer files are the Qwen3.5-4B release and the Laya checkpoint's ModernBERT tokenizer
      (both Apache-2.0); keep LICENSE and NOTICE (bundle root) with them.
- [ ] The baseline vectors were selected on TEST labels (best_adp): label them as such.
- [ ] Re-run `sha256sum -c SHA256SUMS` and `habitus gates --only G8,G10` on the uploaded copy.
- [ ] The distribution name `habitus` may collide with an existing package index project.
"""


REPO_TEXTS = ("LICENSE", "NOTICE")        # copied from the repository to the bundle root


def copy_repo_texts(b):
    """LICENSE and NOTICE of this repository at the bundle root (sources repo:LICENSE, repo:NOTICE)."""
    for name in REPO_TEXTS:
        b.copy(REPO / name, name)


def write_meta(b, hosts_, full, notes, man=None):
    """MANIFEST.json (a new one, or `man` with its files replaced by b.files), README.md, PUBLISH.md and
    SHA256SUMS of the staged bundle."""
    if man is None:
        man = {"name": NAME, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "hosts": list(hosts_),
               "answers_tag": TAG, "tokenizer": {"repo": "Qwen/Qwen3.5-4B", "revision": REVISION,
                                                 "files": list(TOKENIZER_FILES)},
               "laya_tokenizer": ({"dir": f"tokenizer/{LAYA_TOKENIZER_DIR}", "files": list(LAYA_TOKENIZER_FILES)}
                                  if f"tokenizer/{LAYA_TOKENIZER_DIR}/tokenizer.json" in b.files else None),
               "pruned": not full, "excluded": ["cards/<ds>/stats.json (ground-truth counts)", "*.lock"],
               "source_roots": {"research": "the research tree root (paths relative to it)", "repo": "this repository"},
               "notes": notes}
    man["files"] = dict(sorted(b.files.items()))
    (b.stage / "MANIFEST.json").write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8")
    (b.stage / "README.md").write_text(README.format(name=NAME, tag=TAG, tok=TOKENIZER_DIR, laya_tok=LAYA_TOKENIZER_DIR,
                                                     hosts=", ".join(hosts_),
                                                     pruned="all lines" if full else
                                                     "pruned to the keys of every shipped score file"),
                                       encoding="utf-8")
    (b.stage / "PUBLISH.md").write_text(PUBLISH.format(name=NAME), encoding="utf-8")
    rels = sorted(p.relative_to(b.stage).as_posix() for p in b.stage.rglob("*")
                  if p.is_file() and p.name not in ("SHA256SUMS", OWNED))
    (b.stage / "SHA256SUMS").write_text("".join(f"{sha256_file(b.stage / r)}  {r}\n" for r in rels), encoding="utf-8")


def refresh_meta(final):
    """--refresh-meta: LICENSE / NOTICE re-copied from the repository and the metadata rewritten in an
    existing bundle of this script; no source tree is read and every other MANIFEST row is kept."""
    final = Path(final)
    if not (final / OWNED).exists() or not (final / "MANIFEST.json").is_file():
        raise SystemExit(f"{final}: not a bundle written by make_release (no {OWNED} / MANIFEST.json)")
    man = json.loads((final / "MANIFEST.json").read_text(encoding="utf-8"))
    b = Bundle(final, None)
    b.files = dict(man["files"])
    copy_repo_texts(b)
    man["meta_refreshed"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_meta(b, man["hosts"], not man.get("pruned", True), man.get("notes", []), man=man)
    log(f"{final.name}: metadata refreshed ({', '.join(REPO_TEXTS)} + MANIFEST.json, README.md, PUBLISH.md, "
        f"SHA256SUMS); {len(b.files)} files")
    return 0


def write_tar(final):
    tar_p = final.with_suffix(".tar") if final.suffix == "" else Path(str(final) + ".tar")
    tmp = Path(str(tar_p) + ".tmp")

    def norm(ti):
        ti.uid = ti.gid = 0
        ti.uname = ti.gname = ""
        ti.mtime = 0
        return ti
    with tarfile.open(tmp, "w", format=tarfile.PAX_FORMAT) as tf:
        for p in sorted(final.rglob("*")):
            if p.is_file() and p.name != OWNED:
                tf.add(p, arcname=f"{final.name}/{p.relative_to(final).as_posix()}", filter=norm, recursive=False)
    os.replace(tmp, tar_p)
    log(f"tarball {tar_p.name}: {tar_p.stat().st_size / 2**20:.0f} MiB")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hosts", default=",".join(ALLOWED))
    ap.add_argument("--out", default=None, help=f"default <repo>/release/{NAME}")
    ap.add_argument("--full", action="store_true", help="unpruned answer caches")
    ap.add_argument("--tar", action="store_true")
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--force", action="store_true", help="replace an existing bundle written by this script")
    ap.add_argument("--no-baselines", dest="baselines", action="store_false",
                    help="skip baselines/ (G9 bootstrap then needs runs/baselines)")
    ap.add_argument("--research-root", default=None, help="root the MANIFEST source paths are relative to "
                                                          "(default: the parent of paths.pidsmaker_src)")
    ap.add_argument("--refresh-meta", dest="refresh_meta", action="store_true",
                    help="only re-copy LICENSE / NOTICE and rewrite the metadata of an existing bundle")
    a = ap.parse_args(argv)
    if a.refresh_meta:
        return refresh_meta(a.out or REPO / "release" / NAME)
    hosts_ = [h.strip() for h in a.hosts.split(",") if h.strip()]
    bad = [h for h in hosts_ if h not in ALLOWED]
    if bad or not hosts_:
        raise SystemExit(f"--hosts: only the allow-list {ALLOWED} (got {bad or 'none'})")
    for h in hosts_:
        hosts.assert_allowed(h)
    st = config.settings()
    art = config.require("pidsmaker_artifacts", st)
    gt_root = config.require("upstream_gt", st)
    models = config.require("models", st)
    upstream_src = st.paths.upstream_src or str(Path(gt_root).parents[1])
    root = a.research_root or str(Path(config.require("pidsmaker_src", st)).parent)
    final = Path(a.out or REPO / "release" / NAME)
    check_free(final)
    if final.exists() and not a.force:
        raise SystemExit(f"{final} exists: pass --force to rebuild it")
    stage = fresh(Path(str(final) + ".staging"))
    b = Bundle(stage, root)
    inv = json.loads((REPO / "tests" / "golden" / "inventory.json").read_text(encoding="utf-8"))
    notes = []
    t0 = time.time()
    for ds in hosts_:
        spec = hosts.spec(ds)
        copy_scores(b, art, ds, inv)
        copy_precedent(b, art, ds)
        copy_cards(b, art, ds, spec)
        copy_gt(b, gt_root, ds, spec)
        if a.baselines:
            copy_baselines(b, ds, st)
    if not a.baselines:
        notes.append("baselines/ not included (--no-baselines)")
    b.copy(Path(upstream_src) / "LICENSE", "ground_truth/LICENSE")
    copy_repo_texts(b)
    copy_tokenizer(b, models)
    if st.paths.laya_hub and (Path(st.paths.laya_hub) / "typed-decisions" / "tokenizer").is_dir():
        copy_laya_tokenizer(b, st.paths.laya_hub)
    else:
        notes.append(f"tokenizer/{LAYA_TOKENIZER_DIR} not included (paths.laya_hub absent)")
    for ds in hosts_:
        build_nodeindex(b, ds, hosts.spec(ds))
    keys = {ds: None for ds in hosts_} if a.full else collect_keys(b, st, hosts_, a.jobs)
    for ds in hosts_:
        write_answers(b, art, ds, keys[ds], full=a.full)
    write_meta(b, hosts_, a.full, notes)
    if final.exists():
        if not (final / OWNED).exists():
            raise SystemExit(f"{final} was not written by make_release; not replacing it")
        shutil.rmtree(final)
    os.replace(stage, final)
    (final / OWNED).write_text("written by scripts/make_release.py\n", encoding="utf-8")
    log(f"{final.name}: {len(b.files)} files, {sum(r['bytes'] for r in b.files.values()) / 2**20:.0f} MiB "
        f"in {time.time() - t0:.0f}s")
    if a.tar:
        write_tar(final)
    return 0


if __name__ == "__main__":
    sys.exit(main())
