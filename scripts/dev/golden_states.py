"""Golden state / prompt hashes from the ORIGINAL code (dev only; G2 / G3 reference data).

    python scripts/dev/golden_states.py [--out tests/golden/state] [--scratch runs/golden_states]
                                        [--pidsmaker-src DIR] [--pidsmaker-artifacts DIR] [--only DS,..]

This is the only file of the repository that imports the original pidsmaker.reflex code. It runs
that code READ-ONLY:
  - the artifact dir it hands to the original HostData is a fresh scratch dir holding ONLY the
    per-dataset symlinks <scratch>/art/reflex/cards/<ds> -> the original card dirs (the
    transformation dirs are NOT linked: config construction would mkdir through them);
  - kshot.usable_days is monkeypatched to the frozen clean-day lists (no graph is read);
  - KCS1_PRECEDENT_CACHE points at a scratch COPY of the 5 hosts' precedent caches;
  - HABITUS_ROOT / HF_MODULES_CACHE / TMPDIR / cache dirs point into the scratch dir, no GPU;
  - a write-audit hook (plus habitus.guard's, when available) raises on any write, mkdir, rename
    or delete outside the scratch and output dirs, so the source tree provably stays untouched.

Source locations come from configs/local.yaml (paths.pidsmaker_src, paths.pidsmaker_artifacts) or
the command line. Outputs (tests/golden/state/):
  index.json                       the sample list (ds, level, k, day, role, file)
  <ds>_<level>_k<k>_<day>.json.gz  per head signature sha1(serialize(state)) per question (E1: one
                                   state), the head lists, ref, caps and a digest over all states
  text_hashes.json                 sha256 of the frozen question texts (routed.text_hashes() layout)
  prompts.json                     sha1 of the original user_prompt / system_prompt for 200 samples
  sanitize.json                    the original sanitizer pattern string and its live status
  fake_states.json                 state sha1s of the original code on the synthetic KC-S1 host
Only the 5 in-scope datasets are touched (explicit allow-list).
"""

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATASETS = ("CADETS_E3", "THEIA_E3", "CLEARSCOPE_E3", "optc_h501", "ATLASV2_EDR")
# frozen usable clean days (configs/datasets.yaml; checked against habitus.hosts when importable)
CLEAN = {
    "CADETS_E3": ["2018-04-02", "2018-04-03", "2018-04-04", "2018-04-05", "2018-04-07", "2018-04-08",
                  "2018-04-09", "2018-04-10"],
    "THEIA_E3": ["2018-04-03", "2018-04-04", "2018-04-05", "2018-04-09"],
    "CLEARSCOPE_E3": ["2018-04-03", "2018-04-04", "2018-04-05", "2018-04-07", "2018-04-08", "2018-04-09",
                      "2018-04-10"],
    "optc_h501": ["2019-09-19", "2019-09-20", "2019-09-21", "2019-09-22"],
    "ATLASV2_EDR": ["2022-07-16", "2022-07-17", "2022-07-18"],
}
# (ds, level, k label, day): the G2 sample host-days (+ the full CADETS_E3 E2 k2 2018-04-06 of G3)
SAMPLES = (
    ("CADETS_E3", "E2", "2", "2018-04-06"),
    ("CADETS_E3", "E2", "2", "2018-04-02"),
    ("THEIA_E3", "E2", "all", "2018-04-10"),
    ("THEIA_E3", "E2", "all", "2018-04-03"),
    ("CLEARSCOPE_E3", "E2", "all", "2018-04-11"),
    ("CLEARSCOPE_E3", "E2", "2", "2018-04-03"),
    ("CLEARSCOPE_E3", "E2", "0", "2018-04-12"),
    ("CLEARSCOPE_E3", "L4", "2", "2018-04-11"),
    ("optc_h501", "E2", "all", "2019-09-23"),
    ("optc_h501", "E2", "all", "2019-09-19"),
    ("ATLASV2_EDR", "E2", "all", "2022-07-19"),
    ("ATLASV2_EDR", "E2", "all", "2022-07-16"),
)
N_PROMPTS = 200
WRITE_EVENTS = ("os.mkdir", "os.rename", "os.replace", "os.remove", "os.rmdir", "os.symlink", "os.link",
                "os.truncate", "os.chmod", "os.chown", "os.utime", "shutil.copyfile", "shutil.copytree",
                "shutil.rmtree", "shutil.move")


class WriteAuditError(RuntimeError):
    pass


def install_write_audit(roots):
    """sys.addaudithook: any write-mode open / mkdir / rename / delete outside `roots` raises."""
    roots = tuple(os.path.realpath(r).rstrip(os.sep) + os.sep for r in roots)
    flags_w = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC

    def ok(p):
        if isinstance(p, int) or p is None:
            return True
        try:
            p = os.fsdecode(p)
        except TypeError:
            return True
        rp = os.path.realpath(os.path.abspath(p))
        return rp in ("/dev/null",) or any((rp + os.sep).startswith(r) for r in roots)

    def hook(event, args):
        if event == "open":
            path, mode, flags = (list(args) + [None, None, None])[:3]
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or \
                      (isinstance(flags, int) and flags & flags_w)
            if writing and not ok(path):
                raise WriteAuditError(f"write outside the scratch dir refused: {path!r} ({mode!r})")
        elif event in WRITE_EVENTS:
            if event == "os.mkdir" and args and isinstance(args[0], (str, bytes, os.PathLike)) \
                    and os.path.isdir(args[0]):
                return                              # makedirs(exist_ok=True) of an existing dir: no-op
            for p in args[:2]:
                if isinstance(p, (str, bytes, os.PathLike)) and not ok(p):
                    raise WriteAuditError(f"{event} outside the scratch dir refused: {p!r}")
    sys.addaudithook(hook)


def sha1(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def text_bytes(obj):
    if isinstance(obj, str):
        return obj.encode("utf-8")
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def write_json(path, obj, gz=False):
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    tmp = path + ".tmp"
    if gz:
        with open(tmp, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as f:
            f.write(data)
    else:
        with open(tmp, "wb") as f:
            f.write(data + b"\n")
    os.replace(tmp, path)


def golden_name(ds, level, k, day):
    return f"{ds}_{level}_k{k}_{day}.json.gz"


def source_paths(a):
    src, art = a.pidsmaker_src, a.pidsmaker_artifacts
    if not (src and art):
        sys.path.insert(0, os.path.join(REPO, "src"))
        from habitus import config
        s = config.settings()
        src = src or s.paths.pidsmaker_src
        art = art or s.paths.pidsmaker_artifacts
    if not (src and art):
        raise SystemExit("set paths.pidsmaker_src and paths.pidsmaker_artifacts in configs/local.yaml "
                         "(or pass --pidsmaker-src / --pidsmaker-artifacts)")
    return os.path.abspath(src), os.path.abspath(art)


def check_frozen_lists():
    try:
        sys.path.insert(0, os.path.join(REPO, "src"))
        from habitus import hosts
    except ImportError:
        print("[golden] habitus.hosts not importable: using the script's frozen clean-day lists", flush=True)
        return
    for ds in DATASETS:
        got = list(hosts.spec(ds).clean)
        if got != CLEAN[ds]:
            raise SystemExit(f"{ds}: habitus.hosts clean days {got} != frozen {CLEAN[ds]}")


def prepare_scratch(scratch, art_src, datasets):
    if os.path.exists(scratch):
        raise SystemExit(f"{scratch} exists: pass a fresh --scratch")
    for sub in ("art/reflex/cards", "precedent_cache", "habitus_root", "hf_modules", "tmp", "xdg", "mpl"):
        os.makedirs(os.path.join(scratch, sub))
    for ds in datasets:
        cards = os.path.join(art_src, "reflex", "cards", ds)
        if not os.path.isdir(cards):
            raise SystemExit(f"no card dir for {ds}")
        os.symlink(cards, os.path.join(scratch, "art", "reflex", "cards", ds))
        prec = os.path.join(art_src, "reflex", "kcs1", "precedent_cache", ds)
        dst = os.path.join(scratch, "precedent_cache", ds)
        os.makedirs(dst)
        for f in sorted(os.listdir(prec)):
            if f.endswith("_e2.json.gz"):
                shutil.copyfile(os.path.join(prec, f), os.path.join(dst, f))


def shipped(art_src, ds, level, k, day):
    p = os.path.join(art_src, "reflex", "kcs1", "scores", ds, f"{level}_k{k}", f"{day}.json.gz")
    with gzip.open(p, "rt", encoding="utf-8") as f:
        obj = json.load(f)
    return obj["meta"], obj["head"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(REPO, "tests", "golden", "state"))
    ap.add_argument("--scratch", default=os.path.join(REPO, "runs", "golden_states",
                                                      time.strftime("%Y%m%d-%H%M%S")))
    ap.add_argument("--pidsmaker-src", default=None)
    ap.add_argument("--pidsmaker-artifacts", default=None)
    ap.add_argument("--only", default=None, help="comma-separated datasets (default: all 5)")
    ap.add_argument("--no-fake", action="store_true")
    a = ap.parse_args()
    datasets = DATASETS if not a.only else tuple(x for x in a.only.split(",") if x)
    bad = [d for d in datasets if d not in DATASETS]
    if bad:
        raise SystemExit(f"not in scope: {bad}")
    src, art_src = source_paths(a)
    check_frozen_lists()
    scratch, out = os.path.abspath(a.scratch), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    prepare_scratch(scratch, art_src, datasets)
    env = {"KCS1_PRECEDENT_CACHE": os.path.join(scratch, "precedent_cache"),
           "HABITUS_ROOT": os.path.join(scratch, "habitus_root"), "HABITUS_ALLOW_ROOTFS": "1",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_MODULES_CACHE": os.path.join(scratch, "hf_modules"),
           "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": os.path.join(scratch, "tmp"),
           "XDG_CACHE_HOME": os.path.join(scratch, "xdg"), "MPLCONFIGDIR": os.path.join(scratch, "mpl"),
           "KCS1_OUT_TAG": ""}
    os.environ.update(env)
    sys.dont_write_bytecode = True
    install_write_audit([scratch, out])
    try:
        from habitus.guard import install_guard
        install_guard("golden", write_roots=[scratch, out, os.devnull])
        print("[golden] habitus.guard installed", flush=True)
    except Exception as e:                      # the local write audit above still applies
        print(f"[golden] habitus.guard not installed ({type(e).__name__}: {e}): local write audit only",
              flush=True)
    os.chdir(scratch)
    sys.path.insert(0, src)
    t0 = time.time()

    from pidsmaker.reflex import kshot
    from pidsmaker.reflex.anchor.host import HostData
    from pidsmaker.reflex.kcs1 import questions as Q
    from pidsmaker.reflex.kcs1 import sanitize as SAN
    from pidsmaker.reflex.kcs1 import select as SEL
    from pidsmaker.reflex.kcs1 import state as S

    # frozen text and the sanitizer
    items = {"SYSTEM": Q.SYSTEM, "SYSTEM_E2": Q.SYSTEM_E2, "QV_INS": Q.QV_INS, "QV_LEVELS": Q.QV_LEVELS,
             "QT_POS": Q.QT_POS, "QT_NEG": Q.QT_NEG, "QA_POS": Q.QA_POS, "QA_NEG": Q.QA_NEG,
             "QUESTIONS": Q.QUESTIONS, "QUESTIONS_E2": Q.QUESTIONS_E2}
    write_json(os.path.join(out, "text_hashes.json"),
               {k: hashlib.sha256(text_bytes(v)).hexdigest() for k, v in items.items()})
    pat = SAN.pattern()
    write_json(os.path.join(out, "sanitize.json"),
               {"pattern": pat.pattern, "status": SAN.vocab_source(), "special_re": SAN.SPECIAL_RE.pattern,
                "qwen_added": list(SAN.QWEN_ADDED)})
    print(f"[golden] sanitizer status {SAN.vocab_source()}", flush=True)

    index, prompt_pool = [], []
    hosts = {}
    for ds, level, k, day in SAMPLES:
        if ds not in datasets:
            continue
        t1 = time.time()
        host = hosts.get(ds)
        if host is None:
            hosts.clear()
            S._DAYIDX.clear()
            S._REFINDEX.clear()
            kshot.usable_days = lambda cfg, _ds=ds: list(CLEAN[_ds])      # no graph is read
            host = hosts[ds] = HostData(ds, os.path.join(scratch, "art") + os.sep)
            if host.clean != CLEAN[ds]:
                raise SystemExit(f"{ds}: HostData clean {host.clean} != frozen {CLEAN[ds]}")
        meta, shipped_head = shipped(art_src, ds, level, k, day)
        ref, caps, role = list(meta["ref"]), meta["caps"], meta["role"]
        kk = 0 if k == "0" else 2 if k == "2" else None
        h = SEL.head(host, day, ref or None, caps)
        if h != shipped_head:
            raise SystemExit(f"{ds} {level} k{k} {day}: head differs from the shipped score file")
        sigs = SEL.flat(h)
        if level == "E2":
            st = S.StateBuilderE2(host, ref, kk).build(day, sigs)
            texts = {s: {q: S.serialize(x) for q, x in qs.items()} for s, qs in st.items()}
            states = {s: {q: sha1(t) for q, t in qs.items()} for s, qs in texts.items()}
            system = Q.SYSTEM_E2
        else:
            st = S.StateBuilder(host, ref, level, kk).build(day, sigs)
            texts = {s: {"state": S.serialize(x)} for s, x in st.items()}
            states = {s: sha1(t["state"]) for s, t in texts.items()}
            system = Q.SYSTEM
        dig = hashlib.sha1()
        for s in sigs:
            for q, t in texts[s].items():
                dig.update(f"{s}|{q}|{t}\n".encode("utf-8"))
        name = golden_name(ds, level, k, day)
        write_json(os.path.join(out, name),
                   {"ds": ds, "level": level, "k": k, "day": day, "role": role, "ref": ref, "caps": caps,
                    "head": h, "states": states, "digest": dig.hexdigest(), "n_states": len(sigs)}, gz=True)
        index.append({"ds": ds, "level": level, "k": k, "day": day, "role": role, "file": name,
                      "n_states": len(sigs)})
        # prompt samples: every 1/n-th signature of the day, each question and form
        step = max(1, len(sigs) // 6)
        for s in sigs[::step][:6]:
            for q, t in texts[s].items():
                qids = [q] if level == "E2" else ["qv", "qt"]
                for qid in qids:
                    for v in Q.variants(qid):
                        prompt_pool.append({"file": name, "sig": s, "qid": qid, "variant": v,
                                            "state_key": q,
                                            "user_sha1": sha1(Q.user_prompt(t, qid, v)),
                                            "system_sha1": sha1(Q.system_prompt(t, system))})
        print(f"[golden] {name}: {len(sigs)} states, digest {dig.hexdigest()[:12]} ({time.time() - t1:.1f}s)",
              flush=True)
    step = max(1, len(prompt_pool) // N_PROMPTS)
    prompts = prompt_pool[::step][:N_PROMPTS]
    write_json(os.path.join(out, "prompts.json"), {"n": len(prompts), "prompts": prompts})
    write_json(os.path.join(out, "index.json"), {"samples": index, "datasets": list(datasets)})

    if not a.no_fake:
        write_json(os.path.join(out, "fake_states.json"), fake_states(src, S, SEL))
    print(f"[golden] done in {time.time() - t0:.0f}s -> {os.path.relpath(out, REPO)}", flush=True)


def fake_states(src, S, SEL):
    """State sha1s of the original builders on test_kcs1.make_host (E1 L0..L4 and E2, k = 2 and 0)."""
    sys.path.insert(0, os.path.join(src, "tests"))
    import test_kcs1 as T
    T._clear_caches()
    host = T.make_host()
    out = {}
    for ref, k in ((["c1", "c2"], 2), ([], 0)):
        for d in host.test:
            sigs = SEL.flat(SEL.head(host, d, ref or None))
            for lv in ("L0", "L1", "L2", "L3", "L4"):
                st = S.StateBuilder(host, ref, lv, k).build(d, sigs)
                out[f"{lv}|k{k}|{d}"] = {s: sha1(S.serialize(x)) for s, x in st.items()}
            st = S.StateBuilderE2(host, ref, k).build(d, sigs)
            out[f"E2|k{k}|{d}"] = {s: {q: sha1(S.serialize(x)) for q, x in qs.items()} for s, qs in st.items()}
            out[f"head|k{k}|{d}"] = SEL.head(host, d, ref or None)
    x9q = next(s for s in host.nodes("t2").values() if host.card(s)["invariant"].get("file_name") == "x9q")
    st = S.StateBuilderE2(host, ["c1", "c2"], 2).build("t2", [x9q])[x9q]
    texts = {q: S.serialize(x) for q, x in st.items()}
    out["x9q"] = {q: {"sha1": sha1(t), "len": len(t), "neutralised": t.count("(im_end)"),
                      "raw_control": t.count("<|im_end|>")} for q, t in texts.items()}
    return out


if __name__ == "__main__":
    main()
