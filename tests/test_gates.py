"""repro.gates: gate selection (G7 optional), report merging, the runtime guard check of G5 on a
fake ground-truth tree, the G4a expected counts against the shipped inventory, and the G10 parsers
(STATUS blocks, results README table, sha256 lists, provenance / salvage records)."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

from habitus.repro import gates as G
from habitus.repro import reproduce as RP
from habitus.repro.compare import Check

REPO = Path(__file__).resolve().parents[1]


def test_select_keeps_g7_optional():
    want, skipped = G.select()
    assert want == [g for g in G.GATES if g not in ("G7", "G10")] + ["G10"] and "G7" in skipped
    assert want[-1] == "G10" and want[-2] == "G11"                  # G10 reads G11's status of this run
    assert G.select(only=["G7"])[0] == [] and "G7" in G.select(only=["G7"])[1]
    assert G.select(only=["G7"], gpu=0) == (["G7"], {})
    assert G.select(only=["G1", "G7"], gpu=0)[0] == ["G1"]          # G7 runs alone
    assert G.select(skip=["G6", "G9"])[0] == ["G1", "G2", "G3", "G4", "G5", "G8", "G11", "G10"]
    with pytest.raises(SystemExit):
        G.select(only=["G12"])


def test_hb0_compare_rounds_to_the_frozen_decimals():
    want = G.hb0_expected(REPO)
    assert len(want) == 9 and ("CADETS_E3", "k2") in want and ("CADETS_E3", "allc-pre") not in want
    w = want[("CADETS_E3", "k2")]
    got = {"ds": "CADETS_E3", "kplan": "k2", "tp": 12, "fp": 1, "tn": 281516, "fn": 60, "precision": 0.92307,
           "recall": 0.16666, "f1": 0.28235, "mcc": 0.3921834, "threshold": 0.67701, "tie_n": 1, "tie_gt": 1,
           "mcc_ties_neg": 0.3743, "ap": 0.1901, "auc": 0.7621}
    assert float(w["mcc_recomputed"]) == 0.392183
    checks = G.compare_hb0([got], {("CADETS_E3", "k2"): w})
    assert [c.ok for c in checks] == [True], checks[0].detail
    bad = dict(got, fp=2, auc=0.7639)
    (c,) = G.compare_hb0([bad], {("CADETS_E3", "k2"): w})
    assert not c.ok and "fp 2" in c.detail and "auc" in c.detail
    assert not G.compare_hb0([], {("CADETS_E3", "k2"): w})[0].ok


def test_skip_status_for_a_gate_without_its_optional_inputs(tmp_path, monkeypatch):
    def fake(ctx):
        raise G.Skip("catalogs missing")
    monkeypatch.setitem(G.FUNCS, "G11", fake)
    rep = G.run(only=["G11"], out=tmp_path)
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["gates"]["G11"]["status"] == "SKIP" and report["gates"]["G11"]["why"] == "catalogs missing"
    assert rep.ok


def test_expected_counts_match_the_shipped_inventory():
    """EXPECTED_COUNTS (the G4a spec) equals the counts of tests/golden/inventory.json."""
    inv = json.loads((REPO / "tests" / "golden" / "inventory.json").read_text(encoding="utf-8"))["hosts"]
    assert set(inv) == set(G.EXPECTED_COUNTS)
    for ds, rows in inv.items():
        got = {}
        for r in rows:
            n, nn = got.get(r["dir"], (0, 0))
            got[r["dir"]] = (n + 1, nn + (r["role"] == "null"))
        assert got == G.EXPECTED_COUNTS[ds], ds


def test_g2_samples_are_golden_samples():
    idx = REPO / "tests" / "golden" / "state" / "index.json"
    if not idx.exists():
        pytest.skip("tests/golden/state/index.json not generated yet")
    have = {(s["ds"], s["level"], s["k"], s["day"], s["role"]) for s in json.loads(idx.read_text())["samples"]}
    assert set(G.G2_SAMPLES) <= have


def test_guard_blocks_gt_and_eval_on_a_fake_tree(tmp_path):
    """G5c: inside a guarded process a GT open and an habitus.eval import raise LabelGuardError."""
    fake = tmp_path / "gt"
    (fake / "E3-CADETS").mkdir(parents=True)
    f = fake / "E3-CADETS" / "node_probe.csv"
    f.write_text("x\n")
    env = RP.child_env(write_roots=None, extra={"HABITUS_GT": str(fake)})
    log = tmp_path / "guard.log"
    res = RP.run_children([("guard", [sys.executable, "-c", G.GUARD_SCRIPT, str(f)], env, log)], 1)
    assert res["guard"]["rc"] == 0, log.read_text()
    got = G._result_line(log)
    assert got == {"active": True, "gt_open": "blocked", "eval_import": "blocked"}


def test_run_writes_and_merges_the_report(tmp_path, monkeypatch):
    calls = []

    def fake(name, ok):
        def fn(ctx):
            calls.append(name)
            return [Check(f"{name} check", ok)]
        return fn
    monkeypatch.setitem(G.FUNCS, "G1", fake("G1", True))
    monkeypatch.setitem(G.FUNCS, "G8", fake("G8", False))
    rep = G.run(only=["G1"], out=tmp_path)
    assert rep.ok and calls == ["G1"]
    rep = G.run(only=["G8"], out=tmp_path)
    assert not rep.ok
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["gates"]["G1"]["status"] == "PASS" and report["gates"]["G8"]["status"] == "FAIL"
    md = (tmp_path / "report.md").read_text()
    assert "| G1 | PASS |" in md and "| G8 | FAIL |" in md and "G8 check" in md

    def boom(ctx):
        raise RuntimeError("cannot run")
    monkeypatch.setitem(G.FUNCS, "G1", boom)
    rep = G.run(only=["G1"], out=tmp_path)
    assert not rep.ok and "cannot run" in rep.checks[0].detail


def test_g2_uses_release_answers_and_g2u_the_unpruned_ones(tmp_path, monkeypatch):
    seen = []

    def fake_g2(ctx, answers=None):
        seen.append(answers)
        return [Check(f"{'G2u' if answers == 'unpruned' else 'G2'} keys", True)]
    monkeypatch.setattr(G, "g2", fake_g2)
    rep = G.run(only=["G2"], answers="unpruned", out=tmp_path)
    assert rep.ok and seen == ["release", "unpruned"]
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["gates"]["G2"]["status"] == "PASS" and report["gates"]["G2u"]["status"] == "PASS"
    seen.clear()
    G.run(only=["G2"], out=tmp_path)
    assert seen == ["release"]


def test_status_blocks_and_readme_table():
    text = "a\n<!-- STATUS:BEGIN -->\n| x |\n|---|\n<!-- STATUS:END -->\nb\n"
    assert G.status_blocks(text) == ["| x |\n|---|"]
    assert G.status_blocks("none") == []
    table = ("| Paper item | Numbers | File | Command | Status |\n|---|---|---|---|---|\n"
             "| main | .382 | `habitus/main_config.csv`, `habitus/summary_means.json` | reproduce | REPRODUCED, G1 + G4 |\n"
             "| rules | - | not_included.md | - | EXCLUDED by framing |\n")
    rows = G.readme_table(table)
    assert rows == [("`habitus/main_config.csv`, `habitus/summary_means.json`", "REPRODUCED, G1 + G4"),
                    ("not_included.md", "EXCLUDED by framing")]
    assert G._paths_of(rows[0][0]) == ["habitus/main_config.csv", "habitus/summary_means.json"]
    assert G._paths_of("a.csv + b.csv") == ["a.csv", "b.csv"]
    assert G._paths_of("habitus/ablation_e1_e2.csv (E2 k2 composite)") == ["habitus/ablation_e1_e2.csv"]
    assert G._paths_of("–") == [] and G._paths_of("none") == []
    assert G._exists([REPO], "tests/golden/*.json") and not G._exists([REPO], "tests/nothing_*.zz")


def test_verify_sums(tmp_path):
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "f.txt").write_text("hello")
    h = hashlib.sha256(b"hello").hexdigest()
    s = tmp_path / "SUMS"
    s.write_text(f"{h}  d/f.txt\n")
    assert G.verify_sums(s, tmp_path) == (1, [])
    s.write_text(f"{'0' * 64}  d/f.txt\n{h}  d/missing.txt\nnot a line\n")
    n, bad = G.verify_sums(s, tmp_path)
    assert n == 2 and len(bad) == 3


def test_compare_states_against_a_golden(tmp_path):
    golden = {"head": {"process": ["s1"], "file": ["s2"], "netflow": []}, "digest": "d", "n_states": 2,
              "states": {"s1": {"qv": "a", "qt": "b"}, "s2": {"qv": "c", "qt": "d"}}}
    p = tmp_path / "x.jsonl"
    lines = [json.dumps({"meta": {"digest": "d", "n_states": 2}, "head": golden["head"]})]
    lines += [json.dumps({"sig": s, "qid": q, "sha1": h}) for s, per in golden["states"].items() for q, h in per.items()]
    p.write_text("\n".join(lines) + "\n")
    assert all(c.ok for c in G.compare_states(p, golden))
    golden["states"]["s2"]["qt"] = "e"
    assert [c.ok for c in G.compare_states(p, golden)] == [True, False, True]
    e1 = {"head": golden["head"], "digest": "d", "n_states": 2, "states": {"s1": "a", "s2": "c"}}
    p.write_text("\n".join([lines[0], json.dumps({"sig": "s1", "qid": None, "sha1": "a"}),
                            json.dumps({"sig": "s2", "qid": None, "sha1": "c"})]) + "\n")
    assert all(c.ok for c in G.compare_states(p, e1))


def test_provenance_and_salvage_records():
    checks = {c.name: c for c in G.provenance_checks(REPO)}
    assert checks["G10 docs/PROVENANCE.md"].ok, checks["G10 docs/PROVENANCE.md"].detail
    if not (REPO / "runs" / "salvage" / "SALVAGE.json").exists():
        pytest.skip("runs/salvage not present (scripts/dev/salvage.py runs on the build server)")
    assert checks["G10 runs/salvage"].ok, checks["G10 runs/salvage"].detail


def test_salvage_check_is_server_only(tmp_path):
    """Outside the build server (no runs/salvage, paths.pidsmaker_src unset) the check is skipped."""
    import shutil
    (tmp_path / "scripts" / "dev").mkdir(parents=True)
    shutil.copy(REPO / "scripts" / "dev" / "snapshot_sources.py", tmp_path / "scripts" / "dev")
    (tmp_path / "docs").mkdir()
    shutil.copy(REPO / "docs" / "PROVENANCE.md", tmp_path / "docs")
    off = {c.name: c for c in G.provenance_checks(tmp_path, server=False)}
    assert off["G10 runs/salvage"].ok and "skipped" in off["G10 runs/salvage"].detail
    on = {c.name: c for c in G.provenance_checks(tmp_path, server=True)}
    assert not on["G10 runs/salvage"].ok


def test_fresh_dir_refuses_foreign_dirs(tmp_path):
    d = RP.fresh_dir(tmp_path / "out")
    (d / "x").write_text("1")
    d = RP.fresh_dir(tmp_path / "out")                     # ours: recreated empty
    assert not (d / "x").exists() and (d / RP.OWNED).exists()
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("1")
    with pytest.raises(FileExistsError):
        RP.fresh_dir(foreign)
    assert (foreign / "keep.txt").exists()


def test_run_children_caps_parallelism_and_reports_rc(tmp_path):
    env = RP.child_env(write_roots=None)
    specs = [(f"c{i}", [sys.executable, "-c", f"import sys; sys.exit({i % 2})"], env, tmp_path / f"c{i}.log")
             for i in range(4)]
    res = RP.run_children(specs, jobs=9)
    assert {n: r["rc"] for n, r in res.items()} == {"c0": 0, "c1": 1, "c2": 0, "c3": 1}


# --------------------------------------------------------------------------- #
# scripts/make_release.py helpers (the script itself runs in the Gates phase)
# --------------------------------------------------------------------------- #
def _make_release():
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_release", REPO / "scripts" / "make_release.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_release_answer_pruning_keeps_first_record_in_order(tmp_path):
    import gzip
    MR = _make_release()
    art = tmp_path / "research" / "art"
    src = art / "reflex" / "kcs1" / "answers" / MR.TAG
    src.mkdir(parents=True)
    k = ["%040x" % i for i in range(4)]
    lines = [json.dumps({"k": k[2], "z": [1, 0, 0, 0], "lse": 1.0, "n": 2, "cov": 0.9}),
             json.dumps({"k": k[0], "z": [0, 1, 0, 0], "lse": 1.0, "n": 2, "cov": 0.8}),
             "garbage",
             json.dumps({"k": k[2], "z": [9, 9, 9, 9], "lse": 9.0, "n": 2, "cov": 0.1}),
             json.dumps({"k": k[3], "z": [0, 0, 1, 0], "lse": 1.0, "n": 3, "cov": 0.7})]
    (src / "FAKE.jsonl").write_text("\n".join(lines) + "\n")
    b = MR.Bundle(MR.fresh(tmp_path / "stage"), tmp_path / "research")
    MR.write_answers(b, art, "FAKE", {k[2], k[3]})
    out = b.stage / "answers" / MR.TAG / "FAKE.jsonl.gz"
    got = gzip.open(out, "rt").read().splitlines()
    assert got == [lines[0], lines[4]]                     # first record of a key, original order
    assert out.read_bytes()[4:8] == b"\0\0\0\0"            # gzip mtime 0
    row = b.files[f"answers/{MR.TAG}/FAKE.jsonl.gz"]
    assert row["lines"] == 2 and row["source_lines"] == 5 and row["unparsed_lines"] == 1
    assert row["source"] == f"research:art/reflex/kcs1/answers/{MR.TAG}/FAKE.jsonl"
    first = out.read_bytes()
    MR.write_answers(b, art, "FAKE", {k[2], k[3]})
    assert out.read_bytes() == first                        # deterministic
    with pytest.raises(SystemExit, match="not in the source cache"):
        MR.write_answers(b, art, "FAKE", {k[1]})


def test_release_copy_refuses_stats_and_locks_and_records_sources(tmp_path):
    MR = _make_release()
    root = tmp_path / "research"
    (root / "cards").mkdir(parents=True)
    for n in ("stats.json", "x.jsonl.lock", "2022-07-19.json.gz"):
        (root / "cards" / n).write_bytes(b"{}")
    b = MR.Bundle(MR.fresh(tmp_path / "stage"), root)
    for n in ("stats.json", "x.jsonl.lock"):
        with pytest.raises(SystemExit, match="refusing"):
            b.copy(root / "cards" / n, f"cards/{n}")
    b.copy(root / "cards" / "2022-07-19.json.gz", "cards/X/2022-07-19.json.gz")
    row = b.files["cards/X/2022-07-19.json.gz"]
    assert row["source"] == "research:cards/2022-07-19.json.gz" and row["sha256"] == row["source_sha256"]
    with pytest.raises(SystemExit):
        b.source_rel("/" + "elsewhere/file")
    with pytest.raises(SystemExit, match="not created by make_release"):
        foreign = tmp_path / "foreign"
        foreign.mkdir()
        MR.fresh(foreign)


def test_release_metadata_is_scan_clean(tmp_path):
    """MANIFEST.json / README.md / PUBLISH.md / SHA256SUMS of a bundle carry no server path or host name."""
    from habitus.repro import scan as SC
    MR = _make_release()
    root = tmp_path / "research"
    (root / "d").mkdir(parents=True)
    (root / "d" / "f.csv").write_text("1\n")
    b = MR.Bundle(MR.fresh(tmp_path / "stage"), root)
    b.copy(root / "d" / "f.csv", "ground_truth/orthrus/E3-CADETS/f.csv")
    MR.write_meta(b, list(MR.ALLOWED), False, [])
    files = [b.stage / n for n in G.RELEASE_TEXT]
    assert all(f.exists() for f in files)
    for fn in (SC.scan_paths, SC.scan_secrets, SC.scan_emails, SC.scan_hosts):
        assert fn(files, root=b.stage) == [], fn.__name__
    n, bad = G.verify_sums(b.stage / "SHA256SUMS", b.stage)
    assert n == 4 and bad == []                            # f.csv, MANIFEST.json, README.md, PUBLISH.md


def test_release_carries_license_and_notice_and_refresh_meta_keeps_rows(tmp_path):
    """LICENSE / NOTICE of the repo at the bundle root (Apache-2.0 4(a)/(d) for the tokenizer files);
    --refresh-meta adds them to an existing bundle and keeps every other MANIFEST row."""
    MR = _make_release()
    root = tmp_path / "research"
    (root / "d").mkdir(parents=True)
    (root / "d" / "f.csv").write_text("1\n")
    b = MR.Bundle(MR.fresh(tmp_path / "stage"), root)
    b.copy(root / "d" / "f.csv", "ground_truth/orthrus/E3-CADETS/f.csv")
    MR.write_meta(b, list(MR.ALLOWED), False, [])
    marker = (b.stage / MR.OWNED).read_text()
    (b.stage / MR.OWNED).unlink()
    with pytest.raises(SystemExit, match="not a bundle"):
        MR.refresh_meta(b.stage)
    (b.stage / MR.OWNED).write_text(marker)
    before = json.loads((b.stage / "MANIFEST.json").read_text(encoding="utf-8"))
    assert MR.refresh_meta(b.stage) == 0
    man = json.loads((b.stage / "MANIFEST.json").read_text(encoding="utf-8"))
    for name in ("LICENSE", "NOTICE"):
        assert (b.stage / name).read_bytes() == (REPO / name).read_bytes()
        assert man["files"][name]["source"] == f"repo:{name}"
    assert man["files"]["ground_truth/orthrus/E3-CADETS/f.csv"] == before["files"]["ground_truth/orthrus/E3-CADETS/f.csv"]
    assert man["created"] == before["created"] and "meta_refreshed" in man
    n, bad = G.verify_sums(b.stage / "SHA256SUMS", b.stage)
    assert n == 6 and bad == []
    pub = (b.stage / "PUBLISH.md").read_text(encoding="utf-8")
    assert "only when ground_truth is published too" in pub and "LICENSE and NOTICE" in pub


def test_release_ships_the_laya_tokenizer_under_the_dm_laya_dir(tmp_path):
    """make_release copies the 2 Laya tokenizer files to <tokenizers>/laya.RELEASE_TOKENIZER_DIR."""
    from habitus.dm import laya as LA
    MR = _make_release()
    root = tmp_path / "research"
    src = root / "hub" / "typed-decisions" / "tokenizer"
    src.mkdir(parents=True)
    for n in MR.LAYA_TOKENIZER_FILES:
        (src / n).write_text("{}\n")
    b = MR.Bundle(MR.fresh(tmp_path / "stage"), root)
    MR.copy_laya_tokenizer(b, root / "hub")
    assert MR.LAYA_TOKENIZER_DIR == LA.RELEASE_TOKENIZER_DIR
    assert sorted(b.files) == [f"tokenizer/{LA.RELEASE_TOKENIZER_DIR}/{n}" for n in sorted(MR.LAYA_TOKENIZER_FILES)]
    MR.write_meta(b, list(MR.ALLOWED), False, [])
    man = json.loads((b.stage / "MANIFEST.json").read_text(encoding="utf-8"))
    assert man["laya_tokenizer"]["dir"] == f"tokenizer/{LA.RELEASE_TOKENIZER_DIR}"


def test_data_gates_fail_fast_when_their_inputs_are_missing(tmp_path, monkeypatch):
    """Without the release bundle G1/G2/G3/G4/G6/G9 fail on one 'inputs' check naming the missing
    paths and the prerelease view, and spawn no child."""
    from habitus import config

    st = config.load_settings(overrides={"paths": {"data": str(tmp_path / "no_release")}}, env={})
    monkeypatch.setattr(RP, "settings", lambda: st)
    monkeypatch.setattr(RP, "run_children", lambda *a, **k: pytest.fail("a child was spawned"))
    ctx = G.Ctx(tmp_path / "out", 1, "release", None, None)
    for g in ("G1", "G2", "G3", "G4", "G6", "G9"):
        checks = G.FUNCS[g](ctx)
        assert [c.name for c in checks] == [f"{g} inputs"] and not checks[0].ok, g
        assert "configs/prerelease.yaml" in checks[0].detail and "cards" in checks[0].detail, g
    (tmp_path / "no_release" / "cards").mkdir(parents=True)
    (tmp_path / "no_release" / "precedent_cache").mkdir()
    assert G.preflight(ctx, "G3") == [] and G.preflight(ctx, "G2u") == []
    assert "answers" in G.preflight(ctx, "G2")[0].detail
