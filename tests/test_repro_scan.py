"""repro.scan: the G5 / G8 scanners find what they must, honour their allow-lists and never match
their own source. Every forbidden string below is assembled at run time, so this file stays clean."""

import os
import textwrap
from pathlib import Path

import pytest

from habitus.repro import scan as S

REPO = Path(__file__).resolve().parents[1]
SLASH = "/"
HOME = SLASH + "ho" + "me" + SLASH
TMP = SLASH + "tm" + "p" + SLASH
DISK = SLASH + "DA" + "TA"
AT = "@"


def _w(tmp_path, name, text):
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_paths_found_and_fixture_paths_tolerated(tmp_path):
    scratch = "scratch" + "pad"
    bad = _w(tmp_path, "a.py", f"X = '{HOME}someuser/project/x'\nY = '{TMP}session-12/out'\nZ = '{DISK}/kien'\n"
                               f"V = 'my {scratch}/x'\n")
    hits = S.scan_paths([bad], root=tmp_path)
    assert {h.rule for h in hits} == {"home", "tmp", "former_data_disk", "scratch_dir"}
    ok = _w(tmp_path, "fixture.py", f"P = '{TMP}x9q'\nQ = '{HOME}u/out.txt'\nR = 'runs{TMP}x/y'\nS = '{HOME}'\n"
                                    f"T = '/data/app/'\nU = '${{CYRAG_ROOT}}/PIDSMaker'\n")
    assert S.scan_paths([ok], root=tmp_path) == []


def test_this_machines_home_is_matched_literally(tmp_path):
    home = os.path.expanduser("~")
    if home.count("/") < 2:
        pytest.skip("no absolute home directory")
    p = _w(tmp_path, "b.md", f"see {home}/x\n")
    assert "this_home" in {h.rule for h in S.scan_paths([p], root=tmp_path)}


def test_secrets_env_files_and_size(tmp_path):
    key = "s" + "k-" + "A1b2C3d4E5f6G7h8I9j0K1l2"
    tok = "h" + "f_" + "a" * 34
    lines = [f"k = {key}", "OPEN" + "ROUTER_KEY=1", "Authorization: Bea" + "rer abc", "api" + "_key = 'zzz'", tok]
    p = _w(tmp_path, "c.txt", "\n".join(lines) + "\n")
    rules = {h.rule for h in S.scan_secrets([p], root=tmp_path)}
    assert rules == {"sk_key", "openrouter", "bearer", "api_key_assign", "hf_token"}
    env = _w(tmp_path, ".en" + "v.local", "A=1\n")
    assert [h.rule for h in S.scan_secrets([env], root=tmp_path)] == ["env_file"]
    big = tmp_path / "big.bin"
    big.write_bytes(b"\0" * 2048)
    assert [h.rule for h in S.scan_secrets([big], root=tmp_path, max_bytes=1024)] == ["too_large"]
    clean = _w(tmp_path, "d.py", "task_identifier_abcdefghijklmnop = 1\nhf_hub_download = None\n")
    assert S.scan_secrets([clean], root=tmp_path) == []


def test_emails_skip_notice_and_license(tmp_path):
    addr = "someone" + AT + "example" + ".org"
    a = _w(tmp_path, "README.md", f"contact {addr}\n")
    n = _w(tmp_path, "NOTICE", f"Copyright {addr}\n")
    lic = _w(tmp_path, "LICENSE", f"{addr}\n")
    assert [h.file for h in S.scan_emails([a, n, lic], root=tmp_path)] == ["README.md"]
    code = _w(tmp_path, "e.py", "@property\ndef f(x):\n    return x@w\n")
    assert S.scan_emails([code], root=tmp_path) == []


def test_email_scan_is_never_given_card_text():
    """G8 scans tracked files + the NON-card release text only (cards hold e-mail-like log strings)."""
    from habitus.repro import gates as G
    assert set(G.RELEASE_TEXT) == {"MANIFEST.json", "SHA256SUMS", "README.md", "PUBLISH.md"}
    assert not any("card" in f or "answers" in f or "scores" in f for f in G.RELEASE_TEXT)


def test_out_of_scope_hosts(tmp_path):
    names = ["CLEARSCOPE" + "_E5", "cadets" + "_e5", "E5" + "-THEIA", "CARBANAK" + "V2_EDR", "carbanak" + "v2_edr",
             "optc_h" + "051", "optc_h" + "201", "h20" + "1", "TRACE" + "_E3", "FIVEDIRECTIONS" + "_E3",
             "atlasv2" + "_h1", "ATLASV2" + "_H1", "ATLASV2" + "_H2", "atlasv2" + "-h12_x"]
    p = _w(tmp_path, "f.md", "\n".join(names) + "\n")
    hits = S.scan_hosts([p], root=tmp_path)
    assert len(hits) == len(names)
    ok = _w(tmp_path, "g.md", "CADETS_E3 THEIA_E3 CLEARSCOPE_E3 optc_h501 ATLASV2_EDR E3-CADETS h501 "
                              "the carbanak campaign NOT_A_HOST\n")
    assert S.scan_hosts([ok], root=tmp_path) == []
    assert S.redact_hosts("x " + names[0] + " y") == "x [out-of-scope host] y"
    assert S.hosts_in("a " + names[3]) == [names[3]]
    assert S.hosts_in("Ground_Truth/orthrus/" + "atlasv2" + "_h1/") == ["atlasv2" + "_h1"]


def test_scanners_do_not_match_their_own_source():
    me = [REPO / "src" / "habitus" / "repro" / "scan.py", REPO / "src" / "habitus" / "repro" / "gates.py",
          Path(__file__)]
    for fn in (S.scan_paths, S.scan_secrets, S.scan_emails, S.scan_hosts):
        assert fn(me, root=REPO) == [], fn.__name__
    # the label tokens of scan.py / gates.py are allow-listed, but scan.py is char-classed anyway
    text = (REPO / "src" / "habitus" / "repro" / "scan.py").read_text(encoding="utf-8")
    assert S.scan_text(text, S.LABEL_PATTERNS, kind="label") == []


def test_b8_files_are_clean():
    mine = ["src/habitus/cli.py", "src/habitus/repro/__init__.py", "src/habitus/repro/compare.py",
            "src/habitus/repro/reproduce.py", "scripts/make_release.py", "scripts/dev/snapshot_sources.py",
            "scripts/dev/salvage.py", "pyproject.toml", "requirements-lock.txt", "Makefile", "LICENSE", "NOTICE",
            "README.md", "docs/REPRODUCE.md", "docs/PROVENANCE.md", ".gitignore", "tests/test_cli.py",
            "tests/test_gates.py", "tests/test_repro_compare.py", "tests/test_repro_scan.py"]
    files = [REPO / f for f in mine if (REPO / f).exists()]
    for fn in (S.scan_paths, S.scan_secrets, S.scan_emails, S.scan_hosts):
        assert fn(files, root=REPO) == [], fn.__name__


def test_label_scan_allow_list(tmp_path):
    pkg = tmp_path / "habitus"
    call = "g" + "t_nodes("
    _w(pkg, "eval/labels.py", f"def f(ds):\n    return {call}ds)\n")
    _w(pkg, "config.py", "KEY = 'ground" + "_truth'\n")
    _w(pkg, "cli.py", f"x = {call}1)\n")
    _w(pkg, "decide/bad.py", f"y = {call}1)\nz = settings.paths.ground" + "_truth\n")
    _w(pkg, "data/fine.py", "def f(a):\n    return a.gt_total + 1\n")
    hits = S.scan_labels(pkg)
    assert sorted((h.file, h.rule) for h in hits) == [("habitus/decide/bad.py", "gt_nodes"),
                                                      ("habitus/decide/bad.py", "gt_path")]
    script = _w(tmp_path, "scripts/other.py", "p = 'Ground" + "_Truth/orthrus'\n")
    rel = _w(tmp_path, "scripts/make_release.py", "p = 'Ground" + "_Truth/orthrus'\n")
    hits = S.scan_labels(pkg, extra_files=[script, rel], repo_root=tmp_path)
    assert {h.file for h in hits} == {"habitus/decide/bad.py", "scripts/other.py"}


def _pkg(tmp_path, files):
    root = tmp_path / "src" / "habitus"
    for rel, text in files.items():
        _w(root, rel, textwrap.dedent(text))
    return root


def test_import_graph_rules(tmp_path):
    root = _pkg(tmp_path, {
        "__init__.py": "",
        "cli.py": "from habitus import decide\ndef f():\n    import habitus.eval.labels\n",
        "decide/__init__.py": "from habitus.decide import routed\n",
        "decide/routed.py": "import numpy\nfrom habitus.dm import jev\ndef g():\n    from habitus.eval import metrics\n",
        "dm/__init__.py": "",
        "dm/jev.py": "def load():\n    import torch\n",
        "dm/laya.py": "import torch\n",
        "data/__init__.py": "",
        "data/upstream.py": "def cfg():\n    import pidsmaker.config\n",
        "data/cards.py": "import pidsmaker.utils\nfrom pidsmaker.reflex import kshot\n",
        "state/__init__.py": "",
        "state/x.py": "def f():\n    import torch\n",
        "eval/__init__.py": "",
        "eval/metrics.py": "from . import labels\n",
        "eval/labels.py": "",
        "repro/__init__.py": "",
        "repro/gates.py": "import importlib\nM = importlib.import_module('habitus.eval.tables')\n",
    })
    g = S.import_graph(root)
    assert {i.target for i in g["habitus.eval.metrics"]} == {"habitus.eval.labels"}
    hits = {(h.file, h.rule) for h in S.import_problems(g, torch_free_roots=("habitus.cli", "habitus.dm.jev"))}
    assert hits == {
        ("habitus/decide/routed.py", "eval_import"),             # decide never imports eval
        ("habitus/data/cards.py", "pidsmaker_outside_upstream"),
        ("habitus/data/cards.py", "forbidden_upstream"),
        ("habitus/state/x.py", "torch_outside_dm_train"),
    }
    # dm.laya imports torch at module level: fine unless a torch-free entry point reaches it eagerly
    hits = S.import_problems(g, torch_free_roots=("habitus.dm.laya",))
    assert ("habitus/dm/laya.py", "torch_at_import_time") in {(h.file, h.rule) for h in hits}


def test_the_package_import_graph_parses():
    g = S.import_graph(REPO / "src" / "habitus")
    assert "habitus.cli" in g and "habitus.repro.scan" in g
    assert not [i for imps in g.values() for i in imps if i.target.startswith("<syntax error")]
    # the CLI imports nothing of habitus at module level (the guard goes first)
    assert not [i for i in g["habitus.cli"] if not i.lazy and i.target.startswith("habitus")]


def test_tracked_files_and_under():
    files = S.tracked_files(REPO)
    rel = {f.relative_to(REPO).as_posix() for f in files}
    assert "src/habitus/repro/scan.py" in rel
    assert not any(r.startswith(("runs/", "release/")) or r == "configs/local.yaml" for r in rel)
    sub = S.under(files, REPO, ("src/", "NOTICE"))
    assert all(f.relative_to(REPO).as_posix().startswith("src/") or f.name == "NOTICE" for f in sub)
