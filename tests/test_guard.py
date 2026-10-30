"""Label guard and write audit (habitus.guard). The hook cannot be removed, so every hook test runs in
a subprocess."""

import json
import os
import subprocess
import sys
import textwrap

import pytest

from habitus import guard as G

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT_DIR = "Ground" + "_Truth"


def _run(code, tmp_path, env_extra=None):
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "src"), PYTHONDONTWRITEBYTECODE="1",
               HABITUS_RUNS=str(tmp_path / "runs"), **(env_extra or {}))
    p = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=env, capture_output=True, text=True,
                       cwd=str(tmp_path), timeout=120)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------------- #
# Pure checks
# --------------------------------------------------------------------------- #
def test_blocked_reason_patterns(tmp_path):
    roots = (str(tmp_path / "bundle" / "gt"),)
    assert G.blocked_reason(tmp_path / "bundle" / "gt" / "x.csv", roots) == "configured ground-truth directory"
    assert G.blocked_reason(f"/a/{GT_DIR}/orthrus/E3-X/node.csv", ()) == "ground-truth tree"
    assert G.blocked_reason(f"/a/{GT_DIR.lower()}/x", ()) == "ground-truth tree"
    assert G.blocked_reason("/a/_logs/g" + "t_x.pkl", ()) == "upstream label pickle"
    assert G.blocked_reason("/a/b__n2a.pkl", ()) == "node-to-attack map"
    assert G.blocked_reason("/a/host" + "_labels.csv", ()) == "label file"
    assert G.blocked_reason("/a/cards/X/stat" + "s.json", ()) == "card statistics"
    for ok in ("/a/_expand" + "_labels.py", "/a/b/cards/X/2018-04-02.json.gz", "/a/gt.pkl", "/a/labelling.py",
               str(tmp_path / "bundle" / "gtx" / "y.csv")):
        assert G.blocked_reason(ok, roots) is None, ok
    assert G.blocked_import("habitus.eval") and G.blocked_import("habitus.eval.metrics")
    assert not G.blocked_import("habitus.evaluation") and not G.blocked_import("habitus.decide")


def test_label_roots_from_settings(tmp_path):
    from habitus.config import load_settings
    s = load_settings(local=False, env={"HABITUS_GT": str(tmp_path / "g")},
                      overrides={"paths": {"upstream_gt": str(tmp_path / "u")}})
    assert G.label_roots(s) == (str(tmp_path / "g"), str(tmp_path / "u"))


def test_exempt_commands():
    assert G.EXEMPT == ("eval", "reproduce", "gates")


# --------------------------------------------------------------------------- #
# The hook (subprocess)
# --------------------------------------------------------------------------- #
def test_guard_refuses_gt_reads_and_eval_imports(tmp_path):
    gt = tmp_path / "gt"
    gt.mkdir()
    (gt / "node.csv").write_text("x\n")
    tree = tmp_path / GT_DIR
    tree.mkdir()
    (tree / "a.csv").write_text("x\n")
    (tmp_path / "fine.txt").write_text("ok\n")
    res = _run(f"""
        import json, importlib, os
        from habitus import guard
        out = {{"installed": guard.install_guard("decide"), "active": guard.guard_active()}}
        def tryit(f):
            try:
                f(); return "ok"
            except guard.LabelGuardError:
                return "refused"
        out["gt_root"] = tryit(lambda: open({str(gt / 'node.csv')!r}).read())
        out["gt_os_open"] = tryit(lambda: os.close(os.open({str(gt / 'node.csv')!r}, os.O_RDONLY)))
        out["gt_tree"] = tryit(lambda: open({str(tree / 'a.csv')!r}).read())
        out["fine"] = tryit(lambda: open({str(tmp_path / 'fine.txt')!r}).read())
        out["import_stmt"] = tryit(lambda: exec("import habitus.eval.metrics"))
        out["import_module"] = tryit(lambda: importlib.import_module("habitus.eval"))
        out["same_again"] = guard.install_guard("decide")
        try:
            guard.install_guard("state"); out["other"] = "ok"
        except ValueError:
            out["other"] = "raised"
        print(json.dumps(out))
    """, tmp_path, {"HABITUS_GT": str(gt)})
    assert res == {"installed": True, "active": "decide", "gt_root": "refused", "gt_os_open": "refused",
                   "gt_tree": "refused", "fine": "ok", "import_stmt": "refused", "import_module": "refused",
                   "same_again": True, "other": "raised"}


def test_exempt_command_installs_no_label_guard(tmp_path):
    gt = tmp_path / "gt"
    gt.mkdir()
    (gt / "node.csv").write_text("x\n")
    res = _run(f"""
        import json
        from habitus import guard
        out = {{"installed": guard.install_guard("eval"), "active": guard.guard_active()}}
        out["read"] = open({str(gt / 'node.csv')!r}).read()
        print(json.dumps(out))
    """, tmp_path, {"HABITUS_GT": str(gt)})
    assert res == {"installed": False, "active": None, "read": "x\n"}


def test_label_code_imported_before_guard_is_an_error(tmp_path):
    res = _run("""
        import json, sys, types
        sys.modules["habitus.eval"] = types.ModuleType("habitus.eval")
        from habitus import guard
        try:
            guard.install_guard("decide"); out = "installed"
        except guard.LabelGuardError:
            out = "refused"
        print(json.dumps(out))
    """, tmp_path)
    assert res == "refused"


def test_write_audit(tmp_path):
    ok_root = tmp_path / "allowed"
    ok_root.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    (other / "existing.txt").write_text("r\n")
    res = _run(f"""
        import json, os, tempfile
        from habitus import guard
        guard.install_guard("decide", write_roots=[{str(ok_root)!r}])
        def tryit(f):
            try:
                f(); return "ok"
            except guard.WriteAuditError:
                return "refused"
        A, O = {str(ok_root)!r}, {str(other)!r}
        out = {{}}
        out["write_in"] = tryit(lambda: open(os.path.join(A, "x.txt"), "w").write("1"))
        out["mkdir_in"] = tryit(lambda: os.makedirs(os.path.join(A, "d", "e")))
        out["rename_in"] = tryit(lambda: os.replace(os.path.join(A, "x.txt"), os.path.join(A, "y.txt")))
        out["write_out"] = tryit(lambda: open(os.path.join(O, "x.txt"), "w"))
        out["append_out"] = tryit(lambda: open(os.path.join(O, "existing.txt"), "a"))
        out["os_open_out"] = tryit(lambda: os.open(os.path.join(O, "z"), os.O_WRONLY | os.O_CREAT))
        out["mkdir_out"] = tryit(lambda: os.mkdir(os.path.join(O, "d")))
        out["rename_out"] = tryit(lambda: os.rename(os.path.join(A, "y.txt"), os.path.join(O, "y.txt")))
        out["remove_out"] = tryit(lambda: os.remove(os.path.join(O, "existing.txt")))
        out["read_out"] = tryit(lambda: open(os.path.join(O, "existing.txt")).read())
        out["files_out"] = sorted(os.listdir(O))
        out["roots"] = guard.write_roots() is not None
        print(json.dumps(out))
    """, tmp_path)
    assert res == {"write_in": "ok", "mkdir_in": "ok", "rename_in": "ok", "write_out": "refused",
                   "append_out": "refused", "os_open_out": "refused", "mkdir_out": "refused",
                   "rename_out": "refused", "remove_out": "refused", "read_out": "ok",
                   "files_out": ["existing.txt"], "roots": True}


def test_write_audit_sees_through_symlinks_and_works_for_exempt(tmp_path):
    ok_root = tmp_path / "allowed"
    ok_root.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    os.symlink(other, ok_root / "escape")
    res = _run(f"""
        import json, os
        from habitus import guard
        out = {{"label": guard.install_guard("gates", write_roots=[{str(ok_root)!r}])}}
        try:
            open(os.path.join({str(ok_root)!r}, "escape", "x"), "w"); out["escape"] = "ok"
        except guard.WriteAuditError:
            out["escape"] = "refused"
        print(json.dumps(out))
    """, tmp_path)
    assert res == {"label": False, "escape": "refused"}
    assert not (other / "x").exists()


def test_write_audit_allows_the_null_device(tmp_path):
    """torch's import of dill opens os.devnull 'r+b': the audit must not refuse it (it writes
    nothing), and the pure check must agree."""
    ok_root = tmp_path / "allowed"
    ok_root.mkdir()
    res = _run(f"""
        import json, os
        from habitus import guard
        guard.install_guard("decide", write_roots=[{str(ok_root)!r}])
        out = {{"pure": guard.write_violation(os.devnull, guard.write_roots())}}
        try:
            with open(os.devnull, "r+b") as fh:
                fh.write(b"x")
            out["devnull"] = "ok"
        except guard.WriteAuditError:
            out["devnull"] = "refused"
        print(json.dumps(out))
    """, tmp_path)
    assert res == {"pure": False, "devnull": "ok"}


def test_install_write_audit_needs_roots():
    with pytest.raises(ValueError):
        G.install_write_audit([])
