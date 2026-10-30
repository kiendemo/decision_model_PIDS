"""Evidence cards (habitus.data.cards and the vendored graph / resolver / identity / profile modules).

Fixtures pin signature() and render() bytes; a synthetic day pins the card builder; the provenance
test compares every vendored definition with the original source (AST, docstrings ignored); with the
PIDSMaker artifacts configured, one real day (ATLASV2_EDR 2022-07-19) is rebuilt into runs/ and must
be JSON-equal to the shipped card and link files."""

import ast
import os

import networkx as nx
import pytest

from habitus.data import cards as C
from habitus.data import graphs as GR
from habitus.data import identity as I
from habitus.data import profiles as P
from habitus.data import resolver as R
from habitus.io import load_json

from conftest import asset, missing_asset

PROC = {"v": 2, "type": "process", "os": "linux",
        "invariant": {"file_name": "x9q", "executable": "/tmp/x9q", "command_line": "/tmp/x9q -c <n>",
                      "parent_name": "bash", "image_dir": "staging", "image_deleted": "yes"},
        "behaviour": {"ops": {"connect:net-external": "1", "exec:staging": "1", "write:staging": "2-7"},
                      "children": ["sh"], "write_dirs": ["staging"], "write_exts": ["sh"],
                      "exec_dirs": ["staging"], "peers": {"external": "2-7", "internal": "1"},
                      "ports": ["4444", "443"], "flags": ["executed_file_written_today", "wrote_file_executed_later"]}}
FILE = {"v": 2, "type": "file", "os": "windows",
        "invariant": {"path": "C:/users/u/appdata/local/temp/<hex:32>.exe", "dir": "staging", "ext": "exe"},
        "behaviour": {"ops": {"exec:process": "1", "write:process": "1"}, "writers": ["a.exe"], "executors": ["a.exe"],
                      "readers": "2-7", "flags": ["executed_after_write"]}}
NET = {"v": 2, "type": "netflow", "os": "freebsd", "invariant": {"remote_ip": "", "remote_port": "", "endpoint": "unknown"},
       "behaviour": {"ops": {"connect:process": "1"}, "processes": ["nginx"]}}

# from the ORIGINAL reflex/cards.py signature() / render() (extracted by AST and run on these cards)
SIGS = {"process": "8e5a33d6a23b62cd", "file": "39fb23b9aeb33e99", "netflow": "47e748727f08ee7a"}
RENDER = {
    "process": ("Node type: process on a linux host\nFile name: x9q\nExecutable: /tmp/x9q\nCommand line: /tmp/x9q -c <n>\n"
                "Started by: bash\nExecutable directory class: staging\nExecutable deleted from disk: yes\n"
                "Activity during the day (operation:object class = count): connect:net-external=1, exec:staging=1, "
                "write:staging=2-7\nChildren: sh\nWrite dirs: staging\nWrite exts: sh\nExec dirs: staging\n"
                "Ports: 4444, 443\nNetwork peers by class: external=2-7, internal=1\n"
                "Observed: executed file written today, wrote file executed later"),
    "file": ("Node type: file on a windows host\nPath: C:/users/u/appdata/local/temp/<hex:32>.exe\nDirectory class: staging\n"
             "Extension: exe\nActivity during the day (operation:object class = count): exec:process=1, write:process=1\n"
             "Writers: a.exe\nExecutors: a.exe\nDistinct reading processes: 2-7\nObserved: executed after write"),
    "netflow": ("Node type: netflow on a freebsd host\nEndpoint class: unknown\n"
                "Activity during the day (operation:object class = count): connect:process=1\nProcesses: nginx"),
}
RENDER_INV = {
    "process": ("Node type: process on a linux host\nFile name: x9q\nExecutable: /tmp/x9q\nCommand line: /tmp/x9q -c <n>\n"
                "Started by: bash\nExecutable directory class: staging\nExecutable deleted from disk: yes"),
    "file": ("Node type: file on a windows host\nPath: C:/users/u/appdata/local/temp/<hex:32>.exe\nDirectory class: staging\n"
             "Extension: exe"),
    "netflow": "Node type: netflow on a freebsd host\nEndpoint class: unknown",
}


@pytest.mark.parametrize("card", [PROC, FILE, NET], ids=lambda c: c["type"])
def test_signature_and_render_fixtures(card):
    t = card["type"]
    assert C.signature(card) == SIGS[t]
    assert C.render(card) == RENDER[t]
    assert C.render(card, sections=("invariant",)) == RENDER_INV[t]
    reordered = {k: card[k] for k in reversed(list(card))}
    assert C.signature(reordered) == SIGS[t]                 # canonical JSON: key order is irrelevant


def test_constants():
    assert C.TOP == 5 and C.CARD_VERSION == 2 and I.MAX_CMD == 300
    assert C.LINK_VERBS == {"spawn": 1, "write": 2, "create": 2, "rename": 2, "exec": 3, "connect": 4, "send": 4}
    assert C.LINK_CODES == {"1": "spawn", "2": "write", "3": "exec", "4": "net"}
    assert GR.CREATION_OPS == frozenset({"EVENT_CLONE", "EVENT_FORK", "CREATE", "PROCESS_EXECUTED"})


def test_bins_top_ext_abstract():
    assert [C.bin_count(n) for n in (0, 1, 2, 7, 8, 31, 32, 127, 128, 10 ** 6)] == \
        ["0", "1", "2-7", "2-7", "8-31", "8-31", "32-127", "32-127", "128+", "128+"]
    assert C._top({"b": 2, "a": 2, "c": 5, "d": 1, "e": 1, "f": 1}) == ["c", "a", "b", "d", "e"]
    assert C._top({"x": 1, "y": 3}, 1) == ["y"]
    assert C.ext_of("/tmp/A.TaR.GZ") == "gz" and C.ext_of("/bin/ls") == "" and C.ext_of("/x/.bashrc") == ""
    assert C.ext_of(r"C:\Windows\System32\cmd.EXE") == "exe" and C.ext_of(None) == ""
    assert C.abstract_text("/tmp/0123456789abcdef0123456789abcdef.bin") == "/tmp/<hex:32>.bin"
    assert C.abstract_text("{12345678-1234-1234-1234-123456789abc}") == "<guid>"
    assert C.abstract_text("job 20180412 run") == "job <n> run"
    assert C.abstract_text("tok aB3dE5gH7jK9mN1pQ end") == "tok <rand:17> end"
    assert C.abstract_text("") == "" and C.abstract_text(None) is None
    assert len(C.abstract_text("a" * 500)) == I.MAX_CMD


def test_verb_and_dir_class():
    tc, optc, cb = P.profile_for("THEIA_E3"), P.profile_for("optc_h501"), P.profile_for("ATLASV2_EDR")
    assert tc is P.DARPA_TC and optc is P.OPTC and cb is P.CARBON_BLACK_EDR
    assert P.profile_for("CADETS_E3") is P.DARPA_TC and P.profile_for("NOT_A_HOST") is P.DARPA_TC
    assert C.verb("EVENT_CLONE", "subject", "subject", tc) == "spawn"
    assert C.verb("EVENT_SIGNAL", "subject", "subject", tc) == "ipc"
    assert C.verb("EVENT_EXECUTE", "file", "subject", tc) == "exec"
    assert C.verb("START", "subject", "netflow", optc) == "connect"
    assert C.verb("MESSAGE", "netflow", "subject", optc) == "recv"
    assert C.verb("FILE_DELETED", "subject", "file", cb) == "delete"
    assert C.verb("EVENT_WRITE", "subject", "netflow", tc) == "send"
    assert C.verb("EVENT_READ", "subject", "netflow", tc) == "recv"
    assert C.verb("EVENT_MMAP", "subject", "file", tc) == "load"
    assert C.verb("REGISTRY_SET", "subject", "file", cb) == "registry"
    assert C.verb("EVENT_FOO", "subject", "file", tc) == "other"
    assert C.dir_class("/tmp/x", tc) == "staging" and C.dir_class("/usr/bin/x", tc) == "system"
    assert C.dir_class("/home/u/.cache/x", tc) == "cache" and C.dir_class("/etc/passwd", tc) == "config"
    assert C.dir_class("/data/data/com.app/f", tc) == "application" and C.dir_class(None, tc) == "none"
    assert C.dir_class("C:/Windows/System32/cmd.exe", cb) == "system"
    assert C.dir_class("C:/Program Files/x/y.exe", cb) == "application"
    assert C.dir_class("C:/Users/u/AppData/Local/Temp/x.exe", cb) == "staging"
    assert C.dir_class("/device/harddiskvolume2/users/u/x.txt", optc) == "home"
    assert C.dir_class("D:/x", cb) == "other"


def test_identity():
    pi = I.process_identity("windows", "/Device/HarddiskVolume1/Windows/System32/cmd.exe",
                            "cmd.exe /c 1234567 {12345678-1234-1234-1234-123456789abc}", parent=("C:/p/explorer.exe", None))
    assert pi.image == "C:/Windows/System32/cmd.exe" and pi.name == "cmd.exe" and pi.parent == "explorer.exe"
    assert pi.cmdline == "cmd.exe /c <n> <guid>"
    assert I.process_identity("freebsd", None, "nginx", exec_path=None).state() == {"os": "freebsd", "file_name": "nginx"}
    assert I.process_identity("linux", "/bin/sh (deleted)", "/bin/sh").state() == {"os": "linux", "file_name": "sh",
                                                                                   "executable": "/bin/sh"}


def test_resolver_tables_and_fail_hard(tmp_path):
    r = R.SemanticResolver.from_tables("THEIA_E3", subject={1: ("/bin/bash", "bash")}, file={10: "/tmp/a"},
                                       netflow={20: ("128.55.12.9", "53", "4000"), 21: ("8.8.8.8", "53", "1")},
                                       host_ip="128.55.12.73")
    assert r.resolve("1", "subject").path == "/bin/bash"
    assert r.resolve("10", "file").path == "/tmp/a" and r.resolve("11", "file").path is None
    assert r.resolve("20", "netflow").endpoint_class == R.EP_ENCLAVE
    assert r.resolve("21", "netflow").endpoint_class == R.EP_EXTERNAL
    e = r.resolve("2", "subject", label="/usr/bin/curl curl")             # no node-table row: label parse
    assert (e.path, e.cmd, e.resolved) == ("/usr/bin/curl", "curl", False)
    assert R.classify_endpoint("127.0.0.1", None) == R.EP_LOOPBACK and R.classify_endpoint("x", None) == R.EP_UNKNOWN
    with pytest.raises(R.ResolverUnavailable, match="no resolver cache"):
        R.SemanticResolver("THEIA_E3", tmp_path / "none.pkl")
    with pytest.raises(R.ResolverUnavailable, match="refusing to fall back"):
        R.SemanticResolver("THEIA_E3", tmp_path / "none.pkl",
                           db={"database": "x", "host": "127.0.0.1", "user": "u", "password": "p", "port": 9})
    assert not (tmp_path / "none.pkl").exists()


# --------------------------------------------------------------------------- #
# A synthetic day through the full builder (torch window files)
# --------------------------------------------------------------------------- #
def _window():
    G = nx.MultiDiGraph()
    for n, t in (("1", "subject"), ("2", "subject"), ("10", "file"), ("20", "netflow")):
        G.add_node(n, node_type=t, label="")
    for u, v, op, t in (("1", "2", "EVENT_CLONE", 1), ("2", "10", "EVENT_WRITE", 2), ("10", "2", "EVENT_EXECUTE", 3),
                        ("2", "20", "EVENT_CONNECT", 4), ("2", "20", "EVENT_SENDTO", 5)):
        G.add_edge(u, v, label=op, time=t, event_uuid=f"e{t}", y=1)
    return G


def _resolver():
    return R.SemanticResolver.from_tables(
        "THEIA_E3", subject={1: ("/usr/bin/bash", "bash -l"), 2: ("/tmp/x9q", "/tmp/x9q -c 1234567")},
        file={10: "/tmp/payload.sh"}, netflow={20: ("6.6.6.6", "4444", "50000")}, host_ip="128.55.12.73")


EXPECTED = {
    "1": {"v": 2, "type": "process", "os": "linux",
          "invariant": {"file_name": "bash", "executable": "/usr/bin/bash", "command_line": "bash -l",
                        "image_dir": "system"},
          "behaviour": {"ops": {"spawn:process": "1"}, "children": ["x9q"]}},
    "2": {"v": 2, "type": "process", "os": "linux",
          "invariant": {"file_name": "x9q", "executable": "/tmp/x9q", "command_line": "/tmp/x9q -c <n>",
                        "parent_name": "bash", "image_dir": "staging"},
          "behaviour": {"ops": {"connect:net-external": "1", "exec:staging": "1", "send:net-external": "1",
                                "write:staging": "1"},
                        "write_dirs": ["staging"], "write_exts": ["sh"], "exec_dirs": ["staging"],
                        "peers": {"external": "1"}, "ports": ["4444"],
                        "flags": ["executed_file_written_today", "wrote_file_executed_later"]}},
    "10": {"v": 2, "type": "file", "os": "linux", "invariant": {"path": "/tmp/payload.sh", "dir": "staging", "ext": "sh"},
           "behaviour": {"ops": {"exec:process": "1", "write:process": "1"}, "writers": ["x9q"],
                         "executors": ["x9q"], "flags": ["executed_after_write"]}},
    "20": {"v": 2, "type": "netflow", "os": "linux",
           "invariant": {"remote_ip": "6.6.6.6", "remote_port": "4444", "endpoint": "external"},
           "behaviour": {"ops": {"connect:process": "1", "send:process": "1"}, "processes": ["x9q"]}},
}


def test_build_day_synthetic(tmp_path):
    torch = pytest.importorskip("torch")
    day = "2018-04-03"
    gdir = tmp_path / "nx" / f"graph_{day}"
    gdir.mkdir(parents=True)
    torch.save(_window(), gdir / f"{day} 00:00:00.000000000~{day} 00:15:00.000000000")
    empty = nx.MultiDiGraph()
    empty.add_node("99", node_type="subject", label="")
    torch.save(empty, gdir / f"{day} 00:15:00.000000000~{day} 00:30:00.000000000")
    paths = GR.graph_paths(tmp_path / "nx", day)
    assert [os.path.basename(p)[11:19] for p in paths] == ["00:00:00", "00:15:00"]
    for _w, G, _span in GR.windows(paths):                     # the label attribute never survives loading
        assert all("y" not in d for _u, _v, d in G.edges(data=True))
    out = tmp_path / "cards"
    C.build_day("THEIA_E3", str(tmp_path / "nx"), _resolver(), day, str(out))
    raw = load_json(C.card_path(out, "THEIA_E3", day))
    assert (raw["day"], raw["dataset"], raw["version"]) == (day, "THEIA_E3", 2)
    got = {n: raw["cards"][s] for n, s in raw["nodes"].items()}
    assert got == EXPECTED
    assert all(C.signature(c) == s for s, c in raw["cards"].items())
    links = load_json(C.link_path(out, "THEIA_E3", day))
    # connect and send to the same netflow are two (cause, effect, verb) records with one code: the
    # original keeps both rows (dedupe is on the verb, not on the code)
    assert links == {"verbs": C.LINK_CODES, "links": [[1, 2, 1], [2, 10, 2], [2, 20, 4], [2, 20, 4], [10, 2, 3]]}
    assert not os.path.exists(os.path.join(out, "THEIA_E3", "stats.json"))


def test_card_and_link_paths_do_not_mkdir(tmp_path):
    p = C.card_path(tmp_path / "c", "THEIA_E3", "2018-04-03")
    q = C.link_path(tmp_path / "c", "THEIA_E3", "2018-04-03")
    assert p == os.path.join(str(tmp_path), "c", "THEIA_E3", "2018-04-03.json.gz")
    assert q == os.path.join(str(tmp_path), "c", "THEIA_E3", "links", "2018-04-03.json.gz")
    assert not (tmp_path / "c").exists()


def test_build_refuses_read_only_output(settings):
    if not settings.paths.pidsmaker_artifacts:
        pytest.skip("no paths.pidsmaker_artifacts configured")
    from habitus.config import ConfigError
    with pytest.raises(ConfigError, match="read-only"):
        C.build("ATLASV2_EDR", os.path.join(settings.paths.pidsmaker_artifacts, "reflex", "cards"),
                days=["2022-07-19"], force=True)
    with pytest.raises(ValueError, match="not card days"):
        C.build("ATLASV2_EDR", os.path.join(settings.paths.runs, "cards_x"), days=["2022-07-21"])


# --------------------------------------------------------------------------- #
# Provenance: every vendored definition equals the original source (AST, docstrings dropped)
# --------------------------------------------------------------------------- #
def _strip_docs(node):
    for n in ast.walk(node):
        body = getattr(n, "body", None)
        if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef, ast.Module)) and body and \
                isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and \
                isinstance(body[0].value.value, str):
            n.body = body[1:] or [ast.Pass()]
    return node


def _defs(path):
    with open(path, encoding="utf-8") as f:
        tree = _strip_docs(ast.parse(f.read()))
    out = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            out[n.name] = n
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = n.value
    return out


def _members(cls):
    return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}


VENDORED = {
    "cards": ("pidsmaker/reflex/cards.py", C.__file__,
              ("TOP", "LINK_VERBS", "LINK_CODES", "CARD_VERSION", "_VERB_RULES", "verb", "_WIN_SYSTEM", "_WIN_APP",
               "_WIN_HOME", "_POSIX_SYSTEM", "_POSIX_APP", "_POSIX_HOME", "_POSIX_CONFIG", "dir_class", "_HEX",
               "_MIXED", "_DIGITS", "abstract_text", "ext_of", "bin_count", "_top", "_Node", "DayCards",
               "signature", "_LABEL", "render")),
    "identity": ("pidsmaker/reflex/identity.py", I.__file__,
                 ("_DEVICE", "_GUID", "_LONG_NUM", "MAX_CMD", "_clean_path", "normalize_cmd", "ProcessIdentity",
                  "process_identity")),
    "profiles": ("pidsmaker/defense/profiles.py", P.__file__,
                 ("normalize_path", "POSIX_STAGING", "WINDOWS_STAGING", "HostProfile", "DARPA_TC", "OPTC",
                  "CARBON_BLACK_EDR", "profile_for", "in_staging_dir")),
    "resolver": ("pidsmaker/defense/semantics.py", R.__file__,
                 ("EP_LOOPBACK", "EP_PRIVATE", "EP_ENCLAVE", "EP_EXTERNAL", "EP_UNKNOWN", "ENCLAVE_PREFIX_LEN",
                  "UNKNOWN_FILE_KEY", "_NULLISH", "_clean", "basename", "dirname", "Entity", "classify_endpoint")),
}


@pytest.mark.provenance
@pytest.mark.parametrize("mod", list(VENDORED))
def test_vendored_definitions_equal_original(settings, mod):
    root = asset(settings.paths.pidsmaker_src, "paths.pidsmaker_src")
    rel, mine, names = VENDORED[mod]
    orig, new = _defs(asset(os.path.join(root, rel), rel)), _defs(mine)
    for name in names:
        assert ast.dump(orig[name]) == ast.dump(new[name]), f"{mod}.{name} differs from {rel}"


@pytest.mark.provenance
def test_vendored_resolver_methods_and_profile_table(settings):
    root = asset(settings.paths.pidsmaker_src, "paths.pidsmaker_src")
    orig = _defs(asset(os.path.join(root, "pidsmaker/defense/semantics.py"), "semantics.py"))
    new = _defs(R.__file__)
    om, nm = _members(orig["SemanticResolver"]), _members(new["SemanticResolver"])
    for name in ("_pick_host_ip", "resolve", "_resolve_uncached", "_from_label"):
        assert ast.dump(om[name]) == ast.dump(nm[name]), name
    # the DB pass is the original's up to the cache write
    ob, nb = om["_build_from_db"].body, nm["_build_from_db"].body
    k = next(i for i, s in enumerate(ob) if isinstance(s, ast.Assign) and ast.unparse(s).startswith("self.available"))
    assert [ast.dump(s) for s in ob[1:k + 1]] == [ast.dump(s) for s in nb[1:k + 1]]
    # profile table: the in-scope rows of the original
    orig_table = _defs(asset(os.path.join(root, "pidsmaker/defense/profiles.py"), "profiles.py"))["_BY_DATASET"]
    orig_rows = {k.value: v.id for k, v in zip(orig_table.keys, orig_table.values)}
    from habitus.hosts import DATASETS
    for ds in DATASETS:
        want = orig_rows.get(ds.lower(), "DARPA_TC")
        assert P.profile_for(ds).name == getattr(P, want).name, ds


@pytest.mark.provenance
def test_kcs1_fake_cards_equal_original(settings):
    """tests/fakes.make_kcs1_days == the original test_kcs1.make_host days / links (its _card /
    make_host source run with the original signature())."""
    import hashlib
    import json

    import numpy as np
    root = asset(settings.paths.pidsmaker_src, "paths.pidsmaker_src")
    with open(asset(os.path.join(root, "tests", "test_kcs1.py"), "test_kcs1.py"), encoding="utf-8") as f:
        tk = ast.parse(f.read())
    with open(os.path.join(root, "pidsmaker", "reflex", "cards.py"), encoding="utf-8") as f:
        cd = ast.parse(f.read())
    pick = [n for n in tk.body if (isinstance(n, ast.FunctionDef) and n.name in ("_card", "make_host")) or
            (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "HOSTILE" for t in n.targets))]
    pick += [n for n in cd.body if isinstance(n, ast.FunctionDef) and n.name == "signature"]
    captured = {}

    def fake_host(days, links, ds="KCS1T"):
        captured.update(days=days, links=links, ds=ds)
        return None
    ns = {"np": np, "hashlib": hashlib, "json": json, "FakeHost": fake_host}
    exec(compile(ast.Module(body=pick, type_ignores=[]), "original_test_kcs1", "exec"), ns)
    ns["make_host"]()
    from fakes import HOSTILE, make_kcs1_days, make_kcs1_host
    days, links = make_kcs1_days()
    assert HOSTILE == ns["HOSTILE"]
    assert json.dumps(days, sort_keys=False) == json.dumps(captured["days"], sort_keys=False)
    assert json.dumps(links) == json.dumps(captured["links"])
    h = make_kcs1_host()
    assert (h.ds, h.os, h.clean, h.test, h.art) == ("KCS1T", "linux", ["c1", "c2"], ["t1", "t2"], "/nonexistent/")


# --------------------------------------------------------------------------- #
# Assets: rebuild one real day from the PIDSMaker graphs + resolver cache (read-only inputs)
# --------------------------------------------------------------------------- #
def test_rebuild_atlas_day_equals_release(settings):
    art = settings.paths.pidsmaker_artifacts
    if not art:
        pytest.skip("paths.pidsmaker_artifacts not configured (server-only check)")
    ds, day = "ATLASV2_EDR", "2022-07-19"
    refs = [d for d in (settings.paths.cards, os.path.join(art, "reflex", "cards"))
            if os.path.exists(C.card_path(d, ds, day))]
    if not refs:
        missing_asset(f"no shipped card file for {ds} {day}")
    asset(os.path.join(art, "defense", "semantics", f"{ds}.pkl"), "resolver cache")
    from habitus.data import upstream
    try:
        gdir = upstream.graphs_dir(ds, settings=settings)
    except Exception as e:  # noqa: BLE001
        missing_asset(f"PIDSMaker graphs / upstream config unavailable: {e}")
    out = os.path.join(settings.paths.runs, "rebuild_cards")
    assert C.build(ds, out, days=[day], force=True, graphs_dir=gdir, settings=settings) == [day]
    for ref in refs:
        assert load_json(C.card_path(out, ds, day)) == load_json(C.card_path(ref, ds, day)), ref
        assert load_json(C.link_path(out, ds, day)) == load_json(C.link_path(ref, ds, day)), ref


# --------------------------------------------------------------------------- #
# Upstream wiring (no upstream import: the subprocess is captured)
# --------------------------------------------------------------------------- #
def test_build_graphs_script_db_name_split():
    import importlib.util
    from habitus.data import upstream as U
    spec = importlib.util.spec_from_file_location("build_graphs_script", U.BUILD_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                              # imports nothing from pidsmaker at load time
    assert mod._split_db_name(["--db-name", "optc_h501", "orthrus", "optc_h501", "--x", "1"]) == \
        ("optc_h501", ["orthrus", "optc_h501", "--x", "1"])
    assert mod._split_db_name(["orthrus", "CADETS_E3"]) == (None, ["orthrus", "CADETS_E3"])


def test_graphs_subprocess_command(monkeypatch, settings):
    from habitus.config import ConfigError
    from habitus.data import upstream as U
    seen = {}

    def fake_run(cmd, env=None, check=None):
        seen.update(cmd=cmd, env=env)
        return 0
    monkeypatch.setattr(U.subprocess, "run", fake_run)
    U.graphs_subprocess("optc_h501", "/opt/env/bin/python",
                        {"host": "db", "user": "u", "password": None, "port": 5433}, settings=settings)
    cmd = seen["cmd"]
    assert cmd[:4] == ["/opt/env/bin/python", U.BUILD_SCRIPT, "--db-name", "optc_h501"]
    assert cmd[4:6] == ["orthrus", "optc_h501"] and cmd[6] == "--artifact_dir"
    assert cmd[7].startswith(os.path.join(settings.paths.runs, "pidsmaker"))
    assert cmd[8:] == ["--database_host", "db", "--database_user", "u", "--database_port", "5433"]
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == "" and seen["env"]["WANDB_MODE"] == "disabled"
    assert U.DB_PASSWORD_ENV not in seen["env"] or seen["env"][U.DB_PASSWORD_ENV] == os.environ.get(U.DB_PASSWORD_ENV)
    U.graphs_subprocess("optc_h501", "/py", {"host": "db", "user": "u", "password": "s3cret", "port": 5433},
                        settings=settings)                # the password: child environment only, never argv
    assert "s3cret" not in seen["cmd"] and "--database_password" not in seen["cmd"]
    assert seen["env"][U.DB_PASSWORD_ENV] == "s3cret"
    with pytest.raises(ConfigError, match="never as a command-line argument"):
        U.graphs_subprocess("CADETS_E3", "/py", ["--database_password", "x"], settings=settings)
    assert U.db_args({"host": "h", "password": "p"}) == ["--database_host", "h", "--database_password", "p"]
    assert U.db_args({"host": "h", "password": "p"}, password=False) == ["--database_host", "h"]
    U.graphs_subprocess("CADETS_E3", "/py", ["--database_host", "h"], settings=settings)
    assert seen["cmd"][-2:] == ["--database_host", "h"] and seen["cmd"][3] == "cadets_e3"
    if settings.paths.pidsmaker_artifacts:
        with pytest.raises(ConfigError, match="read-only"):
            U.graphs_subprocess("CADETS_E3", "/py", artifacts=settings.paths.pidsmaker_artifacts, settings=settings)
    from habitus.hosts import ForbiddenDataset
    with pytest.raises(ForbiddenDataset):
        U.graphs_subprocess("NOT_A_HOST", "/py", settings=settings)
    assert U.db_params("optc_h501", host="h", port="5433") == {"database": "optc_h501", "host": "h",
                                                              "user": "postgres", "password": "postgres", "port": 5433}
