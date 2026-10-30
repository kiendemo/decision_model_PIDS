"""habitus.cli: the command tree parses, the label guard is installed first for every non-exempt
group (and not for eval / reproduce / gates), reproduce / gates spawn guarded children, forbidden
GPUs are refused, and the key collector enumerates questions without answering them."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from habitus import cli

REPO = Path(__file__).resolve().parents[1]
SRC = str(REPO / "src")


def parse(*argv):
    return cli.build_parser().parse_args(list(argv))


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("argv", [
    ["data", "graphs", "--ds", "CADETS_E3"],
    ["data", "resolver", "--ds", "CADETS_E3", "--db-host", "localhost"],
    ["data", "cards", "--ds", "THEIA_E3", "--days", "2018-04-03,2018-04-04"],
    ["data", "manifest", "--ds", "optc_h501"],
    ["data", "index", "--ds", "ATLASV2_EDR"],
    ["data", "verify", "--ds", "ATLASV2_EDR"],
    ["state", "build", "--ds", "CADETS_E3", "--level", "E2", "--k", "2", "--day", "2018-04-06"],
    ["questions", "render", "--ds", "CLEARSCOPE_E3", "--level", "L4", "--k", "2", "--day", "2018-04-11"],
    ["questions", "keys", "--ds", "CLEARSCOPE_E3", "--from-inventory", "--answers", "unpruned"],
    ["dm", "info", "--dm", "jev-qwen3.5-4b"],
    ["dm", "smoke", "--dm", "laya-typed-zs", "--gpu", "1", "--n", "8"],
    ["dm", "smoke", "--dm", "jev-qwen3.5-4b", "--gpu", "0", "--full-day"],
    ["dm", "info", "--dm", "HB", "--fold", "all"],
    ["dm", "info", "--dm", "laya-typed-ens3", "--ds", "THEIA_E3"],
    ["data", "cards", "--ds", "THEIA_E3", "--pidsmaker-artifacts", "art"],
    ["data", "manifest", "--ds", "optc_h501", "--pidsmaker-artifacts", "art"],
    ["data", "graphs", "--ds", "CADETS_E3", "--db-password-file", "pw"],
    ["decide", "habit", "--ds", "THEIA_E3", "--dm", "laya-typed-zs", "--fold", "all"],
    ["gates", "--only", "G11"],
    ["decide", "routed", "--ds", "CADETS_E3,THEIA_E3", "--from-inventory"],
    ["decide", "routed", "--ds", "CADETS_E3", "--level", "E2,L4", "--ks", "all,2", "--no-nulls", "--gpu", "0"],
    ["decide", "hb0", "--ds", "THEIA_E3", "--kplans", "2,allc-pre"],
    ["decide", "habit", "--ds", "THEIA_E3", "--arm", "HB-Z", "--fold", "loho-THEIA_E3"],
    ["train", "items", "--fold", "all"],
    ["train", "jev", "--fold", "loho-THEIA_E3", "--gpu", "0", "--n", "100", "--variant", "head"],
    ["train", "laya", "--fold", "loho-THEIA_E3", "--gpu", "1", "--n", "100", "--layout", "mq"],
    ["eval", "confusion", "--ds", "CADETS_E3", "--field", "1-qa"],
    ["eval", "tables", "--source", "cards"],
    ["eval", "operating", "--what", "budget,boot"],
    ["eval", "baselines"],
    ["eval", "habit", "--ds", "CADETS_E3", "--levels", "HB0"],
    ["reproduce", "--level", "replay", "--jobs", "3"],
    ["gates", "--only", "G1,G5", "--answers", "unpruned"],
])
def test_every_command_parses(argv):
    a = parse(*argv)
    assert a.group == argv[0] and callable(a.func)


def test_defaults_cache_only_cpu():
    a = parse("decide", "routed", "--ds", "CADETS_E3", "--from-inventory")
    assert a.cache_only is True and a.gpu is None and a.dm == "jev-qwen3.5-4b" and a.nulls is True
    assert cli.device_of(a) == "cpu" and not cli.forwards_model(a)
    a = parse("decide", "routed", "--ds", "CADETS_E3", "--gpu", "1")
    assert cli.device_of(a) == "cuda:1" and cli.forwards_model(a)
    assert cli.forwards_model(parse("dm", "smoke", "--dm", "jev-qwen3.5-4b", "--gpu", "0"))
    assert not cli.forwards_model(parse("train", "items", "--fold", "all"))
    assert not cli.forwards_model(parse("questions", "keys", "--ds", "CADETS_E3"))
    assert parse("reproduce").level == "tables"
    with pytest.raises(SystemExit):
        parse("decide", "routed", "--ds", "CADETS_E3", "--cache-only", "--gpu", "1")   # exclusive
    with pytest.raises(SystemExit):
        parse("reproduce", "--level", "everything")


def test_argument_helpers():
    assert cli.parse_ks("all,2,0") == [None, 2, 0]
    with pytest.raises(SystemExit):
        cli.parse_ks("1")
    assert cli.parse_kplans("2,allc-pre,k2") == ["k2", "allc-pre"]
    with pytest.raises(SystemExit):
        cli.parse_kplans("all")                    # the oracle on CADETS_E3: say allc-pre
    assert cli.parse_levels("E2,L4") == ["E2", "L4"]
    with pytest.raises(SystemExit):
        cli.parse_levels("L5")
    assert cli.ds_list("CADETS_E3, ATLASV2_EDR") == ["CADETS_E3", "ATLASV2_EDR"]
    with pytest.raises(Exception):
        cli.ds_list("NOT_A_HOST")


# --------------------------------------------------------------------------- #
# the guard, per command (child processes: a hook cannot be removed)
# --------------------------------------------------------------------------- #
PROBE = r"""
import json, os, sys
import habitus.guard as G
from habitus import cli
seen = {}
orig = G.install_guard
def wrapped(command, **kw):
    seen["installed_for"] = command
    seen["modules_before"] = sorted(m for m in sys.modules if m.startswith("habitus"))
    return orig(command, **kw)
G.install_guard = wrapped
def probe(a, st):
    out = sys.argv[1]
    with open(out, "w") as fh:
        json.dump({**seen, "active": G.guard_active(), "roots": G.write_roots(),
                   "cvd": os.environ.get("CUDA_VISIBLE_DEVICES"), "tmpdir": os.environ.get("TMPDIR"),
                   "triton": os.environ.get("TRITON_CACHE_DIR")}, fh)
for name in [n for n in dir(cli) if n.startswith("cmd_")]:
    setattr(cli, name, probe)
sys.exit(cli.main(sys.argv[2:]))
"""


def _probe(tmp_path, argv, env_extra=None, out=None):
    out = out or tmp_path / "probe.json"
    env = dict(os.environ, PYTHONPATH=SRC)
    env.pop(cli.WRITE_ROOTS_ENV, None)
    env.pop(cli.OPEN_LOG_ENV, None)
    env.update(env_extra or {})
    r = subprocess.run([sys.executable, "-c", PROBE, str(out), *argv], env=env, cwd=str(REPO),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(out.read_text())


@pytest.mark.parametrize("argv,group", [
    (["state", "build", "--ds", "CADETS_E3", "--k", "2", "--day", "2018-04-06"], "state"),
    (["questions", "keys", "--ds", "CADETS_E3", "--from-inventory"], "questions"),
    (["decide", "routed", "--ds", "CADETS_E3", "--from-inventory"], "decide"),
    (["data", "index", "--ds", "CADETS_E3"], "data"),
    (["dm", "info", "--dm", "jev-qwen3.5-4b"], "dm"),
    (["train", "items", "--fold", "all"], "train"),
])
def test_guard_installed_first_for_guarded_groups(tmp_path, argv, group):
    got = _probe(tmp_path, argv)
    assert got["installed_for"] == group and got["active"] == group
    early = [m for m in got["modules_before"] if m not in ("habitus", "habitus.cli", "habitus.guard")]
    assert not early, f"habitus modules imported before the guard: {early}"
    assert got["cvd"] == ""                          # a cache-only command hides CUDA


@pytest.mark.parametrize("argv", [["eval", "tables"], ["reproduce"], ["gates", "--only", "G8"]])
def test_exempt_groups_are_not_guarded(tmp_path, argv):
    got = _probe(tmp_path, argv)
    assert "installed_for" not in got and got["active"] is None


def test_write_roots_reach_the_guard(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    got = _probe(tmp_path, ["decide", "routed", "--ds", "CADETS_E3", "--from-inventory"],
                 {cli.WRITE_ROOTS_ENV: str(root), "TRITON_CACHE_DIR": str(tmp_path / "elsewhere")},
                 out=root / "probe.json")
    assert got["roots"] and any(r.rstrip("/").endswith("/runs") for r in got["roots"])
    # temp files and the Triton JIT cache (a cold cache writes on the first CUDA forward) land inside
    assert got["tmpdir"] == str(root / "tmp") and got["triton"] == str(root / "triton")
    # and the audit refuses a write outside the roots
    with pytest.raises(AssertionError, match="WriteAuditError"):
        _probe(tmp_path, ["decide", "routed", "--ds", "CADETS_E3", "--from-inventory"],
               {cli.WRITE_ROOTS_ENV: str(root)}, out=tmp_path / "outside.json")


def test_open_log_records_opened_files(tmp_path):
    log = tmp_path / "open.log"
    target = tmp_path / "some_input.txt"
    target.write_text("x")
    code = PROBE.replace("def probe(a, st):\n", f"def probe(a, st):\n    open({str(target)!r}).read()\n")
    env = dict(os.environ, PYTHONPATH=SRC, **{cli.OPEN_LOG_ENV: str(log)})
    r = subprocess.run([sys.executable, "-c", code, str(tmp_path / "p.json"), "dm", "info", "--dm", "x"],
                       env=env, cwd=str(REPO), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert str(target) in log.read_text().splitlines()


def test_reproduce_spawns_guarded_cache_only_children(tmp_path):
    from habitus.repro import reproduce as RP
    specs = RP.replay_children(tmp_path / "root", tmp_path, hosts_=("ATLASV2_EDR",))
    (name, argv, env, log), = specs
    assert argv[:3] == [sys.executable, "-m", "habitus.cli"]
    assert argv[3:5] == ["decide", "routed"] and "--from-inventory" in argv and "--cache-only" in argv
    assert "--gpu" not in argv
    assert env[cli.WRITE_ROOTS_ENV] and env[cli.OPEN_LOG_ENV].endswith("open_ATLASV2_EDR.log")
    assert env["CUDA_VISIBLE_DEVICES"] == "" and SRC in env["PYTHONPATH"].split(os.pathsep)
    assert "decide" not in cli.EXEMPT                    # so the child installs the label guard


# --------------------------------------------------------------------------- #
# GPU refusal
# --------------------------------------------------------------------------- #
def _settings(forbidden):
    from habitus import config
    return config.load_settings(overrides={"forbidden_gpus": list(forbidden)}, local=False, env={})


@pytest.mark.parametrize("cvd,gpu,ok", [(None, 2, False), ("0,2", 1, False), ("GPU-1234abcd", 0, False),
                                        ("1", 1, False), (None, 0, True), ("2,1", 1, True)])
def test_forbidden_gpu_refused(monkeypatch, cvd, gpu, ok):
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    if cvd is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cvd)
    a = parse("dm", "smoke", "--dm", "jev-qwen3.5-4b", "--gpu", str(gpu))
    if ok:
        assert cli.check_gpu(a, _settings([2])) == f"cuda:{gpu}"
    else:
        with pytest.raises(SystemExit, match="refused"):
            cli.check_gpu(a, _settings([2]))


def test_cpu_needs_no_gpu_resolution(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    a = parse("decide", "routed", "--ds", "CADETS_E3", "--from-inventory")
    assert cli.check_gpu(a, _settings([0, 1, 2])) == "cpu"


# --------------------------------------------------------------------------- #
# key collection without answers
# --------------------------------------------------------------------------- #
class _KeysOnly:
    tag = "fake"

    def keys(self, qs):
        return {q.qkey: tuple(f"{q.qkey}#{f.name}" for f in q.forms) for q in qs}

    def refused(self, qs):
        return {}

    def spec(self):
        return {"prefix_share": True, "batch_size": 32, "grouping": "direct"}


def _q(qkey, n=3):
    from habitus.dm.base import Form, TypedQuestion
    opts = tuple(f"o{i}" for i in range(n))
    fwd = Form("fwd", "sys", "{}", "Q?", opts, tuple(range(n)))
    rev = Form("rev", "sys", "{}", "Q?", opts[::-1], tuple(range(n))[::-1])
    return TypedQuestion(qkey, "qv", "score", (fwd, rev), qkey.split("|")[0] + "|sig")


def test_key_collector_answers_nothing_but_records_every_key():
    col = cli.KeyCollector(_KeysOnly())
    dec = col.decide([_q("s1|qv"), _q("s2|qv", 4)])
    assert col.keys == {"s1|qv#fwd", "s1|qv#rev", "s2|qv#fwd", "s2|qv#rev"}
    d = dec["s2|qv"]
    assert d.forms == ("fwd", "rev") and d.cov_min == 1.0 and d.letter_probs["fwd"] == ((0.25,) * 4,)
    assert [x[0] for x in col.per_question] == ["s1|qv", "s2|qv"]
    assert col.spec()["grouping"] == "direct" and col.refused([]) == {}


def test_answer_keys_reader(tmp_path):
    import gzip
    p = tmp_path / "a.jsonl.gz"
    k1, k2 = "a" * 40, "b" * 40
    with gzip.open(p, "wt") as fh:
        fh.write(json.dumps({"k": k1, "z": [1, 2, 3, 4], "lse": 5, "n": 4, "cov": 0.9}) + "\n")
        fh.write("not json\n")
        fh.write(json.dumps({"z": [0], "k": k2}) + "\n")
    assert cli.answer_keys(p) == {k1, k2}


# --------------------------------------------------------------------------- #
# train / dm / decide option handling
# --------------------------------------------------------------------------- #
def test_train_jev_run_dir_follows_the_backbone(monkeypatch):
    """The default run directory of `train jev` is the one the arm's configs/dm file reads."""
    import yaml
    assert cli.jev_run_dir("lora", "Qwen/Qwen3.5-4B") == "HB"
    assert cli.jev_run_dir("head", "Qwen/Qwen3.5-4B") == "HB.head"
    assert cli.jev_run_dir("lora", "Qwen/Qwen3.5-0.8B") == "HB.08b"
    with pytest.raises(SystemExit):
        cli.jev_run_dir("head", "Qwen/Qwen3.5-0.8B")
    arms = cli.habit_arms()
    for (variant, backbone), d in cli.JEV_RUN_DIRS.items():
        cfg = yaml.safe_load((REPO / "configs" / "dm" / f"{arms[d]}.yaml").read_text(encoding="utf-8"))
        assert (cfg.get("adapter") or cfg.get("run")) == f"runs/adapters/<fold>/{d}"
        assert cfg["backbone"] == backbone and cfg["layout"] == ("head" if variant == "head" else "letter")
    from habitus.train import jev as TJ
    seen = {}
    monkeypatch.setattr(cli, "_training_set", lambda a, st: [])
    monkeypatch.setattr(TJ, "train_jev", lambda items, inner, out, **kw: seen.update(out=out, **kw) or {})
    st = cli._settings()
    a = parse("train", "jev", "--fold", "loho-THEIA_E3", "--gpu", "0", "--n", "10", "--backbone", "Qwen/Qwen3.5-0.8B")
    cli.cmd_train_jev(a, st)
    assert seen["out"] == os.path.join(st.paths.runs, "adapters", "loho-THEIA_E3", "HB.08b")
    lora = yaml.safe_load((REPO / "configs" / "train" / "jev-lora.yaml").read_text(encoding="utf-8"))
    assert seen["recipe"] == lora["recipe"] and "variant" not in seen["recipe"]


def test_train_laya_seeds_default_to_the_yaml(monkeypatch):
    from habitus.train import laya as TL
    seen = {}
    monkeypatch.setattr(cli, "_training_set", lambda a, st: [])
    monkeypatch.setattr(TL, "train_laya", lambda items, inner, out, **kw: seen.update(kw) or {})
    st = cli._settings()
    cli.cmd_train_laya(parse("train", "laya", "--fold", "all", "--gpu", "0", "--n", "5", "--layout", "mq"), st)
    assert seen["seeds"] == (0,) and seen["recipe"]["layout"] == "mq"
    cli.cmd_train_laya(parse("train", "laya", "--fold", "all", "--gpu", "0", "--n", "5"), st)
    assert seen["seeds"] == (0, 1, 2)
    cli.cmd_train_laya(parse("train", "laya", "--fold", "all", "--gpu", "0", "--n", "5", "--seeds", "4"), st)
    assert seen["seeds"] == (4,)


def test_registries_come_from_habitus_dm_and_hosts():
    from habitus import dm, hosts
    assert cli.habit_arms() == {a.name: a.config for a in dm.ARMS.values() if a.questions == "habit"}
    assert cli.folds() == tuple(hosts.FOLDS)
    assert cli.habit_arm("HB-L.zs") == "HB-L.zs" == cli.habit_arm("laya-typed-zs")
    assert cli.habit_arm("jev-qwen3.5-0.8b-lora") == "HB.08b"
    with pytest.raises(SystemExit):
        cli.habit_arm("jev-qwen3.5-4b")                   # the routed main config is no habit arm
    assert cli.dm_config("HB") == "jev-qwen3.5-4b-lora" and cli.dm_config("jev-qwen3.5-4b") == "jev-qwen3.5-4b"
    assert parse("decide", "habit", "--ds", "THEIA_E3", "--dm", "laya-typed-zs", "--fold", "all").arm == "laya-typed-zs"
    with pytest.raises(SystemExit):                       # --family is gone: render lists routed questions only
        parse("questions", "render", "--ds", "CLEARSCOPE_E3", "--k", "2", "--day", "2018-04-11", "--family", "habit")


def test_dm_info_takes_a_fold_for_a_trained_arm(capsys):
    """`dm info` of a trained arm resolves its run directory with --fold / --ds (no ValueError)."""
    st = cli._settings()
    cli.cmd_dm_info(parse("dm", "info", "--dm", "HB", "--fold", "loho-THEIA_E3"), st)
    info = json.loads(capsys.readouterr().out)
    assert info["config_name"] == "jev-qwen3.5-4b-lora" and info["fold"] == "loho-THEIA_E3"
    assert "give fold= or ds=" not in info.get("error", "")
    cli.cmd_dm_info(parse("dm", "info", "--dm", "laya-typed-ens3", "--ds", "THEIA_E3"), st)
    info = json.loads(capsys.readouterr().out)
    assert "give fold= or ds=" not in info.get("error", "")
    with pytest.raises(SystemExit):
        cli.cmd_dm_info(parse("dm", "info", "--dm", "HB", "--fold", "nope"), st)


def test_config_error_is_one_line(monkeypatch):
    """main() turns a ConfigError into a one-line exit message (eval: exempt, no guard in this process)."""
    from habitus.config import ConfigError

    def boom(a, st):
        raise ConfigError("upstream PIDSMaker is not importable")
    monkeypatch.setattr(cli, "cmd_eval_baselines", boom)
    with pytest.raises(SystemExit, match="^configuration error: upstream PIDSMaker is not importable$"):
        cli.main(["eval", "baselines"])


def test_db_password_never_reaches_a_child_argv(monkeypatch, tmp_path):
    pw = tmp_path / "pw"
    pw.write_text("s3cret\n", encoding="utf-8")
    a = parse("data", "graphs", "--ds", "CADETS_E3", "--db-password-file", str(pw))
    monkeypatch.delenv(cli.DB_PASSWORD_ENV, raising=False)
    assert cli._db_password(a) == "s3cret"
    monkeypatch.setenv(cli.DB_PASSWORD_ENV, "fromenv")
    assert cli._db_password(parse("data", "graphs", "--ds", "CADETS_E3")) == "fromenv"
    assert cli._db_password(a) == "s3cret"                  # the file wins over the environment


def test_smoke_stats_and_the_repinned_thresholds():
    """G7: the subset smoke is a liveness check at p99 |d letter prob| <= 0.08 (bf16 batch-makeup noise
    ~0.06), the full-day replay is held to 1e-3."""
    from habitus.dm.base import Decision
    assert cli.JEV_SMOKE_P99 == 0.08 and cli.JEV_FULL_DAY_P99 == 1e-3

    def dec(p):
        probs = {"f": (tuple(p),)}
        return Decision(qkey="q", layout="letter", forms=("f",), letter_probs=probs, canon_probs=probs,
                        cov={"f": 1.0}, cov_min=1.0)
    same = cli.smoke_stats([(dec((0.7, 0.3)), dec((0.7, 0.3)))] * 10, "jev")
    assert same["p99_abs_delta"] == 0.0 and same["frac_exact"] == 1.0 and same["argmax_agree"] == 1.0
    assert cli.smoke_ok(same, cli.JEV_FULL_DAY_P99)
    noisy = cli.smoke_stats([(dec((0.64, 0.36)), dec((0.7, 0.3)))] * 10, "jev")
    assert abs(noisy["p99_abs_delta"] - 0.06) < 1e-9 and noisy["frac_exact"] == 0.0
    assert cli.smoke_ok(noisy, cli.JEV_SMOKE_P99) and not cli.smoke_ok(noisy, cli.JEV_FULL_DAY_P99)
    flipped = cli.smoke_stats([(dec((0.4, 0.6)), dec((0.7, 0.3)))], "jev")
    assert flipped["argmax_agree"] == 0.0 and not cli.smoke_ok(flipped, cli.JEV_SMOKE_P99)
    laya = cli.smoke_stats([(dec((0.5, 0.5)), None)], "laya")
    assert laya["p99_abs_delta"] is None and cli.smoke_ok(laya, cli.JEV_SMOKE_P99)
    bad = cli.smoke_stats([(dec((0.5, 0.6)), None)], "laya")
    assert not cli.smoke_ok(bad, cli.JEV_SMOKE_P99)
