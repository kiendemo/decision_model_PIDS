"""The ONLY module that imports upstream PIDSMaker, and only inside functions.

    cfg = orthrus_cfg("CADETS_E3")              # upstream orthrus task config for the dataset
    graphs_dir("CADETS_E3")                      # <artifacts>/transformation/<ds>/transformation/<h>/nx/
    graphs_subprocess("CADETS_E3")               # construction + transformation in the upstream env

In-process use needs the upstream `pidsmaker.config` package (yacs, PyYAML; it imports
pidsmaker.utils as well): either installed in this environment, or a source tree at
settings.paths.upstream_src (else paths.pidsmaker_src) that is put on sys.path. Bytecode writing is
switched off before importing from a source tree, so nothing is written into it.

The task config is built with `artifact_dir` under runs/ (upstream set_task_paths creates log
directories under it), and cfg.dataset.database is set to HostSpec.db_name programmatically:
upstream 2289cd9 names the optc_h501 database 'optc_501' while the dump restore creates
'optc_h501' (docs/upstream.patch). The task hash does not depend on the database name.

Graph construction imports all of upstream pidsmaker.main (torch, wandb, every task), so it never
runs in-process: graphs_subprocess runs scripts/build_graphs.py under settings.paths.pidsmaker_python.
The database password reaches that child in its environment ($HABITUS_DB_PASSWORD, readable by the
owner only), never on its command line, where other users of a shared server could read it in `ps`.
"""

import os
import subprocess
import sys

from habitus import hosts
from habitus.config import REPO_ROOT, ConfigError, check_writable, settings as _settings
from habitus.log import log

MODEL = "orthrus"
SCRATCH_ART = "pidsmaker_art"          # runs/<SCRATCH_ART>: upstream's task-path log directories
BUILT_ART = "pidsmaker"                # runs/<BUILT_ART>: graphs built by `habitus data graphs`
BUILD_SCRIPT = os.path.join(REPO_ROOT, "scripts", "build_graphs.py")
DB_DEFAULTS = {"host": "postgres", "user": "postgres", "password": "postgres", "port": 5432}
DB_PASSWORD_ENV = "HABITUS_DB_PASSWORD"   # scripts/build_graphs.py reads it (never an argv element)


def upstream_tree(s=None):
    """The source tree that provides `pidsmaker` (paths.upstream_src, else paths.pidsmaker_src), or None."""
    s = s or _settings()
    return s.paths.upstream_src or s.paths.pidsmaker_src


def _pidsmaker_config(s=None):
    """Import upstream pidsmaker.config (lazy). Raises ConfigError when it is not importable."""
    try:
        import pidsmaker.config as pc  # noqa: F401  (already installed or already on sys.path)
        return pc
    except ImportError:
        pass
    tree = upstream_tree(s)
    if not tree or not os.path.isdir(os.path.join(tree, "pidsmaker", "config")):
        raise ConfigError("upstream PIDSMaker is not importable: install it in this environment or set "
                          "paths.upstream_src in configs/local.yaml (docs/DATA.md)")
    sys.dont_write_bytecode = True          # never write __pycache__ into a read-only source tree
    if tree not in sys.path:
        sys.path.insert(0, tree)
    import pidsmaker.config as pc
    return pc


def db_params(ds, host=None, user=None, password=None, port=None):
    """Postgres connection parameters for the dataset's restored database."""
    spec = hosts.spec(ds)
    return {"database": spec.db_name,
            "host": DB_DEFAULTS["host"] if host is None else host,
            "user": user or DB_DEFAULTS["user"],
            "password": DB_DEFAULTS["password"] if password is None else password,
            "port": int(port or DB_DEFAULTS["port"])}


def db_args(db=None, *, password=True):
    """Upstream CLI arguments for a db_params dict (empty: upstream defaults). password=False leaves
    the password out (a child process gets it through $HABITUS_DB_PASSWORD instead)."""
    if not db:
        return []
    out = []
    for key in ("host", "user", "password", "port"):
        if key == "password" and not password:
            continue
        if db.get(key) is not None:
            out += [f"--database_{key}", str(db[key])]
    return out


_cli_db_args = db_args


def scratch_artifact_dir(s=None):
    s = s or _settings()
    return os.path.join(s.paths.runs, SCRATCH_ART)


def orthrus_cfg(ds, *, artifact_dir=None, db=None, settings=None):
    """The upstream orthrus task config of `ds`, exactly as the PIDSMaker CLI builds it, with
    cfg.dataset.database (and database_all_file) set to HostSpec.db_name."""
    s = settings or _settings()
    spec = hosts.spec(ds)
    art = artifact_dir or scratch_artifact_dir(s)
    check_writable(art, s)                  # set_task_paths creates directories under it
    pc = _pidsmaker_config(s)
    args = pc.get_runtime_required_args(args=[MODEL, ds, "--artifact_dir", art.rstrip("/") + "/"] + db_args(db))
    cfg = pc.get_yml_cfg(args)
    cfg.dataset.database = spec.db_name
    cfg.dataset.database_all_file = spec.db_name
    return cfg


def graphs_dir(ds, *, artifacts=None, cfg=None, settings=None):
    """The dataset's transformed-graph directory. `artifacts` (default: the first of runs/pidsmaker
    and paths.pidsmaker_artifacts that holds the task-hash directory) is the PIDSMaker artifact root
    that holds the graphs; the hash path comes from the upstream config."""
    s = settings or _settings()
    cfg = cfg or orthrus_cfg(ds, settings=s)
    rel = os.path.relpath(cfg.transformation._graphs_dir, cfg._artifact_dir)
    if artifacts is not None:
        return os.path.join(artifacts, rel)
    cands = [os.path.join(s.paths.runs, BUILT_ART)]
    if s.paths.pidsmaker_artifacts:
        cands.append(s.paths.pidsmaker_artifacts)
    for root in cands:
        p = os.path.join(root, rel)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(f"{ds}: no transformed graphs at {rel} under {cands}: run `habitus data graphs`")


def graphs_subprocess(ds, python=None, db_args=(), *, artifacts=None, settings=None, check=True):
    """Run upstream construction + transformation for `ds` in its own interpreter:
        <python> scripts/build_graphs.py --db-name <db> orthrus <ds> --artifact_dir <artifacts> <db_args>
    `python` defaults to paths.pidsmaker_python; `artifacts` to runs/pidsmaker (never a read-only
    input). `db_args`: upstream CLI arguments (["--database_host", H, ...]) or a db_params dict.
    Returns the CompletedProcess."""
    s = settings or _settings()
    spec = hosts.spec(ds)
    python = python or s.paths.pidsmaker_python
    if not python:
        raise ConfigError("paths.pidsmaker_python is not set: the interpreter of the upstream PIDSMaker "
                          "environment (docs/DATA.md)")
    art = check_writable(artifacts or os.path.join(s.paths.runs, BUILT_ART), s)
    os.makedirs(art, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", WANDB_MODE="disabled", PYTHONDONTWRITEBYTECODE="1")
    if isinstance(db_args, dict):
        extra = _cli_db_args(db_args, password=False)
        if db_args.get("password") is not None:
            env[DB_PASSWORD_ENV] = str(db_args["password"])
    else:
        extra = [str(a) for a in (db_args or ())]
        if "--database_password" in extra:
            raise ConfigError("graphs_subprocess: pass the database password in a db_params dict (it goes to the "
                              f"child's environment as ${DB_PASSWORD_ENV}), never as a command-line argument")
    cmd = [python, BUILD_SCRIPT, "--db-name", spec.db_name, MODEL, ds, "--artifact_dir",
           art.rstrip("/") + "/"] + extra
    tree = upstream_tree(s)
    if tree:
        env["PYTHONPATH"] = tree + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    log(f"{ds}: {' '.join(cmd[:6])} ... (artifacts {art})", "data.upstream")
    return subprocess.run(cmd, env=env, check=check)
