"""Run ONLY the upstream PIDSMaker graph stages (construction + transformation) for one dataset.

Runs in the upstream PIDSMaker environment (it imports all of pidsmaker.main: torch, wandb and every
task), never inside the habitus process. `habitus data graphs` starts it as a subprocess:

    <paths.pidsmaker_python> scripts/build_graphs.py --db-name optc_h501 orthrus optc_h501 \
        --artifact_dir runs/pidsmaker/ [--database_host H --database_user U ...]

--db-name sets cfg.dataset.database (and database_all_file) after the config is built: upstream
2289cd9 names the optc_h501 database 'optc_501', the dump restore creates 'optc_h501'. The task hash
does not depend on it. Everything after --db-name is passed to upstream get_runtime_required_args.
The database password comes from $HABITUS_DB_PASSWORD (never this script's command line, which other
users can read in `ps`) and is added to the upstream arguments inside this process.
Kept compatible with Python 3.9 (the upstream image).
"""

import os
import sys
import time

DB_PASSWORD_ENV = "HABITUS_DB_PASSWORD"


def _split_db_name(argv):
    if "--db-name" not in argv:
        return None, list(argv)
    i = argv.index("--db-name")
    if i + 1 >= len(argv):
        raise SystemExit("--db-name needs a value")
    return argv[i + 1], argv[:i] + argv[i + 2:]


def main(argv=None):
    db_name, rest = _split_db_name(sys.argv[1:] if argv is None else argv)
    password = os.environ.get(DB_PASSWORD_ENV)
    if password and "--database_password" not in rest:
        rest = rest + ["--database_password", password]

    from pidsmaker.config import get_runtime_required_args, get_yml_cfg
    from pidsmaker.config.pipeline import set_task_to_done, update_task_paths_to_restart
    from pidsmaker.main import get_task_to_module
    from pidsmaker.utils.utils import log

    args = get_runtime_required_args(args=rest)
    cfg = get_yml_cfg(args)
    if db_name:
        cfg.dataset.database = db_name
        cfg.dataset.database_all_file = db_name
    should_restart = update_task_paths_to_restart(cfg)
    modules = get_task_to_module(cfg)
    for task in ("construction", "transformation"):
        start = time.time()
        entry = modules[task]
        if not should_restart.get(task, True):
            log(f"[build_graphs] {task}: already done, skipping")
            continue
        log(f"[build_graphs] running {task} (database {cfg.dataset.database}) ...")
        entry["module"].main(cfg)
        set_task_to_done(entry["task_path"])
        log(f"[build_graphs] {task} done in {time.time() - start:.0f}s")
    log(f"[build_graphs] graphs at: {cfg.transformation._graphs_dir}")


if __name__ == "__main__":
    main()
