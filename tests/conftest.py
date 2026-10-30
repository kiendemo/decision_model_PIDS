"""Shared pytest setup for every builder's tests.

  - HABITUS_RUNS points at a fresh temporary directory for the whole session (removed at exit), so
    no test writes into the repo's runs/ or anywhere else by default.
  - Thread caps (OMP / MKL / OpenBLAS / torch) default to 8 on this shared server.
  - HABITUS_REQUIRE_ASSETS=1: a test that would skip for a missing local asset (release bundle,
    tokenizer, PIDSMaker artifacts) FAILS instead (G6).
  - Tests marked `local_assets` need files the release bundle does not hold (the Laya typed-decisions
    weights under paths.laya_hub, the role-table catalogs under paths.catalogs). They skip without
    them, and fail only under HABITUS_REQUIRE_LOCAL_ASSETS=1 (G6 sets it on the build server, where
    paths.pidsmaker_src is configured). docs/REPRODUCE.md says how to fetch them.
  - HABITUS_PROVENANCE=1: run tests marked `provenance` (they read the original source trees through
    paths.pidsmaker_src / paths.upstream_src / paths.laya_src; never written).
The `tests/` directory is on sys.path (rootdir-relative imports: `from fakes import ...`).
"""

import os
import shutil
import sys
import tempfile

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TESTS_DIR)
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")

if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)
_SRC = os.path.join(REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

for _v in THREAD_VARS:
    os.environ.setdefault(_v, "8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True          # never write __pycache__ next to a read-only source file

_RUNS = tempfile.mkdtemp(prefix="habitus-test-runs-")
os.environ["HABITUS_RUNS"] = _RUNS


def require_assets():
    return os.environ.get("HABITUS_REQUIRE_ASSETS", "") not in ("", "0")


def provenance_enabled():
    return os.environ.get("HABITUS_PROVENANCE", "") == "1"


def missing_asset(msg):
    """Skip for a missing local asset, or fail under HABITUS_REQUIRE_ASSETS."""
    if require_assets():
        pytest.fail(f"HABITUS_REQUIRE_ASSETS: {msg}")
    pytest.skip(msg)


def require_local_assets():
    return os.environ.get("HABITUS_REQUIRE_LOCAL_ASSETS", "") not in ("", "0")


def missing_local_asset(msg):
    """Skip for a missing asset outside the release bundle (model weights, catalogs); fail only under
    HABITUS_REQUIRE_LOCAL_ASSETS (not under HABITUS_REQUIRE_ASSETS alone)."""
    if require_local_assets():
        pytest.fail(f"HABITUS_REQUIRE_LOCAL_ASSETS: {msg}")
    pytest.skip(msg)


def asset(path, what="asset"):
    """`path` when it exists; otherwise skip (fail under HABITUS_REQUIRE_ASSETS)."""
    if not path or not os.path.exists(path):
        missing_asset(f"missing {what}: {path}")
    return path


def pytest_configure(config):
    config.addinivalue_line("markers", "provenance: compares with the original sources (HABITUS_PROVENANCE=1)")
    config.addinivalue_line("markers", "slow: reads the full card files of every host")
    config.addinivalue_line("markers", "local_assets: needs model weights / catalogs outside the release bundle "
                            "(fails only under HABITUS_REQUIRE_LOCAL_ASSETS=1)")
    try:
        import torch
        torch.set_num_threads(min(8, torch.get_num_threads()))
    except Exception:  # noqa: BLE001 - torch is optional for most tests
        pass
    try:
        from habitus import config as _config
        _config.reset_settings()
    except ImportError:
        pass


def pytest_collection_modifyitems(config, items):
    if provenance_enabled():
        return
    skip = pytest.mark.skip(reason="provenance test: set HABITUS_PROVENANCE=1 (needs the original sources)")
    for item in items:
        if "provenance" in item.keywords:
            item.add_marker(skip)


def pytest_unconfigure(config):
    shutil.rmtree(_RUNS, ignore_errors=True)


@pytest.fixture(scope="session")
def runs_dir():
    """The session's HABITUS_RUNS directory."""
    return _RUNS


@pytest.fixture
def settings():
    from habitus.config import settings as _settings
    return _settings()


def real_cards_dir():
    """A cards dir holding all five hosts: the release (settings.paths.cards), else the configured
    PIDSMaker artifacts' reflex/cards (read only). Skips (fails under REQUIRE_ASSETS) without one."""
    from habitus import hosts
    from habitus.config import settings as _settings
    s = _settings()
    cands = [s.paths.cards]
    if s.paths.pidsmaker_artifacts:
        cands.append(os.path.join(s.paths.pidsmaker_artifacts, "reflex", "cards"))
    for c in cands:
        if all(os.path.isdir(os.path.join(c, ds)) for ds in hosts.DATASETS):
            return c
    missing_asset(f"no cards dir with all five hosts among {cands}")


@pytest.fixture(scope="session")
def cards_dir_real():
    return real_cards_dir()


@pytest.fixture(scope="session")
def nodeindex_real(cards_dir_real):
    """The node index of all five hosts (45 files), built once per session under HABITUS_RUNS."""
    from habitus import hosts
    from habitus.data import nodeindex
    out = os.path.join(_RUNS, "nodeindex_real")
    files = {ds: nodeindex.build(ds, cards_dir_real, out) for ds in hosts.DATASETS}
    return out, files
