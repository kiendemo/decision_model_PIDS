"""Settings: the only place where HABITUS resolves file-system paths.

Merge order (later wins):
  1. configs/default.yaml                       repo-relative defaults
  2. the main config, when one is given         `habitus --config PATH`, else $HABITUS_CONFIG
  3. configs/local.yaml, when present           gitignored; the only place for server paths
  4. `overrides` (tests)                         a dict with the same layout as the YAML
  5. environment variables                       ENV_PATHS below, HABITUS_FORBIDDEN_GPUS

forbidden_gpus is the UNION of every layer (default.yaml, the main config, local.yaml, overrides and
$HABITUS_FORBIDDEN_GPUS): a later layer or the environment can add a forbidden GPU, never remove one.

Path values may reference other path keys (`${data}/cards`) or environment variables
(`${CYRAG_ROOT}`). A relative path from a YAML file is relative to `home`; a relative `home` is
relative to the repo root; a relative path from the environment is relative to the current
directory. Computing a path never creates it.

No resolved path may lie on the former data disk or on another user's data disk (/D[A]TA, /D[A]TA2,
...; ConfigError). The check is a string check on the absolute path: the file system is never touched
for it.

Read-only inputs: the server inputs of READ_ONLY_PATHS, plus models, laya_hub and catalogs whenever
they resolve outside paths.home (a shared model store or a catalog checkout in another tree is
verified and read, never updated). config.check_writable refuses any output under them.
"""

import os
import re
from dataclasses import dataclass, fields

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CONFIG_DIR = os.path.join(REPO_ROOT, "configs")
DEFAULT_CONFIG = os.path.join(CONFIG_DIR, "default.yaml")
LOCAL_CONFIG = os.path.join(CONFIG_DIR, "local.yaml")
CONFIG_ENV = "HABITUS_CONFIG"

# the former data disk (I/O errors) and the other users' data disks (/D[A]TA2, /D[A]TA3, ...): character
# class so that repository scans do not match this line
FORMER_DATA_DISK = re.compile(r"^/D[A]TA[0-9]*(/|$)")

CORE_PATHS = ("home", "data", "cards", "nodeindex", "answers", "scores", "precedent", "ground_truth",
              "tokenizers", "baselines", "runs", "models", "laya_hub", "catalogs")
# server-only inputs (null by default; real values only in configs/local.yaml). Everything here is
# read, never written.
OPTIONAL_PATHS = ("pidsmaker_src", "pidsmaker_artifacts", "upstream_artifacts", "upstream_gt",
                  "research_results", "paper_derive", "laya_src", "pidsmaker_python", "upstream_src")
READ_ONLY_PATHS = ("pidsmaker_src", "pidsmaker_artifacts", "upstream_artifacts", "upstream_gt",
                   "research_results", "paper_derive", "laya_src", "upstream_src")
# model and catalog stores: read-only whenever they resolve outside paths.home
READ_ONLY_OUTSIDE_HOME = ("models", "laya_hub", "catalogs")
# the release bundle's read-only parts: no output (score file, cache overlay) may be written under them
RELEASE_PATHS = ("data", "scores", "cards", "answers", "precedent", "nodeindex", "ground_truth")

ENV_PATHS = {
    "HABITUS_HOME": "home",
    "HABITUS_DATA": "data",
    "HABITUS_RUNS": "runs",
    "HABITUS_GT": "ground_truth",
    "HF_HOME": "models",
    "HABITUS_LAYA_HUB": "laya_hub",
    "HABITUS_CATALOGS": "catalogs",
    "PIDSMAKER_ARTIFACTS": "pidsmaker_artifacts",
    "PIDSMAKER_UPSTREAM_ARTIFACTS": "upstream_artifacts",
}
GPU_ENV = "HABITUS_FORBIDDEN_GPUS"
TOP_KEYS = ("paths", "device", "threads", "forbidden_gpus", "offline")
_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ConfigError(ValueError):
    """An invalid configuration: unknown key, unresolvable reference, or a forbidden path."""


@dataclass(frozen=True)
class Paths:
    home: str
    data: str
    cards: str
    nodeindex: str
    answers: str
    scores: str
    precedent: str
    ground_truth: str
    tokenizers: str
    baselines: str
    runs: str
    models: str
    laya_hub: str
    catalogs: str
    pidsmaker_src: str = None
    pidsmaker_artifacts: str = None
    upstream_artifacts: str = None
    upstream_gt: str = None
    research_results: str = None
    paper_derive: str = None
    laya_src: str = None
    pidsmaker_python: str = None
    upstream_src: str = None

    def as_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def read_only_roots(self):
        """The configured server inputs that HABITUS must never write under: READ_ONLY_PATHS, plus
        models / laya_hub / catalogs when they lie outside paths.home."""
        roots = [getattr(self, k) for k in READ_ONLY_PATHS if getattr(self, k)]
        roots += [getattr(self, k) for k in READ_ONLY_OUTSIDE_HOME
                  if getattr(self, k) and not is_under(getattr(self, k), self.home)]
        return tuple(roots)


@dataclass(frozen=True)
class Settings:
    paths: Paths
    device: str = "cpu"
    threads: int = 8
    forbidden_gpus: tuple = ()
    offline: bool = True
    sources: tuple = ()          # the YAML files merged, in order

    def as_dict(self):
        return {"paths": self.paths.as_dict(), "device": self.device, "threads": self.threads,
                "forbidden_gpus": list(self.forbidden_gpus), "offline": self.offline,
                "sources": list(self.sources)}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _read_yaml(path):
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return raw


def _merge(base, layer, where):
    for k, v in (layer or {}).items():
        if k not in TOP_KEYS:
            raise ConfigError(f"{where}: unknown key {k!r} (allowed: {', '.join(TOP_KEYS)})")
        if k == "forbidden_gpus":           # a union over the layers: never cleared by a later one
            base["forbidden_gpus"] = tuple(sorted(set(base.get("forbidden_gpus", ())) | set(_parse_gpus(v, where))))
            continue
        if k == "paths":
            if not isinstance(v, dict):
                raise ConfigError(f"{where}: 'paths' must be a mapping")
            for pk, pv in v.items():
                if pk not in CORE_PATHS + OPTIONAL_PATHS:
                    raise ConfigError(f"{where}: unknown path key {pk!r}")
                base["paths"][pk] = (pv, where)
        else:
            base[k] = v


def _parse_gpus(v, where):
    if v is None or v == "":
        return ()
    if isinstance(v, str):
        v = [x for x in re.split(r"[,\s]+", v.strip()) if x]
    if isinstance(v, int):
        v = [v]
    try:
        return tuple(sorted({int(x) for x in v}))
    except (TypeError, ValueError):
        raise ConfigError(f"{where}: forbidden_gpus must be a list of integers, got {v!r}") from None


def _interpolate(key, raw, env, stack=()):
    """Expand ${path key} and ${ENV} references in raw[key][0]."""
    if key in stack:
        raise ConfigError(f"path reference cycle: {' -> '.join(stack + (key,))}")
    value, where = raw[key]
    if value is None:
        return None, where
    if not isinstance(value, str):
        raise ConfigError(f"{where}: paths.{key} must be a string or null, got {value!r}")

    def sub(m):
        name = m.group(1)
        if name in raw:
            v, _w = _interpolate(name, raw, env, stack + (key,))
            if v is None:
                raise ConfigError(f"{where}: paths.{key} references ${{{name}}}, which is null")
            return v
        if name in env:
            return env[name]
        raise ConfigError(f"{where}: paths.{key} references ${{{name}}}: neither a path key nor an "
                          f"environment variable (configs/local.example.yaml placeholders must be replaced)")

    return _REF.sub(sub, value), where


def _absolute(key, value, where, home):
    value = os.path.expanduser(value)
    if os.path.isabs(value):
        return os.path.normpath(value)
    if where == "env":
        return os.path.abspath(value)
    if key == "home":
        return os.path.normpath(os.path.join(REPO_ROOT, value))
    return os.path.normpath(os.path.join(home, value))


def check_path(key, path):
    """ConfigError when `path` lies on the former data disk or another user's data disk (string check
    only)."""
    if path is not None and FORMER_DATA_DISK.match(os.path.abspath(path)):
        raise ConfigError(f"paths.{key} = {path} lies on the former data disk or another user's data disk, "
                          f"which must not be used")
    return path


def load_settings(path=None, overrides=None, *, local=True, env=None):
    """Settings from default.yaml [+ `path`] [+ local.yaml] [+ overrides] [+ env].

    path       a main config merged over default.yaml (None: $HABITUS_CONFIG when set)
    overrides  a dict with the YAML layout ({"paths": {...}, "device": ...}), merged after local.yaml
    local      merge configs/local.yaml when it exists
    env        the environment to read (default os.environ; {} isolates a test)
    """
    env = dict(os.environ if env is None else env)
    if path is None:
        path = env.get(CONFIG_ENV) or None
    merged = {"paths": {k: (None, "default") for k in CORE_PATHS + OPTIONAL_PATHS}}
    sources = []
    for p, use in ((DEFAULT_CONFIG, True), (path, path is not None),
                   (LOCAL_CONFIG, local and os.path.exists(LOCAL_CONFIG))):
        if not use or p is None:
            continue
        p = os.path.abspath(os.fspath(p))
        if sources and p == sources[0]:
            continue
        _merge(merged, _read_yaml(p), p)
        sources.append(p)
    if overrides:
        _merge(merged, overrides, "overrides")
    for var, key in ENV_PATHS.items():
        if env.get(var):                    # cwd-relative, made absolute before ${...} expansion
            merged["paths"][key] = (os.path.abspath(os.path.expanduser(env[var])), "env")
    if env.get(GPU_ENV) is not None:        # adds to the configured list (never removes)
        merged["forbidden_gpus"] = tuple(sorted(set(merged.get("forbidden_gpus", ()))
                                                | set(_parse_gpus(env[GPU_ENV], GPU_ENV))))

    raw = merged["paths"]
    missing = [k for k in CORE_PATHS if raw[k][0] in (None, "")]
    if missing:
        raise ConfigError(f"paths {missing} have no value (configs/default.yaml is incomplete)")
    expanded = {k: _interpolate(k, raw, env) for k in raw}
    home_v, home_w = expanded["home"]
    home = _absolute("home", home_v, home_w, None)
    resolved = {}
    for k, (v, w) in expanded.items():
        if v is None or v == "":
            resolved[k] = None
            continue
        resolved[k] = home if k == "home" else _absolute(k, v, w, home)
        check_path(k, resolved[k])

    device = str(merged.get("device", "cpu"))
    try:
        threads = int(merged.get("threads", 8))
    except (TypeError, ValueError):
        raise ConfigError(f"threads must be an integer, got {merged.get('threads')!r}") from None
    if threads < 1:
        raise ConfigError(f"threads must be >= 1, got {threads}")
    offline = merged.get("offline", True)
    if not isinstance(offline, bool):
        raise ConfigError(f"offline must be true or false, got {offline!r}")
    return Settings(paths=Paths(**resolved), device=device, threads=threads,
                    forbidden_gpus=_parse_gpus(merged.get("forbidden_gpus"), "forbidden_gpus"),
                    offline=offline, sources=tuple(sources))


# --------------------------------------------------------------------------- #
# The process-wide settings
# --------------------------------------------------------------------------- #
_CACHE = {}


def settings():
    """The process-wide Settings (loaded once: default.yaml, $HABITUS_CONFIG, local.yaml, env)."""
    s = _CACHE.get("s")
    if s is None:
        s = _CACHE["s"] = load_settings()
    return s


def reset_settings():
    """Forget the cached Settings (tests); the next settings() call reloads."""
    _CACHE.clear()


def require(key, s=None):
    """The configured optional path `key`, or ConfigError naming the local.yaml key to set."""
    s = s or settings()
    v = getattr(s.paths, key)
    if not v:
        raise ConfigError(f"paths.{key} is not configured: set it in configs/local.yaml "
                          f"(see configs/local.example.yaml)")
    return v


def is_under(path, root):
    p, r = os.path.abspath(path), os.path.abspath(root)
    return p == r or p.startswith(r.rstrip(os.sep) + os.sep)


def check_writable(path, s=None):
    """ConfigError when `path` lies under a configured read-only server input (the source tree)."""
    s = s or settings()
    for root in s.paths.read_only_roots():
        if is_under(path, root):
            raise ConfigError(f"{path} lies under the read-only input {root}: HABITUS never writes there")
    check_path("output", path)
    return os.path.abspath(path)
