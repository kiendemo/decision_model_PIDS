"""Decision models: the registry of backends and arms.

    from habitus.dm import get_dm, get_arm
    dm = get_dm("jev-qwen3.5-4b", ds="CADETS_E3")                 # MAIN, cache-only by default
    dm = get_arm("HB", ds="THEIA_E3", fold="loho-THEIA_E3")       # a trained habit arm

A name resolves to configs/dm/<name>.yaml. The backend is cfg['dm'] plus a suffix from
cfg['layout'] (letter -> 'jev', head -> 'jev-head', la -> 'laya', mq -> 'laya-mq'). The laya
backends are imported by module name only when asked for, so this package imports no torch and no
laya code.

Config rules: every key except the registry keys (dm, layout, questions, ...) is a constructor
argument; path values may hold '<fold>'; a relative path whose first part is 'runs' resolves under
settings.paths.runs, any other relative path under the repository root; cache_tag 'auto' is the
habit tag of the backbone + adapter for habit questions (engine tag otherwise). cache_only defaults
to True and device to settings.device.

Fold safety: an arm backed by trained runs reads each run's habitus.json and refuses a run trained
for another fold, and a fold that trains on the scored dev host (its own pre-deployment days).
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from habitus.dm.base import (LETTERS, CacheMiss, Decision, DecisionModel, Form, Refused,  # noqa: F401
                             TypedQuestion)

__all__ = ["get_dm", "get_arm", "ARMS", "Arm", "BACKENDS", "folds", "fold_for", "check_run_fold", "check_fold_scores",
           "load_config", "resolve", "Form", "TypedQuestion", "Decision", "DecisionModel", "CacheMiss", "Refused",
           "LETTERS"]

BACKEND_PATHS = {
    "jev": ("habitus.dm.jev", "JevDecisionModel"),
    "jev-head": ("habitus.dm.jev_head", "JevHeadDecisionModel"),
    "laya": ("habitus.dm.laya", "LayaDecisionModel"),
    "laya-mq": ("habitus.dm.laya_mq", "LayaMQDecisionModel"),
}
LAYOUT_SUFFIX = {"letter": "", "head": "-head", "la": "", "mq": "-mq"}
FAMILY_LAYOUTS = {"jev": ("letter", "head"), "laya": ("la", "mq")}
REGISTRY_KEYS = ("dm", "layout", "questions", "role", "description")
RENAMES = {"jev-head": {"run": "run_dir"}}
RUN_KEYS = ("adapter", "run", "run_dir", "runs")      # values that are trained run directories
FOLD = "<fold>"
RUN_CONFIG = "habitus.json"


class _Backends(Mapping):
    """{backend key: DecisionModel class}, each imported on first access."""

    def __getitem__(self, key):
        if key not in BACKEND_PATHS:
            raise KeyError(f"unknown decision-model backend {key!r}; known {sorted(BACKEND_PATHS)}")
        mod, name = BACKEND_PATHS[key]
        return getattr(importlib.import_module(mod), name)

    def __iter__(self):
        return iter(BACKEND_PATHS)

    def __len__(self):
        return len(BACKEND_PATHS)


BACKENDS = _Backends()


@dataclass(frozen=True)
class Arm:
    name: str
    config: str
    questions: str                 # 'routed' | 'habit'
    trained: bool


ARMS = {a.name: a for a in (
    Arm("MAIN", "jev-qwen3.5-4b", "routed", False),
    Arm("HB-Z", "jev-qwen3.5-4b-habit", "habit", False),
    Arm("HB", "jev-qwen3.5-4b-lora", "habit", True),
    Arm("HB.head", "jev-qwen3.5-4b-head", "habit", True),
    Arm("HB.08b", "jev-qwen3.5-0.8b-lora", "habit", True),
    Arm("HB-L.s0", "laya-typed-s0", "habit", True),
    Arm("HB-L", "laya-typed-ens3", "habit", True),
    Arm("HB-L.mq", "laya-typed-mq", "habit", True),
    Arm("HB-L.zs", "laya-typed-zs", "habit", False),
)}


def _settings(settings=None):
    if settings is not None:
        return settings
    from habitus.config import settings as _s
    return _s()


def repo_root(settings=None):
    """The repository root: settings.paths.home when it holds configs/dm, else the source checkout."""
    home = getattr(getattr(_settings(settings), "paths", None), "home", None)
    if home is not None and (Path(str(home)) / "configs" / "dm").is_dir():
        return Path(str(home))
    return Path(__file__).resolve().parents[3]


def config_path(name, settings=None):
    return repo_root(settings) / "configs" / "dm" / f"{name}.yaml"


def load_config(name, settings=None):
    import yaml
    p = config_path(name, settings)
    if not p.is_file():
        raise FileNotFoundError(f"no decision-model config {name!r} ({p})")
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{p}: not a mapping")
    return cfg


def backend_key(cfg):
    fam, layout = cfg.get("dm"), cfg.get("layout")
    if fam not in FAMILY_LAYOUTS or layout not in FAMILY_LAYOUTS[fam]:
        raise ValueError(f"config dm={fam!r} layout={layout!r}: expected one of {FAMILY_LAYOUTS}")
    return fam + LAYOUT_SUFFIX[layout]


# --------------------------------------------------------------------------- #
# Folds and run records
# --------------------------------------------------------------------------- #
def folds():
    """{fold: training hosts}: loho-<dev host> = the other dev hosts + the train-only host; all."""
    for mod in ("habitus.hosts", "habitus.questions.habit.constants"):
        try:
            f = getattr(importlib.import_module(mod), "FOLDS", None)
        except ImportError:
            continue
        if f:
            return {k: tuple(v) for k, v in f.items()}
    raise RuntimeError("no fold table (habitus.hosts.FOLDS / habitus.questions.habit.constants.FOLDS)")


def dev_hosts():
    return tuple(f[len("loho-"):] for f in folds() if f.startswith("loho-"))


def fold_for(ds):
    """The fold whose models score `ds`: loho-<ds> for a dev host, else 'all' (the train-only host's
    rows are host-seen)."""
    return f"loho-{ds}" if ds in dev_hosts() else "all"


def check_fold_scores(fold, ds):
    """Refuse a fold that trains on the scored dev host."""
    fs = folds()
    if fold not in fs:
        raise ValueError(f"fold {fold!r} not in {sorted(fs)}")
    if ds is not None and ds in dev_hosts() and ds in fs[fold]:
        raise ValueError(f"{ds}: fold {fold} trains on {ds} (its own pre-deployment days); a dev host is "
                         f"scored by loho-{ds} only")


def run_record(path):
    """habitus.json of a run directory (or of the parent of its adapter/ directory); {} for none."""
    d = os.path.abspath(str(path))
    for cand in (os.path.join(d, RUN_CONFIG), os.path.join(os.path.dirname(d), RUN_CONFIG)):
        if os.path.isfile(cand):
            with open(cand, encoding="utf-8") as f:
                return json.load(f)
    return {}


def check_run_fold(path, fold):
    """The run record of `path`, refused when missing or trained for another fold."""
    rec = run_record(path)
    if not rec:
        raise ValueError(f"{path}: no {RUN_CONFIG} run record, its fold cannot be checked")
    if rec.get("fold") != fold:
        raise ValueError(f"{path} was trained for fold {rec.get('fold')!r}, not {fold!r}")
    return rec


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
def _has_fold(v):
    if isinstance(v, str):
        return FOLD in v
    if isinstance(v, (list, tuple)):
        return any(_has_fold(x) for x in v)
    return False


def _resolve_path(v, fold, settings):
    if not isinstance(v, str):
        return v
    v = v.replace(FOLD, fold) if fold is not None else v
    if FOLD in v:
        raise ValueError(f"path {v!r} needs a fold")
    p = Path(v)
    if p.is_absolute() or "/" not in v:
        return v
    parts = p.parts
    if parts[0] == "runs":
        return str(Path(str(settings.paths.runs)).joinpath(*parts[1:]))
    return str(repo_root(settings) / p)


def _is_pathlike_key(k):
    return k in RUN_KEYS or k in ("ensemble",)


def resolve(cfg, *, ds=None, device=None, cache_only=None, settings=None, fold=None):
    """(backend key, constructor kwargs) of a config (module docstring)."""
    s = _settings(settings)
    cfg = dict(cfg)
    key = backend_key(cfg)
    questions = cfg.get("questions")
    for old, new in RENAMES.get(key, {}).items():
        if old in cfg:
            cfg[new] = cfg.pop(old)
    kw = {k: v for k, v in cfg.items() if k not in REGISTRY_KEYS}
    needs_fold = any(_has_fold(v) for k, v in kw.items() if _is_pathlike_key(k))
    if needs_fold and fold is None:
        if ds is None:
            raise ValueError(f"config {cfg} names trained runs: give fold= or ds=")
        fold = fold_for(ds)
    for k in list(kw):
        if _is_pathlike_key(k):
            v = kw[k]
            if isinstance(v, (list, tuple)):
                kw[k] = [_resolve_path(x, fold, s) for x in v]
            else:
                kw[k] = _resolve_path(v, fold, s)
    runs = []
    for k in RUN_KEYS:
        v = kw.get(k)
        if v:
            runs += list(v) if isinstance(v, (list, tuple)) else [v]
    runs = [r for r in runs if os.path.isabs(str(r)) or os.sep in str(r)]   # a Laya init name is no run
    if runs:
        if fold is None:
            fold = fold_for(ds) if ds is not None else None
        if fold is None:
            raise ValueError("a trained arm needs fold= or ds=")
        check_fold_scores(fold, ds)
        for r in runs:
            check_run_fold(r, fold)
    if kw.get("cache_tag") == "auto":
        kw["cache_tag"] = None
        if key == "jev" and questions == "habit":
            from habitus.dm.jev import DEFAULT_BACKBONE, habit_tag, resolve_adapter
            backbone = kw.get("backbone", DEFAULT_BACKBONE)
            ad = kw.get("adapter")
            kw["cache_tag"] = habit_tag(backbone, resolve_adapter(ad, backbone) if ad else None)
    kw["device"] = device if device is not None else kw.get("device", getattr(s, "device", None) or "cpu")
    kw["cache_only"] = bool(cache_only) if cache_only is not None else bool(kw.get("cache_only", True))
    kw["ds"] = ds
    kw["settings"] = s
    if fold is not None:
        kw["fold"] = fold
    return key, kw


def get_dm(name_or_cfg, *, ds=None, device=None, cache_only=None, settings=None, fold=None, **extra):
    """The DecisionModel of a config name (configs/dm/<name>.yaml) or a config mapping. `extra` are
    further constructor arguments (e.g. forward=, cache=)."""
    s = _settings(settings)
    cfg = load_config(name_or_cfg, s) if isinstance(name_or_cfg, str) else dict(name_or_cfg)
    key, kw = resolve(cfg, ds=ds, device=device, cache_only=cache_only, settings=s, fold=fold)
    kw.update(extra)
    cls = BACKENDS[key]
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    if not any(p.kind == p.VAR_KEYWORD for p in params.values()):
        unknown = sorted(k for k in kw if k not in params)
        hard = [k for k in unknown if k not in ("ds", "settings", "cache_only", "device", "dtype", "fold")]
        if hard:
            where = name_or_cfg if isinstance(name_or_cfg, str) else "mapping"
            raise ValueError(f"{cls.__name__} takes no {hard} (config {where})")
        kw = {k: v for k, v in kw.items() if k in params}
    return cls(**kw)


def get_arm(name, *, ds=None, fold=None, device=None, cache_only=None, settings=None, **extra):
    """The DecisionModel of a registered arm (ARMS)."""
    if name not in ARMS:
        raise ValueError(f"unknown arm {name!r}; known {sorted(ARMS)}")
    return get_dm(ARMS[name].config, ds=ds, device=device, cache_only=cache_only, settings=settings, fold=fold,
                  **extra)
