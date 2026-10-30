"""Ground-truth labels: the ONLY label reader of habitus.

    from habitus.eval import labels
    gt, campaigns = labels.gt_nodes("ATLASV2_EDR")       # set of node ids, {GT file name: set}
    y, n_total = labels.host_labels(host)                # y over host.universe, |GT| incl. cardless nodes
    groups = labels.attack_groups(host)                  # {GT file name: ids} or None (one group)

GT = the last CSV column of every file of HostSpec.gt_files (CONFIG ORDER), resolved under
settings.paths.ground_truth (an absolute path; the original resolved a cwd-relative Ground_Truth/orthrus).
A missing GT file raises (the original skipped it silently). Only the allow-listed datasets are readable:
hosts.assert_allowed runs before any path is built.

A ground-truth node without a test-day card is outside the universe; host_labels counts it in n_total so
that every metric treats it as a miss (FN).

Test doubles: a host object that provides gt() -> (ids inside the universe, n_total) or campaigns() ->
{group: ids} is used as is (the real habitus.data.host.Host has neither).
"""

from __future__ import annotations

import csv
import os
import weakref
from pathlib import Path

import numpy as np

_GT_CACHE: dict = {}
_Y = weakref.WeakKeyDictionary()


def _settings(settings=None):
    if settings is not None:
        return settings
    from habitus.config import settings as _s
    return _s()


def _spec(ds):
    from habitus.hosts import spec
    return spec(ds)


def gt_root(settings=None) -> Path:
    """Absolute directory of the GT CSVs (settings.paths.ground_truth)."""
    p = getattr(_settings(settings).paths, "ground_truth", None)
    if p is None:
        raise FileNotFoundError("settings.paths.ground_truth is not set")
    return Path(os.path.expanduser(str(p))).resolve()


def gt_files(ds, *, root=None, settings=None) -> list[Path]:
    """Absolute paths of the GT files of `ds`, in HostSpec.gt_files (config) order."""
    from habitus.hosts import assert_allowed
    assert_allowed(ds)
    base = Path(os.path.expanduser(str(root))).resolve() if root is not None else gt_root(settings)
    return [base / rel for rel in _spec(ds).gt_files]


def _read_gt(path: Path) -> frozenset:
    if not path.is_file():
        raise FileNotFoundError(f"ground-truth file missing: {path}")
    with open(path, newline="", encoding="utf-8") as fh:
        return frozenset(int(row[-1]) for row in csv.reader(fh) if row)


def gt_nodes(ds, *, root=None, settings=None):
    """(every GT node id of `ds`, {GT file basename: node ids}) with the campaigns in config order."""
    files = gt_files(ds, root=root, settings=settings)
    key = (ds, tuple(str(p) for p in files))
    camp = _GT_CACHE.get(key)
    if camp is None:
        camp = {}
        for p in files:
            camp[os.path.basename(str(p))] = _read_gt(p)
        _GT_CACHE[key] = camp
    out = {k: set(v) for k, v in camp.items()}
    return (set().union(*out.values()) if out else set()), out


def _host_gt(host, settings=None):
    """(GT ids inside host.universe, |GT| incl. the nodes without a test-day card)."""
    fn = getattr(host, "gt", None)
    if callable(fn):                                   # a test double only
        return fn()
    gt, _camp = gt_nodes(host.ds, settings=settings)
    ids = np.fromiter(gt, dtype=np.int64, count=len(gt))
    inside = ids[host.upos(ids) >= 0]
    return set(inside.tolist()), len(gt)


def _check_spec(host, y, n_total):
    """Frozen universe / GT sizes of an allow-listed host (HostSpec) must match what was read."""
    from habitus.hosts import DATASETS
    if getattr(host, "ds", None) not in DATASETS or callable(getattr(host, "gt", None)):
        return
    hs = _spec(host.ds)
    want = {"test_day_nodes": len(host.universe), "gt_total": int(n_total), "gt_in_universe": int(y.sum())}
    bad = {k: (getattr(hs, k), v) for k, v in want.items()
           if getattr(hs, k, None) is not None and int(getattr(hs, k)) != v}
    if bad:
        raise ValueError(f"{host.ds}: universe / GT sizes differ from configs/datasets.yaml (frozen, read): {bad}")


def host_labels(host, settings=None):
    """(y aligned with host.universe, |GT_total|), cached per host object."""
    try:
        hit = _Y.get(host)
    except TypeError:
        hit = None
    if hit is None:
        gt_in, n_total = _host_gt(host, settings)
        y = np.isin(host.universe, np.fromiter(gt_in, dtype=np.int64, count=len(gt_in)))
        _check_spec(host, y, n_total)
        hit = (y, n_total)
        try:
            _Y[host] = hit
        except TypeError:
            pass
    return hit


def attack_groups(host, settings=None):
    """{group: set of node ids} when the label source names more than one attack group (one group per GT
    file, config order), else None."""
    fn = getattr(host, "campaigns", None)
    if callable(fn):
        groups = fn()
    else:
        _all, groups = gt_nodes(host.ds, settings=settings)
    groups = {str(k): {int(x) for x in v} for k, v in (groups or {}).items()}
    return groups if len(groups) > 1 else None


def clear_cache():
    _GT_CACHE.clear()
    _Y.clear()
