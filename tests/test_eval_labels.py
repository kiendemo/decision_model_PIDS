"""Ground-truth reading (habitus.eval.labels): allow-list first, absolute root, config order, a missing
file raises, frozen sizes checked; the real GT files only when present (asset test)."""

import os
from pathlib import Path

import numpy as np
import pytest

from habitus.eval import labels as L
from habitus.hosts import ForbiddenDataset, spec


@pytest.fixture(autouse=True)
def _fresh():
    L.clear_cache()
    yield
    L.clear_cache()


def _write_gt(root, ds, ids_per_file, skip=()):
    for i, rel in enumerate(spec(ds).gt_files):
        if rel in skip:
            continue
        p = Path(root) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(f"x,{t},{n}\n" for t, n in enumerate(ids_per_file[i])) + "\n", encoding="utf-8")


def test_allow_list_runs_first(tmp_path):
    with pytest.raises(ForbiddenDataset):
        L.gt_nodes("NOT_A_HOST", root=tmp_path)
    with pytest.raises(ForbiddenDataset):
        L.gt_files("NOT_A_HOST", root=tmp_path)


def test_paths_are_absolute_and_in_config_order(tmp_path):
    files = L.gt_files("ATLASV2_EDR", root=tmp_path)
    assert all(p.is_absolute() for p in files)
    assert [p.relative_to(tmp_path.resolve()).as_posix() for p in files] == list(spec("ATLASV2_EDR").gt_files)


def test_campaigns_keep_config_order_and_last_column(tmp_path):
    ds = "ATLASV2_EDR"
    n = len(spec(ds).gt_files)
    ids = [[1000 * (n - i) + j for j in range(i + 1)] for i in range(n)]          # file i holds i + 1 ids
    _write_gt(tmp_path, ds, ids)
    gt, camp = L.gt_nodes(ds, root=tmp_path)
    assert list(camp) == [os.path.basename(r) for r in spec(ds).gt_files]
    assert [sorted(v) for v in camp.values()] == [sorted(x) for x in ids]
    assert gt == set().union(*map(set, ids))
    camp["atlasv2_edr_s1.csv"].add(-1)                       # the returned sets are copies
    assert -1 not in L.gt_nodes(ds, root=tmp_path)[1]["atlasv2_edr_s1.csv"]


def test_missing_file_raises(tmp_path):
    ds = "CADETS_E3"
    rels = spec(ds).gt_files
    _write_gt(tmp_path, ds, [[1], [2], [3]], skip=(rels[1],))
    with pytest.raises(FileNotFoundError):
        L.gt_nodes(ds, root=tmp_path)


class _Tiny:
    ds = "TINY"
    universe = np.array([10, 11, 12, 13], dtype=np.int64)

    def upos(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        pos = np.searchsorted(self.universe, ids)
        ok = pos < len(self.universe)
        ok[ok] = self.universe[pos[ok]] == ids[ok]
        return np.where(ok, pos, -1)

    def gt(self):
        return {11, 13}, 3

    def campaigns(self):
        return {"a.csv": {11}, "b.csv": {13, 99}}


def test_host_labels_and_attack_groups_of_a_test_double():
    h = _Tiny()
    y, n_total = L.host_labels(h)
    assert y.tolist() == [False, True, False, True] and n_total == 3
    assert L.host_labels(h)[0] is y                          # cached per host object
    assert L.attack_groups(h) == {"a.csv": {11}, "b.csv": {13, 99}}

    class One(_Tiny):
        def campaigns(self):
            return {"all": {11, 13}}
    assert L.attack_groups(One()) is None                    # one group: no per-attack split


def test_host_labels_check_the_frozen_sizes(tmp_path, monkeypatch):
    """A real host's labels are checked against configs/datasets.yaml (test-day nodes, GT sizes)."""
    ds = "ATLASV2_EDR"
    _write_gt(tmp_path, ds, [[11]] + [[13]] * (len(spec(ds).gt_files) - 1))
    monkeypatch.setattr(L, "gt_root", lambda settings=None: tmp_path.resolve())

    class Fake(_Tiny):
        gt = None                                            # no test-double hook: the GT files are read
    Fake.ds = ds
    with pytest.raises(ValueError, match="datasets.yaml"):
        L.host_labels(Fake())


# --------------------------------------------------------------------------- #
# Asset test: the shipped GT files of the 5 hosts
# --------------------------------------------------------------------------- #
def _gt_root_or_skip():
    try:
        root = L.gt_root()
    except FileNotFoundError:
        root = None
    if root is None or not root.is_dir():
        if os.environ.get("HABITUS_REQUIRE_ASSETS", "") not in ("", "0"):
            pytest.fail(f"release GT missing: {root}")
        pytest.skip(f"release GT missing: {root}")
    return root


@pytest.mark.parametrize("ds", ["CADETS_E3", "THEIA_E3", "CLEARSCOPE_E3", "optc_h501", "ATLASV2_EDR"])
def test_shipped_gt_sizes(ds):
    _gt_root_or_skip()
    gt, camp = L.gt_nodes(ds)
    assert len(gt) == spec(ds).gt_total
    assert list(camp) == [os.path.basename(r) for r in spec(ds).gt_files]
