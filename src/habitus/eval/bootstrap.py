"""Paired, stratified ENTITY bootstrap of the oracle node MCC (CPU only).

    from habitus.eval import bootstrap
    b = bootstrap.boot("ATLASV2_EDR", root=scores_root, operating_dir=runs_operating)   # boot_<DS>.json content
    summary = bootstrap.merge({ds: b, ...})                                            # deltas, points
    bootstrap.write_csv(summary, "bootstrap_mcc.csv")

Systems per host (all on the HABITUS universe): the E2 composite / base / acq_all and E1 L2 / L4 at k2
and kall, 'HABITUS main' (the main configuration), form1 / form2 (vec_<DS>.npz written by
operating.forms) and 'best baseline' (runs/baselines/<ds>_best_baseline.npz written by
eval.baselines.export: the PIDSMaker run with the best own-node-set MCC at best_adp, aligned to the
universe with unscored nodes ranked last).

Resampling unit = entity (a kappa pool, Host.ukey_codes). Two strata, each resampled with replacement
keeping its size: P = the entities holding at least one positive node of the universe plus one singleton
per positive node WITHOUT a card; N = every other entity. One resample = an integer multiplicity per
entity; a node weighs its entity's multiplicity. Resample b is drawn from SeedSequence([seed, b]) alone,
so every system of a host sees the SAME draws and nothing depends on the system list or the chunking
(B = 2000, seeds 101..105 for CADETS_E3..ATLASV2_EDR). The oracle threshold is re-selected in every
resample; ties at the cut count positive (metrics.best_mcc). Optimal cuts lie at ends of tie blocks
holding a positive, so per system the ranking is cut into J segments ending at the J positive tie
blocks and stored as two sparse (J x entity) count matrices: positives per block (G) and all nodes per
segment (A); a chunk of resamples is then G @ M and A @ M plus cumsums.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HOSTS = ("CADETS_E3", "THEIA_E3", "CLEARSCOPE_E3", "optc_h501", "ATLASV2_EDR")
KN = {0: "k0", 2: "k2", None: "kall"}
N_BOOT = 2000
CHUNK = 25
SEED = {"CADETS_E3": 101, "THEIA_E3": 102, "CLEARSCOPE_E3": 103, "optc_h501": 104, "ATLASV2_EDR": 105}
# best PIDSMaker baseline per host at best_adp (own-node-set MCC); verified when a baseline vector is read
BEST_BASELINE_MCC = {"CADETS_E3": 0.404, "THEIA_E3": 0.272, "CLEARSCOPE_E3": 0.126, "optc_h501": 0.176,
                     "ATLASV2_EDR": 0.130}
PAIRS = [("HABITUS main", "best baseline"), ("form2", "HABITUS main"), ("form1", "HABITUS main")]
for _kn in ("k2", "kall"):
    PAIRS += [(f"E2 composite {_kn}", f"E2 base {_kn}"), (f"E2 composite {_kn}", f"E2 acq_all {_kn}"),
              (f"E2 composite {_kn}", f"E1 L4 {_kn}"), (f"E2 composite {_kn}", f"E1 L2 {_kn}"),
              (f"E2 base {_kn}", f"E1 L4 {_kn}")]
del _kn


def _log(msg):
    try:
        from habitus.log import log
        log(msg, "eval.bootstrap")
    except ImportError:
        print(f"[eval.bootstrap] {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# Strata / Ranked
# --------------------------------------------------------------------------- #
def codes_of(keys):
    """Entity codes as metrics.pool makes them: integer keys used as they are when dense, else np.unique codes."""
    keys = np.asarray(keys)
    dense = (np.issubdtype(keys.dtype, np.integer) and len(keys)
             and keys.min() >= 0 and keys.max() <= 4 * len(keys))
    return keys.astype(np.int64) if dense else np.unique(keys, return_inverse=True)[1].astype(np.int64)


class Strata:
    """Entity columns of the multiplicity matrix: [P entities in U | cardless singletons | N entities]."""

    def __init__(self, pos, keys, n_total=None):
        pos = np.asarray(pos, dtype=bool)
        codes = codes_of(keys)
        if codes.shape != pos.shape or pos.ndim != 1:
            raise ValueError(f"keys {codes.shape} and positives {pos.shape} must be equal 1-d arrays")
        n_in = int(pos.sum())
        n_total = n_in if n_total is None else int(n_total)
        if n_total < n_in:
            raise ValueError(f"n_total={n_total} < {n_in} positives in the universe")
        m = int(codes.max()) + 1 if len(codes) else 0
        size = np.bincount(codes, minlength=m)
        used = size > 0
        is_pos = np.zeros(m, dtype=bool)
        is_pos[codes[pos]] = True
        pe, be = np.flatnonzero(is_pos), np.flatnonzero(used & ~is_pos)
        self.n_in, self.n_total, self.n_miss = n_in, n_total, n_total - n_in
        self.n_pos_ent = len(pe)                       # P entities inside the universe
        self.n_p = len(pe) + self.n_miss               # P stratum incl. the cardless singletons
        self.n_b = len(be)
        self.n_cols = self.n_p + self.n_b
        col = np.full(m, -1, dtype=np.int64)
        col[pe] = np.arange(len(pe))
        col[be] = self.n_p + np.arange(len(be))
        self.pos, self.codes, self.used = pos, codes, used
        self.node_col = col[codes]
        self.pool_col = col[np.flatnonzero(used)]      # metrics.pool order: used codes ascending
        # denominators: positives per column (node level; a cardless singleton = 1) and positive pools per column
        self.node_pos = np.bincount(self.node_col[pos], minlength=self.n_cols).astype(np.float64)
        self.node_pos[self.n_pos_ent:self.n_p] = 1.0
        self.pool_pos = np.zeros(self.n_cols)
        self.pool_pos[:self.n_p] = 1.0
        self.sizes = {"pos_entities": int(len(pe)), "cardless": int(self.n_miss), "benign_entities": int(len(be)),
                      "nodes": int(len(pos)), "positives_in_u": n_in,
                      "nodes_in_pos_entities": int(size[pe].sum()) if len(pe) else 0,
                      "largest_pos_entity_nodes": int(size[pe].max()) if len(pe) else 0,
                      "largest_entity_nodes": int(size.max()) if m else 0}

    def ones(self):
        return np.ones((self.n_cols, 1))

    def draws(self, b0, b1, seed=0):
        """(n_cols, b1 - b0) multiplicities of resamples b0..b1-1; resample b uses SeedSequence([seed, b])."""
        M = np.empty((self.n_cols, b1 - b0))
        for j, b in enumerate(range(b0, b1)):
            rng = np.random.default_rng([int(seed), int(b)])
            if self.n_p:
                M[:self.n_p, j] = np.bincount(rng.integers(0, self.n_p, self.n_p), minlength=self.n_p)
            if self.n_b:
                M[self.n_p:, j] = np.bincount(rng.integers(0, self.n_b, self.n_b), minlength=self.n_b)
        return M


class Ranked:
    """One score vector's tie-block structure over the prefix that can hold positives."""

    def __init__(self, s, pos, col, n_cols):
        from scipy import sparse
        s, pos, col = np.asarray(s, dtype=np.float64), np.asarray(pos, dtype=bool), np.asarray(col)
        if s.shape != pos.shape or s.shape != col.shape or s.ndim != 1:
            raise ValueError("scores, positives and columns must be equal 1-d arrays")
        if np.isnan(s).any():
            raise ValueError("NaN score")
        self.J, self.prefix, self.nnz = 0, 0, 0
        self.G = self.A = None
        if not pos.any():
            return
        order = np.argsort(-s, kind="mergesort")      # metrics._blocks' order
        ss, ps = s[order], pos[order]
        end = np.r_[ss[1:] != ss[:-1], True]
        blk = np.cumsum(np.r_[0, end[:-1]])           # tie-block id per sorted position
        last = int(np.flatnonzero(end)[blk[np.flatnonzero(ps)[-1]]])   # end of the last positive block
        blk, ps = blk[:last + 1], ps[:last + 1]
        pb = np.unique(blk[ps])                       # positive blocks, ascending
        seg = np.searchsorted(pb, blk)                # segment j = (end of pb[j-1], end of pb[j]]
        c = col[order[:last + 1]]
        self.J, self.prefix = len(pb), last + 1
        self.A = sparse.csr_matrix((np.ones(last + 1), (seg, c)), shape=(self.J, n_cols))
        self.G = sparse.csr_matrix((np.ones(int(ps.sum())), (seg[ps], c[ps])), shape=(self.J, n_cols))
        self.nnz = int(self.A.nnz + self.G.nnz)

    def ap(self, M, total):
        """Weighted AP per column of M (n_cols x B); total (B,) = weighted positives incl. cardless."""
        if not self.J:
            return np.zeros(M.shape[1])
        g = np.asarray(self.G @ M)
        n = np.cumsum(np.asarray(self.A @ M), axis=0)
        tp = np.cumsum(g, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.where(g > 0, g * tp / n, 0.0)
        num = np.ascontiguousarray(t.T).sum(1)
        total = np.asarray(total, dtype=np.float64)
        return num / np.where(total > 0, total, np.nan)


def totals(st, M):
    """(node-level, entity-level) positive totals per resample: weighted positives incl. the cardless
    singletons. Only the P stratum holds positives; einsum, no BLAS."""
    p = st.n_p
    return (np.einsum("i,ij->j", st.node_pos[:p], M[:p]), np.einsum("i,ij->j", st.pool_pos[:p], M[:p]))


def _best_mcc_weighted(rk, st, negcol, M):
    """Oracle node MCC per resample (columns of M). Optimal cuts lie at ends of tie blocks holding a
    positive (Ranked's segments); ties at the cut count positive, as metrics.best_mcc."""
    if not rk.J:
        return np.zeros(M.shape[1])
    g = np.asarray(rk.G @ M)
    n = np.cumsum(np.asarray(rk.A @ M), axis=0)
    tp = np.cumsum(g, axis=0)
    P = totals(st, M)[0][None, :]
    Ng = np.einsum("i,ij->j", negcol, M)[None, :]
    fp = n - tp
    fn = P - tp
    tn = Ng - fp
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(den > 0, num / den, 0.0)
    return np.maximum(m.max(axis=0), 0.0)


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def universe_sha1(universe):
    return hashlib.sha1(np.ascontiguousarray(np.asarray(universe, dtype=np.int64)).tobytes()).hexdigest()


def baseline_npz_path(ds, settings=None, runs=None):
    """runs/baselines/<ds>_best_baseline.npz, else the release copy under settings.paths.baselines."""
    if runs is not None:
        return Path(runs) / f"{ds}_best_baseline.npz"
    from habitus.config import settings as _s
    st = settings or _s()
    p = Path(str(st.paths.runs)) / "baselines" / f"{ds}_best_baseline.npz"
    if p.exists():
        return p
    return Path(str(st.paths.baselines)) / f"{ds}_best_baseline.npz"


def read_baseline(path, host):
    """(universe-aligned scores, info) of a best-baseline npz; the universe must be the host's."""
    with np.load(path, allow_pickle=False) as z:
        s = np.asarray(z["s_u"], dtype=np.float64)
        info = {"own_mcc": float(z["own_mcc"]), "log": str(z["log"]), "system": str(z["system"]),
                "epoch": int(z["epoch"])}
        sha = str(z["universe_sha1"])
    if s.shape != (len(host.universe),) or sha != universe_sha1(host.universe):
        raise ValueError(f"{path}: baseline vector is not aligned with the {host.ds} universe")
    return s, info


def boot(ds, *, root=None, operating_dir=None, baseline=None, host=None, source="nodeindex", settings=None,
         n_boot=N_BOOT):
    """Bootstrap draws of the oracle node MCC of every system of `ds` (the boot_<DS>.json content)."""
    from habitus.decide import scorefile as S
    from habitus.eval.labels import host_labels
    from habitus.eval.metrics import confusion
    from habitus.eval.operating import _host, _scores_root, runs_dir
    from habitus.eval.tables import main_k
    t0 = time.time()
    root = _scores_root(root, settings)
    host = _host(ds, host, source, settings)
    y, n_total = host_labels(host)
    systems, info = {}, {}
    for k in (2, None):
        by_day, missing = S.load_by_day(ds, "E2", k, host.test, root)
        assert not missing
        systems[f"E2 composite {KN[k]}"] = S.node_scores(host, by_day, "composite")
        systems[f"E2 base {KN[k]}"] = S.node_scores(host, by_day, "base")
        # hard-tier exemption removed: the bounded acquittal applied to every node that QA was asked about
        systems[f"E2 acq_all {KN[k]}"] = S.node_scores(host, {
            d: {s: (max(float(r["qv"]), float(r["qt"])) * (1.0 - 0.5 * float(r["qa"])) if r.get("qa") is not None
                    else max(float(r["qv"]), float(r["qt"]))) for s, r in recs.items()} for d, recs in by_day.items()})
        for lv in ("L2", "L4"):
            bd, miss = S.load_by_day(ds, lv, k, host.test, root)
            if not miss and bd:
                systems[f"E1 {lv} {KN[k]}"] = S.node_scores(host, bd, "composite")
    systems["HABITUS main"] = systems[f"E2 composite {KN[main_k(ds)]}"]
    op = Path(operating_dir) if operating_dir is not None else runs_dir(settings)
    vp = op / f"vec_{ds}.npz"                         # form1 / form2 written by operating.forms
    if vp.exists():
        with np.load(vp) as z:
            for k_ in z.files:
                systems[k_] = np.asarray(z[k_], dtype=np.float64)
    else:
        _log(f"[{ds}] no {vp.name}: form1 / form2 not bootstrapped (run operating.forms first)")
    bp = Path(baseline) if baseline is not None else baseline_npz_path(ds, settings)
    s_b, binfo = read_baseline(bp, host)
    assert abs(binfo["own_mcc"] - BEST_BASELINE_MCC[ds]) < 6e-4, (ds, binfo)
    systems["best baseline"] = s_b
    info["best baseline"] = binfo
    st = Strata(y, host.ukey_codes, n_total)
    negcol = np.bincount(st.node_col[~st.pos], minlength=st.n_cols).astype(np.float64)
    ranked = {n: Ranked(v, st.pos, st.node_col, st.n_cols) for n, v in systems.items()}
    point = {}
    ones = st.ones()
    for n, rk in ranked.items():
        point[n] = float(_best_mcc_weighted(rk, st, negcol, ones)[0])
        ref = confusion(host, systems[n])["mcc"] or 0.0
        assert abs(point[n] - ref) < 1e-9, (ds, n, point[n], ref)
    arr = {n: np.empty(n_boot) for n in systems}
    for b0 in range(0, n_boot, CHUNK):
        b1 = min(n_boot, b0 + CHUNK)
        M = st.draws(b0, b1, SEED[ds])
        for n, rk in ranked.items():
            arr[n][b0:b1] = _best_mcc_weighted(rk, st, negcol, M)
        if b0 % 500 == 0:
            _log(f"[{ds}] boot {b1}/{n_boot} ({time.time() - t0:.0f}s)")
    return {"ds": ds, "n_boot": n_boot, "seed": SEED[ds], "sizes": st.sizes, "point": point, "info": info,
            "boot": {n: [round(float(x), 6) for x in a] for n, a in arr.items()},
            "seconds": round(time.time() - t0, 1)}


# --------------------------------------------------------------------------- #
# Merge: deltas and points with 95% percentile intervals
# --------------------------------------------------------------------------- #
def _ci(a):
    a = np.asarray(a, dtype=np.float64)
    return [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]


def merge(boots, pairs=PAIRS):
    """{ds: boot dict} -> {"boot_deltas": {"A - B": {ds: {point, ci95, p_le_0, p_ge_0}, "mean5": ...}},
    "boot_points": {ds: {system: {point, ci95}, "_info": ...}}}. p = P(delta <= 0); the five-host mean
    delta uses the per-host resamples (independent seeds)."""
    have = [ds for ds in HOSTS if ds in boots]
    summ = {"boot_deltas": {}, "boot_points": {}}
    for a, b in pairs:
        per, deltas = {}, []
        for ds in have:
            bt = boots[ds]["boot"]
            if a not in bt or b not in bt:
                continue
            d = np.asarray(bt[a]) - np.asarray(bt[b])
            pt = boots[ds]["point"][a] - boots[ds]["point"][b]
            per[ds] = {"point": pt, "ci95": _ci(d), "p_le_0": float((d <= 0).mean()), "p_ge_0": float((d >= 0).mean())}
            deltas.append(d)
        if len(deltas) == len(have) and have:
            m = np.mean(deltas, axis=0)
            per["mean5"] = {"point": float(np.mean([per[ds]["point"] for ds in have])), "ci95": _ci(m),
                            "p_le_0": float((m <= 0).mean())}
        summ["boot_deltas"][f"{a} - {b}"] = per
    for ds in have:
        summ["boot_points"][ds] = {n: {"point": v, "ci95": _ci(boots[ds]["boot"][n])}
                                   for n, v in boots[ds]["point"].items()}
        summ["boot_points"][ds]["_info"] = boots[ds]["info"]
    return summ


CSV_COLS = ("kind", "name", "ds", "point", "ci95_lo", "ci95_hi", "p_le_0", "p_ge_0")


def csv_rows(summary):
    """bootstrap_mcc.csv rows: every delta (per host and mean5), then every point MCC_H."""
    rows = []
    for name, per in summary["boot_deltas"].items():
        for ds in list(HOSTS) + ["mean5"]:
            r = per.get(ds)
            if r is None:
                continue
            rows.append({"kind": "delta", "name": name, "ds": ds, "point": r["point"], "ci95_lo": r["ci95"][0],
                         "ci95_hi": r["ci95"][1], "p_le_0": r["p_le_0"], "p_ge_0": r.get("p_ge_0")})
    for ds in HOSTS:
        pts = summary["boot_points"].get(ds)
        if not pts:
            continue
        for n, r in pts.items():
            if n == "_info":
                continue
            rows.append({"kind": "point", "name": n, "ds": ds, "point": r["point"], "ci95_lo": r["ci95"][0],
                         "ci95_hi": r["ci95"][1]})
    return rows


def write_csv(summary, path):
    """bootstrap_mcc.csv (floats as repr)."""
    from habitus.eval.derived import csv_text
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(csv_text(csv_rows(summary), CSV_COLS).encode("utf-8"))
    return path


def load_boots(paths):
    """{ds: boot dict} from boot_<DS>.json files."""
    out = {}
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        out[d["ds"]] = d
    return out


def strip_systems(boot_obj, drop):
    """A boot dict without the systems in `drop` (point / boot keys); the draws of the others are unchanged."""
    out = dict(boot_obj)
    for key in ("point", "boot"):
        if key in out:
            out[key] = {k: v for k, v in out[key].items() if k not in drop}
    return out
