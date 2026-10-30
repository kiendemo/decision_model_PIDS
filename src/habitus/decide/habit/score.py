"""Habit-channel composite: route values + attestation tier (+ s_hat) -> node-day scores -> score files.

    base_of(qv, qt, s_hat)                      max(qv, qt, s_hat)  (HB0: max(qv, qt))
    composite(base, qa, hard, tier, ntype)      (s, a): s = base * (1 - LAMBDA * a)
    rows_for_day(host, ref, head, routes, tiers, habit)   one CompositeRow per head member node
    write(sd, root) / read(path) / load_days(ds, ldir, days, root)   <root>/scores/<ds>/<level dir>/<day>.json.gz
    node_scores(host, by_day)                   universe-aligned max over the test days
    entity_scores(host, s)                      max over the kappa pool (Host.ukey_codes)
    null_pool(host, by_day)                     (null node-day values, n null days) for bars.bars

Acquittal weight a (fixed a priori, never tuned):
    a = 0              in the hard tier (state.builder HARD bits, carried by the E2 route as "hard") or
                       when QA was not asked (qa None; every k = 0 route)
    a = qa             the node is LINEAGE-attested (attest.lineage_tiers)
    a = C_K * qa       any other tier (C_K = 0.5)
so s <= base, s >= (1 - LAMBDA * C_K) base = 0.75 base unless LINEAGE, s >= (1 - LAMBDA) base = 0.5 base
always, and s == base when hard or qa is None. The E2 composite is the special case "every node
LINEAGE" (decide.composite.compose_e2).

Because a higher tier can only LOWER s, and attest.lineage_tiers can only be revoked (never
upgraded) by added links or nodes, an edge-adding attacker cannot lower a node's composite through
the acquittal (fail-safe under mimicry).

Score files (types.ScoreDay; level dir = types.level_dir(level, kplan, arm)): one row per head member
node (per node-day, not per signature), the routes they were read from and the fallback of the
non-head signatures (state.select.fallback_value, in (-1, -0.5], below every row). Writes are atomic
(tmp + rename), meta first, compact JSON; same inputs -> byte-identical files (no timestamp).
Writes go under the habit root (runs/habit) only: decide.scorefile.check_writable refuses the
release directory and every read-only source tree.

Label-free: inputs are route values, tiers and habit statistics computed from cards, links and the
reference days; this module never opens a label file or imports label code.
"""

import gzip
import json
import math
import os

import numpy as np

from habitus.decide.composite import LAMBDA
from habitus.decide.habit.null2 import default_root
from habitus.decide.scorefile import check_writable, upos
from habitus.questions.habit import types as T
from habitus.questions.habit.constants import C_K
from habitus.questions.habit.head import members
from habitus.state.select import TYPES, fallback_rank

ROUTE_KEYS = ("qv", "qt", "qa", "hard", "cov_min")


# --------------------------------------------------------------------------- #
# The composite
# --------------------------------------------------------------------------- #
def _unit(x, what):
    x = float(x)
    if not (math.isfinite(x) and 0.0 <= x <= 1.0):
        raise ValueError(f"{what} {x!r} outside [0, 1]")
    return x


def base_of(qv, qt, s_hat=None):
    """base = max(qv, qt, s_hat); s_hat None (HB0 / HB-X, or s_hat not admitted) -> max(qv, qt).
    Every input must be a finite probability-scale value in [0, 1]."""
    b = max(_unit(qv, "qv"), _unit(qt, "qt"))
    if s_hat is not None:
        b = max(b, _unit(s_hat, "s_hat"))
    return b


def acquittal(qa, hard, tier, ntype="process"):
    """The acquittal weight a (module docstring)."""
    if ntype not in T.NODE_TYPES:
        raise ValueError(f"node type {ntype!r}")
    tier = T.Attest(int(tier))
    if hard or qa is None:
        return 0.0
    qa = _unit(qa, "qa")
    return qa if tier == T.Attest.LINEAGE else C_K * qa


def composite(base, qa, hard, tier, ntype="process"):
    """(s, a) of one node-day: s = base * (1 - LAMBDA * a), a = acquittal(...). Non-finite or
    out-of-range inputs raise."""
    base = _unit(base, "base")
    a = acquittal(qa, hard, tier, ntype)
    return base * (1.0 - LAMBDA * a), a


def _route(routes, sig):
    r = routes.get(sig)
    if r is None:
        raise KeyError(f"head signature {sig} has no route")
    return r


def rows_for_day(host, ref, head, routes, tiers, habit=None):
    """One CompositeRow per member node of every head signature (head.members: ALL members, node id
    order). `routes` = {sig: {qv, qt, qa, hard, cov_min}} (retrofit.e2_routes),
    `tiers` = {node: Attest} (attest.lineage_tiers over the members; a member without a tier raises),
    `habit` = {node: NodeHabit} (its s_hat enters base; a node without one has s_hat None)."""
    mem = members(host, ref.day, head)
    rows = []
    for sig in sorted(mem):
        r = _route(routes, sig)
        ntype = host.card(sig)["type"]
        qv, qt, qa, hard = r["qv"], r["qt"], r.get("qa"), bool(r.get("hard"))
        for n in mem[sig].tolist():
            if n not in tiers:
                raise KeyError(f"{host.ds} {ref.day}: node {n} ({sig}) has no attestation tier")
            h = habit.get(n) if habit else None
            s_hat = None if h is None else h.s_hat
            base = base_of(qv, qt, s_hat)
            tier = T.Attest(int(tiers[n]))
            s, a = composite(base, qa, hard, tier, ntype)
            rows.append(T.CompositeRow(int(n), sig, ntype, float(qv), float(qt), None if qa is None else float(qa),
                                       hard, tier, None if s_hat is None else float(s_hat), base, a, s))
    rows.sort(key=lambda x: x.node)
    return rows


def head_fallback(host, day, head):
    """{non-head sig: state.select fallback value} of `day` (the ScoreDay "fallback" block)."""
    inh = {s for t in TYPES for s in head.get(t, [])}
    return {s: v for s, v in sorted(fallback_rank(host, day).items()) if s not in inh}


def tier_census(rows):
    """{ntype: {tier name: nodes}} of a day's rows (every tier listed, zeros included)."""
    out = {t: {a.name: 0 for a in T.Attest} for t in T.NODE_TYPES}
    for r in rows:
        out[r.ntype][T.Attest(int(r.tier)).name] += 1
    return out


# --------------------------------------------------------------------------- #
# Score files
# --------------------------------------------------------------------------- #
def ldir_of(meta):
    """The level directory of a score-file meta block."""
    return T.level_dir(meta["level"], meta["kplan"], meta.get("arm"))


def day_path(ds, ldir, day, root=None):
    """<root>/scores/<ds>/<ldir>/<day>.json.gz (root None = null2.default_root()); no directory is created."""
    return os.path.join(default_root() if root is None else os.fspath(root), "scores", ds, ldir, f"{day}.json.gz")


def write(sd, root=None):
    """Write one ScoreDay atomically to day_path(ds, level dir, day, root); returns the path."""
    meta = sd.meta
    path = check_writable(day_path(meta["ds"], ldir_of(meta), meta["day"], root))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    data = json.dumps(sd.to_json(), separators=(",", ":"), allow_nan=False).encode("utf-8")
    with open(tmp, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as f:
        f.write(data)                       # no name / time in the gzip header: byte-identical files
    os.replace(tmp, path)
    return path


def read(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return T.ScoreDay.from_json(json.load(f))


def load_days(ds, ldir, days, root=None):
    """({day: ScoreDay} for the days whose file exists, [missing days])."""
    out, missing = {}, []
    for d in days:
        p = day_path(ds, ldir, d, root)
        if os.path.exists(p):
            out[d] = read(p)
        else:
            missing.append(d)
    return out, missing


# --------------------------------------------------------------------------- #
# Node, entity and null aggregation
# --------------------------------------------------------------------------- #
def _check_days(host, by_day, role):
    """Every ScoreDay is this host's, of `role`, and all share one (level, kplan, arm, caps, head rule)."""
    keys = set()
    for d, sd in by_day.items():
        m = sd.meta
        if m["ds"] != host.ds or m["day"] != d or m["role"] != role:
            raise ValueError(f"score file of {m['ds']} {m['day']} ({m['role']}) given as {host.ds} {d} ({role})")
        keys.add((m["level"], m["kplan"], str(m.get("arm")), json.dumps(m["caps"], sort_keys=True), m["head_rule"]))
    if len(keys) > 1:
        raise ValueError(f"{host.ds}: score files of different levels / plans / caps mixed: {sorted(keys)}")


def node_scores(host, by_day, column="s", days=None):
    """Max over the test days of each universe node's node-day value (head row or rarity fallback),
    aligned to host.universe. Every test day must be in `by_day` (a missing day would silently fall
    back to rarity); a NaN (None row value) or an uncovered universe node raises."""
    days = list(host.test) if days is None else list(days)
    missing = [d for d in days if d not in by_day]
    if missing:
        raise KeyError(f"{host.ds}: no score files for test days {missing}")
    _check_days(host, {d: by_day[d] for d in days}, "test")
    s = np.full(len(host.universe), -np.inf)
    for d in days:
        ids, v = by_day[d].node_values(host, column)
        if np.isnan(v).any():
            raise ValueError(f"{host.ds} {d}: column {column!r} has {int(np.isnan(v).sum())} None rows")
        pos = upos(host.universe, ids)
        ok = pos >= 0
        np.maximum.at(s, pos[ok], v[ok])
    if not np.isfinite(s).all():
        raise ValueError(f"{int((~np.isfinite(s)).sum())} universe nodes got no score from days {days}")
    return s


def entity_scores(host, s):
    """(pool code per entity, max score per entity): universe nodes pooled by kappa
    (Host.ukey_codes), codes ascending."""
    s = np.asarray(s, dtype=np.float64)
    codes = np.asarray(host.ukey_codes, dtype=np.int64)
    if codes.shape != s.shape:
        raise ValueError("one score per universe node")
    m = int(codes.max()) + 1 if len(codes) else 0
    ps = np.full(m, -np.inf)
    np.maximum.at(ps, codes, s)
    used = np.zeros(m, dtype=bool)
    used[codes] = True
    return np.nonzero(used)[0], ps[used]


def day_values(host, by_day, column="s", role=None):
    """{day: (node ids, value per node)} (ScoreDay.node_values: head rows + fallback), date order."""
    if role is not None:
        _check_days(host, by_day, role)
    return {d: by_day[d].node_values(host, column) for d in sorted(by_day)}


def null_pool(host, by_day, column="s"):
    """(null node-day values, number of null days) over {null day: ScoreDay}: every node of every
    null day (head rows + fallback), the pool decide.null.pool builds for E2."""
    _check_days(host, by_day, "null")
    vals = [v for _d, (_ids, v) in day_values(host, by_day, column).items()]
    v = np.concatenate(vals) if vals else np.zeros(0)
    if np.isnan(v).any():
        raise ValueError(f"{host.ds}: null column {column!r} has None rows")
    return v, len(vals)


__all__ = ["ROUTE_KEYS", "base_of", "acquittal", "composite", "rows_for_day", "head_fallback", "tier_census",
           "ldir_of", "day_path", "write", "read", "load_days", "node_scores", "entity_scores", "day_values",
           "null_pool"]
