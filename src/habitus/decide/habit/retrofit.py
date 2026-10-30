"""HB0 (zero GPU): the habit channel's decision layer applied to the frozen routed E2 score files.

    from habitus.decide.habit import retrofit
    raw = retrofit.e2_day("CADETS_E3", "k2", "2018-04-06", ref=ref)   # one frozen E2 file, ref checked
    sd = retrofit.retrofit_day(host, ref, "k2", prof, "test")          # -> types.ScoreDay (level HB0)
    retrofit.run("CADETS_E3", kplans=("k2",))                          # test + null days

E2 kept every head signature's routed answers {qv, qt, qa, hard, cov_min} in the release score files
<paths.scores>/<ds>/E2_k<k>/<day>.json.gz. HB0 re-reads them, without any model call, and changes
only the decision layer:
  acquittal  per NODE, a = qa only for a LINEAGE-attested node (attest.lineage_tiers), a = C_K * qa
             otherwise, 0 in the hard tier / without QA (score.composite)
  cap        the two-tier per-day rule at B = 20 (bars.alerts)
Heads, states and answers are E2's own (the file's head is re-derived with head.q0 and must be
equal), so every HB0 vs E2 comparison is paired on identical units.

Plans: E2_DIRS maps a habit k plan to the E2 directory standing for it. k2 = decide.null.lodo_plan
(host, 2) (test days: reference shots(2); null days: each shot day with the other one); allc-pre =
LODO inside the pre-deployment window, which equals the routed k = all (E2_kall) only where
predeploy(host) == host.clean - not CADETS_E3, whose clean list holds 04-07..04-10 (after its first
test day): refused there. k0 = no reference (E2_k0 has no QA: s = base); its null days are the
host's own pre-deployment days. The plan comes from null2.plan / null2.ref_for, and e2_day checks
every E2 file's meta reference against it, so a mismatch fails loudly.

Outputs: <root>/scores/<ds>/HB0_<kplan>/<day>.json.gz through score.write (root = runs/habit).

Label-free: reads E2 score files (route answers), cards, links and pre-deployment reference days.
"""

import os
import time

from habitus.decide.composite import LAMBDA
from habitus.decide.habit import null2, score
from habitus.io import read_json_gz
from habitus.log import log
from habitus.questions.habit import types as T
from habitus.questions.habit.attest import lineage_tiers
from habitus.questions.habit.constants import C_K
from habitus.questions.habit.head import members, q0
from habitus.questions.habit.profile2 import get_profile2
from habitus.state.select import TYPES

LEVEL = "HB0"
E2_DIRS = {"k0": "E2_k0", "k2": "E2_k2", "allc-pre": "E2_kall"}
E2_KLABEL = {"k0": "0", "k2": "2", "allc-pre": "all"}          # routed meta "k" of each E2 directory
KMODE = {"k0": "k0", "k2": "k2", "allc-pre": "allc-pre"}       # Ref.k_mode of each plan
ATTEST, A_NET = "strict", "split"                              # the one decision rule (score-file meta)
ROLES = T.ROLES


# --------------------------------------------------------------------------- #
# Plans
# --------------------------------------------------------------------------- #
def ldir(kplan):
    return T.level_dir(LEVEL, kplan)


def kall_valid(host):
    """E2_kall stands for allc-pre only when the pre-deployment window is the whole clean list."""
    return list(T.predeploy(host)) == list(host.clean)


def plan(host, kplan):
    """[(Ref, role)]: null2.plan_refs (null days first, date order; then every test day)."""
    if kplan not in E2_DIRS:
        raise ValueError(f"HB0 retrofits E2 plans {sorted(E2_DIRS)}, not {kplan!r}")
    return null2.plan_refs(host, kplan)


# --------------------------------------------------------------------------- #
# E2 files
# --------------------------------------------------------------------------- #
def e2_root(settings=None):
    """The frozen routed score files: settings.paths.scores (read-only)."""
    from habitus.decide.scorefile import default_root
    return default_root(settings)


def e2_path(ds, kplan, day, root=None):
    if kplan not in E2_DIRS:
        raise ValueError(f"no E2 directory stands for k plan {kplan!r} ({sorted(E2_DIRS)})")
    return os.path.join(os.fspath(root) if root is not None else e2_root(), ds, E2_DIRS[kplan], f"{day}.json.gz")


def e2_day(ds, kplan, day, root=None, ref=None, role=None):
    """The raw frozen E2 file {"meta", "head", "scores", "fallback"} of (ds, kplan, day). Checks its
    meta (level E2, ds, day, k label) and, when given, that its reference equals `ref`'s (the plan's)
    and its role equals `role`. FileNotFoundError when the file does not exist."""
    p = e2_path(ds, kplan, day, root)
    raw = read_json_gz(p)
    m = raw.get("meta", {})
    want = {"level": "E2", "ds": ds, "day": day, "k": E2_KLABEL[kplan]}
    bad = {k: (m.get(k), v) for k, v in want.items() if str(m.get(k)) != str(v)}
    if bad:
        raise ValueError(f"{p}: meta {bad} (found, wanted)")
    if ref is not None:
        if ref.ds != ds or ref.day != day:
            raise ValueError(f"{p}: asked for {ref.ds} {ref.day}")
        if sorted(m.get("ref") or []) != list(ref.ref_days):
            raise ValueError(f"{p}: E2 reference {sorted(m.get('ref') or [])} is not the plan's {list(ref.ref_days)}")
    if role is not None and m.get("role") != role:
        raise ValueError(f"{p}: E2 role {m.get('role')!r}, the plan says {role!r}")
    return raw


def e2_routes(raw):
    """{sig: {"qv", "qt", "qa", "hard", "cov_min"}} of an E2 file (values checked: qv / qt in [0, 1],
    qa None or in [0, 1]); every head signature must have one."""
    out = {}
    for s, r in sorted(raw["scores"].items()):
        qa = r.get("qa")
        out[s] = {"qv": score._unit(r["qv"], "qv"), "qt": score._unit(r["qt"], "qt"),
                  "qa": None if qa is None else score._unit(qa, "qa"), "hard": bool(r.get("hard")),
                  "cov_min": None if r.get("cov_min") is None else float(r["cov_min"])}
    miss = [s for t in TYPES for s in raw["head"].get(t, []) if s not in out]
    if miss:
        raise KeyError(f"{len(miss)} head signatures without an E2 route, e.g. {miss[:3]}")
    return out


# --------------------------------------------------------------------------- #
# Tiers and profiles
# --------------------------------------------------------------------------- #
def profile(host, ref_days):
    """profile2.get_profile2 (None at k = 0)."""
    return get_profile2(host, list(ref_days)) if ref_days else None


def tiers_for(host, ref, h, prof):
    """{node: Attest} for every member of the head `h` on ref.day. k = 0 (prof None): no reference,
    every node NONE (tier0 is NONE without a profile); else attest.lineage_tiers."""
    mem = members(host, ref.day, h)
    nodes = sorted(int(n) for v in mem.values() for n in v.tolist())
    if prof is None:
        return {n: T.Attest.NONE for n in nodes}
    t = lineage_tiers(host, ref.day, prof, nodes=nodes)
    miss = [n for n in nodes if n not in t]
    if miss:
        raise KeyError(f"{host.ds} {ref.day}: lineage_tiers gave no tier to {len(miss)} head nodes")
    return {n: T.Attest(int(t[n])) for n in nodes}


# --------------------------------------------------------------------------- #
# One day
# --------------------------------------------------------------------------- #
def retrofit_day(host, ref, kplan, prof, role, e2=None, tiers=None, check_head=True, raw=None):
    """HB0 ScoreDay of one (host, ref) from its E2 file (e2: the E2 score root, default
    settings.paths.scores). `tiers`: precomputed tiers_for(host, ref, head, prof); `check_head`:
    re-derive the head with head.q0 at the file's caps and require equality; `raw`: the file already
    read by e2_day with this ref and role."""
    if kplan not in E2_DIRS:
        raise ValueError(f"HB0 retrofits E2 plans {sorted(E2_DIRS)}, not {kplan!r}")
    if ref.k_mode != KMODE[kplan]:
        raise ValueError(f"plan {kplan} with a {ref.k_mode} reference")
    if kplan == "allc-pre" and not kall_valid(host):
        raise ValueError(f"{host.ds}: E2_kall is not allc-pre (pre-deployment {T.predeploy(host)} != clean "
                         f"{host.clean}); allc-pre needs fresh routes (GPU)")
    if role not in ROLES:
        raise ValueError(f"role {role!r}")
    raw = raw if raw is not None else e2_day(host.ds, kplan, ref.day, e2, ref=ref, role=role)
    m = raw["meta"]
    caps = {t: int(c) for t, c in m["caps"].items()}
    h = {t: list(raw["head"].get(t, [])) for t in TYPES}
    if check_head and q0(host, ref, caps) != h:
        raise ValueError(f"{host.ds} {ref.day}: the E2 file's head is not head.q0 at caps {caps}")
    routes = e2_routes(raw)
    routes = {s: routes[s] for t in TYPES for s in h[t]}
    if tiers is None:
        tiers = tiers_for(host, ref, h, prof)
    rows = score.rows_for_day(host, ref, h, routes, tiers, None)
    meta = T.base_meta(ref, LEVEL, kplan, role, caps, head_rule="Q0", arm=None,
                       attest=ATTEST, a_net=A_NET, c_k=C_K, **{"lambda": LAMBDA},
                       e2={"dir": E2_DIRS[kplan], "engine": m.get("engine"), "written": m.get("written")},
                       n_head={t: len(h[t]) for t in TYPES}, n_rows=len(rows), tiers=score.tier_census(rows))
    return T.ScoreDay.from_rows(meta, h, routes, rows, score.head_fallback(host, ref.day, h))


# --------------------------------------------------------------------------- #
# A host
# --------------------------------------------------------------------------- #
def default_host(ds, settings=None):
    """Host(ds) over the release cards, with every clean day kept in its day LRU."""
    from habitus.data.host import Host
    host = Host(ds, settings=settings) if settings is not None else Host(ds)
    keep = getattr(host, "keep_clean_days", None)
    return keep() or host if callable(keep) else host


def run(ds, kplans=("k2", "allc-pre"), root=None, *, e2=None, host=None, roles=("test", "null"), settings=None):
    """Write the HB0 score files of `ds` for every (kplan, role) day under <root>/scores (root None =
    runs/habit); returns the paths written. allc-pre is refused (logged, nothing written) where
    E2_kall is not allc-pre (CADETS_E3)."""
    from habitus.decide.habit.channel import parse_kplans
    kplans = parse_kplans(kplans)
    host = host if host is not None else default_host(ds, settings)
    if host.ds != ds:
        raise ValueError(f"host for {ds} reports ds {host.ds}")
    root = root if root is not None else null2.default_root(settings)
    written = []
    for kplan in kplans:
        if kplan == "allc-pre" and not kall_valid(host):
            log(f"{ds} allc-pre refused: E2_kall is not allc-pre here (clean days after the first test day; "
                "fresh routes need a GPU)", "habit.retrofit")
            continue
        t0 = time.time()
        for ref, role in plan(host, kplan):
            if role not in roles:
                continue
            raw = e2_day(ds, kplan, ref.day, e2, ref=ref, role=role)
            h = {t: list(raw["head"].get(t, [])) for t in TYPES}
            prof = profile(host, ref.ref_days)
            tiers = tiers_for(host, ref, h, prof)
            sd = retrofit_day(host, ref, kplan, prof, role, e2=e2, tiers=tiers, raw=raw)
            written.append(score.write(sd, root))
            log(f"{ds} {kplan} {ref.day} ({role}): {sum(len(v) for v in h.values())} head sigs", "habit.retrofit")
        log(f"{ds} {kplan}: done in {time.time() - t0:.1f}s", "habit.retrofit")
    return written


__all__ = ["LEVEL", "E2_DIRS", "ldir", "kall_valid", "plan", "e2_root", "e2_path", "e2_day", "e2_routes", "profile",
           "tiers_for", "retrofit_day", "default_host", "run"]
