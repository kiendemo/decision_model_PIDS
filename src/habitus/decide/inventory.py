"""Score jobs: which (level, k, day, reference, role) score files exist, and the rule that makes them.

    ScoreJob(ds, level, k, day, ref, role)   one score file; k in {0, 2, None (= all)}, ref a tuple
    scan(ds, root)       the score files under <root>/<ds>/ as rows {dir, day, role, ref, caps},
                         from the first 64 KiB meta of each file only
    inventory(ds, root)  scan() as ScoreJobs, checked: every null file's reference is
                         lodo_plan(host, k)'s for its day, every test file's is ref_for(host, k)
    jobs(host, ...)      the jobs the null rules give (below); G4a asserts
                         set(jobs(...)) == set(inventory(...)) on the 5 hosts. Replay iterates
                         inventory(), never the rules.

Test jobs: E2 at every k; E1 at the levels HostSpec.e1_levels[k<k>] lists.
Null jobs (NULL_RULES, mirrored in configs/pipeline.yaml):
  E2   nulls at k = 2 and k = all on every host; at k = 0 only where HostSpec.k0_nulls
  L2   nulls at k = 2 and k = all on every host
  L4   nulls at k = 2 and k = all on every host
  L3   nulls at k = 0 only where HostSpec.k0_nulls
  no null files for L0 and L1, nor for L3 at k >= 1.
A null job at k is one per lodo_plan(host, k) entry: (clean day, its LODO reference).

The rules describe the shipped results (which commands were run per host), not a design choice.
Nothing here reads labels.
"""

import os
from dataclasses import dataclass

from habitus.decide.null import lodo_plan
from habitus.decide.scorefile import default_root, klabel, parse_k, read_meta

E2 = "E2"
LEVELS = (E2, "L0", "L1", "L2", "L3", "L4")
KS = (0, 2, None)
ROLES = ("test", "null")
SUFFIX = ".json.gz"
ALL_HOSTS, K0_NULLS = "all_hosts", "k0_nulls"
NULL_RULES = {
    E2: {"k2": ALL_HOSTS, "kall": ALL_HOSTS, "k0": K0_NULLS},
    "L2": {"k2": ALL_HOSTS, "kall": ALL_HOSTS},
    "L4": {"k2": ALL_HOSTS, "kall": ALL_HOSTS},
    "L3": {"k0": K0_NULLS},
}


@dataclass(frozen=True)
class ScoreJob:
    ds: str
    level: str                  # 'E2' | 'L0'..'L4'
    k: object                   # 0 | 2 | None (= all)
    day: str
    ref: tuple                  # reference days, in lodo_plan / ref_for order
    role: str                   # 'test' | 'null'

    @property
    def dir(self):
        return f"{self.level}_k{klabel(self.k)}"

    def row(self):
        return {"dir": self.dir, "day": self.day, "role": self.role, "ref": list(self.ref)}


def kkey(k):
    """'k0' | 'k2' | 'kall': the key of HostSpec.e1_levels and NULL_RULES."""
    return f"k{klabel(k)}"


def _order(job):
    return (LEVELS.index(job.level), KS.index(job.k), ROLES.index(job.role), job.day)


def sort_jobs(js):
    """Level, k, test before null, day: the order run() scores them in (one builder per reference)."""
    return sorted(js, key=_order)


# --------------------------------------------------------------------------- #
# Host facts
# --------------------------------------------------------------------------- #
class SpecHost:
    """The day lists of a HostSpec behind the Host API that lodo_plan / ref_for need (ds, clean,
    test, shots), so the inventory can be checked without loading any card."""

    def __init__(self, spec):
        self.spec, self.ds = spec, spec.ds
        self.clean, self.test = list(spec.clean), list(spec.test)

    def shots(self, k):
        return list(self.clean) if k is None else self.clean[:k]


def _hostspec(host=None, ds=None, spec=None):
    if spec is not None:
        return spec
    from habitus import hosts
    attr = getattr(host, "spec", None)
    if isinstance(attr, hosts.HostSpec):
        return attr
    return hosts.spec(ds if host is None else host.ds)


def spec_host(ds):
    """SpecHost of an allowed dataset (raises for any other name)."""
    from habitus import hosts
    hosts.assert_allowed(ds)
    return SpecHost(hosts.spec(ds))


# --------------------------------------------------------------------------- #
# The shipped inventory
# --------------------------------------------------------------------------- #
def level_dirs():
    """Every score-dir name a host may have: <level>_k<k> for the 6 levels and 3 ks."""
    return [f"{lv}_k{klabel(k)}" for lv in LEVELS for k in KS]


def _parse_dir(name):
    level, _, kl = name.rpartition("_k")
    return level, parse_k(kl)[0]


def scan(ds, root=None):
    """[{dir, day, role, ref, caps}] of every score file under <root>/<ds>/, sorted by job order.
    Reads only each file's meta (first 64 KiB). Only the score-dir names of level_dirs() are
    looked at, and anything else in <root>/<ds>/ raises (an inventory is never partial)."""
    from habitus import hosts
    hosts.assert_allowed(ds)
    root = default_root() if root is None else str(root)
    base = os.path.join(root, ds)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"no score dir {base}")
    known = set(level_dirs())
    extra = sorted(set(os.listdir(base)) - known)
    if extra:
        raise ValueError(f"{base}: unexpected entries {extra}")
    rows = []
    for name in level_dirs():
        d = os.path.join(base, name)
        if not os.path.isdir(d):
            continue
        level, k = _parse_dir(name)
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(SUFFIX):
                raise ValueError(f"{d}: unexpected file {fn}")
            day = fn[: -len(SUFFIX)]
            m = read_meta(os.path.join(d, fn))
            got = (m.get("ds"), m.get("level"), m.get("k"), m.get("day"))
            if got != (ds, level, klabel(k), day):
                raise ValueError(f"{d}/{fn}: meta (ds, level, k, day) {got} does not match its path")
            if m.get("role") not in ROLES:
                raise ValueError(f"{d}/{fn}: role {m.get('role')!r}")
            rows.append({"dir": name, "day": day, "role": m["role"], "ref": list(m.get("ref") or []),
                         "caps": m.get("caps")})
    return sorted(rows, key=lambda r: _order(_job(ds, r)))


def _job(ds, row):
    level, k = _parse_dir(row["dir"])
    return ScoreJob(ds, level, k, row["day"], tuple(row["ref"]), row["role"])


def inventory(ds, root=None, host=None, caps=None):
    """The ScoreJobs of the score files under <root>/<ds>/ (root None = settings.paths.scores),
    in job order. Asserts every null file's reference is lodo_plan(host, k)[day]'s and every test
    file's is ref_for(host, k) (exact order), and, when `caps` is given, that every file's caps
    equal it. `host`: anything with ds / clean / test / shots (default: the HostSpec's lists)."""
    from habitus.decide.routed import ref_for
    host = spec_host(ds) if host is None else host
    out = []
    plans = {}
    for r in scan(ds, root):
        j = _job(ds, r)
        if caps is not None and r["caps"] != caps:
            raise ValueError(f"{ds} {r['dir']}/{r['day']}: caps {r['caps']} != {caps}")
        if j.role == "null":
            if j.k not in plans:
                plans[j.k] = {c: tuple(ref) for c, ref in lodo_plan(host, j.k)}
            want = plans[j.k].get(j.day)
            if want is None:
                raise ValueError(f"{ds} {r['dir']}/{r['day']}: a null file on a day outside lodo_plan(k={klabel(j.k)})")
        else:
            if j.day not in host.test:
                raise ValueError(f"{ds} {r['dir']}/{r['day']}: a test file on a non-test day")
            want = tuple(ref_for(host, j.k))
        if j.ref != want:
            raise ValueError(f"{ds} {r['dir']}/{r['day']} ({j.role}): ref {list(j.ref)} != {list(want)}")
        out.append(j)
    return out


# --------------------------------------------------------------------------- #
# The rules
# --------------------------------------------------------------------------- #
def _null_allowed(level, k, spec):
    rule = NULL_RULES.get(level, {}).get(kkey(k))
    if rule == ALL_HOSTS:
        return True
    if rule == K0_NULLS:
        return bool(spec.k0_nulls)
    return False


def jobs(host, *, levels=LEVELS, ks=KS, nulls=True, spec=None):
    """The ScoreJobs of `levels` x `ks` on `host` (a Host, a SpecHost or a dataset name), in job
    order: test jobs (E2 at every k, E1 levels per HostSpec.e1_levels) and, with nulls, the
    null jobs NULL_RULES gives."""
    if isinstance(host, str):
        host = spec_host(host)
    spec = _hostspec(host, spec=spec)
    from habitus.decide.routed import ref_for
    bad = [lv for lv in levels if lv not in LEVELS] + [k for k in ks if k not in KS]
    if bad:
        raise ValueError(f"unknown levels / ks {bad}")
    out = []
    for lv in levels:
        for k in ks:
            if lv != E2 and lv not in tuple(spec.e1_levels.get(kkey(k), ())):
                continue
            ref = tuple(ref_for(host, k))
            out += [ScoreJob(host.ds, lv, k, d, ref, "test") for d in host.test]
            if nulls and _null_allowed(lv, k, spec):
                out += [ScoreJob(host.ds, lv, k, c, tuple(r), "null") for c, r in lodo_plan(host, k)]
    return sort_jobs(out)


def rows(js):
    """JSON rows of ScoreJobs (job order), the format of tests/golden/inventory.json."""
    return [j.row() for j in sort_jobs(js)]
