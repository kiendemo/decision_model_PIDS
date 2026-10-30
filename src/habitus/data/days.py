"""Day enumeration from the PIDSMaker graphs: attack-free days, day masses, usable clean days.

Used ONLY by `habitus data manifest`, to recompute the usable clean days from a user's rebuilt
graphs and compare them with the frozen lists in habitus.hosts / configs/datasets.yaml. Scoring never
calls this module: it reads the frozen lists.

  _benign_dates  train + val dates minus attack dates (original defense/kb/build_kg._benign_dates)
  day_mass       bytes of the day's transformed graph files (original reflex/kshot.day_mass)
  usable_days    attack-free days, date order, without days holding < MIN_DAY_MASS of the median
                 clean-day bytes (original reflex/kshot.usable_days)
"""

import os
import statistics

from habitus.data.graphs import graph_paths

MIN_DAY_MASS = 0.10     # a usable day holds >= 10% of the host's median clean-day graph volume


def _as_spec(x):
    if isinstance(x, str):
        from habitus.hosts import spec
        return spec(x)
    return x


def _dates(x):
    """(train, val, attack windows) from a dataset name, a HostSpec or an upstream PIDSMaker cfg."""
    x = _as_spec(x)
    ds = getattr(x, "dataset", None)
    if ds is not None and hasattr(ds, "train_dates"):           # upstream cfg
        return list(ds.train_dates), list(ds.val_dates), [tuple(w) for w in ds.attack_to_time_window]
    return list(x.train_dates), list(x.val_dates), [tuple(w) for w in x.attack_windows]


def _benign_dates(x, split="all"):
    """Attack-free days. `split`: "all" (train + val), "train" or "val"."""
    train, val, windows = _dates(x)
    attack_dates = {tw[1][:10] for tw in windows}
    pool = {"all": train + val, "train": train, "val": val}[split]
    benign = [d for d in pool if d not in attack_dates]
    # Leakage guards (hard asserts).
    assert set(benign).isdisjoint(attack_dates), (
        f"LEAKAGE: benign dates {benign} intersect attack dates {attack_dates}")
    assert set(benign).issubset(set(train) | set(val)), "benign dates must come from train/val only"
    return benign


def _graphs_dir(spec, graphs_dir, settings):
    if graphs_dir is not None:
        return graphs_dir
    from habitus.data.upstream import graphs_dir as _gd
    return _gd(spec.ds, settings=settings)


def day_mass(graphs_dir, day):
    """Bytes of the day's transformed graphs: 0 for a day the collector did not record."""
    if not os.path.isdir(graphs_dir):
        raise FileNotFoundError(f"graph directory {graphs_dir} does not exist")
    return sum(os.path.getsize(p) for p in graph_paths(graphs_dir, day) if os.path.exists(p))


def usable_days(x, graphs_dir=None, *, settings=None):
    """Attack-free days in date order, without the (near-)empty ones. `x`: a dataset name or a
    HostSpec; graphs_dir defaults to data.upstream.graphs_dir(ds)."""
    spec = _as_spec(x)
    gdir = _graphs_dir(spec, graphs_dir, settings)
    dates = sorted(_benign_dates(spec))
    mass = {d: day_mass(gdir, d) for d in dates}
    med = statistics.median(mass.values()) if mass else 0
    return [d for d in dates if mass[d] > 0 and mass[d] >= MIN_DAY_MASS * med]


def manifest(x, graphs_dir=None, *, settings=None):
    """Recomputed day facts of a host from its graphs, next to the frozen ones."""
    spec = _as_spec(x)
    gdir = _graphs_dir(spec, graphs_dir, settings)
    benign = sorted(_benign_dates(spec))
    test = sorted(spec.configured_test)
    mass = {d: day_mass(gdir, d) for d in benign + test}
    usable = usable_days(spec, gdir)
    first = min(spec.configured_test)
    pre = [d for d in usable if d < first]
    return {
        "ds": spec.ds, "graphs_dir": os.fspath(gdir), "benign": benign, "day_mass": mass,
        "usable_clean": usable, "frozen_clean": list(spec.clean),
        "predeploy": pre, "frozen_predeploy": list(spec.predeploy),
        "card_days": benign + test, "frozen_card_days": list(spec.card_days),
        "ok": usable == list(spec.clean) and pre == list(spec.predeploy) and benign + test == list(spec.card_days),
    }
