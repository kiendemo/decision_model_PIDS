"""Candidate head and fallback rank (fixed a priori, label-free).

The decision model scores only a HEAD of each day's card signatures; every other signature gets a
fallback rank strictly below every decision score, so AP / AUC still cover the whole test universe.

    n_s      nodes carrying signature s on the day (day-local count, no reference, no label)
    process  every signature (cap 4000; beyond the cap: n_s asc, then sig)
    netflow  cap 6000: (k >= 1: kappa unseen in the reference first), n_s asc, sig
    file     cap 2000: (k >= 1: kappa unseen first), "changed on disk" first (an op key starting
             with write/create/rename/delete/exec, or any flag), n_s asc, sig
    k = 0    no reference: the unseen-first key is dropped, the order is otherwise the same
    fallback -1 + 0.5 / (1 + n_s): rarer is higher, always in (-1, -0.5], below every decision score
             (decision scores are probabilities in [0, 1])

Nothing here was chosen by looking at a metric. The same rule picks the head of a clean day when its
null scores are computed (with the reference of the test day it stands in for). The head and the
fallback depend on the day's cards, the reference's identity keys (kappa) and nothing else, so a test
day and a clean day are treated identically.
"""

import weakref
from collections import Counter

CAPS = {"process": 4000, "file": 2000, "netflow": 6000}
TYPES = ("process", "file", "netflow")
CHANGED = ("write", "create", "rename", "delete", "exec")

_KAPPAS = weakref.WeakKeyDictionary()      # host -> {tuple(ref days): frozenset(kappa)}


def fallback_value(n_s):
    """Fallback score of a non-head signature carried by n_s nodes that day."""
    return -1.0 + 0.5 * (1.0 / (1.0 + n_s))


def day_counts(host, day):
    """sig -> n_s (nodes carrying it on `day`)."""
    return Counter(host.nodes(day).values())


def changed_on_disk(card):
    b = card.get("behaviour", {})
    return bool(b.get("flags")) or any(k.startswith(CHANGED) for k in b.get("ops", {}))


def ref_days_of(ref):
    """The reference days a `ref` argument stands for, when it says (None = unknown / no ref)."""
    if ref is None:
        return None
    if hasattr(ref, "kappa_days"):
        r = getattr(ref, "ref", None)
        return list(r) if r is not None else None
    return list(ref)


def ref_kappas(host, ref):
    """The reference's kappa membership test: None (k = 0, no reference) or a container whose
    `in` is True for a kappa seen on >= 1 reference day. `ref` = None, a HostProfile
    (`kappa_days`) or a list of reference days."""
    if ref is None:
        return None
    kd = getattr(ref, "kappa_days", None)
    if kd is not None:
        return {k for k, n in kd.items() if n > 0} or None
    days = tuple(sorted(ref))
    if not days:
        return None
    try:
        per = _KAPPAS.setdefault(host, {})
    except TypeError:                       # a test double that cannot be weak-referenced
        per = {}
    if days not in per:
        per[days] = frozenset(host.inv_key(s) for d in days for s in set(host.nodes(d).values()))
    return per[days]


def head(host, day, ref=None, caps=CAPS):
    """{"process": [sig], "file": [sig], "netflow": [sig]} in priority order, each list capped.
    `ref`: None (k = 0), a HostProfile, or the list of reference days (k >= 1)."""
    days = ref_days_of(ref)
    if days is not None and day in days:
        raise ValueError(f"{day} is a reference day: a head is never drawn on its own reference")
    seen = ref_kappas(host, ref)
    n = day_counts(host, day)
    cards = host.day(day)["cards"]
    by_type = {t: [] for t in TYPES}
    for s in n:
        by_type[cards[s]["type"]].append(s)
    out = {}
    for t in TYPES:
        sigs = by_type[t]
        if t == "process":
            key = lambda s: (n[s], s)
        elif t == "netflow":
            key = (lambda s: (n[s], s)) if seen is None else \
                (lambda s: (host.inv_key(s) in seen, n[s], s))
        else:
            if seen is None:
                key = lambda s: (not changed_on_disk(cards[s]), n[s], s)
            else:
                key = lambda s: (host.inv_key(s) in seen, not changed_on_disk(cards[s]), n[s], s)
        out[t] = sorted(sigs, key=key)[: int(caps.get(t, 0))]
    return out


def flat(h):
    """Head dict -> one list (process, file, netflow order)."""
    return [s for t in TYPES for s in h.get(t, [])]


def limit_caps(limit, caps=CAPS):
    """Every head type capped at min(cap, limit); limit None = the caps unchanged. Used by the item
    export (train.items), whose `limit` parameter is recorded in the item manifest."""
    return dict(caps) if limit is None else {t: min(int(c), int(limit)) for t, c in caps.items()}


def fallback_rank(host, day):
    """{sig: fallback score} for every signature on `day` (the caller uses it for non-head sigs)."""
    return {s: fallback_value(c) for s, c in day_counts(host, day).items()}
