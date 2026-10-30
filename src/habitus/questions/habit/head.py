"""The habit channel's candidate head: Q0, the routed E2 head unchanged.

    from habitus.questions.habit.head import q0, members
    h = q0(host, ref)                      # {"process": [sig], "file": [sig], "netflow": [sig]}
    m = members(host, ref.day, h)          # {sig: sorted node ids} - EVERY member, not representatives

Q0 is state.select.head unchanged (caps P4000 / F2000 / N6000; processes by (n_s, sig); netflows
kappa-unseen first; files kappa-unseen first, then changed on disk): every habit vs E2 comparison is
paired on the same heads and states. Non-head signatures get state.select.fallback_value(n_s).
Robustness claims are CONDITIONAL on head membership: the file cap binds on most hosts, and
membership is then a signature-hash lottery that one padded edge re-rolls (reported, not claimed).
"""

import numpy as np

from habitus.state.select import CAPS, TYPES, flat, head

HEAD_RULES = ("Q0",)


def q0(host, ref, caps=CAPS):
    """Q0 = state.select.head(host, ref.day, ref days or None (k = 0), caps). `ref`: types.Ref."""
    return head(host, ref.day, list(ref.ref_days) or None, caps)


def head_sigs(h):
    """Head dict -> one list (process, file, netflow order) = state.select.flat."""
    return flat(h)


def members(host, day, h):
    """{sig: sorted np.int64 node ids carrying it on `day`} for every head signature (all members:
    per-node observed sets and per-node composites cover every one of them)."""
    want = set(flat(h))
    out = {}
    for n, s in host.nodes(day).items():
        if s in want:
            out.setdefault(s, []).append(n)
    return {s: np.sort(np.asarray(v, dtype=np.int64)) for s, v in out.items()}


__all__ = ["HEAD_RULES", "TYPES", "q0", "head_sigs", "members"]
