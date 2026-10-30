"""The calibration layer shared by every trained decision model (jev LoRA / head, laya la / mq).

Loss. Every trainer minimises the same weighted log score over the offered letters:
    L = sum_i w_i * [-sum_o t_io * log softmax(z_i)_o] / normaliser          (soft_ce)
with t_i the item's soft target in ITS letter order and w_i its sample weight; the normaliser is
sum_i w_i unless the trainer passes `scale` (train.jev: rows of the optimizer batch x mean weight).

Temperatures. One T per readout form ("inclusion-2", "choice-4"), fitted by weighted NLL on the
INNER-day items only (role "inner": the last pre-deployment day of each training host, never a
training day) and clamped to T_CLAMP = [0.5, 3]. The NLL is convex in beta = 1 / T, so a
golden-section search over beta in [1 / 3, 2] finds the clamped optimum deterministically.
Temperatures are written to run records and used ONLY by calibration_report (NLL, ECE, tail
coverage); every served answer and every statistic is at T = 1.

Reports (calibration_report): weighted NLL (fitted T and T = 1), ECE over 15 equal bins of the
top-letter confidence, the tail log score (log score at T = 1 of present inclusion items whose
comparator p_yes `ref_p` is < 0.05; all present items without a comparator), the rare-present AUC
(p_yes at T = 1 of those tail items against absent items, weighted Mann-Whitney) and tail coverage
(observed present rate among items with calibrated p_yes <= 1e-2 / 1e-3 must be <= 3x the predicted).

Items are questions.habit.types.Item (family, letters, target, weight, role, variant, qkey); logits
are 1-D arrays in the item's letter order.
"""

from __future__ import annotations

import math

import numpy as np

T_CLAMP = (0.5, 3.0)                  # = habit constants T_CLAMP (tested)
FORMS = ("inclusion-2", "choice-4")
TAIL_CUT = 0.05                       # tail items: comparator p_yes < 0.05
COVERAGE_CUTS = (1e-2, 1e-3)
COVERAGE_MAX = 3.0
ECE_BINS = 15
EPS = 1e-12


def _types():
    from habitus.questions.habit import types as T
    return T


def form_of(item):
    """Readout form of an item: "<family kind>-<offered letters>" ("inclusion-2" | "choice-4")."""
    T = _types()
    return f"{T.FAMILY_SPEC[T.Family.of(item.family)].kind}-{len(item.letters)}"


def soft_ce(z, target, mask, w, scale=None):
    """sum_i w_i * [-sum_o t_io log softmax(z_i)_o] / (sum_i w_i, or `scale`): the weighted log score
    over the offered letters (mask True). Masked letters take no probability mass and their target
    must be 0. z, target, mask: [B, K]; w: [B]."""
    import torch
    z = z.float().masked_fill(~mask, float("-inf"))
    logp = torch.log_softmax(z, dim=-1)
    per = -(target.float() * torch.where(mask, logp, torch.zeros_like(logp))).sum(-1)
    den = w.sum() if scale is None else torch.as_tensor(float(scale), device=z.device)
    return (per * w.float()).sum() / den


def _softmax(z, t=1.0):
    z = np.asarray(z, dtype=np.float64) / float(t)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _by_form(logits, items):
    out = {}
    for z, it in zip(logits, items):
        z = np.asarray(z, dtype=np.float64)
        if len(z) != len(it.letters):
            raise ValueError(f"item {it.qkey}: {len(z)} logits for {len(it.letters)} letters")
        out.setdefault(form_of(it), []).append((z, it))
    return out


def _nll(Z, Tg, W, t):
    lp = np.log(np.maximum(_softmax(Z, t), EPS))
    return float((W * -(Tg * lp).sum(-1)).sum() / W.sum())


def fit_temperatures(logits, items):
    """{"inclusion-2": T, "choice-4": T}: per readout form the T minimising the weighted NLL of the
    INNER-day items (role "inner"; anything else raises), clamped to T_CLAMP; 1.0 for a form without
    items. Golden-section search on beta = 1 / T."""
    bad = sorted({it.role for it in items if it.role != "inner"})
    if bad:
        raise ValueError(f"temperatures are fitted on inner-day items only, got roles {bad}")
    lo_t, hi_t = T_CLAMP
    out = {f: 1.0 for f in FORMS}
    for form, rows in _by_form(logits, items).items():
        Z = np.stack([z for z, _ in rows])
        Tg = np.array([it.target for _, it in rows], dtype=np.float64)
        W = np.array([it.weight for _, it in rows], dtype=np.float64)
        a, b = 1.0 / hi_t, 1.0 / lo_t
        g = (math.sqrt(5) - 1) / 2
        c, d = b - g * (b - a), a + g * (b - a)
        fc, fd = _nll(Z, Tg, W, 1 / c), _nll(Z, Tg, W, 1 / d)
        for _ in range(80):
            if fc <= fd:
                b, d, fd = d, c, fc
                c = b - g * (b - a)
                fc = _nll(Z, Tg, W, 1 / c)
            else:
                a, c, fc = c, d, fd
                d = a + g * (b - a)
                fd = _nll(Z, Tg, W, 1 / d)
        cands = [(_nll(Z, Tg, W, t), t) for t in (2.0 / (a + b), lo_t, hi_t)]    # the clamp ends exactly
        out[form] = round(float(min(hi_t, max(lo_t, min(cands)[1]))), 6)
    return out


def weighted_auc(pos, pos_w, neg, neg_w):
    """P(score of a positive > score of a negative) + 0.5 P(tie), pairs weighted by w_pos * w_neg."""
    pos, neg = np.asarray(pos, dtype=np.float64), np.asarray(neg, dtype=np.float64)
    pos_w, neg_w = np.asarray(pos_w, dtype=np.float64), np.asarray(neg_w, dtype=np.float64)
    if not len(pos) or not len(neg) or pos_w.sum() <= 0 or neg_w.sum() <= 0:
        return None
    o = np.argsort(neg, kind="stable")
    ns, cw = neg[o], np.concatenate([[0.0], np.cumsum(neg_w[o])])
    lo = np.searchsorted(ns, pos, side="left")
    hi = np.searchsorted(ns, pos, side="right")
    win = cw[lo] + 0.5 * (cw[hi] - cw[lo])
    return float((pos_w * win).sum() / (pos_w.sum() * neg_w.sum()))


def _ece(conf, acc, w, bins=ECE_BINS):
    if not len(conf) or w.sum() <= 0:
        return None
    b = np.minimum((conf * bins).astype(int), bins - 1)
    tot = 0.0
    for k in range(bins):
        m = b == k
        if m.any():
            wk = w[m].sum()
            tot += wk * abs(float((w[m] * acc[m]).sum() / wk) - float((w[m] * conf[m]).sum() / wk))
    return float(tot / w.sum())


def calibration_report(logits, items, temps, ref_p=None, tail_cut=TAIL_CUT):
    """nll / nll_t1, ece15, tail_log_score, rare_auc, tail_coverage (+ per form). `ref_p`: {qkey: a
    comparator's p_yes}; tail items = present inclusion items with ref_p < tail_cut (items without a
    ref_p are left out of the tail); without ref_p every present item is a tail item. Present =
    target share of "yes" > 0, absent = 0. Weighted by item weight; the tail log score and the AUC
    use T = 1 (the statistic's scale), NLL / ECE / coverage the fitted temperatures."""
    to_canonical = _types().to_canonical
    temps = {**{f: 1.0 for f in FORMS}, **(temps or {})}
    by_form, pooled = {}, {"nll": [], "nll_t1": [], "w": [], "conf": [], "acc": []}
    inc = {"p1": [], "pT": [], "t": [], "w": [], "tail": [], "pres": []}
    n_noref = 0
    for form, rows in _by_form(logits, items).items():
        Z = np.stack([z for z, _ in rows])
        Tg = np.array([it.target for _, it in rows], dtype=np.float64)
        W = np.array([it.weight for _, it in rows], dtype=np.float64)
        P1, PT = _softmax(Z), _softmax(Z, temps.get(form, 1.0))
        nll = -(Tg * np.log(np.maximum(PT, EPS))).sum(-1)
        nll1 = -(Tg * np.log(np.maximum(P1, EPS))).sum(-1)
        top = PT.argmax(-1)
        conf, acc = PT[np.arange(len(PT)), top], Tg[np.arange(len(Tg)), top]
        by_form[form] = {"n": len(rows), "T": temps.get(form, 1.0),
                         "nll": float((W * nll).sum() / W.sum()), "nll_t1": float((W * nll1).sum() / W.sum()),
                         "ece15": _ece(conf, acc, W)}
        for k, v in (("nll", nll), ("nll_t1", nll1), ("w", W), ("conf", conf), ("acc", acc)):
            pooled[k].append(v)
        if not form.startswith("inclusion"):
            continue
        for r, (_z, it) in enumerate(rows):
            y1 = to_canonical(P1[r], it.variant)[0]
            yT = to_canonical(PT[r], it.variant)[0]
            ty = to_canonical(it.target, it.variant)[0]
            pres = ty > 0
            if ref_p is None:
                tail = pres
            elif it.qkey in ref_p:
                tail = pres and float(ref_p[it.qkey]) < tail_cut
            else:
                tail = False
                n_noref += pres
            for k, v in (("p1", y1), ("pT", yT), ("t", ty), ("w", it.weight), ("tail", tail), ("pres", pres)):
                inc[k].append(v)
    rep = {"n": len(items), "temperatures": temps, "by_form": by_form,
           "tail_ref": "none: every present item" if ref_p is None else f"comparator p_yes < {tail_cut}",
           "tail_no_ref": int(n_noref)}
    if pooled["w"]:
        W = np.concatenate(pooled["w"])
        rep["nll"] = float((W * np.concatenate(pooled["nll"])).sum() / W.sum())
        rep["nll_t1"] = float((W * np.concatenate(pooled["nll_t1"])).sum() / W.sum())
        rep["ece15"] = _ece(np.concatenate(pooled["conf"]), np.concatenate(pooled["acc"]), W)
    else:
        rep.update({"nll": None, "nll_t1": None, "ece15": None})
    a = {k: np.asarray(v, dtype=bool if k in ("tail", "pres") else np.float64) for k, v in inc.items()}
    tail, absent = a["tail"], ~a["pres"]
    if tail.any():
        p, t, w = np.clip(a["p1"][tail], EPS, 1 - EPS), a["t"][tail], a["w"][tail]
        rep["tail_log_score"] = float((w * -(t * np.log(p) + (1 - t) * np.log(1 - p))).sum() / w.sum())
    else:
        rep["tail_log_score"] = None
    rep["n_tail"], rep["n_absent"] = int(tail.sum()), int(absent.sum())
    rep["rare_auc"] = weighted_auc(a["p1"][tail], a["w"][tail], a["p1"][absent], a["w"][absent])
    cov = {}
    for cut in COVERAGE_CUTS:
        m = a["pT"] <= cut
        if not m.any():
            cov[f"{cut:g}"] = {"n": 0, "predicted": None, "observed": None, "ratio": None, "ok": None}
            continue
        w = a["w"][m]
        pred, obs = float((w * a["pT"][m]).sum() / w.sum()), float((w * a["t"][m]).sum() / w.sum())
        ratio = obs / pred if pred > 0 else (math.inf if obs > 0 else 0.0)
        cov[f"{cut:g}"] = {"n": int(m.sum()), "predicted": pred, "observed": obs, "ratio": ratio,
                           "ok": bool(obs <= COVERAGE_MAX * pred)}
    rep["tail_coverage"] = cov
    return rep
