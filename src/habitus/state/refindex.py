"""RefIndex: the nearest reference cards of the same type (precedent sections).

Two exact cosine indices over the distinct cards of a set of reference days (rendered text deduped,
smallest signature kept): the invariant and the behaviour tokens of `tokens`, hashed into 2^18
columns (token pattern \\S+), sublinear TF-IDF with the IDF fitted on the distinct section texts of
the reference, L2-normalised. Similarities are rounded to 1e-9 before ranking, so an answer depends
on the query card alone and never on the batch it was asked in. nearest() takes neighbours
alternately from the invariant list (ordered by invariant similarity, then behaviour similarity,
then index) and the behaviour list (the converse), skipping repeats.

Only built on a precedent-cache miss (habitus.state.precedent); scikit-learn / scipy are imported here.
"""

import re

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

from habitus.data.cards import render

N_FEATURES = 2 ** 18
E = 3                       # default number of neighbours
TYPES = ("process", "file", "netflow")
SECTIONS = {"inv": ("invariant",), "beh": ("behaviour",)}

_SPLIT = re.compile(r"[/\\ .:=\-_]+")
_WS = re.compile(r"\s")
_HASHER = HashingVectorizer(n_features=N_FEATURES, token_pattern=r"\S+", lowercase=False,
                            alternate_sign=False, norm=None, dtype=np.float64)


def tokens(card, sections):
    """Card text: the type, key=value per scalar field plus up to 30 of its pieces split on path /
    word separators, list items, and dict entries with and without their value. Whitespace inside a
    token becomes '_' (tokens are whitespace-separated)."""
    out = [card["type"]]
    for sec in sections:
        s3 = sec[:3]
        for k, v in (card.get(sec) or {}).items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    out += (f"{s3}:{k}:{k2}={v2}", f"{s3}:{k}:{k2}")
            elif isinstance(v, list):
                out += (f"{s3}:{k}={it}" for it in v)
            else:
                v = str(v)
                out.append(f"{s3}:{k}={v}")
                out += (f"{s3}:{k}~{p}" for p in _SPLIT.split(v)[:30] if p)
    return " ".join(_WS.sub("_", t) for t in out)


def _weigh(X, idf):
    """Sublinear TF-IDF, unnormalised (TfidfTransformer(sublinear_tf=True) before its L2 step)."""
    W = X.astype(np.float64, copy=True)
    W.data = (1.0 + np.log(W.data)) * idf[W.indices]
    return W


class RefIndex:
    """The distinct cards (rendered text deduped, smallest signature kept) of a set of reference
    days, with an L2-normalised sublinear TF-IDF matrix per section (IDF over the distinct section
    texts of the reference) and per type."""

    def __init__(self, host, ref):
        self._hasher, self._weigh, self._tokens = _HASHER, _weigh, tokens
        keep = {}
        for d in ref:
            cards = host.day(d)["cards"]
            for s in set(host.nodes(d).values()):
                txt = render(cards[s])
                if txt not in keep or s < keep[txt]:
                    keep[txt] = s
        self.sigs = sorted(keep.values())
        cards = [host.card(s) for s in self.sigs]
        types = np.array([c["type"] for c in cards], dtype=object)
        self.rows = {t: np.flatnonzero(types == t) for t in TYPES}
        self.idf, self.W = {}, {}
        for sec, secs in SECTIONS.items():
            docs = [tokens(c, secs) for c in cards]
            X = _HASHER.transform(docs).tocsr() if docs else None
            if X is None:
                continue
            first = sorted({d: i for i, d in reversed(list(enumerate(docs)))}.values())
            df = np.bincount(X[first].indices, minlength=X.shape[1])
            self.idf[sec] = np.log((1.0 + len(first)) / (1.0 + df)) + 1.0
            W = normalize(_weigh(X, self.idf[sec]))
            self.W[sec] = {t: W[r] for t, r in self.rows.items() if len(r)}
        self.n = len(self.sigs)

    def _query(self, cards, sec, t):
        Q = self._hasher.transform([self._tokens(c, SECTIONS[sec]) for c in cards]).tocsr()
        Q = normalize(self._weigh(Q, self.idf[sec]))
        return np.round((Q @ self.W[sec][t].T).toarray(), 9)

    @staticmethod
    def _top(a, b, n):
        """Indices of the n best rows by (a desc, b desc, index asc); the whole tie block at the
        n-th value of `a` is considered."""
        cand = np.arange(len(a)) if len(a) <= n else np.flatnonzero(a >= np.partition(a, -n)[-n])
        return cand[np.lexsort((cand, -b[cand], -a[cand]))][:n].tolist()

    def nearest(self, cards, t, e=E):
        """[[example signature, ...] per card] (same type t, alternating invariant / behaviour
        neighbours, distinct)."""
        rows = self.rows.get(t, np.zeros(0, np.int64))
        if not len(rows) or not cards:
            return [[] for _ in cards]
        a, b = self._query(cards, "inv", t), self._query(cards, "beh", t)
        out = []
        for i in range(len(cards)):
            li, lb = self._top(a[i], b[i], e), self._top(b[i], a[i], e)
            ex = []
            for j in (x for pair in zip(li, lb) for x in pair):
                if j not in ex:
                    ex.append(j)
                if len(ex) == e:
                    break
            for j in li + lb:                        # fewer than e distinct in the interleave
                if len(ex) == e:
                    break
                if j not in ex:
                    ex.append(j)
            out.append([self.sigs[rows[j]] for j in ex])
        return out
