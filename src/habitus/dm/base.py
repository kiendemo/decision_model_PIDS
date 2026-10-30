"""The decision-model contract (pinned; every backend and question family codes against it).

A DecisionModel answers a TYPED question (<= 4 options) about a compiled host-reference state. It
makes one forward pass per presented form and returns a T = 1 distribution over the offered
options. A question family (questions.routed, questions.habit) builds the TypedQuestions and
combines the answered forms; the model never sees question semantics beyond the form texts.

    Form           one presentation of a question: system turn, state text, question text, options
                   in LETTER order and the canonical index of each letter
    TypedQuestion  one question: its answer key, family, Laya question type, forms, prefix group
    Decision       per-form, per-member letter- and canonical-order probabilities + coverage
    DecisionModel  the backend protocol (jev: decoder letter logits; laya: encoder [MASK] scorer)

Form.user() is the user turn of the routed and the habit questions, byte for byte: the KC-S1
user_prompt layout (State / Question / Options / Answer with one letter) that the habit questions
reuse with the same glue strings. Every answer-cache key is the sha1 of a prompt containing it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence, runtime_checkable

LETTERS = ("A", "B", "C", "D")
STATE_HEAD = "State:\n"
QUESTION_HEAD = "\n\nQuestion: "
OPTIONS_HEAD = "\nOptions:\n"
ANSWER_HEAD = "\nAnswer with one letter: "
THINK_SUFFIX = "</think>\n\n"
LAYOUTS = ("letter", "head", "la", "mq")
MAX_OPTIONS = len(LETTERS)


@dataclass(frozen=True)
class Form:
    """One presentation (one option order) of a question.

    name      'fwd' | 'rev' (qv, habit) or 'pos' | 'neg' (qt, qa)
    system    the exact system turn
    state     the exact compact-JSON state / view text
    question  the exact instruction text (already formatted)
    options   option texts in LETTER order (2..4)
    canon     canon[i] = canonical option index of letter i (a permutation of range(len(options)))
    """
    name: str
    system: str
    state: str
    question: str
    options: tuple
    canon: tuple

    def __post_init__(self):
        opts = tuple(str(o) for o in self.options)
        canon = tuple(int(c) for c in self.canon)
        object.__setattr__(self, "options", opts)
        object.__setattr__(self, "canon", canon)
        if not 2 <= len(opts) <= MAX_OPTIONS:
            raise ValueError(f"form {self.name!r}: {len(opts)} options (2..{MAX_OPTIONS})")
        if sorted(canon) != list(range(len(opts))):
            raise ValueError(f"form {self.name!r}: canon {canon} is not a permutation of {len(opts)} options")
        for f in ("name", "system", "state", "question"):
            if not isinstance(getattr(self, f), str):
                raise TypeError(f"form field {f} must be a str")

    @property
    def letters(self) -> tuple:
        return LETTERS[:len(self.options)]

    def user(self) -> str:
        """The user turn: byte-identical to the KC-S1 and the habit user_prompt."""
        L = self.letters
        return (STATE_HEAD + self.state + QUESTION_HEAD + self.question + OPTIONS_HEAD
                + "\n".join(f"{x}: {t}" for x, t in zip(L, self.options))
                + ANSWER_HEAD + ", ".join(L))


@dataclass(frozen=True)
class TypedQuestion:
    """One question about one state.

    qkey    routed: f"{sig}|{qid}"; habit: HabitQuestion.qkey ("<view sha1>|<family>|<bucket or ->")
    family  'qv' | 'qt' | 'qa' | a habit Family value ('P1'..'N2')
    qtype   Laya question type: 'score' (qv) | 'noul' (qt, qa) | 'choice' (habit)
    forms   >= 1 forms; forms[0] is the canonical one (routed: variants(qid) order; habit: ORDERS)
    group   prefix-share / encode-once key: routed f"{sig}|sig" (qv + qt) or f"{sig}|qa"; habit
            f"{view.kind}|{view.sha1}"
    """
    qkey: str
    family: str
    qtype: str
    forms: tuple
    group: str

    def __post_init__(self):
        forms = tuple(self.forms)
        object.__setattr__(self, "forms", forms)
        if not forms or not all(isinstance(f, Form) for f in forms):
            raise ValueError(f"question {self.qkey}: forms must be a non-empty sequence of Form")
        names = [f.name for f in forms]
        if len(set(names)) != len(names):
            raise ValueError(f"question {self.qkey}: repeated form names {names}")
        if len({len(f.options) for f in forms}) != 1:
            raise ValueError(f"question {self.qkey}: forms offer different numbers of options")

    @property
    def n_options(self) -> int:
        return len(self.forms[0].options)


@dataclass(frozen=True)
class Decision:
    """The answer to one TypedQuestion.

    layout        'letter' | 'head' | 'la' | 'mq'
    forms         the forms actually answered (MQ: ('fwd',))
    letter_probs  form -> per-member probabilities in LETTER order, T = 1
    canon_probs   same, canonical option order
    cov           form -> coverage (jev letter: the cached record's 'cov', 6 dp; head / la / mq: 1.0)
    cov_min       min(cov.values())
    """
    qkey: str
    layout: str
    forms: tuple
    letter_probs: Mapping = field(default_factory=dict)
    canon_probs: Mapping = field(default_factory=dict)
    cov: Mapping = field(default_factory=dict)
    cov_min: float = 1.0

    def __post_init__(self):
        if self.layout not in LAYOUTS:
            raise ValueError(f"decision {self.qkey}: layout {self.layout!r} not in {LAYOUTS}")
        object.__setattr__(self, "forms", tuple(self.forms))
        for name, m in (("letter_probs", self.letter_probs), ("canon_probs", self.canon_probs), ("cov", self.cov)):
            if set(m) != set(self.forms):
                raise ValueError(f"decision {self.qkey}: {name} forms {sorted(m)} != {sorted(self.forms)}")

    @property
    def members(self) -> int:
        return len(self.letter_probs[self.forms[0]])


class CacheMiss(RuntimeError):
    """Cache-only mode met prompts without a cached answer (raised before any model import)."""

    def __init__(self, keys, msg=None):
        self.keys = list(keys)
        super().__init__(msg or f"{len(self.keys)} answer(s) not in the cache (cache-only), "
                                f"first {self.keys[:3]}")


class Refused(ValueError):
    """Questions a backend refuses (never cut); reasons = {qkey: why}."""

    def __init__(self, reasons, msg=None):
        self.reasons = dict(reasons)
        super().__init__(msg or f"{len(self.reasons)} question(s) refused, e.g. "
                                f"{dict(list(self.reasons.items())[:2])}")


@runtime_checkable
class DecisionModel(Protocol):
    family: str            # 'jev' | 'laya'
    layout: str            # 'letter' | 'head' | 'la' | 'mq'
    tag: str               # answer-cache namespace

    def spec(self) -> dict: ...

    def refused(self, qs: Sequence[TypedQuestion]) -> dict: ...

    def decide(self, qs: Sequence[TypedQuestion]) -> dict: ...

    def close(self) -> None: ...


def canonical(probs, canon):
    """Letter-order probabilities -> canonical option order (canon[i] = canonical index of letter i)."""
    p = tuple(float(x) for x in probs)
    if len(p) != len(canon):
        raise ValueError(f"{len(p)} probabilities for {len(canon)} letters")
    out = [0.0] * len(p)
    for i, c in enumerate(canon):
        out[c] = p[i]
    return tuple(out)


def unique_questions(qs):
    """{qkey: TypedQuestion} in first-seen order; a repeated qkey must carry equal forms."""
    out = {}
    for q in qs:
        prev = out.get(q.qkey)
        if prev is None:
            out[q.qkey] = q
        elif prev.forms != q.forms:
            raise ValueError(f"two different questions share the answer key {q.qkey}")
    return out
