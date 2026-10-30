"""Special-token neutralisation of every string a state carries.

Command lines, paths and endpoints are chosen by whoever ran the program, and both decision-model
tokenizers turn literal control-token text into the real control token. Laya (ModernBERT BPE) maps
"[SEP]" / "[CLS]" to its segment ids, and Qwen3.5 maps "<|im_end|>", "</think>", "<tool_call>" ...
to its chat-control ids, so a command line could end the state early or close the user turn inside
the prompt. Every string leaf (and dict key) is rewritten before serialisation, keeping it readable:
the bracket characters of the token are replaced,

    "[SEP]" -> "(SEP)"   "<|im_end|>" -> "(im_end)"   "</think>" -> "(/think)"   "|||IP_ADDRESS|||" -> "(IP_ADDRESS)"

Coverage = [CLS] [SEP] [PAD] [UNK] [MASK], any <|...|>, <think> </think> <tool_call> </tool_call>,
every entry of Qwen's added vocabulary (QWEN_ADDED is the frozen Qwen3.5 snapshot), and Laya's
non-whitespace added vocabulary ([unused0]..[unused82], |||IP_ADDRESS||| ...). Laya's whitespace-run
added tokens are content, not control, and are left alone.

Substitution runs to a fixed point (every pass deletes bracket characters, so it terminates): a
nesting such as "<|im_<|x|>end|>" cannot leave a token behind. Strings without any trigger character
are returned untouched without a regex pass.

The live check: the pinned tokenizer's tokenizer_config.json (Qwen/Qwen3.5-4B @ 851bf6e8, from
settings.paths.tokenizers or the HF snapshot under settings.paths.models) is read once. Its added
vocabulary must equal the frozen snapshot (pattern_status() == "live==snapshot"); anything else makes
pattern() raise, because the state bytes (and every answer-cache key) would change. "snapshot" (no
pinned tokenizer found) is accepted only after allow_frozen_snapshot(True), for unit tests.
"""

import json
import re
from pathlib import Path

# Qwen3.5 (0.8B / 4B share it) tokenizer_config.json added_tokens_decoder, ids 248044..248076,
# frozen so that state bytes never depend on whether a tokenizer is present.
QWEN_ADDED = (
    "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>", "<|object_ref_end|>",
    "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<|vision_start|>",
    "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>", "<tool_call>", "</tool_call>",
    "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>",
    "<tool_response>", "</tool_response>", "<think>", "</think>", "<|audio_start|>", "<|audio_end|>",
    "<tts_pad>", "<tts_text_bos>", "<tts_text_eod>", "<tts_text_bos_single>", "<|audio_pad|>")
QWEN_BACKBONE = "Qwen/Qwen3.5-4B"
QWEN_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
TOKENIZER_DIR = "Qwen3.5-4B@851bf6e8"          # under settings.paths.tokenizers (release bundle)

_BASE = (r"\[(?:CLS|SEP|PAD|UNK|MASK|unused\d{1,3})\]",     # BERT-style specials, Laya [unusedN]
         r"<\|[^\s<>|]{1,64}\|>",                            # any <|...|> control token
         r"\|\|\|[A-Z_]{1,32}\|\|\|",                        # Laya PII placeholders |||IP_ADDRESS|||
         r"</?think>", r"</?tool_call>")


class SanitizerError(RuntimeError):
    """The live tokenizer vocabulary does not match the frozen snapshot (or no tokenizer was found)."""


def _compile(literals):
    lits = sorted({re.escape(s) for s in literals if s.strip()}, key=lambda s: (-len(s), s))
    return re.compile("|".join(_BASE + tuple(lits)))


SPECIAL_RE = _compile(QWEN_ADDED)
_TRIGGER = frozenset("[<|")
_STATE = {"re": None, "trigger": None, "live": None, "source": None}
_ALLOW = {"frozen": False}


def _settings():
    from habitus import config
    s = config.settings
    return s() if callable(s) and not hasattr(s, "paths") else s


def tokenizer_config_candidates(settings=None):
    """The pinned tokenizer_config.json locations, in lookup order: the release tokenizer dir, then the
    HF hub snapshot at the pinned revision."""
    out = []
    try:
        paths = (settings or _settings()).paths
    except Exception:                                   # no config available: no live tokenizer
        return out
    tok = getattr(paths, "tokenizers", None)
    if tok:
        out.append(Path(tok) / TOKENIZER_DIR / "tokenizer_config.json")
    models = getattr(paths, "models", None)
    if models:
        name = "models--" + QWEN_BACKBONE.replace("/", "--")
        for hub in (Path(models) / "hub", Path(models)):
            out.append(hub / name / "snapshots" / QWEN_REVISION / "tokenizer_config.json")
    return out


def qwen_added_vocab(settings=None):
    """Contents of the pinned tokenizer's added vocabulary (no transformers import); None if not found."""
    for f in tokenizer_config_candidates(settings):
        try:
            with open(f, encoding="utf-8") as fh:
                dec = json.load(fh).get("added_tokens_decoder") or {}
            _STATE["source"] = str(f)
            return [v["content"] for v in dec.values()]
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def _resolve():
    """Read the live vocabulary once; fixed for the life of the process so states stay byte-identical."""
    if _STATE["live"] is None:
        live = qwen_added_vocab()
        extra = [s for s in (live or ()) if s.strip() and not SPECIAL_RE.fullmatch(s)]
        _STATE["re"] = _compile(QWEN_ADDED + tuple(extra)) if extra else SPECIAL_RE
        _STATE["trigger"] = _TRIGGER | {s[0] for s in extra}
        _STATE["live"] = "snapshot" if live is None else ("live+%d" % len(extra) if extra else "live==snapshot")
        _STATE["extra"] = extra
    return _STATE


def pattern_status():
    """'live==snapshot' (pinned tokenizer found, nothing beyond the frozen snapshot), 'live+N'
    (N entries beyond it) or 'snapshot' (no pinned tokenizer found)."""
    return _resolve()["live"]


def allow_frozen_snapshot(flag=True):
    """Unit tests only: accept status 'snapshot' (no tokenizer on disk). 'live+N' always raises."""
    _ALLOW["frozen"] = bool(flag)


def pattern(allow_frozen_snapshot=None):
    """The compiled neutraliser. Raises SanitizerError unless pattern_status() is 'live==snapshot'
    ('snapshot' passes only when allowed for unit tests)."""
    st = _resolve()
    live = st["live"]
    if live != "live==snapshot":
        allow = _ALLOW["frozen"] if allow_frozen_snapshot is None else allow_frozen_snapshot
        if not (live == "snapshot" and allow):
            raise SanitizerError(
                f"sanitizer vocabulary status {live!r}"
                + (f" (extra entries {st.get('extra', [])[:8]})" if live.startswith("live+") else
                   " (pinned Qwen3.5-4B tokenizer_config.json not found)")
                + ": states would not be byte-identical to the shipped answer cache")
    return st["re"]


def _neutral(m):
    return "(" + m.group(0).strip("<>[]|") + ")"


def sanitize_str(s):
    pat = pattern()
    if not any(c in s for c in _STATE["trigger"]):
        return s
    while True:
        s, n = pat.subn(_neutral, s)
        if not n:
            return s


def sanitize(obj):
    """A new object with every string leaf (and string dict key) neutralised; other scalars unchanged.
    Dict insertion order and list order are preserved (serialisation order is the state schema)."""
    if isinstance(obj, str):
        return sanitize_str(obj)
    if isinstance(obj, dict):
        return {(sanitize_str(k) if isinstance(k, str) else k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(sanitize(v) for v in obj)
    return obj


def special_ids_laya(tok):
    """Ids that must never occur in Laya's state segment: its specials ([UNK] [CLS] [SEP] [PAD] [MASK],
    <|padding|>, <|endoftext|>) and every non-whitespace added token ([unusedN], |||IP_ADDRESS||| ...)."""
    return frozenset(set(tok.all_special_ids) | {i for s, i in tok.get_added_vocab().items() if s.strip()})
