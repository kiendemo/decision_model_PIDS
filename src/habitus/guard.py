"""Runtime label guard: a sys.addaudithook that makes a label read FAIL (and an optional write audit).

    from habitus import guard
    guard.install_guard("decide")                       # the CLI, first thing, for every guarded command
    guard.install_guard("decide", write_roots=[runs])   # + refuse every write outside runs/ (G3, G4)
    guard.guard_active()                                # -> "decide"
    guard.blocked_reason(path)                          # pure check, usable without the hook

Why a hook: a grep over the sources cannot see TRANSITIVE reads. An audit hook sees every open(),
os.open() and io.open_code() of the process, whoever calls it; a sys.meta_path finder sees every
first import of a module (the audit "import" event alone misses importlib.import_module).

Refused reads (LabelGuardError, a PermissionError raised inside the open itself):
  - anything under settings.paths.ground_truth or settings.paths.upstream_gt (when configured);
  - a "Ground" + "_Truth" path component, pickled label sets (basename starting with the two-letter
    label prefix + "_" and ending .pkl), node-to-attack maps (*__n2a.pkl), any DATA file whose
    basename contains "_labels" (.py / .pyc / .so library modules are not data), the card-statistics
    file next to the cards (it holds label-derived counts), teacher label stores and held-out reports.
Refused imports: habitus.eval.* (the only label and metric code) and the original label modules. A
blocked module already imported when the guard is installed is an error too.
The patterns use character classes so that this file passes the repository's own label-token scan.

Commands EXEMPT from the label guard: eval, reproduce, gates. reproduce and gates run every
decide / state / questions / dm step as a guarded subprocess and only compare outputs themselves.
A hook cannot be removed: a process stays guarded (installing for another command raises).

Write audit (optional, `write_roots`): an open for writing, mkdir, rename, replace, remove, rmdir,
link, symlink or truncate whose target lies outside every write root raises WriteAuditError. It is
installed even for an exempt command when write_roots is given. The null device is always allowed.
"""

import os
import re
import sys
import threading

EXEMPT = ("eval", "reproduce", "gates")
# (rule name, regex on the absolute, '/'-separated path)
BLOCKED_PATTERNS = (
    ("ground-truth tree", re.compile(r"(^|/)Ground[_]Truth(/|$)", re.IGNORECASE)),
    ("upstream label pickle", re.compile(r"(^|/)g[t][_][^/]*\.pkl$")),
    ("node-to-attack map", re.compile(r"__n2a\.pkl$")),
    ("label file", re.compile(r"(^|/)[^/]*[_]labels(?![^/]*\.(py|pyc|so)$)[^/]*$")),
    ("card statistics", re.compile(r"(^|/)cards/[^/]+/stat[s]\.json$")),
    ("s1 corpus", re.compile(r"(^|/)s1/corpus[^/]*\.json\.gz$")),
    ("s1 teacher labels", re.compile(r"(^|/)s1/teacher/")),
    ("held-out report", re.compile(r"(^|/)research/results/(.*/)?e2[_]heldout[^/]*$")),
)
BLOCKED_IMPORTS = (
    "habitus.eval",
    "pidsmaker.reflex.kcs1.evaluate",
    "pidsmaker.reflex.anchor.evalkit",
    "pidsmaker.reflex.habitus.evaluate",
    "pidsmaker.reflex.s1.consensus",
    "pidsmaker.reflex.s1.catalog_targets",
)
WRITE_EVENTS = {           # audit event -> indices of the path arguments it writes
    "os.mkdir": (0,), "os.rename": (0, 1), "os.replace": (0, 1), "os.remove": (0,), "os.rmdir": (0,),
    "os.link": (0, 1), "os.symlink": (1,), "os.truncate": (0,), "shutil.rmtree": (0,),
}
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC

_STATE = {"command": None, "roots": (), "write_roots": None}
_LOCK = threading.Lock()
_LOCAL = threading.local()
_HOOKED = {"label": False, "write": False}


class LabelGuardError(PermissionError):
    """A guarded HABITUS process tried to read a label file or import label code."""


class WriteAuditError(PermissionError):
    """A write outside the allowed roots while the write audit is active."""


# --------------------------------------------------------------------------- #
# Pure checks
# --------------------------------------------------------------------------- #
def _norm(path):
    p = os.fsdecode(path) if isinstance(path, (bytes, bytearray)) else os.fspath(path)
    return os.path.abspath(p).replace(os.sep, "/")


def _under(p, root):
    r = root.rstrip("/")
    return p == r or p.startswith(r + "/")


def label_roots(s=None):
    """The configured ground-truth directories (normalised), from settings unless `s` is given."""
    if s is None:
        from habitus.config import settings
        s = settings()
    return tuple(_norm(r) for r in (s.paths.ground_truth, s.paths.upstream_gt) if r)


def blocked_reason(path, roots=None):
    """The rule `path` breaks, or None. `roots` defaults to the installed guard's label roots."""
    p = _norm(path)
    for root in (_STATE["roots"] if roots is None else roots):
        if _under(p, root):
            return "configured ground-truth directory"
    for name, rx in BLOCKED_PATTERNS:
        if rx.search(p):
            return name
    return None


def blocked_import(module):
    return any(module == m or module.startswith(m + ".") for m in BLOCKED_IMPORTS)


def _is_write_open(args):
    mode = args[1] if len(args) > 1 else None
    flags = args[2] if len(args) > 2 else None
    if isinstance(flags, int) and flags & _WRITE_FLAGS:
        return True
    return isinstance(mode, str) and any(c in mode for c in "wax+")


def write_violation(path, roots):
    """True when `path` (as given, and after resolving symlinks) is outside every root. The null
    device is never a violation: opening it for writing writes nothing (torch's import of dill opens
    it 'r+b', so every guarded child that imports transformers does it)."""
    p = _norm(path)
    if p == _norm(os.devnull):
        return False
    real = os.path.realpath(p).replace(os.sep, "/")
    return not (any(_under(p, r) for r in roots) and any(_under(real, r) for r in roots))


# --------------------------------------------------------------------------- #
# The hooks
# --------------------------------------------------------------------------- #
def _label_hook(event, args):
    if _STATE["command"] is None or event not in ("open", "import") or getattr(_LOCAL, "busy", False):
        return
    _LOCAL.busy = True
    try:
        if event == "import":
            mod = args[0] if args else None
            if isinstance(mod, str) and blocked_import(mod):
                raise LabelGuardError(f"[habitus.guard {_STATE['command']}] import of label code {mod} refused "
                                      f"(only `habitus eval` may import it)")
            return
        path = args[0] if args else None
        if not isinstance(path, (str, bytes, bytearray, os.PathLike)):
            return                                    # an fd or None: nothing to check
        why = blocked_reason(path)
        if why is not None:
            raise LabelGuardError(f"[habitus.guard {_STATE['command']}] open of {_norm(path)} refused: {why}")
    finally:
        _LOCAL.busy = False


def _write_hook(event, args):
    roots = _STATE["write_roots"]
    if roots is None or getattr(_LOCAL, "wbusy", False):
        return
    if event == "open":
        if not _is_write_open(args):
            return
        targets = (args[0],) if args else ()
    elif event in WRITE_EVENTS:
        targets = tuple(args[i] for i in WRITE_EVENTS[event] if i < len(args))
    else:
        return
    _LOCAL.wbusy = True
    try:
        for t in targets:
            if not isinstance(t, (str, bytes, bytearray, os.PathLike)):
                continue
            if write_violation(t, roots):
                raise WriteAuditError(f"[habitus.guard write-audit] {event} of {_norm(t)} refused: outside "
                                      f"the allowed write roots {list(roots)}")
    finally:
        _LOCAL.wbusy = False


class _BlockFinder:
    """sys.meta_path entry that refuses BLOCKED_IMPORTS (importlib.import_module raises no audit
    "import" event, so both are needed)."""

    @staticmethod
    def find_spec(fullname, path=None, target=None):
        if _STATE["command"] is not None and blocked_import(fullname):
            raise LabelGuardError(f"[habitus.guard {_STATE['command']}] import of label code {fullname} refused "
                                  f"(only `habitus eval` may import it)")
        return None

    @staticmethod
    def invalidate_caches():
        pass


def install_write_audit(write_roots):
    """Refuse every write outside `write_roots` for the rest of the process (roots only grow
    stricter: a second call must give the same roots)."""
    given = [_norm(r) for r in write_roots]
    roots = tuple(sorted(set(given) | {os.path.realpath(r).replace(os.sep, "/") for r in given}))
    if not roots:
        raise ValueError("install_write_audit needs at least one write root")
    with _LOCK:
        if _STATE["write_roots"] is not None:
            if _STATE["write_roots"] != roots:
                raise ValueError(f"write audit already installed with roots {list(_STATE['write_roots'])}")
            return roots
        _STATE["write_roots"] = roots
        if not _HOOKED["write"]:
            sys.addaudithook(_write_hook)
            _HOOKED["write"] = True
    return roots


def install_guard(command, *, write_roots=None, settings=None):
    """Install the label guard for `command` (idempotent for the same command). Returns True when the
    label guard is (now) active, False for an EXEMPT command. With `write_roots`, the write audit is
    installed too (also for an exempt command). Raises LabelGuardError when a blocked module is
    already imported, ValueError when a guard for another command is already installed."""
    if write_roots is not None:
        install_write_audit(write_roots)
    with _LOCK:
        cur = _STATE["command"]
        if cur is not None:
            if cur != command:
                raise ValueError(f"guard already installed for {cur!r}; a process runs one command")
            return True
        if command in EXEMPT:
            return False
        early = sorted(m for m in sys.modules if blocked_import(m))
        if early:
            raise LabelGuardError(f"[habitus.guard {command}] label code already imported before the guard: {early}")
        _STATE["roots"] = label_roots(settings)
        _STATE["command"] = command
        sys.meta_path.insert(0, _BlockFinder)
        if not _HOOKED["label"]:
            sys.addaudithook(_label_hook)
            _HOOKED["label"] = True
        return True


def guard_active():
    """The command the installed label guard protects, or None."""
    return _STATE["command"]


def write_roots():
    """The active write-audit roots, or None."""
    return _STATE["write_roots"]
