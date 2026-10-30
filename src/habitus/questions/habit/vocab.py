"""Closed vocabularies and the attested-name rule.

    from habitus.questions.habit import vocab
    vocab.entity_words(card, tier)          # typed words; raw strings only for an IDENTITY tier
    vocab.role_class(card, tier)            # "unknown" unless the identity is identity-attested
    vocab.cmd_shape("/tmp/x -c <n>")        # 'n/short' (closed grammar CMD_SHAPE_RE)

Every word a habit view prints about an entity comes from a closed list here, except the raw
name / path / command line of an IDENTITY-attested identity (its exact kappa' / path kappa was seen
on a reference day and carries no placeholder). An attacker chooses the strings of the entities it
creates; mapped onto typed words they can only pick a class, never write text the model reads.
Hence byte invariance: for a card that is NOT identity-attested, entity_words(card) is a function of
(type, os, image_dir / dir class, extension class, the shape words) only, so renaming the program or
file (its basename: file_name, the last component of executable, argv[0] of the command line, a
file's basename) while keeping the extension class, or injecting text into those basenames, leaves
it byte-identical. The shape words read the ARGUMENTS of a command line (argv[1:]) and the DIRECTORY
part of a path for exactly that reason; they are Atau (typed words of attacker-choosable strings).
argv[0] is a leading double-quoted token, else the first whitespace-separated token: a program
renamed to a name with a space inside an UNQUOTED command line shifts its arguments (the one rename
the shape words do not absorb).

Vocabularies
  dir       system application home staging cache config other none        (state.sections.DIR_CLASSES)
  endpoint  loopback private enclave internal external unknown              (state.sections.ENDPOINTS)
  port      well-known registered ephemeral none                            (state.profile.port_class)
  ext       executable script library archive document data none            (state.sections EXT_CLASSES)
  role      ROLE_CLASSES, from the frozen catalog table below
  shapes    cmd_shape  "<marks>/<argument length>"   marks = placeholder kinds in argv[1:] or "plain"
            path_shape "<marks>/<directory depth>"   marks = placeholder kinds in the directory part
                                                     (+ "dot": a hidden directory component) or "plain"

The extension class here reads the EXTENSION ONLY (state.sections.ext_class applied to the invariant
alone): the routed state calls an extensionless file that was executed today "executable", which is
behaviour, and a habit view must not move with test-day behaviour.

Role table. A process's role class is looked up by its lower-cased basename in a table built by code
from four public catalogs under settings.paths.catalogs (fetched by scripts/fetch_catalogs.sh at
pinned versions; no evaluation data, no hand-picked name): Debian bookworm Contents-amd64 (the
archive section of every program under bin / sbin / libexec / games), GTFOBins functions (posix),
LOLBAS command categories and the xcyclopedia install locations (Windows). First matching rule wins,
rules in ROLE_RULES order:

  posix    shell       Debian section "shells"
           interpreter Debian section in INTERPRETER_SECTIONS (language runtimes)
           exec-proxy  GTFOBins function shell / command / reverse-shell / bind-shell
           transfer    GTFOBins function download / upload
           dual-use    any other GTFOBins function (file-read, file-write, library-load, ...)
           network     Debian section net / web / mail / comm / httpd / news
           admin       Debian section admin
           devel       Debian section devel / libdevel / vcs / debug
           system      Debian section utils / libs / oldlibs / kernel / otherosfs / misc
           application any other Debian section
  windows  exec-proxy  LOLBAS category Execute / AWL Bypass / UAC Bypass / Compile
           transfer    LOLBAS category Download / Upload
           dual-use    any other LOLBAS category (Copy, ADS, Dump, Credentials, Encode, ...)
           system      xcyclopedia location under C:/Windows
           application xcyclopedia location under C:/Program Files
A basename listed by several entries takes the first rule any of them matches. The posix table
serves linux, freebsd and android hosts. No Windows catalog tags a shell, so cmd.exe and
powershell.exe carry their LOLBAS roles (transfer, exec-proxy). The table is applied ONLY to
identity-attested identities and to class pooling (role_class): an unattested payload named
setup.exe borrows nothing. It is built lazily on first use (~3 s) and checked against
ROLE_TABLE_SHA256: missing catalogs or a changed catalog / rule raise RoleTableError (role words
would otherwise differ from the frozen runs). catalog_hashes() lists the sha256 of every catalog
file it reads.

Label-free: public catalogs (function tags, never campaign text) and card invariants only.
"""

import gzip
import hashlib
import json
import os
import re
import zipfile
from collections.abc import Mapping
from itertools import combinations

from habitus.data.cards import ext_of
from habitus.hosts import OS_FAMILY
from habitus.questions.habit import types as T
from habitus.questions.habit.attest import PLACEHOLDER_KINDS, placeholder_kinds
from habitus.state.profile import LEVEL_CUTS
from habitus.state.profile import port_class as _port_class
from habitus.state.sections import (CMD_CHARS, DIR_CLASSES, ENDPOINTS, EXE_CHARS, EXT_WORDS, PATH_CHARS,
                                    _word, basename)
from habitus.state.sections import ext_class as _state_ext_class

DIR_ORDER, EP_ORDER, PORT_ORDER = T.DIR_ORDER, T.EP_ORDER, T.PORT_ORDER
OS_WORDS = tuple(sorted(set(OS_FAMILY.values()))) + ("unknown",)
FREQ_WORDS = tuple(name for _cut, name in LEVEL_CUTS) + ("low", "none")      # Profile2.frequency

# --------------------------------------------------------------------------- #
# Shapes: closed grammars over the placeholder kinds of cards.abstract_text
# --------------------------------------------------------------------------- #
CMD_LEN_BINS = ((0, "empty"), (31, "short"), (127, "medium"))     # argv[1:] chars <= bound -> word
CMD_LEN_WORDS = tuple(w for _b, w in CMD_LEN_BINS) + ("long",)
DEPTH_WORDS = ("depth-0", "depth-1", "depth-2", "depth-3", "depth-4+")
CMD_MARKS = PLACEHOLDER_KINDS                                       # guid, hex, rand, n
PATH_MARKS = PLACEHOLDER_KINDS + ("dot",)                           # + a hidden directory component
_SPLIT = re.compile(r"[\\/]+")


def _marks_re(kinds):
    """Regex alternation of every non-empty subsequence of `kinds` joined by '+' in kinds order."""
    alts = ["+".join(c) for r in range(len(kinds), 0, -1) for c in combinations(kinds, r)]
    return "(?:plain|" + "|".join(re.escape(a) for a in alts) + ")"


CMD_SHAPE_RE = re.compile(r"^(?:none|" + _marks_re(CMD_MARKS) + "/(?:" + "|".join(map(re.escape, CMD_LEN_WORDS))
                          + "))$")
PATH_SHAPE_RE = re.compile(r"^(?:none|" + _marks_re(PATH_MARKS) + "/(?:" + "|".join(map(re.escape, DEPTH_WORDS))
                           + "))$")


def _marks(found, kinds):
    got = [k for k in kinds if k in found]
    return "+".join(got) if got else "plain"


def _all_shapes(kinds, tails):
    marks = ["plain"] + ["+".join(c) for r in range(1, len(kinds) + 1) for c in combinations(kinds, r)]
    return ("none",) + tuple(f"{m}/{t}" for m in marks for t in tails)


def cmd_args(cmdline):
    """argv[1:] of a (normalised) command line as one string: the first token is dropped (a quoted
    first token up to its closing quote), so a renamed program never changes it."""
    s = str(cmdline or "").strip()
    if not s:
        return ""
    if s[0] == '"':
        j = s.find('"', 1)
        return "" if j < 0 else s[j + 1:].strip()
    parts = s.split(None, 1)
    return parts[1].strip() if len(parts) > 1 else ""


def cmd_shape(cmdline):
    """'none' without a command line, else '<marks>/<length word>' of its arguments argv[1:]:
    marks = the placeholder kinds present (guid, hex, rand, n in that order) or 'plain'; length
    word = empty (0 chars) | short (<= 31) | medium (<= 127) | long. Matches CMD_SHAPE_RE."""
    if not str(cmdline or "").strip():
        return "none"
    args = cmd_args(cmdline)
    n = len(args)
    word = next((w for b, w in CMD_LEN_BINS if n <= b), "long")
    return f"{_marks(placeholder_kinds(args), CMD_MARKS)}/{word}"


def path_dirs(path):
    """Directory components of a path (both separators; the basename and empty parts dropped)."""
    parts = [p for p in _SPLIT.split(str(path or "").strip()) if p]
    return parts[:-1]


def path_shape(path):
    """'none' without a path, else '<marks>/<depth word>' of its DIRECTORY part: marks = the
    placeholder kinds present in the directory components (+ 'dot' when one starts with '.') or
    'plain'; depth = number of directory components (depth-0 .. depth-4+). The basename never
    enters, so a renamed file keeps its shape. Matches PATH_SHAPE_RE."""
    if not str(path or "").strip():
        return "none"
    dirs = path_dirs(path)
    found = set(placeholder_kinds("/".join(dirs)))
    if any(d.startswith(".") and d not in (".", "..") for d in dirs):
        found.add("dot")
    return f"{_marks(found, PATH_MARKS)}/{DEPTH_WORDS[min(len(dirs), len(DEPTH_WORDS) - 1)]}"


CMD_SHAPES = _all_shapes(CMD_MARKS, CMD_LEN_WORDS)
PATH_SHAPES = _all_shapes(PATH_MARKS, DEPTH_WORDS)


# --------------------------------------------------------------------------- #
# Class words
# --------------------------------------------------------------------------- #
def canon_dir(word):
    """A dir class word clamped to DIR_ORDER ('' -> none, unknown text -> other)."""
    return _word(word, DIR_CLASSES, "none", "other")


def canon_endpoint(word):
    """An endpoint class word clamped to EP_ORDER ('' / unknown text -> unknown)."""
    return _word(word, ENDPOINTS, "unknown", "unknown")


def port_class(port):
    """well-known | registered | ephemeral | none (state.profile.port_class)."""
    return _port_class(port)


def ext_word(ext):
    """Extension -> EXT_WORDS (state.sections.ext_class on the extension alone; '' -> none)."""
    return _state_ext_class({"invariant": {"ext": str(ext or "").lower()}})


def ext_class(card):
    """Extension class from the INVARIANT only: a file's ext field; a process's image extension
    (executable, else file_name); 'none' for a netflow. Never reads behaviour (module docstring)."""
    t, inv = card["type"], card["invariant"]
    if t == "file":
        return ext_word(inv.get("ext"))
    if t == "process":
        return ext_word(ext_of(inv.get("executable") or inv.get("file_name") or ""))
    return "none"


# --------------------------------------------------------------------------- #
# Role table (C4)
# --------------------------------------------------------------------------- #
def catalogs_root(settings=None):
    """settings.paths.catalogs (scripts/fetch_catalogs.sh writes it)."""
    if settings is None:
        from habitus import config
        s = config.settings
        settings = s() if callable(s) and not hasattr(s, "paths") else s
    return os.fspath(settings.paths.catalogs)


class RoleTableError(RuntimeError):
    """The catalogs are missing or the role table built from them is not the frozen one."""


ROLE_CLASSES = ("shell", "interpreter", "exec-proxy", "transfer", "dual-use", "network", "admin", "devel",
                "system", "application", "unknown")
ROLE_RULES = ROLE_CLASSES[:-1]              # precedence: a smaller index wins
INTERPRETER_SECTIONS = frozenset(("interpreters", "python", "perl", "ruby", "php", "javascript", "lisp", "ocaml",
                                  "haskell", "java", "gnu-r", "cli-mono", "tcl"))
NETWORK_SECTIONS = frozenset(("net", "web", "mail", "comm", "httpd", "news"))
DEVEL_SECTIONS = frozenset(("devel", "libdevel", "vcs", "debug"))
SYSTEM_SECTIONS = frozenset(("utils", "libs", "oldlibs", "kernel", "otherosfs", "misc"))
GTFO_EXEC = frozenset(("shell", "command", "reverse-shell", "bind-shell"))
GTFO_TRANSFER = frozenset(("download", "upload"))
LOLBAS_EXEC = frozenset(("Execute", "AWL Bypass", "UAC Bypass", "Compile"))
LOLBAS_TRANSFER = frozenset(("Download", "Upload"))
DEB_PROGRAM = re.compile(r"^(?:usr/)?(?:s?bin|libexec|games)/[^/]+$")
WIN_SYSTEM = re.compile(r"^c:/windows/", re.IGNORECASE)
WIN_APP = re.compile(r"^c:/program files( \(x86\))?/", re.IGNORECASE)
POSIX_OS = ("linux", "freebsd", "android")
CATALOG_INPUTS = ("Contents-amd64.gz", "xcyclopedia.json.zip", "gtfobins/_gtfobins/*", "lolbas/yml/*/*.yml")
# sha256 of the canonical JSON of build_role_table() over the catalogs as found on 2026-10-21
# (36,982 posix + 2,144 Windows basenames).
ROLE_TABLE_SHA256 = "0494521304faa22625db72b195dbb0d15134204feff9e25c65a826d1f7c9741a"


def _catalog_files(root=None):
    """Sorted relative paths of every catalog file the role table reads."""
    root = catalogs_root() if root is None else os.fspath(root)
    out = []
    for pat in CATALOG_INPUTS:
        if "*" not in pat:
            if os.path.isfile(os.path.join(root, pat)):
                out.append(pat)
            continue
        d, tail = os.path.split(pat)
        if tail == "*":
            base = os.path.join(root, d)
            out += [os.path.join(d, n) for n in sorted(os.listdir(base)) if os.path.isfile(os.path.join(base, n))] \
                if os.path.isdir(base) else []
        else:                                            # lolbas/yml/<group>/<name>.yml
            top = os.path.join(root, os.path.dirname(d))
            if not os.path.isdir(top):
                continue
            for g in sorted(os.listdir(top)):
                gd = os.path.join(top, g)
                if os.path.isdir(gd):
                    out += [os.path.join(os.path.dirname(d), g, n) for n in sorted(os.listdir(gd))
                            if n.endswith(".yml")]
    return sorted(out)


def catalog_hashes(root=None):
    """{relative path: sha256} of every catalog file the role table reads."""
    root = catalogs_root() if root is None else os.fspath(root)
    out = {}
    for rel in _catalog_files(root):
        h = hashlib.sha256()
        with open(os.path.join(root, rel), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        out[rel] = h.hexdigest()
    return out


def _take(table, name, role):
    """Keep the higher-precedence role of a basename listed more than once."""
    name = name.strip().lower()
    if not name:
        return
    old = table.get(name)
    if old is None or ROLE_RULES.index(role) < ROLE_RULES.index(old):
        table[name] = role


def _section(loc):
    """Debian archive section of one 'area/section/package' or 'section/package' location."""
    parts = loc.split("/")
    return parts[-2] if len(parts) >= 2 else ""


def _deb_role(section):
    if section == "shells":
        return "shell"
    if section in INTERPRETER_SECTIONS:
        return "interpreter"
    if section in NETWORK_SECTIONS:
        return "network"
    if section == "admin":
        return "admin"
    if section in DEVEL_SECTIONS:
        return "devel"
    if section in SYSTEM_SECTIONS:
        return "system"
    return "application"


def _gtfo_role(functions):
    if functions & GTFO_EXEC:
        return "exec-proxy"
    if functions & GTFO_TRANSFER:
        return "transfer"
    return "dual-use" if functions else None


def _lolbas_role(categories):
    if categories & LOLBAS_EXEC:
        return "exec-proxy"
    if categories & LOLBAS_TRANSFER:
        return "transfer"
    return "dual-use" if categories else None


def _gtfobins(root):
    import yaml
    base = os.path.join(root, "gtfobins", "_gtfobins")
    raw = {}
    for name in sorted(os.listdir(base)):
        with open(os.path.join(base, name), encoding="utf-8") as f:
            meta = yaml.safe_load(f.read().split("---")[1]) or {}
        raw[name] = meta
    out = {}
    for name, meta in raw.items():
        if meta.get("alias"):                            # one hop: awk -> mawk, sh-like aliases
            meta = raw.get(meta["alias"], {})
        out[name] = frozenset((meta.get("functions") or {}).keys())
    return out


def _lolbas(root):
    import yaml
    out = {}
    for rel in _catalog_files(root):
        if not rel.startswith("lolbas/"):
            continue
        with open(os.path.join(root, rel), encoding="utf-8") as f:
            meta = yaml.safe_load(f) or {}
        name = str(meta.get("Name") or "").strip().lower()
        if name:
            cats = frozenset(str(c.get("Category")) for c in meta.get("Commands") or [] if c.get("Category"))
            out[name] = out.get(name, frozenset()) | cats
    return out


def build_role_table(root=None):
    """{"posix": {basename: role}, "windows": {basename: role}} from the catalogs (module docstring
    rules). Deterministic: sorted inputs, precedence-resolved duplicates."""
    root = catalogs_root() if root is None else os.fspath(root)
    posix, windows = {}, {}
    gtfo = _gtfobins(root)
    with gzip.open(os.path.join(root, "Contents-amd64.gz"), "rt", errors="ignore") as f:
        for line in f:
            parts = line.rsplit(None, 1)
            if len(parts) != 2 or not DEB_PROGRAM.match(parts[0]):
                continue
            secs = [_section(loc) for loc in parts[1].split(",")]
            role = min((_deb_role(s) for s in secs), key=ROLE_RULES.index)
            _take(posix, basename(parts[0]), role)
    for name, funcs in gtfo.items():
        role = _gtfo_role(funcs)
        if role is not None:
            _take(posix, name, role)                     # outranks every section but shells / interpreters
    for name, cats in _lolbas(root).items():
        role = _lolbas_role(cats)
        if role is not None:
            _take(windows, name, role)
    z = zipfile.ZipFile(os.path.join(root, "xcyclopedia.json.zip"))
    xc = json.loads(z.read("strontic-xcyclopedia.json").decode("utf-8-sig", "ignore"))
    for key in sorted(xc):
        v = xc[key]
        if not isinstance(v, dict) or not v.get("file_path"):
            continue
        p = str(v["file_path"]).replace("\\", "/")
        role = "system" if WIN_SYSTEM.match(p) else "application" if WIN_APP.match(p) else None
        if role is not None:
            _take(windows, basename(p), role)
    return {"posix": dict(sorted(posix.items())), "windows": dict(sorted(windows.items()))}


def role_table_sha256(table):
    return hashlib.sha256(json.dumps(table, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class _RoleTable(Mapping):
    """ROLE_TABLE: os family -> {lower-case basename: role}, built from settings.paths.catalogs on
    first access and checked against ROLE_TABLE_SHA256. Missing catalogs or a mismatch raise
    RoleTableError: a view rendered with other role words would not be the frozen runs' view.
    use_table() installs an explicit table instead (unit-test doubles only; status() reports it)."""

    def __init__(self):
        self._t, self._sha, self._source = None, None, None

    def _load(self):
        if self._t is None:
            root = catalogs_root()
            if not os.path.isfile(os.path.join(root, "Contents-amd64.gz")):
                raise RoleTableError(f"role-table catalogs missing under {root} (paths.catalogs): run "
                                     "scripts/fetch_catalogs.sh")
            built = build_role_table(root)
            sha = role_table_sha256(built)
            if sha != ROLE_TABLE_SHA256:
                raise RoleTableError(f"role table sha256 {sha} != frozen {ROLE_TABLE_SHA256}: a catalog or "
                                     "rule changed")
            self._install(built, sha, root)
        return self._t

    def _install(self, built, sha, source):
        t = {os_: built["posix"] for os_ in POSIX_OS}
        t["windows"] = built["windows"]
        self._t, self._sha, self._source = t, sha, source

    def use_table(self, built):
        """Install `built` = {"posix": {...}, "windows": {...}} without the catalogs (unit-test doubles;
        a frozen table matches only when its sha256 equals ROLE_TABLE_SHA256)."""
        self._install({"posix": dict(built.get("posix") or {}), "windows": dict(built.get("windows") or {})},
                      role_table_sha256(built), "explicit")

    def reset(self):
        self._t, self._sha, self._source = None, None, None

    def status(self):
        """{"available", "sha256", "frozen", "source", "sizes"} (builds the table if needed)."""
        t = self._load()
        return {"available": self._sha == ROLE_TABLE_SHA256, "sha256": self._sha, "frozen": ROLE_TABLE_SHA256,
                "source": self._source, "sizes": {"posix": len(t["linux"]), "windows": len(t["windows"])}}

    def __getitem__(self, os_):
        return self._load()[os_]

    def __iter__(self):
        return iter(self._load())

    def __len__(self):
        return len(self._load())


ROLE_TABLE = _RoleTable()


def program_name(card):
    """Lower-case basename a process's role is looked up by: file_name, else the executable's."""
    inv = card["invariant"]
    return str(inv.get("file_name") or basename(inv.get("executable") or "") or "").strip().lower()


def role_class(card, tier):
    """ROLE_CLASSES word of a process card, "unknown" unless `tier` (T.Attest) is identity-attested,
    and for every non-process card or name absent from the table."""
    if card["type"] != "process" or not T.Attest(tier).identity_attested:
        return "unknown"
    table = ROLE_TABLE.get(card.get("os") or "", None)
    if not table:
        return "unknown"
    return table.get(program_name(card), "unknown")


# --------------------------------------------------------------------------- #
# Entity words (the attested-name rule)
# --------------------------------------------------------------------------- #
TYPED_KEYS = {"process": ("type", "os", "image_dir", "ext_class", "role", "cmd_shape"),
              "file": ("type", "os", "dir", "ext_class", "path_shape"),
              "netflow": ("type", "os", "endpoint", "port_class")}
RAW_KEYS = {"process": ("file_name", "executable", "command_line"), "file": ("path",), "netflow": ()}
ENTITY_KEYS = {t: TYPED_KEYS[t] + RAW_KEYS[t] for t in T.NODE_TYPES}
RAW_CHARS = {"command_line": CMD_CHARS, "executable": EXE_CHARS, "path": PATH_CHARS}   # state section_entity


def attested_raw(tier):
    """The attested-name rule: raw strings may be rendered only for an identity-attested tier."""
    return T.Attest(tier).identity_attested


def entity_words(card, tier):
    """The habit-view entity section of one card (keys in ENTITY_KEYS order): typed words always,
    raw strings (state.sections section_entity truncation) only when `tier` is identity-attested and
    the field is non-empty. A netflow address is never rendered."""
    t, inv = card["type"], card["invariant"]
    os_ = card.get("os") or "unknown"
    if t == "process":
        out = {"type": t, "os": os_, "image_dir": canon_dir(inv.get("image_dir")),
               "ext_class": ext_class(card), "role": role_class(card, tier),
               "cmd_shape": cmd_shape(inv.get("command_line"))}
    elif t == "file":
        out = {"type": t, "os": os_, "dir": canon_dir(inv.get("dir")), "ext_class": ext_class(card),
               "path_shape": path_shape(inv.get("path"))}
    else:
        out = {"type": t, "os": os_, "endpoint": canon_endpoint(inv.get("endpoint")),
               "port_class": port_class(inv.get("remote_port"))}
    if attested_raw(tier):
        for k in RAW_KEYS[t]:
            v = str(inv.get(k) or "")
            if v:
                out[k] = v[:RAW_CHARS[k]] if k in RAW_CHARS else v
    return out


VOCAB_WORDS = frozenset(DIR_ORDER + EP_ORDER + PORT_ORDER + tuple(EXT_WORDS) + ROLE_CLASSES + T.NODE_TYPES
                        + OS_WORDS + FREQ_WORDS + CMD_SHAPES + PATH_SHAPES)


def typed_values(words):
    """The typed (non-raw) values of an entity_words dict, for check h / IFC tests."""
    t = words.get("type")
    return [words[k] for k in TYPED_KEYS.get(t, ()) if k in words]


__all__ = ["DIR_ORDER", "EP_ORDER", "PORT_ORDER", "EXT_WORDS", "ext_class", "ext_word", "port_class", "cmd_shape",
           "path_shape", "cmd_args", "CMD_SHAPE_RE", "PATH_SHAPE_RE", "CMD_SHAPES", "PATH_SHAPES", "ROLE_TABLE",
           "ROLE_CLASSES", "RoleTableError", "catalogs_root", "catalog_hashes", "build_role_table", "role_table_sha256", "role_class",
           "program_name", "entity_words", "ENTITY_KEYS", "TYPED_KEYS", "RAW_KEYS", "attested_raw", "canon_dir",
           "canon_endpoint", "VOCAB_WORDS", "typed_values"]
