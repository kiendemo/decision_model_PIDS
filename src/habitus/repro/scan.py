"""Static scanners behind gates G5 (import isolation, label hygiene, host allow-list) and G8
(paths, secrets, e-mail addresses).

    scan_paths(files)    absolute server paths: the home tree, the temp tree, the former data-disk
                         mount and the session scratch directory
    scan_secrets(files)  API-key shapes, bearer tokens, api-key assignments, env files
    scan_emails(files)   e-mail addresses (NOTICE / LICENSE excepted). Callers pass git-tracked files
                         and the NON-card release text only: card text legitimately holds e-mail-like
                         strings from the DARPA / OpTC / ATLASv2 logs
    scan_hosts(files)    names of hosts outside the 5-host allow-list
    scan_labels(src)     label operations outside habitus/eval/ (explicit file allow-list for mentions
                         that are not reads)
    import_graph(src)    AST import graph of the package; import_problems(graph) applies the G5 rules

Every pattern is char-classed (a literal such as "/h[o]me/") so that this file, and any file that
quotes a pattern, never matches itself. Each scanner returns a list of Hit; an empty list is clean.
The scanners also take plain text (scan_text / redact) for writers that scrub their own output.
"""

from __future__ import annotations

import ast
import gzip
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

MAX_TEXT_BYTES = 64 << 20          # text files above this are not decoded (reported as 'large')
MAX_TRACKED_BYTES = 95 * (1 << 20)  # G8: no tracked file over 95 MB


@dataclass(frozen=True)
class Hit:
    kind: str           # path | secret | email | host | label | import | file
    file: str           # path as given (repo-relative when the caller passes relative paths)
    line: int           # 1-based line, 0 = the whole file
    rule: str           # pattern / rule name
    match: str          # the matched text (truncated)

    def as_dict(self):
        return asdict(self)

    def __str__(self):
        return f"{self.kind}:{self.rule} {self.file}:{self.line}: {self.match}"


# --------------------------------------------------------------------------- #
# Patterns (char-classed: none of them matches its own source text)
# --------------------------------------------------------------------------- #
_LB = r"(?<![\w.~${}-])"            # not part of a longer (relative) path or a placeholder

# A server path is a user's home directory or a named temp directory (at least one directory level):
# short synthetic fixture paths of the test ports (a temp-dir payload, a 1-letter user home) are card
# content, not server paths. This machine's own home directory is matched literally (this_home).
PATH_PATTERNS = {
    "home": re.compile(_LB + r"/h[o]me/[\w.-]{3,}/"),
    "tmp": re.compile(_LB + r"/t[m]p/[\w.-]{3,}/"),
    "former_data_disk": re.compile(_LB + r"/D[A]TA[0-9]?(?![\w-])"),
    "scratch_dir": re.compile(r"scratch[p]ad", re.I),
}



def _this_home():
    """This machine's home directory as a literal pattern (computed at run time, never stored)."""
    home = os.path.expanduser("~").rstrip("/")
    if not home or home == "~" or home.count("/") < 2:
        return None
    return re.compile(re.escape(home) + r"(?=/|\b)")


_HOME_RX = _this_home()
if _HOME_RX is not None:
    PATH_PATTERNS["this_home"] = _HOME_RX

SECRET_PATTERNS = {
    "sk_key": re.compile(r"(?<![A-Za-z0-9])s[k]-[A-Za-z0-9_-]{20,}"),
    "openrouter": re.compile(r"OPEN[R]OUTER"),
    "bearer": re.compile(r"Bea[r]er "),
    "api_key_assign": re.compile(r"api[_-]?ke[y]\s*[=:]\s*['\"]?[A-Za-z0-9]", re.I),
    "hf_token": re.compile(r"(?<![A-Za-z0-9])h[f]_[A-Za-z0-9]{30,}"),
}
SECRET_FILE_NAMES = (re.compile(r"^\.en[v]"), re.compile(r"open[r]outer\.en[v]$", re.I))

EMAIL = re.compile(r"(?<![\w.+%-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?![\w-])")
EMAIL_ALLOWED_FILES = ("NOTICE", "LICENSE")

# Hosts outside the allow-list (CADETS_E3, THEIA_E3, CLEARSCOPE_E3, optc_h501, ATLASV2_EDR): the
# engagement-5 hosts, the other engagement-3 performers, the other OpTC hosts and the second ATLASv2
# family, in the spellings the source tree uses (dataset keys, DB names, GT dirs).
_TC = r"(?:THEIA|CADETS|CLEARSCOPE|TRACE|FIVEDIRECTIONS)"
OOS_HOST = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:"
    + _TC + r"[ _-]?E[5]"
    + r"|(?:TRACE|FIVEDIRECTIONS)[ _-]?E[3]"
    + r"|E[5][ _-]?" + _TC
    + r"|E[3][ _-]?(?:TRACE|FIVEDIRECTIONS)"
    + r"|CARBANA[K][ _-]?V2\w*"
    + r"|optc[ _-]?h?(?:0[5]1|2[0]1)"
    + r"|h(?:0[5]1|2[0]1)"
    + r"|ATLASV[2][ _-]?H[0-9]+\w*"
    + r")(?![A-Za-z0-9])")

LABEL_PATTERNS = {
    "gt_path": re.compile(r"ground[_]truth", re.I),
    "upstream_gt_key": re.compile(r"upstream[_]gt\b"),
    "gt_call": re.compile(r"(?<![\w])g[t]\s*\("),
    "gt_nodes": re.compile(r"\bg[t]_nodes\s*\("),
    "host_labels": re.compile(r"\bhost[_]labels\s*\("),
    "attack_groups": re.compile(r"\battack[_]groups\s*\("),
    "y_truth_var": re.compile(r"\by[_]truth\b"),
    "eval_labels_import": re.compile(r"habitus\.eval\.label[s]\b|from\s+habitus\.eval\s+import\s+[^\n]*\blabel[s]\b"),
}
# Files (relative to the scanned root's parent 'src/habitus' or the repo) whose mentions are not
# label reads (correction 12): the settings key, the HostSpec field, the guard's blocked patterns,
# the scanners' token lists, the eval wiring of the CLI, the GT copy of make_release.
LABEL_ALLOWED = ("config.py", "hosts.py", "guard.py", "repro/scan.py", "repro/gates.py", "cli.py",
                 "scripts/make_release.py",
                 # eval-side, server-only scripts (label readers by design, like habitus/eval/)
                 "scripts/export_baselines.py", "scripts/results/build_results.py")
LABEL_ALLOWED_DIRS = ("eval/",)

FORBIDDEN_IMPORTS = ("pidsmaker.reflex", "pidsmaker.defense", "pidsmaker.attacks", "pidsmaker.detection")
PIDSMAKER_MODULES = ("habitus.data.upstream",)
TORCH_MODULES = ("habitus.dm", "habitus.train", "habitus.data.graphs", "habitus.eval.baselines")
# torch-free entry points: the CPU replay ([replay] extra, no torch) imports these, so their EAGER import
# closure (module-level imports, package __init__ files included) must not reach torch
TORCH_FREE_ROOTS = ("habitus.cli", "habitus.repro.reproduce", "habitus.repro.gates", "habitus.dm",
                    "habitus.dm.jev", "habitus.dm.cache", "habitus.dm.base", "habitus.decide.routed",
                    "habitus.decide.inventory", "habitus.decide.scorefile", "habitus.state.builder",
                    "habitus.state.ladder", "habitus.questions.routed", "habitus.data.host", "habitus.eval.tables",
                    "habitus.eval.operating", "habitus.eval.bootstrap", "habitus.eval.metrics")
EVAL_IMPORTERS = ("habitus.eval", "habitus.repro", "habitus.cli")


# --------------------------------------------------------------------------- #
# Text access
# --------------------------------------------------------------------------- #
def read_text(path):
    """The text of a file, or None for binary / undecodable / very large files. A .gz file is
    decompressed (release metadata and golden fixtures are scanned through their content)."""
    p = Path(path)
    try:
        if p.stat().st_size > MAX_TEXT_BYTES:
            return None
        if p.suffix == ".gz":
            with gzip.open(p, "rb") as fh:
                raw = fh.read(MAX_TEXT_BYTES + 1)
        else:
            raw = p.read_bytes()
    except (OSError, EOFError):
        return None
    if b"\x00" in raw[:8192]:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _rel(path, root=None):
    p = Path(path)
    if root is not None:
        try:
            return p.resolve().relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            pass
    return p.as_posix()


def scan_text(text, patterns, *, kind, file="<text>"):
    """Hits of `patterns` ({name: compiled regex} or one regex) in `text`, line by line."""
    if isinstance(patterns, re.Pattern):
        patterns = {kind: patterns}
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        for name, rx in patterns.items():
            for m in rx.finditer(line):
                hits.append(Hit(kind, str(file), i, name, m.group(0)[:120]))
    return hits


def redact(text, patterns, repl):
    """`text` with every match of `patterns` ({name: regex} or one regex) replaced by `repl`."""
    if isinstance(patterns, re.Pattern):
        patterns = {"x": patterns}
    for rx in patterns.values():
        text = rx.sub(repl, text)
    return text


def hosts_in(text):
    """Distinct out-of-scope host names found in `text` (as written)."""
    return sorted({m.group(0) for m in OOS_HOST.finditer(text)})


def redact_hosts(text, repl="[out-of-scope host]"):
    return OOS_HOST.sub(repl, text)


def _scan_files(files, patterns, kind, root=None, skip=()):
    hits = []
    for f in files:
        rel = _rel(f, root)
        if rel in skip or Path(rel).name in skip:
            continue
        text = read_text(f)
        if text is None:
            continue
        hits += scan_text(text, patterns, kind=kind, file=rel)
    return hits


# --------------------------------------------------------------------------- #
# The scanners
# --------------------------------------------------------------------------- #
def scan_paths(files, *, root=None, allow=()):
    """Absolute server paths in `files` (G8). `allow`: relative paths / names to skip."""
    return _scan_files(files, PATH_PATTERNS, "path", root, allow)


def scan_secrets(files, *, root=None, allow=(), max_bytes=MAX_TRACKED_BYTES):
    """Key shapes in the text of `files`, plus env files and files over `max_bytes` (G8)."""
    hits = _scan_files(files, SECRET_PATTERNS, "secret", root, allow)
    for f in files:
        rel = _rel(f, root)
        name = Path(rel).name
        if any(rx.search(name) for rx in SECRET_FILE_NAMES):
            hits.append(Hit("secret", rel, 0, "env_file", name))
        try:
            size = Path(f).stat().st_size
        except OSError:
            continue
        if max_bytes is not None and size > max_bytes:
            hits.append(Hit("file", rel, 0, "too_large", f"{size} bytes"))
    return hits


def scan_emails(files, *, root=None, allow=EMAIL_ALLOWED_FILES):
    """E-mail addresses in `files` except NOTICE / LICENSE (G8). Never pass card / link / answer /
    score data: their text comes from the audit logs (correction 11)."""
    return _scan_files(files, EMAIL, "email", root, allow)


def scan_hosts(files, *, root=None, allow=()):
    """Out-of-scope host names in `files` (G5d)."""
    return _scan_files(files, OOS_HOST, "host", root, allow)


def _label_allowed(rel):
    rel = rel.replace(os.sep, "/")
    for d in LABEL_ALLOWED_DIRS:
        if rel.startswith(d) or f"/{d}" in rel:
            return True
    return any(rel == a or rel.endswith("/" + a) for a in LABEL_ALLOWED)


def scan_labels(src_root, *, extra_files=(), repo_root=None):
    """Label operations (GT path reads, gt / label calls, label imports) in every .py file under
    `src_root` (the habitus package dir) and in `extra_files` (e.g. scripts/*.py), outside
    habitus/eval/ and the explicit allow-list LABEL_ALLOWED (G5b)."""
    src_root = Path(src_root)
    files = sorted(src_root.rglob("*.py"))
    hits = []
    for f in files:
        rel = f.relative_to(src_root).as_posix()
        if _label_allowed(rel):
            continue
        text = read_text(f)
        if text is not None:
            hits += scan_text(text, LABEL_PATTERNS, kind="label", file=f"{src_root.name}/{rel}")
    for f in extra_files:
        rel = _rel(f, repo_root)
        if _label_allowed(rel):
            continue
        text = read_text(f)
        if text is not None:
            hits += scan_text(text, LABEL_PATTERNS, kind="label", file=rel)
    return hits


# --------------------------------------------------------------------------- #
# Import graph
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Import:
    module: str         # importing module (habitus.x.y)
    target: str         # imported module (absolute)
    line: int
    lazy: bool          # inside a function body (not executed at import time)
    dynamic: bool = False   # importlib.import_module / __import__ with a literal name


def _module_name(path, src_root):
    rel = Path(path).relative_to(Path(src_root).parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve(module, is_pkg, level, name):
    if level == 0:
        return name or ""
    base = module.split(".")
    if not is_pkg:
        base = base[:-1]
    if level > 1:
        base = base[: len(base) - (level - 1)]
    return ".".join([*base, name] if name else base)


class _Visitor(ast.NodeVisitor):
    def __init__(self, module, is_pkg):
        self.module, self.is_pkg, self.depth, self.out = module, is_pkg, 0, []

    def _fn(self, node):
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1

    visit_FunctionDef = visit_AsyncFunctionDef = visit_Lambda = _fn

    def visit_Import(self, node):
        for a in node.names:
            self.out.append(Import(self.module, a.name, node.lineno, self.depth > 0))

    def visit_ImportFrom(self, node):
        base = _resolve(self.module, self.is_pkg, node.level, node.module)
        for a in node.names:
            # "from pkg import mod" may import a submodule: record both spellings' longest form
            self.out.append(Import(self.module, f"{base}.{a.name}" if a.name != "*" else base, node.lineno,
                                   self.depth > 0))

    def visit_Call(self, node):
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
        if name in ("import_module", "__import__") and node.args and isinstance(node.args[0], ast.Constant) \
                and isinstance(node.args[0].value, str):
            self.out.append(Import(self.module, node.args[0].value, node.lineno, self.depth > 0, True))
        self.generic_visit(node)


def import_graph(src_root):
    """{module: [Import]} for every .py file under `src_root` (the habitus package directory)."""
    src_root = Path(src_root)
    graph = {}
    for f in sorted(src_root.rglob("*.py")):
        mod = _module_name(f, src_root)
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        except SyntaxError as e:
            graph[mod] = [Import(mod, f"<syntax error: {e.msg}>", e.lineno or 0, False)]
            continue
        v = _Visitor(mod, f.name == "__init__.py")
        v.visit(tree)
        graph[mod] = v.out
    return graph


def _in(mod, prefixes):
    return any(mod == p or mod.startswith(p + ".") for p in prefixes)


def _owner(target, modules):
    """The module an import target names: itself, or its longest prefix that is a module
    ("habitus.dm.get_dm" -> "habitus.dm")."""
    parts = target.split(".")
    for n in range(len(parts), 0, -1):
        m = ".".join(parts[:n])
        if m in modules:
            return m
    return None


def eager_closure(graph, root):
    """{module: parent} of every habitus module imported at import time of `root` (module-level imports
    and the package __init__ chain), for the chain report."""
    seen, todo = {}, [(root, None)]
    while todo:
        m, parent = todo.pop()
        if m in seen or m not in graph:
            continue
        seen[m] = parent
        parts = m.split(".")
        todo += [(".".join(parts[:n]), m) for n in range(1, len(parts))]
        for imp in graph[m]:
            if not imp.lazy:
                o = _owner(imp.target, graph)
                if o is not None:
                    todo.append((o, m))
    return seen


def _chain(closure, m):
    out = [m]
    while closure.get(out[-1]):
        out.append(closure[out[-1]])
    return " <- ".join(out)


def import_problems(graph, torch_free_roots=TORCH_FREE_ROOTS):
    """The G5a rules over an import_graph():
    - no import of pidsmaker.reflex / defense / attacks / detection anywhere;
    - pidsmaker imported only lazily (inside functions) and only in habitus.data.upstream;
    - torch imported only in habitus.dm / train / data.graphs / eval.baselines, and never at import
      time of a torch-free entry point (TORCH_FREE_ROOTS: the CPU replay runs without torch);
    - habitus.eval imported only by habitus.eval, habitus.repro and habitus.cli (so never by
      habitus.decide or habitus.dm)."""
    hits = []
    reported = set()
    for root in torch_free_roots:
        if root not in graph:
            continue
        clo = eager_closure(graph, root)
        for m in clo:
            for imp in graph[m]:
                if not imp.lazy and _in(imp.target, ("torch",)) and (m, imp.line) not in reported:
                    reported.add((m, imp.line))
                    hits.append(Hit("import", m.replace(".", "/") + ".py", imp.line, "torch_at_import_time",
                                    f"{imp.target} via {_chain(clo, m)}"))
    for mod, imps in sorted(graph.items()):
        f = mod.replace(".", "/") + ".py"
        for imp in imps:
            t = imp.target
            if t.startswith("<syntax error"):
                hits.append(Hit("import", f, imp.line, "syntax", t))
                continue
            if _in(t, FORBIDDEN_IMPORTS):
                hits.append(Hit("import", f, imp.line, "forbidden_upstream", t))
            elif _in(t, ("pidsmaker",)) and (not imp.lazy or not _in(mod, PIDSMAKER_MODULES)):
                hits.append(Hit("import", f, imp.line, "pidsmaker_outside_upstream", t))
            if _in(t, ("torch",)) and not _in(mod, TORCH_MODULES):
                hits.append(Hit("import", f, imp.line, "torch_outside_dm_train", t))
            if _in(t, ("habitus.eval",)) and not _in(mod, EVAL_IMPORTERS):
                hits.append(Hit("import", f, imp.line, "eval_import", t))
    return hits


# --------------------------------------------------------------------------- #
# File sets
# --------------------------------------------------------------------------- #
def tracked_files(repo_root):
    """Files git tracks or would track (cached + untracked not ignored), as absolute Paths. Before
    the first commit this is exactly what a commit of the working tree would contain."""
    repo_root = Path(repo_root)
    out = subprocess.run(["git", "-C", str(repo_root), "ls-files", "-z", "--cached", "--others",
                          "--exclude-standard"], capture_output=True, check=True)
    names = sorted({n for n in out.stdout.decode("utf-8").split("\0") if n})
    return [repo_root / n for n in names if (repo_root / n).is_file()]


def under(files, repo_root, tops):
    """The subset of `files` whose repo-relative path starts with one of `tops` (a dir with a
    trailing '/' or an exact file name)."""
    repo_root = Path(repo_root).resolve()
    keep = []
    for f in files:
        rel = _rel(f, repo_root)
        if any(rel == t or (t.endswith("/") and rel.startswith(t)) for t in tops):
            keep.append(f)
    return keep
