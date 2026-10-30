"""Comparators of `habitus reproduce` and the gates: every number either matches or is reported.

    Check / Report           one PASS/FAIL line per compared quantity; Report.table() prints them
    compare_csv_bytes        byte identity of a CSV (line endings and quoting included), except the
                             declared columns (G1: `note`)
    compare_csv_numeric      integers exact, floats within a tolerance (G9: repr-written floats, 1e-12)
    compare_text             exact text, except declared regex spans (the md "Generated <time>" stamp)
    compare_json             ints / strings / None exact, floats within a tolerance, recursively
    compare_score_file       one recomputed score file vs the shipped one, per level, over the keys
                             present in the shipped file (G4b)

The CSV tokenizer below follows the csv module's default (excel) dialect: ',' delimiter, '"' quote
with doubling, records ended by CRLF, LF or CR; fields are compared as raw bytes.
"""

from __future__ import annotations

import csv
import io
import math
import re
from dataclasses import asdict, dataclass, field

E1_LEVELS = ("L0", "L1", "L2", "L3", "L4")
E1_KEYS = ("qv", "qt", "cov_min")
META_IGNORE = ("seconds", "written", "e2.spec")
MAX_DETAIL = 20


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    got: object = None
    want: object = None

    @property
    def status(self):
        return "PASS" if self.ok else "FAIL"

    def as_dict(self):
        d = asdict(self)
        d["status"] = self.status
        return d


@dataclass
class Report:
    title: str = "habitus"
    checks: list = field(default_factory=list)

    def add(self, name, ok, detail="", got=None, want=None):
        c = Check(name, bool(ok), detail, got, want)
        self.checks.append(c)
        return c

    def extend(self, checks, prefix=""):
        for c in checks:
            if prefix:
                c.name = f"{prefix}{c.name}"
            self.checks.append(c)
        return self

    @property
    def ok(self):
        return bool(self.checks) and all(c.ok for c in self.checks)

    @property
    def n_fail(self):
        return sum(1 for c in self.checks if not c.ok)

    def table(self):
        w = max([len(c.name) for c in self.checks] + [5])
        lines = [f"{'check':<{w}}  status  detail"]
        for c in self.checks:
            d = c.detail
            if c.got is not None or c.want is not None:
                d = (d + " " if d else "") + f"got {c.got!r} want {c.want!r}"
            lines.append(f"{c.name:<{w}}  {c.status:<6}  {d}")
        lines.append(f"{self.title}: {len(self.checks) - self.n_fail} PASS, {self.n_fail} FAIL")
        return "\n".join(lines)

    def as_dict(self):
        return {"title": self.title, "ok": self.ok, "n_fail": self.n_fail,
                "checks": [c.as_dict() for c in self.checks]}


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
def csv_records(data: bytes):
    """[(raw fields, terminator)] of a CSV byte string; a quoted field keeps its quotes."""
    out, i, n = [], 0, len(data)
    while i < n:
        fields = []
        while True:
            if data[i:i + 1] == b'"':
                j = i + 1
                while True:
                    q = data.find(b'"', j)
                    if q < 0:
                        raise ValueError(f"unterminated quoted field at byte {i}")
                    if data[q + 1:q + 2] == b'"':
                        j = q + 2
                        continue
                    break
                fields.append(data[i:q + 1])
                i = q + 1
            else:
                j = i
                while j < n and data[j] not in b',\r\n':
                    j += 1
                fields.append(data[i:j])
                i = j
            if data[i:i + 1] == b",":
                i += 1
                continue
            if data[i:i + 2] == b"\r\n":
                term, i = b"\r\n", i + 2
            elif data[i:i + 1] in (b"\n", b"\r"):
                term, i = data[i:i + 1], i + 1
            elif i >= n:
                term = b""
            else:
                raise ValueError(f"malformed CSV at byte {i}: {data[i:i + 20]!r}")
            break
        out.append((fields, term))
    return out


def _unquote(raw: bytes) -> str:
    s = raw.decode("utf-8")
    if s.startswith('"') and s.endswith('"') and len(s) >= 2:
        s = s[1:-1].replace('""', '"')
    return s


def compare_csv_bytes(new: bytes, ref: bytes, *, ignore_cols=(), name="csv"):
    """Byte identity of two CSVs except the columns in `ignore_cols` (by header name): header,
    record count, every record terminator and every other raw field must be equal."""
    if new == ref:
        return [Check(f"{name}: bytes", True, f"{len(ref)} bytes identical")]
    if not ignore_cols:
        return [Check(f"{name}: bytes", False, _first_diff(new, ref))]
    try:
        a, b = csv_records(new), csv_records(ref)
    except (ValueError, UnicodeDecodeError) as e:
        return [Check(f"{name}: parse", False, str(e))]
    checks = []
    if not b:
        return [Check(f"{name}: header", False, "reference is empty")]
    header = [_unquote(x) for x in b[0][0]]
    skip = {i for i, h in enumerate(header) if h in set(ignore_cols)}
    same = bool(a) and a[0][0] == b[0][0]
    checks.append(Check(f"{name}: header", same, "", None if same else ([_unquote(x) for x in a[0][0]] if a else None),
                        None if same else header))
    checks.append(Check(f"{name}: records", len(a) == len(b), "", len(a), len(b)))
    terms = sum(1 for (_, ta), (_, tb) in zip(a, b) if ta != tb)
    checks.append(Check(f"{name}: line endings", terms == 0 and len(a) == len(b),
                        f"{terms} records end differently" if terms else
                        f"all {repr(b[0][1])[2:-1] or 'EOF'}-terminated as the reference"))
    bad = []
    for r, ((fa, _), (fb, _)) in enumerate(zip(a, b)):
        if len(fa) != len(fb):
            bad.append(f"record {r}: {len(fa)} fields vs {len(fb)}")
            continue
        for c, (x, y) in enumerate(zip(fa, fb)):
            if c not in skip and x != y:
                col = header[c] if c < len(header) else c
                bad.append(f"record {r} col {col}: {x.decode('utf-8', 'replace')!r} vs {y.decode('utf-8', 'replace')!r}")
    checks.append(Check(f"{name}: fields", not bad,
                        (f"{len(bad)} differ: " + "; ".join(bad[:MAX_DETAIL])) if bad else
                        f"every field equal except {sorted(ignore_cols)}"))
    return checks


def _first_diff(a: bytes, b: bytes):
    n = min(len(a), len(b))
    i = next((j for j in range(n) if a[j] != b[j]), n)
    return f"{len(a)} vs {len(b)} bytes; first difference at byte {i}: {a[i:i + 40]!r} vs {b[i:i + 40]!r}"


def _num(s):
    try:
        return int(s), "int"
    except ValueError:
        pass
    try:
        return float(s), "float"
    except ValueError:
        return s, "str"


def compare_csv_numeric(new, ref, *, tol=1e-12, ignore_cols=(), name="csv"):
    """Cell-wise CSV comparison: integers exact, floats |delta| <= tol (nan == nan), other text
    exact; same header, same row order. `new` / `ref`: bytes or str."""
    def rows(x):
        if isinstance(x, bytes):
            x = x.decode("utf-8")
        return list(csv.reader(io.StringIO(x, newline="")))
    try:
        a, b = rows(new), rows(ref)
    except (csv.Error, UnicodeDecodeError) as e:
        return [Check(f"{name}: parse", False, str(e))]
    if not b:
        return [Check(f"{name}: header", False, "reference is empty")]
    same = bool(a) and a[0] == b[0]
    checks = [Check(f"{name}: header", same, "", None if same else (a[0] if a else None), None if same else b[0]),
              Check(f"{name}: rows", len(a) == len(b), "", len(a) - 1, len(b) - 1)]
    header = b[0]
    skip = {i for i, h in enumerate(header) if h in set(ignore_cols)}
    bad, worst = [], 0.0
    for r, (ra, rb) in enumerate(zip(a[1:], b[1:]), 1):
        if len(ra) != len(rb):
            bad.append(f"row {r}: {len(ra)} cells vs {len(rb)}")
            continue
        for c, (x, y) in enumerate(zip(ra, rb)):
            if c in skip or x == y:
                continue
            (vx, tx), (vy, ty) = _num(x), _num(y)
            col = header[c] if c < len(header) else c
            if tx == ty == "int":
                bad.append(f"row {r} {col}: {x} vs {y}")
            elif "str" in (tx, ty):
                bad.append(f"row {r} {col}: {x!r} vs {y!r}")
            else:
                if math.isnan(vx) and math.isnan(vy):
                    continue
                d = abs(vx - vy) if math.isfinite(vx) and math.isfinite(vy) else (0.0 if vx == vy else math.inf)
                worst = max(worst, d)
                if d > tol:
                    bad.append(f"row {r} {col}: {x} vs {y} (|d| {d:.3g})")
    checks.append(Check(f"{name}: cells", not bad,
                        (f"{len(bad)} differ: " + "; ".join(bad[:MAX_DETAIL])) if bad else
                        f"ints exact, floats max |d| {worst:.3g} <= {tol:g}"))
    return checks


# --------------------------------------------------------------------------- #
# Text / JSON
# --------------------------------------------------------------------------- #
GENERATED_STAMP = re.compile(r"Generated \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")


def compare_text(new: str, ref: str, *, ignore=(), name="text"):
    """Exact text equality after replacing the spans of the `ignore` regexes by a placeholder in both."""
    a, b = new, ref
    for rx in ignore:
        rx = re.compile(rx) if isinstance(rx, str) else rx
        a, b = rx.sub("<ignored>", a), rx.sub("<ignored>", b)
    if a == b:
        return [Check(f"{name}: text", True, f"{len(b.splitlines())} lines identical"
                      + (f" (declared exceptions: {len(ignore)} pattern(s))" if ignore else ""))]
    la, lb = a.splitlines(), b.splitlines()
    i = next((j for j in range(min(len(la), len(lb))) if la[j] != lb[j]), min(len(la), len(lb)))
    return [Check(f"{name}: text", False, f"{len(la)} vs {len(lb)} lines; first difference at line {i + 1}: "
                                         f"{(la[i] if i < len(la) else '<eof>')[:120]!r} vs "
                                         f"{(lb[i] if i < len(lb) else '<eof>')[:120]!r}")]


def json_diffs(new, ref, *, tol=0.0, path="", ignore=(), out=None, limit=200):
    """Differences between two JSON values: ints / bools / strings / None exact, a float on either
    side within `tol` (nan == nan), lists element-wise, dicts by key. `ignore`: dotted key paths."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if path in ignore:
        return out
    if isinstance(ref, dict):
        if not isinstance(new, dict):
            out.append(f"{path or '<root>'}: {type(new).__name__} vs dict")
            return out
        for k in sorted(set(ref) | set(new), key=str):
            p = f"{path}.{k}" if path else str(k)
            if p in ignore:
                continue
            if k not in new:
                out.append(f"{p}: missing")
            elif k not in ref:
                out.append(f"{p}: extra")
            else:
                json_diffs(new[k], ref[k], tol=tol, path=p, ignore=ignore, out=out, limit=limit)
        return out
    if isinstance(ref, (list, tuple)):
        if not isinstance(new, (list, tuple)) or len(new) != len(ref):
            out.append(f"{path}: list of {len(new) if isinstance(new, (list, tuple)) else type(new).__name__} "
                       f"vs {len(ref)}")
            return out
        for i, (x, y) in enumerate(zip(new, ref)):
            json_diffs(x, y, tol=tol, path=f"{path}[{i}]", ignore=ignore, out=out, limit=limit)
        return out
    if isinstance(ref, bool) or isinstance(new, bool) or ref is None or new is None or isinstance(ref, str):
        if new != ref or type(new) is not type(ref):
            out.append(f"{path}: {new!r} vs {ref!r}")
        return out
    if isinstance(ref, int) and isinstance(new, int):
        if new != ref:
            out.append(f"{path}: {new} vs {ref}")
        return out
    if isinstance(ref, (int, float)) and isinstance(new, (int, float)):
        x, y = float(new), float(ref)
        if math.isnan(x) and math.isnan(y):
            return out
        d = abs(x - y) if math.isfinite(x) and math.isfinite(y) else (0.0 if x == y else math.inf)
        if d > tol:
            out.append(f"{path}: {new!r} vs {ref!r} (|d| {d:.3g})")
        return out
    if new != ref:
        out.append(f"{path}: {new!r} vs {ref!r}")
    return out


def compare_json(new, ref, *, tol=1e-12, ignore=(), name="json"):
    d = json_diffs(new, ref, tol=tol, ignore=tuple(ignore))
    return [Check(f"{name}: values", not d, (f"{len(d)} differ: " + "; ".join(d[:MAX_DETAIL])) if d else
                  f"equal (ints exact, floats |d| <= {tol:g})")]


# --------------------------------------------------------------------------- #
# Score files
# --------------------------------------------------------------------------- #
def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def compare_score_file(new: dict, ref: dict, *, name=None, meta_ignore=META_IGNORE):
    """One recomputed score file vs the shipped one (G4b). Returns (checks, max |delta|).
    - head and fallback equal (fallback floats exact);
    - the same signatures; per record only the keys of the SHIPPED record are compared and no other
      key may be written (E1: exactly qv, qt, cov_min); numbers |delta| == 0.0, bool / None equal;
    - meta: every key the shipped meta has, except `meta_ignore` (dotted paths), equal."""
    rm, nm = ref.get("meta", {}), new.get("meta", {})
    level = rm.get("level")
    name = name or f"{rm.get('ds')}/{level}_k{rm.get('k')}/{rm.get('day')}"
    checks = [Check(f"{name}: head", new.get("head") == ref.get("head")),
              Check(f"{name}: fallback", new.get("fallback") == ref.get("fallback"),
                    "" if new.get("fallback") == ref.get("fallback") else
                    f"{len(new.get('fallback') or {})} vs {len(ref.get('fallback') or {})} entries")]
    rs, ns = ref.get("scores", {}), new.get("scores", {})
    missing, extra = sorted(set(rs) - set(ns)), sorted(set(ns) - set(rs))
    checks.append(Check(f"{name}: signatures", not missing and not extra,
                        f"{len(rs)} signatures" if not missing and not extra else
                        f"{len(missing)} missing, {len(extra)} extra (e.g. {(missing or extra)[:3]})"))
    worst, bad, keyerr = 0.0, [], []
    e1 = level in E1_LEVELS
    for s, rr in rs.items():
        nr = ns.get(s)
        if nr is None:
            continue
        if isinstance(rr, dict):
            want_keys = set(rr)
            if e1 and want_keys - set(E1_KEYS):
                keyerr.append(f"{s}: shipped E1 keys {sorted(want_keys)}")
            if set(nr) != want_keys:
                keyerr.append(f"{s}: keys {sorted(nr)} vs {sorted(want_keys)}")
            items = [(k, nr.get(k), rr[k]) for k in rr]
        else:
            items = [("value", nr, rr)]
        for k, x, y in items:
            if _is_num(x) and _is_num(y):
                d = abs(float(x) - float(y))
                if math.isnan(d):
                    d = 0.0 if (math.isnan(float(x)) and math.isnan(float(y))) else math.inf
                worst = max(worst, d)
                if d != 0.0:
                    bad.append(f"{s}.{k}: {x!r} vs {y!r}")
            elif x != y or type(x) is not type(y):
                bad.append(f"{s}.{k}: {x!r} vs {y!r}")
    checks.append(Check(f"{name}: record keys", not keyerr,
                        ("; ".join(keyerr[:MAX_DETAIL])) if keyerr else
                        ("E1 records {qv, qt, cov_min}" if e1 else "shipped record keys")))
    checks.append(Check(f"{name}: records", not bad, (f"{len(bad)} differ, max |d| {worst:.3g}: "
                                                       + "; ".join(bad[:MAX_DETAIL])) if bad else
                        f"max |d| {worst!r}"))
    mign = tuple(meta_ignore)
    d = []
    for k in rm:
        if k in mign:
            continue
        if k not in nm:
            d.append(f"meta.{k}: missing")
            continue
        sub = tuple(p[len(k) + 1:] for p in mign if p.startswith(k + "."))
        d += [f"meta.{k}" + ("" if x.startswith(("[", ":")) else ".") + x
              for x in json_diffs(nm[k], rm[k], tol=0.0, ignore=sub)]
    checks.append(Check(f"{name}: meta", not d, "; ".join(d[:MAX_DETAIL]) if d else
                        f"every shipped key equal except {sorted(mign)}"))
    return checks, worst
