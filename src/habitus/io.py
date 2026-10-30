"""JSON / JSON.gz helpers.

Writers are atomic (tmp file + os.replace) and deterministic (gzip header with mtime 0 and no file
name), and create the parent directory of the file they write. Path computation lives elsewhere and
never creates directories.

The JSON text is the same as the original writers produced, so the decompressed bytes match:
  dump_json      json.dump(obj, f)                          (reflex.common.dump_json: card / link files)
  write_json_gz  json.dump(obj, f, separators=(",", ":"))  (kcs1.score.write_json_gz: score files)
"""

import gzip
import hashlib
import io
import json
import os

def _tmp(path):
    return f"{path}.tmp{os.getpid()}"


def _mkparent(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _open_write_text(tmp, gz):
    """A text handle on `tmp`; gzip with a deterministic header when `gz`."""
    if not gz:
        return open(tmp, "w", encoding="utf-8")
    raw = open(tmp, "wb")
    g = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9)
    return _GzText(raw, g)


class _GzText(io.TextIOWrapper):
    """TextIOWrapper over a GzipFile that also closes the underlying raw file."""

    def __init__(self, raw, g):
        super().__init__(g, encoding="utf-8")
        self._raw_file = raw

    def close(self):
        try:
            super().close()
        finally:
            self._raw_file.close()


def _write(obj, path, **dump_kw):
    path = os.fspath(path)
    _mkparent(path)
    tmp = _tmp(path)
    try:
        with _open_write_text(tmp, path.endswith(".gz")) as f:
            json.dump(obj, f, **dump_kw)
        os.replace(tmp, path)          # a reader never sees a half-written file
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return path


def dump_json(obj, path):
    """Write `obj` as JSON (gzip when the name ends in .gz), default separators."""
    return _write(obj, path)


def load_json(path):
    path = os.fspath(path)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def write_json_gz(obj, path):
    """Write `obj` as compact JSON (separators ',' ':'); gzip when the name ends in .gz."""
    return _write(obj, path, separators=(",", ":"))


def read_json_gz(path):
    with gzip.open(os.fspath(path), "rt", encoding="utf-8") as f:
        return json.load(f)


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(os.fspath(path), "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def json_sha256(path):
    """sha256 of a JSON file's canonical form (sort_keys, compact): equal for equal content."""
    obj = load_json(path)
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
