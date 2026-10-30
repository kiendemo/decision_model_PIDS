"""AnswerCache: first record wins, bad lines skipped, read-only base without a lock, flocked overlay."""

import gzip
import json
import os

import pytest

from habitus.dm.cache import AnswerCache, cache_paths, finite_record


def _rec(k, z0=1.0, cov=0.9):
    return {"k": k, "z": [z0, 0.5, 0.25, -1.0], "lse": 3.0, "n": 2, "cov": cov}


def _write(path, lines, gz=False):
    data = "".join((x if isinstance(x, str) else json.dumps(x)) + "\n" for x in lines)
    if gz:
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(data)
    else:
        path.write_text(data, encoding="utf-8")


def test_first_record_wins_and_bad_lines_are_skipped(tmp_path):
    base = tmp_path / "X.jsonl.gz"
    _write(base, [_rec("a", 1.0), "not json", '{"k":"nan","z":[NaN,0,0,0],"lse":1.0,"n":4,"cov":0.5}',
                  _rec("a", 9.0), {"z": [0, 0, 0, 0], "lse": 1, "n": 2, "cov": 1}, '"a string"', _rec("b", 2.0)],
           gz=True)
    over = tmp_path / "runs" / "answers" / "t" / "X.jsonl"
    over.parent.mkdir(parents=True)
    _write(over, [_rec("a", 7.0), _rec("c", 3.0)])
    c = AnswerCache("t", "X", base=base, overlay=over, read_only=True)
    assert c.get("a")["z"][0] == 1.0 and c.get("b")["z"][0] == 2.0 and c.get("c")["z"][0] == 3.0
    assert c.get("nan") is None and "k" not in c.get("a") and len(c) == 3
    assert c.stats["dup"] == 2 and c.stats["bad"] == 4
    with pytest.raises(PermissionError):
        c.put("d", _rec("d"))
    c.close()
    assert sorted(os.listdir(tmp_path)) == ["X.jsonl.gz", "runs"]                 # no lock next to the base
    assert sorted(os.listdir(over.parent)) == ["X.jsonl"]                          # a reader creates no lock


def test_overlay_put_flush_lock_and_reload(tmp_path):
    base = tmp_path / "base.jsonl"
    _write(base, [_rec("a", 1.0)])
    over = tmp_path / "ov" / "X.jsonl"
    c = AnswerCache("t", "X", base=base, overlay=over)
    assert not over.parent.exists()                                               # nothing created on open
    kept = c.put("a", {"z": [5.0, 0, 0, 0], "lse": 1.0, "n": 2, "cov": 0.5})
    assert kept["z"][0] == 1.0 and c.pending == 0                                 # the existing record wins
    c.put("b", {"z": [2.0, 1.0, 0.5, -1.0], "lse": 9.0, "n": 4, "cov": 0.25, "extra": 1})
    assert (over.parent / "X.jsonl.lock").exists() and c.pending == 1
    with pytest.raises(RuntimeError, match="held by another writer"):
        AnswerCache("t", "X", overlay=over).put("z", {"z": [0, 0, 0, 0], "lse": 0, "n": 2, "cov": 1})
    with pytest.raises(ValueError):
        c.put("bad", {"z": [float("nan"), 0, 0, 0], "lse": 0, "n": 2, "cov": 1})
    c.flush()
    lines = over.read_text(encoding="utf-8").splitlines()
    assert lines == ['{"k":"b","z":[2.0,1.0,0.5,-1.0],"lse":9.0,"n":4,"cov":0.25}']
    with open(over, "a", encoding="utf-8") as f:                                  # a crashed writer's partial line
        f.write('{"k":"c","z":[1')
    c.put("d", {"z": [1.0, 2.0, 3.0, 4.0], "lse": 5.0, "n": 3, "cov": 0.1})
    c.close()
    raw = over.read_text(encoding="utf-8")
    assert raw.endswith('{"k":"d","z":[1.0,2.0,3.0,4.0],"lse":5.0,"n":3,"cov":0.1}\n') and '[1\n{"k":"d"' in raw
    again = AnswerCache("t", "X", base=base, overlay=over, read_only=True)
    assert set(again._load()) == {"a", "b", "d"} and again.stats["bad"] == 1
    w = AnswerCache("t", "X", overlay=over)                                       # the lock was released
    w.put("e", {"z": [0, 0, 0, 0], "lse": 0.0, "n": 2, "cov": 1.0})
    w.close()


def test_in_memory_cache_and_paths(tmp_path):
    c = AnswerCache("t", None)
    c.put("a", {"z": [0, 0, 0, 0], "lse": 0.0, "n": 2, "cov": 1.0})
    c.flush()
    c.close()
    assert c.get("a")["n"] == 2 and list(tmp_path.iterdir()) == []
    assert finite_record({"z": [1, 2], "lse": 0, "cov": 1})
    assert not finite_record({"z": [1], "lse": float("inf"), "cov": 1})

    class P:
        answers = str(tmp_path / "answers")
        runs = str(tmp_path / "runs")

    class S:
        paths = P

    base, over = cache_paths("qwen3.5-4b", "X", S)
    assert base is None and str(over) == str(tmp_path / "runs" / "answers" / "qwen3.5-4b" / "X.jsonl")
    (tmp_path / "answers" / "qwen3.5-4b").mkdir(parents=True)
    _write(tmp_path / "answers" / "qwen3.5-4b" / "X.jsonl.gz", [_rec("a")], gz=True)
    assert cache_paths("qwen3.5-4b", "X", S)[0].name == "X.jsonl.gz"
    assert not (tmp_path / "runs").exists()


def test_original_laya_record_is_accepted(tmp_path, capsys):
    """An original-format Laya line {k, z, n} (encoder.py _Member.run) loads, completed as
    dm.laya.cache_record would write it; a jev-style record missing only one field stays bad."""
    import math
    base = tmp_path / "hbl-x.jsonl"
    _write(base, ['{"k":"abc","z":[1.23456,-0.5],"n":2}', '{"k":"short","z":[1.0,2.0],"n":3}',
                  '{"k":"nolse","z":[1.0,2.0,0.0,0.0],"n":2,"cov":0.5}'])
    c = AnswerCache("hbl-x", "CADETS_E3", base=base, read_only=True)
    rec = c.get("abc")
    want = round(max(1.23456, -0.5) + math.log(math.exp(0.0) + math.exp(-0.5 - 1.23456)), 5)
    assert rec == {"z": [1.23456, -0.5], "n": 2, "lse": want, "cov": 1.0}
    assert c.get("short") is None and c.get("nolse") is None
    assert c.stats["loaded"] == 1 and c.stats["bad"] == 2
    assert "2 line(s)" in capsys.readouterr().err
    try:
        from habitus.dm.laya import cache_record
    except ImportError:                     # numpy only; laya imports no torch at module level
        return
    assert cache_record([1.23456, -0.5]) == rec
