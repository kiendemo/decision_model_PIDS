"""repro.compare: byte-exact CSV comparison (line endings, quoting), numeric CSV / JSON tolerance,
and the per-level score-file comparison of G4b (E2 record keys vs E1 {qv, qt, cov_min})."""

import copy
import csv
import io

from habitus.repro import compare as C

COLS = ["ds", "level", "mcc", "note"]


def _csv(rows, lineterminator="\r\n"):
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=COLS, lineterminator=lineterminator)
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue().encode("utf-8")


ROWS = [{"ds": "A", "level": "E2", "mcc": 0.405488, "note": ""},
        {"ds": "B", "level": "L4", "mcc": 0.1, "note": "x, with a comma"}]


def _ok(checks):
    return all(c.ok for c in checks)


def test_identical_bytes_pass():
    assert _ok(C.compare_csv_bytes(_csv(ROWS), _csv(ROWS)))


def test_crlf_sensitivity():
    crlf, lf = _csv(ROWS), _csv(ROWS, "\n")
    assert crlf.count(b"\r\n") == 3 and b"\r\n" not in lf
    assert not _ok(C.compare_csv_bytes(lf, crlf))
    checks = C.compare_csv_bytes(lf, crlf, ignore_cols=("note",))
    bad = [c for c in checks if not c.ok]
    assert [c.name for c in bad] == ["csv: line endings"]


def test_declared_column_is_the_only_exception():
    other = copy.deepcopy(ROWS)
    other[1]["note"] = "a different note"
    assert not _ok(C.compare_csv_bytes(_csv(other), _csv(ROWS)))
    assert _ok(C.compare_csv_bytes(_csv(other), _csv(ROWS), ignore_cols=("note",)))
    other[0]["mcc"] = 0.405489
    checks = C.compare_csv_bytes(_csv(other), _csv(ROWS), ignore_cols=("note",))
    assert [c.name for c in checks if not c.ok] == ["csv: fields"]


def test_quoting_difference_is_a_difference():
    ref = b'ds,level,mcc,note\r\nA,E2,0.1,x\r\n'
    quoted = b'ds,level,mcc,note\r\n"A",E2,0.1,x\r\n'
    checks = C.compare_csv_bytes(quoted, ref, ignore_cols=("note",))
    assert not _ok(checks)


def test_csv_records_tokenizer():
    data = b'a,"b,c","d ""q"" e"\r\n"multi\r\nline",,x\n'
    recs = C.csv_records(data)
    assert [f for f, _ in recs] == [[b"a", b'"b,c"', b'"d ""q"" e"'], [b'"multi\r\nline"', b"", b"x"]]
    assert [t for _, t in recs] == [b"\r\n", b"\n"]
    assert [C._unquote(x) for x in recs[0][0]] == ["a", "b,c", 'd "q" e']


def test_numeric_csv_tolerance():
    ref = "k,n,x\r\na,3,0.1\r\nb,4,0.30000000000000004\r\n"
    assert _ok(C.compare_csv_numeric(ref.replace("0.30000000000000004", "0.3"), ref, tol=1e-12))
    assert not _ok(C.compare_csv_numeric(ref.replace(",3,", ",4,"), ref))           # ints exact
    assert not _ok(C.compare_csv_numeric(ref.replace("0.1", "0.1000001"), ref))
    assert not _ok(C.compare_csv_numeric(ref.replace("a,", "z,"), ref))              # text exact
    assert _ok(C.compare_csv_numeric("k,x\r\na,nan\r\n", "k,x\r\na,nan\r\n"))


def test_json_ints_exact_floats_within_tolerance():
    ref = {"n": 5, "m": [0.1, 0.2], "s": "x", "z": None, "b": True}
    assert _ok(C.compare_json({"n": 5, "m": [0.1 + 1e-15, 0.2], "s": "x", "z": None, "b": True}, ref, tol=1e-12))
    assert not _ok(C.compare_json({**ref, "n": 6}, ref))
    assert not _ok(C.compare_json({**ref, "b": 1}, ref))
    assert not _ok(C.compare_json({**ref, "extra": 1}, ref))
    assert not _ok(C.compare_json({**ref, "m": [0.1]}, ref))
    assert C.json_diffs({"a": {"spec": "x", "l": 1}}, {"a": {"spec": "y", "l": 1}}, ignore=("a.spec",)) == []


def test_text_generated_stamp_is_the_declared_exception():
    a = "# T\n\nGenerated 2026-10-29 18:45:14 by `x`.\nbody\n"
    b = a.replace("18:45:14", "19:00:00")
    assert not _ok(C.compare_text(b, a))
    assert _ok(C.compare_text(b, a, ignore=(C.GENERATED_STAMP,)))
    assert not _ok(C.compare_text(b.replace("body", "Body"), a, ignore=(C.GENERATED_STAMP,)))


def _e2():
    meta = {"ds": "X", "day": "d", "level": "E2", "k": "2", "ref": ["a", "b"], "role": "test",
            "caps": {"process": 4, "file": 2, "netflow": 6}, "n_states": 2, "n_unique_states": 2,
            "n_qa": 2, "n_hard": 0, "e2": {"spec": "pidsmaker/.../SPEC_E2.md", "lambda": 0.5, "routes": {"qv": ["entity"]}},
            "seconds": {"states": 1.0, "engine": 2.0}, "engine": "qwen3.5-4b", "prefix_share": True,
            "batch_size": 32, "grouping": "qvqt", "low_cov": 0, "written": "2026-10-17 11:51:07"}
    rec = {"qv": 0.25, "qt": 0.5, "qa": 0.5, "hard": False, "cov_min": 0.98, "base": 0.5, "composite": 0.375}
    return {"meta": meta, "head": {"process": ["s1", "s2"], "file": [], "netflow": []},
            "scores": {"s1": dict(rec), "s2": {**rec, "qa": None, "composite": 0.5}}, "fallback": {"s3": -0.75}}


def _e1():
    f = _e2()
    f["meta"]["level"] = "L4"
    f["meta"]["grouping"] = "state"
    for k in ("n_qa", "n_hard", "e2"):
        del f["meta"][k]
    f["scores"] = {s: {"qv": r["qv"], "qt": r["qt"], "cov_min": r["cov_min"]} for s, r in f["scores"].items()}
    return f


def test_score_file_equal_except_run_facts():
    ref = _e2()
    new = copy.deepcopy(ref)
    new["meta"].update(seconds={"states": 9.9, "engine": 0.1}, written="now")
    new["meta"]["e2"]["spec"] = "docs/DESIGN.md"
    checks, worst = C.compare_score_file(new, ref)
    assert _ok(checks) and worst == 0.0


def test_score_file_any_delta_fails():
    ref = _e2()
    new = copy.deepcopy(ref)
    new["scores"]["s1"]["qv"] = 0.2500000000000001                # one ulp: still a failure
    checks, worst = C.compare_score_file(new, ref)
    assert not _ok(checks) and worst > 0
    new = copy.deepcopy(ref)
    new["scores"]["s2"]["qa"] = 0.0                         # None (QA not asked) vs a number
    assert not _ok(C.compare_score_file(new, ref)[0])
    new = copy.deepcopy(ref)
    new["meta"]["e2"]["lambda"] = 0.25                      # e2 is compared except its spec
    assert [c.name.split(": ")[1] for c in C.compare_score_file(new, ref)[0] if not c.ok] == ["meta"]
    new = copy.deepcopy(ref)
    new["head"]["file"] = ["s9"]
    assert not _ok(C.compare_score_file(new, ref)[0])
    new = copy.deepcopy(ref)
    new["fallback"]["s3"] = -0.5
    assert not _ok(C.compare_score_file(new, ref)[0])


def test_e1_records_are_qv_qt_cov_min_only():
    ref = _e1()
    assert _ok(C.compare_score_file(copy.deepcopy(ref), ref)[0])
    new = copy.deepcopy(ref)
    for r in new["scores"].values():
        r["composite"] = max(r["qv"], r["qt"])          # an E1 file must not carry E2 keys
    bad = [c.name.split(": ")[1] for c in C.compare_score_file(new, ref)[0] if not c.ok]
    assert bad == ["record keys"]
    new = copy.deepcopy(ref)
    new["meta"]["n_qa"] = 2                              # extra NEW meta keys are not compared
    assert _ok(C.compare_score_file(new, ref)[0])
    new = copy.deepcopy(ref)
    del new["meta"]["grouping"]                         # but every SHIPPED key must be there
    assert not _ok(C.compare_score_file(new, ref)[0])


def test_e2_shipped_keys_drive_the_comparison():
    ref = _e2()
    new = copy.deepcopy(ref)
    del new["scores"]["s1"]["base"]
    assert not _ok(C.compare_score_file(new, ref)[0])
    new = copy.deepcopy(ref)
    new["scores"].pop("s2")
    bad = [c.name.split(": ")[1] for c in C.compare_score_file(new, ref)[0] if not c.ok]
    assert bad == ["signatures"]


def test_report_table_and_status():
    r = C.Report("t")
    r.add("a", True, "fine")
    r.add("b", False, "", got=1, want=2)
    assert not r.ok and r.n_fail == 1
    t = r.table()
    assert "PASS" in t and "FAIL" in t and "got 1 want 2" in t and t.endswith("1 PASS, 1 FAIL")
    assert r.as_dict()["checks"][1]["status"] == "FAIL"
    assert not C.Report("empty").ok
