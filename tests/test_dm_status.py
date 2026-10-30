"""The STATUS MATRIX: one source constant, embedded verbatim in docs/DECISION_MODELS.md."""

import pytest

import dm_helpers as H
from habitus import dm as D
from habitus.dm import status as S

DOC = H.REPO / "docs" / "DECISION_MODELS.md"


def test_doc_block_equals_render_markdown():
    text = DOC.read_text(encoding="utf-8")
    assert S.extract(text) == S.render_markdown()
    assert S.embed(text) == text


def test_matrix_rows():
    rows = S.STATUS_MATRIX
    allowed = {S.REPORTED, S.SUPPORTED_NE, S.FROZEN, S.SUPPORTED, S.UNSUPPORTED, S.FROZEN_SEL, S.REPRODUCED}
    assert all(r.status in allowed and r.questions in S.QUESTION_SETS for r in rows)
    configs = {a.config for a in D.ARMS.values()}
    assert all(r.config == "-" or r.config in configs for r in rows)
    by_arm = {(r.questions, r.arm): r for r in rows}
    assert by_arm[("routed", "MAIN (jev stock Qwen3.5-4B)")].status == S.REPORTED
    assert sum(r.status == S.REPORTED for r in rows) == 1
    habit = {r.arm: r for r in rows if r.questions == "habit"}
    for arm in ("HB-Z", "HB", "HB.head", "HB.08b", "HB-L.s0", "HB-L", "HB-L.mq"):
        assert habit[arm].status == S.FROZEN and habit[arm].config == D.ARMS[arm].config
    assert habit["HB-L.zs"].status == S.FROZEN_SEL and habit["HB-L.full"].status == S.UNSUPPORTED
    assert habit["HB-L.full"].config == "-" and "HB-L.full" not in D.ARMS
    assert [r for r in rows if r.questions == "routed" and r.status == S.SUPPORTED_NE]
    hb0 = [r for r in rows if r.questions == "hb0"]
    assert len(hb0) == 1 and hb0[0].status == S.REPRODUCED == "REPRODUCED (G11)"
    assert "G11" in hb0[0].note and "fetch_catalogs.sh" in hb0[0].note and "ROLE_TABLE_SHA256" in hb0[0].note
    from habitus.repro import gates as G
    assert "G11" in G.GATES and G.HB0_ROWS == hb0[0].evidence


def test_render_extract_embed_and_cli(tmp_path, capsys):
    md = S.render_markdown()
    assert md.endswith("\n") and md.count("\n") == len(S.STATUS_MATRIX) + 2 and S.render_markdown() == md
    doc = tmp_path / "x.md"
    doc.write_text(f"# T\n\n{S.BEGIN}\nold\n{S.END}\n\ntail\n", encoding="utf-8")
    assert S.extract(doc.read_text()) == "old\n"
    assert S.main(["--check", str(doc)]) == 1
    assert S.main(["--write", str(doc)]) == 0
    assert S.main(["--check", str(doc)]) == 0
    text = doc.read_text()
    assert text.startswith("# T\n\n") and text.endswith("\n\ntail\n") and S.extract(text) == md
    capsys.readouterr()
    assert S.main(["--render"]) == 0 and capsys.readouterr().out == md
    with pytest.raises(ValueError):
        S.extract("no markers")
    with pytest.raises(ValueError):
        S.extract(f"{S.BEGIN}\n{S.END}\n{S.BEGIN}\n{S.END}")


def test_module_cli_render():
    r = H.run_python("import runpy, sys; sys.argv = ['x', '--render']; "
                     "runpy.run_module('habitus.dm.status', run_name='__main__')")
    assert r.returncode == 0, r.stderr[-1000:]
    assert r.stdout == S.render_markdown()
