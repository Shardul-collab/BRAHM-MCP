"""scan() asks; process() writes. Uncertainty is an output, never a guess."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

import batch


def _xy(path, x, y, header="# 2Theta Intensity"):
    path.write_text(header + "\n" + "\n".join(f"{a} {b}" for a, b in zip(x, y)))


def _series(tmp_path, n=3, npts=60):
    x = [10 + 0.5 * i for i in range(npts)]
    for k in range(1, n + 1):
        _xy(tmp_path / f"ZnSe_S{k}_XRD.xy", x, [100 + k + i for i in range(npts)])
    return x


def test_scan_writes_nothing(tmp_path):
    _series(tmp_path)
    before = sorted(p.name for p in tmp_path.iterdir())
    batch.scan(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_scan_skips_files_that_are_not_data(tmp_path):
    """The 2026-09-19 harness handed VIDUR a README.md and it tried to parse it
    as UV-Vis."""
    _series(tmp_path, n=1)
    (tmp_path / "README.md").write_text("# notes")
    (tmp_path / "config.json").write_text("{}")
    plan = batch.scan(tmp_path)
    assert {s["file"] for s in plan["skipped"]} == {"README.md", "config.json"}


def test_scan_infers_the_series_and_samples(tmp_path):
    _series(tmp_path)
    plan = batch.scan(tmp_path)
    (g,) = [g for g in plan["series"] if len(g["members"]) == 3]
    assert sorted(m["sample"] for m in g["members"]) == ["S1", "S2", "S3"]


def test_scan_reports_the_parameters_it_will_not_assume(tmp_path):
    _series(tmp_path)
    plan = batch.scan(tmp_path)
    assert "wavelength_a" in plan["parameters_needed"]
    assert "not assumed" in plan["parameters_needed"]["wavelength_a"]


def test_process_writes_one_csv_per_file_with_raw_first(tmp_path):
    _series(tmp_path)
    out = batch.process(tmp_path, {"params": {"wavelength_a": 1.5406}})
    assert len(out["written"]) == 3
    for w in out["written"]:
        assert w["columns"][:3] == ["2Theta", "raw", "norm_max"]
        assert "d_A" in w["columns"] and "q_invA" in w["columns"]


def test_process_writes_a_wide_csv_when_the_axes_match(tmp_path):
    _series(tmp_path)
    out = batch.process(tmp_path, {})
    assert len(out["wide"]) == 1
    wide = out["wide"][0]
    assert sorted(wide["samples"]) == ["S1", "S2", "S3"]
    rows = list(csv.reader(open(Path(out["out_dir"]) / wide["csv"])))
    assert rows[0] == ["2Theta", "S1", "S2", "S3"]
    assert len(rows) == 61


def test_process_refuses_a_wide_csv_when_the_axes_differ_and_says_so(tmp_path):
    _xy(tmp_path / "ZnSe_S1_XRD.xy", [10 + 0.5 * i for i in range(60)], range(60))
    _xy(tmp_path / "ZnSe_S2_XRD.xy", [11 + 0.7 * i for i in range(60)], range(60))
    out = batch.process(tmp_path, {})
    assert out["wide"] == []
    assert any("x-axes differ by more than" in n["note"] for n in out["notes"])


def test_process_honours_a_technique_override(tmp_path):
    f = tmp_path / "mystery_A_run.dat"
    f.write_text("\n".join(f"{100 + i} {i}" for i in range(50)))
    plan = batch.scan(tmp_path)
    assert plan["files"][0]["technique"] in (None, "Unknown", "Uncertain")
    assert plan["questions"] and plan["questions"][0]["kind"] == "technique"
    out = batch.process(tmp_path, {"techniques": {f.name: "Raman"}})
    assert len(out["written"]) == 1
    assert out["written"][0]["technique"] == "Raman"


def test_unresolved_files_are_skipped_not_guessed(tmp_path):
    f = tmp_path / "mystery_A_run.dat"
    f.write_text("\n".join(f"{100 + i} {i}" for i in range(50)))
    out = batch.process(tmp_path, {})
    assert out["written"] == []
    # The file IS skipped and the reason names the uncertainty; the exact
    # wording comes from the router, so assert the behaviour, not the string.
    assert len(out["skipped"]) == 1
    assert out["skipped"][0]["file"] == f.name
    assert "uncertain" in out["skipped"][0]["why"].lower()


def test_a_not_yet_supported_technique_is_named_not_attempted(tmp_path):
    f = tmp_path / "cell_A_XPS.dat"
    f.write_text("\n".join(f"{100 + i} {i}" for i in range(50)))
    out = batch.process(tmp_path, {"techniques": {f.name: "XPS"}})
    assert out["written"] == []
    assert "no parser in VIDUR yet" in out["skipped"][0]["why"]


def test_meta_json_records_outstanding_questions(tmp_path):
    _series(tmp_path)
    out = batch.process(tmp_path, {})
    meta = json.loads((Path(out["out_dir"]) / "_vidur_meta.json").read_text())
    assert "questions_outstanding" in meta and "written" in meta


def test_questions_travel_with_their_evidence(tmp_path):
    """VIDUR has no interface: the caller is a model. A question it cannot
    answer from the file is useless, so each one carries the filename tokens,
    the head of the file, and the column labels."""
    f = tmp_path / "mystery_A_run.dat"
    f.write_text("Reading,Unit\n" + "\n".join(f"{i},x" for i in range(5)))
    plan = batch.scan(tmp_path)
    q = [q for q in plan["questions"] if q["kind"] == "technique"][0]
    ev = q["evidence"]
    assert ev["name_tokens"] == ["mystery", "A", "run"]
    assert ev["column_labels"] == ["Reading", "Unit"]
    assert ev["head"]


def test_a_technique_with_a_parser_is_not_listed_as_unsupported():
    """IV/IT stayed in NOT_YET_SUPPORTED after their parser was written, so 17
    real files were identified at confidence 1.00 and then skipped."""
    from router import _get_parser_map
    for t in batch.NOT_YET_SUPPORTED:
        assert t not in _get_parser_map(), f"{t} has a parser but is listed unsupported"
    for t in batch.KNOWN_TECHNIQUES:
        assert t in _get_parser_map(), f"{t} is advertised but has no parser"


def test_a_constant_the_file_states_is_not_asked_for(tmp_path):
    """A PANalytical .xrdml carries <kAlpha1>. Asking the operator for it is the
    same mistake as assuming Cu Ka1 -- found 2026-09-19 through the live
    connector, on a file whose head showed the wavelength it was asking about."""
    f = tmp_path / "S1_XRD.xrdml"
    f.write_text(
        '<?xml version="1.0"?><xrdMeasurements><xrdMeasurement>'
        '<usedWavelength><kAlpha1 unit="Angstrom">1.5405980</kAlpha1></usedWavelength>'
        '<scan><dataPoints><positions axis="2Theta"><startPosition>10</startPosition>'
        '<endPosition>50</endPosition></positions>'
        '<intensities unit="counts">10 20 30 40 50</intensities>'
        '</dataPoints></scan></xrdMeasurement></xrdMeasurements>')
    plan = batch.scan(tmp_path)
    assert plan["files"][0]["wavelength_a"] == pytest.approx(1.540598)
    assert "wavelength_a" not in plan["parameters_needed"]
    out = batch.process(tmp_path, {})          # no params passed at all
    assert "d_A" in out["written"][0]["columns"]


def test_evidence_is_carried_only_where_a_question_needs_it(tmp_path):
    """17 files x (12 head lines + 21 column labels) made one real scan
    unreadable through the connector."""
    _series(tmp_path, n=3)
    plan = batch.scan(tmp_path)
    assert all(f.get("evidence") is None for f in plan["files"])


def test_an_answered_question_is_not_still_outstanding(tmp_path):
    """Reporting a question the caller just answered trains the operator to
    ignore the field."""
    f = tmp_path / "mystery_A_run.dat"
    f.write_text("\n".join(f"{100 + i} {i}" for i in range(50)))
    out = batch.process(tmp_path, {"techniques": {f.name: "Raman"}})
    assert out["written"]
    assert [q for q in out["questions_outstanding"] if q.get("file") == f.name] == []


def test_wide_csv_pairs_setpoints_that_are_nominally_equal(tmp_path):
    """A source-meter returns -1.0000072717667 and -0.9999924898148 for the same
    nominal -1 V. Demanding equality to 1e-6 refused a wide CSV for every real
    I-V series -- measured on 17 files, 2026-09-19."""
    import numpy as np
    base = np.linspace(-1, 1, 21)
    for k, jitter in enumerate((1e-5, -8e-6), start=1):
        rows = "\n".join(f"{v + jitter} {1e-6 * i}" for i, v in enumerate(base))
        (tmp_path / f"ZnSe_S{k}_XRD.xy").write_text("# 2Theta Intensity\n" + rows)
    out = batch.process(tmp_path, {})
    assert len(out["wide"]) == 1
    w = out["wide"][0]
    assert w["max_axis_deviation"] < w["axis_tolerance"]
