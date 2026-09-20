"""End-to-end detection + routing on synthetic files, one per technique.

VIDUR had no tests at all before 2026-09-19; this is the regression floor for
"the pipeline runs and picks the right parser".
"""
import numpy as np
import pytest

from extractor import extract
from auto_detector import detect, CONFIDENCE_THRESHOLD
from router import route


def _run(path):
    data = extract(str(path))
    det = detect(data)
    return det, route(det, data)


def test_xrd_xy_is_detected_as_xrd(tmp_path):
    f = tmp_path / "scan.xy"
    f.write_text("# 2Theta Intensity\n" + "\n".join(
        f"{10 + 0.5 * i} {100 + i}" for i in range(80)))
    det, res = _run(f)
    assert det["technique"] == "XRD"
    assert det["confidence"] >= CONFIDENCE_THRESHOLD
    assert res["parsed_data"]["axis_name"] == "2Theta"
    assert res["error"] is None


def test_uvvis_ascii_is_detected_as_uvvis(tmp_path):
    f = tmp_path / "spec.csv"
    f.write_text("Wavelength (nm),Absorbance\n" + "\n".join(
        f"{200 + i},{0.5 + i * 0.001}" for i in range(400)))
    det, res = _run(f)
    assert det["technique"] == "UV-Vis", det
    assert res["parsed_data"]["axis_name"] == "Wavelength_nm"


def test_eds_msa_is_detected_as_sem_eds(tmp_path):
    f = tmp_path / "spec.msa"
    f.write_text(
        "#FORMAT      : EMSA/MAS Spectral Data File\n"
        "#XUNITS      : keV\n#YUNITS      : Counts\n"
        "#SIGNALTYPE  : EDS\n#NPOINTS     : 10\n#SPECTRUM    :\n"
        + "\n".join(f"{i * 0.01}, {i * 3}" for i in range(10)))
    det, res = _run(f)
    assert det["technique"] == "SEM_EDX", det


def test_detection_returns_ranked_candidates(tmp_path):
    f = tmp_path / "scan.xy"
    f.write_text("# 2Theta Intensity\n" + "\n".join(
        f"{10 + 0.5 * i} {100 + i}" for i in range(80)))
    det, _ = _run(f)
    scores = [c["score"] for c in det["candidates"]]
    assert scores == sorted(scores, reverse=True)
    # Not a fixed count: the parser set grows (sourcemeter was added 2026-09-19).
    from auto_detector import _load_parsers
    assert len(det["candidates"]) == len(_load_parsers())


def test_unreadable_file_does_not_claim_a_technique(tmp_path):
    f = tmp_path / "junk.bin"
    f.write_bytes(b"\x00\x01\x02not instrument data at all\xff" * 4)
    det, res = _run(f)
    assert det["technique"] in ("Unknown", "Uncertain"), det


# ── header comments must not change the technique ────────────────────────────

def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body)
    return str(p)


def test_leading_comments_do_not_change_detected_technique(tmp_path):
    """
    2026-09-20 regression. xrd.can_parse() matched its "angle"/"q" axis-label
    patterns against data["text"] with re.M -- but extractor.extract() flattens
    newlines out of `text`, so `^` only ever matched the start of the whole
    file. A label on line 1 scored XRD 0.60; the same columns under two "#"
    comment lines scored XRD 0.05 and lost to SEM_EDX at 0.35, whose only
    evidence was that an x range of 0.5-1.5 looks like keV.

    The same data must not come back as two different techniques because one
    copy has a header comment.
    """
    import extractor
    from auto_detector import detect

    rows = "Q,I(Q)\n0.5 100\n1.0 200\n1.5 150\n"
    bare      = _write(tmp_path, "bare.xy", rows)
    commented = _write(tmp_path, "commented.xy", "# version 2.0\n# columns 2\n" + rows)

    a = detect(extractor.extract(bare))
    b = detect(extractor.extract(commented))

    assert a["technique"] == b["technique"] == "XRD"
    assert a["confidence"] == b["confidence"]


def test_axis_label_is_read_from_the_extracted_headers(tmp_path):
    """
    The label the scorer needs is already isolated in data["table_headers"];
    it must be scored from there, not re-derived from raw text. A file whose
    label reaches the headers but never appears at the start of `text` still
    has to produce the axis_label signal.
    """
    import extractor
    from parsers import xrd

    p = _write(tmp_path, "angle.xy",
               "; exported by the diffractometer\nAngle,Counts\n10 100\n20 200\n30 150\n")
    score, signals = xrd.can_parse(extractor.extract(p))
    assert any(s.startswith("axis_label:") for s in signals), signals
    assert score >= 0.6
