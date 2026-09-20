"""VIDUR against real instrument files (downloaded 2026-09-19).

The synthetic fixtures elsewhere in this directory test one behaviour each.
These test that the agent survives files written by actual instruments, and --
where the file states its own axis -- that VIDUR reproduces it rather than
approximating it. Every one of these failed before the 2026-09-19 pass.

Data lives outside the repo, in agents/shani/logs/vidur_2026-09-19/testdata/,
so these skip when it is absent. Provenance is in that directory's README.
"""
import re
from pathlib import Path

import pytest

from extractor import extract
from auto_detector import detect
from router import route

DATA = (Path(__file__).resolve().parents[2]
        / "shani/logs/vidur_2026-09-19/testdata")
pytestmark = pytest.mark.skipif(not DATA.is_dir(), reason="real test data not present")


def _run(name):
    f = DATA / name
    if not f.is_file():
        pytest.skip(f"{name} not downloaded")
    d = extract(str(f))
    det = detect(d)
    return det, route(det, d)


@pytest.mark.parametrize("name", [
    "test_scan.xrdml", "XRD-918-16_10.xrdml", "m54313_om2th_10.xrdml",
])
def test_real_xrdml_axis_matches_the_file_s_own_positions(name):
    """All three use <intensities>, which the parser did not look for -- they
    failed with "<counts> element not found in XRDML"."""
    det, res = _run(name)
    assert res["technique"] == "XRD"
    pd = res["parsed_data"]
    assert pd["metadata"]["axis_calibrated"] is True

    text = (DATA / name).read_text(errors="ignore")
    start = float(re.search(r"<startPosition>([-\d.]+)</startPosition>", text).group(1))
    end = float(re.search(r"<endPosition>([-\d.]+)</endPosition>", text).group(1))
    assert pd["axis"][0] == pytest.approx(start, abs=0.05)
    assert pd["axis"][-1] == pytest.approx(end, abs=0.05)


def test_real_bruker_raw_is_recognised_as_xrd_and_refuses_honestly():
    """RAW4.00 scored 0.2 ("Uncertain") and then failed in the ASCII loader with
    a message about ASCII, which is not what went wrong."""
    det, res = _run("TwoTheta_scan_scrambled.raw")
    assert res["technique"] == "XRD"
    assert det["confidence"] >= 0.6
    assert "RAW4.00" in res["error"]
    assert "ASCII" not in res["error"]


def test_real_jasco_raman_txt_parses_despite_header_and_trailing_block():
    """LS4.txt: 18-line keyword header, XYDATA, then a trailing metadata block.
    Detected as Raman at confidence 1.0, then failed to load."""
    det, res = _run("LS4.txt")
    assert res["technique"] == "Raman"
    pd = res["parsed_data"]
    text = (DATA / "LS4.txt").read_text(errors="ignore")
    npoints = int(re.search(r"NPOINTS\s+(\d+)", text).group(1))
    firstx = float(re.search(r"FIRSTX\s+([\d.]+)", text).group(1))
    lastx = float(re.search(r"LASTX\s+([\d.]+)", text).group(1))
    assert len(pd["axis"]) == npoints
    assert pd["axis"][0] == pytest.approx(firstx, abs=0.5)
    assert pd["axis"][-1] == pytest.approx(lastx, abs=0.5)


def test_a_low_confidence_guess_is_not_reported_as_a_technique():
    """A Galactic .spc scored 0.1 and came back as a confident "SEM_EDX"."""
    det, res = _run("barbsvd.spc")
    assert det["technique"] == "Uncertain"
    assert res["technique"] == "Uncertain"
    assert res.get("best_guess") == "SEM_EDX"


def test_real_perkinelmer_sp_axis_is_not_fabricated():
    """The .sp header is not parsed, so the axis must be channel indices, not
    the linspace(800, 200) nm this used to invent."""
    det, res = _run("spectra.sp")
    pd = res["parsed_data"]
    assert pd["metadata"]["axis_calibrated"] is False
    assert pd["axis_name"] == "channel"
    assert pd["axis"][0] == 0.0
