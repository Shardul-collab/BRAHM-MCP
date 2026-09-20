"""Where a file DOES carry its calibration, the parser must use it -- and where it
does not, it must not substitute a guess.

`sem_eds._parse_bruker_spx` silently fell back to 10.0 eV/channel when <CalibLin>
was absent, which put every peak at an energy the file never stated.
`xrd._parse_xrdml`'s index fallback was labelled "2Theta"/degrees.
"""
import numpy as np
from parsers import xrd, sem_eds

XRDML = """<?xml version="1.0"?>
<xrdMeasurements><xrdMeasurement><scan><dataPoints>
{positions}
<counts>10 20 30 40 50</counts>
</dataPoints></scan></xrdMeasurement></xrdMeasurements>"""

POS = """<positions axis="2Theta"><startPosition>20.0</startPosition>
<endPosition>60.0</endPosition></positions>"""


def test_xrdml_uses_the_real_start_and_end_positions(tmp_path):
    f = tmp_path / "s.xrdml"
    f.write_text(XRDML.format(positions=POS))
    out = xrd._parse_xrdml(str(f))
    assert out["metadata"]["axis_calibrated"] is True
    assert out["axis_name"] == "2Theta"
    assert out["axis"][0] == 20.0 and out["axis"][-1] == 60.0


def test_xrdml_without_positions_is_marked_uncalibrated(tmp_path):
    f = tmp_path / "s.xrdml"
    f.write_text(XRDML.format(positions=""))
    out = xrd._parse_xrdml(str(f))
    assert out["metadata"]["axis_calibrated"] is False
    assert out["axis_name"] == "channel"
    assert out["axis"] == [0.0, 1.0, 2.0, 3.0, 4.0]


SPX = """<?xml version="1.0"?><TRTSpectrum>{calib}
<Channels>100 200 300 400</Channels></TRTSpectrum>"""


def test_spx_uses_caliblin_when_present(tmp_path):
    f = tmp_path / "s.spx"
    f.write_text(SPX.format(calib="<CalibLin>20.0</CalibLin>"))
    out = sem_eds._parse_bruker_spx(str(f))
    assert out["metadata"]["axis_calibrated"] is True
    assert out["axis"] == [0.0, 0.02, 0.04, 0.06]


def test_spx_without_caliblin_does_not_guess_10ev_per_channel(tmp_path):
    f = tmp_path / "s.spx"
    f.write_text(SPX.format(calib=""))
    out = sem_eds._parse_bruker_spx(str(f))
    assert out["metadata"]["axis_calibrated"] is False
    assert out["axis"] == [0.0, 1.0, 2.0, 3.0]   # not 0.00, 0.01, 0.02, 0.03 keV
