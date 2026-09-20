"""The binary parsers used to invent their x-axis.

Found 2026-09-19: four of the six binary paths built the axis with a hardcoded
`linspace`/step and then called it degrees, cm-1, nm or keV --

    xrd._parse_bruker_raw   np.linspace(10.0, 80.0, n)     -> "2Theta" / degrees
    raman._parse_wdf        np.linspace(100.0, 3500.0, n)  -> "RamanShift_cm-1"
    uvvis._parse_pe_sp      np.linspace(800.0, 200.0, n)   -> "Wavelength_nm"
    sem_eds._parse_edax_spc np.arange(n) * 0.01            -> "Energy_keV"

None of those numbers came from the file. A peak position read off any of them
was fabricated. Three of the four also filtered the intensity array *before*
pairing it with the axis, so every dropped point shifted the rest.

These tests fail against the old code.
"""
import struct
import numpy as np
import pytest

from parsers import xrd, raman, uvvis, sem_eds


def _binary(path, header_bytes, values, dtype=np.float32, magic=b""):
    header = magic + b"\x00" * (header_bytes - len(magic))
    path.write_bytes(header + np.asarray(values, dtype=dtype).tobytes())
    return str(path)


def test_bruker_raw_does_not_invent_a_2theta_range(tmp_path):
    f = _binary(tmp_path / "s.raw", 712, [10.0, 20.0, 30.0, 40.0], magic=b"RAW1.01")
    out = xrd._parse_bruker_raw(f)
    assert out["metadata"]["axis_calibrated"] is False
    assert out["axis_name"] == "channel"
    assert out["axis"] == [0.0, 1.0, 2.0, 3.0]
    assert out["axis"][-1] != 80.0          # the old fabricated endpoint


def test_wdf_does_not_invent_a_raman_shift_range(tmp_path):
    f = _binary(tmp_path / "s.wdf", 512, [5.0, 6.0, 7.0])
    out = raman._parse_wdf(f)
    assert out["metadata"]["axis_calibrated"] is False
    assert out["axis"] == [0.0, 1.0, 2.0]
    assert 3500.0 not in out["axis"]        # the old fabricated endpoint


def test_pe_sp_does_not_invent_a_wavelength_range(tmp_path):
    f = _binary(tmp_path / "s.sp", 0x1000, [0.1, 0.2, 0.3])
    out = uvvis._parse_pe_sp(f)
    assert out["metadata"]["axis_calibrated"] is False
    assert out["axis"] == [0.0, 1.0, 2.0]
    assert 800.0 not in out["axis"]         # the old fabricated start


def test_edax_spc_does_not_invent_an_energy_scale(tmp_path):
    f = _binary(tmp_path / "s.spc", 4096, [3, 4, 5], dtype=np.uint32)
    out = sem_eds._parse_edax_spc(f)
    assert out["metadata"]["axis_calibrated"] is False
    assert out["axis"] == [0.0, 1.0, 2.0]   # not 0.00, 0.01, 0.02 keV


def test_uncalibrated_axes_are_never_labelled_with_a_physical_unit(tmp_path):
    cases = [
        (xrd._parse_bruker_raw,     _binary(tmp_path / "a.raw", 712, [1.0, 2.0], magic=b"RAW1.01")),
        (raman._parse_wdf,          _binary(tmp_path / "b.wdf", 512, [1.0, 2.0])),
        (uvvis._parse_pe_sp,        _binary(tmp_path / "c.sp", 0x1000, [0.1, 0.2])),
        (sem_eds._parse_edax_spc,   _binary(tmp_path / "d.spc", 4096, [1, 2], np.uint32)),
    ]
    for fn, path in cases:
        out = fn(path)
        assert out["metadata"]["units"] == "index", out["metadata"]
        assert out["axis_name"] == "channel", out


def test_unsupported_raw_version_names_itself_instead_of_falling_through(tmp_path):
    """A real Bruker RAW4.00 was handed to the v3 offset and, when that failed,
    to the ASCII loader, which reported "Could not parse ... as XRD ASCII data".
    Measured on FAIRmat's TwoTheta_scan_scrambled.raw, 2026-09-19."""
    f = tmp_path / "v4.raw"
    f.write_bytes(b"RAW4.00\x00" + b"\x00" * 704 + np.array([1.0, 2.0], dtype=np.float32).tobytes())
    with pytest.raises(ValueError, match="RAW4.00"):
        xrd._parse_bruker_raw(str(f))
