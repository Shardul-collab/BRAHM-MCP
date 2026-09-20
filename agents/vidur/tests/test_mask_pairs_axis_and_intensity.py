"""_out() must mask axis and intensity together.

The binary paths filtered the intensity array before an axis existed for it, so a
dropped point did not drop its axis value -- it shifted every later point onto
the wrong one. Silent, and invisible in the output shape.
"""
import numpy as np
from parsers import xrd, raman, uvvis, sem_eds


def test_negative_values_are_kept_and_only_nonfinite_is_dropped():
    """VIDUR prepares data for plotting; it must not decide which points are
    real. Value-range predicates (intensity >= 0, and a (wl > 100) & (wl < 3000)
    window in uvvis) used to delete points silently -- measured 2026-09-19,
    the window removed the first 30 rows of a 1024-row LabSpec file.
    Baseline-corrected XRD and Raman are negative on purpose."""
    axis = np.array([10.0, 20.0, 30.0, 40.0])
    inten = np.array([100.0, -5.0, 300.0, np.nan])
    out = xrd._out(axis, inten, "test")
    assert out["axis"] == [10.0, 20.0, 30.0]          # 40.0 carried a NaN
    assert out["intensity"] == [100.0, -5.0, 300.0]   # the negative survives
    assert out["metadata"]["dropped_nonfinite"] == 1


def test_no_parser_applies_a_value_range_filter():
    axis = np.array([1.0, 2.0, 3.0])
    inten = np.array([-50.0, 0.0, 50.0])
    for mod in (xrd, raman, uvvis, sem_eds):
        out = mod._out(axis, inten, "test")
        assert out["intensity"] == [-50.0, 0.0, 50.0], mod.__name__
        assert out["metadata"]["dropped_nonfinite"] == 0, mod.__name__


def test_every_parser_out_returns_equal_length_axis_and_intensity():
    axis = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    inten = np.array([1.0, np.nan, 3.0, -1.0, 5.0])
    for mod in (xrd, raman, uvvis, sem_eds):
        out = mod._out(axis, inten, "test")
        assert len(out["axis"]) == len(out["intensity"]), mod.__name__
