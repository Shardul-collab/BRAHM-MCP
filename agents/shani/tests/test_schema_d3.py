"""
Decision D3 / D11 (2026-09-11): categories and gates for quantities the
original 20-category schema could not hold. Every value below is a real item
the model produced on a replay of the last full S5 run over workflow 1.
"""
import os
import sys

SHANI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SHANI not in sys.path:
    sys.path.insert(0, SHANI)

from tools.extract_research_knowledge import is_valid_knowledge, is_valid_value  # noqa: E402
from services.llm_service import LLMService                                      # noqa: E402

QT = {"in2se3", "raman", "xrd"}


def test_flux_values_have_a_home():
    for v in ["1.26 × 10^13 cm^-2 s^-1 [F_In]", "Se:In flux ratio of ~1.6 [In2Se3 growth]",
              "growth rate of 0.1 Å/s [Bi2Se3 and In2Se3 growth]", "2 × 10^-7 Torr [In BEP]"]:
        assert is_valid_knowledge("growth_flux", v, ""), v
    assert not is_valid_knowledge("growth_flux", "indium flux [constant]", "")


def test_dimensionless_dielectric_values():
    assert is_valid_knowledge("optical_property", "17 [bulk In2Se3 dielectric constant]", "")
    assert is_valid_knowledge("optical_property", "6.29 [dielectric constant of ultra-thin In2Se3]", "")
    assert is_valid_knowledge("optical_property", "2.7 eV [optical bandgap]", "")
    # still no home for a diffraction FWHM
    assert not is_valid_knowledge("optical_property", "0.079° [RC’s FWHM]", "")


def test_ferroelectric_and_photodetector_values():
    assert is_valid_knowledge("ferroelectric_property", "3.2 μC/cm2 [remanent polarization]", "")
    assert is_valid_knowledge("ferroelectric_property", "8.5 pm/V [d33]", "")
    assert is_valid_knowledge("photodetector_metric", "5 nA [dark current]", "")
    assert is_valid_knowledge("photodetector_metric", "1.2 × 10^10 Jones [detectivity]", "")
    assert is_valid_knowledge("photodetector_metric", "0.26 s [rise time]", "")
    assert not is_valid_knowledge("photodetector_metric", "not specified [dark current]", "")


def test_invented_categories_are_mapped_when_unambiguous():
    out = LLMService(None)._validate([
        {"category": "carrier_density", "value": "1e18 cm-3"},
        {"category": "dark_current", "value": "5 nA"},
        {"category": "growth_method", "value": "MBE"},
        {"category": "mobility", "value": "20 cm2/Vs"},     # ambiguous: stays dropped
    ])
    cats = [o["category"] for o in out]
    assert cats == ["electrical_property", "photodetector_metric", "synthesis_method"]


def test_categorical_values_do_not_need_a_query_term():
    for cat, v in [("application", "photodetectors"), ("application", "solar cells"),
                   ("characterization", "HAADF-STEM"), ("defect_type", "excess Se")]:
        assert is_valid_value(v, QT, cat), (cat, v)
        assert not is_valid_value(v, QT), (cat, v)   # unchanged for other callers
    assert not is_valid_value("et", QT, "application")


def test_dielectric_property_alias_d12():
    # 2026-09-11 pm rebuild: paper 20's headline value arrived as 'dielectric_property' and was dropped
    out = LLMService(None)._validate([{"category": "dielectric_property", "value": "εr = 17 [dielectric constant]"}])
    assert out and out[0]["category"] == "optical_property"
    assert is_valid_knowledge("optical_property", out[0]["value"], "")
