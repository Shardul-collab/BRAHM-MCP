"""L4 claim plan (2026-09-11 pm): ranges must state the real extremes, never pool unknown units."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ganesh.writing.grounding import Evidence            # noqa: E402
from ganesh.writing.claims import build_claims, number_and_unit, number_range_and_unit  # noqa: E402


def ev(i, pid, cat, val):
    return Evidence(f"E{i}", pid, cat, val, f"The value was {val}.")


def test_range_states_true_minimum_and_maximum_beyond_eight_items():
    temps = [150, 200, 250, 280, 300, 320, 340, 360, 380, 550]
    evs = [ev(i, i, "growth_temperature", f"{t} °C [substrate temperature]") for i, t in enumerate(temps, 1)]
    c = [x for x in build_claims(evs) if x.kind in ("range", "disagreement")][0]
    assert "150 °C" in c.statement and "550 °C" in c.statement
    assert len(c.evidence) == 8 and {e.value.split()[0] for e in c.evidence} >= {"150", "550"}
    assert "Across 8 papers" in c.statement          # counts the papers actually cited


def test_range_value_uses_its_upper_bound():
    evs = [ev(1, 1, "growth_temperature", "150 to 550 °C [range]"), ev(2, 2, "growth_temperature", "300 °C")]
    c = [x for x in build_claims(evs) if x.kind in ("range", "disagreement")][0]
    assert c.statement.startswith("Across 2 papers") and "span 150 to 550 °C [E1]." in c.statement
    assert number_range_and_unit("150 to 550 °C")[:2] == (150.0, 550.0)


def test_unknown_units_are_not_pooled_into_a_range():
    evs = [ev(1, 1, "electrical_property", "7 ms [photoconductivity rise]"),
           ev(2, 2, "electrical_property", "~1x1014 cm-2 (p-type) [sheet density]")]
    assert not [x for x in build_claims(evs) if x.kind in ("range", "disagreement")]


def test_unit_prefixes_are_scaled():
    assert abs(number_and_unit("2.82 µA/W")[0] - 2.82e-6) < 1e-12
    assert number_and_unit("44.6 A/W") == (44.6, "A/W")
    assert number_and_unit("7 ms")[1] == "s"
    evs = [ev(1, 1, "photoresponsivity", "5 mA/W"), ev(2, 2, "photoresponsivity", "2 A/W")]
    c = [x for x in build_claims(evs) if x.kind in ("range", "disagreement")][0]
    assert "span 5 mA/W [E1] to 2 A/W [E2]" in c.statement


def test_broad_category_ranges_are_per_quantity():
    evs = [ev(1, 1, "optical_property", "1.28 eV [band gap for β-In2Se3]"),
           ev(2, 2, "optical_property", "1.45 eV [optical bandgap]"),
           ev(3, 3, "optical_property", "-5.86 eV/-7.27 eV [band edges]")]
    rng = [x for x in build_claims(evs) if x.kind in ("range", "disagreement")]
    assert len(rng) == 1 and "band gap values span 1.28 eV [E1] to 1.45 eV [E2]" in rng[0].statement
    assert any(x.kind == "single" and x.evidence[0].eid == "E3" for x in build_claims(evs))


def test_misfiled_rows_do_not_form_a_range():
    evs = [ev(1, 1, "subthreshold_swing", "rise and decay times of τr = 0.6 ms"),
           ev(2, 2, "subthreshold_swing", "14 ms [built-up time constant]")]
    assert not [x for x in build_claims(evs) if x.kind in ("range", "disagreement")]


def test_values_beyond_the_cap_are_not_dropped():
    temps = [150, 200, 250, 280, 300, 320, 340, 360, 380, 550, 600]
    evs = [ev(i, i, "growth_temperature", f"{t} °C") for i, t in enumerate(temps, 1)]
    used = {e.eid for c in build_claims(evs) for e in c.evidence}
    assert used == {e.eid for e in evs}


def test_ordinal_degree_sign_is_a_degree():
    assert number_and_unit("950 ºC [substrate temperature]") == (950.0, "°C")
