"""
Parser tests for the Vishwakarma QE output parsers.

Added 2026-09-09. Vishwakarma had no tests at all before this; the only
verification was docker/smoke_test_vishwakarma.sh, which runs pw.x directly
and never exercises a line of Python.

Fixture policy — read this before trusting a green run:
  * parse_pw and parse_dos are checked against REAL output captured from
    jobs in agents/vishwakarma/jobs/ (see fixtures/). Those are trustworthy.
  * parse_ph, parse_neb and parse_hp are checked against SYNTHETIC fixtures
    written by hand from the QE output format, because there has never been
    a ph/neb/hp job in jobs/ to capture. They prove the parser does what it
    intends on the shape it expects — they do NOT prove that shape matches
    what those binaries really emit. Replace them with captured output the
    first time a real run of each completes.
"""

import pytest

from vishwakarma import output_parser as op


# ─── Synthetic fixtures ───────────────────────────────────────────────────────

HP_TABLE_OUT = """
     =--------------------------------------------=
             Hubbard parameters of DFT+U(+V)
     =--------------------------------------------=

       site n.  type  label  spin  new_type  new_label  Hubbard U (eV)
         1       1     Ni     1        1        Ni        6.7788
         2       1     Ni    -1        2        Ni        6.7231

     JOB DONE.
"""

HP_INLINE_OUT = """
     Computed Hubbard U (eV) = 4.5600
     JOB DONE.
"""

NEB_OUT = """
     activation energy (->) =   0.812345 eV
     image:   1    -25.500000 eV
     image:   2    -24.900000 eV
     image:   3    -24.687655 eV
     neb: convergence achieved in  42 iterations
"""


# ─── parse_hp ─────────────────────────────────────────────────────────────────

def test_parse_hp_reads_site_table():
    r = op.parse_hp(HP_TABLE_OUT)
    assert r["code"] == "hp"
    assert r["u_values_ev"] == pytest.approx([6.7788, 6.7231])
    assert [s["site"] for s in r["sites"]] == [1, 2]
    assert [s["label"] for s in r["sites"]] == ["Ni", "Ni"]


def test_parse_hp_reads_inline_form():
    """The shape the old inline handler regex matched — must keep working."""
    r = op.parse_hp(HP_INLINE_OUT)
    assert r["u_values_ev"] == pytest.approx([4.56])


def test_parse_hp_empty_output_is_not_converged():
    r = op.parse_hp("")
    assert r["u_values_ev"] == []
    assert r["converged"] is False


def test_hp_is_registered_in_dispatcher():
    """
    parse() had no "hp" entry, so an hp job silently fell through to
    parse_pw and returned a pw-shaped dict full of Nones.
    """
    assert op.parse(HP_TABLE_OUT, code="hp")["code"] == "hp"


# ─── parse_neb ────────────────────────────────────────────────────────────────

def test_parse_neb_extracts_barrier_and_path():
    r = op.parse_neb(NEB_OUT)
    assert r["converged"] is True
    assert r["activation_ev"] == pytest.approx(0.812345)
    assert len(r["path_energies_ev"]) == 3
    assert r["reaction_ev"] == pytest.approx(0.812345, abs=1e-3)


# ─── Real captured output ─────────────────────────────────────────────────────

def test_parse_pw_on_real_scf(real_pw_output):
    r = op.parse_pw(real_pw_output)
    assert r["code"] == "pw"
    assert r["converged"] is True
    assert r["total_energy_ry"] is not None
    assert r["total_energy_ev"] is not None
    # eV/Ry conversion must stay self-consistent. parse_pw rounds the eV
    # value to 6 dp, so compare at that resolution rather than exactly.
    assert r["total_energy_ev"] == pytest.approx(
        r["total_energy_ry"] * op.RY_TO_EV, abs=1e-6
    )


def test_check_job_success_on_real_scf(real_pw_output):
    ok, reason = op.check_job_success(real_pw_output)
    assert ok is True, reason
