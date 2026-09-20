"""
Regression tests for S4.5 normalisation.

The bug these exist for: run_normalisation deleted every section whose name
matched NOISE_EXACT, unconditionally and per row. On the 2026-09-09 In2Se3 run
that removed 11 of 19 sections and left SIX of nine downloaded papers with zero
PaperContent — invisible to S5 and to every agent reading the corpus. The
corpus was then built from 1-3 papers' worth of text instead of 9.

Deleting noise is right. Deleting a paper's LAST section is not: when every
section looks like noise, the likely cause is S4 mislabelling them, not a paper
that genuinely contains nothing.
"""

import os
import sqlite3
import sys
import tempfile

import pytest

TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)


def _make_db(papers):
    """papers: {paper_id: [section_name, ...] or [(name, body), ...]} -> temp DB."""
    path = tempfile.mktemp(suffix=".db")
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE Paper (id INTEGER PRIMARY KEY, workflow_id INTEGER);
        CREATE TABLE PaperContent (
            id INTEGER PRIMARY KEY, paper_id INTEGER,
            section_name TEXT, content TEXT
        );
        """
    )
    for paper_id, sections in papers.items():
        conn.execute("INSERT INTO Paper VALUES (?, 99)", (paper_id,))
        for entry in sections:
            name, body = entry if isinstance(entry, tuple) else (entry, f"body of {entry}")
            conn.execute(
                "INSERT INTO PaperContent (paper_id, section_name, content) VALUES (?,?,?)",
                (paper_id, name, body),
            )
    conn.commit()
    conn.close()
    return path


def _run(db_path):
    os.environ["SHANI_DB_PATH"] = db_path
    import importlib
    import normalise_paper_content as npc
    importlib.reload(npc)          # re-read DB_PATH from the env
    npc.run_normalisation(99)


def _sections(db_path, paper_id):
    conn = sqlite3.connect(db_path)
    try:
        return [r[0] for r in conn.execute(
            "SELECT section_name FROM PaperContent WHERE paper_id=? ORDER BY id",
            (paper_id,))]
    finally:
        conn.close()


def test_noise_removed_when_real_content_remains():
    db = _make_db({1: ["references", "Results and Discussion"]})
    _run(db)
    assert _sections(db, 1) == ["Results and Discussion"]


def test_paper_of_only_noise_keeps_everything():
    """The regression: all-noise must be a no-op, not a wipe."""
    db = _make_db({1: ["references", "acknowledgements"]})
    _run(db)
    assert sorted(_sections(db, 1)) == ["acknowledgements", "references"]


def test_single_noise_section_survives():
    """One section, and it is noise. Deleting it empties the paper."""
    db = _make_db({1: ["references"]})
    _run(db)
    assert _sections(db, 1) == ["references"]


def test_papers_are_independent():
    """A rescued paper must not stop noise being cleaned from its neighbour."""
    db = _make_db({
        1: ["references", "acknowledgements"],          # all noise -> kept
        2: ["references", "Experimental Methods"],      # mixed     -> cleaned
    })
    _run(db)
    assert sorted(_sections(db, 1)) == ["acknowledgements", "references"]
    assert _sections(db, 2) == ["Methods"] or _sections(db, 2) == ["Experimental Methods"]
    assert "references" not in _sections(db, 2)


def test_no_content_at_all_is_not_an_error():
    db = _make_db({1: []})
    _run(db)
    assert _sections(db, 1) == []


def test_explicit_workflow_id_is_required():
    import normalise_paper_content as npc
    with pytest.raises(ValueError):
        npc.run_normalisation(None)


# ─── idempotency and the preamble rule ────────────────────────────────────────
#
# The second regression, found 2026-09-10. normalise_name() maps a long
# keyword-free heading (a paper title used as a section header) to "preamble",
# and "preamble" was itself on the noise list. Because deletion ran BEFORE
# renaming, such a section survived one run and was deleted on the next — and
# S5 calls run_normalisation() every time it executes. Six sections disappeared
# from workflow 1 that way; paper 20 lost 14 KB of its 17 KB and was then
# skipped by the relevance gate, which reads the preamble.

# A real heading from workflow 1. Long, and carrying none of KEEP_PATTERNS.
# Until 2026-09-11 normalise_name() renamed such headings to "preamble"; they
# now keep their names and front matter is judged by content (D1).
_TITLE_HEADING = "thickness_dependent_dielectric_constant_of_few_layer_in2se3_nano_flakes"


def test_second_run_changes_nothing():
    """The property that was violated: normalisation must reach a fixed point."""
    db = _make_db({1: [_TITLE_HEADING, "Results and Discussion", "references"]})
    _run(db)
    after_first = _sections(db, 1)
    _run(db)
    assert _sections(db, 1) == after_first


def test_title_heading_is_not_deleted_by_a_later_run():
    db = _make_db({1: [_TITLE_HEADING, "Results and Discussion"]})
    _run(db)
    _run(db)
    _run(db)
    assert _TITLE_HEADING in _sections(db, 1)


# ─── Front matter by content (D1, 2026-09-11) ─────────────────────────────────
# Real texts from the S4 dry run of workflow 1.

_P14_FRONT = ("GaAs(111)B Maria Hilse,1,2,∗ Justin Rodriguez,2,3 Jennifer Gray,2 Jinyuan Yao,2,3 Shaoqing\n"
              "Ding,2,3 Derrick Shao Heng Liu,1,2 Mo Li,4 Joshua Young,4 Ying Liu2,3,×, and Roman\n"
              "Engel-Herbert 1,2,5,† 1 Department of Materials Science and Engineering, The Pennsylvania State University,\n"
              "University Park, PA 16802, USA. 2 The Materials Research Institute, The Pennsylvania State University, University Park, PA\n"
              "16802, USA. ∗ mxh752@psu.edu")
_P15_PREAMBLE = ("Fabrication of γ -In2Se3-Based Photodetector\nUsing RF Magnetron Sputtering and\n"
                 "Dependent Properties Yogesh Hase, Vidhika Sharma, Mohit Prasad, Rahul Aher, Shruti Shah, Vidya Doiphode,\n"
                 "Ashish Waghmare, Ashvini Punde, Pratibha Shinde, Swati Rahane, Bharat Bade, Somnath Ladhane,\n"
                 "Habib Pathan, Shashikant P. Patole , and Sandesh R. Jadkar Abstract—Metal chalcogenide indium selenide (In2Se3)\n"
                 "is attracting increasing research interest for photodetector applications. Here, γ -In2Se3 thin\n"
                 "films were prepared at various deposition pressures using the RF magnetron sputtering.")
_P1_SI = ("To fit the time response of the photodetector device in Fig. 6e, a simple exponential behavior was used: "
          "The detailed analysis of the time-resolved surface photovoltage (TR-SPV) is performed following Refs. [3,4].")


def _content(db_path, paper_id, name):
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT content FROM PaperContent WHERE paper_id=? AND section_name=?",
                           (paper_id, name)).fetchone()
        return row and row[0]
    finally:
        conn.close()


def test_front_matter_is_relabelled_not_deleted():
    db = _make_db({1: [("perties_in_inse_films_grown_by_molecular_beam_epitaxy_o", _P14_FRONT),
                       ("Results and Discussion", "x" * 20000)]})
    _run(db)
    assert sorted(_sections(db, 1)) == ["Results and Discussion", "front_matter"]


def test_abstract_is_split_out_of_the_author_block():
    db = _make_db({1: [("preamble", _P15_PREAMBLE), ("methods", "y" * 3000)]})
    _run(db)
    assert sorted(_sections(db, 1)) == ["abstract", "front_matter", "methods"]
    assert _content(db, 1, "abstract").startswith("Abstract—Metal chalcogenide")
    assert "Yogesh Hase" in _content(db, 1, "front_matter")
    after_first = _sections(db, 1)
    _run(db)
    assert _sections(db, 1) == after_first          # still a fixed point


def test_body_section_with_a_long_heading_is_kept():
    """Paper 1's SI section was dropped as 'preamble' because its heading was long."""
    heading = "resolved_surface_photovoltage_tr_spv_and_photodetector_response_tim"
    db = _make_db({1: [(heading, _P1_SI), ("methods", "y" * 3000)]})
    _run(db)
    assert heading in _sections(db, 1)


def test_large_preamble_is_the_paper_and_is_kept():
    """S4 segmentation collapsed: preamble holds the document. Keep it."""
    db = _make_db({1: [("preamble", "x" * 14000), ("methods", "y" * 2600)]})
    _run(db)
    assert sorted(_sections(db, 1)) == ["methods", "preamble"]


def test_preamble_is_kept_when_it_is_the_only_section():
    db = _make_db({1: [("preamble", "short front matter")]})
    _run(db)
    assert _sections(db, 1) == ["preamble"]


# ─── Reference lists inside body sections (D9, 2026-09-11) ────────────────────
_P15_CONCLUSION_HEAD = ("The γ -In2Se3 thin films were successfully synthesized using RF magnetron sputtering. "
                        "The photoresponsivity increases from 0.5 to 4.52 µA/W with an increase in temperature "
                        "from −90 ◦C to +90 ◦C.")
_REF = "[{n}] J.-W. Seo, C. Caneau, R. Bhat, and I. Adesida, “Application of indium-tin-oxide for MSM photodetectors,” IEEE Photon. Technol. Lett., vol. 5, pp. 1313–1315, 1993.\n"
_BIOS = "Ms. Punde has been awarded the Mahatma Jyotiba Phule Research Fellowship."


def test_reference_list_inside_conclusion_is_cut():
    refs = "".join(_REF.format(n=n) for n in range(12, 30))
    db = _make_db({1: [("conclusion", _P15_CONCLUSION_HEAD + " " + refs + _BIOS)]})
    _run(db)
    assert _content(db, 1, "conclusion") == _P15_CONCLUSION_HEAD
    _run(db)
    assert _content(db, 1, "conclusion") == _P15_CONCLUSION_HEAD


def test_in_text_citations_are_not_a_reference_list():
    body = " ".join(f"Films were grown at {300 + n} °C as in [{n}]. The phase was confirmed by XRD [{n + 1}]."
                    for n in range(1, 30))
    db = _make_db({1: [("results", body)]})
    _run(db)
    assert _content(db, 1, "results") == body
