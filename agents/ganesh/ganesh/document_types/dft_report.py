"""
ganesh/document_types/dft_report.py
"""
DFT_REPORT_SECTIONS = [
    {"section_name": "Introduction",        "section_type": "intro",      "depends_on": [],                       "target_word_count": 500},
    {"section_name": "Computational Setup", "section_type": "body",       "depends_on": ["Introduction"],         "target_word_count": 700},
    {"section_name": "Crystal Structure",   "section_type": "body",       "depends_on": ["Computational Setup"],  "target_word_count": 600},
    {"section_name": "Electronic Structure","section_type": "body",       "depends_on": ["Crystal Structure"],    "target_word_count": 800},
    {"section_name": "Band Structure",      "section_type": "body",       "depends_on": ["Electronic Structure"], "target_word_count": 700},
    {"section_name": "Density of States",   "section_type": "body",       "depends_on": ["Band Structure"],       "target_word_count": 700},
    {"section_name": "Phonon Properties",   "section_type": "body",       "depends_on": ["Density of States"],    "target_word_count": 600},
    {"section_name": "Discussion",          "section_type": "body",       "depends_on": ["Phonon Properties"],    "target_word_count": 700},
    {"section_name": "Conclusion",          "section_type": "conclusion", "depends_on": ["Discussion"],           "target_word_count": 400},
]
