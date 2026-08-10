"""
ganesh/document_types/literature_review.py
"""
LITERATURE_REVIEW_SECTIONS = [
    {"section_name": "Introduction",         "section_type": "intro",       "depends_on": [],                        "target_word_count": 600},
    {"section_name": "Background",           "section_type": "body",        "depends_on": ["Introduction"],          "target_word_count": 800},
    {"section_name": "Materials Overview",   "section_type": "body",        "depends_on": ["Background"],            "target_word_count": 700},
    {"section_name": "Synthesis Methods",    "section_type": "body",        "depends_on": ["Materials Overview"],    "target_word_count": 900},
    {"section_name": "Characterization",     "section_type": "body",        "depends_on": ["Synthesis Methods"],     "target_word_count": 900},
    {"section_name": "Properties & Results", "section_type": "body",        "depends_on": ["Characterization"],      "target_word_count": 1000},
    {"section_name": "Discussion",           "section_type": "body",        "depends_on": ["Properties & Results"],  "target_word_count": 800},
    {"section_name": "Research Gaps",        "section_type": "body",        "depends_on": ["Discussion"],            "target_word_count": 600},
    {"section_name": "Conclusion",           "section_type": "conclusion",  "depends_on": ["Research Gaps"],         "target_word_count": 500},
]
