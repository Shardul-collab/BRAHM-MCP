"""
ganesh/document_types/manuscript_draft.py
"""
MANUSCRIPT_DRAFT_SECTIONS = [
    {"section_name": "Abstract",           "section_type": "abstract",   "depends_on": [],                      "target_word_count": 300},
    {"section_name": "Introduction",       "section_type": "intro",      "depends_on": ["Abstract"],            "target_word_count": 800},
    {"section_name": "Methods",            "section_type": "body",       "depends_on": ["Introduction"],        "target_word_count": 900},
    {"section_name": "Results",            "section_type": "body",       "depends_on": ["Methods"],             "target_word_count": 1000},
    {"section_name": "Discussion",         "section_type": "body",       "depends_on": ["Results"],             "target_word_count": 800},
    {"section_name": "Conclusion",         "section_type": "conclusion", "depends_on": ["Discussion"],          "target_word_count": 400},
]
