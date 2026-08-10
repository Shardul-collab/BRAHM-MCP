"""
ganesh/document_types/research_report.py
"""
RESEARCH_REPORT_SECTIONS = [
    {"section_name": "Executive Summary",  "section_type": "intro",      "depends_on": [],                      "target_word_count": 400},
    {"section_name": "Introduction",       "section_type": "intro",      "depends_on": ["Executive Summary"],   "target_word_count": 600},
    {"section_name": "Methodology",        "section_type": "body",       "depends_on": ["Introduction"],        "target_word_count": 700},
    {"section_name": "Results",            "section_type": "body",       "depends_on": ["Methodology"],         "target_word_count": 900},
    {"section_name": "Discussion",         "section_type": "body",       "depends_on": ["Results"],             "target_word_count": 700},
    {"section_name": "Conclusion",         "section_type": "conclusion", "depends_on": ["Discussion"],          "target_word_count": 400},
]
