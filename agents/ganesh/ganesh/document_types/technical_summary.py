"""
ganesh/document_types/technical_summary.py
"""
TECHNICAL_SUMMARY_SECTIONS = [
    {"section_name": "Overview",           "section_type": "intro",      "depends_on": [],              "target_word_count": 300},
    {"section_name": "Key Findings",       "section_type": "body",       "depends_on": ["Overview"],    "target_word_count": 600},
    {"section_name": "Technical Details",  "section_type": "body",       "depends_on": ["Key Findings"],"target_word_count": 800},
    {"section_name": "Recommendations",    "section_type": "conclusion", "depends_on": ["Technical Details"], "target_word_count": 400},
]
