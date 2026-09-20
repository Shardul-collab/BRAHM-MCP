"""
Keyword matching for the rule-based extractors (S2_75 and S5).

Both extractors used to test `keyword in text` — a raw substring test. For the
short acronyms that make up most of the characterisation vocabulary that is
wrong, and measurably so on the live In2Se3 corpus (2026-09-11):

    "bet"  matched "between" / "beta"        -> BET on 9 of 9 full-text papers,
                                                none of which used BET
    "tem"  matched "temperature" / "system"  -> TEM
    "sem"  matched "semiconductor"           -> SEM
    "led"  matched "controlled" / "revealed" -> LED as an application
    "ups"  matched "groups", "eds" "hundreds", "tga" "outgassed",
    "aes"  matched "Alfa Aesar"

28 of the 92 abstract-path rows and 27 S5 rule rows had no whole-word support
for their label anywhere in the text they were extracted from.

Rule: a short alphanumeric keyword (an acronym or a short word) must match as
a whole word, optionally pluralised ("TEMs"). Longer keywords are left as
substring matches, because several are deliberately truncated stems
("infrared spectroscop", "photoluminescen") and a substring of 7+ letters does
not collide with ordinary English the way "tem" does. Keywords that carry their
own padding (" cv ", " pl ") or punctuation ("v_se") keep substring behaviour.
"""
import re
from functools import lru_cache

SHORT_KEYWORD_MAX = 6


@lru_cache(maxsize=4096)
def _word_re(keyword: str):
    return re.compile(r"(?<![a-z0-9])" + re.escape(keyword) + r"s?(?![a-z0-9])")


def keyword_in(keyword: str, text_lower: str) -> bool:
    """True when `keyword` (lower-case) occurs in `text_lower` (lower-case)."""
    if keyword.isalnum() and len(keyword) <= SHORT_KEYWORD_MAX:
        return _word_re(keyword).search(text_lower) is not None
    return keyword in text_lower
