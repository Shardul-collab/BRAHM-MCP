"""
Front matter judged by CONTENT, not by heading shape or size (decision D1,
2026-09-11).

Until then normalisation decided "front matter" in two name/size steps: a
heading of more than 8 tokens with no keyword was renamed "preamble", and a
preamble of <=4,000 chars and <30% of the paper was dropped. A dry run of S4 on
8 workflow-1 papers showed what that measures:

    paper 9   title block + affiliations          1,460  dropped   (right)
    paper 14  author list + affiliations            772  dropped   (right)
    paper 13  affiliations + ABSTRACT             2,963  dropped   (abstract lost)
    paper 15  title + authors + ABSTRACT          2,047  dropped   (abstract lost)
    paper 17  front matter + ABSTRACT             3,512  dropped   (abstract lost)
    paper 1   SI section "Time-resolved SPV..."   1,910  dropped   (body lost)
    paper 1   author list, 8-token heading          778  KEPT      (sent to S5)
    paper 6   author list under a methods heading   469  KEPT      (sent to S5)

4 of 9 full-text papers ended up with none of their abstract in PaperContent.

Here a line is front matter when it carries an e-mail, a metadata label
("Running title", "Corresponding author", ...), an affiliation with an address,
or an author list. A section is front matter when such lines are at least
FRONT_MATTER_MIN_SHARE of its characters. On the 8 dry-run papers (97
sections) every body section scored <= 0.082 and every pure front-matter
section >= 0.827; the mixed paper-13 section scored 0.574 and is handled by
the Abstract-marker split instead.
"""
import re

FRONT_MATTER_MIN_SHARE = 0.5

_AFFIL = re.compile(
    r"\b(Department|Dept\.|University|Universit[äa]t|Institute|Institut|Laborator(?:y|ies)|"
    r"College|School of|Cent(?:er|re)\b|Faculty|Academy|Platform|Consortium|Division|"
    r"Program(?:me)?\b|Leibniz|GmbH|Inc\.)", re.I)
_PLACE = re.compile(
    r"\b(USA|U\.S\.A\.|United States|China|Germany|Japan|Korea|India|Portugal|Belgium|France|"
    r"Spain|Italy|UK|United Kingdom|Saudi Arabia|Taiwan|Singapore|Canada|Australia|"
    r"Switzerland|Netherlands|Sweden)\b|\b[A-Z]{2}\s\d{5}\b|\b\d{5}(?:-\d{4})?\b")
_META = re.compile(
    r"running title|running authors|corresponding author|electronic mail|e-mail|email:|orcid|"
    r"published online|these authors contributed|\*\s*to whom", re.I)
_EMAIL = re.compile(r"\S+@\S+\.\w+")
# "Ryan Trice1," / "Maria C. Tamargo1,2*" — a name carrying an affiliation mark
_MARKED_NAME = re.compile(
    r"\b[A-Z][a-z]+(?:[- ][A-Z]\.)?(?:\s[A-Z][a-z]+)+\s?(?:\d[\d,]*|[a-d]\b|\*|†|,\s*\d)")
# "Yogesh Hase, Vidhika Sharma, Mohit Prasad," — an unmarked comma-separated list
_LISTED_NAME = re.compile(r"\b[A-Z][a-z]+(?:\s[A-Z]\.)?\s[A-Z][a-z]+(?:,|\s+and\b)")

# "Abstract—Metal ...", "ABSTRACT The weak ...", "Abstract: ..."
ABSTRACT_MARKER = re.compile(
    r"(?:(?<=\s)|^)(?:ABSTRACT|Abstract)\s*(?:[:.—–\-]\s*|\n|(?=[A-Z]))")


def is_front_matter_line(line: str) -> bool:
    if _EMAIL.search(line) or _META.search(line):
        return True
    if _AFFIL.search(line) and (_PLACE.search(line) or len(_AFFIL.findall(line)) >= 2):
        return True
    if len(_MARKED_NAME.findall(line)) >= 3 or len(_LISTED_NAME.findall(line)) >= 4:
        return True
    return False


def front_matter_share(text: str) -> float:
    lines = [ln for ln in (text or "").split("\n") if ln.strip()]
    total = sum(len(ln) for ln in lines)
    if not total:
        return 0.0
    return sum(len(ln) for ln in lines if is_front_matter_line(ln)) / total


def is_front_matter(text: str) -> bool:
    return front_matter_share(text) >= FRONT_MATTER_MIN_SHARE


def split_at_abstract(text: str):
    """(before, from_marker_on) if an Abstract marker is present after some text, else None."""
    m = ABSTRACT_MARKER.search(text or "")
    if not m or m.start() == 0 or not text[:m.start()].strip():
        return None
    return text[:m.start()].rstrip(), text[m.start():].lstrip()
