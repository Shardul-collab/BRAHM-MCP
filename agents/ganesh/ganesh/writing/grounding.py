"""
Deterministic grounding verifier for GANESH prose (2026-09-11).

The TRL-5 bar for BRAHM is that >=90% of the review's cited claims are
supported by their source and no value is fabricated. Until now grounding was
a prompt instruction ("do not invent numbers") checked by an LLM critic of the
same kind as the writer, and the evidence given to the writer carried no paper
IDs, so nothing it wrote could be traced.

Here every evidence line has an ID ([E14]) bound to one paper and one source
sentence, the writer must cite IDs, and this module checks - with no model -
that each number and each chemical formula in a sentence appears in the
evidence that sentence cites.

Sentence statuses:
  supported     - makes a checkable claim, cites evidence, every number and
                  formula is found in the cited evidence
  unsupported   - cites evidence, but a number or formula is not in it
  uncited       - contains a number or formula and cites nothing
  bad_citation  - cites an ID that does not exist
  narrative     - no number, no formula, no citation (transitions, framing)
  cited_text    - cites evidence, no number/formula to check (qualitative claim;
                  counted as cited, verifiable only by the audit)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

# º (ordinal indicator) is how several PDFs write the degree sign: "950 ºC"
_FOLD = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉⁻−–º", "01234567890123456789---°")
_CITE_RE = re.compile(r"\[(E\d+(?:\s*[,;]\s*E\d+)*)\]")
_FORMULA_RE = re.compile(r"(?<![A-Za-z0-9])((?:[A-Z][a-z]?\d{0,2}){2,6})(?![a-z0-9])")
_NUMBER_RE = re.compile(r"(?<![A-Za-z\d.])(\d+(?:\.\d+)?)")
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
_ELEMENTS = set("""H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As
Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb
Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi""".split())


def fold(text: str) -> str:
    return (text or "").translate(_FOLD)


def normalise_output(text: str) -> str:
    """
    Plain-text formulas and units from model output. gemma4:12b writes
    $\\text{In}_2\\text{Se}_3$ (measured 2026-09-11); the verifier, the claim
    plan and a reader of plain text all expect In2Se3.
    """
    t = text or ""
    # stage A 2026-09-11 pm: gemma4 also wrote \kappa and \tau_{spv}
    for name, ch in (("kappa", "κ"), ("tau", "τ"), ("delta", "δ"), ("epsilon", "ε"), ("varepsilon", "ε"),
                     ("lambda", "λ"), ("sigma", "σ"), ("rho", "ρ"), ("theta", "θ"), ("omega", "ω"), ("Omega", "Ω")):
        t = re.sub(r"\\" + name + r"(?![A-Za-z])", ch, t)
    t = re.sub(r"\\(?:text|mathrm|rm)\{([^{}]*)\}", r"\1", t)
    t = re.sub(r"_\{([^{}]*)\}", r"\1", t)
    t = re.sub(r"\^\{([^{}]*)\}", r"^\1", t)
    t = re.sub(r"(?<=[A-Za-z])_(\d+)", r"\1", t)
    t = t.replace("\\circ", "°").replace("^°", "°").replace("\\times", "×").replace("\\mu", "μ")
    t = t.replace("\\alpha", "α").replace("\\beta", "β").replace("\\gamma", "γ").replace("\\,", " ")
    t = re.sub(r"\$([^$]{0,80})\$", r"\1", t)
    # hosted models (Groq, 2026-09-11 pm) add markdown emphasis and narrow no-break spaces
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t).replace("\u202f", " ").replace("\u2009", " ")
    return re.sub(r"[ \t]{2,}", " ", t)


# Every Roman numeral character is also an element symbol (I iodine, V vanadium,
# C carbon, ...), so "Phase II" parsed as iodine-iodine and "Phase IV" as
# iodine-vanadium. Found 2026-09-20 when the tightened subject check flagged
# "the peak temperature for Phase II ... Phase IV" for missing formulas.
# Only STRICT Roman numerals are excluded, so real compounds built from the same
# letters survive: VC (vanadium carbide) and CV are not valid Roman numerals.
_ROMAN_RE = re.compile(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")


def is_roman_numeral(tok: str) -> bool:
    return bool(tok) and tok.isupper() and bool(_ROMAN_RE.fullmatch(tok))


def is_formula(tok: str) -> bool:
    if is_roman_numeral(tok):
        return False
    parts = re.findall(r"([A-Z][a-z]?)(\d{0,2})", tok)
    return ("".join(a + b for a, b in parts) == tok and len(parts) >= 2
            and all(a in _ELEMENTS for a, _ in parts))


@dataclass
class Evidence:
    eid: str                     # "E14"
    paper_id: int
    category: str
    value: str
    sentence: str = ""
    source: str = "full text"    # or "abstract"
    title: str = ""
    year: Optional[int] = None
    doi: str = ""
    paper_formulas: str = ""     # formulas in the paper's title + abstract (material context)

    @property
    def text(self) -> str:
        return fold(f"{self.value} {self.sentence}")

    @property
    def formula_text(self) -> str:
        """Material context for this row: what the PAPER wrote, at the level of
        this row.

        2026-09-20, from the TRL-5 audit. Two things used to be in here and are
        deliberately not any more:

        `self.value` — a row's own extracted value is BRAHM's label for the row,
        not a sentence the paper wrote. E236's value is literally "α-In2Se3"
        while its evidence sentence is about Bi2Se3, so the row satisfied the
        subject-formula check by itself and "α-In2Se3 on B substrates displayed
        anti-phase domains" passed. That was 1 of the 2 UNSUPPORTED claims in
        the 50-claim audit.

        `self.paper_formulas` (title + abstract) — this says "the paper mentions
        In2Se3 somewhere", which is a paper-level fact being used to license a
        sentence-level claim. Paper 6 IS an In2Se3 paper, which is why dropping
        `value` alone did not catch the sentence above.

        The title stays: it states what the reported work is.

        Measured on document 2 (372 supported before): dropping `value` alone
        flags 3, dropping both flags 4, and the 4th is the audit's claim 5.
        Sentence-only flags 19 and is too strict.
        """
        return fold(f"{self.sentence} {self.title}")


@dataclass
class SentenceCheck:
    text: str
    status: str
    cited: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)   # numbers / formulas not found


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_RE.split((text or "").strip()) if s.strip()]


def mask_formulas(text: str) -> str:
    """Blank out chemical formulas so their digits (the 3 in In2Se3) are never
    read as reported values - on the claim side and on the evidence side."""
    return _FORMULA_RE.sub(lambda m: " " if is_formula(m.group(1)) else m.group(0), text)


def formulas_in(text: str) -> set:
    return {f for f in _FORMULA_RE.findall(text) if is_formula(f)}


def claims_in(sentence: str):
    """(numbers, formulas) the sentence asserts, citation markers removed."""
    s = fold(_CITE_RE.sub(" ", sentence))
    formulas = [f for f in _FORMULA_RE.findall(s) if is_formula(f)]
    s_wo = mask_formulas(s)
    s_wo = re.sub(r"\b\d+(?:st|nd|rd|th)\b", " ", s_wo)          # ordinals
    s_wo = re.sub(r"\b(?:19|20)\d{2}\b", " ", s_wo)               # years
    numbers = _NUMBER_RE.findall(s_wo)
    return numbers, formulas


def _num_in(n: str, text: str) -> bool:
    return re.search(r"(?<![\d.])" + re.escape(n) + r"(?![\d])", text) is not None


def check_sentence(sentence: str, evidence: Dict[str, Evidence],
                   always_allowed: Iterable[str] = ()) -> SentenceCheck:
    cited = [e.strip() for grp in _CITE_RE.findall(sentence) for e in re.split(r"[,;]", grp)]
    numbers, formulas = claims_in(sentence)
    if not cited:
        formulas = [f for f in formulas if f not in set(always_allowed)]
    if any(c not in evidence for c in cited):
        return SentenceCheck(sentence, "bad_citation", cited,
                             [c for c in cited if c not in evidence])
    if not numbers and not formulas:
        return SentenceCheck(sentence, "cited_text" if cited else "narrative", cited)
    if not cited:
        return SentenceCheck(sentence, "uncited", cited, numbers + formulas)
    pool = " ".join(evidence[c].text for c in cited)
    num_pool = mask_formulas(pool)
    pool_formulas = formulas_in(" ".join(evidence[c].formula_text for c in cited))
    missing = [n for n in numbers if not _num_in(n, num_pool)] + [f for f in formulas if f not in pool_formulas]
    return SentenceCheck(sentence, "unsupported" if missing else "supported", cited, missing)


def check_text(text: str, evidence: Dict[str, Evidence], always_allowed: Iterable[str] = ()):
    return [check_sentence(s, evidence, always_allowed) for s in split_sentences(text)]


def summarise(checks: List[SentenceCheck]) -> dict:
    n = {k: 0 for k in ("supported", "unsupported", "uncited", "bad_citation", "narrative", "cited_text")}
    for c in checks:
        n[c.status] += 1
    checkable = n["supported"] + n["unsupported"] + n["uncited"] + n["bad_citation"]
    n["checkable"] = checkable
    n["grounding_rate"] = (n["supported"] / checkable) if checkable else 1.0
    return n


def failing(checks: List[SentenceCheck]) -> List[SentenceCheck]:
    return [c for c in checks if c.status in ("unsupported", "uncited", "bad_citation")]


def strip_failing(text: str, checks: List[SentenceCheck]) -> str:
    bad = {c.text for c in failing(checks)}
    return " ".join(s for s in split_sentences(text) if s not in bad)
