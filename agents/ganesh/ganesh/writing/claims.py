"""
L4 claim plan: the cross-paper reasoning is computed from the evidence before
any model writes (2026-09-11).

The structural finding behind this: small models write fluent summaries but
are weak at critical comparison. So the comparisons a review is judged on -
where papers agree, where their numbers differ, which methods are shared,
what nobody measured - are derived here, deterministically, and the model is
only asked to put a given claim into prose with the evidence attached.

Heuristic constants are marked as such; they decide which claims get a
paragraph, never whether a number is true (that is the verifier's job).
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional

from ganesh.writing.grounding import Evidence, fold

NUMERIC_CATEGORIES = {
    "growth_temperature", "growth_duration", "chamber_pressure", "gas_flow", "growth_flux",
    "annealing_condition", "doping_parameter", "optical_property", "electrical_property",
    "ferroelectric_property", "photodetector_metric", "photoresponsivity", "field_effect_mobility",
    "on_off_ratio", "threshold_voltage", "subthreshold_swing", "contact_resistance",
}
DISAGREEMENT_RATIO = 1.5      # heuristic: max/min above this is worth discussing
MAX_EVIDENCE_PER_CLAIM = 8

_PHASE_RE = re.compile(r"(?<![A-Za-z])(α|β′|β'|β|γ|δ|κ|ε|alpha|beta[- ]prime|beta|gamma|delta|kappa|epsilon|3R|2H|wurtzite)"
                       r"\s*[-‐]?\s*(?=In2Se3|InSe|In₂Se₃|phase|polymorph|polytype|\s)", re.I)
_PHASE_NORM = {"alpha": "α", "beta": "β", "beta prime": "β′", "beta-prime": "β′", "β'": "β′",
               "gamma": "γ", "delta": "δ", "kappa": "κ", "epsilon": "ε"}
_NUMBER = re.compile(r"(\d+(?:\.\d+)?)\s*(?:[x×]\s*10\s*\^?\s*([-−]?\d+))?")
_UNITS = [  # (canonical, pattern) - searched after the first number
    ("°C", r"°\s*C|℃|\bdeg(?:rees)?\s*C\b"), ("K", r"\bK\b"),
    ("Torr", r"\bTorr\b"), ("mbar", r"\bmbar\b"), ("Pa", r"\bPa\b"),
    ("sccm", r"\bsccm\b"), ("cm⁻²s⁻¹", r"cm\s*\^?\s*-?2\s*s\s*\^?\s*-?1|cm-2\s*s-1"),
    ("Å/s", r"Å\s*/\s*s|A\s*/\s*s\b"), ("nm/min", r"nm\s*/\s*min"),
    ("min", r"\bmin(?:utes?)?\b"), ("h", r"\bh(?:ours?)?\b"), ("ms", r"\bms\b"), ("μs", r"[μµu]s\b"),
    ("s", r"\bs(?:econds?)?\b"),
    ("eV", r"\beV\b"), ("nm", r"\bnm\b"), ("μm", r"[μµu]m\b"),
    ("A/W", r"[mμµ]?A\s*/\s*W"), ("Jones", r"\bJones\b"), ("pm/V", r"pm\s*/\s*V"),
    ("μC/cm²", r"[μµu]C\s*/?\s*cm"), ("kV/cm", r"kV\s*/\s*cm"), ("V", r"\bV\b"),
    ("cm²/Vs", r"cm\s*2\s*/\s*V|cm²\s*/\s*V"), ("mV/dec", r"mV\s*/\s*dec"), ("A", r"[pnμµm]?A\b"), ("%", r"%"),
]
# Broad categories hold several quantities; a range is only stated within one quantity.
BROAD_CATEGORIES = {"optical_property", "electrical_property", "ferroelectric_property", "photodetector_metric"}
_QUANTITY = [
    ("band gap", r"band\s*-?\s*gap|optical gap|\bE\s*g\b"), ("PL peak", r"\bPL\b|photolumin"),
    ("dielectric constant", r"dielectric|permittiv|ε\s*r"), ("refractive index", r"refractive"),
    ("detectivity", r"detectivity|jones"), ("response time", r"\brise|\bdecay|response time|τ"),
    ("dark current", r"dark current"), ("photocurrent", r"photocurrent"), ("mobility", r"mobility"),
    ("carrier density", r"carrier (?:density|concentration)|cm\s*-\s*3|cm⁻³"),
    ("remanent polarization", r"polari[sz]ation|μC"), ("coercive field", r"coercive"),
    ("piezoelectric coefficient", r"\bd33\b|pm\s*/\s*V"), ("resistivity", r"resistiv"),
]
DIMENSIONLESS_OK = {"dielectric constant", "refractive index"}


def quantity_of(e) -> str:
    text = fold(e.value)
    return next((q for q, pat in _QUANTITY if re.search(pat, text, re.I)), "")
# Heuristic evidence hygiene: a claim is only built from rows whose unit fits
# the category (the extractor sometimes files '1.5 nm [roughness]' as a
# growth_duration). Rows that do not fit are left out of L4 claims only.
_UNIT_FITS = {
    "growth_temperature": {"°C", "K"}, "annealing_condition": {"°C", "K", "min", "h", "s"},
    "growth_duration": {"min", "h", "s"}, "chamber_pressure": {"Torr", "mbar", "Pa"},
    "gas_flow": {"sccm"}, "growth_flux": {"cm⁻²s⁻¹", "Å/s", "nm/min", "Torr", "mbar", "Pa", ""},
    "photoresponsivity": {"A/W"}, "field_effect_mobility": {"cm²/Vs"}, "subthreshold_swing": {"mV/dec"},
    "on_off_ratio": {""},
}
PROPERTY_PATTERNS = {   # corpus-level gap detection for configured properties
    "ferroelectricity": r"ferroelectric|polari[sz]ation|piezo|\bd33\b|coercive",
    "photoresponsivity": r"responsivity|photoresponse|a\s*/\s*w",
    "bandgap": r"band\s*-?gap|optical gap",
}


@dataclass
class Claim:
    kind: str                     # range | disagreement | consensus | single | gap
    statement: str
    evidence: List[Evidence] = field(default_factory=list)
    note: str = ""

    @property
    def papers(self):
        return sorted({e.paper_id for e in self.evidence})


def phase_of(e: Evidence) -> Optional[str]:
    m = _PHASE_RE.search(fold(f"{e.value} {e.sentence}"))
    if not m:
        return None
    p = m.group(1).lower()
    return _PHASE_NORM.get(p, m.group(1))


_PREFIX = {"p": 1e-12, "n": 1e-9, "μ": 1e-6, "µ": 1e-6, "u": 1e-6, "m": 1e-3}
_RANGE_RE = re.compile(r"^\s*[~∼≈]?\s*(\d+(?:\.\d+)?)\s*(?:to|-|–|—)\s*[~∼≈]?\s*(\d+(?:\.\d+)?)")


def _scaled(x: float, canon: str, rest: str):
    """Prefixed current/responsivity/time units -> base unit (2.82 µA/W -> 2.82e-6 A/W)."""
    if canon in ("A/W", "A"):
        m = re.search(r"([pnμµum]?)A\b", rest)
        return x * _PREFIX.get(m.group(1), 1.0) if m and m.group(1) else x, canon
    if canon == "ms":
        return x * 1e-3, "s"
    if canon == "μs":
        return x * 1e-6, "s"
    return x, canon


def number_and_unit(value: str):
    x, _, unit = number_range_and_unit(value)
    return x, unit


def number_range_and_unit(value: str):
    """(low, high, unit) of a value; high == low unless the value is a range like '150 to 550 °C'."""
    core = re.sub(r"\[[^\]]*\]", " ", fold(value))
    m = _NUMBER.search(core)
    if not m:
        return None, None, ""
    x = float(m.group(1)) * (10 ** int(m.group(2).replace("−", "-")) if m.group(2) else 1)
    hi = x
    r = _RANGE_RE.match(core[m.start():])
    if r and not m.group(2):
        hi = float(r.group(2))
    rest = core[m.start():]
    for canon, pat in _UNITS:
        if re.search(pat, rest):
            lo_s, unit = _scaled(x, canon, rest)
            hi_s, _ = _scaled(hi, canon, rest)
            return lo_s, hi_s, unit
    return x, hi, ""


def _label(v: str) -> str:
    return re.sub(r"\s*\[[^\]]*\]", "", v).strip().lower()


def _span_selection(items):
    """At most MAX_EVIDENCE_PER_CLAIM items that keep the true minimum and maximum and as many
    papers as possible (was: the 8 lowest values, so the stated maximum was not the maximum)."""
    items = sorted(items, key=lambda t: (t[0], t[1]))
    if len(items) <= MAX_EVIDENCE_PER_CLAIM:
        return items
    lo = items[0]
    hi = max(items, key=lambda t: t[1])
    chosen = [lo] + ([hi] if hi is not lo else [])
    seen = {t[2].paper_id for t in chosen}
    for t in items:
        if len(chosen) >= MAX_EVIDENCE_PER_CLAIM:
            break
        if t not in chosen and t[2].paper_id not in seen:
            chosen.append(t); seen.add(t[2].paper_id)
    for t in items:
        if len(chosen) >= MAX_EVIDENCE_PER_CLAIM:
            break
        if t not in chosen:
            chosen.append(t)
    return sorted(chosen, key=lambda t: (t[0], t[1]))


def build_claims(evidence: List[Evidence], gaps: dict | None = None) -> List[Claim]:
    """gaps: {property: number_of_papers_with_evidence} computed corpus-wide by the caller."""
    claims: List[Claim] = []
    numeric = defaultdict(list)
    categorical = defaultdict(list)
    single_by_paper = defaultdict(list)
    for e in evidence:
        x, hi, unit = number_range_and_unit(e.value)
        if e.category in NUMERIC_CATEGORIES and x is not None:
            if e.category in _UNIT_FITS and unit not in _UNIT_FITS[e.category]:
                continue
            qty = quantity_of(e) if e.category in BROAD_CATEGORIES else e.category.replace("_", " ")
            if (e.category in BROAD_CATEGORIES and not qty) or \
               (not unit and e.category != "on_off_ratio" and qty not in DIMENSIONLESS_OK):
                # no recognised quantity or unit: nothing says these values are commensurable
                # ("7 ms" and "1x1014 cm-2" were pooled as one electrical-property range)
                single_by_paper[e.paper_id].append(e)
                continue
            numeric[(qty, unit)].append((x, hi, e))
        elif e.category not in NUMERIC_CATEGORIES:
            categorical[(e.category, _label(e.value))].append(e)

    for (cat, unit), items in sorted(numeric.items(), key=lambda kv: -len({t[2].paper_id for t in kv[1]})):
        papers = {t[2].paper_id for t in items}
        if len(papers) < 2:
            single_by_paper[next(iter(papers))].extend(t[2] for t in items)
            continue
        all_items = items
        items = _span_selection(items)
        evs = [t[2] for t in items]
        lo_item = min(items, key=lambda t: t[0])
        hi_item = max(items, key=lambda t: t[1])
        lo, hi = lo_item[0], hi_item[1]
        name = cat
        core = lambda e: e.value.split('[')[0].strip()
        papers = {e.paper_id for e in evs}
        # the extremes carry their IDs: hosted writers (Groq, 2026-09-11 pm) restated the claim as an
        # uncited topic sentence ("span 150 °C to 950 °C across the literature") that failed the verifier
        stmt = (f"Across {len(papers)} papers, reported {name} values span {core(lo_item[2])} [{lo_item[2].eid}]"
                + (f" to {core(hi_item[2])} [{hi_item[2].eid}]." if hi_item is not lo_item else "."))
        phases = sorted({p for p in (phase_of(e) for e in evs) if p})
        if phases:
            stmt += " The evidence associates values with phases: " + ", ".join(phases) + "."
        kind, note = "range", ""
        # between-paper spread uses each value's midpoint, so one paper's own range
        # ("150 to 550 °C") does not read as a disagreement between papers
        mids = [(t[0] + t[1]) / 2 for t in items]
        if min(mids) > 0 and max(mids) / min(mids) > DISAGREEMENT_RATIO:
            kind = "disagreement"
            note = ("The values differ substantially between papers. Explain the difference only with conditions "
                    "stated in the evidence (substrate, phase, method, thickness); if the evidence gives no reason, "
                    "say the reports differ without speculating.")
        claims.append(Claim(kind, stmt, evs, note))
        # values beyond the cap used to be dropped from the plan (13 of 36 Synthesis items, 2026-09-11)
        rest = sorted((t for t in all_items if t not in items), key=lambda t: (t[0], t[1]))
        for i in range(0, len(rest), MAX_EVIDENCE_PER_CLAIM):
            claims.append(Claim("range", f"Describe further reports of {name}; each lies within the range "
                                         f"discussed in the previous paragraph.",
                                [t[2] for t in rest[i:i + MAX_EVIDENCE_PER_CLAIM]]))

    for (cat, label), evs in sorted(categorical.items(), key=lambda kv: -len({e.paper_id for e in kv[1]})):
        papers = {e.paper_id for e in evs}
        if len(papers) >= 2:
            one_per_paper = list({e.paper_id: e for e in evs}.values())[:MAX_EVIDENCE_PER_CLAIM]
            claims.append(Claim("consensus",
                                f"{len(papers)} papers report the {cat.replace('_', ' ')} '{label}'.", one_per_paper))
        else:
            single_by_paper[next(iter(papers))].append(evs[0])

    for pid, evs in sorted(single_by_paper.items(), key=lambda kv: -len(kv[1])):
        for i in range(0, len(evs), MAX_EVIDENCE_PER_CLAIM):
            claims.append(Claim("single", "Describe these findings, which all come from one paper, as that "
                                          "paper's results. Do not compare them with other work.",
                                evs[i:i + MAX_EVIDENCE_PER_CLAIM]))

    for prop, n in (gaps or {}).items():
        if n <= 1:
            claims.append(Claim("gap", f"Quantitative evidence on {prop} is "
                                       f"{'absent' if n == 0 else 'limited to a single paper'} in this corpus.", []))
    return claims


def corpus_property_coverage(conn, workflow_id: int, properties) -> dict:
    """Papers with any knowledge row matching each configured property."""
    rows = conn.execute("SELECT k.paper_id, k.category || ' ' || k.value || ' ' || COALESCE(k.sentence,'') "
                        "FROM ResearchKnowledge k JOIN Paper p ON p.id = k.paper_id WHERE p.workflow_id = ?",
                        (workflow_id,)).fetchall()
    out = {}
    for prop in properties:
        pat = re.compile(PROPERTY_PATTERNS.get(prop.lower(), re.escape(prop.lower())), re.I)
        out[prop] = len({pid for pid, t in rows if pat.search(fold(t))})
    return out
