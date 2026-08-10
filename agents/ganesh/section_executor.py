"""
ganesh/section_executor.py
===========================
SectionExecutor: the write → critic → reviser loop for a single section.
"""

from __future__ import annotations

import json
import re
import traceback
from datetime import datetime
from typing import Dict, List, Optional

from ganesh.section_graph import SectionNode, SectionStatus


def _sanitize_json_escapes(text: str) -> str:
    """
    Escape backslashes not part of a valid JSON escape sequence.
    Handles LLM critic output that echoes LaTeX-style notation
    (e.g. \\(900^\\circ\\)C) inside JSON string values, which breaks json.loads().
    Ported from agents/ganesh/ganesh/section_executor.py (the unused nested
    copy) -- that copy had this fix, the live file did not.
    """
    return re.sub(r'\\(?!["\\\\/bfnrtu])', r'\\\\', text)


DEFAULT_QUALITY_THRESHOLD = 7.5
DEFAULT_MAX_ITERATIONS    = 4

QUALITY_DIMENSIONS = [
    "scientific_accuracy",
    "narrative_coherence",
    "brief_compliance",
    "evidence_integration",
    "logical_progression",
    "cross_section_fit",
]

DEFAULT_DIMENSION_WEIGHTS = {
    "scientific_accuracy":   0.25,
    "narrative_coherence":   0.20,
    "brief_compliance":      0.20,
    "evidence_integration":  0.15,
    "logical_progression":   0.10,
    "cross_section_fit":     0.10,
}


class CritiqueRecord:
    def __init__(self, scores, issues, overall_score, weights=None):
        self.scores        = scores
        self.issues        = issues
        self.overall_score = overall_score
        self.weights       = weights or DEFAULT_DIMENSION_WEIGHTS

    @classmethod
    def from_llm_output(cls, raw: dict, weights=None) -> "CritiqueRecord":
        w = weights or DEFAULT_DIMENSION_WEIGHTS
        scores = raw.get("scores", {})
        overall = sum(
            scores.get(dim, 0.0) * w.get(dim, 0.0)
            for dim in QUALITY_DIMENSIONS
        )
        return cls(
            scores        = scores,
            issues        = raw.get("issues", []),
            overall_score = round(overall, 2),
            weights       = w,
        )

    def actionable_issues(self, min_severity: int = 5) -> List[Dict]:
        return [i for i in self.issues if i.get("severity", 10) >= min_severity]


class SectionExecutor:
    def __init__(
        self,
        repo,
        document_id:       int,
        context_bundle:    dict,
        llm_client=None,
        quality_threshold: float  = DEFAULT_QUALITY_THRESHOLD,
        max_iterations:    int    = DEFAULT_MAX_ITERATIONS,
        critic_persona:    Optional[str] = None,
    ):
        self.repo              = repo
        self.document_id       = document_id
        self.context_bundle    = context_bundle
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            from ganesh.llm_client import call_llm as _default_llm
            self.llm_client = _default_llm
        self.quality_threshold = quality_threshold
        self.max_iterations    = max_iterations
        self.critic_persona    = critic_persona or "scientific peer reviewer"

    def run(self, section: SectionNode) -> None:
        section_id   = section.section_id
        section_name = section.section_name
        brief        = section.brief

        print(f"\n[GANESH] Section Executor: {section_name}")

        best_draft_id    = None
        best_score       = 0.0
        iteration        = 0
        current_draft_id = None

        prior_sections = self._get_prior_approved_sections()

        # Evidence is fixed for this section across all iterations — compute
        # once and pass identically to writer, reviser, and critic, so the
        # critic checks against exactly what the writer was given.
        evidence_summary = self._format_section_evidence(brief.get("section_name", ""))

        while iteration < self.max_iterations:

            iteration += 1
            print(f"  [{section_name}] Iteration {iteration}/{self.max_iterations}")

            self._update_section_status(section_id, SectionStatus.DRAFTING)

            draft_content = self._call_writer(
                brief             = brief,
                prior_sections    = prior_sections,
                previous_draft    = self._get_draft_content(current_draft_id),
                critique          = self._get_critique(current_draft_id) if current_draft_id else None,
                evidence_summary  = evidence_summary,
            )

            current_draft_id = self._save_draft(section_id, iteration, draft_content)

            self._update_section_status(section_id, SectionStatus.UNDER_REVIEW)

            critique_raw = self._call_critic(
                section_brief     = brief,
                draft_content     = draft_content,
                prior_sections    = prior_sections,
                evidence_summary  = evidence_summary,
            )

            critique = CritiqueRecord.from_llm_output(critique_raw)
            self._save_critique(section_id, current_draft_id, critique)

            print(f"  [{section_name}] Critic score: {critique.overall_score:.1f} / 10.0")

            if critique.overall_score > best_score:
                best_score    = critique.overall_score
                best_draft_id = current_draft_id

            if critique.overall_score >= self.quality_threshold:
                print(f"  [{section_name}] ✅ Threshold met ({critique.overall_score:.1f} ≥ {self.quality_threshold})")
                self._approve_section(section_id, best_draft_id, best_score)
                return {
                    'approved': True,
                    'final_score': best_score,
                    'iterations': iteration,
                    'below_threshold': False,
                }

            if iteration < self.max_iterations:
                self._update_section_status(section_id, SectionStatus.REVISING)
                prev_draft_id = current_draft_id

                revised_content = self._call_reviser(
                    draft_content     = draft_content,
                    critique          = critique,
                    brief             = brief,
                    evidence_summary  = evidence_summary,
                )

                current_draft_id = self._save_draft(section_id, iteration + 1, revised_content)
                self._save_revision(
                    section_id    = section_id,
                    from_draft_id = prev_draft_id,
                    to_draft_id   = current_draft_id,
                    critique_id   = self._get_latest_critique_id(section_id),
                    summary       = f"Applied {len(critique.actionable_issues())} critique issues",
                )

        print(
            f"  [{section_name}] ⚠️  Max iterations reached. "
            f"Approving best draft (score={best_score:.1f})."
        )
        self._approve_section(section_id, best_draft_id, best_score, below_threshold=True)
        return {
            'approved': True,
            'final_score': best_score,
            'iterations': iteration,
            'below_threshold': True,
        }

    def _call_writer(
        self,
        brief:             dict,
        prior_sections:    List[dict],
        previous_draft:    Optional[str],
        critique:          Optional[dict],
        evidence_summary:  str,
    ) -> str:

        prior_context = "\n\n".join(
            f"[{s['section_name']} — already approved]\n{s['content']}"
            for s in prior_sections
        )

        grounding_rule = """GROUNDING RULE — READ CAREFULLY:
You must not invent, estimate, or substitute typical/textbook values for any
specific number (mobility, voltage, temperature, ratio, peak position,
thickness, etc.). Use ONLY numeric values that appear, verbatim or as a
direct paraphrase, in the AVAILABLE EVIDENCE below.

BEFORE writing, scan every line of AVAILABLE EVIDENCE for concrete numbers,
materials, and methods. If ANY specific values are present anywhere in the
evidence (temperatures, mobilities, ratios, thicknesses, durations, etc.),
you MUST use them explicitly in your writing — do not claim quantitative
data is unavailable if it is present below, even if it appears under a
different label than expected. Only state that quantitative data was not
available if you have checked, line by line, that the evidence below truly
contains no numbers relevant to this section topic.

The document title and any internal labels (e.g. version tags, patch names,
test identifiers) are administrative metadata, NOT scientific content. Do
not treat them as real materials, devices, methods, or cite fabricated
sources for them."""

        if previous_draft and critique:
            issues_text = "\n".join(
                f"- [{i.get('dimension','')}] {i.get('issue','')} → {i.get('suggestion','')}"
                for i in (critique.get("issues") or [])
            )
            prompt = f"""You are a scientific writer producing one section of a research document.

{grounding_rule}

SECTION: {brief.get('section_name', '')}
TYPE: {brief.get('section_type', '')}
TARGET LENGTH: approximately {brief.get('target_word_count', 500)} words

SECTION BRIEF:
{brief.get('brief', '')}

QUALITY CRITERIA:
{chr(10).join(f'- {c}' for c in brief.get('quality_criteria', []))}

AVAILABLE EVIDENCE:
{evidence_summary}

ALREADY-APPROVED SECTIONS (for coherence):
{prior_context or 'None yet.'}

YOU ARE REVISING THE FOLLOWING DRAFT.
PREVIOUS DRAFT:
{previous_draft}

CRITIQUE TO ADDRESS:
{issues_text}

Produce an improved version of the draft that specifically addresses each critique point.
Write ONLY the section content — no headers, no meta-commentary.
"""
        else:
            prompt = f"""You are a scientific writer producing one section of a research document.

{grounding_rule}

SECTION: {brief.get('section_name', '')}
TYPE: {brief.get('section_type', '')}
TARGET LENGTH: approximately {brief.get('target_word_count', 500)} words

SECTION BRIEF:
{brief.get('brief', '')}

QUALITY CRITERIA:
{chr(10).join(f'- {c}' for c in brief.get('quality_criteria', []))}

AVAILABLE EVIDENCE:
{evidence_summary}

ALREADY-APPROVED SECTIONS (for coherence):
{prior_context or 'None yet.'}

Write the section content now.
Write ONLY the section content — no headers, no meta-commentary.
"""

        return self.llm_client(prompt, max_tokens=1500)

    def _call_critic(
        self,
        section_brief:     dict,
        draft_content:     str,
        prior_sections:    List[dict],
        evidence_summary:  str,
    ) -> dict:

        dimensions_text = "\n".join(f"- {d}" for d in QUALITY_DIMENSIONS)

        prompt = f"""You are a {self.critic_persona} evaluating a section of a scientific document.

SECTION BRIEF:
{json.dumps(section_brief, indent=2)}

EVIDENCE THAT WAS PROVIDED TO THE WRITER:
{evidence_summary}

DRAFT TO EVALUATE:
{draft_content}

MANDATORY GROUNDING CHECK (do this before scoring):
Identify every specific numeric claim in the draft (mobility, voltage,
temperature, ratio, peak position, thickness, percentages, etc.). For each
one, check whether it appears, verbatim or as a direct paraphrase, in the
EVIDENCE THAT WAS PROVIDED above. Any numeric claim that does NOT trace back
to the supplied evidence is a fabrication, regardless of how scientifically
plausible it sounds. Report every such fabrication as an issue with
dimension="scientific_accuracy" and severity >= 8. A draft with any
untraceable fabricated number must score scientific_accuracy no higher than
3/10, even if the prose otherwise reads well.
Also check whether the draft treats the document title or any internal
version/patch labels as if they were real scientific content — flag this
the same way if found.

EVALUATE THE DRAFT on these dimensions (score each 0–10):
{dimensions_text}

For each issue found, provide: dimension, issue description, and a specific suggestion.

Respond ONLY with valid JSON in this exact schema:
{{
    "scores": {{
        "scientific_accuracy": <float 0-10>,
        "narrative_coherence": <float 0-10>,
        "brief_compliance": <float 0-10>,
        "evidence_integration": <float 0-10>,
        "logical_progression": <float 0-10>,
        "cross_section_fit": <float 0-10>
    }},
    "issues": [
        {{
            "dimension": "<dimension name>",
            "severity": <int 1-10>,
            "issue": "<description of the problem>",
            "suggestion": "<specific actionable fix>"
        }}
    ]
}}
No preamble. No markdown fences. Only the JSON object.
"""

        raw_response = self.llm_client(prompt)

        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            cleaned = cleaned.rsplit("```", 1)[0]

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            with open("/tmp/critic_json_debug.log", "a") as _dbg:
                _dbg.write(f"\n{'='*80}\n")
                _dbg.write(f"TIMESTAMP: {datetime.utcnow().isoformat()}\n")
                _dbg.write(f"FAST-PATH ERROR: {e}\n")
                _dbg.write(f"--- RAW CLEANED (pre-sanitize) ---\n{cleaned}\n")
            sanitized = _sanitize_json_escapes(cleaned)
            try:
                result = json.loads(sanitized)
                with open("/tmp/critic_json_debug.log", "a") as _dbg:
                    _dbg.write("SANITIZE RESULT: SUCCESS\n")
                return result
            except json.JSONDecodeError as e2:
                with open("/tmp/critic_json_debug.log", "a") as _dbg:
                    _dbg.write(f"SANITIZE RESULT: FAILED — {e2}\n")
                    _dbg.write(f"--- SANITIZED TEXT ---\n{sanitized}\n")
                raise

    def _call_reviser(
        self,
        draft_content:     str,
        critique:          CritiqueRecord,
        brief:             dict,
        evidence_summary:  str,
    ) -> str:
        return self._call_writer(
            brief             = brief,
            prior_sections    = self._get_prior_approved_sections(),
            previous_draft    = draft_content,
            critique          = {"issues": critique.issues},
            evidence_summary  = evidence_summary,
        )

    def _save_draft(self, section_id: int, version: int, content: str) -> int:
        word_count = len(content.split())
        with self.repo.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO GaneshDraft (section_id, version, content, word_count, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (section_id, version, content, word_count, datetime.utcnow().isoformat()),
            )
            return cursor.lastrowid

    def _save_critique(self, section_id, draft_id, critique) -> int:
        with self.repo.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO GaneshCritique
                    (section_id, draft_id, scope, scores_json, issues_json, overall_score, created_at)
                VALUES (?, ?, 'section', ?, ?, ?, ?)
                """,
                (
                    section_id,
                    draft_id,
                    json.dumps(critique.scores),
                    json.dumps(critique.issues),
                    critique.overall_score,
                    datetime.utcnow().isoformat(),
                ),
            )
            return cursor.lastrowid

    def _save_revision(self, section_id, from_draft_id, to_draft_id, critique_id, summary) -> None:
        with self.repo.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO GaneshRevision
                    (section_id, from_draft_id, to_draft_id, critique_id, changes_summary, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (section_id, from_draft_id, to_draft_id, critique_id, summary, datetime.utcnow().isoformat()),
            )

    def _approve_section(self, section_id, best_draft_id, best_score, below_threshold=False) -> None:
        now = datetime.utcnow().isoformat()
        with self.repo.transaction() as cursor:
            cursor.execute(
                "UPDATE GaneshSection SET status = 'approved', quality_score = ?, updated_at = ? WHERE id = ?",
                (best_score, now, section_id),
            )
        if below_threshold:
            with self.repo.transaction() as cursor:
                cursor.execute(
                    "UPDATE GaneshDocument SET quality_flag = 'below_threshold', updated_at = ? WHERE id = ?",
                    (now, self.document_id),
                )

    def _update_section_status(self, section_id: int, status: str) -> None:
        with self.repo.transaction() as cursor:
            cursor.execute(
                "UPDATE GaneshSection SET status = ?, updated_at = ? WHERE id = ?",
                (status, datetime.utcnow().isoformat(), section_id),
            )

    def _get_draft_content(self, draft_id: Optional[int]) -> Optional[str]:
        if draft_id is None:
            return None
        row = self.repo.fetch_one("SELECT content FROM GaneshDraft WHERE id = ?", (draft_id,))
        return row["content"] if row else None

    def _get_critique(self, draft_id: Optional[int]) -> Optional[dict]:
        if draft_id is None:
            return None
        row = self.repo.fetch_one(
            "SELECT scores_json, issues_json FROM GaneshCritique WHERE draft_id = ?",
            (draft_id,),
        )
        if not row:
            return None
        return {
            "scores": json.loads(row["scores_json"] or "{}"),
            "issues": json.loads(row["issues_json"] or "[]"),
        }

    def _get_latest_critique_id(self, section_id: int) -> Optional[int]:
        row = self.repo.fetch_one(
            "SELECT id FROM GaneshCritique WHERE section_id = ? ORDER BY id DESC LIMIT 1",
            (section_id,),
        )
        return row["id"] if row else None

    def _get_prior_approved_sections(self) -> List[dict]:
        rows = self.repo.fetch_all(
            """
            SELECT gs.section_name, gs.exec_order, gd.content
            FROM GaneshSection gs
            JOIN GaneshDraft gd ON gd.section_id = gs.id
            WHERE gs.document_id = ?
              AND gs.status = 'approved'
              AND gd.version = (
                  SELECT MAX(version) FROM GaneshDraft WHERE section_id = gs.id
              )
            ORDER BY gs.exec_order ASC
            """,
            (self.document_id,),
        )
        return [dict(r) for r in rows]

    def _format_section_evidence(self, section_name: str) -> str:
        """Pull pre-indexed evidence for this section from context_bundle.

        Findings (S5_5 synthesized, numeric) are prioritized ahead of raw
        ResearchKnowledge fragments, since findings carry actual quantitative
        values while raw knowledge rows are often bare technique names with
        no context. Confirmed 2026-07-02: a flat rows[:10] cutoff landed
        entirely inside the raw-knowledge portion for sections with large
        evidence pools, silently dropping every finding before the LLM ever
        saw them.
        """
        if not self.context_bundle:
            return "No evidence available."
        section_map = self.context_bundle.get("section_evidence_map", {})
        rows = section_map.get(section_name, [])
        if not rows:
            summary = self.context_bundle.get("knowledge_summary", {})
            lines = []
            for cat, vals in list(summary.items())[:4]:
                lines.append(f"{cat}: {', '.join(str(v) for v in vals[:5])}")
            return "\n".join(lines) or "No evidence available."

        findings    = [r for r in rows if r.get("category") == "finding"]
        non_finding = [r for r in rows if r.get("category") != "finding"]
        ordered     = findings + non_finding

        MAX_EVIDENCE_ROWS = 20
        lines = []
        for r in ordered[:MAX_EVIDENCE_ROWS]:
            if r.get("category") == "finding":
                val        = r.get("value", "")
                prop_name  = r.get("property_name", "")
                prop_value = r.get("property_value", "")
                extra = f" ({prop_name}: {prop_value})" if prop_name else ""
                lines.append(f"[finding] {val}{extra}")
            else:
                cat = r.get("category", "")
                val = r.get("value", "")
                ctx = str(r.get("context") or "")[:150]
                lines.append(f"[{cat}] {val} — {ctx}")
        return "\n".join(lines)

    def _format_evidence(self, evidence_refs: List[str]) -> str:
        if not evidence_refs or not self.context_bundle:
            return "No specific evidence pre-loaded — draw from general context bundle."

        fragments = []
        for ref in evidence_refs:
            content = self.context_bundle.get("evidence", {}).get(ref)
            if content:
                fragments.append(f"[{ref}]\n{content}")

        return "\n\n".join(fragments) if fragments else "See general context bundle."
