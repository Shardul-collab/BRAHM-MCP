from datetime import datetime
from repositories.repository import Repository


# ============================================================
# RESEARCH KNOWLEDGE REPOSITORY
#
# Changes vs previous version:
# - create_research_knowledge() now accepts source_type and
#   confidence — two new columns added to ResearchKnowledge.
#
#   source_type: where the knowledge came from.
#     'abstract'  = S2_75 (extract_lightweight_knowledge)
#     'pdf'       = S5 full PDF extraction
#     'pattern'   = rule-based regex in S5
#     'rule'      = keyword rule in S5
#     'llm'       = LLM extraction
#     'fallback'  = last-resort fallback
#
#   confidence: extraction confidence level.
#     'low'    = abstract-only (S2_75 output)
#     'medium' = rule or LLM from PDF (S5 output)
#     'high'   = reserved for future verified sources
#
# - Both default to None for backward compatibility.
#   Existing callers that do not pass these arguments continue
#   to work. S5, S6, S7 treat NULL as 'medium'.
#
# - create_lightweight_knowledge() added as a dedicated
#   insert function for S2_75. Identical to
#   create_research_knowledge() but always sets:
#     source_type = 'abstract'
#     confidence  = 'low' or 'medium' (caller decides)
#
# - get_knowledge_for_paper() unchanged — returns all columns
#   including the new ones via SELECT *.
# ============================================================


def create_research_knowledge(
    repo: Repository,
    paper_id: int,
    category: str,
    value: str,
    section_source: str | None,
    source_type: str | None = None,
    confidence: str | None = None
):
    """
    Inserts one ResearchKnowledge row.

    section_source: legacy field — which section of the paper
                    the knowledge came from, or extraction method.
    source_type:    new — overall origin ('abstract', 'pdf', etc.)
    confidence:     new — extraction confidence ('low', 'medium').

    Returns:
        int: newly created knowledge ID
    """
    timestamp = datetime.utcnow().isoformat()

    with repo.transaction() as cursor:
        cursor.execute(
            """
            INSERT INTO ResearchKnowledge (
                paper_id,
                category,
                value,
                section_source,
                source_type,
                confidence,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (
                paper_id,
                category,
                value,
                section_source,
                source_type,
                confidence,
                timestamp
            )
        )

        return cursor.lastrowid


def create_lightweight_knowledge(
    repo: Repository,
    paper_id: int,
    category: str,
    value: str,
    sentence: str | None,
    confidence: str
):
    """
    Dedicated insert for S2_75 (extract_lightweight_knowledge).

    Always sets:
      section_source = 'abstract'
      source_type    = 'abstract'
      confidence     = caller-supplied ('low' or 'medium')

    sentence: the abstract sentence the value was extracted from.
              None for rule-based extractions with no sentence.

    Returns:
        int: newly created knowledge ID
    """
    timestamp = datetime.utcnow().isoformat()

    with repo.transaction() as cursor:
        cursor.execute(
            """
            INSERT INTO ResearchKnowledge (
                paper_id,
                category,
                value,
                section_source,
                sentence,
                source_type,
                confidence,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                paper_id,
                category,
                value,
                "abstract",
                sentence,
                "abstract",
                confidence,
                timestamp
            )
        )

        return cursor.lastrowid


def get_knowledge_for_paper(repo: Repository, paper_id: int):
    """
    Returns all ResearchKnowledge rows for a paper.
    Includes source_type and confidence columns.
    """
    rows = repo.fetch_all(
        """
        SELECT *
        FROM ResearchKnowledge
        WHERE paper_id = ?
        """,
        (paper_id,)
    )

    return [dict(r) for r in rows]


def paper_has_knowledge(repo: Repository, paper_id: int) -> bool:
    """
    Returns True if any ResearchKnowledge entries exist for
    this paper. Used by S2_75 to skip papers already processed.
    """
    row = repo.fetch_one(
        """
        SELECT 1
        FROM ResearchKnowledge
        WHERE paper_id = ?
        LIMIT 1
        """,
        (paper_id,)
    )

    return row is not None
