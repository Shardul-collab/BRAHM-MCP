# api/models.py

"""
Pydantic request and response models for the Chitragupta API.

All models use strict field definitions so FastAPI can generate accurate
OpenAPI schemas and validate responses before they leave the server.

AUDIT ADDITIONS
---------------
PendingWritesResponse       — replaces bare dict on GET /entries/{name}/pending
                              (audit fix MIN-1)
TranscriptionJobResponse    — replaces response_model=dict on POST /voice/transcribe
                              (audit fix MIN-2)
TranscriptionResultResponse — replaces response_model=dict on GET /voice/transcribe/{job_id}
                              (audit fix MIN-2)

PaginatedEntriesResponse    — next_cursor field added; offset removed in favour of
                              true cursor-based pagination (audit fix MAJ-1).
                              The `offset` field is kept as a deprecated Optional[int]
                              so existing clients that read but do not rely on it
                              do not break.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── Database models ───────────────────────────────────────────────────────────

class FieldDefinition(BaseModel):
    """A single field in a database schema."""
    name:     str
    type:     str
    options:  list[str] | None = None


class CreateDatabaseRequest(BaseModel):
    database_name:  str
    parent_page_id: str
    fields:         list[FieldDefinition]


class DatabaseResponse(BaseModel):
    database_name:       str
    notion_database_id:  str | None = None
    fields:              list[dict[str, Any]]


class DatabaseListResponse(BaseModel):
    databases: list[str]


class DriftResponse(BaseModel):
    database_name: str
    has_drift:     bool
    issues:        list[str]



class CreateRelationRequest(BaseModel):
    database_a: str
    database_b: str
    relation_name: str
    bidirectional: bool = False
    reverse_relation_name: str | None = None


class LastValuesResponse(BaseModel):
    database_name: str
    last_values: dict

class SkipCountsResponse(BaseModel):
    database_name: str
    skip_counts: dict

class StatusResponse(BaseModel):
    ok:      bool
    message: str


# ── Schema inference models ───────────────────────────────────────────────────

class InferSchemaRequest(BaseModel):
    description: str = Field(
        ...,
        description="Natural language description of what to track.",
        min_length=1,
    )


class InferSchemaResponse(BaseModel):
    fields:      list[dict[str, Any]]
    description: str


# ── Entry models ──────────────────────────────────────────────────────────────

class LogEntryRequest(BaseModel):
    fields: dict[str, Any] = Field(
        ...,
        description="Map of field name → value matching the database schema.",
    )


class LogEntryResponse(BaseModel):
    status:        str
    submission_id: str
    database_name: str
    message:       str


class PaginatedEntriesResponse(BaseModel):
    """
    Response for GET /entries/{name}.

    MAJ-1 FIX: `next_cursor` replaces the old offset-based pagination.
    Pass next_cursor as the `cursor` query param on the next request.
    A null next_cursor means you are on the last page.

    `offset` is kept as a deprecated optional field so existing read-only
    clients do not break, but it is no longer populated by the endpoint.
    """
    database_name: str
    items:         list[dict[str, Any]]
    total:         int   = Field(description="Number of items returned in this page.")
    limit:         int
    next_cursor:   str | None = Field(
        default=None,
        description="Opaque cursor — pass as `cursor` query param for the next page. "
                    "Null means this is the last page.",
    )
    # Deprecated — kept for backward compatibility only
    offset:        int | None = Field(
        default=None,
        description="Deprecated. Was used for offset-based pagination; "
                    "always null now that cursor pagination is in use.",
    )


class PendingWritesResponse(BaseModel):
    """
    Response for GET /entries/{name}/pending.

    MIN-1 FIX: replaces the previous bare dict return so the endpoint has a
    proper Swagger schema and FastAPI can validate the response.
    """
    database_name: str
    pending_count: int
    entries:       list[dict[str, Any]]


# ── Voice models ──────────────────────────────────────────────────────────────

class TranscriptResponse(BaseModel):
    """Legacy synchronous transcription response (kept for compatibility)."""
    transcript:       str
    is_hallucination: bool
    duration_seconds: float | None = None


class TranscriptionJobResponse(BaseModel):
    """
    Response for POST /voice/transcribe (202 Accepted).

    MIN-2 FIX: replaces response_model=dict with a typed model.
    """
    status:  str
    job_id:  str
    message: str


class TranscriptionResultResponse(BaseModel):
    """
    Response for GET /voice/transcribe/{job_id}.

    MIN-2 FIX: replaces response_model=dict with a typed model.

    Fields present depend on job status:
      pending / running  → only status is guaranteed
      done               → status + transcript + is_hallucination + duration_seconds
      failed             → status + error
    """
    job_id:           str
    status:           str
    transcript:       str | None   = None
    is_hallucination: bool | None  = None
    duration_seconds: float | None = None
    error:            str | None   = None


class VoiceLogEntryResponse(BaseModel):
    status:           str
    database_name:    str
    submission_id:    str
    transcript:       str
    extracted_fields: dict[str, Any]


# ── Analysis models ───────────────────────────────────────────────────────────

class AnalysisResponse(BaseModel):
    database_name: str
    insights:      dict[str, Any] = Field(
        description="Pattern analysis output from the pattern_analyzer module. "
                    "Structure varies by database schema and analyzer version.",
    )

# ── Context models (for GANESH /context/* endpoints) ─────────────────────────

class WorkflowSummary(BaseModel):
    workflow_id:   int
    name:          str
    status:        str
    current_stage: str
    papers:        int
    knowledge_rows: int

class WorkflowsResponse(BaseModel):
    ready_count: int
    workflows:   list[WorkflowSummary]

class KnowledgeEntry(BaseModel):
    category: str
    value:    str
    sentence: str | None = None
    paper_id: int

class ContextPackage(BaseModel):
    workflow_ids:      list[int]
    document_type:     str
    total_papers:      int
    total_knowledge:   int
    knowledge:         dict[str, list[KnowledgeEntry]]   # category → entries
    top_papers:        list[dict[str, Any]]

class KnowledgeSummaryResponse(BaseModel):
    workflow_ids:  list[int]
    by_category:   list[dict[str, Any]]
    total_rows:    int
    ready_papers:  int
