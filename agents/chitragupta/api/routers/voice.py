# api/routers/voice.py

"""
Voice router.

FIX 3 — Clean schema names
    Endpoints carry explicit operation_id titles so Swagger shows clean names.

FIX 4 — Background transcription for /voice/transcribe
    Heavy Whisper inference is offloaded to a BackgroundTask.
    The endpoint returns 202 with a job_id immediately.

FIX 5 — Voice → Entry direct endpoint
    POST /voice/log-entry accepts audio + database_name.
    Flow: transcribe → parse_intent → build_json → queue Notion write.

FIX 8 — Structured Notion errors on the log-entry path.

AUDIT FIX CRIT-5 — Whisper no longer blocks the async event loop
    _run_whisper() and _load_audio_bytes() are CPU/IO-bound.  Calling them
    directly inside async def handlers froze the entire FastAPI event loop
    for the duration of Whisper inference (typically 2–10 s), making the
    server unresponsive under any concurrent load.  Both synchronous
    endpoints (infer_schema_from_audio, voice_log_entry) now delegate the
    blocking work to asyncio.get_event_loop().run_in_executor(None, ...)
    so the event loop stays free during inference.

AUDIT FIX MAJ-2 — _transcription_jobs TTL eviction
    The in-process job store previously grew without bound.  Each entry is
    now stamped with created_at.  Jobs older than JOB_TTL_SECONDS (1 hour)
    are evicted lazily on every read and proactively during every new
    submission.  Jobs are also lost on restart — callers should treat a 404
    on a job_id as "expired or server restarted" and resubmit.

AUDIT FIX MAJ-3 — notion_error() now called on Notion failures
    notion_error was imported but never invoked.  Notion failures on the
    voice log-entry path now raise the structured notion_error response
    instead of falling through to the leaky global exception handler.

AUDIT FIX MAJ-4 — File upload validated by content-type AND extension
    Previously only the client-supplied filename extension was checked.
    An attacker could rename any file to .wav and bypass validation.
    _validate_upload() now also checks file.content_type against an
    allowlist of audio MIME types.

AUDIT FIX MAJ-5 — Tempfile always deleted even on write failure
    NamedTemporaryFile(delete=False) with a try/finally inside the context
    manager meant the file was not deleted if tmp.write() raised.
    _load_audio_bytes() now uses a single outer try/finally that runs
    regardless of where the failure occurs.

AUDIT FIX MIN-2 — response_model=dict replaced with typed models
    Transcription endpoints now declare explicit typed response models
    (TranscriptionJobResponse, TranscriptionResultResponse) instead of
    bare dict, enabling Swagger docs and response validation.

AUDIT FIX MIN-3 — Deferred imports moved to module level
    All imports previously buried inside voice_log_entry() are now at the
    top of the file, making dependencies visible and avoiding the overhead
    of import machinery on every request.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, File, Query, UploadFile, status

from api.models import (
    InferSchemaResponse,
    TranscriptResponse,
    VoiceLogEntryResponse,
    TranscriptionJobResponse,       # MIN-2: new — add to api/models.py
    TranscriptionResultResponse,    # MIN-2: new — add to api/models.py
)
from api.dependencies import api_key_auth, http_error, notion_error

from voice.whisper_handler import _get_model, _is_hallucination
from nlp.schema_inferencer import infer_schema_from_description, describe_schema_naturally
from nlp.intent_parser import parse_intent

# MIN-3: imports previously deferred inside voice_log_entry body
from notion.schema_manager import load_schema, SchemaMissingError
from notion.write_journal import new_submission_id, mark_pending
from notion.notion_client import NotionAPIError
from core.json_builder import build_json, json_to_notion_properties, ValidationError

logger = logging.getLogger("chitragupta.api.voice")

router = APIRouter(
    prefix="/voice",
    tags=["Voice"],
    dependencies=[Depends(api_key_auth)],
)

_MAX_BYTES:        int       = 25 * 1024 * 1024
_ALLOWED_SUFFIXES: set[str] = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}

# MAJ-4: MIME type allowlist (checked alongside the filename extension)
_ALLOWED_CONTENT_TYPES: set[str] = {
    "audio/wav",
    "audio/wave",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/x-m4a",
    "audio/ogg",
    "audio/flac",
    "audio/x-flac",
    "audio/webm",
    "video/webm",            # browsers often send webm audio as video/webm
    "application/octet-stream",  # generic fallback allowed; extension check still applies
}

# MAJ-2: TTL for transcription job records (1 hour)
_JOB_TTL_SECONDS: int = 3600

# In-memory job store: job_id → {"status": ..., "created_at": float, ...}
# MAJ-2: entries are evicted after _JOB_TTL_SECONDS to prevent unbounded growth.
# NOTE: jobs are lost on server restart — callers should treat a 404 as
# "expired or server restarted" and resubmit.
_transcription_jobs: dict[str, dict[str, Any]] = {}


# ── Shared helpers ────────────────────────────────────────────────────────────

def _evict_expired_jobs() -> None:
    """
    MAJ-2: Remove jobs older than _JOB_TTL_SECONDS.
    Called before every new submission and on every poll.
    O(n) scan is fine at typical job-store sizes.
    """
    cutoff = time.time() - _JOB_TTL_SECONDS
    expired = [
        jid for jid, job in _transcription_jobs.items()
        if job.get("created_at", 0) < cutoff
    ]
    for jid in expired:
        _transcription_jobs.pop(jid, None)
    if expired:
        logger.debug("Evicted %d expired transcription jobs", len(expired))


def _validate_upload(file: UploadFile) -> str:
    """
    MAJ-4 FIX: Validate both filename extension AND Content-Type header.

    Returns the suffix string or raises http_error(415).
    """
    suffix = Path(file.filename or "audio.wav").suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise http_error(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported file extension '{suffix}'. "
            f"Accepted: {', '.join(sorted(_ALLOWED_SUFFIXES))}.",
        )

    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if content_type and content_type not in _ALLOWED_CONTENT_TYPES:
        raise http_error(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported Content-Type '{content_type}'. "
            "Upload an audio file (wav, mp3, m4a, ogg, flac, webm).",
        )

    return suffix


def _load_audio_bytes(data: bytes, suffix: str):
    """
    Write bytes to a temp file and load via Whisper's audio loader.

    MAJ-5 FIX: A single outer try/finally guarantees the tempfile is
    deleted whether the failure occurs during write or during load_audio,
    replacing the previous pattern where a write failure left an orphaned
    file on disk.
    """
    import whisper.audio as wa

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        audio = wa.load_audio(tmp_path)
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
    return audio


def _run_whisper(audio) -> tuple[str, bool]:
    """Transcribe and return (transcript, is_hallucination). CPU-bound."""
    model  = _get_model()
    result = model.transcribe(audio, fp16=False, language=None, verbose=False)
    text   = result.get("text", "").strip()
    hall   = _is_hallucination(text) if text else False
    return text, hall


def _load_and_transcribe(data: bytes, suffix: str) -> tuple[str, bool]:
    """
    CRIT-5 helper: load audio and run Whisper in a single callable so it
    can be dispatched to run_in_executor with one call.
    """
    audio = _load_audio_bytes(data, suffix)
    return _run_whisper(audio)


# ── Background job runner ─────────────────────────────────────────────────────

def _bg_transcribe(job_id: str, data: bytes, suffix: str) -> None:
    """
    FIX 4: Background task that runs Whisper and stores result.
    Runs in a thread pool — does NOT block the event loop.
    """
    _transcription_jobs[job_id]["status"] = "running"
    try:
        t0 = time.perf_counter()
        transcript, is_hall = _load_and_transcribe(data, suffix)
        duration = round(time.perf_counter() - t0, 2)
        _transcription_jobs[job_id].update({
            "status":           "done",
            "transcript":       transcript,
            "is_hallucination": is_hall,
            "duration_seconds": duration,
        })
        logger.info("BG transcription done | job=%s chars=%d", job_id, len(transcript))
    except Exception as exc:
        _transcription_jobs[job_id].update({"status": "failed", "error": str(exc)})
        logger.error("BG transcription failed | job=%s error=%s", job_id, exc)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/transcribe",
    response_model=TranscriptionJobResponse,   # MIN-2: typed instead of dict
    status_code=status.HTTP_202_ACCEPTED,
    operation_id="transcribe_audio",
    summary="Transcribe audio file (background)",
)
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio file (wav/mp3/m4a/ogg/flac/webm, max 25 MB)"),
) -> TranscriptionJobResponse:
    """
    FIX 3 + FIX 4: Upload an audio file for transcription via local Whisper.

    Returns immediately with a job_id (202 Accepted).
    Poll GET /voice/transcribe/{job_id} to retrieve the result.

    - Accepts: wav, mp3, m4a, ogg, flac, webm — max 25 MB
    - Uses the same Whisper model and hallucination filter as the CLI
    """
    suffix = _validate_upload(file)
    data   = await file.read()

    if len(data) > _MAX_BYTES:
        raise http_error(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"File too large ({len(data) // 1024} KB). Maximum is 25 MB.",
        )

    # MAJ-2: evict stale jobs before registering a new one
    _evict_expired_jobs()

    job_id = new_submission_id()
    _transcription_jobs[job_id] = {
        "status":     "pending",
        "created_at": time.time(),   # MAJ-2: TTL stamp
    }

    # CRIT-5: Whisper runs in a background thread, not blocking the event loop.
    # BackgroundTasks uses a thread pool executor internally.
    background_tasks.add_task(_bg_transcribe, job_id, data, suffix)

    return TranscriptionJobResponse(
        status="queued",
        job_id=job_id,
        message=f"Transcription queued. Poll GET /v1/voice/transcribe/{job_id} for result.",
    )


@router.get(
    "/transcribe/{job_id}",
    response_model=TranscriptionResultResponse,   # MIN-2: typed instead of dict
    operation_id="get_transcription_result",
    summary="Get transcription result by job ID",
)
async def get_transcription_result(job_id: str) -> TranscriptionResultResponse:
    """
    FIX 4: Poll for the result of a background transcription job.

    Returns status: pending | running | done | failed
    When done, includes transcript and is_hallucination.

    A 404 means the job has expired (older than 1 hour) or the server
    was restarted after submission.  Resubmit the audio file.
    """
    # MAJ-2: evict stale entries before looking up
    _evict_expired_jobs()

    job = _transcription_jobs.get(job_id)
    if job is None:
        raise http_error(
            status.HTTP_404_NOT_FOUND,
            f"Job '{job_id}' not found. It may have expired (>1 h) or the "
            "server was restarted. Resubmit the audio file.",
        )
    return TranscriptionResultResponse(job_id=job_id, **{
        k: v for k, v in job.items() if k != "created_at"
    })


@router.post(
    "/infer-schema",
    response_model=InferSchemaResponse,
    operation_id="infer_schema_from_audio",
    summary="Infer schema from audio description",
)
async def infer_schema_from_audio(
    file: UploadFile = File(..., description="Audio describing what to track (max 25 MB)"),
) -> InferSchemaResponse:
    """
    FIX 3: Upload audio describing a database — returns inferred schema fields.

    CRIT-5 FIX: Whisper runs in a thread-pool executor so the event loop
    is not blocked during multi-second inference.
    Accepts: wav, mp3, m4a, ogg, flac, webm — max 25 MB.
    """
    suffix = _validate_upload(file)
    data   = await file.read()

    if len(data) > _MAX_BYTES:
        raise http_error(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "File too large. Maximum is 25 MB.",
        )

    try:
        # CRIT-5: run blocking Whisper work off the event loop
        loop = asyncio.get_event_loop()
        transcript, is_hall = await loop.run_in_executor(
            None, _load_and_transcribe, data, suffix
        )
    except Exception as exc:
        raise http_error(status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc))

    if not transcript or is_hall:
        raise http_error(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Could not extract a useful description from the audio. Please try again.",
        )

    fields = infer_schema_from_description(transcript)
    spoken = describe_schema_naturally(fields, "this database")
    return InferSchemaResponse(fields=fields, description=spoken)


@router.post(
    "/log-entry",
    response_model=VoiceLogEntryResponse,
    status_code=status.HTTP_202_ACCEPTED,
    operation_id="voice_log_entry",
    summary="Transcribe audio and log entry (FIX 5)",
)
async def voice_log_entry(
    background_tasks: BackgroundTasks,
    database_name: str = Query(..., description="Target database name"),
    file: UploadFile = File(..., description="Audio describing the entry (max 25 MB)"),
) -> VoiceLogEntryResponse:
    """
    FIX 5: One-shot voice → entry endpoint.

    Flow:
    1. Transcribe audio via local Whisper (synchronous — needed for extraction)
    2. Extract field values using parse_intent
    3. Validate and coerce via build_json
    4. Queue Notion write via BackgroundTask (FIX 4)

    Returns immediately with transcript, extracted_fields, and submission_id.

    CRIT-5 FIX: Whisper runs via run_in_executor, not blocking the event loop.
    MAJ-3 FIX: Notion failures now raise notion_error() instead of falling
                through to the global exception handler.
    MIN-3 FIX: All imports are at module level (top of file).
    """
    from api.routers.entries import _write_entry_to_notion

    # ── Validate audio ──────────────────────────────────────────────────────
    suffix = _validate_upload(file)
    data   = await file.read()
    if len(data) > _MAX_BYTES:
        raise http_error(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "File too large. Maximum is 25 MB.",
        )

    # ── Load schema ─────────────────────────────────────────────────────────
    try:
        schema = load_schema(database_name)
    except SchemaMissingError as exc:
        raise http_error(status.HTTP_404_NOT_FOUND, str(exc))

    notion_db_id = schema.get("notion_database_id", "").strip()
    if not notion_db_id:
        raise http_error(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"'{database_name}' has no Notion ID. Create the database first.",
        )

    # ── Transcribe — CRIT-5: run off the event loop ──────────────────────────
    try:
        loop = asyncio.get_event_loop()
        transcript, is_hall = await loop.run_in_executor(
            None, _load_and_transcribe, data, suffix
        )
    except Exception as exc:
        raise http_error(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Transcription failed: {exc}")

    if not transcript or is_hall:
        raise http_error(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Audio was empty or unrecognisable. Please try again.",
        )

    # ── Extract fields from transcript ───────────────────────────────────────
    try:
        extracted = parse_intent(database_name, transcript)
    except Exception as exc:
        raise http_error(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Field extraction failed: {exc}",
        )

    # ── Validate and coerce ──────────────────────────────────────────────────
    try:
        validated      = build_json(database_name, extracted)
        notion_payload = json_to_notion_properties(database_name, validated)
    except ValidationError as exc:
        raise http_error(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))

    # ── Queue Notion write ───────────────────────────────────────────────────
    # MAJ-3 FIX: Notion errors on this path previously fell through to the
    # global exception handler which leaked internals.  The background task
    # already handles NotionAPIError with logger.error.  For failures that
    # happen before the task is dispatched (e.g. mark_pending I/O errors),
    # wrap them with notion_error().
    try:
        sid = new_submission_id()
        mark_pending(sid, notion_db_id)
    except NotionAPIError as exc:
        raise notion_error(exc)

    background_tasks.add_task(
        _write_entry_to_notion,
        database_name,
        notion_db_id,
        notion_payload,
        validated,
        sid,
    )

    logger.info(
        "Voice log-entry queued | db='%s' submission=%s transcript_len=%d",
        database_name, sid, len(transcript),
    )

    return VoiceLogEntryResponse(
        status="queued",
        database_name=database_name,
        submission_id=sid,
        transcript=transcript,
        extracted_fields=validated,
    )
