# api/routers/relations.py

"""
Relations router — create relations between two Notion databases.

Endpoints
---------
POST /relations  — create a relation field in database_a pointing to database_b
"""

from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, status

from api.models import CreateRelationRequest, StatusResponse
from api.dependencies import api_key_auth, http_error

from notion.relation_manager import create_relation, RelationError
from notion.schema_manager import SchemaMissingError

logger = logging.getLogger("chitragupta.api.relations")

router = APIRouter(
    prefix="/relations",
    tags=["Relations"],
    dependencies=[Depends(api_key_auth)],
)


@router.post("", response_model=StatusResponse, status_code=status.HTTP_201_CREATED)
async def create_relation_endpoint(body: CreateRelationRequest) -> StatusResponse:
    """
    Create a relation field in database_a that points to database_b.

    Optionally creates a reverse relation in database_b when
    bidirectional=true and reverse_relation_name is provided.

    Mirrors the CLI 'Link Databases' feature.
    """
    if body.database_a == body.database_b:
        raise http_error(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Source and target databases must be different.",
        )

    try:
        create_relation(
            database_a=body.database_a,
            database_b=body.database_b,
            relation_name=body.relation_name,
            bidirectional=body.bidirectional,
            reverse_relation_name=body.reverse_relation_name,
        )
    except SchemaMissingError as exc:
        raise http_error(status.HTTP_404_NOT_FOUND, str(exc))
    except RelationError as exc:
        raise http_error(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))
    except Exception as exc:
        logger.exception("API: relation creation failed")
        raise http_error(status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc))

    logger.info(
        "API: relation created | %s → %s field='%s'",
        body.database_a, body.database_b, body.relation_name,
    )
    return StatusResponse(
        ok=True,
        message=(
            f"Relation '{body.relation_name}' created in '{body.database_a}' "
            f"→ '{body.database_b}'."
        ),
    )
