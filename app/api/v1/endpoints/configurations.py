"""Client configuration CRUD endpoints.

Allows creating, reading, updating, and deleting client configurations
that drive how conversations are analyzed.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import verify_api_key
from app.models.database import ClientConfigurationModel, get_db_session
from app.models.schemas.configuration import (
    ClientConfigurationCreate,
    ClientConfigurationResponse,
    ClientConfigurationUpdate,
)
from app.services.conversation_analyzer import get_config, invalidate_config_cache

router = APIRouter(
    prefix="/configurations",
    tags=["Configurations"],
    dependencies=[Depends(verify_api_key)],
)


def _row_to_response(row: ClientConfigurationModel) -> ClientConfigurationResponse:
    """Convert a database row to the API response schema."""
    config_data = row.config_data or {}
    return ClientConfigurationResponse(
        client_id=row.client_id,
        client_name=row.client_name,
        domain=row.domain,
        description=row.description,
        products=config_data.get("products", []),
        compliance_policies=config_data.get("compliance_policies", []),
        risk_triggers=config_data.get("risk_triggers", []),
        quality_criteria=config_data.get("quality_criteria", {}),
        call_outcome_categories=config_data.get("call_outcome_categories", []),
        created_at=row.created_at.isoformat() if row.created_at else None,
        updated_at=row.updated_at.isoformat() if row.updated_at else None,
    )


@router.get(
    "",
    response_model=list[ClientConfigurationResponse],
    summary="List all configurations",
)
async def list_configurations(
    db: AsyncSession = Depends(get_db_session),
):
    """Return all stored client configurations."""
    result = await db.execute(select(ClientConfigurationModel))
    rows = result.scalars().all()
    return [_row_to_response(r) for r in rows]


@router.get(
    "/{config_id}",
    response_model=ClientConfigurationResponse,
    summary="Get a configuration",
)
async def get_configuration(
    config_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """Get a specific client configuration by ID.

    Searches both database and default JSON configurations.
    """
    try:
        config = await get_config(config_id, db)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration '{config_id}' not found",
        )

    # If from default file, return without timestamps
    return ClientConfigurationResponse(
        client_id=config.client_id,
        client_name=config.client_name,
        domain=config.domain,
        description=config.description,
        products=config.products,
        compliance_policies=config.compliance_policies,
        risk_triggers=config.risk_triggers,
        quality_criteria=config.quality_criteria,
        call_outcome_categories=config.call_outcome_categories,
    )


@router.post(
    "",
    response_model=ClientConfigurationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a configuration",
)
async def create_configuration(
    payload: ClientConfigurationCreate,
    db: AsyncSession = Depends(get_db_session),
):
    """Create a new client configuration.

    The configuration will be stored in the database and can be used
    for subsequent conversation analyses by referencing its client_id.
    """
    # Check for duplicates
    existing = await db.execute(
        select(ClientConfigurationModel).where(
            ClientConfigurationModel.client_id == payload.client_id
        )
    )
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Configuration '{payload.client_id}' already exists",
        )

    # Build the config_data JSON blob
    config_data = {
        "products": payload.products,
        "compliance_policies": [p.model_dump() for p in payload.compliance_policies],
        "risk_triggers": payload.risk_triggers,
        "quality_criteria": payload.quality_criteria.model_dump(),
        "call_outcome_categories": payload.call_outcome_categories,
    }

    row = ClientConfigurationModel(
        client_id=payload.client_id,
        client_name=payload.client_name,
        domain=payload.domain,
        description=payload.description,
        config_data=config_data,
    )
    db.add(row)
    await db.flush()
    await db.refresh(row)
    invalidate_config_cache(payload.client_id)

    return _row_to_response(row)


@router.put(
    "/{config_id}",
    response_model=ClientConfigurationResponse,
    summary="Update a configuration",
)
async def update_configuration(
    config_id: str,
    payload: ClientConfigurationUpdate,
    db: AsyncSession = Depends(get_db_session),
):
    """Update an existing client configuration.

    Only provided fields are updated — omit fields to keep existing values.
    """
    result = await db.execute(
        select(ClientConfigurationModel).where(
            ClientConfigurationModel.client_id == config_id
        )
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration '{config_id}' not found",
        )

    # Update top-level fields
    update_data = payload.model_dump(exclude_none=True)

    if "client_name" in update_data:
        row.client_name = update_data["client_name"]
    if "domain" in update_data:
        row.domain = update_data["domain"]
    if "description" in update_data:
        row.description = update_data["description"]

    # Update config_data sub-fields
    config_data = dict(row.config_data) if row.config_data else {}

    if "products" in update_data:
        config_data["products"] = update_data["products"]
    if "compliance_policies" in update_data:
        config_data["compliance_policies"] = [
            p.model_dump() if hasattr(p, "model_dump") else p
            for p in update_data["compliance_policies"]
        ]
    if "risk_triggers" in update_data:
        config_data["risk_triggers"] = update_data["risk_triggers"]
    if "quality_criteria" in update_data:
        qc = update_data["quality_criteria"]
        config_data["quality_criteria"] = (
            qc.model_dump() if hasattr(qc, "model_dump") else qc
        )
    if "call_outcome_categories" in update_data:
        config_data["call_outcome_categories"] = update_data["call_outcome_categories"]

    row.config_data = config_data
    await db.flush()
    await db.refresh(row)
    invalidate_config_cache(config_id)

    return _row_to_response(row)


@router.delete(
    "/{config_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a configuration",
)
async def delete_configuration(
    config_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """Delete a client configuration by ID."""
    result = await db.execute(
        select(ClientConfigurationModel).where(
            ClientConfigurationModel.client_id == config_id
        )
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration '{config_id}' not found",
        )

    await db.delete(row)
    invalidate_config_cache(config_id)
