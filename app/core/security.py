"""API key authentication dependency.

Simple but production-like API key verification via X-API-Key header.
Health endpoints remain public; all other endpoints require auth.
"""

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.core.config import settings

# Header scheme — shows up in OpenAPI docs with a lock icon
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: str | None = Security(api_key_header),
) -> str:
    """Validate the API key from the request header.

    Args:
        api_key: Value of the X-API-Key header.

    Returns:
        The validated API key.

    Raises:
        HTTPException: 401 if key is missing, 403 if key is invalid.
    """
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide it via the X-API-Key header.",
        )
    if api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )
    return api_key
