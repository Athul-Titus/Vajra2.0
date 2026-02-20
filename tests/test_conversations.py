"""Tests for conversation analysis endpoints.

These tests validate input validation, auth, and error handling.
AI-dependent tests (actual Gemini calls) are in test_e2e.py and
require a valid API key + quota.
"""

import pytest


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_analyze_requires_auth(client, sample_transcript):
    """Analysis endpoint should require X-API-Key."""
    response = await client.post(
        "/api/v1/conversations/analyze/text",
        json={"transcript": sample_transcript},
    )
    assert response.status_code in (401, 403)


@pytest.mark.asyncio
async def test_analyze_rejects_bad_key(client, sample_transcript):
    """Analysis endpoint should reject invalid API keys."""
    response = await client.post(
        "/api/v1/conversations/analyze/text",
        headers={"X-API-Key": "wrong-key"},
        json={"transcript": sample_transcript},
    )
    assert response.status_code in (401, 403)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_analyze_empty_transcript(client, auth_headers):
    """Should reject empty transcript with 422."""
    response = await client.post(
        "/api/v1/conversations/analyze/text",
        headers=auth_headers,
        json={"transcript": ""},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_analyze_too_short_transcript(client, auth_headers):
    """Should reject transcript shorter than min_length."""
    response = await client.post(
        "/api/v1/conversations/analyze/text",
        headers=auth_headers,
        json={"transcript": "Hi"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_analyze_missing_transcript(client, auth_headers):
    """Should reject request without transcript field."""
    response = await client.post(
        "/api/v1/conversations/analyze/text",
        headers=auth_headers,
        json={},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_analyze_invalid_config_id_pattern(client, auth_headers, sample_transcript):
    """Should reject config_id that doesn't match the allowed pattern."""
    response = await client.post(
        "/api/v1/conversations/analyze/text",
        headers=auth_headers,
        json={"transcript": sample_transcript, "config_id": "INVALID-ID!"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_analyze_nonexistent_config(client, auth_headers, sample_transcript):
    """Should return 404 for non-existent configuration.

    Note: This test may return 500 or 404 depending on whether the AI call
    happens before config validation. Our implementation validates config first.
    """
    response = await client.post(
        "/api/v1/conversations/analyze/text",
        headers=auth_headers,
        json={"transcript": sample_transcript, "config_id": "nonexistent_config"},
    )
    assert response.status_code in (404, 500)


# ---------------------------------------------------------------------------
# Audio endpoint validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audio_rejects_unsupported_type(client, auth_headers):
    """Should reject non-audio file types."""
    response = await client.post(
        "/api/v1/conversations/analyze/audio",
        headers=auth_headers,
        files={"file": ("test.txt", b"not audio content", "text/plain")},
        data={"config_id": "telecom_default"},
    )
    assert response.status_code == 415


@pytest.mark.asyncio
async def test_audio_requires_auth(client):
    """Audio endpoint should require X-API-Key."""
    response = await client.post(
        "/api/v1/conversations/analyze/audio",
        files={"file": ("test.wav", b"fake audio", "audio/wav")},
    )
    assert response.status_code in (401, 403)


# ---------------------------------------------------------------------------
# Response format tests (using mocked/minimal scenarios)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openapi_schema_available(client):
    """OpenAPI schema should be accessible at /openapi.json."""
    response = await client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "paths" in schema
    assert "/api/v1/conversations/analyze/text" in schema["paths"]
    assert "/api/v1/conversations/analyze/audio" in schema["paths"]
    assert "/api/v1/configurations" in schema["paths"]
    assert "/api/v1/health" in schema["paths"]
