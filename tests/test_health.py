"""Tests for health endpoint."""

import pytest


@pytest.mark.asyncio
async def test_health_returns_200(client):
    """Health endpoint should always return 200 with status info."""
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "model" in data


@pytest.mark.asyncio
async def test_health_no_auth_required(client):
    """Health endpoint should not require API key auth."""
    response = await client.get("/api/v1/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_root_endpoint(client):
    """Root endpoint should return app info."""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "docs" in data


@pytest.mark.asyncio
async def test_response_has_request_id(client):
    """Every response should include X-Request-ID header."""
    response = await client.get("/api/v1/health")
    assert "x-request-id" in response.headers


@pytest.mark.asyncio
async def test_response_has_timing(client):
    """Every response should include X-Processing-Time-Ms header."""
    response = await client.get("/api/v1/health")
    assert "x-processing-time-ms" in response.headers
    # Should be a valid integer
    int(response.headers["x-processing-time-ms"])
