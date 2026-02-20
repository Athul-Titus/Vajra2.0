"""Tests for configuration CRUD endpoints."""

import pytest


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_config_requires_auth(client):
    """Config endpoints should require X-API-Key header."""
    response = await client.get("/api/v1/configurations")
    assert response.status_code in (401, 403)


@pytest.mark.asyncio
async def test_config_rejects_bad_key(client):
    """Config endpoints should reject invalid API keys."""
    response = await client.get(
        "/api/v1/configurations",
        headers={"X-API-Key": "wrong-key"},
    )
    assert response.status_code in (401, 403)


# ---------------------------------------------------------------------------
# GET default config
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_default_config(client, auth_headers):
    """Should return the built-in telecom_default config."""
    response = await client.get(
        "/api/v1/configurations/telecom_default",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["client_id"] == "telecom_default"
    assert data["domain"] == "telecom"
    assert len(data["compliance_policies"]) == 7
    assert len(data["risk_triggers"]) == 23


@pytest.mark.asyncio
async def test_get_missing_config_returns_404(client, auth_headers):
    """Should return 404 for non-existent config."""
    response = await client.get(
        "/api/v1/configurations/nonexistent_config",
        headers=auth_headers,
    )
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# CREATE config
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_config(client, auth_headers, sample_config_payload):
    """Should create a new configuration and return 201."""
    response = await client.post(
        "/api/v1/configurations",
        headers=auth_headers,
        json=sample_config_payload,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["client_id"] == "test_telecom"
    assert data["client_name"] == "Test Telecom Provider"
    assert len(data["compliance_policies"]) == 1


@pytest.mark.asyncio
async def test_create_duplicate_config_returns_409(client, auth_headers, sample_config_payload):
    """Should return 409 when creating a config with existing client_id."""
    await client.post(
        "/api/v1/configurations",
        headers=auth_headers,
        json=sample_config_payload,
    )
    response = await client.post(
        "/api/v1/configurations",
        headers=auth_headers,
        json=sample_config_payload,
    )
    assert response.status_code == 409


# ---------------------------------------------------------------------------
# LIST configs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_configs_empty(client, auth_headers):
    """Should return empty list when no DB configs exist."""
    response = await client.get("/api/v1/configurations", headers=auth_headers)
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_list_configs_after_create(client, auth_headers, sample_config_payload):
    """Should return created configs in list."""
    await client.post(
        "/api/v1/configurations",
        headers=auth_headers,
        json=sample_config_payload,
    )
    response = await client.get("/api/v1/configurations", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["client_id"] == "test_telecom"


# ---------------------------------------------------------------------------
# UPDATE config
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_config(client, auth_headers, sample_config_payload):
    """Should update specific fields of an existing config."""
    await client.post(
        "/api/v1/configurations",
        headers=auth_headers,
        json=sample_config_payload,
    )

    response = await client.put(
        "/api/v1/configurations/test_telecom",
        headers=auth_headers,
        json={"client_name": "Updated Telecom"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["client_name"] == "Updated Telecom"
    assert data["client_id"] == "test_telecom"  # Unchanged


@pytest.mark.asyncio
async def test_update_missing_config_returns_404(client, auth_headers):
    """Should return 404 when updating non-existent config."""
    response = await client.put(
        "/api/v1/configurations/nonexistent",
        headers=auth_headers,
        json={"client_name": "Test"},
    )
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# DELETE config
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_config(client, auth_headers, sample_config_payload):
    """Should delete an existing config and return 204."""
    await client.post(
        "/api/v1/configurations",
        headers=auth_headers,
        json=sample_config_payload,
    )

    response = await client.delete(
        "/api/v1/configurations/test_telecom",
        headers=auth_headers,
    )
    assert response.status_code == 204

    # Verify it's gone from DB
    response = await client.get("/api/v1/configurations", headers=auth_headers)
    assert len(response.json()) == 0


@pytest.mark.asyncio
async def test_delete_missing_config_returns_404(client, auth_headers):
    """Should return 404 when deleting non-existent config."""
    response = await client.delete(
        "/api/v1/configurations/nonexistent",
        headers=auth_headers,
    )
    assert response.status_code == 404
