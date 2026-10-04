"""Test cases for the /health endpoint."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_endpoint(async_client: AsyncClient):
    """Ensure the health endpoint returns 200 and expected status fields."""
    response = await async_client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "environment" in data
    assert "database_connected" in data
    assert "timestamp" in data

