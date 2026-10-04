"""Test cases for task management endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_task(async_client: AsyncClient):
    """Ensure a new task can be created with a valid goal."""
    payload = {
        "goal": "Find the latest invoice from Acme Corp and extract the total amount.",
        "max_steps": 10,
        "metadata": {"vendor": "Acme Corp"},
    }
    response = await async_client.post("/tasks", json=payload)
    assert response.status_code == 201

    data = response.json()
    assert "id" in data
    assert data["goal"] == payload["goal"]
    assert data["status"] == "PENDING"
    assert data["current_step"] == 0
    assert data["max_steps"] == 10


@pytest.mark.asyncio
async def test_create_task_validation_error(async_client: AsyncClient):
    """Ensure invalid payloads return 422 Unprocessable Entity."""
    payload = {"goal": "ab"}  # too short (< 5 chars)
    response = await async_client.post("/tasks", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_task_details(async_client: AsyncClient):
    """Ensure created task can be retrieved in detail."""
    payload = {
        "goal": "Process Acme Corp invoice #1024 and verify ledger entry.",
        "max_steps": 15,
        "metadata": {"source": "inbox"},
    }
    create_res = await async_client.post("/tasks", json=payload)
    assert create_res.status_code == 201
    task_id = create_res.json()["id"]

    get_res = await async_client.get(f"/tasks/{task_id}")
    assert get_res.status_code == 200

    data = get_res.json()
    assert data["id"] == task_id
    assert data["goal"] == payload["goal"]
    assert data["status"] == "PENDING"
    assert "step_logs" in data
    assert "approvals" in data
    assert "evidence" in data
    assert "state_snapshot" in data
    assert "current_objective" in data
    assert "plan_summary" in data
    assert isinstance(data["step_logs"], list)


@pytest.mark.asyncio
async def test_get_nonexistent_task(async_client: AsyncClient):
    """Ensure querying a nonexistent task ID returns 404."""
    response = await async_client.get("/tasks/non-existent-task-id-1234")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_list_tasks(async_client: AsyncClient):
    """Ensure task listing returns tasks ordered by creation."""
    for i in range(3):
        await async_client.post("/tasks", json={"goal": f"Execute test business process milestone {i}"})

    response = await async_client.get("/tasks")
    assert response.status_code == 200
    tasks = response.json()
    assert len(tasks) >= 3

