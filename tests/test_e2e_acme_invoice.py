"""End-to-End Integration Test for the Acme Corp autonomous invoice processing task."""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.orchestrator import AgentOrchestrator
from app.models.schemas import TaskStatus
from app.services.task_service import TaskService


@pytest.mark.asyncio
async def test_acme_corp_autonomous_task_e2e_with_approval_gate(
    db_session: AsyncSession,
    client: AsyncClient,
):
    """Verifies complete autonomous execution cycle with observation, HITL approval, and outcome verification."""
    # 1. Create Task via API
    task_goal = (
        "Find the latest invoice from Acme Corp, extract the invoice number, "
        "amount and due date, enter the information into the internal finance application, "
        "and verify that it was saved correctly."
    )

    create_resp = await client.post("/tasks", json={"goal": task_goal, "max_steps": 15})
    assert create_resp.status_code == 201
    task_id = create_resp.json()["id"]

    # 2. Trigger Autonomous Execution
    run_resp = await client.post(f"/tasks/{task_id}/run")
    assert run_resp.status_code == 200
    task_data = run_resp.json()

    # The agent should autonomously:
    # a) Find latest Acme Corp invoice (acme_invoice_2024_09.pdf)
    # b) Extract invoice number (INV-ACM-2024-015), amount (12450.50), due date (2024-10-28)
    # c) Attempt high-risk finance invoice creation -> paused by ApprovalGate!
    assert task_data["status"] == TaskStatus.WAITING_FOR_APPROVAL.value
    assert len(task_data["approvals"]) == 1

    pending_approval = task_data["approvals"][0]
    assert pending_approval["status"] == "PENDING"
    assert pending_approval["risk_level"] == "HIGH"
    proposed = pending_approval["proposed_action"]
    assert proposed["tool_name"] == "finance_create_invoice_tool"
    assert proposed["tool_args"]["invoice_number"] == "INV-ACM-2024-015"
    assert float(proposed["tool_args"]["amount"]) == 12450.50

    # 3. Simulate Human Operator Review & Approval via API
    approve_resp = await client.post(
        f"/tasks/{task_id}/approve",
        json={"approved": True, "comment": "Verified against purchase order PO-ACM-889. Approved."},
    )
    assert approve_resp.status_code == 200
    final_task_data = approve_resp.json()

    # 4. Verify Final Outcome & Independence
    assert final_task_data["status"] == TaskStatus.COMPLETED.value
    assert "INV-ACM-2024-015" in final_task_data["result_summary"]

    # Check extracted data memory
    extracted = final_task_data["extracted_data"]
    assert extracted.get("invoice_number") == "INV-ACM-2024-015"
    assert float(extracted.get("amount")) == 12450.50
    assert extracted.get("due_date") == "2024-10-28"
    assert extracted.get("verified_in_finance") is True

    # Check step audit trajectory
    step_logs = final_task_data["step_logs"]
    assert len(step_logs) >= 3
    tool_names = [s["tool_name"] for s in step_logs if s.get("tool_name")]
    assert "document_search_tool" in tool_names
    assert "document_read_tool" in tool_names
    assert "finance_create_invoice_tool" in tool_names

    # Check evidence collection
    evidence_records = final_task_data["evidence"]
    assert len(evidence_records) >= 1
    assert any(ev["evidence_type"] == "outcome_verification" for ev in evidence_records)


@pytest.mark.asyncio
async def test_rejection_halts_task_cleanly(
    db_session: AsyncSession,
    client: AsyncClient,
):
    """Verifies that rejecting a proposed high-risk action halts the task as FAILED."""
    task_goal = "Find latest Acme invoice and submit to finance."
    create_resp = await client.post("/tasks", json={"goal": task_goal})
    task_id = create_resp.json()["id"]

    # Run until paused
    await client.post(f"/tasks/{task_id}/run")

    # Reject
    reject_resp = await client.post(
        f"/tasks/{task_id}/reject",
        json={"approved": False, "comment": "Duplicate entry suspected. Rejected."},
    )
    assert reject_resp.status_code == 200
    rejected_data = reject_resp.json()

    assert rejected_data["status"] == TaskStatus.FAILED.value
    assert "rejected by human supervisor" in rejected_data["result_summary"].lower()
