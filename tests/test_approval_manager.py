"""Unit and integration tests for ApprovalManager, risk levels, and approval API endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.executor import AgentExecutor
from app.agent.state import ActionProposal, ActionType, TaskState
from app.approval.manager import ActionRiskLevel, ApprovalCheckResult, ApprovalManager
from app.main import app
from app.models.schemas import RiskLevel, TaskStatus
from app.models.task import TaskRun
from app.services.task_service import TaskService
from app.tools.base import BaseTool, ToolResult
from app.tools.registry import ToolRegistry


def test_classify_all_4_risk_levels():
    """Verify that actions are accurately classified into the 4 risk tiers:

    READ, LOW_RISK_WRITE, HIGH_RISK_WRITE, and DESTRUCTIVE.
    """
    # 1. Reading invoice -> READ
    read_action = ActionProposal(
        thought="Reading the latest invoice from Acme Corp to extract fields",
        action_type=ActionType.CALL_TOOL,
        tool_name="document_read_tool",
        tool_args={"file_path": "data/invoices/acme_invoice_2024_09.pdf"},
    )
    assert ApprovalManager.classify_risk(read_action) == ActionRiskLevel.READ
    assert ApprovalManager.requires_approval(ActionRiskLevel.READ) is False

    # 2. Creating invoice -> LOW_RISK_WRITE
    create_action = ActionProposal(
        thought="Creating invoice in the internal finance system",
        action_type=ActionType.CALL_TOOL,
        tool_name="finance_create_invoice_tool",
        tool_args={
            "invoice_number": "INV-ACM-2024-001",
            "vendor_name": "Acme Corp",
            "amount": 1250.00,
        },
    )
    assert ApprovalManager.classify_risk(create_action) == ActionRiskLevel.LOW_RISK_WRITE
    assert ApprovalManager.requires_approval(ActionRiskLevel.LOW_RISK_WRITE) is False

    # 3. Changing payment information -> HIGH_RISK_WRITE
    payment_action = ActionProposal(
        thought="Updating vendor routing number and changing payment information",
        action_type=ActionType.CALL_TOOL,
        tool_name="update_payment_details_tool",
        tool_args={"vendor": "Acme Corp", "routing_number": "021000021"},
    )
    assert ApprovalManager.classify_risk(payment_action) == ActionRiskLevel.HIGH_RISK_WRITE
    assert ApprovalManager.requires_approval(ActionRiskLevel.HIGH_RISK_WRITE) is True

    # 4. Deleting invoice -> DESTRUCTIVE
    delete_action = ActionProposal(
        thought="Deleting invoice record from finance ledger permanently",
        action_type=ActionType.CALL_TOOL,
        tool_name="delete_invoice_tool",
        tool_args={"invoice_number": "INV-ACM-2024-001"},
    )
    assert ApprovalManager.classify_risk(delete_action) == ActionRiskLevel.DESTRUCTIVE
    assert ApprovalManager.requires_approval(ActionRiskLevel.DESTRUCTIVE) is True


@pytest.mark.asyncio
async def test_automatic_execution_for_read_and_low_risk_write(db_session: AsyncSession):
    """Verify that READ and LOW_RISK_WRITE actions execute automatically without pausing."""
    task = TaskRun(goal="Read and create invoice", status="RUNNING")
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)

    # READ
    read_action = ActionProposal(
        thought="Read invoice",
        action_type=ActionType.CALL_TOOL,
        tool_name="document_read_tool",
        tool_args={},
    )
    res_read = await ApprovalManager.evaluate_action(db=db_session, task_id=task.id, action=read_action)
    assert res_read.allowed is True
    assert res_read.requires_approval is False
    assert res_read.risk_level == ActionRiskLevel.READ

    # LOW_RISK_WRITE
    create_action = ActionProposal(
        thought="Creating invoice",
        action_type=ActionType.CALL_TOOL,
        tool_name="finance_create_invoice_tool",
        tool_args={"invoice_number": "INV-123"},
    )
    res_write = await ApprovalManager.evaluate_action(db=db_session, task_id=task.id, action=create_action)
    assert res_write.allowed is True
    assert res_write.requires_approval is False
    assert res_write.risk_level == ActionRiskLevel.LOW_RISK_WRITE


@pytest.mark.asyncio
async def test_high_risk_and_destructive_actions_pause_and_create_request(db_session: AsyncSession):
    """Verify that HIGH_RISK_WRITE and DESTRUCTIVE actions pause execution and create ApprovalRequest."""
    task = TaskRun(goal="Modify finances", status="RUNNING")
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)

    # HIGH_RISK_WRITE
    payment_action = ActionProposal(
        thought="Changing payment information to new IBAN",
        action_type=ActionType.CALL_TOOL,
        tool_name="change_payment_information",
        tool_args={"new_iban": "US1234567890"},
    )

    check = await ApprovalManager.evaluate_action(db=db_session, task_id=task.id, action=payment_action)
    assert check.allowed is False
    assert check.requires_approval is True
    assert check.risk_level == ActionRiskLevel.HIGH_RISK_WRITE
    assert check.approval_request_id is not None

    # Verify task state paused to WAITING_FOR_APPROVAL
    updated_task = await TaskService.get_task(db=db_session, task_id=task.id)
    assert updated_task.status == TaskStatus.WAITING_FOR_APPROVAL.value

    # Verify approval request was saved in DB
    pending = await TaskService.get_pending_approval(db=db_session, task_id=task.id)
    assert pending is not None
    assert pending.risk_level == ActionRiskLevel.HIGH_RISK_WRITE.value
    assert "Changing payment information" in pending.justification


@pytest.mark.asyncio
async def test_approve_endpoint_resumes_execution(db_session: AsyncSession):
    """Verify POST /tasks/{task_id}/approve resolves approval and resumes execution."""
    task = TaskRun(goal="Sensitive task", status="WAITING_FOR_APPROVAL")
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)

    # Seed pending approval request
    approval = await TaskService.create_approval_request(
        db=db_session,
        task_id=task.id,
        proposed_action={
            "tool_name": "modify_bank_routing",
            "tool_args": {"routing": "987654"},
            "thought": "Authorizing routing modification",
        },
        risk_level=ActionRiskLevel.HIGH_RISK_WRITE.value,
        justification="Changing bank routing number requires approval.",
    )

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        # Check GET /tasks/{task_id}/approvals
        get_res = await client.get(f"/tasks/{task.id}/approvals")
        assert get_res.status_code == 200
        approvals_list = get_res.json()
        assert len(approvals_list) == 1
        assert approvals_list[0]["id"] == approval.id
        assert approvals_list[0]["status"] == "PENDING"

        # Call POST /tasks/{task_id}/approve
        response = await client.post(
            f"/tasks/{task.id}/approve",
            json={"comment": "Approved by CFO."},
        )
        assert response.status_code == 200
        body = response.json()
        # Confirm approval record resolved
        assert any(
            a["id"] == approval.id and a["status"] == "APPROVED"
            for a in body["approvals"]
        )


@pytest.mark.asyncio
async def test_reject_endpoint_halts_task_cleanly(db_session: AsyncSession):
    """Verify POST /tasks/{task_id}/reject resolves approval as REJECTED and terminates task as FAILED."""
    task = TaskRun(goal="Destructive delete", status="WAITING_FOR_APPROVAL")
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)

    approval = await TaskService.create_approval_request(
        db=db_session,
        task_id=task.id,
        proposed_action={
            "tool_name": "delete_invoice_tool",
            "tool_args": {"invoice_number": "INV-001"},
            "thought": "Deleting invoice permanently",
        },
        risk_level=ActionRiskLevel.DESTRUCTIVE.value,
        justification="Deleting invoice carries DESTRUCTIVE risk.",
    )

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            f"/tasks/{task.id}/reject",
            json={"comment": "Deletion forbidden by auditor."},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == TaskStatus.FAILED.value
        assert "rejected" in body["result_summary"].lower()

        # Confirm approval record status is REJECTED
        assert any(
            a["id"] == approval.id and a["status"] == "REJECTED"
            for a in body["approvals"]
        )


@pytest.mark.asyncio
async def test_approve_endpoint_returns_400_when_no_pending_approval(db_session: AsyncSession):
    """Verify that approving or rejecting when no pending approval exists returns 400 Bad Request."""
    task = TaskRun(goal="Running task", status="RUNNING")
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        res_approve = await client.post(f"/tasks/{task.id}/approve")
        assert res_approve.status_code == 400
        assert "no pending approval request found" in res_approve.json()["detail"].lower()

        res_reject = await client.post(f"/tasks/{task.id}/reject")
        assert res_reject.status_code == 400
        assert "no pending approval request found" in res_reject.json()["detail"].lower()

