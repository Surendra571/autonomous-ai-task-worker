"""Unit tests for the ApprovalGate mechanism."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.state import ActionProposal, ActionType
from app.approval.gate import ApprovalGate
from app.models.schemas import RiskLevel, TaskStatus
from app.models.task import TaskRun
from app.services.task_service import TaskService


@pytest.mark.asyncio
async def test_low_risk_action_allowed_without_approval(db_session: AsyncSession):
    task = TaskRun(goal="Read doc", status="RUNNING")
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)

    action = ActionProposal(
        thought="Reading a document is safe.",
        action_type=ActionType.CALL_TOOL,
        tool_name="document_read_tool",
        tool_args={"file_path": "data/invoices/acme_invoice_2024_09.pdf"},
        risk_level=RiskLevel.LOW,
    )

    check = await ApprovalGate.evaluate_action(
        db=db_session,
        task_id=task.id,
        action=action,
    )

    assert check.allowed is True
    assert check.requires_approval is False


@pytest.mark.asyncio
async def test_high_risk_action_intercepted(db_session: AsyncSession):
    task = TaskRun(goal="Create invoice", status="RUNNING")
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)

    action = ActionProposal(
        thought="Creating a new financial ledger entry.",
        action_type=ActionType.CALL_TOOL,
        tool_name="finance_create_invoice_tool",
        tool_args={
            "invoice_number": "INV-RISK-001",
            "vendor_name": "Acme Corp",
            "amount": 9999.00,
        },
        risk_level=RiskLevel.HIGH,
    )

    check = await ApprovalGate.evaluate_action(
        db=db_session,
        task_id=task.id,
        action=action,
    )

    assert check.allowed is False
    assert check.requires_approval is True
    assert check.approval_request_id is not None

    # Verify task status was updated to WAITING_FOR_APPROVAL
    updated_task = await TaskService.get_task(db_session, task.id)
    assert updated_task.status == TaskStatus.WAITING_FOR_APPROVAL.value

    # Verify approval record was created
    pending = await TaskService.get_pending_approval(db_session, task.id)
    assert pending is not None
    assert pending.id == check.approval_request_id
    assert pending.proposed_action["tool_name"] == "finance_create_invoice_tool"
