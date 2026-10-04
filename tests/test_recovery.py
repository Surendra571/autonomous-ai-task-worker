"""Unit and integration tests for agent failure detection, classification, and recovery behavior."""

import pytest
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.executor import AgentExecutor
from app.agent.recovery import (
    FailureCategory,
    FailureClassifier,
    RecoveryDecision,
    RecoveryManager,
    RecoveryStrategy,
)
from app.agent.state import ActionProposal, ActionType, TaskState
from app.models.schemas import RiskLevel, TaskStatus
from app.models.task import TaskRun
from app.tools.base import BaseTool, ToolResult
from app.tools.registry import ToolRegistry


def test_failure_classifier_all_categories():
    """Verify that FailureClassifier correctly identifies all 10 specified error types."""
    # 1. Timeout
    assert FailureClassifier.classify("Async timeout 30000ms exceeded") == FailureCategory.TIMEOUT
    # 2. Missing element / button
    assert FailureClassifier.classify('Page.wait_for_selector: waiting for locator("[data-testid=\'save-btn\']")') == FailureCategory.MISSING_ELEMENT
    # 3. Browser navigation failure
    assert FailureClassifier.classify("net::ERR_CONNECTION_REFUSED at http://127.0.0.1:8765") == FailureCategory.BROWSER_NAVIGATION_FAILURE
    # 4. Authentication failure
    assert FailureClassifier.classify("401 Unauthorized: Session cookie expired or invalid credentials") == FailureCategory.AUTHENTICATION_FAILURE
    # 5. Duplicate invoice
    assert FailureClassifier.classify("Invoice number 'INV-ACM-2024-001' already exists in finance ledger.") == FailureCategory.DUPLICATE_INVOICE
    # 6. Malformed document
    assert FailureClassifier.classify("Cannot open file as PDF: Corrupt document header") == FailureCategory.MALFORMED_DOCUMENT
    # 7. Missing invoice
    assert FailureClassifier.classify("File not found: data/invoices/umbrella_invoice.pdf does not exist") == FailureCategory.MISSING_INVOICE
    # 8. Finance validation error
    assert FailureClassifier.classify("Finance ledger validation error: Amount must be positive") == FailureCategory.FINANCE_VALIDATION_ERROR
    # 9. Invalid input
    assert FailureClassifier.classify("Input validation failed for tool 'document_search_tool': Field required") == FailureCategory.INVALID_INPUT
    # 10. Generic tool failure
    assert FailureClassifier.classify("Internal OS process error") == FailureCategory.GENERIC_TOOL_FAILURE


@pytest.mark.asyncio
async def test_recovery_missing_button():
    """Test detection and recovery from missing button / selector."""
    manager = RecoveryManager(max_retries=3)
    state = TaskState(task_id="t1", goal="Click submit button")
    action = ActionProposal(
        thought="Clicking invoice submit button",
        action_type=ActionType.CALL_TOOL,
        tool_name="browser_tool",
        tool_args={"action": "click", "selector": "#btn-submit-invoice"},
    )
    error = 'TimeoutError: waiting for locator("#btn-submit-invoice") to be visible'

    # Attempt 1: Re-observe DOM with screenshot
    decision_1 = manager.decide_recovery(state, action, error)
    assert decision_1.failure_category == FailureCategory.MISSING_ELEMENT
    assert decision_1.recovery_strategy == RecoveryStrategy.RE_OBSERVE_ENVIRONMENT
    assert decision_1.recovery_action.tool_args["action"] == "screenshot"
    assert any("ACTION_FAILED" in ev for ev in decision_1.event_sequence)
    assert any("RECOVERY_DECISION" in ev for ev in decision_1.event_sequence)

    # Attempt 2: Switch to alternative tool
    decision_2 = manager.decide_recovery(state, action, error)
    assert decision_2.recovery_strategy == RecoveryStrategy.TRY_ALTERNATIVE_TOOL
    assert decision_2.recovery_action.tool_name == "finance_create_invoice_tool"


@pytest.mark.asyncio
async def test_recovery_temporary_tool_failure(db_session: AsyncSession):
    """Test transient tool failure recovery via bounded retry in AgentExecutor."""
    class FlakyInput(BaseModel):
        dummy: str = ""

    class FlakyTool(BaseTool):
        name: str = "flaky_tool"
        description: str = "Tool that fails once then succeeds"
        input_schema: type[BaseModel] = FlakyInput
        risk_level: str = "LOW"

        def __init__(self):
            self.attempts = 0

        async def execute(self, params: FlakyInput, session=None) -> ToolResult:
            self.attempts += 1
            if self.attempts == 1:
                return ToolResult.fail("Transient socket timeout connecting to remote service")
            return ToolResult.ok(data={"status": "recovered_successfully"})

    tool = FlakyTool()
    registry = ToolRegistry()
    registry.register(tool)

    task = TaskRun(goal="Execute flaky tool", status="PENDING")
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)

    class FlakyPlanner:
        def __init__(self):
            self.calls = 0

        async def decide_next_action(self, state: TaskState) -> ActionProposal:
            self.calls += 1
            if self.calls == 1:
                return ActionProposal(
                    thought="Calling flaky tool",
                    action_type=ActionType.CALL_TOOL,
                    tool_name="flaky_tool",
                    tool_args={"dummy": "val"},
                    risk_level=RiskLevel.LOW,
                )
            return ActionProposal(
                thought="Finished after recovery",
                action_type=ActionType.COMPLETE_TASK,
                tool_name=None,
                tool_args={},
                risk_level=RiskLevel.LOW,
            )

    executor = AgentExecutor(tool_registry=registry, planner=FlakyPlanner(), max_steps=5)
    final_state = await executor.execute_task(db=db_session, task_id=task.id)

    # The tool failed on step 1, executor invoked recovery -> RETRY_SAME_ACTION -> succeeded on step 2!
    assert final_state.current_status == TaskStatus.COMPLETED
    assert tool.attempts == 2
    # Verify execution event sequence recorded in history
    recovery_events = [h for h in final_state.execution_history if h.get("action") == "RECOVERY_EVENT"]
    assert len(recovery_events) >= 1
    assert any("ACTION_FAILED" in ev["result"] for ev in recovery_events)
    assert any("RETRY_SAME_ACTION" in ev["result"] for ev in recovery_events)


@pytest.mark.asyncio
async def test_recovery_invalid_invoice_data():
    """Test automatic sanitization and retry with corrected arguments for validation errors."""
    manager = RecoveryManager(max_retries=3)
    state = TaskState(task_id="t2", goal="Create invoice")
    failed_action = ActionProposal(
        thought="Creating invoice with invalid negative amount and malformed date",
        action_type=ActionType.CALL_TOOL,
        tool_name="finance_create_invoice_tool",
        tool_args={
            "invoice_number": "INV-TEST-009",
            "vendor_name": "Globex",
            "amount": -5500.25,
            "due_date": "2024/11/05",
            "issue_date": "2024.10.01",
        },
        risk_level=RiskLevel.HIGH,
    )
    error = "Finance ledger validation error: amount must be positive and date format invalid"

    decision = manager.decide_recovery(state, failed_action, error)
    assert decision.failure_category == FailureCategory.FINANCE_VALIDATION_ERROR
    assert decision.recovery_strategy == RecoveryStrategy.RETRY_WITH_CORRECTED_ARGS
    assert decision.is_recoverable is True

    # Check argument corrections
    args = decision.recovery_action.tool_args
    assert args["amount"] == 5500.25  # Sanitized to positive
    assert args["due_date"] == "2024-11-05"  # Standardized to YYYY-MM-DD
    assert args["issue_date"] == "2024-10-01"


@pytest.mark.asyncio
async def test_recovery_duplicate_invoice():
    """Test that duplicate invoice detection triggers alternative read/verify tool rather than crashing."""
    manager = RecoveryManager(max_retries=3)
    state = TaskState(task_id="t3", goal="Create and verify invoice")
    failed_action = ActionProposal(
        thought="Inserting invoice into ledger",
        action_type=ActionType.CALL_TOOL,
        tool_name="finance_create_invoice_tool",
        tool_args={
            "invoice_number": "INV-EXISTING-101",
            "vendor_name": "Initech",
            "amount": 3200.0,
        },
        risk_level=RiskLevel.HIGH,
    )
    error = "Finance ledger validation error: Invoice number 'INV-EXISTING-101' already exists."

    decision = manager.decide_recovery(state, failed_action, error)
    assert decision.failure_category == FailureCategory.DUPLICATE_INVOICE
    assert decision.recovery_strategy == RecoveryStrategy.TRY_ALTERNATIVE_TOOL
    assert decision.recovery_action.tool_name == "finance_read_invoice_tool"
    assert decision.recovery_action.tool_args["invoice_number"] == "INV-EXISTING-101"
    assert decision.recovery_action.verification_needed is True

    # Trajectory check
    assert decision.event_sequence == [
        "ACTION_FAILED (finance_create_invoice_tool)",
        "OBSERVE (Diagnosed: DUPLICATE_INVOICE)",
        "RECOVERY_DECISION (TRY_ALTERNATIVE_TOOL)",
        "ALTERNATIVE_ACTION (finance_read_invoice_tool)",
        "VERIFY",
    ]


@pytest.mark.asyncio
async def test_bounded_retries_aborts_cleanly():
    """Verify that exceeding bounded retries aborts with unrecoverable failure."""
    manager = RecoveryManager(max_retries=2)
    state = TaskState(task_id="t4", goal="Failing task")
    action = ActionProposal(
        thought="Failing action",
        action_type=ActionType.CALL_TOOL,
        tool_name="flaky_tool",
        tool_args={},
    )
    error = "Network timeout"

    # Attempt 1 -> retry
    d1 = manager.decide_recovery(state, action, error)
    assert d1.is_recoverable is True
    # Attempt 2 -> retry
    d2 = manager.decide_recovery(state, action, error)
    assert d2.is_recoverable is True
    # Attempt 3 -> exceeded max_retries (2) -> abort!
    d3 = manager.decide_recovery(state, action, error)
    assert d3.is_recoverable is False
    assert d3.recovery_strategy == RecoveryStrategy.ABORT_UNRECOVERABLE
    assert "reached maximum retry budget" in d3.reason
