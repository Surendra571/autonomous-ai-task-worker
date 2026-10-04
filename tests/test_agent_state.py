"""Unit tests for agent state machine and memory models."""

import pytest
from app.agent.state import (
    ActionProposal,
    ActionType,
    ExecutionStep,
    FailureRecord,
    Observation,
    TaskState,
    VerificationResult,
)
from app.models.schemas import RiskLevel, TaskStatus


def test_task_state_initialization():
    state = TaskState(
        task_id="test-task-123",
        goal="Extract invoice and enter into ledger",
        max_steps=10,
    )
    assert state.task_id == "test-task-123"
    assert state.current_step == 0
    assert state.status == TaskStatus.PENDING
    assert state.consecutive_failures == 0
    assert len(state.extracted_data) == 0


def test_state_observation_management():
    state = TaskState(task_id="t1", goal="Test goal")
    obs = Observation(
        step_number=1,
        source="document_read_tool",
        success=True,
        data={"invoice_number": "INV-100", "amount": 500.0},
    )
    state.add_observation(obs)
    assert len(state.recent_observations) == 1
    assert state.recent_observations[0].source == "document_read_tool"


def test_state_failure_tracking_and_recovery():
    state = TaskState(task_id="t2", goal="Test goal")
    state.current_step = 1

    state.record_failure("Network timeout", tool_name="browser_tool")
    assert state.consecutive_failures == 1
    assert len(state.failures) == 1
    assert state.failures[0].error_message == "Network timeout"

    state.record_failure("Second timeout", tool_name="browser_tool")
    assert state.consecutive_failures == 2

    # Successful step resets consecutive failures
    state.record_success()
    assert state.consecutive_failures == 0
    assert len(state.failures) == 2  # Total history preserved


def test_state_extracted_data_update():
    state = TaskState(task_id="t3", goal="Test goal")
    state.update_extracted_data({"vendor": "Acme Corp", "amount": 12450.50})
    assert state.extracted_data["vendor"] == "Acme Corp"
    assert state.extracted_data["amount"] == 12450.50

    state.update_extracted_data("invoice_number", "INV-ACM-2024-015")
    assert state.extracted_data["invoice_number"] == "INV-ACM-2024-015"


def test_action_proposal_validation():
    action = ActionProposal(
        thought="Need to parse PDF",
        action_type=ActionType.CALL_TOOL,
        tool_name="document_read_tool",
        tool_args={"file_path": "data/invoices/test.pdf"},
        risk_level=RiskLevel.LOW,
    )
    assert action.action_type == ActionType.CALL_TOOL
    assert action.tool_name == "document_read_tool"
    assert action.risk_level == RiskLevel.LOW
