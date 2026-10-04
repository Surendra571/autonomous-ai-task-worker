"""Unit and scenario tests for the LLM-based structured planner."""

import json
from unittest.mock import AsyncMock, MagicMock
import pytest
from pydantic import ValidationError

from app.agent.planner import Planner
from app.agent.prompts import PLANNER_SYSTEM_PROMPT, format_planner_user_prompt
from app.agent.state import ActionCall, ActionProposal, ActionType, PlannerDecision, TaskState
from app.models.schemas import RiskLevel, TaskStatus
from app.tools.registry import create_default_registry


@pytest.fixture
def planner():
    """Provides a Planner instance configured with default tools."""
    registry = create_default_registry()
    return Planner(tool_registry=registry)


@pytest.mark.asyncio
async def test_scenario_1_initial_step_document_search(planner: Planner):
    """Scenario 1: Given user goal for Acme Corp, first action is document search."""
    decision = await planner.plan(
        user_goal="Find the latest invoice from Acme Corp, extract details, and save to finance app.",
        current_state={"status": "RUNNING", "current_step": 1, "retry_count": 0},
        execution_history=[],
        latest_observation=None,
        available_tools=planner.tool_registry.get_schemas(),
        known_data={},
        previous_failures=[],
    )

    assert isinstance(decision, PlannerDecision)
    assert decision.action.tool == "document_search_tool"
    assert decision.action.arguments.get("vendor_name") == "Acme Corp"
    assert decision.action.arguments.get("find_latest") is True
    assert decision.requires_approval is False
    assert decision.verification_needed is False


@pytest.mark.asyncio
async def test_scenario_2_document_read_does_not_repeat_search(planner: Planner):
    """Scenario 2: When document path is known, proceed to document_read without repeating search."""
    known = {"latest_invoice_path": "data/invoices/acme_invoice_2024_09.pdf", "vendor_name": "Acme Corp"}
    decision = await planner.plan(
        user_goal="Find the latest invoice from Acme Corp, extract details, and save to finance app.",
        current_state={"status": "RUNNING", "current_step": 2, "retry_count": 0},
        execution_history=[{"action": "document_search_tool", "result": "Found acme_invoice_2024_09.pdf"}],
        latest_observation={"success": True, "data": known},
        available_tools=planner.tool_registry.get_schemas(),
        known_data=known,
        previous_failures=[],
    )

    assert decision.action.tool == "document_read_tool"
    assert decision.action.arguments.get("file_path") == "data/invoices/acme_invoice_2024_09.pdf"
    assert decision.requires_approval is False


@pytest.mark.asyncio
async def test_scenario_3_finance_create_is_risky_requires_approval(planner: Planner):
    """Scenario 3: Entering an invoice is a risky action that requires human approval and verification."""
    extracted = {
        "latest_invoice_path": "data/invoices/acme_invoice_2024_09.pdf",
        "vendor_name": "Acme Corp",
        "invoice_number": "INV-2024-003",
        "amount": 4250.00,
        "currency": "USD",
        "due_date": "2024-10-15",
        "invoice_date": "2024-09-28",
        "pdf_extracted": True,
    }

    decision = await planner.plan(
        user_goal="Find the latest invoice from Acme Corp and enter it into finance.",
        current_state={"status": "RUNNING", "current_step": 3, "retry_count": 0},
        execution_history=[{"action": "document_read_tool", "result": "Extracted INV-2024-003"}],
        latest_observation={"success": True, "data": extracted},
        available_tools=planner.tool_registry.get_schemas(),
        known_data=extracted,
        previous_failures=[],
    )

    assert decision.action.tool == "finance_create_invoice_tool"
    assert decision.action.arguments["invoice_number"] == "INV-2024-003"
    assert decision.action.arguments["amount"] == 4250.00
    assert decision.requires_approval is True
    assert decision.verification_needed is True


@pytest.mark.asyncio
async def test_scenario_4_never_claims_success_without_verification(planner: Planner):
    """Scenario 4: Once invoice is entered, agent verifies before completing."""
    data = {
        "vendor_name": "Acme Corp",
        "invoice_number": "INV-2024-003",
        "amount": 4250.00,
        "currency": "USD",
        "due_date": "2024-10-15",
        "latest_invoice_path": "data/invoices/acme_invoice_2024_09.pdf",
        "pdf_extracted": True,
        "invoice_entered": True,
        "verified_in_finance": False,
    }

    decision = await planner.plan(
        user_goal="Find latest invoice from Acme Corp, save and verify.",
        current_state={"status": "RUNNING", "current_step": 4, "retry_count": 0},
        execution_history=[{"action": "finance_create_invoice_tool", "result": "Invoice created"}],
        latest_observation={"success": True, "data": {"invoice_id": "test-id"}},
        available_tools=planner.tool_registry.get_schemas(),
        known_data=data,
        previous_failures=[],
    )

    # Must NOT complete task yet; must read/verify first
    assert decision.action.tool == "finance_read_invoice_tool"
    assert decision.action.arguments.get("invoice_number") == "INV-2024-003"
    assert decision.verification_needed is True


@pytest.mark.asyncio
async def test_scenario_5_complete_task_after_successful_verification(planner: Planner):
    """Scenario 5: Only when verification is complete does planner choose complete_task."""
    data = {
        "vendor_name": "Acme Corp",
        "invoice_number": "INV-2024-003",
        "amount": 4250.00,
        "pdf_extracted": True,
        "invoice_entered": True,
        "verified_in_finance": True,
    }

    decision = await planner.plan(
        user_goal="Find latest invoice from Acme Corp, save and verify.",
        current_state={"status": "RUNNING", "current_step": 5, "retry_count": 0},
        execution_history=[{"action": "finance_read_invoice_tool", "result": "Verified in finance ledger"}],
        latest_observation={"success": True, "data": {"verified": True}},
        available_tools=planner.tool_registry.get_schemas(),
        known_data=data,
        previous_failures=[],
    )

    assert decision.action.tool == "complete_task"
    assert decision.requires_approval is False


@pytest.mark.asyncio
async def test_scenario_6_tool_failure_recovery_retry(planner: Planner):
    """Scenario 6: When tool encounters an error, planner decides to retry or recover."""
    known = {"latest_invoice_path": "data/invoices/acme_invoice_2024_09.pdf"}
    decision = await planner.plan(
        user_goal="Find latest invoice from Acme Corp and extract.",
        current_state={"status": "RUNNING", "current_step": 2, "retry_count": 0},
        execution_history=[],
        latest_observation={"success": False, "error": "Transient file lock on PDF document_read_tool"},
        available_tools=planner.tool_registry.get_schemas(),
        known_data=known,
        previous_failures=[{"tool": "document_read_tool", "error": "file lock"}],
    )

    assert decision.action.tool == "document_read_tool"
    assert "retrying" in decision.reason.lower()


@pytest.mark.asyncio
async def test_scenario_7_unrecoverable_repeated_failures_stops_task(planner: Planner):
    """Scenario 7: Exceeded retry attempts causes fail_task decision."""
    decision = await planner.plan(
        user_goal="Find latest invoice from Acme Corp.",
        current_state={"status": "RUNNING", "current_step": 4, "retry_count": 3},
        execution_history=[],
        latest_observation={"success": False, "error": "Corrupt file format"},
        available_tools=planner.tool_registry.get_schemas(),
        known_data={"latest_invoice_path": "bad_file.pdf"},
        previous_failures=["error 1", "error 2", "error 3"],
    )

    assert decision.action.tool == "fail_task"
    assert "stopping" in decision.reason.lower() or "failed repeatedly" in decision.reason.lower()


@pytest.mark.asyncio
async def test_scenario_8_missing_information_requests_clarification(planner: Planner):
    """Scenario 8: If user goal omits key context like vendor name, request clarification."""
    decision = await planner.plan(
        user_goal="Process invoice and enter into ledger.",
        current_state={"status": "RUNNING", "current_step": 1, "retry_count": 0},
        execution_history=[],
        latest_observation=None,
        available_tools=planner.tool_registry.get_schemas(),
        known_data={},
        previous_failures=[],
    )

    assert decision.action.tool == "request_clarification"
    assert "question" in decision.action.arguments


def test_pydantic_structured_output_validation():
    """Verify strict Pydantic model validation on PlannerDecision and ActionCall."""
    valid_payload = {
        "reason": "Need to search for Acme invoices.",
        "action": {
            "tool": "document_search_tool",
            "arguments": {"vendor_name": "Acme Corp", "find_latest": True},
        },
        "expected_outcome": "List of invoices",
        "verification_needed": False,
        "requires_approval": False,
    }

    decision = PlannerDecision.model_validate(valid_payload)
    assert decision.action.tool == "document_search_tool"
    assert decision.requires_approval is False

    # Free-form / missing required fields must fail validation
    with pytest.raises(ValidationError):
        PlannerDecision.model_validate({"reason": "Just do it"})

    with pytest.raises(ValidationError):
        PlannerDecision.model_validate({
            "reason": "Testing",
            "action": "free-form string instead of object",
            "expected_outcome": "outcome",
        })


def test_clean_json_response_parsing(planner: Planner):
    """Ensure markdown-wrapped and whitespace-padded JSON is cleaned and parsed properly."""
    raw_markdown = """```json
{
  "reason": "Extract fields from PDF.",
  "action": {
    "tool": "document_read_tool",
    "arguments": {"file_path": "data/invoices/acme.pdf"}
  },
  "expected_outcome": "Invoice data",
  "verification_needed": false,
  "requires_approval": false
}
```"""
    parsed = planner._clean_json_response(raw_markdown)
    assert parsed["action"]["tool"] == "document_read_tool"

    # Non-JSON text should raise ValueError
    with pytest.raises(ValueError):
        planner._clean_json_response("I think we should call document_read_tool now.")


@pytest.mark.asyncio
async def test_llm_structured_output_mocking():
    """Test LLM client integration using a mocked OpenAI chat completion."""
    mock_client = MagicMock()
    mock_chat = MagicMock()
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()

    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = json.dumps({
        "reason": "LLM decided to create invoice.",
        "action": {
            "tool": "finance_create_invoice_tool",
            "arguments": {"invoice_number": "INV-MOCK-99", "amount": 100.0},
        },
        "expected_outcome": "Invoice registered",
        "verification_needed": True,
        "requires_approval": True,
    })
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    planner = Planner(client=mock_client)
    decision = await planner.plan(
        user_goal="Enter mock invoice",
        current_state={},
        execution_history=[],
        latest_observation=None,
        available_tools=[],
        known_data={},
        previous_failures=[],
    )

    assert decision.action.tool == "finance_create_invoice_tool"
    assert decision.requires_approval is True
    assert decision.verification_needed is True
    assert decision.action.arguments["invoice_number"] == "INV-MOCK-99"


@pytest.mark.asyncio
async def test_planner_decide_next_action_interoperability():
    """Ensure decide_next_action bridges TaskState and ActionProposal smoothly."""
    planner = Planner()
    state = TaskState(
        task_id="test-task-1",
        user_goal="Find latest invoice from Globex and process it.",
        current_status=TaskStatus.RUNNING,
    )

    proposal = await planner.decide_next_action(state)
    assert isinstance(proposal, ActionProposal)
    assert proposal.tool_name == "document_search_tool"
    assert proposal.tool_args.get("vendor_name") == "Globex"
    assert state.current_objective is not None
    assert state.plan_summary is not None

