"""Comprehensive unit and integration tests for the dedicated VerificationEngine."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.executor import AgentExecutor
from app.agent.state import ActionProposal, ActionType, TaskState
from app.models.invoice import Invoice
from app.models.schemas import RiskLevel, TaskStatus
from app.models.task import TaskRun
from app.services.task_service import TaskService
from app.tools.registry import ToolRegistry
from app.verification.engine import (
    VerificationCheck,
    VerificationEngine,
    VerificationReport,
)


def test_verification_engine_all_7_checks_pass():
    """Verify that all 7 required criteria pass when stored values match source values."""
    expected = {
        "vendor_name": "Acme Corp",
        "invoice_number": "INV-ACM-2024-001",
        "amount": 12500.50,
        "currency": "USD",
        "due_date": "2024-11-15",
    }
    actual = {
        "vendor_name": "Acme Corp",
        "invoice_number": "INV-ACM-2024-001",
        "amount": 12500.50,
        "currency": "USD",
        "due_date": "2024-11-15",
    }

    report = VerificationEngine.compare_invoice_fields(expected=expected, actual=actual)

    # Must be verified
    assert report.verified is True
    assert len(report.discrepancies) == 0

    # 1. Correct vendor
    assert any(c.name == "correct_vendor" and c.passed for c in report.checks)
    assert any(c.name == "vendor_matches" and c.passed for c in report.checks)

    # 2. Correct invoice number
    assert any(c.name == "correct_invoice_number" and c.passed for c in report.checks)
    assert any(c.name == "invoice_number_matches" and c.passed for c in report.checks)

    # 3. Correct amount
    assert any(c.name == "correct_amount" and c.passed for c in report.checks)
    assert any(c.name == "amount_matches" and c.passed for c in report.checks)

    # 4. Correct currency
    assert any(c.name == "correct_currency" and c.passed for c in report.checks)
    assert any(c.name == "currency_matches" and c.passed for c in report.checks)

    # 5. Correct due date
    assert any(c.name == "correct_due_date" and c.passed for c in report.checks)
    assert any(c.name == "due_date_matches" and c.passed for c in report.checks)

    # 6. Invoice exists in finance system
    assert any(c.name == "invoice_exists" and c.passed for c in report.checks)

    # 7. Stored values match extracted source values
    assert any(c.name == "stored_values_match_source" and c.passed for c in report.checks)

    # Check serialization format matches exact specification:
    # { "verified": true, "checks": [...], "evidence": [...] }
    result_dict = report.to_dict()
    assert result_dict["verified"] is True
    assert isinstance(result_dict["checks"], list)
    assert isinstance(result_dict["evidence"], list)
    assert any(c["name"] == "invoice_exists" and c["passed"] is True for c in result_dict["checks"])
    assert any(c["name"] == "amount_matches" and c["passed"] is True for c in result_dict["checks"])


def test_verification_engine_invoice_not_found():
    """Verify that when the invoice does not exist in the finance ledger, verification fails."""
    expected = {
        "vendor_name": "Globex Corporation",
        "invoice_number": "INV-GLO-9999",
        "amount": 4200.00,
        "currency": "USD",
        "due_date": "2024-12-01",
    }
    actual = None  # Invoice does not exist

    report = VerificationEngine.compare_invoice_fields(expected=expected, actual=actual)

    assert report.verified is False
    assert any(c.name == "invoice_exists" and c.passed is False for c in report.checks)
    assert any(c.name == "stored_values_match_source" and c.passed is False for c in report.checks)
    assert len(report.discrepancies) > 0
    assert "does not exist in the finance application" in report.discrepancies[0]

    result_dict = report.to_dict()
    assert result_dict["verified"] is False
    assert any(c["name"] == "invoice_exists" and c["passed"] is False for c in result_dict["checks"])


def test_verification_engine_field_discrepancies():
    """Verify that field discrepancies (wrong amount, date, vendor, currency) are flagged."""
    expected = {
        "vendor_name": "Initech LLC",
        "invoice_number": "INV-INI-2024-005",
        "amount": 999.99,
        "currency": "USD",
        "due_date": "2024-10-15",
    }
    actual = {
        "vendor_name": "Different Corp",
        "invoice_number": "INV-INI-2024-005",
        "amount": 500.00,  # Mismatched
        "currency": "EUR",  # Mismatched
        "due_date": "2024-11-20",  # Mismatched
    }

    report = VerificationEngine.compare_invoice_fields(expected=expected, actual=actual)

    assert report.verified is False
    assert any(c.name == "invoice_exists" and c.passed is True for c in report.checks)
    assert any(c.name == "vendor_matches" and c.passed is False for c in report.checks)
    assert any(c.name == "amount_matches" and c.passed is False for c in report.checks)
    assert any(c.name == "currency_matches" and c.passed is False for c in report.checks)
    assert any(c.name == "due_date_matches" and c.passed is False for c in report.checks)
    assert any(c.name == "stored_values_match_source" and c.passed is False for c in report.checks)

    assert len(report.discrepancies) >= 4


def test_verification_engine_does_not_trust_planner_claim():
    """Verify that verifier does not trust the planner's assertion and evaluates ground truth."""
    # Planner claims 1000.00 in memory
    planner_fabricated_state = {
        "vendor_name": "Acme Corp",
        "invoice_number": "INV-ACM-FAKE",
        "amount": 1000.00,
        "currency": "USD",
        "due_date": "2024-10-10",
    }
    # But actual record in finance has different amount
    actual_ledger_record = {
        "vendor_name": "Acme Corp",
        "invoice_number": "INV-ACM-FAKE",
        "amount": 2500.00,  # Real database value
        "currency": "USD",
        "due_date": "2024-10-10",
    }

    report = VerificationEngine.compare_invoice_fields(
        expected=planner_fabricated_state,
        actual=actual_ledger_record,
    )

    assert report.verified is False
    assert any(c.name == "amount_matches" and c.passed is False for c in report.checks)


@pytest.mark.asyncio
async def test_verification_engine_database_retrieval_and_evidence(db_session: AsyncSession):
    """Verify that VerificationEngine queries the database independently and attaches evidence."""
    # 1. Create a task in DB
    task = TaskRun(goal="Verify Acme Corp invoice", status="RUNNING")
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)

    # 2. Seed invoice in finance ledger
    invoice = Invoice(
        invoice_number="INV-ACM-DB-001",
        vendor_name="Acme Corp",
        amount=7450.00,
        currency="USD",
        issue_date="2024-06-01",
        due_date="2024-06-30",
        status="PENDING",
    )
    db_session.add(invoice)
    await db_session.commit()

    state = TaskState(
        task_id=task.id,
        goal="Enter and verify Acme invoice",
        extracted_data={
            "vendor_name": "Acme Corp",
            "invoice_number": "INV-ACM-DB-001",
            "amount": 7450.00,
            "currency": "USD",
            "due_date": "2024-06-30",
        },
    )

    report = await VerificationEngine.verify_task(state=state, db=db_session)

    assert report.verified is True
    assert state.extracted_data.get("verified_in_finance") is True
    assert state.latest_verification is not None
    assert state.latest_verification.verified is True

    # Confirm audit evidence was saved into task
    task_fetched = await TaskService.get_task(db_session, task.id)
    assert len(task_fetched.evidence) >= 1
    assert task_fetched.evidence[0].evidence_type == "outcome_verification"


@pytest.mark.asyncio
async def test_verification_engine_browser_dom_extraction():
    """Verify that VerificationEngine can extract invoice fields from browser DOM."""
    mock_page = MagicMock()
    mock_page.content = AsyncMock(return_value="<h1>Invoice Details</h1><span class='invoice-number'>INV-ACM-DOM-001</span>")

    async def mock_query_selector(selector):
        element = MagicMock()
        if "invoice-number" in selector:
            element.inner_text = AsyncMock(return_value="INV-ACM-DOM-001")
        elif "vendor" in selector:
            element.inner_text = AsyncMock(return_value="Acme Corp")
        elif "amount" in selector:
            element.inner_text = AsyncMock(return_value="1500.00")
        elif "currency" in selector:
            element.inner_text = AsyncMock(return_value="USD")
        elif "due-date" in selector:
            element.inner_text = AsyncMock(return_value="2024-12-31")
        else:
            return None
        return element

    mock_page.query_selector = AsyncMock(side_effect=mock_query_selector)

    data = await VerificationEngine.retrieve_from_browser(page=mock_page)
    assert data is not None
    assert data["invoice_number"] == "INV-ACM-DOM-001"
    assert data["vendor_name"] == "Acme Corp"
    assert data["amount"] == "1500.00"


@pytest.mark.asyncio
async def test_executor_completion_requires_verification_and_returns_state_on_failure(
    db_session: AsyncSession,
):
    """Verify that AgentExecutor only completes on successful verification.

    If verification fails, the state returns to planner to decide what to do next.
    """
    task = TaskRun(goal="Enter invoice and verify", status="PENDING")
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)

    # Planner attempts to complete task on step 1 with incorrect amount,
    # then observes the verification failure, corrects the amount on step 2,
    # and succeeds!
    class SelfCorrectingPlanner:
        def __init__(self):
            self.step = 0

        async def decide_next_action(self, state: TaskState) -> ActionProposal:
            self.step += 1
            if self.step == 1:
                # Deliberately propose completion when invoice doesn't exist yet!
                state.extracted_data["invoice_number"] = "INV-NONEXISTENT-001"
                state.extracted_data["amount"] = 999.00
                return ActionProposal(
                    thought="Claiming task is complete prematurely",
                    action_type=ActionType.COMPLETE_TASK,
                    tool_name=None,
                    tool_args={},
                    risk_level=RiskLevel.LOW,
                )
            elif self.step == 2:
                # Planner received verification rejection in observations!
                # Now it fixes the state and points to the valid invoice in ledger
                state.extracted_data["invoice_number"] = "INV-ACM-VALID"
                state.extracted_data["vendor_name"] = "Acme Corp"
                state.extracted_data["amount"] = 5000.00
                state.extracted_data["currency"] = "USD"
                state.extracted_data["due_date"] = "2024-08-15"
                return ActionProposal(
                    thought="Retrying completion after correcting invoice number",
                    action_type=ActionType.COMPLETE_TASK,
                    tool_name=None,
                    tool_args={},
                    risk_level=RiskLevel.LOW,
                )
            return ActionProposal(
                thought="Should not be reached",
                action_type=ActionType.FAIL_TASK,
                tool_name=None,
                tool_args={},
                risk_level=RiskLevel.LOW,
            )

    # Seed the valid invoice in DB
    valid_inv = Invoice(
        invoice_number="INV-ACM-VALID",
        vendor_name="Acme Corp",
        amount=5000.00,
        currency="USD",
        issue_date="2024-08-01",
        due_date="2024-08-15",
        status="PENDING",
    )
    db_session.add(valid_inv)
    await db_session.commit()

    executor = AgentExecutor(
        planner=SelfCorrectingPlanner(),
        max_steps=5,
    )
    final_state = await executor.execute_task(db=db_session, task_id=task.id)

    # Step 1: Failed verification -> loop continued!
    # Step 2: Corrected -> verified -> COMPLETED!
    assert final_state.current_status == TaskStatus.COMPLETED
    assert final_state.extracted_data.get("verified_in_finance") is True

    # Confirm that step 1 verification rejection was recorded in observations
    rejection_observations = [
        obs for obs in final_state.observations
        if obs.source == "VerificationEngine" and not obs.success
    ]
    assert len(rejection_observations) >= 1
    assert "Verification failed" in rejection_observations[0].error
