"""Unit tests for the independent OutcomeVerifier."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.invoice import Invoice
from app.models.schemas import InvoiceCreate
from app.models.task import TaskRun
from app.services.task_service import TaskService
from app.verification.verifier import OutcomeVerifier


def test_verify_invoice_fields_matching():
    expected = {
        "invoice_number": "INV-ACM-2024-015",
        "vendor_name": "Acme Corp",
        "amount": 12450.50,
        "due_date": "2024-10-28",
    }
    actual = {
        "invoice_number": "INV-ACM-2024-015",
        "vendor_name": "Acme Corp",
        "amount": 12450.50,
        "due_date": "2024-10-28",
    }
    result = OutcomeVerifier.verify_invoice_fields(expected, actual)
    assert result.verified is True
    assert len(result.discrepancies) == 0


def test_verify_invoice_fields_discrepancies():
    expected = {
        "invoice_number": "INV-ACM-2024-015",
        "vendor_name": "Acme Corp",
        "amount": 12450.50,
        "due_date": "2024-10-28",
    }
    actual = {
        "invoice_number": "INV-ACM-2024-015",
        "vendor_name": "Acme Corp",
        "amount": 10000.00,  # Mismatched amount
        "due_date": "2024-11-01",  # Mismatched due date
    }
    result = OutcomeVerifier.verify_invoice_fields(expected, actual)
    assert result.verified is False
    assert len(result.discrepancies) == 2
    assert any("Amount mismatch" in d for d in result.discrepancies)
    assert any("Due date mismatch" in d for d in result.discrepancies)


@pytest.mark.asyncio
async def test_verify_saved_invoice_in_db_success(db_session: AsyncSession):
    # Seed task
    task = TaskRun(goal="Verify test", status="RUNNING")
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)

    # Seed invoice
    inv = Invoice(
        invoice_number="INV-VERIFY-001",
        vendor_name="Acme Corp",
        amount=5500.00,
        currency="USD",
        issue_date="2024-05-01",
        due_date="2024-05-31",
        status="PENDING",
    )
    db_session.add(inv)
    await db_session.commit()

    expected = {
        "invoice_number": "INV-VERIFY-001",
        "vendor_name": "Acme Corp",
        "amount": 5500.00,
        "due_date": "2024-05-31",
    }

    result = await OutcomeVerifier.verify_saved_invoice_in_db(
        db=db_session,
        task_id=task.id,
        expected_data=expected,
        invoice_number="INV-VERIFY-001",
    )

    assert result.verified is True
    assert len(result.discrepancies) == 0

    # Verify evidence was added to task
    task_fetched = await TaskService.get_task(db_session, task.id)
    assert len(task_fetched.evidence) == 1
    assert task_fetched.evidence[0].evidence_type == "outcome_verification"
