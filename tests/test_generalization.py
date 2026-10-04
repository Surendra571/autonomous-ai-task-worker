"""Integration and unit tests for task generalization, generic verification strategies, and TaskOutcome contracts."""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.executor import AgentExecutor
from app.agent.planner import Planner
from app.agent.state import TaskOutcome, TaskState
from app.models.invoice import Invoice
from app.models.schemas import TaskStatus
from app.models.task import TaskRun
from app.tools.registry import create_default_registry
from app.verification.strategy import (
    GenericToolVerificationStrategy,
    InvoiceVerificationStrategy,
    VerificationStrategyRegistry,
)


@pytest.mark.asyncio
async def test_generalization_task_1_find_and_verify_document_existence(
    db_session: AsyncSession,
):
    """Task 1: 'Find the latest Acme invoice and verify whether it exists.'
    Demonstrates document discovery and read without financial ledger mutation.
    """
    task = TaskRun(
        goal="Find the latest Acme invoice and verify whether it exists.",
        status="PENDING",
        max_steps=5,
    )
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)

    registry = create_default_registry()
    planner = Planner(tool_registry=registry)
    executor = AgentExecutor(tool_registry=registry, planner=planner, max_steps=5)

    final_state = await executor.execute_task(db=db_session, task_id=task.id)

    assert final_state.current_status == TaskStatus.COMPLETED
    assert final_state.current_step >= 2
    assert "latest_invoice_path" in final_state.extracted_data
    assert "acme" in final_state.extracted_data["latest_invoice_path"].lower()

    # Verify TaskOutcome contract
    assert final_state.outcome is not None
    assert isinstance(final_state.outcome, TaskOutcome)
    assert final_state.outcome.success is True
    assert final_state.outcome.status == TaskStatus.COMPLETED
    assert "latest_invoice_path" in final_state.outcome.facts
    assert final_state.outcome.verification is not None
    assert final_state.outcome.verification.get("verified") is True


@pytest.mark.asyncio
async def test_generalization_task_2_search_finance_for_specific_invoice(
    db_session: AsyncSession,
):
    """Task 2: 'Search the finance system for invoice INV-1042 and report its status.'
    Demonstrates ledger lookup and status reporting.
    """
    # Seed invoice INV-1042
    inv = Invoice(
        invoice_number="INV-1042",
        vendor_name="Acme Corp",
        amount=3450.00,
        currency="USD",
        issue_date="2024-05-10",
        due_date="2024-06-10",
        status="PAID",
    )
    db_session.add(inv)
    await db_session.commit()

    task = TaskRun(
        goal="Search the finance system for invoice INV-1042 and report its status.",
        status="PENDING",
        max_steps=5,
    )
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)

    registry = create_default_registry()
    planner = Planner(tool_registry=registry)
    executor = AgentExecutor(tool_registry=registry, planner=planner, max_steps=5)

    final_state = await executor.execute_task(db=db_session, task_id=task.id)

    assert final_state.current_status == TaskStatus.COMPLETED
    assert len(final_state.tool_results) >= 1
    assert final_state.tool_results[0]["tool"] == "finance_search_tool"
    assert final_state.tool_results[0]["success"] is True

    # Verify outcome contract and working memory facts
    assert final_state.outcome is not None
    assert final_state.outcome.success is True
    assert final_state.outcome.status == TaskStatus.COMPLETED
    assert any("finance_search_tool" in step for step in final_state.completed_steps)
    assert final_state.outcome.verification is not None
    assert final_state.outcome.verification.get("verified") is True


@pytest.mark.asyncio
async def test_generalization_task_3_find_all_unpaid_invoices_for_vendor(
    db_session: AsyncSession,
):
    """Task 3: 'Find all unpaid invoices from Acme and return their invoice numbers and amounts.'
    Demonstrates filter queries and multi-item fact extraction into generic working memory.
    """
    task = TaskRun(
        goal="Find all unpaid invoices from Acme and return their invoice numbers and amounts.",
        status="PENDING",
        max_steps=5,
    )
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)

    registry = create_default_registry()
    planner = Planner(tool_registry=registry)
    executor = AgentExecutor(tool_registry=registry, planner=planner, max_steps=5)

    final_state = await executor.execute_task(db=db_session, task_id=task.id)

    assert final_state.current_status == TaskStatus.COMPLETED
    assert len(final_state.tool_results) >= 1
    assert final_state.tool_results[0]["tool"] == "finance_search_tool"
    assert final_state.tool_results[0]["success"] is True

    # Verify facts and outcome
    assert final_state.outcome is not None
    assert final_state.outcome.success is True
    assert final_state.outcome.status == TaskStatus.COMPLETED
    assert "invoices" in final_state.outcome.facts or "count" in final_state.outcome.facts
    assert final_state.outcome.verification is not None
    assert final_state.outcome.verification.get("verified") is True


@pytest.mark.asyncio
async def test_generalization_task_4_unknown_domain_requests_clarification(
    db_session: AsyncSession,
):
    """Task 4: Out-of-domain task e.g. 'Deploy container to Kubernetes cluster with 5 replicas.'
    Verifies that the planner refuses to hallucinate tools and pauses cleanly for user clarification.
    """
    task = TaskRun(
        goal="Deploy container to Kubernetes cluster with 5 replicas.",
        status="PENDING",
        max_steps=5,
    )
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)

    registry = create_default_registry()
    planner = Planner(tool_registry=registry)
    executor = AgentExecutor(tool_registry=registry, planner=planner, max_steps=5)

    final_state = await executor.execute_task(db=db_session, task_id=task.id)

    # Stopping condition: WAITING_FOR_CLARIFICATION
    assert final_state.current_status == TaskStatus.WAITING_FOR_CLARIFICATION
    assert "clarification" in final_state.result_summary.lower()
    assert len(final_state.tool_results) == 0  # Did not execute any hallucinated tool
    assert final_state.current_step == 1


@pytest.mark.asyncio
async def test_verification_strategy_registry_selection():
    """Verify that VerificationStrategyRegistry dynamically dispatches to the correct strategy."""
    registry = VerificationStrategyRegistry()

    # Invoice creation state
    invoice_creation_state = TaskState(
        task_id="inv-1",
        user_goal="Find Acme invoice and enter into finance system.",
        current_status=TaskStatus.RUNNING,
    )
    invoice_creation_state.extracted_data["invoice_entered"] = True
    selected_1 = registry.select_strategy(invoice_creation_state.user_goal, invoice_creation_state)
    assert isinstance(selected_1, InvoiceVerificationStrategy)

    # Generic search state
    search_state = TaskState(
        task_id="gen-1",
        user_goal="Find all unpaid invoices from Acme and return their amounts.",
        current_status=TaskStatus.RUNNING,
    )
    search_state.completed_steps.append("finance_search_tool")
    search_state.extracted_data["count"] = 2
    selected_2 = registry.select_strategy(search_state.user_goal, search_state)
    assert isinstance(selected_2, GenericToolVerificationStrategy)
