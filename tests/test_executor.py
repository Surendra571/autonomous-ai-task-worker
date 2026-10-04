"""Unit and integration tests for the Autonomous Agent Executor."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.executor import AgentExecutor
from app.agent.planner import Planner
from app.agent.state import ActionCall, ActionProposal, ActionType, PlannerDecision, TaskState
from app.models.schemas import RiskLevel, TaskStatus
from app.models.task import TaskRun
from app.services.task_service import TaskService
from app.tools.base import BaseTool, ToolResult
from app.tools.registry import ToolRegistry


from pydantic import BaseModel, Field


class DummyInput(BaseModel):
    should_fail: bool = False


class DummyTool(BaseTool):
    name: str = "dummy_tool"
    description: str = "Test dummy tool"
    input_schema: type[BaseModel] = DummyInput
    risk_level: str = "LOW"

    async def execute(self, params: DummyInput, session=None, **kwargs) -> ToolResult:
        if params.should_fail:
            return ToolResult(success=False, error="Simulated dummy tool error")
        return ToolResult(success=True, data={"result": "dummy_success"})


class RiskyInput(BaseModel):
    amount: float = 0.0


class RiskyTool(BaseTool):
    name: str = "risky_tool"
    description: str = "Test risky tool"
    input_schema: type[BaseModel] = RiskyInput
    risk_level: str = "HIGH"

    async def execute(self, params: RiskyInput, session=None, **kwargs) -> ToolResult:
        return ToolResult(success=True, data={"status": "executed"})


@pytest.fixture
def custom_registry():
    registry = ToolRegistry()
    registry.register(DummyTool())
    registry.register(RiskyTool())
    return registry


@pytest.mark.asyncio
async def test_executor_records_every_action_and_tool_result(
    db_session: AsyncSession,
    custom_registry: ToolRegistry,
):
    """Verify that every action, tool result, observation, and history entry is recorded."""
    task = TaskRun(goal="Execute dummy actions", status="PENDING", max_steps=10)
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)

    # Mock planner that executes dummy_tool once and then completes
    class MockPlanner:
        def __init__(self):
            self.step = 0

        async def decide_next_action(self, state: TaskState) -> ActionProposal:
            self.step += 1
            if self.step == 1:
                return ActionProposal(
                    thought="Executing first test action",
                    action_type=ActionType.CALL_TOOL,
                    tool_name="dummy_tool",
                    tool_args={"should_fail": False},
                    risk_level=RiskLevel.LOW,
                )
            return ActionProposal(
                thought="All steps finished",
                action_type=ActionType.COMPLETE_TASK,
                tool_name=None,
                tool_args={},
                risk_level=RiskLevel.LOW,
            )

    executor = AgentExecutor(tool_registry=custom_registry, planner=MockPlanner(), max_steps=10)
    final_state = await executor.execute_task(db=db_session, task_id=task.id)

    assert final_state.current_status == TaskStatus.COMPLETED
    assert final_state.current_step == 2
    assert len(final_state.tool_results) == 1
    assert final_state.tool_results[0]["tool"] == "dummy_tool"
    assert final_state.tool_results[0]["success"] is True
    assert len(final_state.execution_history) >= 1
    assert final_state.execution_history[0]["action"] == "dummy_tool"
    assert "dummy_success" in final_state.execution_history[0]["result"]


@pytest.mark.asyncio
async def test_executor_stopping_condition_clarification_required(
    db_session: AsyncSession,
    custom_registry: ToolRegistry,
):
    """Verify executor pauses and stops when planner requests clarification."""
    task = TaskRun(goal="Process vague instruction", status="PENDING")
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)

    class ClarificationPlanner:
        async def decide_next_action(self, state: TaskState) -> ActionProposal:
            return ActionProposal(
                thought="The vendor name is missing.",
                action_type=ActionType.REQUEST_CLARIFICATION,
                tool_name=None,
                tool_args={"question": "Which vendor's invoice should be processed?"},
                risk_level=RiskLevel.LOW,
            )

    executor = AgentExecutor(tool_registry=custom_registry, planner=ClarificationPlanner())
    final_state = await executor.execute_task(db=db_session, task_id=task.id)

    assert final_state.current_status == TaskStatus.WAITING_FOR_CLARIFICATION
    assert "clarification" in final_state.result_summary.lower()


@pytest.mark.asyncio
async def test_executor_stopping_condition_approval_required(
    db_session: AsyncSession,
    custom_registry: ToolRegistry,
):
    """Verify executor pauses when a risky action requires supervisor authorization."""
    task = TaskRun(goal="Execute risky write", status="PENDING")
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)

    class RiskyPlanner:
        async def decide_next_action(self, state: TaskState) -> ActionProposal:
            return ActionProposal(
                thought="Modifying permanent system records.",
                action_type=ActionType.CALL_TOOL,
                tool_name="risky_tool",
                tool_args={"amount": 5000},
                risk_level=RiskLevel.HIGH,
                requires_approval=True,
            )

    executor = AgentExecutor(tool_registry=custom_registry, planner=RiskyPlanner())
    final_state = await executor.execute_task(db=db_session, task_id=task.id)

    assert final_state.current_status == TaskStatus.WAITING_FOR_APPROVAL
    assert final_state.pending_approval_id is not None


@pytest.mark.asyncio
async def test_executor_stopping_condition_unrecoverable_failure(
    db_session: AsyncSession,
    custom_registry: ToolRegistry,
):
    """Verify executor halts when consecutive failures reach threshold."""
    task = TaskRun(goal="Handle repeated errors", status="PENDING")
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)

    class FailingPlanner:
        async def decide_next_action(self, state: TaskState) -> ActionProposal:
            return ActionProposal(
                thought="Attempting action that will fail",
                action_type=ActionType.CALL_TOOL,
                tool_name="dummy_tool",
                tool_args={"should_fail": True},
                risk_level=RiskLevel.LOW,
            )

    executor = AgentExecutor(
        tool_registry=custom_registry,
        planner=FailingPlanner(),
        max_consecutive_failures=2,
    )
    final_state = await executor.execute_task(db=db_session, task_id=task.id)

    assert final_state.current_status == TaskStatus.FAILED
    assert final_state.consecutive_failures >= 2
    assert "consecutive failures" in final_state.result_summary.lower()


@pytest.mark.asyncio
async def test_executor_stopping_condition_max_steps_limit(
    db_session: AsyncSession,
    custom_registry: ToolRegistry,
):
    """Verify executor halts cleanly when configurable MAX_STEPS limit is reached."""
    task = TaskRun(goal="Infinite looping task", status="PENDING", max_steps=4)
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)

    class LoopingPlanner:
        async def decide_next_action(self, state: TaskState) -> ActionProposal:
            return ActionProposal(
                thought=f"Executing loop step {state.current_step}",
                action_type=ActionType.CALL_TOOL,
                tool_name="dummy_tool",
                tool_args={"should_fail": False},
                risk_level=RiskLevel.LOW,
            )

    executor = AgentExecutor(tool_registry=custom_registry, planner=LoopingPlanner(), max_steps=4)
    final_state = await executor.execute_task(db=db_session, task_id=task.id)

    assert final_state.current_status == TaskStatus.FAILED
    assert final_state.current_step == 4
    assert "maximum step budget" in final_state.result_summary.lower()


@pytest.mark.asyncio
async def test_executor_never_completes_without_verification(
    db_session: AsyncSession,
    custom_registry: ToolRegistry,
):
    """Verify that a tool returning success does not mark task complete without verification."""
    task = TaskRun(goal="Enter and verify invoice", status="PENDING")
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)

    class UnverifiedCompletionPlanner:
        def __init__(self):
            self.step = 0

        async def decide_next_action(self, state: TaskState) -> ActionProposal:
            self.step += 1
            if self.step == 1:
                # Agent records that an invoice was entered into memory
                state.extracted_data["invoice_entered"] = True
                state.extracted_data["invoice_number"] = "INV-FAKE-001"
                return ActionProposal(
                    thought="Invoice created in ledger.",
                    action_type=ActionType.CALL_TOOL,
                    tool_name="dummy_tool",
                    tool_args={},
                    risk_level=RiskLevel.LOW,
                )
            # Second step: planner prematurely proposes completion without verification
            return ActionProposal(
                thought="Tool returned success, so I claim goal is complete!",
                action_type=ActionType.COMPLETE_TASK,
                tool_name=None,
                tool_args={},
                risk_level=RiskLevel.LOW,
            )

    executor = AgentExecutor(tool_registry=custom_registry, planner=UnverifiedCompletionPlanner(), max_steps=5)
    # The executor's separate verification should reject premature completion because INV-FAKE-001 doesn't exist in DB
    final_state = await executor.execute_task(db=db_session, task_id=task.id)

    # Must NOT have completed on unverified claim! It recorded failure and continued until max steps
    assert final_state.current_status == TaskStatus.FAILED
    assert any("verification failed" in str(f).lower() for f in final_state.failed_steps)
