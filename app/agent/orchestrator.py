"""Agent Orchestrator driving the autonomous OBSERVE-DECIDE-ACT-VERIFY loop."""

import time
from typing import Any, Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.planner import Planner
from app.agent.state import (
    ActionProposal,
    ActionType,
    ExecutionStep,
    Observation,
    TaskState,
)
from app.approval.gate import ApprovalGate
from app.config.logging import get_logger
from app.config.settings import settings
from app.models.schemas import RiskLevel, TaskStatus
from app.services.task_service import TaskService
from app.tools.registry import ToolRegistry, create_default_registry
from app.verification.verifier import OutcomeVerifier

logger = get_logger("agent.orchestrator")


class AgentOrchestrator:
    """Orchestrates autonomous task execution through structured perception, planning, and action."""

    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        planner: Optional[Planner] = None,
    ):
        self.tool_registry = tool_registry or create_default_registry()
        self.planner = planner or Planner(tool_registry=self.tool_registry)

    async def execute_task(
        self,
        db: AsyncSession,
        task_id: str,
        resumed_action: Optional[ActionProposal] = None,
    ) -> TaskState:
        """Runs the autonomous execution loop for a task until completion, pause for approval, or failure."""
        task_run = await TaskService.get_task(db=db, task_id=task_id, detailed=True)
        if not task_run:
            raise ValueError(f"Task with ID '{task_id}' does not exist.")

        if task_run.status in (TaskStatus.COMPLETED.value, TaskStatus.FAILED.value):
            logger.info("Task already finished", task_id=task_id, status=task_run.status)
            return self._build_state_from_run(task_run)

        # Initialize State
        state = self._build_state_from_run(task_run)
        state.status = TaskStatus.RUNNING
        await TaskService.update_task_status(db=db, task_id=task_id, status=TaskStatus.RUNNING)

        logger.info(
            "Starting autonomous execution loop",
            task_id=task_id,
            goal=state.goal[:80],
            step=state.current_step,
        )

        while state.current_step < state.max_steps:
            state.current_step += 1
            step_start = time.perf_counter()

            # If resuming after human approval was granted, use the resumed action for this step
            is_resumed_step = False
            if resumed_action:
                action = resumed_action
                resumed_action = None
                is_resumed_step = True
            else:
                # -------------------------------------------------------------
                # 1. DECIDE: Planner evaluates state and selects next action
                # -------------------------------------------------------------
                try:
                    action = await self.planner.decide_next_action(state)
                except Exception as e:
                    logger.error("Planner failure", task_id=task_id, error=str(e))
                    state.record_failure(f"Planner error: {str(e)}")
                    if state.consecutive_failures >= settings.MAX_CONSECUTIVE_FAILURES:
                        return await self._fail_task(
                            db, state, f"Exceeded failure threshold during planning: {str(e)}"
                        )
                    continue

            logger.info(
                "Action decided",
                task_id=task_id,
                step=state.current_step,
                action_type=action.action_type.value,
                tool=action.tool_name,
                thought=action.thought[:80],
            )

            # Check for terminal decisions
            if action.action_type == ActionType.COMPLETE_TASK:
                # Perform final independent verification before closing task
                if state.extracted_data.get("invoice_entered") and not state.extracted_data.get("verified_in_finance"):
                    verification = await OutcomeVerifier.verify_saved_invoice_in_db(
                        db=db,
                        task_id=task_id,
                        expected_data=state.extracted_data,
                    )
                    state.latest_verification = verification
                    if not verification.verified:
                        state.record_failure(
                            f"Final verification failed: {', '.join(verification.discrepancies)}"
                        )
                        continue

                return await self._complete_task(db, state, action.thought)

            if action.action_type == ActionType.FAIL_TASK:
                return await self._fail_task(db, state, action.thought)

            # -------------------------------------------------------------
            # 2. APPROVAL GATE: Intercept high-risk actions
            # -------------------------------------------------------------
            tool = self.tool_registry.get(action.tool_name) if action.tool_name else None
            if not is_resumed_step:
                approval_check = await ApprovalGate.evaluate_action(
                    db=db,
                    task_id=task_id,
                    action=action,
                    tool=tool,
                    approved_action_signatures=state.approved_actions,
                )

                if not approval_check.allowed:
                    # Pause execution loop for human authorization
                    state.status = TaskStatus.WAITING_FOR_APPROVAL
                    state.pending_approval_id = approval_check.approval_request_id
                    await TaskService.add_step_log(
                        db=db,
                        task_id=task_id,
                        step_number=state.current_step,
                        phase="DECIDE",
                        thought=f"Paused: {approval_check.reason}",
                        tool_name=action.tool_name,
                        tool_args=action.tool_args,
                        latency_ms=(time.perf_counter() - step_start) * 1000,
                    )
                    logger.info(
                        "Execution paused awaiting approval",
                        task_id=task_id,
                        approval_id=approval_check.approval_request_id,
                    )
                    return state

            # -------------------------------------------------------------
            # 3. ACT: Execute selected tool through ToolRegistry
            # -------------------------------------------------------------
            tool_res = None
            if not tool:
                error_msg = f"Tool '{action.tool_name}' not registered."
                state.record_failure(error_msg, tool_name=action.tool_name)
                await TaskService.add_step_log(
                    db=db,
                    task_id=task_id,
                    step_number=state.current_step,
                    phase="ACT",
                    thought=action.thought,
                    tool_name=action.tool_name,
                    error_message=error_msg,
                    latency_ms=(time.perf_counter() - step_start) * 1000,
                )
                continue

            try:
                # Execute with DB session passed for tools needing database access
                tool_kwargs = {**action.tool_args}
                if "session" in tool.execute.__code__.co_varnames:
                    tool_kwargs["session"] = db

                tool_res = await tool.run(**tool_kwargs)
            except Exception as e:
                logger.error("Tool execution uncaught exception", tool=action.tool_name, error=str(e))
                state.record_failure(str(e), tool_name=action.tool_name)
                await TaskService.add_step_log(
                    db=db,
                    task_id=task_id,
                    step_number=state.current_step,
                    phase="ACT",
                    thought=action.thought,
                    tool_name=action.tool_name,
                    tool_args=action.tool_args,
                    error_message=str(e),
                    latency_ms=(time.perf_counter() - step_start) * 1000,
                )
                continue

            latency_ms = (time.perf_counter() - step_start) * 1000

            # -------------------------------------------------------------
            # 4. OBSERVE: Capture output, evidence, and update state memory
            # -------------------------------------------------------------
            observation = Observation(
                step_number=state.current_step,
                source=action.tool_name,
                success=tool_res.success,
                data=tool_res.data,
                error=tool_res.error,
                evidence=tool_res.evidence,
            )
            state.add_observation(observation)

            # Persist step log to database
            await TaskService.add_step_log(
                db=db,
                task_id=task_id,
                step_number=state.current_step,
                phase="ACT",
                thought=action.thought,
                tool_name=action.tool_name,
                tool_args=action.tool_args,
                tool_result=tool_res.data,
                error_message=tool_res.error,
                latency_ms=latency_ms,
            )

            # Record evidence attachments
            ev_list = []
            if isinstance(tool_res.evidence, list):
                ev_list = tool_res.evidence
            elif isinstance(tool_res.evidence, dict) and tool_res.evidence:
                ev_list = [tool_res.evidence]

            for ev in ev_list:
                if isinstance(ev, dict):
                    await TaskService.add_evidence(
                        db=db,
                        task_id=task_id,
                        evidence_type=ev.get("type", "observation"),
                        description=ev.get("description", f"Evidence from {action.tool_name}"),
                        source_uri_or_path=ev.get("path") or ev.get("url"),
                        payload=ev.get("data") or ev,
                    )

            # -------------------------------------------------------------
            # 5. RECOVER / CONTINUE: Process observation result
            # -------------------------------------------------------------
            if tool_res.success:
                state.record_success()
                self._update_working_memory(state, action.tool_name, tool_res.data)
                # Persist updated extracted facts to DB
                await TaskService.update_task_status(
                    db=db,
                    task_id=task_id,
                    status=TaskStatus.RUNNING,
                    extracted_data=state.extracted_data,
                )

                # ---------------------------------------------------------
                # 6. VERIFY: Trigger independent verification after mutations
                # ---------------------------------------------------------
                if action.tool_name == "finance_create_invoice_tool":
                    verification = await OutcomeVerifier.verify_saved_invoice_in_db(
                        db=db,
                        task_id=task_id,
                        expected_data=state.extracted_data,
                    )
                    state.latest_verification = verification
                    if verification.verified:
                        state.extracted_data["verified_in_finance"] = True
                        logger.info("Invoice verified successfully in finance ledger", task_id=task_id)
                    else:
                        state.record_failure(
                            f"Verification failed after creation: {', '.join(verification.discrepancies)}"
                        )

            else:
                state.record_failure(
                    error_message=tool_res.error or "Action failed",
                    tool_name=action.tool_name,
                )
                if state.consecutive_failures >= settings.MAX_CONSECUTIVE_FAILURES:
                    return await self._fail_task(
                        db,
                        state,
                        f"Halted after {state.consecutive_failures} consecutive failures. Latest error: {tool_res.error}",
                    )

        # Exceeded step limit
        return await self._fail_task(db, state, f"Exceeded maximum step budget of {state.max_steps} steps.")

    def _update_working_memory(self, state: TaskState, tool_name: str, data: Optional[Dict[str, Any]]) -> None:
        """Extracts key business domain entities from tool outputs into working memory."""
        if not data:
            return

        if tool_name == "document_search_tool":
            latest = data.get("latest_invoice") or data.get("latest")
            if latest and isinstance(latest, dict):
                state.extracted_data["latest_invoice_path"] = latest.get("file_path")
                state.extracted_data["latest_invoice_date"] = latest.get("invoice_date") or latest.get("date")

        elif tool_name == "document_read_tool":
            fields = data.get("invoice_data") or data.get("fields")
            if fields and isinstance(fields, dict):
                state.extracted_data.update(fields)
                state.extracted_data["file_name"] = data.get("file_name")
                state.extracted_data["pdf_extracted"] = True

        elif tool_name == "finance_create_invoice_tool":
            state.extracted_data["invoice_entered"] = True
            state.extracted_data["created_invoice_id"] = data.get("invoice_id") or data.get("id")

        elif tool_name == "finance_read_invoice_tool":
            state.extracted_data["verified_in_finance"] = True
            state.extracted_data["saved_record"] = data

    async def _complete_task(self, db: AsyncSession, state: TaskState, summary: str) -> TaskState:
        """Transitions task to COMPLETED and updates persistence."""
        state.status = TaskStatus.COMPLETED
        state.result_summary = summary
        await TaskService.update_task_status(
            db=db,
            task_id=state.task_id,
            status=TaskStatus.COMPLETED,
            result_summary=summary,
            extracted_data=state.extracted_data,
        )
        logger.info("Task completed successfully", task_id=state.task_id, summary=summary[:100])
        return state

    async def _fail_task(self, db: AsyncSession, state: TaskState, reason: str) -> TaskState:
        """Transitions task to FAILED and updates persistence."""
        state.status = TaskStatus.FAILED
        state.result_summary = reason
        await TaskService.update_task_status(
            db=db,
            task_id=state.task_id,
            status=TaskStatus.FAILED,
            result_summary=reason,
            extracted_data=state.extracted_data,
        )
        logger.error("Task failed", task_id=state.task_id, reason=reason)
        return state

    def _build_state_from_run(self, task_run) -> TaskState:
        """Constructs TaskState from SQLAlchemy TaskRun record."""
        return TaskState(
            task_id=task_run.id,
            goal=task_run.goal,
            status=TaskStatus(task_run.status),
            current_step=task_run.current_step,
            max_steps=task_run.max_steps,
            extracted_data=task_run.extracted_data or {},
            result_summary=task_run.result_summary,
        )
