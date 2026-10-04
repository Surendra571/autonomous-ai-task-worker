"""Autonomous Agent Executor driving the 12-step OBSERVE-DECIDE-VALIDATE-ACT-VERIFY loop."""

import time
from typing import Any, Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.planner import Planner
from app.agent.state import (
    ActionProposal,
    ActionType,
    ExecutionStep,
    Observation,
    PlannerDecision,
    TaskState,
)
from app.approval.gate import ApprovalGate
from app.config.logging import get_logger
from app.config.settings import settings
from app.models.schemas import RiskLevel, TaskStatus
from app.services.task_service import TaskService
from app.tools.registry import ToolRegistry, create_default_registry
from app.verification.verifier import OutcomeVerifier
from app.agent.evidence import EvidenceCollector
from app.agent.recovery import RecoveryManager
from app.agent.state import VerificationResult

logger = get_logger("agent.executor")


class AgentExecutor:
    """Executes natural language business tasks through a dynamic autonomous control loop:

    1. Observe current state
    2. Ask planner for the next action
    3. Validate the proposed action
    4. Check approval requirements
    5. Execute the selected tool
    6. Store the ToolResult
    7. Convert the result into an Observation
    8. Update agent state
    9. Determine whether the goal is complete (with separate verification)
    10. If not complete, continue
    11. If an action failed, invoke recovery logic
    12. Stop when:
       - goal verified
       - clarification required
       - approval required
       - unrecoverable failure
       - maximum step limit reached
    """

    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        planner: Optional[Planner] = None,
        max_steps: Optional[int] = None,
        max_consecutive_failures: Optional[int] = None,
    ):
        self.tool_registry = tool_registry or create_default_registry()
        self.planner = planner or Planner(tool_registry=self.tool_registry)
        self.max_steps = max_steps or settings.MAX_STEPS
        self.max_consecutive_failures = (
            max_consecutive_failures or settings.MAX_CONSECUTIVE_FAILURES
        )
        self.recovery_manager = RecoveryManager(max_retries=self.max_consecutive_failures)

    async def execute_task(
        self,
        db: AsyncSession,
        task_id: str,
        resumed_action: Optional[ActionProposal] = None,
    ) -> TaskState:
        """Executes the autonomous loop for the given task until a stopping condition is reached."""
        task_run = await TaskService.get_task(db=db, task_id=task_id, detailed=True)
        if not task_run:
            raise ValueError(f"Task with ID '{task_id}' does not exist.")

        if task_run.status in (TaskStatus.COMPLETED.value, TaskStatus.FAILED.value):
            logger.info("TASK_ALREADY_FINISHED", task_id=task_id, status=task_run.status)
            return self._build_state_from_run(task_run)

        # Initialize State
        state = self._build_state_from_run(task_run)
        if self.max_steps:
            state.max_steps = min(state.max_steps, self.max_steps)

        prev_status = state.current_status
        state.current_status = TaskStatus.RUNNING
        await self._log_transition(db, state, prev_status, TaskStatus.RUNNING)

        logger.info(
            "STARTING_EXECUTION_LOOP",
            task_id=task_id,
            goal=state.user_goal[:80],
            step=state.current_step,
            max_steps=state.max_steps,
        )

        while state.current_step < state.max_steps:
            step_start = time.perf_counter()
            state.current_step += 1

            # -----------------------------------------------------------------
            # 1. OBSERVE CURRENT STATE
            # -----------------------------------------------------------------
            logger.info(
                "LOOP_STEP_1_OBSERVE: Inspecting current state",
                task_id=task_id,
                step=state.current_step,
                max_steps=state.max_steps,
                status=state.current_status.value,
                extracted_keys=list(state.extracted_data.keys()),
                history_length=len(state.execution_history),
                consecutive_failures=state.consecutive_failures,
            )

            # -----------------------------------------------------------------
            # 2. ASK PLANNER FOR THE NEXT ACTION
            # -----------------------------------------------------------------
            is_resumed_step = False
            if resumed_action:
                action = resumed_action
                resumed_action = None
                is_resumed_step = True
                logger.info(
                    "LOOP_STEP_2_DECIDE: Resuming with supervisor-authorized action",
                    task_id=task_id,
                    tool=action.tool_name,
                )
            else:
                logger.info(
                    "LOOP_STEP_2_DECIDE: Querying planner for next action",
                    task_id=task_id,
                    step=state.current_step,
                )
                try:
                    action = await self.planner.decide_next_action(state)
                except Exception as e:
                    logger.error("PLANNER_ERROR: Failed to decide next action", task_id=task_id, error=str(e))
                    state.record_failure(f"Planner error: {str(e)}")
                    if state.consecutive_failures >= self.max_consecutive_failures:
                        return await self._fail_task(
                            db, state, f"Exceeded failure threshold during planning: {str(e)}"
                        )
                    continue

            logger.info(
                "PLANNER_ACTION_DECIDED",
                task_id=task_id,
                step=state.current_step,
                action_type=action.action_type.value,
                tool=action.tool_name,
                reason=action.thought[:100],
                requires_approval=action.requires_approval,
                verification_needed=action.verification_needed,
            )

            # -----------------------------------------------------------------
            # 3. VALIDATE THE PROPOSED ACTION
            # -----------------------------------------------------------------
            logger.info(
                "LOOP_STEP_3_VALIDATE: Validating proposed action",
                action_type=action.action_type.value,
                tool=action.tool_name,
            )

            # Check 1: User Clarification Required
            if action.action_type == ActionType.REQUEST_CLARIFICATION:
                question = action.tool_args.get("question", action.thought)
                logger.info("STOPPING_CONDITION_CLARIFICATION_REQUIRED", task_id=task_id, question=question)
                prev = state.current_status
                state.current_status = TaskStatus.WAITING_FOR_CLARIFICATION
                await self._log_transition(db, state, prev, TaskStatus.WAITING_FOR_CLARIFICATION)
                state.result_summary = f"Waiting for user clarification: {question}"
                await TaskService.add_step_log(
                    db=db,
                    task_id=task_id,
                    step_number=state.current_step,
                    phase="CLARIFY",
                    thought=action.thought,
                    tool_name="request_clarification",
                    tool_args=action.tool_args,
                    latency_ms=(time.perf_counter() - step_start) * 1000,
                )
                await TaskService.save_task_state(db, state)
                return state

            # Check 2: Terminal Failure Proposed by Planner
            if action.action_type == ActionType.FAIL_TASK:
                logger.info("STOPPING_CONDITION_PLANNER_FAIL_TASK", task_id=task_id, reason=action.thought)
                return await self._fail_task(db, state, action.thought)

            # Check 3: Terminal Completion Proposed by Planner (Evaluate in Step 9)
            if action.action_type == ActionType.COMPLETE_TASK:
                completed_state = await self._evaluate_and_complete(db, state, action.thought)
                if completed_state is not None:
                    return completed_state
                continue

            # Check 4: Tool Registration & Schema Validation
            tool = self.tool_registry.get(action.tool_name) if action.tool_name else None
            if not tool:
                error_msg = f"Tool '{action.tool_name}' is not registered in ToolRegistry."
                logger.warning("TOOL_VALIDATION_ERROR", error=error_msg)
                state.record_failure(error_msg, tool_name=action.tool_name)
                await TaskService.add_step_log(
                    db=db,
                    task_id=task_id,
                    step_number=state.current_step,
                    phase="VALIDATE",
                    thought=action.thought,
                    tool_name=action.tool_name,
                    error_message=error_msg,
                    latency_ms=(time.perf_counter() - step_start) * 1000,
                )
                await TaskService.save_task_state(db, state)
                if state.consecutive_failures >= self.max_consecutive_failures:
                    return await self._fail_task(db, state, f"Unrecoverable validation error: {error_msg}")
                continue

            # -----------------------------------------------------------------
            # 4. CHECK APPROVAL REQUIREMENTS
            # -----------------------------------------------------------------
            logger.info("LOOP_STEP_4_APPROVAL_CHECK: Evaluating action risk against approval gate")
            if not is_resumed_step:
                approval_check = await ApprovalGate.evaluate_action(
                    db=db,
                    task_id=task_id,
                    action=action,
                    tool=tool,
                    approved_action_signatures=state.approved_actions,
                )

                if not approval_check.allowed:
                    logger.warning(
                        "STOPPING_CONDITION_APPROVAL_REQUIRED",
                        task_id=task_id,
                        approval_id=approval_check.approval_request_id,
                        reason=approval_check.reason,
                    )
                    prev = state.current_status
                    state.current_status = TaskStatus.WAITING_FOR_APPROVAL
                    state.pending_approval_id = approval_check.approval_request_id
                    await self._log_transition(db, state, prev, TaskStatus.WAITING_FOR_APPROVAL)

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
                    await TaskService.save_task_state(db, state)
                    return state

            # -----------------------------------------------------------------
            # 5. EXECUTE THE SELECTED TOOL
            # -----------------------------------------------------------------
            logger.info(
                "LOOP_STEP_5_ACT: Executing tool",
                tool=action.tool_name,
                args=list(action.tool_args.keys()),
            )
            tool_res = None
            try:
                tool_kwargs = {**action.tool_args}
                if "session" in tool.execute.__code__.co_varnames:
                    tool_kwargs["session"] = db

                tool_res = await tool.run(**tool_kwargs)
            except Exception as e:
                logger.error("TOOL_EXECUTION_EXCEPTION", tool=action.tool_name, error=str(e))
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
                await TaskService.save_task_state(db, state)
                if state.consecutive_failures >= self.max_consecutive_failures:
                    return await self._fail_task(db, state, f"Unrecoverable tool exception: {str(e)}")
                continue

            latency_ms = (time.perf_counter() - step_start) * 1000

            # -----------------------------------------------------------------
            # 6. STORE THE TOOLRESULT
            # -----------------------------------------------------------------
            logger.info(
                "LOOP_STEP_6_STORE_RESULT: Recording tool outcome",
                tool=action.tool_name,
                success=tool_res.success,
                latency_ms=round(latency_ms, 2),
            )
            state.tool_results.append({
                "step": state.current_step,
                "tool": action.tool_name,
                "success": tool_res.success,
                "data": tool_res.data,
                "error": tool_res.error,
                "latency_ms": latency_ms,
            })

            # -----------------------------------------------------------------
            # 7. CONVERT THE RESULT INTO AN OBSERVATION
            # -----------------------------------------------------------------
            logger.info("LOOP_STEP_7_OBSERVATION: Constructing structured observation")
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

            # Persist tangible evidence and collect audit evidence
            evidence_item = EvidenceCollector.collect_tool_evidence(
                task_id=task_id,
                step_number=state.current_step,
                action_name=action.tool_name,
                tool_name=action.tool_name,
                tool_args=action.tool_args,
                tool_result=tool_res,
            )
            state.evidence.append(evidence_item.model_dump())
            if evidence_item.screenshot_path and evidence_item.screenshot_path not in state.screenshots:
                state.screenshots.append(evidence_item.screenshot_path)
            await EvidenceCollector.record_evidence(db=db, evidence=evidence_item)

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

            # -----------------------------------------------------------------
            # 8. UPDATE AGENT STATE
            # -----------------------------------------------------------------
            logger.info("LOOP_STEP_8_UPDATE_STATE: Updating state working memory and execution history")
            action_summary = f"{action.tool_name}({list(action.tool_args.keys())})"
            result_summary = (
                f"Success: {tool_res.data}"
                if tool_res.success
                else f"Failed: {tool_res.error}"
            )
            state.record_action_result(action=action.tool_name, result=result_summary[:200])

            if tool_res.success:
                state.record_success()
                self._update_working_memory(state, action.tool_name, tool_res.data)
                state.completed_steps.append(action.tool_name)

                # Record invoice workflow milestones if applicable
                if action.tool_name == "document_search_tool" and state.extracted_data.get("latest_invoice_path"):
                    m1 = await EvidenceCollector.record_source_invoice_milestone(
                        db=db,
                        task_id=task_id,
                        file_path=state.extracted_data["latest_invoice_path"],
                        invoice_date=state.extracted_data.get("latest_invoice_date"),
                        vendor_name=state.extracted_data.get("vendor_name"),
                        step_number=state.current_step,
                    )
                    state.evidence.append(m1.model_dump())

                elif action.tool_name == "document_read_tool":
                    inv_data = (
                        (tool_res.data.get("invoice_data") if tool_res.data else None)
                        or (tool_res.data if tool_res.data and "invoice_number" in tool_res.data else None)
                        or {k: state.extracted_data.get(k) for k in ("vendor_name", "invoice_number", "amount", "currency", "due_date") if state.extracted_data.get(k)}
                    )
                    if inv_data:
                        m2 = await EvidenceCollector.record_extracted_invoice_milestone(
                            db=db,
                            task_id=task_id,
                            invoice_data=inv_data,
                            file_path=action.tool_args.get("file_path") or state.extracted_data.get("latest_invoice_path"),
                            raw_text=tool_res.data.get("raw_text") if tool_res.data else None,
                            step_number=state.current_step,
                        )
                        state.evidence.append(m2.model_dump())

                elif action.tool_name == "finance_create_invoice_tool":
                    m3 = await EvidenceCollector.record_finance_record_milestone(
                        db=db,
                        task_id=task_id,
                        finance_record=tool_res.data or {},
                        step_number=state.current_step,
                        tool_name=action.tool_name,
                    )
                    state.evidence.append(m3.model_dump())
            else:
                state.record_failure(tool_res.error or "Tool returned failure", tool_name=action.tool_name)

            # Persist state snapshot to database
            await TaskService.save_task_state(db, state)

            # -----------------------------------------------------------------
            # 9. DETERMINE WHETHER THE GOAL IS COMPLETE (Separate Verification)
            # -----------------------------------------------------------------
            logger.info("LOOP_STEP_9_CHECK_COMPLETION: Evaluating goal completion with independent verification")
            # If the action performed was finance_create_invoice_tool, automatically run verification
            if action.tool_name == "finance_create_invoice_tool" and tool_res.success:
                verification = await OutcomeVerifier.verify_saved_invoice_in_db(
                    db=db,
                    task_id=task_id,
                    expected_data=state.extracted_data,
                )
                state.latest_verification = verification
                state.verification_results.append(verification.dict())
                m4 = await EvidenceCollector.record_verification_result_milestone(
                    db=db,
                    task_id=task_id,
                    verification_report=verification,
                    step_number=state.current_step,
                )
                state.evidence.append(m4.model_dump())
                if verification.verified:
                    state.extracted_data["verified_in_finance"] = True
                    logger.info("INDEPENDENT_VERIFICATION_PASSED", task_id=task_id)
                else:
                    logger.warning(
                        "INDEPENDENT_VERIFICATION_DISCREPANCIES",
                        task_id=task_id,
                        discrepancies=verification.discrepancies,
                    )
                    state.record_failure(f"Verification discrepancies: {', '.join(verification.discrepancies)}")
                await TaskService.save_task_state(db, state)

            # Check if task is verified and complete
            if state.extracted_data.get("verified_in_finance") and (
                action.tool_name == "finance_read_invoice_tool" or action.verification_needed
            ):
                if not state.latest_verification or not state.latest_verification.verified:
                    verification_res = VerificationResult(
                        verified=True,
                        verification_type="finance_read_confirmation",
                        details=tool_res.data or {},
                        discrepancies=[],
                    )
                    state.latest_verification = verification_res
                    state.verification_results.append(verification_res.dict())
                    m4 = await EvidenceCollector.record_verification_result_milestone(
                        db=db,
                        task_id=task_id,
                        verification_report=verification_res,
                        step_number=state.current_step,
                    )
                    state.evidence.append(m4.model_dump())

                # Independent verification confirmed!
                logger.info("STOPPING_CONDITION_GOAL_VERIFIED", task_id=task_id)
                return await self._complete_task(
                    db,
                    state,
                    f"Task goal successfully fulfilled and verified for invoice {state.extracted_data.get('invoice_number', 'unknown')}.",
                )

            # -----------------------------------------------------------------
            # 11. IF AN ACTION FAILED, INVOKE RECOVERY LOGIC
            # -----------------------------------------------------------------
            if not tool_res.success:
                logger.warning(
                    "LOOP_STEP_11_RECOVERY: Tool failed; invoking recovery policy",
                    tool=action.tool_name,
                    consecutive_failures=state.consecutive_failures,
                )
                recovery_decision = self.recovery_manager.decide_recovery(
                    state=state,
                    failed_action=action,
                    error_message=tool_res.error or "Tool execution failed",
                )
                for ev_str in recovery_decision.event_sequence:
                    state.record_action_result("RECOVERY_EVENT", ev_str)

                if recovery_decision.is_recoverable and recovery_decision.recovery_action:
                    resumed_action = recovery_decision.recovery_action
                    logger.info(
                        "RECOVERY_POLICY: Scheduled recovery action",
                        strategy=recovery_decision.recovery_strategy.value,
                        recovery_action=resumed_action.tool_name,
                    )
                    await TaskService.save_task_state(db, state)
                elif not recovery_decision.is_recoverable:
                    logger.error("STOPPING_CONDITION_UNRECOVERABLE_FAILURE", task_id=task_id)
                    return await self._fail_task(
                        db,
                        state,
                        recovery_decision.reason,
                    )

            # -----------------------------------------------------------------
            # 10. IF NOT COMPLETE, CONTINUE
            # -----------------------------------------------------------------
            logger.info("LOOP_STEP_10_CONTINUE: Proceeding to next step in execution loop", next_step=state.current_step + 1)

        # ---------------------------------------------------------------------
        # 12. STOPPING CONDITION: MAXIMUM STEP LIMIT REACHED
        # ---------------------------------------------------------------------
        logger.error(
            "STOPPING_CONDITION_MAX_STEPS_REACHED",
            task_id=task_id,
            current_step=state.current_step,
            max_steps=state.max_steps,
        )
        return await self._fail_task(db, state, f"Exceeded maximum step budget of {state.max_steps} steps.")

    async def _evaluate_and_complete(
        self, db: AsyncSession, state: TaskState, summary: str
    ) -> Optional[TaskState]:
        """Enforces that a task is NEVER marked complete without independent verification."""
        logger.info(
            "EVALUATING_COMPLETION_VERIFICATION: Checking verification status before completing",
            task_id=state.task_id,
        )

        needs_verification = (
            state.extracted_data.get("invoice_entered")
            or "invoice_number" in state.extracted_data
            or "invoice" in state.user_goal.lower()
        )
        if needs_verification and not state.extracted_data.get("verified_in_finance"):
            verification = await OutcomeVerifier.verify_saved_invoice_in_db(
                db=db,
                task_id=state.task_id,
                expected_data=state.extracted_data,
            )
            state.latest_verification = verification
            state.verification_results.append(verification.dict())
            m4 = await EvidenceCollector.record_verification_result_milestone(
                db=db,
                task_id=state.task_id,
                verification_report=verification,
                step_number=state.current_step,
            )
            state.evidence.append(m4.model_dump())

            if not verification.verified:
                logger.warning(
                    "COMPLETION_REJECTED: Independent verification failed",
                    discrepancies=verification.discrepancies,
                )
                state.record_failure(f"Final verification failed: {', '.join(verification.discrepancies)}")
                state.add_observation(
                    Observation(
                        step_number=state.current_step,
                        source="VerificationEngine",
                        success=False,
                        data={"discrepancies": verification.discrepancies},
                        error=f"Verification failed: {', '.join(verification.discrepancies)}",
                    )
                )
                await TaskService.save_task_state(db, state)
                if state.consecutive_failures >= self.max_consecutive_failures:
                    return await self._fail_task(
                        db,
                        state,
                        f"Halted after {state.consecutive_failures} consecutive failures during verification.",
                    )
                return None  # Rejection: continue loop so agent can recover/retry!

            state.extracted_data["verified_in_finance"] = True

        return await self._complete_task(db, state, summary)

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
            elif "invoice_number" in data:
                state.extracted_data.update(data)
                state.extracted_data["pdf_extracted"] = True

        elif tool_name == "finance_create_invoice_tool":
            state.extracted_data["invoice_entered"] = True
            state.extracted_data["created_invoice_id"] = data.get("invoice_id") or data.get("id")
            state.extracted_data["saved_record"] = data

        elif tool_name == "finance_read_invoice_tool":
            state.extracted_data["verified_in_finance"] = True
            state.extracted_data["saved_record"] = data
            state.extracted_data["created_invoice_id"] = data.get("id") or data.get("invoice_id") or state.extracted_data.get("created_invoice_id")

        # Generic working memory integration for non-invoice tools
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (str, int, float, bool)) and k not in state.extracted_data:
                    state.extracted_data[k] = v

    async def _complete_task(self, db: AsyncSession, state: TaskState, summary: str) -> TaskState:
        """Transitions task to COMPLETED, logs transition, and persists state."""
        prev = state.current_status
        state.current_status = TaskStatus.COMPLETED
        dossier_text = EvidenceCollector.generate_final_task_summary(state)
        final_summary = f"{summary}\n\n{dossier_text}" if summary not in dossier_text else dossier_text
        state.final_result = final_summary
        await self._log_transition(db, state, prev, TaskStatus.COMPLETED)
        await TaskService.update_task_status(
            db=db,
            task_id=state.task_id,
            status=TaskStatus.COMPLETED,
            result_summary=final_summary,
            extracted_data=state.extracted_data,
        )
        await TaskService.save_task_state(db, state)
        logger.info("TASK_COMPLETED_SUCCESSFULLY", task_id=state.task_id, summary=final_summary[:100])
        return state

    async def _fail_task(self, db: AsyncSession, state: TaskState, reason: str) -> TaskState:
        """Transitions task to FAILED, logs transition, and persists state."""
        prev = state.current_status
        state.current_status = TaskStatus.FAILED
        state.final_result = reason
        await self._log_transition(db, state, prev, TaskStatus.FAILED)
        await TaskService.update_task_status(
            db=db,
            task_id=state.task_id,
            status=TaskStatus.FAILED,
            result_summary=reason,
            extracted_data=state.extracted_data,
        )
        await TaskService.save_task_state(db, state)
        logger.error("TASK_FAILED", task_id=state.task_id, reason=reason)
        return state

    async def _log_transition(
        self,
        db: AsyncSession,
        state: TaskState,
        from_status: Any,
        to_status: Any,
    ) -> None:
        """Logs explicit state transitions observable in application logs."""
        from_str = from_status.value if hasattr(from_status, "value") else str(from_status)
        to_str = to_status.value if hasattr(to_status, "value") else str(to_status)
        logger.info(
            "STATE_TRANSITION",
            task_id=state.task_id,
            from_status=from_str,
            to_status=to_str,
            step=state.current_step,
        )

    def _build_state_from_run(self, task_run) -> TaskState:
        """Constructs TaskState from SQLAlchemy TaskRun record."""
        snapshot = getattr(task_run, "state_snapshot", None) or {}
        if snapshot and isinstance(snapshot, dict) and "task_id" in snapshot:
            return TaskState.model_validate(snapshot)

        return TaskState(
            task_id=task_run.id,
            goal=task_run.goal,
            status=TaskStatus(task_run.status),
            current_step=task_run.current_step,
            max_steps=task_run.max_steps,
            extracted_data=task_run.extracted_data or {},
            result_summary=task_run.result_summary,
            current_objective=getattr(task_run, "current_objective", None),
            plan_summary=getattr(task_run, "plan_summary", None),
        )

