"""Service layer for task creation, retrieval, and state operations."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.task import ApprovalRequest, EvidenceRecord, StepLog, TaskRun
from app.models.schemas import TaskCreate, TaskStatus
from app.config.logging import get_logger

logger = get_logger("service.task")


class TaskService:
    """Encapsulates business operations on Tasks and Execution records."""

    @staticmethod
    async def create_task(
        db: AsyncSession,
        task_in: Optional[TaskCreate] = None,
        goal: Optional[str] = None,
        max_steps: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TaskRun:
        """Creates and persists a new task run."""
        if task_in is not None:
            task_goal = task_in.goal
            task_max_steps = task_in.max_steps or 15
            task_metadata = task_in.metadata or {}
        else:
            task_goal = goal or ""
            task_max_steps = max_steps or 15
            task_metadata = metadata or {}

        task = TaskRun(
            goal=task_goal,
            max_steps=task_max_steps,
            status=TaskStatus.PENDING.value,
            extracted_data=task_metadata,
        )
        db.add(task)
        await db.commit()
        await db.refresh(task)
        logger.info("Task created successfully", task_id=task.id, goal=task.goal[:60])
        return task

    @staticmethod
    async def get_task(db: AsyncSession, task_id: str, detailed: bool = True) -> Optional[TaskRun]:
        """Fetches a task by ID, optionally loading relationships."""
        stmt = select(TaskRun).where(TaskRun.id == task_id)
        if detailed:
            stmt = stmt.options(
                selectinload(TaskRun.step_logs),
                selectinload(TaskRun.approvals),
                selectinload(TaskRun.evidence),
            )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def list_tasks(db: AsyncSession, limit: int = 50, offset: int = 0) -> List[TaskRun]:
        """Lists recent tasks."""
        stmt = (
            select(TaskRun)
            .order_by(TaskRun.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def update_task_status(
        db: AsyncSession,
        task_id: str,
        status: TaskStatus,
        result_summary: Optional[str] = None,
        extracted_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[TaskRun]:
        """Updates task state, summary, and extracted variables."""
        task = await TaskService.get_task(db, task_id, detailed=False)
        if not task:
            return None

        task.status = status.value
        if result_summary is not None:
            task.result_summary = result_summary
        if extracted_data is not None:
            task.extracted_data = {**task.extracted_data, **extracted_data}

        task.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(task)
        return task

    @staticmethod
    async def save_task_state(db: AsyncSession, state: Any) -> Optional[TaskRun]:
        """Persists the complete TaskState snapshot and execution history into the database."""
        task = await TaskService.get_task(db, state.task_id, detailed=False)
        if not task:
            return None

        status_val = (
            state.current_status.value
            if hasattr(state.current_status, "value")
            else str(state.current_status)
        )
        task.status = status_val
        task.current_step = state.current_step
        task.extracted_data = state.extracted_data
        task.result_summary = state.final_result or state.result_summary
        task.current_objective = state.current_objective
        task.plan_summary = state.plan_summary
        task.state_snapshot = state.to_dict() if hasattr(state, "to_dict") else state.model_dump(mode="json")
        task.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(task)
        return task

    @staticmethod
    async def add_step_log(
        db: AsyncSession,
        task_id: str,
        step_number: int,
        phase: str,
        thought: Optional[str] = None,
        tool_name: Optional[str] = None,
        tool_args: Optional[Dict[str, Any]] = None,
        tool_result: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        latency_ms: float = 0.0,
    ) -> StepLog:
        """Appends an execution step to the task trajectory."""
        step = StepLog(
            task_id=task_id,
            step_number=step_number,
            phase=phase,
            thought=thought,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result,
            error_message=error_message,
            latency_ms=latency_ms,
        )
        db.add(step)
        await db.commit()
        await db.refresh(step)
        return step

    @staticmethod
    async def add_evidence(
        db: AsyncSession,
        task_id: str,
        evidence_type: str,
        description: str,
        source_uri_or_path: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> EvidenceRecord:
        """Records tangible evidence of task milestone or verification."""
        evidence = EvidenceRecord(
            task_id=task_id,
            evidence_type=evidence_type,
            description=description,
            source_uri_or_path=source_uri_or_path,
            payload=payload,
        )
        db.add(evidence)
        await db.commit()
        await db.refresh(evidence)
        return evidence

    @staticmethod
    async def create_approval_request(
        db: AsyncSession,
        task_id: str,
        proposed_action: Dict[str, Any],
        risk_level: str = "HIGH",
        justification: str = "Action requires human authorization before execution.",
    ) -> ApprovalRequest:
        """Creates a pending approval request for a high-risk action."""
        approval = ApprovalRequest(
            task_id=task_id,
            proposed_action=proposed_action,
            risk_level=risk_level,
            justification=justification,
            status="PENDING",
            requested_at=datetime.utcnow(),
        )
        db.add(approval)
        await db.commit()
        await db.refresh(approval)
        logger.info("Approval request created", task_id=task_id, approval_id=approval.id, risk=risk_level)
        return approval

    @staticmethod
    async def get_pending_approval(db: AsyncSession, task_id: str) -> Optional[ApprovalRequest]:
        """Retrieves any active pending approval for a given task."""
        stmt = (
            select(ApprovalRequest)
            .where(ApprovalRequest.task_id == task_id, ApprovalRequest.status == "PENDING")
            .order_by(ApprovalRequest.requested_at.desc())
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def get_approval_by_id(db: AsyncSession, approval_id: str) -> Optional[ApprovalRequest]:
        """Fetches an approval request by primary key."""
        stmt = select(ApprovalRequest).where(ApprovalRequest.id == approval_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def resolve_approval(
        db: AsyncSession,
        approval_id: str,
        approved: bool,
        comment: Optional[str] = None,
    ) -> Optional[ApprovalRequest]:
        """Records a human decision (APPROVE/REJECT) on a pending approval request."""
        approval = await TaskService.get_approval_by_id(db, approval_id)
        if not approval:
            return None

        approval.status = "APPROVED" if approved else "REJECTED"
        approval.resolved_at = datetime.utcnow()
        approval.resolution_comment = comment
        await db.commit()
        await db.refresh(approval)
        logger.info(
            "Approval request resolved",
            approval_id=approval.id,
            task_id=approval.task_id,
            status=approval.status,
        )
        return approval

