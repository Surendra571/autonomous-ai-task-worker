"""FastAPI API router defining health and task endpoints."""

from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.db.session import check_database_connection, get_db
from app.models.schemas import (
    ApprovalDecisionRequest,
    ApprovalRequestResponse,
    HealthResponse,
    TaskCreate,
    TaskDetailResponse,
    TaskResponse,
)
from app.services.task_service import TaskService
from app.config.logging import get_logger

logger = get_logger("api.routes")
router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check endpoint",
)
async def get_health() -> HealthResponse:
    """Verifies that the application and its database are operational."""
    db_ok = await check_database_connection()
    return HealthResponse(
        status="ok" if db_ok else "degraded",
        version=settings.APP_VERSION,
        environment=settings.APP_ENV,
        database_connected=db_ok,
        timestamp=datetime.utcnow(),
    )


@router.post(
    "/tasks",
    response_model=TaskResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Tasks"],
    summary="Create a new autonomous task",
)
async def create_task(
    task_in: TaskCreate,
    db: AsyncSession = Depends(get_db),
) -> TaskResponse:
    """Accepts a natural-language business goal and registers a new task run."""
    task = await TaskService.create_task(db=db, task_in=task_in)
    return TaskResponse.model_validate(task)


@router.get(
    "/tasks",
    response_model=List[TaskResponse],
    tags=["Tasks"],
    summary="List recent tasks",
)
async def list_tasks(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
) -> List[TaskResponse]:
    """Retrieves a list of recent task runs."""
    tasks = await TaskService.list_tasks(db=db, limit=limit, offset=offset)
    return [TaskResponse.model_validate(t) for t in tasks]


@router.get(
    "/tasks/{task_id}",
    response_model=TaskDetailResponse,
    tags=["Tasks"],
    summary="Get detailed task trajectory and status",
)
async def get_task(
    task_id: str,
    db: AsyncSession = Depends(get_db),
) -> TaskDetailResponse:
    """Retrieves the full trajectory, observations, approvals, and evidence for a task."""
    task = await TaskService.get_task(db=db, task_id=task_id, detailed=True)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with ID '{task_id}' was not found.",
        )
    return TaskDetailResponse.model_validate(task)


@router.post(
    "/tasks/{task_id}/run",
    response_model=TaskDetailResponse,
    tags=["Tasks"],
    summary="Trigger or resume autonomous task execution",
)
async def run_task(
    task_id: str,
    db: AsyncSession = Depends(get_db),
) -> TaskDetailResponse:
    """Launches the autonomous execution loop for the task."""
    from app.agent.orchestrator import AgentOrchestrator

    orchestrator = AgentOrchestrator()
    try:
        await orchestrator.execute_task(db=db, task_id=task_id)
    except ValueError as e:
        if "does not exist" in str(e).lower():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
        logger.error("Validation/value error while running task", task_id=task_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Execution error: {str(e)}",
        )
    except Exception as e:
        logger.error("Execution error while running task", task_id=task_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Execution error: {str(e)}",
        )

    task = await TaskService.get_task(db=db, task_id=task_id, detailed=True)
    return TaskDetailResponse.model_validate(task)


@router.get(
    "/tasks/{task_id}/approvals",
    response_model=List[ApprovalRequestResponse],
    tags=["Approvals"],
    summary="Get all approval requests for a task",
)
async def get_task_approvals(
    task_id: str,
    db: AsyncSession = Depends(get_db),
) -> List[ApprovalRequestResponse]:
    """Retrieves all approval requests associated with a task."""
    task = await TaskService.get_task(db=db, task_id=task_id, detailed=True)
    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task '{task_id}' not found.")
    return [ApprovalRequestResponse.model_validate(a) for a in task.approvals]


@router.post(
    "/tasks/{task_id}/approve",
    response_model=TaskDetailResponse,
    tags=["Approvals"],
    summary="Approve pending risky action and resume execution",
)
async def approve_task(
    task_id: str,
    decision: Optional[ApprovalDecisionRequest] = None,
    db: AsyncSession = Depends(get_db),
) -> TaskDetailResponse:
    """Authorizes a pending high-risk action and continues autonomous execution."""
    from app.agent.orchestrator import AgentOrchestrator
    from app.agent.state import ActionProposal, ActionType

    pending = await TaskService.get_pending_approval(db=db, task_id=task_id)
    if not pending:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No pending approval request found for task '{task_id}'.",
        )

    # Resolve approval
    comment = decision.comment if (decision and decision.comment) else "Approved by operator."
    await TaskService.resolve_approval(
        db=db,
        approval_id=pending.id,
        approved=True,
        comment=comment,
    )

    # Resume action
    proposed = pending.proposed_action
    resumed_action = ActionProposal(
        thought=f"Resuming human-authorized action: {proposed.get('thought', '')}",
        action_type=ActionType.CALL_TOOL,
        tool_name=proposed.get("tool_name"),
        tool_args=proposed.get("tool_args", {}),
        risk_level=pending.risk_level,
    )

    orchestrator = AgentOrchestrator()
    await orchestrator.execute_task(db=db, task_id=task_id, resumed_action=resumed_action)

    task = await TaskService.get_task(db=db, task_id=task_id, detailed=True)
    return TaskDetailResponse.model_validate(task)


@router.post(
    "/tasks/{task_id}/reject",
    response_model=TaskDetailResponse,
    tags=["Approvals"],
    summary="Reject pending risky action",
)
async def reject_task(
    task_id: str,
    decision: Optional[ApprovalDecisionRequest] = None,
    db: AsyncSession = Depends(get_db),
) -> TaskDetailResponse:
    """Rejects a pending high-risk action and marks task as failed or halts execution."""
    from app.models.schemas import TaskStatus

    pending = await TaskService.get_pending_approval(db=db, task_id=task_id)
    if not pending:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No pending approval request found for task '{task_id}'.",
        )

    comment = decision.comment if (decision and decision.comment) else "Rejected by human supervisor."
    await TaskService.resolve_approval(
        db=db,
        approval_id=pending.id,
        approved=False,
        comment=comment,
    )

    await TaskService.update_task_status(
        db=db,
        task_id=task_id,
        status=TaskStatus.FAILED,
        result_summary=f"Task terminated: High-risk action rejected by human supervisor. Comment: {comment}",
    )

    task = await TaskService.get_task(db=db, task_id=task_id, detailed=True)
    return TaskDetailResponse.model_validate(task)

