"""Approval Gate enforcing human-in-the-loop policies on risky actions."""

from typing import Any, Dict, Optional
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.state import ActionProposal, ActionType
from app.config.logging import get_logger
from app.models.schemas import RiskLevel, TaskStatus
from app.models.task import ApprovalRequest
from app.services.task_service import TaskService
from app.tools.base import BaseTool

logger = get_logger("approval.gate")


class ApprovalCheckResult(BaseModel):
    """Outcome of evaluating an action against the approval gate."""
    allowed: bool
    requires_approval: bool = False
    reason: str
    approval_request_id: Optional[str] = None
    approval_request: Optional[Dict[str, Any]] = None


class ApprovalGate:
    """Interception mechanism preventing unauthorized execution of high-risk actions."""

    # Explicit sensitive actions that always trigger human-in-the-loop validation
    HIGH_RISK_TOOLS = {
        "finance_create_invoice_tool",
    }

    @classmethod
    def is_action_risky(cls, action: ActionProposal, tool: Optional[BaseTool] = None) -> bool:
        """Determines whether an action requires explicit human approval."""
        if action.action_type == ActionType.REQUEST_APPROVAL:
            return True

        if action.risk_level == RiskLevel.HIGH:
            return True

        if tool and tool.risk_level == RiskLevel.HIGH:
            return True

        if action.tool_name in cls.HIGH_RISK_TOOLS:
            return True

        return False

    @classmethod
    async def evaluate_action(
        cls,
        db: AsyncSession,
        task_id: str,
        action: ActionProposal,
        tool: Optional[BaseTool] = None,
        approved_action_signatures: Optional[list[str]] = None,
    ) -> ApprovalCheckResult:
        """Evaluates whether the proposed action is permitted to execute or requires human approval."""
        from app.approval.manager import ApprovalManager

        result = await ApprovalManager.evaluate_action(
            db=db,
            task_id=task_id,
            action=action,
            tool=tool,
            approved_action_signatures=approved_action_signatures,
        )
        return ApprovalCheckResult(
            allowed=result.allowed,
            requires_approval=result.requires_approval,
            reason=result.reason,
            approval_request_id=result.approval_request_id,
            approval_request=result.approval_request,
        )
