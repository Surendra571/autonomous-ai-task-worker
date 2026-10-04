"""Dedicated ApprovalManager enforcing risk level policies and human-in-the-loop workflows."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.state import ActionProposal, ActionType
from app.config.logging import get_logger
from app.models.schemas import ActionRiskLevel, ApprovalStatus, RiskLevel, TaskStatus
from app.models.task import ApprovalRequest
from app.services.task_service import TaskService
from app.tools.base import BaseTool

logger = get_logger("approval.manager")


class ApprovalCheckResult(BaseModel):
    """Outcome of evaluating an action against the approval policy."""
    allowed: bool
    requires_approval: bool = False
    risk_level: ActionRiskLevel = ActionRiskLevel.READ
    reason: str
    approval_request_id: Optional[str] = None
    approval_request: Optional[Dict[str, Any]] = None


class ApprovalManager:
    """Manages action risk classification, human approval interception, and execution resumption.

    Risk Levels:
    - READ: Querying, viewing, searching, or extracting without mutation (e.g. Reading invoice).
    - LOW_RISK_WRITE: Safe, standard business mutations (e.g. Creating invoice).
    - HIGH_RISK_WRITE: Sensitive alterations to financial/payment data (e.g. Changing payment information).
    - DESTRUCTIVE: Irreversible deletions or purges (e.g. Deleting invoice).

    Rules:
    - READ -> automatic (no approval required)
    - LOW_RISK_WRITE -> automatic (no approval required)
    - HIGH_RISK_WRITE -> approval required (pause execution, request approval)
    - DESTRUCTIVE -> approval required (pause execution, request approval)
    """

    # Keyword patterns for risk classification
    DESTRUCTIVE_KEYWORDS = (
        "delete",
        "destroy",
        "drop",
        "purge",
        "wipe",
        "truncate",
        "delete_invoice",
        "remove_invoice",
    )

    HIGH_RISK_WRITE_KEYWORDS = (
        "payment",
        "bank",
        "routing",
        "payout",
        "wire",
        "transfer_funds",
        "change_payment",
        "update_payment",
        "modify_bank",
        "modify_payment",
        "update_bank_details",
    )

    LOW_RISK_WRITE_KEYWORDS = (
        "create_invoice",
        "finance_create_invoice_tool",
        "create",
        "insert",
        "save",
        "draft",
        "post",
        "submit",
        "type",
        "click",
        "select",
    )

    READ_KEYWORDS = (
        "read",
        "search",
        "get",
        "fetch",
        "view",
        "inspect",
        "list",
        "check",
        "open_url",
        "extract_text",
        "screenshot",
        "go_back",
    )

    @classmethod
    def classify_risk(
        cls,
        action: ActionProposal,
        tool: Optional[BaseTool] = None,
    ) -> ActionRiskLevel:
        """Classifies the action into one of four risk levels:

        READ, LOW_RISK_WRITE, HIGH_RISK_WRITE, DESTRUCTIVE.
        """
        # 1. Explicit risk level overrides
        explicit_risk = getattr(action, "risk_level", None)
        if explicit_risk:
            val = str(explicit_risk.value if hasattr(explicit_risk, "value") else explicit_risk).upper()
            if val == "DESTRUCTIVE":
                return ActionRiskLevel.DESTRUCTIVE
            if val in ("HIGH_RISK_WRITE", "HIGH"):
                return ActionRiskLevel.HIGH_RISK_WRITE
            if val in ("LOW_RISK_WRITE",):
                return ActionRiskLevel.LOW_RISK_WRITE
            if val in ("READ",):
                return ActionRiskLevel.READ

        if getattr(action, "requires_approval", False) or action.action_type == ActionType.REQUEST_APPROVAL:
            return ActionRiskLevel.HIGH_RISK_WRITE

        if tool and getattr(tool, "risk_level", None):
            tool_risk = str(tool.risk_level.value if hasattr(tool.risk_level, "value") else tool.risk_level).upper()
            if tool_risk == "DESTRUCTIVE":
                return ActionRiskLevel.DESTRUCTIVE
            if tool_risk in ("HIGH_RISK_WRITE", "HIGH"):
                return ActionRiskLevel.HIGH_RISK_WRITE

        # 2. Check semantic keywords in tool name, thought, and arguments
        tool_name = (action.tool_name or "").lower()
        thought = (action.thought or "").lower()
        target_text = f"{tool_name} {thought} {str(action.tool_args or '').lower()}"

        # Check DESTRUCTIVE
        if any(kw in target_text for kw in cls.DESTRUCTIVE_KEYWORDS):
            return ActionRiskLevel.DESTRUCTIVE

        # Check HIGH_RISK_WRITE
        if any(kw in target_text for kw in cls.HIGH_RISK_WRITE_KEYWORDS):
            return ActionRiskLevel.HIGH_RISK_WRITE

        # Check LOW_RISK_WRITE (e.g. creating invoice)
        if any(kw in tool_name for kw in cls.LOW_RISK_WRITE_KEYWORDS) or any(kw in thought for kw in ("create invoice", "creating invoice")):
            return ActionRiskLevel.LOW_RISK_WRITE

        # Check READ (e.g. reading invoice, searching documents)
        if any(kw in tool_name for kw in cls.READ_KEYWORDS) or any(kw in thought for kw in ("read invoice", "reading invoice", "search")):
            return ActionRiskLevel.READ

        # Fallback based on low risk or default to READ
        if explicit_risk in (RiskLevel.LOW, "LOW"):
            return ActionRiskLevel.LOW_RISK_WRITE

        return ActionRiskLevel.READ

    @classmethod
    def requires_approval(cls, risk_level: ActionRiskLevel) -> bool:
        """Rules:

        READ -> automatic (False)
        LOW_RISK_WRITE -> automatic (False)
        HIGH_RISK_WRITE -> approval (True)
        DESTRUCTIVE -> approval (True)
        """
        return risk_level in (ActionRiskLevel.HIGH_RISK_WRITE, ActionRiskLevel.DESTRUCTIVE)

    @classmethod
    async def evaluate_action(
        cls,
        db: AsyncSession,
        task_id: str,
        action: ActionProposal,
        tool: Optional[BaseTool] = None,
        approved_action_signatures: Optional[List[str]] = None,
    ) -> ApprovalCheckResult:
        """Evaluates whether the action can execute automatically or must be paused for human approval."""
        risk_level = cls.classify_risk(action=action, tool=tool)
        needs_approval = cls.requires_approval(risk_level)
        signature = f"{action.tool_name}:{sorted(action.tool_args.items())}" if action.tool_name else action.thought

        # If previously approved by operator, permit execution
        if approved_action_signatures and signature in approved_action_signatures:
            return ApprovalCheckResult(
                allowed=True,
                requires_approval=False,
                risk_level=risk_level,
                reason="Action signature was previously approved by human supervisor.",
            )

        # Automatic execution for READ and LOW_RISK_WRITE
        if not needs_approval:
            return ApprovalCheckResult(
                allowed=True,
                requires_approval=False,
                risk_level=risk_level,
                reason=f"Action risk level '{risk_level.value}' is authorized for automatic execution.",
            )

        # High-risk action: check if approval already pending in DB
        existing_pending = await TaskService.get_pending_approval(db=db, task_id=task_id)
        if existing_pending:
            return ApprovalCheckResult(
                allowed=False,
                requires_approval=True,
                risk_level=risk_level,
                reason=f"Action is awaiting resolution of existing pending approval request '{existing_pending.id}'.",
                approval_request_id=existing_pending.id,
                approval_request={
                    "id": existing_pending.id,
                    "proposed_action": existing_pending.proposed_action,
                    "risk_level": existing_pending.risk_level,
                    "justification": existing_pending.justification,
                },
            )

        # Create new ApprovalRequest in DB
        justification = (
            f"Action '{action.tool_name or action.action_type.value}' carries {risk_level.value} risk. "
            f"Reasoning: {action.thought}"
        )
        proposed_payload = {
            "tool_name": action.tool_name,
            "tool_args": action.tool_args,
            "thought": action.thought,
            "risk_level": risk_level.value,
            "signature": signature,
        }

        approval = await TaskService.create_approval_request(
            db=db,
            task_id=task_id,
            proposed_action=proposed_payload,
            risk_level=risk_level.value,
            justification=justification,
        )

        # Pause execution by updating task status to WAITING_FOR_APPROVAL
        await TaskService.update_task_status(
            db=db,
            task_id=task_id,
            status=TaskStatus.WAITING_FOR_APPROVAL,
            result_summary=f"Execution paused: Action requires human approval ({risk_level.value}).",
        )

        logger.info(
            "EXECUTION_PAUSED_FOR_APPROVAL",
            task_id=task_id,
            approval_id=approval.id,
            risk_level=risk_level.value,
            tool=action.tool_name,
        )

        return ApprovalCheckResult(
            allowed=False,
            requires_approval=True,
            risk_level=risk_level,
            reason=f"Action requires human authorization ({risk_level.value}). Approval request '{approval.id}' created.",
            approval_request_id=approval.id,
            approval_request={
                "id": approval.id,
                "proposed_action": approval.proposed_action,
                "risk_level": approval.risk_level,
                "justification": approval.justification,
            },
        )

    @classmethod
    async def approve(
        cls,
        db: AsyncSession,
        task_id: str,
        comment: Optional[str] = None,
    ) -> Optional[ApprovalRequest]:
        """Approves a pending approval request and resumes autonomous execution."""
        from app.agent.orchestrator import AgentOrchestrator

        pending = await TaskService.get_pending_approval(db=db, task_id=task_id)
        if not pending:
            return None

        # Resolve approval record in DB
        resolved = await TaskService.resolve_approval(
            db=db,
            approval_id=pending.id,
            approved=True,
            comment=comment or "Approved by operator.",
        )

        # Reconstruct the authorized action
        proposed = pending.proposed_action or {}
        risk_str = pending.risk_level or "LOW"
        try:
            risk = RiskLevel(risk_str)
        except ValueError:
            risk = RiskLevel.LOW

        resumed_action = ActionProposal(
            thought=f"Resuming human-authorized action: {proposed.get('thought', '')}",
            action_type=ActionType.CALL_TOOL,
            tool_name=proposed.get("tool_name"),
            tool_args=proposed.get("tool_args", {}),
            risk_level=risk,
        )

        # Resume execution
        orchestrator = AgentOrchestrator()
        await orchestrator.execute_task(db=db, task_id=task_id, resumed_action=resumed_action)

        return resolved

    @classmethod
    async def reject(
        cls,
        db: AsyncSession,
        task_id: str,
        comment: Optional[str] = None,
    ) -> Optional[ApprovalRequest]:
        """Rejects a pending approval request and marks task as FAILED."""
        pending = await TaskService.get_pending_approval(db=db, task_id=task_id)
        if not pending:
            return None

        resolved = await TaskService.resolve_approval(
            db=db,
            approval_id=pending.id,
            approved=False,
            comment=comment or "Rejected by human supervisor.",
        )

        await TaskService.update_task_status(
            db=db,
            task_id=task_id,
            status=TaskStatus.FAILED,
            result_summary=f"Task terminated: Action rejected by human supervisor. Comment: {comment or 'No comment provided'}",
        )

        logger.info(
            "ACTION_APPROVAL_REJECTED",
            task_id=task_id,
            approval_id=pending.id,
            comment=comment,
        )

        return resolved

