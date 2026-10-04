"""Agent state machine and execution memory models."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.models.schemas import RiskLevel, TaskStatus


class ActionType(str, Enum):
    """Categorization of actions the planner can decide to take."""
    CALL_TOOL = "CALL_TOOL"
    REQUEST_APPROVAL = "REQUEST_APPROVAL"
    REQUEST_CLARIFICATION = "REQUEST_CLARIFICATION"
    COMPLETE_TASK = "COMPLETE_TASK"
    FAIL_TASK = "FAIL_TASK"


class ActionCall(BaseModel):
    """Structured tool call chosen by the planner."""
    tool: str = Field(..., description="Name of the tool to execute, or special actions: 'complete_task', 'fail_task', 'request_clarification', 'request_approval'.")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Arguments to provide to the chosen tool.")


class PlannerDecision(BaseModel):
    """Strict structured decision format returned by the LLM planner."""
    reason: str = Field(..., description="Step-by-step reasoning explaining why this action is chosen.")
    action: ActionCall = Field(..., description="The single next action to execute.")
    expected_outcome: str = Field(..., description="What the agent expects to happen after this action is performed.")
    verification_needed: bool = Field(default=False, description="Whether the outcome of this action requires verification.")
    requires_approval: bool = Field(default=False, description="Whether this action requires human approval before executing.")

    def to_action_proposal(self) -> "ActionProposal":
        """Converts structured PlannerDecision to an ActionProposal for the orchestrator."""
        tool_raw = (self.action.tool or "").strip()
        tool_lower = tool_raw.lower()

        if tool_lower in ("complete_task", "complete"):
            action_type = ActionType.COMPLETE_TASK
            tool_name = None
        elif tool_lower in ("fail_task", "fail"):
            action_type = ActionType.FAIL_TASK
            tool_name = None
        elif tool_lower in ("request_clarification", "clarify", "ask_user"):
            action_type = ActionType.REQUEST_CLARIFICATION
            tool_name = None
        elif tool_lower in ("request_approval", "ask_approval"):
            action_type = ActionType.REQUEST_APPROVAL
            tool_name = None
        else:
            action_type = ActionType.CALL_TOOL
            tool_name = tool_raw

        risk = RiskLevel.HIGH if self.requires_approval else RiskLevel.LOW

        return ActionProposal(
            thought=self.reason,
            action_type=action_type,
            tool_name=tool_name,
            tool_args=self.action.arguments or {},
            risk_level=risk,
            expected_outcome=self.expected_outcome,
            verification_needed=self.verification_needed,
            requires_approval=self.requires_approval,
        )

    @classmethod
    def from_action_proposal(cls, proposal: "ActionProposal") -> "PlannerDecision":
        """Constructs a PlannerDecision from an existing ActionProposal."""
        if proposal.action_type == ActionType.COMPLETE_TASK:
            tool_name = "complete_task"
        elif proposal.action_type == ActionType.FAIL_TASK:
            tool_name = "fail_task"
        elif proposal.action_type == ActionType.REQUEST_CLARIFICATION:
            tool_name = "request_clarification"
        elif proposal.action_type == ActionType.REQUEST_APPROVAL:
            tool_name = "request_approval"
        else:
            tool_name = proposal.tool_name or "noop"

        return cls(
            reason=proposal.thought,
            action=ActionCall(tool=tool_name, arguments=proposal.tool_args or {}),
            expected_outcome=proposal.expected_outcome or "",
            verification_needed=proposal.verification_needed,
            requires_approval=proposal.requires_approval or (proposal.risk_level == RiskLevel.HIGH),
        )


class ActionProposal(BaseModel):
    """Structured decision returned by the planner."""
    thought: str = Field(..., description="Reasoning explaining why this action was chosen.")
    action_type: ActionType = Field(..., description="The type of action to take.")
    tool_name: Optional[str] = Field(None, description="Name of tool to execute if action_type is CALL_TOOL.")
    tool_args: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the selected tool.")
    risk_level: Any = Field(default=RiskLevel.LOW, description="Estimated risk of executing this action.")
    expected_outcome: Optional[str] = Field(None, description="What the agent expects to happen after this action.")
    verification_needed: bool = Field(default=False, description="Whether the action outcome needs verification.")
    requires_approval: bool = Field(default=False, description="Whether human approval is required.")

    @property
    def reason(self) -> str:
        return self.thought

    @property
    def tool(self) -> Optional[str]:
        return self.tool_name

    @property
    def arguments(self) -> Dict[str, Any]:
        return self.tool_args


class Observation(BaseModel):
    """Structured observation captured after executing an action or observing the environment."""
    step_number: int
    source: str = Field(..., description="Origin of observation (e.g., tool name, system, verifier).")
    success: bool = True
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    evidence: Any = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class FailureRecord(BaseModel):
    """Structured log of a failure and the attempted recovery."""
    step_number: int
    error_type: str
    error_message: str
    tool_name: Optional[str] = None
    recovery_attempted: Optional[str] = None
    recovered: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class VerificationResult(BaseModel):
    """Independent outcome verification assessment."""
    verified: bool = Field(..., description="True if the outcome satisfies all goal assertions.")
    verification_type: str = Field(..., description="e.g. cross_check, screen_inspection, database_query")
    checks: List[Dict[str, Any]] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)
    discrepancies: List[str] = Field(default_factory=list, description="Any mismatched fields or validation errors.")
    evidence_uris: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def to_report(self) -> Dict[str, Any]:
        """Returns structured dictionary report."""
        return self.model_dump()


class TaskOutcome(BaseModel):
    """Generic contract for task execution outcomes across any business domain."""
    status: TaskStatus = TaskStatus.COMPLETED
    success: bool = True
    summary: str = Field(default="", description="Summary of outcome.")
    facts: Dict[str, Any] = Field(default_factory=dict, description="Discovered facts and extracted entities.")
    evidence: List[Dict[str, Any]] = Field(default_factory=list, description="Audit artifacts, URLs, and diffs.")
    verification: Optional[Dict[str, Any]] = Field(default=None, description="Verification checks and results.")
    reason: Optional[str] = Field(default=None, description="Detailed explanation of the outcome.")


class ExecutionStep(BaseModel):
    """A complete record of a single turn in the execution loop."""
    step_number: int
    phase: str  # OBSERVE, DECIDE, ACT, VERIFY, RECOVER
    thought: Optional[str] = None
    action: Optional[ActionProposal] = None
    observation: Optional[Observation] = None
    verification: Optional[VerificationResult] = None
    latency_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TaskState(BaseModel):
    """Full working memory and execution state for an autonomous task."""
    task_id: str
    goal: str = Field(default="", description="The natural language goal of the task.")
    status: TaskStatus = TaskStatus.PENDING
    current_step: int = 0
    max_steps: int = 15

    # Structured working memory extracted across actions (e.g. invoice_number, amount)
    extracted_data: Dict[str, Any] = Field(default_factory=dict)
    current_objective: Optional[str] = None
    plan_summary: Optional[str] = None
    retry_count: int = 0
    completed_steps: List[str] = Field(default_factory=list)
    failed_steps: List[str] = Field(default_factory=list)
    execution_history: List[Dict[str, str]] = Field(default_factory=list)
    screenshots: List[str] = Field(default_factory=list)
    approval_requests: List[Dict[str, Any]] = Field(default_factory=list)
    verification_results: List[Dict[str, Any]] = Field(default_factory=list)
    tool_results: List[Dict[str, Any]] = Field(default_factory=list)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)

    # History & Memory
    steps: List[ExecutionStep] = Field(default_factory=list)
    recent_observations: List[Observation] = Field(default_factory=list)
    failures: List[FailureRecord] = Field(default_factory=list)
    consecutive_failures: int = 0

    # Verification and Approval State
    pending_approval_id: Optional[str] = None
    approved_actions: List[str] = Field(default_factory=list)
    latest_verification: Optional[VerificationResult] = None
    result_summary: Optional[str] = None
    outcome: Optional[TaskOutcome] = None

    def __init__(self, **data: Any):
        if "user_goal" in data and "goal" not in data:
            data["goal"] = data.pop("user_goal")
        if "current_status" in data and "status" not in data:
            data["status"] = data.pop("current_status")
        if "final_result" in data and "result_summary" not in data:
            data["result_summary"] = data.pop("final_result")
        if "observations" in data and "recent_observations" not in data:
            data["recent_observations"] = data.pop("observations")
        super().__init__(**data)

    @property
    def user_goal(self) -> str:
        return self.goal

    @user_goal.setter
    def user_goal(self, value: str) -> None:
        self.goal = value

    @property
    def current_status(self) -> TaskStatus:
        return self.status

    @current_status.setter
    def current_status(self, value: TaskStatus) -> None:
        self.status = value

    @property
    def final_result(self) -> Optional[str]:
        return self.result_summary

    @final_result.setter
    def final_result(self, value: Optional[str]) -> None:
        self.result_summary = value

    @property
    def observations(self) -> List[Observation]:
        return self.recent_observations

    def add_observation(self, observation: Observation) -> None:
        """Appends an observation to working memory."""
        self.recent_observations.append(observation)
        if len(self.recent_observations) > 10:
            self.recent_observations = self.recent_observations[-10:]

    def record_failure(self, error_message: str, tool_name: Optional[str] = None, error_type: str = "ExecutionError") -> None:
        """Records an execution error and increments failure count."""
        self.consecutive_failures += 1
        self.failed_steps.append(f"{tool_name or 'unknown'}: {error_message}")
        self.failures.append(
            FailureRecord(
                step_number=self.current_step,
                error_type=error_type,
                error_message=error_message,
                tool_name=tool_name,
            )
        )

    def record_success(self) -> None:
        """Resets consecutive failure counter upon a successful step."""
        self.consecutive_failures = 0

    def record_action_result(self, action: str, result: str) -> None:
        """Records an action and outcome into execution history."""
        self.execution_history.append({"action": action, "result": result})

    def update_extracted_data(self, key_or_dict: Any, value: Any = None) -> None:
        """Updates internal working memory with newly extracted facts."""
        if isinstance(key_or_dict, dict):
            self.extracted_data.update(key_or_dict)
        elif isinstance(key_or_dict, str):
            self.extracted_data[key_or_dict] = value

    def build_outcome(self, summary: Optional[str] = None) -> TaskOutcome:
        """Constructs a standardized TaskOutcome from current state."""
        is_success = self.status == TaskStatus.COMPLETED
        verif = self.latest_verification.model_dump() if self.latest_verification else None
        res_summary = summary or self.result_summary or ("Task completed successfully." if is_success else "Task halted.")
        outcome_obj = TaskOutcome(
            status=self.status,
            success=is_success,
            summary=res_summary,
            facts=dict(self.extracted_data),
            evidence=list(self.evidence),
            verification=verif,
            reason=res_summary,
        )
        self.outcome = outcome_obj
        return outcome_obj

