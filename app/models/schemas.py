"""Pydantic schemas for API requests, responses, and internal DTOs."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
    WAITING_FOR_CLARIFICATION = "WAITING_FOR_CLARIFICATION"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ApprovalStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ActionRiskLevel(str, Enum):
    READ = "READ"
    LOW_RISK_WRITE = "LOW_RISK_WRITE"
    HIGH_RISK_WRITE = "HIGH_RISK_WRITE"
    DESTRUCTIVE = "DESTRUCTIVE"


# ---------------------------------------------------------------------------
# API Request Models
# ---------------------------------------------------------------------------

class TaskCreate(BaseModel):
    """Payload to create and launch an autonomous task."""
    goal: str = Field(
        ...,
        min_length=5,
        description="Natural language instruction of the business task to execute.",
        examples=["Find the latest invoice from Acme Corp, extract the invoice number, amount and due date, enter the information into the internal finance application, and verify that it was saved correctly."]
    )
    max_steps: Optional[int] = Field(default=15, ge=1, le=50, description="Safety limit on execution steps.")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Arbitrary task context metadata.")


class ApprovalDecisionRequest(BaseModel):
    """Payload to approve or reject a pending risky action."""
    approved: Optional[bool] = Field(None, description="True to approve execution, False to reject.")
    comment: Optional[str] = Field(None, description="Optional explanation or reason.")


# ---------------------------------------------------------------------------
# API Response Models
# ---------------------------------------------------------------------------

class StepLogResponse(BaseModel):
    id: str
    step_number: int
    phase: str
    thought: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    latency_ms: float
    created_at: datetime

    model_config = {"from_attributes": True}


class ApprovalRequestResponse(BaseModel):
    id: str
    proposed_action: Dict[str, Any]
    risk_level: str
    justification: str
    status: str
    requested_at: datetime
    resolved_at: Optional[datetime] = None
    resolution_comment: Optional[str] = None

    model_config = {"from_attributes": True}


class EvidenceRecordResponse(BaseModel):
    id: str
    evidence_type: str
    description: str
    source_uri_or_path: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    recorded_at: datetime

    model_config = {"from_attributes": True}


class TaskResponse(BaseModel):
    """Basic task summary response."""
    id: str
    goal: str
    status: TaskStatus
    current_step: int
    max_steps: int
    result_summary: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class TaskDetailResponse(TaskResponse):
    """Detailed task response including step logs, pending approvals, and evidence."""
    current_objective: Optional[str] = None
    plan_summary: Optional[str] = None
    state_snapshot: Optional[Dict[str, Any]] = Field(default_factory=dict)
    extracted_data: Dict[str, Any] = Field(default_factory=dict)
    step_logs: List[StepLogResponse] = Field(default_factory=list)
    approvals: List[ApprovalRequestResponse] = Field(default_factory=list)
    evidence: List[EvidenceRecordResponse] = Field(default_factory=list)

    model_config = {"from_attributes": True}


class HealthResponse(BaseModel):
    """Healthcheck endpoint response."""
    status: str = "ok"
    version: str
    environment: str
    database_connected: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Finance Application Schemas
# ---------------------------------------------------------------------------

class InvoiceStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    PAID = "PAID"
    REJECTED = "REJECTED"


class InvoiceCreate(BaseModel):
    invoice_number: str = Field(..., min_length=2, max_length=64, description="Unique invoice identifier")
    vendor_name: str = Field(..., min_length=2, max_length=128, description="Vendor company name")
    amount: float = Field(..., gt=0, description="Total billed amount")
    currency: str = Field(default="USD", max_length=8)
    issue_date: str = Field(..., description="Issue date in YYYY-MM-DD format")
    due_date: str = Field(..., description="Due date in YYYY-MM-DD format")
    status: str = Field(default="PENDING", description="Status of the invoice")
    description: Optional[str] = Field(None, description="Invoice line item or service description")


class InvoiceResponse(BaseModel):
    id: str
    invoice_number: str
    vendor_name: str
    amount: float
    currency: str
    issue_date: str
    due_date: str
    status: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

