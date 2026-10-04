"""SQLAlchemy models for Task execution, step tracking, approvals, and evidence."""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    JSON,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.models.base import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


class TaskRun(Base):
    """Represents a discrete goal-oriented autonomous task run."""

    __tablename__ = "task_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    goal: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="PENDING", nullable=False)
    current_step: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    max_steps: Mapped[int] = mapped_column(Integer, default=15, nullable=False)

    # Structured working memory and extracted variables
    extracted_data: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    result_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    current_objective: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    plan_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    state_snapshot: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    step_logs: Mapped[List["StepLog"]] = relationship(
        "StepLog", back_populates="task", cascade="all, delete-orphan", order_by="StepLog.step_number"
    )
    approvals: Mapped[List["ApprovalRequest"]] = relationship(
        "ApprovalRequest", back_populates="task", cascade="all, delete-orphan", order_by="ApprovalRequest.requested_at"
    )
    evidence: Mapped[List["EvidenceRecord"]] = relationship(
        "EvidenceRecord", back_populates="task", cascade="all, delete-orphan", order_by="EvidenceRecord.recorded_at"
    )


class StepLog(Base):
    """Audit log capturing a single step in the execution loop."""

    __tablename__ = "step_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    task_id: Mapped[str] = mapped_column(String(36), ForeignKey("task_runs.id"), nullable=False, index=True)
    step_number: Mapped[int] = mapped_column(Integer, nullable=False)
    phase: Mapped[str] = mapped_column(String(32), nullable=False)  # OBSERVE, DECIDE, ACT, VERIFY, RECOVER
    thought: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tool_name: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    tool_args: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    tool_result: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    task: Mapped["TaskRun"] = relationship("TaskRun", back_populates="step_logs")


class ApprovalRequest(Base):
    """Human-in-the-loop approval record for high-risk actions."""

    __tablename__ = "approval_requests"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    task_id: Mapped[str] = mapped_column(String(36), ForeignKey("task_runs.id"), nullable=False, index=True)
    proposed_action: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(16), default="HIGH", nullable=False)
    justification: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="PENDING", nullable=False)  # PENDING, APPROVED, REJECTED
    requested_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    resolution_comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    task: Mapped["TaskRun"] = relationship("TaskRun", back_populates="approvals")


class EvidenceRecord(Base):
    """Tangible evidence verifying goal completion."""

    __tablename__ = "evidence_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    task_id: Mapped[str] = mapped_column(String(36), ForeignKey("task_runs.id"), nullable=False, index=True)
    evidence_type: Mapped[str] = mapped_column(String(32), nullable=False)  # e.g., screenshot, db_record, pdf_diff
    description: Mapped[str] = mapped_column(Text, nullable=False)
    source_uri_or_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    payload: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    task: Mapped["TaskRun"] = relationship("TaskRun", back_populates="evidence")

