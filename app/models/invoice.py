"""SQLAlchemy model for the internal Finance application invoices."""

import uuid
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    DateTime,
    Float,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column
from app.models.base import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


class Invoice(Base):
    """Represents an internal accounts payable invoice record."""

    __tablename__ = "finance_invoices"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    invoice_number: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    vendor_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    currency: Mapped[str] = mapped_column(String(8), default="USD", nullable=False)
    issue_date: Mapped[str] = mapped_column(String(16), nullable=False)  # YYYY-MM-DD
    due_date: Mapped[str] = mapped_column(String(16), nullable=False)    # YYYY-MM-DD
    status: Mapped[str] = mapped_column(String(32), default="PENDING", nullable=False)  # PENDING, PAID, OVERDUE, REJECTED
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
