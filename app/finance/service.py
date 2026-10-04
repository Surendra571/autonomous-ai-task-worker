"""Business logic, database queries, and authentication for the internal Finance application."""

import hmac
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.models.invoice import Invoice
from app.models.schemas import InvoiceCreate
from app.config.logging import get_logger

logger = get_logger("finance.service")


class FinanceService:
    """Manages invoices and authentication for the internal finance portal."""

    # -----------------------------------------------------------------------
    # Authentication & Session Token Management
    # -----------------------------------------------------------------------

    @staticmethod
    def verify_credentials(username: str, password: str) -> bool:
        """Verifies credentials against deterministic test configuration."""
        is_user_valid = hmac.compare_digest(username.strip(), settings.FINANCE_APP_USERNAME)
        is_pass_valid = hmac.compare_digest(password.strip(), settings.FINANCE_APP_PASSWORD)
        return is_user_valid and is_pass_valid

    @staticmethod
    def generate_session_token(username: str) -> str:
        """Generates a signed HMAC session cookie token."""
        timestamp = str(int(time.time()))
        message = f"{username}:{timestamp}"
        signature = hmac.new(
            settings.FINANCE_APP_SECRET_KEY.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"{message}:{signature}"

    @staticmethod
    def validate_session_token(token: Optional[str]) -> Optional[str]:
        """Validates the session cookie. Returns username if valid, None otherwise."""
        if not token or ":" not in token:
            return None
        parts = token.split(":")
        if len(parts) != 3:
            return None
        username, timestamp_str, received_sig = parts
        message = f"{username}:{timestamp_str}"
        expected_sig = hmac.new(
            settings.FINANCE_APP_SECRET_KEY.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(received_sig, expected_sig):
            return None

        # Check token expiration (e.g. 24 hours)
        try:
            timestamp = int(timestamp_str)
            if time.time() - timestamp > 86400:
                return None
        except ValueError:
            return None

        return username

    # -----------------------------------------------------------------------
    # Invoice Operations
    # -----------------------------------------------------------------------

    @staticmethod
    async def list_invoices(
        db: AsyncSession,
        query: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Invoice]:
        """Lists invoices with optional text search and status filter."""
        stmt = select(Invoice)

        if query and query.strip():
            q_term = f"%{query.strip()}%"
            stmt = stmt.where(
                or_(
                    Invoice.invoice_number.ilike(q_term),
                    Invoice.vendor_name.ilike(q_term),
                    Invoice.description.ilike(q_term),
                )
            )

        if status and status.strip() and status.upper() != "ALL":
            stmt = stmt.where(Invoice.status == status.strip().upper())

        stmt = stmt.order_by(Invoice.created_at.desc()).limit(limit).offset(offset)
        result = await db.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def get_invoice(db: AsyncSession, invoice_id: str) -> Optional[Invoice]:
        """Fetches a single invoice by its primary key ID."""
        stmt = select(Invoice).where(Invoice.id == invoice_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def get_invoice_by_number(db: AsyncSession, invoice_number: str) -> Optional[Invoice]:
        """Fetches an invoice by unique invoice number."""
        stmt = select(Invoice).where(Invoice.invoice_number == invoice_number.strip())
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def create_invoice(db: AsyncSession, invoice_in: InvoiceCreate) -> Invoice:
        """Creates and stores a new invoice."""
        existing = await FinanceService.get_invoice_by_number(db, invoice_in.invoice_number)
        if existing:
            raise ValueError(f"Invoice number '{invoice_in.invoice_number}' already exists.")

        invoice = Invoice(
            invoice_number=invoice_in.invoice_number.strip(),
            vendor_name=invoice_in.vendor_name.strip(),
            amount=round(float(invoice_in.amount), 2),
            currency=invoice_in.currency.strip().upper(),
            issue_date=invoice_in.issue_date.strip(),
            due_date=invoice_in.due_date.strip(),
            status=invoice_in.status.strip().upper(),
            description=invoice_in.description.strip() if invoice_in.description else None,
        )
        db.add(invoice)
        await db.commit()
        await db.refresh(invoice)
        logger.info(
            "Invoice created successfully",
            invoice_id=invoice.id,
            invoice_number=invoice.invoice_number,
            vendor=invoice.vendor_name,
            amount=invoice.amount,
        )
        return invoice

    @staticmethod
    async def get_dashboard_metrics(db: AsyncSession) -> Dict[str, Any]:
        """Calculates dashboard KPI metrics."""
        total_stmt = select(func.count(Invoice.id), func.coalesce(func.sum(Invoice.amount), 0.0))
        total_res = await db.execute(total_stmt)
        total_count, total_value = total_res.one()

        pending_stmt = select(func.count(Invoice.id)).where(Invoice.status == "PENDING")
        pending_count = (await db.execute(pending_stmt)).scalar() or 0

        paid_stmt = select(func.count(Invoice.id)).where(Invoice.status == "PAID")
        paid_count = (await db.execute(paid_stmt)).scalar() or 0

        overdue_stmt = select(func.count(Invoice.id)).where(Invoice.status == "OVERDUE")
        overdue_count = (await db.execute(overdue_stmt)).scalar() or 0

        recent_invoices = await FinanceService.list_invoices(db, limit=5)

        return {
            "total_invoices": total_count,
            "total_value": float(total_value),
            "pending_invoices": pending_count,
            "paid_invoices": paid_count,
            "overdue_invoices": overdue_count,
            "recent_invoices": recent_invoices,
        }
