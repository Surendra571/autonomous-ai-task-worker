"""Finance Create Invoice Tool for entering new invoice records into the ledger."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from app.db.session import get_session_factory
from app.finance.service import FinanceService
from app.models.schemas import InvoiceCreate
from app.tools.base import BaseTool, ToolResult


from datetime import datetime
from pydantic import BaseModel, Field, field_validator


class FinanceCreateInvoiceInput(BaseModel):
    """Input parameters for entering an invoice into the finance system."""
    invoice_number: str = Field(..., min_length=2, description="Unique invoice number (e.g. 'INV-ACM-2024-015').")
    vendor_name: str = Field(..., min_length=2, description="Name of the vendor (e.g. 'Acme Corp').")
    amount: float = Field(..., gt=0.0, description="Total invoice amount to record.")
    currency: str = Field(default="USD", description="Currency (defaults to 'USD').")
    issue_date: Optional[str] = Field(default=None, description="Issue date in YYYY-MM-DD format.")
    due_date: str = Field(..., description="Due date in YYYY-MM-DD format.")
    status: str = Field(default="PENDING", description="Initial status: 'PENDING', 'APPROVED', etc.")
    description: Optional[str] = Field(None, description="Optional service notes or line item description.")

    @field_validator("issue_date", mode="before")
    @classmethod
    def set_issue_date(cls, v: Any) -> str:
        if v is None or not str(v).strip():
            return datetime.utcnow().strftime("%Y-%m-%d")
        return str(v).strip()


class FinanceCreateInvoiceTool(BaseTool):
    """
    Inserts a validated vendor invoice into the internal finance ledger.
    Classified as HIGH risk because it mutates accounts payable financial records.
    """

    name = "finance_create_invoice_tool"
    description = (
        "Create and save a new vendor invoice into the internal finance system ledger. "
        "Requires invoice_number, vendor_name, amount, issue_date, and due_date. "
        "High-risk action requiring safety verification."
    )
    input_schema = FinanceCreateInvoiceInput
    risk_level = "HIGH"

    def __init__(self, session_factory=None):
        self.session_factory = session_factory or get_session_factory()

    async def execute(self, params: FinanceCreateInvoiceInput, session: Optional[Any] = None) -> ToolResult:
        if session is not None:
            return await self._create(session, params)

        async with self.session_factory() as sess:
            return await self._create(sess, params)

    async def _create(self, session: Any, params: FinanceCreateInvoiceInput) -> ToolResult:
        try:
            invoice_in = InvoiceCreate(
                invoice_number=params.invoice_number,
                vendor_name=params.vendor_name,
                amount=params.amount,
                currency=params.currency,
                issue_date=params.issue_date,
                due_date=params.due_date,
                status=params.status,
                description=params.description,
            )
            invoice = await FinanceService.create_invoice(session, invoice_in)

            return ToolResult.ok(
                data={
                    "invoice_id": invoice.id,
                    "invoice_number": invoice.invoice_number,
                    "vendor_name": invoice.vendor_name,
                    "amount": invoice.amount,
                    "currency": invoice.currency,
                    "issue_date": invoice.issue_date,
                    "due_date": invoice.due_date,
                    "status": invoice.status,
                    "created_at": invoice.created_at.isoformat(),
                },
                evidence={
                    "saved_in_database": True,
                    "db_primary_key": invoice.id,
                    "invoice_number": invoice.invoice_number,
                    "amount": invoice.amount,
                },
                metadata={"action": "create_invoice", "vendor": invoice.vendor_name},
            )
        except ValueError as e:
            return ToolResult.fail(
                f"Finance ledger validation error: {e}",
                metadata={"invoice_number": params.invoice_number},
            )
        except Exception as e:
            return ToolResult.fail(
                f"Failed to record invoice into finance database: {e}",
                metadata={"invoice_number": params.invoice_number},
            )
