"""Finance Read Invoice Tool for inspecting and verifying saved invoice records."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from app.db.session import get_session_factory
from app.finance.service import FinanceService
from app.tools.base import BaseTool, ToolResult


class FinanceReadInvoiceInput(BaseModel):
    """Input parameters for reading a specific invoice record."""
    invoice_id: Optional[str] = Field(None, description="System UUID identifier of the invoice.")
    invoice_number: Optional[str] = Field(None, description="Unique invoice number string (e.g. 'INV-ACM-2024-001').")


class FinanceReadInvoiceTool(BaseTool):
    """
    Retrieves detailed records for an invoice by ID or invoice number.
    Primarily used for independent post-entry verification.
    """

    name = "finance_read_invoice_tool"
    description = (
        "Retrieve full details of an invoice from the internal finance application using either "
        "invoice_id or invoice_number. Returns all fields for verification."
    )
    input_schema = FinanceReadInvoiceInput
    risk_level = "LOW"

    def __init__(self, session_factory=None):
        self.session_factory = session_factory or get_session_factory()

    async def execute(self, params: FinanceReadInvoiceInput, session: Optional[Any] = None) -> ToolResult:
        if not params.invoice_id and not params.invoice_number:
            return ToolResult.fail("Either 'invoice_id' or 'invoice_number' must be provided.")

        if session is not None:
            return await self._fetch(session, params)

        async with self.session_factory() as sess:
            return await self._fetch(sess, params)

    async def _fetch(self, session: Any, params: FinanceReadInvoiceInput) -> ToolResult:
        invoice = None
        if params.invoice_id:
            invoice = await FinanceService.get_invoice(session, params.invoice_id)
        elif params.invoice_number:
            invoice = await FinanceService.get_invoice_by_number(session, params.invoice_number)

        if not invoice:
            identifier = params.invoice_id or params.invoice_number
            return ToolResult.fail(
                f"Invoice with identifier '{identifier}' was not found in the finance ledger.",
                metadata={"query_identifier": identifier},
            )

        invoice_dict = {
            "invoice_id": invoice.id,
            "invoice_number": invoice.invoice_number,
            "vendor_name": invoice.vendor_name,
            "amount": invoice.amount,
            "currency": invoice.currency,
            "issue_date": invoice.issue_date,
            "due_date": invoice.due_date,
            "status": invoice.status,
            "description": invoice.description,
            "created_at": invoice.created_at.isoformat(),
        }

        return ToolResult.ok(
            data=invoice_dict,
            evidence={
                "verified_in_ledger": True,
                "db_primary_key": invoice.id,
                "ledger_amount": invoice.amount,
                "ledger_due_date": invoice.due_date,
            },
            metadata={"invoice_number": invoice.invoice_number},
        )
