"""Finance Search Tool for querying the internal finance ledger."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session_factory
from app.finance.service import FinanceService
from app.tools.base import BaseTool, ToolResult


class FinanceSearchInput(BaseModel):
    """Input parameters for searching invoices in the internal finance system."""
    query: Optional[str] = Field(None, description="Text to search across vendor names, invoice numbers, or descriptions.")
    status: Optional[str] = Field(None, description="Filter by status: 'PENDING', 'PAID', 'OVERDUE', 'REJECTED', or 'ALL'.")
    limit: Optional[int] = Field(default=20, ge=1, le=100, description="Maximum number of records to return.")


class FinanceSearchTool(BaseTool):
    """
    Queries the internal finance database to search for existing vendor invoice records.
    Useful for checking whether an invoice is already registered.
    """

    name = "finance_search_tool"
    description = (
        "Search recorded vendor invoices in the internal finance system by query text or status. "
        "Returns a list of matching invoice records."
    )
    input_schema = FinanceSearchInput
    risk_level = "LOW"

    def __init__(self, session_factory=None):
        self.session_factory = session_factory or get_session_factory()

    async def execute(self, params: FinanceSearchInput, session: Optional[Any] = None) -> ToolResult:
        if session is not None:
            return await self._search(session, params)

        async with self.session_factory() as sess:
            return await self._search(sess, params)

    async def _search(self, session: Any, params: FinanceSearchInput) -> ToolResult:
        invoices = await FinanceService.list_invoices(
            db=session,
            query=params.query,
            status=params.status,
            limit=params.limit or 20,
        )

        records = [
            {
                "invoice_id": inv.id,
                "invoice_number": inv.invoice_number,
                "vendor_name": inv.vendor_name,
                "amount": inv.amount,
                "currency": inv.currency,
                "issue_date": inv.issue_date,
                "due_date": inv.due_date,
                "status": inv.status,
                "description": inv.description,
            }
            for inv in invoices
        ]

        return ToolResult.ok(
            data={"total_count": len(records), "invoices": records},
            evidence={"matching_invoice_numbers": [r["invoice_number"] for r in records]},
            metadata={"query": params.query, "status": params.status},
        )
