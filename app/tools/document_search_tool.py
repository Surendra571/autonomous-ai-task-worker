"""Document Search Tool for discovering invoice PDFs in the document repository."""

from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.config.settings import settings
from app.documents.parser import scan_and_rank_invoices
from app.tools.base import BaseTool, ToolResult


class DocumentSearchInput(BaseModel):
    """Input parameters for searching invoice documents."""
    vendor: Optional[str] = Field(None, description="Optional vendor name filter (e.g. 'Acme Corp', 'Globex').")
    vendor_name: Optional[str] = Field(None, description="Alternative alias for vendor name filter.")
    query: Optional[str] = Field(None, description="Optional text query to match against invoice data.")
    directory: Optional[str] = Field(None, description="Optional directory path; defaults to configured inbox.")

    @property
    def target_vendor(self) -> Optional[str]:
        return self.vendor or self.vendor_name


class DocumentSearchTool(BaseTool):
    """
    Searches the inbox directory for invoice PDF documents.
    Extracts high-level metadata and returns files ranked chronologically (newest first).
    Allows the agent to dynamically identify documents without hardcoding.
    """

    name = "document_search_tool"
    description = (
        "Search the inbox for invoice PDFs. Can filter by vendor name. "
        "Returns a list of matching files sorted by invoice date (newest first)."
    )
    input_schema = DocumentSearchInput
    risk_level = "LOW"

    def __init__(self, inbox_dir: Optional[Path] = None):
        self.inbox_dir = Path(inbox_dir or settings.INBOX_DIR)

    async def execute(self, params: DocumentSearchInput) -> ToolResult:
        target_dir = Path(params.directory) if params.directory else self.inbox_dir
        if not target_dir.exists():
            return ToolResult.fail(f"Invoice directory '{target_dir}' does not exist.")

        ranked = scan_and_rank_invoices(target_dir, vendor_filter=params.target_vendor)

        # Apply secondary text query filter if provided
        if params.query:
            q_lower = params.query.strip().lower()
            ranked = [
                (path, inv)
                for path, inv in ranked
                if q_lower in path.name.lower()
                or q_lower in inv.vendor_name.lower()
                or q_lower in inv.invoice_number.lower()
            ]

        results = [
            {
                "file_path": str(path.resolve()),
                "file_name": path.name,
                "vendor_name": inv.vendor_name,
                "invoice_number": inv.invoice_number,
                "invoice_date": inv.invoice_date,
                "due_date": inv.due_date,
                "amount": inv.amount,
                "currency": inv.currency,
            }
            for path, inv in ranked
        ]

        latest_match = results[0] if results else None

        return ToolResult.ok(
            data={
                "total_found": len(results),
                "invoices": results,
                "latest_invoice": latest_match,
            },
            evidence={
                "scanned_directory": str(target_dir.resolve()),
                "files_matched": [r["file_name"] for r in results],
            },
            metadata={"vendor_filter": params.vendor, "query": params.query},
        )
