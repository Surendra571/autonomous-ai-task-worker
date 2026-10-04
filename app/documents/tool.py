"""Document Tool interface for the autonomous task worker."""

from pathlib import Path
from typing import Any, Dict, List, Optional
from app.documents.parser import (
    InvoiceData,
    extract_invoice_fields,
    extract_text,
    find_latest_invoice,
    scan_and_rank_invoices,
)
from app.config.settings import settings
from app.config.logging import get_logger

logger = get_logger("documents.tool")


class DocumentTool:
    """Provides high-level document and invoice operations to the agent."""

    def __init__(self, inbox_dir: Optional[Path] = None):
        self.inbox_dir = Path(inbox_dir or settings.INBOX_DIR)

    def extract_text(self, file_path: str | Path) -> str:
        """Extracts text content from a PDF file."""
        return extract_text(file_path)

    def extract_invoice_fields(self, text: str) -> InvoiceData:
        """Extracts structured and validated invoice fields from document text."""
        return extract_invoice_fields(text)

    def parse_invoice(self, file_path: str | Path) -> InvoiceData:
        """Extracts and validates invoice fields directly from a PDF file."""
        text = self.extract_text(file_path)
        return self.extract_invoice_fields(text)

    def find_latest_invoice(self, vendor_name: str, directory: Optional[Path] = None) -> Dict[str, Any]:
        """
        Dynamically discovers the latest invoice for a vendor based on chronological invoice_date.
        Does not hardcode filenames or dates.
        """
        target_dir = directory or self.inbox_dir
        result = find_latest_invoice(target_dir, vendor_name)
        if not result:
            raise FileNotFoundError(f"No invoices found for vendor '{vendor_name}' in {target_dir}")

        path, inv_data = result
        return {
            "file_path": str(path.resolve()),
            "file_name": path.name,
            "invoice_data": inv_data.model_dump(),
        }

    def list_inbox_invoices(self, vendor_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lists all invoice files in the inbox ordered chronologically."""
        ranked = scan_and_rank_invoices(self.inbox_dir, vendor_filter=vendor_filter)
        return [
            {
                "file_path": str(path.resolve()),
                "file_name": path.name,
                "invoice_data": inv.model_dump(),
            }
            for path, inv in ranked
        ]
