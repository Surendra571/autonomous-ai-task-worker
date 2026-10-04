"""Document Read Tool for extracting full text and structured fields from PDF invoices."""

from pathlib import Path
from typing import Any, Dict
from pydantic import BaseModel, Field

from app.documents.parser import extract_invoice_fields, extract_text
from app.tools.base import BaseTool, ToolResult


class DocumentReadInput(BaseModel):
    """Input parameters for reading and parsing an invoice PDF."""
    file_path: str = Field(..., description="Absolute or relative path to the invoice PDF document to read.")


class DocumentReadTool(BaseTool):
    """
    Extracts text and parses structured invoice fields (invoice_number, amount, dates, vendor)
    from a specified invoice PDF using PyMuPDF and Pydantic validation.
    """

    name = "document_read_tool"
    description = (
        "Read and extract structured fields from an invoice PDF. "
        "Returns vendor_name, invoice_number, invoice_date, due_date, amount, and currency."
    )
    input_schema = DocumentReadInput
    risk_level = "LOW"

    async def execute(self, params: DocumentReadInput) -> ToolResult:
        file_path = Path(params.file_path)
        if not file_path.is_file():
            return ToolResult.fail(f"File not found: '{params.file_path}'")

        try:
            raw_text = extract_text(file_path)
            invoice_data = extract_invoice_fields(raw_text)

            return ToolResult.ok(
                data={
                    "file_path": str(file_path.resolve()),
                    "file_name": file_path.name,
                    "invoice_data": invoice_data.model_dump(),
                },
                evidence={
                    "raw_text_excerpt": raw_text[:300] + ("..." if len(raw_text) > 300 else ""),
                    "file_size_bytes": file_path.stat().st_size,
                    "fields_extracted": list(invoice_data.model_dump().keys()),
                },
                metadata={"file_path": str(file_path)},
            )
        except Exception as e:
            return ToolResult.fail(
                f"Failed to parse invoice document '{file_path.name}': {e}",
                metadata={"file_path": str(file_path)},
            )
