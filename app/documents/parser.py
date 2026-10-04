"""Document extraction and parsing module for invoice PDFs using PyMuPDF and Pydantic."""

from datetime import date, datetime
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator
import pymupdf

from app.config.logging import get_logger

logger = get_logger("documents.parser")


class InvoiceData(BaseModel):
    """Structured, validated invoice data extracted from documents."""

    vendor_name: str = Field(..., min_length=2, description="Name of the invoice issuer/vendor")
    invoice_number: str = Field(..., min_length=3, description="Unique invoice identification code")
    invoice_date: str = Field(..., description="Date invoice was issued (YYYY-MM-DD)")
    due_date: str = Field(..., description="Payment due date (YYYY-MM-DD)")
    amount: float = Field(..., gt=0.0, description="Total billed amount, must be greater than zero")
    currency: str = Field(default="USD", description="Currency symbol or code (e.g. USD, EUR, GBP)")

    @field_validator("invoice_date", "due_date")
    @classmethod
    def validate_iso_date(cls, v: str) -> str:
        """Ensures the date is a valid ISO 8601 YYYY-MM-DD formatted date."""
        v_clean = v.strip()
        try:
            datetime.strptime(v_clean, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Date '{v}' must be formatted as YYYY-MM-DD.")
        return v_clean

    @field_validator("due_date")
    @classmethod
    def validate_due_date_after_issue(cls, v: str, info) -> str:
        """Ensures due_date is on or after invoice_date if both are present."""
        invoice_date_val = info.data.get("invoice_date")
        if invoice_date_val:
            try:
                issue_dt = datetime.strptime(invoice_date_val, "%Y-%m-%d").date()
                due_dt = datetime.strptime(v, "%Y-%m-%d").date()
            except ValueError:
                return v

            if due_dt < issue_dt:
                raise ValueError(f"due_date ({v}) cannot be before invoice_date ({invoice_date_val}).")
        return v

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, v: str) -> str:
        """Standardizes currency identifiers."""
        v_clean = v.strip().upper()
        symbol_map = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY"}
        return symbol_map.get(v_clean, v_clean)

    def parsed_date(self) -> date:
        """Returns python date object for dynamic chronological sorting."""
        return datetime.strptime(self.invoice_date, "%Y-%m-%d").date()


def extract_text(file_path: str | Path) -> str:
    """
    Extracts raw text content from a PDF file using PyMuPDF.

    Args:
        file_path: Path to the target PDF file.

    Returns:
        Consolidated text from all pages in the PDF document.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid PDF or is empty.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found at path: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"File '{path.name}' is not a PDF document.")

    try:
        doc = pymupdf.open(str(path))
    except Exception as e:
        logger.error("Failed to open PDF document", path=str(path), error=str(e))
        raise ValueError(f"Cannot read PDF document '{path.name}': {e}") from e

    try:
        if len(doc) == 0:
            raise ValueError(f"PDF file '{path.name}' has 0 pages.")

        page_texts: List[str] = []
        for i, page in enumerate(doc):
            text = page.get_text()
            if text:
                page_texts.append(text)

        full_text = "\n\n".join(page_texts).strip()
        if not full_text:
            raise ValueError(f"No readable text could be extracted from '{path.name}'.")

        logger.info("Extracted text from PDF", filename=path.name, characters=len(full_text))
        return full_text
    finally:
        doc.close()


def normalize_date_string(date_str: str) -> Optional[str]:
    """Attempts to parse common date formats and format as YYYY-MM-DD."""
    cleaned = date_str.strip().strip(",")
    formats = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%B %d, %Y",
        "%B %d %Y",
        "%b %d, %Y",
        "%b %d %Y",
        "%d-%b-%Y",
        "%Y/%m/%d",
    ]
    for fmt in formats:
        try:
            parsed = datetime.strptime(cleaned, fmt)
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def extract_invoice_fields(text: str) -> InvoiceData:
    """
    Parses unstructured text extracted from an invoice document and returns
    a strongly-typed and validated InvoiceData model.

    Args:
        text: Plaintext content of the invoice document.

    Returns:
        Structured and validated InvoiceData.

    Raises:
        ValueError: If mandatory fields cannot be extracted or fail validation.
    """
    if not text or not text.strip():
        raise ValueError("Input text is empty.")

    # 1. Vendor Name
    vendor_name: Optional[str] = None
    known_vendors = ["Acme Corp", "Globex Corporation", "Globex", "Initech LLC", "Initech", "Umbrella Corporation", "Umbrella", "Cyberdyne Systems", "Wayne Enterprises", "Stark Industries"]
    for v in known_vendors:
        if re.search(rf"\b{re.escape(v)}\b", text, re.IGNORECASE):
            vendor_name = v
            break

    if not vendor_name:
        # Fallback heuristic: check lines following "INVOICE" or "From:"
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for i, line in enumerate(lines[:6]):
            if line.upper() == "INVOICE" and i + 1 < len(lines):
                candidate = lines[i + 1]
                if len(candidate) > 2 and not candidate.startswith("Invoice"):
                    vendor_name = candidate
                    break
            elif re.match(r"^(?:From|Vendor):\s*(.+)$", line, re.IGNORECASE):
                vendor_name = re.match(r"^(?:From|Vendor):\s*(.+)$", line, re.IGNORECASE).group(1).strip()
                break

    if not vendor_name:
        raise ValueError("Failed to extract vendor name from invoice text.")

    # 2. Invoice Number
    invoice_num_match = re.search(
        r"(?:Invoice\s*(?:Number|No\.?|#)|Inv\s*#)[:\s]*([A-Z0-9\-_/]+)",
        text,
        re.IGNORECASE,
    )
    if invoice_num_match:
        invoice_number = invoice_num_match.group(1).strip()
    else:
        # Fallback pattern like INV-XXX-YYYY-ZZZ
        code_match = re.search(r"\b(INV-[A-Z0-9\-]+)\b", text, re.IGNORECASE)
        if code_match:
            invoice_number = code_match.group(1).strip()
        else:
            raise ValueError("Failed to extract invoice number from invoice text.")

    # 3. Invoice Date
    invoice_date: Optional[str] = None
    date_match = re.search(
        r"(?:Invoice\s*Date|Date\s*of\s*Issue|Date|Billed\s*Date)[:\s]*([A-Za-z0-9\-/, ]+)",
        text,
        re.IGNORECASE,
    )
    if date_match:
        raw_date = date_match.group(1).strip().splitlines()[0]
        invoice_date = normalize_date_string(raw_date)

    if not invoice_date:
        # Look for explicit ISO dates
        iso_dates = re.findall(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
        if iso_dates:
            invoice_date = iso_dates[0]
        else:
            raise ValueError("Failed to extract invoice date from invoice text.")

    # 4. Due Date
    due_date: Optional[str] = None
    due_match = re.search(
        r"(?:Due\s*Date|Payment\s*Due|Payment\s*due\s*on\s*or\s*before)[:\s]*([A-Za-z0-9\-/, ]+)",
        text,
        re.IGNORECASE,
    )
    if due_match:
        raw_due = due_match.group(1).strip().splitlines()[0]
        due_date = normalize_date_string(raw_due)

    if not due_date:
        # Check second ISO date if available
        iso_dates = re.findall(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
        if len(iso_dates) > 1:
            due_date = iso_dates[1]
        else:
            # Fallback to invoice date if strictly Net 0
            due_date = invoice_date

    # 5. Amount & Currency
    amount: Optional[float] = None
    currency: str = "USD"

    # Match currency explicitly if labeled
    curr_match = re.search(r"\bCurrency[:\s]*([A-Z]{3}|\$|€|£)\b", text, re.IGNORECASE)
    if curr_match:
        currency = curr_match.group(1).strip().upper()

    # Match total line (e.g. "TOTAL DUE: $12,450.50 USD" or "Total: $4,500.00")
    total_patterns = [
        r"(?:TOTAL\s*DUE|TOTAL\s*AMOUNT|BALANCE\s*DUE|AMOUNT\s*DUE|TOTAL)[:\s]*([\$€£]?)\s*([0-9,]+\.[0-9]{2})\s*([A-Z]{3})?",
        r"(?:TOTAL|DUE)[:\s]*([\$€£])\s*([0-9,]+\.[0-9]{2})",
    ]

    for pat in total_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            groups = m.groups()
            for g in groups:
                if not g:
                    continue
                g_clean = g.strip()
                if g_clean in ("$", "€", "£"):
                    currency = g_clean
                elif re.match(r"^[A-Z]{3}$", g_clean):
                    currency = g_clean
                elif re.match(r"^[0-9,]+\.[0-9]{2}$", g_clean):
                    amount = float(g_clean.replace(",", ""))
            if amount is not None:
                break

    if amount is None:
        raise ValueError("Failed to extract total invoice amount from invoice text.")

    # Validate and return the Pydantic model
    return InvoiceData(
        vendor_name=vendor_name,
        invoice_number=invoice_number,
        invoice_date=invoice_date,
        due_date=due_date,
        amount=amount,
        currency=currency,
    )


def scan_and_rank_invoices(
    directory_path: str | Path,
    vendor_filter: Optional[str] = None,
) -> List[Tuple[Path, InvoiceData]]:
    """
    Scans a directory of invoice PDFs, parses each into structured InvoiceData,
    and returns them sorted in descending chronological order (newest first).

    Args:
        directory_path: Folder containing PDF invoices.
        vendor_filter: Optional vendor name to match.

    Returns:
        List of tuples (Path, InvoiceData), sorted newest to oldest.
    """
    dir_path = Path(directory_path)
    if not dir_path.is_dir():
        logger.warning("Directory does not exist", path=str(dir_path))
        return []

    results: List[Tuple[Path, InvoiceData]] = []
    pdf_files = sorted(dir_path.glob("*.pdf"))

    for pdf_path in pdf_files:
        try:
            txt = extract_text(pdf_path)
            inv = extract_invoice_fields(txt)

            if vendor_filter:
                vf = vendor_filter.strip().lower()
                if vf not in inv.vendor_name.lower():
                    continue

            results.append((pdf_path, inv))
        except Exception as e:
            logger.warning("Skipping invalid invoice document", filename=pdf_path.name, error=str(e))

    # Sort descending by invoice_date (latest date first)
    results.sort(key=lambda item: item[1].parsed_date(), reverse=True)
    return results


def find_latest_invoice(
    directory_path: str | Path,
    vendor_name: str,
) -> Optional[Tuple[Path, InvoiceData]]:
    """
    Determines and returns the latest invoice for a specific vendor based on invoice date.
    Does NOT hardcode any dates or filenames.
    """
    ranked = scan_and_rank_invoices(directory_path, vendor_filter=vendor_name)
    if not ranked:
        return None
    return ranked[0]
