"""Tests for document text extraction, invoice field parsing, and chronological ranking."""

from pathlib import Path
import pytest
from pydantic import ValidationError

from app.documents.parser import (
    InvoiceData,
    extract_invoice_fields,
    extract_text,
    find_latest_invoice,
    scan_and_rank_invoices,
)
from app.documents.tool import DocumentTool

INVOICES_DIR = Path("data/invoices")


def test_sample_invoices_exist():
    """Ensure sample invoice PDFs were generated across required vendors."""
    assert INVOICES_DIR.exists()
    pdf_files = list(INVOICES_DIR.glob("*.pdf"))
    assert len(pdf_files) >= 9

    # Verify each required vendor is represented
    filenames = [f.name for f in pdf_files]
    assert any("acme" in f for f in filenames)
    assert any("globex" in f for f in filenames)
    assert any("initech" in f for f in filenames)
    assert any("umbrella" in f for f in filenames)


def test_extract_text_success():
    """Ensure extract_text returns non-empty plaintext from a valid PDF."""
    sample_pdf = INVOICES_DIR / "acme_invoice_2024_09.pdf"
    text = extract_text(sample_pdf)

    assert isinstance(text, str)
    assert "Acme Corp" in text
    assert "INV-ACM-2024-015" in text
    assert "12,450.50" in text
    assert "2024-09-28" in text


def test_extract_text_file_not_found():
    """Ensure extract_text raises FileNotFoundError when file is missing."""
    with pytest.raises(FileNotFoundError):
        extract_text("data/invoices/non_existent_invoice_9999.pdf")


def test_extract_text_invalid_extension(tmp_path):
    """Ensure extract_text raises ValueError on non-PDF file."""
    txt_file = tmp_path / "not_a_pdf.txt"
    txt_file.write_text("Hello world")
    with pytest.raises(ValueError, match="not a PDF document"):
        extract_text(txt_file)


def test_extract_invoice_fields_acme():
    """Verify structured field extraction for the latest Acme Corp invoice."""
    text = extract_text(INVOICES_DIR / "acme_invoice_2024_09.pdf")
    inv = extract_invoice_fields(text)

    assert isinstance(inv, InvoiceData)
    assert inv.vendor_name == "Acme Corp"
    assert inv.invoice_number == "INV-ACM-2024-015"
    assert inv.invoice_date == "2024-09-28"
    assert inv.due_date == "2024-10-28"
    assert inv.amount == 12450.50
    assert inv.currency == "USD"


def test_extract_invoice_fields_all_vendors():
    """Verify field extraction across Globex, Initech, and Umbrella invoices."""
    cases = [
        ("globex_invoice_2024_08.pdf", "Globex", "INV-GLX-2024-205", 8900.00, "2024-08-19"),
        ("initech_invoice_2024_07.pdf", "Initech", "INV-INI-2024-045", 2750.25, "2024-07-22"),
        ("umbrella_invoice_2024_10.pdf", "Umbrella", "INV-UMB-2024-420", 24500.00, "2024-10-02"),
    ]

    for filename, expected_vendor, expected_inv_num, expected_amount, expected_date in cases:
        file_path = INVOICES_DIR / filename
        text = extract_text(file_path)
        inv = extract_invoice_fields(text)

        assert expected_vendor.lower() in inv.vendor_name.lower()
        assert inv.invoice_number == expected_inv_num
        assert inv.amount == expected_amount
        assert inv.invoice_date == expected_date
        assert inv.currency == "USD"


def test_invoice_data_validation():
    """Ensure Pydantic validation catches invalid dates and non-positive amounts."""
    # Invalid amount <= 0
    with pytest.raises(ValidationError):
        InvoiceData(
            vendor_name="Acme Corp",
            invoice_number="INV-001",
            invoice_date="2024-01-15",
            due_date="2024-02-15",
            amount=-100.0,
            currency="USD",
        )

    # Invalid date format
    with pytest.raises(ValidationError):
        InvoiceData(
            vendor_name="Acme Corp",
            invoice_number="INV-001",
            invoice_date="01/15/2024",  # Not ISO YYYY-MM-DD
            due_date="2024-02-15",
            amount=100.0,
            currency="USD",
        )

    # Due date before invoice date
    with pytest.raises(ValidationError):
        InvoiceData(
            vendor_name="Acme Corp",
            invoice_number="INV-001",
            invoice_date="2024-05-15",
            due_date="2024-01-15",  # Earlier than issue date
            amount=100.0,
            currency="USD",
        )


def test_dynamic_latest_invoice_resolution():
    """
    CRITICAL REQUIREMENT:
    Ensure the system dynamically identifies the latest invoice based on invoice date,
    without hardcoding filenames or answers.
    """
    # Acme Corp has 3 invoices: 2024-01-15, 2024-06-12, 2024-09-28
    acme_latest = find_latest_invoice(INVOICES_DIR, "Acme Corp")
    assert acme_latest is not None
    path, inv = acme_latest
    assert inv.invoice_number == "INV-ACM-2024-015"
    assert inv.invoice_date == "2024-09-28"
    assert inv.amount == 12450.50
    assert path.name == "acme_invoice_2024_09.pdf"

    # Globex has 2 invoices: 2024-03-05, 2024-08-19
    globex_latest = find_latest_invoice(INVOICES_DIR, "Globex")
    assert globex_latest is not None
    path, inv = globex_latest
    assert inv.invoice_number == "INV-GLX-2024-205"
    assert inv.invoice_date == "2024-08-19"
    assert path.name == "globex_invoice_2024_08.pdf"

    # Umbrella has 2 invoices: 2024-04-18, 2024-10-02
    umbrella_latest = find_latest_invoice(INVOICES_DIR, "Umbrella")
    assert umbrella_latest is not None
    path, inv = umbrella_latest
    assert inv.invoice_number == "INV-UMB-2024-420"
    assert inv.invoice_date == "2024-10-02"
    assert path.name == "umbrella_invoice_2024_10.pdf"


def test_document_tool_class():
    """Ensure DocumentTool provides high-level convenience methods."""
    tool = DocumentTool(inbox_dir=INVOICES_DIR)

    # Test parse_invoice directly from path
    sample_file = INVOICES_DIR / "acme_invoice_2024_06.pdf"
    data = tool.parse_invoice(sample_file)
    assert data.invoice_number == "INV-ACM-2024-008"
    assert data.amount == 7850.00

    # Test find_latest_invoice via tool
    latest = tool.find_latest_invoice("Acme Corp")
    assert latest["invoice_data"]["invoice_number"] == "INV-ACM-2024-015"
    assert latest["invoice_data"]["amount"] == 12450.50
    assert "acme_invoice_2024_09.pdf" in latest["file_name"]

    # Test list_inbox_invoices
    all_invoices = tool.list_inbox_invoices()
    assert len(all_invoices) >= 9
    # Check that sorting is strictly descending by date
    dates = [item["invoice_data"]["invoice_date"] for item in all_invoices]
    assert dates == sorted(dates, reverse=True)
