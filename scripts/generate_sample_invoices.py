"""Generates realistic sample invoice PDFs for testing document extraction and the autonomous task worker."""

import os
from pathlib import Path
from typing import Any, Dict, List
import pymupdf

OUTPUT_DIR = Path("data/invoices")

INVOICE_DATASETS: List[Dict[str, Any]] = [
    # Acme Corp Invoices (Different dates)
    {
        "filename": "acme_invoice_2024_01.pdf",
        "vendor_name": "Acme Corp",
        "vendor_address": "100 Industrial Parkway, Sector 4, Silicon Valley, CA 94025",
        "invoice_number": "INV-ACM-2024-001",
        "invoice_date": "2024-01-15",
        "due_date": "2024-02-15",
        "currency": "USD",
        "items": [
            ("Cloud Virtual Dedicated Servers (Q1)", 2, 1800.00, 3600.00),
            ("Tier-1 24/7 SLA Incident Support", 1, 900.00, 900.00),
        ],
        "total_amount": 4500.00,
    },
    {
        "filename": "acme_invoice_2024_06.pdf",
        "vendor_name": "Acme Corp",
        "vendor_address": "100 Industrial Parkway, Sector 4, Silicon Valley, CA 94025",
        "invoice_number": "INV-ACM-2024-008",
        "invoice_date": "2024-06-12",
        "due_date": "2024-07-12",
        "currency": "USD",
        "items": [
            ("Database Migration & Replication Services", 1, 5500.00, 5500.00),
            ("Read-Replica High-Availability Node", 1, 2350.00, 2350.00),
        ],
        "total_amount": 7850.00,
    },
    {
        "filename": "acme_invoice_2024_09.pdf",
        "vendor_name": "Acme Corp",
        "vendor_address": "100 Industrial Parkway, Sector 4, Silicon Valley, CA 94025",
        "invoice_number": "INV-ACM-2024-015",
        "invoice_date": "2024-09-28",
        "due_date": "2024-10-28",
        "currency": "USD",
        "items": [
            ("AI Inference GPU Cluster Provisioning (H100)", 1, 9500.00, 9500.00),
            ("Secure VPC Direct Connect Interconnect", 1, 1850.50, 1850.50),
            ("Enterprise Backup Vault (20TB)", 1, 1100.00, 1100.00),
        ],
        "total_amount": 12450.50,
    },

    # Globex Invoices
    {
        "filename": "globex_invoice_2024_03.pdf",
        "vendor_name": "Globex",
        "vendor_address": "842 Global Way, Suite 10, New York, NY 10001",
        "invoice_number": "INV-GLX-2024-101",
        "invoice_date": "2024-03-05",
        "due_date": "2024-04-05",
        "currency": "USD",
        "items": [
            ("Automated Conveyor Sensor Package", 4, 600.00, 2400.00),
            ("Factory Floor Telemetry Hub", 1, 800.00, 800.00),
        ],
        "total_amount": 3200.00,
    },
    {
        "filename": "globex_invoice_2024_08.pdf",
        "vendor_name": "Globex",
        "vendor_address": "842 Global Way, Suite 10, New York, NY 10001",
        "invoice_number": "INV-GLX-2024-205",
        "invoice_date": "2024-08-19",
        "due_date": "2024-09-19",
        "currency": "USD",
        "items": [
            ("Robotic Arm Actuator Assembly", 2, 3800.00, 7600.00),
            ("Precision Calibration Service", 1, 1300.00, 1300.00),
        ],
        "total_amount": 8900.00,
    },

    # Initech Invoices
    {
        "filename": "initech_invoice_2024_02.pdf",
        "vendor_name": "Initech",
        "vendor_address": "4120 Freemont Blvd, Austin, TX 78701",
        "invoice_number": "INV-INI-2024-011",
        "invoice_date": "2024-02-10",
        "due_date": "2024-03-10",
        "currency": "USD",
        "items": [
            ("TPS Report Automation Scripting", 1, 1500.00, 1500.00),
        ],
        "total_amount": 1500.00,
    },
    {
        "filename": "initech_invoice_2024_07.pdf",
        "vendor_name": "Initech",
        "vendor_address": "4120 Freemont Blvd, Austin, TX 78701",
        "invoice_number": "INV-INI-2024-045",
        "invoice_date": "2024-07-22",
        "due_date": "2024-08-22",
        "currency": "USD",
        "items": [
            ("Legacy Payroll System Integration", 1, 2200.00, 2200.00),
            ("Data Migration Validation QA", 1, 550.25, 550.25),
        ],
        "total_amount": 2750.25,
    },

    # Umbrella Invoices
    {
        "filename": "umbrella_invoice_2024_04.pdf",
        "vendor_name": "Umbrella",
        "vendor_address": "Raccoon City Research Park, Unit 8, Arklay County, CO 80439",
        "invoice_number": "INV-UMB-2024-301",
        "invoice_date": "2024-04-18",
        "due_date": "2024-05-18",
        "currency": "USD",
        "items": [
            ("Biological Safety Containment Suite Q2", 1, 14000.00, 14000.00),
            ("Air Filtration & Cryo-cooling Inspection", 1, 4200.00, 4200.00),
        ],
        "total_amount": 18200.00,
    },
    {
        "filename": "umbrella_invoice_2024_10.pdf",
        "vendor_name": "Umbrella",
        "vendor_address": "Raccoon City Research Park, Unit 8, Arklay County, CO 80439",
        "invoice_number": "INV-UMB-2024-420",
        "invoice_date": "2024-10-02",
        "due_date": "2024-11-02",
        "currency": "USD",
        "items": [
            ("Viral Sequencing Telemetry License (Yearly)", 1, 19500.00, 19500.00),
            ("Hazardous Material Transport Protocol (x5)", 5, 1000.00, 5000.00),
        ],
        "total_amount": 24500.00,
    },
]


def create_invoice_pdf(data: Dict[str, Any], output_path: Path) -> None:
    """Generates a professional-looking invoice PDF using PyMuPDF."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)  # Standard A4: 595 x 842 points

    # Header section
    page.draw_rect(pymupdf.Rect(0, 0, 595, 80), color=(0.12, 0.25, 0.69), fill=(0.12, 0.25, 0.69))
    page.insert_text(pymupdf.Point(40, 50), "INVOICE", fontsize=28, color=(1, 1, 1), fontname="helv")

    # Vendor Details
    y = 110
    page.insert_text(pymupdf.Point(40, y), data["vendor_name"], fontsize=14, fontname="helv", color=(0.1, 0.1, 0.1))
    page.insert_text(pymupdf.Point(40, y + 16), data["vendor_address"], fontsize=9, fontname="helv", color=(0.4, 0.4, 0.4))
    page.insert_text(pymupdf.Point(40, y + 30), "Contact: billing@" + data["vendor_name"].lower().replace(" ", "") + ".com", fontsize=9, fontname="helv", color=(0.4, 0.4, 0.4))

    # Meta Details Box (Top Right)
    box_x = 340
    page.draw_rect(pymupdf.Rect(box_x, 95, 555, 175), color=(0.85, 0.88, 0.94), fill=(0.96, 0.97, 1.0))
    page.insert_text(pymupdf.Point(box_x + 15, 115), f"Invoice Number: {data['invoice_number']}", fontsize=10, fontname="helv")
    page.insert_text(pymupdf.Point(box_x + 15, 132), f"Invoice Date: {data['invoice_date']}", fontsize=10, fontname="helv")
    page.insert_text(pymupdf.Point(box_x + 15, 149), f"Due Date: {data['due_date']}", fontsize=10, fontname="helv")
    page.insert_text(pymupdf.Point(box_x + 15, 166), f"Currency: {data['currency']}", fontsize=10, fontname="helv")

    # Bill To Section
    y = 195
    page.insert_text(pymupdf.Point(40, y), "BILL TO:", fontsize=10, fontname="helv", color=(0.3, 0.3, 0.3))
    page.insert_text(pymupdf.Point(40, y + 16), "Internal Enterprise Finance Operations", fontsize=11, fontname="helv")
    page.insert_text(pymupdf.Point(40, y + 30), "Accounts Payable Department - Suite 500", fontsize=9, fontname="helv", color=(0.4, 0.4, 0.4))
    page.insert_text(pymupdf.Point(40, y + 44), "financial-ops@enterprise-corp.internal", fontsize=9, fontname="helv", color=(0.4, 0.4, 0.4))

    # Table Header
    table_top = 270
    page.draw_rect(pymupdf.Rect(40, table_top, 555, table_top + 24), color=(0.2, 0.25, 0.35), fill=(0.2, 0.25, 0.35))
    page.insert_text(pymupdf.Point(50, table_top + 16), "Item Description", fontsize=9, fontname="helv", color=(1, 1, 1))
    page.insert_text(pymupdf.Point(340, table_top + 16), "Qty", fontsize=9, fontname="helv", color=(1, 1, 1))
    page.insert_text(pymupdf.Point(400, table_top + 16), "Unit Price", fontsize=9, fontname="helv", color=(1, 1, 1))
    page.insert_text(pymupdf.Point(490, table_top + 16), "Total", fontsize=9, fontname="helv", color=(1, 1, 1))

    # Line Items
    curr_y = table_top + 42
    for desc, qty, unit_price, line_total in data["items"]:
        page.insert_text(pymupdf.Point(50, curr_y), desc, fontsize=9, fontname="helv")
        page.insert_text(pymupdf.Point(345, curr_y), str(qty), fontsize=9, fontname="helv")
        page.insert_text(pymupdf.Point(400, curr_y), f"${unit_price:,.2f}", fontsize=9, fontname="helv")
        page.insert_text(pymupdf.Point(490, curr_y), f"${line_total:,.2f}", fontsize=9, fontname="helv")

        # Divider line
        page.draw_line(pymupdf.Point(40, curr_y + 8), pymupdf.Point(555, curr_y + 8), color=(0.88, 0.88, 0.88))
        curr_y += 28

    # Summary Totals Box
    total_y = curr_y + 20
    page.draw_rect(pymupdf.Rect(340, total_y, 555, total_y + 60), color=(0.12, 0.25, 0.69), fill=(0.95, 0.97, 1.0))
    page.insert_text(pymupdf.Point(355, total_y + 22), "TOTAL DUE:", fontsize=11, fontname="helv", color=(0.1, 0.1, 0.1))
    page.insert_text(pymupdf.Point(450, total_y + 24), f"${data['total_amount']:,.2f} {data['currency']}", fontsize=14, fontname="helv", color=(0.1, 0.35, 0.15))
    page.insert_text(pymupdf.Point(355, total_y + 44), f"Payment due on or before: {data['due_date']}", fontsize=8, fontname="helv", color=(0.4, 0.4, 0.4))

    # Footer note
    page.draw_line(pymupdf.Point(40, 780), pymupdf.Point(555, 780), color=(0.85, 0.85, 0.85))
    page.insert_text(
        pymupdf.Point(40, 800),
        "Payment Terms: Net 30 days. Remit payments via ACH / Wire Transfer referencing the Invoice Number.",
        fontsize=8,
        fontname="helv",
        color=(0.5, 0.5, 0.5),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    doc.close()
    print(f"Generated: {output_path.name} ({data['vendor_name']} - {data['invoice_date']})")


def generate_all_invoices(target_dir: Path = OUTPUT_DIR) -> List[Path]:
    """Generates all sample invoices in target directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []
    for item in INVOICE_DATASETS:
        file_path = target_dir / item["filename"]
        create_invoice_pdf(item, file_path)
        generated_files.append(file_path)
    return generated_files


if __name__ == "__main__":
    files = generate_all_invoices()
    print(f"\nSuccessfully generated {len(files)} realistic sample invoice PDFs in '{OUTPUT_DIR}'.")
