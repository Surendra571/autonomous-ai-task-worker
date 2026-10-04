"""Database seeder with realistic sample invoices for the simulated finance system."""

from typing import List
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.invoice import Invoice
from app.config.logging import get_logger

logger = get_logger("finance.seed")

SAMPLE_INVOICES = [
    {
        "invoice_number": "INV-ACM-2024-001",
        "vendor_name": "Acme Corp",
        "amount": 4500.00,
        "currency": "USD",
        "issue_date": "2024-01-15",
        "due_date": "2024-02-15",
        "status": "PAID",
        "description": "Q1 Enterprise cloud hosting and infrastructure maintenance",
    },
    {
        "invoice_number": "INV-ACM-2024-002",
        "vendor_name": "Acme Corp",
        "amount": 8750.50,
        "currency": "USD",
        "issue_date": "2024-04-10",
        "due_date": "2024-05-10",
        "status": "PAID",
        "description": "Q2 Managed database clustering and SLA support tier 1",
    },
    {
        "invoice_number": "INV-GLX-2024-089",
        "vendor_name": "Globex Corporation",
        "amount": 12400.00,
        "currency": "USD",
        "issue_date": "2024-05-01",
        "due_date": "2024-06-01",
        "status": "PAID",
        "description": "Industrial automated conveyor robotics hardware inspection",
    },
    {
        "invoice_number": "INV-WYN-2024-312",
        "vendor_name": "Wayne Enterprises",
        "amount": 28000.00,
        "currency": "USD",
        "issue_date": "2024-06-15",
        "due_date": "2024-07-15",
        "status": "PENDING",
        "description": "Annual enterprise penetration testing and zero-trust audit",
    },
    {
        "invoice_number": "INV-STK-2024-904",
        "vendor_name": "Stark Industries",
        "amount": 15200.75,
        "currency": "USD",
        "issue_date": "2024-07-01",
        "due_date": "2024-08-01",
        "status": "PENDING",
        "description": "High-density neural network accelerator modules (x4)",
    },
    {
        "invoice_number": "INV-CYB-2024-118",
        "vendor_name": "Cyberdyne Systems",
        "amount": 6300.00,
        "currency": "USD",
        "issue_date": "2024-03-20",
        "due_date": "2024-04-20",
        "status": "OVERDUE",
        "description": "Firmware security patch licensing and neural coprocessor telemetry",
    },
]


async def seed_invoices(db: AsyncSession) -> int:
    """Seeds the database with sample invoices if currently empty."""
    stmt = select(func.count(Invoice.id))
    count = (await db.execute(stmt)).scalar() or 0

    if count > 0:
        logger.info("Finance database already contains invoices, skipping seeding", count=count)
        return count

    logger.info("Seeding finance database with realistic vendor invoices...")
    seeded_count = 0
    for item in SAMPLE_INVOICES:
        inv = Invoice(
            invoice_number=item["invoice_number"],
            vendor_name=item["vendor_name"],
            amount=item["amount"],
            currency=item["currency"],
            issue_date=item["issue_date"],
            due_date=item["due_date"],
            status=item["status"],
            description=item["description"],
        )
        db.add(inv)
        seeded_count += 1

    await db.commit()
    logger.info("Successfully seeded finance invoices", count=seeded_count)
    return seeded_count
