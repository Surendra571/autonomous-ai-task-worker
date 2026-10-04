"""End-to-end browser automation test for the internal Finance application using Playwright."""

import asyncio
import pytest
import uvicorn
from playwright.async_api import async_playwright

from app.main import app
from app.config.settings import settings


@pytest.fixture
async def live_server():
    """Runs a live Uvicorn HTTP server as an asyncio task in the same event loop."""
    port = 8765
    config = uvicorn.Config(
        app=app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())

    while not server.started:
        await asyncio.sleep(0.05)

    base_url = f"http://127.0.0.1:{port}"
    try:
        yield base_url
    finally:
        server.should_exit = True
        await task


@pytest.mark.asyncio
async def test_browser_full_finance_workflow(live_server):
    """
    Tests the complete end-to-end browser workflow required for the Autonomous AI Task Worker:
    1. Login with test credentials
    2. Navigate to invoices
    3. Search invoices
    4. Open an invoice
    5. Create a new invoice
    6. Submit it
    7. View the created invoice
    8. Verify its fields
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=settings.BROWSER_HEADLESS,
        )
        context = await browser.new_context()
        page = await context.new_page()

        # Step 1: Login
        await page.goto(f"{live_server}/finance/login")
        await page.wait_for_selector('[data-testid="login-form"]')

        await page.fill('[data-testid="login-username"]', settings.FINANCE_APP_USERNAME)
        await page.fill('[data-testid="login-password"]', settings.FINANCE_APP_PASSWORD)
        await page.click('[data-testid="login-submit"]')

        # Verify landing on Dashboard
        await page.wait_for_selector('[data-testid="dashboard-summary"]')
        dashboard_title = await page.text_content('[data-testid="dashboard-title"]')
        assert "Accounts Payable Dashboard" in dashboard_title

        # Step 2: Navigate to Invoices
        await page.click('[data-testid="nav-invoices"]')
        await page.wait_for_selector('[data-testid="invoices-table"]')
        assert await page.is_visible('[data-testid="invoices-table"]')

        # Step 3: Search invoices for "Acme"
        await page.fill('[data-testid="search-input"]', "Acme Corp")
        await page.click('[data-testid="search-button"]')
        await page.wait_for_selector('[data-testid="invoice-row-INV-ACM-2024-001"]')

        # Step 4: Open an existing invoice and inspect
        await page.click('[data-testid="view-invoice-INV-ACM-2024-001"]')
        await page.wait_for_selector('[data-testid="invoice-detail-card"]')
        detail_inv_num = await page.text_content('[data-testid="detail-invoice-number"]')
        assert detail_inv_num.strip() == "INV-ACM-2024-001"

        # Step 5: Navigate to Create Invoice page
        await page.click('[data-testid="nav-create-invoice"]')
        await page.wait_for_selector('[data-testid="create-invoice-form"]')

        test_invoice_num = "INV-ACM-2024-AUTONOMOUS-01"
        test_vendor = "Acme Corp"
        test_amount = "18750.25"
        test_issue_date = "2024-10-01"
        test_due_date = "2024-11-01"
        test_description = "Autonomous AI Task Worker browser automated entry"

        await page.fill('[data-testid="input-invoice-number"]', test_invoice_num)
        await page.fill('[data-testid="input-vendor-name"]', test_vendor)
        await page.fill('[data-testid="input-amount"]', test_amount)
        await page.fill('[data-testid="input-issue-date"]', test_issue_date)
        await page.fill('[data-testid="input-due-date"]', test_due_date)
        await page.fill('[data-testid="input-description"]', test_description)

        # Step 6: Submit it
        await page.click('[data-testid="submit-invoice-btn"]')

        # Step 7: View the created invoice
        await page.wait_for_selector('[data-testid="invoice-detail-card"]')
        await page.wait_for_selector('[data-testid="invoice-saved-success-banner"]')

        # Step 8: Verify its fields independently on the rendered DOM
        created_num = await page.text_content('[data-testid="detail-invoice-number"]')
        created_vendor = await page.text_content('[data-testid="detail-vendor-name"]')
        created_amount = await page.text_content('[data-testid="detail-amount"]')
        created_due_date = await page.text_content('[data-testid="detail-due-date"]')
        created_desc = await page.text_content('[data-testid="detail-description"]')

        assert created_num.strip() == test_invoice_num
        assert created_vendor.strip() == test_vendor
        assert test_amount in created_amount
        assert created_due_date.strip() == test_due_date
        assert test_description in created_desc

        await context.close()
        await browser.close()
