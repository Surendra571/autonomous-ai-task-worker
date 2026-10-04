"""Tests for the simulated internal Finance application."""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.finance.service import FinanceService
from app.finance.seed import seed_invoices


@pytest.mark.asyncio
async def test_login_flow(async_client: AsyncClient):
    """Test login with valid and invalid credentials."""
    # Test invalid credentials
    invalid_res = await async_client.post(
        "/finance/login",
        data={"username": "wrong_user", "password": "wrong_password"},
        follow_redirects=False,
    )
    assert invalid_res.status_code == 401
    assert "Invalid username or password" in invalid_res.text

    # Test valid credentials
    valid_res = await async_client.post(
        "/finance/login",
        data={
            "username": settings.FINANCE_APP_USERNAME,
            "password": settings.FINANCE_APP_PASSWORD,
        },
        follow_redirects=False,
    )
    assert valid_res.status_code == 303
    assert valid_res.headers["location"] == "/finance/dashboard"
    assert settings.FINANCE_APP_COOKIE_NAME in valid_res.cookies


@pytest.mark.asyncio
async def test_unauthenticated_access_redirects(async_client: AsyncClient):
    """Ensure protected finance pages redirect to login for unauthenticated requests."""
    res = await async_client.get("/finance/dashboard", follow_redirects=False)
    assert res.status_code == 303
    assert res.headers["location"] == "/finance/login"

    res = await async_client.get("/finance/invoices", follow_redirects=False)
    assert res.status_code == 303
    assert res.headers["location"] == "/finance/login"


@pytest.mark.asyncio
async def test_dashboard_and_seeding(async_client: AsyncClient):
    """Ensure database seeding and dashboard KPI rendering."""
    # Log in
    login_res = await async_client.post(
        "/finance/login",
        data={
            "username": settings.FINANCE_APP_USERNAME,
            "password": settings.FINANCE_APP_PASSWORD,
        },
    )
    session_cookie = login_res.cookies.get(settings.FINANCE_APP_COOKIE_NAME)

    # Access dashboard
    res = await async_client.get(
        "/finance/dashboard",
        cookies={settings.FINANCE_APP_COOKIE_NAME: session_cookie},
    )
    assert res.status_code == 200
    assert "Accounts Payable Dashboard" in res.text
    assert "TOTAL INVOICES" in res.text


@pytest.mark.asyncio
async def test_create_and_retrieve_invoice(async_client: AsyncClient):
    """Test creating an invoice via HTML form submission and viewing its detail."""
    # Log in
    login_res = await async_client.post(
        "/finance/login",
        data={
            "username": settings.FINANCE_APP_USERNAME,
            "password": settings.FINANCE_APP_PASSWORD,
        },
    )
    session_cookie = login_res.cookies.get(settings.FINANCE_APP_COOKIE_NAME)

    # Submit new invoice form
    invoice_payload = {
        "invoice_number": "INV-ACM-2024-TEST-99",
        "vendor_name": "Acme Corp",
        "amount": "14250.75",
        "currency": "USD",
        "issue_date": "2024-10-01",
        "due_date": "2024-11-01",
        "status": "PENDING",
        "description": "Enterprise software platform maintenance agreement",
    }

    create_res = await async_client.post(
        "/finance/invoices",
        data=invoice_payload,
        cookies={settings.FINANCE_APP_COOKIE_NAME: session_cookie},
        follow_redirects=False,
    )
    assert create_res.status_code == 303
    redirect_url = create_res.headers["location"]
    assert "/finance/invoices/" in redirect_url

    # Follow redirect to detail page
    detail_res = await async_client.get(
        redirect_url,
        cookies={settings.FINANCE_APP_COOKIE_NAME: session_cookie},
    )
    assert detail_res.status_code == 200
    assert "INV-ACM-2024-TEST-99" in detail_res.text
    assert "Acme Corp" in detail_res.text
    assert "14250.75" in detail_res.text
    assert "2024-11-01" in detail_res.text


@pytest.mark.asyncio
async def test_search_and_filter_invoices(async_client: AsyncClient):
    """Test searching invoices by vendor name and filtering by status."""
    login_res = await async_client.post(
        "/finance/login",
        data={
            "username": settings.FINANCE_APP_USERNAME,
            "password": settings.FINANCE_APP_PASSWORD,
        },
    )
    session_cookie = login_res.cookies.get(settings.FINANCE_APP_COOKIE_NAME)

    # Search for Acme Corp
    res = await async_client.get(
        "/finance/invoices?q=Acme",
        cookies={settings.FINANCE_APP_COOKIE_NAME: session_cookie},
    )
    assert res.status_code == 200
    assert "Acme Corp" in res.text

    # Search for nonexistent vendor
    empty_res = await async_client.get(
        "/finance/invoices?q=NonExistentVendor999",
        cookies={settings.FINANCE_APP_COOKIE_NAME: session_cookie},
    )
    assert empty_res.status_code == 200
    assert "No vendor invoices matched your search criteria." in empty_res.text


@pytest.mark.asyncio
async def test_api_endpoints_for_verification(async_client: AsyncClient):
    """Test REST API endpoints used for programmatic independent verification."""
    # Create an invoice via REST API
    payload = {
        "invoice_number": "INV-API-2024-888",
        "vendor_name": "Wayne Enterprises",
        "amount": 9500.0,
        "currency": "USD",
        "issue_date": "2024-09-10",
        "due_date": "2024-10-10",
        "status": "APPROVED",
        "description": "Satellite link encryption firmware",
    }
    create_res = await async_client.post("/finance/api/invoices", json=payload)
    assert create_res.status_code == 201
    created_data = create_res.json()
    assert created_data["invoice_number"] == payload["invoice_number"]
    assert created_data["amount"] == 9500.0

    # Retrieve by ID
    get_res = await async_client.get(f"/finance/api/invoices/{created_data['id']}")
    assert get_res.status_code == 200
    assert get_res.json()["invoice_number"] == payload["invoice_number"]
