"""HTTP routes and HTML server-side views for the internal Finance application."""

from pathlib import Path
from typing import List, Optional
from fastapi import (
    APIRouter,
    Depends,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    status as http_status,
)
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.db.session import get_db
from app.finance.service import FinanceService
from app.models.schemas import InvoiceCreate, InvoiceResponse
from app.config.logging import get_logger

logger = get_logger("finance.routes")

router = APIRouter(prefix="/finance", tags=["Finance Application"])

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def get_current_user(request: Request) -> Optional[str]:
    """Helper to retrieve authenticated username from session cookie."""
    token = request.cookies.get(settings.FINANCE_APP_COOKIE_NAME)
    return FinanceService.validate_session_token(token)


def require_auth(request: Request) -> str:
    """Dependency that requires user authentication or raises 303 Redirect to login."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(
            status_code=http_status.HTTP_303_SEE_OTHER,
            headers={"Location": "/finance/login"},
        )
    return user


# ---------------------------------------------------------------------------
# Authentication Routes
# ---------------------------------------------------------------------------

@router.get("", response_class=HTMLResponse)
async def finance_root(request: Request):
    """Entrypoint redirecting to dashboard or login."""
    user = get_current_user(request)
    if user:
        return RedirectResponse(url="/finance/dashboard", status_code=http_status.HTTP_302_FOUND)
    return RedirectResponse(url="/finance/login", status_code=http_status.HTTP_302_FOUND)


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: Optional[str] = None):
    """Renders the login page."""
    user = get_current_user(request)
    if user:
        return RedirectResponse(url="/finance/dashboard", status_code=http_status.HTTP_302_FOUND)
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"error": error, "user": None},
    )


@router.post("/login", response_class=HTMLResponse)
async def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    """Handles login submission with deterministic credential validation."""
    if FinanceService.verify_credentials(username, password):
        token = FinanceService.generate_session_token(username.strip())
        response = RedirectResponse(url="/finance/dashboard", status_code=http_status.HTTP_303_SEE_OTHER)
        response.set_cookie(
            key=settings.FINANCE_APP_COOKIE_NAME,
            value=token,
            httponly=True,
            samesite="lax",
            max_age=86400,
        )
        logger.info("User logged in successfully", username=username)
        return response

    logger.warning("Failed login attempt", username=username)
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={
            "error": "Invalid username or password. Check test credentials below.",
            "user": None,
        },
        status_code=http_status.HTTP_401_UNAUTHORIZED,
    )


@router.get("/logout")
async def logout(request: Request):
    """Logs out the user by clearing the session cookie."""
    response = RedirectResponse(url="/finance/login", status_code=http_status.HTTP_303_SEE_OTHER)
    response.delete_cookie(key=settings.FINANCE_APP_COOKIE_NAME)
    return response


# ---------------------------------------------------------------------------
# Dashboard and Invoices HTML Views
# ---------------------------------------------------------------------------

@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_view(
    request: Request,
    user: str = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Renders the executive dashboard with key finance metrics."""
    metrics = await FinanceService.get_dashboard_metrics(db)
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={"metrics": metrics, "user": user},
    )


@router.get("/invoices", response_class=HTMLResponse)
async def invoices_list_view(
    request: Request,
    q: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    user: str = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Renders the invoices list with search and filtering capabilities."""
    invoices = await FinanceService.list_invoices(db=db, query=q, status=status)
    return templates.TemplateResponse(
        request=request,
        name="invoices.html",
        context={
            "invoices": invoices,
            "query": q,
            "current_status": status,
            "user": user,
        },
    )


@router.get("/invoices/new", response_class=HTMLResponse)
async def create_invoice_view(
    request: Request,
    user: str = Depends(require_auth),
):
    """Renders the form to create a new vendor invoice."""
    return templates.TemplateResponse(
        request=request,
        name="create_invoice.html",
        context={"user": user, "error": None, "form_data": None},
    )


@router.post("/invoices", response_class=HTMLResponse)
async def create_invoice_submit(
    request: Request,
    invoice_number: str = Form(...),
    vendor_name: str = Form(...),
    amount: float = Form(...),
    currency: str = Form("USD"),
    issue_date: str = Form(...),
    due_date: str = Form(...),
    invoice_status: str = Form("PENDING", alias="status"),
    description: Optional[str] = Form(None),
    user: str = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Handles HTML form submission for a new vendor invoice."""
    form_data = {
        "invoice_number": invoice_number,
        "vendor_name": vendor_name,
        "amount": amount,
        "currency": currency,
        "issue_date": issue_date,
        "due_date": due_date,
        "status": invoice_status,
        "description": description,
    }

    try:
        invoice_in = InvoiceCreate(**form_data)
        invoice = await FinanceService.create_invoice(db, invoice_in)
        return RedirectResponse(
            url=f"/finance/invoices/{invoice.id}?created=true",
            status_code=http_status.HTTP_303_SEE_OTHER,
        )
    except Exception as e:
        logger.warning("Error creating invoice via form", error=str(e))
        return templates.TemplateResponse(
            request=request,
            name="create_invoice.html",
            context={
                "user": user,
                "error": str(e),
                "form_data": form_data,
            },
            status_code=http_status.HTTP_400_BAD_REQUEST,
        )


@router.get("/invoices/{invoice_id}", response_class=HTMLResponse)
async def invoice_detail_view(
    invoice_id: str,
    request: Request,
    created: Optional[bool] = Query(False),
    user: str = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Renders the detailed view of a recorded vendor invoice."""
    invoice = await FinanceService.get_invoice(db, invoice_id)
    if not invoice:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Invoice '{invoice_id}' not found.",
        )

    return templates.TemplateResponse(
        request=request,
        name="invoice_detail.html",
        context={
            "invoice": invoice,
            "success_msg": created,
            "user": user,
        },
    )


# ---------------------------------------------------------------------------
# Programmatic REST API Endpoints (For flexible agent verification)
# ---------------------------------------------------------------------------

@router.get("/api/invoices", response_model=List[InvoiceResponse])
async def api_list_invoices(
    q: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """API endpoint to list invoices programmatically."""
    invoices = await FinanceService.list_invoices(db=db, query=q, status=status, limit=limit, offset=offset)
    return [InvoiceResponse.model_validate(inv) for inv in invoices]


@router.get("/api/invoices/{invoice_id}", response_model=InvoiceResponse)
async def api_get_invoice(
    invoice_id: str,
    db: AsyncSession = Depends(get_db),
):
    """API endpoint to get an invoice by ID."""
    invoice = await FinanceService.get_invoice(db, invoice_id)
    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found.")
    return InvoiceResponse.model_validate(invoice)


@router.post("/api/invoices", response_model=InvoiceResponse, status_code=http_status.HTTP_201_CREATED)
async def api_create_invoice(
    invoice_in: InvoiceCreate,
    db: AsyncSession = Depends(get_db),
):
    """API endpoint to create an invoice."""
    try:
        invoice = await FinanceService.create_invoice(db, invoice_in)
        return InvoiceResponse.model_validate(invoice)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
