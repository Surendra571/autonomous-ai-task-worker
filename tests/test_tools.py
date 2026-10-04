"""Unit tests for each agent tool and the dynamic ToolRegistry."""

from pathlib import Path
import pytest
from playwright.async_api import async_playwright

from app.config.settings import settings
from app.tools.base import BaseTool, ToolResult
from app.tools.browser_tool import BrowserTool
from app.tools.document_search_tool import DocumentSearchTool
from app.tools.document_read_tool import DocumentReadTool
from app.tools.finance_search_tool import FinanceSearchTool
from app.tools.finance_create_invoice_tool import FinanceCreateInvoiceTool
from app.tools.finance_read_invoice_tool import FinanceReadInvoiceTool
from app.tools.registry import ToolRegistry, create_default_registry
from tests.conftest import TestingSessionLocal


# ---------------------------------------------------------------------------
# 1. ToolResult Contract Tests
# ---------------------------------------------------------------------------

def test_tool_result_success():
    """Ensure ToolResult.ok creates valid successful result."""
    res = ToolResult.ok(
        data={"key": "value"},
        evidence={"proof": "screenshot.png"},
        metadata={"tool": "test_tool"},
    )
    assert res.success is True
    assert res.error is None
    assert res.data == {"key": "value"}
    assert res.evidence == {"proof": "screenshot.png"}
    assert res.metadata["tool"] == "test_tool"


def test_tool_result_failure():
    """Ensure ToolResult.fail properly contains error messages without crashing."""
    res = ToolResult.fail("Selector not found", metadata={"attempt": 1})
    assert res.success is False
    assert res.error == "Selector not found"
    assert res.metadata["attempt"] == 1


# ---------------------------------------------------------------------------
# 2. BrowserTool Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_browser_tool_actions(tmp_path):
    """Test all BrowserTool actions (open_url, type, click, extract_text, screenshot)."""
    tool = BrowserTool(artifacts_dir=tmp_path)

    try:
        # 1. open_url using inline data URI
        html_page = """
        <html>
            <head><title>Test App</title></head>
            <body>
                <h1 id="heading">Autonomous AI</h1>
                <input id="test-input" type="text" />
                <button id="test-btn" onclick="document.getElementById('heading').innerText='Clicked!'">Submit</button>
            </body>
        </html>
        """
        data_url = f"data:text/html,{html_page}"
        open_res = await tool.run(action="open_url", url=data_url)
        assert open_res.success is True
        assert open_res.data["title"] == "Test App"

        # 2. extract_text
        extract_res = await tool.run(action="extract_text", selector="#heading")
        assert extract_res.success is True
        assert extract_res.data["extracted_text"] == "Autonomous AI"

        # 3. type
        type_res = await tool.run(action="type", selector="#test-input", text="Agent Input")
        assert type_res.success is True
        assert type_res.data["typed_text"] == "Agent Input"

        # 4. click
        click_res = await tool.run(action="click", selector="#test-btn")
        assert click_res.success is True

        # Verify updated text
        updated_res = await tool.run(action="extract_text", selector="#heading")
        assert updated_res.data["extracted_text"] == "Clicked!"

        # 5. screenshot
        screen_res = await tool.run(action="screenshot", filename="test_browser_tool.png")
        assert screen_res.success is True
        assert "screenshot_path" in screen_res.data
        assert Path(screen_res.data["screenshot_path"]).exists()

        # 6. Invalid action
        bad_res = await tool.run(action="unknown_action_xyz")
        assert bad_res.success is False
        assert "Unknown browser action" in bad_res.error
    finally:
        await tool.close()


# ---------------------------------------------------------------------------
# 3. DocumentSearchTool Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_document_search_tool():
    """Ensure DocumentSearchTool discovers invoice files and ranks chronologically."""
    tool = DocumentSearchTool(inbox_dir=Path("data/invoices"))

    # Search for Acme Corp
    res = await tool.run(vendor="Acme Corp")
    assert res.success is True
    assert res.data["total_found"] == 3
    invoices = res.data["invoices"]
    assert len(invoices) == 3

    # Latest must be 2024-09-28
    assert res.data["latest_invoice"]["invoice_date"] == "2024-09-28"
    assert res.data["latest_invoice"]["invoice_number"] == "INV-ACM-2024-015"

    # Search with non-matching vendor
    empty_res = await tool.run(vendor="NonExistentVendorXYZ")
    assert empty_res.success is True
    assert empty_res.data["total_found"] == 0
    assert empty_res.data["latest_invoice"] is None


# ---------------------------------------------------------------------------
# 4. DocumentReadTool Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_document_read_tool():
    """Ensure DocumentReadTool extracts structured invoice data from PDF."""
    tool = DocumentReadTool()

    target_file = "data/invoices/acme_invoice_2024_09.pdf"
    res = await tool.run(file_path=target_file)
    assert res.success is True
    inv_data = res.data["invoice_data"]
    assert inv_data["vendor_name"] == "Acme Corp"
    assert inv_data["invoice_number"] == "INV-ACM-2024-015"
    assert inv_data["invoice_date"] == "2024-09-28"
    assert inv_data["due_date"] == "2024-10-28"
    assert inv_data["amount"] == 12450.50

    # Non-existent file
    fail_res = await tool.run(file_path="data/invoices/does_not_exist.pdf")
    assert fail_res.success is False
    assert "File not found" in fail_res.error


# ---------------------------------------------------------------------------
# 5. FinanceSearchTool Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_finance_search_tool():
    """Ensure FinanceSearchTool queries the seeded finance database."""
    tool = FinanceSearchTool(session_factory=TestingSessionLocal)

    res = await tool.run(query="Acme Corp")
    assert res.success is True
    assert res.data["total_count"] >= 2
    for inv in res.data["invoices"]:
        assert "Acme" in inv["vendor_name"]


# ---------------------------------------------------------------------------
# 6. FinanceCreateInvoiceTool Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_finance_create_invoice_tool():
    """Ensure FinanceCreateInvoiceTool inserts invoices and enforces safety."""
    tool = FinanceCreateInvoiceTool(session_factory=TestingSessionLocal)

    # Assert risk classification
    assert tool.risk_level == "HIGH"

    # Create new invoice
    payload = {
        "invoice_number": "INV-TOOL-TEST-001",
        "vendor_name": "Acme Corp",
        "amount": 7250.00,
        "currency": "USD",
        "issue_date": "2024-10-01",
        "due_date": "2024-11-01",
        "status": "PENDING",
        "description": "Tool integration test",
    }
    create_res = await tool.run(**payload)
    assert create_res.success is True
    assert create_res.data["invoice_number"] == "INV-TOOL-TEST-001"
    assert create_res.data["amount"] == 7250.00

    # Duplicate invoice_number must fail
    dup_res = await tool.run(**payload)
    assert dup_res.success is False
    assert "already exists" in dup_res.error


# ---------------------------------------------------------------------------
# 7. FinanceReadInvoiceTool Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_finance_read_invoice_tool():
    """Ensure FinanceReadInvoiceTool retrieves invoice details for verification."""
    tool = FinanceReadInvoiceTool(session_factory=TestingSessionLocal)

    # Read seeded invoice by invoice_number
    res = await tool.run(invoice_number="INV-ACM-2024-001")
    assert res.success is True
    assert res.data["invoice_number"] == "INV-ACM-2024-001"
    assert res.data["vendor_name"] == "Acme Corp"
    assert res.data["amount"] == 4500.00
    assert res.evidence["verified_in_ledger"] is True

    # Read non-existent invoice
    fail_res = await tool.run(invoice_number="NON-EXISTENT-INV-999")
    assert fail_res.success is False
    assert "not found" in fail_res.error


# ---------------------------------------------------------------------------
# 8. ToolRegistry Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_registry_lifecycle():
    """Test ToolRegistry discovery, schema generation, and routing."""
    registry = create_default_registry(
        session_factory=TestingSessionLocal,
        inbox_dir=Path("data/invoices"),
    )

    # Verify registration of all 6 tools
    tool_names = registry.list_tools()
    assert "browser_tool" in tool_names
    assert "document_search_tool" in tool_names
    assert "document_read_tool" in tool_names
    assert "finance_search_tool" in tool_names
    assert "finance_create_invoice_tool" in tool_names
    assert "finance_read_invoice_tool" in tool_names

    # Check schemas
    schemas = registry.get_schemas()
    assert len(schemas) == 6
    for s in schemas:
        assert s["type"] == "function"
        assert "name" in s["function"]
        assert "description" in s["function"]
        assert "parameters" in s["function"]
        assert "risk_level" in s["function"]

    # Dispatch tool execution through registry
    exec_res = await registry.execute(
        "document_search_tool",
        {"vendor": "Globex"},
    )
    assert exec_res.success is True
    assert exec_res.data["total_found"] == 2

    # Dispatch to unknown tool
    unknown_res = await registry.execute("imaginary_tool_404", {})
    assert unknown_res.success is False
    assert "not found" in unknown_res.error
