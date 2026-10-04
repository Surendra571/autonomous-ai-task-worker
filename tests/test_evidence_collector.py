"""Comprehensive unit and integration tests for EvidenceCollector.

Verifies:
1. Secret and credential sanitization (passwords, tokens, cookies, auth headers redacted).
2. Action evidence collection: screenshot, URL, page title, extracted text, tool name, timestamp, API response, verification result.
3. Explicit linking to execution events (step_number, action_name, task_id).
4. The 4 invoice workflow milestones:
   - Source invoice
   - Extracted invoice information
   - Finance application record
   - Verification result
5. Final task response referencing useful evidence.
6. Clean database persistence and metadata storage.
"""

from datetime import datetime
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.evidence import (
    EvidenceCollector,
    EvidenceItem,
    EvidenceType,
    InvoiceWorkflowEvidence,
    is_sensitive_key,
    sanitize_evidence,
    sanitize_string,
)
from app.agent.executor import AgentExecutor
from app.agent.state import ActionProposal, ActionType, TaskState, VerificationResult
from app.models.schemas import TaskStatus
from app.models.task import EvidenceRecord, TaskRun
from app.services.task_service import TaskService
from app.tools.base import BaseTool, ToolResult
from app.tools.registry import ToolRegistry
from tests.conftest import TestingSessionLocal


# ---------------------------------------------------------------------------
# 1. Secret & Credential Redaction Tests
# ---------------------------------------------------------------------------

def test_is_sensitive_key():
    """Verify sensitive key detection for various casings and naming conventions."""
    assert is_sensitive_key("password") is True
    assert is_sensitive_key("admin_password") is True
    assert is_sensitive_key("api_key") is True
    assert is_sensitive_key("apiKey") is True
    assert is_sensitive_key("ACCESS_TOKEN") is True
    assert is_sensitive_key("session_id") is True
    assert is_sensitive_key("client_secret") is True
    assert is_sensitive_key("authorization") is True
    assert is_sensitive_key("cookies") is True
    assert is_sensitive_key("private_key") is True

    # Safe keys should return False
    assert is_sensitive_key("vendor_name") is False
    assert is_sensitive_key("amount") is False
    assert is_sensitive_key("invoice_number") is False
    assert is_sensitive_key("due_date") is False
    assert is_sensitive_key("status") is False


def test_sanitize_evidence_redacts_credentials():
    """Verify that nested dictionaries and lists have sensitive values scrubbed."""
    dirty_data = {
        "vendor_name": "Globex Corp",
        "invoice_number": "INV-GLX-2024-99",
        "amount": 4500.00,
        "password": "SuperSecretPassword123!",
        "api_key": "sk-live-abcdef1234567890",
        "auth_token": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.dummy",
        "nested": {
            "session_id": "sess_9988776655",
            "safe_description": "Server hardware maintenance",
            "private_key": "-----BEGIN RSA PRIVATE KEY-----...",
            "credentials": {
                "user": "admin",
                "passwd": "myAdminPassword",
            },
        },
        "cookies": ["session=abc123xyz", "csrftoken=998877"],
    }

    clean = sanitize_evidence(dirty_data)

    # Sensitive values must be masked
    assert clean["password"] == "[REDACTED]"
    assert clean["api_key"] == "[REDACTED]"
    assert clean["auth_token"] == "[REDACTED]"
    assert clean["nested"]["session_id"] == "[REDACTED]"
    assert clean["nested"]["private_key"] == "[REDACTED]"
    assert clean["nested"]["credentials"]["passwd"] == "[REDACTED]"
    assert clean["cookies"] == "[REDACTED]"

    # Business data must be preserved intact
    assert clean["vendor_name"] == "Globex Corp"
    assert clean["invoice_number"] == "INV-GLX-2024-99"
    assert clean["amount"] == 4500.00
    assert clean["nested"]["safe_description"] == "Server hardware maintenance"
    assert clean["nested"]["credentials"]["user"] == "admin"


def test_sanitize_string_urls_and_headers():
    """Verify that credentials in URLs and Authorization header tokens are scrubbed."""
    url_with_pass = "http://admin:secretPass123@finance.corp.internal:8000/api/invoices"
    sanitized_url = sanitize_string(url_with_pass)
    assert "secretPass123" not in sanitized_url
    assert "admin:[REDACTED]@" in sanitized_url

    bearer_str = "Authorization: Bearer my_secret_jwt_token_value_here"
    sanitized_bearer = sanitize_string(bearer_str)
    assert "my_secret_jwt_token_value_here" not in sanitized_bearer
    assert "Bearer [REDACTED]" in sanitized_bearer


# ---------------------------------------------------------------------------
# 2. Tool Evidence Collection Tests
# ---------------------------------------------------------------------------

def test_collect_tool_evidence_screenshot():
    """Verify collection of screenshot paths and categorization as SCREENSHOT."""
    tool_res = ToolResult.ok(
        data={"screenshot_path": "artifacts/screenshots/inv_form.png", "filename": "inv_form.png"},
        evidence={"screenshot": "artifacts/screenshots/inv_form.png", "url": "http://localhost:8000/invoices/new"},
    )
    item = EvidenceCollector.collect_tool_evidence(
        task_id="task-test-01",
        step_number=3,
        action_name="browser_tool",
        tool_name="browser_tool",
        tool_args={"action": "screenshot", "filename": "inv_form.png"},
        tool_result=tool_res,
    )

    assert item.task_id == "task-test-01"
    assert item.step_number == 3
    assert item.tool_name == "browser_tool"
    assert item.evidence_type == EvidenceType.SCREENSHOT.value
    assert item.screenshot_path == "artifacts/screenshots/inv_form.png"
    assert item.url == "http://localhost:8000/invoices/new"
    assert "Screenshot captured" in item.description


def test_collect_tool_evidence_browser_navigation():
    """Verify collection of URL, page title, and extracted text."""
    tool_res = ToolResult.ok(
        data={
            "current_url": "http://localhost:8000/invoices",
            "title": "Finance System - Invoices Ledger",
            "extracted_text": "Invoice INV-2024-001 | Acme Corp | $1,250.00",
        },
    )
    item = EvidenceCollector.collect_tool_evidence(
        task_id="task-test-02",
        step_number=2,
        action_name="open_url",
        tool_name="browser_tool",
        tool_args={"action": "open_url", "url": "http://localhost:8000/invoices"},
        tool_result=tool_res,
    )

    assert item.step_number == 2
    assert item.evidence_type == EvidenceType.BROWSER_PAGE.value
    assert item.url == "http://localhost:8000/invoices"
    assert item.page_title == "Finance System - Invoices Ledger"
    assert "INV-2024-001" in item.extracted_text


def test_collect_tool_evidence_scrubs_api_response():
    """Verify that API response payloads have secrets stripped."""
    tool_res = ToolResult.ok(
        data={
            "invoice_id": "inv_rec_12345",
            "invoice_number": "INV-100",
            "api_key": "super_secret_internal_key",
            "session_token": "token_xyz",
        }
    )
    item = EvidenceCollector.collect_tool_evidence(
        task_id="task-test-03",
        step_number=5,
        action_name="finance_create_invoice_tool",
        tool_name="finance_create_invoice_tool",
        tool_args={"invoice_number": "INV-100", "password": "dont_leak_me"},
        tool_result=tool_res,
    )

    assert item.api_response["api_key"] == "[REDACTED]"
    assert item.api_response["session_token"] == "[REDACTED]"
    assert item.payload["tool_args"]["password"] == "[REDACTED]"
    assert item.api_response["invoice_id"] == "inv_rec_12345"


# ---------------------------------------------------------------------------
# 3. Four-Milestone Invoice Workflow Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_four_invoice_workflow_milestones():
    """Verify explicit recording and aggregation of all 4 invoice workflow milestones:

    1. Source invoice
    2. Extracted invoice information
    3. Finance application record
    4. Verification result
    """
    async with TestingSessionLocal() as session:
        # Create a test task run
        task = await TaskService.create_task(session, goal="Process latest Umbrella invoice")
        task_id = task.id

        # Milestone 1: Source invoice
        item1 = await EvidenceCollector.record_source_invoice_milestone(
            db=session,
            task_id=task_id,
            file_path="data/invoices/umbrella_corp_2024_03.pdf",
            file_name="umbrella_corp_2024_03.pdf",
            invoice_date="2024-03-25",
            vendor_name="Umbrella",
            step_number=1,
        )
        assert item1.evidence_type == EvidenceType.SOURCE_INVOICE.value
        assert item1.payload["milestone"] == 1
        assert item1.step_number == 1

        # Milestone 2: Extracted invoice information
        inv_data = {
            "vendor_name": "Umbrella",
            "invoice_number": "INV-UMB-2024-03",
            "amount": 7500.00,
            "currency": "USD",
            "issue_date": "2024-03-25",
            "due_date": "2024-04-25",
        }
        item2 = await EvidenceCollector.record_extracted_invoice_milestone(
            db=session,
            task_id=task_id,
            invoice_data=inv_data,
            file_path="data/invoices/umbrella_corp_2024_03.pdf",
            step_number=2,
        )
        assert item2.evidence_type == EvidenceType.EXTRACTED_INVOICE.value
        assert item2.payload["milestone"] == 2
        assert item2.payload["invoice_data"]["invoice_number"] == "INV-UMB-2024-03"

        # Milestone 3: Finance application record
        fin_record = {
            "invoice_id": "rec_ledger_987",
            "invoice_number": "INV-UMB-2024-03",
            "vendor_name": "Umbrella",
            "amount": 7500.00,
            "status": "PENDING",
        }
        item3 = await EvidenceCollector.record_finance_record_milestone(
            db=session,
            task_id=task_id,
            finance_record=fin_record,
            step_number=3,
        )
        assert item3.evidence_type == EvidenceType.FINANCE_RECORD.value
        assert item3.payload["milestone"] == 3
        assert item3.payload["finance_record"]["invoice_id"] == "rec_ledger_987"

        # Milestone 4: Verification result
        ver_rep = {
            "verified": True,
            "checks": [
                {"name": "invoice_exists", "passed": True},
                {"name": "vendor_matches", "passed": True},
                {"name": "amount_matches", "passed": True},
                {"name": "due_date_matches", "passed": True},
            ],
            "discrepancies": [],
        }
        item4 = await EvidenceCollector.record_verification_result_milestone(
            db=session,
            task_id=task_id,
            verification_report=ver_rep,
            step_number=4,
        )
        assert item4.evidence_type == EvidenceType.VERIFICATION_RESULT.value
        assert item4.payload["milestone"] == 4
        assert item4.payload["verified"] is True

        # Assemble into state and verify workflow evidence dossier
        state = TaskState(
            task_id=task_id,
            goal="Process latest Umbrella invoice",
            status=TaskStatus.COMPLETED,
            extracted_data={
                "latest_invoice_path": "data/invoices/umbrella_corp_2024_03.pdf",
                "latest_invoice_date": "2024-03-25",
                "vendor_name": "Umbrella",
                "invoice_number": "INV-UMB-2024-03",
                "amount": 7500.00,
                "currency": "USD",
                "due_date": "2024-04-25",
                "created_invoice_id": "rec_ledger_987",
            },
            latest_verification=VerificationResult(
                verified=True,
                verification_type="independent_verification",
                checks=ver_rep["checks"],
                discrepancies=[],
            ),
        )

        dossier = EvidenceCollector.build_workflow_evidence(state)
        assert dossier.is_complete is True
        assert dossier.source_invoice["file_path"] == "data/invoices/umbrella_corp_2024_03.pdf"
        assert dossier.extracted_invoice_info["invoice_number"] == "INV-UMB-2024-03"
        assert dossier.finance_application_record["id"] == "rec_ledger_987"
        assert dossier.verification_result["verified"] is True

        summary_text = dossier.to_summary_text()
        assert "1. Source Invoice" in summary_text
        assert "umbrella_corp_2024_03.pdf" in summary_text
        assert "2. Extracted Invoice Information" in summary_text
        assert "INV-UMB-2024-03" in summary_text
        assert "3. Finance Application Record" in summary_text
        assert "rec_ledger_987" in summary_text
        assert "4. Verification Result" in summary_text
        assert "PASSED" in summary_text


# ---------------------------------------------------------------------------
# 4. Final Task Summary & Audit References
# ---------------------------------------------------------------------------

def test_final_task_summary_references_useful_evidence():
    """Verify that generate_final_task_summary references all four milestones."""
    state = TaskState(
        task_id="task-summary-test",
        goal="Find and enter latest Acme Corp invoice",
        status=TaskStatus.COMPLETED,
        extracted_data={
            "latest_invoice_path": "data/invoices/acme_invoice_2024_10.pdf",
            "file_name": "acme_invoice_2024_10.pdf",
            "latest_invoice_date": "2024-10-01",
            "vendor_name": "Acme Corp",
            "invoice_number": "INV-ACM-2024-010",
            "amount": 3200.00,
            "currency": "USD",
            "due_date": "2024-10-31",
            "created_invoice_id": "inv_acme_10",
        },
        latest_verification=VerificationResult(
            verified=True,
            verification_type="cross_check",
            checks=[
                {"name": "invoice_exists", "passed": True},
                {"name": "amount_matches", "passed": True},
            ],
            discrepancies=[],
        ),
    )

    final_summary = EvidenceCollector.generate_final_task_summary(state)

    assert "acme_invoice_2024_10.pdf" in final_summary
    assert "INV-ACM-2024-010" in final_summary
    assert "USD 3200.0" in final_summary or "3200" in final_summary
    assert "inv_acme_10" in final_summary
    assert "PASSED" in final_summary
    assert "invoice_exists" in final_summary


# ---------------------------------------------------------------------------
# 5. Database Persistence & Detailed Task Response Integration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_evidence_persists_cleanly_in_database():
    """Verify that evidence items are persisted to the database and exposed via TaskService."""
    async with TestingSessionLocal() as session:
        task = await TaskService.create_task(session, goal="Audit evidence persistence test")

        item = EvidenceItem(
            task_id=task.id,
            step_number=1,
            action_name="document_search",
            tool_name="document_search_tool",
            evidence_type=EvidenceType.SOURCE_INVOICE.value,
            description="Found latest invoice file",
            source_uri_or_path="data/invoices/initech_2024.pdf",
            payload={"matched_files": ["data/invoices/initech_2024.pdf"]},
        )

        record = await EvidenceCollector.record_evidence(session, item)

        assert record.id is not None
        assert record.task_id == task.id
        assert record.evidence_type == EvidenceType.SOURCE_INVOICE.value
        assert record.source_uri_or_path == "data/invoices/initech_2024.pdf"
        assert record.payload["step_number"] == 1

        # Fetch detailed task record
        detailed_task = await TaskService.get_task(session, task_id=task.id, detailed=True)
        assert len(detailed_task.evidence) >= 1
        ev_rec = detailed_task.evidence[0]
        assert ev_rec.evidence_type == EvidenceType.SOURCE_INVOICE.value
        assert ev_rec.description == "Found latest invoice file"


# ---------------------------------------------------------------------------
# 6. Integration: Executor Collects Evidence Throughout Loop
# ---------------------------------------------------------------------------

class MockSearchTool(BaseTool):
    name = "document_search_tool"
    description = "Searches for invoices"
    input_schema = dict

    async def execute(self, params: dict) -> ToolResult:
        return ToolResult.ok(
            data={
                "vendor_name": "Initech",
                "latest_invoice": {
                    "file_path": "data/invoices/initech_2024_08.pdf",
                    "invoice_date": "2024-08-15",
                    "vendor_name": "Initech",
                },
            },
            evidence={"file": "data/invoices/initech_2024_08.pdf"},
        )


class MockReadTool(BaseTool):
    name = "document_read_tool"
    description = "Reads invoice PDF"
    input_schema = dict

    async def execute(self, params: dict) -> ToolResult:
        return ToolResult.ok(
            data={
                "file_path": "data/invoices/initech_2024_08.pdf",
                "file_name": "initech_2024_08.pdf",
                "invoice_data": {
                    "vendor_name": "Initech",
                    "invoice_number": "INV-INI-2024-08",
                    "amount": 1800.00,
                    "currency": "USD",
                    "issue_date": "2024-08-15",
                    "due_date": "2024-09-15",
                },
            },
            evidence={"extracted_text": "INVOICE INV-INI-2024-08 Total: $1800.00"},
        )


class MockCreateTool(BaseTool):
    name = "finance_create_invoice_tool"
    description = "Creates invoice in ledger"
    input_schema = dict

    async def execute(self, params: dict) -> ToolResult:
        return ToolResult.ok(
            data={
                "invoice_id": "inv_ledger_ini_88",
                "invoice_number": "INV-INI-2024-08",
                "vendor_name": "Initech",
                "amount": 1800.00,
                "currency": "USD",
                "issue_date": "2024-08-15",
                "due_date": "2024-09-15",
                "status": "PENDING",
            }
        )


class MockVerifierTool(BaseTool):
    name = "finance_read_invoice_tool"
    description = "Reads created invoice"
    input_schema = dict

    async def execute(self, params: dict) -> ToolResult:
        return ToolResult.ok(
            data={
                "id": "inv_ledger_ini_88",
                "invoice_number": "INV-INI-2024-08",
                "vendor_name": "Initech",
                "amount": 1800.00,
                "status": "PENDING",
            }
        )


@pytest.mark.asyncio
async def test_executor_records_evidence_in_execution_loop():
    """Verify that AgentExecutor collects evidence during tool execution and produces evidence-rich completion."""
    async with TestingSessionLocal() as session:
        task = await TaskService.create_task(session, goal="Process Initech invoice end-to-end")

        registry = ToolRegistry()
        registry.register(MockSearchTool())
        registry.register(MockReadTool())
        registry.register(MockCreateTool())
        registry.register(MockVerifierTool())

        # Mock planner returning sequential actions
        actions_queue = [
            ActionProposal(
                thought="Search for latest Initech invoice",
                action_type=ActionType.CALL_TOOL,
                tool_name="document_search_tool",
                tool_args={"vendor": "Initech"},
            ),
            ActionProposal(
                thought="Read invoice details from PDF",
                action_type=ActionType.CALL_TOOL,
                tool_name="document_read_tool",
                tool_args={"file_path": "data/invoices/initech_2024_08.pdf"},
            ),
            ActionProposal(
                thought="Enter invoice into finance application",
                action_type=ActionType.CALL_TOOL,
                tool_name="finance_create_invoice_tool",
                tool_args={
                    "vendor_name": "Initech",
                    "invoice_number": "INV-INI-2024-08",
                    "amount": 1800.00,
                    "due_date": "2024-09-15",
                },
            ),
            ActionProposal(
                thought="Read invoice back from finance application to verify",
                action_type=ActionType.CALL_TOOL,
                tool_name="finance_read_invoice_tool",
                tool_args={"invoice_number": "INV-INI-2024-08"},
            ),
        ]

        step_idx = 0

        class MockSequentialPlanner:
            async def decide_next_action(self, state):
                nonlocal step_idx
                if step_idx < len(actions_queue):
                    act = actions_queue[step_idx]
                    step_idx += 1
                    return act
                return ActionProposal(
                    thought="Goal completed",
                    action_type=ActionType.COMPLETE_TASK,
                )

        executor = AgentExecutor(
            tool_registry=registry,
            planner=MockSequentialPlanner(),
            max_steps=10,
        )

        final_state = await executor.execute_task(session, task_id=task.id)

        assert final_state.current_status == TaskStatus.COMPLETED
        # Evidence must be collected across the steps
        assert len(final_state.evidence) >= 4

        # Final result must reference the key evidence milestones
        result_text = final_state.result_summary
        assert "Source Invoice" in result_text
        assert "initech_2024_08.pdf" in result_text
        assert "INV-INI-2024-08" in result_text
        assert "inv_ledger_ini_88" in result_text
        assert "Finance Application Record" in result_text

        # Verify DB records
        detailed_task = await TaskService.get_task(session, task_id=task.id, detailed=True)
        assert len(detailed_task.evidence) >= 4
