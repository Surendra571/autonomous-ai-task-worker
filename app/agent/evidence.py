"""Dedicated EvidenceCollector for capturing, sanitizing, and persisting audit evidence.

Captures screenshots, URLs, page titles, extracted text, tool execution metadata,
API responses, and independent verification results linked directly to execution events.
Enforces credential/secret sanitization and builds structured multi-milestone
evidence dossiers for business workflows.
"""

from datetime import datetime
from enum import Enum
import re
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.state import TaskState
from app.config.logging import get_logger
from app.models.task import EvidenceRecord
from app.services.task_service import TaskService
from app.tools.base import ToolResult

logger = get_logger("agent.evidence")


# ---------------------------------------------------------------------------
# Secret & Credential Sanitization
# ---------------------------------------------------------------------------

SENSITIVE_KEY_PATTERNS = [
    r"password",
    r"passwd",
    r"pwd",
    r"secret",
    r"token",
    r"api[_-]?key",
    r"access[_-]?token",
    r"refresh[_-]?token",
    r"auth",
    r"authorization",
    r"cookie",
    r"cookies",
    r"session",
    r"session[_-]?id",
    r"credential",
    r"credentials",
    r"private[_-]?key",
    r"client[_-]?secret",
]

SENSITIVE_KEY_REGEX = re.compile(
    "|".join(f"(?:{pat})" for pat in SENSITIVE_KEY_PATTERNS),
    re.IGNORECASE,
)

URL_CREDENTIALS_REGEX = re.compile(r"(https?://)([^:]+):([^@]+)(@)", re.IGNORECASE)
BEARER_TOKEN_REGEX = re.compile(r"(Bearer\s+)[A-Za-z0-9_\-\.]{8,}", re.IGNORECASE)
GENERIC_TOKEN_REGEX = re.compile(r"((?:token|api[_-]?key|secret|password)\s*[:=]\s*['\"]?)[^'\"\s,;&]{6,}(['\"]?)", re.IGNORECASE)


def is_sensitive_key(key: Any) -> bool:
    """Checks whether a dictionary key name indicates sensitive or credential data."""
    if not isinstance(key, str):
        key = str(key)
    return bool(SENSITIVE_KEY_REGEX.search(key))


def sanitize_string(text: str) -> str:
    """Redacts passwords, tokens, and authorization credentials from string text."""
    if not text:
        return text
    # Mask credentials embedded in URLs e.g. http://admin:pass123@host
    redacted = URL_CREDENTIALS_REGEX.sub(r"\1\2:[REDACTED]\4", text)
    # Mask Authorization: Bearer <token>
    redacted = BEARER_TOKEN_REGEX.sub(r"\1[REDACTED]", redacted)
    # Mask inline assignments like token="xyz"
    redacted = GENERIC_TOKEN_REGEX.sub(r"\1[REDACTED]\2", redacted)
    return redacted


def sanitize_evidence(data: Any) -> Any:
    """
    Recursively scrubs secrets, credentials, tokens, session cookies,
    and passwords from arbitrary Python data structures.
    """
    if data is None:
        return None

    if isinstance(data, (int, float, bool)):
        return data

    if isinstance(data, str):
        return sanitize_string(data)

    if isinstance(data, dict):
        sanitized_dict: Dict[str, Any] = {}
        for k, v in data.items():
            k_str = str(k)
            if is_sensitive_key(k_str):
                if isinstance(v, dict):
                    sanitized_dict[k_str] = {
                        sub_k: (
                            "[REDACTED]"
                            if is_sensitive_key(sub_k)
                            else sanitize_evidence(sub_v)
                        )
                        for sub_k, sub_v in v.items()
                    }
                else:
                    sanitized_dict[k_str] = "[REDACTED]"
            else:
                sanitized_dict[k_str] = sanitize_evidence(v)
        return sanitized_dict

    if isinstance(data, (list, tuple, set)):
        sanitized_list = [sanitize_evidence(item) for item in data]
        return type(data)(sanitized_list) if isinstance(data, (list, tuple)) else sanitized_list

    if hasattr(data, "model_dump") and callable(data.model_dump):
        return sanitize_evidence(data.model_dump())

    if hasattr(data, "__dict__"):
        return sanitize_evidence(data.__dict__)

    return sanitize_string(str(data))


# ---------------------------------------------------------------------------
# Structured Evidence Models
# ---------------------------------------------------------------------------

class EvidenceType(str, Enum):
    """Categorization of collected audit evidence."""
    SCREENSHOT = "screenshot"
    BROWSER_PAGE = "browser_page"
    SOURCE_INVOICE = "source_invoice"
    EXTRACTED_INVOICE = "extracted_invoice"
    FINANCE_RECORD = "finance_record"
    VERIFICATION_RESULT = "verification_result"
    API_RESPONSE = "api_response"
    TOOL_EXECUTION = "tool_execution"
    MILESTONE = "milestone"


class EvidenceItem(BaseModel):
    """
    Comprehensive structured evidence record representing an action,
    system state snapshot, document, or verification outcome.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = Field(..., description="Task run this evidence belongs to.")
    step_number: Optional[int] = Field(None, description="Execution step/event that produced this evidence.")
    action_name: Optional[str] = Field(None, description="Action or tool invocation name.")
    tool_name: Optional[str] = Field(None, description="Name of the tool that ran.")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of collection.")
    evidence_type: str = Field(default=EvidenceType.TOOL_EXECUTION.value, description="Classification of evidence.")
    description: str = Field(..., description="Human-readable description of what this evidence proves.")

    # Specialized Evidence Fields
    url: Optional[str] = Field(None, description="Browser URL or API endpoint if applicable.")
    page_title: Optional[str] = Field(None, description="Browser page title if applicable.")
    screenshot_path: Optional[str] = Field(None, description="File path to captured screenshot on disk.")
    extracted_text: Optional[str] = Field(None, description="Text snippet extracted from page or document.")
    api_response: Optional[Any] = Field(None, description="Sanitized payload returned by API/backend.")
    verification_result: Optional[Dict[str, Any]] = Field(None, description="Outcome of validation checks.")

    # Persistence Fields
    source_uri_or_path: Optional[str] = Field(None, description="URI or path for storage reference.")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Full sanitized structured payload.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional audit telemetry.")

    def to_db_payload(self) -> Dict[str, Any]:
        """Formats the payload dictionary for storage in the PostgreSQL evidence_records table."""
        data = {
            "step_number": self.step_number,
            "action_name": self.action_name,
            "tool_name": self.tool_name,
            "timestamp": self.timestamp.isoformat(),
            "url": self.url,
            "page_title": self.page_title,
            "screenshot_path": self.screenshot_path,
            "extracted_text": self.extracted_text,
            "api_response": self.api_response,
            "verification_result": self.verification_result,
            "metadata": self.metadata,
        }
        if self.payload:
            data.update(self.payload)
        return sanitize_evidence(data)


class InvoiceWorkflowEvidence(BaseModel):
    """
    Structured four-milestone evidence dossier specifically required for the invoice workflow:
    1. Source invoice
    2. Extracted invoice information
    3. Finance application record
    4. Verification result
    """
    task_id: str
    source_invoice: Optional[Dict[str, Any]] = Field(
        None,
        description="Milestone 1: Source invoice details (file path, filename, date, vendor, size)."
    )
    extracted_invoice_info: Optional[Dict[str, Any]] = Field(
        None,
        description="Milestone 2: Extracted invoice data (vendor, invoice_number, amount, currency, due_date)."
    )
    finance_application_record: Optional[Dict[str, Any]] = Field(
        None,
        description="Milestone 3: Internal finance application record (ledger ID, invoice_number, status, timestamp)."
    )
    verification_result: Optional[Dict[str, Any]] = Field(
        None,
        description="Milestone 4: Independent verification report (verified, checks list, discrepancies)."
    )
    all_evidence_items: List[EvidenceItem] = Field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """True if all 4 required milestones are present and verified."""
        has_all_milestones = all([
            self.source_invoice is not None,
            self.extracted_invoice_info is not None,
            self.finance_application_record is not None,
            self.verification_result is not None,
        ])
        if not has_all_milestones:
            return False
        return bool(self.verification_result and self.verification_result.get("verified") is True)

    def to_summary_dict(self) -> Dict[str, Any]:
        """Provides a clean dictionary of the 4 milestones."""
        return {
            "source_invoice": self.source_invoice,
            "extracted_invoice_info": self.extracted_invoice_info,
            "finance_application_record": self.finance_application_record,
            "verification_result": self.verification_result,
            "is_complete": self.is_complete,
            "total_evidence_records": len(self.all_evidence_items),
        }

    def to_summary_text(self) -> str:
        """Builds a human-readable markdown summary referencing the 4 milestones."""
        lines = ["### Invoice Workflow Verification Dossier", ""]

        # 1. Source invoice
        lines.append("#### 1. Source Invoice")
        if self.source_invoice:
            file_name = self.source_invoice.get("file_name") or self.source_invoice.get("file_path", "N/A")
            vendor = self.source_invoice.get("vendor_name", "N/A")
            inv_date = self.source_invoice.get("invoice_date", "N/A")
            path = self.source_invoice.get("file_path", "N/A")
            lines.append(f"- **File:** `{file_name}`")
            lines.append(f"- **Path:** `{path}`")
            lines.append(f"- **Identified Vendor:** {vendor}")
            lines.append(f"- **Invoice Date:** {inv_date}")
        else:
            lines.append("- *Source invoice not yet identified or recorded.*")

        # 2. Extracted invoice info
        lines.append("")
        lines.append("#### 2. Extracted Invoice Information")
        if self.extracted_invoice_info:
            inv_num = self.extracted_invoice_info.get("invoice_number", "N/A")
            vendor = self.extracted_invoice_info.get("vendor_name", "N/A")
            amt = self.extracted_invoice_info.get("amount", "N/A")
            curr = self.extracted_invoice_info.get("currency", "USD")
            due = self.extracted_invoice_info.get("due_date", "N/A")
            lines.append(f"- **Invoice Number:** `{inv_num}`")
            lines.append(f"- **Vendor:** {vendor}")
            lines.append(f"- **Amount:** {curr} {amt}")
            lines.append(f"- **Due Date:** {due}")
        else:
            lines.append("- *Invoice fields have not been extracted.*")

        # 3. Finance application record
        lines.append("")
        lines.append("#### 3. Finance Application Record")
        if self.finance_application_record:
            rec_id = self.finance_application_record.get("id") or self.finance_application_record.get("invoice_id", "N/A")
            rec_num = self.finance_application_record.get("invoice_number", "N/A")
            status = self.finance_application_record.get("status", "N/A")
            lines.append(f"- **Ledger ID:** `{rec_id}`")
            lines.append(f"- **Ledger Invoice Number:** `{rec_num}`")
            lines.append(f"- **Status:** `{status}`")
        else:
            lines.append("- *No invoice recorded in finance system.*")

        # 4. Verification result
        lines.append("")
        lines.append("#### 4. Verification Result")
        if self.verification_result:
            is_verified = self.verification_result.get("verified", False)
            status_badge = "PASSED" if is_verified else "FAILED"
            lines.append(f"- **Status:** `{status_badge}`")
            checks = self.verification_result.get("checks", [])
            if checks:
                lines.append("- **Independent Checks:**")
                for c in checks:
                    check_name = c.get("name", "check")
                    c_passed = c.get("passed", False)
                    mark = "✓" if c_passed else "✗"
                    lines.append(f"  - [{mark}] `{check_name}`: {'Passed' if c_passed else 'Failed'}")
            discrepancies = self.verification_result.get("discrepancies", [])
            if discrepancies:
                lines.append("- **Discrepancies:**")
                for d in discrepancies:
                    lines.append(f"  - ⚠️ {d}")
        else:
            lines.append("- *Independent verification not yet executed.*")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dedicated EvidenceCollector Engine
# ---------------------------------------------------------------------------

class EvidenceCollector:
    """
    Centralized collector responsible for:
    - Extracting rich evidence from tools, browser sessions, and system outputs.
    - Scrubbing sensitive credentials, tokens, and secrets from all payloads.
    - Persisting audit records linked directly to the execution step/event.
    - Assembling the 4 mandatory invoice workflow milestones.
    - Formatting final audit references for user-facing task responses.
    """

    sanitize_evidence = staticmethod(sanitize_evidence)

    @classmethod
    def collect_tool_evidence(
        cls,
        task_id: str,
        step_number: int,
        action_name: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_result: ToolResult,
        browser_metadata: Optional[Dict[str, Any]] = None,
    ) -> EvidenceItem:
        """
        Extracts all relevant evidence from a tool execution:
        - screenshot
        - URL
        - page title
        - extracted text
        - tool name
        - timestamp
        - relevant API response
        - verification result
        All data is scrubbed of credentials/secrets before returning.
        """
        tool_data = tool_result.data or {}
        tool_ev = tool_result.evidence or {}
        if not isinstance(tool_ev, dict):
            tool_ev = {"raw_evidence": tool_ev}
        b_meta = browser_metadata or {}

        # 1. Screenshot
        screenshot_path = (
            tool_data.get("screenshot_path")
            or tool_ev.get("screenshot")
            or tool_ev.get("screenshot_path")
            or b_meta.get("screenshot_path")
        )

        # 2. URL
        url = (
            tool_args.get("url")
            or tool_data.get("current_url")
            or tool_data.get("url")
            or tool_ev.get("url")
            or b_meta.get("url")
        )

        # 3. Page Title
        page_title = (
            tool_data.get("title")
            or tool_data.get("page_title")
            or tool_ev.get("title")
            or b_meta.get("title")
        )

        # 4. Extracted Text
        extracted_text = (
            tool_data.get("extracted_text")
            or tool_data.get("text")
            or tool_data.get("raw_text")
            or tool_ev.get("raw_text_excerpt")
        )
        if not extracted_text and tool_name == "document_read_tool":
            extracted_text = str(tool_data.get("invoice_data") or "")

        # 5. Relevant API Response
        api_response = None
        if "finance" in tool_name or "api" in tool_name:
            api_response = tool_data

        # 6. Verification Result
        verification_res = None
        if "verification" in tool_data:
            verification_res = tool_data.get("verification")
        elif "verify" in tool_name:
            verification_res = tool_data

        # Determine Primary Evidence Type
        ev_type = EvidenceType.TOOL_EXECUTION.value
        if screenshot_path:
            ev_type = EvidenceType.SCREENSHOT.value
        elif url or page_title:
            ev_type = EvidenceType.BROWSER_PAGE.value
        elif tool_name == "document_search_tool":
            ev_type = EvidenceType.SOURCE_INVOICE.value
        elif tool_name == "document_read_tool":
            ev_type = EvidenceType.EXTRACTED_INVOICE.value
        elif tool_name in ("finance_create_invoice_tool", "finance_read_invoice_tool"):
            ev_type = EvidenceType.FINANCE_RECORD.value
        elif verification_res is not None or "verify" in tool_name:
            ev_type = EvidenceType.VERIFICATION_RESULT.value
        elif api_response is not None:
            ev_type = EvidenceType.API_RESPONSE.value

        # Build clean description
        status_str = "succeeded" if tool_result.success else "failed"
        desc = f"Step {step_number}: {tool_name} ({action_name}) {status_str}."
        if screenshot_path:
            desc += f" Screenshot captured at {screenshot_path}."
        if url:
            desc += f" URL: {url}."

        # Compile payload
        raw_payload = {
            "success": tool_result.success,
            "tool_data": tool_data,
            "tool_evidence": tool_ev,
            "tool_args": tool_args,
            "error": tool_result.error,
        }

        # Sanitize everything
        clean_args = sanitize_evidence(tool_args)
        clean_payload = sanitize_evidence(raw_payload)
        clean_api_res = sanitize_evidence(api_response)
        clean_ver_res = sanitize_evidence(verification_res)
        clean_url = sanitize_string(url) if url else None
        clean_text = sanitize_string(extracted_text) if extracted_text else None

        source_uri = clean_url or (str(screenshot_path) if screenshot_path else None)

        item = EvidenceItem(
            task_id=task_id,
            step_number=step_number,
            action_name=action_name,
            tool_name=tool_name,
            timestamp=datetime.utcnow(),
            evidence_type=ev_type,
            description=desc,
            url=clean_url,
            page_title=page_title,
            screenshot_path=str(screenshot_path) if screenshot_path else None,
            extracted_text=clean_text,
            api_response=clean_api_res,
            verification_result=clean_ver_res,
            source_uri_or_path=source_uri,
            payload=clean_payload,
            metadata={
                "tool_args": clean_args,
                "latency_ms": tool_result.metadata.get("latency_ms", 0.0),
            },
        )
        return item

    @classmethod
    async def record_evidence(
        cls,
        db: AsyncSession,
        evidence: EvidenceItem,
    ) -> EvidenceRecord:
        """Persists a structured evidence item cleanly into PostgreSQL evidence_records table."""
        sanitized_db_payload = evidence.to_db_payload()
        record = await TaskService.add_evidence(
            db=db,
            task_id=evidence.task_id,
            evidence_type=evidence.evidence_type,
            description=evidence.description,
            source_uri_or_path=evidence.source_uri_or_path or evidence.screenshot_path or evidence.url,
            payload=sanitized_db_payload,
        )
        logger.info(
            "EVIDENCE_RECORDED",
            task_id=evidence.task_id,
            evidence_id=record.id,
            type=evidence.evidence_type,
            step=evidence.step_number,
        )
        return record

    # -----------------------------------------------------------------------
    # Four-Milestone Invoice Workflow Recorders
    # -----------------------------------------------------------------------

    @classmethod
    async def record_source_invoice_milestone(
        cls,
        db: AsyncSession,
        task_id: str,
        file_path: str,
        file_name: Optional[str] = None,
        invoice_date: Optional[str] = None,
        vendor_name: Optional[str] = None,
        step_number: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvidenceItem:
        """
        Milestone 1: Records source invoice file discovery and selection.
        """
        payload = {
            "milestone": 1,
            "milestone_name": "Source Invoice",
            "file_path": file_path,
            "file_name": file_name or (file_path.split("/")[-1].split("\\")[-1] if file_path else None),
            "invoice_date": invoice_date,
            "vendor_name": vendor_name,
            "metadata": metadata or {},
        }
        item = EvidenceItem(
            task_id=task_id,
            step_number=step_number,
            action_name="source_invoice_identified",
            tool_name="document_search_tool",
            evidence_type=EvidenceType.SOURCE_INVOICE.value,
            description=f"Milestone 1: Source invoice identified '{payload['file_name']}' for vendor '{vendor_name}' (Date: {invoice_date}).",
            source_uri_or_path=file_path,
            payload=sanitize_evidence(payload),
        )
        await cls.record_evidence(db, item)
        return item

    @classmethod
    async def record_extracted_invoice_milestone(
        cls,
        db: AsyncSession,
        task_id: str,
        invoice_data: Dict[str, Any],
        file_path: Optional[str] = None,
        raw_text: Optional[str] = None,
        step_number: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvidenceItem:
        """
        Milestone 2: Records extracted structured invoice fields.
        """
        clean_inv = sanitize_evidence(invoice_data)
        payload = {
            "milestone": 2,
            "milestone_name": "Extracted Invoice Information",
            "invoice_data": clean_inv,
            "file_path": file_path,
            "raw_text_snippet": (raw_text[:300] if raw_text else None),
            "metadata": metadata or {},
        }
        inv_num = clean_inv.get("invoice_number", "Unknown")
        vendor = clean_inv.get("vendor_name", "Unknown")
        amt = clean_inv.get("amount", "0")
        curr = clean_inv.get("currency", "USD")

        item = EvidenceItem(
            task_id=task_id,
            step_number=step_number,
            action_name="invoice_fields_extracted",
            tool_name="document_read_tool",
            evidence_type=EvidenceType.EXTRACTED_INVOICE.value,
            description=f"Milestone 2: Extracted invoice data for '{inv_num}' ({vendor}): {curr} {amt}.",
            source_uri_or_path=file_path,
            extracted_text=str(clean_inv),
            payload=payload,
        )
        await cls.record_evidence(db, item)
        return item

    @classmethod
    async def record_finance_record_milestone(
        cls,
        db: AsyncSession,
        task_id: str,
        finance_record: Dict[str, Any],
        step_number: Optional[int] = None,
        tool_name: str = "finance_create_invoice_tool",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvidenceItem:
        """
        Milestone 3: Records the internal finance ledger application entry.
        """
        clean_rec = sanitize_evidence(finance_record)
        rec_id = clean_rec.get("invoice_id") or clean_rec.get("id", "Unknown")
        rec_num = clean_rec.get("invoice_number", "Unknown")
        vendor = clean_rec.get("vendor_name", "Unknown")

        payload = {
            "milestone": 3,
            "milestone_name": "Finance Application Record",
            "finance_record": clean_rec,
            "metadata": metadata or {},
        }

        item = EvidenceItem(
            task_id=task_id,
            step_number=step_number,
            action_name="finance_ledger_recorded",
            tool_name=tool_name,
            evidence_type=EvidenceType.FINANCE_RECORD.value,
            description=f"Milestone 3: Recorded in finance ledger with ID '{rec_id}' (Invoice: {rec_num}, Vendor: {vendor}).",
            api_response=clean_rec,
            source_uri_or_path=f"finance://invoices/{rec_id}",
            payload=payload,
        )
        await cls.record_evidence(db, item)
        return item

    @classmethod
    async def record_verification_result_milestone(
        cls,
        db: AsyncSession,
        task_id: str,
        verification_report: Any,
        step_number: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvidenceItem:
        """
        Milestone 4: Records the independent verification engine result.
        """
        if hasattr(verification_report, "to_dict"):
            rep_dict = verification_report.to_dict()
        elif hasattr(verification_report, "model_dump"):
            rep_dict = verification_report.model_dump()
        elif isinstance(verification_report, dict):
            rep_dict = verification_report
        else:
            rep_dict = {"verified": getattr(verification_report, "verified", False)}

        clean_report = sanitize_evidence(rep_dict)
        is_verified = bool(clean_report.get("verified", False))
        checks = clean_report.get("checks", [])
        discrepancies = clean_report.get("discrepancies", [])

        payload = {
            "milestone": 4,
            "milestone_name": "Verification Result",
            "verified": is_verified,
            "checks": checks,
            "discrepancies": discrepancies,
            "full_report": clean_report,
            "metadata": metadata or {},
        }

        status_text = "PASSED" if is_verified else "FAILED"
        item = EvidenceItem(
            task_id=task_id,
            step_number=step_number,
            action_name="independent_outcome_verified",
            tool_name="verification_engine",
            evidence_type=EvidenceType.VERIFICATION_RESULT.value,
            description=f"Milestone 4: Independent verification {status_text} ({len(checks)} checks evaluated, {len(discrepancies)} discrepancies).",
            verification_result=clean_report,
            payload=payload,
        )
        await cls.record_evidence(db, item)
        return item

    # -----------------------------------------------------------------------
    # Milestone Aggregation & Final Reporting
    # -----------------------------------------------------------------------

    @classmethod
    def build_workflow_evidence(cls, state: TaskState) -> InvoiceWorkflowEvidence:
        """
        Assembles the complete 4-milestone dossier from current TaskState working memory.
        """
        ext = state.extracted_data or {}

        # 1. Source invoice
        source_inv = None
        if ext.get("latest_invoice_path") or ext.get("file_path") or ext.get("file_name"):
            path = ext.get("latest_invoice_path") or ext.get("file_path")
            source_inv = {
                "file_path": path,
                "file_name": ext.get("file_name") or (path.split("/")[-1].split("\\")[-1] if path else None),
                "invoice_date": ext.get("latest_invoice_date") or ext.get("invoice_date"),
                "vendor_name": ext.get("vendor_name"),
            }

        # 2. Extracted invoice info
        extracted_info = None
        if ext.get("invoice_number") or ext.get("pdf_extracted"):
            extracted_info = {
                "invoice_number": ext.get("invoice_number"),
                "vendor_name": ext.get("vendor_name"),
                "amount": ext.get("amount"),
                "currency": ext.get("currency", "USD"),
                "issue_date": ext.get("issue_date") or ext.get("invoice_date"),
                "due_date": ext.get("due_date"),
            }

        # 3. Finance application record
        finance_rec = None
        if ext.get("created_invoice_id") or ext.get("saved_record") or ext.get("invoice_entered"):
            saved = ext.get("saved_record") or {}
            finance_rec = {
                "id": ext.get("created_invoice_id") or saved.get("id") or saved.get("invoice_id"),
                "invoice_number": saved.get("invoice_number") or ext.get("invoice_number"),
                "vendor_name": saved.get("vendor_name") or ext.get("vendor_name"),
                "amount": saved.get("amount") or ext.get("amount"),
                "status": saved.get("status", "PENDING"),
            }

        # 4. Verification result
        ver_res = None
        if state.latest_verification:
            if hasattr(state.latest_verification, "model_dump"):
                ver_res = state.latest_verification.model_dump()
            elif hasattr(state.latest_verification, "to_dict"):
                ver_res = state.latest_verification.to_dict()
            elif isinstance(state.latest_verification, dict):
                ver_res = state.latest_verification
        elif state.verification_results:
            ver_res = state.verification_results[-1]

        # Evidence Items from State
        items: List[EvidenceItem] = []
        for ev in state.evidence:
            if isinstance(ev, dict):
                try:
                    items.append(EvidenceItem.model_validate(ev))
                except Exception:
                    pass

        return InvoiceWorkflowEvidence(
            task_id=state.task_id,
            source_invoice=sanitize_evidence(source_inv),
            extracted_invoice_info=sanitize_evidence(extracted_info),
            finance_application_record=sanitize_evidence(finance_rec),
            verification_result=sanitize_evidence(ver_res),
            all_evidence_items=items,
        )

    @classmethod
    def generate_final_task_summary(cls, state: TaskState) -> str:
        """
        Produces a rich, verifiable final response string referencing all 4 key milestones.
        """
        workflow_ev = cls.build_workflow_evidence(state)
        ext = state.extracted_data or {}
        inv_num = ext.get("invoice_number", "Unknown")
        vendor = ext.get("vendor_name", "Unknown")

        header = f"Task completed successfully for invoice {inv_num} ({vendor}).\n"
        dossier_text = workflow_ev.to_summary_text()
        return f"{header}\n{dossier_text}"
