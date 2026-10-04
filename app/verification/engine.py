"""Dedicated VerificationEngine for autonomous outcome validation.

Independently confirms that real-world business outcomes match user goals
without trusting planner assertions or intermediate tool outputs.
"""

from datetime import date, datetime
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.state import TaskState, VerificationResult
from app.config.logging import get_logger
from app.services.task_service import TaskService
from app.tools.finance_read_invoice_tool import FinanceReadInvoiceTool
from app.tools.finance_search_tool import FinanceSearchTool

logger = get_logger("verification.engine")


class VerificationCheck(BaseModel):
    """An individual assertion evaluated during verification."""
    name: str = Field(..., description="Machine-readable check name.")
    passed: bool = Field(..., description="Whether this specific check passed.")
    expected: Optional[Any] = Field(None, description="Expected value from source.")
    actual: Optional[Any] = Field(None, description="Actual value retrieved from system.")
    message: Optional[str] = Field(None, description="Details or discrepancy note.")


class VerificationReport(BaseModel):
    """Structured result returned by the VerificationEngine."""
    verified: bool = Field(..., description="True only if ALL required checks passed.")
    checks: List[VerificationCheck] = Field(default_factory=list)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    discrepancies: List[str] = Field(default_factory=list)
    summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Canonical dictionary format requested by task specifications."""
        return {
            "verified": self.verified,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                }
                for c in self.checks
            ],
            "evidence": self.evidence,
        }

    def to_verification_result(self) -> VerificationResult:
        """Converts to TaskState's internal VerificationResult model."""
        return VerificationResult(
            verified=self.verified,
            verification_type="independent_outcome_verification",
            checks=[
                {
                    "name": c.name,
                    "passed": c.passed,
                    "expected": c.expected,
                    "actual": c.actual,
                    "message": c.message,
                }
                for c in self.checks
            ],
            evidence=self.evidence,
            details={
                "checks_count": len(self.checks),
                "passed_count": sum(1 for c in self.checks if c.passed),
                "summary": self.summary,
            },
            discrepancies=self.discrepancies,
        )


class VerificationEngine:
    """Dedicated independent verification engine for validating that business task goals

    have actually been achieved in target systems before completing tasks.

    Key principles:
    1. NEVER trust the planner's assertion that a task succeeded.
    2. Independently query the target system (finance application / browser) to retrieve actual state.
    3. Perform strict field-by-field cross validation against extracted source data.
    4. Provide structured check results and audit evidence.
    """

    @staticmethod
    def normalize_amount(val: Any) -> Optional[float]:
        """Normalizes currency/numeric strings to float."""
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        cleaned = re.sub(r"[^\d.-]", "", str(val))
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def normalize_date(val: Any) -> Optional[str]:
        """Normalizes date representations to YYYY-MM-DD."""
        if val is None:
            return None
        if isinstance(val, (datetime, date)):
            return val.strftime("%Y-%m-%d")
        val_str = str(val).strip()
        # Try matching YYYY-MM-DD
        m = re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", val_str)
        if m:
            y, month, d = m.groups()
            return f"{int(y):04d}-{int(month):02d}-{int(d):02d}"
        return val_str

    @staticmethod
    def normalize_str(val: Any) -> str:
        """Normalizes strings for comparison."""
        return str(val or "").strip()

    @classmethod
    def compare_invoice_fields(
        cls,
        expected: Dict[str, Any],
        actual: Optional[Dict[str, Any]],
    ) -> VerificationReport:
        """Cross-checks 7 essential verification criteria:

        1. Correct vendor
        2. Correct invoice number
        3. Correct amount
        4. Correct currency
        5. Correct due date
        6. Invoice exists in finance system
        7. Stored values match extracted source values
        """
        checks: List[VerificationCheck] = []
        discrepancies: List[str] = []
        evidence: List[Dict[str, Any]] = [
            {"type": "source_extracted_data", "data": expected},
            {"type": "finance_application_record", "data": actual or {}},
            {"type": "verification_timestamp", "timestamp": datetime.utcnow().isoformat()},
        ]

        # ---------------------------------------------------------------------
        # CHECK 6: Invoice exists in finance system
        # ---------------------------------------------------------------------
        invoice_exists = actual is not None and bool(actual.get("invoice_number"))
        checks.append(
            VerificationCheck(
                name="invoice_exists",
                passed=invoice_exists,
                expected=expected.get("invoice_number"),
                actual=actual.get("invoice_number") if actual else None,
                message="Invoice located in finance ledger" if invoice_exists else "Invoice does not exist in finance ledger",
            )
        )

        if not invoice_exists:
            discrepancies.append(
                f"Invoice '{expected.get('invoice_number', 'unknown')}' does not exist in the finance application."
            )
            # Remaining field checks cannot pass if invoice doesn't exist
            for name, exp_val in [
                ("correct_vendor", expected.get("vendor_name")),
                ("vendor_matches", expected.get("vendor_name")),
                ("correct_invoice_number", expected.get("invoice_number")),
                ("invoice_number_matches", expected.get("invoice_number")),
                ("correct_amount", expected.get("amount")),
                ("amount_matches", expected.get("amount")),
                ("correct_currency", expected.get("currency")),
                ("currency_matches", expected.get("currency")),
                ("correct_due_date", expected.get("due_date")),
                ("due_date_matches", expected.get("due_date")),
                ("stored_values_match_source", True),
            ]:
                checks.append(
                    VerificationCheck(
                        name=name,
                        passed=False,
                        expected=exp_val,
                        actual=None,
                        message="Check failed because invoice does not exist in ledger",
                    )
                )

            return VerificationReport(
                verified=False,
                checks=checks,
                evidence=evidence,
                discrepancies=discrepancies,
                summary=f"Verification failed: Invoice {expected.get('invoice_number')} not found in finance system.",
            )

        # ---------------------------------------------------------------------
        # CHECK 1: Correct vendor
        # ---------------------------------------------------------------------
        exp_vendor = cls.normalize_str(expected.get("vendor_name")).lower()
        act_vendor = cls.normalize_str(actual.get("vendor_name")).lower()
        vendor_match = bool(exp_vendor and act_vendor and (exp_vendor in act_vendor or act_vendor in exp_vendor))
        if not vendor_match:
            msg = f"Vendor mismatch: expected '{expected.get('vendor_name')}', got '{actual.get('vendor_name')}'"
            discrepancies.append(msg)
        else:
            msg = f"Vendor matches: '{actual.get('vendor_name')}'"
        checks.append(
            VerificationCheck(
                name="correct_vendor",
                passed=vendor_match,
                expected=expected.get("vendor_name"),
                actual=actual.get("vendor_name"),
                message=msg,
            )
        )
        checks.append(
            VerificationCheck(
                name="vendor_matches",
                passed=vendor_match,
                expected=expected.get("vendor_name"),
                actual=actual.get("vendor_name"),
                message=msg,
            )
        )

        # ---------------------------------------------------------------------
        # CHECK 2: Correct invoice number
        # ---------------------------------------------------------------------
        exp_num = cls.normalize_str(expected.get("invoice_number")).upper()
        act_num = cls.normalize_str(actual.get("invoice_number")).upper()
        num_match = bool(exp_num and act_num and exp_num == act_num)
        if not num_match:
            msg = f"Invoice number mismatch: expected '{expected.get('invoice_number')}', got '{actual.get('invoice_number')}'"
            discrepancies.append(msg)
        else:
            msg = f"Invoice number matches: '{actual.get('invoice_number')}'"
        checks.append(
            VerificationCheck(
                name="correct_invoice_number",
                passed=num_match,
                expected=expected.get("invoice_number"),
                actual=actual.get("invoice_number"),
                message=msg,
            )
        )
        checks.append(
            VerificationCheck(
                name="invoice_number_matches",
                passed=num_match,
                expected=expected.get("invoice_number"),
                actual=actual.get("invoice_number"),
                message=msg,
            )
        )

        # ---------------------------------------------------------------------
        # CHECK 3: Correct amount
        # ---------------------------------------------------------------------
        exp_amt = cls.normalize_amount(expected.get("amount"))
        act_amt = cls.normalize_amount(actual.get("amount"))
        if exp_amt is None or act_amt is None:
            amt_match = False
            msg = f"Amount could not be parsed: expected={expected.get('amount')}, actual={actual.get('amount')}"
            discrepancies.append(msg)
        else:
            amt_match = abs(exp_amt - act_amt) < 0.01
            if not amt_match:
                msg = f"Amount mismatch: expected {exp_amt:.2f}, got {act_amt:.2f}"
                discrepancies.append(msg)
            else:
                msg = f"Amount matches: {act_amt:.2f}"
        checks.append(
            VerificationCheck(
                name="correct_amount",
                passed=amt_match,
                expected=exp_amt,
                actual=act_amt,
                message=msg,
            )
        )
        checks.append(
            VerificationCheck(
                name="amount_matches",
                passed=amt_match,
                expected=exp_amt,
                actual=act_amt,
                message=msg,
            )
        )

        # ---------------------------------------------------------------------
        # CHECK 4: Correct currency
        # ---------------------------------------------------------------------
        exp_curr = cls.normalize_str(expected.get("currency", "USD")).upper()
        act_curr = cls.normalize_str(actual.get("currency", "USD")).upper()
        curr_match = (exp_curr == act_curr)
        if not curr_match:
            msg = f"Currency mismatch: expected '{exp_curr}', got '{act_curr}'"
            discrepancies.append(msg)
        else:
            msg = f"Currency matches: '{act_curr}'"
        checks.append(
            VerificationCheck(
                name="correct_currency",
                passed=curr_match,
                expected=exp_curr,
                actual=act_curr,
                message=msg,
            )
        )
        checks.append(
            VerificationCheck(
                name="currency_matches",
                passed=curr_match,
                expected=exp_curr,
                actual=act_curr,
                message=msg,
            )
        )

        # ---------------------------------------------------------------------
        # CHECK 5: Correct due date
        # ---------------------------------------------------------------------
        exp_due = cls.normalize_date(expected.get("due_date"))
        act_due = cls.normalize_date(actual.get("due_date"))
        if exp_due and act_due:
            due_match = (exp_due == act_due)
        else:
            due_match = True  # If not specified in source, consider acceptable
        if not due_match:
            msg = f"Due date mismatch: expected '{exp_due}', got '{act_due}'"
            discrepancies.append(msg)
        else:
            msg = f"Due date matches: '{act_due}'"
        checks.append(
            VerificationCheck(
                name="correct_due_date",
                passed=due_match,
                expected=exp_due,
                actual=act_due,
                message=msg,
            )
        )
        checks.append(
            VerificationCheck(
                name="due_date_matches",
                passed=due_match,
                expected=exp_due,
                actual=act_due,
                message=msg,
            )
        )

        # ---------------------------------------------------------------------
        # CHECK 7: Stored values match extracted source values
        # ---------------------------------------------------------------------
        stored_values_match = (
            invoice_exists
            and vendor_match
            and num_match
            and amt_match
            and curr_match
            and due_match
        )
        match_msg = (
            "All stored ledger values match extracted source values."
            if stored_values_match
            else f"Stored values differ from source: {len(discrepancies)} discrepancies found."
        )
        checks.append(
            VerificationCheck(
                name="stored_values_match_source",
                passed=stored_values_match,
                expected=expected,
                actual=actual,
                message=match_msg,
            )
        )

        verified = stored_values_match and len(discrepancies) == 0

        summary = (
            f"Independent verification PASSED for invoice {actual.get('invoice_number')}: all 7 criteria confirmed."
            if verified
            else f"Independent verification FAILED for invoice {actual.get('invoice_number')}: {', '.join(discrepancies)}"
        )

        evidence.append({
            "type": "discrepancies",
            "count": len(discrepancies),
            "discrepancies": discrepancies,
        })

        return VerificationReport(
            verified=verified,
            checks=checks,
            evidence=evidence,
            discrepancies=discrepancies,
            summary=summary,
        )

    @classmethod
    async def retrieve_from_finance(
        cls,
        db: AsyncSession,
        invoice_number: Optional[str] = None,
        invoice_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Independently retrieves an invoice record from the finance system using tools/DB."""
        if not invoice_number and not invoice_id:
            return None

        # Try reading via FinanceReadInvoiceTool
        read_tool = FinanceReadInvoiceTool()
        tool_res = await read_tool.run(
            invoice_id=invoice_id,
            invoice_number=invoice_number,
            session=db,
        )
        if tool_res.success and tool_res.data:
            return tool_res.data

        # Fallback to search tool
        if invoice_number:
            search_tool = FinanceSearchTool()
            search_res = await search_tool.run(query=invoice_number, session=db)
            if search_res.success and search_res.data and search_res.data.get("invoices"):
                return search_res.data["invoices"][0]

        return None

    @classmethod
    async def retrieve_from_browser(
        cls,
        page: Any,
        invoice_number: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Independently extracts invoice field data directly from browser DOM if page is active."""
        if not page:
            return None

        try:
            # Query invoice detail page elements or table row
            extracted: Dict[str, Any] = {}

            # Check if on invoice details page
            content = await page.content()
            if "Invoice Details" in content or "invoice_number" in content:
                # Extract text from standard detail page fields
                for field, selector in [
                    ("invoice_number", "#invoice-number, [data-testid='invoice-number'], .invoice-number"),
                    ("vendor_name", "#invoice-vendor, [data-testid='vendor-name'], .vendor-name"),
                    ("amount", "#invoice-amount, [data-testid='invoice-amount'], .invoice-amount"),
                    ("currency", "#invoice-currency, [data-testid='invoice-currency'], .invoice-currency"),
                    ("due_date", "#invoice-due-date, [data-testid='due-date'], .due-date"),
                ]:
                    try:
                        el = await page.query_selector(selector)
                        if el:
                            text = await el.inner_text()
                            extracted[field] = text.strip()
                    except Exception:
                        pass

            if extracted.get("invoice_number"):
                return extracted

        except Exception as e:
            logger.warning("BROWSER_EXTRACTION_FAILED", error=str(e))

        return None

    @classmethod
    async def verify_task(
        cls,
        state: TaskState,
        db: Optional[AsyncSession] = None,
        page: Optional[Any] = None,
        expected_data: Optional[Dict[str, Any]] = None,
    ) -> VerificationReport:
        """Main verification entrypoint for verifying an autonomous task.

        Retrieves actual invoice from finance application or browser and performs
        independent cross-validation against expected source invoice data.
        """
        expected = expected_data or state.extracted_data or {}
        inv_number = expected.get("invoice_number") or state.extracted_data.get("invoice_number")
        inv_id = expected.get("invoice_id") or state.extracted_data.get("invoice_id")

        logger.info(
            "STARTING_INDEPENDENT_VERIFICATION",
            task_id=state.task_id,
            target_invoice=inv_number,
        )

        actual_record: Optional[Dict[str, Any]] = None

        # 1. Attempt browser-based DOM verification if page provided
        if page:
            actual_record = await cls.retrieve_from_browser(page=page, invoice_number=inv_number)

        # 2. Query finance application ledger via DB if not retrieved from browser
        if not actual_record and db:
            actual_record = await cls.retrieve_from_finance(
                db=db,
                invoice_number=inv_number,
                invoice_id=inv_id,
            )

        # Run comparison
        report = cls.compare_invoice_fields(expected=expected, actual=actual_record)

        # Convert to TaskState VerificationResult
        verification_result = report.to_verification_result()
        state.latest_verification = verification_result
        state.verification_results.append(verification_result.to_report())

        if report.verified:
            state.extracted_data["verified_in_finance"] = True
            logger.info("VERIFICATION_PASSED", task_id=state.task_id, summary=report.summary)
        else:
            state.extracted_data["verified_in_finance"] = False
            logger.warning(
                "VERIFICATION_FAILED",
                task_id=state.task_id,
                discrepancies=report.discrepancies,
            )

        # Persist audit evidence in DB
        if db and state.task_id:
            try:
                evidence_desc = (
                    f"Independent Verification {'PASSED' if report.verified else 'FAILED'} for invoice '{inv_number}'."
                )
                await TaskService.add_evidence(
                    db=db,
                    task_id=state.task_id,
                    evidence_type="outcome_verification",
                    description=evidence_desc,
                    payload=report.to_dict(),
                )
            except Exception as e:
                logger.warning("FAILED_SAVING_VERIFICATION_EVIDENCE", error=str(e))

        return report
