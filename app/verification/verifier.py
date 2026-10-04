"""Independent outcome verification system for cross-validating task results."""

from typing import Any, Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.state import VerificationResult
from app.config.logging import get_logger
from app.services.task_service import TaskService
from app.tools.finance_read_invoice_tool import FinanceReadInvoiceTool
from app.tools.finance_search_tool import FinanceSearchTool

logger = get_logger("verification.verifier")


class OutcomeVerifier:
    """Verifies that actions produced the expected real-world system state."""

    @classmethod
    def verify_invoice_fields(
        cls,
        expected_data: Dict[str, Any],
        actual_data: Dict[str, Any],
    ) -> VerificationResult:
        """Cross-checks fields between the extracted source invoice and the saved ledger record."""
        discrepancies: List[str] = []

        # 1. Invoice Number
        expected_num = str(expected_data.get("invoice_number", "")).strip().upper()
        actual_num = str(actual_data.get("invoice_number", "")).strip().upper()
        if expected_num != actual_num:
            discrepancies.append(
                f"Invoice number mismatch: expected '{expected_num}', actual '{actual_num}'"
            )

        # 2. Vendor Name
        expected_vendor = str(expected_data.get("vendor_name", "")).strip().lower()
        actual_vendor = str(actual_data.get("vendor_name", "")).strip().lower()
        if expected_vendor and actual_vendor and expected_vendor not in actual_vendor and actual_vendor not in expected_vendor:
            discrepancies.append(
                f"Vendor name mismatch: expected '{expected_data.get('vendor_name')}', actual '{actual_data.get('vendor_name')}'"
            )

        # 3. Total Amount
        try:
            expected_amt = float(expected_data.get("amount", 0.0))
            actual_amt = float(actual_data.get("amount", 0.0))
            if abs(expected_amt - actual_amt) > 0.01:
                discrepancies.append(
                    f"Amount mismatch: expected {expected_amt:.2f}, actual {actual_amt:.2f}"
                )
        except (ValueError, TypeError) as e:
            discrepancies.append(f"Unable to compare amounts numerically: {str(e)}")

        # 4. Due Date
        expected_due = str(expected_data.get("due_date", "")).strip()
        actual_due = str(actual_data.get("due_date", "")).strip()
        if expected_due and actual_due and expected_due != actual_due:
            discrepancies.append(
                f"Due date mismatch: expected '{expected_due}', actual '{actual_due}'"
            )

        passed = len(discrepancies) == 0

        details = {
            "expected": expected_data,
            "actual": actual_data,
            "discrepancies": discrepancies,
            "validation_timestamp": str(logger),
        }

        return VerificationResult(
            verified=passed,
            verification_type="independent_field_cross_check",
            details=details,
            discrepancies=discrepancies,
        )

    @classmethod
    async def verify_saved_invoice_in_db(
        cls,
        db: AsyncSession,
        task_id: str,
        expected_data: Dict[str, Any],
        invoice_number: Optional[str] = None,
        invoice_id: Optional[str] = None,
    ) -> VerificationResult:
        """Independently queries the finance ledger and verifies the record against expected source data."""
        target_number = invoice_number or expected_data.get("invoice_number")
        target_id = invoice_id or expected_data.get("invoice_id")

        read_tool = FinanceReadInvoiceTool()
        actual_record: Optional[Dict[str, Any]] = None

        if target_id or target_number:
            tool_res = await read_tool.run(
                invoice_id=target_id,
                invoice_number=target_number,
                session=db,
            )
            if tool_res.success and tool_res.data:
                actual_record = tool_res.data

        if not actual_record and target_number:
            search_tool = FinanceSearchTool()
            search_res = await search_tool.run(query=target_number, session=db)
            if search_res.success and search_res.data and search_res.data.get("invoices"):
                actual_record = search_res.data["invoices"][0]

        if not actual_record:
            res = VerificationResult(
                verified=False,
                verification_type="finance_ledger_lookup",
                details={"target_number": target_number, "target_id": target_id},
                discrepancies=[f"Could not locate invoice with number '{target_number}' in finance ledger."],
            )
            # Record failed verification evidence
            await TaskService.add_evidence(
                db=db,
                task_id=task_id,
                evidence_type="verification_failure",
                description=f"Independent verification failed: Invoice '{target_number}' not found in ledger.",
                payload=res.details,
            )
            return res

        # Run field-by-field cross validation
        result = cls.verify_invoice_fields(expected_data=expected_data, actual_data=actual_record)

        # Record verification evidence in DB
        evidence_desc = (
            f"Independent verification {'PASSED' if result.verified else 'FAILED'} for invoice '{actual_record.get('invoice_number')}'."
        )
        await TaskService.add_evidence(
            db=db,
            task_id=task_id,
            evidence_type="outcome_verification",
            description=evidence_desc,
            payload={
                "verified": result.verified,
                "discrepancies": result.discrepancies,
                "record": actual_record,
            },
        )

        return result
