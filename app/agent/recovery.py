"""Agent failure detection, classification, and recovery subsystem."""

from datetime import datetime
from enum import Enum
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.agent.state import ActionProposal, ActionType, FailureRecord, TaskState
from app.config.logging import get_logger
from app.config.settings import settings
from app.models.schemas import RiskLevel

logger = get_logger("agent.recovery")


class FailureCategory(str, Enum):
    """Fine-grained classification of execution failures."""
    TIMEOUT = "TIMEOUT"
    BROWSER_NAVIGATION_FAILURE = "BROWSER_NAVIGATION_FAILURE"
    MISSING_ELEMENT = "MISSING_ELEMENT"
    INVALID_INPUT = "INVALID_INPUT"
    AUTHENTICATION_FAILURE = "AUTHENTICATION_FAILURE"
    MALFORMED_DOCUMENT = "MALFORMED_DOCUMENT"
    MISSING_INVOICE = "MISSING_INVOICE"
    DUPLICATE_INVOICE = "DUPLICATE_INVOICE"
    FINANCE_VALIDATION_ERROR = "FINANCE_VALIDATION_ERROR"
    GENERIC_TOOL_FAILURE = "GENERIC_TOOL_FAILURE"


class RecoveryStrategy(str, Enum):
    """Categorized recovery remediation actions."""
    RETRY_SAME_ACTION = "RETRY_SAME_ACTION"
    RETRY_WITH_CORRECTED_ARGS = "RETRY_WITH_CORRECTED_ARGS"
    RE_OBSERVE_ENVIRONMENT = "RE_OBSERVE_ENVIRONMENT"
    TRY_ALTERNATIVE_TOOL = "TRY_ALTERNATIVE_TOOL"
    NAVIGATE_TO_DIFFERENT_PAGE = "NAVIGATE_TO_DIFFERENT_PAGE"
    ASK_FOR_CLARIFICATION = "ASK_FOR_CLARIFICATION"
    ABORT_UNRECOVERABLE = "ABORT_UNRECOVERABLE"


class RecoveryDecision(BaseModel):
    """Structured decision explaining how the agent will recover from an error."""
    failure_category: FailureCategory
    recovery_strategy: RecoveryStrategy
    reason: str
    recovery_action: Optional[ActionProposal] = None
    attempt_number: int = 1
    max_retries: int = 3
    is_recoverable: bool = True
    event_sequence: List[str] = Field(default_factory=list)


class FailureClassifier:
    """Analyzes error messages and execution context to diagnose the failure category."""

    @classmethod
    def classify(
        cls,
        error_message: str,
        tool_name: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> FailureCategory:
        err = (error_message or "").lower()

        # 1. Timeout
        if any(w in err for w in ["timeout", "timed out", "wait_for_selector", "waiting for locator"]):
            if any(w in err for w in ["waiting for locator", "selector", "not visible", "no element"]):
                return FailureCategory.MISSING_ELEMENT
            return FailureCategory.TIMEOUT

        # 2. Missing element (button, input, form)
        if any(w in err for w in ["waiting for locator", "selector not found", "missing element", "no node found", "button", "element"]):
            return FailureCategory.MISSING_ELEMENT

        # 3. Browser Navigation Failure
        if any(w in err for w in ["net::err", "navigation failed", "connection refused", "failed to navigate", "err_connection"]):
            return FailureCategory.BROWSER_NAVIGATION_FAILURE

        # 4. Authentication Failure
        if any(w in err for w in ["unauthenticated", "401", "login failed", "invalid credentials", "unauthorized", "redirected to login"]):
            return FailureCategory.AUTHENTICATION_FAILURE

        # 5. Duplicate Invoice
        if any(w in err for w in ["already exists", "duplicate", "unique constraint", "duplicate entry"]):
            return FailureCategory.DUPLICATE_INVOICE

        # 6. Malformed Document
        if any(w in err for w in ["cannot open file as pdf", "corrupt", "bad pdf", "malformed document", "invalid pdf header"]):
            return FailureCategory.MALFORMED_DOCUMENT

        # 7. Missing Invoice / File Not Found
        if any(w in err for w in ["file not found", "no invoices found", "does not exist", "cannot find file", "no such file"]):
            return FailureCategory.MISSING_INVOICE

        # 8. Finance Application Validation Error
        if any(w in err for w in ["finance ledger validation", "amount must be positive", "invalid date format", "422", "validation error"]):
            return FailureCategory.FINANCE_VALIDATION_ERROR

        # 9. Invalid Input
        if any(w in err for w in ["input validation failed", "validation error", "required parameter", "missing required"]):
            return FailureCategory.INVALID_INPUT

        return FailureCategory.GENERIC_TOOL_FAILURE


class RecoveryManager:
    """Orchestrates bounded, policy-driven recovery when an action fails."""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.action_retry_counts: Dict[str, int] = {}

    def decide_recovery(
        self,
        state: TaskState,
        failed_action: ActionProposal,
        error_message: str,
    ) -> RecoveryDecision:
        """Evaluates failure, bounded retries, and selects the optimal recovery strategy."""
        category = FailureClassifier.classify(
            error_message=error_message,
            tool_name=failed_action.tool_name,
        )

        action_key = f"{failed_action.tool_name or 'generic'}:{category.value}"
        current_attempts = self.action_retry_counts.get(action_key, 0) + 1
        self.action_retry_counts[action_key] = current_attempts

        event_seq = [
            f"ACTION_FAILED ({failed_action.tool_name})",
            f"OBSERVE (Diagnosed: {category.value})",
        ]

        # Check Bounded Retries Threshold
        if current_attempts > self.max_retries:
            reason = (
                f"Action '{failed_action.tool_name}' failed with {category.value} and reached maximum "
                f"retry budget / consecutive failures limit ({self.max_retries} attempts). Aborting to prevent infinite loops."
            )
            event_seq.append(f"RECOVERY_DECISION ({RecoveryStrategy.ABORT_UNRECOVERABLE.value})")
            return RecoveryDecision(
                failure_category=category,
                recovery_strategy=RecoveryStrategy.ABORT_UNRECOVERABLE,
                reason=reason,
                recovery_action=ActionProposal(
                    thought=reason,
                    action_type=ActionType.FAIL_TASK,
                    tool_name=None,
                    tool_args={"error": reason},
                    risk_level=RiskLevel.LOW,
                ),
                attempt_number=current_attempts,
                max_retries=self.max_retries,
                is_recoverable=False,
                event_sequence=event_seq,
            )

        # ---------------------------------------------------------------------
        # STRATEGY 1: DUPLICATE INVOICE RECOVERY
        # ---------------------------------------------------------------------
        if category == FailureCategory.DUPLICATE_INVOICE:
            inv_number = failed_action.tool_args.get("invoice_number")
            reason = (
                f"Invoice '{inv_number}' already exists in the finance ledger. "
                f"Recovering by verifying the existing record instead of re-creating."
            )
            event_seq.extend([
                f"RECOVERY_DECISION ({RecoveryStrategy.TRY_ALTERNATIVE_TOOL.value})",
                "ALTERNATIVE_ACTION (finance_read_invoice_tool)",
                "VERIFY",
            ])
            recovery_action = ActionProposal(
                thought=f"Verifying already existing invoice '{inv_number}' in finance ledger.",
                action_type=ActionType.CALL_TOOL,
                tool_name="finance_read_invoice_tool",
                tool_args={"invoice_number": inv_number},
                risk_level=RiskLevel.LOW,
                expected_outcome="Retrieve existing invoice record and confirm matching fields.",
                verification_needed=True,
            )
            return RecoveryDecision(
                failure_category=category,
                recovery_strategy=RecoveryStrategy.TRY_ALTERNATIVE_TOOL,
                reason=reason,
                recovery_action=recovery_action,
                attempt_number=current_attempts,
                max_retries=self.max_retries,
                is_recoverable=True,
                event_sequence=event_seq,
            )

        # ---------------------------------------------------------------------
        # STRATEGY 2: INVALID INPUT / FINANCE VALIDATION ERROR CORRECTION
        # ---------------------------------------------------------------------
        if category in (FailureCategory.FINANCE_VALIDATION_ERROR, FailureCategory.INVALID_INPUT):
            corrected_args = dict(failed_action.tool_args)
            # Correct negative or invalid amount
            if "amount" in corrected_args:
                try:
                    amt = abs(float(corrected_args["amount"]))
                    corrected_args["amount"] = round(amt, 2)
                except Exception:
                    corrected_args["amount"] = 100.0

            # Correct date formatting if issue/due date invalid
            for date_key in ["issue_date", "due_date", "invoice_date"]:
                if date_key in corrected_args:
                    raw_val = str(corrected_args[date_key])
                    # Ensure standard YYYY-MM-DD
                    match = re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", raw_val)
                    if match:
                        corrected_args[date_key] = f"{match.group(1)}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"
                    else:
                        corrected_args[date_key] = "2024-10-15"

            reason = f"Corrected invalid arguments for {failed_action.tool_name} (sanitized amount and date format)."
            event_seq.extend([
                f"RECOVERY_DECISION ({RecoveryStrategy.RETRY_WITH_CORRECTED_ARGS.value})",
                f"ALTERNATIVE_ACTION ({failed_action.tool_name})",
            ])
            return RecoveryDecision(
                failure_category=category,
                recovery_strategy=RecoveryStrategy.RETRY_WITH_CORRECTED_ARGS,
                reason=reason,
                recovery_action=ActionProposal(
                    thought=f"Retrying {failed_action.tool_name} with corrected and sanitized arguments.",
                    action_type=failed_action.action_type,
                    tool_name=failed_action.tool_name,
                    tool_args=corrected_args,
                    risk_level=failed_action.risk_level,
                    expected_outcome="Successful execution with valid payload.",
                    verification_needed=failed_action.verification_needed,
                    requires_approval=failed_action.requires_approval,
                ),
                attempt_number=current_attempts,
                max_retries=self.max_retries,
                is_recoverable=True,
                event_sequence=event_seq,
            )

        # ---------------------------------------------------------------------
        # STRATEGY 3: MISSING ELEMENT / BUTTON (Browser DOM Interaction)
        # ---------------------------------------------------------------------
        if category == FailureCategory.MISSING_ELEMENT:
            if current_attempts == 1:
                # Attempt 1: Re-observe the current environment with screenshot and text extraction
                reason = f"Element was not found on current page. Taking screenshot to re-observe DOM state."
                event_seq.extend([
                    f"RECOVERY_DECISION ({RecoveryStrategy.RE_OBSERVE_ENVIRONMENT.value})",
                    "ALTERNATIVE_ACTION (browser_tool screenshot)",
                ])
                return RecoveryDecision(
                    failure_category=category,
                    recovery_strategy=RecoveryStrategy.RE_OBSERVE_ENVIRONMENT,
                    reason=reason,
                    recovery_action=ActionProposal(
                        thought="Re-observing browser screen state to inspect available elements.",
                        action_type=ActionType.CALL_TOOL,
                        tool_name="browser_tool",
                        tool_args={"action": "screenshot", "filename": f"recovery_step_{state.current_step}.png"},
                        risk_level=RiskLevel.LOW,
                    ),
                    attempt_number=current_attempts,
                    max_retries=self.max_retries,
                    is_recoverable=True,
                    event_sequence=event_seq,
                )
            else:
                # Attempt 2+: Try alternative tool (direct API / finance tool)
                reason = "Element remains missing in browser UI. Switching to alternative direct finance tool."
                event_seq.extend([
                    f"RECOVERY_DECISION ({RecoveryStrategy.TRY_ALTERNATIVE_TOOL.value})",
                    "ALTERNATIVE_ACTION (finance_create_invoice_tool)",
                ])
                inv_data = state.extracted_data
                return RecoveryDecision(
                    failure_category=category,
                    recovery_strategy=RecoveryStrategy.TRY_ALTERNATIVE_TOOL,
                    reason=reason,
                    recovery_action=ActionProposal(
                        thought="Bypassing browser UI failure by using direct finance tool.",
                        action_type=ActionType.CALL_TOOL,
                        tool_name="finance_create_invoice_tool",
                        tool_args={
                            "invoice_number": inv_data.get("invoice_number", "INV-REC-001"),
                            "vendor_name": inv_data.get("vendor_name", "Acme Corp"),
                            "amount": float(inv_data.get("amount", 100.0)),
                            "due_date": inv_data.get("due_date", "2024-10-15"),
                        },
                        risk_level=RiskLevel.HIGH,
                        requires_approval=True,
                    ),
                    attempt_number=current_attempts,
                    max_retries=self.max_retries,
                    is_recoverable=True,
                    event_sequence=event_seq,
                )

        # ---------------------------------------------------------------------
        # STRATEGY 4: BROWSER NAVIGATION / AUTHENTICATION FAILURE
        # ---------------------------------------------------------------------
        if category in (FailureCategory.BROWSER_NAVIGATION_FAILURE, FailureCategory.AUTHENTICATION_FAILURE):
            reason = "Browser navigation or session authentication failed. Navigating back to login page."
            event_seq.extend([
                f"RECOVERY_DECISION ({RecoveryStrategy.NAVIGATE_TO_DIFFERENT_PAGE.value})",
                "ALTERNATIVE_ACTION (browser_tool open_url login)",
            ])
            return RecoveryDecision(
                failure_category=category,
                recovery_strategy=RecoveryStrategy.NAVIGATE_TO_DIFFERENT_PAGE,
                reason=reason,
                recovery_action=ActionProposal(
                    thought="Navigating to finance login page to restore session.",
                    action_type=ActionType.CALL_TOOL,
                    tool_name="browser_tool",
                    tool_args={"action": "open_url", "url": "http://127.0.0.1:8000/finance/login"},
                    risk_level=RiskLevel.LOW,
                ),
                attempt_number=current_attempts,
                max_retries=self.max_retries,
                is_recoverable=True,
                event_sequence=event_seq,
            )

        # ---------------------------------------------------------------------
        # STRATEGY 5: MISSING INVOICE / MALFORMED DOCUMENT
        # ---------------------------------------------------------------------
        if category in (FailureCategory.MISSING_INVOICE, FailureCategory.MALFORMED_DOCUMENT):
            if current_attempts == 1 and failed_action.tool_name == "document_read_tool":
                # Relax search or find another document
                reason = "Specified invoice PDF is missing or malformed. Re-searching document directory for alternatives."
                event_seq.extend([
                    f"RECOVERY_DECISION ({RecoveryStrategy.TRY_ALTERNATIVE_TOOL.value})",
                    "ALTERNATIVE_ACTION (document_search_tool)",
                ])
                return RecoveryDecision(
                    failure_category=category,
                    recovery_strategy=RecoveryStrategy.TRY_ALTERNATIVE_TOOL,
                    reason=reason,
                    recovery_action=ActionProposal(
                        thought="Re-running document search with broad query.",
                        action_type=ActionType.CALL_TOOL,
                        tool_name="document_search_tool",
                        tool_args={"vendor_name": state.extracted_data.get("vendor_name", "Acme"), "find_latest": True},
                        risk_level=RiskLevel.LOW,
                    ),
                    attempt_number=current_attempts,
                    max_retries=self.max_retries,
                    is_recoverable=True,
                    event_sequence=event_seq,
                )
            else:
                reason = f"Invoice document unavailable or corrupt. Requesting user clarification."
                event_seq.extend([
                    f"RECOVERY_DECISION ({RecoveryStrategy.ASK_FOR_CLARIFICATION.value})",
                    "ALTERNATIVE_ACTION (request_clarification)",
                ])
                return RecoveryDecision(
                    failure_category=category,
                    recovery_strategy=RecoveryStrategy.ASK_FOR_CLARIFICATION,
                    reason=reason,
                    recovery_action=ActionProposal(
                        thought=reason,
                        action_type=ActionType.REQUEST_CLARIFICATION,
                        tool_name=None,
                        tool_args={"question": "The specified invoice file was not found or is corrupt. Please provide a valid invoice document."},
                        risk_level=RiskLevel.LOW,
                    ),
                    attempt_number=current_attempts,
                    max_retries=self.max_retries,
                    is_recoverable=True,
                    event_sequence=event_seq,
                )

        # ---------------------------------------------------------------------
        # STRATEGY 6: TIMEOUT / GENERIC TEMPORARY TOOL FAILURE
        # ---------------------------------------------------------------------
        reason = f"Transient {category.value} encountered on {failed_action.tool_name}. Retrying action (attempt {current_attempts}/{self.max_retries})."
        event_seq.extend([
            f"RECOVERY_DECISION ({RecoveryStrategy.RETRY_SAME_ACTION.value})",
            f"ALTERNATIVE_ACTION ({failed_action.tool_name})",
        ])
        return RecoveryDecision(
            failure_category=category,
            recovery_strategy=RecoveryStrategy.RETRY_SAME_ACTION,
            reason=reason,
            recovery_action=ActionProposal(
                thought=reason,
                action_type=failed_action.action_type,
                tool_name=failed_action.tool_name,
                tool_args=failed_action.tool_args,
                risk_level=failed_action.risk_level,
                expected_outcome=f"Successful execution on retry attempt {current_attempts}.",
                verification_needed=failed_action.verification_needed,
                requires_approval=failed_action.requires_approval,
            ),
            attempt_number=current_attempts,
            max_retries=self.max_retries,
            is_recoverable=True,
            event_sequence=event_seq,
        )
