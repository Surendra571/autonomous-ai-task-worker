"""Verification strategies for independent task outcome validation."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.state import TaskState, VerificationResult
from app.config.logging import get_logger
from app.verification.verifier import OutcomeVerifier

logger = get_logger("verification.strategy")


class VerificationStrategy(ABC):
    """Abstract interface defining independent verification for task outcomes."""

    @abstractmethod
    def can_verify(self, task: Any, state: TaskState) -> bool:
        """Determines whether this strategy can verify the given task/state."""
        pass

    @abstractmethod
    async def verify(
        self,
        task: Any,
        state: TaskState,
        db: Optional[AsyncSession] = None,
    ) -> VerificationResult:
        """Independently verifies whether the goal criteria have been met."""
        pass


class InvoiceVerificationStrategy(VerificationStrategy):
    """Dedicated verification strategy for accounts payable and invoice ledger workflows."""

    def can_verify(self, task: Any, state: TaskState) -> bool:
        """Applies when invoice creation, mutation, or ledger entry workflow was performed or recorded."""
        if state.extracted_data.get("invoice_entered"):
            return True
        if "finance_create_invoice_tool" in state.completed_steps:
            return True
        goal_lower = str(state.user_goal).lower()
        is_entry_goal = any(w in goal_lower for w in ["enter", "create", "save", "submit", "record", "post"])
        if is_entry_goal and ("invoice" in goal_lower or "invoice_number" in state.extracted_data):
            return True
        if "invoice_number" in state.extracted_data and state.extracted_data.get("saved_record"):
            return True
        return False

    async def verify(
        self,
        task: Any,
        state: TaskState,
        db: Optional[AsyncSession] = None,
    ) -> VerificationResult:
        """Executes 7-point deep verification against the database and source invoice."""
        logger.info(
            "Executing InvoiceVerificationStrategy 7-point cross-check",
            task_id=state.task_id,
            invoice_number=state.extracted_data.get("invoice_number"),
        )
        report = await OutcomeVerifier.verify_saved_invoice_in_db(
            db=db,
            task_id=state.task_id,
            expected_data=state.extracted_data,
        )
        return report


class GenericToolVerificationStrategy(VerificationStrategy):
    """General-purpose verification strategy validating declared goal criteria and tool outputs."""

    def can_verify(self, task: Any, state: TaskState) -> bool:
        """Fallback strategy that can evaluate any tool-based task."""
        return True

    async def verify(
        self,
        task: Any,
        state: TaskState,
        db: Optional[AsyncSession] = None,
    ) -> VerificationResult:
        """Validates that execution succeeded, necessary facts were acquired, and no unrecovered failures exist."""
        logger.info(
            "Executing GenericToolVerificationStrategy",
            task_id=state.task_id,
            completed_steps=len(state.completed_steps),
        )
        discrepancies: List[str] = []
        checks: List[Dict[str, Any]] = []

        # Check 1: Action Execution Integrity
        has_completed_steps = len(state.completed_steps) > 0
        checks.append({
            "name": "actions_executed_successfully",
            "passed": has_completed_steps,
            "details": f"Completed steps: {len(state.completed_steps)}",
        })
        if not has_completed_steps:
            discrepancies.append("No successful tool execution steps recorded.")

        # Check 2: Working Memory Facts Acquired
        has_facts = len(state.extracted_data) > 0
        checks.append({
            "name": "target_facts_acquired",
            "passed": has_facts,
            "details": f"Acquired facts count: {len(state.extracted_data)}",
        })
        if not has_facts:
            discrepancies.append("No objective domain facts or query results were gathered in working memory.")

        # Check 3: Clean Execution State (No unrecovered errors)
        no_failures = state.consecutive_failures == 0
        checks.append({
            "name": "zero_unrecovered_failures",
            "passed": no_failures,
            "details": f"Consecutive failures: {state.consecutive_failures}",
        })
        if not no_failures:
            discrepancies.append(f"Task halted with {state.consecutive_failures} unrecovered consecutive failures.")

        # Check 4: Latest Tool Outcome Status
        latest_tool_success = False
        if state.tool_results:
            latest_tool_success = bool(state.tool_results[-1].get("success", False))
        elif state.observations:
            latest_tool_success = bool(getattr(state.observations[-1], "success", False))
        else:
            latest_tool_success = has_completed_steps

        checks.append({
            "name": "latest_action_succeeded",
            "passed": latest_tool_success,
            "details": f"Latest action status: {'success' if latest_tool_success else 'failed'}",
        })
        if not latest_tool_success:
            discrepancies.append("Latest tool invocation failed or returned an error.")

        all_passed = all(c["passed"] for c in checks)

        return VerificationResult(
            verified=all_passed,
            verification_type="generic_objective_verification",
            checks=checks,
            details={
                "extracted_keys": list(state.extracted_data.keys()),
                "completed_steps": state.completed_steps,
            },
            discrepancies=discrepancies,
        )


class VerificationStrategyRegistry:
    """Registry managing available verification strategies."""

    def __init__(self, strategies: Optional[List[VerificationStrategy]] = None):
        self.strategies: List[VerificationStrategy] = strategies or [
            InvoiceVerificationStrategy(),
            GenericToolVerificationStrategy(),
        ]

    def register(self, strategy: VerificationStrategy, prepend: bool = True) -> None:
        """Registers a new domain-specific verification strategy."""
        if prepend:
            self.strategies.insert(0, strategy)
        else:
            self.strategies.append(strategy)

    def select_strategy(self, task: Any, state: TaskState) -> VerificationStrategy:
        """Selects the first matching strategy capable of verifying the task outcome."""
        for strategy in self.strategies:
            if strategy.can_verify(task, state):
                return strategy
        return self.strategies[-1]
