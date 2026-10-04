"""LLM-based Planner and decision maker for the Autonomous AI Task Worker."""

import json
import re
from typing import Any, Dict, List, Optional
from openai import AsyncOpenAI
from pydantic import ValidationError

from app.agent.prompts import PLANNER_SYSTEM_PROMPT, format_planner_user_prompt
from app.agent.state import ActionCall, ActionProposal, ActionType, PlannerDecision, TaskState
from app.config.logging import get_logger
from app.config.settings import settings
from app.models.schemas import RiskLevel
from app.tools.registry import ToolRegistry

logger = get_logger("agent.planner")


class Planner:
    """Decides the single next action to take given the goal, state, history, observations, and available tools."""

    def __init__(self, tool_registry: Optional[ToolRegistry] = None, client: Optional[AsyncOpenAI] = None):
        self.tool_registry = tool_registry or ToolRegistry()
        self.client = client
        if self.client is None and settings.OPENAI_API_KEY and settings.OPENAI_API_KEY != "mock-key-for-development":
            self.client = AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL,
            )

    def _clean_json_response(self, text: str) -> Dict[str, Any]:
        """Cleans and extracts JSON payload from LLM response text, rejecting free-form non-JSON."""
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        text = text.strip()

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
        else:
            raise ValueError(f"No JSON object found in response text: '{text[:100]}'")

        return json.loads(text)

    async def plan(
        self,
        user_goal: str,
        current_state: Any,
        execution_history: List[Any],
        latest_observation: Optional[Any],
        available_tools: List[Dict[str, Any]],
        known_data: Dict[str, Any],
        previous_failures: List[Any],
    ) -> PlannerDecision:
        """Determines the single next action using the LLM or autonomous heuristic policy."""
        # 1. Attempt LLM Structured Output if client is configured
        if self.client:
            try:
                decision = await self._call_llm(
                    user_goal=user_goal,
                    current_state=current_state,
                    execution_history=execution_history,
                    latest_observation=latest_observation,
                    available_tools=available_tools,
                    known_data=known_data,
                    previous_failures=previous_failures,
                )
                if decision:
                    return decision
            except Exception as e:
                logger.warning(
                    "LLM planner call failed or returned invalid output; activating autonomous fallback policy",
                    error=str(e),
                )

        # 2. Deterministic heuristic planning policy fallback
        return self._heuristic_plan(
            user_goal=user_goal,
            current_state=current_state,
            execution_history=execution_history,
            latest_observation=latest_observation,
            available_tools=available_tools,
            known_data=known_data,
            previous_failures=previous_failures,
        )

    async def _call_llm(
        self,
        user_goal: str,
        current_state: Any,
        execution_history: List[Any],
        latest_observation: Optional[Any],
        available_tools: List[Dict[str, Any]],
        known_data: Dict[str, Any],
        previous_failures: List[Any],
    ) -> PlannerDecision:
        """Invokes OpenAI LLM and enforces Pydantic structured output validation."""
        tools_schema = json.dumps(available_tools, indent=2)
        system_content = PLANNER_SYSTEM_PROMPT.format(tools_schema=tools_schema)
        user_content = format_planner_user_prompt(
            user_goal=user_goal,
            current_state=current_state,
            execution_history=execution_history,
            latest_observation=latest_observation,
            available_tools=available_tools,
            known_data=known_data,
            previous_failures=previous_failures,
        )

        response = await self.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            timeout=30.0,
        )

        raw_text = response.choices[0].message.content or ""
        parsed_dict = self._clean_json_response(raw_text)
        return PlannerDecision.model_validate(parsed_dict)

    async def decide_next_action(self, state: TaskState) -> ActionProposal:
        """Convenience method accepting TaskState and returning ActionProposal for the orchestrator."""
        latest_obs = None
        if state.observations:
            last = state.observations[-1]
            if isinstance(last, dict):
                latest_obs = last
            elif hasattr(last, "model_dump"):
                latest_obs = last.model_dump()
            elif hasattr(last, "dict"):
                latest_obs = last.dict()
            else:
                latest_obs = str(last)

        status_str = (
            state.current_status.value
            if hasattr(state.current_status, "value")
            else str(state.current_status)
        )

        failures = [
            f.model_dump() if hasattr(f, "model_dump") else (f.dict() if hasattr(f, "dict") else str(f))
            for f in state.failed_steps
        ]

        decision = await self.plan(
            user_goal=state.user_goal,
            current_state={
                "task_id": state.task_id,
                "status": status_str,
                "current_step": state.current_step,
                "max_steps": state.max_steps,
                "current_objective": state.current_objective,
                "plan_summary": state.plan_summary,
                "retry_count": state.retry_count,
            },
            execution_history=state.execution_history,
            latest_observation=latest_obs,
            available_tools=self.tool_registry.get_schemas(),
            known_data=state.extracted_data,
            previous_failures=failures,
        )

        # Update state objective and plan summary based on planner decision
        state.current_objective = decision.expected_outcome
        if not state.plan_summary:
            state.plan_summary = decision.reason

        return decision.to_action_proposal()

    def _heuristic_plan(
        self,
        user_goal: str,
        current_state: Any,
        execution_history: List[Any],
        latest_observation: Optional[Any],
        available_tools: List[Dict[str, Any]],
        known_data: Dict[str, Any],
        previous_failures: List[Any],
    ) -> PlannerDecision:
        """Autonomous heuristic decision policy tailored to document & ledger workflows."""
        goal_lower = user_goal.lower() if user_goal else ""

        # Extract retry count if available
        retry_count = 0
        if isinstance(current_state, dict):
            retry_count = current_state.get("retry_count", 0)
        elif hasattr(current_state, "retry_count"):
            retry_count = current_state.retry_count

        # ------------------------------------------------------------------
        # RULE 5 & 3: REACT TO OBSERVATION & HANDLE FAILURES
        # ------------------------------------------------------------------
        obs_data: Dict[str, Any] = {}
        obs_success = True
        obs_error = None
        if isinstance(latest_observation, dict):
            obs_success = latest_observation.get("success", True)
            obs_error = latest_observation.get("error")
            obs_data = latest_observation.get("data") or {}
        elif hasattr(latest_observation, "data"):
            obs_data = getattr(latest_observation, "data") or {}
            obs_success = getattr(latest_observation, "success", True)
            obs_error = getattr(latest_observation, "error", None)

        if not obs_success or obs_error:
            if retry_count >= 2 or len(previous_failures) >= 3:
                return PlannerDecision(
                    reason=f"Action failed repeatedly with error '{obs_error}'. Stopping execution to prevent infinite loop.",
                    action=ActionCall(tool="fail_task", arguments={"error": obs_error or "Repeated action failure"}),
                    expected_outcome="Task halted due to unrecoverable tool error.",
                    verification_needed=False,
                    requires_approval=False,
                )
            # Decide retry with alternative strategy or retry action
            return PlannerDecision(
                reason=f"Previous action encountered error '{obs_error}'. Retrying operation (attempt {retry_count + 1}).",
                action=ActionCall(
                    tool="document_search_tool" if "search" in str(obs_error).lower() else "document_read_tool",
                    arguments={"file_path": known_data.get("latest_invoice_path", "")}
                    if "document_read_tool" in str(obs_error)
                    else {"vendor_name": "Acme Corp", "find_latest": True},
                ),
                expected_outcome="Recover from transient tool error and obtain required document data.",
                verification_needed=False,
                requires_approval=False,
            )

        # Merge newly observed data into known_data if not yet present
        active_memory = dict(known_data) if known_data else {}
        if obs_data:
            if "latest_invoice_path" in obs_data and "latest_invoice_path" not in active_memory:
                active_memory["latest_invoice_path"] = obs_data["latest_invoice_path"]
            if "invoice_data" in obs_data:
                inv = obs_data["invoice_data"]
                active_memory.update(inv)
                active_memory["pdf_extracted"] = True
            elif "invoice_number" in obs_data:
                active_memory.update(obs_data)

        # Identify vendor name
        vendor = "Acme Corp"
        if "globex" in goal_lower:
            vendor = "Globex"
        elif "initech" in goal_lower:
            vendor = "Initech"
        elif "umbrella" in goal_lower:
            vendor = "Umbrella"
        elif active_memory.get("vendor_name"):
            vendor = active_memory["vendor_name"]

        # ------------------------------------------------------------------
        # STEP 5: Complete Task (Only after verified in finance!)
        # ------------------------------------------------------------------
        if active_memory.get("verified_in_finance", False):
            inv_number = active_memory.get("invoice_number", "Unknown")
            return PlannerDecision(
                reason=(
                    f"Invoice {inv_number} from {vendor} was successfully extracted, entered into finance, "
                    f"and verified to match all document fields."
                ),
                action=ActionCall(
                    tool="complete_task",
                    arguments={"summary": f"Completed processing and verification of {vendor} invoice {inv_number}."},
                ),
                expected_outcome="All goal requirements and verification criteria satisfied.",
                verification_needed=False,
                requires_approval=False,
            )

        # ------------------------------------------------------------------
        # STEP 4: Independent verification before claiming completion
        # ------------------------------------------------------------------
        if active_memory.get("invoice_entered", False):
            inv_number = active_memory.get("invoice_number")
            return PlannerDecision(
                reason=f"Invoice {inv_number} was created. Verifying ledger record independently before completing task.",
                action=ActionCall(
                    tool="finance_read_invoice_tool",
                    arguments={"invoice_number": inv_number},
                ),
                expected_outcome=f"Verify that saved invoice {inv_number} matches extracted amount and due date.",
                verification_needed=True,
                requires_approval=False,
            )

        # ------------------------------------------------------------------
        # STEP 3: Enter invoice into finance application (RISKY - Requires Approval)
        # ------------------------------------------------------------------
        if active_memory.get("pdf_extracted") or (active_memory.get("invoice_number") and active_memory.get("amount")):
            inv_number = active_memory.get("invoice_number")
            amount = float(active_memory.get("amount", 0.0))
            currency = active_memory.get("currency", "USD")
            due_date = active_memory.get("due_date", "2024-10-15")
            issue_date = (
                active_memory.get("invoice_date")
                or active_memory.get("issue_date")
                or "2024-09-28"
            )

            return PlannerDecision(
                reason=f"Invoice {inv_number} extracted. Entering record of ${amount:.2f} {currency} into internal finance application.",
                action=ActionCall(
                    tool="finance_create_invoice_tool",
                    arguments={
                        "invoice_number": inv_number,
                        "vendor_name": active_memory.get("vendor_name", vendor),
                        "amount": amount,
                        "currency": currency,
                        "issue_date": issue_date,
                        "due_date": due_date,
                        "description": f"Imported invoice {inv_number} from document repository",
                    },
                ),
                expected_outcome=f"Invoice {inv_number} successfully registered in internal finance database.",
                verification_needed=True,
                requires_approval=True,  # RISKY: Modifying financial ledgers requires human supervisor approval
            )

        # ------------------------------------------------------------------
        # RULE 6: REQUEST CLARIFICATION IF REQUIRED INFO IS MISSING
        # ------------------------------------------------------------------
        has_vendor = any(v in goal_lower for v in ["acme", "globex", "initech", "umbrella"])
        if not has_vendor and "latest_invoice_path" not in active_memory and not active_memory.get("vendor_name"):
            return PlannerDecision(
                reason="The user goal does not specify which vendor or company invoice to process.",
                action=ActionCall(
                    tool="request_clarification",
                    arguments={"question": "Please specify the vendor name (e.g. Acme Corp, Globex, Initech, Umbrella)."},
                ),
                expected_outcome="Receive vendor clarification from user to proceed with search.",
                verification_needed=False,
                requires_approval=False,
            )

        # ------------------------------------------------------------------
        # STEP 2: Read document & extract structured fields (Do NOT repeat)
        # ------------------------------------------------------------------
        if "latest_invoice_path" in active_memory:
            latest_path = active_memory["latest_invoice_path"]
            return PlannerDecision(
                reason=f"Invoice document located at {latest_path}. Need to extract invoice number, amount, and due date.",
                action=ActionCall(
                    tool="document_read_tool",
                    arguments={"file_path": latest_path},
                ),
                expected_outcome="Structured invoice fields extracted: invoice_number, amount, due_date, currency.",
                verification_needed=False,
                requires_approval=False,
            )

        # ------------------------------------------------------------------
        # STEP 1: Search for document (Do NOT repeat if already found)
        # ------------------------------------------------------------------
        return PlannerDecision(
            reason=f"Goal requires processing invoice from {vendor}. First discover invoices and identify the latest one.",
            action=ActionCall(
                tool="document_search_tool",
                arguments={"vendor_name": vendor, "find_latest": True},
            ),
            expected_outcome=f"Find all invoice PDFs for {vendor} and return the latest file path.",
            verification_needed=False,
            requires_approval=False,
        )
