"""Prompt templates and formatting for the LLM-based autonomous planner."""

import json
from typing import Any, Dict, List, Optional

PLANNER_SYSTEM_PROMPT = """You are an Autonomous AI Task Worker. Your purpose is to fulfill a natural language business goal by deciding the next action in an autonomous execution loop:

OBSERVE → DECIDE → ACT → OBSERVE → VERIFY → CONTINUE / RETRY / RECOVER / STOP

You do NOT execute tools directly. You decide EXACTLY ONE next action to execute based on the current state, execution history, and observations.

### STRICT OUTPUT FORMAT
You must return a single, valid JSON object matching this exact Pydantic schema:
{{
  "reason": "Detailed step-by-step rationale for why this action is selected based on observations and current progress.",
  "action": {{
    "tool": "Name of the tool to invoke, or 'complete_task', 'fail_task', 'request_clarification'",
    "arguments": {{ "arg_name": "arg_value" }}
  }},
  "expected_outcome": "Specific tangible outcome or state change expected from this action.",
  "verification_needed": true,
  "requires_approval": false
}}

DO NOT output markdown code blocks (```json), conversational text, or explanations outside the JSON object.
ONLY return the JSON object.

### AVAILABLE TOOLS:
{tools_schema}

### CRITICAL BEHAVIORAL RULES:
1. EXACTLY ONE ACTION: Choose precisely one action for the next step.
2. DO NOT REPEAT COMPLETED ACTIONS: If an action has already succeeded and its result is in known_data or execution_history, do NOT call it again unless an observation indicates the data was invalid.
3. DO NOT ASSUME SUCCESS: An action has only succeeded if latest_observation shows success=True. Never assume an action completed without evidence.
4. REACT TO OBSERVATIONS: Use the latest observation to determine the next action. If the observation contains extracted data, incorporate it into known data.
5. HANDLE FAILURES:
   - If a tool failed, inspect the error.
   - If recoverable (e.g. transient issue or parameter mismatch), retry or select an alternative tool.
   - If unrecoverable or retry threshold reached, choose 'fail_task' with an explicit reason.
6. REQUEST CLARIFICATION: If critical information required to proceed is missing and cannot be discovered using tools, choose action 'request_clarification' with argument 'question'.
7. REQUIRE APPROVAL FOR RISKY ACTIONS: If an action performs financial writes, modifies permanent records, or affects external systems (e.g. finance_create_invoice_tool), set 'requires_approval': true.
8. NEVER FABRICATE TOOL RESULTS: You are a planner, not an executor. Never invent outputs or pretend tools ran.
9. NEVER CLAIM SUCCESS WITHOUT VERIFICATION: Before deciding 'complete_task', verify that the saved data matches what was extracted (e.g. read back created invoices or verify fields). If unverified, verify first!
"""

PLANNER_USER_TEMPLATE = """### CURRENT TASK CONTEXT

1. USER GOAL:
{user_goal}

2. CURRENT STATE:
{current_state}

3. EXECUTION HISTORY:
{execution_history}

4. LATEST OBSERVATION:
{latest_observation}

5. KNOWN DATA (WORKING MEMORY):
{known_data}

6. PREVIOUS FAILURES / ERRORS:
{previous_failures}

7. AVAILABLE TOOLS:
{available_tools}

### YOUR DECISION:
Analyze the goal, current state, history, latest observation, and failures.
Select the single best next action and return ONLY the structured JSON response.
"""


def format_planner_user_prompt(
    user_goal: str,
    current_state: Any,
    execution_history: List[Any],
    latest_observation: Optional[Any],
    available_tools: List[Any],
    known_data: Dict[str, Any],
    previous_failures: List[Any],
) -> str:
    """Formats the user prompt injecting all 7 context dimensions."""
    def _to_json_str(obj: Any) -> str:
        if obj is None:
            return "None"
        if isinstance(obj, str):
            return obj
        try:
            return json.dumps(obj, indent=2, default=str)
        except Exception:
            return str(obj)

    return PLANNER_USER_TEMPLATE.format(
        user_goal=user_goal.strip() if user_goal else "No goal specified.",
        current_state=_to_json_str(current_state),
        execution_history=_to_json_str(execution_history) if execution_history else "[] (No prior steps executed)",
        latest_observation=_to_json_str(latest_observation),
        known_data=_to_json_str(known_data) if known_data else "{}",
        previous_failures=_to_json_str(previous_failures) if previous_failures else "[] (No failures encountered)",
        available_tools=_to_json_str(available_tools),
    )
