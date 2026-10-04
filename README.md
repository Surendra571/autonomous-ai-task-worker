# Autonomous AI Task Worker

An autonomous, goal-directed AI agent designed to execute real-world enterprise operations. Rather than acting as a simple chatbot or executing rigid hardcoded scripts, this worker dynamically plans actions, uses browser and document automation tools, seeks human approval before committing high-risk changes, independently verifies ground-truth outcomes, and autonomously recovers from execution failures.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Demo](#demo)
- [Architecture](#architecture)
- [Agent Execution Loop](#agent-execution-loop)
- [Tools](#tools)
- [Agent State](#agent-state)
- [Reliability](#reliability)
- [Verification](#verification)
- [Human Approval](#human-approval)
- [Generalization](#generalization)
- [Tech Stack](#tech-stack)
- [Setup](#setup)
- [Environment Variables](#environment-variables)
- [Running](#running)
- [Example Task](#example-task)
- [Testing](#testing)
- [Known Limitations](#known-limitations)
- [Assumptions](#assumptions)
- [Future Improvements](#future-improvements)
- [AI Tools Used](#ai-tools-used)

---

# Project Overview

### The Problem
Traditional business process automation (RPA) relies on brittle, deterministic scripts that break whenever web layouts change, documents vary in structure, or network glitches occur. Conversely, typical LLM chat assistants can generate text or suggest code, but they lack agency: they cannot directly interact with web applications, inspect internal databases, persist state across multi-step operations, or recover from runtime exceptions. Crucially, raw LLMs hallucinate success, often claiming an action was completed when an underlying API or web form silently failed.

### The Solution
The **Autonomous AI Task Worker** bridges this gap. It receives high-level natural language instructions (such as *"Find the latest Acme invoice, extract key details, enter it into finance, and verify it was saved"*), formulates a step-by-step hypothesis, and enters a dynamic **OBSERVE → DECIDE → ACT → VERIFY** loop. 

Key pillars:
1. **Dynamic Decision Making**: Tools are selected dynamically by a planner using strict Pydantic schemas, not hardcoded sequences.
2. **Human-in-the-Loop (HITL) Safety Gate**: Financial mutations and destructive writes automatically pause execution and require human operator sign-off.
3. **Independent Ground-Truth Verification**: The agent never trusts a tool's return code as proof of task completion; a separate verification engine confirms that expected data actually exists in the destination ledger.
4. **Generalization Across Workflows**: The same engine handles document discovery, financial querying, and out-of-domain requests without task-specific branching.

---

# Demo

The primary reference workflow demonstrates autonomous accounts payable invoice processing:

```text
User Instruction:
"Find the latest invoice from Acme Corp, extract the invoice number, amount and due date,
enter the information into the internal finance application, and verify that it was saved correctly."
```

### End-to-End Execution Sequence:
1. **Document Discovery**: The agent invokes `document_search_tool` to scan invoice repositories and identify the latest PDF (`acme_invoice_2024_09.pdf`).
2. **Data Extraction**: The agent invokes `document_read_tool` using PyMuPDF to extract structured invoice metadata (`INV-ACM-2024-015`, `$12,450.50 USD`, due date `2024-10-28`).
3. **Safety Interception (HITL)**: The agent prepares to call `finance_create_invoice_tool`. The `ApprovalGate` classifies this as `HIGH_RISK_WRITE`, halts execution, sets status to `WAITING_FOR_APPROVAL`, and creates a pending approval request.
4. **Human Review & Approval**: An operator inspects the proposed invoice parameters via `POST /tasks/{id}/approve`.
5. **Execution Resumption**: The executor resumes, registers the invoice in the internal finance system, and captures API response metadata.
6. **Independent 7-Point Verification**: Before marking the task `COMPLETED`, `InvoiceVerificationStrategy` queries the database and DOM to verify:
   - Invoice exists in the ledger
   - Correct vendor name (`Acme Corp`)
   - Correct invoice number (`INV-ACM-2024-015`)
   - Correct amount (`12450.50`)
   - Correct currency (`USD`)
   - Correct due date (`2024-10-28`)
   - Stored values match extracted source document values
7. **Task Completion**: Upon verification success, the agent transitions to `COMPLETED` and attaches a standardized `TaskOutcome` with audit evidence.

---

# Architecture

The system is built on a clean, layered architecture separating intent decision, tool execution, safety gating, and outcome verification:

```text
+-------------------------------------------------------------------------------+
|                                FastAPI REST API                               |
|              (/tasks, /tasks/{id}/run, /tasks/{id}/approve, /health)           |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
|                             AgentExecutor Loop                                |
|   +-----------------------------------------------------------------------+   |
|   | 1. Observe State & Working Memory                                     |   |
|   | 2. Decide Next Action (Planner + ToolRegistry Schemas)                |   |
|   | 3. Validate Proposed Action Schema                                    |   |
|   | 4. Evaluate Safety Risk Level (ApprovalGate)                         |   |
|   |    --> If HIGH/DESTRUCTIVE: Pause & await POST /tasks/{id}/approve    |   |
|   | 5. Execute Selected Tool via ToolRegistry                             |   |
|   | 6. Collect ToolResult & Store Observation                             |   |
|   | 7. Ingest Business Facts into Working Memory                          |   |
|   | 8. Check RecoveryManager if Action Failed                             |   |
|   | 9. Independent Outcome Verification (VerificationStrategyRegistry)    |   |
|   | 10. Stop when: Verified | Clarification Required | Limit Reached      |   |
|   +-----------------------------------------------------------------------+   |
+-------------------------------------------------------------------------------+
           |                            |                           |
           v                            v                           v
+-----------------------+   +-----------------------+   +-----------------------+
|     ToolRegistry      |   |     ApprovalGate      |   |  VerificationEngine   |
|-----------------------|   |-----------------------|   |-----------------------|
| - BrowserTool         |   | Risk Levels:          |   | Strategies:           |
| - DocumentSearchTool  |   | - READ (Auto)         |   | - InvoiceStrategy     |
| - DocumentReadTool    |   | - LOW_WRITE (Auto)    |   |   (7-point deep check)|
| - FinanceSearchTool   |   | - HIGH_WRITE (Pause)  |   | - GenericToolStrategy |
| - FinanceCreateTool   |   | - DESTRUCTIVE (Pause) |   |   (Objective criteria)|
| - FinanceReadTool     |   +-----------------------+   +-----------------------+
+-----------------------+               |
           |                            v
           |               +-------------------------+
           |               |   Human Operator / UI   |
           |               +-------------------------+
           v
+-------------------------------------------------------------------------------+
|                         External & Application Boundary                       |
|   - Playwright Headless Chromium (Web UI Automation & Visual Evidence)        |
|   - PyMuPDF / Fitz (Document Parsing & Text Extraction)                       |
|   - Internal Finance Ledger (FastAPI + SQLAlchemy Async Engine)               |
|   - Audit Store (PostgreSQL / SQLite: Steps, Evidence, Milestones, Screenshots)|
+-------------------------------------------------------------------------------+
```

---

# Agent Execution Loop

The core execution engine implements an explicit 12-step autonomous loop:

$$\text{OBSERVE} \longrightarrow \text{DECIDE} \longrightarrow \text{VALIDATE} \longrightarrow \text{GATE} \longrightarrow \text{ACT} \longrightarrow \text{EVALUATE} \longrightarrow \text{VERIFY} \longrightarrow \text{RECOVER}$$

1. **OBSERVE**: Inspects the current `TaskState`, history of completed steps, recent tool observations, and extracted working memory facts.
2. **DECIDE**: Prompts the `Planner` with the user goal, current state, and available tool schemas. The planner returns a structured Pydantic `PlannerDecision` containing exactly one next action.
3. **VALIDATE**: Checks that the proposed action matches a registered tool and conforms to the tool's input schema.
4. **GATE (Approval Check)**: The `ApprovalGate` evaluates the risk tier of the action. If `HIGH_RISK_WRITE` or `DESTRUCTIVE`, the task state transitions to `WAITING_FOR_APPROVAL` and the loop pauses cleanly.
5. **ACT**: Executes the tool asynchronously with runtime timeouts and error capturing. Tools return a structured `ToolResult` containing `success`, `data`, `error`, `evidence`, and `metadata`. Errors are never silently swallowed.
6. **OBSERVE (Result Ingestion)**: Converts the `ToolResult` into an immutable `Observation`, persists step execution logs, updates working memory with newly extracted entities, and collects visual or data evidence.
7. **VERIFY**: When a milestone or completion proposal occurs, the executor invokes `VerificationStrategyRegistry`. A task is **never** completed merely because a tool succeeded.
8. **RECOVER**: If a tool fails, `RecoveryManager` classifies the failure into one of 10 error categories and emits a recovery decision (bounded retry, argument correction, environment re-observation, or alternative tool).
9. **TERMINATE / CONTINUE**: Halts execution cleanly when:
   - The goal is independently verified (`COMPLETED`)
   - Human operator approval is needed (`WAITING_FOR_APPROVAL`)
   - Ambiguous requirements require operator input (`WAITING_FOR_CLARIFICATION`)
   - Consecutive failures exceed limits or max step budget is reached (`FAILED`)

---

# Tools

All tools inherit from [`BaseTool`](file:///d:/FDE_ASSIGNMENTS/autonomous-ai-task-worker/app/tools/base.py) and expose `name`, `description`, `input_schema` (Pydantic), `risk_level`, and an async `execute()` method returning a standardized `ToolResult`:

| Tool Name | Risk Level | Description | Key Supported Actions |
| :--- | :---: | :--- | :--- |
| **`BrowserTool`** | `READ` / `LOW` | Automates web applications using Playwright headless Chromium. Captures visual screenshots and DOM text. | `open_url`, `click`, `type`, `select`, `extract_text`, `screenshot`, `go_back` |
| **`DocumentSearchTool`** | `READ` | Scans local and configured document repositories for invoices matching a vendor. | Scans PDF files, extracts date metadata, resolves the most recent document (`find_latest=True`). |
| **`DocumentReadTool`** | `READ` | Parses PDF documents using PyMuPDF (`fitz`), extracting raw text and regex-parsing structured fields. | Extracts invoice numbers, vendor names, dates, line items, amounts, and currency. |
| **`FinanceSearchTool`** | `READ` | Queries the internal finance ledger database. | Filters invoices by vendor, status (`PAID`, `PENDING`, `OVERDUE`), or invoice number query string. |
| **`FinanceCreateInvoiceTool`** | `HIGH_RISK_WRITE` | Registers a newly received invoice into the internal finance ledger database. | Inserts invoice record with duplicate detection, due date validation, and amount verification. |
| **`FinanceReadInvoiceTool`** | `READ` | Reads and verifies existing invoice records directly from the finance database. | Retrieves stored record by `invoice_number` or database ID for audit comparisons. |

Tools are managed dynamically by [`ToolRegistry`](file:///d:/FDE_ASSIGNMENTS/autonomous-ai-task-worker/app/tools/registry.py), which exports OpenAPI-compatible JSON schemas to the planner.

---

# Agent State

The execution state is tracked by [`TaskState`](file:///d:/FDE_ASSIGNMENTS/autonomous-ai-task-worker/app/agent/state.py), a centralized model that persists across tool steps and system restarts.

### State Model Fields:
- `task_id`: Unique UUID for the execution.
- `user_goal`: Original natural-language instruction.
- `current_status`: Current lifecycle phase (`PENDING`, `RUNNING`, `WAITING_FOR_APPROVAL`, `WAITING_FOR_CLARIFICATION`, `COMPLETED`, `FAILED`).
- `current_objective`: Planner's immediate tactical milestone.
- `plan_summary`: High-level strategic plan formulated by the planner.
- `current_step` & `max_steps`: Step counter and configurable budget limit (default: 20).
- `completed_steps` & `failed_steps`: Audit lists of completed and failed tool actions.
- `observations`: Sliding window of recent structured observations returned by tools.
- `extracted_data`: Dynamic working memory storing facts, file paths, invoice numbers, amounts, and query results.
- `tool_results`: Chronological history of raw tool execution payloads, latencies, and return statuses.
- `execution_history`: Explicit high-level ledger of actions and outcomes (e.g., `[{"action": "document_search_tool", "result": "Found 3 Acme invoices"}]`).
- `screenshots`: File paths to all captured screenshots.
- `evidence`: Audit dossier containing source document pointers, extracted fields, ledger IDs, and verification records.
- `latest_verification`: Most recent `VerificationResult` from the verification engine.
- `outcome`: Standardized `TaskOutcome` contract (`status`, `success`, `summary`, `facts`, `evidence`, `verification`, `reason`).

State snapshots are persisted asynchronously to PostgreSQL/SQLite after every loop iteration via [`TaskService.save_task_state`](file:///d:/FDE_ASSIGNMENTS/autonomous-ai-task-worker/app/services/task_service.py).

---

# Reliability

The agent is engineered to handle real-world operational faults gracefully without getting stuck in infinite loops.

### Bounded Failures and Limits
- **Max Step Budget**: Hard stop at `MAX_STEPS` (default: 20) prevents runaway executions.
- **Consecutive Failure Threshold**: `MAX_CONSECUTIVE_FAILURES` (default: 3). If an unrecoverable failure occurs 3 times consecutively, the task halts cleanly as `FAILED`.
- **Tool Timeouts**: LLM calls and browser navigations enforce strict 30-second timeouts.

### Recovery Manager
[`RecoveryManager`](file:///d:/FDE_ASSIGNMENTS/autonomous-ai-task-worker/app/agent/recovery.py) inspects failed steps and maps errors to specific recovery strategies:

| Failure Type | Detected Condition | Autonomous Recovery Strategy |
| :--- | :--- | :--- |
| **Missing Button / Element** | CSS or XPath selector not found in DOM | Re-observe DOM state, try alternative semantic selector or fallback tool. |
| **Temporary Tool Failure** | Network timeout or transient socket error | Bounded exponential retry of the same action (up to 3 times). |
| **Invalid Invoice Data** | Missing invoice number or corrupted fields | Re-parse document with secondary regex extraction or request operator clarification. |
| **Duplicate Invoice** | Finance ledger raises unique constraint error | Switch from `create` to `read` to check if the record already exists and verify status. |
| **Navigation Error** | Browser redirects to error or login page | Re-authenticate, re-observe current URL, and re-navigate to destination. |

Every recovery attempt is recorded as an observable event sequence:
$$\text{ACTION\_FAILED} \longrightarrow \text{OBSERVE} \longrightarrow \text{RECOVERY\_DECISION} \longrightarrow \text{ALTERNATIVE\_ACTION} \longrightarrow \text{VERIFY}$$

---

# Verification

### Why Tool Success $\neq$ Task Success
A fundamental design flaw in naive agents is assuming that an HTTP `200 OK` or a successful tool exit code equates to goal achievement. In enterprise applications:
- Web forms can display success banners while failing backend validation.
- Database inserts can be rolled back by subsequent transactions.
- Silent coercion can truncate amounts or corrupt due dates.

### Independent Verification Engine
The agent separates **execution** from **verification**:
1. When the planner proposes task completion, [`AgentExecutor._evaluate_and_complete`](file:///d:/FDE_ASSIGNMENTS/autonomous-ai-task-worker/app/agent/executor.py#L552-L604) queries the [`VerificationStrategyRegistry`](file:///d:/FDE_ASSIGNMENTS/autonomous-ai-task-worker/app/verification/strategy.py#L152-L173).
2. For invoice creation workflows, [`InvoiceVerificationStrategy`](file:///d:/FDE_ASSIGNMENTS/autonomous-ai-task-worker/app/verification/strategy.py#L33-L67) executes a 7-point deep cross-check against the live database and browser DOM:
   - Confirms the record exists in the finance database.
   - Cross-verifies vendor, invoice number, amount, currency, and due date against the raw source invoice.
   - Generates an auditable `VerificationReport` with individual check passes/failures and discrepancy logs.
3. If discrepancies are found, completion is **rejected**. The failure is appended to working memory as a `VerificationEngine` observation, returning control to the agent to recover or self-correct.

---

# Human Approval

The [`ApprovalGate`](file:///d:/FDE_ASSIGNMENTS/autonomous-ai-task-worker/app/approval/gate.py) enforces enterprise authorization boundaries based on four explicit risk levels:

```text
  READ / LOW_RISK_WRITE  ----->  Automatic Execution
  HIGH_RISK_WRITE        ----->  Paused -> ApprovalRequest Created -> Awaits Human Sign-Off
  DESTRUCTIVE            ----->  Paused -> ApprovalRequest Created -> Awaits Human Sign-Off
```

### Risk Classification:
- **`READ`** (`Automatic`): Safe data retrieval (e.g., `document_search_tool`, `document_read_tool`, `finance_read_invoice_tool`, browser navigation).
- **`LOW_RISK_WRITE`** (`Automatic`): Non-destructive state updates (e.g., updating local draft, caching metadata).
- **`HIGH_RISK_WRITE`** (`Approval Required`): Mutating financial records or creating invoices (`finance_create_invoice_tool`).
- **`DESTRUCTIVE`** (`Approval Required`): Deleting records or altering banking information.

### Approval Lifecycle:
1. When a high-risk tool is proposed, the executor pauses and transitions task status to `WAITING_FOR_APPROVAL`.
2. An `ApprovalRequest` record is persisted with proposed action arguments and risk rationale.
3. The human operator reviews the request via REST API:
   - `POST /tasks/{id}/approve`: Resumes execution and commits the action.
   - `POST /tasks/{id}/reject`: Halts the task cleanly as `FAILED` with an audit comment.

---

# Generalization

The agent is not an invoice-only script. The architecture decouples reasoning, memory, tools, and verification so the same agent handles diverse natural language instructions:

### Scenario 1: Document Discovery & Verification
- **Goal**: *"Find the latest Acme invoice and verify whether it exists."*
- **Behavior**: Scans document directory, reads metadata, extracts key fields, and verifies existence without writing to the finance ledger.
- **Verification**: Verified via `GenericToolVerificationStrategy` confirming successful discovery and fact extraction.

### Scenario 2: Finance Ledger Status Query
- **Goal**: *"Search the finance system for invoice INV-1042 and report its status."*
- **Behavior**: Evaluates goal, executes `finance_search_tool(query="INV-1042")`, extracts status into working memory, and returns verified status report.

### Scenario 3: Filtered Vendor Inquiries
- **Goal**: *"Find all unpaid invoices from Acme and return their invoice numbers and amounts."*
- **Behavior**: Calls `finance_search_tool(vendor_name="Acme Corp", status="PENDING")`, collects invoice array into working memory facts, and reports results.

### Scenario 4: Out-of-Domain Safety & Clarification
- **Goal**: *"Deploy container to Kubernetes cluster with 5 replicas."*
- **Behavior**: Inspects registered tools, detects no matching capability, refuses to hallucinate tools, and halts cleanly in `WAITING_FOR_CLARIFICATION`.

---

# Tech Stack

- **Core Backend**: Python 3.12, FastAPI, Uvicorn
- **Data Validation & Schemas**: Pydantic v2, Pydantic Settings
- **Database & Persistence**: SQLAlchemy 2.0 (AsyncIO), PostgreSQL (`asyncpg`), SQLite fallback (`aiosqlite`)
- **Web Automation**: Playwright (Async Chromium)
- **Document Processing**: PyMuPDF (`fitz`)
- **LLM Integration**: OpenAI Async API (`gpt-4o-mini`, temperature `0.0`), structured JSON responses
- **Structured Logging**: Structlog (JSON in production, colored console in dev)
- **Testing & Quality**: Pytest, Pytest-AsyncIO, HTTPX, Coverage

---

# Setup

### Prerequisites
- Python 3.12+
- `uv` (recommended) or standard `pip`
- Git

### Local Installation Commands

```bash
# 1. Clone the repository
git clone  https://github.com/Surendra571/autonomous-ai-task-worker.git
cd autonomous-ai-task-worker

# 2. Create and activate a Python 3.12 virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Upgrade pip and install package with development dependencies
pip install --upgrade pip
pip install -e ".[dev]"

# 4. Install Playwright Chromium browser binaries
playwright install chromium
```

---

# Environment Variables

Copy the sample environment file to create your local `.env`:

```bash
cp .env.example .env
```

### Configuration Keys:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `APP_ENV` | `development` | Environment mode (`development` / `production`). |
| `DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@localhost:5432/task_worker` | Async PostgreSQL database connection URL. |
| `USE_SQLITE_FALLBACK` | `true` | Automatically falls back to local SQLite if PostgreSQL is unavailable. |
| `OPENAI_API_KEY` | `mock-key-for-development` | OpenAI API key. Leave as mock key to run in autonomous offline mode. |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Compatible API endpoint base URL. |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM model identifier. |
| `MAX_STEPS` | `20` | Maximum execution step budget per task run. |
| `MAX_CONSECUTIVE_FAILURES`| `3` | Maximum consecutive failures before halting as unrecoverable. |
| `BROWSER_HEADLESS` | `true` | Runs Playwright Chromium in headless mode. |

---

# Running

### Option 1: Local Development Server

```bash
# Start FastAPI server on port 8000
uvicorn app.main:app --reload --port 8000
```
Interactive Swagger API documentation will be available at:
`http://localhost:8000/docs`

### Option 2: Docker Compose

```bash
# Build and run application and PostgreSQL database containers
docker compose up --build
```

---

# Example Task

### 1. Create a Task
```bash
curl -X POST "http://localhost:8000/tasks" \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Find the latest invoice from Acme Corp, extract the invoice number, amount and due date, enter the information into the internal finance application, and verify that it was saved correctly.",
    "max_steps": 15
  }'
```
*Response returns task details and a generated `id` (e.g. `c4b12345-...`).*

### 2. Trigger Autonomous Execution
```bash
curl -X POST "http://localhost:8000/tasks/c4b12345-.../run"
```
*The agent searches documents, reads the invoice, and pauses at `WAITING_FOR_APPROVAL`.*

### 3. Review & Approve High-Risk Invoice Creation
```bash
curl -X POST "http://localhost:8000/tasks/c4b12345-.../approve" \
  -H "Content-Type: application/json" \
  -d '{
    "approved": true,
    "comment": "Verified against PO-889. Approved."
  }'
```
*The agent resumes, inserts the record, performs 7-point independent verification, and completes.*

### 4. Fetch Completed Task Dossier & Evidence
```bash
curl -X GET "http://localhost:8000/tasks/c4b12345-..."
```

---

# Testing

The test suite covers unit tools, approval gates, planning prompts, browser automation, recovery policies, independent verification, and end-to-end task runs.

### Run All 95 Tests:
```bash
pytest -v
```

### Run Specific Test Suites:
```bash
# Run End-to-End Autonomous Invoice Workflow
pytest tests/test_e2e_acme_invoice.py -v

# Run Task Generalization Scenarios
pytest tests/test_generalization.py -v

# Run Independent Verification Engine Tests
pytest tests/test_verification_engine.py -v

# Run Recovery Policy Tests
pytest tests/test_recovery.py -v

# Run with Test Coverage Summary
pytest --cov=app --cov-report=term-missing
```

---

# Known Limitations

- **Complex PDF Layouts**: The document parser uses PyMuPDF text and regex heuristics, which work reliably on standard computer-generated invoices but may require vision models (VLM/OCR) for low-resolution scanned documents or rotated receipts.
- **Concurrent Task Isolation**: Tasks run concurrently within async Python tasks; however, multi-worker distributed locking is not yet implemented for horizontal multi-node scaling.
- **Single-Turn Human Approval**: The current approval gate is binary (Approve/Reject). It does not yet support inline parameter editing by the operator prior to approval.
- **DOM Selector Brittleness**: Web automation targets semantic CSS selectors and text fallbacks; highly obfuscated or dynamic canvas elements require visual grounding models.

---

# Assumptions

1. **Target Vendor Documents**: Sample test invoices are located in `./data/invoices` with standardized vendor names (`Acme Corp`, `Globex`, `Initech`, `Umbrella`).
2. **Finance Application Schema**: The internal finance database exposes standard invoice fields (`invoice_number`, `vendor_name`, `amount`, `currency`, `issue_date`, `due_date`, `status`).
3. **Storage Fallback**: If PostgreSQL is unreachable, the system automatically falls back to SQLite (`./data/task_worker.db`) to ensure local development and testing are friction-free.

---

# Future Improvements

With additional development time, the following features would be prioritized:
1. **Vision-Language Model (VLM) Tool**: Integrate multimodal models (e.g. Gemini / GPT-4o Vision) to parse handwritten receipts, graphical charts, and complex nested invoice tables.
2. **Distributed Queue Worker**: Migrate execution orchestration from local asyncio tasks to Temporal or Celery backed by Redis/RabbitMQ for enterprise-grade durability.
3. **Webhook Notifications**: Integrate Slack / Microsoft Teams webhooks to ping human supervisors directly when an approval request is triggered.
4. **Interactive Browser Session Takeover**: Allow a human supervisor to open a remote VNC/Playwright session to solve CAPTCHAs or 2FA challenges when encountered.

---

# AI Tools Used

In accordance with academic and submission integrity guidelines, the following AI tools were utilized during development:
- **Google Antigravity & Claude 3.5 Sonnet / Claude 3.7**: Used as agentic pair-programming assistants for drafting boilerplate Pydantic schemas, generating mock test fixtures, brainstorming failure taxonomy scenarios, and conducting adversarial code reviews.
- **OpenAI GPT-4o-mini**: Integrated as the runtime LLM reasoning engine for dynamic action planning and structured JSON output generation.
- **GitHub Copilot**: Used for inline autocompletion of test cases and docstrings.

