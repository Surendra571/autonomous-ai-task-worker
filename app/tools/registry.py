"""Dynamic Tool Registry for managing, discovering, and executing agent tools."""

from typing import Any, Dict, List, Optional
from app.tools.base import BaseTool, ToolResult
from app.tools.browser_tool import BrowserTool
from app.tools.document_search_tool import DocumentSearchTool
from app.tools.document_read_tool import DocumentReadTool
from app.tools.finance_search_tool import FinanceSearchTool
from app.tools.finance_create_invoice_tool import FinanceCreateInvoiceTool
from app.tools.finance_read_invoice_tool import FinanceReadInvoiceTool
from app.config.logging import get_logger

logger = get_logger("tools.registry")


class ToolRegistry:
    """
    Central registry that dynamically exposes available tools to the planner
    and routes structured tool calls to concrete tool instances.
    Enforces the pattern: Planner -> ToolCall -> ToolRegistry -> Tool -> ToolResult.
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Registers a tool instance in the catalog."""
        if tool.name in self._tools:
            logger.warning("Overwriting existing tool registration", tool_name=tool.name)
        self._tools[tool.name] = tool
        logger.debug("Tool registered", tool_name=tool.name, risk_level=tool.risk_level)

    def unregister(self, tool_name: str) -> None:
        """Removes a tool from the catalog."""
        self._tools.pop(tool_name, None)

    def get(self, tool_name: str) -> Optional[BaseTool]:
        """Retrieves a registered tool by name."""
        return self._tools.get(tool_name)

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Alias for get(tool_name)."""
        return self.get(tool_name)

    def has(self, tool_name: str) -> bool:
        """Checks if a tool is registered."""
        return tool_name in self._tools

    def list_tools(self) -> List[str]:
        """Returns names of all registered tools."""
        return list(self._tools.keys())

    def get_schemas(self) -> List[Dict[str, Any]]:
        """
        Dynamically extracts tool specifications and schemas
        to present to the LLM planner.
        """
        return [tool.get_schema() for tool in self._tools.values()]

    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """
        Routes and executes a structured tool call.
        Guarantees that errors are caught, formatted into ToolResult, and never swallowed.
        """
        tool = self.get(tool_name)
        if not tool:
            available = ", ".join(self.list_tools())
            err_msg = f"Tool '{tool_name}' not found. Available tools: [{available}]"
            logger.warning("Execution attempted on unregistered tool", tool_name=tool_name)
            return ToolResult.fail(
                error=err_msg,
                metadata={"requested_tool": tool_name, "available_tools": self.list_tools()},
            )

        logger.info("Executing tool via registry", tool_name=tool_name, risk_level=tool.risk_level)
        return await tool.run(**arguments)


def create_default_registry(
    session_factory=None,
    browser_page=None,
    browser_context=None,
    inbox_dir=None,
) -> ToolRegistry:
    """
    Factory function initializing the complete standard tool catalog:
    1. BrowserTool
    2. DocumentSearchTool
    3. DocumentReadTool
    4. FinanceSearchTool
    5. FinanceCreateInvoiceTool
    6. FinanceReadInvoiceTool
    """
    registry = ToolRegistry()

    registry.register(BrowserTool(page=browser_page, context=browser_context))
    registry.register(DocumentSearchTool(inbox_dir=inbox_dir))
    registry.register(DocumentReadTool())
    registry.register(FinanceSearchTool(session_factory=session_factory))
    registry.register(FinanceCreateInvoiceTool(session_factory=session_factory))
    registry.register(FinanceReadInvoiceTool(session_factory=session_factory))

    return registry
