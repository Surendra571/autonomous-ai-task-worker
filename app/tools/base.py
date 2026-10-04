"""Generic Base Tool interface and ToolResult definition."""

from abc import ABC, abstractmethod
from datetime import datetime
import time
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field

from app.config.logging import get_logger

logger = get_logger("tools.base")


class ToolResult(BaseModel):
    """
    Standardized result contract returned by all tools.
    Never silently swallows errors; guarantees auditability.
    """
    success: bool = Field(..., description="True if the tool executed without error, False otherwise.")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Structured payload produced by the tool.")
    error: Optional[str] = Field(default=None, description="Detailed error message if execution failed.")
    evidence: Optional[Dict[str, Any]] = Field(default=None, description="Verifiable proof (e.g. screenshot path, file diff, raw text).")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution telemetry (latency_ms, tool_name, action, timestamp).")

    @classmethod
    def ok(
        cls,
        data: Optional[Dict[str, Any]] = None,
        evidence: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ToolResult":
        """Convenience constructor for successful execution."""
        return cls(
            success=True,
            data=data or {},
            evidence=evidence,
            metadata=metadata or {},
        )

    @classmethod
    def fail(
        cls,
        error: str,
        data: Optional[Dict[str, Any]] = None,
        evidence: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ToolResult":
        """Convenience constructor for failed execution."""
        return cls(
            success=False,
            error=error,
            data=data,
            evidence=evidence,
            metadata=metadata or {},
        )


class BaseTool(ABC):
    """
    Abstract base class for all tools accessible by the autonomous agent.
    Every tool must define name, description, input schema, execute(), and result schema.
    """

    name: str
    description: str
    input_schema: Type[BaseModel]
    result_schema: Type[BaseModel] = ToolResult
    risk_level: str = "LOW"  # LOW, MEDIUM, HIGH

    def get_schema(self) -> Dict[str, Any]:
        """Returns JSON schema specification for LLM tool-calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema.model_json_schema(),
                "risk_level": self.risk_level,
            },
        }

    async def run(self, session: Optional[Any] = None, **kwargs) -> ToolResult:
        """
        Public execution wrapper with validation, exception containment, and telemetry.
        Never silently swallows errors.
        """
        start_time = time.perf_counter()
        meta = {
            "tool_name": self.name,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Extract session if passed inside kwargs
        session_arg = kwargs.pop("session", session)

        # 1. Validate inputs against input_schema
        try:
            validated_args = self.input_schema(**kwargs)
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            meta["latency_ms"] = round(latency_ms, 2)
            err_msg = f"Input validation failed for tool '{self.name}': {e}"
            logger.warning(err_msg, tool=self.name, args=kwargs)
            return ToolResult.fail(error=err_msg, metadata=meta)

        # 2. Execute concrete tool logic
        try:
            import inspect
            sig = inspect.signature(self.execute)
            if "session" in sig.parameters:
                result = await self.execute(validated_args, session=session_arg)
            else:
                result = await self.execute(validated_args)
            latency_ms = (time.perf_counter() - start_time) * 1000
            result.metadata.update(meta)
            result.metadata["latency_ms"] = round(latency_ms, 2)
            return result
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            meta["latency_ms"] = round(latency_ms, 2)
            err_msg = f"Unhandled exception executing tool '{self.name}': {e}"
            logger.error(err_msg, tool=self.name, exc_info=True)
            return ToolResult.fail(error=err_msg, metadata=meta)

    @abstractmethod
    async def execute(self, params: Any) -> ToolResult:
        """Concrete tool implementation to be defined by subclasses."""
        pass
