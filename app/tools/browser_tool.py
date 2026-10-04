"""Browser automation tool for the autonomous agent using Playwright."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from app.config.settings import settings
from app.tools.base import BaseTool, ToolResult
from app.config.logging import get_logger

logger = get_logger("tools.browser")


class BrowserToolInput(BaseModel):
    """Input parameters for BrowserTool actions."""
    action: str = Field(
        ...,
        description="Action to perform: 'open_url', 'click', 'type', 'select', 'extract_text', 'screenshot', 'go_back'",
    )
    url: Optional[str] = Field(None, description="Target URL when action is 'open_url'.")
    selector: Optional[str] = Field(None, description="CSS or data-testid selector for click, type, select, extract_text.")
    text: Optional[str] = Field(None, description="Text string to type into an input field.")
    value: Optional[str] = Field(None, description="Option value to select in a dropdown.")
    filename: Optional[str] = Field(None, description="Optional custom filename for screenshot.")


class BrowserTool(BaseTool):
    """
    Playwright-powered web automation tool.
    Executes actions such as open_url, click, type, select, extract_text, screenshot, go_back.
    """

    name = "browser_tool"
    description = (
        "Automate web interactions using a browser. "
        "Supported actions: open_url(url), click(selector), type(selector, text), "
        "select(selector, value), extract_text(selector?), screenshot(filename?), go_back()."
    )
    input_schema = BrowserToolInput
    risk_level = "LOW"

    def __init__(
        self,
        page: Optional[Page] = None,
        context: Optional[BrowserContext] = None,
        artifacts_dir: Optional[Path] = None,
    ):
        self._page = page
        self._context = context
        self._playwright = None
        self._browser = None
        self.artifacts_dir = Path(artifacts_dir or settings.ARTIFACTS_DIR)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    async def _ensure_page(self) -> Page:
        """Lazily initialize browser and page if not injected."""
        if self._page is not None and not self._page.is_closed():
            return self._page

        if self._context is not None:
            self._page = await self._context.new_page()
            return self._page

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=settings.BROWSER_HEADLESS,
        )
        self._context = await self._browser.new_context()
        self._page = await self._context.new_page()
        return self._page

    async def close(self) -> None:
        """Clean up browser instances if managed internally."""
        if self._page and not self._page.is_closed():
            await self._page.close()
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def execute(self, params: BrowserToolInput) -> ToolResult:
        page = await self._ensure_page()
        action = params.action.strip().lower()

        if action == "open_url":
            if not params.url:
                return ToolResult.fail("Parameter 'url' is required for action 'open_url'.")
            response = await page.goto(params.url, wait_until="domcontentloaded")
            title = await page.title()
            return ToolResult.ok(
                data={
                    "current_url": page.url,
                    "title": title,
                    "http_status": response.status if response else 200,
                },
                metadata={"action": "open_url", "url": params.url},
            )

        elif action == "click":
            if not params.selector:
                return ToolResult.fail("Parameter 'selector' is required for action 'click'.")
            await page.wait_for_selector(params.selector, timeout=7000)
            await page.click(params.selector)
            return ToolResult.ok(
                data={"current_url": page.url, "selector_clicked": params.selector},
                metadata={"action": "click", "selector": params.selector},
            )

        elif action == "type":
            if not params.selector:
                return ToolResult.fail("Parameter 'selector' is required for action 'type'.")
            if params.text is None:
                return ToolResult.fail("Parameter 'text' is required for action 'type'.")
            await page.wait_for_selector(params.selector, timeout=7000)
            await page.fill(params.selector, params.text)
            return ToolResult.ok(
                data={"selector": params.selector, "typed_text": params.text},
                metadata={"action": "type", "selector": params.selector},
            )

        elif action == "select":
            if not params.selector or params.value is None:
                return ToolResult.fail("Parameters 'selector' and 'value' are required for action 'select'.")
            await page.wait_for_selector(params.selector, timeout=7000)
            selected = await page.select_option(params.selector, value=params.value)
            return ToolResult.ok(
                data={"selector": params.selector, "selected_options": selected},
                metadata={"action": "select", "selector": params.selector, "value": params.value},
            )

        elif action == "extract_text":
            if params.selector:
                await page.wait_for_selector(params.selector, timeout=7000)
                text_content = await page.text_content(params.selector)
            else:
                text_content = await page.content()

            return ToolResult.ok(
                data={
                    "current_url": page.url,
                    "extracted_text": text_content.strip() if text_content else "",
                },
                metadata={"action": "extract_text", "selector": params.selector},
            )

        elif action == "screenshot":
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = params.filename or f"screenshot_{timestamp}.png"
            if not filename.endswith(".png"):
                filename += ".png"
            file_path = self.artifacts_dir / filename
            await page.screenshot(path=str(file_path), full_page=True)

            return ToolResult.ok(
                data={"screenshot_path": str(file_path.resolve()), "filename": filename},
                evidence={"screenshot": str(file_path.resolve()), "url": page.url},
                metadata={"action": "screenshot", "filename": filename},
            )

        elif action == "go_back":
            await page.go_back(wait_until="domcontentloaded")
            return ToolResult.ok(
                data={"current_url": page.url, "title": await page.title()},
                metadata={"action": "go_back"},
            )

        else:
            return ToolResult.fail(
                f"Unknown browser action '{params.action}'. "
                "Supported: open_url, click, type, select, extract_text, screenshot, go_back."
            )
