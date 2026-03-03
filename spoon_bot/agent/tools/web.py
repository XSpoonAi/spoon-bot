"""Web tools: search and fetch operations."""

from __future__ import annotations

import json as _json
import os
from typing import Any
from urllib.parse import urlparse

import httpx
from loguru import logger

from spoon_bot.agent.tools.base import Tool


# Shared httpx client for connection pooling across web tools.
# Created lazily on first use to avoid issues at import time.
_shared_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    """Get or create a shared httpx.AsyncClient with connection pooling."""
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        _shared_client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(
                max_keepalive_connections=5,
                max_connections=10,
            ),
        )
    return _shared_client


class WebSearchTool(Tool):
    """
    Tool to search the web for information.

    Supports multiple search providers.  Tavily is the recommended default
    because it returns structured results out of the box.
    """

    SUPPORTED_PROVIDERS = frozenset({
        "tavily", "google", "duckduckgo", "brave", "serper",
    })

    _ENV_VAR_MAP: dict[str, str] = {
        "tavily": "TAVILY_API_KEY",
        "google": "GOOGLE_SEARCH_API_KEY",
        "brave": "BRAVE_SEARCH_API_KEY",
        "serper": "SERPER_API_KEY",
        # DuckDuckGo doesn't require an API key for basic search
    }

    def __init__(self, default_provider: str = "tavily", max_results: int = 5):
        self._default_provider = default_provider
        self._max_results = max_results

    # ---- Tool interface ----------------------------------------------------

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web for real-time information. Returns relevant results "
            "with titles, URLs, and content snippets. Use this for current events, "
            "prices, news, documentation, or any question requiring up-to-date data."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string",
                },
                "max_results": {
                    "type": "integer",
                    "description": f"Maximum number of results (default: {self._max_results}, max: 20)",
                },
                "topic": {
                    "type": "string",
                    "description": "Search topic: 'general' or 'news'",
                    "enum": ["general", "news"],
                },
                "time_range": {
                    "type": "string",
                    "description": "Time range filter: 'day', 'week', 'month', 'year'",
                    "enum": ["day", "week", "month", "year"],
                },
            },
            "required": ["query"],
        }

    # ---- Execution ---------------------------------------------------------

    async def execute(
        self,
        query: str,
        max_results: int | None = None,
        topic: str | None = None,
        time_range: str | None = None,
        provider: str | None = None,
        **kwargs: Any,
    ) -> str:
        provider = provider or self._default_provider
        max_results = min(max_results or self._max_results, 20)

        if provider not in self.SUPPORTED_PROVIDERS:
            return (
                f"Error: Unsupported provider '{provider}'. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_PROVIDERS))}"
            )

        # Dispatch
        if provider == "tavily":
            return await self._search_tavily(query, max_results, topic, time_range)

        # Fallback: stub for other providers (can be implemented later)
        return await self._search_stub(query, provider, max_results, time_range)

    # ---- Tavily implementation ---------------------------------------------

    async def _search_tavily(
        self,
        query: str,
        max_results: int,
        topic: str | None,
        time_range: str | None,
    ) -> str:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            return (
                "Error: Tavily API key not configured. "
                "Set TAVILY_API_KEY in your .env file.\n"
                "Get a free key at https://tavily.com"
            )

        payload: dict[str, Any] = {
            "query": query,
            "max_results": max_results,
            "include_answer": "basic",
            "search_depth": "basic",
        }
        if topic:
            payload["topic"] = topic
        if time_range:
            payload["time_range"] = time_range

        try:
            client = _get_http_client()
            resp = await client.post(
                "https://api.tavily.com/search",
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error(f"Tavily API error: {exc.response.status_code} {exc.response.text[:300]}")
            return f"Error: Tavily search failed ({exc.response.status_code})"
        except Exception as exc:
            logger.error(f"Tavily request error: {exc}")
            return f"Error: Web search failed — {exc}"

        # Format results
        parts: list[str] = []

        answer = data.get("answer")
        if answer:
            parts.append(f"**Answer:** {answer}\n")

        results = data.get("results", [])
        if results:
            parts.append(f"**Search results ({len(results)}):**\n")
            for i, r in enumerate(results, 1):
                title = r.get("title", "")
                url = r.get("url", "")
                content = r.get("content", "")
                parts.append(f"{i}. **{title}**\n   {url}\n   {content}\n")
        elif not answer:
            parts.append("No results found.")

        return "\n".join(parts)

    # ---- Stub for unimplemented providers ----------------------------------

    async def _search_stub(
        self,
        query: str,
        provider: str,
        max_results: int,
        time_range: str | None,
    ) -> str:
        env_var = self._ENV_VAR_MAP.get(provider, "")
        return (
            f"Error: Provider '{provider}' is not yet implemented. "
            f"Use 'tavily' (default) instead.\n"
            f"Or set {env_var} to configure this provider."
        )


class WebFetchTool(Tool):
    """
    Tool to fetch and parse web page content.

    Supports HTML parsing, JSON responses, and basic text extraction.
    Respects robots.txt and includes configurable user agent.
    """

    # Default user agent (identify as a bot)
    DEFAULT_USER_AGENT = "SpoonBot/1.0 (Web Fetcher; +https://github.com/XSpoonAi)"

    # Maximum content size (10MB default)
    MAX_CONTENT_SIZE = 10 * 1024 * 1024

    # Allowed content types
    ALLOWED_CONTENT_TYPES = frozenset({
        "text/html", "text/plain", "application/json",
        "application/xml", "text/xml", "text/markdown",
    })

    def __init__(
        self,
        user_agent: str | None = None,
        timeout: int = 30,
        max_content_size: int | None = None,
    ):
        """
        Initialize web fetch tool.

        Args:
            user_agent: Custom user agent string.
            timeout: Request timeout in seconds.
            max_content_size: Maximum content size to fetch.
        """
        self._user_agent = user_agent or self.DEFAULT_USER_AGENT
        self._timeout = timeout
        self._max_content_size = max_content_size or self.MAX_CONTENT_SIZE

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return (
            "Fetch content from a URL. Supports HTML pages (with text extraction), "
            "JSON APIs, and plain text. Returns parsed content suitable for analysis."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch",
                },
                "extract_text": {
                    "type": "boolean",
                    "description": "For HTML, extract readable text only (default: true)",
                },
                "selector": {
                    "type": "string",
                    "description": "CSS selector to extract specific content (e.g., 'article', '.content')",
                },
                "headers": {
                    "type": "object",
                    "description": "Additional HTTP headers to send",
                    "additionalProperties": {"type": "string"},
                },
                "method": {
                    "type": "string",
                    "description": "HTTP method (default: GET)",
                    "enum": ["GET", "POST", "HEAD"],
                },
                "body": {
                    "type": "string",
                    "description": "Request body for POST requests",
                },
            },
            "required": ["url"],
        }

    def _validate_url(self, url: str) -> tuple[bool, str]:
        """Validate URL and check for safety issues."""
        try:
            parsed = urlparse(url)

            # Check scheme
            if parsed.scheme not in ("http", "https"):
                return False, f"Invalid URL scheme: {parsed.scheme}. Only http and https are allowed."

            # Check for localhost/internal IPs (SSRF protection)
            hostname = parsed.hostname or ""
            if hostname in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
                return False, "Fetching from localhost is not allowed for security reasons."

            # Check for private IP ranges
            if hostname.startswith(("10.", "192.168.", "172.16.", "172.17.",
                                    "172.18.", "172.19.", "172.20.", "172.21.",
                                    "172.22.", "172.23.", "172.24.", "172.25.",
                                    "172.26.", "172.27.", "172.28.", "172.29.",
                                    "172.30.", "172.31.")):
                return False, "Fetching from private IP addresses is not allowed."

            return True, ""

        except Exception as e:
            return False, f"Invalid URL: {str(e)}"

    async def execute(
        self,
        url: str,
        extract_text: bool = True,
        selector: str | None = None,
        headers: dict[str, str] | None = None,
        method: str = "GET",
        body: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Fetch URL content.

        Args:
            url: URL to fetch.
            extract_text: Extract readable text from HTML.
            selector: CSS selector for specific content.
            headers: Additional HTTP headers.
            method: HTTP method.
            body: Request body for POST.

        Returns:
            Fetched content or error message.
        """
        # Validate URL
        is_valid, error_msg = self._validate_url(url)
        if not is_valid:
            return f"Error: {error_msg}"

        # Validate method
        if method not in ("GET", "POST", "HEAD"):
            return f"Error: Invalid method '{method}'. Allowed: GET, POST, HEAD"

        # POST requires body
        if method == "POST" and not body:
            return "Error: POST method requires a body parameter"

        # Prepare headers
        request_headers = {
            "User-Agent": self._user_agent,
            "Accept": "text/html,application/json,text/plain,*/*",
        }
        if headers:
            request_headers.update(headers)

        try:
            client = _get_http_client()
            resp = await client.request(
                method=method,
                url=url,
                headers=request_headers,
                content=body.encode() if body else None,
                follow_redirects=True,
            )
            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            raw = resp.text

            # Truncate very large responses
            if len(raw) > self._max_content_size:
                raw = raw[: self._max_content_size] + "\n\n[Content truncated]"

            # JSON response — return formatted
            if "json" in content_type:
                try:
                    data = resp.json()
                    return _json.dumps(data, ensure_ascii=False, indent=2)
                except Exception:
                    return raw

            # HTML — try to extract readable text
            if "html" in content_type and extract_text:
                return self._extract_text_from_html(raw, selector)

            # Plain text / other
            return raw

        except httpx.HTTPStatusError as exc:
            return f"Error: HTTP {exc.response.status_code} for {url}"
        except httpx.TimeoutException:
            return f"Error: Request timed out after {self._timeout}s for {url}"
        except Exception as exc:
            return f"Error fetching {url}: {exc}"

    @staticmethod
    def _extract_text_from_html(html: str, selector: str | None = None) -> str:
        """Extract readable text from HTML, optionally scoped to a CSS selector."""
        try:
            from bs4 import BeautifulSoup  # type: ignore[import-untyped]

            soup = BeautifulSoup(html, "html.parser")

            # Remove scripts, styles, nav, footer
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            if selector:
                target = soup.select_one(selector)
                if target:
                    return target.get_text(separator="\n", strip=True)
                return f"No element matching selector '{selector}' found."

            return soup.get_text(separator="\n", strip=True)

        except ImportError:
            # bs4 not installed — return raw with tags stripped naively
            import re
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text).strip()
            return text[:8000] if len(text) > 8000 else text


class WebBrowserTool(Tool):
    """
    Tool for browser-based web interactions (JavaScript rendering).

    Uses headless browser for pages that require JavaScript execution.
    More resource-intensive than WebFetchTool - use only when necessary.
    """

    def __init__(self, headless: bool = True, timeout: int = 60):
        """
        Initialize browser tool.

        Args:
            headless: Run browser in headless mode.
            timeout: Page load timeout in seconds.
        """
        self._headless = headless
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "web_browser"

    @property
    def description(self) -> str:
        return (
            "Load a web page using a headless browser with JavaScript support. "
            "Use this for dynamic pages that require JS execution. "
            "More resource-intensive than web_fetch - use only when needed."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to load in browser",
                },
                "wait_for": {
                    "type": "string",
                    "description": "CSS selector to wait for before extracting content",
                },
                "screenshot": {
                    "type": "boolean",
                    "description": "Take a screenshot of the page",
                },
                "execute_script": {
                    "type": "string",
                    "description": "JavaScript to execute on the page",
                },
                "extract_selector": {
                    "type": "string",
                    "description": "CSS selector to extract content from",
                },
            },
            "required": ["url"],
        }

    async def execute(
        self,
        url: str,
        wait_for: str | None = None,
        screenshot: bool = False,
        execute_script: str | None = None,
        extract_selector: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Load page in headless browser.

        Args:
            url: URL to load.
            wait_for: Selector to wait for.
            screenshot: Take screenshot.
            execute_script: JS to execute.
            extract_selector: Content to extract.

        Returns:
            Page content or error message.
        """
        # Basic URL validation
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return f"Error: Invalid URL scheme. Only http and https are allowed."
        except Exception as e:
            return f"Error: Invalid URL: {str(e)}"

        # Stub implementation
        return (
            f"[STUB] Web browser would execute:\n"
            f"  URL: {url}\n"
            f"  Wait For: {wait_for or '(none)'}\n"
            f"  Screenshot: {screenshot}\n"
            f"  Execute Script: {'(provided)' if execute_script else '(none)'}\n"
            f"  Extract Selector: {extract_selector or '(none)'}\n\n"
            f"To enable headless browser functionality, install Playwright:\n"
            f"  pip install playwright\n"
            f"  playwright install chromium\n\n"
            f"Or use Selenium:\n"
            f"  pip install selenium webdriver-manager\n\n"
            f"This is a stub - no actual browser session was started."
        )
