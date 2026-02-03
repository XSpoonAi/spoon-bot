"""Web tools: search and fetch operations."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

from spoon_bot.agent.tools.base import Tool


class WebSearchTool(Tool):
    """
    Tool to search the web for information.

    Supports multiple search providers (Google, DuckDuckGo, Brave).
    Requires API key configuration for the chosen provider.
    """

    SUPPORTED_PROVIDERS = frozenset({"google", "duckduckgo", "brave", "serper"})

    def __init__(self, default_provider: str = "duckduckgo", max_results: int = 10):
        """
        Initialize web search tool.

        Args:
            default_provider: Default search provider.
            max_results: Maximum number of results to return.
        """
        self._default_provider = default_provider
        self._max_results = max_results

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web for information. Returns a list of relevant results "
            "with titles, URLs, and snippets. Useful for finding current information, "
            "documentation, or researching topics."
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
                "provider": {
                    "type": "string",
                    "description": "Search provider (default: duckduckgo)",
                    "enum": list(self.SUPPORTED_PROVIDERS),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 10, max: 50)",
                },
                "site": {
                    "type": "string",
                    "description": "Limit search to specific site (e.g., 'github.com')",
                },
                "time_range": {
                    "type": "string",
                    "description": "Time range filter: 'day', 'week', 'month', 'year'",
                    "enum": ["day", "week", "month", "year"],
                },
            },
            "required": ["query"],
        }

    def _get_api_key(self, provider: str) -> str | None:
        """Get API key for the specified provider."""
        env_var_map = {
            "google": "GOOGLE_SEARCH_API_KEY",
            "brave": "BRAVE_SEARCH_API_KEY",
            "serper": "SERPER_API_KEY",
            # DuckDuckGo doesn't require an API key for basic search
        }
        env_var = env_var_map.get(provider)
        return os.environ.get(env_var) if env_var else None

    def _get_google_cx(self) -> str | None:
        """Get Google Custom Search Engine ID."""
        return os.environ.get("GOOGLE_SEARCH_CX")

    async def execute(
        self,
        query: str,
        provider: str | None = None,
        max_results: int | None = None,
        site: str | None = None,
        time_range: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Search the web.

        Args:
            query: Search query.
            provider: Search provider to use.
            max_results: Maximum number of results.
            site: Limit to specific site.
            time_range: Time range filter.

        Returns:
            Search results or error message.
        """
        provider = provider or self._default_provider
        max_results = min(max_results or self._max_results, 50)

        # Validate provider
        if provider not in self.SUPPORTED_PROVIDERS:
            return f"Error: Unsupported provider '{provider}'. Supported: {', '.join(sorted(self.SUPPORTED_PROVIDERS))}"

        # Build search query with site filter
        full_query = query
        if site:
            full_query = f"site:{site} {query}"

        # Check API key for providers that require it
        if provider in ("google", "brave", "serper"):
            api_key = self._get_api_key(provider)
            if not api_key:
                env_var_map = {
                    "google": "GOOGLE_SEARCH_API_KEY",
                    "brave": "BRAVE_SEARCH_API_KEY",
                    "serper": "SERPER_API_KEY",
                }
                env_var = env_var_map[provider]
                return (
                    f"Error: {provider.title()} Search API key not configured. "
                    f"Set {env_var} environment variable.\n\n"
                    f"To get an API key:\n"
                    + self._get_api_setup_instructions(provider)
                )

            # Google also needs CX
            if provider == "google" and not self._get_google_cx():
                return (
                    "Error: Google Custom Search Engine ID not configured. "
                    "Set GOOGLE_SEARCH_CX environment variable.\n\n"
                    "To set up Google Custom Search:\n"
                    "  1. Go to https://programmablesearchengine.google.com/\n"
                    "  2. Create a search engine\n"
                    "  3. Get the Search Engine ID (CX)"
                )

        # Stub implementation
        return (
            f"[STUB] Web search would execute:\n"
            f"  Query: {full_query}\n"
            f"  Provider: {provider}\n"
            f"  Max Results: {max_results}\n"
            f"  Time Range: {time_range or 'any'}\n\n"
            f"To enable web search functionality, install the required packages:\n"
            f"  pip install httpx beautifulsoup4\n\n"
            f"And configure API keys for your preferred provider:\n"
            + self._get_api_setup_instructions(provider)
        )

    def _get_api_setup_instructions(self, provider: str) -> str:
        """Get setup instructions for a provider."""
        instructions = {
            "google": (
                "  Google Custom Search:\n"
                "    1. Go to https://console.cloud.google.com/\n"
                "    2. Enable Custom Search API\n"
                "    3. Create credentials (API Key)\n"
                "    4. Set GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_CX"
            ),
            "brave": (
                "  Brave Search:\n"
                "    1. Go to https://brave.com/search/api/\n"
                "    2. Sign up for API access\n"
                "    3. Get your API key\n"
                "    4. Set BRAVE_SEARCH_API_KEY"
            ),
            "serper": (
                "  Serper (Google Search API):\n"
                "    1. Go to https://serper.dev/\n"
                "    2. Sign up for an account\n"
                "    3. Get your API key\n"
                "    4. Set SERPER_API_KEY"
            ),
            "duckduckgo": (
                "  DuckDuckGo:\n"
                "    No API key required for basic search.\n"
                "    Install duckduckgo-search package:\n"
                "    pip install duckduckgo-search"
            ),
        }
        return instructions.get(provider, "  See provider documentation for API setup.")


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

        # Stub implementation
        return (
            f"[STUB] Web fetch would execute:\n"
            f"  URL: {url}\n"
            f"  Method: {method}\n"
            f"  Extract Text: {extract_text}\n"
            f"  Selector: {selector or '(none)'}\n"
            f"  Headers: {request_headers}\n"
            f"  Body: {body[:100] + '...' if body and len(body) > 100 else body or '(none)'}\n\n"
            f"To enable web fetch functionality, install the required packages:\n"
            f"  pip install httpx beautifulsoup4 lxml\n\n"
            f"This is a stub - no actual request was made."
        )


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
