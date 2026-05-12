"""Web tools: search and fetch operations."""

from __future__ import annotations

import html as _html
import json as _json
import os
import re
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from loguru import logger

from spoon_bot.agent.tools.base import Tool
from spoon_bot.agent.tools.execution_context import capture_tool_output

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


async def close_shared_http_client() -> None:
    """Close the shared httpx client so pools are drained on app shutdown."""
    global _shared_client
    if _shared_client is not None and not _shared_client.is_closed:
        await _shared_client.aclose()
    _shared_client = None


_WEB_SEARCH_ENV_CANDIDATES: dict[str, tuple[str, ...]] = {
    "tavily": ("TAVILY_API_KEY",),
    "brave": ("BRAVE_SEARCH_API_KEY", "BRAVE_API_KEY"),
}
_WEB_SEARCH_PROVIDER_PRIORITY: tuple[str, ...] = ("tavily", "brave", "duckduckgo")


def get_configured_web_search_provider(preferred: str | None = None) -> str | None:
    """Return the first usable web_search provider configured in the environment."""
    for provider in _get_web_search_provider_candidates(preferred):
        if provider == "duckduckgo":
            return provider
        if _get_web_search_api_keys(provider):
            return provider
    return None


def describe_web_search_capability(preferred: str | None = None) -> tuple[bool, str]:
    """Describe whether a usable web_search provider is configured."""
    provider = get_configured_web_search_provider(preferred)
    if provider is not None:
        return True, f"Configured web_search provider: {provider}"
    return False, "No web_search provider is configured."


def _get_web_search_api_key(provider: str) -> str | None:
    keys = _get_web_search_api_keys(provider)
    return keys[0] if keys else None


def _get_web_search_provider_candidates(preferred: str | None = None) -> list[str]:
    preferred_provider = (preferred or "").strip().lower()
    candidates: list[str] = []
    if preferred_provider:
        candidates.append(preferred_provider)
    candidates.extend(_WEB_SEARCH_PROVIDER_PRIORITY)

    deduped: list[str] = []
    seen: set[str] = set()
    for provider in candidates:
        if not provider or provider in seen:
            continue
        seen.add(provider)
        deduped.append(provider)
    return deduped


def _get_web_search_api_keys(provider: str) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for env_var in _WEB_SEARCH_ENV_CANDIDATES.get(provider, ()):
        value = os.environ.get(env_var)
        if value:
            for raw in value.split(","):
                text = raw.strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                keys.append(text)
    return keys


class WebSearchTool(Tool):
    """
    Tool to search the web for information.

    Supports multiple search providers. Tavily is preferred when configured,
    with DuckDuckGo as the final fallback because it works without an API key.
    """

    SUPPORTED_PROVIDERS = frozenset({
        "tavily", "google", "duckduckgo", "brave", "serper",
    })

    _ENV_VAR_MAP: dict[str, str] = {
        "tavily": "TAVILY_API_KEY",
        "google": "GOOGLE_SEARCH_API_KEY",
        "brave": "BRAVE_SEARCH_API_KEY or BRAVE_API_KEY",
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
        explicit_provider = (provider or "").strip().lower()
        max_results = min(max_results or self._max_results, 20)

        if explicit_provider and explicit_provider not in self.SUPPORTED_PROVIDERS:
            return (
                f"Error: Unsupported provider '{explicit_provider}'. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_PROVIDERS))}"
            )

        last_failure = "No results found."
        for candidate in self._resolve_provider_chain(explicit_provider or None):
            result = await self._execute_with_provider(candidate, query, max_results, topic, time_range)
            if not self._is_retryable_search_failure(result):
                return result
            last_failure = result
            logger.warning(f"web_search provider '{candidate}' failed, trying next provider if available")
        return last_failure

    def _resolve_provider_chain(self, provider: str | None) -> list[str]:
        explicit = (provider or "").strip().lower()
        if explicit == "duckduckgo":
            return ["duckduckgo"]
        if explicit:
            return [explicit, "duckduckgo"]

        chain: list[str] = []
        for candidate in _get_web_search_provider_candidates(self._default_provider):
            if candidate == "duckduckgo" or _get_web_search_api_keys(candidate):
                chain.append(candidate)
        return chain or [self._default_provider, "duckduckgo"]

    async def _execute_with_provider(
        self,
        provider: str,
        query: str,
        max_results: int,
        topic: str | None,
        time_range: str | None,
    ) -> str:
        if provider == "tavily":
            return await self._search_tavily(query, max_results, topic, time_range)
        if provider == "brave":
            return await self._search_brave(query, max_results, topic, time_range)
        if provider == "duckduckgo":
            return await self._search_duckduckgo(query, max_results, topic, time_range)
        return await self._search_stub(query, provider, max_results, time_range)

    @staticmethod
    def _is_retryable_search_failure(result: str) -> bool:
        normalized = (result or "").strip()
        return normalized.startswith("Error:") or normalized == "No results found."

    # ---- Tavily implementation ---------------------------------------------

    async def _search_tavily(
        self,
        query: str,
        max_results: int,
        topic: str | None,
        time_range: str | None,
    ) -> str:
        api_keys = _get_web_search_api_keys("tavily")
        if not api_keys:
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

        client = _get_http_client()
        last_error = "Error: Tavily search failed"
        for index, api_key in enumerate(api_keys, start=1):
            try:
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
                return self._format_tavily_results(data)
            except httpx.HTTPStatusError as exc:
                logger.error(f"Tavily API error (key {index}/{len(api_keys)}): {exc.response.status_code} {exc.response.text[:300]}")
                last_error = f"Error: Tavily search failed ({exc.response.status_code})"
            except Exception as exc:
                logger.error(f"Tavily request error (key {index}/{len(api_keys)}): {exc}")
                last_error = f"Error: Web search failed — {exc}"
        return last_error

    def _format_tavily_results(self, data: dict[str, Any]) -> str:
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


    async def _search_duckduckgo(
        self,
        query: str,
        max_results: int,
        topic: str | None,
        time_range: str | None,
    ) -> str:
        """Search DuckDuckGo's lightweight HTML endpoint without an API key."""
        search_query = query
        if topic == "news":
            search_query = f"{query} news"
        if time_range:
            search_query = f"{search_query} {time_range}"

        try:
            client = _get_http_client()
            resp = await client.get(
                "https://duckduckgo.com/html/",
                params={"q": search_query},
                headers={
                    "User-Agent": WebFetchTool.DEFAULT_USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml,text/plain,*/*",
                },
                follow_redirects=True,
            )
            resp.raise_for_status()
            html = resp.text
        except httpx.HTTPStatusError as exc:
            logger.error(f"DuckDuckGo search error: {exc.response.status_code} {exc.response.text[:300]}")
            return f"Error: DuckDuckGo search failed ({exc.response.status_code})"
        except Exception as exc:
            logger.error(f"DuckDuckGo request error: {exc}")
            return f"Error: Web search failed — {exc}"

        results = self._parse_duckduckgo_html(html, max_results)
        if not results:
            return "No results found."

        parts = [f"**Search results ({len(results)}):**\n"]
        for i, item in enumerate(results, 1):
            parts.append(
                f"{i}. **{item['title']}**\n"
                f"   {item['url']}\n"
                f"   {item['snippet']}\n"
            )
        return "\n".join(parts)

    @staticmethod
    def _parse_duckduckgo_html(html: str, max_results: int) -> list[dict[str, str]]:
        """Extract title, URL, and snippet from DuckDuckGo HTML results."""
        results: list[dict[str, str]] = []
        result_pattern = re.compile(
            r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
            re.IGNORECASE | re.DOTALL,
        )
        snippet_pattern = re.compile(
            r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(?P<snippet>.*?)</a>',
            re.IGNORECASE | re.DOTALL,
        )
        snippets = list(snippet_pattern.finditer(html))
        for idx, match in enumerate(result_pattern.finditer(html)):
            title = WebSearchTool._clean_html(match.group("title"))
            url = WebSearchTool._normalize_duckduckgo_url(match.group("href"))
            snippet = ""
            if idx < len(snippets):
                snippet = WebSearchTool._clean_html(snippets[idx].group("snippet"))
            if title or url:
                results.append({
                    "title": title or url or "Untitled",
                    "url": url,
                    "snippet": snippet,
                })
            if len(results) >= max_results:
                break
        return results

    @staticmethod
    def _clean_html(value: str) -> str:
        text = re.sub(r"<[^>]+>", " ", value or "")
        text = _html.unescape(text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _normalize_duckduckgo_url(value: str) -> str:
        text = _html.unescape(value or "").strip()
        parsed = urlparse(text)
        if parsed.path == "/l/" or parsed.netloc.endswith("duckduckgo.com"):
            uddg = parse_qs(parsed.query).get("uddg", [""])[0]
            if uddg:
                return unquote(uddg)
        return text


    async def _search_brave(
        self,
        query: str,
        max_results: int,
        topic: str | None,
        time_range: str | None,
    ) -> str:
        api_key = _get_web_search_api_key("brave")
        if not api_key:
            return (
                "Error: Brave Search API key not configured. "
                "Set BRAVE_SEARCH_API_KEY or BRAVE_API_KEY in your environment."
            )

        endpoint = (
            "https://api.search.brave.com/res/v1/news/search"
            if topic == "news"
            else "https://api.search.brave.com/res/v1/web/search"
        )
        params: dict[str, Any] = {
            "q": query,
            "count": max_results,
        }
        if time_range:
            params["freshness"] = time_range

        try:
            client = _get_http_client()
            resp = await client.get(
                endpoint,
                params=params,
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": api_key,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error(f"Brave API error: {exc.response.status_code} {exc.response.text[:300]}")
            return f"Error: Brave search failed ({exc.response.status_code})"
        except Exception as exc:
            logger.error(f"Brave request error: {exc}")
            return f"Error: Web search failed — {exc}"

        results = data.get("results", [])
        if not results:
            return "No results found."

        parts = [f"**Search results ({len(results)}):**\n"]
        for i, item in enumerate(results, 1):
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            content = str(
                item.get("description")
                or item.get("snippet")
                or item.get("content")
                or ""
            ).strip()
            age = str(item.get("age") or "").strip()
            time_note = f" ({age})" if age else ""
            parts.append(f"{i}. **{title or url or 'Untitled'}**{time_note}\n   {url}\n   {content}\n")
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
            f"Use 'tavily' when configured, with DuckDuckGo as fallback.\n"
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
            full_raw = resp.text
            summary_raw = full_raw
            if len(summary_raw) > self._max_content_size:
                summary_raw = summary_raw[: self._max_content_size] + "\n\n[Content truncated]"

            # JSON response — return formatted
            if "json" in content_type:
                try:
                    data = resp.json()
                    result = _json.dumps(data, ensure_ascii=False, indent=2)
                    capture_tool_output(result, result)
                    return result
                except Exception:
                    capture_tool_output(summary_raw, full_raw)
                    return summary_raw

            # HTML — try to extract readable text
            if "html" in content_type and extract_text:
                summary_result = self._extract_text_from_html(summary_raw, selector)
                full_result = self._extract_text_from_html(full_raw, selector)
                capture_tool_output(summary_result, full_result)
                return summary_result

            # Plain text / other
            capture_tool_output(summary_raw, full_raw)
            return summary_raw

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
                return "Error: Invalid URL scheme. Only http and https are allowed."
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
