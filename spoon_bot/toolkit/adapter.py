"""Adapter for spoon-toolkits integration."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING
from uuid import uuid4

from loguru import logger

from spoon_bot.agent.tools.base import Tool
from spoon_bot.agent.tools.execution_context import get_tool_owner
if TYPE_CHECKING:
    pass


@dataclass
class _BackgroundToolkitJob:
    job_id: str
    tool_name: str
    task: asyncio.Task[Any]
    owner_key: str = "default"
    status: str = "running"
    result: str | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None


_TOOLKIT_BACKGROUND_JOBS: dict[str, _BackgroundToolkitJob] = {}


class ToolkitToolWrapper(Tool):
    """Wrapper that adapts spoon-toolkit tools to spoon-bot Tool interface."""

    def __init__(self, toolkit_tool: Any, timeout_seconds: float = 3600.0):
        """
        Initialize wrapper.

        Args:
            toolkit_tool: Instance from spoon-toolkits.
            timeout_seconds: Foreground wait budget before the tool is left running
                in the background.
        """
        self._tool = toolkit_tool
        self._timeout_seconds = timeout_seconds

    @property
    def name(self) -> str:
        return getattr(self._tool, "name", self._tool.__class__.__name__)

    @property
    def description(self) -> str:
        base = getattr(self._tool, "description", "No description")
        return (
            f"{base} Background actions are also supported: execute, list_jobs, "
            "job_status, job_output, terminate_job."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        # Try to get parameters from tool
        base_schema: dict[str, Any]
        if hasattr(self._tool, "parameters"):
            base_schema = dict(self._tool.parameters)
        elif hasattr(self._tool, "input_schema"):
            base_schema = dict(self._tool.input_schema)
        elif hasattr(self._tool, "args_schema"):
            # Pydantic schema
            schema = self._tool.args_schema
            if hasattr(schema, "model_json_schema"):
                base_schema = schema.model_json_schema()
            elif hasattr(schema, "schema"):
                base_schema = schema.schema()
            else:
                base_schema = {"type": "object", "properties": {}}
        else:
            base_schema = {"type": "object", "properties": {}}

        properties = dict(base_schema.get("properties", {}))
        properties.update({
            "action": {
                "type": "string",
                "enum": ["execute", "list_jobs", "job_status", "job_output", "terminate_job"],
                "description": "Tool action. Defaults to execute.",
                "default": "execute",
            },
            "job_id": {
                "type": "string",
                "description": "Background toolkit job ID for status/output/terminate actions",
            },
        })
        return {
            "type": "object",
            "properties": properties,
        }

    async def _invoke_tool(self, call_kwargs: dict[str, Any]) -> Any:
        if hasattr(self._tool, "arun"):
            return await self._tool.arun(**call_kwargs)
        if hasattr(self._tool, "execute"):
            if asyncio.iscoroutinefunction(self._tool.execute):
                return await self._tool.execute(**call_kwargs)
            return await asyncio.to_thread(self._tool.execute, **call_kwargs)
        if hasattr(self._tool, "run"):
            return await asyncio.to_thread(self._tool.run, **call_kwargs)
        if callable(self._tool):
            return await asyncio.to_thread(self._tool, **call_kwargs)
        raise TypeError(f"Tool '{self.name}' is not callable")

    async def _refresh_job(self, job: _BackgroundToolkitJob) -> _BackgroundToolkitJob:
        if job.status != "running":
            return job
        if not job.task.done():
            return job
        try:
            result = await asyncio.shield(job.task)
            job.result = str(result) if result is not None else "Success"
            job.status = "completed"
            job.finished_at = time.time()
        except asyncio.CancelledError:
            job.error = "Cancellation requested"
            job.status = "cancelled"
            job.finished_at = time.time()
        except Exception as exc:
            job.error = str(exc)
            job.status = "failed"
            job.finished_at = time.time()
        return job

    def _attach_background_job_lifecycle(self, job: _BackgroundToolkitJob) -> None:
        """Keep background job status in sync without requiring polling actions."""

        def _on_done(done_task: asyncio.Task[Any]) -> None:
            if job.status != "running":
                return
            try:
                result = done_task.result()
                job.result = str(result) if result is not None else "Success"
                job.status = "completed"
            except asyncio.CancelledError:
                job.error = "Cancellation requested"
                job.status = "cancelled"
            except Exception as exc:  # noqa: BLE001
                job.error = str(exc)
                job.status = "failed"
            finally:
                job.finished_at = time.time()
                self._prune_completed_jobs(owner_key=job.owner_key)

        if job.task.done():
            _on_done(job.task)
            return
        job.task.add_done_callback(_on_done)

    @staticmethod
    def _is_terminal_status(status: str) -> bool:
        return status in {"completed", "failed", "cancelled", "terminated"}

    def _prune_completed_jobs(
        self,
        *,
        owner_key: str,
        keep_completed: int = 20,
    ) -> None:
        completed = [
            job
            for job in _TOOLKIT_BACKGROUND_JOBS.values()
            if job.owner_key == owner_key
            and job.tool_name == self.name
            and self._is_terminal_status(job.status)
        ]
        completed.sort(key=lambda j: j.finished_at or j.created_at, reverse=True)
        for stale_job in completed[keep_completed:]:
            _TOOLKIT_BACKGROUND_JOBS.pop(stale_job.job_id, None)

    async def _handle_background_action(self, action: str, job_id: str | None) -> str:
        owner_key = get_tool_owner()
        self._prune_completed_jobs(owner_key=owner_key)

        if action == "list_jobs":
            matching_jobs = [
                job for job in _TOOLKIT_BACKGROUND_JOBS.values()
                if job.tool_name == self.name and job.owner_key == owner_key
            ]
            if not matching_jobs:
                return f"No background jobs for toolkit tool '{self.name}'"
            lines = [f"Background jobs for toolkit tool '{self.name}':"]
            for job in matching_jobs:
                await self._refresh_job(job)
                lines.append(f"[{job.job_id}] {job.status}")
            return "\n".join(lines)

        if not job_id:
            return "Error: 'job_id' is required for this action"

        job = _TOOLKIT_BACKGROUND_JOBS.get(job_id)
        if job is None or job.tool_name != self.name or job.owner_key != owner_key:
            return f"Error: Background toolkit job not found: {job_id}"

        await self._refresh_job(job)

        if action == "job_status":
            details = job.result if job.result is not None else (job.error or "No output yet")
            return f"job_id: {job.job_id}\nstatus: {job.status}\noutput:\n{details}"

        if action == "job_output":
            if job.result is not None:
                return job.result
            if job.error is not None:
                return f"Error: {job.error}"
            return f"Background toolkit job {job.job_id} is still running; no final output yet"

        if action == "terminate_job":
            if job.task.done():
                await self._refresh_job(job)
                return f"Background toolkit job {job.job_id} already finished with status={job.status}"
            job.task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(job.task), timeout=5.0)
            except asyncio.CancelledError:
                pass
            except asyncio.TimeoutError:
                logger.warning(
                    f"Toolkit job {job.job_id} did not stop promptly after cancellation request"
                )
            await self._refresh_job(job)
            if job.status == "running":
                return (
                    f"Termination requested for toolkit job {job.job_id}, but the underlying work is still running. "
                    "This tool type does not expose incremental output or guaranteed hard-kill support."
                )
            if job.status == "cancelled":
                job.status = "terminated"
                job.finished_at = time.time()
            return f"Background toolkit job {job.job_id} stopped with status={job.status}"

        return f"Error: Unknown action '{action}'"

    async def execute(self, **kwargs: Any) -> str:
        """Execute the toolkit tool.

        Returns:
            Tool execution result as string.
            In case of error, returns a user-friendly error message.
        """
        action = kwargs.pop("action", "execute")
        job_id = kwargs.pop("job_id", None)

        if action != "execute":
            return await self._handle_background_action(action, job_id)

        try:
            owner_key = get_tool_owner()
            bg_task = asyncio.create_task(self._invoke_tool(kwargs))
            try:
                result = await asyncio.wait_for(asyncio.shield(bg_task), timeout=self._timeout_seconds)
                return str(result) if result is not None else "Success"
            except asyncio.TimeoutError:
                job = _BackgroundToolkitJob(
                    job_id=f"tk_{uuid4().hex[:10]}",
                    tool_name=self.name,
                    task=bg_task,
                    owner_key=owner_key,
                )
                _TOOLKIT_BACKGROUND_JOBS[job.job_id] = job
                self._attach_background_job_lifecycle(job)
                self._prune_completed_jobs(owner_key=owner_key)
                logger.warning(
                    f"Toolkit tool {self.name} exceeded foreground wait budget "
                    f"({self._timeout_seconds:g}s) and was left running in the background as {job.job_id}"
                )
                return (
                    f"Toolkit tool '{self.name}' exceeded the foreground wait budget "
                    f"({self._timeout_seconds:g}s) and is still running in the background.\n"
                    f"job_id: {job.job_id}\n"
                    "Use action='job_status' to inspect it, action='job_output' to fetch the final result when ready, "
                    "or action='terminate_job' to request cancellation."
                )
        except asyncio.CancelledError:
            logger.warning(f"Toolkit tool {self.name} was cancelled")
            return f"Error: Tool '{self.name}' was cancelled"
        except ConnectionError as e:
            logger.error(f"Toolkit tool {self.name} connection error: {e}")
            return f"Error: Connection failed for tool '{self.name}'"
        except PermissionError as e:
            logger.error(f"Toolkit tool {self.name} permission error: {e}")
            return f"Error: Permission denied for tool '{self.name}'"
        except ValueError as e:
            logger.error(f"Toolkit tool {self.name} value error: {e}")
            return f"Error: Invalid arguments for tool '{self.name}': {e}"
        except Exception as e:
            error_type = type(e).__name__
            logger.error(f"Toolkit tool {self.name} error ({error_type}): {e}")
            return f"Error executing '{self.name}': {str(e)}"


class ToolkitAdapter:
    """
    Adapter for loading and using spoon-toolkits tools.

    Supports lazy loading to avoid import errors when toolkit is not installed.
    """

    def __init__(self):
        """Initialize toolkit adapter."""
        self._tools: list[Tool] = []
        self._loaded_categories: set[str] = set()

    def load_crypto_tools(self) -> list[Tool]:
        """Load crypto data tools."""
        if "crypto" in self._loaded_categories:
            return [t for t in self._tools if t.name.startswith("crypto_")]

        tools = []
        try:
            from spoon_toolkits.crypto.crypto_data_tools import (
                GetTokenPriceTool,
                Get24hStatsTool,
                GetKlineDataTool,
                PriceThresholdAlertTool,
                SuddenPriceIncreaseTool,
                LendingRateMonitorTool,
            )

            toolkit_tools = [
                GetTokenPriceTool(),
                Get24hStatsTool(),
                GetKlineDataTool(),
                PriceThresholdAlertTool(),
                SuddenPriceIncreaseTool(),
                LendingRateMonitorTool(),
            ]

            for tool in toolkit_tools:
                wrapped = ToolkitToolWrapper(tool)
                tools.append(wrapped)
                self._tools.append(wrapped)

            self._loaded_categories.add("crypto")
            logger.info(f"Loaded {len(tools)} crypto tools")

        except Exception as e:
            logger.warning(f"spoon-toolkits crypto tools not available: {e}")

        return tools

    def load_blockchain_tools(self) -> list[Tool]:
        """Load blockchain data tools (Chainbase, ThirdWeb)."""
        if "blockchain" in self._loaded_categories:
            return [t for t in self._tools if "chain" in t.name.lower() or "thirdweb" in t.name.lower()]

        tools = []
        try:
            from spoon_toolkits.data_platforms.chainbase import (
                GetLatestBlockNumberTool,
                GetBlockByNumberTool,
                GetTransactionByHashTool,
                GetAccountTransactionsTool,
                GetAccountTokensTool,
                GetAccountBalanceTool,
            )

            toolkit_tools = [
                GetLatestBlockNumberTool(),
                GetBlockByNumberTool(),
                GetTransactionByHashTool(),
                GetAccountTransactionsTool(),
                GetAccountTokensTool(),
                GetAccountBalanceTool(),
            ]

            for tool in toolkit_tools:
                wrapped = ToolkitToolWrapper(tool)
                tools.append(wrapped)
                self._tools.append(wrapped)

            self._loaded_categories.add("blockchain")
            logger.info(f"Loaded {len(tools)} blockchain tools")

        except Exception as e:
            logger.warning(f"spoon-toolkits blockchain tools not available: {e}")

        return tools

    def load_security_tools(self) -> list[Tool]:
        """Load security tools (GoPlusLabs)."""
        if "security" in self._loaded_categories:
            return [t for t in self._tools if "security" in t.name.lower()]

        tools = []
        try:
            from spoon_toolkits.security.gopluslabs import get_token_risk_and_security_data

            # Wrap the function as a tool
            class TokenSecurityTool(Tool):
                @property
                def name(self) -> str:
                    return "token_security_check"

                @property
                def description(self) -> str:
                    return "Check token risk and security data using GoPlusLabs"

                @property
                def parameters(self) -> dict[str, Any]:
                    return {
                        "type": "object",
                        "properties": {
                            "token_address": {"type": "string", "description": "Token contract address"},
                            "chain_id": {"type": "string", "description": "Chain ID (e.g., '1' for Ethereum)"},
                        },
                        "required": ["token_address", "chain_id"],
                    }

                async def execute(self, **kwargs: Any) -> str:
                    import asyncio
                    result = await asyncio.to_thread(
                        get_token_risk_and_security_data,
                        kwargs.get("token_address"),
                        kwargs.get("chain_id"),
                    )
                    return str(result)

            security_tool = TokenSecurityTool()
            tools.append(security_tool)
            self._tools.append(security_tool)

            self._loaded_categories.add("security")
            logger.info(f"Loaded {len(tools)} security tools")

        except Exception as e:
            logger.warning(f"spoon-toolkits security tools not available: {e}")

        return tools

    def load_social_tools(self) -> list[Tool]:
        """Load social media tools."""
        if "social" in self._loaded_categories:
            return [t for t in self._tools if "social" in t.name.lower()]

        tools = []
        try:
            from spoon_toolkits.social_media import TwitterTool

            twitter_tool = ToolkitToolWrapper(TwitterTool())
            tools.append(twitter_tool)
            self._tools.append(twitter_tool)

            self._loaded_categories.add("social")
            logger.info(f"Loaded {len(tools)} social tools")

        except Exception as e:
            logger.warning(f"spoon-toolkits social tools not available: {e}")

        return tools

    def load_all(self, timeout_per_category: float = 5.0) -> list[Tool]:
        """Load all available toolkit tools **in parallel**.

        All categories are submitted concurrently so the total wall-clock
        time is bounded by the slowest category, not the sum of all.

        Args:
            timeout_per_category: Max seconds to spend loading each category.
                Categories that exceed this are skipped silently so startup
                is not blocked by slow network calls.
        """
        import concurrent.futures

        all_tools: list[Tool] = []
        loaders = [
            ("crypto", self.load_crypto_tools),
            ("blockchain", self.load_blockchain_tools),
            ("security", self.load_security_tools),
            ("social", self.load_social_tools),
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(loaders)) as pool:
            future_map = {
                pool.submit(loader_fn): category
                for category, loader_fn in loaders
            }

            try:
                for future in concurrent.futures.as_completed(
                    future_map, timeout=timeout_per_category + 1.0
                ):
                    category = future_map[future]
                    try:
                        tools = future.result(timeout=0.1)
                        all_tools.extend(tools)
                    except concurrent.futures.TimeoutError:
                        logger.warning(
                            f"Toolkit '{category}' loading timed out "
                            f"({timeout_per_category}s), skipping"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Toolkit '{category}' loading failed: {e}"
                        )
            except TimeoutError:
                # Some categories didn't finish in time — log and move on
                for future, category in future_map.items():
                    if not future.done():
                        logger.warning(
                            f"Toolkit '{category}' loading timed out "
                            f"({timeout_per_category}s), skipping"
                        )
                        future.cancel()

        return all_tools

    def get_loaded_tools(self) -> list[Tool]:
        """Get all loaded tools."""
        return self._tools.copy()

    @property
    def loaded_categories(self) -> set[str]:
        """Get set of loaded categories."""
        return self._loaded_categories.copy()
