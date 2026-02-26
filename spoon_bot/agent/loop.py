"""
Agent loop: the core processing engine using spoon-core SDK.

This module provides the main agent interface, integrating spoon-core's
ChatBot, SpoonReactMCP, and SkillManager with spoon-bot's native OS tools.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncGenerator

from loguru import logger

# Import spoon-core SDK components (required)
try:
    from spoon_ai.chat import ChatBot
    from spoon_ai.schema import Message, ToolCall as CoreToolCall
    from spoon_ai.llm.interface import LLMResponse
    from spoon_ai.agents.spoon_react_mcp import SpoonReactMCP
    from spoon_ai.agents.spoon_react_skill import SpoonReactSkill
    from spoon_ai.tools import BaseTool, ToolManager
    from spoon_ai.skills import SkillManager

    SPOON_CORE_AVAILABLE = True
except ImportError as e:
    logger.error(f"spoon-core SDK is required: {e}")
    raise ImportError(
        "spoon-bot requires spoon-core SDK. Install with: pip install spoon-ai"
    ) from e

try:
    from spoon_ai.tools.mcp_tool import MCPTool
    MCP_TOOL_AVAILABLE = True
    _MCP_TOOL_IMPORT_ERROR: Exception | None = None
except ImportError as e:
    MCPTool = None  # type: ignore[assignment]
    MCP_TOOL_AVAILABLE = False
    _MCP_TOOL_IMPORT_ERROR = e
    logger.warning(
        f"MCPTool import failed ({e}). MCP integrations are disabled; AgentLoop remains available."
    )

# Import spoon-bot native tools and components
from spoon_bot.agent.context import ContextBuilder
from spoon_bot.agent.tools.registry import (
    CORE_TOOLS,
    ToolRegistry,
)
from spoon_bot.agent.tools.shell import ShellTool
from spoon_bot.agent.tools.filesystem import (
    ReadFileTool,
    WriteFileTool,
    EditFileTool,
    ListDirTool,
)
from spoon_bot.agent.tools.self_config import (
    ActivateToolTool,
    SelfConfigTool,
    MemoryManagementTool,
    SelfUpgradeTool,
)
from spoon_bot.agent.tools.web3 import (
    BalanceCheckTool,
    TransferTool,
    SwapTool,
    ContractCallTool,
)
from spoon_bot.agent.tools.web import WebSearchTool, WebFetchTool
from spoon_bot.agent.tools.document import DocumentParseTool
from spoon_bot.config import AgentLoopConfig, MemSearchConfig, validate_agent_loop_params, resolve_context_window
from spoon_bot.services.spawn import SpawnTool
from spoon_bot.session.manager import SessionManager
from spoon_bot.session.store import create_session_store
from spoon_bot.memory.store import MemoryStore
from spoon_bot.exceptions import (
    SpoonBotError,
    APIKeyMissingError,
    ContextOverflowError,
    LLMError,
    LLMConnectionError,
    LLMTimeoutError,
    SkillActivationError,
    user_friendly_error,
)
from spoon_bot.services.git import GitManager
from spoon_bot.defaults import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_SHELL_TIMEOUT,
    DEFAULT_MAX_OUTPUT,
    DEFAULT_SESSION_KEY,
)

if TYPE_CHECKING:
    from spoon_bot.session.manager import Session


class AgentLoop:
    """
    The agent loop is the core processing engine using spoon-core SDK.

    It uses:
    - spoon-core's ChatBot for LLM interactions
    - spoon-core's SpoonReactMCP/SpoonReactSkill for agent orchestration
    - spoon-core's SkillManager for skill lifecycle
    - spoon-core's MCPTool for MCP server integration
    - spoon-bot's native OS tools (shell, filesystem, etc.)
    """

    # Task-agnostic next_step_prompt that replaces the crypto-centric default.
    # The original spoon-core template ("You can interact with the Neo blockchain …
    # Pick tools by matching the user's request to the tool names …") is injected
    # as a USER message every step, causing the LLM to waste iterations echoing
    # policy text instead of doing useful work.  This minimal version avoids that.
    DEFAULT_NEXT_STEP_PROMPT = (
        "Continue with the next step. "
        "When the task is fully complete, provide your final answer."
    )

    def __init__(
        self,
        workspace: Path | str | None = None,
        model: str | None = None,
        provider: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        shell_timeout: int = DEFAULT_SHELL_TIMEOUT,
        max_output: int = DEFAULT_MAX_OUTPUT,
        session_key: str = DEFAULT_SESSION_KEY,
        skill_paths: list[Path | str] | None = None,
        mcp_config: dict[str, dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        enable_skills: bool = True,
        auto_commit: bool = True,
        enabled_tools: set[str] | None = None,
        tool_profile: str | None = None,
        session_store_backend: str = "file",
        session_store_dsn: str | None = None,
        session_store_db_path: str | None = None,
        context_window: int | None = None,
        memsearch_config: MemSearchConfig | dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the agent loop.

        Args:
            workspace: Path to workspace directory.
            model: Model name to use.
            provider: LLM provider (anthropic, openai, etc.)
            api_key: API key for the LLM provider.
            base_url: Custom base URL for the provider.
            max_iterations: Maximum tool call iterations.
            shell_timeout: Shell command timeout in seconds.
            max_output: Maximum output characters for shell.
            session_key: Session identifier for persistence.
            skill_paths: Additional paths to search for skills.
            mcp_config: MCP server configurations.
            system_prompt: Custom system prompt.
            enable_skills: Whether to enable skill system.
            auto_commit: Whether to auto-commit workspace changes after each message.
            enabled_tools: Explicit set of tool names to enable. None = all.
            tool_profile: Named profile ('coding', 'web3', 'research', 'full').
            session_store_backend: Session storage backend ('file', 'sqlite', 'postgres').
            session_store_dsn: PostgreSQL DSN for 'postgres' backend.
            session_store_db_path: SQLite DB path for 'sqlite' backend.
            context_window: Override context window in tokens (auto-resolved from model if None).
        """
        # Validate parameters
        try:
            self._config = validate_agent_loop_params(
                workspace=workspace,
                model=model,
                max_iterations=max_iterations,
                shell_timeout=shell_timeout,
                max_output=max_output,
                session_key=session_key,
                skill_paths=skill_paths,
                mcp_config=mcp_config,
            )
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise ValueError(f"Invalid AgentLoop configuration: {e}") from e

        # Store config (apply defaults from spoon_bot.defaults if not provided)
        self.workspace = self._config.workspace
        self.model = model or DEFAULT_MODEL
        self.provider = provider or DEFAULT_PROVIDER
        self.api_key = api_key
        self.base_url = base_url
        self.max_iterations = self._config.max_iterations
        self.shell_timeout = self._config.shell_timeout
        self.max_output = self._config.max_output
        self.session_key = self._config.session_key
        self._enable_skills = enable_skills
        self._mcp_config = mcp_config or {}
        self._system_prompt = system_prompt
        self._auto_commit = auto_commit

        # Context window — auto-resolved from model when not explicit
        self.context_window = resolve_context_window(model, context_window)
        logger.info(f"Context window: {self.context_window:,} tokens (model={model})")

        # spoon-core components (initialized later)
        self._chatbot: ChatBot | None = None
        self._agent: SpoonReactMCP | SpoonReactSkill | None = None
        self._skill_manager: SkillManager | None = None
        self._mcp_tools: list[MCPTool] = []

        # spoon-bot components
        self.context = ContextBuilder(self.workspace)
        self.tools = ToolRegistry()

        # Session persistence — configurable backend
        _store_backend = session_store_backend or "file"
        _store_db_path = session_store_db_path
        if _store_backend == "sqlite" and not _store_db_path:
            _store_db_path = str(self.workspace / "sessions.db")
        try:
            _session_store = create_session_store(
                backend=_store_backend,
                workspace=self.workspace,
                db_path=_store_db_path,
                dsn=session_store_dsn,
            )
            self.sessions = SessionManager(workspace=self.workspace, store=_session_store)
            logger.info(f"Session store: {_store_backend}")
        except Exception as exc:
            logger.warning(f"Session store '{_store_backend}' init failed ({exc}), falling back to file")
            self.sessions = SessionManager(self.workspace)

        # Memory store — semantic (memsearch) or file-based
        self._memsearch_config: MemSearchConfig | None = None
        if memsearch_config is not None:
            if isinstance(memsearch_config, dict):
                self._memsearch_config = MemSearchConfig(**memsearch_config)
            else:
                self._memsearch_config = memsearch_config

        if self._memsearch_config and self._memsearch_config.enabled:
            try:
                from spoon_bot.memory.semantic_store import SemanticMemoryStore
                self.memory = SemanticMemoryStore(
                    self.workspace,
                    embedding_provider=self._memsearch_config.embedding_provider,
                    embedding_model=self._memsearch_config.get_embedding_model(),
                    embedding_api_key=self._memsearch_config.get_embedding_api_key(),
                    embedding_base_url=self._memsearch_config.get_embedding_base_url(),
                    milvus_uri=self._memsearch_config.milvus_uri,
                    collection=self._memsearch_config.collection,
                )
                logger.info("Using SemanticMemoryStore (memsearch)")
            except ImportError:
                logger.warning("memsearch not installed, falling back to file-based memory")
                self.memory = MemoryStore(self.workspace)
        else:
            self.memory = MemoryStore(self.workspace)

        self._git = GitManager(self.workspace) if auto_commit else None

        # Skill paths
        self._skill_paths = [self.workspace / "skills"]
        if skill_paths:
            self._skill_paths.extend([Path(p) for p in skill_paths])

        # Session
        self._session = self.sessions.get_or_create(self.session_key)

        # Inject memory context
        memory_context = self.memory.get_memory_context()
        if memory_context:
            self.context.set_memory_context(memory_context)

        # Register native tools
        self._register_native_tools()

        # Apply tool filter: explicit > profile > core default
        if enabled_tools is not None or tool_profile is not None:
            self.tools.set_tool_filter(
                enabled_tools=enabled_tools,
                tool_profile=tool_profile,
            )
        else:
            # Default: only load core tools into the agent.
            # All other tools remain in the registry and can be activated
            # dynamically via add_tool().
            self.tools.set_tool_filter(enabled_tools=set(CORE_TOOLS))

        # Track initialization state
        self._initialized = False

        active_count = len(self.tools)
        total_count = len(self.tools._tools)
        logger.info(
            f"AgentLoop created: model={model}, provider={provider}, "
            f"tools={active_count}/{total_count}, session={session_key}"
        )

    async def initialize(self) -> None:
        """Initialize spoon-core components."""
        if self._initialized:
            return

        # Initialize semantic memory if enabled
        if hasattr(self.memory, 'initialize'):
            await self.memory.initialize()

        # Build system prompt (spoon-bot context + available tool summaries)
        system_prompt = self._system_prompt or self.context.build_system_prompt()

        # Context budget hint
        system_prompt += (
            f"\n\n[Context budget: {self.context_window:,} tokens. "
            f"Be concise when context is limited.]\n"
        )

        # Append available-tools summary so the LLM knows what can be loaded
        inactive_tools = self.tools.get_inactive_tools()
        if inactive_tools:
            system_prompt += self._build_dynamic_tools_prompt(inactive_tools)

        # Create ChatBot (spoon-core LLM interface)
        self._chatbot = ChatBot(
            model_name=self.model,
            llm_provider=self.provider,
            api_key=self.api_key,
            base_url=self.base_url,
            system_prompt=system_prompt,
        )

        # Warn about potential provider/model mismatch that may cause fallback noise
        if self.provider and self.model:
            model_lower = self.model.lower()
            provider_lower = self.provider.lower()
            _provider_model_hints = {
                "anthropic": ["claude"],
                "openai": ["gpt", "o1", "o3", "o4"],
                "deepseek": ["deepseek"],
                "gemini": ["gemini"],
                "openrouter": [],  # openrouter supports all models
            }
            expected_prefixes = _provider_model_hints.get(provider_lower, [])
            if expected_prefixes:
                model_base = model_lower.rsplit("/", 1)[-1]
                if not any(model_base.startswith(p) for p in expected_prefixes):
                    logger.warning(
                        f"Model '{self.model}' may not be native to provider "
                        f"'{self.provider}'. If you see fallback errors in logs, "
                        f"consider using 'openrouter' as provider or matching "
                        f"the model to the provider."
                    )

        # Create MCP tools – expand each server into individual tools (#5)
        if self._mcp_config and not MCP_TOOL_AVAILABLE:
            logger.warning(
                "MCP configuration provided but MCPTool is unavailable "
                f"({_MCP_TOOL_IMPORT_ERROR}); skipping MCP server setup."
            )
        for name, config in self._mcp_config.items():
            if not MCP_TOOL_AVAILABLE:
                break

            mcp_tool = MCPTool(
                name=name,
                description=f"MCP server: {name}",
                mcp_config=config,
            )
            # Try to discover real server tools and create one MCPTool per tool
            if hasattr(mcp_tool, "expand_server_tools"):
                try:
                    expanded = await mcp_tool.expand_server_tools()
                    if expanded:
                        self._mcp_tools.extend(expanded)
                        logger.info(f"MCP server '{name}': expanded to {len(expanded)} tools")
                    else:
                        # Fallback: keep proxy tool (server might be offline)
                        self._mcp_tools.append(mcp_tool)
                        logger.warning(f"MCP server '{name}': no tools discovered, keeping proxy")
                except Exception as exc:
                    logger.warning(f"MCP server '{name}': expansion failed ({exc}), keeping proxy")
                    self._mcp_tools.append(mcp_tool)
            else:
                self._mcp_tools.append(mcp_tool)
                logger.info(
                    f"MCP server '{name}': expand_server_tools() unavailable in this spoon-core version; using proxy."
                )

        # Only pass active (filtered) tools + MCP tools to the agent
        active_tools = list(self.tools.get_active_tools().values()) + self._mcp_tools

        # Common agent kwargs
        agent_kwargs: dict[str, Any] = {
            "llm": self._chatbot,
            "tools": ToolManager(active_tools) if active_tools else None,
            "max_steps": self.max_iterations,
            "system_prompt": system_prompt,
            "next_step_prompt": self.DEFAULT_NEXT_STEP_PROMPT,
        }

        # Create SkillManager if enabled
        if self._enable_skills:
            import inspect
            _sm_sig = inspect.signature(SkillManager.__init__)
            _sm_kwargs: dict[str, Any] = {
                "skill_paths": [str(p) for p in self._skill_paths],
                "llm": self._chatbot,
                "auto_discover": True,
            }
            if "include_default_paths" in _sm_sig.parameters:
                _sm_kwargs["include_default_paths"] = False
            self._skill_manager = SkillManager(**_sm_kwargs)
            agent_kwargs["skill_manager"] = self._skill_manager
            self._agent = SpoonReactSkill(**agent_kwargs)
        else:
            self._agent = SpoonReactMCP(**agent_kwargs)

        # Initialize agent
        await self._agent.initialize()

        # Increase default step timeout for proxy/custom endpoints
        if self.base_url:
            self._agent._default_timeout = 300.0

        self._initialized = True
        active_count = len(self.tools.get_active_tools())
        total_count = len(self.tools._tools)
        logger.info(
            f"AgentLoop initialized: tools={active_count}/{total_count}, "
            f"mcp_servers={len(self._mcp_tools)}, "
            f"skills_enabled={self._enable_skills}"
        )

    def _register_native_tools(self) -> None:
        """Register the default native OS tools."""
        # Shell tool
        self.tools.register(ShellTool(
            timeout=self.shell_timeout,
            max_output=self.max_output,
            working_dir=str(self.workspace),
        ))

        # Filesystem tools
        self.tools.register(ReadFileTool(workspace=self.workspace))
        self.tools.register(WriteFileTool(workspace=self.workspace))
        self.tools.register(EditFileTool(workspace=self.workspace))
        self.tools.register(ListDirTool(workspace=self.workspace))

        # Self-management tools
        self.tools.register(SelfConfigTool())
        memory_tool = MemoryManagementTool()
        memory_tool.set_memory_store(self.memory)
        self.tools.register(memory_tool)
        self.tools.register(SelfUpgradeTool(workspace=self.workspace))

        # Dynamic tool activation — lets the LLM load tools on demand
        self.tools.register(ActivateToolTool(
            activate_fn=self.add_tool,
            list_inactive_fn=lambda: [
                {"name": t.name, "description": t.description}
                for t in self.tools.get_inactive_tools().values()
            ],
        ))

        # Background task tool
        self.tools.register(SpawnTool())

        # Web tools
        self.tools.register(WebSearchTool())
        self.tools.register(WebFetchTool())

        # Document processing tools
        self.tools.register(DocumentParseTool(workspace=self.workspace))

        # Web3 tools
        self.tools.register(BalanceCheckTool())
        self.tools.register(TransferTool())
        self.tools.register(SwapTool())
        self.tools.register(ContractCallTool())

        # Toolkit tools (optional)
        try:
            from spoon_bot.toolkit.adapter import ToolkitAdapter
            toolkit = ToolkitAdapter()
            for tool in toolkit.load_all():
                self.tools.register(tool)
        except ImportError:
            logger.debug("spoon-toolkits not available, skipping toolkit tools")

        logger.debug(f"Registered native tools: {self.tools.list_tools()}")

    def _estimate_token_count(self) -> int:
        """Rough token estimate for current session messages (~4 chars per token)."""
        return sum(len(m.get("content", "")) for m in self._session.messages) // 4

    def _trim_context_if_needed(self) -> None:
        """Auto-trim oldest messages if context budget approaches the limit.

        Keeps at minimum the two most recent messages (last user + assistant pair).
        Raises ContextOverflowError if even after trimming the context is too large.
        """
        if not self._session.messages:
            return

        budget = int(self.context_window * 0.75)
        estimated = self._estimate_token_count()

        if estimated <= budget:
            return

        logger.warning(
            f"Context approaching limit: ~{estimated:,} tokens "
            f"(budget: {budget:,}, window: {self.context_window:,}). "
            f"Trimming oldest messages."
        )

        messages = self._session.messages
        while len(messages) > 2 and self._estimate_token_count() > budget:
            removed = messages.pop(0)
            logger.debug(
                f"Trimmed message: role={removed.get('role')}, "
                f"len={len(removed.get('content', ''))}"
            )

        # If still over the hard limit after trimming, raise
        final_estimate = self._estimate_token_count()
        if final_estimate > self.context_window:
            raise ContextOverflowError(final_estimate, self.context_window)

    async def process(
        self,
        message: str,
        media: list[str] | None = None,
    ) -> str:
        """
        Process a user message and return the agent's response.

        Args:
            message: The user's message.
            media: Optional list of media file paths.

        Returns:
            The agent's response text.
        """
        # Ensure initialized
        if not self._initialized:
            await self.initialize()

        logger.info(f"Processing message: {message[:100]}...")

        # Refresh memory context
        try:
            memory_context = self.memory.get_memory_context()
            if memory_context:
                self.context.set_memory_context(memory_context)
        except Exception as e:
            logger.warning(f"Failed to load memory context: {e}")

        # Trim context if approaching window limit
        self._trim_context_if_needed()

        # Run agent
        try:
            result = await self._agent.run(message)

            # Extract content
            if hasattr(result, "content"):
                final_content = result.content
            else:
                final_content = str(result)

            # Filter out technical execution steps (Step 1:, Step 2:, etc.)
            final_content = self._filter_execution_steps(final_content)

        except Exception as e:
            logger.error(f"Agent processing error: {e}")
            raise

        # Save to session
        try:
            self._session.add_message("user", message)
            self._session.add_message("assistant", final_content)
            self.sessions.save(self._session)
        except Exception as e:
            logger.warning(f"Failed to save session: {e}")

        # Auto-commit workspace changes
        if self._auto_commit and self._git:
            try:
                if self._git.has_changes():
                    self._git.commit(message)
            except Exception as e:
                logger.warning(f"Failed to auto-commit: {e}")

        return final_content

    def _filter_execution_steps(self, content: str) -> str:
        """
        Filter out technical execution steps from agent output.
        Removes lines like "Step 1: Observed output of cmd..."

        Args:
            content: Raw agent output

        Returns:
            Cleaned content without execution steps
        """
        import re

        lines = content.split('\n')
        filtered_lines = []
        skip_until_blank = False

        for line in lines:
            # Skip lines that match execution step patterns
            if re.match(r'^Step \d+:', line):
                skip_until_blank = True
                continue

            # Skip lines that are part of step output
            if skip_until_blank:
                # If we hit a blank line or normal content, stop skipping
                if line.strip() == '' or not line.startswith((' ', '\t', 'Observed', 'Error', 'Security', 'Command:', 'Successfully')):
                    skip_until_blank = False
                    if line.strip():  # Add the line if it's not blank
                        filtered_lines.append(line)
                continue

            # Keep all other lines
            filtered_lines.append(line)

        # Join and clean up excessive blank lines
        result = '\n'.join(filtered_lines)
        result = re.sub(r'\n{3,}', '\n\n', result)  # Replace 3+ newlines with 2
        return result.strip()

    async def stream(
        self,
        message: str,
        media: list[str] | None = None,
        thinking: bool = False,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream a response with typed chunks.

        Args:
            message: User message.
            media: Optional media files.
            thinking: Whether to include thinking output.

        Yields:
            Dicts with keys:
              type:     "content" | "thinking" | "tool_call" | "tool_result" | "done"
              delta:    Incremental text (may be empty for non-text events)
              metadata: Extra context (tool name, args, step number, etc.)
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"Streaming message: {message[:100]}...")

        # Refresh memory context
        try:
            memory_context = self.memory.get_memory_context()
            if memory_context:
                self.context.set_memory_context(memory_context)
        except Exception as e:
            logger.warning(f"Failed to load memory context: {e}")

        full_content = ""

        # Trim context if approaching window limit
        self._trim_context_if_needed()

        # ------------------------------------------------------------------
        # Streaming uses the spoon-core run+stream pattern:
        #   1. Clear task_done + drain output_queue
        #   2. Start run(message) in background — sets task_done on finish
        #   3. Read chunks from output_queue until task_done AND queue empty
        # ------------------------------------------------------------------

        try:
            # 1. Reset streaming state
            self._agent.task_done.clear()
            while not self._agent.output_queue.empty():
                try:
                    await asyncio.wait_for(self._agent.output_queue.get(), timeout=0.1)
                except (asyncio.TimeoutError, Exception):
                    break

            # 2. Start run() in background
            async def _run_and_signal() -> None:
                try:
                    await self._agent.run(request=message)
                except Exception as exc:
                    logger.error(f"Background agent run failed: {exc}")
                    try:
                        await self._agent.output_queue.put({
                            "type": "error",
                            "delta": str(exc),
                            "metadata": {"error": str(exc), "error_code": type(exc).__name__},
                        })
                    except Exception:
                        pass
                finally:
                    self._agent.task_done.set()

            logger.debug(f"Creating bg task, agent state={self._agent.state}")
            bg_task = asyncio.create_task(_run_and_signal())

            # Force a yield to allow the background task to start
            await asyncio.sleep(0)

            # 3. Read output chunks (mirrors fixed BaseAgent.stream logic)
            oq = self._agent.output_queue
            td = self._agent.task_done
            logger.debug(f"output_queue type={type(oq).__name__}, task_done type={type(td).__name__}")
            stream_timeout = 120.0
            deadline = asyncio.get_event_loop().time() + stream_timeout
            chunk_count = 0

            logger.debug(f"Entering stream loop: td={td.is_set()}, qempty={oq.empty()}, qsize={oq.qsize()}")
            while not (td.is_set() and oq.empty()):
                if asyncio.get_event_loop().time() > deadline:
                    logger.warning("Streaming deadline reached, stopping")
                    break

                try:
                    # Use oq.get() without timeout kwarg — works for both
                    # asyncio.Queue and ThreadSafeOutputQueue. Timeout is
                    # handled by the outer asyncio.wait_for.
                    chunk = await asyncio.wait_for(oq.get(), timeout=2.0)
                    chunk_count += 1
                    logger.debug(f"Got chunk #{chunk_count}: type={type(chunk).__name__}, repr={repr(chunk)[:200]}")
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    logger.warning("Streaming cancelled")
                    break
                except Exception as e:
                    logger.warning(f"Queue get error: {type(e).__name__}: {e}")
                    continue

                chunk_type = "content"
                delta = ""
                metadata: dict[str, Any] = {}

                # -- Thinking chunks --
                if hasattr(chunk, "type") and chunk.type == "thinking":
                    chunk_type = "thinking"
                    delta = chunk.content if hasattr(chunk, "content") else str(chunk)
                elif (
                    hasattr(chunk, "metadata")
                    and isinstance(chunk.metadata, dict)
                    and chunk.metadata.get("type") == "thinking"
                ):
                    chunk_type = "thinking"
                    delta = (
                        getattr(chunk, "delta", None)
                        or getattr(chunk, "content", None)
                        or str(chunk)
                    )

                # -- Dict chunks (tool_calls, structured events) --
                elif isinstance(chunk, dict):
                    if chunk.get("type") == "error":
                        yield {
                            "type": "error",
                            "delta": chunk.get("delta", ""),
                            "metadata": chunk.get("metadata", {}),
                        }
                        continue

                    if "tool_calls" in chunk and chunk["tool_calls"]:
                        for tc in chunk["tool_calls"]:
                            # tc may be a ToolCall pydantic object or a dict
                            if isinstance(tc, dict):
                                fn = tc.get("function", {})
                                tc_id = tc.get("id", "")
                                fn_name = fn.get("name", "") if isinstance(fn, dict) else getattr(fn, "name", "")
                                fn_args = fn.get("arguments", "") if isinstance(fn, dict) else getattr(fn, "arguments", "")
                            else:
                                tc_id = getattr(tc, "id", "")
                                fn_obj = getattr(tc, "function", None)
                                fn_name = getattr(fn_obj, "name", "") if fn_obj else ""
                                fn_args = getattr(fn_obj, "arguments", "") if fn_obj else ""
                            yield {
                                "type": "tool_call",
                                "delta": "",
                                "metadata": {
                                    "id": tc_id,
                                    "name": fn_name,
                                    "arguments": fn_args,
                                },
                            }
                        continue
                    # Support both "content" and "delta" keys (#10)
                    text = chunk.get("content") or chunk.get("delta") or ""
                    if text:
                        delta = text
                        full_content += delta

                # -- Object chunks with content --
                elif hasattr(chunk, "content") and chunk.content:
                    delta = chunk.content
                    full_content += delta

                # -- Plain string chunks --
                elif isinstance(chunk, str):
                    delta = chunk
                    full_content += delta

                if delta:
                    yield {"type": chunk_type, "delta": delta, "metadata": metadata}

            logger.debug(f"Stream loop exited: td={td.is_set()}, qempty={oq.empty()}, chunks_received={chunk_count}, full_content_len={len(full_content)}")

            # Ensure background task completes
            try:
                await asyncio.wait_for(bg_task, timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                pass

            # Emit done
            yield {"type": "done", "delta": "", "metadata": {"content": full_content}}

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield {"type": "error", "delta": str(e), "metadata": {"error": str(e)}}
            yield {"type": "done", "delta": "", "metadata": {"error": str(e)}}

        # Save to session only if we got actual content
        if full_content:
            try:
                self._session.add_message("user", message)
                self._session.add_message("assistant", full_content)
                self.sessions.save(self._session)
            except Exception as e:
                logger.warning(f"Failed to save session after streaming: {e}")

    async def process_with_thinking(
        self,
        message: str,
        media: list[str] | None = None,
    ) -> tuple[str, str | None]:
        """
        Process a user message and return the agent's response with thinking content.

        Args:
            message: The user's message.
            media: Optional list of media file paths.

        Returns:
            Tuple of (response_text, thinking_content). thinking_content may be None.
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"Processing message (with thinking): {message[:100]}...")

        # Refresh memory context
        try:
            memory_context = self.memory.get_memory_context()
            if memory_context:
                self.context.set_memory_context(memory_context)
        except Exception as e:
            logger.warning(f"Failed to load memory context: {e}")

        # Trim context if approaching window limit
        self._trim_context_if_needed()

        # Run agent with thinking enabled
        try:
            run_kwargs: dict[str, Any] = {"thinking": True}
            if media:
                run_kwargs["media"] = media
            result = await self._agent.run(message, **run_kwargs)

            # Extract content and thinking
            if hasattr(result, "content"):
                final_content = result.content
            else:
                final_content = str(result)

            thinking_content = None
            if hasattr(result, "thinking_content"):
                thinking_content = result.thinking_content
            elif hasattr(result, "thinking"):
                thinking_content = result.thinking
            elif hasattr(result, "metadata") and isinstance(result.metadata, dict):
                thinking_content = result.metadata.get("thinking")

        except Exception as e:
            logger.error(f"Agent processing error: {e}")
            raise

        # Save to session
        try:
            self._session.add_message("user", message)
            self._session.add_message("assistant", final_content)
            self.sessions.save(self._session)
        except Exception as e:
            logger.warning(f"Failed to save session: {e}")

        # Auto-commit workspace changes
        if self._auto_commit and self._git:
            try:
                if self._git.has_changes():
                    self._git.commit(message)
            except Exception as e:
                logger.warning(f"Failed to auto-commit: {e}")

        return final_content, thinking_content

    # ------------------------------------------------------------------
    # Dynamic prompt helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_dynamic_tools_prompt(inactive_tools: dict[str, "Tool"]) -> str:
        """Build the 'Dynamically Loadable Tools' system-prompt section.

        Lists ALL inactive tools with their descriptions so the AI Agent
        can autonomously decide which to activate. No hardcoded topic
        mapping — the LLM reads tool descriptions and decides for itself.
        """
        lines: list[str] = [
            "\n\n## Dynamically Loadable Tools\n\n"
            "The following specialized tools are registered but not yet active. "
            "Use `activate_tool(action='list')` to refresh this list at any time, "
            "or `activate_tool(action='activate', tool_name='<name>')` to load "
            "a tool. You can activate multiple tools in one call using "
            "comma-separated names.\n\n"
            "**Read each tool's description carefully and activate the ones "
            "you need BEFORE answering the user's question. "
            "Always prefer specialized tools over generic web_search when "
            "a matching tool is available.**\n"
        ]

        for tool in inactive_tools.values():
            lines.append(f"- `{tool.name}`: {tool.description}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Dynamic tool management
    # ------------------------------------------------------------------

    def add_tool(self, name: str) -> bool:
        """
        Dynamically activate a tool and inject it into the running agent.

        The tool must already be registered in the ToolRegistry.  This method
        activates it in the filter, then adds it to the agent's ToolManager
        so it becomes available for the next LLM step.

        Args:
            name: Registered tool name to activate.

        Returns:
            True if the tool was activated, False otherwise.
        """
        tool = self.tools.get(name)
        if tool is None:
            logger.warning(f"add_tool: '{name}' is not registered")
            return False

        if not self.tools.activate_tool(name):
            logger.debug(f"add_tool: '{name}' is already active")
            return False

        # If the agent is initialized, inject the tool into its ToolManager
        if self._agent and hasattr(self._agent, "available_tools"):
            tm: ToolManager = self._agent.available_tools
            if name not in tm.tool_map:
                tm.add_tool(tool)
                logger.info(f"Injected tool '{name}' into running agent")

        return True

    def add_tools(self, *names: str) -> list[str]:
        """
        Activate multiple tools at once.

        Args:
            *names: Tool names to activate.

        Returns:
            List of tool names that were successfully activated.
        """
        activated = []
        for name in names:
            if self.add_tool(name):
                activated.append(name)
        return activated

    def remove_tool(self, name: str) -> bool:
        """
        Dynamically deactivate a tool and remove it from the running agent.

        Args:
            name: Tool name to deactivate.

        Returns:
            True if the tool was deactivated, False otherwise.
        """
        if not self.tools.deactivate_tool(name):
            return False

        # Remove from agent's ToolManager if running
        if self._agent and hasattr(self._agent, "available_tools"):
            tm: ToolManager = self._agent.available_tools
            if name in tm.tool_map:
                tm.remove_tool(name)
                logger.info(f"Removed tool '{name}' from running agent")

        return True

    def get_available_tools(self) -> list[dict[str, Any]]:
        """
        List all registered tools with their active/inactive status.

        Returns:
            List of dicts with name, description, and active flag.
        """
        return self.tools.get_all_tool_summaries()

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._session.clear()
        self.sessions.save(self._session)
        logger.info("Conversation history cleared")

    def get_history(self) -> list[dict[str, str]]:
        """Get current conversation history."""
        return self._session.get_history()

    def remember(self, fact: str, category: str = "Facts") -> None:
        """Add a fact to long-term memory."""
        if not fact or not fact.strip():
            raise ValueError("Fact cannot be empty")
        self.memory.add_memory(fact.strip(), category)

    def note(self, content: str) -> None:
        """Add a note to today's daily file."""
        if not content or not content.strip():
            raise ValueError("Note content cannot be empty")
        self.memory.add_daily_note(content.strip())

    @property
    def skills(self) -> list[str]:
        """Get available skill names."""
        if self._skill_manager:
            return self._skill_manager.list()
        return []

    @property
    def chatbot(self) -> ChatBot | None:
        """Access the underlying ChatBot."""
        return self._chatbot

    @property
    def agent(self) -> SpoonReactMCP | SpoonReactSkill | None:
        """Access the underlying spoon-core agent."""
        return self._agent


async def create_agent(
    model: str | None = None,
    provider: str | None = None,
    api_key: str | None = None,
    workspace: Path | str | None = None,
    session_key: str = "default",
    base_url: str | None = None,
    mcp_config: dict[str, dict[str, Any]] | None = None,
    enable_skills: bool = True,
    auto_commit: bool = True,
    enabled_tools: set[str] | None = None,
    tool_profile: str | None = None,
    memsearch_config: MemSearchConfig | dict[str, Any] | None = None,
    **kwargs: Any,
) -> AgentLoop:
    """
    Create and initialize an AgentLoop.

    This is the recommended way to create an agent.

    By default only core tools (shell, read_file, write_file, edit_file,
    list_dir) are loaded into the agent.  Use ``enabled_tools`` or
    ``tool_profile`` to override, or call ``agent.add_tool(name)`` after
    creation to dynamically inject additional tools.

    Args:
        model: Model name.
        provider: LLM provider (anthropic, openai, etc.)
        api_key: API key for the provider.
        workspace: Workspace directory path.
        session_key: Session identifier.
        base_url: Custom base URL for the provider.
        mcp_config: MCP server configurations.
        enable_skills: Whether to enable skill system.
        auto_commit: Whether to auto-commit workspace changes after each message.
        enabled_tools: Explicit set of tool names to enable. None = core only.
        tool_profile: Named profile ('core', 'coding', 'web3', 'research', 'full').
        **kwargs: Additional arguments for AgentLoop.

    Returns:
        Initialized AgentLoop instance.

    Example:
        >>> agent = await create_agent()
        >>> response = await agent.process("Hello!")

        >>> # Load all tools
        >>> agent = await create_agent(tool_profile="full")

        >>> # Dynamically add a tool after creation
        >>> agent = await create_agent()
        >>> agent.add_tool("web_search")
    """
    # Apply defaults from spoon_bot.defaults
    model = model or DEFAULT_MODEL
    provider = provider or DEFAULT_PROVIDER

    agent = AgentLoop(
        model=model,
        provider=provider,
        api_key=api_key,
        workspace=workspace,
        session_key=session_key,
        base_url=base_url,
        mcp_config=mcp_config,
        enable_skills=enable_skills,
        auto_commit=auto_commit,
        enabled_tools=enabled_tools,
        tool_profile=tool_profile,
        memsearch_config=memsearch_config,
        **kwargs,
    )

    await agent.initialize()
    return agent


# Export spoon-core types for convenience
__all__ = [
    "AgentLoop",
    "create_agent",
    # spoon-core types
    "ChatBot",
    "Message",
    "LLMResponse",
    "SpoonReactMCP",
    "SpoonReactSkill",
    "ToolManager",
    "MCPTool",
    "SkillManager",
]
