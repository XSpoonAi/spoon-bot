"""Integration layer for spoon-core SDK.

This module provides bridges between spoon-bot gateway and spoon-core SDK,
enabling reuse of:
- SpoonReactMCP agent (MCP-enabled ReAct agent)
- ChatBot (LLM management with multi-provider support)
- ToolManager (tool registry and execution)
- SkillManager (skill lifecycle management)
- ERC8004Client (Web3 identity authentication)
- X402PaymentService (payment gating)
- Message/LLMResponse (standardized message types)

spoon-core is a required dependency - no fallbacks.
"""

from __future__ import annotations

import os
from typing import Any, AsyncGenerator

from loguru import logger

# Import spoon-core SDK (required)
try:
    # Core chat and LLM
    from spoon_ai.chat import ChatBot
    from spoon_ai.schema import Message, LLMResponse, ToolCall as CoreToolCall

    # Agents
    from spoon_ai.agents import SpoonReactAI
    from spoon_ai.agents.spoon_react_mcp import SpoonReactMCP
    from spoon_ai.agents.spoon_react_skill import SpoonReactSkill
    from spoon_ai.agents.skill_mixin import SkillEnabledMixin

    # Tools
    from spoon_ai.tools import BaseTool as CoreBaseTool
    from spoon_ai.tools import ToolManager as CoreToolManager
    from spoon_ai.tools.mcp_tool import MCPTool

    # Skills
    from spoon_ai.skills import SkillManager as CoreSkillManager

    # Graph orchestration
    from spoon_ai.graph import StateGraph, CompiledGraph
    from spoon_ai.graph.config import ParallelGroupConfig

    logger.info("spoon-core SDK loaded successfully")

except ImportError as e:
    logger.error(f"spoon-core SDK is required: {e}")
    raise ImportError(
        "spoon-bot requires spoon-core SDK. Install with: pip install spoon-ai"
    ) from e

# Optional modules from spoon-core
try:
    from spoon_ai.identity import ERC8004Client, DIDResolver
    _IDENTITY_AVAILABLE = True
except ImportError:
    ERC8004Client = None
    DIDResolver = None
    _IDENTITY_AVAILABLE = False
    logger.debug("spoon-core identity module not available")

try:
    from spoon_ai.payments import X402PaymentService
    _PAYMENTS_AVAILABLE = True
except ImportError:
    X402PaymentService = None
    _PAYMENTS_AVAILABLE = False
    logger.debug("spoon-core payments module not available")


def is_spoon_core_available() -> bool:
    """Check if spoon-core SDK is available. Always returns True since it's required."""
    return True


def get_available_modules() -> dict[str, bool]:
    """Get dictionary of available spoon-core modules."""
    return {
        "core": True,
        "identity": _IDENTITY_AVAILABLE,
        "payments": _PAYMENTS_AVAILABLE,
    }


class SpoonCoreAgent:
    """
    Wrapper around spoon-core's SpoonReactMCP agent.

    Provides unified interface for agent execution using spoon-core SDK.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        provider: str = "anthropic",
        system_prompt: str | None = None,
        max_steps: int = 15,
        tools: list[Any] | None = None,
        mcp_servers: dict[str, dict] | None = None,
        enable_skills: bool = True,
        skill_paths: list[str] | None = None,
    ):
        """
        Initialize SpoonCoreAgent.

        Args:
            model: Model name to use.
            provider: LLM provider (anthropic, openai, etc.)
            system_prompt: Custom system prompt.
            max_steps: Maximum reasoning steps.
            tools: List of tool instances.
            mcp_servers: MCP server configurations.
            enable_skills: Whether to enable skill system.
            skill_paths: Paths to search for skills.
        """
        self._model = model
        self._provider = provider
        self._system_prompt = system_prompt
        self._max_steps = max_steps
        self._tools = tools or []
        self._mcp_servers = mcp_servers or {}
        self._enable_skills = enable_skills
        self._skill_paths = skill_paths or []

        self._agent: SpoonReactMCP | SpoonReactSkill | None = None
        self._chatbot: ChatBot | None = None
        self._tool_manager: CoreToolManager | None = None
        self._skill_manager: CoreSkillManager | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the agent and all components."""
        if self._initialized:
            return

        # Create ChatBot
        self._chatbot = ChatBot(
            model_name=self._model,
            llm_provider=self._provider,
            system_prompt=self._system_prompt,
        )

        # Create MCP tools
        mcp_tools = []
        for name, config in self._mcp_servers.items():
            mcp_tool = MCPTool(
                name=name,
                description=f"MCP server: {name}",
                mcp_config=config,
            )
            mcp_tools.append(mcp_tool)

        all_tools = self._tools + mcp_tools
        if all_tools:
            self._tool_manager = CoreToolManager(all_tools)

        # Create SkillManager if enabled
        if self._enable_skills:
            self._skill_manager = CoreSkillManager(
                skill_paths=self._skill_paths,
                llm=self._chatbot,
                auto_discover=True,
            )

            # Create SpoonReactSkill agent
            self._agent = SpoonReactSkill(
                llm=self._chatbot,
                tools=self._tool_manager,
                skill_manager=self._skill_manager,
                max_steps=self._max_steps,
            )
        else:
            # Create SpoonReactMCP agent
            self._agent = SpoonReactMCP(
                llm=self._chatbot,
                tools=self._tool_manager,
                max_steps=self._max_steps,
            )

        # Initialize agent
        await self._agent.initialize()

        self._initialized = True
        logger.info(
            f"SpoonCoreAgent initialized: model={self._model}, "
            f"provider={self._provider}, "
            f"mcp_servers={len(mcp_tools)}, "
            f"skills_enabled={self._enable_skills}"
        )

    async def run(
        self,
        message: str,
        session_key: str = "default",
        stream: bool = False,
    ) -> str | AsyncGenerator[str, None]:
        """
        Run the agent with a message.

        Args:
            message: User message.
            session_key: Session identifier.
            stream: Whether to stream response.

        Returns:
            Agent response (string or async generator for streaming).
        """
        if not self._initialized:
            await self.initialize()

        if stream:
            return self._stream(message)
        else:
            result = await self._agent.run(message)
            return result.content if hasattr(result, "content") else str(result)

    async def _stream(self, message: str) -> AsyncGenerator[str, None]:
        """Stream response from spoon-core agent."""
        async for chunk in self._agent.stream(message):
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content
            elif isinstance(chunk, str):
                yield chunk

    @property
    def tools(self) -> list[Any]:
        """Get available tools."""
        if self._tool_manager:
            return list(self._tool_manager.tools.values())
        return self._tools

    @property
    def skills(self) -> list[str]:
        """Get available skill names."""
        if self._skill_manager:
            return self._skill_manager.list()
        return []

    def get_status(self) -> dict[str, Any]:
        """Get agent status."""
        return {
            "initialized": self._initialized,
            "model": self._model,
            "provider": self._provider,
            "tools_count": len(self.tools),
            "skills_count": len(self.skills),
            "mcp_servers": list(self._mcp_servers.keys()),
        }


class SpoonCoreIdentity:
    """
    Wrapper around spoon-core's ERC8004 identity system.

    Provides Web3-based authentication when spoon-core identity module is available.
    """

    def __init__(
        self,
        rpc_url: str | None = None,
        private_key: str | None = None,
    ):
        """
        Initialize identity client.

        Args:
            rpc_url: Ethereum RPC URL.
            private_key: Private key for signing (optional).
        """
        self._rpc_url = rpc_url or os.environ.get("ETH_RPC_URL")
        self._private_key = private_key or os.environ.get("PRIVATE_KEY")
        self._client: Any = None

        if _IDENTITY_AVAILABLE and ERC8004Client:
            try:
                self._client = ERC8004Client(
                    rpc_url=self._rpc_url,
                    private_key=self._private_key,
                )
                logger.info("ERC8004 identity client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize ERC8004 client: {e}")

    @property
    def available(self) -> bool:
        """Check if identity system is available."""
        return self._client is not None

    async def resolve_did(self, did: str) -> dict[str, Any] | None:
        """
        Resolve a DID document.

        Args:
            did: DID string (e.g., "did:spoon:agent:xyz")

        Returns:
            DID document or None if not found.
        """
        if not self._client:
            return None

        try:
            return await self._client.resolve_agent(did)
        except Exception as e:
            logger.error(f"Failed to resolve DID {did}: {e}")
            return None

    async def verify_signature(
        self,
        message: str,
        signature: str,
        address: str,
    ) -> bool:
        """
        Verify a signed message.

        Args:
            message: Original message.
            signature: Signature to verify.
            address: Expected signer address.

        Returns:
            True if signature is valid.
        """
        if not self._client:
            return False

        try:
            return await self._client.verify_signature(message, signature, address)
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False


class SpoonCorePayments:
    """
    Wrapper around spoon-core's X402 payment system.

    Provides payment gating for API endpoints when spoon-core payments module is available.
    """

    def __init__(self):
        """Initialize payment service."""
        self._service: Any = None

        if _PAYMENTS_AVAILABLE and X402PaymentService:
            try:
                self._service = X402PaymentService()
                logger.info("X402 payment service initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize X402 service: {e}")

    @property
    def available(self) -> bool:
        """Check if payment system is available."""
        return self._service is not None

    async def verify_payment(self, payment_header: str) -> dict[str, Any] | None:
        """
        Verify a payment header.

        Args:
            payment_header: X-Payment header value.

        Returns:
            Payment receipt or None if invalid.
        """
        if not self._service:
            return None

        try:
            return await self._service.verify(payment_header)
        except Exception as e:
            logger.error(f"Payment verification failed: {e}")
            return None

    async def get_requirements(self, endpoint: str) -> dict[str, Any]:
        """
        Get payment requirements for an endpoint.

        Args:
            endpoint: API endpoint path.

        Returns:
            Payment requirements specification.
        """
        if not self._service:
            return {"required": False}

        return await self._service.get_requirements(endpoint)


# Export spoon-core types for type hints
__all__ = [
    "is_spoon_core_available",
    "get_available_modules",
    "SpoonCoreAgent",
    "SpoonCoreIdentity",
    "SpoonCorePayments",
    # Re-export spoon-core types
    "ChatBot",
    "Message",
    "LLMResponse",
    "SpoonReactMCP",
    "SpoonReactSkill",
    "CoreToolManager",
    "MCPTool",
    "CoreSkillManager",
    "StateGraph",
    "ParallelGroupConfig",
]
