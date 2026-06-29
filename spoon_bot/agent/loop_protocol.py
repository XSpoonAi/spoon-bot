"""Tool protocol and response helpers for AgentLoop.

ponytail: mixin split only; keep process/stream in loop.py.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

try:
    from spoon_ai.schema import Message, AgentState
except ImportError as e:
    logger.error(f"spoon-core SDK is required: {e}")
    raise ImportError(
        "spoon-bot requires spoon-core SDK. Install with: pip install spoon-ai-sdk"
    ) from e

from spoon_bot.agent.context_snapshot import write_llm_context_snapshot
from spoon_bot.agent.execution_ledger import (
    ExecutionLedger,
)
from spoon_bot.agent.session_compact import (
    build_recent_session_turns_payload,
    build_session_compact_context,
)
from spoon_bot.agent.request_hints import (
    request_is_bare_continuation,
    request_is_plain_continuation_only,
)
from spoon_bot.agent.tools.shell import ShellTool
from spoon_bot.agent.tools.execution_context import (
    bind_request_execution_hints,
    consume_captured_tool_output,
    get_request_execution_hints,
    normalize_tool_arguments,
    sanitize_tool_arguments_for_history,
    track_tool_invocations,
)
from spoon_bot.agent.turn_verifiers import (
    build_tool_event_synthesis_brief,
    build_user_facing_tool_event_answer,
    build_user_facing_tool_evidence_answer,
    dominant_non_latin_scripts,
    final_answer_denies_available_tool_evidence,
    latest_tool_event_has_active_background_job,
    latest_tool_event_has_user_summary_marker,
    read_only_tool_turn_needs_continuation,
    should_run_skill_contract_check,
    skill_contract_needs_continuation,
)
from spoon_bot.exceptions import (
    LLMTimeoutError,
)
from spoon_bot.utils.retry import (
    DEFAULT_RETRY_CONFIG,
    RetryConfig,
    with_provider_retry,
)

if TYPE_CHECKING:
    pass

AgentLoop: Any = None

from spoon_bot.agent.loop_state import _DEFAULT_PROVIDER_ASK_TIMEOUT, _MISSING

_REPEATED_READ_RECOVERY_THRESHOLD = 2
_CLIENT_STREAM_TOOL_RESULT_LIMIT = 80_000
_MODEL_STREAM_TOOL_RESULT_LIMIT = 80_000
_CLIENT_STREAM_TOOL_DELTA_LIMIT = 12_000
_FINAL_ANSWER_SYNTHESIS_TIMEOUT = 45.0
_FINAL_ANSWER_SYNTHESIS_TOOL_EVENT_LIMIT = 16
_TASK_COMPLETION_VERDICT_TIMEOUT = 30.0
_MACHINE_READABLE_FORMAT_RE = re.compile(r"(?i)\b(?:json|yaml|xml|csv|raw)\b")
_FINAL_ANSWER_SYNTHESIS_SYSTEM_PROMPT = (
    "You are the final-answer synthesis stage for an autonomous tool-using "
    "agent. Write only the final answer to the user. Use the supplied tool "
    "evidence, do not call tools, do not invent missing progress, and do not "
    "expose internal runtime markers, raw JSON, tool transcripts, or planning. "
    "The user's request is intent, not proof; never list requested checklist "
    "items as completed unless the evidence explicitly says they completed. "
    "Do not add generic troubleshooting guides, manual-install tutorials, "
    "example commands, or option menus unless those exact next steps appear in "
    "the evidence or the user explicitly asked for guidance. If blocked, state "
    "the evidence-backed blocker and only the evidence-backed next step. "
    "Write the answer as a concise, conversational conclusion for a person: "
    "what happened, what the current state is, and what remains if anything. "
    "Treat local commands, paths, status tokens, and exception text as internal "
    "evidence. Unless the user explicitly asked for raw commands or logs, "
    "explain their meaning instead of copying them literally. "
    "When the evidence contains terminal user-facing summaries or completed "
    "stateful tool results, summarize those results directly; do not turn old "
    "SKILL.md command examples or setup notes into proposed future work. "
    "Match the newest user's natural language unless they explicitly requested "
    "another language or a machine-readable format. Short continuation messages "
    "still define the response language; do not answer in the language of older "
    "requests or tool evidence just because the evidence is written that way. "
    "Keep it concise and human."
)
_TASK_COMPLETION_VERDICT_SYSTEM_PROMPT = (
    "You are a domain-neutral completion verifier for an autonomous "
    "tool-using agent. Compare the newest user request, the assistant draft, "
    "and the supplied tool evidence. Return only compact JSON with keys "
    "`status`, `reason`, and `next_focus`. Use status `needs_continuation` "
    "only when the evidence does not satisfy the requested outcome and another "
    "tool action can reasonably make progress. Use status `complete` when the "
    "evidence satisfies the request, or when the evidence shows a real blocker "
    "that should be reported instead of retried. Use status `awaiting_user` "
    "when the assistant draft is waiting for a user choice, confirmation, "
    "missing value, or permission before another side effect; this is terminal "
    "for the current turn unless the newest user request already clearly "
    "authorized that exact next action. Do not infer user consent from the "
    "assistant draft, prior assistant questions, continuation prompts, or "
    "conversation history. Do not use product-specific rules, route names, "
    "repository names, game names, or natural-language phrase matching. The "
    "user request is intent, not proof. For countable or repeated outcomes, "
    "compare the requested count and scope against concrete tool evidence; a "
    "stage summary is evidence for that stage only, not proof that the "
    "remaining requested repetitions are complete."
)


class LoopProtocolMixin:
    def _extract_last_assistant_content(self) -> str:
        """Extract the last assistant message content from the agent's memory.

        Used as a fallback when toolcall.run() returns "No results" but the
        LLM actually produced meaningful content stored in memory.
        """
        try:
            if not hasattr(self._agent, "memory"):
                return ""
            messages = (
                self._agent.memory.get_messages()
                if hasattr(self._agent.memory, "get_messages")
                else []
            )
            # Walk backwards to find the last assistant message with content
            for msg in reversed(messages):
                role = getattr(msg, "role", None)
                # role may be an enum (Role.ASSISTANT) or a string
                role_str = role.value if hasattr(role, "value") else str(role)
                if role_str != "assistant":
                    continue
                # Prefer .text_content (handles multimodal), fall back to .content
                text = getattr(msg, "text_content", None) or getattr(msg, "content", None)
                if text and isinstance(text, str) and text.strip():
                    # Skip internal sentinel messages
                    if AgentLoop._is_internal_completion_sentinel(text):
                        continue
                    logger.info(f"Extracted fallback content from memory (len={len(text)})")
                    return text.strip()
        except Exception as exc:
            logger.warning(f"Failed to extract content from agent memory: {exc}")
        return ""

    @classmethod
    def _serialize_message_content(cls, content: Any) -> Any:
        """Convert message content into JSON-serializable Python objects."""
        if content is None or isinstance(content, (str, int, float, bool)):
            return content
        if isinstance(content, list):
            return [cls._serialize_message_content(item) for item in content]
        if isinstance(content, dict):
            return {
                str(key): cls._serialize_message_content(value) for key, value in content.items()
            }
        if hasattr(content, "model_dump"):
            try:
                dumped = content.model_dump(exclude_none=True)
            except TypeError:
                dumped = content.model_dump()
            return cls._serialize_message_content(dumped)

        extracted: dict[str, Any] = {}
        for attr in (
            "type",
            "text",
            "image_url",
            "source",
            "file_path",
            "media_type",
            "filename",
            "name",
            "url",
            "detail",
            "data",
        ):
            if hasattr(content, attr):
                extracted[attr] = cls._serialize_message_content(getattr(content, attr))
        if extracted:
            return extracted
        return str(content)

    @classmethod
    def _multimodal_content_summary(cls, content: list[Any], max_text_chars: int) -> str:
        """Summarize multimodal content while dropping heavy binary payloads."""
        serialized = cls._serialize_message_content(content)
        if not isinstance(serialized, list):
            text = str(serialized or "")
            return text[:max_text_chars] if max_text_chars and len(text) > max_text_chars else text

        text_parts: list[str] = []
        image_count = 0
        document_count = 0
        file_refs: list[str] = []

        for block in serialized:
            if isinstance(block, dict):
                block_type = str(block.get("type") or "")
                if block_type == "text":
                    text = str(block.get("text") or "")
                    if text:
                        text_parts.append(text)
                    continue
                if block_type in {"image", "image_url"}:
                    image_count += 1
                    continue
                if block_type == "document":
                    document_count += 1
                    continue
                if block_type == "file":
                    file_path = str(block.get("file_path") or "")
                    if file_path:
                        file_refs.append(Path(file_path).name or file_path)
                    continue
            elif block:
                text_parts.append(str(block))

        text = "\n".join(part for part in text_parts if part).strip()
        if max_text_chars and len(text) > max_text_chars:
            text = text[:max_text_chars] + f"\n...[truncated {len(text) - max_text_chars} chars]"

        summary_parts = [text] if text else []
        if image_count:
            summary_parts.append(
                f"[{image_count} image attachment(s) omitted during context compression]"
            )
        if document_count:
            summary_parts.append(
                f"[{document_count} document attachment(s) omitted during context compression]"
            )
        if file_refs:
            shown = ", ".join(file_refs[:3])
            more = "" if len(file_refs) <= 3 else f" (+{len(file_refs) - 3} more)"
            summary_parts.append(f"[File attachment reference(s): {shown}{more}]")
        if not summary_parts:
            summary_parts.append("[Multimodal content omitted during context compression]")
        return "\n".join(summary_parts)

    @classmethod
    def _compress_message_content(cls, content: Any, max_chars: int) -> Any:
        """Truncate text content and summarize multimodal content for recovery."""
        if isinstance(content, str):
            if len(content) <= max_chars:
                return content
            return content[:max_chars] + f"\n...[truncated {len(content) - max_chars} chars]"
        if isinstance(content, list):
            return cls._multimodal_content_summary(content, max_chars)
        return content

    @classmethod
    def _compact_runtime_message_content(cls, message: Any, max_chars: int) -> Any:
        """Compact older runtime messages without preserving stale assistant authority."""
        role = cls._message_role_value(message)
        content = getattr(message, "content", None)

        if role == "assistant":
            tool_calls = getattr(message, "tool_calls", None) or []
            if tool_calls:
                if not content:
                    return content
                return (
                    "[assistant tool-call turn compacted; exact tool arguments/results "
                    "are recoverable via search_history]"
                )
            if content in (None, ""):
                return content
            if isinstance(content, list):
                return (
                    "[assistant multimodal reply compacted; earlier assistant analysis may be stale. "
                    "Prioritize the latest user request and current tool evidence. "
                    "Use search_history to recover exact earlier user/tool facts; search assistant replies "
                    "only when their literal wording is explicitly needed.]"
                )
            return (
                "[assistant reply compacted; earlier assistant analysis/conclusions may be stale. "
                "Prioritize the latest user request and current tool evidence. "
                "Use search_history to recover exact earlier user/tool facts; search assistant replies "
                "only when their literal wording is explicitly needed.]"
            )

        if role == "tool":
            text_content = content if isinstance(content, str) else ""
            file_header = re.match(r"^\[file:\s*([^\]\n]+)\]\s*\n", text_content.strip())
            if not file_header:
                file_header = re.match(
                    r"^Observed output of cmd [^\n]* execution:\s*\[file:\s*([^\]\n]+)\]\s*\n",
                    text_content.strip(),
                )
            if file_header:
                return (
                    f"[file: {file_header.group(1)}] historical file reference only; "
                    "read the current local file again for exact text"
                )
            compressed = cls._compress_message_content(content, max_chars)
            if compressed == content:
                return content
            if isinstance(compressed, str):
                tool_name = getattr(message, "name", None) or "tool"
                return f"[{tool_name} result compacted]\n{compressed}"
            return compressed

        if role == "user":
            compressed = cls._compress_message_content(content, max_chars)
            if compressed == content:
                return content
            if isinstance(compressed, str):
                return f"[earlier user message compacted]\n{compressed}"
            return compressed

        return cls._compress_message_content(content, max_chars)

    @classmethod
    def _message_content_char_count(cls, content: Any) -> int:
        """Return an approximate character count for text and multimodal payloads."""
        if content is None:
            return 0
        if isinstance(content, str):
            return len(content)
        if not isinstance(content, list):
            return len(str(content))

        total = 0
        serialized = cls._serialize_message_content(content)
        for block in serialized if isinstance(serialized, list) else [serialized]:
            if isinstance(block, dict):
                block_type = str(block.get("type") or "")
                if block_type == "text":
                    total += len(str(block.get("text") or ""))
                    continue
                if block_type == "image_url":
                    total += len(str((block.get("image_url") or {}).get("url") or ""))
                    continue
                if block_type == "image":
                    source = block.get("source") or {}
                    total += len(str(source.get("media_type") or ""))
                    total += len(str(source.get("data") or ""))
                    continue
                if block_type == "document":
                    source = block.get("source") or {}
                    total += len(str(source.get("media_type") or ""))
                    total += len(str(source.get("data") or ""))
                    total += len(str(block.get("filename") or ""))
                    continue
                if block_type == "file":
                    total += len(str(block.get("file_path") or ""))
                    total += len(str(block.get("media_type") or ""))
                    continue
                total += len(json.dumps(block, sort_keys=True, ensure_ascii=True, default=str))
                continue
            total += len(str(block))
        return total

    @classmethod
    def _msg_char_count(cls, msg) -> int:
        """Return total character count of a message's content."""
        return cls._message_content_char_count(getattr(msg, "content", None))

    def _estimate_runtime_tokens(self) -> int:
        """Rough token estimate from the agent's runtime messages (~4 chars/token)."""
        if not self._agent or not hasattr(self._agent, "memory"):
            return 0
        return sum(self._msg_char_count(m) for m in self._agent.memory.messages) // 4

    def _build_session_recall_context(self, current_message: str) -> str:
        """Build a compact same-session context block for follow-up questions."""
        return build_session_compact_context(
            getattr(self, "_session", None),
            current_message,
            resume_latest_user_turn=request_is_bare_continuation(current_message),
        )

    def _is_next_step_user_msg(self, msg) -> bool:
        """True when *msg* looks like an injected next_step_prompt (not a real user message)."""
        role = getattr(msg, "role", None)
        if hasattr(role, "value"):
            role = role.value
        if role != "user":
            return False
        text = msg.content if isinstance(msg.content, str) else ""
        return (
            text == self.DEFAULT_NEXT_STEP_PROMPT
            or text.startswith("[ORIGINAL USER REQUEST]")
            or text.startswith("[INTERNAL ")
            or text.startswith("[TURN PRIORITY]:")
            or text.startswith("## Active Request Context")
            or text.startswith("## Current Session Compact")
            or text.startswith("Focus on the user")
        )

    @staticmethod
    def _callable_accepts_kwarg(func: Any, kwarg: str) -> bool:
        """Return True when *func* can accept *kwarg* as a keyword argument."""
        import inspect

        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):
            return False

        if any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
        ):
            return True
        param = signature.parameters.get(kwarg)
        return bool(
            param
            and param.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        )

    @staticmethod
    def _message_role_value(msg: Any) -> str | None:
        """Return the normalized string role value for a runtime message."""
        if isinstance(msg, dict):
            role = msg.get("role")
        else:
            role = getattr(msg, "role", None)
        return role.value if hasattr(role, "value") else role

    @staticmethod
    def _message_tool_calls_value(msg: Any) -> list:
        """Return tool_calls from object or dict payloads."""
        if isinstance(msg, dict):
            raw = msg.get("tool_calls")
        else:
            raw = getattr(msg, "tool_calls", None)
        return raw if isinstance(raw, list) else []

    @staticmethod
    def _tool_call_id_value(tool_call: Any) -> str | None:
        """Return a tool_call id from object or dict payloads."""
        if isinstance(tool_call, dict):
            value = tool_call.get("id")
        else:
            value = getattr(tool_call, "id", None)
        return str(value) if value else None

    @staticmethod
    def _message_tool_call_id_value(msg: Any) -> str | None:
        """Return a tool result's tool_call_id from object or dict payloads."""
        if isinstance(msg, dict):
            value = msg.get("tool_call_id") or msg.get("id")
        else:
            value = getattr(msg, "tool_call_id", None) or getattr(msg, "id", None)
        return str(value) if value else None

    @staticmethod
    def _set_message_tool_calls_value(msg: Any, value: list | None) -> None:
        """Set an assistant message's tool calls on object or dict payloads."""
        if isinstance(msg, dict):
            msg["tool_calls"] = value
        else:
            msg.tool_calls = value

    @classmethod
    def _collect_offered_tool_call_ids(cls, messages: list) -> set[str]:
        """Collect tool_call ids offered by assistant messages."""
        offered_ids: set[str] = set()
        for msg in messages:
            for tool_call in cls._message_tool_calls_value(msg):
                tool_call_id = cls._tool_call_id_value(tool_call)
                if tool_call_id:
                    offered_ids.add(tool_call_id)
        return offered_ids

    @classmethod
    def _collect_answered_tool_call_ids(cls, messages: list) -> set[str]:
        """Collect tool_call ids answered by tool-result messages."""
        answered_ids: set[str] = set()
        for msg in messages:
            if cls._message_role_value(msg) != "tool":
                continue
            tool_call_id = cls._message_tool_call_id_value(msg)
            if tool_call_id:
                answered_ids.add(tool_call_id)
        return answered_ids

    @classmethod
    def _tool_call_details_by_id(cls, messages: list) -> dict[str, tuple[str, Any]]:
        """Return tool-call name/arguments keyed by id from runtime memory."""
        details: dict[str, tuple[str, Any]] = {}
        for msg in messages:
            if cls._message_role_value(msg) != "assistant":
                continue
            for tool_call in cls._message_tool_calls_value(msg):
                tool_call_id = cls._tool_call_id_value(tool_call)
                if not tool_call_id or tool_call_id in details:
                    continue
                name, arguments = cls._tool_call_name_and_arguments(tool_call)
                details[tool_call_id] = (name, arguments)
        return details

    @classmethod
    def _adjust_message_start_to_preserve_tool_context(
        cls,
        messages: list,
        start_index: int,
        *,
        floor: int = 0,
    ) -> int:
        """Expand a kept-range start backward so tool trajectories stay intact."""
        if not messages:
            return max(0, floor)

        adjusted_index = max(floor, min(start_index, len(messages)))
        if adjusted_index <= floor or adjusted_index >= len(messages):
            return adjusted_index

        kept_answered_ids = cls._collect_answered_tool_call_ids(messages[adjusted_index:])
        if not kept_answered_ids:
            return adjusted_index

        kept_offered_ids = cls._collect_offered_tool_call_ids(messages[adjusted_index:])
        needed_ids = {
            tool_call_id
            for tool_call_id in kept_answered_ids
            if tool_call_id not in kept_offered_ids
        }
        if not needed_ids:
            return adjusted_index

        for index in range(adjusted_index - 1, floor - 1, -1):
            message = messages[index]
            if cls._message_role_value(message) != "assistant":
                continue

            matched_here = False
            for tool_call in cls._message_tool_calls_value(message):
                tool_call_id = cls._tool_call_id_value(tool_call)
                if tool_call_id and tool_call_id in needed_ids:
                    needed_ids.discard(tool_call_id)
                    matched_here = True

            if matched_here:
                adjusted_index = index
            if not needed_ids:
                break

        return adjusted_index

    @classmethod
    def _reorder_tool_messages(cls, messages: list) -> int:
        """Move tool results to immediately follow the issuing assistant turn."""
        if not messages:
            return 0

        claimed_tool_indices: set[int] = set()
        tool_messages_by_assistant_index: dict[int, list] = {}
        assistant_index_by_tool_call_id: dict[str, int] = {}

        for index, message in enumerate(messages):
            if cls._message_role_value(message) != "assistant":
                continue
            for tool_call in cls._message_tool_calls_value(message):
                tool_call_id = cls._tool_call_id_value(tool_call)
                if tool_call_id and tool_call_id not in assistant_index_by_tool_call_id:
                    assistant_index_by_tool_call_id[tool_call_id] = index

        for index, message in enumerate(messages):
            if cls._message_role_value(message) != "tool":
                continue

            tool_call_id = cls._message_tool_call_id_value(message)
            if not tool_call_id:
                continue

            assistant_index = assistant_index_by_tool_call_id.get(tool_call_id)
            if assistant_index is None:
                continue
            tool_messages_by_assistant_index.setdefault(assistant_index, []).append(message)
            claimed_tool_indices.add(index)

        if not claimed_tool_indices:
            return 0

        original_ids = [id(message) for message in messages]
        reordered_messages: list = []
        for index, message in enumerate(messages):
            if index in claimed_tool_indices:
                continue

            reordered_messages.append(message)
            if cls._message_role_value(message) != "assistant":
                continue

            tool_calls = cls._message_tool_calls_value(message)
            if not tool_calls:
                continue

            tool_order = {
                cls._tool_call_id_value(tool_call): position
                for position, tool_call in enumerate(tool_calls)
            }
            matched_tool_messages = tool_messages_by_assistant_index.get(index, [])
            matched_tool_messages.sort(
                key=lambda item: tool_order.get(
                    cls._message_tool_call_id_value(item),
                    len(tool_order),
                )
            )
            reordered_messages.extend(matched_tool_messages)

        reordered_ids = [id(message) for message in reordered_messages]
        if reordered_ids == original_ids:
            return 0

        moved = sum(
            1
            for original_id, reordered_id in zip(original_ids, reordered_ids)
            if original_id != reordered_id
        )
        messages[:] = reordered_messages
        logger.info(f"Reordered {moved} runtime message positions to restore tool adjacency")
        return moved

    @classmethod
    def _repair_tool_pairing(
        cls,
        messages: list,
        *,
        drop_unanswered_tool_calls: bool = True,
    ) -> int:
        """Remove orphaned tool results and tool calls after message deletion.

        Ensures every tool_call_id in a tool-result message has a matching
        tool_calls entry in a preceding assistant message, and vice-versa.
        Also removes tool-role messages with no tool_call_id at all (e.g.
        injected from session history without metadata).
        Without this, the LLM API rejects the conversation.

        When ``drop_unanswered_tool_calls`` is false, this keeps assistant
        tool_calls that are still in flight so live runtime normalization does
        not corrupt the next provider request.

        Returns the number of messages/calls removed.
        """
        removed = 0

        offered_ids = cls._collect_offered_tool_call_ids(messages)
        i = 0
        while i < len(messages):
            msg = messages[i]
            role = cls._message_role_value(msg)
            tool_call_id = cls._message_tool_call_id_value(msg)

            if role == "tool" and not tool_call_id:
                del messages[i]
                removed += 1
                continue
            if tool_call_id and tool_call_id not in offered_ids:
                del messages[i]
                removed += 1
                continue
            i += 1

        if drop_unanswered_tool_calls:
            answered_ids = cls._collect_answered_tool_call_ids(messages)
            for msg in messages:
                tc_list = cls._message_tool_calls_value(msg)
                if not tc_list:
                    continue
                original_len = len(tc_list)
                tc_list[:] = [tc for tc in tc_list if cls._tool_call_id_value(tc) in answered_ids]
                if len(tc_list) < original_len:
                    removed += original_len - len(tc_list)
                if not tc_list:
                    cls._set_message_tool_calls_value(msg, None)

        if removed:
            logger.info(f"Repaired tool pairing: removed {removed} orphaned messages/calls")
        return removed

    @classmethod
    def _normalize_runtime_tool_context(
        cls,
        messages: list,
        *,
        finalized: bool = False,
    ) -> int:
        """Repair runtime tool sequencing without pruning live tool trajectories."""
        normalized = cls._reorder_tool_messages(messages)
        normalized += cls._repair_tool_pairing(
            messages,
            drop_unanswered_tool_calls=finalized,
        )
        return normalized

    def _uses_strict_tool_turn_order(self) -> bool:
        """True for providers/models that reject non-adjacent function call turns.

        OpenAI-compatible providers and Gemini require tool-result messages to immediately
        follow the assistant message that issued the tool_calls.
        """
        provider_raw = getattr(self, "provider", None)
        model_raw = getattr(self, "model", None)
        base_url_raw = getattr(self, "base_url", None)

        provider = provider_raw.strip().lower() if isinstance(provider_raw, str) else ""
        model = model_raw.strip().lower() if isinstance(model_raw, str) else ""
        base_url = base_url_raw.strip().lower() if isinstance(base_url_raw, str) else ""

        native_non_openai_compatible = {
            "anthropic",
            "gemini",
            "google",
            "google_ai_studio",
            "google-ai-studio",
        }
        openai_compatible = provider in {"openai", "openrouter", "deepseek"} or (
            provider not in native_non_openai_compatible and bool(base_url and "/v1" in base_url)
        )
        if openai_compatible:
            return True
        if provider in {"gemini", "google", "google_ai_studio", "google-ai-studio"}:
            return True
        if any(prefix in model for prefix in ("gpt-", "o3", "o4", "openai/")):
            return True
        if "gemini" in model or model.startswith("google/"):
            return True
        if "api.openai.com" in base_url:
            return True
        return "generativelanguage.googleapis.com" in base_url

    def _should_skip_runtime_next_step_prompt(self) -> bool:
        """Avoid synthetic user turns for strict function-turn providers."""
        if not self._uses_strict_tool_turn_order():
            return False
        return self._uses_openai_compatible_tool_api()

    @classmethod
    def _snap_drop_end_to_turn_boundary(
        cls, messages: list, keep_head: int, desired_end: int
    ) -> int:
        """Return an index <= desired_end that sits on a safe boundary.

        A *safe* drop boundary is a position ``k`` where:
          * ``messages[k]`` is ``role="user"`` (start of the next turn), OR
          * ``messages[k]`` is ``role="assistant"`` with no ``tool_calls`` and
            not followed immediately by a ``role="tool"`` message (i.e. the
            tool chain that the previous assistant opened is fully settled
            at the end of index ``k - 1``).

        Snapping ensures we never split an assistant(tool_calls) → tool(result)
        pair, which would leak orphans the provider rejects.  If no boundary
        is found, we fall back to ``keep_head`` (i.e. drop nothing) so the
        safety invariant wins over the token-budget target.
        """
        if desired_end <= keep_head:
            return keep_head
        end = min(desired_end, len(messages))

        def _role(i: int) -> str | None:
            if i < 0 or i >= len(messages):
                return None
            return cls._message_role_value(messages[i])

        def _has_tool_calls(i: int) -> bool:
            if i < 0 or i >= len(messages):
                return False
            tc = getattr(messages[i], "tool_calls", None)
            return bool(tc)

        # Walk backwards from desired_end looking for a clean cut.  The cut
        # index k means "delete messages[keep_head:k]", so messages[k]
        # becomes the new first non-head message.
        for k in range(end, keep_head, -1):
            role = _role(k)
            if role == "user":
                return k
            if role == "assistant" and not _has_tool_calls(k):
                if _role(k + 1) != "tool":
                    return k
        return keep_head

    def _insert_compaction_marker(self, messages: list, dropped: int) -> None:
        """Insert a runtime-only marker telling the model history was trimmed.

        The marker is a ``role="user"`` system-style message that documents
        how many prior messages were removed from runtime memory and points
        the model at the :class:`SearchHistoryTool` as the recovery path.
        It is **not** persisted to the session store — the store still holds
        the full transcript — so adding the marker is idempotent across
        turns.
        """
        if not self._agent or not hasattr(self._agent, "memory"):
            return
        try:
            from spoon_ai.schema import Message, Role  # lazy import
        except Exception:  # pragma: no cover - defensive
            return

        marker_text = (
            "[history-compacted] "
            f"{dropped} older message(s) (including any tool-call/result pairs) "
            "were removed from this turn's in-memory context to fit the token "
            "budget. The persisted session transcript was NOT cleared. The "
            "latest real user request remains authoritative. Earlier assistant "
            "analysis/conclusions in compacted history may be tentative or "
            "stale, so do not treat them as current instructions. If you need "
            "an exact earlier tool result, tool argument, image description, "
            "or user statement, call the `search_history` tool (scope='current'). "
            "Plain earlier assistant replies are omitted there by default; "
            "search them explicitly only when their literal wording matters."
        )
        try:
            marker = Message(role=Role.USER, content=marker_text)
        except Exception:
            return

        insert_at = 1 if messages else 0
        messages.insert(insert_at, marker)

    @staticmethod
    def _is_history_compaction_marker(msg: Any) -> bool:
        """True when *msg* is the runtime-only history-compacted marker."""
        role = AgentLoop._message_role_value(msg)
        if role != "user":
            return False
        content = getattr(msg, "content", None)
        return isinstance(content, str) and content.startswith("[history-compacted]")

    def _latest_real_user_message_index(self, messages: list[Any]) -> int | None:
        """Return the latest user-authored message that must survive compaction intact."""
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if self._message_role_value(message) != "user":
                continue
            if self._is_next_step_user_msg(message):
                continue
            if self._is_history_compaction_marker(message):
                continue
            return index
        return None

    def _compress_runtime_context(
        self,
        *,
        force: bool = False,
        budget_tokens: int | None = None,
    ) -> int:
        """Proactively compress the agent's runtime context.

        Strategy (inspired by Openclaw's context engine):
        1. Drop redundant next_step_prompt user messages (keep only the latest).
        2. Truncate ALL older message content (tool results, assistant, user).
        3. If still over budget, drop entire old message rounds.

        Trigger: estimated tokens > the active runtime compaction budget.
        """
        if not self._agent or not hasattr(self._agent, "memory"):
            return 0

        messages = self._agent.memory.messages
        normalized = self._normalize_runtime_tool_context(messages)
        if len(messages) <= 6:
            return normalized

        estimated = self._estimate_runtime_tokens()
        budget = (
            budget_tokens
            if budget_tokens is not None
            else self._runtime_compaction_trigger_budget()
        )

        if not force and estimated <= budget:
            return normalized

        logger.warning(
            f"Context compression triggered: ~{estimated:,} tokens "
            f"(budget: {budget:,}, window: {self.context_window:,}). "
            f"Messages: {len(messages)}"
        )

        compressed = normalized

        # Phase 1: Remove all but the LAST next_step_prompt user message.
        last_nsp_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if self._is_next_step_user_msg(messages[i]):
                last_nsp_idx = i
                break
        indices_to_remove = []
        for i in range(len(messages)):
            if i != last_nsp_idx and self._is_next_step_user_msg(messages[i]):
                indices_to_remove.append(i)
        for idx in reversed(indices_to_remove):
            del messages[idx]
            compressed += 1
        if indices_to_remove:
            logger.info(f"Phase 1: removed {len(indices_to_remove)} old next_step_prompt messages")

        protected_user_index = self._latest_real_user_message_index(messages)

        # Phase 2: Compact older messages except the first and last 6.
        keep_tail = min(6, len(messages))
        max_content = 300
        for i in range(1, max(1, len(messages) - keep_tail)):
            if protected_user_index is not None and i == protected_user_index:
                continue
            msg = messages[i]
            original_content = getattr(msg, "content", None)
            compressed_content = self._compact_runtime_message_content(msg, max_content)
            if compressed_content != original_content:
                msg.content = compressed_content
                compressed += 1

        # Phase 3: If still over budget, drop oldest rounds (keep first + last 8).
        #
        # *Segment-aware* drop: we only cut along user-turn boundaries so a
        # drop never splits an ``assistant(tool_calls) -> tool(result)``
        # chain. Partial splits would leak through as orphaned tool
        # messages and the LLM provider would reject the whole request.
        # The persisted session store is untouched, so anything dropped
        # here remains reachable via ``search_history``.
        estimated = self._estimate_runtime_tokens()
        dropped_in_phase3 = 0
        if estimated > budget and len(messages) > 12:
            keep_head = 1
            keep_tail_drop = min(8, len(messages) - 1)
            tail_start = len(messages) - keep_tail_drop
            if protected_user_index is not None:
                tail_start = min(tail_start, protected_user_index)
            droppable = max(0, tail_start - keep_head)
            if droppable > 4:
                desired = droppable // 2
                snap_end = self._snap_drop_end_to_turn_boundary(
                    messages, keep_head, keep_head + desired
                )
                keep_start = self._adjust_message_start_to_preserve_tool_context(
                    messages,
                    snap_end,
                    floor=keep_head,
                )
                actual_drop = max(0, keep_start - keep_head)
                if actual_drop > 0:
                    del messages[keep_head : keep_head + actual_drop]
                    compressed += actual_drop
                    dropped_in_phase3 = actual_drop
                    logger.info(
                        f"Phase 3: dropped {actual_drop} oldest messages "
                        f"(segment-snapped from desired {desired})"
                    )

        # Phase 3b: Insert a compaction marker so the model knows prior
        # turns are recoverable via ``search_history``. The marker is
        # runtime-only; it never touches the persisted session.
        if dropped_in_phase3 > 0:
            self._insert_compaction_marker(messages, dropped_in_phase3)

        # Phase 4: Repair tool_use/tool_result pairing broken by message deletion.
        compressed += self._normalize_runtime_tool_context(messages)

        final_est = self._estimate_runtime_tokens()
        logger.info(
            f"Context compression done: {compressed} actions, "
            f"tokens ~{estimated:,} -> ~{final_est:,}, "
            f"messages {len(messages)}"
        )
        return compressed

    def _force_compress_runtime_context(self) -> int:
        """Emergency context compression when normal compression is insufficient.

        Aggressively truncates ALL content and drops messages to get under 40 %
        of context_window.
        """
        if not self._agent or not hasattr(self._agent, "memory"):
            return 0

        messages = self._agent.memory.messages
        compressed = self._normalize_runtime_tool_context(messages)
        if len(messages) <= 4:
            return compressed

        protected_user_index = self._latest_real_user_message_index(messages)

        # Compact older messages aggressively while preserving the latest real user turn.
        for index, msg in enumerate(messages):
            if protected_user_index is not None and index == protected_user_index:
                continue
            original_content = getattr(msg, "content", None)
            compressed_content = self._compact_runtime_message_content(msg, 150)
            if compressed_content != original_content:
                msg.content = compressed_content
                compressed += 1

        # Drop all but first + last 6 messages (segment-aware)
        dropped = 0
        if len(messages) > 8:
            keep_head = 1
            keep_tail = min(6, len(messages) - 1)
            desired_end = len(messages) - keep_tail
            if protected_user_index is not None:
                desired_end = min(desired_end, protected_user_index)
            snap_end = self._snap_drop_end_to_turn_boundary(messages, keep_head, desired_end)
            keep_start = self._adjust_message_start_to_preserve_tool_context(
                messages,
                snap_end,
                floor=keep_head,
            )
            actual_drop = max(0, keep_start - keep_head)
            if actual_drop > 0:
                del messages[keep_head : keep_head + actual_drop]
                compressed += actual_drop
                dropped = actual_drop

        if dropped > 0:
            self._insert_compaction_marker(messages, dropped)

        # Repair tool pairing broken by message deletion
        compressed += self._normalize_runtime_tool_context(messages)

        logger.warning(f"Force-compressed {compressed} messages/results for recovery")
        return compressed

    def _install_anti_loop_tracker(self, base_prompt: str) -> None:
        """Wrap think() for compaction/logging without prompt-based routing."""
        agent = self._agent
        if agent is None:
            return
        self._install_tool_call_protocol_guards()
        self._install_tool_loop_stop_guard()
        self._install_session_progress_persistence_guard()

        agent_loop = self
        original_think = getattr(agent, "_spoon_bot_base_think", None)
        if original_think is None:
            original_think = agent.think
            setattr(agent, "_spoon_bot_base_think", original_think)
        else:
            agent.think = original_think

        async def _tracked_think(*args: Any, **kwargs: Any) -> bool:
            agent_loop._compress_runtime_context()

            desired_next_step_prompt = agent.next_step_prompt or base_prompt
            request_execution_hints = get_request_execution_hints()
            runtime_tool_events = agent_loop._collect_runtime_tool_result_events_from_memory()

            def _latest_real_user_text() -> str:
                runtime_messages = AgentLoop._get_runtime_memory_messages(agent_loop)
                latest_user_index = agent_loop._latest_real_user_message_index(runtime_messages)
                if latest_user_index is None or latest_user_index >= len(runtime_messages):
                    return ""
                return AgentLoop._stringify_stream_payload(
                    AgentLoop._stream_message_attr(
                        runtime_messages[latest_user_index],
                        "content",
                        "",
                    )
                )

            if (
                request_execution_hints
                and should_run_skill_contract_check(runtime_tool_events)
                and AgentLoop._tool_events_have_repeated_read_guardrail(runtime_tool_events[-3:])
                and not latest_tool_event_has_user_summary_marker(runtime_tool_events)
            ):
                latest_user_text = _latest_real_user_text()
                desired_next_step_prompt = AgentLoop._build_repeated_read_recovery_prompt(
                    latest_user_text or base_prompt,
                )
            suppress_runtime_prompt = AgentLoop._should_skip_runtime_next_step_prompt(agent_loop)
            try:
                previous_provider_tail_prompt = vars(agent).get(
                    "_spoon_bot_provider_tail_prompt",
                    _MISSING,
                )
            except Exception:
                previous_provider_tail_prompt = getattr(
                    agent,
                    "_spoon_bot_provider_tail_prompt",
                    _MISSING,
                )
            if suppress_runtime_prompt:
                logger.info(
                    "Suppressing synthetic next_step_prompt user turn for strict tool-call provider"
                )
                agent._spoon_bot_provider_tail_prompt = agent_loop.DEFAULT_NEXT_STEP_PROMPT
                agent.next_step_prompt = None
            else:
                if previous_provider_tail_prompt is not _MISSING:
                    try:
                        delattr(agent, "_spoon_bot_provider_tail_prompt")
                    except Exception:
                        pass
                agent.next_step_prompt = desired_next_step_prompt

            think_timeout = AgentLoop._positive_runtime_budget(
                getattr(agent_loop, "provider_ask_timeout", None),
                _DEFAULT_PROVIDER_ASK_TIMEOUT,
            )
            if getattr(agent_loop, "provider_ask_timeout", None) == 0:
                think_timeout = 0.0

            try:
                if think_timeout > 0:
                    try:
                        result = await asyncio.wait_for(
                            original_think(*args, **kwargs),
                            timeout=think_timeout,
                        )
                    except asyncio.TimeoutError as exc:
                        logger.warning(
                            "Provider think step exceeded {:.1f}s; "
                            "raising retryable provider timeout.",
                            think_timeout,
                        )
                        raise LLMTimeoutError(
                            str(getattr(agent_loop, "provider", "provider")),
                            think_timeout,
                        ) from exc
                else:
                    result = await original_think(*args, **kwargs)
            finally:
                if previous_provider_tail_prompt is _MISSING:
                    try:
                        delattr(agent, "_spoon_bot_provider_tail_prompt")
                    except Exception:
                        pass
                else:
                    agent._spoon_bot_provider_tail_prompt = previous_provider_tail_prompt
                agent.next_step_prompt = desired_next_step_prompt

            _log_agent_reasoning()
            _log_tool_calls()

            return result

        def _log_agent_reasoning():
            """Extract and log the agent's reasoning text from its last response."""
            from spoon_bot.utils.privacy import mask_secrets

            summary = getattr(agent, "last_reasoning_summary", None)
            if isinstance(summary, str) and summary.strip():
                captured_summary = agent_loop._capture_reasoning_text(mask_secrets(summary.strip()))
                if captured_summary:
                    logger.info(f"💭 Agent reasoning: {captured_summary}")
                return
            if not hasattr(agent, "memory") or not agent.memory.messages:
                return
            for msg in reversed(agent.memory.messages[-3:]):
                role = getattr(msg, "role", "")
                if role != "assistant":
                    continue
                content = msg.content if isinstance(msg.content, str) else ""
                if not content or not content.strip():
                    break
                safe_text = mask_secrets(content.strip())
                captured = agent_loop._capture_reasoning_text(safe_text)
                if captured:
                    logger.info(f"💭 Agent reasoning: {captured}")
                break

        def _log_tool_calls():
            """Log each tool call with arguments so TUI can display them."""
            from spoon_bot.utils.privacy import mask_secrets

            if not hasattr(agent, "tool_calls") or not agent.tool_calls:
                return
            for tc in agent.tool_calls:
                fn = getattr(tc, "function", tc)
                name = getattr(fn, "name", "")
                raw_args = getattr(fn, "arguments", "")
                safe_args = mask_secrets(raw_args) if raw_args else ""
                logger.info(f"Tool call: {name}({safe_args})")

        agent.think = _tracked_think

    def _install_tool_call_protocol_guards(self) -> None:
        """Install transport-level guards for provider tool-call invariants."""
        agent = getattr(self, "_agent", None)
        llm = getattr(agent, "llm", None)
        ask_tool = getattr(llm, "ask_tool", None)
        if llm is None or not callable(ask_tool):
            return

        original_ask_tool = getattr(llm, "_spoon_bot_base_ask_tool", None)
        if original_ask_tool is None:
            original_ask_tool = ask_tool
            try:
                setattr(llm, "_spoon_bot_base_ask_tool", original_ask_tool)
            except Exception:
                return

        agent_loop = self

        def _latest_request_message_for_provider() -> str:
            try:
                runtime_messages = AgentLoop._get_runtime_memory_messages(agent_loop)
                latest_user_index = agent_loop._latest_real_user_message_index(runtime_messages)
                if latest_user_index is not None and latest_user_index < len(runtime_messages):
                    content = AgentLoop._stream_message_attr(
                        runtime_messages[latest_user_index],
                        "content",
                        "",
                    )
                    text = AgentLoop._stringify_stream_payload(content).strip()
                    if text:
                        return text
            except Exception:
                pass

            session = getattr(agent_loop, "_session", None)
            raw_messages = getattr(session, "messages", None)
            if not isinstance(raw_messages, list):
                try:
                    getter = getattr(session, "get_messages", None)
                    raw_messages = getter() if callable(getter) else []
                except Exception:
                    raw_messages = []
            for message in reversed(raw_messages or []):
                try:
                    role = str(message.get("role", "")).strip().lower()
                    content = message.get("content", "")
                except AttributeError:
                    role = AgentLoop._message_role_value(message)
                    content = getattr(message, "content", "")
                if role != "user":
                    continue
                text = AgentLoop._stringify_stream_payload(content).strip()
                if text:
                    return text
            return ""

        def _active_provider_system_prompt(current_system_prompt: Any) -> str | None:
            diagnostic: dict[str, Any] = {
                "current_system_has_context": (
                    isinstance(current_system_prompt, str)
                    and "## Active Request Context" in current_system_prompt
                ),
                "used_cached_prompt": False,
                "latest_request_chars": 0,
                "built_prompt": False,
                "error": "",
            }
            active_system_prompt = getattr(
                agent_loop,
                "_active_request_augmented_system_prompt",
                None,
            )
            if isinstance(active_system_prompt, str) and active_system_prompt:
                diagnostic["used_cached_prompt"] = True
                diagnostic["built_prompt"] = True
                agent_loop._last_provider_context_injection = diagnostic
                return active_system_prompt

            base_prompt = (
                current_system_prompt
                if isinstance(current_system_prompt, str) and current_system_prompt.strip()
                else getattr(agent, "system_prompt", None)
            )
            if not isinstance(base_prompt, str) or not base_prompt.strip():
                diagnostic["error"] = "missing_base_system_prompt"
                agent_loop._last_provider_context_injection = diagnostic
                return None

            latest_request = _latest_request_message_for_provider()
            diagnostic["latest_request_chars"] = len(latest_request)
            if not latest_request:
                diagnostic["error"] = "missing_latest_request"
                agent_loop._last_provider_context_injection = diagnostic
                return None
            try:
                request_context = agent_loop._build_request_context_prompt(latest_request)
            except Exception as exc:
                diagnostic["error"] = f"build_request_context_failed:{type(exc).__name__}"
                agent_loop._last_provider_context_injection = diagnostic
                return None
            augmented_prompt = f"{base_prompt}\n\n## Active Request Context\n{request_context}"
            agent_loop._active_request_base_system_prompt = base_prompt
            agent_loop._active_request_augmented_system_prompt = augmented_prompt
            diagnostic["built_prompt"] = True
            agent_loop._last_provider_context_injection = diagnostic
            return augmented_prompt

        async def _guarded_ask_tool(*args: Any, **kwargs: Any):
            should_disable_parallel = agent_loop._should_disable_parallel_tool_calls()
            should_textualize_tool_history = (
                agent_loop._should_textualize_tool_history_for_provider()
            )
            call_args = list(args)
            call_kwargs = dict(kwargs)
            current_system_prompt = call_kwargs.get("system_msg")
            active_system_prompt = _active_provider_system_prompt(current_system_prompt)
            if isinstance(active_system_prompt, str) and active_system_prompt:
                has_request_context = (
                    isinstance(current_system_prompt, str)
                    and "## Active Request Context" in current_system_prompt
                )
                if not has_request_context:
                    call_kwargs["system_msg"] = active_system_prompt
                    try:
                        diagnostic = dict(
                            getattr(agent_loop, "_last_provider_context_injection", {}) or {}
                        )
                        diagnostic["injected"] = True
                        agent_loop._last_provider_context_injection = diagnostic
                    except Exception:
                        pass
            if should_textualize_tool_history:
                if "messages" in call_kwargs:
                    call_kwargs["messages"] = AgentLoop._textualize_tool_history(
                        call_kwargs["messages"]
                    )
                elif call_args:
                    call_args[0] = AgentLoop._textualize_tool_history(call_args[0])

            if agent_loop._provider_requires_user_tail():
                continuation_prompt = getattr(agent, "next_step_prompt", None)
                if not (isinstance(continuation_prompt, str) and continuation_prompt.strip()):
                    try:
                        provider_tail_prompt = vars(agent).get("_spoon_bot_provider_tail_prompt")
                    except Exception:
                        provider_tail_prompt = getattr(
                            agent,
                            "_spoon_bot_provider_tail_prompt",
                            None,
                        )
                    if isinstance(provider_tail_prompt, str) and provider_tail_prompt.strip():
                        continuation_prompt = provider_tail_prompt
                if "messages" in call_kwargs:
                    call_kwargs["messages"] = AgentLoop._ensure_provider_messages_end_with_user(
                        call_kwargs["messages"],
                        continuation_prompt,
                    )
                elif call_args:
                    call_args[0] = AgentLoop._ensure_provider_messages_end_with_user(
                        call_args[0],
                        continuation_prompt,
                    )

            injected_parallel_flag = False
            if should_disable_parallel and "parallel_tool_calls" not in call_kwargs:
                call_kwargs["parallel_tool_calls"] = False
                injected_parallel_flag = True

            if "max_tokens" not in call_kwargs and "max_completion_tokens" not in call_kwargs:
                call_kwargs["max_tokens"] = AgentLoop._tool_call_output_token_budget()

            async def _ask_with_parallel_fallback(request_kwargs: dict[str, Any]):
                retry_config = getattr(agent_loop, "_retry_config", None)
                if not isinstance(retry_config, RetryConfig):
                    retry_config = DEFAULT_RETRY_CONFIG
                ask_timeout = AgentLoop._positive_runtime_budget(
                    getattr(agent_loop, "provider_ask_timeout", None),
                    _DEFAULT_PROVIDER_ASK_TIMEOUT,
                )
                if getattr(agent_loop, "provider_ask_timeout", None) == 0:
                    ask_timeout = 0.0

                try:
                    snapshot_index = int(getattr(agent_loop, "_llm_context_snapshot_index", 0)) + 1
                    agent_loop._llm_context_snapshot_index = snapshot_index
                    snapshot_path = write_llm_context_snapshot(
                        agent_loop,
                        label=f"ask_tool#{snapshot_index}",
                        args=call_args,
                        kwargs=request_kwargs,
                    )
                    if snapshot_path is not None:
                        logger.debug(f"Wrote LLM context snapshot: {snapshot_path}")
                except Exception as exc:
                    logger.debug(f"Failed to write LLM context snapshot: {exc}")

                async def _call_provider(current_kwargs: dict[str, Any]):
                    async def _invoke_original() -> Any:
                        if inspect.iscoroutinefunction(original_ask_tool):
                            return await original_ask_tool(*call_args, **current_kwargs)

                        result = await asyncio.to_thread(
                            original_ask_tool,
                            *call_args,
                            **current_kwargs,
                        )
                        if inspect.isawaitable(result):
                            return await result
                        return result

                    if ask_timeout <= 0:
                        return await _invoke_original()
                    try:
                        return await asyncio.wait_for(
                            _invoke_original(),
                            timeout=ask_timeout,
                        )
                    except asyncio.TimeoutError as exc:
                        logger.warning(
                            "Provider ask_tool call exceeded {:.1f}s; "
                            "retrying if provider retry budget remains.",
                            ask_timeout,
                        )
                        raise LLMTimeoutError(
                            str(getattr(agent_loop, "provider", "provider")), ask_timeout
                        ) from exc

                async def _attempt():
                    try:
                        return await _call_provider(request_kwargs)
                    except Exception as exc:
                        if injected_parallel_flag and "parallel_tool_calls" in str(exc):
                            retry_kwargs = dict(request_kwargs)
                            retry_kwargs.pop("parallel_tool_calls", None)
                            return await _call_provider(retry_kwargs)
                        raise

                def _on_retry(attempt: int, exc: Exception, delay: float) -> None:
                    logger.warning(
                        "Provider ask_tool transient error (attempt {}/{}), "
                        "retrying in {:.1f}s: {}: {}",
                        attempt + 1,
                        retry_config.max_retries + 1,
                        delay,
                        type(exc).__name__,
                        exc,
                    )

                return await with_provider_retry(
                    _attempt,
                    config=retry_config,
                    on_retry=_on_retry,
                )

            try:
                response = await _ask_with_parallel_fallback(call_kwargs)
            except Exception:
                raise

            retry_reason = AgentLoop._tool_response_needs_retry(response)
            if retry_reason:
                retry_kwargs = dict(call_kwargs)
                current_budget = retry_kwargs.get("max_tokens") or retry_kwargs.get(
                    "max_completion_tokens"
                )
                try:
                    current_budget_int = int(current_budget) if current_budget is not None else None
                except (TypeError, ValueError):
                    current_budget_int = None
                retry_kwargs["max_tokens"] = AgentLoop._tool_call_retry_token_budget(
                    current_budget_int
                )
                retry_kwargs.pop("max_completion_tokens", None)
                logger.warning(
                    "Retrying provider tool turn because tool-call arguments were incomplete: "
                    f"{retry_reason}"
                )
                retry_response = await _ask_with_parallel_fallback(retry_kwargs)
                retry_blocker = AgentLoop._tool_response_needs_retry(retry_response)
                if retry_blocker:
                    response = AgentLoop._block_incomplete_tool_calls(retry_response, retry_blocker)
                else:
                    response = retry_response

            if should_disable_parallel:
                AgentLoop._coerce_response_to_single_tool_call(response)
            return response

        try:
            setattr(llm, "ask_tool", _guarded_ask_tool)
        except Exception:
            return

    def _install_tool_loop_stop_guard(self) -> None:
        """Stop the core action loop when a tool emits a hard guardrail result."""
        agent = getattr(self, "_agent", None)
        execute_tool = getattr(agent, "execute_tool", None)
        if agent is None or not callable(execute_tool):
            return

        original_execute_tool = getattr(agent, "_spoon_bot_base_execute_tool", None)
        if original_execute_tool is None:
            original_execute_tool = execute_tool
            try:
                setattr(agent, "_spoon_bot_base_execute_tool", original_execute_tool)
            except Exception:
                return
        else:
            try:
                setattr(agent, "execute_tool", original_execute_tool)
            except Exception:
                return

        async def _guarded_execute_tool(*args: Any, **kwargs: Any) -> Any:
            result = await original_execute_tool(*args, **kwargs)
            if "stop_tool_loop" in AgentLoop._stringify_stream_payload(result).casefold():
                shutdown_event = getattr(agent, "_shutdown_event", None)
                if hasattr(shutdown_event, "set"):
                    try:
                        shutdown_event.set()
                    except Exception:
                        pass
                try:
                    agent.state = AgentState.FINISHED
                except Exception:
                    pass
                logger.warning(
                    "Tool loop guardrail emitted; stopping the current agent action loop."
                )
            return result

        try:
            setattr(agent, "execute_tool", _guarded_execute_tool)
        except Exception:
            return

    def _install_session_progress_persistence_guard(self) -> None:
        """Persist in-flight tool progress after runtime memory receives it."""
        agent = getattr(self, "_agent", None)
        add_message = getattr(agent, "add_message", None)
        if agent is None or not callable(add_message):
            return

        original_add_message = getattr(agent, "_spoon_bot_base_add_message", None)
        if original_add_message is None:
            original_add_message = add_message
            try:
                setattr(agent, "_spoon_bot_base_add_message", original_add_message)
            except Exception:
                return
        else:
            try:
                setattr(agent, "add_message", original_add_message)
            except Exception:
                return

        agent_loop = self

        async def _guarded_add_message(role: Any, content: Any, *args: Any, **kwargs: Any) -> Any:
            result = await original_add_message(role, content, *args, **kwargs)
            normalized_role = str(role or "").strip().lower()
            if normalized_role in {"assistant", "tool"}:
                start_index = getattr(agent_loop, "_active_turn_memory_start_index", None)
                if isinstance(start_index, int) and start_index >= 0:
                    agent_loop._persist_turn_tool_trace_checkpoint(start_index)
            return result

        try:
            setattr(agent, "add_message", _guarded_add_message)
        except Exception:
            return

    def _uses_openai_compatible_tool_api(self) -> bool:
        """Return true when the active provider accepts OpenAI-style tool kwargs."""
        provider_raw = getattr(self, "provider", None)
        base_url_raw = getattr(self, "base_url", None)
        provider = provider_raw.strip().lower() if isinstance(provider_raw, str) else ""
        base_url = base_url_raw.strip().lower() if isinstance(base_url_raw, str) else ""

        if provider in {"openai", "openrouter", "deepseek"}:
            return True
        if provider in {"anthropic", "gemini", "google", "google_ai_studio", "google-ai-studio"}:
            return False
        return bool(base_url and "/v1" in base_url)

    def _should_disable_parallel_tool_calls(self) -> bool:
        """Prefer one tool result turn at a time for strict OpenAI-compatible APIs."""
        import os as _os

        if getattr(self, "_force_serial_tool_calls", False):
            return True

        raw = _os.getenv("SPOON_BOT_PARALLEL_TOOL_CALLS")
        if raw is not None:
            return raw.strip().lower() in {"0", "false", "no", "off"}
        return self._uses_strict_tool_turn_order() and self._uses_openai_compatible_tool_api()

    def _should_textualize_tool_history_for_provider(self) -> bool:
        """Return true when operators explicitly request text-only tool history.

        Completed OpenAI-style tool-call transcripts are valid provider input as
        long as assistant(tool_calls) and matching tool messages stay adjacent.
        Textualizing them by default makes internal tool evidence look like a
        normal assistant reply and can confuse long autonomous workflows.
        """
        import os as _os

        raw = _os.getenv("SPOON_BOT_TEXTUALIZE_TOOL_HISTORY")
        if raw is not None:
            return raw.strip().lower() in {"1", "true", "yes", "on", "always"}

        return False

    def _provider_requires_user_tail(self) -> bool:
        """Return true for providers that reject assistant-prefill style tails."""
        import os as _os

        raw = _os.getenv("SPOON_BOT_PROVIDER_USER_TAIL")
        if raw is not None:
            return raw.strip().lower() in {"1", "true", "yes", "on", "always"}

        return self._provider_requires_user_tail_legacy_default()

    def _provider_requires_user_tail_legacy_default(self) -> bool:
        """Default to native provider message order unless env opts in.

        Adding a synthetic ``Continue.`` user tail after each tool result hides
        the real user request behind runtime plumbing. Modern tool-call APIs can
        continue from a tool-result tail directly, so the compatibility shim is
        now opt-in through ``SPOON_BOT_PROVIDER_USER_TAIL``.
        """
        return False

    @classmethod
    def _ensure_provider_messages_end_with_user(
        cls,
        messages: Any,
        continuation_prompt: str | None = None,
    ) -> Any:
        """Return provider-bound messages ending in a user turn when required."""
        if not isinstance(messages, list) or not messages:
            return messages

        for message in reversed(messages):
            role = str(cls._message_role_value(message) or "").strip().lower()
            if role == "system":
                continue
            if role == "user":
                return messages
            break
        else:
            return messages

        prompt = (
            continuation_prompt.strip()
            if isinstance(continuation_prompt, str) and continuation_prompt.strip()
            else cls.DEFAULT_NEXT_STEP_PROMPT
        )
        return [*messages, Message(role="user", content=prompt)]

    @classmethod
    def _textualize_tool_history(cls, messages: Any) -> Any:
        """Return provider-safe messages with completed tool turns summarized as text."""
        if not isinstance(messages, list) or not messages:
            return messages

        converted: list[Message] = []
        index = 0
        changed = False
        while index < len(messages):
            msg = messages[index]
            role = cls._message_role_value(msg)
            if role == "assistant" and cls._message_tool_calls_value(msg):
                tool_calls = cls._message_tool_calls_value(msg)
                expected_ids = {
                    cls._tool_call_id_value(tool_call)
                    for tool_call in tool_calls
                    if cls._tool_call_id_value(tool_call)
                }
                results: list[Any] = []
                cursor = index + 1
                while cursor < len(messages):
                    next_msg = messages[cursor]
                    if cls._message_role_value(next_msg) != "tool":
                        break
                    tool_call_id = cls._message_tool_call_id_value(next_msg)
                    if expected_ids and tool_call_id not in expected_ids:
                        break
                    results.append(next_msg)
                    cursor += 1

                converted.append(cls._build_tool_history_summary_message(msg, tool_calls, results))
                changed = True
                index = cursor
                continue

            if role == "tool":
                converted.append(cls._build_standalone_tool_summary_message(msg))
                changed = True
                index += 1
                continue

            converted.append(cls._copy_message_without_provider_tool_fields(msg))
            index += 1

        return converted if changed else messages

    @classmethod
    def _copy_message_without_provider_tool_fields(cls, msg: Any) -> Message:
        role = cls._message_role_value(msg) or "user"
        if role not in {"system", "user", "assistant"}:
            role = "assistant"
        content = cls._stream_message_attr(msg, "content", None)
        if content is None:
            content = cls._stream_message_attr(msg, "text_content", "") or ""
        return Message(role=role, content=content)

    @classmethod
    def _build_tool_history_summary_message(
        cls,
        assistant_msg: Any,
        tool_calls: list[Any],
        tool_results: list[Any],
    ) -> Message:
        from spoon_bot.utils.privacy import mask_secrets

        intro = cls._stream_message_attr(assistant_msg, "text_content", None)
        if not isinstance(intro, str) or not intro:
            intro = cls._stream_message_attr(assistant_msg, "content", "") or ""
        parts: list[str] = []
        if isinstance(intro, str) and intro.strip():
            parts.append(mask_secrets(intro.strip()))
        parts.append("[Tool execution summary]")

        result_by_id = {
            cls._message_tool_call_id_value(result): result
            for result in tool_results
            if cls._message_tool_call_id_value(result)
        }
        for tool_call in tool_calls:
            tool_name, arguments = cls._tool_call_name_and_arguments(tool_call)
            tool_call_id = cls._tool_call_id_value(tool_call)
            result_msg = result_by_id.get(tool_call_id)
            result_text = ""
            if result_msg is not None:
                result_text = cls._stream_message_attr(result_msg, "text_content", None)
                if not isinstance(result_text, str) or not result_text:
                    result_text = cls._stream_message_attr(result_msg, "content", "") or ""
            if len(result_text) > 4000:
                result_text = result_text[:4000] + "\n[truncated]"
            parts.append(
                "- "
                + mask_secrets(str(tool_name or "tool"))
                + "("
                + mask_secrets(str(arguments or ""))
                + "): "
                + mask_secrets(str(result_text or "No tool result was captured."))
            )

        return Message(role="assistant", content="\n".join(parts))

    @classmethod
    def _build_standalone_tool_summary_message(cls, msg: Any) -> Message:
        from spoon_bot.utils.privacy import mask_secrets

        name = cls._stream_message_attr(msg, "name", "") or "tool"
        content = cls._stream_message_attr(msg, "text_content", None)
        if not isinstance(content, str) or not content:
            content = cls._stream_message_attr(msg, "content", "") or ""
        if len(content) > 4000:
            content = content[:4000] + "\n[truncated]"
        return Message(
            role="assistant",
            content=f"[Tool execution summary]\n- {mask_secrets(str(name))}: {mask_secrets(str(content))}",
        )

    @staticmethod
    def _coerce_response_to_single_tool_call(response: Any) -> int:
        """Keep one tool call when the transport requires serial tool-result turns."""
        if isinstance(response, dict):
            tool_calls = response.get("tool_calls")
        else:
            tool_calls = getattr(response, "tool_calls", None)
        if isinstance(tool_calls, tuple):
            tool_call_list = list(tool_calls)
        elif isinstance(tool_calls, list):
            tool_call_list = tool_calls
        else:
            return 0
        if len(tool_call_list) <= 1:
            return 0

        dropped = len(tool_call_list) - 1
        try:
            if isinstance(response, dict):
                response["tool_calls"] = tool_call_list[:1]
            else:
                response.tool_calls = tool_call_list[:1]
        except Exception:
            return 0

        metadata = (
            response.get("metadata")
            if isinstance(response, dict)
            else getattr(response, "metadata", None)
        )
        if isinstance(metadata, dict):
            metadata["serial_tool_calls_enforced"] = True
            metadata["dropped_parallel_tool_calls"] = dropped
        logger.info(
            f"Serial tool-call guard kept 1 tool call and deferred {dropped} parallel tool call(s)"
        )
        return dropped

    @staticmethod
    def _tool_call_output_token_budget() -> int:
        """Return the default completion budget for tool-producing turns."""
        raw = os.getenv("SPOON_BOT_TOOL_CALL_MAX_TOKENS")
        if raw is None:
            return 12_000
        try:
            return max(1_024, min(200_000, int(raw.strip())))
        except (TypeError, ValueError):
            return 12_000

    @staticmethod
    def _tool_call_retry_token_budget(current: int | None = None) -> int:
        """Return the retry completion budget for truncated tool-producing turns."""
        raw = os.getenv("SPOON_BOT_TOOL_CALL_RETRY_MAX_TOKENS")
        if raw is not None:
            try:
                return max(1_024, min(200_000, int(raw.strip())))
            except (TypeError, ValueError):
                pass
        base = current or AgentLoop._tool_call_output_token_budget()
        return max(base * 2, 32_768)

    @staticmethod
    def _tool_call_arguments_json_error(arguments: Any) -> str | None:
        """Return an error string when tool-call arguments are not complete JSON."""
        if arguments is None:
            return None
        if isinstance(arguments, dict):
            return None
        if not isinstance(arguments, str):
            return f"unsupported argument type {type(arguments).__name__}"
        text = arguments.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return f"{exc.msg} at char {exc.pos}"
        if not isinstance(parsed, dict):
            return "tool arguments must decode to a JSON object"
        return None

    @staticmethod
    def _tool_call_name_and_raw_arguments(tool_call: Any) -> tuple[str, Any]:
        """Return a tool-call name and argument payload from object or dict shapes."""
        fn = getattr(tool_call, "function", None) or (
            tool_call.get("function") if isinstance(tool_call, dict) else None
        )
        if fn is not None:
            name = getattr(fn, "name", None) or (fn.get("name") if isinstance(fn, dict) else None)
            arguments = getattr(fn, "arguments", None) or (
                fn.get("arguments") if isinstance(fn, dict) else None
            )
            return str(name or ""), arguments
        name = getattr(tool_call, "name", None) or (
            tool_call.get("name") if isinstance(tool_call, dict) else None
        )
        arguments = getattr(tool_call, "arguments", None) or (
            tool_call.get("arguments") if isinstance(tool_call, dict) else None
        )
        return str(name or ""), arguments

    @staticmethod
    def _tool_response_needs_retry(response: Any) -> str | None:
        """Return why a tool response should be retried before executing tools."""
        tool_calls = getattr(response, "tool_calls", None)
        if not isinstance(tool_calls, list) or not tool_calls:
            return None

        finish_reason = (
            str(
                getattr(response, "finish_reason", None)
                or getattr(response, "native_finish_reason", None)
                or ""
            )
            .strip()
            .lower()
        )
        if finish_reason in {"length", "max_tokens", "max_output_tokens"}:
            return f"finish_reason={finish_reason}"

        for tool_call in tool_calls:
            tool_name, arguments = AgentLoop._tool_call_name_and_raw_arguments(tool_call)
            error = AgentLoop._tool_call_arguments_json_error(arguments)
            if error:
                return f"{tool_name or 'tool'} arguments are incomplete JSON: {error}"
        return None

    @staticmethod
    def _block_incomplete_tool_calls(response: Any, reason: str) -> Any:
        """Prevent execution of tool calls that the provider returned incomplete."""
        message = (
            "Tool call generation was truncated before valid JSON arguments were complete. "
            f"Reason: {reason}. Retry the request with smaller tool payloads or shorter "
            "file writes instead of executing partial arguments."
        )
        try:
            response.tool_calls = []
        except Exception:
            pass
        try:
            response.content = message
        except Exception:
            pass
        metadata = getattr(response, "metadata", None)
        if isinstance(metadata, dict):
            metadata["incomplete_tool_calls_blocked"] = True
            metadata["incomplete_tool_call_reason"] = reason
        return response

    def _restore_agent_think(self) -> None:
        """Restore the agent's base think() implementation after a request."""
        agent = self._agent
        if agent is None:
            return
        original_think = getattr(agent, "_spoon_bot_base_think", None)
        if original_think is not None:
            agent.think = original_think

    def _reset_reasoning_capture(self) -> None:
        """Reset request-scoped reasoning captured from tracked think logs."""
        self._latest_reasoning_excerpt = None
        self._pending_reasoning_chunks = []

    def _reset_runtime_notices(self) -> None:
        """Reset request-scoped runtime notices surfaced to clients."""
        self._pending_runtime_notices = []

    def _queue_runtime_notice(self, **notice: Any) -> None:
        """Queue a request-scoped runtime notice for later streaming."""
        pending = getattr(self, "_pending_runtime_notices", None)
        if not isinstance(pending, list):
            pending = []
            self._pending_runtime_notices = pending
        payload = {key: value for key, value in notice.items() if value is not None}
        if payload:
            pending.append(payload)

    def _drain_runtime_notices(self) -> list[dict[str, Any]]:
        """Return queued runtime notices and clear the request buffer."""
        pending = getattr(self, "_pending_runtime_notices", None)
        if not isinstance(pending, list):
            self._pending_runtime_notices = []
            return []
        notices = [item for item in pending if isinstance(item, dict)]
        self._pending_runtime_notices = []
        return notices

    @staticmethod
    def _runtime_notice_to_stream_event(
        notice: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Convert an internal runtime notice into a stream event."""
        if not isinstance(notice, dict):
            return None

        kind = str(notice.get("kind") or "").strip().lower()
        if kind != "runtime_compaction":
            return None

        stage = str(notice.get("stage") or "").strip().lower()
        if stage == "overflow_retry":
            delta = (
                "Context window exceeded. Earlier history was compacted and the "
                "agent resumed the latest request."
            )
        else:
            delta = (
                "Context window near limit. Earlier history was compacted before "
                "continuing the latest request."
            )

        metadata = dict(notice)
        metadata.setdefault("visible", True)
        return {
            "type": "notice",
            "delta": delta,
            "metadata": metadata,
        }

    def _capture_reasoning_text(self, text: str | None) -> str | None:
        """Store a reasoning excerpt so gateway transports can reuse it."""
        normalized = str(text or "").strip()
        if not normalized:
            return None
        if normalized == self._latest_reasoning_excerpt:
            return normalized
        self._latest_reasoning_excerpt = normalized
        self._pending_reasoning_chunks.append(normalized)
        return normalized

    def _drain_reasoning_chunks(self) -> list[str]:
        """Return pending reasoning excerpts and clear the queue."""
        pending = [
            text
            for text in self._pending_reasoning_chunks
            if isinstance(text, str) and text.strip()
        ]
        self._pending_reasoning_chunks = []
        return pending

    @staticmethod
    def _normalize_comparable_text(text: str | None) -> str:
        """Collapse whitespace so duplicate text can be compared reliably."""
        return " ".join(str(text or "").split())

    @staticmethod
    def _resolve_stream_fallback_delta(
        streamed_text: str | None,
        final_text: str | None,
    ) -> tuple[str, str]:
        """Return merged final content plus the missing delta to emit."""
        streamed = str(streamed_text or "")
        final = str(final_text or "")
        if not streamed or not final:
            return final, final

        prefix_candidates = [streamed]
        trimmed_streamed = streamed.rstrip()
        if trimmed_streamed and trimmed_streamed != streamed:
            prefix_candidates.append(trimmed_streamed)

        for prefix in prefix_candidates:
            if prefix and final.startswith(prefix):
                return final, final[len(prefix) :]

        return streamed + final, final

    @staticmethod
    def _normalized_text_with_source_end_offsets(text: str) -> tuple[str, list[int]]:
        """Normalize whitespace while retaining source end offsets for each char."""
        normalized_chars: list[str] = []
        source_end_offsets: list[int] = []
        in_whitespace = False
        for index, char in enumerate(text):
            if char.isspace():
                if normalized_chars and not in_whitespace:
                    normalized_chars.append(" ")
                    source_end_offsets.append(index + 1)
                    in_whitespace = True
                continue
            normalized_chars.append(char)
            source_end_offsets.append(index + 1)
            in_whitespace = False

        while normalized_chars and normalized_chars[-1] == " ":
            normalized_chars.pop()
            source_end_offsets.pop()
        return "".join(normalized_chars), source_end_offsets

    @staticmethod
    def _whitespace_free_text_with_source_end_offsets(text: str) -> tuple[str, list[int]]:
        """Remove whitespace while retaining source end offsets for each char."""
        normalized_chars: list[str] = []
        source_end_offsets: list[int] = []
        for index, char in enumerate(text):
            if char.isspace():
                continue
            normalized_chars.append(char)
            source_end_offsets.append(index + 1)
        return "".join(normalized_chars), source_end_offsets

    @staticmethod
    def _remove_all_whitespace(text: str | None) -> str:
        return "".join(str(text or "").split())

    @staticmethod
    def _looks_like_incomplete_repeated_stream_prefix(
        incoming_text: str | None,
        already_emitted_text: str | None,
    ) -> bool:
        incoming_normalized = AgentLoop._remove_all_whitespace(incoming_text)
        emitted_normalized = AgentLoop._remove_all_whitespace(already_emitted_text)
        return (
            len(incoming_normalized) >= 12
            and len(emitted_normalized) >= 24
            and len(incoming_normalized) < len(emitted_normalized)
            and emitted_normalized.startswith(incoming_normalized)
        )

    @staticmethod
    def _trim_repeated_stream_prefix(
        incoming_text: str | None,
        already_emitted_text: str | None,
    ) -> str:
        """Remove a repeated prefix that has already been streamed to the user."""
        incoming = str(incoming_text or "")
        if not incoming:
            return incoming

        emitted = str(already_emitted_text or "")
        for prefix in (emitted, emitted.rstrip()):
            if prefix and incoming.startswith(prefix):
                return incoming[len(prefix) :].lstrip()

        emitted_normalized = AgentLoop._normalize_comparable_text(already_emitted_text)
        if len(emitted_normalized) < 24:
            return incoming

        incoming_normalized, incoming_source_ends = (
            AgentLoop._normalized_text_with_source_end_offsets(incoming)
        )
        if not incoming_normalized:
            return incoming

        matched_len = 0
        if incoming_normalized.startswith(emitted_normalized):
            matched_len = len(emitted_normalized)
        else:
            min_match_len = 24
            max_start = max(0, len(emitted_normalized) - 2000)
            for start in range(max_start, len(emitted_normalized) - min_match_len + 1):
                suffix = emitted_normalized[start:]
                if incoming_normalized.startswith(suffix):
                    matched_len = len(suffix)
                    break

        if matched_len < 24 or matched_len > len(incoming_source_ends):
            emitted_compact = AgentLoop._remove_all_whitespace(already_emitted_text)
            incoming_compact, incoming_compact_source_ends = (
                AgentLoop._whitespace_free_text_with_source_end_offsets(incoming)
            )
            matched_compact_len = 0
            if incoming_compact.startswith(emitted_compact):
                matched_compact_len = len(emitted_compact)
            else:
                min_match_len = 24
                max_start = max(0, len(emitted_compact) - 2000)
                for start in range(max_start, len(emitted_compact) - min_match_len + 1):
                    suffix = emitted_compact[start:]
                    if incoming_compact.startswith(suffix):
                        matched_compact_len = len(suffix)
                        break
            if matched_compact_len < 24 or matched_compact_len > len(incoming_compact_source_ends):
                return incoming
            cut_at = incoming_compact_source_ends[matched_compact_len - 1]
            return incoming[cut_at:].lstrip()
        cut_at = incoming_source_ends[matched_len - 1]
        return incoming[cut_at:].lstrip()

    def _looks_like_duplicate_thinking(
        self,
        thinking_text: str | None,
        content_text: str | None,
    ) -> bool:
        """Return True when a thinking payload is effectively the final answer."""
        normalized_thinking = self._normalize_comparable_text(thinking_text)
        normalized_content = self._normalize_comparable_text(content_text)
        if not normalized_thinking or not normalized_content:
            return False
        if normalized_thinking == normalized_content:
            return True
        shorter, longer = sorted(
            (normalized_thinking, normalized_content),
            key=len,
        )
        return len(shorter) >= 64 and shorter in longer

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

        if not content:
            return content or ""

        lines = content.split("\n")
        filtered_lines = []
        skip_until_blank = False

        for line in lines:
            # Skip lines that match execution step patterns
            if re.match(r"^Step \d+:", line):
                skip_until_blank = True
                continue

            # Skip lines that are part of step output
            if skip_until_blank:
                # If we hit a blank line or normal content, stop skipping
                if line.strip() == "" or not line.startswith(
                    (" ", "\t", "Observed", "Error", "Security", "Command:", "Successfully")
                ):
                    skip_until_blank = False
                    if line.strip():  # Add the line if it's not blank
                        filtered_lines.append(line)
                continue

            # Keep all other lines
            filtered_lines.append(line)

        # If everything was filtered (e.g. all lines were "Step N: ..."), fall back to
        # extracting the inline content from the last Step line so we never return "".
        if not filtered_lines and lines:
            for raw_line in reversed(lines):
                m = re.match(r"^Step \d+:\s*(.+)", raw_line)
                if m and m.group(1).strip():
                    filtered_lines = [m.group(1).strip()]
                    break
            if not filtered_lines:
                # Last-resort: return original content unchanged
                return content.strip()

        # Join and clean up excessive blank lines
        result = "\n".join(filtered_lines)
        result = re.sub(r"\n{3,}", "\n\n", result)  # Replace 3+ newlines with 2
        return result.strip()

    @staticmethod
    def _should_replace_stream_error_preamble(
        content: str,
        *,
        saw_tool_call: bool,
        saw_content_after_tool_call: bool,
    ) -> bool:
        """Return True when provider failure left only a stale pre-tool preamble."""
        if not str(content or "").strip():
            return True
        return saw_tool_call and not saw_content_after_tool_call

    @staticmethod
    def _is_internal_completion_sentinel(content: Any) -> bool:
        """Return True for provider/runtime completion markers that are not answers."""
        text = str(content or "").strip()
        return text in {
            "Task completed",
            "Task completed based on finish_reason signal",
            "Thinking completed. No action needed. Task finished.",
        }

    @staticmethod
    def _looks_like_tool_call_protocol_text(content: str) -> bool:
        """Return True when plain text contains model/tool protocol markup.

        Some OpenAI-compatible providers can surface their tool-call control
        language as reasoning text instead of structured ``tool_calls``. That
        markup is not user-visible content and, more importantly, it means the
        runtime did not execute the intended tool call.
        """
        text = str(content or "")
        if not text.strip():
            return False
        compact = re.sub(r"\s+", "", text)
        lower = compact.casefold()
        if "dsml" in lower:
            return any(
                marker in lower
                for marker in (
                    "tool_call",
                    "toolcalls",
                    "invoke",
                    "parameter",
                    "name=",
                )
            )
        if "<tool_call" in lower or "<toolcalls" in lower or "<tool_calls" in lower:
            return any(marker in lower for marker in ("name=", "function", "arguments", "invoke"))
        if "<invoke" in lower and "name=" in lower:
            return True
        return False

    @staticmethod
    def _looks_like_tool_call_protocol_fragment(content: str) -> bool:
        """Return True for partial streamed protocol tags that must not leak."""
        text = str(content or "")
        if not text.strip():
            return False
        compact = re.sub(r"\s+", "", text)
        lower = compact.casefold()
        if "dsml" in lower and "<" in lower:
            return True
        return any(
            marker in lower
            for marker in (
                "<tool_call",
                "<toolcalls",
                "<tool_calls",
                "<invoke",
                "<|tool",
            )
        )

    @staticmethod
    def _looks_like_pseudo_tool_call_text(content: str) -> bool:
        """Return True when plain text is pretending that tools were called."""
        text = str(content or "")
        if AgentLoop._looks_like_tool_call_protocol_text(text):
            return True
        if "Observed output of cmd" not in text or "execution:" not in text:
            return False
        return bool(
            re.search(
                r"(?im)^\s*(?:[-*]\s*)?`?[a-z_][a-z0-9_]*\s*"
                r"\([^)\n]{0,700}\)`?\s*:\s*Observed output of cmd\b.*\bexecution:\s*",
                text,
            )
        )

    @staticmethod
    def _build_pseudo_tool_call_repair_prompt(
        user_request: str,
        pseudo_content: str,
    ) -> str:
        """Build an internal retry prompt after the model emitted fake tool text."""
        return (
            "[INTERNAL TOOL-CALL REPAIR]\n"
            "Your previous assistant output wrote tool-call protocol markup or "
            "tool-call-shaped text as plain text. Those actions were NOT executed by "
            "the runtime, and any claimed outputs in that text are invalid.\n\n"
            "The invalid text is intentionally omitted from this repair context so it cannot "
            "be mistaken for evidence. Continue the latest user request from the real tool "
            "messages already in memory by calling tools through the tool-call API. Do not "
            "describe a tool call, do not write protocol tags or `tool_name(...)` in text, "
            "and do not claim success until actual tool results have been returned. If the "
            "needed tool is unavailable or fails, report that concrete blocker.\n\n"
            f"Latest user request:\n{user_request}"
        )

    @staticmethod
    def _build_repeated_read_recovery_prompt(
        user_request: str,
        *,
        request_context: str = "",
    ) -> str:
        """Return the neutral continuation token used by the core loop."""
        return AgentLoop.DEFAULT_NEXT_STEP_PROMPT

    @staticmethod
    def _build_history_search_budget_recovery_prompt(
        user_request: str,
        tool_result_events: list[dict[str, Any]],
        *,
        request_context: str = "",
    ) -> str:
        """Return the neutral continuation token used by the core loop."""
        return AgentLoop.DEFAULT_NEXT_STEP_PROMPT

    @staticmethod
    def _drop_pseudo_tool_call_assistant_messages(self, start_index: int) -> int:
        """Remove runtime assistant messages that contain fake tool-call transcripts."""
        if not isinstance(start_index, int) or start_index < 0:
            return 0
        try:
            messages = AgentLoop._get_runtime_memory_messages(self)
        except Exception:
            return 0
        if not isinstance(messages, list) or start_index >= len(messages):
            return 0

        removed = 0
        for index in range(len(messages) - 1, start_index - 1, -1):
            msg = messages[index]
            role = AgentLoop._stream_message_role(msg).lower()
            if role != "assistant":
                continue
            if AgentLoop._message_tool_calls_value(msg):
                continue
            content = AgentLoop._stream_message_attr(msg, "text_content", None)
            if content in (None, ""):
                content = AgentLoop._stream_message_attr(msg, "content", "")
            if AgentLoop._looks_like_pseudo_tool_call_text(str(content or "")):
                del messages[index]
                removed += 1
        return removed

    @staticmethod
    def _extract_run_result_text(result: Any) -> str:
        """Normalize a spoon-core run result to plain text."""
        if hasattr(result, "content") and result.content is not None:
            return str(result.content or "")
        if hasattr(result, "content"):
            return str(result) if str(result) != "None" else ""
        if result is None:
            return ""
        return str(result)

    @staticmethod
    def _looks_like_raw_tool_transcript_leak(content: str) -> bool:
        """Return True when provider fallback text is dominated by tool transcript artifacts."""
        text = str(content or "")
        if "Observed output of cmd" in text:
            if len(text) >= 4_000:
                return True
            if re.match(r"(?is)^\s*Observed output of cmd\b.*\bexecution:\s*", text):
                return True
            if AgentLoop._looks_like_pseudo_tool_call_text(text):
                return True
            return bool(
                re.search(
                    r"(?is)(?:^|.)\s*Step\s+\d+:\s*Observed output of cmd\b.*\bexecution:\s*",
                    text,
                )
            )

        marker_count = sum(
            1
            for pattern in (
                r"(?im)^\s*Thought process\s*$",
                r"(?im)^\s*Tool\s*[·:]\s*",
                r"(?im)^\s*Input\s*$",
                r"(?im)^\s*Output\s*$",
                r'(?im)^\s*"\s*action\s*"\s*:',
                r'(?im)^\s*"\s*arguments\s*"\s*:',
                r'(?im)^\s*"\s*result\s*"\s*:',
            )
            if re.search(pattern, text)
        )
        if marker_count >= 4 and re.search(r"(?im)^\s*Output\s*$", text):
            return True
        return False

    @staticmethod
    def _compact_tool_evidence_text(value: Any, *, limit: int = 700) -> str:
        text = AgentLoop._stringify_stream_payload(value)
        text = re.sub(r"^Observed output of cmd [^\n]* execution:\s*", "", text.strip())
        file_header = re.match(r"^\[file:\s*([^\]\n]+)\]\s*\n", text)
        if file_header:
            return (
                f"[file: {file_header.group(1)}] "
                "historical file reference in fallback; read the current local file again for exact text"
            )
        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        if lines:
            text = "\n".join(lines)
        text = AgentLoop._mask_user_visible_text(text)
        if len(text) <= limit:
            return text
        head = text[: max(120, limit // 2)].rstrip()
        tail = text[-max(120, limit // 3) :].lstrip()
        return f"{head}\n... (tool output truncated)\n{tail}"

    def _build_raw_tool_transcript_leak_response(
        self,
        start_index: int,
        *,
        user_message: str | None = None,
    ) -> str:
        """Build a user-facing fallback without exposing raw tool transcript text."""
        try:
            messages = AgentLoop._get_runtime_memory_messages(self)
        except Exception:
            messages = []
        if not isinstance(messages, list):
            messages = []
        if isinstance(start_index, int) and start_index >= 0:
            messages = messages[start_index:]

        tool_outputs: list[str] = []
        for msg in messages:
            if AgentLoop._stream_message_role(msg).lower() != "tool":
                continue
            content = AgentLoop._stream_message_attr(msg, "text_content", None)
            if content in (None, ""):
                content = AgentLoop._stream_message_attr(msg, "content", "")
            if content not in (None, ""):
                tool_outputs.append(str(content))

        return build_user_facing_tool_evidence_answer(
            tool_outputs[-8:],
            user_message=user_message,
        )

    @staticmethod
    def _looks_like_internal_scratchpad_text(text: str) -> bool:
        """Return True for short legacy ASCII provider-surfaced planning notes."""
        compact = " ".join(str(text or "").strip().split())
        if len(compact) < 12 or len(compact) > 500:
            return False

        ascii_chars = sum(1 for ch in compact if ord(ch) < 128)
        if ascii_chars / max(len(compact), 1) < 0.70:
            return False
        return bool(
            re.search(
                r"(?ix)"
                r"\b("
                r"need\s+(?:to|answer|respond|reply|summarize|mention|follow|continue|inspect|check|use|run|find|resolve|handle|tool)|"
                r"i\s+(?:need|should|have\s+to)|"
                r"i(?:'|’)ll\s+(?:run|check|inspect|fetch|use|look|read|verify|confirm|execute|search|open|review|try|handle|continue|see|start)|"
                r"we\s+(?:need|should|may|can)|"
                r"user\s+(?:asks|asked|wants|requested|said)|"
                r"let\s+me\s+(?:start|check|inspect|fetch|run|use|look|read|verify|confirm|execute|search|open|review|try|handle|continue|see)|"
                r"let(?:'|’)s|likely|maybe"
                r")\b",
                compact,
            )
        ) or (
            bool(re.search(r"(?i)\bneed\s+\S+", compact))
            and bool(
                re.search(
                    r"(?i)\b(?:check|inspect|run|use|call|try|maybe|likely|let(?:'|’)s|tool|command)\b",
                    compact,
                )
            )
        )

    @staticmethod
    def _sanitize_internal_guardrail_text(text: str) -> str:
        """Map internal loop-control markers to user-facing wording."""
        value = str(text or "")
        replacements = [
            (
                r"READ_FILE_CACHE_HIT:\s*requested file range already available in this request\.[^\n]*",
                "The requested file range is already available in this request; continuing from the existing context.",
            ),
            (
                r"STOP_TOOL_LOOP:\s*Error:\s*redundant file read suppressed\.[^\n]*",
                "The duplicate file read was skipped because that content range was already available.",
            ),
            (
                r"STOP_TOOL_LOOP:\s*Error:\s*duplicate tool invocation suppressed\.[^\n]*",
                "The duplicate action was skipped because the same action had already run.",
            ),
            (
                r"STOP_TOOL_LOOP:\s*Error:\s*repeated side-effecting tool series suppressed\.[^\n]*",
                "The repeated external action was skipped.",
            ),
            (
                r"STOP_TOOL_LOOP:\s*Error:\s*consecutive tool failures suppressed\.[^\n]*",
                "Repeated tool failures were stopped.",
            ),
            (
                r"redundant file read suppressed",
                "duplicate file read skipped",
            ),
            (
                r"duplicate tool invocation suppressed",
                "duplicate action skipped",
            ),
            (
                r"repeated side-effecting tool series suppressed",
                "repeated external action skipped",
            ),
            (
                r"consecutive tool failures suppressed",
                "repeated tool failures stopped",
            ),
        ]
        for pattern, replacement in replacements:
            value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)
        return value

    @staticmethod
    def _needs_tool_evidence_synthesis(
        content: str | None,
        *,
        user_message: str | None = None,
    ) -> bool:
        """Return True when current terminal text is absent or unsafe to preserve."""
        text = str(content or "").strip()
        if (
            not text
            or text in {"No results", "NO_CONCISE_TOOL_EVIDENCE"}
            or AgentLoop._is_internal_completion_sentinel(text)
        ):
            return True
        if AgentLoop._looks_like_internal_scratchpad_text(text):
            return True
        if AgentLoop._looks_like_raw_tool_transcript_leak(text):
            return True
        if AgentLoop._looks_like_pseudo_tool_call_text(text):
            return True
        if AgentLoop._final_answer_script_mismatch(user_message, text):
            return True
        return False

    @staticmethod
    def _deterministic_ledger_final(
        ledger: ExecutionLedger | None,
        *,
        max_chars: int = 5000,
    ) -> str:
        """Return a final-answer-safe ledger summary when tool facts exist."""
        if not isinstance(ledger, ExecutionLedger):
            return ""
        if not (ledger.has_stateful_progress() or ledger.open_blockers):
            return ""
        return ledger.render_user_facing_summary(max_chars=max_chars)

    @staticmethod
    def _should_use_deterministic_ledger_final(
        content: Any,
        *,
        user_message: str | None = None,
    ) -> bool:
        """Return True when the model answer is unusable and ledger facts must carry final output.

        The ledger summary is intentionally conservative and mechanical. It is a
        fallback for empty or contaminated answers, not a replacement for a
        grounded natural-language answer synthesized from the same evidence.
        """
        text = str(content or "").strip()
        if not text:
            return True
        if text in {"NO_CONCISE_TOOL_EVIDENCE", "No results"}:
            return True
        if AgentLoop._is_internal_completion_sentinel(text):
            return True
        if AgentLoop._looks_like_internal_scratchpad_text(text):
            return True
        if AgentLoop._looks_like_raw_tool_transcript_leak(text):
            return True
        if AgentLoop._looks_like_pseudo_tool_call_text(text):
            return True
        if AgentLoop._final_answer_script_mismatch(user_message, text):
            return True
        return False

    @staticmethod
    def _mask_user_visible_text(text: str) -> str:
        """Mask secrets before content is streamed, persisted, or finalized."""
        try:
            from spoon_bot.utils.privacy import mask_secrets

            masked = mask_secrets(str(text or ""))
        except Exception:
            masked = str(text or "")
        return AgentLoop._sanitize_internal_guardrail_text(masked)

    @staticmethod
    def _strip_leaked_scratchpad_prefix(content: str) -> str:
        """Remove a short internal planning preamble from a final answer.

        Some OpenAI-compatible providers can surface scratchpad-style text as a
        normal content segment. This is output hygiene, not task routing: only
        an initial, mostly-ASCII planning note is removed, and only when a
        substantive answer remains.
        """
        text = str(content or "")
        if not text.strip():
            return text

        def _strip_once(value: str) -> str:
            for match in re.finditer(r"[\u4e00-\u9fff]", value[:800]):
                if match.start() <= 0:
                    continue
                prefix = value[: match.start()]
                if re.search(r"[\u4e00-\u9fff]", prefix):
                    continue
                suffix = value[match.start() :]
                if suffix.strip() and AgentLoop._looks_like_internal_scratchpad_text(prefix):
                    return suffix.lstrip()

            sentence_match = re.match(
                r"^((?:[^\n.!?。！？]{1,240}[.!?。！？]\s*){1,3})(\S[\s\S]*)$",
                value,
            )
            if sentence_match:
                prefix, suffix = sentence_match.group(1), sentence_match.group(2)
                if suffix.strip() and AgentLoop._looks_like_internal_scratchpad_text(prefix):
                    return suffix.lstrip()
            return value

        for _ in range(4):
            stripped = _strip_once(text)
            if stripped == text:
                return text
            text = stripped
        return text

    def _finalize_response_content(
        self,
        message: str,
        content: str,
        *,
        turn_memory_start_index: int,
    ) -> str:
        """Apply generic execution-step filtering without prompt-derived dispatch."""
        if AgentLoop._looks_like_raw_tool_transcript_leak(content):
            return AgentLoop._mask_user_visible_text(
                AgentLoop._build_raw_tool_transcript_leak_response(
                    self,
                    turn_memory_start_index,
                    user_message=message,
                )
            )
        filtered = AgentLoop._filter_execution_steps(self, content)
        cleaned = AgentLoop._strip_leaked_scratchpad_prefix(filtered)
        return AgentLoop._mask_user_visible_text(cleaned)

    @staticmethod
    def _stream_message_attr(message: Any, key: str, default: Any = None) -> Any:
        """Read a message field from either dict or object runtime payloads."""
        if isinstance(message, dict):
            return message.get(key, default)
        return getattr(message, key, default)

    @staticmethod
    def _stream_message_role(message: Any) -> str:
        """Normalize runtime message roles to plain strings."""
        role = AgentLoop._stream_message_attr(message, "role", "")
        return role.value if hasattr(role, "value") else str(role or "")

    @staticmethod
    def _stringify_stream_payload(payload: Any) -> str:
        """Serialize structured tool payloads for websocket metadata."""
        if payload is None:
            return ""
        if isinstance(payload, str):
            return payload
        if isinstance(payload, (dict, list)):
            try:
                return json.dumps(payload, ensure_ascii=False)
            except Exception:
                return str(payload)
        return str(payload)

    @staticmethod
    def _tool_call_arguments_key(arguments: Any) -> str:
        """Normalize tool-call arguments for captured-output matching."""
        return normalize_tool_arguments(arguments)

    @staticmethod
    def _tool_call_arguments_display(tool_name: Any, arguments: Any) -> str:
        """Return tool-call arguments suitable for UI/session history."""
        return sanitize_tool_arguments_for_history(str(tool_name or ""), arguments)

    @staticmethod
    def _cap_stream_metadata_text(value: Any, *, limit: int = 200_000) -> tuple[str, bool, int]:
        """Return a websocket-safe text payload plus truncation metadata."""
        text = AgentLoop._stringify_stream_payload(value)
        original_len = len(text)
        if original_len <= limit:
            return text, False, original_len
        marker = "\n... (stream output middle truncated) ...\n"
        available = limit - len(marker)
        if available <= 0:
            return text[:limit], True, original_len
        head_limit = max(1, available // 2)
        tail_limit = max(1, available - head_limit)
        return (
            text[:head_limit].rstrip() + marker + text[-tail_limit:].lstrip(),
            True,
            original_len,
        )

    @staticmethod
    def _client_visible_tool_result_text(value: Any) -> tuple[str, bool]:
        """Return the current tool result shown to the client.

        Historical compaction may replace old file bodies with references, but
        the active tool result should carry the file text it just read. This
        keeps the UI and persisted stream aligned with what the model can use.
        """
        text = AgentLoop._stringify_stream_payload(value).strip()
        text = AgentLoop._mask_user_visible_text(text)
        return text, False

    @staticmethod
    def _model_visible_tool_result_text(value: Any) -> tuple[str, bool]:
        """Return the current tool result available to the model.

        The model needs the latest file contents it explicitly read in this
        turn. Old file bodies are removed by session compaction instead of being
        dropped from the active tool result.
        """
        text = AgentLoop._stringify_stream_payload(value).strip()
        text = AgentLoop._mask_user_visible_text(text)
        return text, False

    @staticmethod
    def _merge_stream_tool_result_metadata(
        metadata: dict[str, Any],
        *,
        streamed_result: str,
        captured_output: Any | None,
    ) -> dict[str, Any]:
        """Prefer captured full tool output while retaining model-visible summary."""
        merged = dict(metadata)
        summary_result = streamed_result
        full_result = streamed_result

        if captured_output is not None:
            captured_summary = getattr(captured_output, "summary_output", "") or summary_result
            captured_full = getattr(captured_output, "full_output", "") or captured_summary
            summary_result = captured_summary
            full_result = captured_full
            captured_metadata = getattr(captured_output, "metadata", None)
            if isinstance(captured_metadata, dict):
                for key, value in captured_metadata.items():
                    if value is not None and key not in merged:
                        merged[key] = value

        guardrail_message = AgentLoop._tool_loop_suppression_message_from_text(
            full_result or summary_result
        )
        client_result, omitted_body = AgentLoop._client_visible_tool_result_text(
            full_result or summary_result
        )
        client_summary, omitted_summary_body = AgentLoop._client_visible_tool_result_text(
            summary_result or client_result
        )
        tool_name = str(merged.get("name") or merged.get("tool") or "").strip()
        model_source = (
            full_result
            if tool_name == "read_file" and full_result
            else summary_result or client_summary or client_result
        )
        model_summary, omitted_model_summary_body = AgentLoop._model_visible_tool_result_text(
            model_source
        )

        if guardrail_message:
            merged["guardrail_stop"] = True
            merged["guardrail_message"] = guardrail_message
        if client_result:
            stream_full_result, stream_truncated, stream_original_len = (
                AgentLoop._cap_stream_metadata_text(
                    client_result,
                    limit=_CLIENT_STREAM_TOOL_RESULT_LIMIT,
                )
            )
            merged["result"] = stream_full_result
            merged["content"] = stream_full_result
            merged["output"] = stream_full_result
            merged["full_result"] = stream_full_result
            merged["full_content"] = stream_full_result
            merged["full_output"] = stream_full_result
            if stream_truncated:
                merged["stream_output_truncated"] = True
                merged["stream_output_original_chars"] = stream_original_len
        if model_summary:
            stream_summary, summary_truncated, summary_original_len = (
                AgentLoop._cap_stream_metadata_text(
                    model_summary,
                    limit=_MODEL_STREAM_TOOL_RESULT_LIMIT,
                )
            )
            merged["model_result"] = stream_summary
            merged["model_content"] = stream_summary
            merged["model_output"] = stream_summary
            if summary_truncated:
                merged["model_output_truncated"] = True
                merged["model_output_original_chars"] = summary_original_len
        if full_result and summary_result and full_result != summary_result:
            merged["result_truncated_for_model"] = True
        return merged

    def _remember_stream_tool_result_metadata(
        self,
        tool_call_id: Any,
        metadata: dict[str, Any],
    ) -> None:
        """Retain structural tool-result metadata for session trace persistence."""
        key = str(tool_call_id or "").strip()
        if not key or not isinstance(metadata, dict):
            return

        selected: dict[str, Any] = {}
        for field in (
            "status",
            "returncode",
            "return_code",
            "exit_code",
            "guardrail_stop",
            "guardrail_reason",
            "guardrail_message",
            "tool_outcome",
        ):
            if field in metadata and metadata[field] is not None:
                selected[field] = metadata[field]
        if not selected:
            return

        cache = getattr(self, "_stream_tool_result_metadata_by_id", None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(self, "_stream_tool_result_metadata_by_id", cache)
        cache[key] = selected

    def _stream_tool_result_metadata_for_trace(self, tool_call_id: Any) -> dict[str, Any]:
        key = str(tool_call_id or "").strip()
        if not key:
            return {}
        cache = getattr(self, "_stream_tool_result_metadata_by_id", None)
        if not isinstance(cache, dict):
            return {}
        value = cache.get(key)
        return dict(value) if isinstance(value, dict) else {}

    @staticmethod
    def _stream_tool_result_event_tool_name(event: dict[str, Any]) -> str:
        """Return a normalized tool name from a structural tool result event."""
        metadata = dict(event.get("metadata") or {})
        return str(metadata.get("name") or metadata.get("tool") or "").strip().lower()

    @staticmethod
    def _stream_tool_result_event_arguments(event: dict[str, Any]) -> Any:
        """Return decoded tool arguments from a structural tool result event."""
        metadata = dict(event.get("metadata") or {})
        value = metadata.get("arguments") or metadata.get("input") or metadata.get("args") or ""
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return ""
            try:
                parsed = json.loads(stripped)
            except Exception:
                return stripped
            return parsed
        return value

    @staticmethod
    def _stream_tool_result_event_is_context_only(event: dict[str, Any]) -> bool:
        """Return True for setup/inspection evidence that should not dominate finals."""
        tool_name = AgentLoop._stream_tool_result_event_tool_name(event)
        if tool_name in {
            "read_file",
            "list_dir",
            "grep",
            "web_fetch",
            "web_search",
            "skill_marketplace",
            "search_history",
        }:
            return True
        if tool_name == "shell":
            args = AgentLoop._stream_tool_result_event_arguments(event)
            if isinstance(args, dict):
                command = str(args.get("command") or "").strip()
            else:
                command = str(args or "").strip()
            return bool(command) and ShellTool.command_is_plain_read_only_inspection(command)
        return False

    @staticmethod
    def _select_final_answer_synthesis_events(
        tool_result_events: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Choose evidence for final answer synthesis without replaying old context.

        Read-only skill/file evidence helps the model choose actions during the
        turn, but after state-changing tools have run it can pollute the final
        answer with setup recipes. Keep it only when there is no stateful tool
        evidence to summarize.
        """
        events = [event for event in tool_result_events if isinstance(event, dict)]
        if not events:
            return []
        stateful_events = [
            event
            for event in events
            if not AgentLoop._stream_tool_result_event_is_context_only(event)
        ]
        selected = stateful_events or events
        return selected[-_FINAL_ANSWER_SYNTHESIS_TOOL_EVENT_LIMIT:]

    @staticmethod
    def _stream_tool_result_event_summary(event: dict[str, Any], *, limit: int = 2400) -> str:
        """Build a compact, user-visible summary from a structural tool result event."""
        metadata = dict(event.get("metadata") or {})
        tool_name = str(metadata.get("name") or metadata.get("tool") or "tool").strip() or "tool"
        payload = (
            metadata.get("model_result")
            or metadata.get("model_output")
            or metadata.get("model_content")
            or metadata.get("output")
            or metadata.get("result")
            or metadata.get("content")
            or metadata.get("full_output")
            or metadata.get("full_result")
            or event.get("delta")
        )
        text = AgentLoop._stringify_stream_payload(payload).strip()
        text = AgentLoop._mask_user_visible_text(text)
        if len(text) > limit:
            text = text[:limit].rstrip() + "\n... (tool output truncated)"
        if not text:
            text = "completed without text output"
        return f"Tool `{tool_name}` output:\n{text}"

    @staticmethod
    def _stream_tool_result_visible_delta(delta: Any, metadata: dict[str, Any]) -> str:
        """Return the user-visible WebSocket delta for a tool_result chunk."""
        text = AgentLoop._stringify_stream_payload(delta)
        if not text:
            text = AgentLoop._stringify_stream_payload(
                metadata.get("model_result")
                or metadata.get("model_output")
                or metadata.get("model_content")
                or ""
            )
        visible, _ = AgentLoop._client_visible_tool_result_text(text)
        capped, _, _ = AgentLoop._cap_stream_metadata_text(
            visible,
            limit=_CLIENT_STREAM_TOOL_DELTA_LIMIT,
        )
        return capped

    @staticmethod
    def _is_tool_loop_suppression_event(event: dict[str, Any]) -> bool:
        """Return True for tool guardrail results that should end the current loop."""
        metadata = dict(event.get("metadata") or {})
        if metadata.get("guardrail_stop") is True:
            return True
        payload = (
            metadata.get("model_result")
            or metadata.get("model_output")
            or metadata.get("model_content")
            or metadata.get("output")
            or metadata.get("result")
            or metadata.get("content")
            or metadata.get("full_output")
            or metadata.get("full_result")
            or event.get("delta")
        )
        text = AgentLoop._stringify_stream_payload(payload).lower()
        return "stop_tool_loop" in text

    @staticmethod
    def _tool_loop_suppression_message_from_text(value: Any) -> str | None:
        """Return a user-facing message for a raw STOP_TOOL_LOOP payload."""
        text = AgentLoop._stringify_stream_payload(value).lower()
        if "stop_tool_loop" not in text:
            return None
        if "exact requested shell command failed" in text:
            return None
        if "history search budget" in text:
            return "I stopped searching old conversation history and continued from the latest request."
        if "consecutive tool failures suppressed" in text:
            return "I stopped retrying after repeated failures."
        if "repeated tool failure pattern suppressed" in text:
            return "I stopped retrying after the same failure pattern repeated."
        if "repeated side-effecting tool series suppressed" in text:
            return "I stopped before repeating the same external action."
        if "repeated shell file read suppressed" in text:
            return "I stopped after a repeated shell file inspection produced no new action."
        if "duplicate shell inspection suppressed" in text:
            return "I stopped after a repeated status/inspection command produced no new action."
        if "redundant file read suppressed" in text:
            return "I skipped a file read whose content was already available."
        if "duplicate tool invocation suppressed" in text:
            return "I skipped a repeated action that had already run."
        return "I stopped before repeating the same action."

    @staticmethod
    def _extract_exact_command_failure_blocker(event: dict[str, Any]) -> str | None:
        """Extract a user-facing blocker from an exact-command STOP_TOOL_LOOP shell result."""
        metadata = dict(event.get("metadata") or {})
        tool_name = str(metadata.get("name") or metadata.get("tool") or "").strip().lower()
        if tool_name != "shell":
            return None

        payload = (
            metadata.get("model_result")
            or metadata.get("model_output")
            or metadata.get("model_content")
            or metadata.get("output")
            or metadata.get("result")
            or metadata.get("content")
            or metadata.get("full_output")
            or metadata.get("full_result")
            or event.get("delta")
        )
        text = AgentLoop._stringify_stream_payload(payload).strip()
        if "STOP_TOOL_LOOP: Exact requested shell command failed." not in text:
            return None

        cleaned = re.sub(
            r"^STOP_TOOL_LOOP: Exact requested shell command failed\.[^\n]*\n?",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        if cleaned:
            return AgentLoop._mask_user_visible_text(cleaned)
        return "The exact requested shell command failed before the task could continue."

    @staticmethod
    def _tool_loop_suppression_message(event: dict[str, Any]) -> str | None:
        """Return a user-facing message for an internal STOP_TOOL_LOOP guardrail."""
        metadata = dict(event.get("metadata") or {})
        guardrail_message = metadata.get("guardrail_message")
        if isinstance(guardrail_message, str) and guardrail_message.strip():
            return guardrail_message.strip()
        payload = (
            metadata.get("model_result")
            or metadata.get("model_output")
            or metadata.get("model_content")
            or metadata.get("output")
            or metadata.get("result")
            or metadata.get("content")
            or metadata.get("full_output")
            or metadata.get("full_result")
            or event.get("delta")
        )
        return AgentLoop._tool_loop_suppression_message_from_text(payload)

    @staticmethod
    def _extract_current_session_fact_check_blocker(text: str) -> str | None:
        """Extract the user-facing current-session fact-check blocker."""
        value = str(text or "").strip()
        lower = value.casefold()
        marker = "current-session fact check required"
        index = lower.find(marker)
        if index < 0:
            return None
        blocker = value[index:].strip()
        blocker = re.sub(r"\s+", " ", blocker).strip()
        if not blocker:
            return None
        return f"{blocker} No external/live-state lookup was used as proof of the prior action."

    @staticmethod
    def _extract_tool_suppression_user_response(
        tool_result_events: list[dict[str, Any]],
    ) -> str | None:
        """Convert STOP_TOOL_LOOP events into a clean final answer."""
        suppression_message: str | None = None
        for event in reversed(tool_result_events):
            message = AgentLoop._tool_loop_suppression_message(event)
            if message:
                suppression_message = message
                break
        if not suppression_message:
            return None

        for event in reversed(tool_result_events):
            if AgentLoop._is_tool_loop_suppression_event(event):
                continue
            summary = AgentLoop._stream_tool_result_event_summary(event)
            if summary:
                fact_check_blocker = AgentLoop._extract_current_session_fact_check_blocker(summary)
                if fact_check_blocker:
                    return fact_check_blocker
                return (
                    f"{suppression_message}\n\n"
                    "Latest available result:\n\n"
                    f"{summary}\n\n"
                    "No additional duplicate action was run."
                )

        return f"{suppression_message} No new result was produced."

    @staticmethod
    def _build_tool_loop_fallback_response(
        tool_result_events: list[dict[str, Any]],
        *,
        reason: str,
        user_message: str | None = None,
    ) -> str:
        """Return a bounded final answer when a tool loop never yields final content."""
        if reason == "tool_suppression":
            for event in reversed(tool_result_events):
                exact_blocker = AgentLoop._extract_exact_command_failure_blocker(event)
                if exact_blocker:
                    return exact_blocker
            suppression_response = AgentLoop._extract_tool_suppression_user_response(
                tool_result_events
            )
            if suppression_response:
                return suppression_response

        return build_user_facing_tool_event_answer(
            tool_result_events[-6:],
            incomplete=True,
            user_message=user_message,
        )

    @staticmethod
    def _final_answer_script_mismatch(
        user_message: str | None,
        final_content: str | None,
    ) -> bool:
        """Detect clear script mismatches without classifying prompt routes."""
        if (
            not isinstance(user_message, str)
            or not user_message.strip()
            or not isinstance(final_content, str)
            or not final_content.strip()
            or _MACHINE_READABLE_FORMAT_RE.search(user_message)
        ):
            return False

        user_scripts = set(dominant_non_latin_scripts(user_message))
        if not user_scripts:
            return False
        final_scripts = set(dominant_non_latin_scripts(final_content))
        return not bool(user_scripts & final_scripts)

    async def _synthesize_final_answer_from_tool_events(
        self,
        tool_result_events: list[dict[str, Any]],
        *,
        user_message: str | None,
        incomplete: bool = False,
        fallback_text: str = "",
    ) -> str:
        """Ask the configured model to write the final user-facing answer from evidence."""
        synthesis_events = AgentLoop._select_final_answer_synthesis_events(tool_result_events)
        evidence_brief = build_tool_event_synthesis_brief(
            synthesis_events,
            incomplete=incomplete,
            user_message=user_message,
        )
        active_ledger = getattr(self, "_active_execution_ledger", None)
        ledger_fallback = ""
        if isinstance(active_ledger, ExecutionLedger):
            ledger_context = active_ledger.render_context(max_chars=8000)
            if ledger_context.strip():
                evidence_brief = (
                    f"[STRUCTURED VERIFIED EVIDENCE]\n{ledger_context}\n{evidence_brief}"
                )
            ledger_fallback = active_ledger.render_user_facing_summary(max_chars=5000)
        deterministic_fallback = (
            ledger_fallback
            or str(fallback_text or "").strip()
            or build_user_facing_tool_event_answer(
                synthesis_events,
                incomplete=incomplete,
                user_message=user_message,
            )
        )
        chatbot = getattr(self, "_chatbot", None)
        manager = getattr(chatbot, "llm_manager", None)
        chat = getattr(manager, "chat", None)
        ask = getattr(chatbot, "ask", None)
        if not callable(chat) and not callable(ask):
            return deterministic_fallback

        retry_config = getattr(self, "_retry_config", None)
        if not isinstance(retry_config, RetryConfig):
            retry_config = DEFAULT_RETRY_CONFIG

        async def _chat_with_brief(brief: str, *, label: str) -> Any:
            messages = [
                Message(role="system", content=_FINAL_ANSWER_SYNTHESIS_SYSTEM_PROMPT),
                Message(role="user", content=brief),
            ]

            async def _do_synthesis() -> Any:
                async def _invoke() -> Any:
                    if callable(chat):
                        result = chat(
                            messages=messages,
                            provider=getattr(self, "provider", None),
                        )
                    else:
                        result = ask(
                            messages=[Message(role="user", content=brief)],
                            system_msg=_FINAL_ANSWER_SYNTHESIS_SYSTEM_PROMPT,
                        )
                    if inspect.isawaitable(result):
                        return await result
                    return result

                return await asyncio.wait_for(
                    _invoke(),
                    timeout=_FINAL_ANSWER_SYNTHESIS_TIMEOUT,
                )

            return await with_provider_retry(
                _do_synthesis,
                config=retry_config,
                on_retry=lambda attempt, exc, delay: logger.warning(
                    f"{label} transient error "
                    f"(attempt {attempt + 1}/{retry_config.max_retries + 1}), "
                    f"retrying in {delay:.1f}s: {type(exc).__name__}: {exc}"
                ),
            )

        try:
            response = await _chat_with_brief(
                evidence_brief,
                label="Final-answer synthesis",
            )
        except Exception as exc:
            logger.warning(f"Final-answer synthesis failed; using fallback: {exc}")
            return deterministic_fallback

        synthesized = AgentLoop._extract_run_result_text(response).strip()
        if not synthesized:
            content = getattr(response, "content", "")
            synthesized = str(content or "").strip()
        if not synthesized:
            return deterministic_fallback
        if AgentLoop._final_answer_script_mismatch(user_message, synthesized):
            logger.warning(
                "Final-answer synthesis returned a script mismatch; "
                "requesting a language repair from the synthesis model."
            )
        synthesized = AgentLoop._mask_user_visible_text(
            AgentLoop._strip_leaked_scratchpad_prefix(synthesized)
        )
        if AgentLoop._final_answer_script_mismatch(user_message, synthesized):
            language_repair_brief = (
                evidence_brief
                + "\n\n[SYNTHESIS LANGUAGE CHECK]\n"
                + "The previous draft did not match the newest user's language. "
                + "Rewrite the final answer from the same verified evidence, "
                + "using the newest user's language and no raw tool transcript."
            )
            try:
                language_repair_response = await _chat_with_brief(
                    language_repair_brief,
                    label="Final-answer synthesis language repair",
                )
            except Exception as exc:
                logger.warning(
                    f"Final-answer synthesis language repair failed; using fallback: {exc}"
                )
                return deterministic_fallback
            repaired_language = AgentLoop._extract_run_result_text(language_repair_response).strip()
            if not repaired_language:
                content = getattr(language_repair_response, "content", "")
                repaired_language = str(content or "").strip()
            repaired_language = AgentLoop._mask_user_visible_text(
                AgentLoop._strip_leaked_scratchpad_prefix(repaired_language)
            )
            if repaired_language and not AgentLoop._final_answer_script_mismatch(
                user_message,
                repaired_language,
            ):
                synthesized = repaired_language
            elif deterministic_fallback and not AgentLoop._final_answer_script_mismatch(
                user_message,
                deterministic_fallback,
            ):
                return deterministic_fallback
        if final_answer_denies_available_tool_evidence(synthesized, synthesis_events):
            repair_brief = (
                evidence_brief
                + "\n\n[SYNTHESIS QUALITY CHECK]\n"
                + "The previous draft denied or ignored available tool evidence. "
                + "Rewrite the final answer from the verified chronological evidence "
                + "and direct user-facing evidence above. If one requested detail is "
                + "not confirmed, say that narrowly while still reporting the "
                + "confirmed completed facts."
            )
            try:
                repair_response = await _chat_with_brief(
                    repair_brief,
                    label="Final-answer synthesis repair",
                )
            except Exception as exc:
                logger.warning(f"Final-answer synthesis repair failed; using fallback: {exc}")
                return deterministic_fallback

            repaired = AgentLoop._extract_run_result_text(repair_response).strip()
            if not repaired:
                content = getattr(repair_response, "content", "")
                repaired = str(content or "").strip()
            repaired = AgentLoop._mask_user_visible_text(
                AgentLoop._strip_leaked_scratchpad_prefix(repaired)
            )
            if (
                repaired
                and not AgentLoop._final_answer_script_mismatch(user_message, repaired)
                and not final_answer_denies_available_tool_evidence(
                    repaired,
                    synthesis_events,
                )
            ):
                return repaired
            logger.warning(
                "Final-answer synthesis denied available tool evidence after repair; "
                "using deterministic evidence summary."
            )
            return deterministic_fallback
        return synthesized

    def _session_evidence_synthesis_events(
        self,
        tool_result_events: list[dict[str, Any]],
        *,
        user_message: str | None,
    ) -> list[dict[str, Any]]:
        """Append same-session compact evidence for prior-state answer synthesis."""
        synthesis_events = list(tool_result_events)
        try:
            raw_messages = (
                self._session.get_messages()
                if hasattr(self._session, "get_messages")
                else self._session.get_history()
            )
        except Exception as exc:
            logger.debug(f"Session turn evidence load skipped: {exc}")
            raw_messages = []

        try:
            payload = build_recent_session_turns_payload(
                raw_messages,
                limit=8,
                max_content_length=700,
            )
        except Exception as exc:
            logger.debug(f"Session turn evidence build skipped: {exc}")
            payload = {}

        appended_turn_evidence = False
        turn_payloads = payload.get("substantive_turns") if isinstance(payload, dict) else []
        if not isinstance(turn_payloads, list):
            turn_payloads = []
        for index, turn in enumerate(turn_payloads[:8], start=1):
            if not isinstance(turn, dict):
                continue
            lines = [
                f"Recent session turn {index}:",
                f"user_request: {turn.get('user_request') or ''}",
            ]
            invoked = turn.get("invoked_skills")
            if invoked:
                lines.append(f"invoked_skills: {invoked}")
            if turn.get("latest_stateful_tool_result"):
                lines.append(
                    f"latest_stateful_tool_result: {turn.get('latest_stateful_tool_result')}"
                )
            recent_results = turn.get("recent_tool_results")
            if isinstance(recent_results, list) and recent_results:
                for result_index, result in enumerate(recent_results[:3], start=1):
                    lines.append(f"recent_tool_result_{result_index}: {result}")
            if turn.get("assistant_summary"):
                lines.append(f"assistant_summary: {turn.get('assistant_summary')}")
            synthesis_events.append(
                {
                    "metadata": {
                        "name": "current_session_turn",
                        "full_output": "\n".join(str(line) for line in lines if str(line).strip()),
                    }
                }
            )
            appended_turn_evidence = True

        if not appended_turn_evidence:
            try:
                session_compact_evidence = self._build_session_recall_context(
                    user_message or ""
                ).strip()
            except Exception as exc:
                logger.debug(f"Session compact evidence build skipped: {exc}")
                session_compact_evidence = ""
        else:
            session_compact_evidence = ""
        if session_compact_evidence:
            synthesis_events.append(
                {
                    "metadata": {
                        "name": "current_session_compact",
                        "full_output": session_compact_evidence,
                    }
                }
            )
        return synthesis_events

    @staticmethod
    def _extract_json_object_text(text: str | None) -> str:
        """Extract a single JSON object from model text."""
        value = str(text or "").strip()
        if not value:
            return ""
        if value.startswith("```"):
            lines = value.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            value = "\n".join(lines).strip()
        if value.startswith("{") and value.endswith("}"):
            return value
        start = value.find("{")
        end = value.rfind("}")
        if start >= 0 and end > start:
            return value[start : end + 1]
        return ""

    @staticmethod
    def _skill_contract_evidence_fingerprint(
        events: list[dict[str, Any]],
    ) -> tuple[tuple[str, str, str], ...]:
        fingerprint: set[tuple[str, str, str]] = set()
        for event in events:
            metadata = dict(event.get("metadata") or {})
            name = str(metadata.get("name") or metadata.get("tool") or "").strip()
            arguments = AgentLoop._stringify_stream_payload(
                metadata.get("arguments") or metadata.get("input") or metadata.get("args") or ""
            )
            summary = AgentLoop._stream_tool_result_event_summary(event, limit=240)
            fingerprint.add(
                (
                    name,
                    AgentLoop._normalize_comparable_text(arguments),
                    AgentLoop._normalize_comparable_text(summary),
                )
            )
        return tuple(sorted(fingerprint))

    @staticmethod
    def _skill_contract_tool_call_brief(
        events: list[dict[str, Any]],
        *,
        limit: int = 10,
    ) -> str:
        """Summarize observed tool calls without interpreting task intent."""
        rows: list[str] = []
        seen: set[tuple[str, str]] = set()
        for event in events:
            metadata = dict(event.get("metadata") or {})
            name = str(metadata.get("name") or metadata.get("tool") or "").strip()
            if not name:
                continue
            arguments = AgentLoop._stringify_stream_payload(
                metadata.get("arguments") or metadata.get("input") or metadata.get("args") or ""
            ).strip()
            key = (
                name,
                AgentLoop._normalize_comparable_text(arguments),
            )
            if key in seen:
                continue
            seen.add(key)
            if arguments:
                arguments = AgentLoop._compress_message_content(arguments, 360)
                rows.append(f"- {name} input={arguments}")
            else:
                rows.append(f"- {name}")
        if not rows:
            return "- No tool calls captured."
        return "\n".join(rows[-limit:])

    @staticmethod
    def _continuation_tool_event_brief(event: dict[str, Any], *, limit: int = 1200) -> list[str]:
        """Return compact, non-body evidence lines for continuation prompts."""
        metadata = dict(event.get("metadata") or {})
        name = str(metadata.get("name") or metadata.get("tool") or "").strip()
        arguments = AgentLoop._stringify_stream_payload(
            metadata.get("arguments") or metadata.get("input") or metadata.get("args") or ""
        ).strip()
        heading = name or "tool_result"
        if arguments:
            heading += " input=" + AgentLoop._compress_message_content(arguments, 360)

        summary = AgentLoop._stream_tool_result_event_summary(event, limit=limit)
        if name == "read_file":
            first_line = ""
            for raw_line in str(summary or "").splitlines():
                if raw_line.strip():
                    first_line = raw_line.strip()
                    break
            body_ref = (
                first_line
                if first_line.startswith("[file:")
                else "file content was read successfully"
            )
            return [
                f"- {heading}",
                f"{body_ref}. Use this as the current local file state; do not answer by repeating the file body.",
            ]
        if name in {"list_dir", "grep", "search_history", "web_search", "web_fetch"}:
            if summary:
                return [
                    f"- {heading}",
                    AgentLoop._compress_message_content(summary, min(limit, 360)),
                ]
            return [f"- {heading}"]
        if summary:
            return [
                f"- {heading}",
                AgentLoop._compress_message_content(summary, limit),
            ]
        return [f"- {heading}"]

    @staticmethod
    def _skill_contract_continuation_attempt_limit() -> int:
        raw = os.getenv("SPOON_BOT_SKILL_COMPLETION_MAX_CONTINUATIONS")
        if raw is not None:
            try:
                return max(0, int(raw.strip()))
            except ValueError:
                pass
        return 12

    @staticmethod
    def _task_completion_continuation_attempt_limit() -> int:
        raw = os.getenv("SPOON_BOT_TASK_COMPLETION_MAX_CONTINUATIONS")
        if raw is not None:
            try:
                return max(0, int(raw.strip()))
            except ValueError:
                pass
        return 20

    @staticmethod
    def _request_is_plain_bounded_continuation(
        authoritative_message: str,
        request_execution_hints: dict[str, Any] | None = None,
    ) -> bool:
        if (
            isinstance(request_execution_hints, dict)
            and "plain_continuation" in request_execution_hints
        ):
            return bool(request_execution_hints.get("plain_continuation"))
        return request_is_plain_continuation_only(authoritative_message)

    @staticmethod
    def _plain_continuation_can_auto_continue_same_unit(
        tool_result_events: list[dict[str, Any]],
    ) -> bool:
        """Allow only same-unit monitoring, not a new repeated side-effect unit."""
        return latest_tool_event_has_active_background_job(tool_result_events)

    @staticmethod
    def _parse_task_completion_verdict(value: Any) -> dict[str, str] | None:
        text = AgentLoop._stringify_stream_payload(value).strip()
        if not text:
            return None
        parsed: Any
        try:
            parsed = json.loads(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                return None
            try:
                parsed = json.loads(text[start : end + 1])
            except Exception:
                return None
        if not isinstance(parsed, dict):
            return None
        status = str(parsed.get("status") or "").strip().casefold()
        if status not in {"complete", "needs_continuation", "awaiting_user"}:
            return None
        return {
            "status": status,
            "reason": str(parsed.get("reason") or "").strip(),
            "next_focus": ""
            if status == "awaiting_user"
            else str(parsed.get("next_focus") or "").strip(),
        }

    def _build_task_completion_verdict_brief(
        self,
        *,
        authoritative_message: str,
        final_content: str | None,
        tool_result_events: list[dict[str, Any]],
    ) -> str:
        lines = [
            "[TASK COMPLETION VERIFICATION]",
            "Decide whether another tool-using step is needed.",
            "",
            "[NEWEST USER REQUEST]",
            str(authoritative_message or "").strip(),
            "",
            "[TURN BOUNDARY RULE]",
            "Only the newest user request can authorize a new side effect. The "
            "assistant draft is not a user reply. If the assistant draft is "
            "asking the user for a decision, confirmation, missing value, or "
            "permission before continuing, mark the turn as awaiting_user unless "
            "the newest user request already explicitly authorized that exact "
            "next action and the evidence is still incomplete.",
        ]
        if request_is_plain_continuation_only(authoritative_message):
            lines.extend(
                [
                    "",
                    "[BOUNDED CONTINUATION RULE]",
                    "The newest request is only a continuation acknowledgement; "
                    "it adds no new count, target, or scope. Once this turn has "
                    "produced a concrete tool result for the selected prior task, "
                    "mark the turn complete or awaiting_user instead of using an "
                    "older repeated/countable request as permission to start "
                    "another similar side-effect unit. The only exception is "
                    "same-unit monitoring of an already running background job.",
                ]
            )
        if final_content and str(final_content).strip():
            lines.extend(
                [
                    "",
                    "[ASSISTANT DRAFT]",
                    AgentLoop._compress_message_content(str(final_content).strip(), 1800),
                ]
            )
        active_ledger = getattr(self, "_active_execution_ledger", None)
        if isinstance(active_ledger, ExecutionLedger):
            ledger_context = active_ledger.render_context(max_chars=5000)
            if ledger_context.strip():
                lines.extend(["", "[STRUCTURED TOOL EVIDENCE]", ledger_context])

        evidence_lines: list[str] = []
        for event in tool_result_events[-12:]:
            metadata = dict(event.get("metadata") or {})
            name = str(metadata.get("name") or metadata.get("tool") or "").strip()
            arguments = AgentLoop._stringify_stream_payload(
                metadata.get("arguments") or metadata.get("input") or metadata.get("args") or ""
            ).strip()
            heading = name or "tool_result"
            if arguments:
                heading += " input=" + AgentLoop._compress_message_content(arguments, 260)
            evidence_lines.append(f"- {heading}")
            summary = AgentLoop._stream_tool_result_event_summary(event, limit=900)
            if summary:
                evidence_lines.append(AgentLoop._compress_message_content(summary, 900))
        if evidence_lines:
            lines.extend(["", "[RECENT TOOL EVENTS]", *evidence_lines])

        lines.extend(
            [
                "",
                "[OUTPUT FORMAT]",
                '{"status":"complete|needs_continuation|awaiting_user","reason":"short evidence-based reason","next_focus":"next tool objective or empty"}',
            ]
        )
        return "\n".join(line for line in lines if line is not None)

    async def _evaluate_task_completion_verdict(
        self,
        *,
        authoritative_message: str,
        final_content: str | None,
        tool_result_events: list[dict[str, Any]],
    ) -> dict[str, str]:
        """Ask a generic verifier whether the tool-backed request is terminal."""
        if not tool_result_events:
            return {"status": "complete", "reason": "No tool evidence.", "next_focus": ""}
        if any(AgentLoop._is_tool_loop_suppression_event(event) for event in tool_result_events):
            return {
                "status": "complete",
                "reason": "Tool guardrail produced a blocker to report.",
                "next_focus": "",
            }
        if latest_tool_event_has_active_background_job(tool_result_events):
            return {
                "status": "needs_continuation",
                "reason": "Latest evidence is a running background job, not a terminal outcome.",
                "next_focus": (
                    "Inspect or monitor the existing background job/output. Do not "
                    "rerun the same command; if no new evidence is available, "
                    "report the current running or partial state."
                ),
            }

        final_text = str(final_content or "").strip()
        if not final_text or final_text in {"No results", "NO_CONCISE_TOOL_EVIDENCE"}:
            return {
                "status": "needs_continuation",
                "reason": "No terminal assistant content after tool use.",
                "next_focus": "",
            }
        if read_only_tool_turn_needs_continuation(final_content, tool_result_events):
            return {
                "status": "needs_continuation",
                "reason": (
                    "Only read-only tool evidence is present and the assistant "
                    "draft is raw tool output, not a grounded terminal answer."
                ),
                "next_focus": (
                    "Continue from the current files/tool evidence. If the "
                    "latest request requires a workspace change, call the "
                    "appropriate mutation tool; otherwise answer from the "
                    "read-only evidence in user-facing prose."
                ),
            }

        chatbot = getattr(self, "_chatbot", None)
        manager = getattr(chatbot, "llm_manager", None)
        chat = getattr(manager, "chat", None)
        ask = getattr(chatbot, "ask", None)
        if not callable(chat) and not callable(ask):
            if latest_tool_event_has_user_summary_marker(tool_result_events):
                return {
                    "status": "complete",
                    "reason": (
                        "Latest tool evidence contains a user-facing summary and "
                        "no verifier model is available."
                    ),
                    "next_focus": "",
                }
            return {
                "status": "complete",
                "reason": "No verifier model available.",
                "next_focus": "",
            }

        brief = self._build_task_completion_verdict_brief(
            authoritative_message=authoritative_message,
            final_content=final_content,
            tool_result_events=tool_result_events,
        )
        messages = [
            Message(role="system", content=_TASK_COMPLETION_VERDICT_SYSTEM_PROMPT),
            Message(role="user", content=brief),
        ]

        try:

            async def _invoke() -> Any:
                if callable(chat):
                    result = chat(
                        messages=messages,
                        provider=getattr(self, "provider", None),
                    )
                else:
                    result = ask(
                        messages=[Message(role="user", content=brief)],
                        system_msg=_TASK_COMPLETION_VERDICT_SYSTEM_PROMPT,
                    )
                if inspect.isawaitable(result):
                    return await result
                return result

            response = await asyncio.wait_for(
                _invoke(),
                timeout=_TASK_COMPLETION_VERDICT_TIMEOUT,
            )
        except Exception as exc:
            logger.debug(f"Task completion verifier skipped: {exc}")
            return {"status": "complete", "reason": "Verifier unavailable.", "next_focus": ""}

        response_text = AgentLoop._extract_run_result_text(response).strip()
        if not response_text:
            response_text = str(getattr(response, "content", "") or "").strip()
        parsed = AgentLoop._parse_task_completion_verdict(response_text)
        if parsed is None:
            logger.debug(
                f"Task completion verifier returned unparsable output: {response_text[:300]}"
            )
            return {"status": "complete", "reason": "Verifier output unparsable.", "next_focus": ""}
        return parsed

    async def _evaluate_skill_completion_verdict(
        self,
        *,
        authoritative_message: str,
        final_content: str | None,
        tool_result_events: list[dict[str, Any]],
    ) -> dict[str, str]:
        """Use only structured event state to decide if another model step is needed."""
        if not should_run_skill_contract_check(tool_result_events):
            return {"status": "complete", "reason": "No skill contract evidence.", "next_focus": ""}
        active_ledger = getattr(self, "_active_execution_ledger", None)
        if (
            isinstance(active_ledger, ExecutionLedger)
            and active_ledger.has_stateful_progress()
            and str(final_content or "").strip()
            and str(final_content or "").strip() not in {"No results", "NO_CONCISE_TOOL_EVIDENCE"}
        ):
            return {
                "status": "complete",
                "reason": "Ledger contains stateful tool evidence and terminal content.",
                "next_focus": "",
            }
        if skill_contract_needs_continuation(final_content, tool_result_events):
            return {
                "status": "needs_continuation",
                "reason": "Skill-backed turn has no terminal response or only setup/read evidence.",
                "next_focus": "",
            }
        return {
            "status": "complete",
            "reason": "Skill turn has terminal content or progress.",
            "next_focus": "",
        }

    async def _continue_task_until_terminal(
        self,
        *,
        authoritative_message: str,
        request_execution_hints: dict[str, Any],
        final_content: str,
        tool_result_events: list[dict[str, Any]],
        retry_runner: Callable[..., Awaitable[Any]],
        run_kwargs: dict[str, Any],
        memory_start_index: int,
        label: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        attempts = 0
        limit = AgentLoop._task_completion_continuation_attempt_limit()
        plain_continuation = AgentLoop._request_is_plain_bounded_continuation(
            authoritative_message,
            request_execution_hints,
        )
        while attempts < limit and tool_result_events:
            if (
                plain_continuation
                and not AgentLoop._plain_continuation_can_auto_continue_same_unit(
                    tool_result_events,
                )
            ):
                break
            verdict = await self._evaluate_task_completion_verdict(
                authoritative_message=authoritative_message,
                final_content=final_content,
                tool_result_events=tool_result_events,
            )
            if not isinstance(verdict, dict) or verdict.get("status") != "needs_continuation":
                break

            before_fingerprint = AgentLoop._skill_contract_evidence_fingerprint(
                tool_result_events,
            )
            attempts += 1
            final_content = await self._run_process_task_continuation(
                authoritative_message=authoritative_message,
                request_execution_hints=request_execution_hints,
                tool_result_events=tool_result_events,
                retry_runner=retry_runner,
                run_kwargs=run_kwargs,
                label=f"{label}_{attempts}",
                previous_draft=final_content,
                continuation_reason=verdict.get("reason"),
                continuation_focus=verdict.get("next_focus"),
            )
            tool_result_events = self._collect_runtime_tool_result_events_from_memory(
                memory_start_index
            )
            after_fingerprint = AgentLoop._skill_contract_evidence_fingerprint(
                tool_result_events,
            )
            if after_fingerprint == before_fingerprint:
                logger.warning("Task continuation produced no new tool evidence; stopping.")
                active_ledger = getattr(self, "_active_execution_ledger", None)
                if isinstance(active_ledger, ExecutionLedger):
                    if not (active_ledger.has_stateful_progress() or active_ledger.file_reads):
                        active_ledger.record_blocker(
                            tool_name="agent_loop",
                            reason="continuation_without_tool_progress",
                            summary=(
                                "The request still needed another tool step, but "
                                "the internal continuation produced no new verified "
                                "tool evidence."
                            ),
                        )
                    ledger_summary = active_ledger.render_user_facing_summary(max_chars=5000)
                    if ledger_summary:
                        final_content = ledger_summary
                break
        return final_content, tool_result_events

    async def _continue_skill_contract_until_terminal(
        self,
        *,
        authoritative_message: str,
        request_execution_hints: dict[str, Any],
        final_content: str,
        tool_result_events: list[dict[str, Any]],
        retry_runner: Callable[..., Awaitable[Any]],
        run_kwargs: dict[str, Any],
        memory_start_index: int,
        label: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        attempts = 0
        limit = AgentLoop._skill_contract_continuation_attempt_limit()
        plain_continuation = AgentLoop._request_is_plain_bounded_continuation(
            authoritative_message,
            request_execution_hints,
        )
        while (
            attempts < limit
            and tool_result_events
            and should_run_skill_contract_check(tool_result_events)
        ):
            if (
                plain_continuation
                and not AgentLoop._plain_continuation_can_auto_continue_same_unit(
                    tool_result_events,
                )
            ):
                break
            verdict = await self._evaluate_skill_completion_verdict(
                authoritative_message=authoritative_message,
                final_content=final_content,
                tool_result_events=tool_result_events,
            )
            if not isinstance(verdict, dict) or verdict.get("status") != "needs_continuation":
                break

            before_fingerprint = AgentLoop._skill_contract_evidence_fingerprint(
                tool_result_events,
            )
            attempts += 1
            final_content = await self._run_process_skill_contract_continuation(
                authoritative_message=authoritative_message,
                request_execution_hints=request_execution_hints,
                tool_result_events=tool_result_events,
                retry_runner=retry_runner,
                run_kwargs=run_kwargs,
                label=f"{label}_{attempts}",
                previous_draft=final_content,
                continuation_reason=verdict.get("reason"),
                continuation_focus=verdict.get("next_focus"),
            )
            tool_result_events = self._collect_runtime_tool_result_events_from_memory(
                memory_start_index
            )
            after_fingerprint = AgentLoop._skill_contract_evidence_fingerprint(
                tool_result_events,
            )
            if after_fingerprint == before_fingerprint:
                logger.warning(
                    "Skill continuation produced no new tool evidence; "
                    "stopping bounded continuation."
                )
                break
        return final_content, tool_result_events

    def _get_runtime_memory_messages(self) -> list[Any]:
        """Return runtime memory messages exposed by the active inner agent."""
        if not hasattr(self._agent, "memory") or self._agent.memory is None:
            return []

        memory = self._agent.memory
        messages = getattr(memory, "messages", None)
        if isinstance(messages, list):
            return messages

        if hasattr(memory, "get_messages"):
            try:
                messages = memory.get_messages()
                if isinstance(messages, list):
                    return messages
            except Exception as exc:
                logger.debug(f"Failed to read runtime memory via get_messages(): {exc}")

        return []

    def _collect_runtime_tool_result_events_from_memory(
        self,
        start_index: int = 0,
    ) -> list[dict[str, Any]]:
        """Collect tool result events from runtime memory for non-stream flows."""
        messages = AgentLoop._get_runtime_memory_messages(self)
        if start_index < 0 or start_index > len(messages):
            start_index = 0
        tool_call_details = AgentLoop._tool_call_details_by_id(messages)

        events: list[dict[str, Any]] = []
        for msg in messages[start_index:]:
            if AgentLoop._stream_message_role(msg) != "tool":
                continue

            tool_call_id = AgentLoop._stream_message_attr(
                msg, "tool_call_id", ""
            ) or AgentLoop._stream_message_attr(msg, "id", "")
            result_payload = AgentLoop._stream_message_attr(msg, "text_content", None)
            if result_payload in (None, ""):
                result_payload = AgentLoop._stream_message_attr(msg, "content", "")
            serialized_result = AgentLoop._stringify_stream_payload(result_payload)
            tool_name = AgentLoop._stream_message_attr(msg, "name", "")
            tool_arguments = ""
            if tool_call_id:
                inferred_name, inferred_arguments = tool_call_details.get(tool_call_id, ("", ""))
                if inferred_name and not tool_name:
                    tool_name = inferred_name
                if inferred_arguments not in (None, ""):
                    tool_arguments = AgentLoop._tool_call_arguments_key(inferred_arguments)
            metadata: dict[str, Any] = {
                "name": tool_name,
                "result": serialized_result,
                "content": serialized_result,
                "output": serialized_result,
                "full_result": serialized_result,
                "full_content": serialized_result,
                "full_output": serialized_result,
                "model_result": serialized_result,
                "model_content": serialized_result,
                "model_output": serialized_result,
            }
            if tool_arguments:
                metadata["arguments"] = AgentLoop._tool_call_arguments_display(
                    tool_name,
                    tool_arguments,
                )
            if tool_call_id:
                metadata["id"] = tool_call_id
                metadata["tool_call_id"] = tool_call_id
            events.append(
                {
                    "type": "tool_result",
                    "delta": serialized_result,
                    "metadata": metadata,
                }
            )
        return events

    @staticmethod
    def _tool_events_have_repeated_read_guardrail(
        events: list[dict[str, Any]],
    ) -> bool:
        """Return True when tool evidence shows repeated read suppression."""
        for event in events:
            metadata = dict(event.get("metadata") or {})
            if str(metadata.get("name") or "").strip() != "read_file":
                continue
            payload = (
                metadata.get("model_result")
                or metadata.get("model_output")
                or metadata.get("model_content")
                or metadata.get("output")
                or metadata.get("result")
                or metadata.get("content")
                or event.get("delta")
            )
            text = AgentLoop._stringify_stream_payload(payload).lower()
            if (
                "repeated redundant read_file suppressed" in text
                or "requested file range was already provided" in text
                or "requested file range already available" in text
                or "already available in this request" in text
                or "file content already available" in text
            ):
                return True
        return False

    @staticmethod
    def _tool_events_have_history_search_budget(
        events: list[dict[str, Any]],
    ) -> bool:
        """Return True when history lookup budget is exhausted for this turn."""
        for event in events:
            metadata = dict(event.get("metadata") or {})
            if str(metadata.get("name") or "").strip() != "search_history":
                continue
            payload = (
                metadata.get("model_result")
                or metadata.get("model_output")
                or metadata.get("model_content")
                or metadata.get("output")
                or metadata.get("result")
                or metadata.get("content")
                or event.get("delta")
            )
            text = AgentLoop._stringify_stream_payload(payload).strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if isinstance(parsed, dict) and parsed.get("budget_exhausted") is True:
                return True
            if "history search budget reached for this request" in text.casefold():
                return True
        return False

    async def _run_process_repeated_read_recovery(
        self,
        *,
        authoritative_message: str,
        request_execution_hints: dict[str, Any],
        retry_runner: Callable[..., Awaitable[Any]],
        run_kwargs: dict[str, Any],
    ) -> str:
        """Retry once for non-stream flows when repeated read_file consumed the loop."""
        AgentLoop._drain_agent_output_queue(self)
        self._reset_agent_state_for_retry()
        repair_prompt = AgentLoop._build_repeated_read_recovery_prompt(
            authoritative_message,
        )
        await self._agent.add_message("user", repair_prompt)
        self._agent.next_step_prompt = repair_prompt

        previous_force_serial = bool(getattr(self, "_force_serial_tool_calls", False))
        self._force_serial_tool_calls = True
        try:
            with (
                bind_request_execution_hints(request_execution_hints),
                track_tool_invocations(max_repeats=1),
            ):
                result = await AgentLoop._run_agent_with_context_overflow_recovery(
                    self,
                    label="process_repeated_read_recovery",
                    retry_runner=retry_runner,
                    **run_kwargs,
                )
        finally:
            if previous_force_serial:
                self._force_serial_tool_calls = True
            elif hasattr(self, "_force_serial_tool_calls"):
                delattr(self, "_force_serial_tool_calls")

        return AgentLoop._extract_run_result_text(result)

    async def _run_process_history_search_budget_recovery(
        self,
        *,
        authoritative_message: str,
        request_execution_hints: dict[str, Any],
        tool_result_events: list[dict[str, Any]],
        retry_runner: Callable[..., Awaitable[Any]],
        run_kwargs: dict[str, Any],
        label: str,
    ) -> str:
        """Retry once after unproductive history lookup consumes the turn."""
        AgentLoop._drain_agent_output_queue(self)
        self._reset_agent_state_for_retry()
        repair_prompt = AgentLoop._build_history_search_budget_recovery_prompt(
            authoritative_message,
            tool_result_events,
        )
        await self._agent.add_message("user", repair_prompt)
        self._agent.next_step_prompt = repair_prompt

        recovery_hints = dict(request_execution_hints)
        recovery_hints["history_search_budget_exhausted"] = True
        previous_force_serial = bool(getattr(self, "_force_serial_tool_calls", False))
        self._force_serial_tool_calls = True
        try:
            with (
                bind_request_execution_hints(recovery_hints),
                track_tool_invocations(max_repeats=1),
            ):
                result = await AgentLoop._run_agent_with_context_overflow_recovery(
                    self,
                    label=label,
                    retry_runner=retry_runner,
                    **run_kwargs,
                )
        finally:
            if previous_force_serial:
                self._force_serial_tool_calls = True
            elif hasattr(self, "_force_serial_tool_calls"):
                delattr(self, "_force_serial_tool_calls")

        return AgentLoop._extract_run_result_text(result)

    @staticmethod
    def _build_skill_contract_continuation_prompt(
        authoritative_message: str,
        tool_result_events: list[dict[str, Any]],
        *,
        previous_draft: str | None = None,
        continuation_reason: str | None = None,
        continuation_focus: str | None = None,
    ) -> str:
        """Build a generic continuation prompt grounded in tool evidence."""
        lines = [
            "[SKILL CONTRACT CONTINUATION]",
            AgentLoop.DEFAULT_NEXT_STEP_PROMPT,
            "",
            "[LATEST USER REQUEST]",
            str(authoritative_message or "").strip(),
        ]
        if previous_draft and str(previous_draft).strip():
            lines.extend(
                [
                    "",
                    "[PREVIOUS ASSISTANT DRAFT OR PLAN]",
                    AgentLoop._compress_message_content(str(previous_draft).strip(), 1600),
                ]
            )
        if continuation_reason:
            lines.extend(["", "[CONTINUATION REASON]", str(continuation_reason).strip()])
        if continuation_focus:
            lines.extend(["", "[CONTINUATION FOCUS]", str(continuation_focus).strip()])

        evidence_lines = []
        for event in tool_result_events[-10:]:
            evidence_lines.extend(AgentLoop._continuation_tool_event_brief(event, limit=1200))
        if evidence_lines:
            lines.extend(["", "[RECENT TOOL EVIDENCE]", *evidence_lines])
        if request_is_plain_continuation_only(authoritative_message):
            lines.extend(
                [
                    "",
                    "[BOUNDED CONTINUATION LIMIT]",
                    "The latest user message adds no new count, target, or "
                    "scope. Continue only the same bounded unit already in "
                    "progress. Do not start another repeated/countable unit "
                    "from an older request; report current status or ask for "
                    "explicit scope after this unit is no longer actively "
                    "running.",
                ]
            )
        lines.append("")
        lines.append(
            "Continue from the evidence above. If the requested workflow is "
            "already terminal, answer now; otherwise call the next appropriate "
            "tool from the active skill contract."
        )
        return "\n".join(line for line in lines if line is not None)

    @staticmethod
    def _build_task_continuation_prompt(
        authoritative_message: str,
        tool_result_events: list[dict[str, Any]],
        *,
        previous_draft: str | None = None,
        continuation_reason: str | None = None,
        continuation_focus: str | None = None,
    ) -> str:
        """Build a generic continuation prompt from tool evidence."""
        lines = [
            "[TASK CONTINUATION]",
            AgentLoop.DEFAULT_NEXT_STEP_PROMPT,
            "",
            "[LATEST USER REQUEST]",
            str(authoritative_message or "").strip(),
        ]
        if previous_draft and str(previous_draft).strip():
            lines.extend(
                [
                    "",
                    "[PREVIOUS ASSISTANT DRAFT - NOT USER INPUT]",
                    AgentLoop._compress_message_content(str(previous_draft).strip(), 1600),
                ]
            )
        if continuation_reason:
            lines.extend(["", "[CONTINUATION REASON]", str(continuation_reason).strip()])
        if continuation_focus:
            lines.extend(["", "[CONTINUATION FOCUS]", str(continuation_focus).strip()])

        evidence_lines = []
        for event in tool_result_events[-10:]:
            evidence_lines.extend(AgentLoop._continuation_tool_event_brief(event, limit=1200))
        if evidence_lines:
            lines.extend(["", "[RECENT TOOL EVIDENCE]", *evidence_lines])
        plain_continuation = request_is_plain_continuation_only(authoritative_message)
        if plain_continuation:
            lines.extend(
                [
                    "",
                    "[BOUNDED CONTINUATION LIMIT]",
                    "The latest user message adds no new count, target, or "
                    "scope. Continue only the same bounded unit already in "
                    "progress. Do not start another repeated/countable unit "
                    "from an older request; report current status or ask for "
                    "explicit scope after this unit is no longer actively "
                    "running.",
                ]
            )
        lines.append("")
        lines.append(
            "Continue from the evidence above. If the requested outcome is "
            "already terminal or evidence shows a real blocker, answer now. "
            "Otherwise call the next appropriate tool. Do not repeat completed "
            "tool calls; use existing files and tool evidence as the current state."
        )
        lines.append(
            "Do not treat the previous assistant draft, any question in that "
            "draft, or this internal continuation prompt as user consent. If the "
            "previous draft is waiting for the user to choose, confirm, provide a "
            "missing value, or permit another side effect, stop with that waiting "
            "answer unless the latest user request already clearly authorized the "
            "exact next action."
        )
        if plain_continuation:
            lines.append(
                "Do not inherit a count or repeated-action scope from the prior "
                "request when the newest request is only a continuation "
                "acknowledgement. The prior request is an anchor for state, not "
                "permission to perform multiple new side-effect units."
            )
        else:
            lines.append(
                "For countable or repeated requests, continue until the concrete tool "
                "evidence satisfies the requested count and scope, or until tool "
                "evidence shows a real blocker. Treat stage summaries as progress "
                "for the stage they describe, not as proof of the remaining stages."
            )
        lines.append(
            "If the request asks to create, update, build, deploy, start, or verify "
            "a workspace artifact and the evidence so far is only read/list/search "
            "output, the next step must be a mutation or execution tool such as "
            "write_file, edit_file, shell, service_expose, or an explicit blocker "
            "grounded in tool evidence."
        )
        return "\n".join(line for line in lines if line is not None)

    async def _run_process_skill_contract_continuation(
        self,
        *,
        authoritative_message: str,
        request_execution_hints: dict[str, Any],
        tool_result_events: list[dict[str, Any]],
        retry_runner: Callable[..., Awaitable[Any]],
        run_kwargs: dict[str, Any],
        label: str,
        previous_draft: str | None = None,
        continuation_reason: str | None = None,
        continuation_focus: str | None = None,
    ) -> str:
        """Retry once when a skill turn stops after contract setup."""
        AgentLoop._drain_agent_output_queue(self)
        self._reset_agent_state_for_retry()
        repair_prompt = AgentLoop._build_skill_contract_continuation_prompt(
            authoritative_message,
            tool_result_events,
            previous_draft=previous_draft,
            continuation_reason=continuation_reason,
            continuation_focus=continuation_focus,
        )
        await self._agent.add_message("user", repair_prompt)
        self._agent.next_step_prompt = repair_prompt

        previous_force_serial = bool(getattr(self, "_force_serial_tool_calls", False))
        self._force_serial_tool_calls = True
        try:
            with (
                bind_request_execution_hints(request_execution_hints),
                track_tool_invocations(max_repeats=1),
            ):
                result = await AgentLoop._run_agent_with_context_overflow_recovery(
                    self,
                    label=label,
                    retry_runner=retry_runner,
                    **run_kwargs,
                )
        finally:
            if previous_force_serial:
                self._force_serial_tool_calls = True
            elif hasattr(self, "_force_serial_tool_calls"):
                delattr(self, "_force_serial_tool_calls")
        return AgentLoop._extract_run_result_text(result)

    async def _run_process_task_continuation(
        self,
        *,
        authoritative_message: str,
        request_execution_hints: dict[str, Any],
        tool_result_events: list[dict[str, Any]],
        retry_runner: Callable[..., Awaitable[Any]],
        run_kwargs: dict[str, Any],
        label: str,
        previous_draft: str | None = None,
        continuation_reason: str | None = None,
        continuation_focus: str | None = None,
    ) -> str:
        """Retry once when a tool-backed turn stopped before requested outcome."""
        AgentLoop._drain_agent_output_queue(self)
        self._reset_agent_state_for_retry()
        repair_prompt = AgentLoop._build_task_continuation_prompt(
            authoritative_message,
            tool_result_events,
            previous_draft=previous_draft,
            continuation_reason=continuation_reason,
            continuation_focus=continuation_focus,
        )
        await self._agent.add_message("user", repair_prompt)
        self._agent.next_step_prompt = repair_prompt

        previous_force_serial = bool(getattr(self, "_force_serial_tool_calls", False))
        self._force_serial_tool_calls = True
        try:
            with (
                bind_request_execution_hints(request_execution_hints),
                track_tool_invocations(max_repeats=1),
            ):
                result = await AgentLoop._run_agent_with_context_overflow_recovery(
                    self,
                    label=label,
                    retry_runner=retry_runner,
                    **run_kwargs,
                )
        finally:
            if previous_force_serial:
                self._force_serial_tool_calls = True
            elif hasattr(self, "_force_serial_tool_calls"):
                delattr(self, "_force_serial_tool_calls")
        return AgentLoop._extract_run_result_text(result)

    def _collect_stream_tool_result_events(
        self,
        start_index: int,
        emitted_tool_result_ids: set[str],
        *,
        tool_output_capture_scope: str | None,
        tool_call_arguments_by_id: dict[str, str],
    ) -> tuple[list[dict[str, Any]], int]:
        """Collect newly-added tool result messages from runtime memory."""
        messages = AgentLoop._get_runtime_memory_messages(self)
        if start_index < 0 or start_index > len(messages):
            start_index = 0

        events: list[dict[str, Any]] = []
        next_index = start_index
        tool_call_details = AgentLoop._tool_call_details_by_id(messages)
        for index, msg in enumerate(messages[start_index:], start=start_index):
            if AgentLoop._stream_message_role(msg) != "tool":
                next_index = index + 1
                continue

            tool_call_id = AgentLoop._stream_message_attr(
                msg, "tool_call_id", ""
            ) or AgentLoop._stream_message_attr(msg, "id", "")
            if tool_call_id and tool_call_id not in tool_call_arguments_by_id:
                inferred_name, inferred_arguments = tool_call_details.get(tool_call_id, ("", ""))
                if inferred_arguments not in (None, ""):
                    tool_call_arguments_by_id[tool_call_id] = AgentLoop._tool_call_arguments_key(
                        inferred_arguments
                    )
            if tool_call_id and tool_call_id in emitted_tool_result_ids:
                next_index = index + 1
                continue

            result_payload = AgentLoop._stream_message_attr(msg, "text_content", None)
            if result_payload in (None, ""):
                result_payload = AgentLoop._stream_message_attr(msg, "content", "")
            serialized_result = AgentLoop._stringify_stream_payload(result_payload)
            tool_name = AgentLoop._stream_message_attr(msg, "name", "")
            if tool_call_id and not tool_name:
                inferred_name, _ = tool_call_details.get(tool_call_id, ("", ""))
                if inferred_name:
                    tool_name = inferred_name
            captured_output = consume_captured_tool_output(
                tool_output_capture_scope,
                tool_name=tool_name,
                arguments=tool_call_arguments_by_id.get(tool_call_id, ""),
            )

            metadata: dict[str, Any] = {"name": tool_name}
            arguments = tool_call_arguments_by_id.get(tool_call_id, "")
            if arguments:
                metadata["arguments"] = AgentLoop._tool_call_arguments_display(
                    tool_name,
                    arguments,
                )
            if tool_call_id:
                metadata["id"] = tool_call_id
                metadata["tool_call_id"] = tool_call_id
                emitted_tool_result_ids.add(tool_call_id)
            metadata = AgentLoop._merge_stream_tool_result_metadata(
                metadata,
                streamed_result=serialized_result,
                captured_output=captured_output,
            )
            self._remember_stream_tool_result_metadata(tool_call_id, metadata)
            visible_delta = AgentLoop._stream_tool_result_visible_delta("", metadata)

            events.append(
                {
                    "type": "tool_result",
                    "delta": visible_delta,
                    "metadata": metadata,
                }
            )
            next_index = index + 1

        return events, next_index
