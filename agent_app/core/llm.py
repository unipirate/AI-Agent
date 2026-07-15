from __future__ import annotations

import json
import logging
from collections.abc import Generator
from typing import Any, Protocol

from openai import OpenAI

from agent_app.config import Settings
from agent_app.llm_profiles import preset_for
from agent_app.models import Plan, StreamChunk, StreamResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt (natural conversational style, no JSON instruction)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a helpful AI assistant with access to tools.
Use tools when the user asks to perform actions (inspect files, search the web, move files).
For normal conversation, reply directly in the user's language.
When using move_file, always ask for confirmation first.
Keep responses concise and helpful.
"""

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function calling format)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories at a given path within the sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to list. Defaults to current directory.",
                        "default": ".",
                    },
                    "max_items": {
                        "type": "integer",
                        "description": "Maximum number of items to return.",
                        "default": 50,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information using a query string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return.",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_file",
            "description": "Move or rename a file within the sandbox. Requires user confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "src": {
                        "type": "string",
                        "description": "Source file path (relative to sandbox root).",
                    },
                    "dst": {
                        "type": "string",
                        "description": "Destination file path (relative to sandbox root).",
                    },
                },
                "required": ["src", "dst"],
            },
        },
    },
]

# Anthropic uses a slightly different tool schema format
ANTHROPIC_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "list_files",
        "description": "List files and directories at a given path within the sandbox.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to list. Defaults to current directory.",
                    "default": ".",
                },
                "max_items": {
                    "type": "integer",
                    "description": "Maximum number of items to return.",
                    "default": 50,
                },
            },
        },
    },
    {
        "name": "search_web",
        "description": "Search the web for information using a query string.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "move_file",
        "description": "Move or rename a file within the sandbox. Requires user confirmation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "src": {
                    "type": "string",
                    "description": "Source file path (relative to sandbox root).",
                },
                "dst": {
                    "type": "string",
                    "description": "Destination file path (relative to sandbox root).",
                },
            },
            "required": ["src", "dst"],
        },
    },
]

CONFIRMATION_TOOLS = frozenset({"move_file"})

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class Planner(Protocol):
    def plan(self, user_text: str, history: list[dict[str, str]] | None = None) -> Plan: ...

    def plan_stream(
        self, user_text: str, history: list[dict[str, str]] | None = None
    ) -> Generator[StreamChunk | StreamResult, None, None]: ...

    def update_settings(self, settings: Settings) -> None: ...


# ---------------------------------------------------------------------------
# Fallback rule-based planner (no LLM)
# ---------------------------------------------------------------------------


def _rule_based_plan(user_text: str) -> Plan:
    text = user_text.lower()
    if any(k in text for k in ("search", "网页", "web", "查一下", "新闻")):
        return Plan(mode="tool", tool_name="search_web", tool_args={"query": user_text})
    if any(k in text for k in ("list", "文件", "文档", "目录", "folder")):
        return Plan(mode="tool", tool_name="list_files", tool_args={"path": "."})
    if any(k in text for k in ("move", "移动", "整理到")):
        return Plan(
            mode="tool",
            tool_name="move_file",
            tool_args={"src": "", "dst": ""},
            requires_confirmation=True,
            message="我可以帮你移动文件，但需要你给出 src 和 dst 路径。",
        )
    return Plan(mode="respond", message="我在这。你可以让我列出文件，或者帮你搜索网页信息。")


def _provider_label(settings: Settings) -> str:
    if settings.llm_provider_id:
        return preset_for(settings.llm_provider_id).display_name
    if settings.llm_base_url:
        return "本地 LLM"
    if settings.llm_api_key:
        return "云端 LLM"
    return "LLM"


# ---------------------------------------------------------------------------
# OpenAI-compatible Planner (streaming + function calling)
# ---------------------------------------------------------------------------


class OpenAICompatPlanner:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = self._build_client(settings)

    def _build_client(self, settings: Settings) -> OpenAI | None:
        if settings.llm_base_url:
            return OpenAI(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key or "local",
            )
        if settings.llm_api_key:
            return OpenAI(api_key=settings.llm_api_key)
        return None

    def update_settings(self, settings: Settings) -> None:
        self._settings = settings
        self._client = self._build_client(settings)

    def plan_stream(
        self, user_text: str, history: list[dict[str, str]] | None = None
    ) -> Generator[StreamChunk | StreamResult, None, None]:
        if not self._client:
            result_plan = _rule_based_plan(user_text)
            yield StreamResult(plan=result_plan, full_text=result_plan.message)
            return

        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_text})

        try:
            response = self._client.chat.completions.create(
                model=self._settings.llm_model,
                messages=messages,  # type: ignore[arg-type]
                tools=TOOL_DEFINITIONS,  # type: ignore[arg-type]
                temperature=0,
                stream=True,
            )
        except Exception as exc:
            exc_msg = str(exc).lower()
            # Local models may not support function calling — retry without tools
            if self._settings.llm_base_url and (
                "tool" in exc_msg or "function" in exc_msg or "not supported" in exc_msg
            ):
                logger.warning("Local model does not support tools, falling back to plain stream")
                yield from self._plan_stream_no_tools(messages)
                return

            label = _provider_label(self._settings)
            logger.exception("LLM streaming call failed provider=%s", label)
            hint = (
                f"请确认 {label} 服务可用，且 API Key / base_url 配置正确。"
                if not self._settings.llm_base_url
                else f"请确认本地服务已启动（{self._settings.llm_base_url}）。"
            )
            error_msg = f"{label} 调用失败。\n{hint}"
            yield StreamResult(
                plan=Plan(mode="respond", message=error_msg),
                full_text=error_msg,
            )
            return

        full_text = ""
        tool_call_buffers: dict[int, dict[str, str]] = {}

        try:
            for chunk in response:
                choice = chunk.choices[0] if chunk.choices else None  # type: ignore[union-attr]
                if not choice:
                    continue

                delta = choice.delta

                if delta.content:
                    full_text += delta.content
                    logger.debug("stream text chunk len=%d", len(delta.content))
                    yield StreamChunk(text=delta.content)

                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        buf = tool_call_buffers.setdefault(idx, {"name": "", "arguments": ""})
                        if tc_delta.function and tc_delta.function.name:
                            buf["name"] += tc_delta.function.name
                        if tc_delta.function and tc_delta.function.arguments:
                            buf["arguments"] += tc_delta.function.arguments

                if choice.finish_reason:
                    logger.debug("stream finish_reason=%s", choice.finish_reason)
                    break
        except Exception as exc:
            logger.exception("Error during OpenAI stream iteration")
            error_msg = f"流式响应中断: {exc}"
            yield StreamResult(
                plan=Plan(mode="respond", message=error_msg),
                full_text=full_text or error_msg,
            )
            return

        if tool_call_buffers:
            buf = next(iter(tool_call_buffers.values()))
            tool_name = buf["name"]
            try:
                tool_args = json.loads(buf["arguments"]) if buf["arguments"] else {}
            except json.JSONDecodeError:
                logger.error("Failed to parse tool arguments: %s", buf["arguments"])
                error_msg = "工具参数解析失败，请重试。"
                yield StreamResult(
                    plan=Plan(mode="respond", message=error_msg),
                    full_text=error_msg,
                )
                return

            requires_confirmation = tool_name in CONFIRMATION_TOOLS
            plan = Plan(
                mode="tool",
                tool_name=tool_name,
                tool_args=tool_args,
                requires_confirmation=requires_confirmation,
                message=full_text,
            )
            yield StreamResult(plan=plan, full_text=full_text)
        else:
            plan = Plan(mode="respond", message=full_text)
            yield StreamResult(plan=plan, full_text=full_text)

    def plan(self, user_text: str, history: list[dict[str, str]] | None = None) -> Plan:
        """Non-streaming fallback implemented by consuming plan_stream()."""
        result: Plan | None = None
        for item in self.plan_stream(user_text, history):
            if isinstance(item, StreamResult):
                result = item.plan
        if result is None:
            return Plan(mode="respond", message="流式生成未返回结果。")
        return result

    def _plan_stream_no_tools(
        self, messages: list[dict[str, Any]]
    ) -> Generator[StreamChunk | StreamResult, None, None]:
        """Fallback streaming without tools for local models that don't support function calling."""
        try:
            response = self._client.chat.completions.create(  # type: ignore[union-attr]
                model=self._settings.llm_model,
                messages=messages,  # type: ignore[arg-type]
                temperature=0,
                stream=True,
            )
        except Exception as exc:
            label = _provider_label(self._settings)
            logger.exception("Fallback (no tools) stream failed provider=%s", label)
            error_msg = f"{label} 调用失败（无工具模式）。"
            yield StreamResult(plan=Plan(mode="respond", message=error_msg), full_text=error_msg)
            return

        full_text = ""
        try:
            for chunk in response:
                choice = chunk.choices[0] if chunk.choices else None  # type: ignore[union-attr]
                if not choice:
                    continue
                delta = choice.delta
                if delta.content:
                    full_text += delta.content
                    yield StreamChunk(text=delta.content)
                if choice.finish_reason:
                    break
        except Exception:
            logger.exception("Error during no-tools stream iteration")

        plan = Plan(mode="respond", message=full_text)
        yield StreamResult(plan=plan, full_text=full_text)


# ---------------------------------------------------------------------------
# Anthropic Planner (streaming + tool_use)
# ---------------------------------------------------------------------------


class AnthropicPlanner:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = self._build_client(settings)

    def _build_client(self, settings: Settings) -> Any | None:
        if not settings.llm_api_key:
            return None
        import anthropic

        return anthropic.Anthropic(api_key=settings.llm_api_key)

    def update_settings(self, settings: Settings) -> None:
        self._settings = settings
        self._client = self._build_client(settings)

    def plan_stream(
        self, user_text: str, history: list[dict[str, str]] | None = None
    ) -> Generator[StreamChunk | StreamResult, None, None]:
        if not self._client:
            result_plan = _rule_based_plan(user_text)
            yield StreamResult(plan=result_plan, full_text=result_plan.message)
            return

        messages: list[dict[str, str]] = []
        if history:
            start = next(
                (i for i, m in enumerate(history) if m["role"] == "user"),
                len(history),
            )
            messages.extend(history[start:])
        messages.append({"role": "user", "content": user_text})

        try:
            stream_ctx = self._client.messages.stream(
                model=self._settings.llm_model or "claude-sonnet-4-20250514",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=messages,
                tools=ANTHROPIC_TOOL_DEFINITIONS,
            )
        except Exception:
            logger.exception("Claude streaming call failed")
            error_msg = "Claude 调用失败。\n请确认 Anthropic API Key 与模型名称正确。"
            yield StreamResult(
                plan=Plan(mode="respond", message=error_msg),
                full_text=error_msg,
            )
            return

        full_text = ""
        current_tool_name = ""
        tool_input_json = ""
        in_tool_use_block = False
        final_plan: Plan | None = None

        try:
            with stream_ctx as stream:
                for event in stream:
                    event_type = getattr(event, "type", "")

                    if event_type == "content_block_start":
                        block = getattr(event, "content_block", None)
                        if block and getattr(block, "type", "") == "tool_use":
                            in_tool_use_block = True
                            current_tool_name = getattr(block, "name", "")
                            tool_input_json = ""
                            logger.debug("Anthropic tool_use block start: %s", current_tool_name)

                    elif event_type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        if not delta:
                            continue
                        delta_type = getattr(delta, "type", "")

                        if delta_type == "text_delta":
                            text = getattr(delta, "text", "")
                            if text:
                                full_text += text
                                logger.debug("Anthropic text delta len=%d", len(text))
                                yield StreamChunk(text=text)

                        elif delta_type == "input_json_delta":
                            partial = getattr(delta, "partial_json", "")
                            if partial:
                                tool_input_json += partial

                    elif event_type == "content_block_stop":
                        if in_tool_use_block:
                            try:
                                tool_args = json.loads(tool_input_json) if tool_input_json else {}
                            except json.JSONDecodeError:
                                logger.error(
                                    "Failed to parse Anthropic tool args: %s",
                                    tool_input_json,
                                )
                                tool_args = {}
                            requires_confirmation = current_tool_name in CONFIRMATION_TOOLS
                            final_plan = Plan(
                                mode="tool",
                                tool_name=current_tool_name,
                                tool_args=tool_args,
                                requires_confirmation=requires_confirmation,
                                message=full_text,
                            )
                            in_tool_use_block = False

                    elif event_type == "message_stop":
                        logger.debug("Anthropic message_stop")
                        break

        except Exception as exc:
            logger.exception("Error during Anthropic stream iteration")
            error_msg = f"Claude 流式响应中断: {exc}"
            yield StreamResult(
                plan=Plan(mode="respond", message=error_msg),
                full_text=full_text or error_msg,
            )
            return

        if final_plan:
            yield StreamResult(plan=final_plan, full_text=full_text)
        else:
            plan = Plan(mode="respond", message=full_text)
            yield StreamResult(plan=plan, full_text=full_text)

    def plan(self, user_text: str, history: list[dict[str, str]] | None = None) -> Plan:
        """Non-streaming fallback implemented by consuming plan_stream()."""
        result: Plan | None = None
        for item in self.plan_stream(user_text, history):
            if isinstance(item, StreamResult):
                result = item.plan
        if result is None:
            return Plan(mode="respond", message="流式生成未返回结果。")
        return result


# ---------------------------------------------------------------------------
# Factory + backward-compatible wrapper
# ---------------------------------------------------------------------------


def build_planner(settings: Settings) -> Planner:
    if settings.llm_provider_id == "claude":
        return AnthropicPlanner(settings)
    return OpenAICompatPlanner(settings)


class LLMPlanner:
    """Backward-compatible wrapper that delegates to the appropriate planner."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._delegate = build_planner(settings)

    def update_settings(self, settings: Settings) -> None:
        provider_changed = settings.llm_provider_id != self._settings.llm_provider_id
        self._settings = settings
        if provider_changed:
            self._delegate = build_planner(settings)
        else:
            self._delegate.update_settings(settings)

    def plan(self, user_text: str, history: list[dict[str, str]] | None = None) -> Plan:
        return self._delegate.plan(user_text, history)

    def plan_stream(
        self, user_text: str, history: list[dict[str, str]] | None = None
    ) -> Generator[StreamChunk | StreamResult, None, None]:
        yield from self._delegate.plan_stream(user_text, history)
