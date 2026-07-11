from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol

from openai import OpenAI

from agent_app.config import Settings
from agent_app.llm_profiles import preset_for
from agent_app.models import Plan

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an AI agent planner.
Return strict JSON only with this schema:
{
  "mode": "respond" | "tool",
  "message": "string",
  "tool_name": "list_files" | "search_web" | "move_file" | null,
  "tool_args": { ... },
  "requires_confirmation": true | false
}

Rules:
- Use mode=respond for normal chat.
- Use tool=list_files when user asks to inspect local files.
- Use tool=search_web when user asks for web information.
- Use tool=move_file only when user asks to move/organize files.
- move_file must set requires_confirmation=true.
- Keep tool_args minimal and valid.
"""


class Planner(Protocol):
    def plan(self, user_text: str, history: list[dict[str, str]] | None = None) -> Plan: ...

    def update_settings(self, settings: Settings) -> None: ...


def _extract_json(text: str) -> str:
    """Local models often wrap JSON in markdown fences."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _parse_plan(raw: str) -> Plan:
    try:
        data = json.loads(_extract_json(raw))
        return Plan(
            mode=data.get("mode", "respond"),
            message=data.get("message", ""),
            tool_name=data.get("tool_name"),
            tool_args=data.get("tool_args") or {},
            requires_confirmation=bool(data.get("requires_confirmation", False)),
        )
    except json.JSONDecodeError:
        return Plan(
            mode="respond",
            message="我暂时没能解析任务计划。你可以换种说法，或先让我执行一个简单动作（如列出文件）。",
        )


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


class OpenAICompatPlanner:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = self._build_client(settings)

    def _build_client(self, settings: Settings) -> OpenAI | None:
        # Local servers (Ollama/LM Studio) accept any non-empty api_key.
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

    def plan(self, user_text: str, history: list[dict[str, str]] | None = None) -> Plan:
        if not self._client:
            return _rule_based_plan(user_text)

        try:
            messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": user_text})

            response = self._client.chat.completions.create(
                model=self._settings.llm_model,
                messages=messages,
                temperature=0,
            )
        except Exception as exc:
            label = _provider_label(self._settings)
            logger.exception("LLM call failed provider=%s", label)
            hint = (
                f"请确认 {label} 服务可用，且 API Key / base_url 配置正确。"
                if not self._settings.llm_base_url
                else f"请确认本地服务已启动（{self._settings.llm_base_url}）。"
            )
            return Plan(mode="respond", message=f"{label} 调用失败。\n{hint}")

        raw = (response.choices[0].message.content or "").strip()
        return _parse_plan(raw)


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

    def plan(self, user_text: str, history: list[dict[str, str]] | None = None) -> Plan:
        if not self._client:
            return _rule_based_plan(user_text)

        try:
            messages: list[dict[str, str]] = []
            if history:
                # Anthropic requires messages to start with role="user";
                # drop any leading assistant messages from trimmed history
                start = next(
                    (i for i, m in enumerate(history) if m["role"] == "user"),
                    len(history),
                )
                messages.extend(history[start:])
            messages.append({"role": "user", "content": user_text})

            response = self._client.messages.create(
                model=self._settings.llm_model or "claude-sonnet-4-20250514",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
        except Exception:
            logger.exception("Claude call failed")
            return Plan(
                mode="respond",
                message="Claude 调用失败。\n请确认 Anthropic API Key 与模型名称正确。",
            )

        parts = []
        for block in response.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        raw = "".join(parts).strip()
        return _parse_plan(raw)


def build_planner(settings: Settings) -> Planner:
    if settings.llm_provider_id == "claude":
        return AnthropicPlanner(settings)
    return OpenAICompatPlanner(settings)


class LLMPlanner:
    """Backward-compatible wrapper."""

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
