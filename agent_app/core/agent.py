from __future__ import annotations

import logging
from collections.abc import Generator
from dataclasses import asdict
from typing import Any
from uuid import uuid4

from agent_app.config import Settings
from agent_app.core.llm import LLMPlanner
from agent_app.llm_profiles import LlmProfile, resolve_profile_llm
from agent_app.models import AgentReply, Plan, ProposedAction, StreamChunk, StreamResult
from agent_app.tools.file_tools import list_files, move_file
from agent_app.tools.web_tools import search_web

logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.current_profile: LlmProfile | None = None
        self.planner = LLMPlanner(settings)
        self.pending_actions: dict[str, ProposedAction] = {}

    def apply_llm_profile(self, profile: LlmProfile) -> str:
        self.current_profile = profile
        settings, message = resolve_profile_llm(profile, self.settings)
        self.settings = settings
        self.planner.update_settings(settings)
        logger.info(
            "Applied LLM profile provider=%s model=%s",
            profile.provider_id,
            settings.llm_model,
        )
        return message

    def handle_user_message(
        self, user_text: str, history: list[dict[str, str]] | None = None
    ) -> AgentReply:
        """Non-streaming entry point (kept for backward compatibility)."""
        logger.info("User message received (%d chars)", len(user_text))
        plan = self.planner.plan(user_text, history)
        return self._execute_plan(plan)

    def handle_user_message_stream(
        self, user_text: str, history: list[dict[str, str]] | None = None
    ) -> Generator[StreamChunk | AgentReply, None, None]:
        """Streaming entry point: yields StreamChunks during generation, then a final AgentReply."""
        logger.info("User message received (stream, %d chars)", len(user_text))

        for item in self.planner.plan_stream(user_text, history):
            if isinstance(item, StreamChunk):
                yield item
            elif isinstance(item, StreamResult):
                plan = item.plan

                if plan.mode == "tool" and plan.tool_name:
                    yield StreamChunk(
                        text=f"\n[调用 {plan.tool_name}...]\n",
                        chunk_type="tool_status",
                    )
                    logger.debug("Executing tool=%s after stream", plan.tool_name)

                reply = self._execute_plan(plan)

                if plan.mode == "tool" and plan.tool_name and not reply.pending_action:
                    yield StreamChunk(
                        text="[工具执行完成]\n",
                        chunk_type="tool_status",
                    )

                logger.debug(
                    "yield final AgentReply, message len=%d", len(reply.message)
                )
                yield reply

    def approve_action(self, action_id: str) -> AgentReply:
        action = self.pending_actions.pop(action_id, None)
        if not action:
            return AgentReply("没有找到待确认动作，可能已经过期。")

        result = self._run_tool(action.tool_name, action.args)
        return AgentReply(f"已执行确认动作。\n{result}", tool_name=action.tool_name)

    def reject_action(self, action_id: str) -> AgentReply:
        action = self.pending_actions.pop(action_id, None)
        if not action:
            return AgentReply("没有找到待确认动作，可能已经过期。")
        return AgentReply(f"已取消动作: {action.description}")

    def _execute_plan(self, plan: Plan) -> AgentReply:
        if plan.mode == "respond":
            return AgentReply(plan.message or "好的。")

        if plan.mode == "tool" and plan.tool_name:
            if plan.requires_confirmation:
                action = ProposedAction(
                    action_id=str(uuid4()),
                    description=plan.message or f"执行工具 {plan.tool_name}",
                    tool_name=plan.tool_name,
                    args=plan.tool_args,
                )
                self.pending_actions[action.action_id] = action
                details = f"{action.description}\n工具: {action.tool_name}\n参数: {asdict(action)['args']}"
                return AgentReply(
                    message=f"该操作需要确认：\n{details}",
                    pending_action=action,
                )

            result = self._run_tool(plan.tool_name, plan.tool_args)
            return AgentReply(result, tool_name=plan.tool_name)

        return AgentReply("我没能识别到可执行动作，请换一种描述试试。")

    def _run_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        logger.info("Running tool=%s args=%s", tool_name, args)
        if tool_name == "list_files":
            path = str(args.get("path", "."))
            max_items = int(args.get("max_items", 50))
            return list_files(
                allowed_root=self.settings.allowed_root, path=path, max_items=max_items
            )

        if tool_name == "move_file":
            src = str(args.get("src", ""))
            dst = str(args.get("dst", ""))
            return move_file(allowed_root=self.settings.allowed_root, src=src, dst=dst)

        if tool_name == "search_web":
            query = str(args.get("query", "")).strip()
            max_results = int(args.get("max_results", 5))
            return search_web(
                tavily_api_key=self.settings.tavily_api_key,
                query=query,
                max_results=max_results,
            )

        logger.warning("Unknown tool requested: %s", tool_name)
        return f"未知工具: {tool_name}"
