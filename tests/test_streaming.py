"""Integration tests for the streaming pipeline.

Tests the full flow: mock LLM -> plan_stream -> Agent -> StreamChunk/AgentReply
without network access or real API keys.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_app.config import Settings
from agent_app.core.agent import Agent
from agent_app.core.llm import OpenAICompatPlanner
from agent_app.models import AgentReply, Plan, StreamChunk, StreamResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def settings() -> Settings:
    return Settings(
        llm_base_url="http://localhost:8080/v1",
        llm_api_key="test-key",
        llm_model="test-model",
        llm_provider_id="",
        tavily_api_key="",
        allowed_root=Path("/tmp/test-sandbox"),
    )


@pytest.fixture
def agent(settings: Settings) -> Agent:
    return Agent(settings)


# ---------------------------------------------------------------------------
# Mock streaming responses
# ---------------------------------------------------------------------------


def _make_text_stream_chunks(text: str, chunk_size: int = 5):
    """Simulate OpenAI streaming text response as a list of mock chunks."""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk_text = text[i : i + chunk_size]
        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = chunk_text
        mock_chunk.choices[0].delta.tool_calls = None
        mock_chunk.choices[0].finish_reason = None
        chunks.append(mock_chunk)

    # Final chunk with finish_reason
    final = MagicMock()
    final.choices = [MagicMock()]
    final.choices[0].delta.content = None
    final.choices[0].delta.tool_calls = None
    final.choices[0].finish_reason = "stop"
    chunks.append(final)
    return chunks


def _make_tool_call_stream_chunks(tool_name: str, arguments_json: str):
    """Simulate OpenAI streaming tool call response."""
    chunks = []

    # First chunk: tool call name
    c1 = MagicMock()
    c1.choices = [MagicMock()]
    c1.choices[0].delta.content = None
    tc_delta1 = MagicMock()
    tc_delta1.index = 0
    tc_delta1.function.name = tool_name
    tc_delta1.function.arguments = ""
    c1.choices[0].delta.tool_calls = [tc_delta1]
    c1.choices[0].finish_reason = None
    chunks.append(c1)

    # Split arguments into chunks
    mid = len(arguments_json) // 2
    for part in [arguments_json[:mid], arguments_json[mid:]]:
        c = MagicMock()
        c.choices = [MagicMock()]
        c.choices[0].delta.content = None
        tc_delta = MagicMock()
        tc_delta.index = 0
        tc_delta.function.name = None
        tc_delta.function.arguments = part
        c.choices[0].delta.tool_calls = [tc_delta]
        c.choices[0].finish_reason = None
        chunks.append(c)

    # Final chunk
    final = MagicMock()
    final.choices = [MagicMock()]
    final.choices[0].delta.content = None
    final.choices[0].delta.tool_calls = None
    final.choices[0].finish_reason = "tool_calls"
    chunks.append(final)
    return chunks


# ---------------------------------------------------------------------------
# Tests: Planner layer
# ---------------------------------------------------------------------------


class TestOpenAICompatPlannerStream:
    def test_text_stream(self, settings: Settings) -> None:
        """Pure text response streams chunks then yields StreamResult."""
        planner = OpenAICompatPlanner(settings)
        text = "你好，我可以帮你做很多事情。"
        mock_chunks = _make_text_stream_chunks(text)

        with patch.object(planner._client, "chat") as mock_chat:
            mock_chat.completions.create.return_value = iter(mock_chunks)

            items = list(planner.plan_stream("hello"))

        stream_chunks = [i for i in items if isinstance(i, StreamChunk)]
        stream_results = [i for i in items if isinstance(i, StreamResult)]

        assert len(stream_chunks) > 0
        assert len(stream_results) == 1

        reconstructed = "".join(c.text for c in stream_chunks)
        assert reconstructed == text

        result = stream_results[0]
        assert result.plan.mode == "respond"
        assert result.plan.message == text
        assert result.full_text == text

    def test_tool_call_stream(self, settings: Settings) -> None:
        """Tool call response accumulates arguments correctly."""
        planner = OpenAICompatPlanner(settings)
        args_json = '{"query": "latest news"}'
        mock_chunks = _make_tool_call_stream_chunks("search_web", args_json)

        with patch.object(planner._client, "chat") as mock_chat:
            mock_chat.completions.create.return_value = iter(mock_chunks)

            items = list(planner.plan_stream("search latest news"))

        stream_results = [i for i in items if isinstance(i, StreamResult)]
        assert len(stream_results) == 1

        result = stream_results[0]
        assert result.plan.mode == "tool"
        assert result.plan.tool_name == "search_web"
        assert result.plan.tool_args == {"query": "latest news"}

    def test_no_client_fallback(self) -> None:
        """Without a client, falls back to rule-based plan."""
        settings = Settings(
            llm_base_url="",
            llm_api_key="",
            llm_model="",
            llm_provider_id="",
            tavily_api_key="",
            allowed_root=Path("/tmp"),
        )
        planner = OpenAICompatPlanner(settings)

        items = list(planner.plan_stream("list files"))
        stream_results = [i for i in items if isinstance(i, StreamResult)]
        assert len(stream_results) == 1
        assert stream_results[0].plan.mode == "tool"
        assert stream_results[0].plan.tool_name == "list_files"


# ---------------------------------------------------------------------------
# Tests: Agent layer
# ---------------------------------------------------------------------------


class TestAgentStream:
    def test_text_reply_stream(self, agent: Agent) -> None:
        """Agent yields StreamChunks then a final AgentReply for text responses."""
        text = "Hello, how can I help?"

        def mock_plan_stream(user_text, history=None):
            for char in text:
                yield StreamChunk(text=char)
            yield StreamResult(plan=Plan(mode="respond", message=text), full_text=text)

        with patch.object(agent.planner, "plan_stream", side_effect=mock_plan_stream):
            items = list(agent.handle_user_message_stream("hi"))

        chunks = [i for i in items if isinstance(i, StreamChunk)]
        replies = [i for i in items if isinstance(i, AgentReply)]

        assert len(chunks) == len(text)
        assert len(replies) == 1
        assert replies[0].message == text
        assert replies[0].tool_name is None

    def test_tool_call_stream(self, agent: Agent) -> None:
        """Agent yields tool status chunks and final AgentReply with tool_name."""

        def mock_plan_stream(user_text, history=None):
            yield StreamResult(
                plan=Plan(mode="tool", tool_name="list_files", tool_args={"path": "."}),
                full_text="",
            )

        with patch.object(agent.planner, "plan_stream", side_effect=mock_plan_stream):
            with patch.object(agent, "_run_tool", return_value="file1.txt\nfile2.txt"):
                items = list(agent.handle_user_message_stream("list files"))

        status_chunks = [i for i in items if isinstance(i, StreamChunk) and i.chunk_type == "tool_status"]
        replies = [i for i in items if isinstance(i, AgentReply)]

        assert len(status_chunks) == 2  # tool_call_start + tool_call_end
        assert len(replies) == 1
        assert replies[0].tool_name == "list_files"
        assert "file1.txt" in replies[0].message

    def test_confirmation_required(self, agent: Agent) -> None:
        """When tool requires confirmation, no tool_call_end status is emitted."""

        def mock_plan_stream(user_text, history=None):
            yield StreamResult(
                plan=Plan(
                    mode="tool",
                    tool_name="move_file",
                    tool_args={"src": "a.txt", "dst": "b.txt"},
                    requires_confirmation=True,
                    message="Move a.txt to b.txt?",
                ),
                full_text="",
            )

        with patch.object(agent.planner, "plan_stream", side_effect=mock_plan_stream):
            items = list(agent.handle_user_message_stream("move a.txt to b.txt"))

        replies = [i for i in items if isinstance(i, AgentReply)]
        assert len(replies) == 1
        assert replies[0].pending_action is not None
        assert replies[0].tool_name is None  # Not executed yet

    def test_plan_non_streaming_uses_stream(self, agent: Agent) -> None:
        """Non-streaming plan() should still work by consuming plan_stream()."""
        text = "I'm here to help."

        def mock_plan_stream(user_text, history=None):
            yield StreamChunk(text="I'm ")
            yield StreamChunk(text="here to help.")
            yield StreamResult(plan=Plan(mode="respond", message=text), full_text=text)

        with patch.object(agent.planner._delegate, "plan_stream", side_effect=mock_plan_stream):
            result = agent.handle_user_message("hello")

        assert result.message == text
