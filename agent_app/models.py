from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProposedAction:
    action_id: str
    description: str
    tool_name: str
    args: dict[str, Any]


@dataclass
class AgentReply:
    message: str
    pending_action: ProposedAction | None = None
    tool_name: str | None = None


@dataclass
class Plan:
    mode: str
    message: str = ""
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    requires_confirmation: bool = False


@dataclass
class StreamChunk:
    """A single streaming token or status indicator.

    chunk_type:
      - "text": normal token text for incremental UI display
      - "tool_status": status message (e.g. tool call start/end)
    """

    text: str
    chunk_type: str = "text"


@dataclass
class StreamResult:
    """Final structured result emitted at end of a streaming plan.

    Separation of concerns:
      - plan: structured decision (mode, tool_name, tool_args, etc.)
      - full_text: raw accumulated text from all StreamChunks yielded during the stream

    When plan.mode == "respond", plan.message and full_text have the same content.
    full_text is retained as the exact record of what was displayed to the user,
    while plan.message may undergo post-processing in the future.
    """

    plan: Plan
    full_text: str
