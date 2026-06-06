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


@dataclass
class Plan:
    mode: str
    message: str = ""
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    requires_confirmation: bool = False
