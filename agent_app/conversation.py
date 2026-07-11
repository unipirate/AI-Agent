from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent_app.token_counter import (
    MAX_HISTORY_TOKENS,
    MAX_TOOL_CONTENT_CHARS,
    count_tokens,
    trim_history_by_tokens,
)

logger = logging.getLogger(__name__)

CONVERSATIONS_DIR: Path = Path("~/.ai-agent/conversations").expanduser()
MAX_PERSISTED_MESSAGES: int = 200


@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: str
    tool_name: str | None = None
    token_count: int = 0


@dataclass
class SessionMeta:
    session_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


@dataclass
class ChatSession:
    session_id: str
    title: str
    messages: list[ChatMessage]
    created_at: str
    updated_at: str

    def add_user_message(self, content: str, model: str = "") -> None:
        tokens = count_tokens(content, model)
        msg = ChatMessage(
            role="user",
            content=content,
            timestamp=_now_iso(),
            token_count=tokens,
        )
        self.messages.append(msg)
        self.updated_at = _now_iso()
        if len(self.messages) == 1:
            self.title = content[:30].strip() or "New Chat"

    def add_assistant_message(self, content: str, model: str = "") -> None:
        tokens = count_tokens(content, model)
        msg = ChatMessage(
            role="assistant",
            content=content,
            timestamp=_now_iso(),
            token_count=tokens,
        )
        self.messages.append(msg)
        self.updated_at = _now_iso()

    def add_tool_message(self, tool_name: str, content: str, model: str = "") -> None:
        content = content[:MAX_TOOL_CONTENT_CHARS]
        tokens = count_tokens(content, model)
        msg = ChatMessage(
            role="tool",
            content=content,
            timestamp=_now_iso(),
            tool_name=tool_name,
            token_count=tokens,
        )
        self.messages.append(msg)
        self.updated_at = _now_iso()

    def get_history_for_llm(
        self, max_tokens: int = MAX_HISTORY_TOKENS, model: str = ""
    ) -> list[dict[str, str]]:
        """Return recent messages that fit within *max_tokens*, for LLM consumption.

        Ensures alternating user/assistant roles (required by Anthropic API)
        by merging consecutive same-role messages.
        """
        if not self.messages:
            return []
        all_msgs: list[dict[str, str]] = []
        for m in self.messages:
            if m.role == "tool":
                entry = {"role": "user", "content": f"[Tool: {m.tool_name}]\n{m.content}"}
            else:
                entry = {"role": m.role, "content": m.content}

            if all_msgs and all_msgs[-1]["role"] == entry["role"]:
                all_msgs[-1]["content"] += "\n\n" + entry["content"]
            else:
                all_msgs.append(entry)

        return trim_history_by_tokens(all_msgs, max_tokens, model)

    def save(self) -> None:
        CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
        path = CONVERSATIONS_DIR / f"{self.session_id}.json"
        persisted = self.messages[-MAX_PERSISTED_MESSAGES:]
        data = {
            "session_id": self.session_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [asdict(m) for m in persisted],
        }
        _atomic_write_json(path, data)

    def rename(self, new_title: str) -> None:
        self.title = new_title
        self.updated_at = _now_iso()

    def to_meta(self) -> SessionMeta:
        return SessionMeta(
            session_id=self.session_id,
            title=self.title,
            created_at=self.created_at,
            updated_at=self.updated_at,
            message_count=len(self.messages),
        )

    @classmethod
    def load(cls, session_id: str) -> ChatSession | None:
        path = CONVERSATIONS_DIR / f"{session_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            messages = [
                ChatMessage(
                    role=m["role"],
                    content=m["content"],
                    timestamp=m.get("timestamp", ""),
                    tool_name=m.get("tool_name"),
                    token_count=m.get("token_count", 0),
                )
                for m in data.get("messages", [])
            ]
            return cls(
                session_id=data["session_id"],
                title=data.get("title", "Untitled"),
                messages=messages,
                created_at=data.get("created_at", ""),
                updated_at=data.get("updated_at", ""),
            )
        except (json.JSONDecodeError, KeyError, OSError):
            logger.exception("Failed to load session %s", session_id)
            return None

    @classmethod
    def new_session(cls) -> ChatSession:
        now = _now_iso()
        return cls(
            session_id=str(uuid4()),
            title="New Chat",
            messages=[],
            created_at=now,
            updated_at=now,
        )


@dataclass
class SessionIndex:
    """Manages the index of all sessions stored in index.json."""

    active_session_id: str | None = None
    sessions: list[SessionMeta] = field(default_factory=list)

    def list_sessions(self) -> list[SessionMeta]:
        return list(self.sessions)

    def get_active_session_id(self) -> str | None:
        return self.active_session_id

    def set_active_session_id(self, session_id: str) -> None:
        self.active_session_id = session_id

    def add_session(self, meta: SessionMeta) -> None:
        self.sessions.insert(0, meta)
        self.active_session_id = meta.session_id

    def remove_session(self, session_id: str) -> str:
        """Remove a session and return the next session_id to activate.

        If the list becomes empty, creates a new session as fallback.
        """
        idx = next((i for i, s in enumerate(self.sessions) if s.session_id == session_id), None)
        if idx is not None:
            self.sessions.pop(idx)

        # Delete the session file on disk
        session_file = CONVERSATIONS_DIR / f"{session_id}.json"
        try:
            session_file.unlink(missing_ok=True)
        except OSError:
            logger.warning("Failed to delete session file %s", session_file)

        if not self.sessions:
            new = ChatSession.new_session()
            new.save()
            self.add_session(new.to_meta())
            self.save()
            return new.session_id

        # Pick adjacent session: prefer next (same index), else previous
        if idx is not None:
            next_idx = min(idx, len(self.sessions) - 1)
        else:
            next_idx = 0
        target_id = self.sessions[next_idx].session_id
        self.active_session_id = target_id
        self.save()
        return target_id

    def update_session_meta(self, meta: SessionMeta) -> None:
        for i, s in enumerate(self.sessions):
            if s.session_id == meta.session_id:
                self.sessions[i] = meta
                return

    def get_meta(self, session_id: str) -> SessionMeta | None:
        for s in self.sessions:
            if s.session_id == session_id:
                return s
        return None

    def save(self) -> None:
        CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
        path = CONVERSATIONS_DIR / "index.json"
        data = {
            "active_session_id": self.active_session_id,
            "sessions": [asdict(s) for s in self.sessions],
        }
        _atomic_write_json(path, data)

    @classmethod
    def load(cls) -> SessionIndex:
        path = CONVERSATIONS_DIR / "index.json"
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            sessions = [
                SessionMeta(
                    session_id=s["session_id"],
                    title=s.get("title", "Untitled"),
                    created_at=s.get("created_at", ""),
                    updated_at=s.get("updated_at", ""),
                    message_count=s.get("message_count", 0),
                )
                for s in data.get("sessions", [])
            ]
            return cls(
                active_session_id=data.get("active_session_id"),
                sessions=sessions,
            )
        except (json.JSONDecodeError, KeyError, OSError):
            logger.exception("Failed to load session index")
            return cls()


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON to *path* atomically via temp file + rename."""
    try:
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=".as_tmp_", suffix=".json")
        try:
            with open(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            Path(tmp_path).replace(path)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise
    except OSError:
        logger.exception("Failed to write %s", path)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
