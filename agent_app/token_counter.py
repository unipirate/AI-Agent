from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

MAX_HISTORY_TOKENS: int = 3000
MAX_TOOL_CONTENT_CHARS: int = 8000
_SINGLE_MSG_TOKEN_CAP: int = 1500

try:
    import tiktoken

    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False


@lru_cache(maxsize=8)
def _get_encoding(model: str) -> Any:
    """Cached tiktoken encoder lookup."""
    return tiktoken.encoding_for_model(model)


def count_tokens(text: str, model: str = "") -> int:
    """Estimate token count for *text*.

    Uses tiktoken when available and model is OpenAI-compatible.
    Falls back to len(text) // 4 heuristic otherwise.
    """
    if _TIKTOKEN_AVAILABLE and _is_tiktoken_model(model):
        try:
            enc = _get_encoding(model)
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


def truncate_text_to_tokens(text: str, max_tokens: int, model: str = "") -> str:
    """Truncate *text* to fit within *max_tokens*, appending ellipsis if cut."""
    if _TIKTOKEN_AVAILABLE and _is_tiktoken_model(model):
        try:
            enc = _get_encoding(model)
            token_ids = enc.encode(text)
            if len(token_ids) <= max_tokens:
                return text
            return str(enc.decode(token_ids[:max_tokens])) + "…[截断]"
        except Exception:
            pass
    char_limit = max_tokens * 4
    if len(text) <= char_limit:
        return text
    return text[:char_limit] + "…[截断]"


def trim_history_by_tokens(
    messages: list[dict[str, Any]],
    max_tokens: int = MAX_HISTORY_TOKENS,
    model: str = "",
) -> list[dict[str, Any]]:
    """Return the longest recent suffix of *messages* fitting within *max_tokens*.

    Iterates from newest to oldest, accumulating token counts.
    Always keeps at least the most recent message (truncated if oversized).
    Individual messages exceeding _SINGLE_MSG_TOKEN_CAP are truncated to
    prevent a single long message from monopolizing the entire context window.
    """
    if not messages:
        return []

    result: list[dict[str, Any]] = []
    budget = max_tokens

    for msg in reversed(messages):
        content = msg.get("content", "")
        tokens = count_tokens(content, model)

        if tokens > _SINGLE_MSG_TOKEN_CAP:
            content = truncate_text_to_tokens(content, _SINGLE_MSG_TOKEN_CAP, model)
            tokens = _SINGLE_MSG_TOKEN_CAP
            msg = {**msg, "content": content}

        if result and tokens > budget:
            break
        result.append(msg)
        budget -= tokens

    result.reverse()
    return result


def _is_tiktoken_model(model: str) -> bool:
    """Return True if tiktoken likely supports this model name."""
    if not model:
        return False
    prefixes = ("gpt-", "o1-", "o3-", "text-", "davinci", "curie", "babbage", "ada")
    return any(model.startswith(p) for p in prefixes)
