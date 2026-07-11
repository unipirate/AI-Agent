from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

MAX_HISTORY_TOKENS: int = 3000
MAX_TOOL_CONTENT_CHARS: int = 8000

try:
    import tiktoken

    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False


def count_tokens(text: str, model: str = "") -> int:
    """Estimate token count for *text*.

    Uses tiktoken when available and model is OpenAI-compatible.
    Falls back to len(text) // 4 heuristic otherwise.
    """
    if _TIKTOKEN_AVAILABLE and _is_tiktoken_model(model):
        try:
            enc = tiktoken.encoding_for_model(model)
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


def trim_history_by_tokens(
    messages: list[dict[str, Any]],
    max_tokens: int = MAX_HISTORY_TOKENS,
    model: str = "",
) -> list[dict[str, Any]]:
    """Return the longest recent suffix of *messages* fitting within *max_tokens*.

    Iterates from newest to oldest, accumulating token counts.
    Always keeps at least the most recent message regardless of size.
    """
    if not messages:
        return []

    result: list[dict[str, Any]] = []
    budget = max_tokens

    for msg in reversed(messages):
        tokens = count_tokens(msg.get("content", ""), model)
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
