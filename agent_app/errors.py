from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def format_user_error(context: str, exc: Exception) -> str:
    """Return a user-facing error string that does not echo exception details."""
    logger.exception("%s", context.rstrip("。"))
    return f"{context}（{type(exc).__name__}）"
