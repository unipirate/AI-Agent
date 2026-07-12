from __future__ import annotations

import logging
import tkinter as tk
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)


class BackgroundRunner:
    """Run blocking agent/LLM work off the Tk main thread."""

    def __init__(self, root: tk.Misc) -> None:
        self._root = root
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agent-worker")

    def submit(
        self,
        work: Callable[[], T],
        on_success: Callable[[T], None],
        *,
        on_error: Callable[[Exception], None] | None = None,
        on_finished: Callable[[], None] | None = None,
    ) -> Future[T]:
        future = self._executor.submit(work)

        def poll() -> None:
            if not future.done():
                self._root.after(100, poll)
                return

            def finish() -> None:
                try:
                    result = future.result()
                except Exception as exc:
                    logger.exception("Background task failed")
                    if on_error:
                        on_error(exc)
                else:
                    on_success(result)
                if on_finished:
                    on_finished()

            self._root.after(0, finish)

        self._root.after(100, poll)
        return future

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
