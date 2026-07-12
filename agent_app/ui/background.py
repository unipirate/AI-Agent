from __future__ import annotations

import logging
import queue
import tkinter as tk
from collections.abc import Generator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, TypeVar

from agent_app.models import AgentReply, StreamChunk

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

    def submit_streaming(
        self,
        work: Callable[[], Generator[StreamChunk | AgentReply, None, None]],
        on_chunk: Callable[[str], None],
        on_success: Callable[[AgentReply], None],
        *,
        on_error: Callable[[Exception], None] | None = None,
        on_finished: Callable[[], None] | None = None,
    ) -> None:
        """Submit a streaming generator task.

        Queue protocol (no sentinel needed):
          - str: text chunk to display incrementally
          - AgentReply: final result, signals stream end
          - Exception: error, signals stream end
        """
        q: queue.Queue[str | AgentReply | Exception] = queue.Queue()

        def worker() -> None:
            try:
                gen = work()
                for item in gen:
                    if isinstance(item, StreamChunk):
                        q.put(item.text)
                    elif isinstance(item, AgentReply):
                        q.put(item)
            except Exception as exc:
                logger.exception("Streaming worker failed")
                q.put(exc)

        self._executor.submit(worker)

        def poll() -> None:
            batch: list[str] = []
            try:
                while True:
                    item = q.get_nowait()
                    if isinstance(item, AgentReply):
                        if batch:
                            on_chunk("".join(batch))
                        self._root.after(0, lambda reply=item: _finish_success(reply))
                        return
                    if isinstance(item, Exception):
                        if batch:
                            on_chunk("".join(batch))
                        self._root.after(0, lambda exc=item: _finish_error(exc))
                        return
                    batch.append(item)
            except queue.Empty:
                pass

            if batch:
                on_chunk("".join(batch))
                logger.debug("poll drained %d chunks, %d chars", len(batch), sum(len(s) for s in batch))

            self._root.after(50, poll)

        def _finish_success(reply: AgentReply) -> None:
            on_success(reply)
            if on_finished:
                on_finished()

        def _finish_error(exc: Exception) -> None:
            if on_error:
                on_error(exc)
            if on_finished:
                on_finished()

        self._root.after(50, poll)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
