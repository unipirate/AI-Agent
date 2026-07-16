from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from typing import Callable

from agent_app.conversation import SessionIndex, SessionMeta
from agent_app.ui.theme import (
    COLOR_ACCENT,
    COLOR_BG,
    COLOR_MUTED,
    COLOR_SURFACE,
    COLOR_TEXT,
)


class SessionPanel(ttk.Frame):
    """Left sidebar showing the list of conversation sessions."""

    def __init__(
        self,
        parent: tk.Widget,
        session_index: SessionIndex,
        *,
        active_session_id: str,
        on_switch: Callable[[str], None],
        on_new: Callable[[], None],
        on_delete: Callable[[str], None],
        on_rename: Callable[[str, str], None],
    ) -> None:
        super().__init__(parent, width=220)
        self.pack_propagate(False)

        self._index = session_index
        self._active_id = active_session_id
        self._on_switch = on_switch
        self._on_new = on_new
        self._on_delete = on_delete
        self._on_rename = on_rename

        self._build_ui()
        self._populate()

    def _build_ui(self) -> None:
        header = ttk.Frame(self)
        header.pack(fill=tk.X, padx=8, pady=(10, 6))

        ttk.Label(header, text="会话列表", style="Heading.TLabel").pack(side=tk.LEFT)

        new_btn = ttk.Button(header, text="＋", width=3, command=self._on_new)
        new_btn.pack(side=tk.RIGHT)

        separator = ttk.Separator(self, orient=tk.HORIZONTAL)
        separator.pack(fill=tk.X, padx=8)

        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._canvas = tk.Canvas(
            container,
            bg=COLOR_BG,
            highlightthickness=0,
            width=200,
        )
        scrollbar = ttk.Scrollbar(
            container, orient=tk.VERTICAL, command=self._canvas.yview
        )
        self._scrollable = ttk.Frame(self._canvas)

        self._scrollable.bind(
            "<Configure>",
            lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")),
        )

        self._canvas_window = self._canvas.create_window(
            (0, 0), window=self._scrollable, anchor="nw"
        )
        self._canvas.configure(yscrollcommand=scrollbar.set)

        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._canvas.bind("<Configure>", self._on_canvas_resize)

    def _on_canvas_resize(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        self._canvas.itemconfig(self._canvas_window, width=event.width)

    def _populate(self) -> None:
        for widget in self._scrollable.winfo_children():
            widget.destroy()

        sessions = self._index.list_sessions()
        for meta in sessions:
            self._create_session_item(meta)

    def _create_session_item(self, meta: SessionMeta) -> None:
        is_active = meta.session_id == self._active_id
        bg = COLOR_SURFACE if is_active else COLOR_BG
        fg = COLOR_ACCENT if is_active else COLOR_TEXT

        item_frame = tk.Frame(self._scrollable, bg=bg, cursor="hand2")
        item_frame.pack(fill=tk.X, padx=4, pady=2)

        title_label = tk.Label(
            item_frame,
            text=meta.title[:25] or "Untitled",
            bg=bg,
            fg=fg,
            font=("PingFang SC", 12, "bold" if is_active else "normal"),
            anchor="w",
        )
        title_label.pack(fill=tk.X, padx=8, pady=(6, 0))

        info_text = f"{meta.message_count} 条消息"
        info_label = tk.Label(
            item_frame,
            text=info_text,
            bg=bg,
            fg=COLOR_MUTED,
            font=("PingFang SC", 10),
            anchor="w",
        )
        info_label.pack(fill=tk.X, padx=8, pady=(0, 6))

        sid = meta.session_id
        for widget in (item_frame, title_label, info_label):
            widget.bind("<Button-1>", lambda e, s=sid: self._on_switch(s))
            widget.bind("<Button-2>", lambda e, s=sid: self._show_context_menu(e, s))
            widget.bind("<Button-3>", lambda e, s=sid: self._show_context_menu(e, s))

    def _show_context_menu(self, event: tk.Event, session_id: str) -> None:  # type: ignore[type-arg]
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="重命名", command=lambda: self._do_rename(session_id))
        menu.add_command(label="删除", command=lambda: self._do_delete(session_id))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _do_rename(self, session_id: str) -> None:
        current_title = ""
        for s in self._index.list_sessions():
            if s.session_id == session_id:
                current_title = s.title
                break
        new_title = simpledialog.askstring(
            "重命名会话",
            "输入新名称：",
            initialvalue=current_title,
            parent=self,
        )
        if new_title and new_title.strip():
            self._on_rename(session_id, new_title.strip())

    def _do_delete(self, session_id: str) -> None:
        if not messagebox.askyesno("删除会话", "确定要删除这个会话吗？", parent=self):
            return
        self._on_delete(session_id)

    def refresh(self) -> None:
        """Reload the session list from the index."""
        self._populate()

    def set_active(self, session_id: str) -> None:
        """Update which session is highlighted as active."""
        self._active_id = session_id
        self._populate()
