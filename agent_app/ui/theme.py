from __future__ import annotations

import platform
import tkinter as tk
from tkinter import ttk

APP_NAME = "AI小助手"

# Glass-blue palette: translucent light blue with frosted-glass feel.
COLOR_BG = "#e8f4fc"
COLOR_SURFACE = "#f0f8ff"
COLOR_GLASS = "#dceefb"
COLOR_GLASS_BORDER = "#b8d8f0"
COLOR_TEXT = "#1a3a5c"
COLOR_MUTED = "#5a7a96"
COLOR_ACCENT = "#3b8dd4"
COLOR_ACCENT_HOVER = "#2b7bc0"
COLOR_ACCENT_PRESS = "#1f6aab"
COLOR_CHAT_BG = "#f5faff"
COLOR_CHAT_FG = "#1a2e42"
COLOR_CHAT_INSERT = "#3b8dd4"
COLOR_HIGHLIGHT = "#d0e9f8"
COLOR_SIDEBAR_BG = "#daeaf7"
COLOR_SEPARATOR = "#c4ddef"


def center_window(window: tk.Tk | tk.Toplevel, width: int, height: int) -> None:
    """Place a window at the center of the primary screen."""
    window.update_idletasks()
    screen_w = int(window.winfo_screenwidth())
    screen_h = int(window.winfo_screenheight())
    pos_x = max(0, (screen_w - width) // 2)
    pos_y = max(0, (screen_h - height) // 2)
    window.geometry(f"{width}x{height}+{pos_x}+{pos_y}")


def apply_bright_theme(root: tk.Misc) -> ttk.Style:
    """Apply a frosted glass blue ttk theme shared by all app windows."""
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    if isinstance(root, (tk.Tk, tk.Toplevel)):
        root.configure(bg=COLOR_BG)
        if platform.system() == "Darwin":
            try:
                root.attributes("-transparent", False)
            except tk.TclError:
                pass

    base_font = ("PingFang SC", 13)
    title_font = ("PingFang SC", 16, "bold")
    heading_font = ("PingFang SC", 13, "bold")

    style.configure(
        ".",
        background=COLOR_BG,
        foreground=COLOR_TEXT,
        font=base_font,
        borderwidth=0,
    )
    style.configure("TFrame", background=COLOR_BG)
    style.configure("TLabel", background=COLOR_BG, foreground=COLOR_TEXT)
    style.configure("Muted.TLabel", background=COLOR_BG, foreground=COLOR_MUTED)
    style.configure(
        "Title.TLabel",
        background=COLOR_BG,
        foreground=COLOR_TEXT,
        font=title_font,
    )
    style.configure(
        "Heading.TLabel",
        background=COLOR_BG,
        foreground=COLOR_TEXT,
        font=heading_font,
    )
    style.configure("TLabelframe", background=COLOR_GLASS, foreground=COLOR_TEXT, borderwidth=1)
    style.configure("TLabelframe.Label", background=COLOR_GLASS, foreground=COLOR_TEXT)

    style.configure(
        "TEntry",
        fieldbackground=COLOR_SURFACE,
        foreground=COLOR_TEXT,
        borderwidth=1,
        relief="flat",
    )
    style.configure(
        "TCombobox",
        fieldbackground=COLOR_SURFACE,
        foreground=COLOR_TEXT,
        borderwidth=1,
    )

    style.configure(
        "TButton",
        background=COLOR_GLASS,
        foreground=COLOR_TEXT,
        padding=(14, 8),
        borderwidth=1,
        relief="flat",
    )
    style.map(
        "TButton",
        background=[("active", COLOR_HIGHLIGHT), ("pressed", COLOR_GLASS_BORDER)],
        foreground=[("disabled", COLOR_MUTED)],
        relief=[("pressed", "flat")],
    )

    style.configure(
        "Accent.TButton",
        background=COLOR_ACCENT,
        foreground="#ffffff",
        padding=(14, 8),
        borderwidth=0,
    )
    style.map(
        "Accent.TButton",
        background=[("active", COLOR_ACCENT_HOVER), ("pressed", COLOR_ACCENT_PRESS)],
    )

    style.configure("TRadiobutton", background=COLOR_BG, foreground=COLOR_TEXT)
    style.configure("TCheckbutton", background=COLOR_BG, foreground=COLOR_TEXT)

    style.configure(
        "Sidebar.TFrame",
        background=COLOR_SIDEBAR_BG,
    )
    style.configure(
        "Sidebar.TLabel",
        background=COLOR_SIDEBAR_BG,
        foreground=COLOR_TEXT,
    )
    style.configure(
        "Sidebar.TButton",
        background=COLOR_SIDEBAR_BG,
        foreground=COLOR_TEXT,
        padding=(10, 6),
        borderwidth=0,
    )
    style.map(
        "Sidebar.TButton",
        background=[("active", COLOR_GLASS), ("pressed", COLOR_HIGHLIGHT)],
    )

    style.configure(
        "TSeparator",
        background=COLOR_SEPARATOR,
    )

    return style


def style_chat_text(widget: tk.Text) -> None:
    """Style the chat text widget with a frosted glass appearance."""
    widget.configure(
        bg=COLOR_CHAT_BG,
        fg=COLOR_CHAT_FG,
        insertbackground=COLOR_CHAT_INSERT,
        selectbackground=COLOR_HIGHLIGHT,
        relief=tk.FLAT,
        padx=12,
        pady=12,
        font=("PingFang SC", 13),
        highlightthickness=1,
        highlightbackground=COLOR_GLASS_BORDER,
        highlightcolor=COLOR_ACCENT,
        borderwidth=0,
    )
