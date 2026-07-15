from __future__ import annotations

import tkinter as tk
from tkinter import ttk

APP_NAME = "AI小助手"

# Bright, light palette for desktop UI.
COLOR_BG = "#f4f6fb"
COLOR_SURFACE = "#ffffff"
COLOR_TEXT = "#1f2937"
COLOR_MUTED = "#5b6472"
COLOR_ACCENT = "#2563eb"
COLOR_CHAT_BG = "#ffffff"
COLOR_CHAT_FG = "#111827"
COLOR_CHAT_INSERT = "#2563eb"


def center_window(window: tk.Tk | tk.Toplevel, width: int, height: int) -> None:
    """Place a window at the center of the primary screen."""
    window.update_idletasks()
    screen_w = int(window.winfo_screenwidth())
    screen_h = int(window.winfo_screenheight())
    pos_x = max(0, (screen_w - width) // 2)
    pos_y = max(0, (screen_h - height) // 2)
    window.geometry(f"{width}x{height}+{pos_x}+{pos_y}")


def apply_bright_theme(root: tk.Misc) -> ttk.Style:
    """Apply a light ttk theme shared by all app windows."""
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    if isinstance(root, (tk.Tk, tk.Toplevel)):
        root.configure(bg=COLOR_BG)

    base_font = ("PingFang SC", 13)
    title_font = ("PingFang SC", 14, "bold")
    heading_font = ("PingFang SC", 12, "bold")

    style.configure(".", background=COLOR_BG, foreground=COLOR_TEXT, font=base_font)
    style.configure("TFrame", background=COLOR_BG)
    style.configure("TLabel", background=COLOR_BG, foreground=COLOR_TEXT)
    style.configure("Muted.TLabel", background=COLOR_BG, foreground=COLOR_MUTED)
    style.configure("Title.TLabel", background=COLOR_BG, foreground=COLOR_TEXT, font=title_font)
    style.configure("Heading.TLabel", background=COLOR_BG, foreground=COLOR_TEXT, font=heading_font)
    style.configure("TLabelframe", background=COLOR_BG, foreground=COLOR_TEXT)
    style.configure("TLabelframe.Label", background=COLOR_BG, foreground=COLOR_TEXT)
    style.configure("TEntry", fieldbackground=COLOR_SURFACE, foreground=COLOR_TEXT)
    style.configure("TCombobox", fieldbackground=COLOR_SURFACE, foreground=COLOR_TEXT)
    style.configure(
        "TButton",
        background=COLOR_SURFACE,
        foreground=COLOR_TEXT,
        padding=(12, 7),
        borderwidth=1,
    )
    style.map(
        "TButton",
        background=[("active", "#e8eef8"), ("pressed", "#dbeafe")],
        foreground=[("disabled", COLOR_MUTED)],
    )
    style.configure(
        "Accent.TButton",
        background=COLOR_ACCENT,
        foreground="#ffffff",
        padding=(12, 7),
    )
    style.map(
        "Accent.TButton",
        background=[("active", "#1d4ed8"), ("pressed", "#1e40af")],
    )
    style.configure("TRadiobutton", background=COLOR_BG, foreground=COLOR_TEXT)
    return style


def style_chat_text(widget: tk.Text) -> None:
    widget.configure(
        bg=COLOR_CHAT_BG,
        fg=COLOR_CHAT_FG,
        insertbackground=COLOR_CHAT_INSERT,
        selectbackground="#dbeafe",
        relief=tk.FLAT,
        padx=10,
        pady=10,
        font=("PingFang SC", 13),
    )
