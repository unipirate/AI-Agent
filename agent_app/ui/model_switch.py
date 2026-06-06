from __future__ import annotations

import logging
import tkinter as tk
from tkinter import ttk
from typing import Callable

from agent_app.llm_profiles import (
    LlmProfile,
    ProfileStore,
    PROVIDER_ORDER,
    PROVIDER_PRESETS,
    draft_profile,
    is_local_provider,
    preset_for,
    profile_summary,
    provider_id_from_display,
    save_active_profile,
    test_profile_connection,
)
from agent_app.secrets import load_api_key, mask_api_key
from agent_app.ui.background import BackgroundRunner
from agent_app.errors import format_user_error

logger = logging.getLogger(__name__)


def _present_toplevel(window: tk.Toplevel) -> None:
    """Ensure a Toplevel is visible (macOS hides it when the root is withdrawn)."""
    window.update_idletasks()
    window.deiconify()
    window.lift()
    window.attributes("-topmost", True)
    window.after(200, lambda: window.attributes("-topmost", False))
    window.focus_force()


class ModelSwitchDialog(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Misc,
        store: ProfileStore,
        *,
        mode: str = "startup",
        on_apply: Callable[[LlmProfile, ProfileStore, str | None], str] | None = None,
    ) -> None:
        super().__init__(parent)
        self.store = store
        self.mode = mode
        self.on_apply = on_apply
        self.result: LlmProfile | None = None
        self._profile_id = store.get_active().id if store.get_active() else None

        self.title("选择大模型")
        self.geometry("520x420")
        self.resizable(False, False)

        active = store.get_active() or LlmProfile.from_preset("local_mlx")
        self.display_name_var = tk.StringVar(value=active.display_name)
        self.base_url_var = tk.StringVar(value=active.base_url or "")
        self.model_var = tk.StringVar(value=active.model)
        self.api_key_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self._runner = BackgroundRunner(parent)

        self._build_ui()
        self._load_profile(active)
        self._on_provider_change()

        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.bind("<Escape>", lambda _event: self._on_cancel())
        self._setup_modal()

    def _setup_modal(self) -> None:
        parent = self.master
        if parent.state() != "withdrawn":
            self.transient(parent)
        _present_toplevel(self)
        self.grab_set()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self, padding=16)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Provider").grid(row=0, column=0, sticky=tk.W, pady=(0, 4))
        provider_names = [PROVIDER_PRESETS[pid].display_name for pid in PROVIDER_ORDER]
        self.provider_combo = ttk.Combobox(
            frame,
            values=provider_names,
            state="readonly",
            width=42,
        )
        active = self.store.get_active()
        if active:
            self.provider_combo.set(preset_for(active.provider_id).display_name)
        self.provider_combo.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=(0, 10))
        self.provider_combo.bind("<<ComboboxSelected>>", self._on_provider_change)

        ttk.Label(frame, text="显示名称").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.display_name_var, width=44).grid(
            row=3, column=0, columnspan=2, sticky=tk.EW, pady=(0, 10)
        )

        ttk.Label(frame, text="Base URL").grid(row=4, column=0, sticky=tk.W)
        self.base_url_entry = ttk.Entry(frame, textvariable=self.base_url_var, width=44)
        self.base_url_entry.grid(row=5, column=0, columnspan=2, sticky=tk.EW, pady=(0, 10))

        ttk.Label(frame, text="Model").grid(row=6, column=0, sticky=tk.W)
        self.model_combo = ttk.Combobox(frame, textvariable=self.model_var, width=42)
        self.model_combo.grid(row=7, column=0, columnspan=2, sticky=tk.EW, pady=(0, 10))

        ttk.Label(frame, text="API Key").grid(row=8, column=0, sticky=tk.W)
        key_row = ttk.Frame(frame)
        key_row.grid(row=9, column=0, columnspan=2, sticky=tk.EW, pady=(0, 10))
        self.key_hint = ttk.Label(key_row, text="未设置", foreground="#666")
        self.key_hint.pack(side=tk.LEFT)
        self.api_key_entry = ttk.Entry(key_row, textvariable=self.api_key_var, show="*", width=34)
        self.api_key_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        ttk.Label(frame, textvariable=self.status_var, foreground="#444").grid(
            row=10, column=0, columnspan=2, sticky=tk.W, pady=(0, 10)
        )

        buttons = ttk.Frame(frame)
        buttons.grid(row=11, column=0, columnspan=2, sticky=tk.EW)
        self.test_btn = ttk.Button(buttons, text="测试连接", command=self._on_test)
        self.test_btn.pack(side=tk.LEFT)
        ttk.Button(buttons, text="取消", command=self._on_cancel).pack(side=tk.RIGHT, padx=(8, 0))

        primary_text = "保存并继续" if self.mode == "startup" else "应用"
        ttk.Button(buttons, text=primary_text, command=self._on_confirm).pack(side=tk.RIGHT)

        frame.columnconfigure(0, weight=1)

    def _load_profile(self, profile: LlmProfile) -> None:
        self._profile_id = profile.id
        self.display_name_var.set(profile.display_name)
        self.base_url_var.set(profile.base_url or "")
        self.model_var.set(profile.model)
        self.api_key_var.set("")
        stored_key = load_api_key(profile.id)
        self.key_hint.configure(text=f"已保存: {mask_api_key(stored_key)}")

    def _current_draft(self) -> LlmProfile:
        provider_id = provider_id_from_display(self.provider_combo.get())
        return draft_profile(
            profile_id=self._profile_id,
            provider_id=provider_id,
            display_name=self.display_name_var.get(),
            base_url=self.base_url_var.get(),
            model=self.model_var.get(),
        )

    def _on_provider_change(self, _event: object | None = None) -> None:
        provider_id = provider_id_from_display(self.provider_combo.get())
        preset = preset_for(provider_id)

        existing = None
        if self._profile_id:
            for profile in self.store.profiles:
                if profile.id == self._profile_id:
                    existing = profile
                    break

        if existing and existing.provider_id == provider_id:
            profile = existing
        else:
            profile = LlmProfile.from_preset(provider_id)
            self._profile_id = profile.id

        self.display_name_var.set(profile.display_name)
        self.base_url_var.set(profile.base_url or "")
        self.model_var.set(profile.model)
        self.api_key_var.set("")
        self.key_hint.configure(text=f"已保存: {mask_api_key(load_api_key(profile.id))}")

        models = list(preset.default_models)
        self.model_combo.configure(values=models if models else ())
        if models and not self.model_var.get():
            self.model_var.set(models[0])

        is_local = is_local_provider(provider_id)
        is_custom = provider_id == "custom"
        is_cloud_preset = not is_local and not is_custom and not preset.uses_anthropic

        if is_cloud_preset or preset.uses_anthropic:
            self.base_url_entry.configure(state=tk.DISABLED)
            if preset.base_url:
                self.base_url_var.set(preset.base_url)
            else:
                self.base_url_var.set("")
        else:
            self.base_url_entry.configure(state=tk.NORMAL)

        if is_local:
            self.api_key_entry.configure(state=tk.DISABLED)
        else:
            self.api_key_entry.configure(state=tk.NORMAL)

    def _on_test(self) -> None:
        profile = self._current_draft()
        api_key = self.api_key_var.get().strip() or None
        self.status_var.set("测试中…")
        self.test_btn.configure(state=tk.DISABLED)

        def work() -> tuple[bool, str]:
            return test_profile_connection(profile, api_key)

        def on_success(result: tuple[bool, str]) -> None:
            ok, message = result
            self.status_var.set(f"✓ {message}" if ok else message)

        def on_finished() -> None:
            self.test_btn.configure(state=tk.NORMAL)

        self._runner.submit(
            work,
            on_success,
            on_error=lambda exc: self.status_var.set(format_user_error("测试连接失败。", exc)),
            on_finished=on_finished,
        )

    def _on_confirm(self) -> None:
        profile = self._current_draft()
        new_key = self.api_key_var.get().strip() or None
        self.store = save_active_profile(self.store, profile, new_key)

        if self.on_apply:
            self.on_apply(profile, self.store, new_key)

        self.result = profile
        self.status_var.set("已保存。")
        self._runner.shutdown()
        self.destroy()

    def _on_cancel(self) -> None:
        if self.mode == "startup":
            active = self.store.get_active()
            self.result = active
        else:
            self.result = None
        self._runner.shutdown()
        self.destroy()


def show_model_switch_dialog(
    parent: tk.Misc,
    store: ProfileStore,
    *,
    mode: str = "startup",
    on_apply: Callable[[LlmProfile, ProfileStore, str | None], str] | None = None,
) -> LlmProfile | None:
    dialog = ModelSwitchDialog(parent, store, mode=mode, on_apply=on_apply)
    parent.update_idletasks()
    parent.wait_window(dialog)
    return dialog.result


class ModelSwitcherBar(ttk.Frame):
    def __init__(
        self,
        parent: tk.Misc,
        store: ProfileStore,
        profile: LlmProfile,
        on_switch: Callable[[LlmProfile], None],
    ) -> None:
        super().__init__(parent)
        self.store = store
        self.profile = profile
        self.on_switch = on_switch

        self.summary_var = tk.StringVar(value=f"当前：{profile_summary(profile)}")
        ttk.Label(self, textvariable=self.summary_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(self, text="切换模型…", command=self._open_dialog).pack(side=tk.RIGHT)

    def update_profile(self, profile: LlmProfile) -> None:
        self.profile = profile
        self.summary_var.set(f"当前：{profile_summary(profile)}")

    def _open_dialog(self) -> None:
        result = show_model_switch_dialog(
            self.winfo_toplevel(),
            self.store,
            mode="switch",
        )
        if result:
            self.on_switch(result)
