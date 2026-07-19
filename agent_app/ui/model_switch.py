from __future__ import annotations

import logging
import tkinter as tk
from collections.abc import Callable
from tkinter import ttk

from agent_app.errors import format_user_error
from agent_app.llm_profiles import (
    DiscoveredLocalModel,
    LlmProfile,
    ProfileStore,
    discover_running_local_models,
    draft_profile,
    is_local_provider,
    preset_for,
    profile_summary,
    save_active_profile,
    test_profile_connection,
)
from agent_app.secrets import load_api_key, mask_api_key
from agent_app.ui.background import BackgroundRunner
from agent_app.ui.theme import APP_NAME, apply_bright_theme, center_window

logger = logging.getLogger(__name__)

MINERU_KEYRING_KEY = "mineru_token"
LOCAL_GROUP_KEY = "local_group"
DIALOG_WIDTH = 560
DIALOG_HEIGHT = 540

# Sorted alphabetically by button label (plan requirement).
PROVIDER_BUTTONS: tuple[tuple[str, str], ...] = (
    ("ChatGPT", "openai"),
    ("Claude", "claude"),
    ("DeepSeek", "deepseek"),
    ("Gemini", "gemini"),
    ("本地大模型", LOCAL_GROUP_KEY),
    ("自定义", "custom"),
)


def _present_toplevel(window: tk.Toplevel) -> None:
    """Ensure a Toplevel is visible (macOS hides it when the root is withdrawn)."""
    center_window(window, DIALOG_WIDTH, DIALOG_HEIGHT)
    window.update_idletasks()
    window.deiconify()
    window.lift()
    window.attributes("-topmost", True)
    window.after(200, lambda: window.attributes("-topmost", False))
    window.focus_force()


def _entry_key_for_profile(profile: LlmProfile) -> str:
    if is_local_provider(profile.provider_id):
        return LOCAL_GROUP_KEY
    return profile.provider_id


def _config_title_for_entry(entry_key: str) -> str:
    if entry_key == LOCAL_GROUP_KEY:
        return "配置 · 本地大模型"
    for label, key in PROVIDER_BUTTONS:
        if key == entry_key:
            return f"配置 · {label}"
    return "配置大模型"


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
        _active = store.get_active()
        self._profile_id = _active.id if _active else None
        self._entry_key = ""
        self._active_provider_id = "local_mlx"
        self._local_options: dict[str, DiscoveredLocalModel] = {}

        self.title(f"{APP_NAME} · 选择大模型")
        self.geometry(f"{DIALOG_WIDTH}x{DIALOG_HEIGHT}")
        self.resizable(False, False)
        apply_bright_theme(self)
        center_window(self, DIALOG_WIDTH, DIALOG_HEIGHT)

        active = store.get_active() or LlmProfile.from_preset("local_mlx")
        self.display_name_var = tk.StringVar(value=active.display_name)
        self.base_url_var = tk.StringVar(value=active.base_url or "")
        self.model_var = tk.StringVar(value=active.model)
        self.local_model_var = tk.StringVar()
        self.api_key_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self._config_title_var = tk.StringVar()
        self._mineru_enabled_var = tk.BooleanVar(value=False)
        self._mineru_token_var = tk.StringVar()
        self._runner = BackgroundRunner(parent)

        self._container = ttk.Frame(self, padding=16)
        self._container.pack(fill=tk.BOTH, expand=True)
        self._picker_frame = ttk.Frame(self._container)
        self._config_frame = ttk.Frame(self._container)

        self._build_picker_page()
        self._build_config_page()
        self._show_picker()

        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.bind("<Escape>", lambda _event: self._on_cancel())
        self._setup_modal()

    def _setup_modal(self) -> None:
        parent = self.master
        if hasattr(parent, "state") and parent.state() != "withdrawn":
            self.transient(parent)  # type: ignore[call-overload]
        _present_toplevel(self)
        self.grab_set()

    def _build_picker_page(self) -> None:
        ttk.Label(self._picker_frame, text="选择大模型", style="Title.TLabel").pack(
            anchor=tk.W, pady=(0, 4)
        )

        active = self.store.get_active()
        current_key = _entry_key_for_profile(active) if active else None
        if current_key and active:
            ttk.Label(
                self._picker_frame,
                text=f"当前：{profile_summary(active)}",
                style="Muted.TLabel",
            ).pack(anchor=tk.W, pady=(0, 12))

        grid = ttk.Frame(self._picker_frame)
        grid.pack(fill=tk.BOTH, expand=True)

        for index, (label, entry_key) in enumerate(PROVIDER_BUTTONS):
            row, col = divmod(index, 2)
            button_label = label
            if entry_key == current_key:
                button_label = f"{label}（当前）"
            btn = ttk.Button(
                grid,
                text=button_label,
                command=lambda key=entry_key: self._show_config(key),  # type: ignore[misc]
                width=22,
            )
            btn.grid(row=row, column=col, padx=6, pady=6, sticky=tk.EW)

        for col in range(2):
            grid.columnconfigure(col, weight=1)

        # MinerU Token optional section
        mineru_frame = ttk.LabelFrame(self._picker_frame, text="PDF 解析（可选）", padding=8)
        mineru_frame.pack(fill=tk.X, pady=(12, 0))

        check_row = ttk.Frame(mineru_frame)
        check_row.pack(fill=tk.X)
        self._mineru_check = ttk.Checkbutton(
            check_row,
            text="配置 MinerU Token（用于高精度 PDF 提取）",
            variable=self._mineru_enabled_var,
            command=self._toggle_mineru_token,
        )
        self._mineru_check.pack(anchor=tk.W)

        self._mineru_token_frame = ttk.Frame(mineru_frame)
        self._mineru_hint = ttk.Label(
            self._mineru_token_frame,
            text="免费获取：https://mineru.net/apiManage/token",
            style="Muted.TLabel",
        )
        self._mineru_hint.pack(anchor=tk.W, pady=(4, 2))
        token_row = ttk.Frame(self._mineru_token_frame)
        token_row.pack(fill=tk.X)
        self._mineru_key_hint = ttk.Label(token_row, text="未设置", style="Muted.TLabel")
        self._mineru_key_hint.pack(side=tk.LEFT)
        self._mineru_token_entry = ttk.Entry(
            token_row, textvariable=self._mineru_token_var, show="*", width=34
        )
        self._mineru_token_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        self._load_mineru_token_state()

        footer = ttk.Frame(self._picker_frame)
        footer.pack(fill=tk.X, pady=(16, 0))
        ttk.Button(footer, text="取消", command=self._on_cancel).pack(side=tk.RIGHT)

    def _build_config_page(self) -> None:
        header = ttk.Frame(self._config_frame)
        header.pack(fill=tk.X, pady=(0, 12))
        ttk.Button(header, text="← 返回", command=self._show_picker).pack(side=tk.LEFT)
        ttk.Label(header, textvariable=self._config_title_var, style="Heading.TLabel").pack(
            side=tk.LEFT, padx=(12, 0)
        )

        form = ttk.Frame(self._config_frame)
        form.pack(fill=tk.BOTH, expand=True)
        form.columnconfigure(0, weight=1)

        self._display_name_frame = ttk.Frame(form)
        self._display_name_frame.grid(row=0, column=0, sticky=tk.EW, pady=(0, 10))
        ttk.Label(self._display_name_frame, text="显示名称").pack(anchor=tk.W)
        ttk.Entry(self._display_name_frame, textvariable=self.display_name_var, width=44).pack(
            fill=tk.X, expand=True
        )

        self._base_url_frame = ttk.Frame(form)
        self._base_url_frame.grid(row=1, column=0, sticky=tk.EW, pady=(0, 10))
        ttk.Label(self._base_url_frame, text="Base URL").pack(anchor=tk.W)
        self.base_url_entry = ttk.Entry(
            self._base_url_frame, textvariable=self.base_url_var, width=44
        )
        self.base_url_entry.pack(fill=tk.X, expand=True)

        self._model_frame = ttk.Frame(form)
        self._model_frame.grid(row=2, column=0, sticky=tk.EW, pady=(0, 10))
        model_header = ttk.Frame(self._model_frame)
        model_header.pack(fill=tk.X)
        self._model_label = ttk.Label(model_header, text="Model")
        self._model_label.pack(side=tk.LEFT, anchor=tk.W)
        self.rescan_btn = ttk.Button(
            model_header, text="重新扫描", command=self._discover_local_models
        )
        self.rescan_btn.pack(side=tk.RIGHT)
        self.model_combo = ttk.Combobox(self._model_frame, textvariable=self.model_var, width=42)
        self.model_combo.pack(fill=tk.X, expand=True)
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_combo_selected)

        self._api_key_frame = ttk.Frame(form)
        self._api_key_frame.grid(row=3, column=0, sticky=tk.EW, pady=(0, 10))
        ttk.Label(self._api_key_frame, text="API Key").pack(anchor=tk.W)
        key_row = ttk.Frame(self._api_key_frame)
        key_row.pack(fill=tk.X, expand=True)
        self.key_hint = ttk.Label(key_row, text="未设置", style="Muted.TLabel")
        self.key_hint.pack(side=tk.LEFT)
        self.api_key_entry = ttk.Entry(key_row, textvariable=self.api_key_var, show="*", width=34)
        self.api_key_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        ttk.Label(form, textvariable=self.status_var, style="Muted.TLabel").grid(
            row=4, column=0, sticky=tk.W, pady=(0, 10)
        )

        buttons = ttk.Frame(self._config_frame)
        buttons.pack(fill=tk.X)
        self.test_btn = ttk.Button(buttons, text="测试连接", command=self._on_test)
        self.test_btn.pack(side=tk.LEFT)
        ttk.Button(buttons, text="取消", command=self._on_cancel).pack(side=tk.RIGHT, padx=(8, 0))

        primary_text = "保存并继续" if self.mode == "startup" else "应用"
        ttk.Button(
            buttons, text=primary_text, style="Accent.TButton", command=self._on_confirm
        ).pack(side=tk.RIGHT)

    def _show_picker(self) -> None:
        self._config_frame.pack_forget()
        self._picker_frame.pack(fill=tk.BOTH, expand=True)
        self.status_var.set("")

    def _show_config(self, entry_key: str) -> None:
        self._entry_key = entry_key
        self.status_var.set("")
        self._config_title_var.set(_config_title_for_entry(entry_key))

        if entry_key == LOCAL_GROUP_KEY:
            self._update_field_visibility()
            self._discover_local_models()
        else:
            self._apply_provider(entry_key)
            self._update_field_visibility()

        self._picker_frame.pack_forget()
        self._config_frame.pack(fill=tk.BOTH, expand=True)

    def _update_field_visibility(self) -> None:
        is_local_group = self._entry_key == LOCAL_GROUP_KEY
        is_custom = self._active_provider_id == "custom" and not is_local_group

        if is_local_group:
            self._display_name_frame.grid_remove()
            self._base_url_frame.grid_remove()
            self._api_key_frame.grid_remove()
            self._model_frame.grid()
            self._model_label.configure(text="选择已加载的本地模型")
            self.rescan_btn.pack(side=tk.RIGHT)
            self.model_combo.configure(textvariable=self.local_model_var)
            return

        self.rescan_btn.pack_forget()
        self.model_combo.configure(textvariable=self.model_var)

        if is_custom:
            self._display_name_frame.grid()
        else:
            self._display_name_frame.grid_remove()

        if is_custom:
            self._base_url_frame.grid()
        else:
            self._base_url_frame.grid_remove()

        self._model_frame.grid()
        self._model_label.configure(text="Model")
        self._api_key_frame.grid()

    def _on_model_combo_selected(self, _event: object | None = None) -> None:
        if self._entry_key != LOCAL_GROUP_KEY:
            return
        label = self.local_model_var.get().strip()
        if label:
            self._apply_local_selection(label)

    def _apply_local_selection(self, label: str) -> None:
        option = self._local_options.get(label)
        if option is None:
            return

        self._active_provider_id = option.provider_id
        self.base_url_var.set(option.base_url)
        self.model_var.set(option.model_id)
        self.display_name_var.set(preset_for(option.provider_id).display_name)

        existing = None
        if self._profile_id:
            for profile in self.store.profiles:
                if profile.id == self._profile_id:
                    existing = profile
                    break

        if not (existing and existing.provider_id == option.provider_id):
            profile = LlmProfile.from_preset(option.provider_id)
            self._profile_id = profile.id

    def _discover_local_models(self) -> None:
        active = self.store.get_active()
        saved_model = active.model if active and is_local_provider(active.provider_id) else ""
        saved_base = active.base_url if active and is_local_provider(active.provider_id) else ""

        self.model_combo.configure(state=tk.DISABLED, values=())
        self.local_model_var.set("")
        self.status_var.set("正在扫描已加载的本地模型…")
        self.rescan_btn.configure(state=tk.DISABLED)

        def work() -> list[DiscoveredLocalModel]:
            return discover_running_local_models()

        def on_success(options: list[DiscoveredLocalModel]) -> None:
            self._local_options = {option.label: option for option in options}
            if not options:
                self.model_combo.configure(state=tk.DISABLED, values=())
                self.status_var.set("未检测到已启动的本地服务。")
                return

            labels = tuple(option.label for option in options)
            self.model_combo.configure(values=labels, state="readonly")

            matched_label = None
            for option in options:
                if option.model_id == saved_model and option.base_url == saved_base:
                    matched_label = option.label
                    break

            if matched_label:
                self.local_model_var.set(matched_label)
                self._apply_local_selection(matched_label)
                self.status_var.set(f"已发现 {len(options)} 个可用模型。")
            elif len(options) == 1:
                self.local_model_var.set(options[0].label)
                self._apply_local_selection(options[0].label)
                self.status_var.set("已发现 1 个可用模型。")
            else:
                self.status_var.set(f"已发现 {len(options)} 个可用模型，请选择。")

        def on_error(exc: Exception) -> None:
            self.model_combo.configure(state=tk.DISABLED, values=())
            self.status_var.set(format_user_error("扫描本地服务失败。", exc))

        def on_finished() -> None:
            self.rescan_btn.configure(state=tk.NORMAL)

        self._runner.submit(work, on_success, on_error=on_error, on_finished=on_finished)

    def _load_profile(self, profile: LlmProfile) -> None:
        self._profile_id = profile.id
        self.display_name_var.set(profile.display_name)
        self.base_url_var.set(profile.base_url or "")
        self.model_var.set(profile.model)
        self.api_key_var.set("")
        stored_key = load_api_key(profile.id)
        self.key_hint.configure(text=f"已保存: {mask_api_key(stored_key)}")

    def _current_draft(self) -> LlmProfile:
        return draft_profile(
            profile_id=self._profile_id,
            provider_id=self._active_provider_id,
            display_name=self.display_name_var.get(),
            base_url=self.base_url_var.get(),
            model=self.model_var.get(),
        )

    def _apply_provider(self, provider_id: str) -> None:
        self._active_provider_id = provider_id
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

        self._load_profile(profile)

        is_custom = provider_id == "custom"
        is_cloud_preset = not is_custom and not preset.uses_anthropic

        models = list(preset.default_models)
        self.model_combo.configure(
            values=models if models else (),
            state="readonly" if models else tk.NORMAL,
        )
        if models and not self.model_var.get():
            self.model_var.set(models[0])

        if is_cloud_preset or preset.uses_anthropic:
            self.base_url_entry.configure(state=tk.DISABLED)
            if preset.base_url:
                self.base_url_var.set(preset.base_url)
            else:
                self.base_url_var.set("")
        else:
            self.base_url_entry.configure(state=tk.NORMAL)

        self.api_key_entry.configure(state=tk.NORMAL)

    def _on_test(self) -> None:
        if self._entry_key == LOCAL_GROUP_KEY and not self.local_model_var.get().strip():
            self.status_var.set("请先选择一个本地模型。")
            return

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
        if self._entry_key == LOCAL_GROUP_KEY and not self.local_model_var.get().strip():
            self.status_var.set("请先选择一个本地模型。")
            return

        profile = self._current_draft()
        new_key = self.api_key_var.get().strip() or None
        self.store = save_active_profile(self.store, profile, new_key)
        self._save_mineru_token()

        if self.on_apply:
            self.on_apply(profile, self.store, new_key)

        self.result = profile
        self.status_var.set("已保存。")
        self._runner.shutdown()
        self.destroy()

    def _on_cancel(self) -> None:
        # Startup: dismiss without saving → caller should exit (not open main window).
        self.result = None
        self._runner.shutdown()
        self.destroy()

    # ------ MinerU Token helpers ------

    def _load_mineru_token_state(self) -> None:
        """Load existing MinerU token from keyring and update UI state."""
        from agent_app.secrets import load_api_key, mask_api_key

        stored = load_api_key(MINERU_KEYRING_KEY)
        if stored:
            self._mineru_enabled_var.set(True)
            self._mineru_token_frame.pack(fill=tk.X, pady=(4, 0))
            self._mineru_key_hint.configure(text=f"已保存: {mask_api_key(stored)}")
        else:
            self._mineru_enabled_var.set(False)
            self._mineru_token_frame.pack_forget()

    def _toggle_mineru_token(self) -> None:
        """Show/hide the MinerU token input based on checkbox state."""
        if self._mineru_enabled_var.get():
            self._mineru_token_frame.pack(fill=tk.X, pady=(4, 0))
        else:
            self._mineru_token_frame.pack_forget()

    def _save_mineru_token(self) -> None:
        """Persist MinerU token to keyring if provided."""
        from agent_app.secrets import delete_api_key, save_api_key

        if not self._mineru_enabled_var.get():
            delete_api_key(MINERU_KEYRING_KEY)
            return

        new_token = self._mineru_token_var.get().strip()
        if new_token:
            save_api_key(MINERU_KEYRING_KEY, new_token)


def load_mineru_token() -> str | None:
    """Load MinerU token from keyring (utility for use outside the dialog)."""
    from agent_app.secrets import load_api_key

    return load_api_key(MINERU_KEYRING_KEY)


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
