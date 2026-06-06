from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from agent_app.core.agent import Agent
from agent_app.errors import format_user_error
from agent_app.llm_profiles import LlmProfile, ProfileStore, profile_summary
from agent_app.models import AgentReply
from agent_app.ui.background import BackgroundRunner
from agent_app.ui.model_switch import ModelSwitcherBar


class AgentDesktopApp:
    def __init__(
        self,
        root: tk.Tk,
        agent: Agent,
        *,
        profile_store: ProfileStore,
        active_profile: LlmProfile,
    ) -> None:
        self.root = root
        self.agent = agent
        self.profile_store = profile_store
        self.active_profile = active_profile
        self.pending_action_id: str | None = None
        self._runner = BackgroundRunner(root)
        self._busy = False

        self.root.title("Local AI Agent")
        self.root.geometry("920x640")
        self.root.minsize(760, 520)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._append("system", "Agent 已启动。你可以让我列目录、搜索网页，或执行待确认文件操作。")

    def _on_close(self) -> None:
        self._runner.shutdown()
        self.root.destroy()

    def show_llm_status(self, status: str) -> None:
        self._append("system", status)

    def initialize_llm_profile(self, profile: LlmProfile) -> None:
        self.active_profile = profile
        self.switcher.update_profile(profile)
        self._append("system", f"正在连接模型：{profile_summary(profile)}…")
        self._set_busy(True)

        self._runner.submit(
            lambda: self.agent.apply_llm_profile(profile),
            self.show_llm_status,
            on_error=lambda exc: self._append("system", format_user_error("连接模型失败。", exc)),
            on_finished=lambda: self._set_busy(False),
        )

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        self.switcher = ModelSwitcherBar(
            frame,
            self.profile_store,
            self.active_profile,
            on_switch=self._on_profile_switch,
        )
        self.switcher.pack(fill=tk.X, pady=(0, 10))

        self.chat = tk.Text(frame, wrap=tk.WORD, state=tk.DISABLED)
        self.chat.pack(fill=tk.BOTH, expand=True)

        bottom = ttk.Frame(frame)
        bottom.pack(fill=tk.X, pady=(10, 0))

        self.input_var = tk.StringVar()
        self.input = ttk.Entry(bottom, textvariable=self.input_var)
        self.input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input.bind("<Return>", self._on_send)

        self.send_btn = ttk.Button(bottom, text="发送", command=self._on_send)
        self.send_btn.pack(side=tk.LEFT, padx=(8, 0))

        actions = ttk.Frame(frame)
        actions.pack(fill=tk.X, pady=(10, 0))
        self.approve_btn = ttk.Button(actions, text="批准动作", command=self._on_approve, state=tk.DISABLED)
        self.approve_btn.pack(side=tk.LEFT)
        self.reject_btn = ttk.Button(actions, text="拒绝动作", command=self._on_reject, state=tk.DISABLED)
        self.reject_btn.pack(side=tk.LEFT, padx=(8, 0))

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        send_state = tk.DISABLED if busy else tk.NORMAL
        self.send_btn.configure(state=send_state)
        self.input.configure(state=send_state)
        if busy:
            self.approve_btn.configure(state=tk.DISABLED)
            self.reject_btn.configure(state=tk.DISABLED)

    def _on_profile_switch(self, profile: LlmProfile) -> None:
        self.active_profile = profile
        self.switcher.update_profile(profile)
        self._append("system", f"正在切换模型：{profile_summary(profile)}…")
        self._set_busy(True)

        def work() -> str:
            return self.agent.apply_llm_profile(profile)

        self._runner.submit(
            work,
            lambda status: self._append("system", f"已切换模型：{profile_summary(profile)}\n{status}"),
            on_error=lambda exc: self._append("system", format_user_error("切换模型失败。", exc)),
            on_finished=lambda: self._set_busy(False),
        )

    def _set_action_buttons(self, enabled: bool) -> None:
        if self._busy:
            return
        state = tk.NORMAL if enabled else tk.DISABLED
        self.approve_btn.configure(state=state)
        self.reject_btn.configure(state=state)

    def _on_send(self, _event: object | None = None) -> None:
        if self._busy:
            return
        text = self.input_var.get().strip()
        if not text:
            return
        self.input_var.set("")
        self._append("you", text)
        self._set_busy(True)
        self.pending_action_id = None
        self._set_action_buttons(False)

        self._runner.submit(
            lambda: self.agent.handle_user_message(text),
            self._show_agent_reply,
            on_error=lambda exc: self._append("system", format_user_error("处理消息失败。", exc)),
            on_finished=lambda: self._set_busy(False),
        )

    def _show_agent_reply(self, reply: AgentReply) -> None:
        self._append("agent", reply.message)
        if reply.pending_action:
            self.pending_action_id = reply.pending_action.action_id
            self._set_action_buttons(True)
        else:
            self.pending_action_id = None
            self._set_action_buttons(False)

    def _on_approve(self) -> None:
        if self._busy or not self.pending_action_id:
            return
        action_id = self.pending_action_id
        self.pending_action_id = None
        self._set_action_buttons(False)
        self._set_busy(True)

        self._runner.submit(
            lambda: self.agent.approve_action(action_id),
            lambda reply: self._append("agent", reply.message),
            on_error=lambda exc: self._append("system", format_user_error("批准动作失败。", exc)),
            on_finished=lambda: self._set_busy(False),
        )

    def _on_reject(self) -> None:
        if self._busy or not self.pending_action_id:
            return
        action_id = self.pending_action_id
        self.pending_action_id = None
        self._set_action_buttons(False)
        self._set_busy(True)

        self._runner.submit(
            lambda: self.agent.reject_action(action_id),
            lambda reply: self._append("agent", reply.message),
            on_error=lambda exc: self._append("system", format_user_error("拒绝动作失败。", exc)),
            on_finished=lambda: self._set_busy(False),
        )

    def _append(self, role: str, content: str) -> None:
        self.chat.configure(state=tk.NORMAL)
        self.chat.insert(tk.END, f"[{role}] {content}\n\n")
        self.chat.see(tk.END)
        self.chat.configure(state=tk.DISABLED)
