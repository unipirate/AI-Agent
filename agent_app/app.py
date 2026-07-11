from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from agent_app.conversation import ChatSession, SessionIndex
from agent_app.core.agent import Agent
from agent_app.errors import format_user_error
from agent_app.llm_profiles import LlmProfile, ProfileStore, profile_summary
from agent_app.models import AgentReply
from agent_app.ui.background import BackgroundRunner
from agent_app.ui.model_switch import ModelSwitcherBar
from agent_app.ui.session_panel import SessionPanel
from agent_app.ui.theme import APP_NAME, center_window, style_chat_text


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

        self._session_index = SessionIndex.load()
        self._session = self._load_or_create_session()

        self.root.title(APP_NAME)
        self.root.geometry("1100x640")
        self.root.minsize(900, 520)
        center_window(self.root, 1100, 640)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._replay_session()
        self._append("system", f"{APP_NAME} 已启动。你可以让我列目录、搜索网页，或执行待确认文件操作。")

    def _load_or_create_session(self) -> ChatSession:
        active_id = self._session_index.get_active_session_id()
        if active_id:
            session = ChatSession.load(active_id)
            if session:
                return session
        session = ChatSession.new_session()
        session.save()
        self._session_index.add_session(session.to_meta())
        self._session_index.save()
        return session

    def _replay_session(self) -> None:
        """Replay existing messages from the loaded session into the chat display."""
        if not self._session.messages:
            return
        self._append("system", "── 已恢复上次会话 ──")
        for msg in self._session.messages:
            if msg.role == "user":
                self._append("you", msg.content)
            elif msg.role == "assistant":
                self._append("agent", msg.content)
            elif msg.role == "tool":
                self._append("system", f"[Tool: {msg.tool_name}] {msg.content[:200]}")

    def _on_close(self) -> None:
        self._session.save()
        self._session_index.update_session_meta(self._session.to_meta())
        self._session_index.save()
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
        outer = ttk.Frame(self.root)
        outer.pack(fill=tk.BOTH, expand=True)

        # Session sidebar
        self.session_panel = SessionPanel(
            outer,
            self._session_index,
            active_session_id=self._session.session_id,
            on_switch=self._on_session_switch,
            on_new=self._on_new_session,
            on_delete=self._on_delete_session,
            on_rename=self._on_rename_session,
        )
        self.session_panel.pack(side=tk.LEFT, fill=tk.Y)

        # Main content area
        frame = ttk.Frame(outer, padding=12)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.switcher = ModelSwitcherBar(
            frame,
            self.profile_store,
            self.active_profile,
            on_switch=self._on_profile_switch,
        )
        self.switcher.pack(fill=tk.X, pady=(0, 10))

        self.chat = tk.Text(frame, wrap=tk.WORD, state=tk.DISABLED)
        style_chat_text(self.chat)
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

        model = self.active_profile.model if self.active_profile else ""
        self._session.add_user_message(text, model)
        history = self._session.get_history_for_llm(model=model)
        # Exclude the last message (current user input) from history since agent will see it as user_text
        history_for_llm = history[:-1] if history else None

        self._runner.submit(
            lambda: self.agent.handle_user_message(text, history_for_llm or None),
            self._show_agent_reply,
            on_error=lambda exc: self._append("system", format_user_error("处理消息失败。", exc)),
            on_finished=lambda: self._set_busy(False),
        )

    def _show_agent_reply(self, reply: AgentReply) -> None:
        self._append("agent", reply.message)

        model = self.active_profile.model if self.active_profile else ""
        self._session.add_assistant_message(reply.message, model)
        self._session.save()
        self._session_index.update_session_meta(self._session.to_meta())
        self._session_index.save()
        self.session_panel.refresh()

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

    def _on_session_switch(self, session_id: str) -> None:
        """Switch to a different session."""
        if session_id == self._session.session_id:
            return
        self._session.save()
        self._session_index.update_session_meta(self._session.to_meta())
        self._session_index.set_active_session_id(session_id)
        self._session_index.save()

        loaded = ChatSession.load(session_id)
        if loaded is None:
            self._append("system", "加载会话失败。")
            return
        self._session = loaded
        self._clear_chat()
        self._replay_session()
        self.session_panel.set_active(session_id)

    def _on_new_session(self) -> None:
        """Create a new empty session and switch to it."""
        self._session.save()
        self._session_index.update_session_meta(self._session.to_meta())

        new = ChatSession.new_session()
        new.save()
        self._session_index.add_session(new.to_meta())
        self._session_index.save()
        self._session = new
        self._clear_chat()
        self._append("system", "新会话已创建。")
        self.session_panel.refresh()

    def _on_delete_session(self, session_id: str) -> None:
        """Delete a session and switch to the next available."""
        next_id = self._session_index.remove_session(session_id)
        if session_id == self._session.session_id:
            loaded = ChatSession.load(next_id)
            self._session = loaded if loaded else ChatSession.new_session()
            self._clear_chat()
            self._replay_session()
        self.session_panel.refresh()
        self.session_panel.set_active(self._session.session_id)

    def _on_rename_session(self, session_id: str, new_title: str) -> None:
        """Rename a session."""
        if session_id == self._session.session_id:
            self._session.rename(new_title)
            self._session.save()
            self._session_index.update_session_meta(self._session.to_meta())
        else:
            loaded = ChatSession.load(session_id)
            if loaded:
                loaded.rename(new_title)
                loaded.save()
                self._session_index.update_session_meta(loaded.to_meta())
        self._session_index.save()
        self.session_panel.refresh()

    def _clear_chat(self) -> None:
        """Clear the chat display."""
        self.chat.configure(state=tk.NORMAL)
        self.chat.delete("1.0", tk.END)
        self.chat.configure(state=tk.DISABLED)

    def _append(self, role: str, content: str) -> None:
        self.chat.configure(state=tk.NORMAL)
        self.chat.insert(tk.END, f"[{role}] {content}\n\n")
        self.chat.see(tk.END)
        self.chat.configure(state=tk.DISABLED)
