from __future__ import annotations

import tkinter as tk
from pathlib import Path

from agent_app.app import AgentDesktopApp
from agent_app.config import load_settings
from agent_app.core.agent import Agent
from agent_app.llm_profiles import bootstrap_from_env, load_profile_store
from agent_app.log_config import configure_logging
from agent_app.ui.model_switch import show_model_switch_dialog


def ensure_allowed_root(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    configure_logging()
    base_settings = load_settings()
    ensure_allowed_root(base_settings.allowed_root)

    store = load_profile_store()
    if store is None:
        store = bootstrap_from_env(base_settings)

    root = tk.Tk()
    root.withdraw()

    agent = Agent(base_settings)

    profile = show_model_switch_dialog(root, store, mode="startup")
    if profile is None:
        profile = store.get_active()
    if profile is None:
        root.destroy()
        return

    root.deiconify()

    app = AgentDesktopApp(
        root,
        agent,
        profile_store=store,
        active_profile=profile,
    )
    app.initialize_llm_profile(profile)
    root.mainloop()


if __name__ == "__main__":
    main()
