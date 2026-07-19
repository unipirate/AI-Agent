from __future__ import annotations

import tkinter as tk
from dataclasses import replace
from pathlib import Path

from agent_app.app import AgentDesktopApp
from agent_app.config import load_settings
from agent_app.core.agent import Agent
from agent_app.llm_profiles import bootstrap_from_env, load_profile_store
from agent_app.log_config import configure_logging
from agent_app.ui.model_switch import load_mineru_token, show_model_switch_dialog
from agent_app.ui.theme import APP_NAME, apply_bright_theme


def ensure_allowed_root(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# Application bootstrap entry point
def main() -> None:
    configure_logging()
    base_settings = load_settings()
    ensure_allowed_root(base_settings.allowed_root)

    store = load_profile_store()
    if store is None:
        store = bootstrap_from_env(base_settings)

    root = tk.Tk()
    root.withdraw()
    apply_bright_theme(root)
    root.title(APP_NAME)

    agent = Agent(base_settings)

    profile = show_model_switch_dialog(root, store, mode="startup")
    if profile is None:
        root.destroy()
        return

    # Merge MinerU token from keyring into settings (keyring takes priority over .env)
    keyring_mineru = load_mineru_token()
    if keyring_mineru:
        updated_settings = replace(base_settings, mineru_token=keyring_mineru)
        agent.settings = updated_settings

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
