from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    llm_base_url: str | None
    llm_api_key: str | None
    llm_model: str
    llm_provider_id: str | None
    tavily_api_key: str | None
    allowed_root: Path


def load_settings() -> Settings:
    load_dotenv()
    root = os.getenv("AGENT_ALLOWED_ROOT", "~/Documents/AI-Agent-Sandbox")

    # Local LLM (Ollama / LM Studio) usually exposes an OpenAI-compatible /v1 endpoint.
    base_url = os.getenv("LLM_BASE_URL") or None
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or None
    model = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or ""
    if not base_url and api_key and not model:
        model = "gpt-4o-mini"

    return Settings(
        llm_base_url=base_url,
        llm_api_key=api_key,
        llm_model=model,
        llm_provider_id=None,
        tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
        allowed_root=Path(root).expanduser().resolve(),
    )
