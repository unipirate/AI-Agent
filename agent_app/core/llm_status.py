from __future__ import annotations

import logging
from dataclasses import replace

import requests
from agent_app.config import Settings

logger = logging.getLogger(__name__)


def _parse_models_list(data: dict[str, object]) -> list[str]:
    items = data.get("data", [])
    if not isinstance(items, list):
        return []
    return [item.get("id", "") for item in items if isinstance(item, dict) and item.get("id")]


def fetch_server_models(base_url: str, timeout: int = 3) -> list[str]:
    """List models advertised by the server (may include cached installs, not only loaded)."""
    base = base_url.rstrip("/")
    resp = requests.get(f"{base}/models", timeout=timeout)
    resp.raise_for_status()
    return _parse_models_list(resp.json())


def fetch_active_server_models(base_url: str, timeout: int = 3) -> list[str]:
    """Prefer in-memory / running models; fall back to full catalog when unsupported."""
    base = base_url.rstrip("/")

    try:
        resp = requests.get(f"{base}/models/loaded", timeout=timeout)
        if resp.ok:
            models = _parse_models_list(resp.json())
            if models:
                logger.debug("Active models from /models/loaded at %s: %d", base_url, len(models))
                return models
    except requests.RequestException:
        logger.debug("No /models/loaded endpoint at %s", base_url)

    root = base[: -len("/v1")] if base.endswith("/v1") else base
    try:
        resp = requests.get(f"{root}/api/ps", timeout=timeout)
        if resp.ok:
            payload = resp.json()
            models = [
                item.get("name", "") for item in payload.get("models", []) if item.get("name")
            ]
            if models:
                logger.debug("Active models from Ollama /api/ps at %s: %d", base_url, len(models))
                return models
    except requests.RequestException:
        logger.debug("No Ollama /api/ps endpoint at %s", base_url)

    return fetch_server_models(base_url, timeout=timeout)


def resolve_local_llm(settings: Settings) -> tuple[Settings, str]:
    """Pick the model currently served on the local endpoint."""
    if not settings.llm_base_url:
        return settings, _cloud_status(settings)

    try:
        model_ids = fetch_server_models(settings.llm_base_url)
    except requests.RequestException:
        logger.exception("Failed to connect to local LLM at %s", settings.llm_base_url)
        return settings, (
            f"无法连接 {settings.llm_base_url}。"
            "请先启动本地 LLM 服务（mlx_lm / Ollama / LM Studio）。"
        )

    if not model_ids:
        return settings, f"已连接 {settings.llm_base_url}，但未返回可用模型。"

    active = model_ids[0]
    configured = settings.llm_model.strip()

    if len(model_ids) == 1:
        if configured and configured != active:
            resolved = replace(settings, llm_model=active)
            return resolved, (
                f"已自动切换到当前运行的模型: {active}（配置中为 {configured}，换模型后无需改配置）"
            )
        resolved = replace(settings, llm_model=active)
        return resolved, f"已连接本地 LLM，当前模型: {active}"

    if configured and configured in model_ids:
        return settings, f"已连接本地 LLM，使用指定模型: {configured}"

    resolved = replace(settings, llm_model=active)
    return resolved, f"已连接本地 LLM，自动选择: {active}（可用: {', '.join(model_ids)}）"


def _cloud_status(settings: Settings) -> str:
    if settings.llm_api_key:
        if settings.llm_provider_id:
            return f"云端 LLM · {settings.llm_provider_id} ({settings.llm_model or '默认'})"
        return f"云端 LLM ({settings.llm_model})"
    return "规则模式（未配置 LLM）"


def describe_llm_backend(settings: Settings) -> str:
    if settings.llm_base_url:
        return f"本地 LLM ({settings.llm_base_url})"
    return _cloud_status(settings)


def check_llm_connection(settings: Settings) -> str:
    resolved_settings, message = resolve_local_llm(settings)
    return message
