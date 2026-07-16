from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import uuid4

import requests

from agent_app.config import Settings
from agent_app.core.llm_status import (
    fetch_active_server_models,
    fetch_server_models,
    resolve_local_llm,
)
from agent_app.secrets import load_api_key, save_api_key

logger = logging.getLogger(__name__)

PROFILES_DIR = Path.home() / ".ai-agent"
PROFILES_PATH = PROFILES_DIR / "profiles.json"

LOCAL_PROVIDER_IDS = frozenset({"local_mlx", "local_ollama", "local_lmstudio"})


@dataclass(frozen=True)
class ProviderPreset:
    provider_id: str
    display_name: str
    base_url: str | None
    default_models: tuple[str, ...]
    env_key_var: str | None = None
    uses_anthropic: bool = False


PROVIDER_PRESETS: dict[str, ProviderPreset] = {
    "local_mlx": ProviderPreset(
        provider_id="local_mlx",
        display_name="本地 · mlx_lm",
        base_url="http://localhost:8080/v1",
        default_models=(),
    ),
    "local_ollama": ProviderPreset(
        provider_id="local_ollama",
        display_name="本地 · Ollama",
        base_url="http://localhost:11434/v1",
        default_models=(),
    ),
    "local_lmstudio": ProviderPreset(
        provider_id="local_lmstudio",
        display_name="本地 · LM Studio",
        base_url="http://localhost:1234/v1",
        default_models=(),
    ),
    "deepseek": ProviderPreset(
        provider_id="deepseek",
        display_name="DeepSeek",
        base_url="https://api.deepseek.com/v1",
        default_models=("deepseek-chat", "deepseek-reasoner"),
        env_key_var="DEEPSEEK_API_KEY",
    ),
    "openai": ProviderPreset(
        provider_id="openai",
        display_name="ChatGPT",
        base_url=None,
        default_models=("gpt-4o", "gpt-4o-mini"),
        env_key_var="OPENAI_API_KEY",
    ),
    "gemini": ProviderPreset(
        provider_id="gemini",
        display_name="Gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        default_models=("gemini-2.0-flash", "gemini-1.5-pro"),
        env_key_var="GEMINI_API_KEY",
    ),
    "claude": ProviderPreset(
        provider_id="claude",
        display_name="Claude",
        base_url=None,
        default_models=("claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022"),
        env_key_var="ANTHROPIC_API_KEY",
        uses_anthropic=True,
    ),
    "custom": ProviderPreset(
        provider_id="custom",
        display_name="自定义",
        base_url="",
        default_models=(),
        env_key_var="LLM_API_KEY",
    ),
}

PROVIDER_ORDER = (
    "local_mlx",
    "local_ollama",
    "local_lmstudio",
    "deepseek",
    "openai",
    "gemini",
    "claude",
    "custom",
)


@dataclass
class LlmProfile:
    id: str
    provider_id: str
    display_name: str
    base_url: str | None
    model: str

    @classmethod
    def from_preset(cls, provider_id: str) -> LlmProfile:
        preset = PROVIDER_PRESETS[provider_id]
        default_model = preset.default_models[0] if preset.default_models else ""
        return cls(
            id=str(uuid4()),
            provider_id=provider_id,
            display_name=preset.display_name,
            base_url=preset.base_url,
            model=default_model,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> LlmProfile:
        return cls(
            id=str(data["id"]),
            provider_id=str(data["provider_id"]),
            display_name=str(data["display_name"]),
            base_url=str(data["base_url"]) if data.get("base_url") else None,
            model=str(data.get("model", "")),
        )


@dataclass
class ProfileStore:
    active_profile_id: str
    profiles: list[LlmProfile] = field(default_factory=list)

    def get_active(self) -> LlmProfile | None:
        for profile in self.profiles:
            if profile.id == self.active_profile_id:
                return profile
        return self.profiles[0] if self.profiles else None

    def upsert(self, profile: LlmProfile) -> None:
        for index, existing in enumerate(self.profiles):
            if existing.id == profile.id:
                self.profiles[index] = profile
                self.active_profile_id = profile.id
                return
        self.profiles.append(profile)
        self.active_profile_id = profile.id


def is_local_provider(provider_id: str) -> bool:
    return provider_id in LOCAL_PROVIDER_IDS


@dataclass(frozen=True)
class DiscoveredLocalModel:
    provider_id: str
    base_url: str
    model_id: str
    label: str


def discover_running_local_models(timeout: int = 2) -> list[DiscoveredLocalModel]:
    """Probe known local endpoints and return actively loaded models when supported."""
    discovered: list[DiscoveredLocalModel] = []
    for provider_id in sorted(LOCAL_PROVIDER_IDS):
        preset = PROVIDER_PRESETS[provider_id]
        base_url = preset.base_url
        if not base_url:
            continue
        try:
            model_ids = fetch_active_server_models(base_url, timeout=timeout)
        except requests.RequestException:
            logger.debug(
                "Local LLM endpoint unavailable provider=%s url=%s",
                provider_id,
                base_url,
            )
            continue
        service_name = preset.display_name.replace("本地 · ", "")
        for model_id in model_ids:
            discovered.append(
                DiscoveredLocalModel(
                    provider_id=provider_id,
                    base_url=base_url,
                    model_id=model_id,
                    label=f"{service_name} · {model_id}",
                )
            )
    discovered.sort(key=lambda item: item.label.lower())
    return discovered


def preset_for(provider_id: str) -> ProviderPreset:
    return PROVIDER_PRESETS[provider_id]


def resolve_api_key(profile: LlmProfile) -> str | None:
    stored = load_api_key(profile.id)
    if stored:
        return stored

    preset = preset_for(profile.provider_id)
    if preset.env_key_var:
        env_value = (
            os.getenv(preset.env_key_var) or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        )
        if env_value:
            return env_value

    if is_local_provider(profile.provider_id):
        return "local"
    return None


def profile_to_settings(profile: LlmProfile, base: Settings | None = None) -> Settings:
    preset = preset_for(profile.provider_id)
    api_key = resolve_api_key(profile)

    if is_local_provider(profile.provider_id):
        base_url = profile.base_url or preset.base_url
    elif profile.provider_id == "custom":
        base_url = profile.base_url or None
    elif preset.uses_anthropic:
        base_url = None
    else:
        base_url = preset.base_url

    allowed_root = (
        base.allowed_root
        if base
        else Path(os.getenv("AGENT_ALLOWED_ROOT", "~/Documents/AI-Agent-Sandbox"))
        .expanduser()
        .resolve()
    )
    tavily = base.tavily_api_key if base else os.getenv("TAVILY_API_KEY") or None

    return Settings(
        llm_base_url=base_url,
        llm_api_key=api_key,
        llm_model=profile.model,
        llm_provider_id=profile.provider_id,
        tavily_api_key=tavily,
        allowed_root=allowed_root,
    )


def profile_summary(profile: LlmProfile) -> str:
    preset = preset_for(profile.provider_id)
    model = profile.model or "（自动探测）"
    return f"{preset.display_name} · {model}"


def load_profile_store() -> ProfileStore | None:
    if not PROFILES_PATH.exists():
        return None
    data = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
    profiles = [LlmProfile.from_dict(item) for item in data.get("profiles", [])]
    if not profiles:
        return None
    return ProfileStore(
        active_profile_id=data.get("active_profile_id") or profiles[0].id,
        profiles=profiles,
    )


def save_profile_store(store: ProfileStore) -> None:
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "active_profile_id": store.active_profile_id,
        "profiles": [profile.to_dict() for profile in store.profiles],
    }
    PROFILES_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _infer_local_provider(base_url: str) -> str:
    if ":11434" in base_url:
        return "local_ollama"
    if ":1234" in base_url:
        return "local_lmstudio"
    if ":8080" in base_url:
        return "local_mlx"
    return "custom"


def bootstrap_from_env(settings: Settings) -> ProfileStore:
    if settings.llm_base_url:
        provider_id = _infer_local_provider(settings.llm_base_url)
        preset = preset_for(provider_id)
        profile = LlmProfile(
            id=str(uuid4()),
            provider_id=provider_id,
            display_name=preset.display_name if provider_id != "custom" else "自定义",
            base_url=settings.llm_base_url,
            model=settings.llm_model,
        )
    elif settings.llm_api_key:
        profile = LlmProfile(
            id=str(uuid4()),
            provider_id="openai",
            display_name="ChatGPT",
            base_url=None,
            model=settings.llm_model or "gpt-4o-mini",
        )
        save_api_key(profile.id, settings.llm_api_key)
    else:
        profile = LlmProfile.from_preset("local_mlx")

    store = ProfileStore(active_profile_id=profile.id, profiles=[profile])
    save_profile_store(store)
    return store


def provider_id_from_display(display_name: str) -> str:
    for provider_id in PROVIDER_ORDER:
        if PROVIDER_PRESETS[provider_id].display_name == display_name:
            return provider_id
    return "custom"


def draft_profile(
    *,
    profile_id: str | None,
    provider_id: str,
    display_name: str,
    base_url: str,
    model: str,
) -> LlmProfile:
    preset = preset_for(provider_id)
    resolved_base_url: str | None = base_url.strip() or (preset.base_url or "")
    if provider_id != "custom" and not is_local_provider(provider_id) and not preset.uses_anthropic:
        resolved_base_url = preset.base_url or ""
    if preset.uses_anthropic:
        resolved_base_url = None
    elif is_local_provider(provider_id) or provider_id == "custom":
        resolved_base_url = base_url.strip() or (preset.base_url or None)
    return LlmProfile(
        id=profile_id or str(uuid4()),
        provider_id=provider_id,
        display_name=display_name.strip() or preset.display_name,
        base_url=resolved_base_url,
        model=model.strip(),
    )


def save_active_profile(
    store: ProfileStore,
    profile: LlmProfile,
    new_api_key: str | None = None,
) -> ProfileStore:
    apply_profile_key(profile.id, new_api_key)
    store.upsert(profile)
    save_profile_store(store)
    return store


def apply_profile_key(profile_id: str, new_key: str | None) -> None:
    if new_key and new_key.strip():
        save_api_key(profile_id, new_key.strip())


def test_profile_connection(
    profile: LlmProfile, api_key_override: str | None = None
) -> tuple[bool, str]:
    preset = preset_for(profile.provider_id)
    api_key = (
        api_key_override.strip()
        if api_key_override and api_key_override.strip()
        else resolve_api_key(profile)
    )

    if preset.uses_anthropic:
        if not api_key:
            return False, "请先填写 Anthropic API Key。"
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
            model = profile.model or preset.default_models[0]
            client.messages.create(
                model=model,
                max_tokens=16,
                messages=[{"role": "user", "content": "ping"}],
            )
            return True, f"已连接 Claude，模型: {model}"
        except Exception:
            logger.exception("Claude connection test failed")
            return False, "Claude 连接失败，请检查 API Key 与模型名称。"

    if is_local_provider(profile.provider_id):
        base_url = profile.base_url or preset.base_url
        if not base_url:
            return False, "请填写 base_url。"
        try:
            model_ids = fetch_server_models(base_url)
        except requests.RequestException:
            logger.exception("Local LLM connection test failed base_url=%s", base_url)
            return False, f"无法连接 {base_url}，请确认本地服务已启动。"
        if not model_ids:
            return False, f"已连接 {base_url}，但未返回可用模型。"
        return True, f"已连接本地服务，可用模型: {', '.join(model_ids[:5])}"

    if profile.provider_id == "custom":
        if not profile.base_url:
            return False, "请填写 base_url。"
        if not api_key:
            return False, "请填写 API Key。"

    if not api_key:
        return False, f"请先填写 API Key，或在 .env 中设置 {preset.env_key_var or 'LLM_API_KEY'}。"

    base_url = profile.base_url or preset.base_url
    try:
        from openai import OpenAI

        openai_client: OpenAI
        if base_url:
            openai_client = OpenAI(base_url=base_url, api_key=api_key or "local")
            try:
                model_ids = fetch_server_models(base_url)
                if model_ids:
                    return True, f"已连接，可用模型: {', '.join(model_ids[:5])}"
            except requests.RequestException:
                logger.warning(
                    "GET /models failed for %s; falling back to completion probe", base_url
                )
        else:
            openai_client = OpenAI(api_key=api_key)

        model = profile.model or (
            preset.default_models[0] if preset.default_models else "gpt-4o-mini"
        )
        openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        return True, f"已连接，模型: {model}"
    except Exception:
        logger.exception("Cloud LLM connection test failed provider=%s", profile.provider_id)
        return False, "连接失败，请检查 API Key、base_url 与网络。"


def resolve_profile_llm(profile: LlmProfile, base: Settings) -> tuple[Settings, str]:
    settings = profile_to_settings(profile, base)
    if is_local_provider(profile.provider_id):
        return resolve_local_llm(settings)
    if profile.provider_id == "custom" and settings.llm_base_url:
        return resolve_local_llm(settings)

    preset = preset_for(profile.provider_id)
    if preset.uses_anthropic and not settings.llm_api_key:
        return settings, "Claude 未配置 API Key，请在设置中填写。"
    if not settings.llm_api_key:
        return settings, f"{preset.display_name} 未配置 API Key，将使用规则模式。"
    model = settings.llm_model or (preset.default_models[0] if preset.default_models else "")
    return settings, f"已选择 {preset.display_name}，模型: {model or '默认'}"
