from __future__ import annotations

import logging

import keyring
from keyring.errors import KeyringError

logger = logging.getLogger(__name__)

SERVICE_NAME = "ai-agent"


def save_api_key(profile_id: str, api_key: str) -> None:
    keyring.set_password(SERVICE_NAME, profile_id, api_key)


def load_api_key(profile_id: str) -> str | None:
    try:
        return keyring.get_password(SERVICE_NAME, profile_id)
    except KeyringError as exc:
        logger.warning("Failed to load API key for profile=%s: %s", profile_id, exc)
        return None


def delete_api_key(profile_id: str) -> None:
    try:
        keyring.delete_password(SERVICE_NAME, profile_id)
    except KeyringError as exc:
        logger.warning("Failed to delete API key for profile=%s: %s", profile_id, exc)


def has_api_key(profile_id: str) -> bool:
    return bool(load_api_key(profile_id))


def mask_api_key(key: str | None) -> str:
    if not key:
        return "未设置"
    if len(key) <= 4:
        return "****"
    return f"****{key[-4:]}"
