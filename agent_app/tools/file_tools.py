from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _resolve_inside_root(root: Path, user_path: str) -> Path:
    target = (root / user_path).resolve()
    if root not in target.parents and target != root:
        raise ValueError(f"Path escapes allowed root: {target}")
    return target


def list_files(*, allowed_root: Path, path: str = ".", max_items: int = 50) -> str:
    root = allowed_root.resolve()
    try:
        target = _resolve_inside_root(root, path)
    except ValueError as exc:
        logger.warning("list_files rejected path=%r: %s", path, exc)
        return str(exc)

    if not target.exists():
        return f"路径不存在: {target}"
    if not target.is_dir():
        return f"不是目录: {target}"

    try:
        items: list[str] = []
        for child in sorted(target.iterdir(), key=lambda p: p.name.lower()):
            icon = "DIR" if child.is_dir() else "FILE"
            items.append(f"[{icon}] {child.name}")
            if len(items) >= max_items:
                items.append("... (truncated)")
                break
    except OSError as exc:
        logger.exception("list_files failed path=%s", target)
        return f"读取目录失败: {type(exc).__name__}"

    header = f"目录: {target}\n共显示 {min(len(items), max_items)} 条"
    body = "\n".join(items) if items else "(empty)"
    return f"{header}\n{body}"


def move_file(*, allowed_root: Path, src: str, dst: str) -> str:
    if not src.strip() or not dst.strip():
        return "move_file 需要 src 和 dst 参数。"

    root = allowed_root.resolve()
    try:
        src_path = _resolve_inside_root(root, src)
        dst_path = _resolve_inside_root(root, dst)
    except ValueError as exc:
        logger.warning("move_file rejected src=%r dst=%r: %s", src, dst, exc)
        return str(exc)

    if not src_path.exists():
        return f"源文件不存在: {src_path}"

    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        src_path.rename(dst_path)
    except OSError as exc:
        logger.exception("move_file failed src=%s dst=%s", src_path, dst_path)
        return f"移动文件失败: {type(exc).__name__}"

    logger.info("move_file ok src=%s dst=%s", src_path, dst_path)
    return f"已移动: {src_path} -> {dst_path}"
