from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _resolve_inside_root(root: Path, user_path: str) -> Path:
    target = (root / user_path).resolve()
    if root not in target.parents and target != root:
        raise ValueError(f"Path escapes allowed root: {target}")
    return target


def _validate_pdf_path(allowed_root: Path, path: str) -> Path | str:
    """Validate and resolve the PDF path. Returns Path on success, error string on failure."""
    if not path.strip():
        return "read_pdf 需要 path 参数。"

    root = allowed_root.resolve()
    try:
        target = _resolve_inside_root(root, path)
    except ValueError as exc:
        logger.warning("read_pdf rejected path=%r: %s", path, exc)
        return str(exc)

    if not target.exists():
        return f"文件不存在: {target}"
    if not target.is_file():
        return f"不是普通文件: {target}"
    if target.suffix.lower() != ".pdf":
        return f"不是 PDF 文件: {target.name}"

    return target


def _extract_with_mineru(
    target: Path, mineru_token: str | None, use_precision: bool, pages: str | None
) -> str:
    """Call MinerU SDK to extract PDF content. Returns markdown or error string."""
    try:
        from mineru import MinerU
    except ImportError:
        logger.error("mineru-open-sdk not installed")
        return ""

    try:
        client = MinerU(mineru_token) if mineru_token else MinerU()

        if use_precision and mineru_token:
            kwargs: dict[str, str | bool] = {}
            if pages:
                kwargs["pages"] = pages
            result = client.extract(str(target), **kwargs)
        else:
            result = client.flash_extract(str(target))

        return result.markdown or ""

    except Exception as exc:
        logger.exception("read_pdf MinerU extraction failed path=%s", target)
        error_type = type(exc).__name__
        return f"__ERROR__PDF 提取失败 ({error_type}): {exc}"


def read_pdf(
    *,
    allowed_root: Path,
    mineru_token: str | None,
    path: str,
    mode: str = "auto",
    pages: str | None = None,
    max_chars: int = 50_000,
) -> str:
    """Extract text from a PDF file using MinerU Cloud SDK.

    Args:
        allowed_root: Sandbox root directory.
        mineru_token: MinerU API token (None = flash-only mode).
        path: PDF file path relative to sandbox root.
        mode: "auto" | "flash" | "precision".
        pages: Optional page range, e.g. "1-10".
        max_chars: Maximum characters to return.

    Returns:
        Extracted Markdown text or user-friendly error message.
    """
    validated = _validate_pdf_path(allowed_root, path)
    if isinstance(validated, str):
        return validated
    target = validated

    use_precision = (mode == "precision") or (mode == "auto" and bool(mineru_token))
    markdown = _extract_with_mineru(target, mineru_token, use_precision, pages)

    if markdown.startswith("__ERROR__"):
        return markdown.removeprefix("__ERROR__")
    if not markdown:
        return "错误: mineru-open-sdk 未安装。请运行 pip install mineru-open-sdk"

    truncated = len(markdown) > max_chars
    if truncated:
        markdown = markdown[:max_chars]

    header = f"PDF: {target.name} ({len(markdown)} chars{'，已截断' if truncated else ''})"
    if not markdown.strip():
        return f"{header}\n(未提取到文本内容，PDF 可能为纯图片或已加密)"

    return f"{header}\n{markdown}"
