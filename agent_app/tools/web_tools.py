from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)


def search_web(*, tavily_api_key: str | None, query: str, max_results: int = 5) -> str:
    if not query.strip():
        return "search_web 需要 query 参数。"

    if not tavily_api_key:
        logger.warning("search_web called without TAVILY_API_KEY")
        return "未配置 TAVILY_API_KEY，当前仅完成工具骨架。配置后可返回真实搜索结果。"

    url = "https://api.tavily.com/search"
    payload: dict[str, Any] = {
        "api_key": tavily_api_key,
        "query": query,
        "max_results": max_results,
        "include_answer": True,
    }
    try:
        logger.info("search_web query=%r max_results=%d", query, max_results)
        resp = requests.post(url, json=payload, timeout=25)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.exception("search_web request failed")
        return f"搜索失败: {exc}"

    data = resp.json()
    answer = data.get("answer", "")
    results = data.get("results", [])

    lines = [f"查询: {query}"]
    if answer:
        lines.append(f"总结: {answer}")

    for item in results[:max_results]:
        title = item.get("title") or "(no title)"
        link = item.get("url") or ""
        snippet = item.get("content") or ""
        lines.append(f"- {title}\n  {link}\n  {snippet[:180]}")

    return "\n".join(lines)
