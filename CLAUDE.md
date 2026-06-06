# CLAUDE.md

## 1. Tech Stack
- Language: Python 3.13+ (Strict Type Hinting required)
- Desktop UI: Tkinter (ttk)
- LLM clients: OpenAI-compatible SDK (`openai`), Anthropic SDK (`anthropic`)
- Config: `python-dotenv`, `keyring`, `~/.ai-agent/profiles.json`
- HTTP: `requests`

## 2. Build & Test Commands
- Create venv: `python3 -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt`
- Run app: `python -m agent_app.main`

## 3. Code Style & Design Patterns
- Use `@dataclass` for settings, state, and shared data models.
- Module boundaries:
  - `app.py`, `ui/` — Tkinter UI only (widgets, events, display). Use `ui/background.py` for off-thread work.
  - `core/` — agent orchestration and LLM planners (pure Python, NO UI imports).
  - `llm_profiles.py`, `secrets.py`, `config.py`, `log_config.py` — configuration, persistence, logging setup.
  - `errors.py` — user-facing error formatting (no secret leakage).
  - `tools/` — tool implementations.
- UI must call business helpers (e.g. `save_active_profile`) instead of duplicating persistence or connection logic.
- **Concurrency & UI Thread**: Always isolate LLM calls and tools from the UI thread using `threading` or `concurrent.futures`.
- **Diagnostics**: Use Python's built-in `logging` module. Avoid raw `print()` statements. Log tool inputs/outputs and LLM errors clearly.

> **Current tool dispatch** (until BaseTool refactor): `core/agent._run_tool()` uses if-elif branches based on tool name. New tools can be added by extending that function and updating the planner's tool schema in `core/llm.py`.

**Key resolution order** (implemented in `llm_profiles.resolve_api_key()`):
1. System keyring (via `secrets.load_api_key(profile_id)`)
2. Per-provider `.env` variable (e.g., `DEEPSEEK_API_KEY`)
3. Legacy `LLM_API_KEY` or `OPENAI_API_KEY`
4. Return `None` — cloud providers become unavailable; local providers default to `"local"`

> **未来**: 计划在 keyring 与 .env 之间插入加密文件层，见 Roadmap。

## 4. Code Style Details
- **Type annotations**: All functions (especially public APIs) must have explicit type annotations (use Python 3.13+ `type` and `typing`).
- **Naming**: Classes `PascalCase`, functions/variables `snake_case`, constants `UPPER_CASE`.

## 5. STRICT Constraints (Guardrails)
- DO NOT refactor unrelated code when fixing a specific bug.
- DO NOT add external dependencies without explicit confirmation.
- NEVER delete existing comments or documentation unless explicitly asked.
- Maintain existing architecture boundaries (do not mix UI with business logic).
- **NO UI Control in Background**: NEVER read from or write to Tkinter widgets or UI variables (e.g., `StringVar.set()`, `btn.config()`) from a non-GUI thread. Update UI states safely via `root.after()`.
- **File Sandbox Enforcement**: All file operations in `tools/` MUST validate that target paths remain strictly inside `AGENT_ALLOWED_ROOT` (default `~/Documents/AI-Agent-Sandbox`) using `Path.resolve()` to prevent directory traversal.
- NEVER execute blocking HTTP requests or LLM streaming directly on the Tkinter main thread.
- NEVER hardcode API keys. Store secrets in the system keychain via `secrets.py`. `~/.ai-agent/profiles.json` holds profile metadata only (no keys). Optional developer fallback: `.env` env vars.

## 6. Approved Runtime Dependencies
- `openai`, `requests`, `python-dotenv` — existing baseline
- `anthropic`, `keyring` — approved for Claude adapter and API key storage (model-switch feature)

## 7. Project Layout
agent_app/
├── main.py              # entry point
├── app.py               # main app class (UI init, event wiring)
├── config.py            # .env loading → Settings dataclass
├── llm_profiles.py      # provider presets, profile CRUD, profiles.json
├── secrets.py           # keyring read/write, mask_api_key
├── errors.py            # user-facing error formatting (no secret leakage)
├── log_config.py        # logging setup
├── models.py            # shared dataclasses (Plan, AgentReply, etc.)
├── ui/
│   ├── background.py    # BackgroundRunner (ThreadPoolExecutor + root.after)
│   └── model_switch.py  # ModelSwitchDialog, ModelSwitcherBar
├── core/
│   ├── agent.py         # Agent orchestration, tool dispatch
│   ├── llm.py           # OpenAICompatPlanner, AnthropicPlanner, LLMPlanner
│   └── llm_status.py    # local model detection, connection status
└── tools/
    ├── file_tools.py    # list_files, move_file
    └── web_tools.py     # search_web (Tavily)

## 8. Environment Variables (`.env.example`)
- `LLM_BASE_URL` — local LLM endpoint (e.g. `http://localhost:8080/v1`)
- `LLM_MODEL` — model name override (optional, auto-detected for local)
- `LLM_API_KEY` / `OPENAI_API_KEY` — legacy fallback (prefer keyring via UI)
- `DEEPSEEK_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` — per-provider `.env` fallback
- `TAVILY_API_KEY` — web search tool
- `AGENT_ALLOWED_ROOT` — sandbox directory for file tools (default `~/Documents/AI-Agent-Sandbox`)

## 9. Error Handling & User Feedback
- All LLM calls, tool executions, and I/O operations must be wrapped in `try/except`.
- **UI Error Propagation**: Background thread exceptions must be caught, logged, and safely propagated to the UI thread via `root.after(0, callback)` to alert the user.
- Never silently swallow exceptions. At a minimum, call `logger.error()` or `logger.exception()`.

## 10. Security
- **Secret Masking**: NEVER print full API keys or Bearer tokens in logs, UI output, or exception messages. Show only last four characters (e.g. `****xYzW`) via `mask_api_key()`.
- **Error Filtering**: Leverage `errors.py` to ensure user-facing UI messages (chat area, status bar) format raw system exceptions into safe, readable text that cannot leak secrets or operational details.
---

## Roadmap (未实现，计划目标)

以下条目当前代码尚未实现。实现后移入对应正式章节。

- **Docstrings**: Core modules, tool functions, and complex logic should have Google-style docstrings. (→ 移入 §4)
- **Testing**: `pytest tests/` — 建立测试目录和基本用例。(→ 新增 §)
- **Linter & Formatter**: `black agent_app/ && isort agent_app/` — 统一格式化。(→ 新增 §)
- **Type Check**: `mypy agent_app/ --strict` — 静态类型验证。(→ 新增 §)
- **LOG_LEVEL 环境变量**: 支持 `LOG_LEVEL=DEBUG` 动态控制日志级别。(→ 移入 §8)
- **BaseTool 抽象**: 工具统一继承 `BaseTool`，实现 `name` / `description` / `execute`，并通过 registry 注册。(→ 替代当前 `_run_tool` 分支)
- **Mock LLM for Testing**: 编写 mock planner 类，离线测试 UI 行为、不消耗 API 额度。
- **get_key_source()**: 在 `secrets.py` 中增加函数，返回 key 来源（keyring / .env / 未配置）用于调试。
- **Secret `__repr__` masking**: 含密钥的 dataclass/类重写 `__repr__` 返回占位符。
- **Streaming**: 在后台线程中使用 `stream=True`，通过 `root.after()` 分片推送至 UI。(→ 移入 §3)
- **`_safe_call` helper**: 在 `BackgroundRunner` 中封装 `root.after(0, ...)` 简化 UI 回调。(→ 移入 `ui/background.py`)
- **加密文件 Fallback 层**: 在 `secrets.py` 的 key 解析链路中增加中间层：keyring → 加密文件（`~/.ai-agent/secrets.enc`）→ `.env`。适用于无系统钥匙串的环境（Docker / 无桌面 Linux）。加密方案待定（Fernet + 机器派生密钥 或 用户主密码）。实现后更新 §3 Key resolution order 为三层。(→ 移入 §10)
