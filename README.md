# Local AI Agent (Desktop)

A Python desktop AI agent scaffold with:

- Native desktop window (Tkinter), no browser required
- Agent core with pluggable tools
- File organization tool (safe by default)
- Web search tool (API-backed with graceful fallback)
- Confirmation flow for risky actions

## Quick start

1. Create and activate virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment:

   ```bash
   cp .env.example .env
   ```

   Then edit `.env` and set LLM provider (see **Local LLM** below).
   Optional: `TAVILY_API_KEY` for web search tool.

4. Run desktop app:

   ```bash
   python -m agent_app.main
   ```

## Project layout

```text
agent_app/
  main.py            # app entrypoint
  app.py             # tkinter UI
  config.py          # env and runtime config
  models.py          # shared dataclasses
  core/
    agent.py         # orchestrates planning + tool calls
    llm.py           # LLM client abstraction
  tools/
    file_tools.py    # local document utilities
    web_tools.py     # web search utility
```

## Current behavior (MVP scaffold)

- You can chat with the agent in a native desktop window.
- Agent chooses between:
  - normal reply
  - list files in allowed directory
  - web search
  - move file (requires explicit confirmation in UI)
- If no LLM is configured, it falls back to simple rule-based planning.

## Local LLM (mlx_lm / Ollama / LM Studio)

This project uses the OpenAI Python SDK with a configurable `base_url`.
Most local runtimes expose an OpenAI-compatible API at `/v1`.

### mlx_lm server (Apple Silicon)

1. Start your model server:

   ```bash
   python -m mlx_lm server \
     --model /Users/gaofuquan/.cache/modelscope/hub/models/okwinds/Qwen2.5-Coder-7B-Instruct-MLX-4bit \
     --port 8080
   ```

2. Set `.env` (port fixed, model auto-detected):

   ```env
   LLM_BASE_URL=http://localhost:8080/v1
   ```

   If you run different models one at a time on the same port, you do **not**
   need to edit `.env` each time. Restart Agent after switching models.

3. Verify endpoint:

   ```bash
   curl http://localhost:8080/v1/models
   ```

### Ollama

1. Install and start Ollama: https://ollama.com
2. Pull a model:

   ```bash
   ollama pull qwen2.5:7b
   ```

3. Set `.env`:

   ```env
   LLM_BASE_URL=http://localhost:11434/v1
   LLM_MODEL=qwen2.5:7b
   ```

4. Verify endpoint:

   ```bash
   curl http://localhost:11434/v1/models
   ```

### LM Studio

1. Load a model in LM Studio
2. Start the local server (default port `1234`)
3. Set `.env`:

   ```env
   LLM_BASE_URL=http://localhost:1234/v1
   LLM_MODEL=<exact model id shown in LM Studio>
   ```

### Cloud OpenAI (alternative)

Leave `LLM_BASE_URL` empty and set:

```env
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
```

## Safety defaults

- File operations are restricted to `AGENT_ALLOWED_ROOT`.
- Risky actions require approve/reject.
- The move-file tool is available only through confirmation flow.

## Next suggested steps

- Add markdown/PDF parsing to build a document organizer pipeline.
- Add local vector store for knowledge memory.
- Swap rule fallback with structured function calling for stronger planning quality.
