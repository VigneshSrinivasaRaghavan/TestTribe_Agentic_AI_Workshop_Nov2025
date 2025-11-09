"""
Core LLM client (thin wrapper).

This module provides a single high-level function `chat(messages)` used by
agents to call an LLM provider and return assistant text. The implementation
keeps things intentionally small for Day‑1 teaching: it supports two providers
(`ollama` and `openai`) via a provider switch driven by environment
variables (loaded from `.env`).

Design goals (Day-1):
- Minimal, readable code students can extend.
- Provider-agnostic caller API: callers pass OpenAI-style `messages` and get
  back assistant text.
- Avoid advanced per-model knobs (e.g., `temperature`) to reduce student
  friction and 400/compatibility errors during demos.

Environment configuration (loaded via `python-dotenv`):
- `PROVIDER` (default: "ollama") — which provider to use. Values: "ollama" or "openai".
- `MODEL` — model id to request (e.g. `gpt-4o-mini`, `gpt-5-nano`, or `mistral:latest`).
- `OLLAMA_HOST` — base URL for local Ollama (default: `http://localhost:11434`).
- `OPENAI_API_KEY` — required when `PROVIDER=openai`.
- `LLM_TIMEOUT_S` — request timeout in seconds (default: 60).

Usage (examples):

```py
from src.core import chat

messages = [
    {"role": "system", "content": "You are a helpful QA assistant."},
    {"role": "user", "content": "Summarize the requirement"},
]
out = chat(messages)
```

The function returns the assistant's text (string). For JSON outputs the
caller should validate/parse the returned text (see `src.core.utils` helpers
for parsing and cleanup).
"""

from __future__ import annotations
import os
from typing import List, Dict
import httpx
from dotenv import load_dotenv

load_dotenv(override=True)

PROVIDER = (os.getenv("PROVIDER") or "ollama").strip().lower()
MODEL = (os.getenv("MODEL") or "mistral:latest").strip()
OLLAMA_HOST = (os.getenv("OLLAMA_HOST") or "http://localhost:11434").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""
TIMEOUT_S = int(os.getenv("LLM_TIMEOUT_S") or "60")
# No temperature handling for Day-1: keep payloads simple and compatible.
LLM_TEMPERATURE = None

Message = Dict[str, str]


def chat(messages: List[Message], timeout: int = TIMEOUT_S) -> str:
    """Send `messages` to the configured LLM provider and return assistant text.

    This thin, provider-agnostic helper keeps the interface simple for Day-1
    teaching: callers pass OpenAI-style `messages` and get back the assistant's
    `content` string. Validation is intentionally minimal to keep code readable.

    Args:
        messages: List of message dicts with `role` and `content`.
        timeout: Request timeout in seconds.

    Returns:
        str: Assistant text returned by the selected provider.

    Raises:
        ValueError: If `messages` is empty or not a list.
        RuntimeError: For provider-specific failures (missing keys, empty replies).
        NotImplementedError: If `PROVIDER` is not supported.
    """
    if not isinstance(messages, list) or not messages:
        raise ValueError(
            "messages must be a non-empty list of {'role','content'} dicts."
        )

    if PROVIDER == "ollama":
        url = f"{OLLAMA_HOST.rstrip('/')}/api/chat"
        payload = {"model": MODEL, "messages": messages, "stream": False}
        with httpx.Client(timeout=timeout) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            msg = (data.get("message") or {}).get("content")
            if not msg:
                raise RuntimeError(
                    "Ollama returned empty content. Check model and host."
                )
            return msg

    elif PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is missing but PROVIDER=openai.")
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {"model": MODEL, "messages": messages, "temperature": 0}
        with httpx.Client(timeout=timeout) as client:
            r = client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError("OpenAI returned no choices. Check model and key.")
            return (choices[0].get("message") or {}).get("content") or ""

    else:
        raise NotImplementedError("Unsupported PROVIDER. Use 'ollama' or 'openai'.")
