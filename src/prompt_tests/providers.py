"""Model providers. Every provider turns (system, prompt) into text plus latency."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx

_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_URL = _host if _host.startswith("http") else f"http://{_host}"


@dataclass
class Completion:
    text: str
    latency_s: float
    cached: bool = False


class Provider(Protocol):
    name: str

    def complete(
        self, prompt: str, system: str | None = None, temperature: float = 0.0
    ) -> Completion: ...


class OllamaProvider:
    def __init__(self, model: str, url: str = OLLAMA_URL):
        self.model, self.url = model, url
        self.name = f"ollama:{model}"

    def complete(self, prompt, system=None, temperature=0.0) -> Completion:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        t0 = time.time()
        resp = httpx.post(
            f"{self.url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=600,
        )
        resp.raise_for_status()
        return Completion(resp.json()["message"]["content"], time.time() - t0)


class OpenAICompatProvider:
    """Any OpenAI-compatible server: llama.cpp, vLLM, LM Studio, LocalAI."""

    def __init__(self, model: str, base_url: str, api_key: str = "none"):
        self.model, self.base_url, self.api_key = model, base_url.rstrip("/"), api_key
        self.name = f"openai:{model}"

    def complete(self, prompt, system=None, temperature=0.0) -> Completion:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        t0 = time.time()
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": messages, "temperature": temperature},
            timeout=600,
        )
        resp.raise_for_status()
        return Completion(resp.json()["choices"][0]["message"]["content"], time.time() - t0)


def make_provider(spec: str) -> Provider:
    """'ollama:qwen2.5:3b' or 'openai:model@http://localhost:8080/v1'."""
    kind, _, rest = spec.partition(":")
    if kind == "ollama":
        return OllamaProvider(rest)
    if kind == "openai":
        model, _, url = rest.partition("@")
        return OpenAICompatProvider(model, url or "http://localhost:8080/v1")
    raise ValueError(f"unknown provider spec {spec!r}; use ollama:<model> or openai:<model>@<url>")


class CachedProvider:
    """Wraps a provider with a SQLite response cache, so you can iterate on scorers and
    rubrics without paying for generation again. Temperature 0 only by default."""

    def __init__(self, inner: Provider, path: Path):
        self.inner, self.name = inner, inner.name
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, text TEXT, latency REAL)"
        )
        self.lock = threading.Lock()

    def complete(self, prompt, system=None, temperature=0.0, sample: int = 0) -> Completion:
        key = hashlib.sha256(
            json.dumps([self.name, system, prompt, temperature, sample]).encode()
        ).hexdigest()
        with self.lock:
            row = self.db.execute(
                "SELECT text, latency FROM cache WHERE key = ?", (key,)
            ).fetchone()
        if row:
            return Completion(row[0], row[1], cached=True)
        out = self.inner.complete(prompt, system, temperature)
        with self.lock:
            self.db.execute(
                "INSERT OR REPLACE INTO cache VALUES (?,?,?)", (key, out.text, out.latency_s)
            )
            self.db.commit()
        return out
