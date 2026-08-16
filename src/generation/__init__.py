"""LLM provider — interface and Ollama implementation for Qwen (spec §78).

Supports swappable backends: Ollama (default for CPU), Transformers (GPU).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol, runtime_checkable

import httpx

from src.config import get_settings

logger = logging.getLogger(__name__)


@runtime_checkable
class LanguageModel(Protocol):
    """Interface for language model providers (spec §78)."""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        ...

    @property
    def model_name(self) -> str:
        ...


class OllamaProvider:
    """LLM provider using Ollama API for local model inference.

    Uses the Qwen2.5:3b model by default. Ollama handles quantization
    and CPU/GPU optimization automatically.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        settings = get_settings()
        self._model = model or settings.llm_model
        self._base_url = base_url or settings.ollama_base_url
        self._temperature = temperature if temperature is not None else settings.temperature
        self._max_tokens = max_tokens or settings.max_tokens
        self._client = httpx.Client(timeout=120.0)

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from a prompt using Ollama API."""
        system_prompt = kwargs.get("system_prompt", "")
        temperature = kwargs.get("temperature", self._temperature)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": self._max_tokens,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
        except httpx.ConnectError:
            logger.error(
                "Cannot connect to Ollama at %s. "
                "Make sure Ollama is running: 'ollama serve'",
                self._base_url,
            )
            raise
        except Exception as e:
            logger.error("Ollama generation failed: %s", e)
            raise

    def is_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            resp = self._client.get(f"{self._base_url}/api/tags")
            if resp.status_code != 200:
                return False
            models = resp.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            # Check if our model is available (with or without :latest tag)
            return any(
                self._model in name or name.startswith(self._model.split(":")[0])
                for name in model_names
            )
        except Exception:
            return False


def get_llm_provider() -> LanguageModel:
    """Factory for the configured LLM provider."""
    settings = get_settings()
    if settings.llm_provider.value == "ollama":
        return OllamaProvider()
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
