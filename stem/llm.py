"""Provider-agnostic LLM client wrapping Anthropic and OpenAI."""

from __future__ import annotations

import os
from typing import Iterator


class LLMClient:
    """
    Thin adapter over Anthropic and OpenAI SDKs.

    Set STEM_PROVIDER=openai (or anthropic) in .env to switch backends.
    The underlying SDK is imported lazily so only the one you use needs to be installed.

    Methods
    -------
    complete(system, user, max_tokens)        — single non-streaming completion
    stream_tokens(messages, system, max_tokens) — yields text tokens for live output
    """

    def __init__(self, provider: str, model: str) -> None:
        if provider not in ("anthropic", "openai"):
            raise ValueError(f"provider must be 'anthropic' or 'openai', got {provider!r}")
        self.provider = provider
        self.model = model

        if provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic()
        else:
            import openai
            self._client = openai.OpenAI()

    @classmethod
    def from_env(cls, model: str | None = None) -> LLMClient:
        """Create from STEM_PROVIDER and STEM_MODEL environment variables."""
        provider = os.getenv("STEM_PROVIDER", "anthropic")
        if model is None:
            defaults = {"anthropic": "claude-sonnet-4-6", "openai": "gpt-4o"}
            model = os.getenv("STEM_MODEL", defaults.get(provider, "gpt-4o"))
        return cls(provider=provider, model=model)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def complete(self, system: str, user: str, max_tokens: int = 2048) -> str:
        """Single non-streaming completion. Returns the response text."""
        if self.provider == "anthropic":
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
            )
            return response.content[0].text.strip()
        else:
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return (response.choices[0].message.content or "").strip()

    def stream_tokens(
        self,
        messages: list[dict],
        system: str = "",
        max_tokens: int = 4096,
    ) -> Iterator[str]:
        """Yield text tokens as they arrive (for streaming output to the terminal)."""
        if self.provider == "anthropic":
            kwargs: dict = dict(model=self.model, max_tokens=max_tokens, messages=messages)
            if system:
                kwargs["system"] = [
                    {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
                ]
            with self._client.messages.stream(**kwargs) as s:
                yield from s.text_stream
        else:
            msgs = ([{"role": "system", "content": system}] if system else []) + messages
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=msgs,
                stream=True,
            )
            for chunk in response:
                text = chunk.choices[0].delta.content or ""
                if text:
                    yield text
