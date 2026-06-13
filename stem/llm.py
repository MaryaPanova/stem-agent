"""LLM client with tool-use support, plus a scriptable fake for tests.

The agent needs *tool calling*, not just text completion, so this is a real tool-use
client (Anthropic-primary). ``ScriptedLLM`` lets the whole rollout / evolution / eval
stack run deterministically with no API key — that is what the test suite uses.

Honesty note: the previous version of this project advertised "switch providers, no code
changes". That was never quite true once tool use is involved. This client supports
Anthropic; the seam is here if someone wants to add OpenAI tool-use, but it is not done.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolUse:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResponse:
    """Normalised result of one model turn."""

    text: str = ""
    tool_uses: list[ToolUse] = field(default_factory=list)
    stop_reason: str = "end_turn"

    @property
    def wants_tool(self) -> bool:
        return bool(self.tool_uses)


# Convenience builders for scripted tests / fixtures -------------------------


def say(text: str) -> LLMResponse:
    return LLMResponse(text=text, stop_reason="end_turn")


def call(_tool: str, _id: str = "t", **arguments: Any) -> LLMResponse:
    """Build a tool-call response. ``_tool``/``_id`` are underscored so a tool argument
    may itself be named ``tool``/``id``/``name`` without colliding."""
    return LLMResponse(tool_uses=[ToolUse(id=_id, name=_tool, input=arguments)],
                       stop_reason="tool_use")


class LLMClient:
    """Anthropic-backed tool-use client."""

    def __init__(self, provider: str, model: str, client: Any | None = None) -> None:
        if provider != "anthropic":
            raise ValueError(
                f"provider {provider!r} not supported for tool use; use 'anthropic'"
            )
        self.provider = provider
        self.model = model
        if client is not None:
            self._client = client
        else:
            import anthropic

            self._client = anthropic.Anthropic()

    @classmethod
    def from_env(cls, model: str | None = None) -> "LLMClient":
        provider = os.getenv("STEM_PROVIDER", "anthropic")
        if model is None:
            model = os.getenv("STEM_MODEL", "claude-sonnet-4-6")
        return cls(provider=provider, model=model)

    def run(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens,
            messages=messages,
        )
        if system:
            kwargs["system"] = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]
        if tools:
            kwargs["tools"] = tools

        resp = self._client.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_uses: list[ToolUse] = []
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                tool_uses.append(ToolUse(id=block.id, name=block.name, input=dict(block.input)))

        return LLMResponse(
            text="".join(text_parts),
            tool_uses=tool_uses,
            stop_reason=getattr(resp, "stop_reason", "end_turn"),
        )


class ScriptedLLM:
    """A drop-in replacement for ``LLMClient`` that returns canned turns.

    ``script`` is either:
      * a list of ``LLMResponse`` — popped in order, or
      * a callable ``(system, messages, tools) -> LLMResponse`` — for smarter fakes that
        react to the conversation (used to simulate a "competent" specialist in tests).
    """

    def __init__(
        self,
        script: list[LLMResponse] | Callable[..., LLMResponse],
        model: str = "scripted",
    ) -> None:
        self.model = model
        self.provider = "scripted"
        self._script = script
        self._queue = list(script) if isinstance(script, list) else None
        self.calls: list[dict[str, Any]] = []  # recorded for assertions

    def run(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        self.calls.append({"system": system, "messages": messages, "tools": tools})
        if callable(self._script):
            return self._script(system, messages, tools)
        assert self._queue is not None
        if not self._queue:
            return say("")  # nothing left to say -> ends the rollout
        return self._queue.pop(0)
