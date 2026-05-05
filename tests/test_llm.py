"""Tests for the LLMClient provider adapter."""

from unittest.mock import MagicMock, patch

import pytest

from stem.llm import LLMClient


# ------------------------------------------------------------------
# Construction and from_env
# ------------------------------------------------------------------

def test_invalid_provider_raises():
    with pytest.raises(ValueError, match="provider"):
        LLMClient(provider="gemini", model="x")


def test_from_env_defaults_to_anthropic(monkeypatch):
    monkeypatch.delenv("STEM_PROVIDER", raising=False)
    monkeypatch.delenv("STEM_MODEL", raising=False)
    with patch("anthropic.Anthropic"):
        client = LLMClient.from_env()
    assert client.provider == "anthropic"
    assert client.model == "claude-sonnet-4-6"


def test_from_env_uses_openai_when_set(monkeypatch):
    monkeypatch.setenv("STEM_PROVIDER", "openai")
    monkeypatch.delenv("STEM_MODEL", raising=False)
    with patch("openai.OpenAI"):
        client = LLMClient.from_env()
    assert client.provider == "openai"
    assert client.model == "gpt-4o"


def test_from_env_respects_stem_model(monkeypatch):
    monkeypatch.setenv("STEM_PROVIDER", "openai")
    monkeypatch.setenv("STEM_MODEL", "gpt-4o-mini")
    with patch("openai.OpenAI"):
        client = LLMClient.from_env()
    assert client.model == "gpt-4o-mini"


# ------------------------------------------------------------------
# complete — Anthropic
# ------------------------------------------------------------------

def test_complete_anthropic_returns_text():
    with patch("anthropic.Anthropic") as Mock:
        mock_sdk = MagicMock()
        Mock.return_value = mock_sdk
        mock_sdk.messages.create.return_value.content = [MagicMock(text="  answer  ")]

        llm = LLMClient(provider="anthropic", model="test")
        result = llm.complete(system="sys", user="hi")

    assert result == "answer"


def test_complete_anthropic_passes_cache_control():
    with patch("anthropic.Anthropic") as Mock:
        mock_sdk = MagicMock()
        Mock.return_value = mock_sdk
        mock_sdk.messages.create.return_value.content = [MagicMock(text="ok")]

        llm = LLMClient(provider="anthropic", model="test")
        llm.complete(system="sys", user="hi")

    call_kwargs = mock_sdk.messages.create.call_args.kwargs
    assert call_kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}


# ------------------------------------------------------------------
# complete — OpenAI
# ------------------------------------------------------------------

def test_complete_openai_returns_text():
    with patch("openai.OpenAI") as Mock:
        mock_sdk = MagicMock()
        Mock.return_value = mock_sdk
        mock_sdk.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="  answer  "))
        ]

        llm = LLMClient(provider="openai", model="test")
        result = llm.complete(system="sys", user="hi")

    assert result == "answer"


def test_complete_openai_puts_system_first_in_messages():
    with patch("openai.OpenAI") as Mock:
        mock_sdk = MagicMock()
        Mock.return_value = mock_sdk
        mock_sdk.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="ok"))
        ]

        llm = LLMClient(provider="openai", model="test")
        llm.complete(system="be helpful", user="hello")

    msgs = mock_sdk.chat.completions.create.call_args.kwargs["messages"]
    assert msgs[0] == {"role": "system", "content": "be helpful"}
    assert msgs[1] == {"role": "user", "content": "hello"}


# ------------------------------------------------------------------
# stream_tokens — OpenAI
# ------------------------------------------------------------------

def test_stream_tokens_openai_yields_chunks():
    with patch("openai.OpenAI") as Mock:
        mock_sdk = MagicMock()
        Mock.return_value = mock_sdk

        def make_chunk(text):
            c = MagicMock()
            c.choices = [MagicMock(delta=MagicMock(content=text))]
            return c

        mock_sdk.chat.completions.create.return_value = iter([
            make_chunk("Hello"),
            make_chunk(" world"),
            make_chunk(""),   # empty chunk should be skipped
        ])

        llm = LLMClient(provider="openai", model="test")
        tokens = list(llm.stream_tokens(messages=[{"role": "user", "content": "hi"}]))

    assert tokens == ["Hello", " world"]


def test_stream_tokens_openai_prepends_system():
    with patch("openai.OpenAI") as Mock:
        mock_sdk = MagicMock()
        Mock.return_value = mock_sdk
        mock_sdk.chat.completions.create.return_value = iter([])

        llm = LLMClient(provider="openai", model="test")
        list(llm.stream_tokens(
            messages=[{"role": "user", "content": "hi"}],
            system="you are helpful",
        ))

    msgs = mock_sdk.chat.completions.create.call_args.kwargs["messages"]
    assert msgs[0] == {"role": "system", "content": "you are helpful"}


# ------------------------------------------------------------------
# stream_tokens — Anthropic
# ------------------------------------------------------------------

def test_stream_tokens_anthropic_uses_stream_context():
    with patch("anthropic.Anthropic") as Mock:
        mock_sdk = MagicMock()
        Mock.return_value = mock_sdk

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Hello", " world"])
        mock_sdk.messages.stream.return_value = mock_stream

        llm = LLMClient(provider="anthropic", model="test")
        tokens = list(llm.stream_tokens(messages=[{"role": "user", "content": "hi"}]))

    assert tokens == ["Hello", " world"]
