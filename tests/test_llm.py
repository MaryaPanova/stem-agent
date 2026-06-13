"""ScriptedLLM + response builders + the Anthropic adapter's parsing (mocked client)."""

from stem.llm import LLMClient, LLMResponse, ScriptedLLM, call, say


def test_builders():
    assert say("hi").text == "hi" and not say("hi").wants_tool
    r = call("buy", quantity=2)
    assert r.wants_tool and r.tool_uses[0].name == "buy" and r.tool_uses[0].input == {"quantity": 2}


def test_scripted_queue_and_exhaustion():
    llm = ScriptedLLM([say("a"), call("t")])
    assert llm.run("", []).text == "a"
    assert llm.run("", []).wants_tool
    assert llm.run("", []).text == ""  # exhausted -> empty turn ends rollouts
    assert len(llm.calls) == 3


def test_scripted_callable():
    llm = ScriptedLLM(lambda system, messages, tools: say(f"saw {len(messages)} msgs"))
    assert llm.run("sys", [{"role": "user", "content": "x"}]).text == "saw 1 msgs"


class _FakeBlock:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeResp:
    def __init__(self, content, stop_reason="tool_use"):
        self.content = content
        self.stop_reason = stop_reason


class _FakeAnthropic:
    class messages:  # noqa: N801
        @staticmethod
        def create(**kwargs):
            return _FakeResp([
                _FakeBlock(type="text", text="thinking"),
                _FakeBlock(type="tool_use", id="abc", name="buy", input={"quantity": 1}),
            ])


def test_anthropic_adapter_parses_text_and_tool_use():
    client = LLMClient(provider="anthropic", model="x", client=_FakeAnthropic())
    resp = client.run("system", [{"role": "user", "content": "go"}], tools=[{"name": "buy"}])
    assert isinstance(resp, LLMResponse)
    assert resp.text == "thinking"
    assert resp.tool_uses[0].name == "buy" and resp.tool_uses[0].input == {"quantity": 1}
