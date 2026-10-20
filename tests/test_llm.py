import io
import sys
import types
import urllib.error

import pytest

from agentkernel import LLMError, OpenAICompatLLM, Usage, make_llm
from agentkernel.llm import AnthropicLLM, LLMResponse, _to_openai_messages, request_digest

MSGS = [{"role": "user", "content": "hi"}]
TOOL = {"name": "calculator", "description": "math", "input_schema": {"type": "object"}}


def test_request_digest_is_stable_and_sensitive_to_every_part():
    base = request_digest("sys", MSGS, [TOOL])
    assert base == request_digest("sys", [dict(m) for m in MSGS], [dict(TOOL)])
    assert base != request_digest("sys2", MSGS, [TOOL])
    assert base != request_digest("sys", [{"role": "user", "content": "hello"}], [TOOL])
    assert base != request_digest("sys", MSGS, [])
    assert request_digest("s", MSGS, None) == request_digest("s", MSGS, [])


def test_response_helpers():
    r = LLMResponse([{"type": "text", "text": "a"}, {"type": "tool_use", "id": "1", "name": "t"}])
    assert r.text == "a" and r.tool_calls[0].name == "t" and r.tool_calls[0].input == {}
    assert Usage(1, 2) + Usage(3, 4) == Usage(4, 6) and Usage(1, 2).total == 3


def test_openai_message_translation():
    msgs = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "checking"},
                {
                    "type": "tool_use",
                    "id": "c1",
                    "name": "calculator",
                    "input": {"expression": "1"},
                },
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "1"}]},
    ]
    out = _to_openai_messages("SYS", msgs)
    assert out[0] == {"role": "system", "content": "SYS"} and out[1]["content"] == "q"
    assert out[2]["tool_calls"][0]["function"]["arguments"] == '{"expression": "1"}'
    assert out[3] == {"role": "tool", "tool_call_id": "c1", "content": "1"}


def test_openai_compatible_generate_parses_text_tools_and_usage(monkeypatch):
    llm = OpenAICompatLLM("m")
    reply = {
        "choices": [
            {
                "message": {
                    "content": "hi",
                    "tool_calls": [
                        {"function": {"name": "calculator", "arguments": '{"expression": "1"}'}},
                        {"id": "x", "function": {"name": "t", "arguments": "{broken"}},
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 4},
    }
    seen = {}
    monkeypatch.setattr(llm, "_post", lambda p: seen.update(p) or reply)
    resp = llm.generate(system="s", messages=MSGS, tools=[TOOL])
    assert resp.text == "hi" and resp.stop_reason == "tool_use" and resp.usage == Usage(12, 4)
    assert [c.id for c in resp.tool_calls] == ["call_0", "x"]  # a missing id gets a stable one
    assert resp.tool_calls[0].input == {"expression": "1"} and resp.tool_calls[1].input == {}
    assert seen["tools"][0]["function"]["parameters"] == {"type": "object"}


def test_http_failures_become_clear_errors(monkeypatch):
    llm = OpenAICompatLLM("m", base_url="http://x/v1", max_retries=0, timeout=3)

    def deny(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 401, "no", {}, io.BytesIO(b"bad key"))

    monkeypatch.setattr("urllib.request.urlopen", deny)
    with pytest.raises(LLMError, match="invalid API key"):
        llm.generate(system="s", messages=MSGS)

    def down(req, timeout):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr("urllib.request.urlopen", down)
    with pytest.raises(LLMError, match="Is the server running"):
        llm.generate(system="s", messages=MSGS)

    def slow(req, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr("urllib.request.urlopen", slow)
    with pytest.raises(LLMError, match="within 3s"):
        llm.generate(system="s", messages=MSGS)


def test_transient_errors_are_retried(monkeypatch):
    llm = OpenAICompatLLM("m", max_retries=2)
    monkeypatch.setattr("time.sleep", lambda s: None)
    attempts = {"n": 0}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}'

    def flaky(req, timeout):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise urllib.error.HTTPError(req.full_url, 429, "slow", {}, io.BytesIO(b""))
        return Resp()

    monkeypatch.setattr("urllib.request.urlopen", flaky)
    assert llm.generate(system="s", messages=MSGS).text == "ok" and attempts["n"] == 2


def test_factory():
    assert make_llm("openai", "llama", base_url="http://h/v1").model == "llama"
    with pytest.raises(LLMError, match="Set --model"):
        make_llm("openai", "claude-sonnet-5-5")
    with pytest.raises(LLMError, match="unknown provider"):
        make_llm("nope", "m")


def test_anthropic_adapter_with_a_fake_sdk(monkeypatch):
    class Block:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class Messages:
        def create(self, **kw):
            Messages.kw = kw
            return types.SimpleNamespace(
                content=[
                    Block(type="text", text="hello"),
                    Block(type="tool_use", id="tu", name="calculator", input={"expression": "1"}),
                ],
                stop_reason="tool_use",
                usage=types.SimpleNamespace(input_tokens=11, output_tokens=7),
            )

    class Anthropic:
        def __init__(self, **kw):
            self.messages = Messages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=Anthropic))
    resp = AnthropicLLM("m", api_key="k").generate(system="s", messages=MSGS, tools=[TOOL])
    assert resp.text == "hello" and resp.tool_calls[0].name == "calculator"
    assert resp.usage == Usage(11, 7) and Messages.kw["tools"] == [TOOL]


def test_anthropic_without_the_sdk_explains_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(LLMError, match="agentkernel\\[anthropic\\]"):
        AnthropicLLM()
