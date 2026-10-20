"""Model layer: a tiny protocol, two adapters, and request digests for deterministic replay.

Messages use the Anthropic content-block format as the canonical internal representation.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from .util import digest


class LLMError(RuntimeError):
    """A provider failure with a human-readable message."""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens
        )


@dataclass
class LLMResponse:
    content: list[dict[str, Any]]
    stop_reason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)

    @property
    def text(self) -> str:
        return "".join(b["text"] for b in self.content if b.get("type") == "text")

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [
            ToolCall(b["id"], b["name"], dict(b.get("input") or {}))
            for b in self.content
            if b.get("type") == "tool_use"
        ]


class LLM(Protocol):
    model: str

    def generate(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 2048,
    ) -> LLMResponse: ...


def request_digest(
    system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
) -> str:
    """Fingerprint of everything sent to the model. Replay compares these to catch drift."""
    return digest({"system": system, "messages": messages, "tools": tools or []})


# --------------------------------------------------------------------------- OpenAI-compatible
def _to_openai_messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
        elif m["role"] == "assistant":
            text = "".join(b["text"] for b in content if b.get("type") == "text")
            calls = [
                {
                    "id": b["id"],
                    "type": "function",
                    "function": {"name": b["name"], "arguments": json.dumps(b["input"])},
                }
                for b in content
                if b.get("type") == "tool_use"
            ]
            msg: dict[str, Any] = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        else:
            for b in content:
                if b.get("type") == "tool_result":
                    body = (
                        b["content"] if isinstance(b["content"], str) else json.dumps(b["content"])
                    )
                    out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": body})
                elif b.get("type") == "text":
                    out.append({"role": "user", "content": b["text"]})
    return out


class OpenAICompatLLM:
    """Any OpenAI-compatible /chat/completions endpoint (Ollama, Groq, Gemini, vLLM, ...)."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        api_key: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode(), headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as exc:
                body = exc.read().decode(errors="replace")[:300]
                if exc.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    time.sleep(2**attempt)
                    continue
                hint = " (invalid API key?)" if exc.code in (401, 403) else ""
                raise LLMError(f"{self.base_url} returned HTTP {exc.code}{hint}: {body}") from exc
            except urllib.error.URLError as exc:
                if isinstance(exc.reason, TimeoutError):
                    raise LLMError(f"no response from {url} within {self.timeout:g}s") from exc
                if attempt < self.max_retries:
                    time.sleep(2**attempt)
                    continue
                raise LLMError(f"cannot reach {url}: {exc.reason}. Is the server running?") from exc
            except TimeoutError as exc:
                raise LLMError(f"no response from {url} within {self.timeout:g}s") from exc
        raise LLMError("unreachable")  # pragma: no cover

    def generate(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _to_openai_messages(system, messages),
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["input_schema"],
                    },
                }
                for t in tools
            ]
        data = self._post(payload)
        choice = data["choices"][0]
        message = choice["message"]
        content: list[dict[str, Any]] = []
        if message.get("content"):
            content.append({"type": "text", "text": message["content"]})
        for i, tc in enumerate(message.get("tool_calls") or []):
            raw = tc["function"].get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except json.JSONDecodeError:
                args = {}
            content.append(
                {
                    "type": "tool_use",
                    "id": tc.get("id") or f"call_{i}",
                    "name": tc["function"]["name"],
                    "input": args if isinstance(args, dict) else {},
                }
            )
        stop = {"tool_calls": "tool_use", "length": "max_tokens"}.get(
            choice.get("finish_reason") or "", "end_turn"
        )
        u = data.get("usage") or {}
        return LLMResponse(
            content, stop, Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
        )


# --------------------------------------------------------------------------- Anthropic
class AnthropicLLM:
    """Anthropic Messages API (needs the optional `anthropic` extra)."""

    def __init__(
        self, model: str = "claude-sonnet-5-5", api_key: str | None = None, max_retries: int = 4
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise LLMError("install the extra: pip install 'agentkernel[anthropic]'") from exc
        self.model = model
        self._api_error = getattr(anthropic, "APIError", None)
        self._client = anthropic.Anthropic(api_key=api_key, max_retries=max_retries)

    def generate(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        try:
            resp = self._client.messages.create(**kwargs)
        except Exception as exc:
            if self._api_error is not None and isinstance(exc, self._api_error):
                status = getattr(exc, "status_code", None)
                if status == 401:
                    raise LLMError(
                        "Anthropic API key is invalid. Check ANTHROPIC_API_KEY."
                    ) from exc
                if "credit balance" in str(exc).lower():
                    raise LLMError("Anthropic account has no credit (Plans & Billing).") from exc
                raise LLMError(f"Anthropic API error ({status}): {exc}") from exc
            raise
        content: list[dict[str, Any]] = []
        for block in resp.content:
            if block.type == "text":
                content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                content.append(
                    {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
                )
        usage = Usage(resp.usage.input_tokens, resp.usage.output_tokens)
        return LLMResponse(content, resp.stop_reason or "end_turn", usage)


def make_llm(
    provider: str,
    model: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 120.0,
) -> LLM:
    """provider: 'anthropic' or 'openai' (any OpenAI-compatible endpoint)."""
    if provider == "anthropic":
        return AnthropicLLM(model, api_key=api_key)
    if provider == "openai":
        if model.startswith("claude"):
            raise LLMError("Set --model (or AGENTKERNEL_MODEL) to a model your provider serves.")
        return OpenAICompatLLM(
            model, base_url or "http://localhost:11434/v1", api_key, timeout=timeout
        )
    raise LLMError(f"unknown provider '{provider}' (use 'anthropic' or 'openai')")
