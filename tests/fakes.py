"""Deterministic fakes: scripted models and a crash that bypasses ordinary error handling."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

from agentkernel.llm import LLMError, LLMResponse, Usage

_ids = itertools.count(1)


def text(t: str, tokens: tuple[int, int] = (10, 5)) -> LLMResponse:
    return LLMResponse([{"type": "text", "text": t}], "end_turn", Usage(*tokens))


def calls(*specs: tuple[str, dict[str, Any]], tokens: tuple[int, int] = (10, 5)) -> LLMResponse:
    content = [
        {"type": "tool_use", "id": f"toolu_{next(_ids)}", "name": n, "input": i} for n, i in specs
    ]
    return LLMResponse(content, "tool_use", Usage(*tokens))


class ScriptedLLM:
    """Returns queued responses (or raises queued exceptions) in order; records every call."""

    model = "fake"

    def __init__(self, script: list[LLMResponse | Exception]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        if not self.script:
            raise AssertionError("ScriptedLLM ran out of responses")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FuncLLM:
    model = "fake-func"

    def __init__(self, fn: Callable[[dict[str, Any]], LLMResponse]) -> None:
        self.fn = fn
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        return self.fn(kwargs)


class SimulatedCrash(BaseException):
    """Stands in for a killed process: BaseException so no tool/kernel handler swallows it."""


__all__ = ["FuncLLM", "LLMError", "ScriptedLLM", "SimulatedCrash", "calls", "text"]
