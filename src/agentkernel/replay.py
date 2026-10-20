"""Deterministic replay: re-run a recorded run against the CURRENT code, without a model or tools.

The recorded model responses and tool results are fed back through the live kernel. If the
prompt construction, tool schemas, message handling or policy changed since the run was
recorded, the replay diverges, and the first divergence is reported. This turns any real run
into a regression test for the agent's *logic* (everything except the model's randomness), with
no API key and no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .journal import Event, Journal
from .llm import LLMResponse, Usage
from .policy import Policy
from .runtime import Kernel
from .tools import Tool, ToolRegistry, ToolResult
from .util import digest


class ReplayError(Exception):
    """The run cannot be replayed (unknown run, or it was never started)."""


class _Exhausted(Exception):
    """The replayed code wanted more model responses or tool results than were recorded."""


@dataclass
class Divergence:
    index: int
    kind: str
    expected: Any
    actual: Any
    message: str


@dataclass
class ReplayReport:
    run_id: str
    compared: int
    divergences: list[Divergence] = field(default_factory=list)
    integrity_problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.divergences and not self.integrity_problems


class RecordedLLM:
    model = "recorded"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)

    def generate(self, **_: Any) -> LLMResponse:
        if not self._responses:
            raise _Exhausted("the code asked for more model calls than were recorded")
        return self._responses.pop(0)


def signature(events: list[Event]) -> list[tuple[Any, ...]]:
    """The behaviour-relevant essence of a run: no timestamps, hashes, or free-text reasons."""
    out: list[tuple[Any, ...]] = []
    for ev in events:
        d = ev.data
        if ev.kind == "llm_response":
            out.append(("llm_response", d["step"], d["request_digest"], digest(d["content"])[:12]))
        elif ev.kind == "policy_decision":
            out.append(("policy_decision", d["call"], d["decision"]))
        elif ev.kind == "approval_requested":
            out.append(("approval_requested", d["call"]))
        elif ev.kind == "approval_resolved":
            out.append(("approval_resolved", d["call"], bool(d["approved"])))
        elif ev.kind == "tool_result":
            out.append(("tool_result", d["call"], bool(d["is_error"]), digest(d["content"])[:12]))
        elif ev.kind == "run_suspended":
            out.append(("run_suspended", tuple(d.get("pending", []))))
        elif ev.kind == "run_resumed":
            out.append(("run_resumed",))
        elif ev.kind == "run_finished":
            out.append(("run_finished", d["status"], digest(d.get("output", ""))[:12]))
    return out


def _explain(expected: tuple[Any, ...] | None, actual: tuple[Any, ...] | None) -> str:
    if expected is None and actual is not None:
        return f"replay produced an extra event {actual[0]!r} that the recording does not have"
    if actual is None and expected is not None:
        return f"the recording has an event {expected[0]!r} that the replay never produced"
    if expected is None or actual is None:  # both missing: nothing to explain
        return "no difference"
    if expected[0] != actual[0]:
        return f"event order changed: recorded {expected[0]!r}, replay produced {actual[0]!r}"
    kind = expected[0]
    if kind == "llm_response" and expected[2] != actual[2]:
        return (
            f"the request for model step {expected[1]} differs from the recording: the system "
            "prompt, tool schemas or message construction changed"
        )
    if kind == "policy_decision":
        return (
            f"policy decision for call {expected[1]} changed: "
            f"recorded {expected[2]}, now {actual[2]}"
        )
    if kind == "tool_result":
        return f"the result recorded for call {expected[1]} no longer matches"
    return f"{kind} differs from the recording"


def replay(
    journal: Journal,
    run_id: str,
    registry: ToolRegistry,
    policy: Policy,
    *,
    system_prompt: str = "You are a careful assistant.",
) -> ReplayReport:
    events = journal.events(run_id)
    if not events or events[0].kind != "run_started":
        raise ReplayError(f"run '{run_id}' not found or never started")
    started = events[0].data

    responses = [
        LLMResponse(
            e.data["content"],
            e.data.get("stop_reason", "end_turn"),
            Usage(e.data["usage"]["input"], e.data["usage"]["output"]),
        )
        for e in events
        if e.kind == "llm_response"
    ]
    recorded = {
        e.data["call"]: ToolResult(e.data["content"], bool(e.data["is_error"]))
        for e in events
        if e.kind == "tool_result"
    }
    approvals = {
        e.data["call"]: bool(e.data["approved"]) for e in events if e.kind == "approval_resolved"
    }

    def executor(key: str, tool: Tool, args: dict[str, Any]) -> ToolResult:
        if key not in recorded:
            raise _Exhausted(f"no recorded result for call {key}")
        return recorded[key]

    scratch = Journal(":memory:")
    kernel = Kernel(
        RecordedLLM(responses),
        registry,
        scratch,
        policy,
        system_prompt=system_prompt,
        max_steps=started.get("max_steps", 20),
        token_budget=started.get("token_budget", 200_000),
        executor=executor,
    )
    new_id = kernel.start(started["task"])
    stalled = False
    try:
        result = kernel.advance(new_id)
        while result.status == "suspended" and result.pending:
            key = result.pending[0].call_key
            if key not in approvals:
                break  # the original run is also still waiting here
            kernel.approve(new_id, key, approved=approvals[key], by="replay")
            result = kernel.advance(new_id)
    except _Exhausted:
        stalled = True

    original = signature(events)
    replayed = signature(scratch.events(new_id))
    report = ReplayReport(run_id, compared=max(len(original), len(replayed)))
    report.integrity_problems = journal.verify(run_id)
    for i in range(max(len(original), len(replayed))):
        exp = original[i] if i < len(original) else None
        act = replayed[i] if i < len(replayed) else None
        if exp != act:
            report.divergences.append(
                Divergence(i, (exp or act or ("?",))[0], exp, act, _explain(exp, act))
            )
            if len(report.divergences) >= 20:
                break
    if stalled and not report.divergences:
        report.divergences.append(
            Divergence(len(replayed), "exhausted", None, None, "replay needed more recorded data")
        )
    scratch.close()
    return report
