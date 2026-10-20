"""Run state is *derived*: replay the journal's events and you get the exact state back.

There is no mutable "current state" to lose in a crash. `rebuild` is the only place that knows
how events map to state, and `build_messages` is the only place that turns state into the
conversation sent to the model, so replay and live execution cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .journal import Event
from .llm import Usage
from .policy import Action, Decision
from .tools import UNTRUSTED


@dataclass
class ToolResultRecord:
    content: str
    is_error: bool
    status: str  # ok | error | denied | uncertain


@dataclass
class CallState:
    key: str  # "<step>.<index>": unique within a run (providers may reuse their own ids)
    provider_id: str
    name: str
    input: dict[str, Any]
    step: int
    capabilities: tuple[str, ...] = ()
    decision: Decision | None = None
    approval_requested: bool = False
    approval: bool | None = None
    started: bool = False
    result: ToolResultRecord | None = None


@dataclass
class StepState:
    number: int
    content: list[dict[str, Any]]
    stop_reason: str
    calls: list[CallState] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(b["text"] for b in self.content if b.get("type") == "text")


@dataclass
class RunState:
    run_id: str
    task: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    steps: list[StepState] = field(default_factory=list)
    suspended: bool = False
    finished: bool = False
    final_status: str = ""
    output: str = ""
    usage: Usage = field(default_factory=Usage)
    caps_seen: set[str] = field(default_factory=set)

    @property
    def status(self) -> str:
        if self.finished:
            return self.final_status
        return "suspended" if self.suspended else "running"

    @property
    def last_text(self) -> str:
        return self.steps[-1].text if self.steps else ""

    def call(self, key: str) -> CallState | None:
        for step in self.steps:
            for call in step.calls:
                if call.key == key:
                    return call
        return None


def rebuild(run_id: str, events: list[Event]) -> RunState:
    state = RunState(run_id=run_id)
    calls: dict[str, CallState] = {}
    for ev in events:
        d = ev.data
        kind = ev.kind
        if kind == "run_started":
            state.task = d["task"]
            state.meta = d
        elif kind == "llm_response":
            step = StepState(d["step"], d["content"], d.get("stop_reason", "end_turn"))
            blocks = [b for b in d["content"] if b.get("type") == "tool_use"]
            for i, block in enumerate(blocks):
                call = CallState(
                    key=f"{step.number}.{i}",
                    provider_id=block["id"],
                    name=block["name"],
                    input=block.get("input") or {},
                    step=step.number,
                )
                step.calls.append(call)
                calls[call.key] = call
            state.steps.append(step)
            u = d.get("usage") or {}
            state.usage = state.usage + Usage(u.get("input", 0), u.get("output", 0))
        elif kind == "policy_decision" and d["call"] in calls:
            call = calls[d["call"]]
            call.decision = Decision(Action(d["decision"]), d["reason"], d["source"])
            call.capabilities = tuple(d.get("capabilities", ()))
        elif kind == "approval_requested" and d["call"] in calls:
            calls[d["call"]].approval_requested = True
        elif kind == "approval_resolved" and d["call"] in calls:
            calls[d["call"]].approval = bool(d["approved"])
        elif kind == "tool_started" and d["call"] in calls:
            calls[d["call"]].started = True
            state.caps_seen.update(d.get("capabilities", ()))
        elif kind == "tool_result" and d["call"] in calls:
            calls[d["call"]].result = ToolResultRecord(
                d["content"], bool(d["is_error"]), d.get("status", "ok")
            )
        elif kind == "run_suspended":
            state.suspended = True
        elif kind == "run_resumed":
            state.suspended = False
        elif kind == "run_finished":
            state.finished = True
            state.final_status = d["status"]
            state.output = d.get("output", "")
    return state


def wrap_output(name: str, content: str, untrusted: bool) -> str:
    """Tool results are data, never instructions; untrusted sources are marked as such."""
    safe = content.replace("</tool_output>", "&lt;/tool_output&gt;")
    flag = ' untrusted="true"' if untrusted else ""
    return f'<tool_output tool="{name}"{flag}>\n{safe}\n</tool_output>'


def build_messages(state: RunState) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "user", "content": state.task}]
    for step in state.steps:
        messages.append(
            {"role": "assistant", "content": step.content or [{"type": "text", "text": "(empty)"}]}
        )
        if not step.calls:
            continue
        blocks: list[dict[str, Any]] = []
        for call in step.calls:
            if call.result is None:  # the step is not settled yet; nothing more to send
                return messages
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": call.provider_id,
                "content": wrap_output(
                    call.name, call.result.content, UNTRUSTED in call.capabilities
                ),
            }
            if call.result.is_error:
                block["is_error"] = True
            blocks.append(block)
        messages.append({"role": "user", "content": blocks})
    return messages
