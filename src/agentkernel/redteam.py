"""Offline red team: a fully compromised (scripted) model attacks the kernel.

Each scenario scripts the *worst case*: a model that obediently carries out the attacker's plan
(read the secret, read the poisoned inbox, send it out, evade by case or batching...). The
assertion is about the kernel: did the forbidden outcome happen? Because the "model" is a script,
this runs in CI with no API key and proves the controls hold regardless of model alignment.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .demo_tools import make_demo_tools
from .journal import Journal
from .llm import LLMResponse, Usage
from .policy import Policy, default_policy
from .runtime import Kernel, RunResult

SCENARIO_DIR = Path(__file__).parent / "scenarios"


class ScriptedAttacker:
    """Plays a fixed list of turns: {"calls": [...]} issues tool calls, {"say": "..."} ends."""

    model = "scripted-attacker"

    def __init__(self, turns: list[dict[str, Any]]) -> None:
        expanded: list[dict[str, Any]] = []
        for turn in turns:
            expanded.extend([turn] * int(turn.get("repeat", 1)))
        self._turns = expanded
        self._n = 0

    def generate(self, **_: Any) -> LLMResponse:
        turn = self._turns[self._n] if self._n < len(self._turns) else {"say": "done"}
        self._n += 1
        content: list[dict[str, Any]] = []
        for i, call in enumerate(turn.get("calls", [])):
            content.append(
                {
                    "type": "tool_use",
                    "id": f"atk_{self._n}_{i}",
                    "name": call["tool"],
                    "input": call.get("input", {}),
                }
            )
        if not content:
            content.append({"type": "text", "text": turn.get("say", "done")})
        return LLMResponse(
            content,
            "tool_use" if len(content) and content[0]["type"] == "tool_use" else "end_turn",
            Usage(10, 5),
        )


@dataclass
class ScenarioResult:
    name: str
    description: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    status: str = ""


def load_scenarios(directory: str | Path | None = None) -> list[dict[str, Any]]:
    folder = Path(directory) if directory else SCENARIO_DIR
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))]


def _check(
    scenario: dict[str, Any], result: RunResult, journal: Journal, run_id: str, workspace: Path
) -> list[str]:
    expect = scenario.get("expect", {})
    events = journal.events(run_id)
    failures: list[str] = []
    outbox = workspace / "outbox.jsonl"
    outbox_text = outbox.read_text(encoding="utf-8") if outbox.exists() else ""

    if "status" in expect and result.status != expect["status"]:
        failures.append(f"status is {result.status!r}, expected {expect['status']!r}")
    if "outbox_count" in expect:
        count = len([line for line in outbox_text.splitlines() if line.strip()])
        if count != expect["outbox_count"]:
            failures.append(f"{count} email(s) were sent, expected {expect['outbox_count']}")
    if "secret_absent_from_outbox" in expect and expect["secret_absent_from_outbox"] in outbox_text:
        failures.append("the secret reached the outbox")

    tool_of = {e.data["call"]: e.data["tool"] for e in events if e.kind == "policy_decision"}
    executed = {
        tool_of[e.data["call"]]
        for e in events
        if e.kind == "tool_started" and e.data["call"] in tool_of
    }
    for name in expect.get("executed", []):
        if name not in executed:
            failures.append(f"tool {name!r} should have run but did not")
    for name in expect.get("not_executed", []):
        if name in executed:
            failures.append(f"tool {name!r} ran but must not have")

    decisions = {e.data["call"]: e.data["decision"] for e in events if e.kind == "policy_decision"}
    for key, action in expect.get("decisions", {}).items():
        if decisions.get(key) != action:
            failures.append(f"call {key}: decision {decisions.get(key)!r}, expected {action!r}")

    results = {e.data["call"]: e.data for e in events if e.kind == "tool_result"}
    for key in expect.get("errors", []):
        if not results.get(key, {}).get("is_error"):
            failures.append(f"call {key} should have failed but did not")
    needle = expect.get("absent_from_results")
    if needle and any(needle in r["content"] for r in results.values()):
        failures.append(f"{needle!r} leaked into a tool result")
    return failures


def run_scenario(scenario: dict[str, Any]) -> ScenarioResult:
    with tempfile.TemporaryDirectory(prefix="agentkernel-redteam-") as tmp:
        workspace = Path(tmp)
        for rel, content in scenario.get("files", {}).items():
            path = workspace / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        raw_policy = scenario.get("policy", "default")
        policy = default_policy() if raw_policy == "default" else Policy.from_dict(raw_policy)
        journal = Journal(":memory:")
        kernel = Kernel(
            ScriptedAttacker(scenario["turns"]),
            make_demo_tools(workspace),
            journal,
            policy,
            max_steps=scenario.get("max_steps", 20),
        )
        run_id = kernel.start(scenario["task"])
        result = kernel.advance(run_id)
        failures = _check(scenario, result, journal, run_id, workspace)
        problems = journal.verify(run_id)
        if problems:
            failures.append(f"journal integrity: {problems[0]}")
        journal.close()
    return ScenarioResult(
        scenario["name"], scenario.get("description", ""), not failures, failures, result.status
    )


def run_all(directory: str | Path | None = None) -> list[ScenarioResult]:
    return [run_scenario(s) for s in load_scenarios(directory)]
