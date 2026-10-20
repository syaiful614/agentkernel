import pytest

from agentkernel import (
    Action,
    Journal,
    Kernel,
    LLMResponse,
    Policy,
    ReplayError,
    Rule,
    ToolRegistry,
    Usage,
    default_policy,
    replay,
    tool,
)
from agentkernel.demo_tools import calculator, make_demo_tools
from agentkernel.replay import signature
from fakes import ScriptedLLM, text

PROMPT = "You are a careful assistant."
EMAIL = {"to": "manager@example.com", "subject": "s", "body": "b"}


def fixed(name, args, call_id="t1"):
    block = {"type": "tool_use", "id": call_id, "name": name, "input": args}
    return LLMResponse([block], "tool_use", Usage(10, 5))


def record(tmp_path, script, *, approve=(), policy=None, registry=None):
    journal = Journal()
    registry = registry or make_demo_tools(tmp_path)
    kernel = Kernel(ScriptedLLM(script), registry, journal, policy or default_policy())
    run_id = kernel.start("do the thing")
    kernel.advance(run_id)
    for key in approve:
        kernel.approve(run_id, key)
        kernel.resume(run_id)
    return journal, run_id, registry


def arithmetic_run(tmp_path):
    return record(tmp_path, [fixed("calculator", {"expression": "6*7"}), text("42")])


def test_an_unchanged_agent_replays_identically(tmp_path):
    journal, run_id, registry = arithmetic_run(tmp_path)
    report = replay(journal, run_id, registry, default_policy(), system_prompt=PROMPT)
    assert report.ok and report.divergences == [] and report.compared == 5


def test_replay_never_runs_the_tools_again(tmp_path):
    counter = {"n": 0}

    @tool
    def count_me() -> str:
        """Counts its own executions."""
        counter["n"] += 1
        return "counted"

    registry = ToolRegistry([count_me])
    policy = Policy([Rule("*", Action.ALLOW)], Action.ALLOW)
    journal, run_id, _ = record(
        tmp_path, [fixed("count_me", {}), text("done")], policy=policy, registry=registry
    )
    assert counter["n"] == 1
    assert replay(journal, run_id, registry, policy, system_prompt=PROMPT).ok
    assert counter["n"] == 1  # replayed from the journal; no side effects


def test_runs_with_approvals_replay_including_the_suspension(tmp_path):
    script = [fixed("send_email", EMAIL), text("sent")]
    journal, run_id, registry = record(tmp_path, script, approve=["1.0"])
    assert replay(journal, run_id, registry, default_policy(), system_prompt=PROMPT).ok


def test_a_run_still_waiting_for_a_human_replays_up_to_the_wait(tmp_path):
    journal, run_id, registry = record(tmp_path, [fixed("send_email", EMAIL)])
    assert replay(journal, run_id, registry, default_policy(), system_prompt=PROMPT).ok


def test_the_same_behaviour_gives_the_same_signature(tmp_path):
    first = arithmetic_run(tmp_path)
    second = arithmetic_run(tmp_path)
    assert signature(first[0].events(first[1])) == signature(second[0].events(second[1]))


def test_changing_the_system_prompt_is_caught_at_the_first_model_call(tmp_path):
    journal, run_id, registry = arithmetic_run(tmp_path)
    report = replay(journal, run_id, registry, default_policy(), system_prompt="Different prompt.")
    assert not report.ok
    first = report.divergences[0]
    assert first.kind == "llm_response" and "system prompt" in first.message


def test_changing_the_tool_schemas_is_caught(tmp_path):
    journal, run_id, registry = arithmetic_run(tmp_path)

    @tool
    def extra() -> str:
        """A new tool the model would now be offered."""
        return ""

    bigger = ToolRegistry([*registry, extra])
    report = replay(journal, run_id, bigger, default_policy(), system_prompt=PROMPT)
    assert not report.ok and "tool schemas" in report.divergences[0].message


def test_a_policy_change_that_flips_a_decision_is_caught(tmp_path):
    journal, run_id, registry = arithmetic_run(tmp_path)
    stricter = Policy([Rule("calculator", Action.DENY)], Action.DENY)
    report = replay(journal, run_id, registry, stricter, system_prompt=PROMPT)
    assert not report.ok
    message = next(d.message for d in report.divergences if d.kind == "policy_decision")
    assert "call 1.0 changed" in message and "recorded allow, now deny" in message


def test_a_policy_that_would_now_ask_diverges_instead_of_hanging(tmp_path):
    journal, run_id, registry = arithmetic_run(tmp_path)
    asking = Policy([Rule("calculator", Action.ASK)], Action.ASK)
    report = replay(journal, run_id, registry, asking, system_prompt=PROMPT)
    assert not report.ok and any("recorded allow, now ask" in d.message for d in report.divergences)


def test_tampering_is_reported_as_an_integrity_problem(tmp_path):
    journal, run_id, registry = arithmetic_run(tmp_path)
    journal._conn.execute(
        "UPDATE events SET data = replace(data, '42', '43') "
        "WHERE run_id = ? AND kind = 'tool_result'",
        (run_id,),
    )
    report = replay(journal, run_id, registry, default_policy(), system_prompt=PROMPT)
    assert (
        not report.ok
        and report.integrity_problems
        and "tool_result" in report.integrity_problems[0]
    )


def test_unknown_runs_cannot_be_replayed(tmp_path):
    with pytest.raises(ReplayError, match="not found"):
        replay(Journal(), "nope", ToolRegistry([calculator]), default_policy())
