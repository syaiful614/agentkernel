import json

import pytest

from agentkernel import PolicyError, run_all, run_scenario
from agentkernel.redteam import ScriptedAttacker, load_scenarios

SCENARIOS = {s["name"]: s for s in load_scenarios()}


def test_the_suite_is_substantial_and_well_formed():
    assert len(SCENARIOS) >= 8
    for s in SCENARIOS.values():
        assert {"name", "task", "turns", "expect"} <= set(s) and s["description"]


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_every_attack_is_contained(name):
    result = run_scenario(SCENARIOS[name])
    assert result.passed, result.failures


def _without_the_guard(name):
    open_policy = {"default": "allow", "trifecta": "allow", "rules": []}
    return {**SCENARIOS[name], "policy": open_policy}


@pytest.mark.parametrize(
    "name", ["exfil_via_poisoned_inbox", "poisoned_inbox_first", "all_three_legs_in_one_turn"]
)
def test_the_exfiltration_scenarios_really_depend_on_the_guard(name):
    """Mutation check: switch the protection off and the very same attack must succeed."""
    result = run_scenario(_without_the_guard(name))
    assert not result.passed
    assert any("email(s) were sent" in f or "reached the outbox" in f for f in result.failures)


def test_the_allowlist_scenario_depends_on_its_rule():
    scenario = dict(SCENARIOS["recipient_allowlist_blocks_before_asking"])
    policy = json.loads(json.dumps(scenario["policy"]))
    policy["rules"] = [r for r in policy["rules"] if r.get("action") != "deny"]
    result = run_scenario({**scenario, "policy": policy})
    assert not result.passed


def test_the_path_scenario_depends_on_the_sandbox(tmp_path):
    """If the tool read the file, the secret would show up in a result and the check would fail."""
    scenario = SCENARIOS["path_traversal_is_contained"]
    leaked = {**scenario, "turns": [{"calls": [{"tool": "read_notes", "input": {"name": "x"}}]}]}
    leaked["files"] = {**scenario["files"], "notes/x": "TOP-SECRET-OUTSIDE"}
    assert not run_scenario(leaked).passed  # a legitimate read of that text is detected


def test_scripted_attacker_expands_repeats_and_then_stops():
    attacker = ScriptedAttacker([{"calls": [{"tool": "t"}], "repeat": 2}, {"say": "bye"}])
    kinds = [attacker.generate().content[0]["type"] for _ in range(4)]
    assert kinds == ["tool_use", "tool_use", "text", "text"]  # an exhausted script just says done


def test_custom_scenario_folders_and_failures_are_reported(tmp_path):
    bad = {
        "name": "expects_an_email",
        "description": "wrongly expects mail to be sent",
        "task": "x",
        "turns": [{"say": "nothing happens"}],
        "expect": {"outbox_count": 1},
    }
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    (result,) = run_all(tmp_path)
    assert not result.passed and "0 email(s) were sent, expected 1" in result.failures[0]


def test_an_invalid_scenario_policy_is_rejected():
    broken = {**SCENARIOS["runaway_loop_is_capped"], "policy": {"default": "sometimes"}}
    with pytest.raises(PolicyError):
        run_scenario(broken)
