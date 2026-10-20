import json

import pytest

from agentkernel import (
    EXFIL,
    MUTATES,
    PRIVATE,
    UNTRUSTED,
    Action,
    Policy,
    PolicyError,
    Rule,
    default_policy,
    tool,
)


@tool(capabilities=[EXFIL, MUTATES])
def send_email(to: str, subject: str = "", body: str = "") -> str:
    """Send mail."""
    return "ok"


@tool(capabilities=[PRIVATE])
def read_notes(name: str) -> str:
    """Read notes."""
    return "x"


def decide(policy, name, args, tool_obj, seen=()):
    return policy.evaluate(name, args, tool_obj, seen)


def test_first_matching_rule_wins_and_default_applies_otherwise():
    p = Policy([Rule("read_*", Action.ALLOW, "ok"), Rule("read_*", Action.DENY)], Action.ASK)
    assert decide(p, "read_notes", {}, read_notes).action is Action.ALLOW
    d = decide(p, "mystery", {}, None)
    assert d.action is Action.ASK and d.source == "default"


def test_globs_are_case_sensitive_and_exact():
    p = Policy([Rule("send_email", Action.ALLOW)], Action.DENY)
    assert decide(p, "send_email", {}, None).action is Action.ALLOW
    for name in ("SEND_EMAIL", "send_email ", "send-email", "xsend_email"):
        assert decide(p, name, {}, None).action is Action.DENY


def test_taint_and_capability_conditions():
    p = Policy(
        [
            Rule("write_*", Action.ASK, when_tainted=True),
            Rule("write_*", Action.ALLOW, when_tainted=False),
            Rule("post_*", Action.DENY, after=(PRIVATE,)),
        ],
        Action.ASK,
    )
    assert decide(p, "write_file", {}, None, seen=[UNTRUSTED]).action is Action.ASK
    assert decide(p, "write_file", {}, None, seen=[]).action is Action.ALLOW
    assert decide(p, "post_x", {}, None, seen=[PRIVATE]).action is Action.DENY
    assert decide(p, "post_x", {}, None, seen=[]).action is Action.ASK


def test_argument_conditions_fail_closed():
    p = Policy(
        [
            Rule("send_email", Action.DENY, "outside", unless_args={"to": r"@example\.com$"}),
            Rule("send_email", Action.ALLOW),
        ]
    )
    assert decide(p, "send_email", {"to": "a@example.com"}, send_email).action is Action.ALLOW
    assert decide(p, "send_email", {"to": "a@evil.test"}, send_email).action is Action.DENY
    assert decide(p, "send_email", {}, send_email).action is Action.DENY  # missing arg: closed
    only = Policy([Rule("send_email", Action.DENY, when_args={"to": "evil"})], Action.ALLOW)
    assert decide(only, "send_email", {"to": "x@evil.test"}, send_email).action is Action.DENY
    assert decide(only, "send_email", {}, send_email).action is Action.ALLOW


def test_trifecta_guard_refuses_the_final_leg_even_if_a_rule_allows_it():
    allow_all = Policy([Rule("*", Action.ALLOW)], Action.ALLOW)
    seen = [PRIVATE, UNTRUSTED]
    d = decide(allow_all, "send_email", {}, send_email, seen)
    assert d.action is Action.DENY and d.source == "trifecta"
    # either leg alone is fine: the guard only fires when all three meet
    assert decide(allow_all, "send_email", {}, send_email, [PRIVATE]).action is Action.ALLOW
    assert decide(allow_all, "send_email", {}, send_email, [UNTRUSTED]).action is Action.ALLOW
    # and it only restricts tools that can actually send something out
    assert decide(allow_all, "read_notes", {}, read_notes, seen).action is Action.ALLOW


def test_trifecta_guard_can_only_tighten():
    deny_rule = Policy([Rule("send_email", Action.DENY, "no")], Action.ASK, Action.ASK)
    d = decide(deny_rule, "send_email", {}, send_email, [PRIVATE, UNTRUSTED])
    assert d.action is Action.DENY and d.source == "rule:0"  # the guard never loosens a deny

    asking = Policy([Rule("send_email", Action.ALLOW)], Action.ASK, trifecta=Action.ASK)
    assert decide(asking, "send_email", {}, send_email, [PRIVATE, UNTRUSTED]).action is Action.ASK

    off = Policy([Rule("send_email", Action.ALLOW)], Action.ASK, trifecta=Action.ALLOW)
    assert decide(off, "send_email", {}, send_email, [PRIVATE, UNTRUSTED]).action is Action.ALLOW


def test_a_single_tool_with_all_three_capabilities_is_caught():
    @tool(capabilities=[PRIVATE, UNTRUSTED, EXFIL])
    def everything() -> str:
        """Does it all."""
        return ""

    p = Policy([Rule("*", Action.ALLOW)], Action.ALLOW)
    assert decide(p, "everything", {}, everything, []).source == "trifecta"


def test_default_policy_shape():
    p = default_policy()
    assert decide(p, "calculator", {}, None).action is Action.ALLOW
    assert decide(p, "read_notes", {}, read_notes).action is Action.ALLOW
    assert decide(p, "send_email", {}, send_email).action is Action.ASK
    assert decide(p, "something_new", {}, None).action is Action.ASK
    assert p.trifecta is Action.DENY


def test_roundtrip_digest_and_file_loading(tmp_path):
    p = default_policy()
    again = Policy.from_dict(json.loads(json.dumps(p.to_dict())))
    assert again.digest() == p.digest() and len(again.rules) == len(p.rules)
    changed = Policy.from_dict(p.to_dict())
    changed.default = Action.DENY
    assert changed.digest() != p.digest()

    path = tmp_path / "policy.json"
    path.write_text(json.dumps(p.to_dict()), encoding="utf-8-sig")  # Windows editors add a BOM
    assert Policy.from_file(path).digest() == p.digest()


@pytest.mark.parametrize(
    "data,msg",
    [
        ({"rules": [{"action": "allow"}]}, "needs a string 'tool'"),
        ({"rules": [{"tool": "x", "action": "maybe"}]}, "allow, ask or deny"),
        ({"rules": [{"tool": "x", "action": "allow", "after": "private"}]}, "'after'"),
        ({"rules": [{"tool": "x", "action": "allow", "when_tainted": "yes"}]}, "when_tainted"),
        ({"rules": [{"tool": "x", "action": "allow", "when_args": {"a": "("}}]}, "invalid regex"),
        ({"rules": [{"tool": "x", "action": "allow", "unless_args": []}]}, "must map"),
        ({"default": "sometimes"}, "allow, ask or deny"),
    ],
)
def test_invalid_policies_are_rejected_clearly(data, msg):
    with pytest.raises(PolicyError, match=msg):
        Policy.from_dict(data)


def test_unreadable_policy_files(tmp_path):
    with pytest.raises(PolicyError, match="cannot read"):
        Policy.from_file(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{oops")
    with pytest.raises(PolicyError, match="not valid JSON"):
        Policy.from_file(bad)
