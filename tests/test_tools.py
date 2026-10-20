from typing import Annotated

import pytest
from pydantic import Field

from agentkernel import EXFIL, MUTATES, PRIVATE, ToolError, ToolRegistry, tool
from agentkernel.demo_tools import calculator, make_demo_tools


@tool
def add(a: Annotated[int, Field(description="first")], b: int = 1) -> int:
    """Add two integers."""
    return a + b


def test_schema_comes_from_type_hints():
    api = add.to_api()
    assert api["name"] == "add" and api["description"] == "Add two integers."
    assert api["input_schema"]["properties"]["a"]["description"] == "first"
    assert api["input_schema"]["required"] == ["a"] and "title" not in api["input_schema"]


def test_validation_errors_and_exceptions_become_error_results():
    assert add.run({"a": 2, "b": 3}).content == "5"
    bad = add.run({"a": "x"})
    assert bad.is_error and "Invalid arguments" in bad.content

    @tool
    def boom() -> str:
        """Fails."""
        raise RuntimeError("kaput")

    @tool
    def polite() -> str:
        """Fails politely."""
        raise ToolError("not allowed here")

    assert boom.run({}).content == "RuntimeError: kaput"
    assert polite.run({}).content == "not allowed here" and polite.run({}).is_error


def test_output_is_truncated():
    @tool
    def big() -> str:
        """Huge."""
        return "x" * 9000

    assert "truncated 1000 chars" in big.run({}).content


def test_capabilities_are_validated_and_idempotency_fails_safe():
    @tool(capabilities=[MUTATES])
    def writes() -> str:
        """Writes."""
        return "ok"

    @tool(capabilities=[PRIVATE])
    def reads() -> str:
        """Reads."""
        return "ok"

    @tool(capabilities=[MUTATES], idempotent=True)
    def safe_write() -> str:
        """Writes idempotently."""
        return "ok"

    assert writes.idempotent is False  # mutating tools are not repeated blindly
    assert reads.idempotent is True and safe_write.idempotent is True
    with pytest.raises(ValueError, match="unknown capabilities"):

        @tool(capabilities=["reads_everything"])
        def nope() -> str:
            """Nope."""
            return ""


def test_missing_docstring_and_duplicates_rejected():
    def nodoc(x: int) -> int:
        return x

    with pytest.raises(ValueError):
        tool(nodoc)
    reg = ToolRegistry([add])
    with pytest.raises(ValueError):
        reg.register(add)
    assert "add" in reg and len(reg) == 1 and reg.names == ["add"]


@pytest.mark.parametrize("expr,expected", [("(17*23)+sqrt(144)", "403"), ("2**10", "1024")])
def test_calculator(expr, expected):
    assert calculator.run({"expression": expr}).content == expected


@pytest.mark.parametrize("expr", ["__import__('os').system('id')", "open('x')", "2**100000"])
def test_calculator_never_executes_code(expr):
    assert calculator.run({"expression": expr}).is_error


def test_demo_tools_carry_the_right_labels(tmp_path):
    reg = make_demo_tools(tmp_path)
    labels = {t.name: set(t.capabilities) for t in reg}
    assert labels["read_notes"] == {PRIVATE} and labels["list_notes"] == {PRIVATE}
    assert labels["read_inbox"] == {"ingests_untrusted"}
    assert labels["send_email"] == {EXFIL, MUTATES} and not reg.get("send_email").idempotent
    assert labels["calculator"] == set()


def test_notes_tool_cannot_leave_its_folder(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "a.txt").write_text("hello")
    (tmp_path / "secret.txt").write_text("outside")
    reg = make_demo_tools(tmp_path)
    assert reg.get("read_notes").run({"name": "a.txt"}).content == "hello"
    escaped = reg.get("read_notes").run({"name": "../secret.txt"})
    assert escaped.is_error and "outside" not in escaped.content
