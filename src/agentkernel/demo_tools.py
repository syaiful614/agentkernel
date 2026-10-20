"""A small assistant workspace that reproduces the lethal trifecta safely, on local files.

  notes/*.txt   private data           read_notes / list_notes   -> reads_private
  inbox.txt     attacker-reachable     read_inbox                -> ingests_untrusted
  outbox.jsonl  the way out            send_email                -> can_exfiltrate + mutates

Nothing is ever sent anywhere: "sending" appends a line to outbox.jsonl.
"""

from __future__ import annotations

import ast
import json
import math
import operator
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

from .tools import EXFIL, MUTATES, PRIVATE, UNTRUSTED, ToolRegistry, tool

_BINOPS: dict[type, Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_FUNCS: dict[str, Callable[..., Any]] = {"sqrt": math.sqrt, "abs": abs, "round": round}


def _safe_eval(node: ast.AST) -> Any:
    """Whitelist evaluator: arithmetic only, never eval()."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("exponent too large")
        return _BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _safe_eval(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCS
        and not node.keywords
    ):
        return _FUNCS[node.func.id](*[_safe_eval(a) for a in node.args])
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


@tool
def calculator(expression: Annotated[str, Field(description="e.g. '(17*23)+sqrt(144)'")]) -> str:
    """Evaluate an arithmetic expression exactly (+ - * / // % **, sqrt, abs, round)."""
    if len(expression) > 200:
        raise ValueError("expression too long")
    result = _safe_eval(ast.parse(expression.strip(), mode="eval"))
    return f"{result:.12g}" if isinstance(result, float) else str(result)


def make_demo_tools(root: str | Path) -> ToolRegistry:
    base = Path(root).resolve()
    notes_dir = (base / "notes").resolve()

    def notes_path(name: str) -> Path:
        target = (notes_dir / name).resolve()
        if not target.is_relative_to(notes_dir):
            raise PermissionError("path escapes the notes folder")
        return target

    @tool(capabilities=[PRIVATE])
    def list_notes() -> str:
        """List the names of the user's private notes."""
        if not notes_dir.is_dir():
            return "(no notes)"
        return "\n".join(sorted(p.name for p in notes_dir.iterdir() if p.is_file())) or "(no notes)"

    @tool(capabilities=[PRIVATE])
    def read_notes(name: Annotated[str, Field(description="Note file name, e.g. todo.txt")]) -> str:
        """Read one of the user's private notes."""
        return notes_path(name).read_text(encoding="utf-8", errors="replace")

    @tool(capabilities=[UNTRUSTED])
    def read_inbox() -> str:
        """Read the user's inbox (messages written by other people)."""
        inbox = base / "inbox.txt"
        return inbox.read_text(encoding="utf-8", errors="replace") if inbox.exists() else "(empty)"

    @tool(capabilities=[EXFIL, MUTATES])
    def send_email(to: str, subject: str, body: str) -> str:
        """Send an email on the user's behalf."""
        with (base / "outbox.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"to": to, "subject": subject, "body": body}) + "\n")
        return f"email queued to {to}"

    return ToolRegistry([calculator, list_notes, read_notes, read_inbox, send_email])
