"""Tools with *capability labels*: the trust root of the security policy.

Every tool declares what it can do to the world. The policy engine reasons about those labels
(never about what the model says it is doing), so a mislabelled tool is the one way to defeat
the guard. Labels are therefore explicit, fail-safe, and part of the tool definition.

  reads_private      returns data the user would not want to leak (notes, files, secrets)
  ingests_untrusted  returns content an attacker may control (email, web pages, uploads)
  can_exfiltrate     can send information out of the system (email, HTTP, publish, write public)
  mutates            changes the outside world (implies "not safe to repeat" by default)
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, get_type_hints

from pydantic import BaseModel, ValidationError, create_model

PRIVATE = "reads_private"
UNTRUSTED = "ingests_untrusted"
EXFIL = "can_exfiltrate"
MUTATES = "mutates"
CAPABILITIES = frozenset({PRIVATE, UNTRUSTED, EXFIL, MUTATES})

MAX_OUTPUT_CHARS = 8_000


def clean_schema(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {
            k: clean_schema(v)
            for k, v in schema.items()
            if not (k == "title" and isinstance(v, str))
        }
    if isinstance(schema, list):
        return [clean_schema(v) for v in schema]
    return schema


def truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


class ToolError(Exception):
    """Raise from a tool to report a failure to the model without a type-name prefix."""


@dataclass
class ToolResult:
    content: str
    is_error: bool = False


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[..., Any]
    model: type[BaseModel]
    capabilities: frozenset[str] = field(default_factory=frozenset)
    idempotent: bool = True

    def to_api(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def run(self, raw: dict[str, Any] | None) -> ToolResult:
        try:
            args = self.model.model_validate(raw or {})
        except ValidationError as exc:
            detail = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or 'input'}: {e['msg']}" for e in exc.errors()
            )
            return ToolResult(f"Invalid arguments for '{self.name}': {detail}", True)
        kwargs = {k: getattr(args, k) for k in type(args).model_fields}
        try:
            out = self.fn(**kwargs)
        except ToolError as exc:
            return ToolResult(truncate(str(exc)), True)
        except Exception as exc:  # a buggy tool must not kill the run
            return ToolResult(f"{type(exc).__name__}: {exc}", True)
        if not isinstance(out, str):
            out = json.dumps(out, default=str, ensure_ascii=False)
        return ToolResult(truncate(out))


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    capabilities: Iterable[str] = (),
    idempotent: bool | None = None,
) -> Any:
    """Turn a typed function into a `Tool`.

    `idempotent` defaults to False when the tool `mutates` (fail-safe: after a crash the kernel
    will not re-run it blindly) and to True otherwise.
    """
    caps = frozenset(capabilities)
    unknown = caps - CAPABILITIES
    if unknown:
        raise ValueError(f"unknown capabilities {sorted(unknown)}; use {sorted(CAPABILITIES)}")

    def build(f: Callable[..., Any]) -> Tool:
        desc = description or inspect.getdoc(f)
        if not desc:
            raise ValueError(f"Tool '{f.__name__}' needs a docstring/description for the model")
        hints = get_type_hints(f, include_extras=True)
        fields: dict[str, Any] = {}
        for pname, param in inspect.signature(f).parameters.items():
            default = ... if param.default is inspect.Parameter.empty else param.default
            fields[pname] = (hints.get(pname, str), default)
        model = create_model(f"{f.__name__}_args", **fields)
        return Tool(
            name=name or f.__name__,
            description=desc.strip(),
            input_schema=clean_schema(model.model_json_schema()),
            fn=f,
            model=model,
            capabilities=caps,
            idempotent=(MUTATES not in caps) if idempotent is None else idempotent,
        )

    return build(fn) if fn is not None else build


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools:
            self.register(t)

    def register(self, t: Tool) -> None:
        if t.name in self._tools:
            raise ValueError(f"Duplicate tool name: {t.name}")
        self._tools[t.name] = t

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self) -> Any:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.to_api() for t in self._tools.values()]
