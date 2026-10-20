"""Security policy: declarative rules + a guard against the "lethal trifecta".

The lethal trifecta (a run that can see private data, ingests attacker-controlled content, and
can send information out) is the structural cause of most agent data-exfiltration incidents:
a prompt injection hidden in untrusted content tells the model to read the private data and
send it out. No amount of prompting reliably prevents that, so the kernel does not rely on the
model behaving. It tracks which capabilities a run has *exercised* and refuses the final leg.

Evaluation order for each tool call:
  1. first matching rule wins (otherwise the default action),
  2. then the trifecta guard may only make the outcome *stricter* (allow < ask < deny).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from .tools import EXFIL, PRIVATE, UNTRUSTED, Tool
from .util import digest


class PolicyError(ValueError):
    """The policy definition is invalid."""


class Action(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


_STRICTNESS = {Action.ALLOW: 0, Action.ASK: 1, Action.DENY: 2}


@dataclass(frozen=True)
class Decision:
    action: Action
    reason: str
    source: str  # "rule:<index>" | "default" | "trifecta" | "unknown_tool"


@dataclass(frozen=True)
class Rule:
    """One policy rule. All conditions must hold for the rule to apply.

    tool         glob over the tool name (case-sensitive)
    when_tainted True/False: whether untrusted content has entered the run
    after        capabilities that must already have been exercised in the run
    when_args    {arg: regex} each argument must be present and match
    unless_args  {arg: regex} rule applies unless the argument is present and matches
                 (a missing argument therefore fails closed: the rule applies)
    """

    tool: str
    action: Action
    reason: str = ""
    when_tainted: bool | None = None
    after: tuple[str, ...] = ()
    when_args: Mapping[str, str] = field(default_factory=dict)
    unless_args: Mapping[str, str] = field(default_factory=dict)

    def matches(
        self, name: str, args: Mapping[str, Any], tainted: bool, seen: frozenset[str]
    ) -> bool:
        if not fnmatchcase(name, self.tool):
            return False
        if self.when_tainted is not None and self.when_tainted != tainted:
            return False
        if not set(self.after) <= seen:
            return False
        for arg, pattern in self.when_args.items():
            if arg not in args or not re.search(pattern, str(args[arg])):
                return False
        for arg, pattern in self.unless_args.items():
            if arg in args and re.search(pattern, str(args[arg])):
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"tool": self.tool, "action": self.action.value}
        if self.reason:
            out["reason"] = self.reason
        if self.when_tainted is not None:
            out["when_tainted"] = self.when_tainted
        if self.after:
            out["after"] = list(self.after)
        if self.when_args:
            out["when_args"] = dict(self.when_args)
        if self.unless_args:
            out["unless_args"] = dict(self.unless_args)
        return out


def _action(value: Any, where: str) -> Action:
    try:
        return Action(value)
    except ValueError:
        raise PolicyError(f"{where}: action must be allow, ask or deny (got {value!r})") from None


def _patterns(value: Any, where: str) -> dict[str, str]:
    if not isinstance(value, dict) or not all(isinstance(v, str) for v in value.values()):
        raise PolicyError(f"{where} must map argument names to regex strings")
    for pattern in value.values():
        try:
            re.compile(pattern)
        except re.error as exc:
            raise PolicyError(f"{where}: invalid regex {pattern!r}: {exc}") from exc
    return dict(value)


@dataclass
class Policy:
    rules: list[Rule] = field(default_factory=list)
    default: Action = Action.ASK
    trifecta: Action = Action.DENY  # ALLOW switches the guard off (not recommended)

    def evaluate(
        self,
        name: str,
        args: Mapping[str, Any],
        tool: Tool | None,
        seen: Iterable[str],
    ) -> Decision:
        seen_set = frozenset(seen)
        tainted = UNTRUSTED in seen_set
        decision = Decision(self.default, "no rule matched; default policy", "default")
        for index, rule in enumerate(self.rules):
            if rule.matches(name, args, tainted, seen_set):
                reason = rule.reason or f"matched rule {index}"
                decision = Decision(rule.action, reason, f"rule:{index}")
                break
        if tool is not None and EXFIL in tool.capabilities:
            effective = seen_set | tool.capabilities
            if {PRIVATE, UNTRUSTED, EXFIL} <= effective:
                if _STRICTNESS[self.trifecta] > _STRICTNESS[decision.action]:
                    decision = Decision(
                        self.trifecta,
                        "lethal-trifecta guard: this run has handled private data and untrusted "
                        "content, so it may not also send information out",
                        "trifecta",
                    )
        return decision

    # ------------------------------------------------------------------ (de)serialisation
    def to_dict(self) -> dict[str, Any]:
        return {
            "default": self.default.value,
            "trifecta": self.trifecta.value,
            "rules": [r.to_dict() for r in self.rules],
        }

    def digest(self) -> str:
        return digest(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Policy:
        if not isinstance(data, Mapping):
            raise PolicyError("a policy must be a JSON object")
        rules: list[Rule] = []
        for i, raw in enumerate(data.get("rules", [])):
            where = f"rules[{i}]"
            if not isinstance(raw, Mapping) or not isinstance(raw.get("tool"), str):
                raise PolicyError(f"{where}: needs a string 'tool' glob")
            after = raw.get("after", [])
            if not isinstance(after, list) or not all(isinstance(a, str) for a in after):
                raise PolicyError(f"{where}: 'after' must be a list of capability names")
            tainted = raw.get("when_tainted")
            if tainted is not None and not isinstance(tainted, bool):
                raise PolicyError(f"{where}: 'when_tainted' must be true or false")
            rules.append(
                Rule(
                    tool=raw["tool"],
                    action=_action(raw.get("action"), where),
                    reason=str(raw.get("reason", "")),
                    when_tainted=tainted,
                    after=tuple(after),
                    when_args=_patterns(raw.get("when_args", {}), f"{where}.when_args"),
                    unless_args=_patterns(raw.get("unless_args", {}), f"{where}.unless_args"),
                )
            )
        return cls(
            rules=rules,
            default=_action(data.get("default", "ask"), "default"),
            trifecta=_action(data.get("trifecta", "deny"), "trifecta"),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> Policy:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8-sig"))  # tolerate a BOM
        except OSError as exc:
            raise PolicyError(f"cannot read policy {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise PolicyError(f"policy {path} is not valid JSON: {exc}") from exc
        return cls.from_dict(data)


def default_policy() -> Policy:
    """Conservative defaults: reads are free, anything outbound needs a human, guard is on."""
    return Policy(
        rules=[
            Rule("calculator", Action.ALLOW, "pure computation"),
            Rule("list_*", Action.ALLOW, "read-only listing"),
            Rule("read_*", Action.ALLOW, "read-only access"),
            Rule("send_*", Action.ASK, "outbound communication needs a human"),
        ],
        default=Action.ASK,
        trifecta=Action.DENY,
    )
