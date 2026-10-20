"""Command line: run | list | show | approve | deny | resume | replay | verify | redteam."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .demo_tools import make_demo_tools
from .journal import Event, Journal
from .llm import LLM, LLMError, make_llm
from .policy import Policy, PolicyError, default_policy
from .redteam import run_all
from .replay import ReplayError, replay
from .runtime import Kernel, KernelError, RunResult

SYSTEM_PROMPT = (
    "You are a personal assistant with access to the user's private notes, their inbox and "
    "their email. Use the tools to do what the user asks, and answer concisely."
)


class CLIError(Exception):
    """Invalid command-line usage."""


class _NoLLM:
    model = "none"

    def generate(self, **_: Any) -> Any:
        raise LLMError("this command does not call a model")


def _preview(value: Any, limit: int = 70) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def format_event(ev: Event) -> str:
    d = ev.data
    kind = ev.kind
    if kind == "run_started":
        detail = f"{_preview(d['task'])}  [model {d.get('model')}, {len(d.get('tools', []))} tools]"
    elif kind == "llm_response":
        calls = [b["name"] for b in d["content"] if b.get("type") == "tool_use"]
        what = f"calls {', '.join(calls)}" if calls else f"says {_preview(_text(d['content']))}"
        detail = f"step {d['step']}: {what} ({d['usage']['input'] + d['usage']['output']} tokens)"
    elif kind == "policy_decision":
        verdict = d["decision"].upper()
        detail = f"{d['call']} {d['tool']} -> {verdict} ({d['source']}) {_preview(d['reason'], 60)}"
    elif kind == "approval_requested":
        detail = f"{d['call']} {d['tool']}({_preview(d['input'], 50)}) needs a human"
    elif kind == "approval_resolved":
        verdict = "APPROVED" if d["approved"] else "REJECTED"
        detail = f"{d['call']} {verdict} by {d.get('by', '?')}"
    elif kind == "tool_started":
        detail = f"{d['call']} started" + (" (retry)" if d.get("retry") else "")
    elif kind == "tool_result":
        detail = f"{d['call']} [{d['status']}] {_preview(d['content'], 60)}"
    elif kind == "run_suspended":
        detail = f"waiting for approval of {', '.join(d.get('pending', []))}"
    elif kind == "run_finished":
        detail = f"{d['status']}: {_preview(d.get('output', ''))}"
    else:
        detail = _preview(d, 80)
    return f"{ev.seq:>4}  {kind:<18} {detail}"


def _text(content: list[dict[str, Any]]) -> str:
    return "".join(b.get("text", "") for b in content if b.get("type") == "text")


# --------------------------------------------------------------------------- wiring
def _journal(args: argparse.Namespace) -> Journal:
    return Journal(args.db)


def _policy(args: argparse.Namespace) -> Policy:
    return Policy.from_file(args.policy) if args.policy else default_policy()


def _llm(args: argparse.Namespace) -> LLM:
    provider = args.provider or os.environ.get("AGENTKERNEL_PROVIDER", "openai")
    model = args.model or os.environ.get("AGENTKERNEL_MODEL")
    if provider == "anthropic":
        model = model or "claude-sonnet-5-5"
        key = os.environ.get("ANTHROPIC_API_KEY")
    else:
        if not model:
            raise LLMError("Set --model (or AGENTKERNEL_MODEL), e.g. an Ollama model name.")
        key = os.environ.get("AGENTKERNEL_API_KEY")
    return make_llm(
        provider,
        model,
        api_key=key,
        base_url=args.base_url or os.environ.get("AGENTKERNEL_BASE_URL"),
        timeout=float(os.environ.get("AGENTKERNEL_TIMEOUT", "120")),
    )


def _kernel(args: argparse.Namespace, journal: Journal, *, with_llm: bool) -> Kernel:
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    on_event = (lambda ev: print(format_event(ev), file=sys.stderr)) if args.verbose else None
    return Kernel(
        _llm(args) if with_llm else _NoLLM(),
        make_demo_tools(workspace),
        journal,
        _policy(args),
        system_prompt=SYSTEM_PROMPT,
        max_steps=args.max_steps,
        token_budget=args.token_budget,
        on_event=on_event,
    )


def _report(result: RunResult) -> int:
    if result.status == "suspended":
        print(f"[waiting] run {result.run_id} needs a human decision", file=sys.stderr)
        for p in result.pending:
            # Never truncate what a human is asked to approve: a payload hidden past a preview
            # limit would be approved unseen.
            print(f"  call {p.call_key}: {p.tool}", file=sys.stderr)
            print(json.dumps(p.input, ensure_ascii=False, indent=2), file=sys.stderr)
            print(f"  why: {p.reason}", file=sys.stderr)
            first = f"  approve: agentkernel approve {result.run_id} {p.call_key}"
            print(
                f"{first}    deny: agentkernel deny {result.run_id} {p.call_key}", file=sys.stderr
            )
        print(f"  then:    agentkernel resume {result.run_id}", file=sys.stderr)
        return 3
    print(result.output)
    print(
        f"\n[{result.status} | run {result.run_id} | {result.steps} steps | "
        f"{result.usage.total} tokens]",
        file=sys.stderr,
    )
    return 0 if result.ok else 2


# --------------------------------------------------------------------------- commands
def _cmd_run(args: argparse.Namespace) -> int:
    journal = _journal(args)
    return _report(_kernel(args, journal, with_llm=True).run(args.task))


def _cmd_resume(args: argparse.Namespace) -> int:
    journal = _journal(args)
    return _report(_kernel(args, journal, with_llm=True).resume(args.run_id))


def _decide(args: argparse.Namespace, approved: bool) -> int:
    journal = _journal(args)
    kernel = _kernel(args, journal, with_llm=False)
    kernel.approve(args.run_id, args.call, approved=approved, by="cli", note=args.note or "")
    verdict = "approved" if approved else "rejected"
    print(f"{verdict} call {args.call}. Continue with: agentkernel resume {args.run_id}")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    runs = _journal(args).runs(args.limit)
    if not runs:
        print("(no runs yet)")
    for r in runs:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(r.created_ts))
        print(f"{r.run_id}  {when}  {r.status:<16} {_preview(r.task, 60)}")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    journal = _journal(args)
    if journal.get_run(args.run_id) is None:
        raise KernelError(f"unknown run '{args.run_id}'")
    for ev in journal.events(args.run_id):
        if args.json:
            print(json.dumps({"seq": ev.seq, "ts": ev.ts, "kind": ev.kind, "data": ev.data}))
        else:
            print(format_event(ev))
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    journal = _journal(args)
    if journal.get_run(args.run_id) is None:
        raise KernelError(f"unknown run '{args.run_id}'")
    problems = journal.verify(args.run_id)
    if not problems:
        print(f"run {args.run_id}: journal intact ({len(journal.events(args.run_id))} events)")
        return 0
    for p in problems:
        print(f"TAMPERED: {p}")
    return 1


def _cmd_replay(args: argparse.Namespace) -> int:
    journal = _journal(args)
    registry = make_demo_tools(Path(args.workspace))
    report = replay(journal, args.run_id, registry, _policy(args), system_prompt=SYSTEM_PROMPT)
    if report.ok:
        print(f"run {args.run_id}: replay matches the recording ({report.compared} events)")
        return 0
    for problem in report.integrity_problems:
        print(f"INTEGRITY: {problem}")
    for d in report.divergences[:5]:
        print(f"DIVERGES at event {d.index}: {d.message}")
    return 1


def _cmd_redteam(args: argparse.Namespace) -> int:
    results = run_all(args.scenarios)
    for r in results:
        print(f"[{'PASS' if r.passed else 'FAIL'}] {r.name:<42} -> {r.status}")
        for failure in r.failures:
            print(f"        - {failure}")
    passed = sum(r.passed for r in results)
    print(f"\n{passed}/{len(results)} attack scenarios contained")
    return 0 if passed == len(results) else 1


# --------------------------------------------------------------------------- parser
def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", default=".agentkernel/journal.db", help="journal database")
    common.add_argument("--workspace", default="workspace", help="folder with notes/ and inbox.txt")
    common.add_argument("--policy", help="policy JSON file (default: built-in conservative policy)")
    common.add_argument("--provider", choices=["openai", "anthropic"])
    common.add_argument("--model")
    common.add_argument("--base-url", help="OpenAI-compatible endpoint (default: local Ollama)")
    common.add_argument("--max-steps", type=int, default=20)
    common.add_argument("--token-budget", type=int, default=200_000)
    common.add_argument("-v", "--verbose", action="store_true", help="print events as they happen")

    parser = argparse.ArgumentParser(prog="agentkernel", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", parents=[common], help="start a run")
    p.add_argument("task")
    p.set_defaults(fn=_cmd_run)

    p = sub.add_parser("resume", parents=[common], help="continue a suspended or interrupted run")
    p.add_argument("run_id")
    p.set_defaults(fn=_cmd_resume)

    for name, approved, text in (("approve", True, "approve"), ("deny", False, "reject")):
        p = sub.add_parser(name, parents=[common], help=f"{text} one pending call")
        p.add_argument("run_id")
        p.add_argument("call", help="call key shown by `run`, e.g. 2.0")
        p.add_argument("--note")
        p.set_defaults(fn=lambda a, ok=approved: _decide(a, ok))

    p = sub.add_parser("list", parents=[common], help="list runs")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(fn=_cmd_list)

    p = sub.add_parser("show", parents=[common], help="print a run's timeline")
    p.add_argument("run_id")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=_cmd_show)

    p = sub.add_parser("verify", parents=[common], help="check a run's hash chain")
    p.add_argument("run_id")
    p.set_defaults(fn=_cmd_verify)

    p = sub.add_parser("replay", parents=[common], help="replay a run against the current code")
    p.add_argument("run_id")
    p.set_defaults(fn=_cmd_replay)

    p = sub.add_parser("redteam", help="run the offline attack scenarios")
    p.add_argument("--scenarios", help="folder of scenario JSON files (default: built-in)")
    p.set_defaults(fn=_cmd_redteam)

    args = parser.parse_args(argv)
    try:
        return int(args.fn(args))
    except (LLMError, KernelError, PolicyError, ReplayError, CLIError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
