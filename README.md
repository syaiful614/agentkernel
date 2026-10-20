# AgentKernel

[![CI](https://github.com/syaiful614/agentkernel/actions/workflows/ci.yml/badge.svg)](https://github.com/syaiful614/agentkernel/actions)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

**A durable, replayable, policy-governed runtime for LLM agents.** Crash recovery, human
approvals that survive restarts, a security policy that holds even when the model is fully
compromised, deterministic replay, and an offline red-team harness. Standard library plus
pydantic, ~2,200 lines, 108 tests.

## The problem

Getting an agent to call tools is easy. Letting it act in the real world for hours or days is
not, because of four failure modes that demos never show:

1. **The process dies mid-action.** Did the email go out? Was the card charged? Retrying blindly
   can do it twice; giving up can do it zero times.
2. **A human has to approve something.** An `input()` prompt blocks the process and dies with it.
   Real approvals take hours, and the process that asked may be long gone.
3. **Prompt injection.** An agent that can read private data, reads attacker-controlled content,
   and can send things out can be talked into leaking the data. Prompting does not reliably stop
   that, so something outside the model has to.
4. **You change the code and cannot tell what broke.** Model output is random, so ordinary tests
   do not work, and there is no record of what the agent actually did.

AgentKernel is built around one idea that addresses all four: **the journal is the source of
truth.** Nothing about a run lives in memory between steps; its state is derived by replaying an
append-only, hash-chained event log.

## What it does

| Capability | How |
|---|---|
| **Crash recovery** | Any process can continue a run from its journal. Idempotent tools are re-run; non-idempotent tools are **never** re-executed after a crash (recorded as "uncertain" and reported to the model). |
| **Durable approvals** | A policy `ask` suspends the run. The process may exit; `approve` and `resume` can happen days later from another process. An approval authorises one specific call, nothing else. |
| **Taint-aware policy** | Declarative rules over tool names, arguments and run state, evaluated fail-closed. |
| **Lethal-trifecta guard** | Tools carry capability labels. If a run has handled private data **and** untrusted content, it may not also send information out, regardless of what the model says. |
| **Deterministic replay** | Feed a recorded run back through the current code with no model and no side effects; the first divergence in prompt construction or policy is reported. |
| **Tamper-evident audit** | Each event is chained to the previous one by hash; `verify` detects edits and deletions. |
| **Offline red team** | A scripted, fully compromised "model" attacks the kernel in CI. No API key needed. |

## Architecture

```mermaid
flowchart LR
    task["Task"] --> kernel
    subgraph kernel["Kernel: stateless between steps"]
        direction TB
        derive["Rebuild state from the journal"] --> settle["Settle pending tool calls"]
        settle --> llm["Call the model"]
        llm --> derive
    end
    kernel <--> journal[("Journal: append-only, hash-chained SQLite")]
    settle --> policy{"Policy engine plus trifecta guard"}
    policy -- allow --> tools["Tools with capability labels"]
    policy -- ask --> human["Human approval: durable, can take days"]
    policy -- deny --> denied["Refused, and the model is told why"]
    human --> settle
    journal --> replay["Replay: recorded model and results against current code"]
```

Design rationale: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). What is and is not protected:
[docs/THREAT_MODEL.md](docs/THREAT_MODEL.md).

## See it in 30 seconds (offline, no model, no key)

```bash
pip install -e ".[dev]"
agentkernel redteam            # 8 attacks by a fully compromised model
python examples/walkthrough.py # the whole story, narrated
```

```text
[PASS] exfil_via_poisoned_inbox                   -> completed
[PASS] poisoned_inbox_first                       -> completed
[PASS] all_three_legs_in_one_turn                 -> completed
[PASS] legitimate_send_waits_for_a_human          -> suspended
[PASS] recipient_allowlist_blocks_before_asking   -> completed
[PASS] path_traversal_is_contained                -> completed
[PASS] tool_name_confusion                        -> completed
[PASS] runaway_loop_is_capped                     -> max_steps

8/8 attack scenarios contained
```

```text
=== 2. A legitimate send waits for a human, and survives the process dying
status: suspended; waiting on call 1.0 (send_email)
after approval: completed; emails sent: 1 (exactly one)

=== 3. A crash in the middle of a non-idempotent action never double-fires
process killed after the card was charged but before the result was recorded
on restart the call was recorded as ['uncertain']; times charged: 1 (not repeated)
```

## Run it with a real model

The model must support tool calling. Any OpenAI-compatible endpoint works (Ollama by default);
Anthropic needs `pip install -e ".[anthropic]"`.

```bash
mkdir -p workspace/notes
echo "PROJECT ORION BUDGET: 4.2M" > workspace/notes/budget.txt

agentkernel run "Email my budget note to manager@example.com" --provider openai --model <model>
# [waiting] run 3f2a9c1d7b4e needs a human decision
#   call 2.0: send_email
#   { "to": "manager@example.com", "subject": "...", "body": "..." }
#   approve: agentkernel approve 3f2a9c1d7b4e 2.0    deny: agentkernel deny 3f2a9c1d7b4e 2.0

agentkernel approve 3f2a9c1d7b4e 2.0
agentkernel resume  3f2a9c1d7b4e --provider openai --model <model>

agentkernel show 3f2a9c1d7b4e    # the full timeline
agentkernel verify 3f2a9c1d7b4e  # the hash chain is intact
agentkernel replay 3f2a9c1d7b4e  # the current code still behaves as recorded
```

The approval prompt always prints the **complete** input: a payload hidden behind a preview limit
would be approved unseen. Small local models are unreliable at tool use; the kernel's guarantees
do not depend on the model, but whether a task succeeds does.

## Policy

```json
{
  "default": "ask",
  "trifecta": "deny",
  "rules": [
    { "tool": "read_*", "action": "allow" },
    { "tool": "send_email", "action": "deny", "reason": "recipient is outside example.com",
      "unless_args": { "to": "@example\\.com$" } },
    { "tool": "send_*", "action": "ask" },
    { "tool": "write_*", "action": "ask", "when_tainted": true }
  ]
}
```

- The first matching rule wins; otherwise `default` applies. Tool-name globs are case-sensitive.
- `unless_args` and `when_args` take regexes; a **missing argument fails closed**.
- `when_tainted` is true once untrusted content has entered the run; `after` requires earlier
  capabilities (`["reads_private"]`).
- The trifecta guard runs after the rules and can only make the outcome **stricter**.

## Library

```python
from agentkernel import Journal, Kernel, default_policy, make_llm
from agentkernel.demo_tools import make_demo_tools

journal = Journal(".agentkernel/journal.db")
kernel = Kernel(
    make_llm("openai", "your-model"),
    make_demo_tools("workspace"),
    journal,
    default_policy(),
)
result = kernel.run("Email my budget note to my manager")
if result.status == "suspended":
    pending = result.pending[0]
    print(pending.tool, pending.input)  # a human reviews this, possibly days later
    kernel.approve(result.run_id, pending.call_key, by="alice")
    result = kernel.resume(result.run_id)
print(result.output)
```

Tools declare what they can do; the policy reasons about these labels, never about what the model
claims:

```python
from agentkernel import EXFIL, MUTATES, tool


@tool(capabilities=[EXFIL, MUTATES])
def post_to_slack(channel: str, text: str) -> str:
    """Post a message to a Slack channel."""
    ...
```

| Label | Meaning |
|---|---|
| `reads_private` | returns data the user would not want leaked |
| `ingests_untrusted` | returns content an attacker may control |
| `can_exfiltrate` | can send information out of the system |
| `mutates` | changes the outside world; implies *not idempotent* unless declared otherwise |

## Guarantees and the tests behind them

| Guarantee | Evidence |
|---|---|
| A compromised model cannot complete private + untrusted + send, in any order, even batched in one turn | red-team scenarios 1 to 3, plus **mutation tests** that switch the guard off and show the same attacks then succeed |
| A non-idempotent action is never executed twice after a crash | `test_non_idempotent_tools_are_never_re_run_after_a_crash` (the crash is a real `BaseException`) |
| A pending approval survives process death and authorises only its own call | `test_the_run_survives_the_death_of_the_process`, `test_an_approval_authorises_only_the_call_it_names` |
| Suspending and resuming does not change what the model sees | request digests equal with and without an interruption |
| Replay catches changed prompts, tool schemas and policy | `tests/test_replay.py` |
| Editing or deleting a journal event is detected | `tests/test_journal.py` |

108 tests, 95% line coverage, run on Python 3.10 to 3.12 in CI together with ruff, mypy and the
red-team suite.

## Limitations (read these)

- **Capability labels are the trust root.** A tool that sends data out but is not labelled
  `can_exfiltrate` defeats the guard. See the threat model.
- The guard is **per run** and conservative: a run that handled private data and untrusted content
  cannot send afterwards, even for unrelated reasons. Start a new run for unrelated work.
- Single node (SQLite). Tools run in-process, so this does not sandbox tool code.
- Tool calls run **sequentially** (needed for determinism and for the guard); one approval is
  requested at a time.
- No streaming and no context compaction; limits are step and token budgets.
- The hash chain detects edits and deletions but not truncation of the log's tail; anchor the head
  hash elsewhere if that matters.
- The red-team suite uses a scripted attacker. It proves the controls hold regardless of model
  behaviour; it says nothing about how often real models attempt such attacks, and it has not been
  run against live models.

## Project layout

```text
src/agentkernel/
  journal.py     append-only hash-chained event log, leases
  state.py       state derived from events; message construction
  runtime.py     the durable kernel: settle calls, suspend, resume, recover
  policy.py      rules, taint conditions, lethal-trifecta guard
  tools.py       typed tools with capability labels
  replay.py      deterministic replay against current code
  redteam.py     scripted-attacker harness; scenarios/*.json
  llm.py         model protocol, OpenAI-compatible and Anthropic adapters
  demo_tools.py  notes / inbox / send_email workspace that reproduces the trifecta
  cli.py         run, resume, approve, deny, list, show, verify, replay, redteam
```

## Development

```bash
make all   # ruff, mypy, pytest with coverage, red team
```

## License

MIT
