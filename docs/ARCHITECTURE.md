# Architecture and design decisions

AgentKernel is a small runtime (~2,200 lines) built so that an agent can act in the real world
for a long time and still be recoverable, reviewable and testable. Everything follows from one
decision: the journal is the source of truth.

## 1. Event sourcing: state is derived, never stored

A run is a sequence of events in an append-only SQLite table. `state.rebuild` turns those events
into the run's state, and `state.build_messages` turns state into the conversation sent to the
model. There is no mutable "current state" to lose in a crash, and live execution, crash
recovery, approval resumption and replay all go through the same two functions, so they cannot
drift apart.

Events: `run_started`, `llm_response`, `policy_decision`, `approval_requested`,
`approval_resolved`, `tool_started`, `tool_result`, `run_suspended`, `run_resumed`, `llm_error`,
`run_finished`.

Each event carries a hash over (previous hash, sequence number, kind, data, timestamp). Editing
or removing an event breaks the chain at that point and `verify` reports it. Limit: truncating
the tail of the log is not detectable by the chain alone.

## 2. The lifecycle of one tool call

```text
llm_response -> policy_decision -+-> deny -----------------------------> tool_result (denied)
                                 |
                                 +-> ask -> approval_requested -> run_suspended
                                 |            ... a human decides, maybe days later ...
                                 |          approval_resolved -+-> rejected -> tool_result (denied)
                                 |                             +-> approved --+
                                 |                                            |
                                 +-> allow -----------------------------------+
                                                                              v
                                                         tool_started -> tool_result
```

Calls identify themselves as `<step>.<index>` (for example `2.0`), not by the provider's id,
because providers reuse ids across steps. Calls in one model turn run **sequentially, in order**.
That costs latency and buys two things: replay is deterministic, and a capability exercised by an
earlier call is visible to the policy check of a later one, so batching the three legs of an
attack into a single turn cannot evade the guard.

## 3. Crash semantics

A crash leaves `tool_started` without `tool_result`. On resume:

| Tool | After a crash |
|---|---|
| idempotent | executed again (`tool_started` is journaled with `retry: true`) |
| not idempotent | **not** executed again; recorded as `uncertain` and the model is told the outcome is unknown |

This is at-most-once for mutating tools, deliberately: running a payment or an email twice is
usually worse than asking. A tool is non-idempotent by default when it carries the `mutates`
label. A crash at any other point (after the model replied but before the policy decision, for
example) is recovered without calling the model again, because the next step is derived from the
journal.

Model failures are different: an `llm_error` event is recorded and the run stays resumable, since
nothing was half-written.

## 4. Leases

To advance a run a worker must hold its lease (a row with an owner and an expiry, taken with
`BEGIN IMMEDIATE`). A live worker blocks others; a crashed worker's lease expires. The lease is
renewed at every step.

## 5. Policy and the trifecta guard

Rules are evaluated first-match-wins, then a default. Tool-name globs are case-sensitive and
exact. Argument conditions fail closed: `unless_args` applies the rule when the argument is
missing; `when_args` does not apply it. The guard is evaluated last and only ever tightens:

```text
decision = first matching rule, else default
if the tool can_exfiltrate and (capabilities exercised in this run + the tool's own)
   include reads_private, ingests_untrusted and can_exfiltrate:
    decision = max(decision, trifecta setting)        # allow < ask < deny
```

Capabilities count as exercised when a tool *starts*, not when it is requested, so a denied read
never taints a run. Rationale and the attack it targets: [THREAT_MODEL.md](THREAT_MODEL.md).

## 6. Replay

`replay` creates a scratch journal and runs the **current** kernel against the recorded model
responses and tool results. It then compares a behaviour signature (request digests, policy
decisions, approvals, results, final status) with the recording and reports the first
divergence. A changed system prompt, tool schema, message construction or policy shows up as a
specific message, with no model, no network and no side effects. It tests the agent's logic, not
the model's randomness.

## 7. Trade-offs, stated plainly

| Decision | Gain | Cost |
|---|---|---|
| Event-sourced state | recovery, audit, replay from one mechanism | every step re-reads the journal; fine at this scale, not for huge runs |
| Sequential tool calls | determinism; the guard sees earlier calls | slower than parallel execution |
| At-most-once for mutating tools | no double side effects | the model must handle "outcome unknown" |
| One approval at a time | later calls are decided with correct state | more round trips for multi-call turns |
| Capability labels | policy reasons about facts, not model claims | a wrong label defeats the guard |
| SQLite | zero-ops, transactional, cross-process | single node |
| Stdlib plus pydantic | small, auditable | no async, no provider SDK features |

## 8. Deliberately not here

Streaming, context compaction, parallel tool execution, sandboxing of tool code, network egress
control, authentication and multi-tenancy. They are orthogonal to the guarantees above and are
better added as layers than baked into the core.
