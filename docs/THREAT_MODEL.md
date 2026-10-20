# Threat model

## Goal

Prevent **prompt-injection-driven data exfiltration and unintended actions** from succeeding,
even when the model is fully controlled by the attacker. The kernel does not try to make the
model honest or correct; it limits what a dishonest or mistaken model can make the system do.

## Assets

- Private data returned by tools (notes, files, records).
- Side effects in the outside world (messages, payments, writes).
- The integrity of the audit log.

## Adversary

- Controls the content returned by any tool labelled `ingests_untrusted` (email bodies, web
  pages, uploaded files).
- May fully control the model's output. This is the assumed worst case: the red-team suite
  scripts a model that obediently carries out the attacker's plan.
- Cannot change the kernel code, the policy file, the tool implementations or their labels, and
  cannot write to the journal database directly (if they can, the hash chain makes edits
  detectable).

## Trust assumptions

1. **Capability labels are correct.** They are the trust root: the policy reasons about labels,
   not about model claims.
2. The policy file and tool code are controlled by the operator.
3. The person approving actions reads what they approve.
4. The journal file is protected by ordinary file permissions.

## Controls

| Threat | Control | Evidence |
|---|---|---|
| Injected content makes the model read private data and send it out | lethal-trifecta guard: private + untrusted + exfiltrate cannot meet in one run | red-team scenarios 1 and 2, mutation tests |
| The same attack with legs in a different order | the guard tracks exercised capabilities, not order | scenario 2 |
| All three legs batched into one model turn | sequential execution; earlier calls are visible to later checks | scenario 3 |
| Tool-name tricks (`SEND_EMAIL`, trailing space, look-alikes) | exact, case-sensitive match; unknown tools are denied and never reach a tool | scenario 7, policy tests |
| Sending to an attacker-chosen destination | argument rules that fail closed | scenario 5, policy tests |
| Reading outside a tool's folder | path containment inside the tool | scenario 6 (tool level) |
| Runaway loops and cost | step limit and token budget | scenario 8, runtime tests |
| Actions nobody reviewed | `ask` suspends the run; approval is bound to one call | runtime tests |
| A malicious payload hidden from the approver | the approval prompt prints the complete input | CLI test |
| Double side effects after a crash | non-idempotent tools are never re-executed | runtime tests |
| Two workers acting on one run | leases | journal and runtime tests |
| Editing the record of what happened | hash-chained journal | journal tests |
| Tool output that instructs the model | results wrapped as data, untrusted sources marked | defence in depth; **not** a guarantee |

## Residual risks

- **A mislabelled tool defeats the guard.** Any tool that can make information visible outside the
  trust boundary must carry `can_exfiltrate`: email, HTTP requests, publishing, writing to shared
  or public locations, even writing to a file an attacker can read.
- **The guard needs all three legs.** An injection that causes a harmful action *without* private
  data or an outbound channel (deleting a record, say) is not covered by it; use `deny` and
  `ask` rules and approvals for such tools.
- **Conservative by design.** After a run has handled private data and untrusted content, it may
  not send anything, even for unrelated reasons. Start a new run for unrelated work.
- **Approval fatigue and social engineering.** A human who approves everything is not a control.
  Keep `ask` rare and meaningful.
- **The model can still be wrong or leak data in its answer to the user.** That is not exfiltration
  to a third party, but the user may forward it.
- **Tail truncation of the journal** is not detectable by the chain alone.
- **Scripted attacker only.** The suite shows the controls hold for the modelled attacks; it is not
  a proof, and it has not been run against live models.

## Out of scope

Sandboxing of tool code (tools run in-process), network egress control, authentication,
multi-tenancy, secret management, and supply-chain attacks on dependencies.
