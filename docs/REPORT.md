# Engineering Report: IncidentZero

Name: Ayesha Tahir Awan
Roll number: 25I-7631
Model: `openai/gpt-oss-20b` via Groq
Limits: as shipped in `configs/limits.json`, being 14 LLM calls, 28 tool calls and 120 seconds

## 1. Architecture

The runtime follows one rule. The model proposes and Python authorises.
`AgentController` decides whether a call is well formed, permitted, affordable,
repetitive, approved, stale, retryable or terminal, so a wrong or adversarial model
cannot turn a proposal into an unsafe action or a false closure.

`AgentState` holds the run's memory separately from the message list: the active
plan and revision, evidence ids, the latest `world_version`, the evidence id and
result of the most recent `verify_recovery`, the previous release observed per
service, the call budgets, approval and revision history, and a terminal status.

Three derived facts do real work. Any successful mutating tool clears the
recovery-verified flag, so an action after a passing check forces a new verification
before closure. The previous-release map is filled only from real `get_deployments`
results, making a rollback target checkable against evidence. The diagnostics map
records which of metrics, logs, deployments and dependencies has been read per
service, so the controller can answer what the agent knew before an action runs.

`_execute_tool_call` is the only boundary to the simulator, and its gate order is
deliberate: schema validation, loop guard, tool budget, state preconditions, risk and
human approval, then execution. Validation is free and catches invented tools first.
Approval costs human time, so it runs last, immediately before the only line allowed
to call `ToolRegistry.execute`.

Anything Python refuses is returned in the same shape as a real tool result, so the
model receives refusals through the same channel as observations and one classifier
handles both.

Retries, budgets and thresholds come from `configs/limits.json`. Risk comes from
`configs/risk_policy.json`, read on every call. No threshold is a literal in the
runtime, so lowering a budget or raising a tool's risk changes behaviour with no code
change. `TraceRecorder` writes JSONL and can echo to the console. The API key is
never logged.

## 2. Planning and re-planning strategy

Before the first model turn the controller runs a deterministic platform survey,
reading health for every service in `configs/services.json`, critical path first, then
metrics for any component reporting unhealthy.

This is evidence gathering, not diagnosis. It reports which component is degraded
and nothing about why or which remediation applies.

It exists because the budgets differ wildly in cost. A tool call takes milliseconds
and there are 28. A model round trip takes 15 to 25 seconds against a 120 second
limit, barely six usable turns. One early run checked seven healthy services and
never reached the degraded one.

`Planner.create` then produces an explicit plan with a hypothesis, at least two
subgoals with success signals, and a verification step, which is appended if the
model omits one. A deterministic fallback plan means a bad plan never stops the run.

`ReplanPolicy.classify` sorts every tool result into one of five response classes.
Those are `continue`, `retry`, `re_observe`, `reject` and `replan`.
`should_replan()` remains the boolean that the grading contract requires.

Re-planning is triggered by:

- approval denial
- a blocked loop
- `verify_recovery` returning `criteria_met=false`
- a non-retryable action failure
- a transient failure of a mutating action
- a second remediation that ran but reported the cause untouched
- any unrecognised status

That second-to-last trigger is R4's "new evidence contradicts the active
hypothesis". The simulator states it outright in the effect string of an action
applied to the wrong cause, so it is observable rather than inferred. The first one is
surfaced to the model rather than spent on a revision, since one wrong action is cheap
to correct. A second means the hypothesis is wrong, so the plan is revised.

A revision always increments the counter and is generated from the trigger plus a
state summary. A failed revision still advances it, so repeated failure is visible in
the trace and leads to escalation instead of a silent loop.

## 3. Failure handling

The controller does not branch on the simulator's `retryable` flag. That flag is a
truthful statement about the error. It says nothing about whether the action is
still justified.

`stale_precondition` is the clearest case. It arrives marked `retryable=true`, but
resending with a fresher version number is exactly wrong. On staleness the controller
re-observes the affected service itself, then tells the model not to resend without
re-justifying the action.

The opposite case also exists. `"Recovery criteria are not met"` arrives with
`retryable=false`, yet it is the most recoverable situation in the simulator.

Unknown statuses fall through to re-plan, so an unfamiliar result can never cause a
blind retry or an unchecked execution.

**Malformed model output.** Rejected by JSON schema validation before execution and
returned as a corrective observation naming the problem.

**Transient tool failures.** On read-only observations these are absorbed inside the
execution boundary, so the model receives only the successful result. Surfacing the
blip previously cost a decision turn re-requesting data the controller already had.
Mutating actions are excluded, because after one fails the world may have moved.

**Transient model failures.** Bounded retry, three attempts, capped exponential
backoff, injected sleeper. Each attempt consumes LLM budget because each is a real
request.

Two provider behaviours needed specific handling.

Groq returns HTTP 400 with code `tool_use_failed` when the model emits a malformed
tool call. The request was valid and only the generation failed, so this is
classified as transient. A run should not abort on one unlucky decode.

Retrying at the original near-deterministic temperature reproduced the identical
malformed output three times. Sampling now widens after each consecutive generation
failure and resets on success, so a retry is a genuinely different attempt.

A permanent error such as a 401 or a bad configuration aborts safely instead.

**Impossible incidents and exhausted budget.** Both route through `_wrap_up`, which
spends reserved calls on a final `verify_recovery` and closes only if the simulator
confirms the criteria, otherwise escalating with the evidence gathered.
`escalate_incident` has no version precondition, so it is always a safe exit.

Human approval wait is measured and excluded from the runtime limit, which bounds
the agent's work and not how fast an approver types.

## 4. Safety and stopping

Approval is required for whatever the risk policy marks high or critical, currently
`rollback_deployment`, `shift_traffic` and `failover_database`. An unknown tool
defaults to critical, so a new tool is gated rather than waved through.

A denial is recorded, returned as an observation, and never executed. The tests
assert this by checking `world_version` did not move, which a log line alone would
not establish.

A high or critical action is refused before the approval gate unless diagnostic
evidence has been observed for the service it targets. Health does not count, since
the survey reads health everywhere and counting it would make the check vacuous. The
rule follows the risk policy rather than a tool list, so promoting a tool to high
makes it demand evidence with no code change. It is scoped to mutating tools, so a
missing policy cannot deadlock the agent into needing evidence before it may
observe.

Proof of recovery is narrow and enforced in Python as well as by the simulator.
`close_incident` is refused locally unless a current `verify_recovery` reported
`criteria_met=true` and the call cites that exact evidence id. Any mutating action
since that verification invalidates it. Model prose claiming success is treated as a
stall. The controller nudges once and then escalates.

Across every test scenario the run lands on one of the five contract statuses,
including prose-only models, invented tools, observation loops and exhausted
budgets.

## 5. Evaluation

Elapsed time excludes human approval wait.

| Metric | public-a | public-b | public-c | public-b, approval denied |
|---|---|---|---|---|
| Root cause | cache corruption | bad deploy | capacity spike | bad deploy |
| Terminal outcome | resolved | resolved | resolved | escalated |
| LLM requests | 7 / 14 | 7 / 14 | 7 / 14 | 9 / 14 |
| Tool calls | 18 / 28 | 18 / 28 | 20 / 28 | 20 / 28 |
| Model decision turns | 6 | 5 | 6 | 7 |
| Plan revisions | 0 | 0 | 0 | 1 |
| High risk proposed / approved / executed | 0 / 0 / 0 | 1 / 1 / 1 | 0 / 0 / 0 | 1 / 0 / 0 |
| Recovery proof | EV-0017, close ok | EV-0017, close ok | EV-0019, close ok | EV-0020, `criteria_met=false` |
| Redundant calls | 0 | 0 | 0 | 0 |
| Elapsed, agent only | 91 s | 71 s | 109 s | 126 s |

The ticket was misleading in all three public scenarios. public-a and public-b both
name `order-service` as the suspect and carry an operator note blaming
`payment-service`. The real components were `redis-cache` and `checkout-service`.
The agent followed evidence in every case.

In public-a the first remediation was an ineffective restart. The contradiction check
caught it from the effect string and corrected the model, which then chose
`clear_cache`, with no plan revision needed. The denied run passed the runtime target
at 126 seconds and the wrap-up path handled it: a final verification reported
`criteria_met=false`, so it escalated with evidence rather than closing.

## 6. Three failure traces

### A wrong first action, then recovery. public-a.

The plan correctly identified the degraded `redis-cache`, but the first remediation
was `restart_service`, which returned `ok` with the effect `"service restarted; no
confirmed root-cause remediation"`. The controller read that effect, told the model
the cause was untouched, and the model followed with `clear_cache`. Verification then
passed and the incident closed.

### Approval denied, alternatives exhausted, honest escalation. public-b with denial.

The model proposed `rollback_deployment` with deployment evidence behind it. The
gateway denied it. The action never executed.

The denial drove a plan revision, which chose the lower-risk `restart_service`. It
returned `ok`, but the effect string reported no root-cause remediation, because the
true fix was the rollback the human had refused.

At the deadline the controller ran a final verification, saw `criteria_met=false`, and
escalated with evidence instead of claiming success.

This is the correct outcome. The agent did not bypass the human and did not
manufacture a resolution.

### A hallucinated rollback target. public-b, an earlier run.

The model proposed rolling back `checkout-service` to `2.3.9` without ever calling
`get_deployments`. No such version existed. The simulator rejected it, but only after
a human had been asked to approve an invented action, and recovering cost a plan
revision, a second approval round and several calls.

The fix was a Python precondition, later generalised to every high risk action. The
re-run needed one approval instead of two, and agent time fell from over a minute to
16 seconds.

## 7. Limitations

**The survey trades model autonomy for wall-clock headroom.**

Handing the model the set of degraded components removes a genuine part of the
investigation. It is evidence and not diagnosis, and a human responder would read a dashboard
rather than poll ten services. But with many degraded components it would be noise.

**Wall-clock is the binding constraint, not the call budget.**

Runs finish using 7 to 9 of the 14 LLM calls, but 71 to 126 seconds against a 120
second target. Performance therefore depends on provider latency we do not control.
The denied run escalated on the clock with five LLM calls still unused.

**Re-planning is coarse.**

A revision regenerates the whole plan rather than editing the failing step. Two
revisions cost 20 percent of the LLM budget, and in the denied run the second
produced a hypothesis barely different from the first. Step-level revision would be
cheaper and better grounded.

**The loop guard is purely syntactic.**

It fingerprints the tool name plus sorted arguments, so it catches exact repeats but
not semantically equivalent ones. Scaling to 3 then 4 then 5 replicas is not
detected. The guard also never resets, so a re-justified repeat after the world
changes is blocked rather than reconsidered.
