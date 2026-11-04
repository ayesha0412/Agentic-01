# AI Assistance Declaration

Name: Ayesha Tahir Awan
Roll number: 25I-7631

## Tools used

- Claude Sonnet 5 model is used in the code generation. 
- Also chatgpt and co pilot has been used alongside.

## What I used them for

I used the assistant as a coding partner. I set the direction and reviewed every
piece before it went into the repository.

What I did:

- Set the build order. Policy classes first, then state, then the approval gate,
  then the controller loop, then tests.
- Approved or sent back each design before it was written.
- Required an explanation of every component before moving to the next one.
- Ran all the live Groq evaluations myself. Those runs produced the traces.
- Read the traces and worked out what was going wrong in each failed run.
- Wrote `LoopGuard.record` myself. That is the argument fingerprint and the
  repeat counting.
- Decided where the commit boundaries went.

What the assistant did:

- Wrote most of the implementation code from those decisions.
- Explained the assignment specification and the simulator contract.
- Drafted the tests.

The assistant wrote the bulk of `recovery.py`, `policies.py` apart from
`LoopGuard`, `state.py`, `controller.py`, `planner.py`, `config.py`, `prompts.py`,
the console echo in `trace.py`, and the tests in `tests/student/`.

Most of the design work happened because a live run failed. The bootstrap survey,
the in-line tool retry, the temperature escalation on retry, and the rollback
precondition all exist because I ran a scenario, it went wrong, and we read the
trace together. None of those were planned in advance.

## Two suggestions I rejected or changed

**1. Inconsistent field naming in `AgentState`.**

The proposed fields were `latest_verify_evidence_id` and
`last_verify_criteria_met`. Two names for the same idea in one dataclass. I sent it
back and asked for one convention. Both became `latest_`. That also matches the
`latest_world_version` field which was already there.

**2. Matching on the literal string `"transient_error"`.**

The classifier branched on exact status strings. I did not trust that. I asked how
we knew the simulator would always return that exact word.

We checked instead of assuming. Every status the engine and the registry can emit
was listed. There are only five. They are `ok`, `error`, `transient_error`,
`stale_precondition` and `validation_error`. So the match is safe.

The useful part came out of the check. The real protection is not the string
match. It is the catch-all `return REPLAN` at the bottom of `classify()`. An
unfamiliar status falls through to re-planning. Re-planning never retries blindly
and never executes anything. That line matters more than the ones above it.

## One AI-generated or AI-assisted bug I personally diagnosed

**Symptom.**

A `public-c` run hit the wall-clock limit and escalated. I read the trace. The
model had called `get_metrics` on `checkout-service` twice. Once at 52.6 seconds
and again at 68.6 seconds. A `tool_retry` event sat between them. Each model round
trip cost about 16 seconds out of a 120 second limit. So one duplicated
observation was expensive.

I asked why the model was retrying something the controller had already retried.

**Cause.**

The order in `run()` was wrong. The failed tool result went into the conversation
first, through `_append_tool_result`. Only then did `_handle_tool_retry` run. The
controller retried and got the data. It appended that data as a separate `system`
message.

So the model saw two things. Its own tool call had failed. A system message held
some data. It treated the failure as the answer to its request and the system
message as background noise. Then it called `get_metrics` again.

**Fix.**

The retry moved inside `_execute_tool_call`. It now happens before the result is
returned. The model only ever receives the successful observation. It never learns
the blip happened.

Mutating actions are excluded from this on purpose. If a restart or a rollback
fails then the world may have moved. That decision has to be made again. It must
not be repeated silently.

Both halves are pinned in `tests/student/test_recovery_and_stopping.py`. A
`FlakyRegistry` injects transient failures on demand. One test proves the retry
happens. Another test proves a failed action is never repeated.

## Code ownership statement

I can explain every submitted component, its failure behavior, and the trade-offs I
chose. I understand that the TA may ask me to modify the code during viva.

Signature / typed name: Ayesha Tahir Awan
