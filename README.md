# IncidentZero

**Assignment 1 - Agentic Artificial Intelligence (Fall 2026)**
**Domain:** bounded autonomous SRE / production-incident response
**Mode:** individual assignment, plain Python + Groq SDK, no agent framework
**Submitted by:** Ayesha Tahir Awan, 25I-7631

A bounded autonomous incident commander for a local simulated production platform.
The model proposes; Python authorises. `AgentController` decides whether a proposed
call is well formed, permitted, affordable, repetitive, approved, stale, retryable
or terminal, so a model that is wrong or adversarial cannot turn a proposal into an
unsafe action or a false incident closure.

The simulator, tool schemas and configuration files under `configs/`,
`incidentzero/environment/` and `incidentzero/tools/` are the protected starter
boundary and are unmodified; `python scripts/check_protected_integrity.py` verifies
this. The agent runtime under `incidentzero/agent/` is the submitted work.

See `REPORT.md` for the engineering report and evaluation, and `AI_USAGE.md` for the
AI assistance declaration.

## Setup and run commands


Requires **Python 3.11+**.

```bash
# 1. environment
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. credentials  (the .env file is NOT included in the submission)
cp .env.example .env               # Windows: copy .env.example .env
#    then set GROQ_API_KEY=<your own key> inside .env
```

`.env.example` already carries `GROQ_MODEL=openai/gpt-oss-20b` and
`STUDENT_ID=25I-7631`, so only the key needs filling in. With `STUDENT_ID` set,
`--student-id` can be omitted from the run commands below.

### Tests, offline, no Groq quota consumed

```bash
pytest -q tests/public -m infrastructure   # 4 starter infrastructure tests
pytest -q tests/public                     # 8 public tests, all passing
pytest -q tests/student                    # 91 student tests, all passing
pytest -q tests/                           # 99 total
```

### Compliance checks

```bash
python scripts/check_banned_imports.py     # no agent-framework imports
python scripts/check_protected_integrity.py # protected files unmodified
python scripts/count_todos.py              # 0 TODO(A1) markers remaining
```

### Live runs against Groq

```bash
python -m incidentzero.cli run --student-id 25I-7631 --scenario public-a
python -m incidentzero.cli run --student-id 25I-7631 --scenario public-b   # type APPROVE at the prompt
python -m incidentzero.cli run --student-id 25I-7631 --scenario public-c
```

Each run prints a live event feed and writes a JSONL trace to
`traces/25I-7631_<scenario>.jsonl`.

Useful flags:

| Flag | Effect |
|---|---|
| `--approval console` | default, prompts a human for high/critical actions |
| `--approval approve` | auto-approve, for unattended runs |
| `--approval deny` | auto-deny, to demonstrate denial handling and re-planning |
| `--quiet` | suppress the live feed, print only the final outcome |
| `--model <name>` | override the model (default `openai/gpt-oss-20b`) |

### Included traces

| File | Outcome |
|---|---|
| `traces/25I-7631_public-a.jsonl` | resolved, cache corruption, `clear_cache` |
| `traces/25I-7631_public-b.jsonl` | resolved, bad deploy, approved `rollback_deployment` |
| `traces/25I-7631_public-c.jsonl` | resolved, capacity spike, `scale_service` |
| `traces/25I-7631_public-b_approval-denied.jsonl` | escalated, rollback denied, alternatives failed verification |

See `REPORT.md` for the evaluation table and failure analysis, and `AI_USAGE.md` for
the AI assistance declaration.

---

## Stable contract

Read `docs/CONTRACTS.md` before editing. Hidden grading assumes those public interfaces still exist. You may refactor internally, but do not delete or rename required public classes/functions.

## Protected areas

Do not modify the simulator to make scenarios easier. The grader uses clean copies and additional hidden scenarios. In particular, do not depend on private fields or anything named `_oracle`, `_root_cause`, or `_scenario_spec`.

The point is to build a robust **controller**, not to reverse-engineer the answer from the simulator source.
