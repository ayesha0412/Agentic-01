from __future__ import annotations

import json
import time
from typing import Any

from incidentzero.approval.gateway import ApprovalGateway
from incidentzero.config import Limits, load_services
from incidentzero.domain.models import AgentOutcome, ModelReply, ToolCall
from incidentzero.model.base import ModelClient
from incidentzero.model.errors import PermanentModelError, TransientModelError
from incidentzero.telemetry.budget import BudgetExceeded, BudgetManager
from incidentzero.telemetry.trace import TraceRecorder
from incidentzero.tools.registry import ToolRegistry

from .planner import Planner
from .policies import (
    CONTRADICTED,
    REJECT,
    REPLAN,
    RETRY,
    RE_OBSERVE,
    LoopGuard,
    ReplanPolicy,
    RiskPolicy,
)
from .prompts import SYSTEM_PROMPT
from .recovery import RetryPolicy
from .state import MUTATING_TOOLS, AgentState


class AgentController:
    """Bounded autonomous incident controller.

    The model proposes; this class authorises. Every decision that has consequences --
    whether a call is well formed, allowed, affordable, repetitive, approved, stale,
    retryable or terminal -- is made here in Python, so that an incorrect, unlucky or
    adversarial model cannot turn a proposal into an unsafe action or a false closure.
    """

    #: Tool calls held back so the agent can always verify-and-close or escalate.
    RESERVED_TOOL_CALLS = 2
    #: How many prose-only model replies to tolerate before terminating safely.
    MAX_NO_TOOL_CALL_REPLIES = 2

    def __init__(
        self,
        model: ModelClient,
        tools: ToolRegistry,
        approval: ApprovalGateway,
        budget: BudgetManager,
        trace: TraceRecorder,
        limits: Limits | None = None,
        risk_policy_path: str = "configs/risk_policy.json",
    ) -> None:
        self.model = model
        self.tools = tools
        self.approval = approval
        self.budget = budget
        self.trace = trace
        self.limits = limits or Limits.load()
        self.state = AgentState()
        self.planner = Planner(model)
        self.risk = RiskPolicy(risk_policy_path)
        self.replan_policy = ReplanPolicy()
        # Thresholds come from configuration, never from literals in the runtime.
        self.loop_guard = LoopGuard(self.limits.max_same_action_repeats)
        self.retry_policy = RetryPolicy(max_attempts=self.limits.max_consecutive_model_retries)
        self._tool_retry_counts: dict[str, int] = {}
        self._started_at = time.monotonic()
        # Wall-clock spent blocked on a human approver, excluded from the runtime limit.
        self._approval_wait_s = 0.0

    def _model_decide(self) -> ModelReply:
        """One model decision, with bounded retry and correct budget accounting.

        Each attempt is a real provider request, so each attempt consumes LLM budget
        (assignment Q23). Budget is consumed inside the retried callable rather than
        around it, which means a run cannot exceed its request budget by retrying.
        """
        def attempt() -> ModelReply:
            self.budget.consume_llm()
            return self.model.decide(self.state.messages, self.tools.groq_tools)

        reply = self.retry_policy.call_model(attempt)
        self.state.budget_snapshot = {
            "llm_used": self.budget.llm_calls,
            "llm_max": self.budget.max_llm_calls,
            "tool_used": self.budget.tool_calls,
            "tool_max": self.budget.max_tool_calls,
        }
        self.trace.record(
            "model_reply",
            {
                "attempts": self.retry_policy.last_attempts,
                "finish_reason": reply.finish_reason,
                "usage": reply.usage,
                "content": (reply.content or "")[:400],
                "tool_calls": [{"name": c.name, "arguments": c.arguments} for c in reply.tool_calls],
                "remaining_llm": self.budget.remaining_llm,
                "remaining_tools": self.budget.remaining_tools,
                "elapsed_s": round(self._elapsed(), 1),
            },
        )
        return reply

    def _refusal(self, call: ToolCall, status: str, message: str) -> dict[str, Any]:
        """Build a tool-result-shaped observation for a request Python refused.

        Refusals are shaped exactly like simulator results so that the model sees them
        through the same channel as real observations, and so ReplanPolicy can classify
        them with the same code path. The world version is read from the environment
        rather than invented, and no evidence id is issued because nothing was observed.
        """
        return {
            "status": status,
            "tool": call.name,
            "world_version": self.tools.environment.world_version,
            "evidence_id": None,
            "data": None,
            "retryable": False,
            "message": message,
        }

    def _execute_tool_call(self, call: ToolCall) -> dict[str, Any]:
        """Validate, gate, approve if required, execute, trace, and return one observation.

        Order matters. Schema validation is free and catches invented tools before
        anything else happens; the loop guard and budget check are cheap refusals; human
        approval is the most expensive gate and therefore runs last, immediately before
        the single line that is allowed to touch the simulator.
        """
        ok, error = self.tools.validate(call.name, call.arguments)
        if not ok:
            self.trace.record(
                "validation_failure",
                {"tool": call.name, "arguments": call.arguments, "error": error},
            )
            return self._refusal(
                call,
                "validation_error",
                f"{error}. Correct the arguments against the tool schema and try again.",
            )

        if self.loop_guard.record(call.name, call.arguments):
            self.trace.record(
                "loop_blocked",
                {"tool": call.name, "arguments": call.arguments},
            )
            return self._refusal(
                call,
                "loop_blocked",
                "This exact action has already been attempted and did not change the "
                "outcome. Choose a different action or escalate.",
            )

        if self.budget.remaining_tools <= 0:
            self.trace.record("budget_blocked", {"tool": call.name})
            return self._refusal(
                call, "budget_blocked", "Tool-call budget exhausted; terminate safely."
            )

        # R8 lists state preconditions as part of the pre-execution check, so they are
        # enforced here at the boundary rather than by the caller. Running before the
        # approval gate means an action the controller can already tell is unjustified
        # never costs an approver's attention.
        precondition = self._precondition_failure(call)
        if precondition:
            self.trace.record(
                "precondition_failure", {"tool": call.name, "message": precondition}
            )
            return self._refusal(call, "precondition_failed", precondition)

        # Risk is read from configuration on every call, so changing risk_policy.json
        # changes behaviour with no code change.
        if self.risk.requires_human_approval(call.name):
            justification = str(call.arguments.get("reason") or "").strip()
            self.trace.record(
                "approval_requested",
                {
                    "tool": call.name,
                    "risk": self.risk.risk(call.name).value,
                    "arguments": call.arguments,
                    "justification": justification,
                },
            )
            waiting_from = time.monotonic()
            granted = bool(self.approval.approve(call.name, call.arguments, justification))
            waited = time.monotonic() - waiting_from
            self._approval_wait_s += waited
            self.state.record_approval(call.name, granted, justification)
            self.trace.record(
                "approval_result",
                {"tool": call.name, "granted": granted, "waited_s": round(waited, 1)},
            )
            if not granted:
                return self._refusal(
                    call,
                    "approval_denied",
                    "A human approver denied this action. It was not executed. Revise "
                    "the plan with a lower-risk alternative or escalate with evidence.",
                )

        self.budget.consume_tool()
        result = self.tools.execute(call.name, call.arguments)
        self.trace.record(
            "tool_result",
            {"call": {"name": call.name, "arguments": call.arguments}, "result": result},
        )

        # A transient failure of a read-only observation is absorbed here rather than
        # handed back to the model. Re-reading changes nothing in the world, so the
        # model only needs the evidence -- surfacing the blip makes it spend a decision
        # turn re-requesting data the controller has already fetched. Mutating actions
        # are deliberately excluded: after one fails the world may have moved, so the
        # decision has to be re-made rather than repeated.
        attempt = 1
        while (
            result.get("status") == "transient_error"
            and result.get("retryable") is True
            and call.name not in MUTATING_TOOLS
            and attempt < self.limits.max_consecutive_model_retries
            and self.budget.remaining_tools > self.RESERVED_TOOL_CALLS
        ):
            attempt += 1
            self.trace.record("tool_retry", {"tool": call.name, "attempt": attempt})
            self.budget.consume_tool()
            result = self.tools.execute(call.name, call.arguments)
            self.trace.record(
                "tool_result",
                {
                    "call": {"name": call.name, "arguments": call.arguments},
                    "result": result,
                    "retry_attempt": attempt,
                },
            )

        if result.get("status") == "transient_error":
            self.trace.record(
                "tool_retry_exhausted", {"tool": call.name, "attempts": attempt}
            )
            result = dict(result)
            result["message"] = (
                f"{result.get('message', '')} This observation failed {attempt} time(s). "
                "Obtain equivalent evidence from a different observation tool."
            )
        return result

    def _append_assistant(self, reply: ModelReply) -> None:
        msg: dict[str, Any] = {"role": "assistant", "content": reply.content}
        if reply.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
                for call in reply.tool_calls
            ]
        self.state.messages.append(msg)

    def _append_tool_result(self, call: ToolCall, result: dict[str, Any]) -> None:
        self.state.messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result, ensure_ascii=False),
        })

    # -- controller-initiated observation ------------------------------------------
    def _controller_observe(self, name: str, arguments: dict[str, Any], why: str) -> dict[str, Any]:
        """Run an observation the controller decided on, not the model.

        These deliberately skip the loop guard. The loop guard exists to stop the
        *model* from repeating itself; a controller-initiated re-observation is
        already bounded by the budget and by the number of rejections that can occur.
        """
        if self.budget.remaining_tools <= 0:
            return self._refusal(ToolCall("ctl", name, arguments), "budget_blocked", "No tool budget left.")
        self.budget.consume_tool()
        result = self.tools.execute(name, arguments)
        self.trace.record(
            "controller_observation",
            {"tool": name, "arguments": arguments, "why": why, "result": result},
        )
        self.state.observe_result(result)
        self.state.messages.append(
            {
                "role": "system",
                "content": (
                    f"Controller re-observation ({why}): {json.dumps(result, ensure_ascii=False)}"
                ),
            }
        )
        return result

    def _bootstrap_survey(self) -> dict[str, Any]:
        """Read health across the platform before the model takes its first turn.

        This is evidence gathering, not diagnosis. Health says which component is
        degraded; it says nothing about why, or which of the six remediations is
        correct, which remains the model's job. It exists because the two budgets
        are wildly asymmetric: a tool call costs milliseconds, while a model round
        trip costs tens of seconds against a hard wall-clock limit. Letting the model
        discover an unhealthy service one round trip at a time spent entire runs
        confirming that healthy services were healthy. A human responder would open a
        dashboard rather than query ten services one at a time; this is that
        dashboard. Metrics are pulled only for components that are actually degraded,
        so a healthy platform costs almost nothing.
        """
        def read(tool: str, service: str) -> dict[str, Any]:
            """One survey read, absorbing the simulator's injected transient failures."""
            result: dict[str, Any] = {}
            for _ in range(self.limits.max_consecutive_model_retries):
                if self.budget.remaining_tools <= self.RESERVED_TOOL_CALLS + 2:
                    return result
                self.budget.consume_tool()
                result = self.tools.execute(tool, {"service": service})
                self.state.observe_result(result)
                if result.get("status") != "transient_error":
                    return result
                self.trace.record("tool_retry", {"tool": tool, "service": service, "phase": "survey"})
            return result

        health: dict[str, Any] = {}
        degraded: list[str] = []
        for name in load_services():
            result = read("get_service_health", name)
            if result.get("status") != "ok":
                continue
            data = result.get("data") or {}
            health[name] = {"healthy": data.get("healthy"), "replicas": data.get("replicas"), "version": data.get("version")}
            if data.get("healthy") is False:
                degraded.append(name)

        metrics: dict[str, Any] = {}
        for name in degraded:
            result = read("get_metrics", name)
            if result.get("status") == "ok":
                metrics[name] = result.get("data")

        survey = {"health": health, "degraded": degraded, "metrics": metrics}
        self.trace.record("bootstrap_survey", survey)
        return survey

    def _revise_plan(self, trigger: dict[str, Any]) -> None:
        """Advance the plan to its next revision and tell the model what changed."""
        if self.state.plan is None or self.budget.remaining_llm <= 1:
            return
        self.budget.consume_llm()
        self.state.plan = self.planner.revise(
            self.state.plan, trigger, json.dumps(self.state.summary(), default=str)
        )
        self.state.record_plan_revision(str(trigger.get("reason", trigger)), self.state.plan.hypothesis)
        self.trace.record(
            "plan_revised",
            {
                "revision": self.state.plan.revision,
                "trigger": trigger,
                "hypothesis": self.state.plan.hypothesis,
                "steps": [s.objective for s in self.state.plan.steps],
            },
        )
        self.state.messages.append(
            {
                "role": "system",
                "content": (
                    f"Plan revised to revision {self.state.plan.revision}. "
                    f"Hypothesis: {self.state.plan.hypothesis}. "
                    f"Remaining steps: {[s.objective for s in self.state.plan.steps]}"
                ),
            }
        )

    def _safe_escalate(self, reason: str) -> dict[str, Any]:
        """Deterministic terminal escalation that does not need the model.

        Used when the budget, the clock or repeated failure means the agent cannot
        safely continue. escalate_incident carries no world-version precondition, so
        it cannot be rejected as stale -- it is always available as a safe exit.
        """
        padded = reason if len(reason) >= 20 else f"{reason} (autonomous remediation halted)"
        result = self.tools.execute(
            "escalate_incident",
            {"reason": padded, "evidence_ids": self.state.evidence_ids[-12:]},
        )
        self.trace.record("safe_escalation", {"reason": padded, "result": result})
        self.state.observe_result(result)
        return result

    def _wrap_up(self, why: str) -> AgentOutcome:
        """Terminate safely when time or budget runs out, without wasting a near-win.

        Remediation may already have worked, so the last reserved calls are spent on
        the only evidence that can prove it: one verify_recovery, and a close if the
        criteria are met. Closure still requires genuine simulator evidence -- this
        path cannot manufacture a resolved status. Otherwise the run escalates with
        whatever evidence was gathered, which is a correct outcome, not a failure.
        """
        self.trace.record("wrap_up", {"why": why, "remaining_tools": self.budget.remaining_tools})
        if self.budget.remaining_tools >= 2:
            verify = self._controller_observe("verify_recovery", {}, why=f"final check: {why}")
            if verify.get("status") == "ok" and self.state.latest_verify_criteria_met:
                close = self.tools.execute(
                    "close_incident",
                    {
                        "summary": (
                            "Recovery criteria verified after bounded autonomous remediation; "
                            f"closing because {why}."
                        ),
                        "evidence_ids": [self.state.latest_verify_evidence_id],
                        "expected_world_version": self.state.latest_world_version,
                        "reason": "Objective recovery criteria were met on the final verification.",
                    },
                )
                self.trace.record("wrap_up_close", {"result": close})
                self.state.observe_result(close)
                if close.get("status") == "ok":
                    return self._outcome(
                        "resolved",
                        "Incident closed on the final verification "
                        f"(evidence {self.state.latest_verify_evidence_id}).",
                    )
        self._safe_escalate(f"Autonomous remediation halted because {why}.")
        return self._outcome("escalated", f"Escalated because {why}.")

    def _outcome(self, status: str, summary: str) -> AgentOutcome:
        self.state.terminate(status, summary)
        self.trace.record(
            "terminal",
            {
                "status": status,
                "summary": summary,
                "llm_calls": self.budget.llm_calls,
                "tool_calls": self.budget.tool_calls,
                "plan_revisions": self.state.plan_revisions,
                "approvals": self.state.approvals,
                "evidence_ids": self.state.evidence_ids,
            },
        )
        return AgentOutcome(
            status=status,
            summary=summary,
            llm_calls=self.budget.llm_calls,
            tool_calls=self.budget.tool_calls,
            final_world_version=self.state.latest_world_version,
            evidence_ids=self.state.evidence_ids,
            trace_path=str(self.trace.path),
        )

    # -- preconditions the controller enforces before an action is even attempted ----
    def _precondition_failure(self, call: ToolCall) -> str | None:
        """Reject actions whose preconditions the controller can check locally.

        Closing without a current successful verification is the invariant that
        separates a proven recovery from a model claiming success, so it is checked
        here rather than relying on the simulator to refuse it.
        """
        if call.name == "close_incident" and not self.state.latest_verify_criteria_met:
            return (
                "close_incident requires a current successful verify_recovery. "
                "Call verify_recovery first and cite its evidence id."
            )
        if call.name == "close_incident":
            cited = call.arguments.get("evidence_ids") or []
            if self.state.latest_verify_evidence_id not in cited:
                return (
                    "close_incident must cite the evidence id of the most recent "
                    f"verify_recovery ({self.state.latest_verify_evidence_id})."
                )

        if call.name == "rollback_deployment":
            # A rollback target is a fact about the platform, not something the model
            # may recall. In a live run the model proposed rolling back to a version
            # it had never observed, and a human was asked to approve it. Checking the
            # target against observed deployment evidence keeps an unevidenced
            # high-risk action away from the approver entirely.
            service = call.arguments.get("service")
            target = call.arguments.get("target_version")
            observed = self.state.observed_previous_version.get(str(service))
            if observed is None:
                return (
                    f"No deployment evidence has been observed for {service}. Call "
                    "get_deployments first so the rollback target is grounded in evidence."
                )
            if target != observed:
                return (
                    f"target_version {target!r} is not the known previous release for "
                    f"{service}. get_deployments reported {observed!r}. Use the observed "
                    "version rather than a remembered one."
                )

        # R3: a high or critical action must be supported by relevant observed
        # evidence. This is the general form of the rollback rule above, driven by the
        # risk policy rather than a hard-coded tool list, so promoting a tool to high
        # in configs/risk_policy.json immediately makes it demand evidence too.
        # It is scoped to mutating tools so that a missing or empty policy, which makes
        # everything critical, cannot deadlock the agent by demanding evidence before
        # it is allowed to observe anything.
        if call.name in MUTATING_TOOLS and self.risk.requires_human_approval(call.name):
            service = str(call.arguments.get("service", ""))
            if not self.state.observed_diagnostics.get(service):
                return (
                    f"No diagnostic evidence has been observed for {service}, so a "
                    f"{self.risk.risk(call.name).value}-risk action is not justified. "
                    "Read its metrics, logs, deployments or dependencies first."
                )
        return None

    def _elapsed(self) -> float:
        """Seconds of agent work so far, excluding time spent waiting on a human.

        The runtime limit bounds what the agent does, not how quickly an approver
        answers. Counting the wait would let a slow human turn a correct, in-budget
        run into a timeout, and would make the limit depend on who is at the keyboard.
        """
        return time.monotonic() - self._started_at - self._approval_wait_s

    def _out_of_time(self) -> bool:
        return self._elapsed() > self.limits.max_runtime_seconds

    def run(self) -> AgentOutcome:
        self._started_at = time.monotonic()
        self._approval_wait_s = 0.0
        self.state.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Investigate the active production incident, mitigate it safely, verify recovery, then close it; otherwise escalate with evidence."},
        ]
        no_tool_call_replies = 0
        try:
            # Bootstrap with one real observation so the plan is based on environment evidence.
            self.budget.consume_tool()
            incident = self.tools.execute("get_incident", {})
            self.state.observe_result(incident)
            self.trace.record("bootstrap_incident", incident)
            self.state.messages.append({"role": "system", "content": f"Current incident evidence: {json.dumps(incident)}"})

            # Cheap deterministic evidence before the first expensive model turn.
            survey = self._bootstrap_survey()
            self.state.messages.append(
                {
                    "role": "system",
                    "content": (
                        "Platform survey already performed for you (do not re-request this "
                        f"data): {json.dumps(survey, ensure_ascii=False)}\n"
                        "Components listed under 'degraded' are the ones actually unhealthy; "
                        "every other component reported healthy. Start from this evidence, "
                        "determine the failure mode of the degraded components using logs, "
                        "deployments or dependencies, then act."
                    ),
                }
            )

            self.budget.consume_llm()
            self.state.plan = self.planner.create({"incident": incident, "survey": survey})
            self.trace.record(
                "plan_created",
                {
                    "revision": self.state.plan.revision,
                    "hypothesis": self.state.plan.hypothesis,
                    "steps": [
                        {"step_id": s.step_id, "objective": s.objective, "success_signal": s.success_signal}
                        for s in self.state.plan.steps
                    ],
                },
            )
            self.state.messages.append(
                {
                    "role": "system",
                    "content": (
                        f"Active plan (revision {self.state.plan.revision}). "
                        f"Hypothesis: {self.state.plan.hypothesis}. "
                        f"Steps: {[s.objective for s in self.state.plan.steps]}"
                    ),
                }
            )

            while not self.state.is_terminal:
                # --- budget and clock guards, before spending anything ---------------
                if self._out_of_time():
                    self.trace.record(
                        "deadline_reached",
                        {
                            "elapsed_s": round(self._elapsed(), 1),
                            "approval_wait_s": round(self._approval_wait_s, 1),
                        },
                    )
                    return self._wrap_up("the wall-clock runtime limit was reached")
                if self.budget.remaining_tools <= self.RESERVED_TOOL_CALLS or self.budget.remaining_llm <= 1:
                    self.trace.record(
                        "budget_warning",
                        {"remaining_llm": self.budget.remaining_llm, "remaining_tools": self.budget.remaining_tools},
                    )
                    return self._wrap_up("the remaining call budget is insufficient to continue safely")
                if self.budget.remaining_llm <= self.limits.warning_llm_calls_remaining:
                    self.trace.record("budget_warning", {"remaining_llm": self.budget.remaining_llm})

                # --- ask the model for one proposal ----------------------------------
                try:
                    reply = self._model_decide()
                except TransientModelError as exc:
                    self.trace.record("model_retries_exhausted", {"error": str(exc)})
                    self._safe_escalate("The model provider remained unavailable after bounded retries.")
                    return self._outcome("escalated", f"Escalated after transient model failures: {exc}")
                except PermanentModelError as exc:
                    self.trace.record("model_permanent_error", {"error": str(exc)})
                    self._safe_escalate("A permanent model/configuration error stopped autonomous remediation.")
                    return self._outcome("aborted", f"Aborted on permanent model error: {exc}")

                self._append_assistant(reply)

                # --- the model answered in prose instead of acting -------------------
                if not reply.tool_calls:
                    no_tool_call_replies += 1
                    self.trace.record(
                        "model_stopped_without_action",
                        {"content": (reply.content or "")[:400], "occurrence": no_tool_call_replies},
                    )
                    if no_tool_call_replies >= self.MAX_NO_TOOL_CALL_REPLIES:
                        self._safe_escalate(
                            "The model stopped proposing actions before recovery was verified."
                        )
                        return self._outcome(
                            "escalated",
                            "Escalated: model produced narrative instead of actions; text is not proof of recovery.",
                        )
                    # Natural-language claims are never operational success (Q25).
                    self.state.messages.append(
                        {
                            "role": "system",
                            "content": (
                                "Your message contained no tool call. Narrative text is not "
                                "operational evidence and cannot resolve the incident. Either "
                                "call a tool that advances the plan, run verify_recovery, or "
                                "call escalate_incident with the evidence collected so far."
                            ),
                        }
                    )
                    continue

                # Parallel tool calls are disabled in the Groq client, so one call per turn.
                call = reply.tool_calls[0]

                result = self._execute_tool_call(call)

                world_before = self.state.latest_world_version
                self.state.observe_result(result)
                if self.state.latest_world_version != world_before:
                    # A new world_version means earlier observations no longer
                    # describe the platform, so re-reading one is fresh work
                    # rather than a repeat.
                    self.loop_guard.note_world_change()
                self._append_tool_result(call, result)

                # --- terminal tool outcomes -----------------------------------------
                if call.name == "close_incident" and result.get("status") == "ok":
                    return self._outcome(
                        "resolved",
                        "Incident closed after verify_recovery reported criteria_met=true "
                        f"(evidence {self.state.latest_verify_evidence_id}).",
                    )
                if call.name == "escalate_incident" and result.get("status") == "ok":
                    return self._outcome("escalated", "Incident escalated with collected evidence.")

                # --- classify the result and respond in kind -------------------------
                decision = self.replan_policy.classify(result)
                self.trace.record(
                    "control_decision",
                    {"tool": call.name, "status": result.get("status"), "decision": decision},
                )

                if decision == RETRY:
                    self._handle_tool_retry(call, result)
                elif decision == RE_OBSERVE:
                    self._handle_stale_world(call, result)
                elif decision == REJECT:
                    # The corrective message is already in the tool result the model sees.
                    continue
                elif decision == CONTRADICTED:
                    self._handle_contradiction(call, result)
                elif decision == REPLAN:
                    self._revise_plan(
                        {
                            "reason": result.get("status"),
                            "tool": call.name,
                            "message": result.get("message"),
                            "data": result.get("data"),
                        }
                    )

            return self._outcome("failed", "Run ended without an explicit terminal action.")

        except BudgetExceeded as exc:
            self.trace.record("budget_exhausted", {"error": str(exc)})
            return self._outcome("budget_exhausted", str(exc))
        except Exception as exc:  # never leave the run without a terminal status
            self.trace.record("unexpected_error", {"error": f"{type(exc).__name__}: {exc}"})
            return self._outcome("failed", f"Unexpected controller error: {type(exc).__name__}: {exc}")

    # -- failure handlers ----------------------------------------------------------
    def _handle_tool_retry(self, call: ToolCall, result: dict[str, Any]) -> None:
        """Handle a transient failure that in-line retry could not absorb.

        Observation retries happen inside _execute_tool_call, so anything reaching
        here is either a mutating action -- where the world may have moved and the
        decision must be remade rather than repeated -- or an observation whose
        retries were exhausted, where the model needs to find another evidence path.
        """
        if call.name in MUTATING_TOOLS:
            self._revise_plan({"reason": "transient_action_failure", "tool": call.name})
            return
        self.state.messages.append(
            {
                "role": "system",
                "content": (
                    f"{call.name} kept failing transiently and was already retried by the "
                    "controller. Obtain equivalent evidence through a different "
                    "observation tool instead of repeating it."
                ),
            }
        )

    def _handle_contradiction(self, call: ToolCall, result: dict[str, Any]) -> None:
        """React to a remediation that ran but reported the cause is untouched.

        The first such result is surfaced to the model rather than spent on a full
        revision, because one ineffective action is cheap to correct and a revision
        costs an LLM call out of fourteen. A second one means the hypothesis itself is
        wrong rather than the chosen action, so the plan is revised.
        """
        effect = str((result.get("data") or {}).get("effect", ""))
        self.state.ineffective_actions.append(call.name)
        self.trace.record(
            "hypothesis_contradicted",
            {
                "tool": call.name,
                "effect": effect,
                "hypothesis": self.state.plan.hypothesis if self.state.plan else None,
                "ineffective_so_far": self.state.ineffective_actions,
            },
        )
        if len(self.state.ineffective_actions) >= 2:
            self._revise_plan(
                {
                    "reason": "hypothesis_contradicted",
                    "tool": call.name,
                    "effect": effect,
                    "ineffective_actions": self.state.ineffective_actions,
                }
            )
            return
        self.state.messages.append(
            {
                "role": "system",
                "content": (
                    f"{call.name} executed successfully but the simulator reported: "
                    f"\"{effect}\". That is evidence against the current hypothesis, not "
                    "progress. Do not repeat this action or try a larger version of it "
                    "without new evidence. Gather evidence that distinguishes the real "
                    "cause, or choose the remediation that matches the evidence you "
                    "already have."
                ),
            }
        )

    def _handle_stale_world(self, call: ToolCall, result: dict[str, Any]) -> None:
        """React to stale_precondition by re-observing, never by bumping the version.

        The rejected action is re-examined against fresh evidence: the controller
        refreshes the affected service and hands the decision back to the model with
        an explicit instruction that the action must be re-justified, not repeated.
        """
        service = call.arguments.get("service")
        self.trace.record(
            "stale_world",
            {
                "tool": call.name,
                "expected": result.get("expected"),
                "actual": result.get("actual"),
                "service": service,
            },
        )
        if isinstance(service, str):
            self._controller_observe("get_service_health", {"service": service}, why="stale world recheck")
            if self.budget.remaining_tools > self.RESERVED_TOOL_CALLS:
                self._controller_observe("get_metrics", {"service": service}, why="stale world recheck")
        self.state.messages.append(
            {
                "role": "system",
                "content": (
                    f"Your {call.name} was rejected: the world moved from version "
                    f"{result.get('expected')} to {result.get('actual')} after your "
                    "observation. Fresh state is above. Do not resend the same action with "
                    "a new version number unless the fresh evidence still justifies it; "
                    f"the newest observed world version is {self.state.latest_world_version}."
                ),
            }
        )
