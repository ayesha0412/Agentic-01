from __future__ import annotations

from typing import Any

from incidentzero.domain.models import AgentPlan, PlanStep
from incidentzero.model.base import ModelClient


PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string"},
        "rationale_summary": {"type": "string"},
        "steps": {
            "type": "array",
            "minItems": 2,
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "step_id": {"type": "string"},
                    "objective": {"type": "string"},
                    "success_signal": {"type": "string"},
                },
                "required": ["step_id", "objective", "success_signal"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["hypothesis", "rationale_summary", "steps"],
    "additionalProperties": False,
}


class Planner:
    def __init__(self, model: ModelClient) -> None:
        self.model = model

    #: Used when the model cannot produce a usable plan. The agent must still start
    #: from an explicit, evidence-first plan rather than acting with no plan at all.
    FALLBACK_HYPOTHESIS = (
        "Root cause unknown. The incident ticket is treated as an unverified lead "
        "until observations identify the failing component."
    )
    FALLBACK_STEPS = [
        ("S1", "Observe health, metrics and logs across the checkout critical path.",
         "The failing component and its failure mode are identified from observations."),
        ("S2", "Correlate the failure mode with deployment, capacity, cache or database evidence.",
         "One hypothesis is supported by at least two independent observations."),
        ("S3", "Apply the least risky remediation supported by that evidence.",
         "The remediation returns ok."),
        ("S4", "Run verify_recovery and close only on criteria_met, otherwise re-plan or escalate.",
         "verify_recovery reports criteria_met=true, or the incident is escalated with evidence."),
    ]

    def _fallback_plan(self, reason: str) -> AgentPlan:
        return AgentPlan(
            hypothesis=self.FALLBACK_HYPOTHESIS,
            steps=[PlanStep(step_id=i, objective=o, success_signal=s) for i, o, s in self.FALLBACK_STEPS],
            rationale_summary=f"Deterministic fallback plan used because: {reason}",
        )

    def _parse(self, raw: dict[str, Any]) -> AgentPlan:
        """Validate a model-produced plan beyond what the JSON schema guarantees."""
        steps = [
            PlanStep(
                step_id=str(row["step_id"]),
                objective=str(row["objective"]),
                success_signal=str(row["success_signal"]),
            )
            for row in raw["steps"]
        ]
        if len(steps) < 2:
            raise ValueError("a plan needs at least two subgoals")
        # R2 requires a final verification step. Models routinely end a plan on the
        # remediation, so rather than rejecting an otherwise usable plan the controller
        # appends the step the requirement demands.
        if "verify" not in steps[-1].objective.lower() + steps[-1].success_signal.lower():
            steps.append(
                PlanStep(
                    step_id=f"S{len(steps) + 1}",
                    objective="Run verify_recovery, then close on criteria_met or escalate with evidence.",
                    success_signal="verify_recovery reports criteria_met=true and close_incident succeeds.",
                )
            )
        return AgentPlan(
            hypothesis=str(raw["hypothesis"]),
            steps=steps,
            rationale_summary=str(raw.get("rationale_summary", "")),
        )

    def create(self, incident_observation: dict[str, Any], context: list[dict[str, Any]] | None = None) -> AgentPlan:
        """Create an explicit initial plan grounded in the bootstrap observation.

        The plan is required before any remediation, so a model failure here must not
        stop the run: an explicit deterministic fallback keeps the agent evidence-first
        rather than letting it act with no plan.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "Create a short SRE investigation-and-remediation plan for a local "
                    "simulated platform. Do not assume the ticket's suspected root cause "
                    "is correct.\n"
                    "Rules:\n"
                    "- Every step must be achievable with the local tools: reading health, "
                    "metrics, logs, deployments, dependencies and runbooks, then restarting, "
                    "scaling, clearing cache, rolling back, failing over or shifting traffic.\n"
                    "- Do not plan staging deployments, code changes, hotfixes, post-mortems "
                    "or long-term monitoring; they are not available.\n"
                    "- Investigate before remediating, and finish with a verification step."
                ),
            },
            {"role": "user", "content": f"Incident observation: {incident_observation}"},
        ]
        if context:
            messages.extend(context)
        try:
            return self._parse(self.model.structured(messages, "incident_plan", PLAN_SCHEMA))
        except Exception as exc:
            return self._fallback_plan(f"{type(exc).__name__}: {exc}")

    def revise(self, current: AgentPlan, trigger: dict[str, Any], state_summary: str) -> AgentPlan:
        """Produce the next revision of the plan in response to a concrete trigger.

        A revision is not "ask the model again". The model is shown what it already
        believed, what actually happened, and what evidence has accumulated, and is
        required to respond to that. The revision counter always advances, even when
        the model fails to return a usable plan, so that the trace shows a decision
        point was reached.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are revising an in-flight SRE incident plan. The previous plan "
                    "did not survive contact with the environment. Produce the next plan "
                    "revision.\n"
                    "Rules:\n"
                    "- Do not restate the failed hypothesis unless the new evidence still "
                    "supports it; if you keep it, the steps must change.\n"
                    "- Keep subgoals that are already satisfied; do not re-investigate "
                    "what has been observed.\n"
                    "- Only reference tools that exist in the local tool surface.\n"
                    "- If the evidence indicates the cause is outside the managed "
                    "platform, or no safe action remains, plan to escalate with evidence."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Previous hypothesis: {current.hypothesis}\n"
                    f"Previous steps: {[s.objective for s in current.steps]}\n"
                    f"Revision trigger: {trigger}\n"
                    f"Agent state: {state_summary}"
                ),
            },
        ]
        try:
            raw = self.model.structured(messages, "incident_plan", PLAN_SCHEMA)
            steps = [PlanStep(**row) for row in raw["steps"]]
            revised = AgentPlan(
                hypothesis=raw["hypothesis"],
                steps=steps,
                revision=current.revision + 1,
                rationale_summary=raw["rationale_summary"],
            )
        except Exception:
            # A failed revision must still advance the counter: the controller has to
            # be able to see that re-planning was attempted and did not help, so that
            # repeated failures lead to escalation instead of a silent retry loop.
            revised = AgentPlan(
                hypothesis=current.hypothesis,
                steps=current.steps,
                revision=current.revision + 1,
                rationale_summary=(
                    f"Revision {current.revision + 1} could not be generated by the "
                    f"model; previous plan retained. Trigger: {trigger}"
                ),
            )
        return revised
