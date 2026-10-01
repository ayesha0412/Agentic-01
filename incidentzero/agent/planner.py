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

    def create(self, incident_observation: dict[str, Any], context: list[dict[str, Any]] | None = None) -> AgentPlan:
        """Create an explicit initial plan.

        The baseline is intentionally thin. Improve validation, retry behavior, grounding,
        and budget integration in your submission.
        """
        messages = [
            {"role": "system", "content": "Create a short SRE investigation-and-remediation plan. Do not assume the ticket's suspected root cause is correct."},
            {"role": "user", "content": f"Incident observation: {incident_observation}"},
        ]
        raw = self.model.structured(messages, "incident_plan", PLAN_SCHEMA)
        steps = [PlanStep(**row) for row in raw["steps"]]
        return AgentPlan(hypothesis=raw["hypothesis"], steps=steps, rationale_summary=raw["rationale_summary"])

    def revise(self, current: AgentPlan, trigger: dict[str, Any], state_summary: str) -> AgentPlan:
        # TODO(A1): implement a grounded plan revision that increments revision,
        # preserves completed work when appropriate, and does not repeat a failed plan blindly.
        return current
