from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from incidentzero.domain.models import AgentPlan


@dataclass
class AgentState:
    messages: list[dict[str, Any]] = field(default_factory=list)
    plan: AgentPlan | None = None
    evidence_ids: list[str] = field(default_factory=list)
    latest_world_version: int | None = None
    last_tool_results: list[dict[str, Any]] = field(default_factory=list)
    repeated_actions: dict[str, int] = field(default_factory=dict)
    status: str = "running"

    def observe_result(self, result: dict[str, Any]) -> None:
        evidence = result.get("evidence_id")
        if evidence:
            self.evidence_ids.append(evidence)
        version = result.get("world_version")
        if isinstance(version, int):
            self.latest_world_version = version
        self.last_tool_results.append(result)
        self.last_tool_results = self.last_tool_results[-8:]
