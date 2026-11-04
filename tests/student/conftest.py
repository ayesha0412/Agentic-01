"""Shared fixtures for the student test suite.

Every test here runs offline. The model is replaced by ScriptedModelClient so that
controller behaviour can be asserted against a model that is deliberately wrong,
malformed or unhelpful, and so the suite consumes no Groq quota.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from incidentzero.agent.controller import AgentController
from incidentzero.approval.gateway import AlwaysApproveGateway
from incidentzero.config import Limits
from incidentzero.domain.models import ModelReply, ToolCall
from incidentzero.environment.engine import SimulationEnvironment
from incidentzero.model.scripted import ScriptedModelClient
from incidentzero.telemetry.budget import BudgetManager
from incidentzero.telemetry.trace import TraceRecorder
from incidentzero.tools.registry import ToolRegistry

#: A structured plan the scripted model can return, valid against PLAN_SCHEMA.
VALID_PLAN = {
    "hypothesis": "A degraded component on the checkout path is causing customer errors.",
    "rationale_summary": "Survey evidence identifies the degraded component.",
    "steps": [
        {"step_id": "S1", "objective": "Confirm the failure mode", "success_signal": "logs identify a cause"},
        {"step_id": "S2", "objective": "Remediate and verify", "success_signal": "criteria_met is true"},
    ],
}


def tool_reply(name: str, arguments: dict[str, Any], call_id: str = "call-1") -> ModelReply:
    return ModelReply(tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)])


@pytest.fixture
def trace_path(tmp_path: Path) -> Path:
    return tmp_path / "trace.jsonl"


@pytest.fixture
def make_controller(trace_path: Path):
    """Build a controller wired to a scripted model and an in-memory simulator."""

    def _make(
        decisions: list[ModelReply] | None = None,
        *,
        scenario: str = "public-b",
        approval: Any = None,
        limits: Limits | None = None,
        plans: list[dict] | None = None,
        student_id: str = "TEST-001",
        risk_policy_path: str = "configs/risk_policy.json",
    ) -> AgentController:
        limits = limits or Limits.load()
        env = SimulationEnvironment(student_id, scenario)
        return AgentController(
            model=ScriptedModelClient(
                decisions=decisions or [],
                structured_outputs=plans if plans is not None else [VALID_PLAN] * 6,
            ),
            tools=ToolRegistry(env),
            approval=approval or AlwaysApproveGateway(),
            budget=BudgetManager(
                max_llm_calls=limits.max_llm_calls, max_tool_calls=limits.max_tool_calls
            ),
            trace=TraceRecorder(trace_path),
            limits=limits,
            risk_policy_path=risk_policy_path,
        )

    return _make


@pytest.fixture
def read_trace(trace_path: Path):
    """Return every trace row, or only those matching one event name."""

    def _read(event: str | None = None) -> list[dict[str, Any]]:
        if not trace_path.exists():
            return []
        rows = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
        return [r for r in rows if event is None or r["event"] == event]

    return _read
