from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from incidentzero.domain.models import AgentPlan, ToolCall


#: Terminal statuses required by the grading contract, plus the in-flight one.
RUNNING = "running"
RESOLVED = "resolved"
ESCALATED = "escalated"
ABORTED = "aborted"
BUDGET_EXHAUSTED = "budget_exhausted"
FAILED = "failed"

TERMINAL_STATUSES = frozenset({RESOLVED, ESCALATED, ABORTED, BUDGET_EXHAUSTED, FAILED})

#: Observations that count as having investigated a service. Health is excluded on
#: purpose: the bootstrap survey reads health for every service, so counting it would
#: make "have we looked into this component" true everywhere and the check vacuous.
#: These four require the agent to have actually gone and looked at the component.
DIAGNOSTIC_TOOLS = frozenset(
    {"get_metrics", "get_logs", "get_deployments", "get_dependencies"}
)

#: Tools that change the world, so a prior verification no longer describes it.
MUTATING_TOOLS = frozenset(
    {
        "restart_service",
        "scale_service",
        "clear_cache",
        "rollback_deployment",
        "failover_database",
        "shift_traffic",
    }
)


@dataclass(slots=True)
class Intent:
    """An action the model proposed, together with the evidence that motivated it.

    Recording the justification is what makes a stale-world recovery meaningful.
    When an action is rejected because the world moved, the controller can
    re-observe and ask "does the reason I picked this action still hold?" instead
    of blindly substituting the newer version number and firing again.
    """

    call: ToolCall
    justification: str
    observed_world_version: int | None
    evidence_ids: list[str] = field(default_factory=list)


@dataclass
class AgentState:
    messages: list[dict[str, Any]] = field(default_factory=list)
    plan: AgentPlan | None = None
    evidence_ids: list[str] = field(default_factory=list)
    latest_world_version: int | None = None
    last_tool_results: list[dict[str, Any]] = field(default_factory=list)
    status: str = RUNNING

    # --- recovery verification -------------------------------------------------
    # close_incident is refused unless it cites the evidence id of the most recent
    # verify_recovery call, so the id has to survive outside the message history.
    latest_verify_evidence_id: str | None = None
    latest_verify_criteria_met: bool = False

    # --- decision bookkeeping --------------------------------------------------
    pending_intent: Intent | None = None
    approvals: list[dict[str, Any]] = field(default_factory=list)
    plan_revisions: list[dict[str, Any]] = field(default_factory=list)
    executed_actions: list[str] = field(default_factory=list)
    terminal_reason: str | None = None

    # Previous release per service, as actually observed via get_deployments. Used to
    # check a rollback target against evidence instead of against the model's memory.
    observed_previous_version: dict[str, str] = field(default_factory=dict)

    # Which diagnostic observations have been made about each service. This is what
    # lets the controller answer "what did the agent know when it chose this action"
    # before the action runs, rather than only afterwards from the trace.
    observed_diagnostics: dict[str, list[str]] = field(default_factory=dict)

    # R1 requires the call budgets to be part of the explicit state. BudgetManager
    # remains the authority that enforces them; this is the state's view of them, so a
    # state summary or a trace row shows what the agent had left at that moment.
    budget_snapshot: dict[str, int] = field(default_factory=dict)

    # Remediations that ran successfully but reported no root-cause remediation. Each
    # one is evidence against the hypothesis that motivated it.
    ineffective_actions: list[str] = field(default_factory=list)

    # ---------------------------------------------------------------------------
    @property
    def plan_revision(self) -> int:
        return self.plan.revision if self.plan else 0

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def observe_result(self, result: dict[str, Any]) -> None:
        """Fold one tool result into state.

        Every result carries an evidence id and a world version, so this is the single
        place where the agent's picture of the world advances.
        """
        evidence = result.get("evidence_id")
        if evidence:
            self.evidence_ids.append(evidence)
        version = result.get("world_version")
        if isinstance(version, int):
            self.latest_world_version = version
        self.last_tool_results.append(result)
        self.last_tool_results = self.last_tool_results[-8:]

        tool = result.get("tool")
        if tool in DIAGNOSTIC_TOOLS and result.get("status") == "ok":
            service = (result.get("data") or {}).get("service")
            if isinstance(service, str):
                seen = self.observed_diagnostics.setdefault(service, [])
                if tool not in seen:
                    seen.append(str(tool))

        if result.get("tool") == "get_deployments" and result.get("status") == "ok":
            data = result.get("data") or {}
            service = data.get("service")
            previous = (data.get("previous") or {}).get("version")
            if isinstance(service, str) and isinstance(previous, str):
                self.observed_previous_version[service] = previous

        if result.get("tool") == "verify_recovery" and result.get("status") == "ok":
            data = result.get("data") or {}
            self.latest_verify_evidence_id = evidence
            self.latest_verify_criteria_met = bool(data.get("criteria_met"))
        elif result.get("status") == "ok" and result.get("tool") in MUTATING_TOOLS:
            # Any change to the world invalidates an earlier verification: the incident
            # must be re-verified before it can be closed.
            self.latest_verify_criteria_met = False
            self.executed_actions.append(str(result.get("tool")))

    def record_approval(self, tool_name: str, granted: bool, justification: str) -> None:
        self.approvals.append(
            {"tool": tool_name, "granted": granted, "justification": justification}
        )

    def record_plan_revision(self, trigger: str, hypothesis: str) -> None:
        self.plan_revisions.append(
            {
                "revision": self.plan_revision,
                "trigger": trigger,
                "hypothesis": hypothesis,
            }
        )

    def terminate(self, status: str, reason: str) -> None:
        self.status = status
        self.terminal_reason = reason

    def summary(self) -> dict[str, Any]:
        """Compact view of state for prompts and trace rows (never the raw messages)."""
        return {
            "status": self.status,
            "plan_revision": self.plan_revision,
            "hypothesis": self.plan.hypothesis if self.plan else None,
            "latest_world_version": self.latest_world_version,
            "evidence_count": len(self.evidence_ids),
            "executed_actions": self.executed_actions,
            "latest_verify_criteria_met": self.latest_verify_criteria_met,
            "approvals": self.approvals,
            "observed_diagnostics": self.observed_diagnostics,
            "budget": self.budget_snapshot,
            "ineffective_actions": self.ineffective_actions,
        }

