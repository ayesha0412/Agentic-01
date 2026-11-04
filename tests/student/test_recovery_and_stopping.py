"""Failure classes: stale world, transient failure, false success, budget exhaustion.

These cover what the controller does after something has already gone wrong, which
is where the difference between retrying, re-observing, re-planning and stopping
actually shows up.
"""

from __future__ import annotations

import pytest

from incidentzero.agent.recovery import RetryPolicy
from incidentzero.agent.state import TERMINAL_STATUSES, AgentState
from incidentzero.config import Limits
from incidentzero.domain.models import ModelReply, ToolCall
from incidentzero.model.errors import PermanentModelError, TransientModelError

from .conftest import tool_reply

SCALE_ARGS = {
    "service": "checkout-service",
    "replicas": 4,
    "expected_world_version": 1,
    "reason": "CPU saturation at 99 percent on the degraded component.",
}


# --- failure class 4: the world moved between observation and action --------------
def test_stale_action_is_rejected_and_not_executed(make_controller):
    controller = make_controller()
    # Move the world on with an unrelated action, so the planned version is stale.
    controller._execute_tool_call(ToolCall("c0", "restart_service", {
        "service": "auth-service", "expected_world_version": 1,
        "reason": "Advance the world version for this test.",
    }))
    version_after_change = controller.tools.environment.world_version

    result = controller._execute_tool_call(ToolCall("c1", "scale_service", SCALE_ARGS))

    assert result["status"] == "stale_precondition"
    assert controller.tools.environment.world_version == version_after_change


def test_stale_world_triggers_re_observation_not_a_blind_refire(make_controller, read_trace):
    """A stale rejection must send the agent back to the evidence, not bump the number."""
    controller = make_controller()
    controller._execute_tool_call(ToolCall("c0", "restart_service", {
        "service": "auth-service", "expected_world_version": 1,
        "reason": "Advance the world version for this test.",
    }))
    stale_call = ToolCall("c1", "scale_service", SCALE_ARGS)
    result = controller._execute_tool_call(stale_call)

    version_before = controller.tools.environment.world_version
    controller._handle_stale_world(stale_call, result)

    # Fresh evidence was gathered, and no mutating action was fired.
    assert read_trace("stale_world")
    assert [r["payload"]["tool"] for r in read_trace("controller_observation")] == [
        "get_service_health",
        "get_metrics",
    ]
    assert controller.tools.environment.world_version == version_before
    # And the model is told explicitly not to just resend with a newer version.
    assert "Do not resend the same action" in controller.state.messages[-1]["content"]


def test_stale_is_classified_as_re_observe_despite_being_marked_retryable(make_controller):
    """The simulator marks staleness retryable; the controller must not simply retry."""
    controller = make_controller()
    stale = {"status": "stale_precondition", "retryable": True, "tool": "scale_service"}

    assert stale["retryable"] is True
    assert controller.replan_policy.classify(stale) == "re_observe"
    assert controller.replan_policy.classify(stale) != "retry"


# --- failure class 5: transient failures, bounded ----------------------------------
class FlakyRegistry:
    """Wraps a real registry and fails the first N calls to one tool, transiently."""

    def __init__(self, inner, tool: str, failures: int) -> None:
        self._inner = inner
        self._tool = tool
        self._remaining = failures
        self.executed: list[str] = []

    def __getattr__(self, item):
        return getattr(self._inner, item)

    def execute(self, name, arguments):
        self.executed.append(name)
        if name == self._tool and self._remaining > 0:
            self._remaining -= 1
            return {
                "status": "transient_error", "tool": name,
                "world_version": self._inner.environment.world_version,
                "evidence_id": None, "data": None, "retryable": True,
                "message": "Simulated telemetry backend timeout.",
            }
        return self._inner.execute(name, arguments)


def test_transient_observation_failure_is_absorbed_before_the_model_sees_it(make_controller, read_trace):
    """The model must receive the evidence, not the blip the controller already handled."""
    controller = make_controller()
    controller.tools = FlakyRegistry(controller.tools, "get_logs", failures=1)

    result = controller._execute_tool_call(
        ToolCall("c1", "get_logs", {"service": "checkout-service", "limit": 10})
    )

    assert result["status"] == "ok", "the model should never see the transient failure"
    assert read_trace("tool_retry"), "a retry happened behind the scenes"
    # Retried at least once, and never beyond the configured bound.
    assert 2 <= len(controller.tools.executed) <= controller.limits.max_consecutive_model_retries


def test_observation_retry_is_bounded_and_then_redirects(make_controller, read_trace):
    """If retries cannot rescue the observation, stop and tell the model to try elsewhere."""
    controller = make_controller()
    controller.tools = FlakyRegistry(controller.tools, "get_logs", failures=99)

    result = controller._execute_tool_call(
        ToolCall("c1", "get_logs", {"service": "checkout-service", "limit": 10})
    )

    assert result["status"] == "transient_error"
    assert len(controller.tools.executed) == controller.limits.max_consecutive_model_retries
    assert "different observation tool" in result["message"]
    assert read_trace("tool_retry_exhausted")


def test_a_failed_mutating_action_is_never_silently_repeated(make_controller):
    """Read failures may be repeated; an action failure must be re-decided instead."""
    controller = make_controller()
    controller.tools = FlakyRegistry(controller.tools, "restart_service", failures=1)

    result = controller._execute_tool_call(ToolCall("c1", "restart_service", {
        "service": "checkout-service", "expected_world_version": 1,
        "reason": "Restart the degraded checkout component.",
    }))

    assert result["status"] == "transient_error"
    assert controller.tools.executed == ["restart_service"], "no automatic repeat of an action"


def test_model_retry_is_bounded_and_counts_every_attempt():
    sleeps: list[float] = []
    calls = {"n": 0}
    retry = RetryPolicy(max_attempts=3, sleeper=sleeps.append)

    def always_transient():
        calls["n"] += 1
        raise TransientModelError("429 rate limited")

    with pytest.raises(TransientModelError):
        retry.call_model(always_transient)

    assert calls["n"] == 3, "exactly max_attempts calls, never a fourth"
    assert sleeps == [0.5, 1.0], "no sleep after the final attempt"
    assert retry.last_attempts == 3


def test_permanent_model_error_is_never_retried():
    calls = {"n": 0}
    retry = RetryPolicy(max_attempts=3, sleeper=lambda s: None)

    def permanent():
        calls["n"] += 1
        raise PermanentModelError("401 invalid api key")

    with pytest.raises(PermanentModelError):
        retry.call_model(permanent)
    assert calls["n"] == 1


def test_model_retries_consume_llm_budget(make_controller):
    """A retry is still a provider request, so it must be charged (assignment Q23)."""
    controller = make_controller()
    controller.retry_policy = RetryPolicy(max_attempts=3, sleeper=lambda s: None)

    class AlwaysRateLimited:
        def decide(self, messages, tools):
            raise TransientModelError("429")

    controller.model = AlwaysRateLimited()
    before = controller.budget.llm_calls

    with pytest.raises(TransientModelError):
        controller._model_decide()

    assert controller.budget.llm_calls == before + 3


def test_exhausted_transient_model_failure_escalates_rather_than_crashing(make_controller):
    controller = make_controller()
    controller.retry_policy = RetryPolicy(max_attempts=2, sleeper=lambda s: None)

    class AlwaysRateLimited:
        def decide(self, messages, tools):
            raise TransientModelError("429 rate limited")

        def structured(self, messages, name, schema):
            return {
                "hypothesis": "h", "rationale_summary": "r",
                "steps": [{"step_id": "S1", "objective": "o", "success_signal": "s"},
                          {"step_id": "S2", "objective": "o2", "success_signal": "s2"}],
            }

    controller.model = AlwaysRateLimited()
    controller.planner.model = AlwaysRateLimited()

    outcome = controller.run()
    assert outcome.status == "escalated"
    assert controller.tools.environment._escalated is True


def test_permanent_model_error_aborts_safely(make_controller):
    controller = make_controller()

    class BadKey:
        def decide(self, messages, tools):
            raise PermanentModelError("401 invalid api key")

        def structured(self, messages, name, schema):
            return {
                "hypothesis": "h", "rationale_summary": "r",
                "steps": [{"step_id": "S1", "objective": "o", "success_signal": "s"},
                          {"step_id": "S2", "objective": "o2", "success_signal": "s2"}],
            }

    controller.model = BadKey()
    controller.planner.model = BadKey()

    outcome = controller.run()
    assert outcome.status == "aborted"


# --- failure class 6: an action succeeded but the incident is not fixed ------------
def test_action_ok_is_not_proof_of_recovery(make_controller):
    """A restart on a bad-deploy incident returns ok while the objective stays violated."""
    controller = make_controller(scenario="public-b")
    restart = controller._execute_tool_call(ToolCall("c1", "restart_service", {
        "service": "checkout-service",
        "expected_world_version": controller.tools.environment.world_version,
        "reason": "Restart the degraded checkout component.",
    }))
    assert restart["status"] == "ok"

    verify = controller._execute_tool_call(ToolCall("c2", "verify_recovery", {}))
    controller.state.observe_result(verify)

    assert verify["data"]["criteria_met"] is False
    assert controller.replan_policy.classify(verify) == "replan"


def test_failed_verification_produces_a_real_plan_revision(make_controller, read_trace):
    controller = make_controller(scenario="public-b")
    controller.state.plan = controller.planner.create({"title": "t"})
    revision_before = controller.state.plan.revision

    controller._revise_plan({"reason": "verification_failed", "tool": "verify_recovery"})

    assert controller.state.plan.revision == revision_before + 1
    assert read_trace("plan_revised")
    assert controller.state.plan_revisions[-1]["trigger"] == "verification_failed"


def test_a_mutating_action_invalidates_an_earlier_passing_verification():
    state = AgentState()
    state.observe_result({"tool": "verify_recovery", "status": "ok", "evidence_id": "EV-0009",
                          "world_version": 5, "data": {"criteria_met": True}})
    assert state.latest_verify_criteria_met is True

    state.observe_result({"tool": "restart_service", "status": "ok", "evidence_id": "EV-0010",
                          "world_version": 6, "data": {}})
    assert state.latest_verify_criteria_met is False


# --- failure class 7: premature or unevidenced closure -----------------------------
def test_close_without_verification_is_refused_by_the_controller(make_controller):
    controller = make_controller(scenario="public-b")
    call = ToolCall("c1", "close_incident", {
        "summary": "The incident appears to be fully resolved now.",
        "evidence_ids": ["EV-0001"],
        "expected_world_version": controller.tools.environment.world_version,
        "reason": "Closing without running verify_recovery first.",
    })

    assert controller._precondition_failure(call) is not None
    assert controller.tools.environment._closed is False


def test_close_must_cite_the_latest_verification_evidence(make_controller):
    controller = make_controller(scenario="public-b")
    controller.state.latest_verify_criteria_met = True
    controller.state.latest_verify_evidence_id = "EV-0042"

    stale_citation = ToolCall("c1", "close_incident", {
        "summary": "Closing while citing an older piece of evidence.",
        "evidence_ids": ["EV-0001"],
        "expected_world_version": 1,
        "reason": "Cites evidence that is not the latest verification.",
    })
    assert "EV-0042" in (controller._precondition_failure(stale_citation) or "")

    correct_citation = ToolCall("c2", "close_incident", {
        "summary": "Closing while citing the latest verification evidence.",
        "evidence_ids": ["EV-0042"],
        "expected_world_version": 1,
        "reason": "Cites the most recent verification evidence id.",
    })
    assert controller._precondition_failure(correct_citation) is None


def test_rollback_without_deployment_evidence_never_reaches_the_approver(make_controller, read_trace):
    """A live run proposed rolling back to a version it had never observed."""

    class ExplodingGateway:
        def approve(self, *args, **kwargs):
            raise AssertionError("an unevidenced rollback must not reach the approver")

    controller = make_controller(approval=ExplodingGateway())
    call = ToolCall("c1", "rollback_deployment", {
        "service": "checkout-service", "target_version": "2.3.9",
        "expected_world_version": 1,
        "reason": "Rolling back to a version that was never observed.",
    })

    assert "get_deployments first" in (controller._precondition_failure(call) or "")


def test_rollback_target_must_match_observed_deployment_evidence(make_controller):
    controller = make_controller()
    controller.state.observe_result(
        controller.tools.execute("get_deployments", {"service": "checkout-service"})
    )
    assert controller.state.observed_previous_version["checkout-service"] == "2.4.0"

    def rollback_to(version: str) -> ToolCall:
        return ToolCall("c", "rollback_deployment", {
            "service": "checkout-service", "target_version": version,
            "expected_world_version": 1,
            "reason": "Deployment correlates with the onset of customer errors.",
        })

    assert controller._precondition_failure(rollback_to("2.3.9")) is not None
    assert controller._precondition_failure(rollback_to("2.4.0")) is None


@pytest.mark.parametrize(
    "tool, arguments",
    [
        ("failover_database", {"service": "order-db", "expected_world_version": 1,
                               "reason": "The database looks like the problem to me."}),
        ("shift_traffic", {"service": "checkout-service", "percent_to_secondary": 50,
                           "expected_world_version": 1,
                           "reason": "Shifting traffic away seems like a sensible move."}),
    ],
)
def test_high_risk_action_without_evidence_never_reaches_the_approver(make_controller, tool, arguments):
    """R3: a high or critical action must be supported by observed evidence."""

    class ExplodingGateway:
        def approve(self, *args, **kwargs):
            raise AssertionError("an unevidenced high-risk action must not reach the approver")

    controller = make_controller(approval=ExplodingGateway())
    before = controller.tools.environment.world_version

    result = controller._execute_tool_call(ToolCall("c1", tool, arguments))

    assert result["status"] == "precondition_failed"
    assert controller.tools.environment.world_version == before
    assert controller.replan_policy.classify(result) == "replan"


def test_the_same_action_is_allowed_once_the_service_has_been_investigated(make_controller):
    """The gate demands evidence; it must not block a properly grounded action."""
    controller = make_controller()
    controller.state.observe_result(
        controller._execute_tool_call(ToolCall("seed", "get_metrics", {"service": "order-db"}))
    )

    call = ToolCall("c1", "failover_database", {
        "service": "order-db", "expected_world_version": 1,
        "reason": "Connection saturation observed on the primary.",
    })
    assert controller._precondition_failure(call) is None


def test_health_alone_does_not_count_as_having_investigated_a_service(make_controller):
    """The bootstrap survey reads health everywhere, so health cannot satisfy the gate."""
    controller = make_controller()
    controller.state.observe_result(
        controller._execute_tool_call(
            ToolCall("seed", "get_service_health", {"service": "order-db"})
        )
    )

    call = ToolCall("c1", "failover_database", {
        "service": "order-db", "expected_world_version": 1,
        "reason": "Acting on a health check alone, with no diagnostics.",
    })
    assert controller._precondition_failure(call) is not None


def test_the_evidence_gate_follows_the_risk_policy(make_controller, tmp_path):
    """Demote a tool in the policy and the evidence requirement goes with it."""
    policy = tmp_path / "risk.json"
    policy.write_text('{"failover_database": "medium", "get_metrics": "low"}')
    controller = make_controller(risk_policy_path=str(policy))

    call = ToolCall("c1", "failover_database", {
        "service": "order-db", "expected_world_version": 1,
        "reason": "Medium risk under this policy, so no evidence gate applies.",
    })
    assert controller._precondition_failure(call) is None


def test_observation_tools_are_never_blocked_by_a_policy_that_makes_everything_critical(make_controller, tmp_path):
    """An empty policy makes every tool critical. The agent must still be able to look."""
    policy = tmp_path / "risk.json"
    policy.write_text("{}")
    controller = make_controller(risk_policy_path=str(policy))

    assert controller.risk.requires_human_approval("get_metrics") is True
    # Observation tools are not mutating, so the evidence gate does not apply to them
    # and the agent cannot be deadlocked into needing evidence before it may observe.
    assert controller._precondition_failure(
        ToolCall("c1", "get_metrics", {"service": "order-db"})
    ) is None


# --- failure class 9: the action ran but the evidence contradicts the hypothesis ---
def test_an_ineffective_remediation_is_read_as_evidence_against_the_hypothesis(make_controller):
    """R4: new evidence contradicting the hypothesis must trigger reconsideration."""
    controller = make_controller()
    ineffective = {"status": "ok", "tool": "restart_service",
                   "data": {"effect": "service restarted; no confirmed root-cause remediation"}}
    effective = {"status": "ok", "tool": "restart_service",
                 "data": {"effect": "memory pressure cleared and service recovered"}}

    assert controller.replan_policy.classify(ineffective) == "contradicted"
    assert controller.replan_policy.should_replan(ineffective) is True
    assert controller.replan_policy.classify(effective) == "continue"


def test_first_contradiction_corrects_the_model_without_spending_a_revision(make_controller, read_trace):
    """One ineffective action is cheap to correct. A revision costs an LLM call."""
    controller = make_controller()
    controller.state.plan = controller.planner.create({"title": "t"})
    revision_before = controller.state.plan.revision

    controller._handle_contradiction(
        ToolCall("c1", "restart_service", {"service": "checkout-service"}),
        {"status": "ok", "tool": "restart_service",
         "data": {"effect": "service restarted; no confirmed root-cause remediation"}},
    )

    assert read_trace("hypothesis_contradicted")
    assert controller.state.plan.revision == revision_before
    assert "evidence against the current hypothesis" in controller.state.messages[-1]["content"]


def test_a_second_contradiction_revises_the_plan(make_controller, read_trace):
    """Two ineffective actions mean the hypothesis is wrong, not just the action."""
    controller = make_controller()
    controller.state.plan = controller.planner.create({"title": "t"})
    revision_before = controller.state.plan.revision
    result = {"status": "ok", "tool": "scale_service",
              "data": {"effect": "replica count changed; root cause may remain"}}

    controller._handle_contradiction(ToolCall("c1", "restart_service", {}), result)
    controller._handle_contradiction(ToolCall("c2", "scale_service", {}), result)

    assert controller.state.plan.revision == revision_before + 1
    assert read_trace("plan_revised")
    assert controller.state.plan_revisions[-1]["trigger"] == "hypothesis_contradicted"


# --- requirement checks that no live run happened to exercise ----------------------
def test_every_plan_ends_in_a_verification_step(make_controller):
    """R2 requires a final verification step, and models routinely omit one."""
    controller = make_controller(plans=[{
        "hypothesis": "A bad release is the cause.",
        "rationale_summary": "Errors started with the rollout.",
        "steps": [
            {"step_id": "S1", "objective": "Read deployments", "success_signal": "versions known"},
            {"step_id": "S2", "objective": "Roll back the release", "success_signal": "rollback returns ok"},
        ],
    }])
    plan = controller.planner.create({"title": "t"})

    assert len(plan.steps) == 3
    assert "verify" in plan.steps[-1].objective.lower()


def test_the_state_summary_exposes_the_call_budgets(make_controller):
    """R1 lists the call budgets among the state the agent must track."""
    controller = make_controller(decisions=[tool_reply("get_metrics", {"service": "order-db"})])
    controller.state.plan = controller.planner.create({"title": "t"})
    controller._model_decide()

    budget = controller.state.summary()["budget"]
    assert budget["llm_max"] == controller.budget.max_llm_calls
    assert budget["tool_max"] == controller.budget.max_tool_calls
    assert budget["llm_used"] >= 1


def test_prose_only_model_can_never_resolve_an_incident(make_controller):
    """Natural-language claims of success are not operational evidence (Q25)."""
    controller = make_controller(
        decisions=[ModelReply(content="I am confident the incident is fully resolved.")] * 6
    )
    outcome = controller.run()

    assert outcome.status == "escalated"
    assert controller.tools.environment._closed is False


# --- failure class 8: running out of budget ----------------------------------------
def test_low_budget_ends_in_safe_escalation_not_a_hard_stop(make_controller):
    controller = make_controller(
        decisions=[tool_reply("get_metrics", {"service": "checkout-service"})] * 10,
        limits=Limits(max_llm_calls=3, max_tool_calls=16),
    )
    outcome = controller.run()

    assert outcome.status in TERMINAL_STATUSES
    assert outcome.status == "escalated"
    assert controller.tools.environment._escalated is True
    assert controller.budget.llm_calls <= 3
    assert controller.budget.tool_calls <= 16


def test_every_run_ends_with_an_explicit_terminal_status(make_controller):
    """Whatever the model does, the run must land on one of the five statuses."""
    scenarios = [
        [],                                                        # model runs out of replies
        [ModelReply(content="thinking out loud")] * 5,             # prose only
        [tool_reply("not_a_tool", {})] * 5,                        # invented tool
        [tool_reply("get_metrics", {"service": "checkout-service"})] * 5,  # observation loop
    ]
    for decisions in scenarios:
        controller = make_controller(decisions=decisions)
        outcome = controller.run()
        assert outcome.status in TERMINAL_STATUSES
        assert outcome.trace_path
