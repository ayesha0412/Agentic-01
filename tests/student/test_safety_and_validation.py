"""Failure classes: malformed tool requests, human approval denial, repetition.

These cover the boundary between what the model asks for and what actually reaches
the simulator. In every case the assertion is about the world, not about the log:
a refused action must leave world_version untouched, because that is the only
evidence that nothing was executed.
"""

from __future__ import annotations

import pytest

from incidentzero.approval.gateway import AlwaysApproveGateway, AlwaysDenyGateway
from incidentzero.domain.models import ToolCall

ROLLBACK_ARGS = {
    "service": "checkout-service",
    "target_version": "2.4.0",
    "expected_world_version": 1,
    "reason": "Errors began immediately after release 2.4.1 was deployed.",
}


# --- failure class 1: malformed or invented tool requests -------------------------
@pytest.mark.parametrize(
    "name, arguments, because",
    [
        ("delete_everything", {}, "tool does not exist"),
        ("get_metrics", {}, "required argument 'service' is missing"),
        ("get_logs", {"service": "checkout-service", "limit": "ten"}, "limit must be an integer"),
        ("get_logs", {"service": "checkout-service", "limit": 99}, "limit exceeds the schema maximum"),
        ("scale_service", {"service": "checkout-service", "replicas": 40,
                           "expected_world_version": 1, "reason": "far beyond the allowed range"},
         "replicas exceeds the schema maximum"),
        ("get_runbook", {"topic": "astrology"}, "topic is not in the permitted enum"),
    ],
)
def test_invalid_requests_never_reach_the_simulator(make_controller, name, arguments, because):
    controller = make_controller()
    before = controller.tools.environment.world_version

    result = controller._execute_tool_call(ToolCall("c1", name, arguments))

    assert result["status"] == "validation_error", because
    assert controller.tools.environment.world_version == before
    # A rejection must teach the model something, not just fail.
    assert result["message"]


def test_validation_failure_does_not_consume_tool_budget(make_controller):
    controller = make_controller()
    spent_before = controller.budget.tool_calls

    controller._execute_tool_call(ToolCall("c1", "not_a_real_tool", {}))

    assert controller.budget.tool_calls == spent_before


def test_extra_argument_is_rejected(make_controller):
    """additionalProperties is false, so an invented parameter must not slip through."""
    controller = make_controller()
    result = controller._execute_tool_call(
        ToolCall("c1", "get_metrics", {"service": "checkout-service", "force": True})
    )
    assert result["status"] == "validation_error"


# --- failure class 2: human approval denial ---------------------------------------
def test_denied_high_risk_action_is_not_executed(make_controller, read_trace):
    controller = make_controller(approval=AlwaysDenyGateway())
    # Ground the action first, so this exercises the approval gate rather than the
    # evidence precondition that now sits in front of it. Going through the execution
    # boundary absorbs the transient failures this seed injects.
    controller.state.observe_result(
        controller._execute_tool_call(
            ToolCall("seed", "get_deployments", {"service": "checkout-service"})
        )
    )
    before = controller.tools.environment.world_version

    result = controller._execute_tool_call(ToolCall("c1", "rollback_deployment", ROLLBACK_ARGS))

    assert result["status"] == "approval_denied"
    # The decisive assertion: the simulator never moved, so nothing ran.
    assert controller.tools.environment.world_version == before
    assert controller.state.approvals == [
        {"tool": "rollback_deployment", "granted": False, "justification": ROLLBACK_ARGS["reason"]}
    ]
    assert read_trace("approval_requested") and read_trace("approval_result")


def test_denial_is_classified_as_replan_not_retry(make_controller):
    controller = make_controller(approval=AlwaysDenyGateway())
    result = controller._execute_tool_call(ToolCall("c1", "rollback_deployment", ROLLBACK_ARGS))

    assert controller.replan_policy.classify(result) == "replan"
    assert controller.replan_policy.should_replan(result) is True


def test_medium_risk_action_does_not_ask_for_approval(make_controller, read_trace):
    """Approval is driven by the policy file, so a medium action must run unimpeded."""

    class ExplodingGateway:
        def approve(self, *args, **kwargs):
            raise AssertionError("approval must not be requested for a medium-risk action")

    controller = make_controller(approval=ExplodingGateway())
    result = controller._execute_tool_call(
        ToolCall("c1", "restart_service", {
            "service": "checkout-service",
            "expected_world_version": 1,
            "reason": "Restart the degraded component identified by the survey.",
        })
    )
    assert result["status"] == "ok"
    assert read_trace("approval_requested") == []


def test_risk_levels_come_from_configuration(make_controller, tmp_path):
    """Raising a tool's risk in the policy file must gate it, with no code change."""
    policy = tmp_path / "risk.json"
    policy.write_text('{"restart_service": "critical", "get_metrics": "low"}')

    controller = make_controller(approval=AlwaysDenyGateway(), risk_policy_path=str(policy))
    # restart_service is now critical under this policy, so it needs evidence too.
    controller.state.observe_result(
        controller._execute_tool_call(
            ToolCall("seed", "get_metrics", {"service": "checkout-service"})
        )
    )
    before = controller.tools.environment.world_version

    result = controller._execute_tool_call(
        ToolCall("c1", "restart_service", {
            "service": "checkout-service",
            "expected_world_version": 1,
            "reason": "Restart that is now critical under the modified policy.",
        })
    )
    assert result["status"] == "approval_denied"
    assert controller.tools.environment.world_version == before


def test_unknown_tool_defaults_to_the_most_restrictive_risk(make_controller):
    """A tool missing from the policy must be treated as critical, never as safe."""
    controller = make_controller()
    assert controller.risk.requires_human_approval("some_future_tool") is True


# --- failure class 3: unproductive repetition -------------------------------------
def test_third_identical_action_is_blocked(make_controller):
    controller = make_controller()
    args = {"service": "checkout-service"}

    statuses = [
        controller._execute_tool_call(ToolCall(f"c{i}", "get_metrics", args))["status"]
        for i in range(3)
    ]
    assert statuses == ["ok", "ok", "loop_blocked"]


def test_loop_guard_ignores_argument_ordering(make_controller):
    """The same action written two ways is one action, so the count must carry over."""
    controller = make_controller()
    controller._execute_tool_call(ToolCall("c1", "get_logs", {"service": "order-db", "limit": 5}))
    controller._execute_tool_call(ToolCall("c2", "get_logs", {"limit": 5, "service": "order-db"}))
    third = controller._execute_tool_call(ToolCall("c3", "get_logs", {"service": "order-db", "limit": 5}))

    assert third["status"] == "loop_blocked"


def test_different_arguments_are_not_a_loop(make_controller):
    controller = make_controller()
    for service in ("checkout-service", "payment-service", "order-service"):
        result = controller._execute_tool_call(ToolCall("c", "get_metrics", {"service": service}))
        assert result["status"] == "ok"


def test_blocked_loop_is_classified_as_replan(make_controller):
    controller = make_controller()
    args = {"service": "checkout-service"}
    for _ in range(3):
        result = controller._execute_tool_call(ToolCall("c", "get_metrics", args))
    assert controller.replan_policy.classify(result) == "replan"
