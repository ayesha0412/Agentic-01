from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from incidentzero.domain.models import RiskLevel

from .state import MUTATING_TOOLS


class RiskPolicy:
    def __init__(self, config_path: str | Path = "configs/risk_policy.json") -> None:
        self.mapping = json.loads(Path(config_path).read_text(encoding="utf-8"))

    def risk(self, tool_name: str) -> RiskLevel:
        return RiskLevel(self.mapping.get(tool_name, "critical"))

    def requires_human_approval(self, tool_name: str) -> bool:
        return self.risk(tool_name) in {RiskLevel.HIGH, RiskLevel.CRITICAL}


#: The five ways the controller can respond to a tool result.
CONTINUE = "continue"      # nothing wrong; keep going with the current plan
RETRY = "retry"            # same call, same arguments, bounded number of times
RE_OBSERVE = "re_observe"  # the action may still be right, but our facts are stale
REJECT = "reject"          # the request itself was malformed; correct the model
REPLAN = "replan"          # the strategy is wrong; revise hypothesis/steps
CONTRADICTED = "contradicted"  # the action ran, but evidence says the cause is untouched


class ReplanPolicy:
    """Classify a tool result into one of the five response classes.

    The simulator's own ``retryable`` flag is deliberately NOT used as the control
    signal. It is a truthful statement about the error ("this is not permanent"),
    but it says nothing about whether the *action* is still justified. The clearest
    example is ``stale_precondition``: the simulator marks it ``retryable=True``,
    yet firing the same action again with a fresh version number is exactly the
    wrong move (assignment Q20) -- the world changed, so the decision must be
    re-examined against fresh observations.
    """

    #: Statuses that mean "the world moved under us".
    STALE_STATUSES = {"stale_precondition"}
    #: Statuses the controller synthesises for policy refusals (not simulator errors).
    POLICY_STATUSES = {"approval_denied", "loop_blocked", "budget_blocked"}
    #: Statuses that mean the request never should have been sent.
    REJECT_STATUSES = {"validation_error"}

    #: Phrases the simulator uses when an action ran but did not address the cause.
    #: Matching is substring based and deliberately forgiving: if the wording changes,
    #: the check simply stops firing, which is the behaviour before it existed.
    INEFFECTIVE_EFFECT_MARKERS = (
        "no confirmed root-cause remediation",
        "root cause may remain",
        "symptoms remain",
        "no evidence it addressed",
        "without evidence of corruption",
    )

    def contradicts_hypothesis(self, tool_result: dict[str, Any]) -> bool:
        """True when a remediation succeeded but reported that the cause is untouched.

        R4 lists "new evidence contradicts the active hypothesis" as a re-planning
        trigger. The simulator states this outright in the effect string of an action
        applied to the wrong root cause, so the contradiction is observable rather
        than something that has to be inferred from natural language.
        """
        if (tool_result or {}).get("status") != "ok":
            return False
        effect = str(((tool_result or {}).get("data") or {}).get("effect", "")).lower()
        return any(marker in effect for marker in self.INEFFECTIVE_EFFECT_MARKERS)

    def classify(self, tool_result: dict[str, Any]) -> str:
        status = (tool_result or {}).get("status")
        tool = (tool_result or {}).get("tool")

        if status in self.REJECT_STATUSES:
            return REJECT
        if status in self.STALE_STATUSES:
            return RE_OBSERVE
        if status in self.POLICY_STATUSES:
            return REPLAN
        if status == "transient_error" and tool_result.get("retryable") is True:
            return RETRY

        if status == "ok":
            # An action returning ok is not proof the incident is fixed. The only
            # tool that can speak to recovery is verify_recovery, and if it reports
            # criteria_met=false the current strategy has not worked.
            if tool == "verify_recovery":
                data = tool_result.get("data") or {}
                return CONTINUE if data.get("criteria_met") else REPLAN
            if self.contradicts_hypothesis(tool_result):
                return CONTRADICTED
            return CONTINUE

        # Remaining non-ok statuses are simulator-side refusals: unknown service,
        # wrong rollback target, premature close. None are fixed by repetition.
        return REPLAN

    def should_replan(self, tool_result: dict[str, Any]) -> bool:
        """Stable contract required by the grader.

        RE_OBSERVE is included because re-observing is the first half of
        reconsidering a decision: both mean "do not proceed with the action as
        planned". The controller uses :meth:`classify` for the finer distinction.
        """
        return self.classify(tool_result) in {REPLAN, RE_OBSERVE, CONTRADICTED}


class LoopGuard:
    def __init__(self, max_same_action_repeats: int = 2) -> None:
        self.max_same_action_repeats = max_same_action_repeats
        self._counts: dict[str, int] = {}
        # counts how many times the world has moved during this run
        self._world_epoch = 0

    def note_world_change(self) -> None:
        """Tell the guard that the world changed, so observations count again."""
        self._world_epoch += 1

    def record(self, action_name: str, arguments: dict[str, Any]) -> bool:
        """Return True when the exact same action has repeated too often."""
        # normalize the arguments to a string representation for counting
        normalized_args = json.dumps(arguments, sort_keys=True)
        key = f"{action_name}:{normalized_args}"
        # An action that changes the world is counted for the whole run, because
        # repeating a remediation that did not work is a real loop. An observation
        # only reports what is true now, so once the world has moved the same
        # observation is a new question and gets a fresh count.
        if action_name not in MUTATING_TOOLS:
            key = f"{key}@{self._world_epoch}"
        self._counts[key] = self._counts.get(key, 0) + 1
        return self._counts[key] > self.max_same_action_repeats
