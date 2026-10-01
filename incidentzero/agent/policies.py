from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from incidentzero.domain.models import RiskLevel


class RiskPolicy:
    def __init__(self, config_path: str | Path = "configs/risk_policy.json") -> None:
        self.mapping = json.loads(Path(config_path).read_text(encoding="utf-8"))

    def risk(self, tool_name: str) -> RiskLevel:
        return RiskLevel(self.mapping.get(tool_name, "critical"))

    def requires_human_approval(self, tool_name: str) -> bool:
        return self.risk(tool_name) in {RiskLevel.HIGH, RiskLevel.CRITICAL}


class ReplanPolicy:
    def should_replan(self, tool_result: dict[str, Any]) -> bool:
        # TODO(A1): stale state, denied approval, non-retryable action failure,
        # contradictory evidence and verified non-recovery should not all be treated the same.
        return False


class LoopGuard:
    def __init__(self, max_same_action_repeats: int = 2) -> None:
        self.max_same_action_repeats = max_same_action_repeats
        self._counts: dict[str, int] = {}

    def record(self, action_name: str, arguments: dict[str, Any]) -> bool:
        """Return True when the exact same action has repeated too often."""
        # TODO(A1): create a stable action fingerprint and detect looping.
        return False
