from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path


#: Used only if configs/services.json cannot be read, so a missing config degrades
#: to a still-working agent rather than an empty survey.
_FALLBACK_SERVICES = [
    "edge-gateway", "auth-service", "catalog-service", "cart-service",
    "checkout-service", "payment-service", "inventory-service", "order-service",
    "redis-cache", "order-db",
]


def load_services(path: str | Path = "configs/services.json") -> list[str]:
    """Service names to survey, ordered so the customer-facing path is read first.

    Reading the platform from configuration rather than hard-coding it means a hidden
    scenario with a different service list is surveyed correctly.
    """
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return list(_FALLBACK_SERVICES)
    services = [str(s) for s in raw.get("services", [])] or list(_FALLBACK_SERVICES)
    critical = [str(s) for s in raw.get("critical_path", []) if s in services]
    return critical + [s for s in services if s not in critical]


@dataclass(frozen=True)
class Limits:
    """Resource limits for one agent run, loaded from configs/limits.json.

    Nothing in the runtime hard-codes these numbers. Lowering a limit in the config
    file changes the agent's behaviour -- budget warnings, loop thresholds and retry
    bounds all follow it -- without any code change.
    """

    max_llm_calls: int = 14
    max_tool_calls: int = 28
    max_consecutive_model_retries: int = 3
    max_same_action_repeats: int = 2
    max_runtime_seconds: int = 120
    warning_llm_calls_remaining: int = 3

    @classmethod
    def load(cls, path: str | Path = "configs/limits.json") -> "Limits":
        """Load limits, ignoring unknown keys so a richer config file still works."""
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})
