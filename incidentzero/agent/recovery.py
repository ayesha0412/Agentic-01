from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from incidentzero.model.errors import TransientModelError

T = TypeVar("T")


class RetryPolicy:
    """Bounded retry for transient model/provider failures only.

    A retry is appropriate when the *request* failed for reasons unrelated to the
    decision it carried: a 429, a network blip, a 5xx. Everything else -- a bad
    request, a wrong tool choice, a denied approval -- is a decision problem and
    must be handled by the controller, not by calling the same thing again.
    """

    def __init__(
        self,
        max_attempts: int = 3,
        sleeper: Callable[[float], None] = time.sleep,
        base_delay: float = 0.5,
        max_delay: float = 8.0,
    ) -> None:
        self.max_attempts = max_attempts
        self.sleeper = sleeper
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.last_attempts = 0

    def backoff_delay(self, attempt: int) -> float:
        """Capped exponential backoff: 0.5s, 1s, 2s, ... up to max_delay."""
        return min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)

    def call_model(self, fn: Callable[[], T]) -> T:
        self.last_attempts = 0
        for attempt in range(1, self.max_attempts + 1):
            self.last_attempts = attempt
            try:
                return fn()
            except TransientModelError:
                # Out of attempts: surface the failure so the controller can
                # terminate safely instead of retrying forever.
                if attempt == self.max_attempts:
                    raise
                self.sleeper(self.backoff_delay(attempt))
        # Unreachable: the loop either returns or raises.
        raise AssertionError("retry loop exited without a result")
