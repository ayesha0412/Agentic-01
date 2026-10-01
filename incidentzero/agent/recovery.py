from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from incidentzero.model.errors import TransientModelError

T = TypeVar("T")


class RetryPolicy:
    def __init__(self, max_attempts: int = 3, sleeper: Callable[[float], None] = time.sleep) -> None:
        self.max_attempts = max_attempts
        self.sleeper = sleeper

    def call_model(self, fn: Callable[[], T]) -> T:
        # TODO(A1): bounded retry with backoff for TransientModelError only.
        # Do not retry PermanentModelError, schema mistakes, or an unsafe tool action.
        return fn()
