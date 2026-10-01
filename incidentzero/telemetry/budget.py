from __future__ import annotations

from dataclasses import dataclass


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class BudgetManager:
    max_llm_calls: int = 14
    max_tool_calls: int = 28
    llm_calls: int = 0
    tool_calls: int = 0

    def consume_llm(self) -> None:
        if self.llm_calls >= self.max_llm_calls:
            raise BudgetExceeded("LLM-call budget exhausted")
        self.llm_calls += 1

    def consume_tool(self) -> None:
        if self.tool_calls >= self.max_tool_calls:
            raise BudgetExceeded("Tool-call budget exhausted")
        self.tool_calls += 1

    @property
    def remaining_llm(self) -> int:
        return self.max_llm_calls - self.llm_calls

    @property
    def remaining_tools(self) -> int:
        return self.max_tool_calls - self.tool_calls
