from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class TraceRecorder:
    def __init__(self, path: str | Path, echo: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self.echo = echo
        self._started = time.time()

    def record(self, event: str, payload: dict[str, Any]) -> None:
        row = {"ts": round(time.time(), 3), "event": event, "payload": payload}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        if self.echo:
            line = _console_line(event, payload)
            if line:
                from rich import print as rprint

                rprint(f"[dim]{time.time() - self._started:6.1f}s[/dim] {line}")


#: Events worth showing live, and how to colour them.
_STYLES = {
    "approval_requested": "bold yellow",
    "approval_result": "bold yellow",
    "validation_failure": "red",
    "loop_blocked": "red",
    "precondition_failure": "red",
    "budget_warning": "yellow",
    "stale_world": "yellow",
    "plan_revised": "magenta",
    "terminal": "bold green",
}


def _console_line(event: str, payload: dict[str, Any]) -> str | None:
    """One short human-readable line per trace event, or None to stay quiet."""
    from rich.markup import escape

    def short(value: Any, n: int = 90) -> str:
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        return escape(text[:n] + ("..." if len(text) > n else ""))

    if event == "bootstrap_incident":
        return f"[bold]INCIDENT[/bold] {short((payload.get('data') or {}).get('title'))}"
    if event == "plan_created":
        return f"[magenta]PLAN r{payload.get('revision')}[/magenta] {short(payload.get('hypothesis'))}"
    if event == "plan_revised":
        return f"[magenta]REPLAN -> r{payload.get('revision')}[/magenta] {short(payload.get('hypothesis'))}"
    if event == "model_reply":
        calls = payload.get("tool_calls") or []
        if not calls:
            return f"[yellow]model replied with text, no action[/yellow] {short(payload.get('content'), 60)}"
        return f"[cyan]->[/cyan] {calls[0]['name']} {short(calls[0].get('arguments'), 70)}"
    if event in {"tool_result", "controller_observation"}:
        result = payload.get("result") or {}
        status = result.get("status", "?")
        style = "green" if status == "ok" else "red"
        detail = result.get("data") if status == "ok" else result.get("message")
        prefix = "   " if event == "tool_result" else "   [dim](controller)[/dim] "
        return f"{prefix}[{style}]{status}[/{style}] v{result.get('world_version')} {short(detail, 80)}"
    if event == "control_decision":
        decision = payload.get("decision")
        return None if decision == "continue" else f"   [yellow]decision: {decision}[/yellow]"
    if event in _STYLES:
        return f"[{_STYLES[event]}]{event}[/{_STYLES[event]}] {short(payload, 100)}"
    if event in {"tool_retry", "safe_escalation", "wrap_up", "deadline_reached", "model_retries_exhausted"}:
        return f"[yellow]{event}[/yellow] {short(payload, 90)}"
    return None
