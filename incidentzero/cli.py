from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
from rich import print

from incidentzero.agent.controller import AgentController
from incidentzero.approval.gateway import (
    AlwaysApproveGateway,
    AlwaysDenyGateway,
    ConsoleApprovalGateway,
)
from incidentzero.config import Limits
from incidentzero.environment.engine import SimulationEnvironment
from incidentzero.model.groq_client import GroqModelClient
from incidentzero.telemetry.budget import BudgetManager
from incidentzero.telemetry.trace import TraceRecorder
from incidentzero.tools.registry import ToolRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="incidentzero")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    # Defaults to STUDENT_ID from the environment so a run needs no arguments,
    # while still allowing any roll number to be passed explicitly.
    run.add_argument("--student-id", default=os.getenv("STUDENT_ID"))
    run.add_argument("--scenario", default="public-a")
    run.add_argument("--model", default=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"))
    run.add_argument(
        "--quiet", action="store_true", help="suppress the live event feed"
    )
    run.add_argument(
        "--approval",
        choices=["console", "approve", "deny"],
        default="console",
        help=(
            "console prompts a human (default); approve/deny use the scripted gateways "
            "for unattended runs and for demonstrating denial handling"
        ),
    )
    return parser


GATEWAYS = {
    "console": ConsoleApprovalGateway,
    "approve": AlwaysApproveGateway,
    "deny": AlwaysDenyGateway,
}


def main() -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run":
        if not args.student_id:
            parser.error("--student-id is required, or set STUDENT_ID in .env")
        limits = Limits.load()
        env = SimulationEnvironment(args.student_id, args.scenario)
        registry = ToolRegistry(env)
        trace_path = Path("traces") / f"{args.student_id}_{args.scenario}.jsonl"
        controller = AgentController(
            model=GroqModelClient(model=args.model),
            tools=registry,
            approval=GATEWAYS[args.approval](),
            # Budgets come from configs/limits.json, not from code defaults.
            budget=BudgetManager(
                max_llm_calls=limits.max_llm_calls, max_tool_calls=limits.max_tool_calls
            ),
            trace=TraceRecorder(trace_path, echo=not args.quiet),
            limits=limits,
        )
        outcome = controller.run()
        print("\n[bold]Outcome[/bold]")
        print(json.dumps(asdict(outcome), indent=2, default=str))


if __name__ == "__main__":
    main()
