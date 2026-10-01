from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
from rich import print

from incidentzero.agent.controller import AgentController
from incidentzero.approval.gateway import ConsoleApprovalGateway
from incidentzero.environment.engine import SimulationEnvironment
from incidentzero.model.groq_client import GroqModelClient
from incidentzero.telemetry.budget import BudgetManager
from incidentzero.telemetry.trace import TraceRecorder
from incidentzero.tools.registry import ToolRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="incidentzero")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--student-id", required=True)
    run.add_argument("--scenario", default="public-a")
    run.add_argument("--model", default=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"))
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    if args.command == "run":
        env = SimulationEnvironment(args.student_id, args.scenario)
        registry = ToolRegistry(env)
        trace_path = Path("traces") / f"{args.student_id}_{args.scenario}.jsonl"
        controller = AgentController(
            model=GroqModelClient(model=args.model),
            tools=registry,
            approval=ConsoleApprovalGateway(),
            budget=BudgetManager(),
            trace=TraceRecorder(trace_path),
        )
        outcome = controller.run()
        print("\n[bold]Outcome[/bold]")
        print(json.dumps(asdict(outcome), indent=2, default=str))


if __name__ == "__main__":
    main()
