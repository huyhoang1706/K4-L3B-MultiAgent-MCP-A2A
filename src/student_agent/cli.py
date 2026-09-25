from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from .cases import load_case_set
from .config import Settings
from .contracts import Contracts
from .mcp_gateway import connect_gateway
from .submission import package_submission, validate_artifacts
from .trace import TraceWriter
from .workflow import solve_case


def _root(value: str) -> Path:
    return Path(value).resolve()


async def _show_tools(root: Path) -> None:
    settings = Settings.load(root)
    contracts = Contracts(root / "contracts" / "schemas")
    async with connect_gateway(settings.mcp_endpoint, settings.team_api_key, contracts) as gateway:
        for tool in await gateway.list_tools():
            print(tool)


async def _run(root: Path, concurrency: int = 4, only: list[str] | None = None) -> None:
    settings = Settings.load(root)
    case_set = load_case_set(root)
    contracts = Contracts(root / "contracts" / "schemas")
    output_root = root / "outputs"
    trace_path = root / "traces" / "trace.jsonl"
    output_root.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    for stale in output_root.glob("*.json"):
        stale.unlink()
    trace_path.unlink(missing_ok=True)
    trace = TraceWriter(trace_path, contracts)
    case_ids = [case_id for case_id in case_set.case_ids if not only or case_id in only]
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async with connect_gateway(settings.mcp_endpoint, settings.team_api_key, contracts) as gateway:
        discovered_tools = await gateway.list_tools()
        if not discovered_tools:
            raise RuntimeError("MCP Gateway returned no tools")

        async def run_case(case_id: str) -> None:
            async with semaphore:
                case = case_set.cases[case_id]
                trace.emit(case_id=case_id, event_type="case_received", actor="coordinator")
                output = await solve_case(case, gateway, trace)
                contracts.validate_output(output, f"outputs/{case_id}.json")
                if output.get("case_id") != case_id:
                    raise ValueError(f"solver returned a mismatched case_id for {case_id}")
                target = output_root / f"{case_id}.json"
                temporary = target.with_suffix(".json.tmp")
                temporary.write_text(
                    json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                temporary.replace(target)
                trace.emit(case_id=case_id, event_type="case_finalized", actor="coordinator")
                print(f"done {case_id}: {output['assessment']['primary_issue']}", flush=True)

        results = await asyncio.gather(*(run_case(c) for c in case_ids), return_exceptions=True)
        failures = [
            f"{case_id}: {result}"
            for case_id, result in zip(case_ids, results, strict=True)
            if isinstance(result, BaseException)
        ]
        if failures:
            raise RuntimeError(f"{len(failures)} case(s) failed: " + "; ".join(failures[:5]))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Day09 L3B student workflow")
    result.add_argument("--root", default=".", help="repository root (default: current directory)")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("validate-inputs", help="validate case-set.json and all 100 inputs")
    commands.add_parser("mcp-tools", help="authenticate and list discovered MCP tools")
    run = commands.add_parser("run", help="run the implemented workflow for all cases")
    run.add_argument("--concurrency", type=int, default=4, help="cases processed in parallel")
    run.add_argument("--only", nargs="+", metavar="CASE_ID", help="debug: run only these cases")
    run.add_argument(
        "--call-plan",
        choices=["lean", "full"],
        default=None,
        help="full: 5-6 calls, all required evidence groups (default); lean: <= 2 calls per case",
    )
    commands.add_parser("validate", help="validate outputs and observable trace")
    package = commands.add_parser("package", help="validate and build the submission ZIP")
    package.add_argument("--output", default="dist/submission.zip")
    return result


def main() -> None:
    args = parser().parse_args()
    root = _root(args.root)
    try:
        if args.command == "validate-inputs":
            case_set = load_case_set(root)
            print(
                f"OK: {case_set.variant_id} / {case_set.version} / {len(case_set.case_ids)} cases"
            )
        elif args.command == "mcp-tools":
            asyncio.run(_show_tools(root))
        elif args.command == "run":
            if args.call_plan:
                os.environ["DAY09_CALL_PLAN"] = args.call_plan
            try:
                asyncio.run(_run(root, args.concurrency, args.only))
            except BaseExceptionGroup as group:
                # mcp transport task-group failure: report instead of a raw traceback.
                done = {path.stem for path in (root / "outputs").glob("*.json")}
                missing = [c for c in load_case_set(root).case_ids if c not in done]
                raise RuntimeError(
                    f"MCP session failed ({group.exceptions[0]!r}); "
                    f"{len(missing)} case(s) without output, rerun `day09 run`"
                ) from group
        elif args.command == "validate":
            case_set = load_case_set(root)
            contracts = Contracts(root / "contracts" / "schemas")
            _, trace = validate_artifacts(root, case_set, contracts)
            print(f"OK: {len(case_set.case_ids)} outputs / {len(trace)} trace events")
        elif args.command == "package":
            destination = package_submission(root, root / args.output)
            print(f"OK: {destination}")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
