#!/usr/bin/env python3
"""Build the Protocol-2.1 agentic Core contract from bound replay reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.agentic_core_contract import (  # noqa: E402
    artifact_binding,
    build_agentic_contract_report,
)


def build_report_from_paths(
    *,
    source_suite: Path,
    behavioral: Path,
    task_contracts: Path,
    complexity: Path,
    observed_depth: Path,
    strategy_depth: Path,
    source_grounded: Path,
    source_consumption: Path,
) -> dict[str, Any]:
    paths = {
        "source_suite": source_suite,
        "behavioral": behavioral,
        "task_contracts": task_contracts,
        "complexity": complexity,
        "observed_depth": observed_depth,
        "strategy_depth": strategy_depth,
        "source_grounded": source_grounded,
        "source_consumption": source_consumption,
    }
    payloads = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    return build_agentic_contract_report(
        **payloads,
        input_bindings={
            name: artifact_binding(path) for name, path in paths.items()
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-suite", type=Path, required=True)
    parser.add_argument("--behavioral", type=Path, required=True)
    parser.add_argument("--task-contracts", type=Path, required=True)
    parser.add_argument("--complexity", type=Path, required=True)
    parser.add_argument("--observed-depth", type=Path, required=True)
    parser.add_argument("--strategy-depth", type=Path, required=True)
    parser.add_argument("--source-grounded", type=Path, required=True)
    parser.add_argument("--source-consumption", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_report_from_paths(
        source_suite=args.source_suite,
        behavioral=args.behavioral,
        task_contracts=args.task_contracts,
        complexity=args.complexity,
        observed_depth=args.observed_depth,
        strategy_depth=args.strategy_depth,
        source_grounded=args.source_grounded,
        source_consumption=args.source_consumption,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "n_expected",
                    "n_completed",
                    "n_passed",
                    "n_held",
                    "n_retired",
                )
            },
            indent=2,
        )
    )
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
