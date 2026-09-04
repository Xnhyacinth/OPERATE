#!/usr/bin/env python3
"""Run a diagnostic protocol-2.1 pilot over every working-set backend."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.implementation_identity import implementation_identity  # noqa: E402
from core.protocol21_evidence import artifact_binding  # noqa: E402
from core.protocol21_qualification import (  # noqa: E402
    build_row_qualification_evidence,
)
from domains.registry import get_backend_capability  # noqa: E402

DIFFICULTY_RANK = {"basic": 0, "medium": 1, "high": 2, "extreme": 3}
OUTPUT_NAMES = (
    "pilot_source_suite.json",
    "preflight.json",
    "behavioral.json",
    "source_consumption.json",
    "task_contracts.json",
    "complexity.json",
    "observed_depth.json",
    "strategy_depth.json",
    "source_grounded.json",
    "agentic_contract.json",
    "diagnostic_core.json",
    "pilot_summary.json",
)


def select_pilot_rows(
    rows: list[dict[str, Any]],
    *,
    qualification: bool = False,
    all_live_sumo: bool = False,
    all_input_scenarios: bool = False,
) -> list[dict[str, Any]]:
    if all_input_scenarios:
        return list(rows)
    if qualification:
        rows = [
            row
            for row in rows
            if get_backend_capability(
                str(row.get("backend_kind") or "")
            ).formal_core_allowed
        ]
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row.get("backend_kind") or ""),
            str(row.get("scenario_id") or ""),
        ),
    )
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for row in ordered:
        identity = (
            str(row.get("scenario_id") or ""),
            str(row.get("scenario_signature") or ""),
        )
        backend = str(row.get("backend_kind") or "")
        if not any(
            str(existing.get("backend_kind") or "") == backend
            for existing in selected.values()
        ):
            selected[identity] = row
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain[str(row.get("domain") or "")].append(row)
    for domain_rows in by_domain.values():
        row = max(
            domain_rows,
            key=lambda item: (
                DIFFICULTY_RANK.get(
                    str(item.get("difficulty_level") or ""), -1
                ),
                int(item.get("horizon_ticks") or 0),
                str(item.get("scenario_id") or ""),
            ),
        )
        selected[
            (
                str(row.get("scenario_id") or ""),
                str(row.get("scenario_signature") or ""),
            )
        ] = row
    if all_live_sumo:
        for row in rows:
            if str(row.get("backend_kind") or "") != "sumo":
                continue
            selected[
                (
                    str(row.get("scenario_id") or ""),
                    str(row.get("scenario_signature") or ""),
                )
            ] = row
    return sorted(
        selected.values(),
        key=lambda row: str(row.get("scenario_id") or ""),
    )


def _validate_selection_mode(
    *,
    qualification: bool,
    all_input_scenarios: bool,
) -> str | None:
    if qualification and all_input_scenarios:
        return "all_input_scenarios_is_diagnostic_only"
    return None


def _validate_source_suite(
    suite: dict[str, Any],
    *,
    qualification: bool,
    all_input_scenarios: bool,
) -> str | None:
    status = str(suite.get("status") or "")
    if status == "diagnostic_backend_slice":
        if qualification:
            return "diagnostic_backend_slice_is_not_qualification_input"
        if not all_input_scenarios:
            return "diagnostic_backend_slice_requires_all_input_scenarios"
        if suite.get("diagnostic_only") is not True:
            return "diagnostic_backend_slice_flag_missing"
        if suite.get("leaderboard_eligible") is not False:
            return "diagnostic_backend_slice_must_be_non_eligible"
        for row in suite.get("scenarios") or []:
            if not row.get("scenario_signature"):
                return "diagnostic_backend_slice_row_signature_missing"
            if not row.get("source_key"):
                return "diagnostic_backend_slice_source_key_missing"
            if not row.get("source_denominator_key"):
                return "diagnostic_backend_slice_source_denominator_missing"
            if not row.get("structural_fingerprint") or not row.get(
                "semantic_fingerprint"
            ):
                return "diagnostic_backend_slice_row_fingerprint_missing"
            if not row.get("case_ledger"):
                return "diagnostic_backend_slice_case_ledger_missing"
        return None
    if status != "working_set" or suite.get("leaderboard_eligible") is not False:
        return "qualification_source_must_be_completed_noneligible_working_set"
    return None


def _stage_commands(
    *,
    source: Path,
    output_dir: Path,
    workers: int,
    timeout: int,
    n_rows: int,
    qualification: bool = False,
    max_replays: int = 24,
    max_replay_work_ticks: int = 4096,
    exact_max_calls: int = 10,
    exact_max_replays: int = 512,
    per_action_cap: int = 12,
) -> list[tuple[str, list[str], Path]]:
    py = sys.executable
    paths = {name: output_dir / name for name in OUTPUT_NAMES}
    cache = output_dir / "cache"
    stages = [
        (
            "preflight",
            [
                py,
                str(REPO_ROOT / "scripts/preflight_protocol21_working_set.py"),
                "--source-suite",
                str(source),
                "--output",
                str(paths["preflight.json"]),
                "--expected-count",
                str(n_rows),
                "--require-source-consumption-adapters",
                *(
                    [
                        "--require-formal-core-backends",
                        "--exercise-source-adapters",
                    ]
                    if qualification
                    else []
                ),
            ],
            paths["preflight.json"],
        ),
        (
            "behavioral",
            [
                py,
                str(REPO_ROOT / "scripts/calibrate_core_candidate.py"),
                "--suite",
                str(source),
                "--output",
                str(paths["behavioral.json"]),
                "--workers",
                str(workers),
                "--sample-timeout-seconds",
                str(timeout),
                "--cache-dir",
                str(cache / "behavioral"),
            ],
            paths["behavioral.json"],
        ),
        (
            "source_consumption",
            [
                py,
                str(REPO_ROOT / "scripts/audit_protocol21_source_consumption.py"),
                "--suite",
                str(source),
                "--behavioral",
                str(paths["behavioral.json"]),
                "--output",
                str(paths["source_consumption.json"]),
            ],
            paths["source_consumption.json"],
        ),
        (
            "task_contracts",
            [
                py,
                str(REPO_ROOT / "scripts/calibrate_task_contracts.py"),
                "--suite",
                str(source),
                "--output",
                str(paths["task_contracts.json"]),
                "--agent",
                "oracle_offline",
                "--fallback-agents",
                "greedy_heuristic",
                "--workers",
                str(workers),
                "--sample-timeout-seconds",
                str(timeout),
            ],
            paths["task_contracts.json"],
        ),
        (
            "complexity",
            [
                py,
                str(REPO_ROOT / "scripts/calibrate_core_complexity.py"),
                "--suite",
                str(source),
                "--output",
                str(paths["complexity.json"]),
                "--agents",
                "oracle_offline",
                "greedy_heuristic",
                "wait_only",
                "--workers",
                str(workers),
                "--sample-timeout-seconds",
                str(timeout),
                "--cache-dir",
                str(cache / "complexity"),
                "--max-replays",
                str(max_replays),
                "--max-replay-work-ticks",
                str(max_replay_work_ticks),
                "--exact-max-calls",
                str(exact_max_calls),
                "--exact-max-replays",
                str(exact_max_replays),
                "--per-action-cap",
                str(per_action_cap),
            ],
            paths["complexity.json"],
        ),
        (
            "observed_depth",
            [
                py,
                str(REPO_ROOT / "scripts/audit_observed_reference_depth.py"),
                "--behavioral",
                str(paths["behavioral.json"]),
                "--task-contracts",
                str(paths["task_contracts.json"]),
                "--output",
                str(paths["observed_depth.json"]),
            ],
            paths["observed_depth.json"],
        ),
        (
            "strategy_depth",
            [
                py,
                str(REPO_ROOT / "scripts/audit_strategy_depth_calibration.py"),
                "--input",
                str(paths["complexity.json"]),
                "--output",
                str(paths["strategy_depth.json"]),
            ],
            paths["strategy_depth.json"],
        ),
        (
            "source_grounded",
            [
                py,
                str(REPO_ROOT / "scripts/audit_source_grounded_pipeline.py"),
                "--input",
                str(source),
                "--behavioral",
                str(paths["behavioral.json"]),
                "--source-consumption",
                str(paths["source_consumption.json"]),
                "--task-contracts",
                str(paths["task_contracts.json"]),
                "--complexity",
                str(paths["complexity.json"]),
                "--strategy-depth",
                str(paths["strategy_depth.json"]),
                "--require-protocol21-evidence",
                "--output",
                str(paths["source_grounded.json"]),
            ],
            paths["source_grounded.json"],
        ),
        (
            "agentic_contract",
            [
                py,
                str(REPO_ROOT / "scripts/audit_protocol21_core_contract.py"),
                "--source-suite",
                str(source),
                "--behavioral",
                str(paths["behavioral.json"]),
                "--task-contracts",
                str(paths["task_contracts.json"]),
                "--complexity",
                str(paths["complexity.json"]),
                "--observed-depth",
                str(paths["observed_depth.json"]),
                "--strategy-depth",
                str(paths["strategy_depth.json"]),
                "--source-grounded",
                str(paths["source_grounded.json"]),
                "--source-consumption",
                str(paths["source_consumption.json"]),
                "--output",
                str(paths["agentic_contract.json"]),
            ],
            paths["agentic_contract.json"],
        ),
        (
            "diagnostic_materialization",
            [
                py,
                str(REPO_ROOT / "scripts/materialize_protocol2_core.py"),
                "--source",
                str(source),
                "--behavioral",
                str(paths["behavioral.json"]),
                "--tasks",
                str(paths["task_contracts.json"]),
                "--observed-depth",
                str(paths["observed_depth.json"]),
                "--depth",
                str(paths["strategy_depth.json"]),
                "--source-gate",
                str(paths["source_grounded.json"]),
                "--agentic-contract",
                str(paths["agentic_contract.json"]),
                "--require-protocol21-gates",
                "--output",
                str(paths["diagnostic_core.json"]),
            ],
            paths["diagnostic_core.json"],
        ),
    ]
    return stages


def _status_by_backend(
    report: dict[str, Any],
    source_rows: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    source = {
        (
            str(row.get("scenario_id") or ""),
            str(row.get("scenario_signature") or ""),
        ): row
        for row in source_rows
    }
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in report.get("results") or []:
        identity = (
            str(row.get("scenario_id") or ""),
            str(row.get("scenario_signature") or ""),
        )
        backend = str(
            row.get("backend_kind")
            or source.get(identity, {}).get("backend_kind")
            or ""
        )
        counts[backend].update([str(row.get("status") or "unknown")])
    return {
        key: dict(sorted(value.items()))
        for key, value in sorted(counts.items())
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-suite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sample-timeout-seconds", type=int, default=900)
    parser.add_argument("--qualification", action="store_true")
    parser.add_argument("--all-live-sumo", action="store_true")
    parser.add_argument("--all-input-scenarios", action="store_true")
    parser.add_argument("--max-replays", type=int, default=24)
    parser.add_argument("--max-replay-work-ticks", type=int, default=4096)
    parser.add_argument("--exact-max-calls", type=int, default=10)
    parser.add_argument("--exact-max-replays", type=int, default=512)
    parser.add_argument("--per-action-cap", type=int, default=12)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    selection_error = _validate_selection_mode(
        qualification=args.qualification,
        all_input_scenarios=args.all_input_scenarios,
    )
    if selection_error:
        parser.error(selection_error)
    output_dir = args.output_dir.resolve()
    if output_dir.is_relative_to(REPO_ROOT.resolve()):
        parser.error("--output-dir must be outside the repository")
    if args.workers < 1 or args.sample_timeout_seconds <= 0:
        parser.error("workers and sample timeout must be positive")
    suite = json.loads(args.source_suite.read_text(encoding="utf-8"))
    source_suite_error = _validate_source_suite(
        suite,
        qualification=args.qualification,
        all_input_scenarios=args.all_input_scenarios,
    )
    if source_suite_error:
        print(f"ERROR: {source_suite_error}", file=sys.stderr)
        return 4
    rows = list(suite.get("scenarios") or [])
    selected = select_pilot_rows(
        rows,
        qualification=args.qualification,
        all_live_sumo=args.all_live_sumo,
        all_input_scenarios=args.all_input_scenarios,
    )
    source_path = output_dir / "pilot_source_suite.json"
    stages = _stage_commands(
        source=source_path,
        output_dir=output_dir,
        workers=args.workers,
        timeout=args.sample_timeout_seconds,
        n_rows=len(selected),
        qualification=args.qualification,
        max_replays=args.max_replays,
        max_replay_work_ticks=args.max_replay_work_ticks,
        exact_max_calls=args.exact_max_calls,
        exact_max_replays=args.exact_max_replays,
        per_action_cap=args.per_action_cap,
    )
    print(
        "PILOT_STAGE_ORDER="
        + " -> ".join(name for name, _argv, _output in stages)
    )
    for name, argv, _output in stages:
        print(f"[{name}] " + " ".join(argv))
    if not args.execute:
        print("NO_COMMANDS_EXECUTED=true")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    pilot_suite = {
        key: value for key, value in suite.items() if key != "scenarios"
    }
    pilot_suite.update(
        {
            "schema_version": "2.1-pilot",
            "status": "diagnostic_pilot_source",
            "diagnostic_only": True,
            "leaderboard_eligible": False,
            "n_selected": len(selected),
            "scenarios": selected,
        }
    )
    source_path.write_text(
        json.dumps(pilot_suite, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    stage_statuses: dict[str, dict[str, Any]] = {}
    for name, argv, expected_output in stages:
        completed = subprocess.run(argv, cwd=REPO_ROOT, check=False)
        stage_statuses[name] = {
            "return_code": completed.returncode,
            "output_exists": expected_output.is_file(),
        }
        if not expected_output.is_file():
            return completed.returncode or 3
        json.loads(expected_output.read_text(encoding="utf-8"))
        if completed.returncode != 0 and not args.qualification:
            return completed.returncode

    behavioral = json.loads(
        (output_dir / "behavioral.json").read_text(encoding="utf-8")
    )
    consumption = json.loads(
        (output_dir / "source_consumption.json").read_text(encoding="utf-8")
    )
    grounded = json.loads(
        (output_dir / "source_grounded.json").read_text(encoding="utf-8")
    )
    agentic = json.loads(
        (output_dir / "agentic_contract.json").read_text(encoding="utf-8")
    )
    diagnostic = json.loads(
        (output_dir / "diagnostic_core.json").read_text(encoding="utf-8")
    )
    blockers = Counter(
        blocker
        for row in agentic.get("results") or []
        for blocker in row.get("blockers") or []
    )
    retirements = Counter(
        blocker
        for row in agentic.get("results") or []
        if row.get("status") == "retired"
        for blocker in row.get("blockers") or []
    )
    selected_domains = {
        str(row.get("domain") or "") for row in selected
    }
    selected_backends = {
        str(row.get("backend_kind") or "") for row in selected
    }
    consumption_passed = {
        (
            str(row.get("scenario_id") or ""),
            str(row.get("scenario_signature") or ""),
        )
        for row in consumption.get("results") or []
        if row.get("status") == "passed"
    }
    selected_identities = {
        (
            str(row.get("scenario_id") or ""),
            str(row.get("scenario_signature") or ""),
        )
        for row in selected
    }
    selected_by_identity = {
        (
            str(row.get("scenario_id") or ""),
            str(row.get("scenario_signature") or ""),
        ): row
        for row in selected
    }
    live_identities = {
        (
            str(row.get("scenario_id") or ""),
            str(row.get("scenario_signature") or ""),
        )
        for row in selected
        if str(row.get("backend_kind") or "") == "sumo"
    }
    live_runtime_verified = {
        (
            str(row.get("scenario_id") or ""),
            str(row.get("scenario_signature") or ""),
        )
        for row in behavioral.get("results") or []
        if (
            row.get("backend_kind")
            or selected_by_identity.get(
                (
                    str(row.get("scenario_id") or ""),
                    str(row.get("scenario_signature") or ""),
                ),
                {},
            ).get("backend_kind")
        )
        == "sumo"
        and (row.get("checks") or {}).get("native_backend_executable") is True
    }
    grounded_domains = {
        str(row.get("domain") or "")
        for row in grounded.get("results") or []
        if str(row.get("status") or "").startswith("admitted")
    }
    agentic_domains = {
        str(row.get("domain") or "")
        for row in agentic.get("results") or []
        if row.get("status") == "passed"
    }
    diagnostic_rows = (
        diagnostic.get("scenarios")
        or diagnostic.get("results")
        or []
    )
    diagnostic_domains = {
        str(row.get("domain") or "") for row in diagnostic_rows
    }
    disallowed = {
        backend
        for backend in selected_backends
        if backend == "mock_sumo"
        or not get_backend_capability(backend).formal_core_allowed
    }
    live_sumo_verified = bool(live_identities) and (
        live_identities <= live_runtime_verified
    )
    qualification_checks = {
        "every_selected_backend_source_consumption_passed": (
            selected_identities <= consumption_passed
        ),
        "live_sumo_runtime_verified": live_sumo_verified,
        "source_grounded_domain_coverage": (
            selected_domains <= grounded_domains
        ),
        "agentic_domain_coverage": selected_domains <= agentic_domains,
        "diagnostic_core_domain_coverage": (
            selected_domains <= diagnostic_domains
        ),
        "formal_backend_fidelity_only": not disallowed,
    }
    framework_blockers = {
        "source_contract_missing",
        "source_consumption_unproven",
        "backend_source_evidence_adapter_unimplemented",
        "native_backend_execution_unproven",
        "artifact_identity_mismatch",
        "implementation_tree_mismatch",
    }
    present_framework_blockers = sorted(framework_blockers.intersection(blockers))
    qualification_checks["no_framework_blockers"] = not present_framework_blockers
    qualification_passed = bool(
        args.qualification and all(qualification_checks.values())
    )
    qualification_blockers = [
        name for name, passed in qualification_checks.items() if not passed
    ]
    scenario_quality_blockers = {
        key: value
        for key, value in sorted(blockers.items())
        if key not in framework_blockers
    }
    summary = {
        "schema_version": "1.0",
        "diagnostic_only": True,
        "qualification": bool(args.qualification),
        "qualification_budgets": {
            "max_replays": args.max_replays,
            "max_replay_work_ticks": args.max_replay_work_ticks,
            "exact_max_calls": args.exact_max_calls,
            "exact_max_replays": args.exact_max_replays,
            "per_action_cap": args.per_action_cap,
        },
        "qualification_checks": qualification_checks,
        "qualification_passed": qualification_passed,
        "qualification_status": (
            "passed" if qualification_passed else "blocked"
        ),
        "qualification_blockers": qualification_blockers,
        "live_sumo_runtime_verified": live_sumo_verified,
        "n_scenarios": len(selected),
        "selected_by_backend": dict(
            sorted(
                Counter(str(row.get("backend_kind") or "") for row in selected).items()
            )
        ),
        "selected_by_domain": dict(
            sorted(
                Counter(str(row.get("domain") or "") for row in selected).items()
            )
        ),
        "stage_statuses": stage_statuses,
        "behavioral_by_backend": _status_by_backend(behavioral, selected),
        "source_consumption_by_backend": _status_by_backend(
            consumption, selected
        ),
        "source_grounded_by_backend": _status_by_backend(grounded, selected),
        "agentic_by_backend": _status_by_backend(agentic, selected),
        "blocker_counts": dict(sorted(blockers.items())),
        "retirement_counts": dict(sorted(retirements.items())),
        "framework_blocker_counts": {
            key: blockers[key] for key in present_framework_blockers
        },
        "scenario_quality_blocker_counts": scenario_quality_blockers,
    }
    (output_dir / "pilot_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.qualification:
        grounded_passed = {
            (
                str(row.get("scenario_id") or ""),
                str(row.get("scenario_signature") or ""),
            )
            for row in grounded.get("results") or []
            if str(row.get("status") or "").startswith("admitted")
        }
        agentic_passed = {
            (
                str(row.get("scenario_id") or ""),
                str(row.get("scenario_signature") or ""),
            )
            for row in agentic.get("results") or []
            if row.get("status") == "passed"
        }
        diagnostic_identities = {
            (
                str(row.get("scenario_id") or ""),
                str(row.get("scenario_signature") or ""),
            )
            for row in diagnostic_rows
        }
        checks_by_identity = {}
        for identity, row in selected_by_identity.items():
            backend = str(row.get("backend_kind") or "")
            checks_by_identity[identity] = {
                "source_consumption": identity in consumption_passed,
                "source_grounded": identity in grounded_passed,
                "agentic_contract": identity in agentic_passed,
                "diagnostic_materialization": (
                    identity in diagnostic_identities
                ),
                "formal_backend_fidelity": (
                    backend not in disallowed
                ),
                "live_sumo_runtime": (
                    identity in live_runtime_verified
                    if backend == "sumo"
                    else True
                ),
            }
        evidence_bindings = {
            name: artifact_binding(output_dir / name)
            for name in (
                "behavioral.json",
                "source_consumption.json",
                "source_grounded.json",
                "agentic_contract.json",
                "diagnostic_core.json",
            )
        }
        qualification_rows = build_row_qualification_evidence(
            source_rows=selected,
            checks_by_identity=checks_by_identity,
            evidence_bindings=evidence_bindings,
            implementation_identity=implementation_identity(),
            suite_identity=artifact_binding(source_path),
            suite_qualification_passed=qualification_passed,
        )
        (output_dir / "qualification_rows.json").write_text(
            json.dumps(qualification_rows, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.qualification and not qualification_passed:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
