#!/usr/bin/env python3
"""Static production preflight for the protocol-2.1 working set."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.implementation_identity import implementation_identity  # noqa: E402
from core.protocol21_evidence import artifact_binding, report_rows  # noqa: E402
from core.source_asset_contract import (  # noqa: E402
    resolve_source_asset_contract,
)
from core.suite_identity import verify_scenario_row_against_yaml  # noqa: E402
from domains.registry import (  # noqa: E402
    get_backend_capability,
    get_domain_spec,
    resolve_backend_source_contract_builder,
    resolve_backend_source_evidence_adapter,
)
from evaluation.task_completion import task_completion_contract  # noqa: E402

_OPTIONAL_RUNTIME_UNAVAILABLE_TYPES = frozenset(
    {
        "EgretAcopfUnavailable",
        "Grid2OpUnavailable",
        "LogisticsBackendUnavailable",
        "MicrogridBackendUnavailable",
        "OpenDssRuntimeUnavailable",
        "RcrsSidecarUnavailable",
        "SumoSidecarUnavailable",
    }
)
_OPTIONAL_RUNTIME_MODULE_ROOTS = frozenset(
    {
        "citylearn",
        "commonroad",
        "dsbx",
        "dss",
        "egret",
        "grid2op",
        "libsumo",
        "matpowercaseframes",
        "or_gym",
        "ortools",
        "pandapower",
        "pymgrid",
        "pyomo",
        "pyvrp",
        "simbench",
        "sumolib",
        "traci",
        "vrplib",
    }
)


def _is_optional_runtime_unavailable(exc: BaseException) -> bool:
    """Keep an absent optional simulator runtime pending, not statically fatal.

    A source adapter exercise is a runtime probe.  The backend-specific
    ``*Unavailable`` exceptions mean that the source contract is still
    structurally valid but cannot be executed on this host; later behavioral
    and readiness gates must remain responsible for blocking release.  Other
    exceptions remain fatal so real adapter bugs cannot hide behind a skip.
    """
    return any(
        cls.__name__ in _OPTIONAL_RUNTIME_UNAVAILABLE_TYPES
        for cls in type(exc).mro()
    )


def _environment_closure_blocker(exc: BaseException) -> str | None:
    if _is_optional_runtime_unavailable(exc):
        return "optional_runtime_unavailable"
    if isinstance(exc, ModuleNotFoundError):
        missing_root = str(exc.name or "").partition(".")[0]
        if missing_root in _OPTIONAL_RUNTIME_MODULE_ROOTS:
            return "optional_runtime_dependency_missing"
    detail = str(exc).casefold()
    if (
        "runtime_version_lock_mismatch" in detail
        or "runtime_version_mismatch" in detail
    ):
        return "runtime_version_mismatch"
    return None


def _source_hashes(
    scenario: dict[str, Any],
    *,
    repo_root: Path,
) -> tuple[dict[str, str], list[str], list[str]]:
    contract = resolve_source_asset_contract(scenario, repo_root=repo_root)
    return (
        dict(contract.locked_source_hashes),
        list(contract.missing_required_files),
        list(contract.contract_errors),
    )


def build_preflight(
    *,
    suite: dict[str, Any],
    suite_path: Path,
    expected_count: int,
    require_source_consumption_adapters: bool,
    require_formal_core_backends: bool = False,
    exercise_source_adapters: bool = False,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    rows = report_rows(suite)
    tree = implementation_identity(repo_root)
    identities = [
        (
            str(row.get("scenario_id") or ""),
            str(row.get("scenario_signature") or ""),
        )
        for row in rows
    ]
    multiplicity = Counter(identities)
    results: list[dict[str, Any]] = []
    adapter_coverage: Counter[str] = Counter()
    fidelity_coverage: Counter[str] = Counter()
    source_adapter_coverage: dict[str, dict[str, Any]] = {}

    for row in rows:
        fatal: list[str] = []
        environment: list[str] = []
        runtime_pending = [
            "behavioral_replay_pending",
            "source_consumption_replay_pending",
            "task_contract_replay_pending",
            "complexity_replay_pending",
            "depth_replay_pending",
        ]
        backend_fidelity: dict[str, Any] = {}
        identity = (
            str(row.get("scenario_id") or ""),
            str(row.get("scenario_signature") or ""),
        )
        if not all(identity) or multiplicity[identity] != 1:
            fatal.append("source_row_identity_not_unique")
        raw_path = str(row.get("path") or "")
        path = Path(raw_path)
        scenario_path = path if path.is_absolute() else repo_root / path
        scenario: dict[str, Any] = {}
        if not scenario_path.is_file():
            fatal.append("missing_scenario_yaml")
        else:
            try:
                loaded = yaml.safe_load(
                    scenario_path.read_text(encoding="utf-8")
                )
                if not isinstance(loaded, dict):
                    raise ValueError("not a mapping")
                scenario = loaded
            except Exception:
                fatal.append("scenario_yaml_unparseable")
        if scenario:
            fatal.extend(
                verify_scenario_row_against_yaml(row, path=scenario_path)
            )
            try:
                seed = int(scenario.get("seed"))
                horizon = int(scenario.get("horizon_ticks"))
                if seed < 0 or horizon <= 1:
                    raise ValueError
            except (TypeError, ValueError):
                fatal.append("seed_or_horizon_invalid")
            _hashes, missing, contract_errors = _source_hashes(
                scenario,
                repo_root=repo_root,
            )
            if missing:
                environment.append("required_source_file_missing")
            if contract_errors:
                fatal.append("source_asset_contract_invalid")
            try:
                capability = get_backend_capability(
                    scenario.get("backend_kind")
                )
                adapter_coverage[capability.backend_kind] += 1
                if not capability.fidelity_contract:
                    fatal.append("backend_fidelity_not_declared")
                else:
                    fidelity_coverage[capability.backend_kind] += 1
                if (
                    require_source_consumption_adapters
                    and not getattr(
                        capability,
                        "source_evidence_adapter",
                        getattr(capability, "source_consumption_adapter", None),
                    )
                ):
                    fatal.append("missing_source_consumption_adapter")
                try:
                    resolve_backend_source_contract_builder(capability)
                except (ImportError, AttributeError, TypeError, ValueError) as exc:
                    fatal.append("source_contract_builder_unimplemented")
                    source_adapter_coverage.setdefault(
                        capability.backend_kind,
                        {},
                    )["source_contract_builder_detail"] = str(exc)
                try:
                    resolve_backend_source_evidence_adapter(capability)
                    adapter_callable = True
                except (ImportError, AttributeError, TypeError, ValueError) as exc:
                    adapter_callable = False
                    if require_source_consumption_adapters:
                        fatal.append("source_consumption_adapter_unimplemented")
                    source_adapter_coverage.setdefault(
                        capability.backend_kind,
                        {
                            "registered": bool(
                                getattr(
                                    capability,
                                    "source_evidence_adapter",
                                    getattr(
                                        capability,
                                        "source_consumption_adapter",
                                        None,
                                    ),
                                )
                            ),
                            "callable": False,
                            "exercised": False,
                            "exercise_status": "unimplemented",
                            "detail": str(exc),
                        },
                    )
                if require_formal_core_backends and not capability.formal_core_allowed:
                    fatal.append("backend_formal_fidelity_not_allowed")
                if not capability.observation_tools:
                    fatal.append("native_observation_surface_empty")
                if not capability.control_tools:
                    fatal.append("native_control_surface_empty")
                backend_fidelity = {
                    "runtime_fidelity": capability.runtime_fidelity,
                    "formal_core_allowed": capability.formal_core_allowed,
                    "fidelity_contract": capability.fidelity_contract,
                        "source_consumption_mode": getattr(
                            capability, "source_consumption_mode", None
                        ),
                        "decision_cadence_mode": getattr(
                            capability, "decision_cadence_mode", None
                        ),
                }
                coverage = source_adapter_coverage.setdefault(
                    capability.backend_kind,
                    {
                        "registered": bool(
                            getattr(
                                capability,
                                "source_evidence_adapter",
                                getattr(
                                    capability,
                                    "source_consumption_adapter",
                                    None,
                                ),
                            )
                        ),
                        "callable": adapter_callable,
                        "exercised": False,
                        "exercise_status": "not_requested",
                    },
                )
                if (
                    exercise_source_adapters
                    and adapter_callable
                    and not coverage["exercised"]
                ):
                    env = None
                    try:
                        env = get_domain_spec(
                            str(scenario.get("domain") or "")
                        ).env_factory()()
                        env.reset(scenario, int(scenario.get("seed") or 0))
                        evidence = env.source_consumption_evidence(
                            scenario=scenario
                        )
                        if not isinstance(evidence, dict):
                            raise TypeError("adapter returned non-mapping evidence")
                        coverage.update(
                            {
                                "exercised": True,
                                "exercise_status": (
                                    "runtime_pending"
                                    if evidence.get("status") == "held"
                                    else str(
                                        evidence.get("status") or "unknown"
                                    )
                                ),
                                "evidence_blockers": list(
                                    evidence.get("blockers") or []
                                ),
                            }
                        )
                    except Exception as exc:
                        environment_blocker = _environment_closure_blocker(exc)
                        coverage.update(
                            {
                                "exercised": True,
                                "exercise_status": (
                                    "runtime_pending"
                                    if environment_blocker is not None
                                    else "exception"
                                ),
                                "detail": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        if environment_blocker is not None:
                            if _is_optional_runtime_unavailable(exc):
                                coverage["runtime_unavailable"] = True
                            environment.append(environment_blocker)
                        else:
                            fatal.append("source_adapter_exercise_failed")
                    finally:
                        if env is not None:
                            env.close()
            except KeyError:
                fatal.append("missing_backend_registry")
            perturbations = [
                item
                for item in scenario.get("perturbations") or []
                if isinstance(item, dict)
            ]
            event_ticks = [
                int(item["trigger_tick"])
                for item in perturbations
                if isinstance(item.get("trigger_tick"), int)
            ]
            if event_ticks and not any(
                0 < tick < int(scenario.get("horizon_ticks") or 0)
                for tick in event_ticks
            ):
                fatal.append("all_declared_events_unreachable")
            try:
                contract = task_completion_contract(
                    str(scenario.get("domain") or ""),
                    str(scenario.get("family") or ""),
                )
                if not contract:
                    fatal.append("task_contract_not_declared")
            except Exception:
                fatal.append("task_contract_not_declared")
        results.append(
            {
                "scenario_id": identity[0],
                "scenario_signature": identity[1],
                "domain": row.get("domain"),
                "backend_kind": row.get("backend_kind"),
                "path": raw_path,
                "fatal_blockers": sorted(set(fatal)),
                "environment_blockers": sorted(set(environment)),
                "runtime_pending_checks": runtime_pending,
                "backend_fidelity_contract": backend_fidelity,
                "status": (
                    "fatal"
                    if fatal
                    else "environment_pending"
                    if environment
                    else "runtime_pending"
                ),
            }
        )

    if len(rows) != expected_count:
        results.append(
            {
                "scenario_id": "__suite__",
                "scenario_signature": "",
                "fatal_blockers": ["working_set_count_mismatch"],
                "environment_blockers": [],
                "runtime_pending_checks": [],
                "status": "fatal",
            }
        )
    n_fatal = sum(bool(row["fatal_blockers"]) for row in results)
    n_environment_pending = sum(
        bool(row["environment_blockers"]) and not row["fatal_blockers"]
        for row in results
    )
    n_runtime_pending = sum(
        bool(row["runtime_pending_checks"])
        and not row["fatal_blockers"]
        and not row["environment_blockers"]
        for row in results
    )
    return {
        "schema_version": "1.0",
        "status": (
            "passed"
            if n_fatal == 0 and n_environment_pending == 0
            else "blocked"
        ),
        "n_expected": expected_count,
        "n_completed": len(rows),
        "n_fatal": n_fatal,
        "n_environment_pending": n_environment_pending,
        "n_runtime_pending": n_runtime_pending,
        "n_formal_disallowed": sum(
            not get_backend_capability(
                str(row.get("backend_kind") or "")
            ).formal_core_allowed
            for row in rows
            if str(row.get("backend_kind") or "")
            in {
                *adapter_coverage,
            }
        ),
        "by_domain": dict(
            sorted(Counter(str(row.get("domain") or "") for row in rows).items())
        ),
        "by_backend": dict(
            sorted(
                Counter(str(row.get("backend_kind") or "") for row in rows).items()
            )
        ),
        "backend_adapter_coverage": {
            key: value > 0
            for key, value in sorted(adapter_coverage.items())
        },
        "backend_fidelity_coverage": {
            key: value > 0
            for key, value in sorted(fidelity_coverage.items())
        },
        "source_adapter_coverage": dict(sorted(source_adapter_coverage.items())),
        "results": results,
        "input_bindings": {
            "source_suite": artifact_binding(
                suite_path,
                implementation_tree_sha256=tree[
                    "implementation_tree_sha256"
                ],
            )
        },
        **tree,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument(
        "--require-source-consumption-adapters",
        action="store_true",
    )
    parser.add_argument("--require-formal-core-backends", action="store_true")
    parser.add_argument("--exercise-source-adapters", action="store_true")
    args = parser.parse_args()
    suite_path = args.source_suite.resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    expected_count = (
        args.expected_count
        if args.expected_count is not None
        else len(report_rows(suite))
    )
    report = build_preflight(
        suite=suite,
        suite_path=suite_path,
        expected_count=expected_count,
        require_source_consumption_adapters=(
            args.require_source_consumption_adapters
        ),
        require_formal_core_backends=args.require_formal_core_backends,
        exercise_source_adapters=args.exercise_source_adapters,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "n_expected",
                    "n_completed",
                    "n_fatal",
                    "n_environment_pending",
                    "n_runtime_pending",
                )
            },
            indent=2,
        )
    )
    return 0 if report["status"] == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
