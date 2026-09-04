#!/usr/bin/env python3
"""Run a bounded, candidate-only native probe for the Power Grid queue.

The regular Protocol-2.1 replay is deliberately not replaced by this command.
It advances each queued scenario with a no-op action for a small bounded number
of native ticks, then asks the backend for direct source-consumption evidence.
This makes the runtime question explicit (and reproducible) before the costly
full replay.  The command writes only a non-release ledger; it never changes a
Core suite or a release manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.implementation_identity import implementation_identity  # noqa: E402
from core.pomdp import Action  # noqa: E402
from domains.power_grid.adapter import PowerGridEnvironment  # noqa: E402

_OPTIONAL_RUNTIME_UNAVAILABLE = {
    "EgretAcopfUnavailable",
    "Grid2OpUnavailable",
    "LogisticsBackendUnavailable",
    "MicrogridBackendUnavailable",
    "OpenDssRuntimeUnavailable",
    "RcrsSidecarUnavailable",
    "SumoSidecarUnavailable",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML object required: {path}")
    return value


def _resolve(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else ROOT / path


def _suite_path(item: dict[str, Any]) -> Path:
    command = item.get("command") or []
    if not isinstance(command, list):
        raise ValueError("queue item command must be a list")
    try:
        return Path(str(command[command.index("--source-suite") + 1])).resolve()
    except (ValueError, IndexError) as exc:
        raise ValueError("queue item has no --source-suite binding") from exc


def _runtime_reason(error: BaseException) -> str:
    if type(error).__name__ in _OPTIONAL_RUNTIME_UNAVAILABLE:
        return "runtime_unavailable"
    return "runtime_exception"


def classify_probe(
    *,
    evidence: dict[str, Any] | None,
    trace: dict[str, Any] | None,
    deterministic: bool,
    error: BaseException | None,
) -> dict[str, Any]:
    """Classify one probe with an explicit non-release disposition.

    A native probe pass is *not* Core admission: it remains held until the
    source, task, depth, headroom, and Protocol-2.1 replay gates complete.
    """

    if error is not None:
        reason = _runtime_reason(error)
        return {
            "status": "held_runtime",
            "disposition": "held_runtime",
            "reason_codes": [reason, type(error).__name__],
        }
    if not deterministic:
        return {
            "status": "held_runtime",
            "disposition": "held_runtime",
            "reason_codes": ["nondeterministic_native_trace"],
        }

    evidence = evidence or {}
    trace = trace or {}
    evidence_passed = evidence.get("status") == "passed"
    trace_passed = (
        trace.get("status") == "passed"
        and trace.get("source_state_effect_observed") is True
    )
    if evidence_passed and trace_passed:
        return {
            "status": "native_prefilter_passed",
            "disposition": "held_repair",
            "reason_codes": ["native_source_effect_observed", "full_protocol21_pending"],
        }

    blockers = [
        str(value)
        for value in [*(evidence.get("blockers") or []), *(trace.get("blockers") or [])]
        if value
    ]
    if not blockers:
        blockers = ["native_source_effect_unproven"]
    return {
        "status": "held_repair",
        "disposition": "held_repair",
        "reason_codes": sorted(set(blockers)),
    }


def _run_once(scenario: dict[str, Any], *, max_ticks: int) -> tuple[dict[str, Any], dict[str, Any]]:
    env = PowerGridEnvironment()
    try:
        seed = int(scenario.get("seed") or 0)
        env.reset(scenario, seed)
        ticks = 0
        while ticks < max_ticks and env.tick < env.horizon:
            result = env.step(Action())
            ticks += 1
            if result.done:
                break
        backend = getattr(env, "_backend", None)
        trace_fn = getattr(backend, "protocol21_source_trace", None)
        if not callable(trace_fn):
            raise RuntimeError("backend_protocol21_source_trace_unimplemented")
        trace = trace_fn()
        if not isinstance(trace, dict):
            raise TypeError("backend source trace must be a mapping")
        evidence = env.source_consumption_evidence(scenario=scenario)
        if not isinstance(evidence, dict):
            raise TypeError("source consumption evidence must be a mapping")
        return trace, evidence
    finally:
        env.close()


def _probe_row(row: dict[str, Any], *, max_ticks: int) -> dict[str, Any]:
    scenario_id = str(row.get("scenario_id") or "")
    scenario_signature = str(row.get("scenario_signature") or "")
    scenario_path = _resolve(str(row.get("path") or ""))
    result: dict[str, Any] = {
        "scenario_id": scenario_id,
        "scenario_signature": scenario_signature,
        "path": str(row.get("path") or ""),
        "backend_kind": str(row.get("backend_kind") or ""),
        "source_denominator_key": str(row.get("source_denominator_key") or ""),
        "physical_source_key": str(row.get("physical_source_key") or ""),
        "max_ticks": max_ticks,
        "simulator_calls": 0,
    }
    try:
        scenario = _load_yaml(scenario_path)
        first_trace, first_evidence = _run_once(scenario, max_ticks=max_ticks)
        second_trace, second_evidence = _run_once(scenario, max_ticks=max_ticks)
        deterministic = _digest(first_trace) == _digest(second_trace) and _digest(
            first_evidence
        ) == _digest(second_evidence)
        classification = classify_probe(
            evidence=first_evidence,
            trace=first_trace,
            deterministic=deterministic,
            error=None,
        )
        result.update(
            {
                **classification,
                "deterministic": deterministic,
                "trace_sha256": _digest(first_trace),
                "evidence_sha256": _digest(first_evidence),
                "source_consumption_ticks": list(first_trace.get("consumption_ticks") or []),
                "source_state_effect_observed": first_trace.get(
                    "source_state_effect_observed"
                )
                is True,
                "evidence_blockers": list(first_evidence.get("blockers") or []),
                "trace_blockers": list(first_trace.get("blockers") or []),
                "simulator_calls": 2,
            }
        )
    except Exception as exc:  # noqa: BLE001 - ledger must retain every row
        result.update(
            {
                **classify_probe(
                    evidence=None,
                    trace=None,
                    deterministic=False,
                    error=exc,
                ),
                "deterministic": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    return result


def probe_queue(
    *,
    queue_path: Path,
    output_path: Path,
    max_ticks: int = 4,
    identity_fn: Any | None = None,
) -> dict[str, Any]:
    """Probe every row in a queue and write a candidate-only ledger."""

    if max_ticks < 1:
        raise ValueError("max_ticks must be positive")
    queue = _load_json(queue_path)
    if queue.get("candidate_only") is not True or queue.get("release_admission") is not False:
        raise ValueError("Power Grid queue must be candidate-only and non-release")
    items = queue.get("items")
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise ValueError("queue items must be a list of mappings")

    candidate_suite_binding = (queue.get("input_bindings") or {}).get("candidate_suite")
    if not isinstance(candidate_suite_binding, dict):
        raise ValueError("queue candidate_suite input binding is required")
    candidate_suite_path = Path(str(candidate_suite_binding.get("path") or ""))
    if not candidate_suite_path.is_file():
        raise ValueError("queue candidate_suite path is unavailable")
    candidate_suite_sha256 = _sha256(candidate_suite_path)
    if candidate_suite_binding.get("sha256") != candidate_suite_sha256:
        raise ValueError("queue candidate_suite hash mismatch")

    if identity_fn is None:
        def identity_fn() -> dict[str, Any]:
            return implementation_identity(ROOT)

    implementation_start = dict(identity_fn())

    results: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda value: str(value.get("work_id") or "")):
        shard_record: dict[str, Any] = {
            "work_id": str(item.get("work_id") or ""),
            "stage": str(item.get("stage") or ""),
            "suite_sha256": str((item.get("metadata") or {}).get("suite_sha256") or ""),
            "n_expected": int((item.get("metadata") or {}).get("n_scenarios") or 0),
        }
        try:
            suite_path = _suite_path(item)
            shard_record["suite_path"] = str(suite_path)
            actual_sha = _sha256(suite_path)
            shard_record["actual_suite_sha256"] = actual_sha
            if shard_record["suite_sha256"] != actual_sha:
                raise ValueError("queue_suite_hash_mismatch")
            suite = _load_json(suite_path)
            rows = suite.get("scenarios")
            if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
                raise ValueError("candidate suite rows must be mappings")
            if len(rows) != shard_record["n_expected"]:
                raise ValueError("queue_suite_count_mismatch")
            shard_results = [
                _probe_row(row, max_ticks=max_ticks)
                for row in rows
            ]
        except Exception as exc:  # noqa: BLE001 - preserve shard failure
            shard_results = [
                {
                    "scenario_id": str(item.get("scenario_id") or ""),
                    "scenario_signature": str(item.get("scenario_signature") or ""),
                    "backend_kind": str(item.get("backend") or ""),
                    **classify_probe(
                        evidence=None,
                        trace=None,
                        deterministic=False,
                        error=exc,
                    ),
                    "deterministic": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "simulator_calls": 0,
                }
            ]
        results.extend(shard_results)
        shard_record.update(
            {
                "n_completed": len(shard_results),
                "status_counts": dict(
                    sorted(Counter(str(row.get("status") or "") for row in shard_results).items())
                ),
                "disposition_counts": dict(
                    sorted(
                        Counter(str(row.get("disposition") or "") for row in shard_results).items()
                    )
                ),
            }
        )
        shards.append(shard_record)

    implementation_end = dict(identity_fn())
    implementation_stable = (
        implementation_start.get("implementation_tree_sha256")
        == implementation_end.get("implementation_tree_sha256")
    )
    if not implementation_stable:
        for row in results:
            row["disposition"] = "held_stale_evidence"
            row["status"] = "held_stale_evidence"
            row["reason_codes"] = sorted(
                set(row.get("reason_codes") or []) | {"implementation_tree_drift"}
            )
            row["simulator_calls"] = 0
    status_counts = Counter(str(row.get("status") or "") for row in results)
    disposition_counts = Counter(str(row.get("disposition") or "") for row in results)
    report = {
        "schema_version": "powergrid-native-prefilter-ledger-v1",
        "status": "candidate_native_prefilter_complete",
        "candidate_only": True,
        "release_admission": False,
        "queue_binding": {
            "path": str(queue_path.resolve()),
            "sha256": _sha256(queue_path),
        },
        "source_suite_sha256": candidate_suite_sha256,
        "source_suite_binding": {
            "path": str(candidate_suite_path.resolve()),
            "sha256": candidate_suite_sha256,
        },
        "runtime_version": platform.python_version(),
        "runtime": {
            "python": platform.python_version(),
            "max_noop_ticks": max_ticks,
            "probe_repeats": 2,
            "action": "empty_action",
        },
        "policy": {
            "frozen_core_untouched": True,
            "model_outcomes_used_for_filtering": False,
            "native_prefilter_before_full_protocol21": True,
            "native_prefilter_pass_is_not_core_admission": True,
            "every_input_has_one_terminal_row": len(results)
            == sum(int(shard.get("n_completed") or 0) for shard in shards),
            "implementation_drift_fails_closed": True,
        },
        "n_expected": sum(int(shard.get("n_expected") or 0) for shard in shards),
        "n_completed": len(results),
        "status_counts": dict(sorted(status_counts.items())),
        "disposition_counts": dict(sorted(disposition_counts.items())),
        "shards": shards,
        "results": results,
        "rows": results,
        "implementation_identity_start": implementation_start,
        "implementation_identity_end": implementation_end,
        "implementation_tree_sha256": implementation_end.get("implementation_tree_sha256"),
        "implementation_tree_stable": implementation_stable,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-ticks", type=int, default=4)
    args = parser.parse_args(argv)
    report = probe_queue(
        queue_path=args.queue.resolve(),
        output_path=args.output.resolve(),
        max_ticks=args.max_ticks,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "n_expected": report["n_expected"],
                "n_completed": report["n_completed"],
                "status_counts": report["status_counts"],
                "disposition_counts": report["disposition_counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
