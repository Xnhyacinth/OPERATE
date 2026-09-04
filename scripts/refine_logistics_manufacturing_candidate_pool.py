#!/usr/bin/env python3
"""Close the local Logistics/Manufacturing candidate inventory.

This is a deterministic structural refinement pass, not a release writer.  It
enumerates every locally available raw source unit, collapses mirrors and
seed-only replicas, validates current backend compatibility, and assigns every
unit and independent candidate one terminal disposition.
``ready_for_full_admission`` means that the source/identity/native-control
prefilter is complete; promotion still uses the focused candidate-delta
behavioral replay required by OPERATE.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domains.logistics.seeds.from_jsplib import (  # noqa: E402
    parse_jsplib_instance,
)


DEFAULT_SOURCE_SUITE = ROOT / "release/operate_v0_61_0/protocol21_source_suite.json"
DEFAULT_JSPLIB = ROOT / "works/JSPLIB-Instances"
DEFAULT_REALM = ROOT / "works/REALM-Bench-direct-pilot/datasets/clean/JSSP/J2.json"
DEFAULT_DYNASCHED = ROOT / "works/DynaSchedBench/data"
DEFAULT_PYVRP = ROOT / "works/PyVRP-Instances"
DEFAULT_VRPLIB = ROOT / "works/VRPLIB/tests/data"
DEFAULT_M5 = ROOT / "works/M5"
DEFAULT_ORGYM = ROOT / "works/OR-Gym"
DEFAULT_OUTPUT = (
    ROOT / ".hl/artifacts/operate_v058_logistics_manufacturing_refinement.json"
)

FINAL_DISPOSITIONS = {
    "ready_for_full_admission",
    "held_repair",
    "redesign",
    "secondary",
    "rejected",
}
ACTIONABLE_DYNA_EVENTS = {
    "ARRIVAL",
    "BREAKDOWN",
    "DUE_DATE_CHANGE",
    "ORDER_CANCELLATION",
    "PREVENTIVE_MAINTENANCE",
    "PRIORITY_CHANGE",
    "PTIME_CHANGE",
    "ROUTE_CHANGE",
}
DYNASCHED_EVENT_DRIVEN_REPLAY_CONTRACT = "dynasched_native_boundary_work_ticks_v1"
DYNASCHED_EVENT_DRIVEN_MAX_WORK_TICKS = 32768
SUPPORTED_ROUTING_TYPES = {"CVRP": "pyvrp_cvrp", "VRPTW": "pyvrp_vrptw"}
ROUTING_VARIANT_SEMANTICS = {
    "DCVRP": ["maximum_route_distance"],
    "GVRP": ["client_groups", "optional_group_service"],
    "HFVRP": ["heterogeneous_vehicle_types", "vehicle_specific_cost_capacity"],
    "MDVRPTW": ["multiple_depots", "depot_specific_vehicle_types"],
    "MTVRPTWR": ["multi_trip_reload", "client_release_times"],
    "PCVRPTW": ["optional_clients", "client_prizes"],
    "SDVRPTW": ["split_delivery", "partial_quantity_state"],
    "TSP": ["single_hamiltonian_tour", "distance_only_objective"],
    "VRPB": ["linehaul_backhaul_precedence", "pickup_delivery_load_state"],
}
INTRINSIC_SECONDARY_ROUTING_VARIANTS = {
    "DCVRP": "static_distance_cap_lacks_distinct_long_horizon_operational_process",
    "TSP": "static_single_tour_lacks_distinct_long_horizon_operational_process",
}


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _active_evidence(payload: Any) -> tuple[str, set[str]]:
    strings: list[str] = []
    hashes: set[str] = set()

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, str):
            strings.append(value)
            if ("sha" in key.lower() or key.lower() == "hash") and re.fullmatch(
                r"(?:sha256:)?[0-9a-fA-F]{64}", value
            ):
                hashes.add(value.split(":")[-1].lower())
        elif isinstance(value, dict):
            for nested_key, nested in value.items():
                visit(nested, str(nested_key))
        elif isinstance(value, list):
            for nested in value:
                visit(nested, key)

    visit(payload)
    return "\n".join(strings), hashes


def _attempt(check: str, outcome: str, detail: str) -> dict[str, str]:
    return {
        "check": check,
        "outcome": outcome,
        "detail": detail,
        "phase": "proposed" if outcome == "design_required" else "executed",
    }


def _row(
    *,
    candidate_id: str,
    source_id: str,
    source_unit: str,
    domain: str,
    classification_scope: str,
    disposition: str,
    reason_codes: list[str],
    repair_attempts: list[dict[str, str]],
    evidence: dict[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    if disposition not in FINAL_DISPOSITIONS:
        raise ValueError(f"invalid disposition: {disposition}")
    if not reason_codes:
        raise ValueError(f"{candidate_id} has no reason_codes")
    body = {
        "candidate_id": candidate_id,
        "source_id": source_id,
        "source_unit": source_unit,
        "domain": domain,
        "classification_scope": classification_scope,
        "final_disposition": disposition,
        # Compatibility views used by the focused report tests and older tools.
        "disposition": disposition,
        "reason_codes": sorted(set(reason_codes)),
        "repair_attempts": repair_attempts,
        "evidence": evidence,
    }
    body.update(extra)
    return body


def _raw_and_candidate(
    *, raw: dict[str, Any], candidate: dict[str, Any] | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [raw]
    candidates: list[dict[str, Any]] = []
    if candidate is not None:
        rows.append(candidate)
        candidates.append(candidate)
    return rows, candidates


def _parse_jsplib(path: Path) -> tuple[int, int, int]:
    parsed = parse_jsplib_instance(path)
    return int(parsed["jobs"]), int(parsed["machines"]), int(parsed["operations"])


def _checksum_manifest(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    output: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
            output[parts[1]] = parts[0].lower()
    return output


def _refine_jsplib(
    root: Path, active_text: str, active_hashes: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metadata_path = root / "instances.json"
    if not metadata_path.is_file():
        return [], []
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, list):
        raise ValueError("JSPLIB instances.json must be a list")
    manifest = _checksum_manifest(root / "CHECKSUMS.txt")
    rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for item in sorted(
        (item for item in metadata if isinstance(item, dict)),
        key=lambda item: str(item.get("name") or ""),
    ):
        name = str(item.get("name") or "").strip()
        rel = str(item.get("path") or "")
        path = root / rel
        unit = _relative(path)
        candidate_id = f"jsplib/{name or path.name}"
        attempts = [
            _attempt(
                "source_availability", "passed" if path.is_file() else "failed", unit
            )
        ]
        disposition = "rejected"
        reasons = ["invalid_or_missing_jsplib_source"]
        evidence: dict[str, Any] = {"path": unit}
        if path.is_file():
            digest = _sha256(path)
            evidence["sha256"] = digest
            expected = manifest.get(rel)
            checksum_ok = expected == digest
            attempts.append(
                _attempt(
                    "checksum_lock",
                    "passed" if checksum_ok else "failed",
                    "manifest hash matches source bytes"
                    if checksum_ok
                    else "missing or mismatched CHECKSUMS entry",
                )
            )
            try:
                jobs, machines, operations = _parse_jsplib(path)
                schema_ok = True
            except (OSError, UnicodeError, ValueError) as exc:
                jobs = machines = operations = 0
                schema_ok = False
                evidence["parse_error"] = str(exc)
            attempts.append(
                _attempt(
                    "native_schema",
                    "passed" if schema_ok else "failed",
                    f"{jobs} jobs x {machines} machines",
                )
            )
            evidence.update(
                {"jobs": jobs, "machines": machines, "operations": operations}
            )
            active = f"jsplib_job_shop:{name}" in active_text or digest in active_hashes
            duplicate = digest in seen_hashes
            if duplicate:
                disposition, reasons = "secondary", ["duplicate_operation_graph_bytes"]
            elif active:
                disposition, reasons = "secondary", ["already_in_core"]
            elif not checksum_ok or not schema_ok:
                disposition, reasons = "rejected", ["source_integrity_or_schema_failed"]
            else:
                disposition, reasons = (
                    "ready_for_full_admission",
                    [
                        "independent_locked_operation_graph",
                        "current_jsplib_backend_consumable",
                        "difficulty_is_diagnostic",
                        "procedural_stress_explicitly_labelled",
                    ],
                )
            seen_hashes.add(digest)
        raw = _row(
            candidate_id=f"raw/{candidate_id}",
            source_id="jsplib",
            source_unit=unit,
            domain="manufacturing",
            classification_scope="raw_unit",
            disposition=disposition,
            reason_codes=reasons,
            repair_attempts=attempts,
            evidence=evidence,
        )
        candidate = _row(
            candidate_id=candidate_id,
            source_id="jsplib",
            source_unit=unit,
            domain="manufacturing",
            classification_scope="candidate",
            disposition=disposition,
            reason_codes=reasons,
            repair_attempts=attempts,
            evidence=evidence,
            source_family="jsplib",
            source_metadata={"instance_name": name, **evidence},
            procedural_stress={"label": "procedural", "source_native": False},
        )
        added_rows, added_candidates = _raw_and_candidate(raw=raw, candidate=candidate)
        rows.extend(added_rows)
        candidates.extend(added_candidates)
    return rows, candidates


def _refine_realm(
    path: Path, active_text: str, active_hashes: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not path.is_file():
        return [], []
    source_sha = _sha256(path)
    payload = _load_object(path)
    values = payload.get("instances")
    if not isinstance(values, list):
        raise ValueError("REALM J2 instances must be a list")
    rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    seen_graphs: set[str] = set()
    allowed = {
        "machine_breakdown",
        "power_outage",
        "supply_delay",
        "weather_effect",
        "emergency_shutdown",
    }
    for item in sorted(
        (item for item in values if isinstance(item, dict)),
        key=lambda item: str(item.get("instance_id") or ""),
    ):
        instance_id = str(item.get("instance_id") or "").strip()
        candidate_id = f"realm_j2/{instance_id or 'missing-id'}"
        jobs = int(item.get("num_jobs") or 0)
        machines = int(item.get("num_machines") or 0)
        disruptions = (
            item.get("disruptions") if isinstance(item.get("disruptions"), list) else []
        )
        kinds = sorted(
            {
                str(event.get("type"))
                for event in disruptions
                if isinstance(event, dict) and event.get("type")
            }
        )
        graph_sha = _canonical_sha(item.get("jobs") or [])
        active = instance_id and instance_id in active_text
        schema_ok = bool(instance_id and jobs > 0 and machines > 0 and item.get("jobs"))
        event_ok = bool(kinds and set(kinds) <= allowed)
        duplicate = graph_sha in seen_graphs
        attempts = [
            _attempt("source_lock", "passed", f"J2 file sha256={source_sha}"),
            _attempt(
                "native_job_graph",
                "passed" if schema_ok else "failed",
                f"{jobs}x{machines}",
            ),
            _attempt(
                "source_native_disruption_registry",
                "passed" if event_ok else "failed",
                ",".join(kinds) or "none",
            ),
        ]
        evidence = {
            "path": _relative(path),
            "source_sha256": source_sha,
            "instance_id": instance_id,
            "operation_graph_sha256": graph_sha,
            "jobs": jobs,
            "machines": machines,
            "operations": jobs * machines,
            "source_native_disruption_types": kinds,
        }
        if active or source_sha in active_hashes and instance_id in active_text:
            disposition, reasons = "secondary", ["already_in_core"]
        elif duplicate:
            disposition, reasons = "secondary", ["duplicate_operation_graph"]
        elif not schema_ok or not event_ok:
            disposition, reasons = (
                "rejected",
                ["invalid_native_job_or_disruption_schema"],
            )
        elif kinds != ["machine_breakdown"]:
            attempts.append(
                _attempt(
                    "current_backend_state_transition",
                    "semantic_loss_detected",
                    (
                        "exact executable transition is not implemented for "
                        + ",".join(kinds)
                    ),
                )
            )
            disposition, reasons = (
                "redesign",
                [
                    "source_native_event_backend_semantics_unimplemented",
                    "exact_state_transition_required_before_core",
                ],
            )
        else:
            disposition, reasons = (
                "ready_for_full_admission",
                [
                    "independent_source_native_operation_graph",
                    "source_native_actionable_disruption",
                    "current_realm_sidecar_backend_consumable",
                    "difficulty_is_diagnostic",
                ],
            )
        seen_graphs.add(graph_sha)
        raw = _row(
            candidate_id=f"raw/{candidate_id}",
            source_id="realm_j2",
            source_unit=f"{_relative(path)}#{instance_id}",
            domain="manufacturing",
            classification_scope="raw_unit",
            disposition=disposition,
            reason_codes=reasons,
            repair_attempts=attempts,
            evidence=evidence,
        )
        candidate_extra: dict[str, Any] = {}
        if disposition == "redesign":
            candidate_extra["redesign_spec"] = {
                "required_backend": "jsplib_job_shop_realm_j2",
                "required_controls": [
                    "inspect_job_shop_state",
                    "dispatch_ready_operations",
                    "intervene_on_source_event",
                ],
                "required_semantics": [
                    f"source_native_{kind}_state_transition" for kind in kinds
                ],
                "source_contract": (
                    "the selected J2 row must remain the canonical runtime source"
                ),
            }
        candidate = _row(
            candidate_id=candidate_id,
            source_id="realm_j2",
            source_unit=f"{_relative(path)}#{instance_id}",
            domain="manufacturing",
            classification_scope="candidate",
            disposition=disposition,
            reason_codes=reasons,
            repair_attempts=attempts,
            evidence=evidence,
            source_family="realm_j2",
            source_metadata=evidence,
            procedural_stress={"label": "source_native", "source_native": True},
            **candidate_extra,
        )
        added_rows, added_candidates = _raw_and_candidate(raw=raw, candidate=candidate)
        rows.extend(added_rows)
        candidates.extend(added_candidates)
    return rows, candidates


def _dyna_metrics(bundle: Path) -> dict[str, Any]:
    model = bundle / "input_model.json"
    events = bundle / "events.jsonl"
    static_jobs = bundle / "static_jobs.json"
    static_machines = bundle / "static_machines.json"
    json.loads(model.read_text(encoding="utf-8"))
    jobs_payload = json.loads(static_jobs.read_text(encoding="utf-8"))
    machines_payload = json.loads(static_machines.read_text(encoding="utf-8"))
    jobs = jobs_payload.get("jobs")
    if not isinstance(jobs, dict) or not jobs:
        raise ValueError("static_jobs.json lacks a nonempty jobs object")
    operation_count = 0
    for job in jobs.values():
        if not isinstance(job, dict) or not isinstance(job.get("routing"), list):
            raise ValueError("static job lacks a routing list")
        operation_count += len(job["routing"])
    if operation_count <= 0:
        raise ValueError("static job graph has no operations")
    machines = machines_payload.get("machines")
    if not isinstance(machines, list) or not machines:
        raise ValueError("static_machines.json lacks a nonempty machines list")
    counts: Counter[str] = Counter()
    event_rows: list[dict[str, Any]] = []
    for line in events.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if not isinstance(event, dict) or not event.get("event_type"):
            raise ValueError("event line lacks event_type")
        event_rows.append(event)
        counts[str(event["event_type"])] += 1
    actionable = sorted(set(counts) & ACTIONABLE_DYNA_EVENTS)
    event_count = sum(counts.values())
    route_lengths = {
        str(job_id): len(job["routing"]) for job_id, job in jobs.items()
    }
    operation_work_count = operation_count
    for event in event_rows:
        if str(event.get("event_type") or "") != "ROUTE_CHANGE":
            continue
        job_id = str(event.get("job_id") or "")
        new_routing = event.get("new_routing")
        from_step = event.get("from_step")
        if job_id not in route_lengths or not isinstance(new_routing, list):
            raise ValueError("route change does not match the static job graph")
        if isinstance(from_step, bool) or not isinstance(from_step, int):
            raise ValueError("route change lacks an integer from_step")
        new_length = max(0, from_step) + len(new_routing)
        operation_work_count += max(0, new_length - route_lengths[job_id])
        route_lengths[job_id] = new_length
    estimated_work_ticks = event_count + operation_work_count + 2
    return {
        "input_sha256": _sha256(model),
        "events_sha256": _sha256(events),
        "static_jobs_sha256": _sha256(static_jobs),
        "static_machines_sha256": _sha256(static_machines),
        "event_count": event_count,
        "operation_count": operation_count,
        "operation_work_count": operation_work_count,
        "estimated_horizon_ticks": estimated_work_ticks,
        "event_driven_replay_contract": {
            "contract": DYNASCHED_EVENT_DRIVEN_REPLAY_CONTRACT,
            "estimated_work_ticks": estimated_work_ticks,
            "max_work_ticks": DYNASCHED_EVENT_DRIVEN_MAX_WORK_TICKS,
            "within_budget": (
                estimated_work_ticks <= DYNASCHED_EVENT_DRIVEN_MAX_WORK_TICKS
            ),
        },
        "event_types": sorted(counts),
        "actionable_event_types": actionable,
    }


def _dyna_is_hard(config: str, evidence: dict[str, Any]) -> bool:
    scale = "large" in config or "medium" in config
    rho = re.search(r"rho(\d+)", config)
    high_load = bool(rho and int(rho.group(1)) >= 70)
    dynamic = "dynamic" in config or len(evidence["actionable_event_types"]) >= 2
    return scale and high_load and dynamic and evidence["event_count"] > 0


def _refine_dynasched(
    root: Path, active_text: str, runtime_available: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bundles = (
        sorted(path.parent for path in root.glob("**/input_model.json"))
        if root.is_dir()
        else []
    )
    raw_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_source_graphs: set[tuple[str, str]] = set()
    for bundle in bundles:
        rel = bundle.relative_to(root).as_posix()
        parts = bundle.relative_to(root).parts
        group = "/".join(parts[:-1])
        attempts = [_attempt("source_bundle", "passed", rel)]
        try:
            evidence = _dyna_metrics(bundle)
            valid = True
            attempts.append(
                _attempt(
                    "input_and_event_schema",
                    "passed",
                    f"{evidence['event_count']} events",
                )
            )
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            evidence = {
                "parse_error": str(exc),
                "event_count": 0,
                "actionable_event_types": [],
            }
            valid = False
            attempts.append(_attempt("input_and_event_schema", "failed", str(exc)))
        record = {
            "bundle": bundle,
            "rel": rel,
            "group": group,
            "valid": valid,
            "attempts": attempts,
            "evidence": evidence,
        }
        grouped[group].append(record)

    selected: dict[str, dict[str, Any]] = {}
    for group, records in grouped.items():
        valid = [record for record in records if record["valid"]]
        if valid:
            selected[group] = sorted(
                valid,
                key=lambda record: (
                    not bool(record["evidence"]["actionable_event_types"]),
                    int(record["evidence"]["estimated_horizon_ticks"])
                    > DYNASCHED_EVENT_DRIVEN_MAX_WORK_TICKS,
                    -len(record["evidence"]["actionable_event_types"]),
                    int(record["evidence"]["estimated_horizon_ticks"]),
                    record["rel"],
                ),
            )[0]

    for group in sorted(grouped):
        representative = selected.get(group)
        for record in sorted(grouped[group], key=lambda item: item["rel"]):
            rel = record["rel"]
            attempts = list(record["attempts"])
            if not record["valid"]:
                disposition, reasons = "rejected", ["invalid_dynasched_bundle"]
            elif (
                int(record["evidence"]["estimated_horizon_ticks"])
                > DYNASCHED_EVENT_DRIVEN_MAX_WORK_TICKS
            ):
                attempts.append(
                    _attempt(
                        "formal_replay_budget_profile",
                        "held_for_budget",
                        (
                            "estimated event-driven work exceeds the bounded "
                            f"{DYNASCHED_EVENT_DRIVEN_MAX_WORK_TICKS}-tick formal "
                            "replay budget"
                        ),
                    )
                )
                disposition, reasons = (
                    "held_repair",
                    [
                        "dynasched_event_driven_work_budget_exceeded",
                        "source_bundle_remains_scientifically_valid",
                    ],
                )
            elif record is not representative:
                attempts.append(
                    _attempt(
                        "replica_collapse",
                        "secondary",
                        "one representative per structural config",
                    )
                )
                disposition, reasons = (
                    "secondary",
                    ["seed_replica_of_structural_config"],
                )
            else:
                source_graph = (
                    str(record["evidence"].get("input_sha256") or ""),
                    str(record["evidence"].get("events_sha256") or ""),
                )
                attempts.append(
                    _attempt(
                        "dsbx_runtime",
                        "passed" if runtime_available else "failed",
                        "official dsbx import",
                    )
                )
                if rel in active_text:
                    disposition, reasons = "secondary", ["already_in_core"]
                elif source_graph in seen_source_graphs:
                    attempts.append(
                        _attempt(
                            "effective_source_deduplication",
                            "secondary",
                            "byte-identical input_model and event stream already selected",
                        )
                    )
                    disposition, reasons = (
                        "secondary",
                        ["duplicate_effective_source_graph"],
                    )
                elif not runtime_available:
                    attempts.append(
                        _attempt(
                            "runtime_repair",
                            "external_runtime_unavailable",
                            "install the source-locked dsbx Python package",
                        )
                    )
                    disposition, reasons = (
                        "held_repair",
                        ["external_dynasched_runtime_unavailable"],
                    )
                elif not record["evidence"]["actionable_event_types"]:
                    disposition, reasons = (
                        "secondary",
                        ["valid_source_without_actionable_environment_event"],
                    )
                else:
                    disposition, reasons = (
                        "ready_for_full_admission",
                        [
                            "independent_structural_config",
                            "source_native_actionable_events",
                            "difficulty_is_diagnostic",
                            "official_dynasched_runtime_available",
                        ],
                    )
                seen_source_graphs.add(source_graph)
            evidence = {
                "bundle": rel,
                "structural_config": group,
                "hard_dynamic_load_diagnostic": _dyna_is_hard(
                    group, record["evidence"]
                ),
                **record["evidence"],
            }
            raw = _row(
                candidate_id=f"raw/dynasched/{rel}",
                source_id="dynaschedbench",
                source_unit=rel,
                domain="manufacturing",
                classification_scope="raw_unit",
                disposition=disposition,
                reason_codes=reasons,
                repair_attempts=attempts,
                evidence=evidence,
            )
            raw_rows.append(raw)
        if representative is not None:
            raw = next(
                row for row in raw_rows if row["source_unit"] == representative["rel"]
            )
            candidate_id = f"dynasched/{group}"
            candidates.append(
                _row(
                    candidate_id=candidate_id,
                    source_id="dynaschedbench",
                    source_unit=representative["rel"],
                    domain="manufacturing",
                    classification_scope="candidate",
                    disposition=raw["disposition"],
                    reason_codes=raw["reason_codes"],
                    repair_attempts=raw["repair_attempts"],
                    evidence=raw["evidence"],
                    source_family="dynasched",
                    source_metadata=raw["evidence"],
                    procedural_stress={"label": "source_native", "source_native": True},
                )
            )
    return raw_rows + candidates, candidates


def _routing_header(path: Path) -> tuple[str, str, int]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    name = path.stem
    kind = (
        "VRPTW" if "Solomon" in path.as_posix() or path.suffix.lower() == ".txt" else ""
    )
    dimension = 0
    for line in lines[:100]:
        match = re.match(r"\s*(NAME|TYPE|DIMENSION)\s*:?[\t ]+(.+?)\s*$", line, re.I)
        if not match:
            continue
        key, value = match.group(1).upper(), match.group(2).strip()
        if key == "NAME":
            name = value
        elif key == "TYPE":
            kind = value.upper()
        elif key == "DIMENSION":
            dimension = int(float(value))
    if kind == "VRPTW" and not dimension:
        data_lines = [line for line in lines if re.match(r"\s*\d+\s+[-+]?\d", line)]
        dimension = len(data_lines)
    if not name or not kind or dimension <= 0:
        raise ValueError("missing NAME/TYPE/DIMENSION")
    return name, kind, dimension


def _routing_requires_native_matrix(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="replace").upper()
    return (
        "EDGE_WEIGHT_TYPE" in text
        and "EXPLICIT" in text
        and "EDGE_WEIGHT_SECTION" in text
        and "NODE_COORD_SECTION" not in text
    )


def _routing_files(
    pyvrp_root: Path, vrplib_root: Path
) -> Iterable[tuple[str, str, Path]]:
    if pyvrp_root.is_dir():
        for path in sorted(pyvrp_root.glob("**/*.vrp")):
            variant = path.relative_to(pyvrp_root).parts[0].upper()
            yield "pyvrp_instances", variant, path
    if vrplib_root.is_dir():
        for path in sorted(vrplib_root.glob("**/*.vrp")):
            yield "vrplib_package", "CVRP", path
        for path in sorted(vrplib_root.glob("**/*.vrptw")):
            yield "vrplib_package_lkh_cvrptw", "VRPTW", path
        for path in sorted(vrplib_root.glob("**/Vrp-Set-Solomon/**/*.txt")):
            yield "vrplib_package", "VRPTW", path


def _pyvrp_native_probe(path: Path, *, available: bool) -> dict[str, Any]:
    if not available:
        return {"status": "runtime_unavailable"}
    try:
        import pyvrp

        data = pyvrp.read(path, round_func="round")
        clients = list(data.clients())
        vehicles = list(data.vehicle_types())
        return {
            "status": "passed",
            "num_clients": int(data.num_clients),
            "num_depots": int(data.num_depots),
            "num_vehicle_types": int(data.num_vehicle_types),
            "num_vehicles": int(data.num_vehicles),
            "num_groups": int(data.num_groups),
            "has_time_windows": bool(data.has_time_windows()),
            "has_client_prizes": any(int(client.prize) != 0 for client in clients),
            "has_client_release_times": any(
                int(client.release_time) != 0 for client in clients
            ),
            "has_pickups": any(
                any(value != 0 for value in client.pickup) for client in clients
            ),
            "has_multiple_vehicle_capacities": len(
                {tuple(vehicle.capacity) for vehicle in vehicles}
            )
            > 1,
        }
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def _refine_routing(
    pyvrp_root: Path,
    vrplib_root: Path,
    active_text: str,
    active_hashes: set[str],
    pyvrp_native_available: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parsed: list[dict[str, Any]] = []
    for source_id, variant, path in _routing_files(pyvrp_root, vrplib_root):
        attempts = [_attempt("source_availability", "passed", _relative(path))]
        try:
            name, kind, dimension = _routing_header(path)
            digest = _sha256(path)
            valid = True
            attempts.append(
                _attempt("routing_header", "passed", f"{kind} dimension={dimension}")
            )
        except (OSError, UnicodeError, ValueError) as exc:
            name, kind, dimension, digest, valid = (
                path.stem,
                "INVALID",
                0,
                _sha256(path),
                False,
            )
            attempts.append(_attempt("routing_header", "failed", str(exc)))
        if source_id == "vrplib_package":
            if kind in {"VRPTW", "CVRPTW"}:
                variant = "VRPTW"
            elif kind == "CVRP":
                variant = "CVRP"
            else:
                variant = kind
        parsed.append(
            {
                "source_id": source_id,
                "path": path,
                "name": name,
                "kind": kind,
                "variant": variant,
                "dimension": dimension,
                "sha256": digest,
                "valid": valid,
                "attempts": attempts,
            }
        )
    # Prefer the source tree consumed by current builders, then a stable path.
    parsed.sort(
        key=lambda item: (
            item["name"],
            item["variant"],
            item["source_id"] != "pyvrp_instances",
            str(item["path"]),
        )
    )
    seen_identity: set[tuple[str, str]] = set()
    seen_hash: set[str] = set()
    rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for item in parsed:
        identity = (item["variant"], item["name"])
        unit = _relative(item["path"])
        if item["source_id"] == "pyvrp_instances":
            instance_id = (
                item["path"]
                .relative_to(pyvrp_root / item["variant"])
                .with_suffix("")
                .as_posix()
            )
        else:
            instance_id = item["name"]
        candidate_id = f"routing/{item['variant'].lower()}/{instance_id}"
        resolver_compatible = True
        evidence = {
            "path": unit,
            "sha256": item["sha256"],
            "instance_name": instance_id,
            "source_declared_name": item["name"],
            "routing_type": item["kind"],
            "routing_variant": item["variant"],
            "dimension": item["dimension"],
            "current_source_resolver_compatible": resolver_compatible,
        }
        attempts = list(item["attempts"])
        independent = identity not in seen_identity and item["sha256"] not in seen_hash
        if not item["valid"]:
            disposition, reasons = "rejected", ["invalid_routing_source_schema"]
        elif not independent:
            attempts.append(
                _attempt(
                    "mirror_deduplication",
                    "secondary",
                    "logical instance or byte-identical mirror",
                )
            )
            disposition, reasons = "secondary", ["logical_instance_mirror_duplicate"]
        elif item["sha256"] in active_hashes or (
            item["name"] in active_text
            and SUPPORTED_ROUTING_TYPES.get(item["variant"], "") in active_text
        ):
            disposition, reasons = "secondary", ["already_in_core"]
        elif item["variant"] not in SUPPORTED_ROUTING_TYPES:
            native_probe = _pyvrp_native_probe(
                item["path"], available=pyvrp_native_available
            )
            required_semantics = ROUTING_VARIANT_SEMANTICS.get(
                item["variant"], ["variant_native_objective_and_constraints"]
            )
            evidence["pyvrp_native_probe"] = native_probe
            evidence["required_native_semantics"] = required_semantics
            attempts.extend(
                [
                    _attempt(
                        "pyvrp_native_parse",
                        str(native_probe["status"]),
                        "native PyVRP preserves the source variant"
                        if native_probe["status"] == "passed"
                        else str(
                            native_probe.get("error") or "PyVRP runtime unavailable"
                        ),
                    ),
                    _attempt(
                        "current_backend_semantics",
                        "semantic_loss_detected",
                        ",".join(required_semantics),
                    ),
                ]
            )
            intrinsic_reason = INTRINSIC_SECONDARY_ROUTING_VARIANTS.get(item["variant"])
            if intrinsic_reason:
                disposition, reasons = (
                    "secondary",
                    [
                        "valid_native_source_but_outside_operational_agency_core",
                        intrinsic_reason,
                    ],
                )
            else:
                disposition, reasons = (
                    "redesign",
                    [
                        "routing_variant_requires_native_backend_and_tools",
                        "new_intervenable_operational_state_axis",
                    ],
                )
        elif _routing_requires_native_matrix(item["path"]):
            attempts.append(
                _attempt(
                    "current_backend_semantics",
                    "semantic_loss_detected",
                    "explicit source distance matrix has no coordinate embedding; "
                    "the current route simulator computes Euclidean distances",
                )
            )
            evidence["required_native_semantics"] = ["explicit_source_distance_matrix"]
            disposition, reasons = (
                "redesign",
                [
                    "explicit_distance_matrix_backend_semantics_missing",
                    "source_metric_cannot_be_replaced_by_euclidean_coordinates",
                ],
            )
        else:
            disposition, reasons = (
                "ready_for_full_admission",
                [
                    "independent_locked_routing_graph",
                    "current_routing_backend_consumable",
                    "difficulty_is_diagnostic",
                    "procedural_stress_explicitly_labelled",
                ],
            )
        raw = _row(
            candidate_id=f"raw/{item['source_id']}/{unit}",
            source_id=item["source_id"],
            source_unit=unit,
            domain="logistics",
            classification_scope="raw_unit",
            disposition=disposition,
            reason_codes=reasons,
            repair_attempts=attempts,
            evidence=evidence,
        )
        candidate: dict[str, Any] | None = None
        if independent:
            extra: dict[str, Any] = {}
            if disposition == "redesign":
                extra["redesign_spec"] = {
                    "required_backend": f"pyvrp_{item['variant'].lower()}",
                    "required_controls": [
                        "inspect_native_constraints",
                        "dispatch_native_route",
                        "revise_native_route",
                    ],
                    "source_contract": "preserve variant-native depots, fleet, pickup/delivery, and time-window fields",
                    "required_semantics": ROUTING_VARIANT_SEMANTICS.get(
                        item["variant"], ["variant_native_objective_and_constraints"]
                    ),
                }
            candidate = _row(
                candidate_id=candidate_id,
                source_id=item["source_id"],
                source_unit=unit,
                domain="logistics",
                classification_scope="candidate",
                disposition=disposition,
                reason_codes=reasons,
                repair_attempts=attempts,
                evidence=evidence,
                source_family="routing",
                source_metadata=evidence,
                procedural_stress={"label": "procedural", "source_native": False},
                **extra,
            )
        rows.append(raw)
        if candidate is not None:
            rows.append(candidate)
            candidates.append(candidate)
        seen_identity.add(identity)
        seen_hash.add(item["sha256"])
    return rows, candidates


def _m5_hard_score(values: list[float]) -> float:
    if not values or sum(values) <= 0:
        return -1.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    zero_fraction = sum(value == 0 for value in values) / len(values)
    return (
        math.sqrt(variance) / max(1.0, mean)
        + max(values) / max(1.0, mean)
        + zero_fraction
    )


def _refine_m5(
    root: Path, orgym_root: Path, active_text: str, runtime_available: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sales = root / "sales_train_evaluation.csv"
    if not sales.is_file():
        return [], []
    records: list[dict[str, Any]] = []
    best_by_axis: dict[tuple[str, str], tuple[float, str]] = {}
    with sales.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        day_fields = [
            field for field in reader.fieldnames or [] if re.fullmatch(r"d_\d+", field)
        ]
        selected_days = day_fields[-60:]
        for row in reader:
            key = str(row.get("id") or "")
            values = [float(row[field] or 0.0) for field in selected_days]
            score = _m5_hard_score(values)
            axis = (str(row.get("dept_id") or ""), str(row.get("store_id") or ""))
            record = {
                "key": key,
                "item_id": str(row.get("item_id") or ""),
                "dept_id": axis[0],
                "store_id": axis[1],
                "score": round(score, 8),
                "demand_total": sum(values),
                "demand_max": max(values or [0.0]),
                "start_day": int(selected_days[0].split("_")[1])
                if selected_days
                else 0,
                "end_day": int(selected_days[-1].split("_")[1]) if selected_days else 0,
            }
            records.append(record)
            current = best_by_axis.get(axis)
            rank = (score, key)
            if score >= 0 and (current is None or rank > current):
                best_by_axis[axis] = rank
    selected_keys = {value[1] for value in best_by_axis.values()}
    rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    lock_exists = (root / "source_lock.json").is_file()
    runtime_source_exists = orgym_root.is_dir()
    for record in records:
        key = record["key"]
        unit = f"sales_train_evaluation.csv#{key}"
        candidate_id = f"m5_orgym/{key}/d{record['start_day']}_d{record['end_day']}"
        active = (
            key in active_text
            and f"d{record['start_day']}_d{record['end_day']}" in active_text
        )
        attempts = [
            _attempt(
                "m5_source_lock",
                "passed" if lock_exists else "failed",
                "source_lock.json",
            ),
            _attempt(
                "orgym_runtime",
                "passed" if runtime_available and runtime_source_exists else "failed",
                "InvManagement-v1 runtime",
            ),
            _attempt(
                "hard_stream_selection",
                "selected" if key in selected_keys else "secondary",
                f"dept-store score={record['score']}",
            ),
        ]
        evidence = {
            **record,
            "source_files": [
                "calendar.csv",
                "sales_train_evaluation.csv",
                "sell_prices.csv",
            ],
        }
        if active:
            disposition, reasons = "secondary", ["already_in_core"]
        elif key not in selected_keys:
            disposition, reasons = (
                "secondary",
                ["valid_series_not_hardest_for_department_store_axis"],
            )
        elif not lock_exists:
            disposition, reasons = (
                "held_repair",
                ["external_m5_source_lock_unavailable"],
            )
        elif not runtime_available or not runtime_source_exists:
            disposition, reasons = "held_repair", ["external_orgym_runtime_unavailable"]
        else:
            disposition, reasons = (
                "ready_for_full_admission",
                [
                    "hardest_empirical_demand_stream_per_department_store_axis",
                    "source_locked_m5_window",
                    "current_orgym_backend_consumable",
                    "long_horizon_volatility_and_intermittency_pressure",
                ],
            )
        raw = _row(
            candidate_id=f"raw/m5/{key}",
            source_id="m5_forecasting",
            source_unit=unit,
            domain="logistics",
            classification_scope="raw_unit",
            disposition=disposition,
            reason_codes=reasons,
            repair_attempts=attempts,
            evidence=evidence,
        )
        rows.append(raw)
        if key in selected_keys:
            candidate = _row(
                candidate_id=candidate_id,
                source_id="m5_forecasting",
                source_unit=unit,
                domain="logistics",
                classification_scope="candidate",
                disposition=disposition,
                reason_codes=reasons,
                repair_attempts=attempts,
                evidence=evidence,
                source_family="m5_orgym",
                source_metadata=evidence,
                procedural_stress={"label": "none_required", "source_native": True},
            )
            rows.append(candidate)
            candidates.append(candidate)
    # OR-Gym is one supporting runtime, not 26 MiB of fake data candidates.
    rows.append(
        _row(
            candidate_id="raw/orgym/runtime",
            source_id="orgym",
            source_unit=_relative(orgym_root),
            domain="logistics",
            classification_scope="raw_unit",
            disposition="secondary",
            reason_codes=["support_runtime_not_independent_empirical_source"],
            repair_attempts=[
                _attempt(
                    "runtime_repository",
                    "passed" if runtime_source_exists else "failed",
                    _relative(orgym_root),
                )
            ],
            evidence={
                "runtime_available": runtime_available,
                "source_checkout_available": runtime_source_exists,
            },
        )
    )
    return rows, candidates


def build(
    *,
    source_suite: Path = DEFAULT_SOURCE_SUITE,
    jsplib_root: Path = DEFAULT_JSPLIB,
    realm_path: Path = DEFAULT_REALM,
    dynasched_root: Path = DEFAULT_DYNASCHED,
    pyvrp_root: Path = DEFAULT_PYVRP,
    vrplib_root: Path = DEFAULT_VRPLIB,
    m5_root: Path = DEFAULT_M5,
    orgym_root: Path = DEFAULT_ORGYM,
    dynasched_runtime_available: bool | None = None,
    orgym_runtime_available: bool | None = None,
    pyvrp_native_available: bool | None = None,
) -> dict[str, Any]:
    suite = _load_object(source_suite)
    active_text, active_hashes = _active_evidence(suite)
    if dynasched_runtime_available is None:
        dynasched_runtime_available = importlib.util.find_spec("dsbx") is not None
    if orgym_runtime_available is None:
        orgym_runtime_available = importlib.util.find_spec("or_gym") is not None
    if pyvrp_native_available is None:
        pyvrp_native_available = importlib.util.find_spec("pyvrp") is not None
    tracks = [
        _refine_jsplib(jsplib_root, active_text, active_hashes),
        _refine_realm(realm_path, active_text, active_hashes),
        _refine_dynasched(dynasched_root, active_text, dynasched_runtime_available),
        _refine_routing(
            pyvrp_root,
            vrplib_root,
            active_text,
            active_hashes,
            pyvrp_native_available,
        ),
        _refine_m5(m5_root, orgym_root, active_text, orgym_runtime_available),
    ]
    rows = [row for track_rows, _ in tracks for row in track_rows]
    candidates = [row for _, track_candidates in tracks for row in track_candidates]
    rows.sort(
        key=lambda row: (
            row["source_id"],
            row["classification_scope"],
            row["candidate_id"],
        )
    )
    candidates.sort(key=lambda row: (row["source_id"], row["candidate_id"]))
    raw_units = [row for row in rows if row["classification_scope"] == "raw_unit"]
    disposition_counts = Counter(row["final_disposition"] for row in candidates)
    summary = {
        "n_discovered": len(rows),
        "n_terminal": len(rows),
        "n_unresolved": 0,
        "n_raw_units": len(raw_units),
        "n_independent_candidates": len(candidates),
        "candidate_dispositions": dict(sorted(disposition_counts.items())),
        "raw_units_by_source": dict(
            sorted(Counter(row["source_id"] for row in raw_units).items())
        ),
        "ready_for_full_admission_by_domain": dict(
            sorted(
                Counter(
                    row["domain"]
                    for row in candidates
                    if row["final_disposition"] == "ready_for_full_admission"
                ).items()
            )
        ),
    }
    return {
        "schema_version": "operate-logistics-manufacturing-refinement-v1",
        "inputs": {
            "source_suite": {
                "path": _relative(source_suite),
                "sha256": _sha256(source_suite),
            }
        },
        "status": "complete",
        "classification_contract": {
            "ready_for_full_admission_scope": (
                "source_identity_native_control_structural_prefilter_complete"
            ),
            "promotion_requires": "focused_candidate_delta_behavioral_replay",
            "raw_units_do_not_inflate_candidate_denominator": True,
            "allowed_dispositions": sorted(FINAL_DISPOSITIONS),
        },
        "no_pending_decisions": True,
        "summary": summary,
        "rows": rows,
        "source_units": raw_units,
        "candidates": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-suite", type=Path, default=DEFAULT_SOURCE_SUITE)
    parser.add_argument("--jsplib-root", type=Path, default=DEFAULT_JSPLIB)
    parser.add_argument("--realm-path", type=Path, default=DEFAULT_REALM)
    parser.add_argument("--dynasched-root", type=Path, default=DEFAULT_DYNASCHED)
    parser.add_argument("--pyvrp-root", type=Path, default=DEFAULT_PYVRP)
    parser.add_argument("--vrplib-root", type=Path, default=DEFAULT_VRPLIB)
    parser.add_argument("--m5-root", type=Path, default=DEFAULT_M5)
    parser.add_argument("--orgym-root", type=Path, default=DEFAULT_ORGYM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    report = build(
        source_suite=args.source_suite,
        jsplib_root=args.jsplib_root,
        realm_path=args.realm_path,
        dynasched_root=args.dynasched_root,
        pyvrp_root=args.pyvrp_root,
        vrplib_root=args.vrplib_root,
        m5_root=args.m5_root,
        orgym_root=args.orgym_root,
    )
    if args.execute:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
