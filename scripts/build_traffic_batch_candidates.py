#!/usr/bin/env python3
"""Build candidate-only Traffic batches from source-locked SUMO windows.

The builder separates events observed in the real route schedule from explicit,
deterministic stress overlays.  It does not run SUMO, score candidates, update
Core, or write a release artifact.  Output is shaped as one immutable candidate
queue that can be consumed by ``run_protocol21_candidate_batches.py``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.sidecar.sumo_sidecar import sumo_available  # noqa: E402
from domains.traffic.seeds.schema import TrafficPerturbation  # noqa: E402
from domains.traffic.seeds.sumo365 import (  # noqa: E402
    SUMO365_EXPECTED_FILE_SHA256S,
    SUMO365_LICENSE,
    SUMO365_SERVICE_DATES,
    build_sumo365_traffic_seed,
    sumo365_date_files,
)

QUEUE_SCHEMA_VERSION = "traffic_candidate_batch.v1"
EVENT_SCHEDULE_VERSION = "traffic_deterministic_pressure.v1"
DEFAULT_WINDOW_BEGIN_S = 6 * 3600
DEFAULT_WINDOW_END_S = 10 * 3600
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SourceWindow:
    """One real SUMO demand window plus source locks."""

    source_family: str
    service_date: str
    network_ref: str
    route_ref: str
    sumocfg_ref: str
    window_begin_s: int
    window_end_s: int
    expected_sha256: dict[str, str]
    license: str
    source_url: str
    source_version: str

    def __post_init__(self) -> None:
        if not self.source_family.strip() or not self.service_date.strip():
            raise ValueError("source_family and service_date must be non-empty")
        if self.window_begin_s < 0 or self.window_end_s <= self.window_begin_s:
            raise ValueError("source window must have a positive duration")
        required_hashes = {"network", "route", "sumocfg"}
        if set(self.expected_sha256) != required_hashes:
            raise ValueError("expected_sha256 must contain exactly network, route, and sumocfg")
        if any(not _SHA256_RE.fullmatch(value) for value in self.expected_sha256.values()):
            raise ValueError("expected source hashes must be lowercase SHA-256")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve(path: str, *, repo_root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else repo_root / value


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_lock(
    window: SourceWindow,
    *,
    repo_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    paths = {
        "network": _resolve(window.network_ref, repo_root=repo_root),
        "route": _resolve(window.route_ref, repo_root=repo_root),
        "sumocfg": _resolve(window.sumocfg_ref, repo_root=repo_root),
    }
    observed = {name: _file_sha256(path) for name, path in paths.items()}
    blockers: list[str] = []
    for name in sorted(paths):
        if observed[name] is None:
            blockers.append(f"source_file_missing:{name}")
        elif observed[name] != window.expected_sha256[name]:
            blockers.append(f"source_hash_mismatch:{name}")
    return (
        {
            "locked": not blockers,
            "expected_sha256": dict(sorted(window.expected_sha256.items())),
            "observed_sha256": dict(sorted(observed.items())),
            "source_files": {name: str(path) for name, path in sorted(paths.items())},
            "license": window.license,
            "source_url": window.source_url,
            "source_version": window.source_version,
        },
        blockers,
    )


def _physical_source_identity(window: SourceWindow) -> str:
    """Physical graph identity: dataset family + exact network bytes."""
    return "traffic:sumo:physical:" + _canonical_sha256(
        {
            "source_family": window.source_family,
            "network_sha256": window.expected_sha256["network"],
        }
    )


def _effective_source_identity(window: SourceWindow) -> str:
    """Effective source identity: physical graph + independent route date."""
    return "traffic:sumo:effective:" + _canonical_sha256(
        {
            "physical_source_identity": _physical_source_identity(window),
            "service_date": window.service_date,
            "route_sha256": window.expected_sha256["route"],
        }
    )


def _open_route(path: Path):
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


def _parse_departures(
    route_path: Path,
    *,
    begin_s: int,
    end_s: int,
) -> list[int]:
    departures: list[int] = []
    with _open_route(route_path) as stream:
        for _event, element in ET.iterparse(stream, events=("end",)):
            if element.tag.rsplit("}", 1)[-1] in {"vehicle", "trip", "flow"}:
                raw = element.get("depart") or element.get("begin")
                try:
                    depart = int(float(str(raw)))
                except (TypeError, ValueError):
                    element.clear()
                    continue
                if begin_s <= depart < end_s:
                    departures.append(depart)
            element.clear()
    return sorted(departures)


def _source_arrival_schedule(
    departures: list[int],
    *,
    begin_s: int,
    end_s: int,
    decision_interval_s: int,
) -> list[dict[str, Any]]:
    """Select one strongest five-minute arrival burst from source fields."""
    if not departures:
        return []
    bucket_seconds = 300
    counts = Counter(
        begin_s + ((departure - begin_s) // bucket_seconds) * bucket_seconds
        for departure in departures
    )
    bucket, count = min(
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    return [
        {
            "event_type": "arrival_burst",
            "origin": "source_schedule",
            "source_field": "vehicle.depart",
            "window_begin_s": bucket,
            "window_end_s": min(bucket + bucket_seconds, end_s),
            "trigger_tick": (bucket - begin_s) // decision_interval_s,
            "duration_ticks": max(1, bucket_seconds // decision_interval_s),
            "vehicle_count": count,
        }
    ]


def _declared_pressure_schedule(
    *,
    effective_source_identity: str,
    horizon_ticks: int,
) -> list[dict[str, Any]]:
    """Build deterministic, explicitly synthetic pressure overlays."""
    offset = int(effective_source_identity[-8:], 16) % 2
    templates = (
        ("incident", 2 + offset, 3, False, "network_edge_from_live_inventory"),
        ("signal_failure", 5 + offset, 3, True, "tls_from_live_inventory"),
        ("stale_detector", 8 + offset, 3, True, "detector_corridor_from_binding"),
    )
    return [
        {
            "event_type": event_type,
            "origin": "declared_perturbation",
            "deterministic_schedule": True,
            "trigger_tick": min(trigger_tick, max(1, horizon_ticks - 2)),
            "duration_ticks": min(duration, max(1, horizon_ticks - trigger_tick)),
            "hidden": hidden,
            "target_resolver": target_resolver,
            "source_independence_credit": False,
        }
        for event_type, trigger_tick, duration, hidden, target_resolver in templates
    ]


def _scenario_body(
    window: SourceWindow,
    *,
    source_schedule: list[dict[str, Any]],
    declared_schedule: list[dict[str, Any]],
    source_lock: dict[str, Any],
    effective_source_identity: str,
    physical_source_identity: str,
) -> dict[str, Any]:
    difficulty = "high"
    seed = (
        build_sumo365_traffic_seed(
            seed_id=effective_source_identity,
            service_date=window.service_date,
            family="incident_response",
            difficulty_level=difficulty,
            difficulty_mode="deep_planning",
        )
        if window.source_family == "sumo_ingolstadt_365"
        else None
    )
    if seed is None:
        # Reuse the domain-native shape without making an unsupported provenance
        # claim for arbitrary source families.  The queue remains held_repair
        # until a family-specific corridor/TLS binding is supplied.
        from domains.traffic.seeds.from_lust import build_traffic_seed

        seed = build_traffic_seed(
            seed_id=effective_source_identity,
            family="incident_response",
            seed=int(effective_source_identity[-8:], 16),
            difficulty_level=difficulty,
            difficulty_mode="deep_planning",
        )
        seed.net_ref = window.network_ref
        seed.route_ref = window.route_ref
        seed.provenance.data_source = window.source_family
        seed.provenance.files = [
            window.network_ref,
            window.route_ref,
            window.sumocfg_ref,
        ]
        seed.provenance.url = window.source_url
        seed.provenance.commit = window.source_version
        seed.provenance.license = window.license
        seed.provenance.lock_strategy = "file_sha256"
        seed.provenance.source_locked = True

    horizon_ticks = max(
        12,
        (window.window_end_s - window.window_begin_s) // 300,
    )
    seed.backend_kind = "sumo"
    seed.horizon_ticks = horizon_ticks
    seed.tick_minutes = 5
    seed.backend_config = {
        **seed.backend_config,
        "backend_kind": "sumo",
        "sumo_config_path": window.sumocfg_ref,
        "sumo_extra_args": [
            "--begin",
            str(window.window_begin_s),
            "--end",
            str(window.window_end_s),
        ],
        "source_event_schedule": source_schedule,
        "declared_pressure_schedule": declared_schedule,
        "event_schedule_version": EVENT_SCHEDULE_VERSION,
        "effective_source_identity": effective_source_identity,
        "physical_source_identity": physical_source_identity,
        "source_lock": source_lock,
        "release_ready": False,
        "candidate_only": True,
    }
    perturbation_kind = {
        "incident": "lane_blockage",
        "signal_failure": "signal_failure",
        "stale_detector": "detector_dropout",
    }
    seed.perturbations = [
        TrafficPerturbation(
            kind=perturbation_kind[event["event_type"]],
            trigger_tick=int(event["trigger_tick"]),
            duration_ticks=int(event["duration_ticks"]),
            hidden=bool(event["hidden"]),
            target={"runtime_resolver": event["target_resolver"]},
            intensity=0.75,
            notes=(
                "Deterministic declared stress overlay; does not count as a "
                "real source or independent effective source."
            ),
        )
        for event in declared_schedule
    ]
    body = seed.to_dict()
    # Protocol-2.1 source consumption is explicit: the live backend must load
    # these exact network, demand, and configuration bytes.  Keeping the
    # contract on the materialized YAML lets the generic preflight and source
    # evidence adapters validate traffic candidates without special-casing this
    # builder.
    expected = dict(source_lock.get("expected_sha256") or {})
    body["source_contract"] = {
        "runtime_input": [
            window.network_ref,
            window.route_ref,
            window.sumocfg_ref,
        ],
        "derivation_input": [],
        "implementation_asset": [],
        "metadata": [],
        "license": [],
        "file_sha256s": {
            window.network_ref: expected["network"],
            window.route_ref: expected["route"],
            window.sumocfg_ref: expected["sumocfg"],
        },
        "derived_window": {
            "sha256": _canonical_sha256(
                {
                    "source_family": window.source_family,
                    "service_date": window.service_date,
                    "window_begin_s": window.window_begin_s,
                    "window_end_s": window.window_end_s,
                }
            ),
            "recipe_version": "sumo365_source_schedule_burst_v1",
        },
    }
    return body


def build_batch_plan(
    windows: list[SourceWindow],
    *,
    runtime_available: bool | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Build an order-invariant candidate queue with terminal dispositions."""
    runtime_ok = sumo_available() if runtime_available is None else runtime_available
    by_effective_source: dict[str, list[SourceWindow]] = {}
    for window in windows:
        by_effective_source.setdefault(_effective_source_identity(window), []).append(window)

    candidates: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for effective_source_identity in sorted(by_effective_source):
        grouped = sorted(
            by_effective_source[effective_source_identity],
            key=lambda item: (
                item.window_begin_s,
                item.window_end_s,
                item.network_ref,
                item.route_ref,
            ),
        )
        window = grouped[0]
        for duplicate in grouped[1:]:
            duplicates.append(
                {
                    "effective_source_identity": effective_source_identity,
                    "source_window": duplicate.to_dict(),
                    "work_state": "terminal",
                    "disposition": "secondary_duplicate",
                    "reason": "one_row_per_effective_source_identity",
                }
            )

        source_lock, blockers = _source_lock(window, repo_root=repo_root)
        if not runtime_ok:
            blockers.append("live_sumo_runtime_unavailable")
        physical_source_identity = _physical_source_identity(window)
        source_schedule: list[dict[str, Any]] = []
        declared_schedule: list[dict[str, Any]] = []
        scenario: dict[str, Any] | None = None
        if source_lock["locked"]:
            departures = _parse_departures(
                _resolve(window.route_ref, repo_root=repo_root),
                begin_s=window.window_begin_s,
                end_s=window.window_end_s,
            )
            source_schedule = _source_arrival_schedule(
                departures,
                begin_s=window.window_begin_s,
                end_s=window.window_end_s,
                decision_interval_s=300,
            )
            if not departures:
                blockers.append("source_window_has_no_departures")
            declared_schedule = _declared_pressure_schedule(
                effective_source_identity=effective_source_identity,
                horizon_ticks=max(
                    12,
                    (window.window_end_s - window.window_begin_s) // 300,
                ),
            )
            if departures:
                scenario = _scenario_body(
                    window,
                    source_schedule=source_schedule,
                    declared_schedule=declared_schedule,
                    source_lock=source_lock,
                    effective_source_identity=effective_source_identity,
                    physical_source_identity=physical_source_identity,
                )

        if (
            any(
                blocker.startswith("source_") and blocker != "source_window_has_no_departures"
                for blocker in blockers
            )
            or "live_sumo_runtime_unavailable" in blockers
        ):
            disposition = "held_runtime"
            work_state = "terminal"
        elif "source_window_has_no_departures" in blockers:
            disposition = "retired_intrinsic"
            work_state = "terminal"
        else:
            disposition = "held_repair"
            work_state = "pending"

        candidate_key = (
            f"traffic/traffic_batch/{window.source_family}/{window.service_date}/"
            f"{effective_source_identity.rsplit(':', 1)[-1][:16]}"
        )
        if scenario is not None:
            scenario["seed_id"] = candidate_key
        candidates.append(
            {
                "candidate_key": candidate_key,
                "domain": "traffic",
                "backend_kind": "sumo",
                "stage": "conversion",
                "work_state": work_state,
                "disposition": disposition,
                "blockers": sorted(set(blockers)),
                "effective_source_identity": effective_source_identity,
                "physical_source_identity": physical_source_identity,
                "source_window": window.to_dict(),
                "source_lock": source_lock,
                "source_schedule": source_schedule,
                "declared_perturbations": declared_schedule,
                "scenario": scenario,
                "required_next_stage": ("native_prefilter" if work_state == "pending" else None),
                "resource_tokens": 4,
            }
        )

    body = {
        "schema_version": QUEUE_SCHEMA_VERSION,
        "candidate_only": True,
        "formal_release_ready": False,
        "stage": "conversion",
        "n_input_windows": len(windows),
        "n_candidates": len(candidates),
        "n_secondary_duplicates": len(duplicates),
        "candidates": candidates,
        "secondary_duplicates": sorted(
            duplicates,
            key=lambda row: (
                row["effective_source_identity"],
                row["source_window"]["window_begin_s"],
            ),
        ),
        "invariants": {
            "one_row_per_effective_source_identity": True,
            "declared_perturbations_do_not_create_source_independence": True,
            "missing_runtime_or_hash_is_held_runtime": True,
            "core_or_release_mutation": False,
        },
    }
    body["queue_sha256"] = _canonical_sha256(body)
    return body


def materialize_batch(plan: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    """Write immutable candidate YAMLs plus coordinator-compatible queue JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_dir = output_dir / "yamls"
    yaml_dir.mkdir(exist_ok=True)
    queue = json.loads(json.dumps(plan))
    queue["builder_schema_version"] = queue["schema_version"]
    queue["schema_version"] = "candidate-batch-queue-v1"
    queue["items"] = []
    n_written = 0
    for row in queue["candidates"]:
        scenario = row.pop("scenario")
        row["candidate_yaml"] = None
        if scenario is None:
            continue
        filename = f"{row['effective_source_identity'].rsplit(':', 1)[-1]}.yaml"
        path = yaml_dir / filename
        path.write_text(
            yaml.safe_dump(scenario, sort_keys=False),
            encoding="utf-8",
        )
        row["candidate_yaml"] = str(path.relative_to(output_dir))
        row["candidate_yaml_sha256"] = _file_sha256(path)
        n_written += 1
    for row in queue["candidates"]:
        scenario_signature = (
            _canonical_sha256(
                yaml.safe_load((output_dir / row["candidate_yaml"]).read_text(encoding="utf-8"))
            )[:16]
            if row["candidate_yaml"]
            else _canonical_sha256(row["source_window"])[:16]
        )
        item = {
            "work_id": row["candidate_key"],
            "stage": "static_preflight",
            "work_state": row["work_state"],
            "disposition": row["disposition"],
            "domain": "traffic",
            "backend": "sumo",
            "scenario_id": row["candidate_key"],
            "scenario_signature": scenario_signature,
            "effective_source_identity": row["effective_source_identity"],
            "physical_source_identity": row["physical_source_identity"],
            "candidate_yaml": row["candidate_yaml"],
        }
        if row["work_state"] in {"pending", "failed_retryable"}:
            item["command"] = [
                sys.executable,
                "scripts/build_traffic_batch_candidates.py",
                "--check-candidate",
                str((output_dir / row["candidate_yaml"]).resolve()),
            ]
        queue["items"].append(item)
    queue_path = output_dir / "traffic_candidate_queue.json"
    queue_path.write_text(
        json.dumps(queue, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "queue": str(queue_path),
        "queue_sha256": _file_sha256(queue_path),
        "n_yaml_written": n_written,
    }


def check_candidate_yaml(path: Path) -> dict[str, Any]:
    """Fail-closed static check used as the coordinator's first queue stage."""
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("candidate YAML must contain a mapping")
    blockers: list[str] = []
    backend = loaded.get("backend_config") or {}
    source_events = backend.get("source_event_schedule") or []
    declared_events = backend.get("declared_pressure_schedule") or []
    source_lock = backend.get("source_lock") or {}
    if loaded.get("domain") != "traffic" or loaded.get("backend_kind") != "sumo":
        blockers.append("candidate_not_live_sumo_traffic")
    if not source_lock.get("locked"):
        blockers.append("source_lock_not_green")
    if not source_events or any(
        event.get("origin") != "source_schedule" for event in source_events
    ):
        blockers.append("source_schedule_origin_invalid")
    if not declared_events or any(
        event.get("origin") != "declared_perturbation"
        or event.get("source_independence_credit") is not False
        for event in declared_events
    ):
        blockers.append("declared_perturbation_layer_invalid")
    identities = (
        backend.get("effective_source_identity"),
        backend.get("physical_source_identity"),
    )
    if not all(isinstance(value, str) and value for value in identities):
        blockers.append("source_identity_missing")
    report = {
        "schema_version": "traffic_candidate_static_preflight.v1",
        "candidate_yaml": str(path),
        "candidate_yaml_sha256": _file_sha256(path),
        "status": "passed" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "next_stage": "native_prefilter" if not blockers else None,
    }
    if blockers:
        raise ValueError("traffic candidate static preflight failed: " + ", ".join(blockers))
    return report


def sumo365_windows() -> list[SourceWindow]:
    """Return all locally declared SUMO365 morning windows."""
    windows: list[SourceWindow] = []
    for service_date in SUMO365_SERVICE_DATES:
        files = sumo365_date_files(service_date)
        hashes = SUMO365_EXPECTED_FILE_SHA256S[service_date]
        windows.append(
            SourceWindow(
                source_family="sumo_ingolstadt_365",
                service_date=service_date,
                network_ref=files["network"],
                route_ref=files["route"],
                sumocfg_ref=files["sumocfg"],
                window_begin_s=DEFAULT_WINDOW_BEGIN_S,
                window_end_s=DEFAULT_WINDOW_END_S,
                expected_sha256={key: hashes[key] for key in ("network", "route", "sumocfg")},
                license=SUMO365_LICENSE,
                source_url="https://github.com/TUM-VT/sumo_ingolstadt",
                source_version="e0a95deebe200ff81b6705044d66310d6266d42b",
            )
        )
    return windows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-candidate", type=Path)
    parser.add_argument(
        "--runtime-available",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override live SUMO discovery for planning/audit hosts.",
    )
    parser.add_argument("--plan", action="store_true", help="Print plan without writes.")
    args = parser.parse_args()
    if args.check_candidate is not None:
        print(json.dumps(check_candidate_yaml(args.check_candidate), indent=2, sort_keys=True))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required unless --check-candidate is used")
    plan = build_batch_plan(sumo365_windows(), runtime_available=args.runtime_available)
    if args.plan:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    report = materialize_batch(plan, output_dir=args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
