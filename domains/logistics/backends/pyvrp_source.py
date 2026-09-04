"""Source-locked VRPLIB/Solomon instance resolution for routing backends."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..seeds.vrplib_reader import read_instance


class PyvrpSourceContractError(ValueError):
    """Raised when a declared routing source cannot construct runtime state."""


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _same_number(left: Any, right: Any) -> bool:
    try:
        return abs(float(left) - float(right)) <= 1e-6
    except (TypeError, ValueError):
        return False


def resolve_pyvrp_source_instance(
    *,
    source_path: str,
    source_sha256: str | None,
    instance_kind: str,
    backend_config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Open one explicit source file, parse it, and cross-check baked config."""
    if not source_path:
        raise PyvrpSourceContractError("source_instance_path_missing")
    path = Path(source_path)
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    if not path.is_file():
        raise PyvrpSourceContractError("required_source_file_missing")

    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    expected_sha = str(source_sha256 or "").removeprefix("sha256:")
    if expected_sha and actual_sha != expected_sha:
        raise PyvrpSourceContractError("source_hash_mismatch")

    parsed = read_instance(path)
    nodes = list(parsed.get("nodes") or [])
    depot_index = int(parsed.get("depot_index", 0) or 0)
    if not nodes or not (0 <= depot_index < len(nodes)):
        raise PyvrpSourceContractError("source_parser_output_invalid")
    depot_node = nodes[depot_index]
    clients = [row for index, row in enumerate(nodes) if index != depot_index]

    baked = dict(backend_config.get("network") or {})
    baked_customers = list(baked.get("customers") or [])
    if baked_customers:
        clients = clients[: len(baked_customers)]
    normalized_customers = [
        {
            "id": str(
                baked_customers[index].get("id", f"c{index}")
                if index < len(baked_customers)
                else f"c{index}"
            ),
            "x": float(row["x"]),
            "y": float(row["y"]),
            "demand": float(row.get("demand", 0.0)),
            "tw_early": float(row.get("tw_early", 0.0)),
            "tw_late": float(row.get("tw_late", 1.0e9)),
        }
        for index, row in enumerate(clients)
    ]
    network = {
        "depot": {"x": float(depot_node["x"]), "y": float(depot_node["y"])},
        "customers": normalized_customers,
        "capacity": float(parsed.get("capacity", 0.0) or 0.0),
        "n_vehicles": int(
            baked.get("n_vehicles")
            or parsed.get("n_vehicles")
            or 1
        ),
        "service_time": float(parsed.get("service_time", 0.0) or 0.0),
    }

    if baked:
        baked_depot = dict(baked.get("depot") or {})
        checks = [
            _same_number(baked_depot.get("x"), network["depot"]["x"]),
            _same_number(baked_depot.get("y"), network["depot"]["y"]),
            _same_number(baked.get("capacity"), network["capacity"]),
            _same_number(
                baked.get("service_time", 0.0), network["service_time"]
            ),
            len(baked_customers) == len(normalized_customers),
        ]
        for left, right in zip(
            baked_customers, normalized_customers, strict=True
        ):
            checks.extend(
                _same_number(left.get(key), right.get(key))
                for key in ("x", "y", "demand", "tw_early", "tw_late")
            )
        if not all(checks):
            raise PyvrpSourceContractError("source_window_lineage_mismatch")

    channels = [
        "depot_coordinates",
        "client_coordinates",
        "client_demand",
        "vehicle_capacity",
        "distance_or_edge_cost",
    ]
    if instance_kind == "vrptw":
        channels.extend(["service_duration", "time_window"])
    representation = {
        "instance_kind": instance_kind,
        "source_name": str(parsed.get("name") or path.stem),
        "network": network,
    }
    return {
        "declared_source_path": source_path,
        "source_path": str(path),
        "source_sha256": actual_sha,
        "instance_kind": instance_kind,
        "parser_representation": representation,
        "parser_output_digest": _digest(representation),
        "consumed_channels": channels,
    }
