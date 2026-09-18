"""Source-locked VRPLIB/Solomon instance resolution for routing backends."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ..seeds.vrplib_reader import read_instance


class PyvrpSourceContractError(ValueError):
    """Raised when a declared routing source cannot construct runtime state."""


# The seed builders derive a bounded customer slice of the cited instance so
# the dispatch-wave simulator has a tractable fleet. The slice is a real
# provenance fact: the runtime state is NOT the cited instance. A seed must
# say so; a slice that is neither declared structurally nor stated in the
# provenance note fails closed instead of silently under-modelling.
_INSTANCE_SLICE_NOTE = re.compile(
    r"slice of VRPLIB instance '(?P<instance>[^']+)' \((?P<anchor>[^)]*)\)"
    r"(?:;\s*(?P<summary>[^.;]*))?"
)
_NOTE_CUSTOMER_COUNT = re.compile(r"(\d+)\s+customers")
_DEFAULT_SLICE_REASON = "cited_instance_is_sliced_for_dispatch_wave_simulator"


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _instance_names_match(declared: str, resolved: str) -> bool:
    """Suffix-tolerant instance identity.

    The generated notes cite the instance with its mirror subdirectory
    (``GH200/C1_2_1`` for the parsed ``C1_2_1``), so only the trailing
    component is compared.
    """

    def tail(value: str) -> str:
        return value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].strip()

    return bool(tail(declared)) and tail(declared) == tail(resolved)


def cited_client_count(parsed: dict[str, Any]) -> int:
    """Clients the cited instance declares (``DIMENSION`` minus the depot).

    ``vrplib``/``read_instance`` keep the parsed ``DIMENSION`` header out of
    the normalized dict, so it is re-read from the file header; a sourceless
    parser output falls back to the parsed node count.
    """
    dimension = parsed.get("dimension")
    try:
        dimension_value = int(dimension)
    except (TypeError, ValueError):
        dimension_value = 0
    if dimension_value <= 0:
        dimension_value = len(parsed.get("nodes") or [])
    return max(0, dimension_value - 1)


def declared_instance_slice(
    *,
    backend_config: dict[str, Any],
    provenance_notes: str = "",
) -> dict[str, Any] | None:
    """Return the seed's explicit truncation declaration, if any.

    Two accepted shapes, in preference order:

    1. ``backend_config['instance_slice']`` — the structured builder
       declaration (``instantiated_customers`` / ``cited_clients`` /
       ``reason``). Numeric fields are validated against the realized slice
       by the caller so a stale declaration cannot pass as honest.
    2. The generated ``provenance.notes`` sentence emitted by the routing
       builders (``... slice of VRPLIB instance '<name>' (<anchor>); ...``).
       Released rows carry this and must keep loading; the declared reason is
       the verbatim remainder of the note.
    """
    declared = backend_config.get("instance_slice")
    if isinstance(declared, dict) and declared:
        reason = str(declared.get("reason") or "").strip()
        return {
            "declared": True,
            "declaration_source": "backend_config.instance_slice",
            "reason": reason or _DEFAULT_SLICE_REASON,
            "declared_instantiated_customers": declared.get(
                "instantiated_customers"
            ),
            "declared_cited_clients": declared.get("cited_clients"),
        }
    match = _INSTANCE_SLICE_NOTE.search(str(provenance_notes or ""))
    if match is None:
        return None
    anchor = match.group("anchor").strip()
    summary = (match.group("summary") or "").strip()
    description = "slice of VRPLIB instance"
    if summary:
        description = f"{description}; {summary}"
    if anchor:
        description = f"{description} ({anchor})"
    declared_count = _NOTE_CUSTOMER_COUNT.search(summary)
    return {
        "declared": True,
        "declaration_source": "provenance.notes",
        "reason": description,
        "declared_instance": match.group("instance"),
        "declared_anchor": anchor,
        # The note states the instantiated size in prose; the caller checks it
        # against the realized slice so a hand-written note cannot declare an
        # arbitrary instance or size. ``None`` (no count in the note) is also
        # refused rather than waved through.
        "declared_instantiated_customers": (
            int(declared_count.group(1)) if declared_count else None
        ),
    }


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
    provenance_notes: str = "",
) -> dict[str, Any]:
    """Open one explicit source file, parse it, and cross-check baked config.

    Raises ``PyvrpSourceContractError`` when the seed cites a VRPLIB instance
    but instantiates fewer clients than that instance declares, unless the
    seed explicitly declares the truncation (``backend_config['instance_slice']``
    or the generated ``provenance.notes`` sentence). The runtime never silently
    substitutes a smaller instance for the cited one.
    """
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
    cited_clients = cited_client_count(parsed)
    if baked_customers:
        clients = clients[: len(baked_customers)]

    # ── Citation/instantiation contract ─────────────────────────────────
    # ``baked_customers`` (the instantiated slice) may legitimately be smaller
    # than ``cited_clients`` (the file's DIMENSION) — the dispatch-wave
    # simulator is bounded. That gap is provenance, not a detail: refuse an
    # undeclared slice so a row can never claim the full instance while
    # running a truncated one.
    instantiated = len(clients)
    source_slice = {
        "cited_instance": str(parsed.get("name") or path.stem),
        "cited_clients": cited_clients,
        "instantiated_customers": instantiated,
        "truncated": instantiated < cited_clients,
    }
    if source_slice["truncated"]:
        declaration = declared_instance_slice(
            backend_config=backend_config,
            provenance_notes=provenance_notes,
        )
        if declaration is None:
            raise PyvrpSourceContractError(
                "source_instance_truncated_without_declaration:"
                f"cited={source_slice['cited_instance']}"
                f"({cited_clients} clients) but runtime instantiates "
                f"{instantiated}; declare backend_config['instance_slice'] or "
                "the provenance note, or stop citing the full instance"
            )
        if declaration["declaration_source"] == "provenance.notes":
            # A prose note is hand-editable, so it is checked against the
            # file that was actually parsed: the cited name and the stated
            # size must both be real. Otherwise any candidate row could pass
            # the gate by asserting a different instance or a made-up size.
            if not _instance_names_match(
                str(declaration.get("declared_instance") or ""),
                source_slice["cited_instance"],
            ):
                raise PyvrpSourceContractError(
                    "source_instance_slice_declaration_mismatch:note_cites="
                    f"{declaration.get('declared_instance')!r} "
                    f"resolved={source_slice['cited_instance']!r}"
                )
        declared_instantiated = declaration.get("declared_instantiated_customers")
        if isinstance(declared_instantiated, int) and not isinstance(
            declared_instantiated, bool
        ):
            if declared_instantiated != instantiated:
                raise PyvrpSourceContractError(
                    "source_instance_slice_declaration_stale:declared="
                    f"{declared_instantiated} instantiated={instantiated}"
                )
        elif declaration["declaration_source"] == "provenance.notes":
            raise PyvrpSourceContractError(
                "source_instance_slice_declaration_mismatch:"
                "provenance_note_states_no_customer_count"
            )
        declared_cited = declaration.get("declared_cited_clients")
        if isinstance(declared_cited, int) and not isinstance(
            declared_cited, bool
        ):
            if declared_cited != cited_clients:
                raise PyvrpSourceContractError(
                    "source_instance_slice_declaration_stale:declared_cited="
                    f"{declared_cited} cited={cited_clients}"
                )
        source_slice["truncation_declared"] = True
        source_slice["truncation_reason"] = declaration["reason"]
        source_slice["truncation_declaration_source"] = declaration[
            "declaration_source"
        ]
    else:
        source_slice["truncation_declared"] = False

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
    # ``parser_output_digest`` is a compared source-consumption field, so this
    # payload must stay byte-identical to what released rows hashed. The
    # citation/instantiation facts therefore live only in the sibling
    # ``source_slice`` keys, never inside the hashed representation.
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
        "source_slice": dict(source_slice),
    }
