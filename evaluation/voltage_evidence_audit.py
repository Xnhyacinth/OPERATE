"""Audit bus evidence coverage without changing the frozen completion scorer.

The caller authenticates trajectory and native-unserved evidence bytes. The
population must be frozen independently of those observations and bind the
source hash; a hash does not itself establish that a population is appropriate.
No voltage compliance, service fraction, or replacement F score is emitted.
"""

from __future__ import annotations

import hashlib
import json
import math


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _bus_ids(value: object) -> bool:
    return (
        isinstance(value, list)
        and all(isinstance(key, str) and key.startswith("bus_") for key in value)
        and len(set(value)) == len(value)
    )


def audit_voltage_evidence(
    trace: list[dict],
    *,
    population: dict | None = None,
    expected_population_sha256: str | None = None,
    expected_source_sha256: str | None = None,
    native_unserved: list[dict] | None = None,
) -> dict:
    """Classify required bus-time evidence against a source-bound population.

    Population schema: ``required_voltage_population.v1``, ``source_sha256``,
    ``required_bus_ids``, ``required_ticks``, and ``excluded_bus_ids``
    (bus ID -> source reason). Required ticks are frozen before execution;
    omitted whole observations cannot shrink expected bus-time.
    Supply the independently recorded SHA-256 of its canonical JSON bytes
    (sort_keys=True, separators=(",", ":"), allow_nan=False).

    Optional native-unserved rows have ``tick``, ``bus_ids``, ``evidence_ids``.
    The caller must verify that these authenticated native evidence IDs prove
    lack of supply to those buses. NaN or entity absence cannot prove this.
    Known native unserved slots have complete evidence but are never credited
    as finite observations. Excluded source buses do not enter the population.
    """
    result = {
        "schema_version": "voltage_evidence_coverage.v1",
        "changes_frozen_score": False,
        "status": "unauditable",
        "reason": "authoritative_voltage_population_missing",
    }
    if population is None:
        return result
    if not isinstance(population, dict):
        raise ValueError("voltage_population_invalid")
    required = population.get("required_bus_ids")
    required_ticks = population.get("required_ticks")
    excluded = population.get("excluded_bus_ids")
    if (
        population.get("schema_version") != "required_voltage_population.v1"
        or not _bus_ids(required)
        or not required
        or not isinstance(required_ticks, list)
        or not required_ticks
        or any(type(tick) is not int or tick < 0 for tick in required_ticks)
        or len(set(required_ticks)) != len(required_ticks)
        or not isinstance(excluded, dict)
        or not _bus_ids(list(excluded))
        or any(
            not isinstance(reason, str) or not reason.strip()
            for reason in excluded.values()
        )
        or set(required) & excluded.keys()
    ):
        raise ValueError("voltage_population_invalid")
    if not (
        _sha256(expected_source_sha256)
        and population.get("source_sha256") == expected_source_sha256
    ):
        raise ValueError("voltage_population_source_mismatch")
    encoded = json.dumps(
        population, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    population_hash = hashlib.sha256(encoded).hexdigest()
    if (
        not _sha256(expected_population_sha256)
        or population_hash != expected_population_sha256
    ):
        raise ValueError("voltage_population_hash_mismatch")

    required_ids = set(required)
    required_tick_ids = set(required_ticks)
    unserved_by_tick = {}
    for row in native_unserved or []:
        evidence = row.get("evidence_ids") if isinstance(row, dict) else None
        if (
            not isinstance(row, dict)
            or type(row.get("tick")) is not int
            or row["tick"] not in required_tick_ids
            or not _bus_ids(row.get("bus_ids"))
            or not set(row["bus_ids"]) <= required_ids
            or not isinstance(evidence, list)
            or not evidence
            or any(not isinstance(key, str) or not key for key in evidence)
            or row["tick"] in unserved_by_tick
        ):
            raise ValueError("native_unserved_bus_evidence_invalid")
        unserved_by_tick[row["tick"]] = row

    ticks = []
    seen_ticks = set()
    orphan_ticks = []
    counts = {key: 0 for key in ("finite", "native_unserved", "missing", "unknown")}
    for row in trace:
        if not isinstance(row, dict) or type(row.get("tick")) is not int:
            raise ValueError("voltage_trace_tick_invalid")
        tick = row["tick"]
        if tick in seen_ticks:
            raise ValueError("voltage_trace_tick_duplicate")
        seen_ticks.add(tick)
        observation = row.get("observation")
        if not isinstance(observation, dict):
            raise ValueError("voltage_trace_observation_invalid")
        entities = observation.get("entities", {})
        if not isinstance(entities, dict):
            raise ValueError("voltage_trace_entities_invalid")
        if tick not in required_tick_ids:
            orphan_ticks.append(tick)
            continue
        native = unserved_by_tick.get(tick) or {}
        unserved = set(native.get("bus_ids") or [])
        missing, unknown, finite = [], [], []
        for bus in sorted(required_ids - unserved):
            if bus not in entities:
                missing.append(bus)
                continue
            entity = entities[bus]
            voltage = entity.get("vm_pu") if isinstance(entity, dict) else None
            if (
                type(voltage) in (int, float)
                and math.isfinite(voltage)
                and voltage >= 0
            ):
                finite.append(bus)
            else:
                unknown.append(bus)
        undeclared = sorted(
            key
            for key in entities
            if isinstance(key, str)
            and key.startswith("bus_")
            and key not in required_ids
            and key not in excluded
        )
        classified = {
            "finite": finite,
            "native_unserved": sorted(unserved),
            "missing": missing,
            "unknown": unknown,
        }
        for key, buses in classified.items():
            counts[key] += len(buses)
        ticks.append(
            {
                "tick": tick,
                **{f"{key}_bus_ids": buses for key, buses in classified.items()},
                "native_unserved_evidence_ids": native.get("evidence_ids") or [],
                "undeclared_bus_ids": undeclared,
            }
        )
    if unserved_by_tick.keys() - seen_ticks:
        raise ValueError("native_unserved_bus_tick_outside_trace")
    missing_ticks = sorted(required_tick_ids - seen_ticks)
    complete = (
        bool(ticks)
        and not missing_ticks
        and not orphan_ticks
        and not (counts["missing"] or counts["unknown"])
        and not any(row["undeclared_bus_ids"] for row in ticks)
    )
    result.update(
        status="complete" if complete else "incomplete",
        reason="required_bus_evidence_classified"
        if complete
        else "required_bus_evidence_incomplete",
        source_sha256=expected_source_sha256,
        population_sha256=population_hash,
        scope="source_bound_required_bus_ticks",
        required_bus_ids=sorted(required_ids),
        required_ticks=sorted(required_tick_ids),
        missing_ticks=missing_ticks,
        orphan_ticks=sorted(orphan_ticks),
        required_tick_coverage_complete=not missing_ticks and not orphan_ticks,
        excluded_bus_ids=excluded.copy(),
        expected_bus_time=len(required_ids) * len(required_tick_ids),
        missing_tick_bus_time=len(required_ids) * len(missing_ticks),
        counts=counts,
        ticks=ticks,
    )
    return result
