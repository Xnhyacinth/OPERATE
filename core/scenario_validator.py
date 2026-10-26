"""Scenario YAML schema validation.

Ensures scenario YAMLs have all required fields with correct types before
execution. Malformed YAMLs with wrong keys silently defaulting to 0 was a
known source of silent data corruption in the v0.35-rc2 audit.

Usage::

    from core.scenario_validator import validate_scenario_yaml
    errors = validate_scenario_yaml(scenario)
    if errors:
        for err in errors:
            logger.warning(err)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("scenario_validator")

REQUIRED_FIELDS: dict[str, type | tuple[type, ...]] = {
    "domain": str,
    "family": str,
    "difficulty_mode": str,
    "difficulty_level": str,
    "backend_kind": str,
    "seed": int,
    "horizon_ticks": int,
}

OPTIONAL_FIELDS: dict[str, type | tuple[type, ...]] = {
    "seed_id": str,
    "scenario_id": str,
    "scenario_signature": str,
    "backend_config": dict,
    "load_assignments": (list, dict),
    "provenance": dict,
    "provenance_files": list,
    "source_lock": (dict, str),
    "difficulty_params": dict,
    "fog_config": dict,
    "cascade_config": dict,
    "stakeholder_config": dict,
    "dilemma_config": dict,
    "tool_budget": dict,
    "complexity_metrics": dict,
    "observation_budget_chars": int,
    "description": str,
    "tags": list,
    "tick_minutes": int,
    "tick_seconds": (int, float),
    "clock_contract": dict,
}

VALID_DOMAINS = {
    "power_grid",
    "traffic",
    "microgrid",
    "logistics",
    "datacenter",
    "disaster",
    # Pilot-only native CityLearn integration.  Formal release coverage still
    # remains the five-domain Protocol-2.1 contract.
    "building_energy",
    # Protocol-2.2 pilot: vehicle-level tactical supervision uses a seconds
    # clock while historical domains retain their minute clock contract.
    "autonomous_driving",
}
PUBLIC_DIFFICULTY_MODES = {"time_pressure", "deep_planning"}
# Frozen Core / candidate YAMLs may stamp a source-locked long-horizon mode.
# New materializations should still use the public two-mode set.
LEGACY_DIFFICULTY_MODE_ALIASES = {"source_locked_long_horizon"}
VALID_DIFFICULTY_MODES = PUBLIC_DIFFICULTY_MODES | LEGACY_DIFFICULTY_MODE_ALIASES
PUBLIC_DIFFICULTY_LEVELS = {"basic", "medium", "high", "extreme"}
# Frozen-release compatibility only. New materializations must use the public set.
LEGACY_DIFFICULTY_LEVEL_ALIASES = {"extreme_plus", "cascading"}
VALID_DIFFICULTY_LEVELS = PUBLIC_DIFFICULTY_LEVELS | LEGACY_DIFFICULTY_LEVEL_ALIASES


def _validate_driving_clock(tick_seconds: int | float, clock: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if clock.get("schema_version") != "driving_clock_v1":
        errors.append("clock_contract.schema_version must be 'driving_clock_v1'")
    physics_step = clock.get("physics_step_seconds")
    shield_step = clock.get("shield_step_seconds")
    substeps = clock.get("substeps_per_supervisory_tick")
    if (
        not isinstance(physics_step, int | float)
        or isinstance(physics_step, bool)
        or not math.isfinite(float(physics_step))
        or physics_step <= 0
    ):
        errors.append("clock_contract.physics_step_seconds must be positive")
    if (
        not isinstance(shield_step, int | float)
        or isinstance(shield_step, bool)
        or not math.isfinite(float(shield_step))
        or shield_step <= 0
    ):
        errors.append("clock_contract.shield_step_seconds must be positive")
    elif isinstance(physics_step, int | float) and not math.isclose(
        float(shield_step), float(physics_step), rel_tol=0.0, abs_tol=1e-12
    ):
        errors.append("clock_contract.shield_step_seconds must equal physics_step_seconds")
    if not isinstance(substeps, int) or isinstance(substeps, bool) or substeps <= 0:
        errors.append("clock_contract.substeps_per_supervisory_tick must be a positive integer")
    elif (
        isinstance(physics_step, int | float)
        and physics_step > 0
        and not math.isclose(
            float(tick_seconds),
            float(physics_step) * substeps,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        errors.append("clock_contract.substeps_per_supervisory_tick does not match tick_seconds")
    if clock.get("provider_wall_clock_advances_simulation") is not False:
        errors.append("clock_contract.provider_wall_clock_advances_simulation must be false")
    return errors


def _standard_scenario_identity_from_path(
    source_path: str | Path | None,
) -> dict[str, str] | None:
    if source_path is None:
        return None
    path = Path(source_path)
    parts = path.as_posix().split("/")
    if len(parts) < 6:
        return None
    try:
        idx = parts.index("scenarios")
    except ValueError:
        return None
    tail = parts[idx + 1 :]
    if len(tail) != 5:
        return None
    domain, family, difficulty_mode, difficulty_level, filename = tail
    if domain not in VALID_DOMAINS:
        return None
    if difficulty_mode not in VALID_DIFFICULTY_MODES:
        return None
    if difficulty_level not in VALID_DIFFICULTY_LEVELS:
        return None
    if not filename.endswith(".yaml"):
        return None
    return {
        "domain": domain,
        "family": family,
        "difficulty_mode": difficulty_mode,
        "difficulty_level": difficulty_level,
    }


def canonical_scenario_slug_from_path(source_path: str | Path | None) -> str | None:
    expected_identity = _standard_scenario_identity_from_path(source_path)
    if expected_identity is None or source_path is None:
        return None
    filename_stem = Path(source_path).stem
    return "/".join(
        [
            expected_identity["domain"],
            expected_identity["family"],
            expected_identity["difficulty_mode"],
            expected_identity["difficulty_level"],
            filename_stem,
        ]
    )


def _allowed_seed_ids_for_canonical_path(
    source_path: str | Path | None,
) -> set[str] | None:
    expected_identity = _standard_scenario_identity_from_path(source_path)
    if expected_identity is None or source_path is None:
        return None
    filename_stem = Path(source_path).stem
    domain = expected_identity["domain"]
    family = expected_identity["family"]
    difficulty_mode = expected_identity["difficulty_mode"]
    difficulty_level = expected_identity["difficulty_level"]
    allowed = {filename_stem, canonical_scenario_slug_from_path(source_path)}
    if domain == "logistics":
        allowed.add(f"{family}/{difficulty_mode}/{difficulty_level}/{filename_stem}")
    return {value for value in allowed if isinstance(value, str)}


def validate_scenario_yaml(
    scenario: dict[str, Any], source_path: str | Path | None = None
) -> list[str]:
    """Validate a loaded scenario YAML dict against the expected schema.

    Returns a list of human-readable error strings. An empty list means
    the scenario passes all structural checks.

    Checks performed:
    - Required fields are present with correct types
    - Numeric fields have sensible positive values
    - Enum fields use known values
    - Optional fields (when present) have correct types
    """
    if not isinstance(scenario, dict):
        return [f"scenario YAML must deserialize to a mapping/dict, got {type(scenario).__name__}"]

    errors: list[str] = []

    # ---- Required fields ----
    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in scenario:
            errors.append(f"missing required field: {field}")
        elif not isinstance(scenario[field], expected_type):
            errors.append(
                f"field {field}: expected {expected_type.__name__}, "
                f"got {type(scenario[field]).__name__}"
            )

    # ---- Value range checks ----
    horizon = scenario.get("horizon_ticks")
    if isinstance(horizon, int) and horizon <= 0:
        errors.append(f"horizon_ticks must be positive, got {horizon}")

    if domain := scenario.get("domain"):
        if domain == "autonomous_driving":
            tick_seconds = scenario.get("tick_seconds")
            if (
                not isinstance(tick_seconds, int | float)
                or isinstance(tick_seconds, bool)
                or not math.isfinite(float(tick_seconds))
            ):
                errors.append("missing required field: tick_seconds")
            elif tick_seconds <= 0:
                errors.append(f"tick_seconds must be positive, got {tick_seconds}")
            if "tick_minutes" in scenario:
                errors.append("autonomous_driving uses tick_seconds, not tick_minutes")
            clock = scenario.get("clock_contract")
            if not isinstance(clock, dict):
                errors.append("missing required field: clock_contract")
            elif isinstance(tick_seconds, int | float) and tick_seconds > 0:
                errors.extend(_validate_driving_clock(tick_seconds, clock))
        else:
            tick_min = scenario.get("tick_minutes")
            if not isinstance(tick_min, int):
                errors.append("missing required field: tick_minutes")
            elif tick_min <= 0:
                errors.append(f"tick_minutes must be positive, got {tick_min}")

    seed_val = scenario.get("seed")
    if isinstance(seed_val, int) and seed_val < 0:
        errors.append(f"seed must be non-negative, got {seed_val}")

    # ---- Enum validation ----
    domain = scenario.get("domain")
    if isinstance(domain, str) and domain not in VALID_DOMAINS:
        errors.append(f"unknown domain: {domain!r} (valid: {sorted(VALID_DOMAINS)})")

    mode = scenario.get("difficulty_mode")
    if isinstance(mode, str) and mode not in VALID_DIFFICULTY_MODES:
        errors.append(
            f"unknown difficulty_mode: {mode!r} (valid: {sorted(VALID_DIFFICULTY_MODES)})"
        )

    level = scenario.get("difficulty_level")
    if isinstance(level, str) and level not in VALID_DIFFICULTY_LEVELS:
        errors.append(
            f"unknown difficulty_level: {level!r} (valid: {sorted(VALID_DIFFICULTY_LEVELS)})"
        )

    expected_identity = _standard_scenario_identity_from_path(source_path)
    if expected_identity is not None:
        for field, expected in expected_identity.items():
            actual = scenario.get(field)
            if isinstance(actual, str) and actual != expected:
                errors.append(f"path/body mismatch for {field}: path={expected!r}, body={actual!r}")

        allowed_seed_ids = _allowed_seed_ids_for_canonical_path(source_path)
        actual_seed_id = scenario.get("seed_id")
        if (
            allowed_seed_ids is not None
            and isinstance(actual_seed_id, str)
            and actual_seed_id not in allowed_seed_ids
        ):
            errors.append(
                "canonical seed_id mismatch: "
                f"expected one of {sorted(allowed_seed_ids)!r}, got {actual_seed_id!r}"
            )

        actual_scenario_id = scenario.get("scenario_id")
        expected_scenario_id = canonical_scenario_slug_from_path(source_path)
        if (
            expected_scenario_id is not None
            and isinstance(actual_scenario_id, str)
            and actual_scenario_id != expected_scenario_id
        ):
            errors.append(
                "canonical scenario_id mismatch: "
                f"expected {expected_scenario_id!r}, got {actual_scenario_id!r}"
            )

    # ---- Optional field type checks ----
    for field, expected_type in OPTIONAL_FIELDS.items():
        if (
            field in scenario
            and scenario[field] is not None
            and not isinstance(scenario[field], expected_type)
        ):
            errors.append(
                f"optional field {field}: expected {expected_type}, "
                f"got {type(scenario[field]).__name__}"
            )

    return errors
