"""Traffic-domain scenario validation helpers."""

from __future__ import annotations

from typing import Any


def validate_traffic_scenario_yaml(scenario: dict[str, Any]) -> list[str]:
    """Return traffic-specific validation errors for a loaded scenario mapping."""
    if scenario.get("family") != "vip_priority_dilemma":
        return []

    errors: list[str] = []
    perturbations = scenario.get("perturbations")
    if not isinstance(perturbations, list):
        return ["vip_priority_dilemma requires a perturbations list"]
    perturbation_kinds = {
        item.get("kind") for item in perturbations if isinstance(item, dict)
    }
    for required_kind in ("vip_arrival", "ems_corridor_request"):
        if required_kind not in perturbation_kinds:
            errors.append(
                f"vip_priority_dilemma requires perturbation kind: {required_kind}"
            )

    vip_tick = next(
        (
            item.get("trigger_tick")
            for item in perturbations
            if isinstance(item, dict) and item.get("kind") == "vip_arrival"
        ),
        None,
    )
    ems_tick = next(
        (
            item.get("trigger_tick")
            for item in perturbations
            if isinstance(item, dict) and item.get("kind") == "ems_corridor_request"
        ),
        None,
    )

    dilemmas = scenario.get("dilemmas")
    if not isinstance(dilemmas, list) or not dilemmas:
        errors.append("vip_priority_dilemma requires a non-empty dilemmas list")
        return errors

    dilemma = next(
        (
            item
            for item in dilemmas
            if isinstance(item, dict) and item.get("dilemma_id") == "vip_priority_dilemma"
        ),
        None,
    )
    if dilemma is None:
        errors.append(
            "vip_priority_dilemma requires a dilemma with dilemma_id='vip_priority_dilemma'"
        )
        return errors

    if isinstance(vip_tick, int) and isinstance(ems_tick, int):
        if vip_tick != ems_tick:
            errors.append(
                "vip_priority_dilemma requires VIP and EMS requests to share a trigger_tick"
            )
        elif dilemma.get("trigger_tick") != vip_tick:
            errors.append(
                "vip_priority_dilemma requires dilemma trigger_tick to match the live VIP/EMS request tick"
            )
    return errors
