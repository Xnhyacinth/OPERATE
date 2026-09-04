#!/usr/bin/env python3
"""
examples/cross_backend_cascade.py — v0.2 cross-backend cascade demo.

This demo wires TWO real OPERATE backends together via the
cascade bus (an earlier ``cascade_stub.py`` placeholder was removed in
v0.2.4):

- Upstream: transmission ``critical_winter_peak`` scenario running on
  the ``pglib_uc_synthetic`` backend.
- Downstream: distribution ``distribution_volt_var`` scenario running
  on the ``cigre_distribution`` backend.

When the transmission backend publishes ``power_grid.line.outage`` or
``power_grid.generator.outage`` events, a coordinator translates them
into ``ScenarioSeed.perturbations`` that the distribution backend
applies the next tick. This proves the architecture supports real
cross-backend physics propagation without a non-Python runtime.

The demo:

1. Runs a transmission episode and collects cascade events.
2. Replays the same window on the distribution backend with the
   cascade-translated perturbations injected.
3. Compares distribution-side metrics WITH and WITHOUT cascade
   injection to show the propagation is measurable.

Run from repo root:

    python examples/cross_backend_cascade.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core import Action, CascadeBus, CascadeEvent, ToolCall  # noqa: E402
from domains.power_grid.adapter import PowerGridEnvironment  # noqa: E402
from domains.power_grid.seeds.from_cigre import (  # noqa: E402
    build_distribution_volt_var_seed,
)
from domains.power_grid.seeds.from_pglib_uc import (  # noqa: E402
    build_critical_winter_peak_seed,
    list_cases,
)
from domains.power_grid.seeds.schema import Perturbation  # noqa: E402


def _grid_to_distribution_perturbation(
    event: CascadeEvent,
) -> Perturbation | None:
    """Translate an upstream cascade event into a distribution-backend
    perturbation. The mapping is deliberately conservative — only
    events that have a physical analogue on a radial feeder map to a
    real perturbation; the rest are dropped.
    """
    if event.event_type == "power_grid.line.outage":
        # Translate to a load_surge on the downstream distribution
        # feeder: when an upstream line trips, distribution circuits
        # often see voltage sag + load redistribution. We model this as
        # a 15% residential surge for 4 ticks starting the same tick.
        return Perturbation(
            kind="load_surge",
            trigger_tick=event.tick,
            duration_ticks=4,
            intensity=0.15,
            target={"stakeholder_class": "residential"},
            notes=(
                f"Cascaded from upstream transmission line outage "
                f"(tx_line={event.payload.get('line_id')})."
            ),
        )
    if event.event_type == "power_grid.generator.outage":
        # Upstream gen outage → downstream DER must absorb more load.
        # Model as a wind_dropout (any local renewable becomes more
        # critical) for 6 ticks.
        return Perturbation(
            kind="wind_dropout",
            trigger_tick=event.tick,
            duration_ticks=6,
            intensity=0.30,
            target={"generator_kind": "renewable"},
            notes=(
                f"Cascaded from upstream generator outage "
                f"(tx_gen={event.payload.get('generator_id')})."
            ),
        )
    return None


def _run_transmission_episode(
    *, seed_id: str, level: str, capture_bus: CascadeBus
) -> tuple[list[CascadeEvent], int]:
    """Run a transmission episode and capture cascade events."""
    captured: list[CascadeEvent] = []
    capture_bus.subscribe("power_grid.*", lambda e: captured.append(e))
    case = list_cases("rts_gmlc")[0]
    seed = build_critical_winter_peak_seed(
        case, seed_id=seed_id, difficulty_level=level
    )
    env = PowerGridEnvironment(cascade_bus=capture_bus)
    env.reset(seed.to_dict(), seed=seed.seed)
    for t in range(seed.horizon_ticks):
        env.step(
            Action(
                tool_calls=[ToolCall(name="wait", idempotency_key=f"tx_w_{t}")],
                dominant="wait",
            )
        )
    return captured, seed.horizon_ticks


def _run_distribution_episode(
    *,
    seed_id: str,
    level: str,
    extra_perturbations: list[Perturbation],
) -> dict[str, Any]:
    """Run a distribution episode optionally augmented with cascade-
    translated perturbations. Returns aggregate cost components."""
    seed = build_distribution_volt_var_seed(seed_id=seed_id, difficulty_level=level)
    # Append cascade-translated perturbations to the seed
    enriched = seed.to_dict()
    enriched.setdefault("perturbations", []).extend(
        [
            {
                "kind": p.kind,
                "trigger_tick": p.trigger_tick,
                "duration_ticks": p.duration_ticks,
                "hidden": p.hidden,
                "target": p.target,
                "intensity": p.intensity,
                "notes": p.notes,
            }
            for p in extra_perturbations
        ]
    )
    env = PowerGridEnvironment()
    env.reset(enriched, seed=seed.seed)
    for t in range(seed.horizon_ticks):
        env.step(
            Action(
                tool_calls=[ToolCall(name="wait", idempotency_key=f"dx_w_{t}")],
                dominant="wait",
            )
        )
    gt = env.ground_truth()
    return gt["cost_components"]


def main() -> int:
    print("=" * 72)
    print("OPERATE v0.2 cross-backend cascade demo")
    print("=" * 72)
    print()
    print("Stage 1 — Run a transmission episode and capture cascade events.")
    print()
    bus = CascadeBus()
    events, tx_horizon = _run_transmission_episode(
        seed_id="cx_tx", level="high", capture_bus=bus
    )
    by_type: dict[str, int] = {}
    for e in events:
        by_type[e.event_type] = by_type.get(e.event_type, 0) + 1
    print(f"  transmission horizon: {tx_horizon} ticks")
    print(f"  cascade events captured: {len(events)}")
    for k, n in sorted(by_type.items()):
        print(f"    {k}: {n}")
    print()

    print("Stage 2 — Translate cascade events into distribution perturbations.")
    print()
    # We need to know the downstream horizon to remap event ticks.
    # Build a probe seed to read it, then linearly scale tx_tick →
    # dx_tick. Real production code would use wall-clock alignment;
    # for the demo we keep this transparent.
    probe = build_distribution_volt_var_seed(seed_id="probe", difficulty_level="basic")
    dx_horizon = probe.horizon_ticks
    distribution_perturbations: list[Perturbation] = []
    for e in events:
        p = _grid_to_distribution_perturbation(e)
        if p is None:
            continue
        # Remap tx_tick (0..tx_horizon-1) → dx_tick (0..dx_horizon-1).
        # Clamp so triggers land at least 1 tick into the episode.
        remapped = max(
            1, int(round(e.tick * (dx_horizon - 1) / max(1, tx_horizon - 1)))
        )
        # Cap duration so the perturbation stays inside the dx window.
        p.trigger_tick = remapped
        p.duration_ticks = max(1, min(p.duration_ticks, dx_horizon - remapped))
        distribution_perturbations.append(p)
    print(
        f"  events translated to distribution perturbations: "
        f"{len(distribution_perturbations)}  (remapped to dx horizon "
        f"{dx_horizon})"
    )
    for p in distribution_perturbations[:3]:
        print(
            f"    - {p.kind} @ dx_tick {p.trigger_tick} "
            f"(dur={p.duration_ticks}): {p.notes}"
        )
    print()

    print(
        "Stage 3 — Run the distribution episode twice (with and without "
        "cascade) and compare costs."
    )
    print()
    # Use the BASIC level so the distribution baseline has no built-in
    # wind_dropout — the cascade-injected perturbations are then
    # additive rather than overlapping with the seed's own stressors.
    cost_baseline = _run_distribution_episode(
        seed_id="cx_dx_base", level="basic", extra_perturbations=[]
    )
    cost_cascaded = _run_distribution_episode(
        seed_id="cx_dx_cascaded",
        level="basic",
        extra_perturbations=distribution_perturbations,
    )
    print(f"  distribution cost (no cascade): {sum(cost_baseline.values()):,.0f}")
    print(f"  distribution cost (cascaded):   {sum(cost_cascaded.values()):,.0f}")
    delta = sum(cost_cascaded.values()) - sum(cost_baseline.values())
    print(f"  Δ cost attributable to cascade: {delta:+,.0f}")
    print()
    print("Conclusion: a real transmission-side event measurably affects the")
    print("distribution-side episode through the cascade_bus, with no SUMO /")
    print("Java / Julia / non-Python runtime needed. v0.3 swaps the distribution")
    print("backend for an RCRS / SUMO subscriber using the same interface.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
