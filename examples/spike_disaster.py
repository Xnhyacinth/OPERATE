#!/usr/bin/env python3
"""
examples/spike_disaster.py — v0.3 Phase 3.3 disaster end-to-end spike.

Runs a 24-tick deterministic-policy episode against the
``MockRcrsBackend`` (pure Python, no Docker) on the canonical
``urban_earthquake_M6_24h`` Kobe seed.

The policy is intentionally simple — "send 1 ambulance team each tick
to the zone with the highest reported buried count" — so output is
short and the per-tick reasoning is easy to audit:

- Per-tick zone summary (one line: target zone, buried, fire).
- Final per-zone ``unserved_minutes`` map.
- Final ``cost_components`` dict.

Runnable from repo root WITHOUT Docker / Java / OpenQuake:

    python examples/spike_disaster.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core import Action, ToolCall  # noqa: E402
from domains.disaster.adapter import DisasterEnvironment  # noqa: E402
from domains.disaster.seeds.from_rcrs_kobe import (  # noqa: E402
    build_urban_earthquake_M6_24h_kobe_seed,
)


def _hottest_zone(snap: dict[str, Any]) -> str | None:
    """Pick the zone with the highest reported buried count.

    Defensive: any zone whose ``buried`` is hidden by the fog policy
    reports None — skip those. Returns None when no zone has a
    visible positive buried count (then the spike issues a ``wait``).
    """
    entities = snap.get("entities", {}) or {}
    best_zone: str | None = None
    best_buried = 0
    for zid, ent in entities.items():
        if ent.get("kind") != "zone":
            continue
        buried = ent.get("buried")
        if not isinstance(buried, (int, float)):
            continue
        if buried > best_buried:
            best_buried = int(buried)
            best_zone = zid
    return best_zone


def main(*, n_ticks: int = 24, verbose: bool = True) -> dict[str, Any]:
    """Run the spike and return a result dict.

    Public so ``tests/test_disaster_skeleton.py`` can call ``main()``
    and assert the result is non-empty.
    """
    seed = build_urban_earthquake_M6_24h_kobe_seed(
        seed_id="spike_kobe_v0.3",
        seed=42,
        difficulty_level="medium",
        difficulty_mode="time_pressure",
    )
    env = DisasterEnvironment()
    obs = env.reset(seed.to_dict(), seed=seed.seed)

    if verbose:
        print(
            f"# spike_disaster: seed={seed.seed_id} horizon={env.horizon} "
            f"zones={len(obs.get('entities', {}))}"
        )

    total_ticks = min(n_ticks, env.horizon)
    tick_lines: list[str] = []
    for _ in range(total_ticks):
        snap = env.snapshot()
        target = _hottest_zone(snap)
        if target is None:
            action = Action(tool_calls=[ToolCall(name="wait")], dominant="wait")
        else:
            action = Action(
                tool_calls=[
                    ToolCall(
                        name="dispatch_ambulance",
                        args={"target_zone": target, "n_teams": 1},
                    )
                ],
                dominant="dispatch_ambulance",
            )
        ret = env.step(action)
        # One-line per-tick summary — kept compact so 24 ticks fits in
        # well under 30 lines of output.
        totals = ret.observation.get("totals", {})
        tick_lines.append(
            f"t={env.tick:02d} target={target or '-'} "
            f"buried={totals.get('aggregate_buried', 0)} "
            f"fire={totals.get('aggregate_fire_intensity', 0)} "
            f"teams={totals.get('aggregate_dispatched_teams', 0)}"
        )
        if ret.done:
            break

    gt = env.ground_truth()
    per_zone = gt.get("per_zone_unserved_minutes", {})
    costs = gt.get("cost_components", {})

    if verbose:
        for line in tick_lines:
            print(line)
        print("# per-zone unserved minutes:")
        for zid in sorted(per_zone.keys()):
            print(f"  {zid:18s} {per_zone[zid]:.1f}")
        print("# cost components:")
        for k in sorted(costs.keys()):
            print(f"  {k:28s} {costs[k]:.2f}")

    return {
        "seed_id": seed.seed_id,
        "horizon": env.horizon,
        "ticks_run": min(total_ticks, env.tick),
        "per_zone_unserved_minutes": per_zone,
        "cost_components": costs,
        "tick_lines": tick_lines,
    }


if __name__ == "__main__":  # pragma: no cover
    main()
