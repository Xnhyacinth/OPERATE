#!/usr/bin/env python3
"""Run an explicit native SUMO smoke for the autonomous-driving sibling.

This probe validates the SUMO/TraCI bridge and the backend-owned shield. Core
eligibility is decided from source, replay, safety, and headroom evidence, not
the host operating system or CPU architecture.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.sidecar.sumo_sidecar import probe_sumo_transport  # noqa: E402
from domains.autonomous_driving.adapter import DrivingSeed  # noqa: E402
from domains.autonomous_driving.backends.live_sumo_ego import LiveSumoEgoBackend  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--net", type=Path, required=True)
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--ego", required=True)
    parser.add_argument("--source-bundle", type=Path)
    parser.add_argument("--candidate-id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ticks", type=int, default=2)
    parser.add_argument("--tick-seconds", type=float, default=5.0)
    parser.add_argument("--physics-step-seconds", type=float, default=0.1)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = (args.config, args.net, args.route)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"live SUMO asset missing: {', '.join(missing)}")
    if args.ticks < 1 or args.ticks > 120:
        raise ValueError("--ticks must be between 1 and 120")
    if args.tick_seconds <= 0.0 or args.physics_step_seconds <= 0.0:
        raise ValueError("clock durations must be positive")
    substeps = round(args.tick_seconds / args.physics_step_seconds)
    if substeps < 1 or abs(substeps * args.physics_step_seconds - args.tick_seconds) > 1e-9:
        raise ValueError("tick-seconds must be an exact multiple of physics-step-seconds")

    os.environ["OPERATE_AUTONOMOUS_DRIVING_SUMO_REAL"] = "1"
    transport = probe_sumo_transport()
    if transport not in {"libsumo", "traci"}:
        raise RuntimeError("no native SUMO transport is available")

    seed = DrivingSeed(
        seed_id=f"live-sumo-smoke:{args.ego}",
        family="sustained_highway_risk_supervision",
        horizon_ticks=args.ticks,
        tick_seconds=args.tick_seconds,
        seed=args.seed,
        clock_contract={
            "schema_version": "driving_clock_v1",
            "physics_step_seconds": args.physics_step_seconds,
            "shield_step_seconds": args.physics_step_seconds,
            "substeps_per_supervisory_tick": substeps,
            "provider_wall_clock_advances_simulation": False,
        },
    )
    backend = LiveSumoEgoBackend(
        {
            "execution_mode": "live",
            "sumo_config_path": str(args.config.resolve()),
            "sumo_net_path": str(args.net.resolve()),
            "sumo_route_path": str(args.route.resolve()),
            "ego_vehicle_id": args.ego,
            "source_bundle": str(args.source_bundle.resolve()) if args.source_bundle else None,
            "candidate_id": args.candidate_id,
            "physics_step_seconds": args.physics_step_seconds,
            "physics_substeps_per_tick": substeps,
        }
    )
    started = time.monotonic()
    records: list[dict[str, Any]] = []
    sumo_version: str | None = None
    startup_stdout = io.StringIO()
    try:
        with redirect_stdout(startup_stdout):
            backend.reset(seed)
        sumo_version = backend.native_runtime_version
        for tick in range(args.ticks):
            records.append(backend.tick(tick).to_dict())
        source_evidence = backend.source_consumption_evidence()
    finally:
        backend.close()
    final = records[-1]
    return {
        "schema_version": "autonomous_driving_live_sumo_smoke_v1",
        "status": "verified_native_smoke",
        "admission": "portable_native_smoke",
        "transport": backend.native_transport or transport,
        "sumo_version": sumo_version,
        "startup_messages": startup_stdout.getvalue().splitlines(),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "policy": "cross_platform_runtime_fingerprint_v1",
        },
        "seed": args.seed,
        "ego_vehicle_id": args.ego,
        "ticks": args.ticks,
        "tick_seconds": args.tick_seconds,
        "physics_step_seconds": args.physics_step_seconds,
        "records": records,
        "source_consumption_evidence": source_evidence,
        "final": {
            "collision_count": final["collision_count"],
            "road_departure_count": final["road_departure_count"],
            "shield_intervention_count": final["shield_intervention_count"],
            "mrm_active": final["mrm_active"],
        },
        "wall_clock_seconds_diagnostic": round(time.monotonic() - started, 6),
    }


def main() -> int:
    args = _parser().parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
