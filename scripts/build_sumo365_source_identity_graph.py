#!/usr/bin/env python3
"""Build an independent SUMO365 source-identity graph for native mining."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_sumo_source_graph import audit_sumocfg  # noqa: E402

DEFAULT_ROOT = (
    REPO_ROOT / "works/sumo_ingolstadt/simulation/Ingolstadt SUMO 365"
)


def build_graph(
    *,
    sumocfg_root: Path,
    service_dates: list[str],
    sumo_version: str,
) -> dict[str, Any]:
    """Audit exact recursive asset graphs and emit the miner crosscheck shape."""
    results: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for service_date in sorted(set(service_dates)):
        config = sumocfg_root / f"{service_date}.sumocfg"
        report = audit_sumocfg(
            config,
            service_date=service_date,
            sumo_version=sumo_version,
        )
        if report.get("status") == "blocked":
            results.append(
                {
                    "service_date": service_date,
                    "work_state": "terminal",
                    "status": "blocked",
                    "reason_code": report.get("reason_code")
                    or "traffic_source_identity_mismatch",
                }
            )
            continue
        identity = report["source_identity"]
        candidates.append(
            {
                "service_date": service_date,
                "complete_source_identity_sha256": identity["sha256"],
                "complete_source_identity_payload": identity["payload"],
                "source_graph": report["source_graph"],
                "work_state": "terminal",
                "status": "source_identity_locked",
            }
        )
        results.append(
            {
                "service_date": service_date,
                "work_state": "terminal",
                "status": "source_identity_locked",
            }
        )
    return {
        "schema_version": "sumo365-independent-source-identity-graph-v1",
        "status": "complete" if len(candidates) == len(results) else "blocked",
        "metadata": {
            "sumo_version": sumo_version,
            "transport": "traci_tcp",
        },
        "n_expected": len(results),
        "n_terminal": len(results),
        "candidate_configs": candidates,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sumocfg-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--service-dates", nargs="+", required=True)
    parser.add_argument("--sumo-version", default="SUMO 1.27.1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_graph(
        sumocfg_root=args.sumocfg_root.resolve(),
        service_dates=args.service_dates,
        sumo_version=args.sumo_version,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
