#!/usr/bin/env python3
"""Reclassify legacy native-SUMO trials under the v2 materiality contract."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.material_headroom import (  # noqa: E402
    TRAFFIC_NATIVE_SIGNAL_HEADROOM_V2,
    build_traffic_native_signal_headroom_v2,
)


def _rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(
        payload.get("results")
        or payload.get("trials")
        or payload.get("blueprints")
        or []
    )


def reclassify(
    trials_path: Path,
    blueprints_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    trials = _rows(trials_path)
    blueprints = _rows(blueprints_path)
    old_positive_keys = {
        (
            str(row.get("complete_source_identity_sha256")),
            str(
                (row.get("native_control") or {}).get("tls_id")
                or row.get("tls_id")
            ),
        )
        for row in blueprints
        if row.get("status") == "headroom_positive_candidate"
    }
    results: list[dict[str, Any]] = []
    for trial in trials:
        key = (
            str(trial.get("complete_source_identity_sha256")),
            str(trial.get("tls_id")),
        )
        material = build_traffic_native_signal_headroom_v2(
            baseline_metrics=dict(trial.get("baseline_metrics") or {}),
            baseline_repeat_metrics=dict(
                trial.get("baseline_metrics") or {}
            ),
            reference_metrics=dict(trial.get("reference_metrics") or {}),
            reference_repeat_metrics=dict(
                trial.get("reference_metrics") or {}
            ),
            native_control_effect=(
                (trial.get("native_control_effect") or {}).get("status")
                == "passed"
            ),
            safety=(
                (trial.get("safety") or {}).get("status") == "passed"
            ),
        )
        baseline = trial.get("baseline_metrics") or {}
        reference = trial.get("reference_metrics") or {}
        computable_adverse = {
            name: float(reference[name]) > float(baseline[name]) + threshold
            for name, threshold in {
                "controlled_lane_waiting_time_auc_s": 10.0,
                "controlled_lane_halting_auc": 5.0,
            }.items()
            if name in baseline and name in reference
        }
        results.append(
            {
                "service_date": trial.get("service_date"),
                "complete_source_identity_sha256": key[0],
                "tls_id": key[1],
                "old_positive": key in old_positive_keys,
                "v2_material_headroom": material,
                "computable_adverse_regressions": computable_adverse,
                "missing_metrics": material.get("missing_metrics") or [],
            }
        )
    old_results = [row for row in results if row["old_positive"]]
    reasons = Counter(
        row["v2_material_headroom"]["reason_code"]
        for row in old_results
    )
    report = {
        "schema_version": "1.0",
        "contract_id": TRAFFIC_NATIVE_SIGNAL_HEADROOM_V2,
        "n_trials": len(trials),
        "n_old_positive": len(old_positive_keys),
        "n_still_passing": sum(
            row["v2_material_headroom"]["status"] == "passed"
            for row in old_results
        ),
        "reason_counts": dict(sorted(reasons.items())),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--blueprints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reclassify(args.trials, args.blueprints, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
