from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from domains.traffic.runtime_control_contract import (  # noqa: E402
    SAFETY_CAUSALITY_SCHEMA_VERSION,
    SAFETY_EVENT_IDENTITY_MODE,
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(
    *,
    summary_path: Path,
    trials_path: Path,
    manifest_path: Path,
    source_identity_crosscheck_path: Path,
) -> dict[str, Any]:
    summary = _load(summary_path)
    trials_payload = _load(trials_path)
    manifest = _load(manifest_path)
    source_crosscheck = _load(source_identity_crosscheck_path)
    if not (
        source_crosscheck.get("scope_kind") == "bounded_request"
        and source_crosscheck.get("all_match") is True
    ):
        raise ValueError(
            "source identity crosscheck must pass for bounded_request"
        )
    required = (
        "trial_plan_sha",
        "implementation_tree_sha",
        "source_identity_sha",
    )
    missing = [name for name in required if not manifest.get(name)]
    if missing:
        raise ValueError(
            "execution manifest missing required bindings: "
            + ", ".join(missing)
        )
    trials = list(trials_payload.get("results") or [])
    release_positives = sum(
        row.get("release_candidate_status") == "passed"
        for row in trials
    )
    diagnostic_positives = sum(
        (row.get("diagnostic_headroom_without_safety") or {}).get(
            "status"
        )
        == "passed"
        for row in trials
    )
    initialization_only = sum(
        bool(
            (row.get("safety") or {}).get(
                "initialization_background_events_present"
            )
        )
        and (row.get("safety") or {}).get("status") == "passed"
        for row in trials
    )
    source_background = sum(
        (row.get("safety") or {}).get("reason_code")
        == "traffic_source_safety_background_violation"
        for row in trials
    )
    control_regression = sum(
        (row.get("safety") or {}).get("reason_code")
        == "traffic_control_safety_regression"
        for row in trials
    )
    safety_passed = sum(
        (row.get("safety") or {}).get("status") == "passed"
        for row in trials
    )
    return {
        "schema_version": "1.0",
        "scope": "bounded_probe",
        "status": (
            "diagnostic_candidates_available"
            if diagnostic_positives
            else "blocked"
        ),
        "trial_count": len(trials),
        "positive_count": release_positives,
        "release_positive_count": release_positives,
        "diagnostic_headroom_positive_count": diagnostic_positives,
        "initialization_background_only_trial_count": (
            initialization_only
        ),
        "source_safety_background_trial_count": source_background,
        "control_safety_regression_trial_count": control_regression,
        "safety_passed_trial_count": safety_passed,
        "conclusion": (
            "positive_found_within_bounded_probe"
            if release_positives
            else "no_positive_found_within_bounded_probe"
        ),
        "summary_status": summary.get("status"),
        "bindings": {
            "trial_plan_sha": manifest["trial_plan_sha"],
            "execution_manifest_sha": hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            "implementation_tree_sha": manifest[
                "implementation_tree_sha"
            ],
            "source_identity_sha": manifest["source_identity_sha"],
            "source_identity_crosscheck_file_sha256": hashlib.sha256(
                source_identity_crosscheck_path.read_bytes()
            ).hexdigest(),
            "safety_causality_schema_version": (
                SAFETY_CAUSALITY_SCHEMA_VERSION
            ),
            "safety_event_identity_mode": (
                SAFETY_EVENT_IDENTITY_MODE
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--source-identity-crosscheck",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        summary_path=args.summary,
        trials_path=args.trials,
        manifest_path=args.manifest,
        source_identity_crosscheck_path=(
            args.source_identity_crosscheck
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
