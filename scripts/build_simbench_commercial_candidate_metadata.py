#!/usr/bin/env python3
"""Build one current-tree SimBench commercial candidate metadata record.

The historical native-prefilter summary is a discovery hint only.  This
builder re-resolves the constructor and profile-window identities from the
current SimBench installation and makes no source-consumption, headroom,
safety, replay, or release-admission claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.source_asset_contract import virtual_source_identity_sha256  # noqa: E402
from domains.power_grid.seeds.source_locks import SOURCE_LOCKS  # noqa: E402
from scripts.grounded_candidate_pipeline import rank_simbench_windows  # noqa: E402


DEFAULT_HISTORICAL_SUMMARY = (
    REPO_ROOT / ".hl/artifacts/simbench_plus5_extra_20260815/summary.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / ".hl/artifacts/operate_v058_simbench_commercial_p19680_candidate_metadata.json"
)
TARGET_NETWORK = "simbench:1-MV-comm--0-sw"
TARGET_PROFILE_START_INDEX = 19680
HORIZON_TICKS = 48
PROFILE_STEP = 4
SCENARIO_ID = (
    "power_grid/simbench_mv_commercial_timeseries_control/deep_planning/high/"
    "grounded_1_MV_comm_0_sw_high_p19680_r0p6_h48"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _historical_row(summary: dict[str, Any]) -> dict[str, Any]:
    rows = summary.get("per_window")
    if not isinstance(rows, list):
        raise ValueError("historical summary per_window must be a list")
    matches = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("network") == TARGET_NETWORK
        and int(row.get("profile_start_index") or -1)
        == TARGET_PROFILE_START_INDEX
    ]
    if len(matches) != 1:
        raise ValueError("exactly one historical SimBench commercial p19680 row required")
    return matches[0]


def resolve_current_window() -> dict[str, Any]:
    """Resolve p19680 from current source-bundled SimBench profiles."""
    windows = rank_simbench_windows(
        TARGET_NETWORK,
        horizon_ticks=HORIZON_TICKS,
        limit=128,
    )
    matches = [
        row
        for row in windows
        if int(row.get("profile_start_index") or -1)
        == TARGET_PROFILE_START_INDEX
    ]
    if len(matches) != 1:
        raise ValueError("current SimBench p19680 window is missing or ambiguous")
    return matches[0]


def build_candidate_metadata(
    *,
    historical_summary_path: Path,
    runtime_version: str,
    current_window: dict[str, Any],
) -> dict[str, Any]:
    """Return one candidate-only structural prefilter record."""
    summary = _load_object(historical_summary_path)
    historical = _historical_row(summary)
    if (
        int(current_window.get("profile_start_index") or -1)
        != TARGET_PROFILE_START_INDEX
        or not current_window.get("source_window_sha256")
    ):
        raise ValueError("current SimBench window identity is not p19680")

    dataset_code = TARGET_NETWORK.split(":", 1)[1]
    constructor_uri = f"pandapower-simbench://{dataset_code}@{runtime_version}"
    constructor_sha256 = virtual_source_identity_sha256(constructor_uri)
    if constructor_sha256 is None:
        raise ValueError("current SimBench constructor identity is not lockable")
    source_window_sha256 = str(current_window["source_window_sha256"])
    dataset_lock = SOURCE_LOCKS["simbench"]
    source_identity = {
        "constructor_identity_sha256": constructor_sha256,
        "constructor_uri": constructor_uri,
        "dataset_code": dataset_code,
        "dataset_commit": dataset_lock.commit,
        "dataset_release": dataset_lock.data_release,
        "dataset_version": dataset_lock.version,
        "horizon_ticks": HORIZON_TICKS,
        "network": TARGET_NETWORK,
        "profile_start_index": TARGET_PROFILE_START_INDEX,
        "profile_step": PROFILE_STEP,
        "runtime_version": runtime_version,
        "source_window_sha256": source_window_sha256,
    }
    physical_source_key = json.dumps(
        {
            "backend_kind": "cigre_distribution",
            "required_source_assets": [
                {
                    "declared_path": constructor_uri,
                    "sha256": constructor_sha256,
                }
            ],
            "schema_version": "source_asset_graph_v1",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    effective_source_key = f"simbench_official:{_stable_hash(source_identity)}"
    historical_fields = {
        key: historical.get(key)
        for key in (
            "native_headroom",
            "native_prefilter_pass",
            "reference_native_loss",
            "scenario_id",
            "source_window_sha256",
            "wait_native_loss",
        )
    }
    return {
        "schema_version": "operate-simbench-commercial-candidate-metadata-v1",
        "status": "complete_structural_prefilter",
        "candidate_only": True,
        "release_admission": False,
        "executes_replay": False,
        "bindings": {
            "historical_summary_path": str(historical_summary_path.resolve()),
            "historical_summary_sha256": _sha256(historical_summary_path),
        },
        "policy": {
            "current_tree_source_consumption_required": True,
            "current_tree_behavioral_headroom_required": True,
            "historical_headroom_is_admission_evidence": False,
        },
        "candidates": [
            {
                "candidate_id": SCENARIO_ID,
                "candidate_only": True,
                "release_admission": False,
                "domain": "power_grid",
                "backend_kind": "cigre_distribution",
                "source_family": "simbench_mv_commercial_timeseries_control",
                "physical_source_key": physical_source_key,
                "source_denominator_key": effective_source_key,
                "source_identity": source_identity,
                "structural_axes": [
                    "commercial_mv_topology",
                    "locked_profile_window:p19680",
                    "volt_var_and_redispatch_control",
                ],
                "quality": {
                    "behavioral_headroom": "unknown",
                    "safety": "unknown",
                    "source_consumption": "unknown",
                },
                "disposition": "pending_current_tree_validation",
                "next_stage": "current_tree_source_consumption_and_behavioral_prefilter",
                "historical_evidence": {
                    "status": "historical_unverified_current_tree",
                    **historical_fields,
                },
            }
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--historical-summary",
        type=Path,
        default=DEFAULT_HISTORICAL_SUMMARY,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_candidate_metadata(
        historical_summary_path=args.historical_summary,
        runtime_version=importlib.metadata.version("simbench"),
        current_window=resolve_current_window(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"n_candidates": len(report["candidates"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
