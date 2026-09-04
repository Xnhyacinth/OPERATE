#!/usr/bin/env python3
"""Report truthfully actionable external autonomous-driving source routes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

SOURCE_ROUTES: tuple[dict[str, Any], ...] = (
    {
        "source_id": "waymo_open_motion",
        "dataset_version": "1.3.1 (October 2025)",
        "official_url": "https://waymo.com/open/download/",
        "terms_url": "https://waymo.com/open/terms/",
        "runtime_target": "waymax",
        "runtime_repository_head": "a64dfec9be8576b60d9cecc94f406d9812d4a7d0",
        "access": "login and acceptance of non-commercial dataset terms required",
        "source_role": "closed_loop_branching_source",
        "source_type_core_eligible": True,
    },
    {
        "source_id": "nuplan",
        "dataset_version": "v1.1; devkit v1.2.2",
        "official_url": "https://www.nuplan.org/nuplan",
        "terms_url": "https://www.nuscenes.org/terms-of-use",
        "runtime_target": "nuplan",
        "runtime_repository_head": "e9241677997dd86bfc0bcd44817ab04fe631405b",
        "access": "registration and non-commercial dataset terms required",
        "source_role": "closed_loop_planning_source",
        "source_type_core_eligible": True,
    },
    {
        "source_id": "highd",
        "dataset_version": "60 recordings / 110,500 vehicles",
        "official_url": "https://levelxdata.com/highd-dataset/",
        "terms_url": "https://levelxdata.com/highd-dataset/",
        "runtime_target": "",
        "runtime_repository_head": "719cf28be4ae0c2a82314176e0712d636a3854ec",
        "access": "manual application; non-commercial; no onward raw-data sharing",
        "source_role": "naturalistic_trajectory_source",
        "source_type_core_eligible": True,
        "static_blockers": ["native_adapter_missing"],
    },
    {
        "source_id": "commonroad_reach",
        "dataset_version": "2025.2.1",
        "official_url": "https://github.com/CommonRoad/commonroad-reachable-set",
        "terms_url": "https://github.com/CommonRoad/commonroad-reachable-set/blob/main/LICENSE",
        "runtime_target": "commonroad_reach",
        "runtime_repository_head": "72ad8ab9",
        "access": "public BSD-3-Clause code; Python >=3.10,<3.12",
        "source_role": "shadow_validator",
        "source_type_core_eligible": False,
        "requires_source_asset": False,
        "static_blockers": ["shadow_validator_integration_pending"],
    },
    {
        "source_id": "commonroad_crime",
        "dataset_version": "current official repository head",
        "official_url": "https://github.com/CommonRoad/commonroad-crime",
        "terms_url": "https://github.com/CommonRoad/commonroad-crime/blob/main/LICENSE",
        "runtime_target": "commonroad_crime",
        "runtime_repository_head": "60bebed8",
        "access": "public BSD-3-Clause code; CommonRoad Reach dependency",
        "source_role": "shadow_monitor",
        "source_type_core_eligible": False,
        "requires_source_asset": False,
        "static_blockers": ["shadow_monitor_integration_pending"],
    },
    {
        "source_id": "carla_rss",
        "dataset_version": "CARLA development line; RSS remains experimental",
        "official_url": "https://github.com/carla-simulator/carla",
        "terms_url": "https://carla.readthedocs.io/en/latest/adv_rss/",
        "runtime_target": "carla",
        "runtime_repository_head": "0a5ce0d5",
        "access": "public code with separately licensed simulator assets",
        "source_role": "dev_only_simulator",
        "source_type_core_eligible": False,
        "requires_source_asset": False,
        "static_blockers": ["rss_runtime_integration_pending", "dev_only_runtime"],
    },
    {
        "source_id": "bench2drive",
        "dataset_version": "0.0.4",
        "official_url": "https://github.com/Thinklab-SJTU/Bench2Drive",
        "terms_url": "https://github.com/Thinklab-SJTU/Bench2Drive/blob/0.0.4/LICENSE",
        "runtime_target": "carla",
        "runtime_repository_head": "7ec25d1c",
        "access": "CARLA 0.9.15 synthetic benchmark; non-commercial terms",
        "source_role": "method_transfer_only",
        "source_type_core_eligible": False,
        "requires_source_asset": False,
        "static_blockers": [
            "synthetic_source_not_core_eligible",
            "method_transfer_only",
        ],
    },
)


def build_external_source_report(
    *, runtime_modules: dict[str, bool], local_assets: dict[str, bool]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for source in SOURCE_ROUTES:
        source_id = str(source["source_id"])
        runtime_target = str(source["runtime_target"])
        asset_present = bool(local_assets.get(source_id))
        runtime_present = bool(runtime_target and runtime_modules.get(runtime_target))
        requires_asset = bool(source.get("requires_source_asset", True))
        blockers = [str(value) for value in source.get("static_blockers") or []]
        if requires_asset and not asset_present:
            blockers.extend(["dataset_access_approval_required", "authorized_sample_asset_missing"])
        if runtime_target and not runtime_present:
            blockers.append("native_runtime_not_importable")
        if requires_asset and asset_present and runtime_present:
            blockers.append("native_adapter_conversion_pending")
        integration_rung = "cloned" if requires_asset and asset_present else "pre_cloned"
        ready = requires_asset and asset_present and (runtime_present or not runtime_target)
        if not requires_asset:
            disposition = "held_runtime_or_integration"
        else:
            disposition = "fetch_build_ready" if ready else "held_access_terms"
        rows.append(
            {
                **source,
                "official_source_verified_on": "2026-08-14",
                "local_authorized_asset_present": asset_present,
                "runtime_importable": runtime_present,
                "integration_rung": integration_rung,
                "current_core_eligible": False,
                "bulk_fetch_allowed": False,
                "raw_asset_redistribution_allowed": False,
                "disposition": disposition,
                "blockers": sorted(set(blockers)),
                "next_stage": (
                    "single_authorized_shard_native_conversion"
                    if ready
                    else "isolated_validator_integration"
                    if not requires_asset
                    else "user_license_acceptance_then_single_small_shard"
                ),
            }
        )
    return {
        "schema_version": "autonomous_external_source_probe_v1",
        "status": "candidate_only",
        "sources": rows,
        "summary": {
            "source_count": len(rows),
            "fetch_build_ready": sum(row["disposition"] == "fetch_build_ready" for row in rows),
            "native_candidate_ready": 0,
        },
        "policy": {
            "accept_terms_automatically": False,
            "download_large_archives": False,
            "metadata_is_not_native_conversion": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runtime_modules = {
        "waymax": importlib.util.find_spec("waymax") is not None,
        "nuplan": importlib.util.find_spec("nuplan") is not None,
        "commonroad_reach": importlib.util.find_spec("commonroad_reach") is not None,
        "commonroad_crime": importlib.util.find_spec("commonroad_crime") is not None,
        "carla": importlib.util.find_spec("carla") is not None,
    }
    local_assets = {
        "waymo_open_motion": bool(os.environ.get("OPERATE_WOMD_SAMPLE")),
        "nuplan": bool(os.environ.get("OPERATE_NUPLAN_SAMPLE")),
        "highd": bool(os.environ.get("OPERATE_HIGHD_SAMPLE")),
    }
    report = build_external_source_report(
        runtime_modules=runtime_modules,
        local_assets=local_assets,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
