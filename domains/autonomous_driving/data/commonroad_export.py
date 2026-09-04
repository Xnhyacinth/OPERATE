"""Isolated CommonRoad XML export for the NGSIM conversion pipeline.

CommonRoad 2026.1 pins the legacy protobuf runtime while the main benchmark
uses a newer protobuf through OR-Tools.  Running the file writer in a child
with protobuf's pure-Python implementation keeps both runtime contracts
intact and makes the XML round-trip explicit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from commonroad.common.common_scenario import ScenarioID
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.file_writer import CommonRoadFileWriter
from commonroad.common.util import FileFormat
from commonroad.common.writer.file_writer_interface import OverwriteExistingFile
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.lanelet import Lanelet, LaneletType
from commonroad.scenario.scenario import Scenario, Tag

_COMMONROAD_MAP_NAMES = {
    "us-101": "NGSIM_US101",
    "i-80": "NGSIM_I80",
    "lankershim": "NGSIM_LANKERSHIM",
    "peachtree": "NGSIM_PEACHTREE",
}


def _commonroad_map_name(recording_id: str) -> str:
    try:
        return _COMMONROAD_MAP_NAMES[recording_id.strip().lower()]
    except KeyError as error:
        raise ValueError("ngsim_commonroad_recording_identity_unsupported") from error


def export(payload: dict[str, Any], output: Path, report: Path) -> None:
    lane_count = int(payload["lane_count"])
    route_length = float(payload["route_length_m"])
    map_name = _commonroad_map_name(str(payload["recording_id"]))
    scenario = Scenario(
        0.1,
        ScenarioID(country_id="USA", map_name=map_name, map_id=1),
    )
    x_coordinates = np.array([0.0, route_length], dtype=float)
    for index in range(lane_count):
        center_y = index * 3.6
        center = np.column_stack((x_coordinates, np.full(2, center_y)))
        left = np.column_stack((x_coordinates, np.full(2, center_y + 1.8)))
        right = np.column_stack((x_coordinates, np.full(2, center_y - 1.8)))
        scenario.add_objects(
            Lanelet(
                left,
                center,
                right,
                index + 1,
                lanelet_type={LaneletType.HIGHWAY},
            )
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    CommonRoadFileWriter(
        scenario,
        PlanningProblemSet(),
        author="USDOT FHWA; mechanically converted by OPERATE",
        affiliation="OPERATE",
        source=str(payload["source"]),
        tags={Tag.HIGHWAY},
        file_format=FileFormat.XML,
    ).write_to_file(
        str(output),
        OverwriteExistingFile.SKIP,
        check_validity=True,
    )
    roundtrip, _ = CommonRoadFileReader(str(output)).open()
    observed_lanelets = len(roundtrip.lanelet_network.lanelets)
    if observed_lanelets != lane_count:
        raise ValueError("ngsim_commonroad_roundtrip_lane_count_mismatch")
    report.write_text(
        json.dumps(
            {
                "schema_version": "ngsim_commonroad_export_v2",
                "source_window_sha256": payload["source_window_sha256"],
                "trajectory_sibling": "runtime/log_replay.jsonl",
                "admission_status": "format_roundtrip_only",
                "map_name": map_name,
                "geometry_mode": "synthetic_straight_lane_proxy",
                "core_validator_eligible": False,
                "lanelet_count": observed_lanelets,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("CommonRoad export payload must be an object")
    export(payload, args.output, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
