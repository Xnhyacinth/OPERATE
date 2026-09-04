"""Source acquisition and deterministic materialization for driving data."""

from .contracts import NGSIMSourcePlan, build_ngsim_plan
from .ngsim import (
    materialize_bundle,
    mine_lane_change_windows,
    mine_time_headway_windows,
    mine_windows,
    normalize_csv,
    verify_bundle,
    verify_ngsim_csv,
    verify_source_lock,
)

__all__ = [
    "NGSIMSourcePlan",
    "build_ngsim_plan",
    "materialize_bundle",
    "mine_lane_change_windows",
    "mine_time_headway_windows",
    "mine_windows",
    "normalize_csv",
    "verify_bundle",
    "verify_ngsim_csv",
    "verify_source_lock",
]
