"""Scenario-level provenance locks for public power-grid data sources."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceLock:
    url: str
    commit: str
    lock_strategy: str
    version: str | None = None
    data_release: str | None = None


SOURCE_LOCKS: dict[str, SourceLock] = {
    "pglib_uc": SourceLock(
        url="https://github.com/power-grid-lib/pglib-uc",
        commit="39a7f38cf4703de92f0291f0c873c2e98c789301",
        lock_strategy="git_commit",
    ),
    "rts_gmlc": SourceLock(
        url="https://github.com/GridMod/RTS-GMLC",
        commit="3ece0d3725c844056132393ee252b3083dd4eab4",
        data_release="v0.2.3",
        lock_strategy="git_tag+commit",
    ),
    "grid2op_l2rpn": SourceLock(
        url="https://github.com/Grid2Op/grid2op",
        commit="d74b8e11a238ebea40fd17694529347bb4854d3c",
        version="v1.12.4",
        lock_strategy="git_tag+commit",
    ),
    "cigre_mv_pandapower": SourceLock(
        url="https://pandapower.readthedocs.io/en/latest/networks/cigre.html",
        commit="561b08e01ff12dd40a2e76615412b14b205f0e91",
        lock_strategy="pandapower_git_commit",
    ),
    "pandapower_mv_oberrhein": SourceLock(
        url="https://pandapower.readthedocs.io/en/latest/networks/power_system_test_cases.html",
        commit="561b08e01ff12dd40a2e76615412b14b205f0e91",
        lock_strategy="pandapower_git_commit",
    ),
    "simbench": SourceLock(
        url="https://github.com/e2nIEE/simbench",
        commit="615135cbc04f4576bba6edad8528c1aa7e0a0b10",
        version="v1.6.2",
        data_release="SimBench 1.6.2 bundled grids and profiles",
        lock_strategy="git_tag+commit+pypi_version+simbench_code+profile_window",
    ),
    "pglib_opf": SourceLock(
        url="https://github.com/power-grid-lib/pglib-opf",
        commit="dc6be4b2f85ca0e776952ec22cbd4c22396ea5a3",
        data_release="v23.07",
        lock_strategy="git_tag+commit",
    ),
}


def provenance_lock_kwargs(*source_ids: str) -> dict[str, str]:
    """Return common scenario provenance lock fields for one or more sources."""
    locks = [SOURCE_LOCKS[source_id] for source_id in source_ids]
    return {
        "url": " + ".join(lock.url for lock in locks),
        "commit": " + ".join(lock.commit for lock in locks),
        "lock_strategy": " + ".join(lock.lock_strategy for lock in locks),
    }
