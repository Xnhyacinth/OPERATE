"""
domains.power_grid.seeds.from_rts_gmlc — Build seeds from RTS-GMLC CSVs.

RTS-GMLC (Reliability Test System — Grid Modernization Lab Consortium) is
the NREL/DOE-curated 73-bus reference grid. Its ``SourceData/`` exposes
canonical CSVs (``bus.csv``, ``gen.csv``, ``branch.csv``,
``reserves.csv``) and ``timeseries_pointers.csv`` referencing day-ahead /
real-time load and renewable profiles.

For v0.1 we use RTS-GMLC at the *metadata level* (bus count, generator
mix, branch topology, reserves requirement) and the pglib-uc rts_gmlc/
subset for the actual time-series (since pglib-uc is the curated,
JSON-clean derivative). This gives us a deterministic, self-contained
data path while preserving the RTS-GMLC provenance chain.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ..source_paths import source_root
from .from_pglib_uc import (
    build_critical_winter_peak_seed,
    build_daily_ops_24h_seed,
)
from .schema import Provenance, ScenarioSeed
from .source_locks import provenance_lock_kwargs

RTS_GMLC_ROOT = source_root("RTS-GMLC") / "RTS_Data"
SOURCE_DATA = RTS_GMLC_ROOT / "SourceData"


def load_bus_table() -> list[dict[str, Any]]:
    p = SOURCE_DATA / "bus.csv"
    if not p.exists():
        return []
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_gen_table() -> list[dict[str, Any]]:
    p = SOURCE_DATA / "gen.csv"
    if not p.exists():
        return []
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def topology_summary() -> dict[str, Any]:
    """Quick summary used by tests and provenance attestation."""
    buses = load_bus_table()
    gens = load_gen_table()
    fuel_mix: dict[str, int] = {}
    for g in gens:
        fuel = g.get("Fuel") or g.get("Unit Type") or "unknown"
        fuel_mix[fuel] = fuel_mix.get(fuel, 0) + 1
    return {
        "bus_count": len(buses),
        "gen_count": len(gens),
        "fuel_mix": fuel_mix,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Seed builders that compose pglib-uc rts_gmlc/ time series with RTS-GMLC
# topology metadata
# ─────────────────────────────────────────────────────────────────────────────


def build_daily_ops_24h_seed_from_rts(
    pglib_case_path: Path,
    *,
    seed_id: str,
    seed: int = 42,
    difficulty_mode: str = "time_pressure",
    difficulty_level: str = "basic",
) -> ScenarioSeed:
    """Daily-ops 24h seed using a pglib-uc rts_gmlc/ case for chronics and the
    RTS-GMLC SourceData/ for topology provenance enrichment."""
    seed_obj = build_daily_ops_24h_seed(
        pglib_case_path,
        seed_id=seed_id,
        seed=seed,
        difficulty_mode=difficulty_mode,
        difficulty_level=difficulty_level,
    )
    summary = topology_summary()
    # Enrich provenance with RTS-GMLC topology summary
    enriched = Provenance(
        data_source="rts_gmlc+pglib_uc",
        files=seed_obj.provenance.files
        + [
            "works/RTS-GMLC/RTS_Data/SourceData/bus.csv",
            "works/RTS-GMLC/RTS_Data/SourceData/gen.csv",
        ],
        **provenance_lock_kwargs("rts_gmlc", "pglib_uc"),
        time_window=dict(seed_obj.provenance.time_window),
        license="CC-BY-4.0 (pglib-uc); NREL/DOE attribution (RTS-GMLC)",
        notes=(
            "Time-series from pglib-uc rts_gmlc subset; topology metadata "
            f"({summary['bus_count']} buses, {summary['gen_count']} gens) "
            "from RTS-GMLC SourceData."
        ),
    )
    seed_obj.provenance = enriched
    return seed_obj


def build_critical_winter_peak_seed_from_rts(
    pglib_case_path: Path,
    *,
    seed_id: str,
    seed: int = 42,
    difficulty_mode: str = "time_pressure",
    difficulty_level: str = "basic",
) -> ScenarioSeed:
    seed_obj = build_critical_winter_peak_seed(
        pglib_case_path,
        seed_id=seed_id,
        seed=seed,
        difficulty_mode=difficulty_mode,
        difficulty_level=difficulty_level,
    )
    summary = topology_summary()
    enriched = Provenance(
        data_source="rts_gmlc+pglib_uc",
        files=seed_obj.provenance.files
        + [
            "works/RTS-GMLC/RTS_Data/SourceData/bus.csv",
            "works/RTS-GMLC/RTS_Data/SourceData/gen.csv",
        ],
        **provenance_lock_kwargs("rts_gmlc", "pglib_uc"),
        time_window=dict(seed_obj.provenance.time_window),
        license="CC-BY-4.0 (pglib-uc); NREL/DOE attribution (RTS-GMLC)",
        notes=(
            "Winter-peak time-series from pglib-uc rts_gmlc subset; topology "
            f"metadata ({summary['bus_count']} buses, {summary['gen_count']} "
            "gens) from RTS-GMLC SourceData."
        ),
    )
    seed_obj.provenance = enriched
    return seed_obj
