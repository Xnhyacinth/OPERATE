#!/usr/bin/env python3
"""
scripts/generate_scenarios.py — Generate the v0.1 scenario YAML set.

Produces 3 families × 2 difficulty modes × 4 difficulty levels × N seeds ≈
120 deterministic scenarios. Writes one YAML per scenario plus a
``_registry.json`` index with scenario_signature + provenance.

Run from the repo root:

    python scripts/generate_scenarios.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import yaml  # type: ignore[import]  # noqa: E402

from domains.power_grid.seeds.from_cigre import (  # noqa: E402
    build_distribution_volt_var_seed,
)
from domains.power_grid.seeds.from_l2rpn import build_storm_emergency_6h_seed  # noqa: E402
from domains.power_grid.seeds.from_pglib_uc import (  # noqa: E402
    build_critical_winter_peak_seed,
    build_daily_ops_24h_seed,
    build_reserve_stress_seed,
    build_wind_uncertainty_seed,
    list_ca_cases,
    list_cases,
    list_ferc_cases,
)
from domains.power_grid.seeds.from_rts_real_ts import (  # noqa: E402
    build_daily_ops_real_forecast_seed,
)
from domains.power_grid.seeds.schema import ScenarioSeed  # noqa: E402

DIFFICULTY_MODES = ["time_pressure", "deep_planning"]
DIFFICULTY_LEVELS = ["basic", "medium", "high", "extreme"]
# v0.4 Bucket B (D2): collapse RNG-only seed clones to ONE canonical seed.
# For these deterministic backends the integer seed only perturbs
# fog/tool RNG, not scenario structure (std=0 across seeds on all 8
# complexity_metrics), so 5 seeds were 5 replications of one task — corpus
# inflation with no marginal test coverage. Structural diversity now comes
# from the FULL real case library instead. The structural-seed families
# (storm_idf2023 / storm_wcci_2022 / acopf_dispatch_24h) keep their own
# seeds via their dedicated generators.
SEEDS_PER_VARIANT = 1
SEED_OFFSETS = [42]
# v0.4 Bucket B (D3): critical_winter_peak now uses the genuine cold
# season (Nov–Apr) rather than an arbitrary last-3-months slice, so the
# family name matches its content. RTS-GMLC monthly tags are YYYY-MM-DD.
WINTER_MONTHS = {"01", "02", "03", "04", "11", "12"}

SCENARIOS_ROOT = REPO_ROOT / "scenarios" / "power_grid"


def write_yaml(seed: ScenarioSeed, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = seed.to_dict()
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(body, f, sort_keys=False, allow_unicode=True)


def gen_daily_ops() -> list[dict]:
    """Generate daily_ops_24h scenarios from pglib-uc rts_gmlc cases."""
    cases = list_cases("rts_gmlc")
    # v0.4 Bucket B (D1): use the FULL 12-month rts_gmlc library (was the
    # first 3) — structural diversity from real cases, 1 canonical seed.
    chosen_cases = cases
    rows = []
    for case_path in chosen_cases:
        case_tag = case_path.stem.replace("-", "")
        for mode in DIFFICULTY_MODES:
            for level in DIFFICULTY_LEVELS:
                for _sd, off in enumerate(SEED_OFFSETS):
                    seed_id = f"do_{case_tag}_{mode}_{level}_s{off}"
                    seed = build_daily_ops_24h_seed(
                        case_path,
                        seed_id=seed_id,
                        seed=off,
                        difficulty_mode=mode,
                        difficulty_level=level,
                    )
                    out = (
                        SCENARIOS_ROOT
                        / "daily_ops_24h"
                        / mode
                        / level
                        / f"{seed_id}.yaml"
                    )
                    write_yaml(seed, out)
                    rows.append(_row(seed, out))
    return rows


def gen_winter_peak() -> list[dict]:
    """Generate critical_winter_peak scenarios from pglib-uc winter months."""
    cases = list_cases("rts_gmlc")
    # v0.4 Bucket B (D3): restrict to the genuine cold season (Nov–Apr) so
    # a "critical winter peak" scenario is never built on a July case.
    chosen_cases = [c for c in cases if c.stem.split("-")[1] in WINTER_MONTHS]
    rows = []
    for case_path in chosen_cases:
        case_tag = case_path.stem.replace("-", "")
        for mode in DIFFICULTY_MODES:
            for level in DIFFICULTY_LEVELS:
                for _sd, off in enumerate(SEED_OFFSETS):
                    seed_id = f"wp_{case_tag}_{mode}_{level}_s{off}"
                    seed = build_critical_winter_peak_seed(
                        case_path,
                        seed_id=seed_id,
                        seed=off,
                        difficulty_mode=mode,
                        difficulty_level=level,
                    )
                    out = (
                        SCENARIOS_ROOT
                        / "critical_winter_peak"
                        / mode
                        / level
                        / f"{seed_id}.yaml"
                    )
                    write_yaml(seed, out)
                    rows.append(_row(seed, out))
    return rows


def gen_storm() -> list[dict]:
    """Generate storm_emergency_6h scenarios.

    Public generation uses the four-level ladder. Compound-event mechanics
    for the hardest rung are represented by ``stress_profile`` metadata.
    """
    rows = []
    chronics_ids = [0, 1, 2, 3, 4]
    for cid in chronics_ids:
        for mode in DIFFICULTY_MODES:
            for level in DIFFICULTY_LEVELS:
                for _sd, off in enumerate(SEED_OFFSETS):
                    seed_id = f"st_chron{cid}_{mode}_{level}_s{off}"
                    seed = build_storm_emergency_6h_seed(
                        seed_id=seed_id,
                        seed=off,
                        difficulty_mode=mode,
                        difficulty_level=level,
                        env_name="l2rpn_case14_sandbox",
                        chronics_id=cid,
                        start_step=0,
                    )
                    out = (
                        SCENARIOS_ROOT
                        / "storm_emergency_6h"
                        / mode
                        / level
                        / f"{seed_id}.yaml"
                    )
                    write_yaml(seed, out)
                    rows.append(_row(seed, out))
    return rows


def gen_distribution_volt_var() -> list[dict]:
    """Generate distribution_volt_var scenarios (CIGRE MV + v0.4 topologies).

    v0.4 Bucket B (D1): the distribution tier diversifies from one CIGRE
    MV feeder to TWO real topologies, each at 1 canonical seed:

    - ``distribution_volt_var``          — CIGRE MV (15-bus), byte-identical
      to v0.3 at seed=42 (10 structural cells).
    - ``distribution_volt_var_oberrhein`` — pandapower mv_oberrhein, a real
      ~179-bus German MV feeder; perturbations are scaled up (coordinated
      all-feeder peak) so the larger grid actually reaches voltage
      violations and the agent's shed/curtail decisions discriminate.

    A balanced synthetic LV feeder was evaluated and DROPPED: it is
    voltage-robust by construction (no violations even at +100% surge), so
    it would add non-discriminating padding. Additional distribution
    topologies (OpenDSS IEEE-13/34/123, 3-phase LV) are v0.5 work.

    Each topology: 2 modes × 5 levels × 1 seed = 10 scenarios → 20 total.
    """
    networks = [
        ("cigre_mv_with_der_all", "dvv", "distribution_volt_var"),
        ("mv_oberrhein", "dvvob", "distribution_volt_var_oberrhein"),
    ]
    rows = []
    for network, prefix, family in networks:
        for mode in DIFFICULTY_MODES:
            for level in DIFFICULTY_LEVELS:
                for _sd, off in enumerate(SEED_OFFSETS):
                    seed_id = f"{prefix}_{mode}_{level}_s{off}"
                    seed = build_distribution_volt_var_seed(
                        seed_id=seed_id,
                        seed=off,
                        difficulty_mode=mode,
                        difficulty_level=level,
                        network=network,
                    )
                    out = SCENARIOS_ROOT / family / mode / level / f"{seed_id}.yaml"
                    write_yaml(seed, out)
                    rows.append(_row(seed, out))
    return rows


def gen_daily_ops_real_forecast() -> list[dict]:
    """v0.2 family: daily_ops_real_forecast_24h.

    Uses pglib-uc rts_gmlc cases but overlays a forecast-bias
    perturbation whose intensity is sourced from RTS-GMLC's real
    DA→RT error timeseries. 3 cases × 2 modes × 4 levels × 5 seeds =
    **120 scenarios**.
    """
    # v0.4 Bucket B (D1): full 12-month rts_gmlc library, 1 canonical seed.
    cases = list_cases("rts_gmlc")
    rows = []
    for case_path in cases:
        case_tag = case_path.stem.replace("-", "")
        for mode in DIFFICULTY_MODES:
            for level in DIFFICULTY_LEVELS:
                for _sd, off in enumerate(SEED_OFFSETS):
                    seed_id = f"drf_{case_tag}_{mode}_{level}_s{off}"
                    seed = build_daily_ops_real_forecast_seed(
                        case_path,
                        seed_id=seed_id,
                        seed=off,
                        difficulty_mode=mode,
                        difficulty_level=level,
                    )
                    out = (
                        SCENARIOS_ROOT
                        / "daily_ops_real_forecast_24h"
                        / mode
                        / level
                        / f"{seed_id}.yaml"
                    )
                    write_yaml(seed, out)
                    rows.append(_row(seed, out))
    return rows


def gen_wind_uncertainty() -> list[dict]:
    """Generate wind_uncertainty_24h scenarios from pglib-uc ferc/ cases.

    The FERC subset has paired hw/lw wind variants per month. We pick
    the first 4 months × 2 wind regimes = 8 cases × 2 modes × 5 levels
    × 5 seeds = **400 scenarios**.
    """
    # v0.4 Bucket B (D1): full 24-case ferc library (12 months × hw/lw),
    # 1 canonical seed. (D1 corpus-balance): wind is the largest UC family,
    # and its mode axis is thin (horizon + bias-start only), so it ships
    # time_pressure-only to keep the aggregate-UC tier from swamping the
    # power-flow share. The deep_planning mode is retained on the smaller
    # rts_gmlc families (daily_ops / winter / real_forecast).
    cases = list_ferc_cases()
    chosen_cases = cases
    rows = []
    for case_path in chosen_cases:
        case_tag = (
            case_path.stem.replace("-", "").replace("_hw", "_hw").replace("_lw", "_lw")
        )
        for mode in ["time_pressure"]:
            for level in DIFFICULTY_LEVELS:
                for _sd, off in enumerate(SEED_OFFSETS):
                    seed_id = f"wu_{case_tag}_{mode}_{level}_s{off}"
                    seed = build_wind_uncertainty_seed(
                        case_path,
                        seed_id=seed_id,
                        seed=off,
                        difficulty_mode=mode,
                        difficulty_level=level,
                    )
                    out = (
                        SCENARIOS_ROOT
                        / "wind_uncertainty_24h"
                        / mode
                        / level
                        / f"{seed_id}.yaml"
                    )
                    write_yaml(seed, out)
                    rows.append(_row(seed, out))
    return rows


def gen_reserve_stress() -> list[dict]:
    """Generate reserve_stress_24h scenarios from pglib-uc ca/ cases.

    The California subset has 4 reserves variants (``_reserves_0/1/3/5``)
    per month. We pick the first 3 months as variants so each scenario
    YAML is a real published reserve profile, not a synthesized one.
    """
    # v0.4 Bucket B (D1): full 20-case ca library (5 months × 4 reserves
    # variants), 1 canonical seed; time_pressure-only (see wind rationale)
    # so the two largest UC families don't dominate the power-flow share.
    cases = list_ca_cases()
    chosen_cases = cases
    rows = []
    for case_path in chosen_cases:
        # Preserve the reserves variant suffix (`_0`/`_1`/`_3`/`_5`) so
        # the four reserves-stress files for the same month do not collide
        # in the generated seed_id. v0.1.2 bug: slicing to [:16] collapsed
        # them to identical seed_ids, overwriting YAML files and producing
        # 200 hash-drift audit errors.
        case_tag = case_path.stem.replace("-", "").replace("_reserves_", "_r")
        for mode in ["time_pressure"]:
            for level in DIFFICULTY_LEVELS:
                for _sd, off in enumerate(SEED_OFFSETS):
                    seed_id = f"rs_{case_tag}_{mode}_{level}_s{off}"
                    seed = build_reserve_stress_seed(
                        case_path,
                        seed_id=seed_id,
                        seed=off,
                        difficulty_mode=mode,
                        difficulty_level=level,
                    )
                    out = (
                        SCENARIOS_ROOT
                        / "reserve_stress_24h"
                        / mode
                        / level
                        / f"{seed_id}.yaml"
                    )
                    write_yaml(seed, out)
                    rows.append(_row(seed, out))
    return rows


def _row(seed: ScenarioSeed, out_path: Path) -> dict:
    return {
        "scenario_id": seed.seed_id,
        "family": seed.family,
        "difficulty_mode": seed.difficulty_mode,
        "difficulty_level": seed.difficulty_level,
        "seed": seed.seed,
        "backend_kind": seed.backend_kind,
        "horizon_ticks": seed.horizon_ticks,
        "tick_minutes": seed.tick_minutes,
        "scenario_signature": seed.signature(),
        "complexity_metrics": seed.complexity_metrics(),
        "provenance_files": list(seed.provenance.files),
        "provenance_source": seed.provenance.data_source,
        "provenance_license": seed.provenance.license,
        "path": str(out_path.relative_to(REPO_ROOT)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--family",
        choices=[
            "all",
            "daily_ops_24h",
            "daily_ops_real_forecast_24h",
            "critical_winter_peak",
            "storm_emergency_6h",
            "reserve_stress_24h",
            "wind_uncertainty_24h",
            "distribution_volt_var",
        ],
        default="all",
    )
    args = parser.parse_args()

    print(f"Writing scenarios under {SCENARIOS_ROOT}")
    all_rows: list[dict] = []
    if args.family in {"all", "daily_ops_24h"}:
        all_rows.extend(gen_daily_ops())
    if args.family in {"all", "critical_winter_peak"}:
        all_rows.extend(gen_winter_peak())
    if args.family in {"all", "storm_emergency_6h"}:
        all_rows.extend(gen_storm())
    if args.family in {"all", "reserve_stress_24h"}:
        all_rows.extend(gen_reserve_stress())
    if args.family in {"all", "wind_uncertainty_24h"}:
        all_rows.extend(gen_wind_uncertainty())
    if args.family in {"all", "daily_ops_real_forecast_24h"}:
        all_rows.extend(gen_daily_ops_real_forecast())
    if args.family in {"all", "distribution_volt_var"}:
        all_rows.extend(gen_distribution_volt_var())

    registry = {
        "schema_version": "0.1.0",
        "n_scenarios": len(all_rows),
        "by_family": {
            f: sum(1 for r in all_rows if r["family"] == f)
            for f in {r["family"] for r in all_rows}
        },
        "scenarios": all_rows,
    }
    out_registry = SCENARIOS_ROOT / "_registry.json"
    with open(out_registry, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(all_rows)} scenarios:")
    for family, n in registry["by_family"].items():
        print(f"  {family}: {n}")
    print(f"Registry at {out_registry.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
