#!/usr/bin/env python3
"""
scripts/generate_idf2023_scenarios.py — Generate the v0.3 Phase 3.1
``storm_emergency_6h_idf2023`` family from the IEEE 118-bus Île-de-France
Grid2Op env.

Sweep layout (default 96 scenarios):

  * 4 chronics × 4 levels (basic/medium/high/extreme) ×
    2 modes × 3 seeds = 120 scenarios.

  The v0.3.0 family (3 chronics × 5 levels × 2 modes × 2 seeds = 60) is a
  strict SUBSET of this grid, so re-running reproduces the published 60
  YAMLs byte-identically and only appends the 60 new ones. chronics_id 3
  wraps modulo to a physical chronic in test mode (see
  from_l2rpn_idf2023 docstring); it is a signature axis, not new physics.

Pass ``--n-scenarios N`` to cap; the deterministic enumeration order is
documented in ``_planned_scenarios()`` so a cap of N truncates the
deterministic suffix and stays idempotent across reruns.

The script is **re-runnable**. On each run it:

1. Rewrites every scenario YAML under
   ``scenarios/power_grid/storm_emergency_6h_idf2023/``.
2. Loads the existing ``scenarios/power_grid/_registry.json``.
3. **Removes** every existing row whose family is
   ``storm_emergency_6h_idf2023`` so re-runs don't double-count.
4. Appends the freshly generated rows.
5. Recomputes ``n_scenarios`` and ``by_family``.
6. Writes the registry atomically (temp file + replace) preserving the
   existing JSON style (indent=2, ensure_ascii=False).

Pre-existing scenario hashes are never touched. The 1360 v0.2.4
scenarios stay at byte-identical signatures.

Run from the repo root:

    python scripts/generate_idf2023_scenarios.py            # generate all 60
    python scripts/generate_idf2023_scenarios.py --n-scenarios 24
    python scripts/generate_idf2023_scenarios.py --dry-run  # plan only, no writes
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import yaml  # type: ignore[import]  # noqa: E402

from domains.power_grid.seeds.from_l2rpn_idf2023 import (  # noqa: E402
    build_storm_emergency_6h_idf2023_seed,
    idf2023_chronics_available,
)
from domains.power_grid.seeds.schema import ScenarioSeed  # noqa: E402

FAMILY = "storm_emergency_6h_idf2023"

SCENARIOS_ROOT = REPO_ROOT / "scenarios" / "power_grid"
FAMILY_ROOT = SCENARIOS_ROOT / FAMILY
REGISTRY_PATH = SCENARIOS_ROOT / "_registry.json"

DIFFICULTY_MODES = ["time_pressure", "deep_planning"]
DIFFICULTY_LEVELS = ["basic", "medium", "high", "extreme"]
SEED_OFFSETS = [42, 43, 44]  # v0.3.x Step D: 3 seeds per (chronic, mode, level)


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic enumeration
# ─────────────────────────────────────────────────────────────────────────────


def _planned_scenarios() -> list[dict[str, object]]:
    """Return the full deterministic enumeration of (chronic, mode, level,
    seed) tuples, ordered so truncating to the first N still gives a
    stable subset across runs.

    Ordering: public level (basic to extreme), chronic, mode, then seed.
    This puts the most "basic" coverage early so a small --n-scenarios
    still produces meaningful diversity.
    """
    plan: list[dict[str, object]] = []
    chronics = idf2023_chronics_available()
    for level in DIFFICULTY_LEVELS:
        for cid in chronics:
            for mode in DIFFICULTY_MODES:
                for seed in SEED_OFFSETS:
                    plan.append(
                        {
                            "chronics_id": int(cid),
                            "difficulty_mode": mode,
                            "difficulty_level": level,
                            "seed": int(seed),
                        }
                    )
    return plan


def _seed_id_for(spec: dict[str, object]) -> str:
    return (
        f"stidf_chron{spec['chronics_id']}"
        f"_{spec['difficulty_mode']}"
        f"_{spec['difficulty_level']}"
        f"_s{spec['seed']}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scenario writing
# ─────────────────────────────────────────────────────────────────────────────


def write_yaml(seed: ScenarioSeed, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = seed.to_dict()
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(body, f, sort_keys=False, allow_unicode=True)


def _row(seed: ScenarioSeed, out_path: Path) -> dict:
    """Build a registry row exactly mirroring the schema used by
    ``scripts/generate_scenarios.py:_row`` so audit.py can consume the new
    rows without changes."""
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


def _scenario_yaml_path(spec: dict[str, object], seed_id: str) -> Path:
    return (
        FAMILY_ROOT
        / str(spec["difficulty_mode"])
        / str(spec["difficulty_level"])
        / f"{seed_id}.yaml"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Registry update (atomic, idempotent)
# ─────────────────────────────────────────────────────────────────────────────


def _load_registry() -> dict[str, object]:
    if not REGISTRY_PATH.exists():
        raise SystemExit(
            f"registry not found at {REGISTRY_PATH}; run "
            f"`python scripts/generate_scenarios.py` first to bootstrap"
        )
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


def _write_registry_atomic(registry: dict[str, object]) -> None:
    """Write registry via temp file + os.replace so a partial write
    cannot corrupt the file."""
    tmp = REGISTRY_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
        f.write("\n")  # match existing trailing newline (if any)
    os.replace(tmp, REGISTRY_PATH)


def _replace_family_rows(
    registry: dict[str, object], new_rows: list[dict]
) -> dict[str, object]:
    """Strip every existing row whose family is FAMILY, then append the
    new rows. Recompute n_scenarios and by_family.

    Pre-existing rows for other families are kept byte-for-byte.
    """
    scenarios = list(registry.get("scenarios", []))  # type: ignore[arg-type]
    pruned = [r for r in scenarios if r.get("family") != FAMILY]
    pruned.extend(new_rows)

    by_family: dict[str, int] = {}
    for r in pruned:
        fam = str(r.get("family", "unknown"))
        by_family[fam] = by_family.get(fam, 0) + 1

    registry["n_scenarios"] = len(pruned)
    registry["by_family"] = by_family
    registry["scenarios"] = pruned
    return registry


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate storm_emergency_6h_idf2023 scenarios (IEEE 118-bus "
            "Île-de-France) and merge them into scenarios/power_grid/"
            "_registry.json."
        )
    )
    parser.add_argument(
        "--n-scenarios",
        type=int,
        default=None,
        help=(
            "Cap the number of scenarios generated (deterministic suffix "
            "truncation). Default: all 60."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and exit; write no files.",
    )
    args = parser.parse_args()

    plan = _planned_scenarios()
    full_plan_count = len(plan)
    if args.n_scenarios is not None:
        if args.n_scenarios < 1:
            print(
                f"--n-scenarios must be >= 1 (got {args.n_scenarios})",
                file=sys.stderr,
            )
            return 2
        plan = plan[: args.n_scenarios]

    print(
        f"[plan] family={FAMILY}  "
        f"full_layout={full_plan_count}  "
        f"will_generate={len(plan)}  "
        f"dry_run={args.dry_run}"
    )
    # Group preview (counts by level × mode)
    counts: dict[tuple[str, str], int] = {}
    for spec in plan:
        key = (str(spec["difficulty_level"]), str(spec["difficulty_mode"]))
        counts[key] = counts.get(key, 0) + 1
    for (level, mode), n in sorted(counts.items()):
        print(f"  level={level:<10} mode={mode:<14} n={n}")

    if args.dry_run:
        print("[dry-run] no files written, no registry changes")
        return 0

    # Build seeds + write YAMLs
    new_rows: list[dict] = []
    for spec in plan:
        seed_id = _seed_id_for(spec)
        seed = build_storm_emergency_6h_idf2023_seed(
            seed_id=seed_id,
            seed=int(spec["seed"]),  # type: ignore[arg-type]
            difficulty_mode=str(spec["difficulty_mode"]),
            difficulty_level=str(spec["difficulty_level"]),
            chronics_id=int(spec["chronics_id"]),  # type: ignore[arg-type]
        )
        out_path = _scenario_yaml_path(spec, seed_id)
        write_yaml(seed, out_path)
        new_rows.append(_row(seed, out_path))

    # Idempotent registry merge
    registry = _load_registry()
    pre_total = int(registry.get("n_scenarios", 0))
    pre_family_n = int(
        dict(registry.get("by_family", {})).get(FAMILY, 0)  # type: ignore[arg-type]
    )
    registry = _replace_family_rows(registry, new_rows)
    post_total = int(registry["n_scenarios"])
    post_family_n = int(
        dict(registry["by_family"]).get(FAMILY, 0)  # type: ignore[arg-type]
    )
    _write_registry_atomic(registry)

    print()
    print(
        f"[done] wrote {len(new_rows)} YAML scenarios under {FAMILY_ROOT.relative_to(REPO_ROOT)}"
    )
    print(
        f"[done] registry: total {pre_total} -> {post_total}  "
        f"({FAMILY}: {pre_family_n} -> {post_family_n})"
    )
    print(f"[done] registry path: {REGISTRY_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
