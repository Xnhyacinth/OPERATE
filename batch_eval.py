#!/usr/bin/env python3
"""
batch_eval.py — Multi-agent × multi-scenario × multi-seed batch runner
(thin CLI over ``runner/``).

Parallelizes ``runner.run_one`` across a thread pool (network-bound LLM
calls dominate), aggregates results into a per-agent leaderboard with
bootstrap confidence intervals, and writes a CSV + JSON manifest.

The batch helpers (``expand_scenarios``, ``run_one_safe``,
``_episode_file_logging``) live in ``runner/batch.py``. This module
re-exports them — including the legacy ``_``-private aliases — so every
existing import path (``from batch_eval import _expand_scenarios``,
``from batch_eval import _run_one_safe``, ``from batch_eval import
_episode_file_logging``) keeps resolving.

Usage:

    python batch_eval.py \\
        --agents wait_only greedy_heuristic oracle_offline random \\
        --scenarios "power_grid/daily_ops_24h/time_pressure/basic/*" \\
                    "power_grid/daily_ops_24h/time_pressure/medium/*" \\
        --seeds 42 43 44 \\
        --output-dir batch_results/baselines_run \\
        --max-workers 4
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation import build_leaderboard  # noqa: E402

# ``run_one`` is re-exported so legacy ``from batch_eval import run_one``
# keeps resolving; ``load_scenario_yaml`` is used in ``main()``.
from run import load_scenario_yaml, run_one  # noqa: E402, F401

# P3-2: batch helpers moved verbatim into ``runner/batch.py``. Re-import the
# public names plus the legacy ``_``-private aliases so existing imports
# (``from batch_eval import _expand_scenarios``, ``from batch_eval import
# _run_one_safe``, ``from batch_eval import _episode_file_logging``) keep
# resolving. Call sites in this module still use the ``_``-private names so
# tests that ``monkeypatch.setattr(mod, "_expand_scenarios", ...)`` keep
# working. ``F401`` is intentional on the re-exports.
from runner.batch import _episode_file_logging  # noqa: E402, F401
from runner.batch import expand_scenarios as _expand_scenarios  # noqa: E402, F401
from runner.batch import run_one_safe as _run_one_safe  # noqa: E402, F401

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
LOGGER = logging.getLogger("batch_eval")


def _execution_plan(
    backend_kinds: set[str], requested_workers: int
) -> tuple[str, int]:
    """Return executor isolation and effective concurrency for a batch."""
    workers = max(1, int(requested_workers))
    executor_kind = (
        "process" if backend_kinds.intersection({"sumo", "sumo_ego"}) else "thread"
    )
    if "grid2op" in backend_kinds:
        workers = 1
    return executor_kind, workers


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--agents", nargs="+", required=True)
    p.add_argument(
        "--scenarios",
        nargs="+",
        required=True,
        help="Scenario slugs or glob patterns relative to scenarios/",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-workers", type=int, default=4)
    args = p.parse_args()

    llm_agents = sorted(
        set(args.agents).intersection({"llm_agent", "react_llm", "reflexion_llm"})
    )
    if llm_agents:
        p.error(
            "batch_eval.py is the legacy deterministic-baseline runner and must not "
            "run LLM agents. Use scripts/batch_llm_eval.py for treatment-bound "
            f"LLM evaluation (requested: {', '.join(llm_agents)})."
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = _expand_scenarios(args.scenarios)
    LOGGER.info(
        "expanded %d scenarios from %d patterns", len(scenarios), len(args.scenarios)
    )

    # Grid2Op's ``grid2op.make()`` is NOT thread-safe — concurrent calls
    # corrupt its global class cache and cause >99% failure rate. Force
    # single-worker execution if any selected scenario uses the grid2op
    # backend.
    backend_kinds: set[str] = set()
    for slug in scenarios:
        try:
            body = load_scenario_yaml(slug)
            backend_kinds.add(str(body.get("backend_kind", "")))
        except Exception:
            continue
    executor_kind, effective_workers = _execution_plan(
        backend_kinds, args.max_workers
    )
    if "grid2op" in backend_kinds and args.max_workers > 1:
        LOGGER.warning(
            "grid2op-backed scenario detected; forcing --max-workers=1 "
            "because grid2op.make() corrupts its global class cache under "
            "concurrent instantiation."
        )
    if executor_kind == "process" and effective_workers > 1:
        LOGGER.info(
            "live SUMO backend detected; using process isolation for %d workers",
            effective_workers,
        )
    args.max_workers = effective_workers

    work: list[tuple[str, str, int, dict[str, Any]]] = []
    for s in scenarios:
        for a in args.agents:
            kwargs: dict[str, Any] = {}
            for seed in args.seeds:
                work.append((s, a, seed, kwargs))
    LOGGER.info("total episodes scheduled: %d", len(work))

    results: list[dict[str, Any]] = []
    executor_type = (
        futures.ProcessPoolExecutor
        if executor_kind == "process"
        else futures.ThreadPoolExecutor
    )
    with executor_type(max_workers=args.max_workers) as pool:
        for r in pool.map(_run_one_safe, work):
            results.append(r)

    # Persist all per-episode results
    with open(out_dir / "episodes.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # CSV summary
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "scenario_id",
                "scenario_signature",
                "family",
                "difficulty_mode",
                "difficulty_level",
                "agent_name",
                "seed",
                "status",
                "total_score",
                "raw_total",
                "prevented_loss",
                "foresight_score",
                "chose_fatal_option",
            ]
        )
        for r in results:
            score = r.get("score") or {}
            cf = r.get("counterfactual") or {}
            fs = r.get("foresight") or {}
            w.writerow(
                [
                    r.get("scenario_id"),
                    r.get("scenario_signature"),
                    r.get("family"),
                    r.get("difficulty_mode"),
                    r.get("difficulty_level"),
                    r.get("agent_name"),
                    r.get("seed"),
                    r.get("status", "ok"),
                    score.get("total_score"),
                    score.get("raw_total"),
                    cf.get("prevented_loss"),
                    fs.get("foresight_score"),
                    (r.get("ground_truth_summary") or {}).get("chose_fatal_option"),
                ]
            )

    # Leaderboard
    per_agent_totals: dict[str, list[float]] = {}
    per_agent_fatal: dict[str, list[bool]] = {}
    for r in results:
        if r.get("status") != "ok":
            continue
        a = r["agent_name"]
        per_agent_totals.setdefault(a, []).append(float(r["score"]["total_score"]))
        per_agent_fatal.setdefault(a, []).append(
            bool((r.get("ground_truth_summary") or {}).get("chose_fatal_option", False))
        )
    rows = build_leaderboard(per_agent_totals, fatal_flags=per_agent_fatal)
    leaderboard = [row.to_dict() for row in rows]
    with open(out_dir / "leaderboard.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "leaderboard": leaderboard,
                "n_episodes_total": len(results),
                "n_episodes_ok": sum(1 for r in results if r.get("status") == "ok"),
            },
            f,
            indent=2,
        )

    LOGGER.info("wrote %d episodes; leaderboard:", len(results))
    for row in rows:
        LOGGER.info(
            "  %s: mean=%.2f [95%% CI %.2f, %.2f] over %d episodes",
            row.agent_id,
            row.mean,
            row.ci_lo,
            row.ci_hi,
            row.n_episodes,
        )
    print(f"\nResults in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
