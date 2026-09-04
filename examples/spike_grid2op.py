#!/usr/bin/env python3
"""
examples/spike_grid2op.py — End-to-end smoke spike for the power-grid domain.

Runs a short mock-LLM episode on:

1. The default ``pglib_uc_synthetic`` backend (always available — uses
   real pglib-uc rts_gmlc data).
2. If grid2op is importable: a short episode on the ``l2rpn_case14_sandbox``
   storm scenario, to verify the Grid2Op backend is wired correctly.

Prints a compact summary of:

- scenario id + signature
- per-tick action stream
- final ground-truth cost components
- counterfactual replay vs the wait_only baseline (`prevented_loss`)

Usage:
    python examples/spike_grid2op.py
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

# Make repo root importable when launched directly
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core import (  # noqa: E402
    Action,
    ToolCall,
    run_counterfactual,
    wait_only_policy,
)
from domains.power_grid.adapter import PowerGridEnvironment  # noqa: E402
from domains.power_grid.seeds.from_l2rpn import (  # noqa: E402
    build_storm_emergency_6h_seed,
)
from domains.power_grid.seeds.from_pglib_uc import (  # noqa: E402
    build_daily_ops_24h_seed,
    list_cases,
)


def banner(text: str) -> None:
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def _mock_agent_actions(env: PowerGridEnvironment, horizon: int) -> list[Action]:
    """A very small heuristic 'agent' for the spike.

    - Tick 0: query state + forecast
    - Tick 1: investigate the first generator
    - Tick 2: redispatch a small bump on the first committed generator
    - Tick 3+: alternate between `wait` and `commit_reserve`
    """
    actions: list[Action] = []
    obs0 = env.snapshot()
    first_gen = next(
        (gid for gid, e in obs0["entities"].items() if e.get("kind") == "generator"),
        None,
    )
    actions.append(
        Action(
            tool_calls=[
                ToolCall(name="query_grid_state", idempotency_key="qgs_0"),
                ToolCall(
                    name="forecast_query", args={"horizon": 4}, idempotency_key="fq_0"
                ),
            ],
            dominant="query_grid_state",
        )
    )
    if first_gen is not None:
        actions.append(
            Action(
                tool_calls=[
                    ToolCall(
                        name="investigate_substation",
                        args={"target_id": first_gen},
                        idempotency_key=f"inv_{first_gen}_1",
                    ),
                ],
                dominant="investigate_substation",
            )
        )
        actions.append(
            Action(
                tool_calls=[
                    ToolCall(
                        name="redispatch_generation",
                        args={
                            "generator_id": first_gen,
                            "target_mw": 60,
                            "commit": True,
                        },
                        idempotency_key=f"rd_{first_gen}_2",
                    ),
                ],
                dominant="redispatch_generation",
            )
        )
    for t in range(3, horizon):
        if t % 4 == 0:
            actions.append(
                Action(
                    tool_calls=[
                        ToolCall(
                            name="commit_reserve",
                            args={"mw": 25},
                            idempotency_key=f"cr_{t}",
                        )
                    ],
                    dominant="commit_reserve",
                )
            )
        elif t % 4 == 1:
            actions.append(
                Action(
                    tool_calls=[
                        ToolCall(name="query_grid_state", idempotency_key=f"qgs_{t}")
                    ],
                    dominant="query_grid_state",
                )
            )
        else:
            actions.append(
                Action(
                    tool_calls=[ToolCall(name="wait", idempotency_key=f"w_{t}")],
                    dominant="wait",
                )
            )
    return actions[:horizon]


def _cost_extractor(gt: dict) -> dict[str, float]:
    return dict(gt.get("cost_components", {}))


def run_synthetic_spike() -> None:
    banner("OPERATE spike — backend = pglib_uc_synthetic (always available)")
    case = list_cases("rts_gmlc")[0]
    seed = build_daily_ops_24h_seed(
        case, seed_id="spike_pglib", difficulty_level="medium"
    )
    print(f"  scenario.id        = {seed.seed_id}")
    print(f"  scenario.signature = {seed.signature()}")
    print(f"  data source        = {seed.provenance.data_source}")
    print(
        f"  horizon            = {seed.horizon_ticks} ticks @ {seed.tick_minutes} min"
    )
    print(
        f"  perturbations      = {len(seed.perturbations)} | dilemmas = {len(seed.dilemmas)}"
    )
    print(
        f"  stakeholder classes= {sorted({la.stakeholder_class for la in seed.load_assignments})}"
    )

    env = PowerGridEnvironment()
    obs = env.reset(seed.to_dict(), seed=seed.seed)
    print(f"\n  initial entities   = {len(obs.get('entities', {}))}")
    print(
        f"  initial trust      = {[(g, r['tier']) for g, r in obs['stakeholder_trust'].items()]}"
    )

    actions = _mock_agent_actions(env, seed.horizon_ticks)
    print(f"\n  mock agent actions = {[a.dominant for a in actions]}")

    total_reward = 0.0
    for action in actions:
        ret = env.step(action)
        total_reward += ret.reward
        if ret.done:
            break

    gt = env.ground_truth()
    print(f"\n  per-tick reward sum = {round(total_reward, 3)}")
    print(f"  cost components     = {json.dumps(gt['cost_components'], indent=2)}")
    print(f"  per-load shed (MWh) = {json.dumps(gt['per_load_shed_mwh'], indent=2)}")
    print(f"  dilemmas triggered  = {gt.get('dilemmas_triggered', [])}")
    print(f"  chose fatal option? = {gt.get('chose_fatal_option')}")

    banner("Counterfactual replay (mask actions → wait_only)")
    cf_report = run_counterfactual(
        env_factory=PowerGridEnvironment,
        scenario_config=seed.to_dict(),
        seed=seed.seed,
        actual_actions=actions,
        cost_extractor=_cost_extractor,
        masking_policy=wait_only_policy,
        masking_label="wait_only",
    )
    print(json.dumps(cf_report.to_dict(), indent=2))


def run_grid2op_spike() -> None:
    banner("OPERATE spike — backend = grid2op (optional)")
    try:
        importlib.import_module("grid2op")
    except ImportError:
        print("  grid2op not installed — skipping. Install with:")
        print("    pip install grid2op pandapower lightsim2grid")
        return
    seed = build_storm_emergency_6h_seed(
        seed_id="spike_storm", difficulty_level="basic", seed=42
    )
    print(f"  scenario.id        = {seed.seed_id}")
    print(f"  scenario.signature = {seed.signature()}")
    print(f"  backend.env_name   = {seed.backend_config['env_name']}")

    env = PowerGridEnvironment()
    try:
        obs = env.reset(seed.to_dict(), seed=seed.seed)
    except Exception as exc:
        print(f"  grid2op reset failed: {type(exc).__name__}: {exc}")
        return
    print(f"  initial entities   = {len(obs.get('entities', {}))}")

    actions = _mock_agent_actions(env, seed.horizon_ticks)
    print(f"  mock agent actions = {[a.dominant for a in actions]}")
    for action in actions:
        ret = env.step(action)
        if ret.done:
            break
    gt = env.ground_truth()
    print(
        f"  cost components    = {json.dumps(gt.get('cost_components', {}), indent=2)}"
    )


def main() -> int:
    run_synthetic_spike()
    run_grid2op_spike()
    print("\n[spike] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
