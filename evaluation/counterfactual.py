"""
evaluation.counterfactual — Power-grid-flavoured wrapper over
``core.counterfactual.run_counterfactual``.

Provides the canonical ``cost_extractor`` used by the v0.1 scorer, plus a
convenience function ``power_grid_counterfactual_report`` that the runner
calls right after the actual run finishes.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

from core import (
    Action,
    CounterfactualReport,
    backend_opt_out_report,
    run_counterfactual,
    wait_only_policy,
)
from core.counterfactual import keep_investigations_policy_for_env
from core.pomdp_env import POMDPEnvironment


def power_grid_cost_extractor(ground_truth: dict[str, Any]) -> dict[str, float]:
    """Extract the scoring-relevant cost components from a ground-truth snap.

    Both backends now emit a unified vocabulary (production_cost, shed_penalty,
    balance_error_cost, reserve_violation_cost, safety_violation_cost,
    startup_cost). Older Grid2Op-style keys are still accepted for backward
    compatibility with any pre-v0.1.1 trajectory files.
    """
    components = dict(ground_truth.get("cost_components", {}))
    if "overload_cost" in components and "balance_error_cost" not in components:
        components["balance_error_cost"] = components.pop("overload_cost")
    if "disconnection_cost" in components and "safety_violation_cost" not in components:
        components["safety_violation_cost"] = components.pop("disconnection_cost")
    if "shed_cost" in components and "shed_penalty" not in components:
        components["shed_penalty"] = components.pop("shed_cost")
    return components


def power_grid_counterfactual_report(
    env_factory: Callable[[], POMDPEnvironment],
    scenario_config: dict[str, Any],
    seed: int,
    actual_actions: list[Action],
    masking_policy: str = "wait_only",
    per_action: bool = False,
    per_action_cap: int | None = 20,
    per_action_groups: bool = False,
    per_action_group_cap: int | None = 20,
) -> CounterfactualReport:
    """Run a counterfactual replay against the wait-only or investigations-only
    masking policy. Returns the :class:`CounterfactualReport` dataclass.

    Red line #5 (release contract): every scenario must support masked-action
    replay or explicitly opt out with a machine-readable reason. Prior to
    this check, ``POMDPEnvironment.supports_counterfactual()`` was a defined
    override hook that no real call site actually consulted — every backend
    got a replay attempt regardless of what it returned. This probes it
    first and, when a backend opts out, returns
    :func:`core.backend_opt_out_report` instead of spending a replay on an
    environment that declared itself unsupported.
    """
    probe_env = env_factory()
    readonly_tool_names: set[str] | None = None
    try:
        probe_env.reset(copy.deepcopy(scenario_config), seed)
        if not probe_env.supports_counterfactual():
            return backend_opt_out_report(
                masking_policy=masking_policy,
                backend_domain=getattr(probe_env, "domain", ""),
            )
        if masking_policy == "wait_only":
            mask_fn = wait_only_policy
        elif masking_policy == "keep_investigations":
            # Derive the domain's read-only tool set from the already-reset
            # probe env so the keep-investigations baseline is
            # domain-agnostic (not hardcoded to power-grid tool names).
            mask_fn = keep_investigations_policy_for_env(probe_env)
        else:
            raise ValueError(f"unknown masking policy: {masking_policy}")
        readonly_tool_names = probe_env.readonly_tool_names()
    finally:
        probe_env.close()
    return run_counterfactual(
        env_factory=env_factory,
        scenario_config=scenario_config,
        seed=seed,
        actual_actions=actual_actions,
        cost_extractor=power_grid_cost_extractor,
        masking_policy=mask_fn,
        masking_label=masking_policy,
        per_action=per_action,
        per_action_cap=per_action_cap,
        per_action_groups=per_action_groups,
        per_action_group_cap=per_action_group_cap,
        readonly_tool_names=readonly_tool_names,
    )


# ── Domain-neutral aliases (T0) ──────────────────────────────────────────────
# The cost extractor only reads ``ground_truth['cost_components']`` (emitted by
# every domain, with harmless legacy-key remaps), and the report is already
# ``env_factory``-parameterized — so the same functions serve all v0.7 domains.
# The ``power_grid_*`` names are retained for v0.1–v0.6 import compatibility.
domain_cost_extractor = power_grid_cost_extractor
domain_counterfactual_report = power_grid_counterfactual_report
