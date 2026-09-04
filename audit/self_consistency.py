"""Self-consistency + evidence completeness audit checks.

Runs a quick wait_only episode on each release scenario to verify it
does not crash and produces a finite reward, with evidence completeness
validation.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from audit._common import (
    DEFAULT_REGISTRY_PATH,
    _load_registry,
    _load_yaml,
    _resolve_scenario_path,
)
from audit.episode_cache import active_episode_cache
from baselines import WaitOnlyAgent
from core.protocol21_evidence import canonicalize_repo_owned_paths
from domains.registry import build_backend_records as _build_backend_records
from domains.registry import (
    get_backend_capability,
    get_domain_spec,
    reference_optimum_from_backend_config,
)
from evaluation import (
    ScoringInputs,
    domain_counterfactual_report,
    evaluate_foresight,
    evaluate_task_completion,
    score_episode,
    separate_task_outcome_and_process,
)
from runner.episode import _run_episode_loop


def quick_run(scenario: dict[str, Any], agent_cls) -> dict[str, Any]:
    seed = int(scenario.get("seed", 42))
    capability = get_backend_capability(str(scenario.get("backend_kind") or ""))
    baseline_intervals = [
        value
        for value in (
            int(capability.periodic_scan_every_ticks),
            int(capability.max_review_after_ticks),
        )
        if value > 0
    ]
    # T0: domain-dispatched env factory (defaults to power_grid).
    env = get_domain_spec(scenario.get("domain")).env_factory()()
    try:
        env.reset(scenario, seed=seed)
        agent = agent_cls()
        agent.reset(env, scenario, seed=seed)
        run = _run_episode_loop(
            env=env,
            agent=agent,
            logger=None,
            baseline_scan_interval=(
                min(baseline_intervals) if baseline_intervals else None
            ),
        )
        return {
            "env": env,
            "actions": list(run["actions"]),
            "total_reward": float(run["episode_reward_total"]),
            "event_adaptive_autonomy": dict(run["event_adaptive_autonomy"]),
            "terminal_integrity": dict(run["terminal_integrity"]),
            "world_evolution_records": list(run["world_evolution_records"]),
        }
    except Exception:
        close = getattr(env, "close", None)
        if callable(close):
            close()
        raise


def episode_metrics(
    scenario: dict[str, Any],
    row: dict[str, Any],
    agent_cls: Any,
    *,
    difficulty_level: str,
    scenario_signature: str,
    include_load_classes: bool = True,
    replay_index: int = 0,
) -> dict[str, Any]:
    """Run or reuse a serializable audit episode summary."""

    score_context = {
        "metrics_schema_version": "0.3",
        "trajectory_metrics_version": "0.1",
        "difficulty_level": difficulty_level,
        "scenario_signature": scenario_signature,
        "include_load_classes": include_load_classes,
        "replay_index": replay_index,
    }

    def compute() -> dict[str, Any]:
        run = quick_run(scenario, agent_cls)
        env = run["env"]
        environment_closed = False

        def close_environment() -> None:
            nonlocal environment_closed
            if environment_closed:
                return
            environment_closed = True
            close = getattr(env, "close", None)
            if callable(close):
                close()

        try:
            scoring_inputs, ground_truth, source_consumption = (
                _scoring_inputs_for_quick_run(
                    scenario,
                    row,
                    run,
                    difficulty_level=difficulty_level,
                    scenario_signature=scenario_signature,
                    include_load_classes=include_load_classes,
                    close_environment=close_environment,
                )
            )
            score = score_episode(scoring_inputs)
            evidence = scoring_inputs.evidence_logger
            missing_evidence = [
                dim.name
                for dim in score.dimensions
                if dim.applicable and not dim.evidence_ids
            ]
            tool_calls = evidence.items_by_kind("tool_call") if evidence else []
            logical_state_changes: dict[str, bool] = {}
            logical_call_ticks: dict[str, int] = {}
            effective_tool_names: set[str] = set()
            effective_state_changing_ticks: set[int] = set()
            interaction_ticks: set[int] = set()
            for ordinal, item in enumerate(tool_calls):
                payload = item.payload or {}
                call_id = payload.get("call_id")
                key = str(call_id) if call_id else f"legacy-{ordinal}"
                materialized = str(payload.get("status") or "") != "pending"
                logical_call_ticks.setdefault(key, int(item.tick))
                if payload.get("name") not in {None, "wait", "noop"}:
                    interaction_ticks.add(int(item.tick))
                effective = (
                    payload.get("ok") is not False
                    and bool(payload.get("state_changing"))
                    and materialized
                )
                logical_state_changes[key] = (
                    logical_state_changes.get(key, False) or effective
                )
                if effective:
                    effective_tool_names.add(str(payload.get("name") or "<unknown>"))
                    effective_state_changing_ticks.add(int(item.tick))
            native_dimension_names = {
                "system_survival",
                "economic_cost",
                "safety_violation",
                "optimality_gap",
                "counterfactual_prevention",
            }
            native_dimension_scores = {
                dim.name: float(dim.raw_score)
                for dim in score.dimensions
                if dim.name in native_dimension_names and dim.applicable
            }
            counterfactual = scoring_inputs.counterfactual_report or {}
            task_completion = separate_task_outcome_and_process(
                evaluate_task_completion(
                    scenario=scenario,
                    ground_truth=ground_truth,
                    counterfactual=counterfactual,
                    score=score.to_dict(),
                ),
                scenario=scenario,
            )
            world_records = list(run.get("world_evolution_records") or [])
            terminal_integrity = dict(run.get("terminal_integrity") or {})
            autonomy = dict(run.get("event_adaptive_autonomy") or {})
            traffic_capture: dict[str, Any] = {}
            if str(scenario.get("domain") or "") == "traffic":
                from core.protocol21_traffic_capture import (
                    validate_source_lineage,
                    validate_vehicle_event_capture,
                )

                capture = dict(ground_truth.get("vehicle_control_capture") or {})
                capture_records = [
                    row for row in capture.get("records") or [] if isinstance(row, dict)
                ]
                traffic_capture = {
                    "status": capture.get("status"),
                    "record_count": int(
                        capture.get("record_count", len(capture_records)) or 0
                    ),
                    "returned_record_count": len(capture_records),
                    "truncated": bool(capture.get("truncated")),
                    "complete_vehicle_event_count": sum(
                        validate_vehicle_event_capture(row)["status"] == "complete"
                        for row in capture_records
                    ),
                    "verified_source_lineage_count": sum(
                        validate_source_lineage(row.get("source_lineage") or {})[
                            "lineage_verified"
                        ]
                        for row in capture_records
                    ),
                    "complete_source_identity_sha256": capture.get(
                        "complete_source_identity_sha256"
                    ),
                }
            return {
                "cost": float(
                    sum(
                        float(value)
                        for value in (
                            ground_truth.get("cost_components") or {}
                        ).values()
                    )
                ),
                "total_score": float(score.total_score),
                "raw_total": float(score.raw_total),
                "total_reward": float(run["total_reward"]),
                "missing_evidence_dimensions": missing_evidence,
                "tool_calls": len(tool_calls),
                "successful_non_wait_calls": sum(
                    bool((item.payload or {}).get("ok"))
                    and (item.payload or {}).get("name") != "wait"
                    for item in tool_calls
                ),
                "successful_state_changing_calls": sum(logical_state_changes.values()),
                "effective_tool_names": sorted(effective_tool_names),
                "effective_state_changing_ticks": sorted(
                    effective_state_changing_ticks
                ),
                "interaction_ticks": sorted(interaction_ticks),
                "effective_decision_ticks": len(effective_state_changing_ticks),
                "phase_depth_proxy": len(
                    {
                        logical_call_ticks[key]
                        for key, ok in logical_state_changes.items()
                        if ok
                    }
                ),
                "shortest_strategy_status": "pending_bounded_replay_minimization",
                "required_tool_set_status": "observed_effective_set_only",
                "prevented_loss": float(
                    counterfactual.get("prevented_loss", 0.0) or 0.0
                ),
                "normalized_prevention": float(
                    counterfactual.get("normalized_prevention", 0.0) or 0.0
                ),
                "native_dimension_scores": native_dimension_scores,
                "task_completion": task_completion,
                "world_evolution": {
                    "record_count": len(world_records),
                    "material_exogenous_event_count": sum(
                        bool(row.get("material_exogenous")) for row in world_records
                    ),
                    "source_scheduled_event_count": sum(
                        row.get("origin") == "source_schedule" for row in world_records
                    ),
                },
                "event_adaptive_autonomy": autonomy,
                "terminal_integrity": terminal_integrity,
                "traffic_capture": traffic_capture,
                "source_consumption_evidence": source_consumption,
            }
        finally:
            close_environment()

    cache = active_episode_cache()
    if cache is None:
        return compute()
    return cache.get_or_compute(
        scenario=scenario,
        row=row,
        agent_name=_agent_cache_identity(agent_cls),
        score_context=score_context,
        compute=compute,
    )


def _agent_cache_identity(agent_cls: Any) -> str:
    """Include implementation content so controller edits invalidate cache entries."""
    qualified_name = (
        f"{getattr(agent_cls, '__module__', '')}."
        f"{getattr(agent_cls, '__qualname__', getattr(agent_cls, '__name__', agent_cls))}"
    )
    try:
        source = inspect.getsource(agent_cls).encode("utf-8")
    except (OSError, TypeError):
        source = qualified_name.encode("utf-8")
    return f"{qualified_name}@{hashlib.sha256(source).hexdigest()[:16]}"


_AUDIT_LP_OPTIMUM_CACHE: dict[str, float | None] = {}


def _audit_maybe_lp_optimum(scenario: dict[str, Any]) -> float | None:
    """Runner-parity LP reference for aggregate UC audit quick runs."""
    import audit as _audit_pkg

    backend_kind = str(scenario.get("backend_kind", ""))
    if backend_kind != "pglib_uc_synthetic":
        return None
    case_rel = str((scenario.get("backend_config") or {}).get("case_file", ""))
    if not case_rel:
        return None
    horizon = int(scenario.get("horizon_ticks", 24))
    cache_key = f"{case_rel}@{horizon}"
    if cache_key in _AUDIT_LP_OPTIMUM_CACHE:
        return _AUDIT_LP_OPTIMUM_CACHE[cache_key]
    try:
        from evaluation.lp_oracle import lp_dispatch_optimum

        case_path = _audit_pkg.REPO_ROOT.parent / case_rel
        if not case_path.exists():
            return None
        with open(case_path, encoding="utf-8") as f:
            case = json.load(f)
        result = lp_dispatch_optimum(case, n_periods=horizon)
        if not result.feasible or result.optimum_cost <= 0:
            _AUDIT_LP_OPTIMUM_CACHE[cache_key] = None
            return None
        _AUDIT_LP_OPTIMUM_CACHE[cache_key] = result.optimum_cost
        return result.optimum_cost
    except Exception:
        return None


def _audit_reference_optimum(
    scenario: dict[str, Any],
    env: Any,
    spec: Any,
) -> float | None:
    """Mirror ``run.py`` optimality-gap reference selection for audit runs."""
    if spec.uses_runner_lp_oracle:
        lp_optimum = _audit_maybe_lp_optimum(scenario)
        if (
            lp_optimum is None
            and str(scenario.get("backend_kind", "")) == "pandapower_acopf"
        ):
            backend = getattr(env, "_backend", None)
            ref_fn = getattr(backend, "acopf_reference_optimum", None)
            if callable(ref_fn):
                try:
                    ref_val = float(ref_fn())
                    if ref_val > 0:
                        lp_optimum = ref_val
                except Exception:
                    lp_optimum = None
        return lp_optimum
    return reference_optimum_from_backend_config(env)


def _scoring_inputs_for_quick_run(
    scenario: dict[str, Any],
    row: dict[str, Any],
    run: dict[str, Any],
    *,
    difficulty_level: str,
    scenario_signature: str,
    include_load_classes: bool = True,
    close_environment: Callable[[], None],
) -> tuple[ScoringInputs, dict[str, Any], dict[str, Any]]:
    """Build scorer inputs for audit quick runs with runner-parity context."""
    env = run["env"]
    spec = get_domain_spec(scenario.get("domain"))
    gt = env.ground_truth()
    source_consumption = canonicalize_repo_owned_paths(
        env.source_consumption_evidence(scenario=scenario)
    )
    lp_optimum = _audit_reference_optimum(scenario, env, spec)
    evidence = env.evidence
    backend_tick_records = _build_backend_records(env)
    realized = [
        {**ev.payload, "tick": ev.tick}
        for ev in (evidence.items_by_kind("realized_event") if evidence else [])
    ]
    foresight = evaluate_foresight(evidence).to_dict() if evidence else None
    stakeholder_mgr = env.stakeholders
    dilemma_mgr = env.dilemmas
    completed_tick = getattr(env, "tick", 0)
    close_environment()

    cf = domain_counterfactual_report(
        env_factory=spec.env_factory(),
        scenario_config=scenario,
        seed=int(scenario.get("seed", 42)),
        actual_actions=run["actions"],
    )
    if evidence is not None:
        evidence.log(
            kind="counterfactual_result",
            tick=completed_tick,
            payload={
                "actual_cost": float(cf.actual_cost),
                "counterfactual_cost": float(cf.counterfactual_cost),
                "prevented_loss": float(cf.prevented_loss),
                "applicable": bool(cf.applicable),
                "reason_code": cf.reason_code,
                "masking_policy": cf.masking_policy,
            },
            source="engine",
        )
        if lp_optimum is not None:
            evidence.log(
                kind="lp_oracle",
                tick=completed_tick,
                payload={
                    "lp_optimum_cost": float(lp_optimum),
                    "production_cost": float(
                        gt["cost_components"].get("production_cost", 0.0)
                    ),
                },
                source="engine",
            )
    load_classes = (
        {
            la["load_id"]: la["stakeholder_class"]
            for la in scenario.get("load_assignments", [])
        }
        if include_load_classes
        else {}
    )
    task_counterfactual = cf.to_dict()
    task_counterfactual["_counterfactual_task_tick_records"] = (
        cf.counterfactual_ground_truth.get("_task_tick_records") or []
    )
    scoring_inputs = ScoringInputs(
        backend_tick_records=backend_tick_records,
        realized_events=realized,
        cost_components=gt["cost_components"],
        per_load_shed_mwh=gt.get(spec.equity_shed_key, {}),
        load_classes=load_classes,
        evidence_logger=evidence,
        stakeholder_mgr=stakeholder_mgr,
        dilemma_mgr=dilemma_mgr,
        chose_fatal_option=gt.get("chose_fatal_option", False),
        counterfactual_report=task_counterfactual,
        foresight_summary=foresight,
        lp_optimum=lp_optimum,
        difficulty_level=difficulty_level,
        scenario_signature=scenario_signature,
    )
    return scoring_inputs, gt, source_consumption


def _grid2op_available() -> bool:
    try:
        import grid2op  # type: ignore[import]  # noqa: F401

        return True
    except ImportError:
        return False


def _egret_available() -> bool:
    """True iff the EGRET AC-OPF stack (egret + pyomo) is importable.

    v0.3.4: the four episode-running checks skip ``egret_acopf`` scenarios
    when this is False (documented optional-runtime behaviour, mirroring the
    grid2op skip), so an audit on a host without IPOPT does not crash or fail
    the egret family -- it counts them as skipped. The hash + provenance checks
    always run, so the egret YAMLs are still integrity-verified.
    """
    import importlib.util as _u

    return _u.find_spec("egret") is not None and _u.find_spec("pyomo") is not None


def _pandapower_available() -> bool:
    try:
        import pandapower  # type: ignore[import]  # noqa: F401

        return True
    except ImportError:
        return False


def _runtime_unavailable(
    backend_kind: str,
    g2o: bool,
    egret: bool,
    pandapower: bool,
) -> bool:
    """Whether a scenario must be skipped because its optional backend runtime
    is absent on this host."""
    if backend_kind == "grid2op" and not g2o:
        return True
    if backend_kind == "egret_acopf" and not egret:
        return True
    return backend_kind in {"cigre_distribution", "pandapower_acopf"} and not pandapower


def _scenario_runtime_unavailable(
    scenario: dict[str, Any],
    g2o: bool,
    egret: bool,
    pandapower: bool,
) -> bool:
    """Check runtime and local dataset availability without remote loading."""
    backend_kind = str(scenario.get("backend_kind", ""))
    if _runtime_unavailable(backend_kind, g2o, egret, pandapower):
        return True
    if backend_kind == "sumo":
        return os.getenv("OPERATE_TRAFFIC_BACKEND_REAL") != "1"
    if backend_kind != "grid2op":
        return False
    try:
        import grid2op  # type: ignore[import]

        config = scenario.get("backend_config") or {}
        env_name = str(config.get("env_name") or "l2rpn_case14_sandbox")
        available = set(str(x) for x in grid2op.list_available_local_env())
        return (
            env_name not in available
            and env_name.removesuffix("_small") not in available
        )
    except Exception:
        return True


def check_self_consistency_and_evidence(
    samples_per_family: int = 2,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> tuple[int, int, list[str], int]:
    registry = _load_registry(registry_path)
    issues: list[str] = []
    families: dict[str, list[dict[str, Any]]] = {}
    for row in registry.get("scenarios", []):
        families.setdefault(row["family"], []).append(row)
    n_ok = 0
    n_total = 0
    n_skipped = 0
    g2o = _grid2op_available()
    egret = _egret_available()
    pandapower = _pandapower_available()
    for _fam, rows in families.items():
        for row in rows[:samples_per_family]:
            n_total += 1
            try:
                body = _load_yaml(_resolve_scenario_path(row["path"]))
                # Skip scenarios whose optional backend runtime is unavailable
                # (grid2op / egret_acopf / pandapower) — documented optional-runtime
                # behaviour; not a release defect.
                if _scenario_runtime_unavailable(body, g2o, egret, pandapower):
                    n_skipped += 1
                    continue
                metrics = episode_metrics(
                    body,
                    row,
                    WaitOnlyAgent,
                    difficulty_level=row["difficulty_level"],
                    scenario_signature=row["scenario_signature"],
                )
                if not math.isfinite(metrics["total_reward"]):
                    issues.append(f"{row['scenario_id']}: non-finite total_reward")
                    continue
                bad_dims = metrics["missing_evidence_dimensions"]
                if bad_dims:
                    issues.append(
                        f"{row['scenario_id']}: missing evidence on dims "
                        + ", ".join(bad_dims)
                    )
                    continue
                n_ok += 1
            except Exception as exc:
                issues.append(f"{row['scenario_id']}: {type(exc).__name__}: {exc}")
    return n_ok, n_total, issues, n_skipped
