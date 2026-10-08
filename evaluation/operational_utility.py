"""0.25 continuous, source-scaled operational outcomes; no policy calibration.

Native penalties already encode service importance, lateness and resources.
They enter the objective once. Service and capability profiles explain outcomes,
not extra rewards for tool use, process verbosity or measured episode length.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from evaluation.leaderboard import _macro, PRIMARY_LEADERBOARD_FORMULA_VERSION
from evaluation.native_scorecard import _read
from evaluation.operational_service import measure_service, SERVICE_BACKENDS
from evaluation.scorer import score_system_survival

VERSION = "0.25.0"
REVISION = "source_scaled_operational_outcomes.v2"


def utility(cost: float, scale: float, *, signed: bool) -> float:
    if (
        any(type(x) not in (int, float) or not math.isfinite(x) for x in (cost, scale))
        or scale <= 0
    ):
        raise ValueError("invalid utility cost or source scale")
    if not signed and cost < 0:
        raise ValueError("negative cost in nonnegative objective")
    return 100 * (0.5 - math.atan(cost / scale) / math.pi)


def native_outcome_quality(cost: float, scale: float, *, signed: bool) -> float:
    """Monotone source-normalized settled-cost score, not task attainment.

    Zero cost receives 100; a cost equal to the frozen source exposure receives
    50. Every finite nonnegative cost remains distinguishable, while native hard
    failures are gated to zero by the caller. Signed negative settled cost is
    retained as a raw outcome but receives no unbounded main-score bonus.
    """
    if (
        any(type(x) not in (int, float) or not math.isfinite(x) for x in (cost, scale))
        or scale <= 0
    ):
        raise ValueError("invalid native outcome cost or source scale")
    if not signed and cost < 0:
        raise ValueError("negative cost in nonnegative objective")
    ratio = cost / scale
    if not math.isfinite(ratio):
        raise ValueError("nonfinite native outcome ratio")
    quality = 100 / (1 + ratio) if ratio >= 0 else 100.0
    if not math.isfinite(quality):
        raise ValueError("nonfinite native outcome quality")
    return quality


def capability_strata(scenario: dict) -> dict:
    """Outcome slices selected before observing any model; not causal credit."""
    cfg = scenario.get("backend_config") or {}
    recipe = cfg.get("response_window_recipe") or {}
    pre, trigger = (
        recipe.get("preposition_opportunity_tick"),
        recipe.get("event_trigger_tick"),
    )
    anticipation = type(pre) is int and type(trigger) is int and 0 <= pre < trigger
    hidden = any(
        p.get("hidden") is True for p in scenario.get("perturbations") or []
    ) or bool(cfg.get("hidden_source_event_types"))
    phase = cfg.get("task_contract") or {}
    multi = len(phase.get("phase_ticks") or []) > 1
    intertemporal = scenario["backend_kind"] in {
        "citylearn",
        "orgym_invmgmt",
        "dynasched_flexible_job_shop",
        "pglib_uc_synthetic",
        "pymgrid_economic_dispatch",
        "pandapower_lv",
    }
    return {
        name: dict(
            status="source_declared" if active else "not_applicable",
            definition=definition,
            interpretation="task_outcome_on_source_opportunity_not_isolated_causal_ability",
        )
        for name, active, definition in (
            (
                "anticipatory_opportunity_outcome",
                anticipation,
                "source response recipe has preposition tick before event",
            ),
            (
                "hidden_change_outcome",
                hidden,
                "source contains hidden perturbations or source event types",
            ),
            (
                "multi_stage_outcome",
                multi,
                "source task contract declares multiple phase ticks",
            ),
            (
                "intertemporal_scheduling_outcome",
                intertemporal,
                "native storage, inventory, precedence or commitment couples decisions across time",
            ),
        )
    }


def _completed_runtime(payload: dict, binding: dict, row: dict) -> dict | None:
    """Recover relocated but hash-bound terminal truth, never execute it."""
    descriptor = payload.get("completed_runtime_artifact") or {}
    if not descriptor.get("sha256") or not descriptor.get("path"):
        return None
    snapshot_path = Path(binding["artifacts"]["scoring_inputs_artifact"]["path"])
    candidates = [
        Path(descriptor["path"]),
        snapshot_path.parent / Path(descriptor["path"]).name,
    ]
    for path in dict.fromkeys(candidates):
        if not path.is_file():
            continue
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != descriptor["sha256"]:
            raise ValueError("completed_runtime_hash_mismatch")
        runtime = json.loads(raw)["payload"]
        if any(
            runtime.get("identity", {}).get(k) != row[k]
            for k in ("scenario_signature", "seed")
        ):
            raise ValueError("completed_runtime_identity_mismatch")
        inputs = payload["inputs"]
        if runtime.get("backend_tick_records") != inputs.get(
            "backend_tick_records"
        ) or (runtime.get("ground_truth") or {}).get("cost_components") != inputs.get(
            "cost_components"
        ):
            raise ValueError("completed_runtime_snapshot_mismatch")
        return runtime
    return None


def _recover_settlement(scenario, inputs, trace, runtime):
    if scenario["backend_kind"] == "orgym_invmgmt":
        from evaluation.operational_settlement_inventory import (
            recover_inventory_settlement,
        )

        return recover_inventory_settlement(
            scenario, snapshot_inputs=inputs, trace=trace
        )
    from evaluation.operational_settlement_energy import recover_energy_settlement

    return recover_energy_settlement(
        scenario,
        snapshot_inputs=inputs,
        trace=trace,
        backend_kind=scenario["backend_kind"],
        completed_runtime=runtime,
    )


def _native_survival_records(backend_kind, records, ledger):
    """Use native catastrophe evidence, including authenticated legacy fields.

    Older scoring snapshots omitted backend-specific catastrophe fields. The
    generic survival fallback treats five voltage violations or 200 MW balance
    error as a blackout, which is not the native catastrophe definition for
    these backends. Recover only fields already bound to the episode ledger.
    """
    if all(type(record.get("catastrophic_failure")) is bool for record in records):
        return records, "native_snapshot_marker"
    if any("catastrophic_failure" in record for record in records):
        return None, "mixed_native_catastrophe_schema"

    if backend_kind in {"pglib_uc_synthetic", "citylearn"}:
        if any(
            "done" in record and (type(record["done"]) is not bool or record["done"])
            for record in records
        ):
            return None, "native_catastrophe_status_unknown"
        return [
            {
                **record,
                "catastrophic_failure_applicable": False,
                "catastrophic_failure_inapplicable_reason": (
                    "backend_has_no_native_catastrophe_measure"
                ),
            }
            for record in records
        ], "native_catastrophe_unmodeled"

    engine = [
        entry.get("payload")
        for entry in ledger
        if entry.get("source") == "engine" and entry.get("kind") == "backend_tick"
    ]
    if len(engine) != len(records) or any(not isinstance(p, dict) for p in engine):
        return None, "native_catastrophe_ledger_missing"
    if any(
        record.get("tick") != payload.get("tick")
        for record, payload in zip(records, engine, strict=True)
        if "tick" in record and "tick" in payload
    ):
        return None, "native_catastrophe_clock_mismatch"
    if any(
        type(record[key]) not in (int, float)
        or type(payload[key]) not in (int, float)
        or not math.isfinite(record[key])
        or not math.isfinite(payload[key])
        or not math.isclose(record[key], payload[key], rel_tol=1e-9, abs_tol=1e-9)
        for record, payload in zip(records, engine, strict=True)
        for key in (
            "n_voltage_violations",
            "balance_error_mw",
            "aggregate_demand_mw",
            "aggregate_generation_mw",
        )
        if key in record and key in payload
    ):
        return None, "native_catastrophe_state_mismatch"
    if all(type(payload.get("catastrophic_failure")) is bool for payload in engine):
        return [
            {**record, "catastrophic_failure": payload["catastrophic_failure"]}
            for record, payload in zip(records, engine, strict=True)
        ], "native_ledger_marker"

    if backend_kind in {
        "cigre_distribution",
        "opendss_fresh_feeders",
        "opendss_ieee13",
        "pandapower_lv",
    }:
        recovered = []
        for record, payload in zip(records, engine, strict=True):
            values = [
                value
                for value in (record.get("converged"), payload.get("converged"))
                if type(value) is bool
            ]
            if (
                not values
                or len(set(values)) != 1
                or type(record.get("done")) is not bool
            ):
                return None, "native_convergence_evidence_missing"
            failed = not values[0]
            if backend_kind in {"cigre_distribution", "pandapower_lv"}:
                failed = failed or record["done"]
            recovered.append({**record, "catastrophic_failure": failed})
        return recovered, "native_convergence_recovered"

    if backend_kind == "pymgrid_economic_dispatch":
        recovered = []
        collapsed = False
        for record, payload in zip(records, engine, strict=True):
            if type(payload.get("collapsed")) is not bool:
                return None, "native_collapse_evidence_missing"
            collapsed = collapsed or payload["collapsed"]
            recovered.append(
                {
                    **record,
                    "catastrophic_failure": collapsed or bool(record.get("done")),
                }
            )
        return recovered, "native_collapse_recovered"

    return None, "native_catastrophe_status_unknown"


def evaluate_operational(
    row: dict,
    contract: dict,
    scenario: dict,
    *,
    scoring_mode="legacy_utility",
    acceptance_contract=None,
) -> dict:
    if scoring_mode not in {"legacy_utility", "acceptance", "native_outcome"}:
        raise ValueError("unsupported_operational_scoring_mode")
    result = dict(
        score=None,
        reason="unbound_episode_artifacts",
        evidence_ids=[],
        contract_sha256=contract["contract_sha256"],
        protocol_revision=REVISION,
        interpretation="source_scaled_operational_utility_not_success_probability",
        safety=None,
    )

    def missing(reason):
        return {**result, "reason": reason}

    if any(
        row.get(k) != contract.get(k)
        for k in ("scenario_signature", "seed", "domain", "backend_kind")
    ):
        return missing("operational_identity_mismatch")
    blocker = row.get("blocker")
    if blocker and blocker not in (
        "multiple_comparison_scopes",
        "incomplete_observation_window",
    ):
        return missing(blocker)
    if (
        blocker == "multiple_comparison_scopes"
        and row.get("individual_execution_verified") is not True
    ):
        return missing("individual_execution_identity_unproven")
    binding = row.get("artifact_binding") or {}
    if not (
        binding.get("verified") is True and binding.get("native_cost_bound") is True
    ):
        return result
    payload = _read(binding, "scoring_inputs_artifact")["payload"]
    if any(
        payload.get("identity", {}).get(k) != row[k]
        for k in ("scenario_signature", "seed")
    ):
        return missing("snapshot_operational_identity_mismatch")
    inputs = payload["inputs"]
    ledger = _read(binding, "evidence_ledger_artifact")
    ids = {e["evidence_id"]: e for e in ledger}
    items = (inputs.get("evidence_logger") or {}).get("items")
    if (
        len(ids) != len(ledger)
        or not isinstance(items, list)
        or any(ids.get(e.get("evidence_id")) != e for e in items)
    ):
        return missing("snapshot_evidence_mismatch")
    state_ids = [
        e["evidence_id"]
        for e in ledger
        if e.get("source") == "engine" and e.get("kind") == "backend_tick"
    ]
    records, fatal = (
        inputs.get("backend_tick_records"),
        inputs.get("chose_fatal_option"),
    )
    if (
        not state_ids
        or not isinstance(records, list)
        or not records
        or type(fatal) is not bool
    ):
        return missing("native_state_evidence_missing")
    survival_records, safety_source = _native_survival_records(
        row["backend_kind"], records, ledger
    )
    if survival_records is None:
        return missing(safety_source)
    survival = asdict(score_system_survival(survival_records, evidence_ids=state_ids))
    if not (
        survival["applicable"] is True
        or str(survival.get("reason")).startswith("native_catastrophe_not_modeled:")
    ):
        return missing("native_safety_unknown")
    hard = fatal or (survival["applicable"] is True and survival["floor_violation"])
    result.update(
        safety=dict(
            verified=True,
            hard_failure=hard,
            evidence_ids=state_ids,
            signal_source=safety_source,
        ),
        evidence_ids=state_ids,
    )
    costs = inputs.get("cost_components")
    if (
        not isinstance(costs, dict)
        or not costs
        or any(
            type(v) not in (int, float) or not math.isfinite(v) for v in costs.values()
        )
    ):
        return missing("native_cost_invalid")
    signed_components = {
        "citylearn": {"energy_cost", "terminal_storage_settlement"},
        "orgym_invmgmt": {"inventory_asset_settlement"},
        "pandapower_lv": {"terminal_storage_settlement"},
    }.get(row["backend_kind"], set())
    if any(value < 0 and key not in signed_components for key, value in costs.items()):
        return missing("invalid_native_component_value_domain")
    allowed = contract.get("allowed_cost_components")
    if allowed is not None and not set(costs) <= set(allowed):
        return missing("unknown_native_cost_component")
    absent = set(contract.get("required_cost_components") or []) - set(costs)
    recoverable = {
        "citylearn": {"terminal_storage_settlement"},
        "pandapower_lv": {"terminal_storage_settlement"},
        "orgym_invmgmt": {"inventory_asset_settlement"},
    }.get(row["backend_kind"], set())
    if absent - recoverable:
        return missing("missing_native_cost_component")
    native = row.get("native_outcome") or {}
    if native.get("applicable") is not True:
        return missing(native.get("reason", "native_objective_missing"))
    cost = sum(costs.values())
    scale = contract["scales"]["native_cost"]["value"]
    signed = contract["native_cost_value_domain"] == "signed"
    if row["backend_kind"] == "pymgrid_economic_dispatch":
        if any(
            type(r.get(k)) not in (int, float) or not math.isfinite(r[k])
            for r in records
            for k in ("balance_error_mw", "shed_penalty")
        ):
            return missing("native_state_loss_records_invalid")
        cost = (
            sum(abs(r["balance_error_mw"]) * 200 for r in records)
            + costs["shed_penalty"]
        )
        scale = contract["scales"]["loss"]["value"]
        signed = False
    if not math.isclose(
        cost, native.get("actual_cost", math.nan), rel_tol=1e-10, abs_tol=1e-7
    ):
        return missing("cached_native_objective_mismatch")
    if not signed and cost < 0:
        return missing("negative_native_objective")
    trace = _read(binding, "trajectory_artifact")
    if (
        not trace
        or [r.get("tick") for r in trace] != list(range(1, len(trace) + 1))
        or len(records) != len(trace)
        or len(trace) != binding.get("trace_ticks")
    ):
        return missing("native_trace_clock_invalid")
    recorded_cost = cost
    if absent:
        runtime = (
            _completed_runtime(payload, binding, row)
            if row["backend_kind"] in {"citylearn", "pandapower_lv"}
            else None
        )
        recovery = _recover_settlement(scenario, inputs, trace, runtime)
        additions = recovery.get("components") or {}
        recovery_ids = recovery.get("evidence_ids") or []
        if recovery.get("applicable") is not True:
            return missing(
                recovery.get("reason", "terminal_settlement_recovery_unavailable")
            )
        if (
            set(additions) != absent
            or not recovery_ids
            or any(e not in ids for e in recovery_ids)
            or any(
                type(v) not in (int, float) or not math.isfinite(v)
                for v in additions.values()
            )
        ):
            return missing("terminal_settlement_recovery_evidence_invalid")
        result["terminal_settlement_recovery"] = {
            **recovery,
            "completed_runtime_artifact": payload.get("completed_runtime_artifact")
            if runtime is not None
            else None,
        }
        result["recorded_components"] = costs
        costs = {**costs, **additions}
        cost = sum(costs.values())
        result["evidence_ids"] = list(dict.fromkeys(state_ids + recovery_ids))
    service = measure_service(
        scenario, snapshot_inputs=inputs, trace=trace, source_contract=contract
    )
    if any(e not in ids for e in service.get("evidence_ids", [])):
        return missing("service_evidence_unbound")
    result["service"] = service
    native_cost = cost
    if row["backend_kind"] == "dynasched_flexible_job_shop":
        obs = trace[-1].get("observation") or {}
        makespan = obs.get("makespan")
        completed = service.get("numerator")
        required = service.get("denominator")
        if (
            service.get("score") is None
            or type(makespan) not in (int, float)
            or not math.isfinite(makespan)
            or makespan < 0
        ):
            return missing("source_schedule_settlement_unavailable")
        cost = makespan + 1000.0 * (required - completed)
        result["obligation_settlement"] = dict(
            original_native_cost=native_cost,
            corrected_cost=cost,
            correction=cost - native_cost,
            definition="makespan_plus_1000_times_all_source_operations_not_completed_including_cancelled_and_unarrived",
        )
    complete_early = (
        row["backend_kind"] == "dynasched_flexible_job_shop"
        and service.get("score") == 100
    )
    complete_early = complete_early or (
        row["backend_kind"] == "sumo_ego"
        and records[-1].get("route_progress") == 1.0
        and records[-1].get("done") is True
        and not hard
    )
    if len(trace) > contract["horizon_ticks"] or (
        len(trace) != contract["horizon_ticks"] and not hard and not complete_early
    ):
        return missing("incomplete_observation_window")
    precision = (result.get("terminal_settlement_recovery") or {}).get(
        "cost_precision_bound", 0.0
    )
    if (
        type(precision) not in (int, float)
        or not math.isfinite(precision)
        or precision < 0
    ):
        return missing("terminal_settlement_precision_invalid")
    interval = (
        [0.0, 0.0]
        if hard
        else [
            utility(cost + precision, scale, signed=signed),
            utility(cost - precision, scale, signed=signed),
        ]
    )
    result.update(
        score=0.0 if hard else utility(cost, scale, signed=signed),
        score_interval_from_recovered_precision=interval,
        reason="native_hard_failure" if hard else "source_scaled_native_outcome",
        native_objective=cost,
        native_objective_id=native.get("objective_id"),
        recorded_native_objective=recorded_cost,
        scored_objective_id="fjsp.source_obligation_settled_loss.v1"
        if "obligation_settlement" in result
        else native.get("objective_id"),
        source_scale=scale,
        signed_objective=signed,
        components=costs,
        component_policy="native_weights_once_no_extra_service_penalty",
        scale_sensitivity={
            str(factor): 0.0 if hard else utility(cost, scale * factor, signed=signed)
            for factor in (0.5, 1.0, 2.0)
        },
    )
    if scoring_mode == "native_outcome":
        low_cost = cost - precision
        if not signed:
            low_cost = max(0.0, low_cost)
        quality = 0.0 if hard else native_outcome_quality(cost, scale, signed=signed)
        quality_interval = (
            [0.0, 0.0]
            if hard
            else [
                native_outcome_quality(cost + precision, scale, signed=signed),
                native_outcome_quality(low_cost, scale, signed=signed),
            ]
        )
        legacy_sensitivity = result.pop("scale_sensitivity")
        result.update(
            native_evidence_ids=list(result["evidence_ids"]),
            legacy_utility_score=result["score"],
            legacy_utility_scale_sensitivity=legacy_sensitivity,
            legacy_utility_included_in_primary=False,
            score=quality,
            score_interval=quality_interval,
            scale_sensitivity={
                str(factor): 0.0
                if hard
                else native_outcome_quality(cost, scale * factor, signed=signed)
                for factor in (0.5, 1.0, 2.0)
            },
            attained=None,
            protocol_revision="source_grounded_native_outcome.v3",
            interpretation="settled_native_loss_quality_not_task_attainment",
            reason="native_hard_failure"
            if hard
            else "source_scaled_settled_native_loss",
        )
    if scoring_mode == "acceptance":
        from evaluation.operational_acceptance import score_acceptance
        from evaluation.operational_protocol import objective_identity

        result.update(
            native_evidence_ids=list(result["evidence_ids"]),
            legacy_utility_score=result["score"],
            legacy_utility_interval=result["score_interval_from_recovered_precision"],
            legacy_utility_included_in_primary=False,
            score=None,
            attained=None,
            protocol_revision="frozen_operational_acceptance.v4",
            interpretation="fixed_operational_acceptance_not_behavior_reward",
        )
        if acceptance_contract is None:
            result.update(reason="acceptance_contract_missing", applicable=False)
        else:
            native_terminal_allowed = (
                acceptance_contract.get("horizon_policy", "fixed_horizon")
                == "native_mission_terminal"
                and complete_early
            )
            effective_horizon = (
                len(trace) if native_terminal_allowed else contract["horizon_ticks"]
            )
            precision_components = {
                key: precision
                for key in (result.get("terminal_settlement_recovery") or {}).get(
                    "components", {}
                )
            }
            measurement = {
                **{
                    k: contract[k]
                    for k in ("scenario_signature", "seed", "domain", "backend_kind")
                },
                **objective_identity(contract),
                "scenario_sha256": contract["scenario_sha256"],
                "applicable": True,
                "cost": cost,
                "hard_failure": hard,
                "cost_precision_bound": precision,
                "component_precision_bounds": precision_components,
                "service": service,
                "components": costs,
                "evidence_ids": result["evidence_ids"],
                "horizon_ticks": effective_horizon,
                "configured_horizon_ticks": contract["horizon_ticks"],
                "early_completion_verified": native_terminal_allowed,
                "horizon_policy": acceptance_contract.get(
                    "horizon_policy", "fixed_horizon"
                ),
                "complete_observation_window": len(trace) == contract["horizon_ticks"]
                or native_terminal_allowed,
            }
            acceptance = score_acceptance(
                measurement,
                acceptance_contract,
                snapshot_inputs={**inputs, "evidence_ids": state_ids},
                trace=trace,
                valid_evidence_ids=ids,
            )
            result.update(acceptance)
    return result


def aggregate_operational(
    rows: list[dict], contracts: list[dict], *, models: list[str]
) -> dict:
    suite = {(c["scenario_signature"], c["seed"]): c for c in contracts}
    if not suite or len(suite) != len(contracts) or len(set(models)) != len(models):
        raise ValueError("empty or duplicate operational suite/model")
    indexed = {}
    for row in rows:
        key = (row["model"], row["scenario_signature"], row["seed"])
        if key in indexed:
            raise ValueError("duplicate operational observation")
        if key[0] not in models or key[1:] not in suite:
            raise ValueError("foreign operational observation")
        indexed[key] = row

    def aggregate(scope):
        output = []
        for model in models:
            scores = defaultdict(list)
            sensitivity = {str(f): defaultdict(list) for f in (0.5, 1.0, 2.0)}
            interval_values = [defaultdict(list), defaultdict(list)]
            issues = []
            n_scored = n_hard = 0
            for key, c in scope.items():
                r = indexed.get((model, *key))
                m = (r or {}).get("mission") or {}
                s = m.get("score")
                if (
                    type(s) not in (int, float)
                    or not math.isfinite(s)
                    or not 0 <= s <= 100
                    or not m.get("evidence_ids")
                ):
                    issues.append(
                        dict(case=list(key), reason=m.get("reason", "missing_case"))
                    )
                    continue
                group = (c["domain"], c["backend_kind"], c["source_denominator_key"])
                scores[group].append(s)
                for i, v in enumerate(
                    m.get("score_interval_from_recovered_precision", [s, s])
                ):
                    interval_values[i][group].append(v)
                for f in sensitivity:
                    v = m.get("scale_sensitivity", {}).get(f)
                    if v is not None:
                        sensitivity[f][group].append(v)
                n_scored += 1
                n_hard += int((m.get("safety") or {}).get("hard_failure") is True)
            complete = bool(scope) and n_scored == len(scope) and not issues
            score, sources, backends, domains = (
                _macro(scores) if complete else (None, {}, {}, {})
            )
            output.append(
                dict(
                    model=model,
                    primary_score=score,
                    primary_rank=None,
                    score=score,
                    rank=None,
                    complete=complete,
                    n_expected=len(scope),
                    n_scored=n_scored,
                    score_interval_from_recovered_precision=[
                        _macro(v)[0] for v in interval_values
                    ]
                    if complete
                    else None,
                    n_hard_failures=n_hard,
                    issues=issues,
                    source_scores=sources,
                    backend_scores=backends,
                    domain_scores=domains,
                    scale_sensitivity={
                        f: _macro(v)[0]
                        if complete and sum(map(len, v.values())) == len(scope)
                        else None
                        for f, v in sensitivity.items()
                    },
                )
            )
        ranked = sorted(
            (r for r in output if r["complete"]),
            key=lambda r: (-r["score"], r["model"]),
        )
        prior = None
        rank = None
        for i, r in enumerate(ranked, 1):
            if prior != r["score"]:
                rank = i
            r["primary_rank"] = r["rank"] = rank
            prior = r["score"]
            low, high = r["score_interval_from_recovered_precision"]
            r["quantization_overlap_models"] = [
                other["model"]
                for other in ranked
                if other["model"] != r["model"]
                and low <= other["score_interval_from_recovered_precision"][1]
                and high >= other["score_interval_from_recovered_precision"][0]
            ]
        return dict(models=output, leaderboard=ranked, n_expected=len(scope))

    result = aggregate(suite)
    # Diagnostic means also have a source-fixed denominator; missing measurements
    # cannot improve a model's service or timely-delivery profile.
    for item in result["models"]:
        item["service_profile"] = {}
        for name, backends in (
            ("fulfillment", SERVICE_BACKENDS),
            ("timely_delivery", {"pyvrp_cvrp", "pyvrp_vrptw"}),
        ):
            scope = {
                key: c for key, c in suite.items() if c["backend_kind"] in backends
            }
            values = defaultdict(list)
            count = 0
            for key, c in scope.items():
                r = indexed.get((item["model"], *key)) or {}
                service = (r.get("mission") or {}).get("service") or {}
                measurement = (
                    service
                    if name == "fulfillment"
                    else service.get("timeliness") or {}
                )
                score = measurement.get("score")
                if (
                    type(score) in (int, float)
                    and math.isfinite(score)
                    and 0 <= score <= 100
                    and service.get("evidence_ids")
                ):
                    values[
                        (c["domain"], c["backend_kind"], c["source_denominator_key"])
                    ].append(score)
                    count += 1
            item["service_profile"][name] = dict(
                score=_macro(values)[0] if scope and count == len(scope) else None,
                n_expected=len(scope),
                n_scored=count,
                included_in_primary_separately=False,
            )
    result.update(
        schema_version=REVISION,
        protocol_revision=REVISION,
        evaluation_version=VERSION,
        aggregation=PRIMARY_LEADERBOARD_FORMULA_VERSION,
        suite_cases=len(suite),
        scorable_cases=len(suite),
        scoring_coverage_complete=True,
        score_units="source_scaled_operational_utility_0_100_not_percent_success",
        full_suite_leaderboard=result["leaderboard"],
        slices={
            name: aggregate(
                {
                    k: c
                    for k, c in suite.items()
                    if c.get("strata", {}).get(name, {}).get("status")
                    == "source_declared"
                }
            )
            for name in sorted({n for c in contracts for n in c.get("strata", {})})
        },
        reason_counts=dict(Counter(r["mission"]["reason"] for r in rows)),
        formal_run_certified=False,
        leaderboard_eligible=False,
    )
    return result
