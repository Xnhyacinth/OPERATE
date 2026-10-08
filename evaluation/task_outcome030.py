"""Task outcomes from freshly authenticated 0.28 episodes, without replay.

The caller owns original episode authentication. This extension rechecks source
contracts and native artifact bytes, retaining the historical report unchanged.
"""

from collections import Counter
from copy import deepcopy
import math
from pathlib import Path

import yaml

from evaluation.native_scorecard import _read
from evaluation.operational_outcome027 import (
    aggregate_outcome027,
    native_quality027,
    score_case027,
)
from evaluation.operational_outcome028 import outcome028
from evaluation.operational_weights import _key, _unique, validate_task_weights
from evaluation.trajectory_outcome027 import ROOT, canonical_digest, read_bound
from evaluation.trajectory_outcome028 import load_policy

VERSION = "0.30.0"
REVISION = "source_population_voltage_and_attained_recovery.v1"
VOLTAGE_BACKENDS = {"pandapower_lv", "cigre_distribution"}


def normalize_voltage_components(components, cost, records, contract, *, precision=0.0):
    """Retain non-voltage losses at their original scale and normalize bus-time."""
    nodes, horizon, scale = (
        contract[k] for k in ("source_bus_count", "horizon_ticks", "original_scale")
    )
    if any(type(v) is not int or v <= 0 for v in (nodes, horizon)):
        raise ValueError("invalid_voltage_source_population")
    if not isinstance(components, dict) or "voltage_violation_cost" not in components:
        raise ValueError("voltage_components_missing")
    if (
        any(
            type(v) not in (int, float) or not math.isfinite(v)
            for v in [cost, scale, precision, *components.values()]
        )
        or scale <= 0
        or precision < 0
    ):
        raise ValueError("invalid_voltage_component_number")
    if not math.isclose(
        math.fsum(components.values()), cost, rel_tol=1e-10, abs_tol=1e-7
    ):
        raise ValueError("voltage_component_total_mismatch")
    if not isinstance(records, list) or len(records) != horizon:
        raise ValueError("voltage_component_window_incomplete")
    counts = []
    for tick, record in enumerate(records):
        count = record.get("n_voltage_violations")
        if (
            type(record.get("tick")) is not int
            or record["tick"] != tick
            or type(count) is not int
            or not 0 <= count <= nodes
        ):
            raise ValueError("voltage_component_record_invalid")
        counts.append(count)
    voltage = components["voltage_violation_cost"]
    if voltage != 1200.0 * sum(counts):
        raise ValueError("voltage_component_record_mismatch")
    voltage_scale = 1200.0 * horizon * nodes
    other = math.fsum(v for k, v in components.items() if k != "voltage_violation_cost")
    loss = math.fsum([voltage / voltage_scale, other / scale])
    domain = contract["native_cost_value_domain"]
    if domain not in {"signed", "nonnegative"}:
        raise ValueError("invalid_source_cost_domain")
    signed = domain == "signed"
    quality = native_quality027(loss, 1.0, signed=signed)
    return dict(
        verified=True,
        native_cost=cost,
        native_cost_components=deepcopy(components),
        source_bus_count=nodes,
        horizon_ticks=horizon,
        voltage_scale=voltage_scale,
        original_scale=scale,
        voltage_loss=voltage / voltage_scale,
        other_component_loss=other / scale,
        dimensionless_loss=loss,
        native_quality=quality,
        quality_interval=[
            native_quality027(
                loss + d if signed else max(0.0, loss + d), 1.0, signed=signed
            )
            for d in (precision / scale, -precision / scale)
        ],
        cost_precision_bound=precision,
        signed_quality_saturated=loss < 0,
        precision_interpretation="recorded_point_components_with_inherited_settlement_recovery_bound_not_sampling_uncertainty",
        interpretation="source_bus_time_voltage_exposure_plus_original_scaled_remaining_loss",
    )


def measure_recovery_attainment(records, required):
    """Best nominal dwell in the latest MRM cycle; degraded ticks break dwell."""
    if (
        type(required) is not int
        or required <= 0
        or not isinstance(records, list)
        or not records
    ):
        raise ValueError("recovery_source_or_records_missing")
    mrm_modes = {"mrm_active", "minimal_risk_condition", "recovery_pending"}
    allowed = mrm_modes | {"nominal", "degraded"}
    cycle_maxima = []
    active = recovered = False
    current = maximum = terminal = 0
    previous = None
    for tick, record in enumerate(records):
        mode = record.get("assurance_mode")
        if (
            type(record.get("tick")) is not int
            or record["tick"] != tick
            or mode not in allowed
            or type(record.get("mrm_active")) is not bool
            or record["mrm_active"] != (mode in mrm_modes)
            or type(record.get("catastrophic_failure")) is not bool
        ):
            raise ValueError("native_recovery_state_invalid")
        if mode in mrm_modes:
            if not active or previous not in mrm_modes:
                if active:
                    cycle_maxima.append(maximum)
                active, recovered, maximum = True, False, 0
            current = 0
        elif mode == "nominal":
            if active:
                recovered = True
            current = current + 1 if recovered else 0
            maximum = max(maximum, current)
        else:
            current = 0
        terminal = terminal + 1 if mode == "nominal" else 0
        previous = mode
    if active:
        cycle_maxima.append(maximum)
    credited = min(maximum, required) if active else required
    return dict(
        score=100.0 * credited / required,
        numerator=credited,
        denominator=required,
        mrm_observed=active,
        recovery_required=active,
        attained_nominal_ticks_latest_cycle=maximum,
        attained_nominal_ticks_by_cycle=cycle_maxima,
        terminal_nominal_ticks=terminal,
        terminal_stability_score=100.0 * min(terminal, required) / required,
        terminal_assurance_mode=previous,
        reason="source_required_attained_recovery_dwell"
        if active
        else "no_mrm_guarded_recovery_not_required",
        interpretation="latest_mrm_obligation_attained_dwell_not_terminal_stability_or_route_completion",
        reset_rule="renewed_mrm_resets_attainment;degraded_breaks_consecutive_dwell;pre_mrm_nominal_excluded",
    )


def build_task_outcomes(base: dict, *, root: Path = ROOT) -> dict:
    """Extend an in-memory report freshly produced by the raw trajectory reader."""
    if base.get("evaluation_version") != "0.28.0":
        raise ValueError("task030_requires_authenticated_outcome028")
    raw, ref = read_bound(base["policy"])
    policy = load_policy(Path(ref["path"]), root=root)
    if (
        base["source_contracts"] != policy["source_contracts"]
        or base["task_weight_manifest"] != policy["task_weight_manifest"]
    ):
        raise ValueError("task030_source_policy_mismatch")
    contracts = _unique(base["source_contracts"]["contracts"])
    weights = validate_task_weights(
        base["task_weight_manifest"], base["source_contracts"]
    )
    populations = _unique(policy["voltage_population_contracts"])
    normalizations, recoveries = {}, {}
    for key, contract in contracts.items():
        backend = contract["backend_kind"]
        common = dict(
            scenario_signature=key[0],
            seed=key[1],
            source_contract_sha256=contract["contract_sha256"],
        )
        if backend in VOLTAGE_BACKENDS:
            population = populations[key]
            if (
                population["excluded_bus_ids"]
                or len(population["required_native_ticks"]) != contract["horizon_ticks"]
            ):
                raise ValueError("task030_whole_component_population_unsupported")
            normalization = dict(
                **common,
                population_contract_sha256=population["contract_sha256"],
                source_bus_count=len(population["required_bus_ids"]),
                horizon_ticks=contract["horizon_ticks"],
                original_scale=contract["scales"]["native_cost"]["value"],
                native_cost_value_domain=contract["native_cost_value_domain"],
                voltage_formula="C_voltage/(1200*H*M)",
                remaining_formula="sum(other_native_components)/original_S",
            )
            normalization["contract_sha256"] = canonical_digest(normalization)
            normalizations[key] = normalization
        elif backend == "sumo_ego":
            source_raw, _ = read_bound(
                dict(
                    path=str(Path(root) / contract["scenario_path"]),
                    sha256=contract["scenario_sha256"],
                )
            )
            scenario = yaml.safe_load(source_raw)
            requirement = scenario["backend_config"]["task_requirements"]
            if requirement.get("guarded_recovery_required_if_mrm") is not True:
                raise ValueError("task030_recovery_requirement_missing")
            recovery = dict(
                **common,
                required_stable_dwell_ticks=requirement["required_stable_dwell_ticks"],
                measurement="maximum_consecutive_nominal_dwell_after_native_mrm_in_latest_cycle",
                terminal_stability="separate_diagnostic",
            )
            recovery["contract_sha256"] = canonical_digest(recovery)
            recoveries[key] = recovery
    report = deepcopy(base)
    for row in report["graded_episodes"]:
        key = _key(row)
        old = row["outcome028"]
        result = deepcopy(old)
        normalization = deepcopy(row.get("normalization028"))
        completion = deepcopy(row.get("completion"))
        contract = contracts[key]
        row["backend_kind"] = contract["backend_kind"]
        if (
            old["evidence_qualified"]
            and old["hard_failure"] is not True
            and key in normalizations
        ):
            inputs = _read(row["artifact_binding"], "scoring_inputs_artifact")[
                "payload"
            ]["inputs"]
            mission = row["native_measurement027"]
            precision = (mission.get("terminal_settlement_recovery") or {}).get(
                "cost_precision_bound", 0.0
            )
            normalization = None
            if row.get("C") is not None and mission.get("components"):
                normalization = normalize_voltage_components(
                    mission["components"],
                    row["C"],
                    inputs["backend_tick_records"],
                    normalizations[key],
                    precision=precision,
                )
            result = outcome028(old, normalization)
            if normalization is None and result["hard_failure"] is False:
                result["reason"] = "native_voltage_component_measurement_missing"
        elif (
            old["evidence_qualified"]
            and key in recoveries
            and old["completion_measured"]
        ):
            inputs = _read(row["artifact_binding"], "scoring_inputs_artifact")[
                "payload"
            ]["inputs"]
            attained = measure_recovery_attainment(
                inputs["backend_tick_records"],
                recoveries[key]["required_stable_dwell_ticks"],
            )
            completion.update(attained)
            result = score_case027(
                row["native_measurement027"],
                completion,
                contract,
                evidence_verified=True,
            )
        result.update(evaluation_version=VERSION, protocol_revision=REVISION)
        row.update(
            outcome030=result,
            normalization030=normalization,
            completion030=completion,
            legacy_F028=row.get("F"),
            legacy_N028=row.get("N"),
            legacy_Q028=row.get("Q"),
            F=completion.get("score") if completion else None,
            N=result["native_quality"],
            Q=result["score"],
        )
    models = [r["model"] for r in base["models"]]
    aggregate = aggregate_outcome027(
        [{**r, "outcome027": r["outcome030"]} for r in report["graded_episodes"]],
        contracts,
        weights,
        models=models,
    )
    original_models = {r["model"]: r for r in base["models"]}
    for model in aggregate["models"]:
        for name in ("selected_actual_model_ids", "n_observed"):
            if name in original_models[model["model"]]:
                model[name] = original_models[model["model"]][name]
    report.update(aggregate)
    report.update(
        schema_version="operate_raw_trajectory_outcome030_report.v1",
        evaluation_version=VERSION,
        protocol_revision=REVISION,
        normalization_contracts030=list(normalizations.values()),
        recovery_contracts030=list(recoveries.values()),
        historical_models028=deepcopy(base["models"]),
    )
    rules = dict(
        evaluation_version=VERSION,
        protocol_revision=REVISION,
        voltage_loss="C_voltage/(1200*H*M)+sum(other_native_components)/original_S",
        native_quality="100/(1+loss);signed_negative_loss_saturates_at_100",
        joint_score="verified_hard_failure_zero_else_min(F,N);economic_only_N",
        recovery="latest_MRM_cycle_maximum_consecutive_nominal_supervisory_ticks/source_D;no_MRM_full_conditional_credit",
        reset="renewed_MRM_resets_max;degraded_breaks_streak_only;exclude_pre_MRM_nominal",
        weights="unchanged_authenticated028_task_weight_manifest",
        missingness="unchanged_unknown_is_NA_with_fixed_denominator_bounds",
        precision="retain_recorded_component_rounding_and_inherited_settlement_recovery_bound",
        authority="versioned_evaluator_implementation_bound_by_current_implementation;self_hash_identifies_rules_not_authority",
    )
    rules["rules_sha256"] = canonical_digest(rules)
    report["scoring_rules030"] = rules
    if "counts" in report:
        report["counts"]["numeric_Q"] = sum(
            row["Q"] is not None for row in report["graded_episodes"]
        )
        report["counts"]["complete_models"] = sum(
            row["complete"] for row in report["models"]
        )
    report["missing_by_model"] = {
        model: dict(
            Counter(
                (row.get("completion030") or {}).get("reason")
                or row["outcome030"].get("reason")
                or row.get("status", "missing")
                for row in report["graded_episodes"]
                if row["model"] == model and row["Q"] is None
            )
        )
        for model in models
    }
    if read_bound(ref)[0] != raw:
        raise ValueError("task030_policy_changed")
    return report
