"""Model-independent reference anchors for the 0.31 anchored outcome score.

CPU reference controllers are measured with the native cost, fulfillment and
hard-gate rules that the 0.30 reader applies to model trajectories. Anchors are
derived from source contracts and reference executions only; no model outcome
is read. Missing, nondeterministic or unauthenticated reference evidence fails
closed instead of producing an anchor.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import yaml

from evaluation.mission_contracts import _read_locked
from evaluation.native_objectives import extract_native_objective
from evaluation.operational_completion import measure_completion
from evaluation.operational_outcome028 import normalize_dss_components
from evaluation.operational_service import SERVICE_BACKENDS, measure_service
from evaluation.source_voltage_population import measure_voltage_completion027
from evaluation.task_outcome030 import measure_recovery_attainment
from evaluation.trajectory_outcome027 import canonical_digest

VERSION = "reference_anchors031.v1"
POLICIES = ("wait_only", "greedy_heuristic", "oracle_offline", "random")
LOWER_POLICY = "wait_only"
UPPER_POLICIES = ("oracle_offline", "greedy_heuristic")
EXTRA_UPPER_PREFIX = "citylearn_tariff_reference@"
DSS_BACKENDS = {"opendss_ieee13", "opendss_fresh_feeders"}
LV_BACKENDS = {"pandapower_lv", "cigre_distribution"}
VOLTAGE_BACKENDS = DSS_BACKENDS | LV_BACKENDS


def _finite(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_case_identity(identity: dict, contract: dict) -> None:
    if any(
        identity.get(field) != contract[field]
        for field in ("scenario_signature", "seed")
    ):
        raise ValueError("reference_episode_case_identity_mismatch")


def _validate_policy_identity(episode: dict, policy: str) -> None:
    if policy.startswith(EXTRA_UPPER_PREFIX):
        from baselines.citylearn_tariff_reference import GRID, config_id

        config = episode.get("agent_config")
        if (
            episode.get("agent_name") != EXTRA_UPPER_PREFIX.rstrip("@")
            or config not in GRID
            or policy != f"{EXTRA_UPPER_PREFIX}{config_id(config)}"
        ):
            raise ValueError("reference_episode_policy_identity_mismatch")
    elif policy not in POLICIES or episode.get("agent_name") != policy:
        raise ValueError("reference_episode_policy_identity_mismatch")


def settled_native_cost(
    backend: str, components: dict, records: list, service: dict | None, trace: list
) -> float:
    """The settled native objective, matching the 0.27 reader's definition."""
    if (
        not isinstance(components, dict)
        or not components
        or any(not _finite(v) for v in components.values())
    ):
        raise ValueError("reference_cost_components_invalid")
    cost = math.fsum(components.values())
    if backend == "pymgrid_economic_dispatch":
        if any(
            not _finite(r.get(k))
            for r in records
            for k in ("balance_error_mw", "shed_penalty")
        ):
            raise ValueError("reference_state_loss_records_invalid")
        cost = (
            math.fsum(abs(r["balance_error_mw"]) * 200 for r in records)
            + components["shed_penalty"]
        )
    elif backend == "dynasched_flexible_job_shop":
        makespan = (
            (trace[-1].get("observation") or {}).get("makespan") if trace else None
        )
        if (
            not service
            or service.get("score") is None
            or not _finite(makespan)
            or makespan < 0
        ):
            raise ValueError("reference_schedule_settlement_unavailable")
        cost = makespan + 1000.0 * (service["denominator"] - service["numerator"])
    return float(cost)


def voltage_loss030(
    backend: str,
    components: dict,
    contract: dict,
    *,
    population: dict | None,
    dss: dict | None,
) -> float | None:
    """0.30 dimensionless voltage loss, retained only as a sensitivity input."""
    if backend in LV_BACKENDS:
        nodes, horizon = len(population["required_bus_ids"]), contract["horizon_ticks"]
        other = math.fsum(
            v for k, v in components.items() if k != "voltage_violation_cost"
        )
        return (
            components["voltage_violation_cost"] / (1200.0 * horizon * nodes)
            + other / contract["scales"]["native_cost"]["value"]
        )
    if backend in DSS_BACKENDS:
        return normalize_dss_components(
            components, math.fsum(components.values()), dss
        )["dimensionless_loss"]
    return None


def measure_reference(
    scenario: dict,
    contract: dict,
    *,
    episode: dict,
    inputs: dict,
    trace: list[dict],
    population: dict | None = None,
    dss: dict | None = None,
) -> dict:
    """Measure one complete reference execution with the reader's native rules."""
    _validate_case_identity(episode, contract)
    backend = contract["backend_kind"]
    records = inputs.get("backend_tick_records")
    if not isinstance(records, list) or not isinstance(trace, list) or not trace:
        raise ValueError("reference_native_window_missing")
    components = inputs.get("cost_components")
    ground = (episode.get("ground_truth_summary") or {}).get("cost_components")
    if components != ground:
        raise ValueError("reference_cost_components_snapshot_mismatch")
    native = extract_native_objective(
        {
            **episode,
            "backend_kind": backend,
            "domain": contract["domain"],
            "status": "ok",
        }
    )
    if (
        native.get("reason") != "evidenced_native_objective"
        or type(native.get("hard_failure")) is not bool
    ):
        raise ValueError(
            f"reference_native_objective_unavailable:{native.get('reason')}"
        )
    hard = native["hard_failure"]
    if len(trace) != contract["horizon_ticks"] and not hard:
        complete_early = (
            backend == "sumo_ego"
            and records[-1].get("route_progress") == 1.0
            and records[-1].get("done") is True
        )
        if not complete_early and backend != "dynasched_flexible_job_shop":
            raise ValueError("reference_incomplete_observation_window")
    service = None
    if backend in SERVICE_BACKENDS:
        service = measure_service(
            scenario, snapshot_inputs=inputs, trace=trace, source_contract=contract
        )
    cost = settled_native_cost(backend, components, records, service, trace)
    if backend not in {
        "pymgrid_economic_dispatch",
        "dynasched_flexible_job_shop",
    } and not math.isclose(cost, native["actual_cost"], rel_tol=1e-10, abs_tol=1e-7):
        raise ValueError("reference_native_cost_mismatch")
    if backend == "citylearn":
        fulfillment = None
    elif backend == "sumo_ego":
        required = scenario["backend_config"]["task_requirements"][
            "required_stable_dwell_ticks"
        ]
        fulfillment = measure_recovery_attainment(records, required)["score"]
    elif backend in VOLTAGE_BACKENDS:
        fulfillment = measure_voltage_completion027(
            scenario,
            source_contract=contract,
            population=population,
            snapshot_inputs=inputs,
            trace=trace,
        ).get("score")
    else:
        if service is not None and not service.get("evidence_ids"):
            raise ValueError("reference_service_evidence_missing")
        fulfillment = measure_completion(
            scenario,
            source_contract=contract,
            snapshot_inputs=inputs,
            trace=trace,
            existing_service=service,
        ).get("score")
    if backend != "citylearn" and not _finite(fulfillment) and not hard:
        raise ValueError("reference_fulfillment_unavailable")
    return {
        "C": cost,
        "F": float(fulfillment) if _finite(fulfillment) else None,
        "hard_failure": hard,
        "loss030": voltage_loss030(
            backend, components, contract, population=population, dss=dss
        )
        if backend in VOLTAGE_BACKENDS
        else None,
        "evidence_ids": list(native["evidence_ids"]),
    }


def declared_event_contract(ledger_lines: list[str]) -> list[dict]:
    """Declared perturbations the backend actually published as typed events.

    Exogenous events fire independently of the agent, so the do-nothing ledger
    fixes which declared events a response can be credited against.
    """
    events = {}
    for line in ledger_lines:
        if not line.strip():
            continue
        record = json.loads(line)
        payload = record.get("payload")
        if (
            record.get("kind") != "realized_event"
            or not isinstance(payload, dict)
            or not (
                payload.get("origin") == "declared_perturbation"
                or payload.get("declared_perturbation") is True
            )
            or not isinstance(payload.get("event_id"), str)
        ):
            continue
        deadline = next(
            (
                payload[k]
                for k in (
                    "response_window_end_tick",
                    "response_deadline_tick",
                    "mandatory_response_tick",
                )
                if type(payload.get(k)) is int
            ),
            None,
        )
        events.setdefault(
            payload["event_id"],
            dict(
                event_id=payload["event_id"],
                tick=record["tick"],
                hidden=bool(payload.get("hidden")),
                deadline_tick=deadline,
            ),
        )
    return [events[k] for k in sorted(events)]


def case_anchor(
    case: dict, references: dict, *, h_min: float, service_delta: float
) -> dict:
    """Choose anchors and axis informativeness from reference measurements only."""
    lower = references[LOWER_POLICY]
    applicable = case["backend_kind"] != "citylearn"
    candidates = [
        (references[p]["C"], p)
        for p in sorted(references)
        if (p in UPPER_POLICIES or p.startswith(EXTRA_UPPER_PREFIX))
        and not references[p]["hard_failure"]
        and (not applicable or references[p]["F"] >= lower["F"] - service_delta)
    ]
    c_ref, ref_policy = min(candidates) if candidates else (lower["C"], LOWER_POLICY)
    scale = max(abs(lower["C"]), abs(c_ref), 1e-9)
    cost_headroom = lower["C"] - c_ref
    cost_informative = cost_headroom >= h_min * scale
    service_informative = bool(
        applicable and lower["F"] is not None and lower["F"] <= 100.0 - service_delta
    )
    eligible = cost_informative or service_informative
    anchor = {
        **case,
        "C_wait": lower["C"],
        "F_wait": lower["F"],
        "hard_wait": lower["hard_failure"],
        "C_ref": c_ref,
        "ref_policy": ref_policy,
        "F_ref": references[ref_policy]["F"],
        "loss030_wait": lower["loss030"],
        "loss030_ref": references[ref_policy]["loss030"],
        "cost_scale": scale,
        "cost_headroom": cost_headroom,
        "relative_cost_headroom": cost_headroom / scale,
        "cost_axis_informative": cost_informative,
        "service_applicable": applicable,
        "service_axis_informative": service_informative,
        "headline_eligible": eligible,
        "ineligibility": None
        if eligible
        else {
            "code": "no_reference_headroom",
            "detail": "best_model_independent_reference_does_not_improve_cost_by_h_min_and_do_nothing_meets_full_service",
        },
        "references": references,
    }
    return anchor


def _load_episode_record(
    case_dir: Path,
    policy: str,
    repetition: int,
    records: list[dict],
    root: Path,
    *,
    contract: dict,
):
    matches = [
        r for r in records if r["policy"] == policy and r["repetition"] == repetition
    ]
    if len(matches) != 1 or matches[0].get("status") != "ok":
        raise ValueError(
            f"reference_execution_missing:{case_dir.name}:{policy}:{repetition}"
        )
    record = matches[0]
    episode_path = root / record["episode_path"]
    if _sha256(episode_path) != record["episode_sha256"]:
        raise ValueError("reference_episode_hash_mismatch")
    artifacts = {}
    for artifact in record["artifacts"]:
        path = root / artifact["path"]
        if _sha256(path) != artifact["sha256"]:
            raise ValueError("reference_artifact_hash_mismatch")
        artifacts[path.name] = artifact

    def one(suffix: str) -> Path:
        found = [
            root / a["path"] for name, a in artifacts.items() if name.endswith(suffix)
        ]
        if len(found) != 1:
            raise ValueError(f"reference_artifact_ambiguous:{suffix}")
        return found[0]

    episode = json.loads(episode_path.read_bytes())
    _validate_case_identity(episode, contract)
    _validate_policy_identity(episode, policy)
    for field in ("scenario_signature", "seed", "agent_name", "agent_config"):
        if field in record and record[field] != episode.get(field):
            raise ValueError("reference_record_identity_mismatch")
    payload = json.loads(one(".scoring_inputs.json").read_bytes())["payload"]
    if "identity" in payload:
        identity = payload["identity"]
        if not isinstance(identity, dict) or any(
            field not in identity or identity[field] != episode.get(field)
            for field in ("scenario_signature", "seed", "agent_name", "agent_config")
        ):
            raise ValueError("reference_snapshot_identity_mismatch")
    inputs = payload["inputs"]
    trace = [
        json.loads(line)
        for line in one(".trajectory.jsonl").read_text().splitlines()
        if line.strip()
    ]
    binding = {
        "episode": {"path": record["episode_path"], "sha256": record["episode_sha256"]},
        "scoring_inputs": {
            "path": str(one(".scoring_inputs.json").relative_to(root)),
            "sha256": _sha256(one(".scoring_inputs.json")),
        },
        "trajectory": {
            "path": str(one(".trajectory.jsonl").relative_to(root)),
            "sha256": _sha256(one(".trajectory.jsonl")),
        },
    }
    events = declared_event_contract(one(".evidence.jsonl").read_text().splitlines())
    return episode, inputs, trace, binding, events


def _measure_policy(
    result_path, result, policy, *, scenario, contract, population, dss, root
):
    """Both repetitions of one reference policy; they must agree exactly."""
    _validate_case_identity(result, contract)
    if result["determinism"].get(policy, {}).get("deterministic") is not True:
        raise ValueError(
            f"reference_not_deterministic:{result['scenario_signature']}:{policy}"
        )
    measured, events = [], []
    for repetition in (0, 1):
        episode, inputs, trace, binding, declared = _load_episode_record(
            result_path.parent,
            policy,
            repetition,
            result["episodes"],
            root,
            contract=contract,
        )
        events.append(declared)
        value = measure_reference(
            scenario,
            contract,
            episode=episode,
            inputs=inputs,
            trace=trace,
            population=population,
            dss=dss,
        )
        measured.append((value, binding))
    (first, b0), (second, b1) = measured
    if any(first[k] != second[k] for k in ("C", "F", "hard_failure", "loss030")):
        raise ValueError(
            f"reference_repetitions_disagree:{result['scenario_signature']}:{policy}"
        )
    reference = {k: first[k] for k in ("C", "F", "hard_failure", "loss030")} | {
        "evidence_ids": first["evidence_ids"],
        "executions": [b0, b1],
    }
    return reference, events


def build_reference_anchors(
    reference_dirs: list[Path],
    *,
    policy027: dict,
    policy028: dict,
    root: Path,
    h_min: float,
    service_delta: float,
    extra_reference_dirs: list[Path] = (),
) -> dict:
    """Compile a hash-bound anchor manifest covering every frozen panel case.

    ``extra_reference_dirs`` hold additional fixed reference configurations
    (``EXTRA_UPPER_PREFIX`` labels) for a subset of cases; they join the
    upper-anchor candidates of those cases only.
    """
    extras = {}
    for directory in extra_reference_dirs:
        for result_path in sorted(Path(directory).glob("*/result.json")):
            result = json.loads(result_path.read_bytes())
            key = (result["scenario_signature"], result["seed"])
            labels = sorted(result["determinism"])
            if (
                key in extras
                or not labels
                or any(not label.startswith(EXTRA_UPPER_PREFIX) for label in labels)
            ):
                raise ValueError("extra_reference_case_duplicate_or_foreign_policy")
            extras[key] = (result_path, result, labels)
    contracts = {
        (c["scenario_signature"], c["seed"]): c
        for c in policy027["source_contracts"]["contracts"]
    }
    populations = {
        (c["scenario_signature"], c["seed"]): c
        for c in policy027["voltage_population_contracts"]
    }
    dss = {
        (c["scenario_signature"], c["seed"]): c
        for c in policy028["normalization_contracts"]
    }
    cases = {}
    for directory in reference_dirs:
        for result_path in sorted(Path(directory).glob("*/result.json")):
            result = json.loads(result_path.read_bytes())
            key = (result["scenario_signature"], result["seed"])
            if key not in contracts or key in cases:
                raise ValueError("reference_case_foreign_or_duplicate")
            contract = contracts[key]
            scenario = yaml.safe_load(_read_locked(
                root,
                {"path": contract["scenario_path"], "sha256": contract["scenario_sha256"]},
                label="reference_scenario",
            ))
            measure = dict(
                scenario=scenario,
                contract=contract,
                population=populations.get(key),
                dss=dss.get(key),
                root=root,
            )
            references, event_sets = {}, []
            runs = [(result_path, result, policy) for policy in POLICIES]
            if key in extras:
                extra_path, extra_result, labels = extras.pop(key)
                runs += [(extra_path, extra_result, label) for label in labels]
            for path, run, policy in runs:
                references[policy], events = _measure_policy(
                    path, run, policy, **measure
                )
                event_sets += events
            case = {
                "scenario_signature": key[0],
                "seed": key[1],
                "domain": contract["domain"],
                "task_family": contract.get("task_family"),
                "backend_kind": contract["backend_kind"],
                "source_contract_sha256": contract["contract_sha256"],
            }
            if any(events != event_sets[0] for events in event_sets):
                raise ValueError(f"reference_declared_events_agent_dependent:{key[0]}")
            cases[key] = case_anchor(
                case, references, h_min=h_min, service_delta=service_delta
            )
            cases[key]["declared_events"] = event_sets[0]
    if set(cases) != set(contracts) or extras:
        raise ValueError("reference_anchor_coverage_incomplete")
    rows = [cases[k] for k in sorted(cases)]
    manifest = {
        "schema_version": VERSION,
        "suite_sha256": policy027["source_contracts"]["suite_sha256"],
        "lower_policy": LOWER_POLICY,
        "upper_policies": list(UPPER_POLICIES)
        + ([f"{EXTRA_UPPER_PREFIX}*"] if extra_reference_dirs else []),
        "h_min": h_min,
        "service_delta": service_delta,
        "model_outcomes_used": False,
        "cases": rows,
        "counts": {
            "cases": len(rows),
            "headline_eligible": sum(r["headline_eligible"] for r in rows),
            "cost_axis_informative": sum(r["cost_axis_informative"] for r in rows),
            "service_axis_informative": sum(
                r["service_axis_informative"] for r in rows
            ),
            "cases_with_declared_events": sum(bool(r["declared_events"]) for r in rows),
        },
    }
    manifest["anchors_sha256"] = canonical_digest(manifest)
    return manifest


def validate_reference_anchors(
    manifest: dict, source_contracts: dict
) -> dict[tuple[str, int], dict]:
    """Integrity and coverage check against authenticated source contracts."""
    if manifest.get("schema_version") != VERSION:
        raise ValueError("unsupported_reference_anchor_schema")
    if manifest.get("anchors_sha256") != canonical_digest(
        {k: v for k, v in manifest.items() if k != "anchors_sha256"}
    ):
        raise ValueError("reference_anchor_hash_mismatch")
    if (
        manifest.get("suite_sha256") != source_contracts.get("suite_sha256")
        or manifest.get("model_outcomes_used") is not False
    ):
        raise ValueError("reference_anchor_suite_mismatch")
    contracts = {
        (c["scenario_signature"], c["seed"]): c for c in source_contracts["contracts"]
    }
    anchors = {(c["scenario_signature"], c["seed"]): c for c in manifest["cases"]}
    if anchors.keys() != contracts.keys() or len(anchors) != len(manifest["cases"]):
        raise ValueError("reference_anchor_coverage_mismatch")
    for key, anchor in anchors.items():
        if anchor["source_contract_sha256"] != contracts[key]["contract_sha256"]:
            raise ValueError("reference_anchor_source_contract_mismatch")
        for field in ("C_wait", "C_ref", "cost_scale"):
            if not _finite(anchor[field]):
                raise ValueError("reference_anchor_value_invalid")
    return anchors
