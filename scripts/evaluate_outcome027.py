#!/usr/bin/env python3
"""Candidate 0.27 fixed-panel rescore of explicitly selected frozen episodes.

No provider or episode replay runs. Source population construction may initialize
native source definitions; its execution accounting is reported separately.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.mission_contracts import _digest as contract_digest, _read_locked  # noqa: E402
from evaluation.native_scorecard import _read  # noqa: E402
from evaluation.operational_completion import measure_completion  # noqa: E402
from evaluation.operational_contracts import compile_operational_suite  # noqa: E402
from evaluation.operational_opportunities import (  # noqa: E402
    build_opportunity_report,
    compile_opportunity_ledger,
)
from evaluation.operational_outcome027 import (  # noqa: E402
    aggregate_outcome027,
    score_case027,
)
from evaluation.operational_service import SERVICE_BACKENDS  # noqa: E402
from evaluation.operational_utility import capability_strata, evaluate_operational  # noqa: E402
from evaluation.operational_weights import validate_task_weights  # noqa: E402
from evaluation.source_voltage_population import (  # noqa: E402
    compile_source_voltage_population,
    measure_voltage_completion027,
)
from scripts.audit_offline_main_table026 import audit as audit026  # noqa: E402
from scripts.evaluate_completion026 import _authenticated_service  # noqa: E402

MODULES = (
    "scripts/evaluate_outcome027.py",
    "evaluation/operational_outcome027.py",
    "evaluation/source_voltage_population.py",
    "evaluation/operational_opportunities.py",
    "domains/pandapower_service.py",
    "evaluation/operational_completion.py",
    "evaluation/operational_contracts.py",
    "evaluation/operational_service.py",
    "evaluation/operational_utility.py",
    "evaluation/operational_weights.py",
    "evaluation/mission_contracts.py",
    "evaluation/leaderboard.py",
    "evaluation/native_scorecard.py",
    "evaluation/scorer.py",
    "evaluation/operational_settlement_energy.py",
    "evaluation/operational_settlement_inventory.py",
    "scripts/audit_offline_main_table026.py",
    "scripts/evaluate_completion026.py",
    "domains/microgrid/backends/pandapower_lv.py",
    "domains/power_grid/backends/cigre_distribution.py",
    "domains/power_grid/backends/opendss_ieee13.py",
    "domains/power_grid/backends/opendss_fresh_feeders.py",
)
VOLTAGE_BACKENDS = {
    "pandapower_lv",
    "cigre_distribution",
    "opendss_ieee13",
    "opendss_fresh_feeders",
}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_digest(value: object) -> str:
    return digest(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    )


def locked_json(ref: dict, root: Path) -> dict:
    return json.loads(_read_locked(root, ref, label="candidate_input"))


def _native_inputs(
    row: dict, contract: dict, root: Path, relocations: dict | None = None
) -> tuple[dict, list, list, dict]:
    binding = deepcopy(row["artifact_binding"])
    for descriptor in binding["artifacts"].values():
        path = Path((relocations or {}).get(descriptor["sha256"], descriptor["path"]))
        descriptor["path"] = str(path if path.is_absolute() else root / path)
    snapshot = _read(binding, "scoring_inputs_artifact")
    payload = snapshot["payload"]
    inputs = payload["inputs"]
    trace = _read(binding, "trajectory_artifact")
    ledger = _read(binding, "evidence_ledger_artifact")
    identity = payload.get("identity") or {}
    ids = {item["evidence_id"]: item for item in ledger}
    logged = (inputs.get("evidence_logger") or {}).get("items")
    if (
        any(identity.get(key) != row[key] for key in ("scenario_signature", "seed"))
        or inputs.get("scenario_signature") != contract["scenario_signature"]
        or len(ids) != len(ledger)
        or not isinstance(logged, list)
        or any(ids.get(item.get("evidence_id")) != item for item in logged)
        or len(trace) != binding.get("trace_ticks")
        or len(ledger) != binding.get("evidence_count")
        or not set(row["mission"].get("native_evidence_ids") or []) <= ids.keys()
    ):
        raise ValueError("candidate_selected_native_evidence_mismatch")
    return inputs, trace, ledger, binding


def _window(
    scenario: dict, contract: dict, inputs: dict, trace: list, mission: dict
) -> str | None:
    records = inputs.get("backend_tick_records")
    size = len(trace)
    if (
        not size
        or not isinstance(records, list)
        or len(records) != size
        or any(type(item.get("tick")) is not int for item in records + trace)
        or [item["tick"] for item in records] != list(range(size))
        or [item["tick"] for item in trace] != list(range(1, size + 1))
        or any(
            (item.get("observation") or {}).get("tick") != item["tick"]
            for item in trace
        )
        or size > scenario["horizon_ticks"]
    ):
        return None
    if size == scenario["horizon_ticks"]:
        return "full_source_horizon"
    done = records[-1].get("done") is True
    if not done and scenario["backend_kind"] == "dynasched_flexible_job_shop":
        # Historical generic logistics snapshots can omit native termination.
        # These logger items were authenticated against the complete ledger.
        terminal = [
            item
            for item in (inputs.get("evidence_logger") or {}).get("items", [])
            if item.get("source") == "engine"
            and item.get("kind") == "backend_tick"
            and type(item.get("tick")) is int
            and item["tick"] == size - 1
            and type((item.get("payload") or {}).get("tick")) is int
            and item["payload"]["tick"] == size - 1
        ]
        done = len(terminal) == 1 and terminal[0]["payload"].get("done") is True
    if not done:
        return None
    if (mission.get("safety") or {}).get("hard_failure") is True:
        return "verified_native_hard_terminal"
    final = trace[-1]["observation"]
    if (
        scenario["backend_kind"] == "dynasched_flexible_job_shop"
        and final.get("jobs_completed") == contract["service"].get("jobs_total")
        and final.get("jobs_cancelled") == 0
        and (mission.get("service") or {}).get("score") == 100
    ):
        return "source_jobs_all_completed_native_terminal"
    if (
        scenario["backend_kind"] == "sumo_ego"
        and records[-1].get("route_progress") == 1
    ):
        return "source_route_native_terminal"
    return None


def evaluate(manifest: dict, *, root: Path = ROOT) -> dict:
    root = Path(root)
    start_code = {name: digest((root / name).read_bytes()) for name in MODULES}
    if manifest.get("schema_version") != "operate_candidate_outcome027_inputs.v1":
        raise ValueError("unsupported_candidate_input_manifest")
    suite = manifest["suite"]
    locked_json(suite, root)
    suite_path = Path(suite["path"])
    suite_path = suite_path if suite_path.is_absolute() else root / suite_path
    compiled = compile_operational_suite(suite_path, root=root)
    contracts = {
        (item["scenario_signature"], item["seed"]): item
        for item in compiled["contracts"]
    }
    if len(contracts) != 141:
        raise ValueError("candidate_requires_frozen_lite141")
    lock = locked_json(manifest["constructor_asset_lock"], root)
    lock_hash = canonical_digest(lock)
    scenarios, populations = {}, {}
    for key, contract in contracts.items():
        scenario = yaml.safe_load(
            _read_locked(
                root,
                {
                    "path": contract["scenario_path"],
                    "sha256": contract["scenario_sha256"],
                },
                label="scenario",
            )
        )
        scenarios[key] = scenario
        contract["strata"] = capability_strata(scenario)
        contract.pop("contract_sha256")
        contract["contract_sha256"] = contract_digest(contract)
        if scenario["backend_kind"] in VOLTAGE_BACKENDS:
            populations[key] = compile_source_voltage_population(
                scenario,
                scenario_sha256=contract["scenario_sha256"],
                constructor_asset_lock=lock,
                constructor_lock_sha256=lock_hash,
            )
    aliases, selected, source_audits = {}, [], []
    weight_manifest = None
    for ref in manifest["source_reports"]:
        report = locked_json(ref, root)
        path = Path(ref["path"])
        path = path if path.is_absolute() else root / path
        audit_summary, _ = audit026(report_path=path, suite_path=suite_path, root=root)
        source025 = locked_json(report["source_report"], root)
        old_rows = {
            (row["model"], row["scenario_signature"], row["seed"]): row
            for row in source025["graded_episodes"]
        }
        if report["source_contracts"] != compiled:
            raise ValueError("candidate_source_contract_changed")
        if (
            weight_manifest is not None
            and weight_manifest != report["task_weight_manifest"]
        ):
            raise ValueError("candidate_frozen_weights_changed")
        weight_manifest = report["task_weight_manifest"]
        include = ref["include_models"]
        if (
            not include
            or len(include) != len(set(include))
            or not set(include) <= report["canonical_models"].keys()
        ):
            raise ValueError("candidate_invalid_selected_models")
        for model in include:
            alias = report["canonical_models"][model]
            if model in aliases or alias in aliases.values():
                raise ValueError("candidate_duplicate_model_alias")
            aliases[model] = alias
        for original in report["graded_episodes"]:
            if original["model"] in include:
                identity = (
                    original["model"],
                    original["scenario_signature"],
                    original["seed"],
                )
                selected.append((original, old_rows[identity]))
        source_audits.append(
            {
                "source_report": {"path": ref["path"], "sha256": ref["sha256"]},
                "source025_report": report["source_report"],
                "audit026": audit_summary,
            }
        )
    if not aliases or weight_manifest is None:
        raise ValueError("candidate_empty_model_selection")
    relocations = manifest.get("artifact_relocations", {})
    selected_hashes = {
        item.get("sha256")
        for original, _ in selected
        for item in (original.get("artifact_binding") or {})
        .get("artifacts", {})
        .values()
    }
    if not (
        isinstance(relocations, dict)
        and set(relocations) <= selected_hashes
        and all(isinstance(path, str) and path for path in relocations.values())
    ):
        raise ValueError("candidate_invalid_artifact_relocations")
    weights = validate_task_weights(weight_manifest, compiled)
    output = []
    for original, source025 in selected:
        row = deepcopy(original)
        key = row["scenario_signature"], row["seed"]
        contract, scenario = contracts[key], scenarios[key]
        old_mission = source025["mission"]
        mission = deepcopy(old_mission)
        completion = {
            "applicable": scenario["backend_kind"] != "citylearn",
            "score": None,
            "evidence_ids": [],
            "reason": "native_evidence_unqualified",
        }
        binding = row.get("artifact_binding") or {}
        verified, window = False, None
        opportunity = None
        if binding.get("verified") is True and binding.get("native_cost_bound") is True:
            inputs, trace, ledger, local_binding = _native_inputs(
                row, contract, root, relocations
            )
            raw_row = {
                **source025,
                "domain": contract["domain"],
                "backend_kind": contract["backend_kind"],
                "artifact_binding": local_binding,
            }
            fresh = evaluate_operational(
                raw_row, contract, scenario, scoring_mode="native_outcome"
            )
            # Reuse C only after reconstructing original component/settlement semantics.
            if old_mission.get("native_objective") is not None:
                for field in (
                    "native_objective",
                    "source_scale",
                    "signed_objective",
                    "components",
                    "safety",
                    "native_evidence_ids",
                    "score_interval",
                ):
                    if fresh.get(field) != old_mission.get(field):
                        raise ValueError(
                            f"candidate_original_native_reconstruction_mismatch:{field}:{row['model']}:{key[0]}"
                        )
            mission = fresh
            mission["native_quality_interval"] = fresh.get("score_interval")
            window = _window(scenario, contract, inputs, trace, mission)
            verified = window is not None and bool(fresh.get("native_evidence_ids"))
            if scenario["backend_kind"] in VOLTAGE_BACKENDS:
                completion = measure_voltage_completion027(
                    scenario,
                    source_contract=contract,
                    population=populations[key],
                    snapshot_inputs=inputs,
                    trace=trace,
                )
            else:
                service = (
                    _authenticated_service({"mission": mission}, scenario, contract)
                    if scenario["backend_kind"] in SERVICE_BACKENDS
                    else None
                )
                completion = measure_completion(
                    scenario,
                    source_contract=contract,
                    snapshot_inputs=inputs,
                    trace=trace,
                    existing_service=service,
                    root=root,
                )
            completion["evidence_ids"] = (
                list(
                    completion.get("evidence_ids")
                    or mission.get("native_evidence_ids")
                    or []
                )
                if completion.get("score") is not None
                else []
            )
            if not set(completion["evidence_ids"]) <= {
                item["evidence_id"] for item in ledger
            }:
                raise ValueError("candidate_completion_evidence_unbound")
            opportunity = build_opportunity_report(
                compile_opportunity_ledger(
                    None,
                    evidence_ledger=ledger,
                    identity={"scenario_signature": key[0], "seed": key[1]},
                    expected_contract_sha256=None,
                )
            )
        row.update(
            completion=completion,
            native_measurement027=mission,
            measurement_window027=window,
            operational_opportunities027=opportunity,
        )
        row["outcome027"] = score_case027(
            mission, completion, contract, evidence_verified=verified
        )
        row["legacy_score0261"] = original["mission"].get("score")
        output.append(row)
    scored = aggregate_outcome027(output, contracts, weights, models=list(aliases))
    if start_code != {name: digest((root / name).read_bytes()) for name in MODULES}:
        raise ValueError("candidate_scoring_code_changed_during_evaluation")
    scored.update(
        schema_version="operate_candidate_outcome027_report.v1",
        input_manifest_sha256=canonical_digest(manifest),
        input_manifest_digest_policy="canonical_json",
        source_contracts=compiled,
        task_weight_manifest=weight_manifest,
        canonical_models=aliases,
        source_audits=source_audits,
        voltage_population_contracts=list(populations.values()),
        graded_episodes=output,
        case_reason_counts=dict(Counter(row["outcome027"]["reason"] for row in output)),
        scoring_modules_sha256=start_code,
        scoring_code_identity_scope="listed_evaluator_measurement_and_source_constructor_modules_with_start_end_fence",
        retrospective_only=True,
        formal_run_certified=False,
        candidate_scoring_release_ready=bool(scored["leaderboard"]),
        comparison_policy="inherit_original_frozen_selection_and_compatibility_assumptions",
        native_objective_policy="reconstruct_original_025_components_settlement_and_source_scale_no_new_episode_execution",
        artifact_relocations=relocations,
        artifact_storage_policy="original_selected_descriptors_preserved_hash_identical_relocations_allowed",
        execution_policy={
            "provider_calls": 0,
            "environment_replays": 0,
            "source_population_initialization": "locked_native_definition_construction_may_initialize_source_solver",
            "model_outcomes_used_for_calibration": False,
        },
    )
    return scored


def score_audit_ledger(report: dict):
    weights = {
        (item["scenario_signature"], item["seed"]): item["score_weight"]
        for item in report["task_weight_manifest"]["cases"]
    }
    for row in report["graded_episodes"]:
        yield {
            "model": row["model"],
            "scenario_signature": row["scenario_signature"],
            "seed": row["seed"],
            "score_weight": weights[(row["scenario_signature"], row["seed"])],
            "outcome027": row["outcome027"],
            "completion": row["completion"],
            "measurement_window027": row["measurement_window027"],
            "artifact_binding": row["artifact_binding"],
            "legacy_score0261": row["legacy_score0261"],
        }


def render(report: dict) -> str:
    def number(value):
        return "N/A" if value is None else f"{value:.3f}"

    lines = [
        "# OPERATE candidate 0.27 fixed-panel outcomes",
        "",
        "Lite141 remains a model-informed development panel. Bounds condition on recorded point costs; they are not confidence intervals or point ranks.",
        "",
        "| Model | Valid points | Evidence qualified | Index | Rank | Missing-measurement bounds |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for item in report["models"]:
        low, high = item["missing_measurement_bounds"]
        lines.append(
            f"| {report['canonical_models'][item['model']]} | {item['n_scored']}/{item['n_expected']} | {'yes' if item['valid_for_comparison'] else 'no'} | {number(item['primary_score'])} | {item['primary_rank'] or '—'} | [{low:.3f}, {high:.3f}] |"
        )
    lines.extend(
        [
            "",
            "Recorded safety gates describe their measured scope. Opportunity rates without fixed source contracts remain N/A. Existing execution identities and compatibility assumptions are inherited.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_bytes())
    report = evaluate(manifest)
    ledger = "".join(
        json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n"
        for item in score_audit_ledger(report)
    ).encode()
    report["score_audit_ledger"] = {
        "path": "score_audit_ledger.jsonl",
        "sha256": digest(ledger),
        "rows": len(report["graded_episodes"]),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    (args.output_dir / "score_audit_ledger.jsonl").write_bytes(ledger)
    (args.output_dir / "report.md").write_text(render(report))
    print(render(report))


if __name__ == "__main__":
    main()
