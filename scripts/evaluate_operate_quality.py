#!/usr/bin/env python3
"""Offline OPERATE 0.23 evaluation and measurement-readiness audit.

Uses the existing authenticated 0.22 reader, not its native score as the new
headline. New contracts are hash-bound; missing capability evidence stays N/A.
No provider calls or source-artifact mutations are made.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.agency_measurements import score_agency_contract  # noqa: E402
from evaluation.native_objectives import extract_native_objective  # noqa: E402
from evaluation.operate_quality import VERSION, DIMENSIONS, aggregate_operate_rows  # noqa: E402
from evaluation.task_quality import evaluate_task_quality, backend_readiness  # noqa: E402
from scripts.evaluate_native_quality import evaluate as evaluate_legacy  # noqa: E402
from tools.lite_trajectory_report import _scope, select_latest_attempts  # noqa: E402


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load_measurement_contracts(descriptor, suite_sha256, specs):
    if descriptor is None:
        return {}, {
            "mode": "readiness_audit",
            "reason": "no_additional_measurement_contracts",
        }
    raw = Path(descriptor["path"]).read_bytes()
    if not descriptor.get("sha256") or _digest(raw) != descriptor["sha256"]:
        raise ValueError("measurement contract hash missing or mismatched")
    data = json.loads(raw)
    if (
        data.get("schema_version")
        not in {"operate_measurement_suite.v1", "operate_measurement_suite.v2"}
        or data.get("suite_sha256") != suite_sha256
    ):
        raise ValueError("measurement contract suite identity mismatch")
    contracts = {}
    if not isinstance(data.get("contracts"), list):
        raise ValueError("measurement contracts must be a list")
    multi_axis = data["schema_version"] == "operate_measurement_suite.v2"
    for item in data["contracts"]:
        case = (item.get("scenario_signature"), item.get("seed"))
        key = (*case, item.get("primary_dimension")) if multi_axis else case
        if (
            case not in specs
            or key in contracts
            or item.get("primary_dimension") not in DIMENSIONS
        ):
            raise ValueError(
                "duplicate/outside measurement contract or unknown dimension"
            )
        if item["primary_dimension"] in ("A", "L") and not isinstance(
            item.get("agency_contract"), dict
        ):
            raise ValueError("agency primary dimensions require a declared contract")
        contracts[key] = item
    return contracts, dict(
        path=descriptor["path"],
        sha256=_digest(raw),
        n_contracts=len(contracts),
        multi_axis=multi_axis,
    )


def normalize_action_trace(ledger):
    """Inventory every recorded tool attempt, including rejected state changes.

    Clock coordinates and call ownership come from tool/engine records. Model
    text cannot invent a call, remove failed interventions or assert effects.
    """
    calls, seen = [], set()
    for item in ledger:
        if item.get("kind") != "tool_call" or item.get("source") not in (
            "engine",
            "tool",
        ):
            continue
        p = item.get("payload") or {}
        call = p.get("call_id")
        if (
            not isinstance(call, str)
            or not call
            or call in seen
            or type(p.get("state_changing")) is not bool
            or type(item.get("tick")) is not int
        ):
            raise ValueError("incomplete_or_duplicate_tool_action_inventory")
        seen.add(call)
        consumes = p.get("consumes_evidence_ids") or []
        if not isinstance(consumes, list) or not all(
            isinstance(v, str) for v in consumes
        ):
            raise ValueError("invalid_consumed_evidence_inventory")
        effects = [
            other["evidence_id"]
            for other in ledger
            if other.get("source") == "engine"
            and other.get("kind") != "tool_call"
            and (other.get("payload") or {}).get("call_id") == call
        ]
        calls.append(
            dict(
                tick=item["tick"],
                call_id=call,
                state_changing=p["state_changing"],
                action_evidence_id=item["evidence_id"],
                consumes_evidence_ids=consumes,
                effect_evidence_ids=effects,
            )
        )
    return calls


def _bound_jsonl(binding, name):
    descriptor = binding["artifacts"][name]
    raw = Path(descriptor["path"]).read_bytes()
    if _digest(raw) != descriptor["sha256"]:
        raise ValueError("bound artifact changed during 0.23 evaluation")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def _input_rows(config):
    result = {}
    for label, paths in config["runs"].items():
        rows = []
        filters = config.get("row_filters", {}).get(label, {})
        for path in paths:
            raw = Path(path).read_bytes()
            if _digest(raw) != config.get("input_sha256", {}).get(str(path)):
                raise ValueError(
                    "0.23 requires frozen input hashes for every episode file"
                )
            rows.extend(
                {
                    **row,
                    "_offline_source_path": row.get("_offline_source_path", str(path)),
                }
                for line in raw.splitlines()
                if line.strip()
                for row in [json.loads(line)]
                if all(row.get(k) == v for k, v in filters.items())
            )
        if config.get("attempt_policy", "retain_all") == "latest_started":
            rows, _ = select_latest_attempts(rows)
        keys = [(r.get("scenario_signature"), r.get("seed")) for r in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(
                "0.23 requires one explicitly selected attempt per model/case"
            )
        result[label] = rows
    return result


def _execution_proven(row):
    tree = row.get("implementation_tree_sha256")
    return bool(tree) and row.get(
        "implementation_tree_sha256_start"
    ) == tree == row.get("implementation_tree_sha256_end")


def _execution_usable(row, comparison_policy):
    # Compatibility is explicit; original execution hashes remain in the report.
    # Provider, scenario and physical artifact validation belong to the reader.
    fields = (
        "implementation_tree_sha256",
        "model",
        "agent_profile_sha256",
        "agent_treatment_sha256",
        "interaction_mode",
        "run_semantics_fingerprint",
    )
    return (
        all(isinstance(row.get(k), str) and row[k] for k in fields)
        and all(isinstance(v, str) and v for v in _scope(row))
        and (
            _execution_proven(row)
            or comparison_policy == "latest_framework_user_assumed"
        )
    )


def validate_native_population(suite_path, declared):
    from scripts.compile_operate_measurements import compile_suite

    expected = {
        (e["scenario_signature"], e["seed"], e["primary_dimension"]): e
        for e in compile_suite(suite_path)["contracts"]
    }
    if declared != expected:
        raise ValueError(
            "native measurement population differs from complete source compilation"
        )


def _retro_measurement(
    primary, contract, row, spec, binding, reference, reference_inputs, baseline_cache
):
    import yaml
    from evaluation.legacy_agency_adapter import (
        compile_legacy_agency_contract,
        score_legacy_agency,
    )
    from evaluation.legacy_persistence_adapter import (
        compile_legacy_persistence_contract,
        score_legacy_persistence,
        load_verified_wait_baseline,
    )

    raw = Path(spec["path"]).read_bytes()
    if _digest(raw) != spec["yaml_sha256"]:
        raise ValueError("source scenario changed since suite lock")
    scenario = {
        **yaml.safe_load(raw),
        "scenario_signature": spec["scenario_signature"],
        "seed": spec["seed"],
    }
    compiler = (
        compile_legacy_agency_contract
        if primary == "A"
        else compile_legacy_persistence_contract
    )
    expected = compiler(scenario, scenario_sha256=_digest(raw))
    if expected != contract:
        raise ValueError(
            "retrospective contract differs from locked source compilation"
        )
    reference_key = (reference or {}).get("contract_sha256")
    if reference_key not in baseline_cache:
        baseline = load_verified_wait_baseline(reference, reference_inputs)
        baseline_inputs = None
        if baseline.get("verified"):
            descriptor = baseline["scoring_inputs_artifact"]
            raw = Path(descriptor["path"]).read_bytes()
            if _digest(raw) != descriptor["sha256"]:
                raise ValueError("verified wait snapshot changed during evaluation")
            baseline_inputs = json.loads(raw)["payload"]["inputs"]
        baseline_cache[reference_key] = (baseline, baseline_inputs)
    baseline, baseline_inputs = baseline_cache[reference_key]
    if primary == "L":
        return score_legacy_persistence(
            row, contract, artifact_binding=binding, baseline=baseline
        )
    artifacts = binding["artifacts"]
    snapshot = artifacts["scoring_inputs_artifact"]
    snapshot_raw = Path(snapshot["path"]).read_bytes()
    if _digest(snapshot_raw) != snapshot["sha256"]:
        raise ValueError("bound snapshot changed during evaluation")
    return score_legacy_agency(
        contract,
        snapshot_inputs=json.loads(snapshot_raw)["payload"]["inputs"],
        trace=_bound_jsonl(binding, "trajectory_artifact"),
        evidence_ledger=_bound_jsonl(binding, "evidence_ledger_artifact"),
        artifact_provenance=dict(
            trace_sha256=artifacts["trajectory_artifact"]["sha256"],
            snapshot_sha256=snapshot["sha256"],
            evidence_sha256=artifacts["evidence_ledger_artifact"]["sha256"],
        ),
        verified=True,
        baseline_inputs=baseline_inputs,
        baseline_provenance=baseline,
    )


def select_quality_reference(candidates, legacy_native, comparison_policy):
    if comparison_policy == "strict":
        selected = legacy_native.get("reference_contract_sha256")
        return next(
            (c for c in candidates if selected and c["contract_sha256"] == selected),
            None,
        )
    if comparison_policy != "latest_framework_user_assumed":
        raise ValueError("unknown comparison policy")
    if len({c["contract_sha256"] for c in candidates}) > 1:
        raise ValueError("ambiguous user-assumed quality reference")
    return candidates[0] if candidates else None


def recording_contract_is_bound(ledger, contract, start_tick):
    digest = _digest(
        json.dumps(
            contract, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    )
    return any(
        e.get("source") == "engine"
        and e.get("kind") == "agency_recording_contract"
        and type(e.get("tick")) is int
        and e["tick"] <= start_tick
        and (e.get("payload") or {}).get("version") == "agency_measurement_recording.v1"
        and (e.get("payload") or {}).get("measurement_contract_sha256") == digest
        for e in ledger
    )


def evaluate(config, *, bootstrap=0):
    suite_raw = Path(config["suite"]).read_bytes()
    suite = json.loads(suite_raw)["scenarios"]
    specs = {(s["scenario_signature"], s["seed"]): s for s in suite}
    declared, contract_provenance = load_measurement_contracts(
        config.get("measurement_contracts"), _digest(suite_raw), specs
    )
    if contract_provenance.get("multi_axis"):
        validate_native_population(config["suite"], declared)
    inputs = _input_rows(config)
    # This reader verifies reference executions and artifact hashes. Cross-tree
    # comparability follows the user's explicit existing manifest policy.
    legacy = evaluate_legacy({**config, "require_input_hashes": True})
    refs = {}
    for contract in legacy["reference_contracts"]:
        key = (contract["scenario_signature"], contract["seed"])
        refs.setdefault(key, []).append(contract)
    if contract_provenance.get("multi_axis"):
        assigned_suite = [{**specs[k[:2]], "primary_dimension": k[2]} for k in declared]
        if {k[:2] for k in declared if k[2] == "R"} != set(specs):
            raise ValueError("multi-axis measurement suite must retain every R task")
    else:
        assigned_suite = [
            {
                **s,
                "primary_dimension": declared.get(k, {}).get("primary_dimension", "R"),
            }
            for k, s in specs.items()
        ]
    baseline_cache = {}
    measurements = []
    detailed = []
    for label, rows in inputs.items():
        old_group = legacy["by_model"][label]
        old_rows = {
            (r["scenario_signature"], r["seed"]): r
            for r in old_group["graded_episodes"]
        }
        for original in rows:
            key = (original.get("scenario_signature"), original.get("seed"))
            if key not in specs:
                if config.get("select_suite_members") is True:
                    continue
                raise ValueError("0.23 input contains case outside suite")
            spec = specs[key]
            row = {**spec, **original}
            old = old_rows[key]
            binding = old["artifact_binding"]
            entries = (
                [v for k, v in declared.items() if k[:2] == key]
                if contract_provenance.get("multi_axis")
                else [declared.get(key, {})]
            )
            for entry in entries:
                primary = entry.get("primary_dimension", "R")
                native = extract_native_objective(row)
                proven = _execution_usable(row, legacy["comparison_policy"])
                safety = dict(
                    verified=binding["verified"]
                    and proven
                    and native.get("applicable") is True,
                    hard_failure=native.get("hard_failure"),
                    evidence_ids=native.get("evidence_ids", []),
                )
                if not binding["verified"] or not proven:
                    reason = (
                        "unbound_episode_artifacts"
                        if not binding["verified"]
                        else "execution_identity_unproven_or_changed"
                    )
                    result = dict(score=None, reason=reason, evidence_ids=[])
                elif primary == "R":
                    result = evaluate_task_quality(
                        row,
                        artifact_binding=binding,
                        reference_contract=select_quality_reference(
                            refs.get(key, []),
                            old["native_quality"],
                            legacy["comparison_policy"],
                        ),
                        acceptance_contract=entry.get("result_contract"),
                    )
                elif entry.get("adapter") == "legacy_native":
                    result = _retro_measurement(
                        primary,
                        entry["agency_contract"],
                        row,
                        spec,
                        binding,
                        select_quality_reference(
                            refs.get(key, []),
                            old["native_quality"],
                            legacy["comparison_policy"],
                        ),
                        legacy["reference_inputs"],
                        baseline_cache,
                    )
                else:
                    contract = entry["agency_contract"]
                    # Suite-level identity, not a model-supplied assertion.
                    if any(
                        contract.get(field) != row.get(field)
                        for field in ("scenario_signature", "seed", "backend_kind")
                    ):
                        raise ValueError("agency contract episode identity mismatch")
                    ledger = _bound_jsonl(binding, "evidence_ledger_artifact")
                    trajectory = _bound_jsonl(binding, "trajectory_artifact")
                    ticks = [t.get("observation", {}).get("tick") for t in trajectory]
                    if not ticks or any(type(t) is not int for t in ticks):
                        result = dict(
                            score=None,
                            reason="native_trace_clock_unavailable",
                            evidence_ids=[],
                        )
                        marker = False
                    else:
                        marker = recording_contract_is_bound(
                            ledger, contract, min(ticks)
                        )
                    # Old successful chains cannot prove the complete opportunity ledger.
                    if not marker:
                        result = dict(
                            score=None,
                            reason="complete_agency_recording_contract_missing",
                            evidence_ids=[],
                        )
                    else:
                        try:
                            actions = normalize_action_trace(ledger)
                            agency = score_agency_contract(
                                contract,
                                evidence_ledger=ledger,
                                trace=actions,
                                contract_verified=True,
                                trace_complete=True,
                                coverage_start_tick=min(ticks),
                                coverage_end_tick=max(ticks),
                                verified_evidence_ids={
                                    e["evidence_id"] for e in ledger
                                },
                            )
                            result = agency["dimensions"][primary]
                        except ValueError as exc:
                            result = dict(score=None, reason=str(exc), evidence_ids=[])
                scored = dict(
                    model=label,
                    primary_dimension=primary,
                    scenario_signature=key[0],
                    seed=key[1],
                    measurement=result,
                    safety=safety,
                )
                measurements.append(scored)
                detailed.append(
                    {
                        **scored,
                        "primary_dimension": primary,
                        "artifact_binding": binding,
                        "execution_provenance": old.get("execution_provenance"),
                        "legacy_022_native": old["native_quality"],
                        "legacy_021_total": (row.get("score") or {}).get("total_score"),
                        "legacy_021_primary": (row.get("ranking") or {}).get(
                            "primary_score"
                        ),
                    }
                )
    aggregation_mode = (
        "domain_balanced" if contract_provenance.get("multi_axis") else "axis_first"
    )
    report = aggregate_operate_rows(
        measurements,
        assigned_suite,
        list(inputs),
        bootstrap=bootstrap,
        aggregation_mode=aggregation_mode,
    )
    for label, group in legacy["by_model"].items():
        if group.get("identity_blocker"):
            report["models"][label].update(
                index=None, complete=False, identity_blocker=group["identity_blocker"]
            )
            report["models"][label].pop("ci", None)
            for view in report["weight_sensitivity"]:
                view["scores"][label] = None
                view["ranking"] = []
    if any(not m["complete"] for m in report["models"].values()):
        report.update(complete=False, ranking=[], pairwise=[])
    complete_models = [m for m, group in report["models"].items() if group["complete"]]
    report["complete_model_ranking"] = sorted(
        complete_models, key=lambda m: (-report["models"][m]["index"], m)
    )
    report["ranking_excluded_models"] = [m for m in inputs if m not in complete_models]
    report["complete_model_ranking_scope"] = (
        "only_models_complete_on_the_same_declared_population"
    )
    if bootstrap and complete_models and not report["complete"]:
        cohort = aggregate_operate_rows(
            [r for r in measurements if r["model"] in complete_models],
            assigned_suite,
            complete_models,
            bootstrap=bootstrap,
            aggregation_mode=aggregation_mode,
        )
        for model in complete_models:
            if "ci" in cohort["models"][model]:
                report["models"][model]["ci"] = cohort["models"][model]["ci"]
        report["complete_model_pairwise"] = cohort["pairwise"]
        report["complete_model_inference_reason"] = cohort["inference_reason"]
        report["complete_model_weight_sensitivity"] = cohort["weight_sensitivity"]
    report.update(
        comparison_policy=legacy["comparison_policy"],
        compatibility_user_assumed=legacy["compatibility_user_assumed"],
        measurement_contracts=contract_provenance,
        suite_sha256=_digest(suite_raw),
        episode_inputs=legacy["episode_inputs"],
        reference_inputs=legacy["reference_inputs"],
        graded_episodes=detailed,
        backend_readiness=backend_readiness(),
        protocol_readiness="complete"
        if report["complete"]
        else "measurement_or_execution_gaps",
        reason_counts=dict(Counter(r["measurement"].get("reason") for r in detailed)),
        semantics="evidence_linked_native_outcome_supervision_and_staged_fulfillment",
        index_scope="domain_balanced_native_index_with_limited_agency_scope"
        if contract_provenance.get("multi_axis")
        else "declared_contract_scope",
        dimension_support={
            d: dict(
                n_cases=sum(s["primary_dimension"] == d for s in assigned_suite),
                domains=sorted(
                    {s["domain"] for s in assigned_suite if s["primary_dimension"] == d}
                ),
                backends=sorted(
                    {
                        s["backend_kind"]
                        for s in assigned_suite
                        if s["primary_dimension"] == d
                    }
                ),
                n_physical_sources=len(
                    {
                        s.get("physical_source_key")
                        for s in assigned_suite
                        if s["primary_dimension"] == d
                    }
                ),
            )
            for d in DIMENSIONS
        },
    )
    return report


def render_summary(report):
    lines = [
        "# OPERATE 0.23",
        "",
        f"Comparison policy: {report['comparison_policy']}. Original identities are retained.",
        "",
        "N/A means not a complete three-dimension Index; available-component weights are never redistributed.",
        "",
        "Retrospective A/L populations cover declared microgrid response/stage contracts. The composite is a limited-scope experimental index, not a general memory or cross-domain autonomy score.",
        "",
        "| Model | Overall | R (measured/expected) | A (measured/expected) | L (measured/expected) | All measured/expected | Hard failures / safety measured |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    if report.get("aggregation_mode") == "domain_balanced":
        lines[4:4] = [
            "Overall averages domain scores equally. Each domain uses its source-declared axis weights; the R/A/L columns are separate dimension profiles.",
            "",
            "Effective weight allocation: " + ", ".join(f"{d}={w:.2%}" for d, w in report["effective_dimension_weights"].items()) + ". Full domain-axis weights are recorded in report.json.",
            "",
        ]
    for model, item in report["models"].items():
        cells = []
        for axis in DIMENSIONS:
            d = item["dimensions"][axis]
            score = "N/A" if d["score"] is None else f"{d['score']:.4f}"
            cells.append(f"{score} ({d['n_measured']}/{d['n_expected']})")
        total = "N/A" if item["index"] is None else f"{item['index']:.4f}"
        lines.append(
            f"| {model} | {total} | {' | '.join(cells)} | {item['n_measured']}/{item['n_expected']} | {item['safety']['hard_failures']}/{item['safety']['n_measured']} |"
        )
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=0)
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error("bootstrap must be nonnegative")
    from core.implementation_identity import implementation_identity

    before = implementation_identity(ROOT)
    entry_hash = _digest(Path(__file__).read_bytes())
    raw = args.manifest.read_bytes()
    report = evaluate(json.loads(raw), bootstrap=args.bootstrap)
    after = implementation_identity(ROOT)
    if before["evaluation_runtime_sha256"] != after[
        "evaluation_runtime_sha256"
    ] or entry_hash != _digest(Path(__file__).read_bytes()):
        raise RuntimeError("0.23 implementation changed while evaluating")
    if ".hl" not in args.output.resolve().parts:
        parser.error("output must be under .hl")
    args.output.mkdir(parents=True, exist_ok=False)
    report.update(
        input_manifest_sha256=_digest(raw),
        scoring_implementation=before,
        scoring_entry_sha256=entry_hash,
    )
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    (args.output / "report.md").write_text("\n".join(render_summary(report)) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "evaluation_version": VERSION,
                "complete_models": sum(
                    m["complete"] for m in report["models"].values()
                ),
                "reason_counts": report["reason_counts"],
            }
        )
    )


if __name__ == "__main__":
    main()
