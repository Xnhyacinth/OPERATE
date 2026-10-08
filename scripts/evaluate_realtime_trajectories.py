#!/usr/bin/env python3
"""Reauthenticate original/recovered realtime evidence and report fixed panels.

This offline reader never calls a provider or constructs an environment. Run
and treatment identities remain distinct; its descriptive case-availability
union is not a merged formal run or a logical-primary score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.realtime_clock import classify_realtime_clock  # noqa: E402
from evaluation.realtime_diagnostics import evaluate_realtime_diagnostics  # noqa: E402
from evaluation.realtime_persistence_recovery import (  # noqa: E402
    recover_realtime_persisted_artifact,
    recover_realtime_wire_persisted_artifact,
)
from evaluation.supplementary_analysis import analyze_e3_group  # noqa: E402
from runner.realtime_episode import (  # noqa: E402
    _apply_realtime_artifact_validation,
    _build_evidence_closure,
    _canonical_json,
    _provider_turn_audit_violations,
)
from scripts.batch_realtime_llm_eval import (  # noqa: E402
    aggregate_realtime_scorecard,
    realtime_artifact_eligibility,
)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _wire_hash_reasons(audit: list[dict]) -> set[str]:
    reasons = set()
    for turn in audit:
        for records, payload_key, reason in (
            ("provider_requests", "envelope", "PROVIDER_REQUEST_PAYLOAD_HASH_MISMATCH"),
            (
                "provider_responses",
                "response",
                "PROVIDER_RESPONSE_PAYLOAD_HASH_MISMATCH",
            ),
        ):
            for record in turn.get(records) or []:
                payload = record.get(payload_key)
                digest = hashlib.sha256(
                    json.dumps(
                        payload,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ).encode()
                ).hexdigest()
                if not isinstance(payload, dict) or record.get("sha256") != digest:
                    reasons.add(reason)
    return reasons


def _path(raw: str, root: Path) -> Path:
    path = Path(raw)
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("bound_path_outside_root")
    return resolved


def _read_bound(ref: dict, root: Path) -> bytes:
    raw = _path(ref["path"], root).read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise ValueError("bound_artifact_hash_mismatch")
    if "byte_count" in ref and (
        type(ref["byte_count"]) is not int or ref["byte_count"] != len(raw)
    ):
        raise ValueError("bound_artifact_byte_count_mismatch")
    return raw


def _selected_sources(
    ref: dict, root: Path, expected: int
) -> dict[tuple[str, int], dict]:
    suite = json.loads(_read_bound(ref, root))
    selected = {}
    for row in suite["scenarios"]:
        body = yaml.safe_load(
            _read_bound({"path": row["path"], "sha256": row["yaml_sha256"]}, root)
        )
        if body["horizon_ticks"] != row["horizon_ticks"]:
            raise ValueError("source_horizon_mismatch")
        clock = classify_realtime_clock(body, horizon_ticks=row["horizon_ticks"])
        if not clock["scorecard_eligible"]:
            continue
        key = row["scenario_signature"], row["seed"]
        if key in selected or type(row["seed"]) is not int:
            raise ValueError("source_case_identity_invalid")
        selected[key] = {**row, "tick_interval_s": clock["wall_tick_interval_s"]}
    if len(selected) != expected:
        raise ValueError("fixed_source_denominator_mismatch")
    return selected


def _bind_config(
    cohort: dict, manifest: dict, release: dict, config: dict, sources: dict
) -> None:
    batch = config["batch_treatment_identity"]
    binding = batch["formal_runtime_binding"]
    selection = batch["selection_contract"]
    role = cohort["role"]
    expected_suite = (
        release["formal_realtime_batch_contract"]["suite_manifest_sha256"]
        if role == "Core37"
        else manifest["source_suites"]["E3"]["sha256"]
    )
    if (
        _digest(batch) != config["batch_treatment_sha256"]
        or config["model"] != cohort["model"]
        or batch["model_shard"]["model"] != cohort["model"]
        or batch["formal_manifest_sha256"] != manifest["release_manifest"]["sha256"]
        or binding["manifest_sha256"] != manifest["release_manifest"]["sha256"]
        or binding["formal_core_suite_sha256"]
        != manifest["source_suites"]["Core37"]["sha256"]
        or binding["formal_source_suite_sha256"] != manifest["source_suite"]["sha256"]
        or batch["formal_release_id"] != release["release_id"]
        or batch["suite_sha256"] != expected_suite
        or selection["suite_sha256"] != expected_suite
        or selection["kind"] != ("core" if role == "Core37" else "lite")
        or selection["n_scorecard_rows"] != len(sources)
        or batch["clock"]["tick_interval_policy"] != "native_dt_v1"
        or batch["sampling"]["pass_k"] != 1
    ):
        raise ValueError("original_run_source_or_treatment_binding_mismatch")
    # Missing material settings must not acquire present-day adapter defaults.
    required = {
        "accepted_response_models",
        "model",
        "provider",
        "api_mode",
        "temperature",
        "max_tokens",
        "protocol_repair_max_tokens",
        "model_context_window_tokens",
        "model_max_output_tokens",
        "provider_timeout_s",
        "provider_retry_max_attempts",
        "provider_retry_max_elapsed_s",
        "provider_failure_policy",
        "max_consecutive_provider_failures",
        "persistent_history_max_messages",
        "persistent_context_max_chars",
        "persistent_memory_max_items",
        "stream_chat_completions",
        "tool_choice",
        "tool_choice_supported",
        "prompt_mode",
        "provider_rpm_limit",
        "provider_rpd_limit",
        "provider_rate_limit_scope",
        "reasoning_effort",
        "reasoning_effort_format",
        "thinking_type",
        "base_url",
        "api_version",
        "effective_api_version",
        "responses_base_url",
        "private_provider_route_sha256",
    }
    if not required <= set(batch["model_shard"]):
        raise ValueError("original_frozen_provider_settings_missing")


def _select_episode(
    rows: list[dict], key: tuple[str, int]
) -> tuple[dict | None, dict | None]:
    matching = [
        (i, row)
        for i, row in enumerate(rows)
        if (row["scenario_signature"], row["seed"]) == key
    ]
    terminal = [(i, row) for i, row in matching if row["status"] != "in_flight"]
    healthy = [(i, row) for i, row in terminal if row["status"] == "ok"]
    selected = max(
        healthy or terminal or matching,
        key=lambda item: (item[1].get("invocation_started_at_utc") or "", item[0]),
        default=None,
    )
    return (selected[1] if selected else None, terminal[-1][1] if terminal else None)


def _original_artifact_path(episode: dict, journal: Path) -> Path:
    declared = Path(episode["artifact_path"])
    if declared.is_absolute() and "trajectories" in declared.parts:
        path = journal.parent.joinpath(
            *declared.parts[declared.parts.index("trajectories") :]
        )
    else:
        path = journal.parent / declared
    if not path.resolve().is_relative_to(journal.parent.resolve()):
        raise ValueError("original_artifact_outside_own_run")
    return path.resolve()


def _audit_episode(
    episode: dict,
    source: dict,
    config: dict,
    journal: Path,
    receipt: dict,
    recoveries: dict,
    root: Path,
) -> tuple[dict, dict]:
    path = _original_artifact_path(episode, journal)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != episode["artifact_sha256"]:
        raise ValueError("original_artifact_hash_mismatch")
    original = json.loads(raw)
    identity = original["treatment_identity"]
    batch = config["batch_treatment_identity"]
    if (
        any(
            original[field] != episode[field] or episode[field] != source[field]
            for field in ("scenario_id", "scenario_signature", "seed")
        )
        or episode["batch_treatment_sha256"] != config["batch_treatment_sha256"]
        or episode["implementation_tree_sha256"] != batch["implementation_tree_sha256"]
        or identity["implementation_contract"]["implementation_tree_sha256"]
        != batch["implementation_tree_sha256"]
        or _digest(identity) != original["treatment_sha256"]
        or original["treatment_sha256"] != episode["episode_treatment_sha256"]
        or episode["horizon_ticks"] != source["horizon_ticks"]
        or episode["tick_interval_s"] != source["tick_interval_s"]
        or episode["model"] != config["model"]
        or episode["pass_index"] != 0
    ):
        raise ValueError("original_episode_identity_mismatch")
    ref = {
        "path": str(path.relative_to(root.resolve())),
        "sha256": episode["artifact_sha256"],
        "byte_count": len(raw),
    }
    recovery = recoveries.get((path, episode["artifact_sha256"]))
    closure = original["evidence_closure"]
    artifact = original
    mode = "original_valid_persistence"
    provenance = {"original_artifact": ref}
    wire_stage = (
        receipt["schema_version"]
        == "operate_realtime_historical_wire_derivation_receipt.v1"
    )
    needs_recovery = _digest(closure["ledger"]) != closure["ledger_sha256"] or (
        wire_stage
        and "PROVIDER_REQUEST_PAYLOAD_HASH_MISMATCH"
        in _wire_hash_reasons(original["provider_audit"])
    )
    if needs_recovery:
        if recovery is None:
            raise ValueError("missing_bound_recovery_attestation")
        recover = (
            recover_realtime_wire_persisted_artifact
            if wire_stage
            else recover_realtime_persisted_artifact
        )
        derived, attestation = recover(
            raw,
            artifact_ref=recovery["original_artifact"],
            original_repo_root=receipt["root_candidate"],
        )
        if recovery["status"] != "derived_verified":
            raise ValueError("receipt_recovery_status_mismatch")
        recorded = json.loads(_read_bound(recovery["derived_artifact"], root))
        recorded_attestation = json.loads(_read_bound(recovery["attestation"], root))
        if (
            recorded != derived
            or recorded_attestation != attestation
            or _digest(derived) != recovery["derived_artifact_canonical_sha256"]
        ):
            raise ValueError(
                "derived_artifact_or_attestation_reauthentication_mismatch"
            )
        artifact = derived
        mode = (
            "authenticated_ledger_and_request_preimage_derivation"
            if wire_stage
            else "authenticated_path_preimage_derivation"
        )
        provenance.update(
            derived_artifact=recovery["derived_artifact"],
            attestation=recovery["attestation"],
        )
    elif recovery is not None:
        raise ValueError("unexpected_recovery_for_healthy_persistence")
    closure = artifact["evidence_closure"]
    ledger = closure["ledger"]
    if (
        type(closure["ledger_count"]) is not int
        or len(ledger) != closure["ledger_count"]
        or _digest(ledger) != closure["ledger_sha256"]
    ):
        raise ValueError("evidence_ledger_hash_or_count_mismatch")
    rebuilt = _build_evidence_closure(
        SimpleNamespace(evidence=SimpleNamespace(to_jsonable=lambda: ledger)), artifact
    )
    if rebuilt != closure:
        raise ValueError("evidence_closure_reconstruction_mismatch")
    aliases = tuple(batch["model_shard"]["accepted_response_models"])
    wire = sorted(
        _wire_hash_reasons(artifact["provider_audit"]).union(
            *(
                _provider_turn_audit_violations(
                    row,
                    accepted_response_models=aliases,
                    audit_rows=artifact["provider_audit"],
                )
                for row in artifact["provider_audit"]
            )
        )
    )
    diagnostics = evaluate_realtime_diagnostics(
        events=artifact["events"],
        turns=artifact["turns"],
        transitions=artifact["transitions"],
        lifecycle=artifact["action_lifecycle"],
        interaction_stats=artifact["llm_interaction_stats"],
        evidence_ledger=ledger,
        polling_events=0,
    )
    if diagnostics != original["diagnostics"] or (
        "diagnostics" in episode and episode["diagnostics"] != original["diagnostics"]
    ):
        raise ValueError("diagnostics_reconstruction_mismatch")
    validated = deepcopy(artifact)
    _apply_realtime_artifact_validation(
        validated,
        behavioral_state_settled=artifact["teardown"].get(
            "behavioral_settlement_complete"
        )
        is True,
    )
    reasons = realtime_artifact_eligibility(artifact, episode, config)
    if episode["status"] != "ok":
        reasons.append(f"journal_status:{episode['status']}")
    if validated["artifact_validation"] != original["artifact_validation"]:
        reasons.append("original_validation_reconstruction_mismatch")
    reasons.extend(wire)
    if validated["artifact_validation"]["valid"] is not True:
        reasons.extend(validated["artifact_validation"]["blocker_codes"])
    return artifact, {
        "eligible": not reasons,
        "reasons": sorted(set(reasons)),
        "persistence_mode": mode,
        "origin": provenance,
        "canonical_original_episode_sha256": _digest(episode),
        "original_implementation_tree_sha256": episode["implementation_tree_sha256"],
        "original_episode_treatment_sha256": episode["episode_treatment_sha256"],
        "diagnostics": diagnostics,
        "original_validation": original["artifact_validation"],
        "current_validation": validated["artifact_validation"],
        "wire_issues": wire,
        "primary027": None,
        "merge_with_logical_primary": False,
    }


def build_report(manifest: dict, *, root: Path = ROOT) -> dict:
    if manifest["schema_version"] != "operate_realtime_diagnostics_input.v1":
        raise ValueError("unsupported_realtime_input_manifest")
    models = manifest["models"]
    if (
        not isinstance(models, list)
        or not models
        or len(set(models)) != len(models)
        or any(not isinstance(model, str) or not model for model in models)
    ):
        raise ValueError("fixed_model_denominator_invalid")
    release = json.loads(_read_bound(manifest["release_manifest"], root))
    _read_bound(manifest["source_suite"], root)
    sources = {
        role: _selected_sources(ref, root, manifest["expected_cases"][role])
        for role, ref in manifest["source_suites"].items()
    }
    if set(sources) != {"Core37", "E3"} or not set(sources["E3"]) <= set(
        sources["Core37"]
    ):
        raise ValueError("source_panel_relationship_invalid")
    receipt = json.loads(_read_bound(manifest["derivation_receipt"], root))
    if receipt["schema_version"] not in {
        "operate_realtime_historical_derivation_receipt.v1",
        "operate_realtime_historical_wire_derivation_receipt.v1",
    }:
        raise ValueError("unsupported_realtime_derivation_receipt")
    recoveries = {}
    for row in receipt["rows"]:
        ref = row["original_artifact"]
        key = _path(ref["path"], root), ref["sha256"]
        if key in recoveries:
            raise ValueError("duplicate_recovery_original")
        recoveries[key] = row
    receipt_journals = {
        (row["path"], row["sha256"]) for row in receipt["source_journals"]
    }
    reports, records, artifacts, configs = [], [], {}, {}
    e3_cohort_keys = set()
    for cohort in manifest["cohorts"]:
        role, model = cohort["role"], cohort["model"]
        if role not in sources or model not in manifest["models"]:
            raise ValueError("undeclared_cohort")
        if (
            cohort["journal"]["path"],
            cohort["journal"]["sha256"],
        ) not in receipt_journals:
            raise ValueError("cohort_journal_not_bound_to_receipt")
        journal = _path(cohort["journal"]["path"], root)
        if _path(cohort["run_config"]["path"], root) != journal.with_name(
            "run_config.json"
        ):
            raise ValueError("original_config_outside_own_run")
        config = json.loads(_read_bound(cohort["run_config"], root))
        _bind_config(cohort, manifest, release, config, sources[role])
        batch_hash = config["batch_treatment_sha256"]
        if batch_hash in configs:
            raise ValueError("duplicate_formal_cohort")
        configs[batch_hash] = config
        delay = config["batch_treatment_identity"]["clock"]["response_delivery_delay_s"]
        if role == "E3":
            e3_key = model, delay
            if e3_key in e3_cohort_keys or delay not in {0.0, 1.0, 5.0}:
                raise ValueError("duplicate_or_undeclared_e3_arm")
            e3_cohort_keys.add(e3_key)
        rows = [
            json.loads(line)
            for line in _read_bound(cohort["journal"], root).splitlines()
            if line.strip()
        ]
        if any(
            (row["scenario_signature"], row["seed"]) not in sources[role]
            for row in rows
        ):
            raise ValueError("foreign_source_episode")
        cohort_records = []
        for key, source in sources[role].items():
            episode, latest_terminal = _select_episode(rows, key)
            result = {
                "role": role,
                "model": model,
                "scenario_signature": key[0],
                "seed": key[1],
                "delay_s": delay,
                "batch_treatment_sha256": batch_hash,
                "journal": cohort["journal"],
                "run_config": cohort["run_config"],
                "max_workers": config["batch_treatment_identity"]["scheduler"][
                    "max_workers"
                ],
                "eligible": False,
                "reasons": ["missing"],
                "selected_journal_status": episode["status"] if episode else "missing",
                "original_journal_eligibility_reasons": episode.get(
                    "eligibility_reasons", []
                )
                if episode
                else [],
                "latest_terminal_status": latest_terminal["status"]
                if latest_terminal
                else None,
                "invocation_started_at_utc": episode.get("invocation_started_at_utc")
                if episode
                else None,
                "diagnostics": None,
                "primary027": None,
                "merge_with_logical_primary": False,
            }
            if episode:
                try:
                    if not episode.get("artifact_path"):
                        raise ValueError(
                            f"original_artifact_missing:{episode['status']}"
                        )
                    artifact, audited = _audit_episode(
                        episode, source, config, journal, receipt, recoveries, root
                    )
                    result.update(audited)
                    artifacts[(batch_hash, *key)] = artifact
                except (OSError, ValueError, KeyError, TypeError) as error:
                    result["reasons"] = [f"{type(error).__name__}:{error}"]
            records.append(result)
            cohort_records.append(result)
        jobs = [{"job_key": f"{key[0]}:{key[1]}"} for key in sources[role]]
        score_rows = [
            {
                "job_key": f"{row['scenario_signature']}:{row['seed']}",
                "status": "ok" if row["eligible"] else "ineligible",
                "diagnostics": row["diagnostics"] or {},
            }
            for row in cohort_records
        ]
        reports.append(
            {
                "role": role,
                "model": model,
                "batch_treatment_sha256": batch_hash,
                "journal": cohort["journal"],
                "run_config": cohort["run_config"],
                "max_workers": config["batch_treatment_identity"]["scheduler"][
                    "max_workers"
                ],
                "delay_s": delay,
                "expected_cases": len(sources[role]),
                "eligible_cases": sum(row["eligible"] for row in cohort_records),
                "missing_or_ineligible_cases": sum(
                    not row["eligible"] for row in cohort_records
                ),
                "status_counts": dict(
                    Counter(row["selected_journal_status"] for row in cohort_records)
                ),
                "scorecard_scope": "eligible_subset_of_this_original_run_only",
                "scorecard": aggregate_realtime_scorecard(score_rows, jobs, config),
            }
        )
    e3_groups = []
    for model in manifest["models"]:
        for key in sources["E3"]:
            arms = {
                str(int(row["delay_s"])): row
                for row in records
                if row["role"] == "E3"
                and row["model"] == model
                and (row["scenario_signature"], row["seed"]) == key
            }
            values = [
                artifacts[(row["batch_treatment_sha256"], *key)]
                for row in arms.values()
                if row["eligible"]
            ]
            group = analyze_e3_group(values)
            identities = []
            for row in arms.values():
                frozen = deepcopy(
                    configs[row["batch_treatment_sha256"]]["batch_treatment_identity"]
                )
                frozen["clock"].pop("response_delivery_delay_s")
                identities.append(frozen)
            if len(identities) == 3 and any(
                value != identities[0] for value in identities[1:]
            ):
                group = {
                    "valid": False,
                    "problem": "original_e3_batch_identity_mismatch",
                }
            e3_groups.append(
                {
                    "model": model,
                    "scenario_signature": key[0],
                    "seed": key[1],
                    "arms": {
                        delay: {
                            "eligible": arms[delay]["eligible"],
                            "reasons": arms[delay]["reasons"],
                            "journal_status": arms[delay]["selected_journal_status"],
                            "batch_treatment_sha256": arms[delay][
                                "batch_treatment_sha256"
                            ],
                            "diagnostics": arms[delay]["diagnostics"]
                            if arms[delay]["eligible"]
                            else None,
                        }
                        if delay in arms
                        else {
                            "eligible": False,
                            "reasons": ["missing_e3_arm"],
                            "journal_status": "missing",
                            "batch_treatment_sha256": None,
                            "diagnostics": None,
                        }
                        for delay in ("0", "1", "5")
                    },
                    "group": group,
                }
            )
    availability = []
    for model in manifest["models"]:
        cells = []
        for key in sources["Core37"]:
            candidates = [
                row
                for row in records
                if row["role"] == "Core37"
                and row["model"] == model
                and (row["scenario_signature"], row["seed"]) == key
                and row["selected_journal_status"] == "ok"
            ]
            selected = max(
                candidates,
                key=lambda row: (
                    row["invocation_started_at_utc"] or "",
                    row["journal"]["sha256"],
                ),
                default=None,
            )
            cells.append(
                {
                    "scenario_signature": key[0],
                    "seed": key[1],
                    "eligible": selected["eligible"] if selected else False,
                    "selected_batch_treatment_sha256": selected[
                        "batch_treatment_sha256"
                    ]
                    if selected
                    else None,
                    "reasons": selected["reasons"]
                    if selected
                    else ["no_original_status_ok_candidate"],
                }
            )
        availability.append(
            {
                "model": model,
                "expected_cases": len(cells),
                "eligible_cases": sum(cell["eligible"] for cell in cells),
                "missing_or_ineligible_cases": sum(
                    not cell["eligible"] for cell in cells
                ),
                "selection_rule": "latest_original_status_ok_before_audit_no_score_selection_or_post_audit_fallback",
                "scope": "descriptive_cross_cohort_case_availability_not_merged_formal_score",
                "primary027": None,
                "cells": cells,
            }
        )
    return {
        "schema_version": "operate_realtime_trajectory_diagnostics.v1",
        "input_manifest_canonical_sha256": _digest(manifest),
        "inputs": {
            key: manifest[key]
            for key in (
                "release_manifest",
                "source_suite",
                "source_suites",
                "derivation_receipt",
            )
        },
        "selection_rule": "latest_original_status_ok_before_audit_with_original_nonok_statuses_retained",
        "fixed_denominators": {
            "core_cases_per_model": len(sources["Core37"]),
            "e3_cases_per_model": len(sources["E3"]),
            "e3_models": len(manifest["models"]),
            "e3_groups": len(e3_groups),
            "e3_arm_cells": len(e3_groups) * 3,
        },
        "e3_valid_groups": sum(row["group"]["valid"] for row in e3_groups),
        "e3_groups": e3_groups,
        "cohorts": reports,
        "core_case_availability": availability,
        "rows": records,
        "primary027": None,
        "merge_with_logical_primary": False,
        "execution": {
            "provider_calls": 0,
            "native_episode_replays": 0,
            "environment_constructors": 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    modules = (
        "scripts/evaluate_realtime_trajectories.py",
        "evaluation/realtime_persistence_recovery.py",
        "evaluation/realtime_diagnostics.py",
        "runner/realtime_episode.py",
        "scripts/batch_realtime_llm_eval.py",
        "core/realtime_clock.py",
        "evaluation/supplementary_analysis.py",
    )
    before = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in modules
    }
    raw = args.manifest.read_bytes()
    report = build_report(json.loads(raw), root=args.repo_root.resolve())
    report["input_manifest"] = {
        "path": str(args.manifest),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
    }
    if before != {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in modules
    }:
        raise ValueError("analysis_implementation_changed_during_read")
    report["analysis_implementation"] = before
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "e3_valid_groups": report["e3_valid_groups"],
                "e3_expected_groups": report["fixed_denominators"]["e3_groups"],
                "core_case_availability": [
                    {
                        key: row[key]
                        for key in ("model", "expected_cases", "eligible_cases")
                    }
                    for row in report["core_case_availability"]
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
