"""Fail-closed source-consumption evidence validation for protocol-2.1."""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

from core.source_asset_contract import resolve_source_asset_contract
from domains.registry import get_backend_capability


_NAMED_EVENT_FIELDS = (
    "expected_source_event_ids",
    "observed_source_event_ids",
    "material_source_event_ids",
    "source_event_materiality",
)


def _named_event_evidence_is_valid(evidence: dict[str, Any]) -> bool:
    """Validate an exact native-event causal attestation when one is emitted."""
    exact_contract = bool(evidence.get("formal_admission")) or any(
        field in evidence for field in _NAMED_EVENT_FIELDS
    )
    if not exact_contract:
        return True
    expected = evidence.get("expected_source_event_ids")
    observed = evidence.get("observed_source_event_ids")
    material = evidence.get("material_source_event_ids")
    rows = evidence.get("source_event_materiality")
    if not all(isinstance(values, list) for values in (expected, observed, material, rows)):
        return False
    if (
        not expected
        or len(expected) != len(set(expected))
        or observed != expected
        or material != expected
        or len(rows) != len(expected)
    ):
        return False
    for event_id, row in zip(expected, rows):
        if not isinstance(row, dict):
            return False
        changed_fields = row.get("changed_state_fields")
        before = str(row.get("before_state_digest") or "")
        after = str(row.get("after_state_digest") or "")
        if (
            row.get("event_id") != event_id
            or row.get("state_observation_kind") != "native_backend_readback"
            or row.get("materiality_passed") is not True
            or not isinstance(changed_fields, list)
            or not changed_fields
            or any(not isinstance(field, str) or not field for field in changed_fields)
            or not before
            or not after
            or before == after
        ):
            return False
    return True


def _path_aliases(path: str, *, repo_root: Path) -> tuple[str, ...]:
    """Return stable relative/absolute aliases for a source evidence path."""
    raw = str(path)
    aliases = [raw]
    candidate = Path(raw)
    try:
        resolved = candidate.resolve() if candidate.is_absolute() else (repo_root / candidate).resolve()
        relative = resolved.relative_to(repo_root.resolve()).as_posix()
    except (OSError, ValueError):
        relative = None
    if relative and relative not in aliases:
        aliases.append(relative)
    return tuple(aliases)


def _align_hash_map(
    values: dict[str, Any],
    *,
    declared_keys: set[str],
    repo_root: Path,
) -> dict[str, Any]:
    """Align runtime absolute paths with repository-relative declarations.

    Runtime backends commonly report absolute paths while scenario contracts
    intentionally use reproducible repository-relative paths.  Preserve an
    exact declared spelling when one is present (important for external
    temporary sources), otherwise use the matching alias.
    """
    declared_aliases = {
        key: set(_path_aliases(key, repo_root=repo_root))
        for key in declared_keys
    }
    aligned: dict[str, Any] = {}
    for path, value in values.items():
        raw = str(path)
        aliases = set(_path_aliases(raw, repo_root=repo_root))
        matches = [
            key
            for key, key_aliases in declared_aliases.items()
            if aliases & key_aliases
        ]
        if matches:
            for key in matches:
                aligned[key] = value
        else:
            aligned[raw] = value
    return aligned


def resolve_declared_sources(
    scenario: dict[str, Any],
    *,
    repo_root: Path,
) -> tuple[list[str], dict[str, str], list[str]]:
    contract = resolve_source_asset_contract(scenario, repo_root=repo_root)
    declared = [
        *contract.required_runtime_source_files,
        *contract.required_derivation_source_files,
    ]
    return (
        declared,
        dict(contract.locked_source_hashes),
        list(contract.missing_required_files),
    )


def normalize_runtime_source_evidence(
    *,
    row: dict[str, Any],
    scenario: dict[str, Any],
    replay_evidence: list[dict[str, Any]],
    repo_root: Path,
) -> dict[str, Any]:
    backend_kind = str(scenario.get("backend_kind") or "")
    derived_window = (
        (scenario.get("source_contract") or {}).get("derived_window")
        or {}
    )
    if (
        backend_kind == "orgym_invmgmt"
        or (
            backend_kind == "alibaba_trace_sim"
            and not derived_window.get("sha256")
        )
    ):
        from domains import source_contracts

        builder = getattr(
            source_contracts,
            backend_kind,
        )
        scenario = copy.deepcopy(scenario)
        scenario["source_contract"] = builder(
            scenario, repo_root
        )
    contract = resolve_source_asset_contract(scenario, repo_root=repo_root)
    declared = [
        *contract.required_runtime_source_files,
        *contract.required_derivation_source_files,
    ]
    declared_hashes = dict(contract.locked_source_hashes)
    missing = list(contract.missing_required_files)
    runtime_hashes = {
        path: declared_hashes[path]
        for path in contract.required_runtime_source_files
        if path in declared_hashes
    }
    derivation_hashes = {
        path: declared_hashes[path]
        for path in contract.required_derivation_source_files
        if path in declared_hashes
    }
    base = {
        "scenario_id": row.get("scenario_id"),
        "scenario_signature": row.get("scenario_signature"),
        "backend_kind": row.get("backend_kind"),
        "status": "held",
        "proof_kind": None,
        "declared_source_files": declared,
        "locked_source_hashes": declared_hashes,
        "required_runtime_source_files": list(
            contract.required_runtime_source_files
        ),
        "required_derivation_source_files": list(
            contract.required_derivation_source_files
        ),
        "consumed_source_hashes": {},
        "lineage_source_hashes": {},
        "consumed_channels": [],
        "derived_backend_state_fields": [],
        "consumption_ticks": [],
        "direct_runtime_match": False,
        "derived_window_lineage_match": False,
        "runtime_source_coverage": 0.0,
        "state_effect_observed": False,
        "deterministic_across_replays": False,
        "runtime_trace_observed": False,
        "evidence_from_scenario_config_only": False,
        "direct_runtime_determinism_proven": False,
        "named_events_causally_proven": None,
        "expected_source_event_ids": [],
        "observed_source_event_ids": [],
        "material_source_event_ids": [],
        "source_event_materiality": [],
        "blockers": [],
    }
    try:
        capability = get_backend_capability(scenario.get("backend_kind"))
    except KeyError:
        capability = None
    if capability is not None and not capability.formal_core_allowed:
        base["status"] = "failed"
        base["blockers"] = ["backend_formal_fidelity_not_allowed"]
        return base
    if missing:
        base["status"] = "failed"
        base["blockers"] = ["required_source_file_missing"]
        return base
    if contract.contract_errors:
        base["blockers"] = list(contract.contract_errors)
        return base
    if len(replay_evidence) < 2:
        base["blockers"] = ["source_consumption_replay_pair_missing"]
        return base
    first, second = replay_evidence[:2]
    if not isinstance(first, dict) or not isinstance(second, dict):
        base["blockers"] = ["backend_native_source_trace_invalid"]
        return base
    base.update(
        {
            "expected_source_event_ids": list(
                first.get("expected_source_event_ids") or []
            ),
            "observed_source_event_ids": list(
                first.get("observed_source_event_ids") or []
            ),
            "material_source_event_ids": list(
                first.get("material_source_event_ids") or []
            ),
            "source_event_materiality": copy.deepcopy(
                first.get("source_event_materiality") or []
            ),
        }
    )
    upstream_statuses = {
        str(evidence.get("status") or "")
        for evidence in (first, second)
        if evidence.get("status") is not None
    }
    upstream_blockers = sorted(
        {
            str(blocker)
            for evidence in (first, second)
            for blocker in evidence.get("blockers") or []
            if str(blocker)
        }
    )
    named_event_proofs = [
        evidence.get("named_events_causally_proven")
        for evidence in (first, second)
        if evidence.get("named_events_causally_proven") is not None
    ]
    base["named_events_causally_proven"] = (
        all(value is True for value in named_event_proofs)
        if named_event_proofs
        else None
    )
    if (
        "failed" in upstream_statuses
        or "held" in upstream_statuses
        or upstream_blockers
    ):
        base["status"] = "failed" if "failed" in upstream_statuses else "held"
        base["blockers"] = upstream_blockers or [
            "backend_native_source_trace_not_passed"
        ]
        return base
    if not all(
        _named_event_evidence_is_valid(evidence)
        for evidence in (first, second)
    ):
        base["blockers"] = ["named_source_events_causal_proof_invalid"]
        return base
    if named_event_proofs and not all(
        value is True for value in named_event_proofs
    ):
        base["blockers"] = upstream_blockers or [
            "named_source_events_causal_proof_missing"
        ]
        return base
    comparable_fields = (
        "status",
        "blockers",
        "named_events_causally_proven",
        "expected_source_event_ids",
        "observed_source_event_ids",
        "material_source_event_ids",
        "source_event_materiality",
        "proof_kind",
        "consumed_source_hashes",
        "lineage_source_hashes",
        "consumed_window_sha256",
        "recipe_version",
        "consumed_channels",
        "derived_backend_state_fields",
        "consumption_ticks",
        "opened_source_paths",
        "opened_source_sha256",
        "parser_output_digest",
        "instance_kind",
        "initial_state_digest",
        "source_field_to_state_field_map",
        "runtime_job_window_digest",
        "post_source_state_digests",
        "deterministic_source_trace",
        "source_window",
        "source_state_effect_observed",
        "trace_semantic_digest",
        "complete_source_identity_sha256",
        "runtime_opened_assets",
        "runtime_trace_observed",
        "evidence_from_scenario_config_only",
    )
    deterministic = all(
        first.get(field) == second.get(field) for field in comparable_fields
    )
    deterministic = bool(
        deterministic
        and first.get("deterministic_source_trace") is not False
        and second.get("deterministic_source_trace") is not False
    )
    direct_runtime = first.get("proof_kind") == "direct_runtime_files"
    direct_runtime_determinism_proven = not direct_runtime or bool(
        deterministic
        and all(
            bool(evidence.get("trace_semantic_digest"))
            and bool(evidence.get("post_source_state_digests"))
            and bool(evidence.get("runtime_opened_assets"))
            for evidence in (first, second)
        )
    )
    consumed_hashes = dict(first.get("consumed_source_hashes") or {})
    if (
        not consumed_hashes
        and first.get("proof_kind") == "direct_runtime_files"
        and first.get("runtime_trace_observed") is True
        and first.get("evidence_from_scenario_config_only") is not True
    ):
        opened_paths = {
            alias
            for path in first.get("opened_source_paths") or []
            for alias in _path_aliases(str(path), repo_root=repo_root)
        }
        opened_hashes = _align_hash_map(
            dict(first.get("opened_source_sha256") or {}),
            declared_keys=set(runtime_hashes),
            repo_root=repo_root,
        )
        consumed_hashes = {
            str(path): str(digest)
            for path, digest in opened_hashes.items()
            if str(path) in opened_paths or str(path) in runtime_hashes
        }
    consumed_hashes = _align_hash_map(
        consumed_hashes,
        declared_keys={*runtime_hashes, *derivation_hashes},
        repo_root=repo_root,
    )
    lineage_hashes = _align_hash_map(
        dict(first.get("lineage_source_hashes") or {}),
        declared_keys=set(derivation_hashes),
        repo_root=repo_root,
    )
    declared_window_hash = str(contract.derived_window_sha256 or "")
    consumed_window_hash = str(first.get("consumed_window_sha256") or "")
    direct_match = bool(runtime_hashes) and all(
        consumed_hashes.get(path) == digest
        for path, digest in runtime_hashes.items()
    )
    lineage_match = bool(
        derivation_hashes
        and declared_window_hash
        and consumed_window_hash == declared_window_hash
        and first.get("recipe_version") == contract.recipe_version
        and all(
            lineage_hashes.get(path) == digest
            for path, digest in derivation_hashes.items()
        )
    )
    coverage = (
        sum(
            consumed_hashes.get(path) == digest
            for path, digest in runtime_hashes.items()
        )
        / len(runtime_hashes)
        if runtime_hashes
        else 0.0
    )
    channels = list(first.get("consumed_channels") or [])
    state_fields = list(first.get("derived_backend_state_fields") or [])
    ticks = list(first.get("consumption_ticks") or [])
    state_effect = first.get("state_effect_observed") is True and bool(state_fields)
    runtime_trace = first.get("runtime_trace_observed") is True
    config_only = first.get("evidence_from_scenario_config_only") is True
    base.update(
        {
            "proof_kind": first.get("proof_kind"),
            "consumed_source_hashes": consumed_hashes,
            "lineage_source_hashes": lineage_hashes,
            "declared_window_sha256": declared_window_hash,
            "consumed_window_sha256": consumed_window_hash,
            "direct_runtime_match": direct_match,
            "derived_window_lineage_match": lineage_match,
            "runtime_source_coverage": coverage,
            "consumed_channels": channels,
            "derived_backend_state_fields": state_fields,
            "consumption_ticks": ticks,
            "state_effect_observed": state_effect,
            "deterministic_across_replays": deterministic,
            "runtime_trace_observed": runtime_trace,
            "evidence_from_scenario_config_only": config_only,
            "direct_runtime_determinism_proven": (
                direct_runtime_determinism_proven
            ),
            "runtime_opened_assets": list(
                first.get("runtime_opened_assets") or []
            ),
            "parser_output_digest": first.get("parser_output_digest"),
            "instance_kind": first.get("instance_kind"),
            "initial_state_digest": first.get("initial_state_digest"),
            "post_source_state_digests": list(
                first.get("post_source_state_digests") or []
            ),
            "source_window": dict(first.get("source_window") or {}),
            "source_state_effect_observed": (
                first.get("source_state_effect_observed") is True
            ),
            "deterministic_source_trace": (
                direct_runtime_determinism_proven
                if direct_runtime
                else first.get("deterministic_source_trace") is True
            ),
            "trace_semantic_digest": first.get(
                "trace_semantic_digest"
            ),
        }
    )
    if first.get("declared_source_unused") is True:
        base["status"] = "failed"
        base["blockers"] = ["backend_did_not_consume_declared_source"]
    elif first.get("controlled_intervention_no_effect") is True:
        base["status"] = "failed"
        base["blockers"] = ["controlled_source_intervention_no_effect"]
    elif not (direct_match or lineage_match):
        base["status"] = "failed"
        base["blockers"] = ["source_hash_or_lineage_mismatch"]
    elif not channels:
        base["blockers"] = ["consumed_source_channel_missing"]
    elif config_only:
        base["blockers"] = ["scenario_config_only_source_evidence"]
    elif not runtime_trace:
        base["blockers"] = ["backend_runtime_source_trace_missing"]
    elif not state_effect:
        base["blockers"] = ["source_state_effect_unproven"]
    elif not deterministic:
        base["blockers"] = ["deterministic_source_trace_mismatch"]
    elif not direct_runtime_determinism_proven:
        base["blockers"] = ["direct_runtime_determinism_unproven"]
    else:
        base["status"] = "passed"
    return base


SourceEvidenceExtractor = Callable[..., dict[str, Any]]
