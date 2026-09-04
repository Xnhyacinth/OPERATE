#!/usr/bin/env python3
"""Merge compatible finalized logical-persistent formal LLM shards.

The input shard remains the unit of provider execution and retry.  This merger
is the publication boundary: it revalidates every source shard and episode,
recomputes the cross-model primary inference, and publishes the merged manifest
last as the commit marker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.leaderboard import PrimaryLeaderboardContractError  # noqa: E402
from scripts import batch_llm_eval as batch  # noqa: E402


MERGE_SCHEMA_VERSION = "formal_llm_shard_merge/1.0"
MERGED_LEADERBOARD_SCHEMA_VERSION = "formal_llm_merged_leaderboard/1.0"
FAMILY_SCHEMA_VERSION = "formal_treatment_family/1.0"
FORMAL_CONTRACT_VERSION = "agentic_persistent.v1"
FORMAL_INTERACTION_MODE = "logical_persistent"
FORMAL_AGENTIC_PROFILE = {
    "max_tokens": 32_768,
    "protocol_repair_max_tokens": 8_192,
    "persistent_history_max_messages": 64,
    "persistent_context_max_chars": 512_000,
    "persistent_memory_max_items": 128,
    "provider_timeout_s": 300.0,
    "max_consecutive_provider_failures": 1,
    "provider_failure_policy": "abort",
    "tool_choice": "auto",
    "stream_chat_completions": True,
}

_PROFILE_FIELDS = (
    "harness",
    "within_tick_interaction",
    "max_tokens",
    "persistent_history_max_messages",
    "persistent_context_max_chars",
    "persistent_memory_max_items",
    "protocol_repair_max_tokens",
    "tool_choice",
    "stream_chat_completions",
    "token_count_method",
    "token_count_version",
    "provider_timeout_s",
    "max_consecutive_provider_failures",
    "provider_failure_policy",
)


class FormalShardMergeError(ValueError):
    """The source shards cannot form one formal comparison stratum."""


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalShardMergeError(f"{label} is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise FormalShardMergeError(f"{label} must be a JSON object: {path}")
    return payload


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise FormalShardMergeError(f"{label} must be a positive integer")
    return value


def _sha256_text(value: Any, *, label: str) -> str:
    text = str(value or "")
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise FormalShardMergeError(f"{label} must be a SHA-256 digest")
    return text


def _eligibility_is_green(payload: dict[str, Any]) -> bool:
    eligibility = payload.get("leaderboard_eligibility")
    return bool(
        payload.get("leaderboard_eligible") is True
        and isinstance(eligibility, dict)
        and eligibility.get("eligible") is True
        and eligibility.get("blockers") == []
    )


def _scenario_seed_pairs(manifest: dict[str, Any]) -> list[list[Any]]:
    value = manifest.get("scenario_seed_pairs")
    if not isinstance(value, list) or not value:
        raise FormalShardMergeError("scenario_seed_pairs must be nonempty")
    normalized: list[list[Any]] = []
    for pair in value:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or not isinstance(pair[0], str)
            or not pair[0]
            or pair[0].strip() != pair[0]
            or isinstance(pair[1], bool)
            or not isinstance(pair[1], int)
        ):
            raise FormalShardMergeError("scenario_seed_pairs is invalid")
        normalized.append([pair[0], pair[1]])
    if len({(pair[0], pair[1]) for pair in normalized}) != len(normalized):
        raise FormalShardMergeError("scenario_seed_pairs contains duplicates")
    return sorted(normalized, key=lambda pair: (pair[0], pair[1]))


def _profile(manifest: dict[str, Any]) -> dict[str, Any]:
    profile = {field: manifest.get(field) for field in _PROFILE_FIELDS}
    if any(value is None for value in profile.values()):
        raise FormalShardMergeError("persistent agent profile is incomplete")
    for field in (
        "max_tokens",
        "persistent_history_max_messages",
        "persistent_context_max_chars",
        "persistent_memory_max_items",
        "protocol_repair_max_tokens",
        "max_consecutive_provider_failures",
    ):
        _positive_int(profile[field], label=f"profile.{field}")
    if profile["tool_choice"] not in {"auto", "required"}:
        raise FormalShardMergeError("profile.tool_choice is invalid")
    if profile["provider_failure_policy"] != "abort":
        raise FormalShardMergeError(
            "profile.provider_failure_policy must be abort"
        )
    if not isinstance(profile["stream_chat_completions"], bool):
        raise FormalShardMergeError(
            "profile.stream_chat_completions must be boolean"
        )
    if profile["within_tick_interaction"] is not True:
        raise FormalShardMergeError(
            "profile.within_tick_interaction must be true"
        )
    for field in (
        "harness",
        "tool_choice",
        "token_count_method",
        "token_count_version",
    ):
        if not isinstance(profile[field], str) or not profile[field]:
            raise FormalShardMergeError(f"profile.{field} must be nonempty")
    timeout = profile["provider_timeout_s"]
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or float(timeout) <= 0.0
    ):
        raise FormalShardMergeError("profile.provider_timeout_s must be positive")
    profile["provider_timeout_s"] = float(timeout)
    return profile


def _concurrency(manifest: dict[str, Any]) -> dict[str, int]:
    requested_workers = _positive_int(
        manifest.get("max_workers_requested"),
        label="concurrency.max_workers_requested",
    )
    effective_workers = _positive_int(
        manifest.get("max_workers_effective"),
        label="concurrency.max_workers_effective",
    )
    if requested_workers != effective_workers:
        raise FormalShardMergeError("concurrency requested/effective mismatch")
    return {
        "max_workers_requested": requested_workers,
        "max_workers_effective": effective_workers,
    }


def _formal_treatment_family_projection(
    manifest: dict[str, Any], leaderboard: dict[str, Any]
) -> dict[str, Any]:
    """Project only protocol-wide fields shared across model/provider shards."""

    contract = manifest.get("formal_run_contract")
    if not isinstance(contract, dict):
        raise FormalShardMergeError("formal_run_contract must be an object")
    _concurrency(manifest)
    return {
        "schema_version": FAMILY_SCHEMA_VERSION,
        "suite_manifest_sha256": _sha256_text(
            manifest.get("suite_manifest_sha256"), label="suite hash"
        ),
        "implementation_tree_sha256": _sha256_text(
            manifest.get("implementation_tree_sha256"),
            label="implementation tree hash",
        ),
        "formal_run_contract": contract,
        "formal_run_contract_sha256": _canonical_json_sha256(contract),
        "prompt_mode": manifest.get("prompt_mode"),
        "interaction_mode": manifest.get("interaction_mode"),
        "seed_mode": manifest.get("seed_mode"),
        "scenario_seed_pairs": _scenario_seed_pairs(manifest),
        "scoring_version": manifest.get("scoring_version"),
        "primary_leaderboard_formula_version": leaderboard.get(
            "primary_leaderboard_formula_version"
        ),
        "primary_inference_version": leaderboard.get(
            "primary_inference_version"
        ),
        "evaluation_protocol_version": manifest.get(
            "evaluation_protocol_version"
        ),
        "temperature": manifest.get("temperature"),
        "pass_k": manifest.get("pass_k"),
        "save_trajectories": manifest.get("save_trajectories"),
        "scheduler_mode": manifest.get("scheduler_mode"),
        "persistent_agent_profile": _profile(manifest),
    }


def formal_treatment_family_sha256(
    manifest: dict[str, Any], leaderboard: dict[str, Any]
) -> str:
    """Stable comparison-stratum hash, intentionally model/provider agnostic."""

    return _canonical_json_sha256(
        _formal_treatment_family_projection(manifest, leaderboard)
    )


def _validate_current_formal_contract(manifest: dict[str, Any]) -> None:
    if manifest.get("formal_run") is not True:
        raise FormalShardMergeError("source shard is not a formal run")
    if manifest.get("batch_state") != batch.BATCH_STATE_FINAL:
        raise FormalShardMergeError("source shard is not finalized")
    if manifest.get("finalize_enabled") is not True:
        raise FormalShardMergeError("source shard was not finalized explicitly")
    if (
        manifest.get("git_metadata_available") is not True
        or manifest.get("git_dirty") is not False
    ):
        raise FormalShardMergeError("source shard implementation tree is not clean")
    if not _eligibility_is_green(manifest):
        raise FormalShardMergeError("source shard is not leaderboard eligible")
    if manifest.get("implementation_tree_stable") is not True:
        raise FormalShardMergeError("source implementation tree is not stable")
    if manifest.get("formal_runtime_binding_stable") is not True:
        raise FormalShardMergeError("source formal runtime binding is not stable")

    contract = manifest.get("formal_run_contract")
    if not isinstance(contract, dict):
        raise FormalShardMergeError("formal run contract is missing")
    if contract.get("contract_version") != FORMAL_CONTRACT_VERSION:
        raise FormalShardMergeError(
            f"formal contract version must be {FORMAL_CONTRACT_VERSION}"
        )
    required_contract_fields = {
        "required_model_count_per_shard": 1,
        "minimum_pass_k": 1,
        "minimum_max_workers": 1,
        "maximum_max_workers": 32,
        "required_interaction_mode": FORMAL_INTERACTION_MODE,
        "required_prompt_mode": "strict",
        "required_seed_mode": "scenario",
        "required_scheduler_mode": "global",
        "required_temperature": 0.0,
        "requires_explicit_model_capabilities": True,
        "agentic_profile": FORMAL_AGENTIC_PROFILE,
        "save_trajectories": True,
    }
    for field, expected in required_contract_fields.items():
        actual = contract.get(field)
        if type(actual) is not type(expected) or actual != expected:
            raise FormalShardMergeError(
                f"formal run contract {field} must equal {expected!r}"
            )

    required_manifest_fields = {
        "prompt_mode": "strict",
        "interaction_mode": FORMAL_INTERACTION_MODE,
        "seed_mode": "scenario",
        "scheduler_mode": "global",
        "scoring_version": batch.SCORING_VERSION,
        "evaluation_protocol_version": batch.EVALUATION_PROTOCOL_VERSION,
        "temperature": 0.0,
        "save_trajectories": True,
    }
    for field, expected in required_manifest_fields.items():
        if manifest.get(field) != expected:
            raise FormalShardMergeError(f"{field} must equal {expected!r}")
    temperature = manifest.get("temperature")
    if isinstance(temperature, bool) or not isinstance(temperature, float):
        raise FormalShardMergeError("temperature must be the float 0.0")

    models = manifest.get("models")
    if (
        not isinstance(models, list)
        or len(models) != 1
        or not isinstance(models[0], str)
        or not models[0]
        or models[0].strip() != models[0]
    ):
        raise FormalShardMergeError("each source shard must contain one unique model")
    pass_k = _positive_int(manifest.get("pass_k"), label="pass_k")
    if pass_k < int(contract["minimum_pass_k"]):
        raise FormalShardMergeError("pass_k is below the formal minimum")
    pairs = _scenario_seed_pairs(manifest)
    n_scenarios = _positive_int(
        manifest.get("n_scenarios"), label="n_scenarios"
    )
    if n_scenarios != len(pairs):
        raise FormalShardMergeError("n_scenarios does not match scenario_seed_pairs")
    expected_total = _positive_int(
        manifest.get("expected_total"), label="expected_total"
    )
    if expected_total != n_scenarios * pass_k:
        raise FormalShardMergeError("expected_total does not match the shard grid")
    n_episodes_total = _positive_int(
        manifest.get("n_episodes_total"), label="n_episodes_total"
    )
    n_episodes_ok = _positive_int(
        manifest.get("n_episodes_ok"), label="n_episodes_ok"
    )
    n_episodes_error = manifest.get("n_episodes_error")
    if (
        n_episodes_total != expected_total
        or n_episodes_ok != expected_total
        or isinstance(n_episodes_error, bool)
        or n_episodes_error != 0
    ):
        raise FormalShardMergeError("formal shard episode counts are incomplete")

    concurrency = _concurrency(manifest)
    requested_workers = concurrency["max_workers_requested"]
    if not (
        int(contract["minimum_max_workers"])
        <= requested_workers
        <= int(contract["maximum_max_workers"])
    ):
        raise FormalShardMergeError("concurrency is outside the formal worker range")
    profile = _profile(manifest)
    for field, expected in FORMAL_AGENTIC_PROFILE.items():
        if profile[field] != expected:
            raise FormalShardMergeError(
                f"formal agentic profile {field} must equal {expected!r}"
            )

    model = models[0]
    context_caps = manifest.get("model_context_window_tokens_by_model")
    output_caps = manifest.get("model_max_output_tokens_by_model")
    if (
        not isinstance(context_caps, dict)
        or set(context_caps) != {model}
        or not isinstance(output_caps, dict)
        or set(output_caps) != {model}
    ):
        raise FormalShardMergeError("model capability binding is incomplete")
    context = _positive_int(context_caps[model], label="model context capability")
    output = _positive_int(output_caps[model], label="model output capability")
    if output > context:
        raise FormalShardMergeError("model output capability exceeds context")
    treatment_map = manifest.get("agent_treatment_sha256_by_model")
    if not isinstance(treatment_map, dict) or set(treatment_map) != {model}:
        raise FormalShardMergeError("agent treatment binding is incomplete")
    _sha256_text(treatment_map[model], label="agent treatment hash")


def _validate_leaderboard(
    leaderboard: dict[str, Any], *, model: str, manifest: dict[str, Any]
) -> dict[str, Any]:
    if not _eligibility_is_green(leaderboard):
        raise FormalShardMergeError("source leaderboard is not leaderboard eligible")
    if leaderboard.get("scoring_version") != manifest.get("scoring_version"):
        raise FormalShardMergeError("source leaderboard scoring_version mismatch")
    if leaderboard.get("primary_leaderboard_formula_version") != (
        batch.PRIMARY_LEADERBOARD_FORMULA_VERSION
    ):
        raise FormalShardMergeError("source primary leaderboard formula mismatch")
    if leaderboard.get("primary_inference_version") != batch.PRIMARY_INFERENCE_VERSION:
        raise FormalShardMergeError("source primary inference version mismatch")
    primary = leaderboard.get("primary_leaderboard")
    if not isinstance(primary, list) or len(primary) != 1:
        raise FormalShardMergeError(
            "source leaderboard must contain one primary model row"
        )
    row = primary[0]
    if not isinstance(row, dict) or row.get("model") != model:
        raise FormalShardMergeError("source leaderboard model scope mismatch")
    score = row.get("primary_leaderboard_score")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
    ):
        raise FormalShardMergeError("source primary leaderboard score is invalid")
    return row


def _load_episode_rows(
    shard_dir: Path,
    *,
    manifest: dict[str, Any],
    model: str,
) -> tuple[list[dict[str, Any]], Path]:
    episodes_path = shard_dir / "episodes.jsonl"
    if not episodes_path.is_file():
        raise FormalShardMergeError(f"source episodes.jsonl is missing: {shard_dir}")
    raw_rows: list[dict[str, Any]] = []
    try:
        for line_number, raw_line in enumerate(
            episodes_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise FormalShardMergeError(
                    f"episodes.jsonl line {line_number} is not an object"
                )
            raw_rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalShardMergeError(
            f"episodes.jsonl contains invalid JSON: {episodes_path}"
        ) from exc

    rows = batch.effective_episode_rows_for_analysis(raw_rows)
    treatment_reasons = batch._formal_treatment_binding_reasons(manifest, rows)
    if treatment_reasons:
        raise FormalShardMergeError(
            "source agent treatment binding failed: "
            + ", ".join(treatment_reasons)
        )
    selected = batch._select_rows_for_treatment(rows, manifest)
    if len(selected) != len(rows):
        raise FormalShardMergeError("source contains rows outside its treatment")

    pass_k = int(manifest["pass_k"])
    expected_grid = {
        (str(slug), int(seed), f"pass-{pass_index}")
        for slug, seed in manifest["scenario_seed_pairs"]
        for pass_index in range(pass_k)
    }
    actual_grid: set[tuple[str, int, str]] = set()
    for row in selected:
        if row.get("model") != model or row.get("pass_k") != pass_k:
            raise FormalShardMergeError("source episode grid model/pass_k mismatch")
        cell = (
            str(row.get("scenario_slug") or ""),
            int(row.get("seed", -1)),
            str(row.get("pass_id") or ""),
        )
        if cell in actual_grid:
            raise FormalShardMergeError("source episode grid contains duplicates")
        actual_grid.add(cell)
        eligible, reasons = batch._formal_row_eligibility(
            row,
            required_suite_hash=str(manifest["suite_manifest_sha256"]),
            required_implementation_tree_sha256=str(
                manifest["implementation_tree_sha256"]
            ),
            required_interaction_mode=FORMAL_INTERACTION_MODE,
            verify_artifact_bytes=True,
        )
        if not eligible:
            raise FormalShardMergeError(
                "source episode is not formally eligible: " + ", ".join(reasons)
            )
    if actual_grid != expected_grid:
        raise FormalShardMergeError("source episode grid is incomplete or has extras")
    if len(selected) != manifest.get("n_episodes_total"):
        raise FormalShardMergeError("source episode count differs from manifest")
    if manifest.get("n_episodes_ok") != len(selected):
        raise FormalShardMergeError("source successful episode count differs")
    if manifest.get("n_episodes_error") != 0:
        raise FormalShardMergeError("source shard contains episode errors")
    return selected, episodes_path


def _recompute_primary_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        return batch._primary_leaderboard_payload(rows)
    except (PrimaryLeaderboardContractError, ValueError) as exc:
        raise FormalShardMergeError(
            f"merged primary inference contract failed: {exc}"
        ) from exc


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _ensure_output_available(output_dir: Path) -> None:
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise FormalShardMergeError(
            "output directory must be absent or an empty directory"
        )


def _comparison_mismatch(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> str | None:
    labels = {
        "suite_manifest_sha256": "suite hash",
        "implementation_tree_sha256": "implementation tree",
        "formal_run_contract": "formal run contract",
        "formal_run_contract_sha256": "formal run contract",
        "prompt_mode": "prompt_mode",
        "interaction_mode": "interaction_mode",
        "seed_mode": "seed_mode",
        "scenario_seed_pairs": "scenario seed scope",
        "scoring_version": "scoring_version",
        "primary_leaderboard_formula_version": "primary formula",
        "primary_inference_version": "primary inference",
        "evaluation_protocol_version": "evaluation protocol",
        "temperature": "temperature",
        "pass_k": "pass_k",
        "save_trajectories": "trajectory policy",
        "scheduler_mode": "scheduler_mode",
        "persistent_agent_profile": "persistent agent profile",
    }
    for field, label in labels.items():
        if reference.get(field) != candidate.get(field):
            return label
    return None


def merge_formal_shards(
    shard_dirs: list[Path], *, output_dir: Path
) -> dict[str, Any]:
    """Validate and merge formal single-model shards into one leaderboard."""

    if len(shard_dirs) < 2:
        raise FormalShardMergeError("at least two formal shard directories are required")
    output_dir = output_dir.resolve()
    _ensure_output_available(output_dir)

    source_records: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    models_seen: set[str] = set()
    family_projection: dict[str, Any] | None = None
    reference_manifest: dict[str, Any] | None = None
    input_primary_by_model: dict[str, dict[str, Any]] = {}
    concurrency_by_model: dict[str, dict[str, int]] = {}
    expected_total = 0

    for raw_shard_dir in shard_dirs:
        shard_dir = raw_shard_dir.resolve()
        if not shard_dir.is_dir():
            raise FormalShardMergeError(f"formal shard directory is missing: {shard_dir}")
        manifest_path = shard_dir / "RUN_MANIFEST.json"
        leaderboard_path = shard_dir / "leaderboard.json"
        manifest = _load_json_object(manifest_path, label="RUN_MANIFEST.json")
        leaderboard = _load_json_object(
            leaderboard_path, label="leaderboard.json"
        )
        _validate_current_formal_contract(manifest)
        expected_total += int(manifest["expected_total"])
        model = str(manifest["models"][0])
        if model in models_seen:
            raise FormalShardMergeError(f"duplicate model across shards: {model}")
        models_seen.add(model)
        concurrency_by_model[model] = _concurrency(manifest)
        input_primary_by_model[model] = _validate_leaderboard(
            leaderboard, model=model, manifest=manifest
        )
        projection = _formal_treatment_family_projection(manifest, leaderboard)
        if family_projection is None:
            family_projection = projection
            reference_manifest = manifest
        else:
            mismatch = _comparison_mismatch(family_projection, projection)
            if mismatch is not None:
                raise FormalShardMergeError(
                    f"formal shard {mismatch} mismatch: {shard_dir}"
                )
        rows, episodes_path = _load_episode_rows(
            shard_dir, manifest=manifest, model=model
        )
        all_rows.extend(rows)
        source_records.append(
            {
                "model": model,
                "source_shard_path": Path(
                    os.path.relpath(shard_dir, start=output_dir.parent)
                ).as_posix(),
                "run_manifest_sha256": _file_sha256(manifest_path),
                "leaderboard_sha256": _file_sha256(leaderboard_path),
                "episodes_sha256": _file_sha256(episodes_path),
                "agent_treatment_sha256": manifest[
                    "agent_treatment_sha256_by_model"
                ][model],
            }
        )

    assert family_projection is not None
    assert reference_manifest is not None
    source_records.sort(key=lambda record: str(record["model"]))
    all_rows.sort(
        key=lambda row: (
            str(row.get("model") or ""),
            str(row.get("scenario_slug") or ""),
            int(row.get("seed", -1)),
            str(row.get("pass_id") or ""),
        )
    )
    primary = _recompute_primary_payload(all_rows)
    recomputed_rows = primary.get("leaderboard")
    if not isinstance(recomputed_rows, list):
        raise FormalShardMergeError("recomputed primary leaderboard is invalid")
    recomputed_by_model = {
        str(row.get("model")): row
        for row in recomputed_rows
        if isinstance(row, dict) and row.get("model")
    }
    if set(recomputed_by_model) != models_seen:
        raise FormalShardMergeError("recomputed primary leaderboard model mismatch")
    for model in sorted(models_seen):
        prior_score = float(
            input_primary_by_model[model]["primary_leaderboard_score"]
        )
        merged_score = float(
            recomputed_by_model[model]["primary_leaderboard_score"]
        )
        if not math.isclose(prior_score, merged_score, rel_tol=0.0, abs_tol=1e-9):
            raise FormalShardMergeError(
                f"recomputed primary score differs for model {model}"
            )
    if primary.get("scoring_version") != family_projection["scoring_version"]:
        raise FormalShardMergeError("recomputed primary scoring_version mismatch")
    if primary.get("primary_leaderboard_formula_version") != family_projection[
        "primary_leaderboard_formula_version"
    ]:
        raise FormalShardMergeError("recomputed primary formula mismatch")
    if primary.get("primary_inference_version") != family_projection[
        "primary_inference_version"
    ]:
        raise FormalShardMergeError("recomputed primary inference mismatch")

    family_hash = _canonical_json_sha256(family_projection)
    models = sorted(models_seen)
    concurrency_by_model = {
        model: concurrency_by_model[model] for model in models
    }
    created_at = datetime.now(UTC).isoformat()
    common_output = {
        "formal_treatment_family_sha256": family_hash,
        "suite_manifest_sha256": family_projection["suite_manifest_sha256"],
        "implementation_tree_sha256": family_projection[
            "implementation_tree_sha256"
        ],
        "formal_run_contract_version": FORMAL_CONTRACT_VERSION,
        "models": models,
        "concurrency_by_model": concurrency_by_model,
        "n_shards": len(source_records),
        "scoring_version": primary["scoring_version"],
        "primary_leaderboard_formula_version": primary[
            "primary_leaderboard_formula_version"
        ],
        "primary_inference_version": primary["primary_inference_version"],
        "primary_leaderboard": recomputed_rows,
        "primary_pairwise": primary.get("primary_pairwise", []),
    }
    merged_leaderboard = {
        "schema_version": MERGED_LEADERBOARD_SCHEMA_VERSION,
        "created_at_utc": created_at,
        "leaderboard_eligible": True,
        "leaderboard_eligibility": {"eligible": True, "blockers": []},
        **common_output,
        "primary_inference_n_physical_clusters": primary.get(
            "primary_inference_n_physical_clusters"
        ),
        "n_input_samples": primary.get("n_input_samples", len(all_rows)),
        "source_shards": source_records,
    }
    merged_manifest = {
        "schema_version": MERGE_SCHEMA_VERSION,
        "created_at_utc": created_at,
        "formal_run": True,
        "batch_state": batch.BATCH_STATE_FINAL,
        "leaderboard_eligible": True,
        "leaderboard_eligibility": {"eligible": True, "blockers": []},
        **common_output,
        "formal_run_contract": reference_manifest["formal_run_contract"],
        "formal_treatment_family": family_projection,
        "prompt_mode": family_projection["prompt_mode"],
        "interaction_mode": family_projection["interaction_mode"],
        "seed_mode": family_projection["seed_mode"],
        "scenario_seed_pairs": family_projection["scenario_seed_pairs"],
        "n_scenarios": reference_manifest["n_scenarios"],
        "pass_k": family_projection["pass_k"],
        "scheduler_mode": family_projection["scheduler_mode"],
        "persistent_agent_profile": family_projection[
            "persistent_agent_profile"
        ],
        "expected_total": expected_total,
        "n_episodes_total": len(all_rows),
        "n_episodes_ok": len(all_rows),
        "n_episodes_error": 0,
        "source_shards": source_records,
        "artifacts": {
            "leaderboard_json": "leaderboard.json",
            "run_manifest_json": "RUN_MANIFEST.json",
        },
    }

    _ensure_output_available(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_path = output_dir / "leaderboard.json"
    _atomic_write_json(leaderboard_path, merged_leaderboard)
    merged_manifest["artifacts"]["leaderboard_json_sha256"] = _file_sha256(
        leaderboard_path
    )
    _atomic_write_json(output_dir / "RUN_MANIFEST.json", merged_manifest)
    return merged_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge compatible finalized logical-persistent formal shards."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("shards", nargs="+", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = merge_formal_shards(args.shards, output_dir=args.output_dir)
    except FormalShardMergeError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "models": manifest["models"],
                "formal_treatment_family_sha256": manifest[
                    "formal_treatment_family_sha256"
                ],
                "leaderboard_eligible": True,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
