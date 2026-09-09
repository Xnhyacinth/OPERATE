from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import pytest

from evaluation.scorer import SCORING_VERSION
from scripts import merge_formal_llm_shards as merge
from tests.test_batch_llm_eval import (
    _bind_formal_artifacts,
    _formally_eligible_protocol21_row,
)


def _fake_primary_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    leaderboard = [
        {
            "model": str(row["model"]),
            "primary_leaderboard_score": float(row["test_primary_score"]),
            "primary_task_completion_rate": 1.0,
            "n_samples": 1,
        }
        for row in rows
    ]
    leaderboard.sort(
        key=lambda row: (-row["primary_leaderboard_score"], row["model"])
    )
    return {
        "schema_version": "1.0",
        "scoring_version": SCORING_VERSION,
        "primary_leaderboard_formula_version": (
            "effective_source_backend_domain_macro_v1"
        ),
        "primary_inference_version": (
            "physical_cluster_hierarchical_bootstrap_randomization_v1"
        ),
        "primary_inference_n_physical_clusters": 1,
        "primary_pairwise": [],
        "n_input_samples": len(rows),
        "leaderboard": leaderboard,
    }


def _write_shard(
    root: Path,
    *,
    model: str,
    score: float,
    suite_hash: str = "a" * 64,
    tree_hash: str = "b" * 64,
    contract_version: str = "agentic_persistent.v1",
    prompt_mode: str = "strict",
    interaction_mode: str = "logical_persistent",
    seed_mode: str = "scenario",
    scoring_version: str = SCORING_VERSION,
    pass_k: int = 1,
    scheduler_mode: str = "global",
    workers: int = 4,
    history_messages: int = 64,
) -> Path:
    shard = root / model.replace("/", "_")
    shard.mkdir(parents=True)
    treatment_hash = (model.encode("utf-8").hex() + "0" * 64)[:64]
    row = _formally_eligible_protocol21_row()
    row.update(
        {
            "model": model,
            "agent_name": f"llm_agent/{model}",
            "interaction_mode": interaction_mode,
            "suite_manifest_sha256": suite_hash,
            "implementation_tree_sha256": tree_hash,
            "scenario_slug": "traffic/formal-case",
            "scenario_signature": "scenario-signature",
            "seed": 42,
            "pass_id": "pass-0",
            "pass_index": 0,
            "pass_k": pass_k,
            "test_primary_score": score,
        }
    )
    row["score"]["scoring_version"] = scoring_version
    llm_stats = row["trajectory_summary"]["llm"]
    llm_stats["provider_models"] = [model]
    llm_stats["provider_model_identity_records"] = [
        {
            "schema_version": "provider_model_identity_closure_v1",
            "request_sequence": 1,
            "requested_model": model,
            "observed_models": [model],
            "closure": "exact",
        }
    ]
    row = _bind_formal_artifacts(row, shard, key=model.replace("/", "_"))
    treatment_hash = str(row["agent_treatment_sha256"])
    episodes_path = shard / "episodes.jsonl"
    episodes_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    contract = {
        "contract_version": contract_version,
        "required_model_count_per_shard": 1,
        "minimum_pass_k": 1,
        "minimum_max_workers": 1,
        "maximum_max_workers": 32,
        "required_interaction_mode": "logical_persistent",
        "required_prompt_mode": "strict",
        "required_seed_mode": "scenario",
        "required_scheduler_mode": "global",
        "required_temperature": 0.0,
        "requires_explicit_model_capabilities": True,
        "agentic_profile": copy.deepcopy(merge.FORMAL_AGENTIC_PROFILE),
        "save_trajectories": True,
    }
    manifest = {
        "schema_version": "1.0",
        "formal_run": True,
        "leaderboard_eligible": True,
        "leaderboard_eligibility": {"eligible": True, "blockers": []},
        "batch_state": "final",
        "finalize_enabled": True,
        "git_metadata_available": True,
        "git_dirty": False,
        "models": [model],
        "n_scenarios": 1,
        "scenario_seed_pairs": [["traffic/formal-case", 42]],
        "expected_total": pass_k,
        "n_episodes_total": 1,
        "n_episodes_ok": 1,
        "n_episodes_error": 0,
        "suite_manifest_sha256": suite_hash,
        "implementation_tree_sha256": tree_hash,
        "implementation_tree_stable": True,
        "formal_runtime_binding_stable": True,
        "formal_run_contract": contract,
        "prompt_mode": prompt_mode,
        "interaction_mode": interaction_mode,
        "seed_mode": seed_mode,
        "scoring_version": scoring_version,
        "temperature": 0.0,
        "pass_k": pass_k,
        "scheduler_mode": scheduler_mode,
        "max_workers_requested": workers,
        "max_workers_effective": workers,
        "save_trajectories": True,
        "evaluation_protocol_version": "2.1",
        "harness": "direct_api",
        "within_tick_interaction": True,
        "max_tokens": 64_000,
        "persistent_history_max_messages": history_messages,
        "persistent_context_max_chars": 512_000,
        "persistent_memory_max_items": 128,
        "protocol_repair_max_tokens": 8192,
        "tool_choice": "auto",
        "stream_chat_completions": True,
        "token_count_method": "utf8_bytes_upper_bound",
        "token_count_version": "1",
        "provider_timeout_s": 300.0,
        "max_consecutive_provider_failures": 1,
        "provider_failure_policy": "abort",
        "base_url": f"https://provider.invalid/{model}",
        "model_context_window_tokens_by_model": {model: 192_000},
        "model_max_output_tokens_by_model": {model: 64_000},
        "agent_treatment_sha256_by_model": {model: treatment_hash},
        "artifacts": {"episodes_jsonl": str(episodes_path)},
    }
    leaderboard = {
        "schema_version": "1.0",
        "leaderboard_eligible": True,
        "leaderboard_eligibility": {"eligible": True, "blockers": []},
        "scoring_version": scoring_version,
        "primary_leaderboard_formula_version": (
            "effective_source_backend_domain_macro_v1"
        ),
        "primary_inference_version": (
            "physical_cluster_hierarchical_bootstrap_randomization_v1"
        ),
        "primary_leaderboard": [
            {
                "model": model,
                "primary_leaderboard_score": score,
                "primary_task_completion_rate": 1.0,
                "n_samples": 1,
            }
        ],
        "primary_pairwise": [],
    }
    (shard / "RUN_MANIFEST.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (shard / "leaderboard.json").write_text(
        json.dumps(leaderboard), encoding="utf-8"
    )
    return shard


def _mutate_json(path: Path, mutator: Any) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutator(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_merge_writes_stable_treatment_family_and_recomputed_leaderboard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left = _write_shard(
        tmp_path / "inputs", model="hy3-ioa", score=70.0, workers=32
    )
    right = _write_shard(
        tmp_path / "inputs",
        model="stealth/ox-alpha",
        score=80.0,
        workers=4,
    )
    _mutate_json(
        right / "RUN_MANIFEST.json",
        lambda manifest: (
            manifest["model_context_window_tokens_by_model"].update(
                {"stealth/ox-alpha": 1_000_000}
            ),
            manifest["model_max_output_tokens_by_model"].update(
                {"stealth/ox-alpha": 128_000}
            ),
            manifest.update(max_tokens=128_000),
        ),
    )
    monkeypatch.setattr(merge, "_recompute_primary_payload", _fake_primary_payload)
    write_order: list[str] = []
    original_atomic_write = merge._atomic_write_json

    def recording_write(path: Path, payload: dict[str, Any]) -> None:
        write_order.append(path.name)
        original_atomic_write(path, payload)

    monkeypatch.setattr(merge, "_atomic_write_json", recording_write)

    output = merge.merge_formal_shards(
        [right, left], output_dir=tmp_path / "merged"
    )

    manifest = json.loads(
        (tmp_path / "merged" / "RUN_MANIFEST.json").read_text(encoding="utf-8")
    )
    leaderboard = json.loads(
        (tmp_path / "merged" / "leaderboard.json").read_text(encoding="utf-8")
    )
    assert output == manifest
    assert write_order == ["leaderboard.json", "RUN_MANIFEST.json"]
    assert manifest["leaderboard_eligible"] is True
    assert manifest["models"] == ["hy3-ioa", "stealth/ox-alpha"]
    assert manifest["expected_total"] == 2
    assert leaderboard["primary_leaderboard"][0]["model"] == (
        "stealth/ox-alpha"
    )
    assert leaderboard["formal_treatment_family_sha256"] == manifest[
        "formal_treatment_family_sha256"
    ]
    assert manifest["artifacts"]["leaderboard_json_sha256"] == (
        merge._file_sha256(tmp_path / "merged" / "leaderboard.json")
    )
    projection_text = json.dumps(manifest["formal_treatment_family"], sort_keys=True)
    assert "hy3-ioa" not in projection_text
    assert "ox-alpha" not in projection_text
    assert "base_url" not in projection_text
    assert "model_context_window" not in projection_text
    assert "concurrency" not in projection_text
    assert manifest["concurrency_by_model"] == {
        "hy3-ioa": {
            "max_workers_effective": 32,
            "max_workers_requested": 32,
        },
        "stealth/ox-alpha": {
            "max_workers_effective": 4,
            "max_workers_requested": 4,
        },
    }
    assert leaderboard["concurrency_by_model"] == manifest[
        "concurrency_by_model"
    ]
    assert {record["source_shard_path"] for record in manifest["source_shards"]} == {
        Path(os.path.relpath(left, start=(tmp_path / "merged").parent)).as_posix(),
        Path(os.path.relpath(right, start=(tmp_path / "merged").parent)).as_posix(),
    }
    assert manifest["formal_treatment_family"]["persistent_agent_profile"][
        "protocol_repair_max_tokens"
    ] == 8192
    assert manifest["formal_treatment_family"]["persistent_agent_profile"][
        "tool_choice"
    ] == "auto"
    assert manifest["formal_treatment_family"]["persistent_agent_profile"][
        "stream_chat_completions"
    ] is True
    assert not list((tmp_path / "merged").glob("*.tmp"))


def test_family_hash_and_ranking_are_input_order_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left = _write_shard(tmp_path / "inputs", model="model-a", score=60.0)
    right = _write_shard(tmp_path / "inputs", model="model-b", score=50.0)
    monkeypatch.setattr(merge, "_recompute_primary_payload", _fake_primary_payload)

    first = merge.merge_formal_shards(
        [left, right], output_dir=tmp_path / "merged-first"
    )
    second = merge.merge_formal_shards(
        [right, left], output_dir=tmp_path / "merged-second"
    )

    assert first["formal_treatment_family_sha256"] == second[
        "formal_treatment_family_sha256"
    ]
    assert first["primary_leaderboard"] == second["primary_leaderboard"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda m: m.update(suite_manifest_sha256="c" * 64), "suite"),
        (lambda m: m.update(implementation_tree_sha256="d" * 64), "tree"),
        (
            lambda m: m["formal_run_contract"].update(contract_version="legacy"),
            "contract version",
        ),
        (lambda m: m.update(prompt_mode="debug"), "prompt_mode"),
        (lambda m: m.update(interaction_mode="logical_stateless"), "interaction_mode"),
        (lambda m: m.update(seed_mode="fixed"), "seed_mode"),
        (lambda m: m.update(scoring_version="0.11.0"), "scoring_version"),
        (
            lambda m: m.update(
                pass_k=2,
                expected_total=2,
                n_episodes_total=2,
                n_episodes_ok=2,
            ),
            "pass_k",
        ),
        (lambda m: m.update(scheduler_mode="per_model"), "scheduler_mode"),
        (lambda m: m.update(persistent_history_max_messages=40), "profile"),
        (lambda m: m.update(protocol_repair_max_tokens=2048), "profile"),
        (lambda m: m.update(tool_choice="required"), "profile"),
        (lambda m: m.update(stream_chat_completions=False), "profile"),
        (lambda m: m.update(finalize_enabled=False), "finalized explicitly"),
        (lambda m: m.update(git_dirty=True), "not clean"),
    ],
)
def test_merge_rejects_incompatible_formal_strata_without_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Any,
    message: str,
) -> None:
    left = _write_shard(tmp_path / "inputs", model="model-a", score=60.0)
    right = _write_shard(tmp_path / "inputs", model="model-b", score=50.0)
    _mutate_json(right / "RUN_MANIFEST.json", mutation)
    monkeypatch.setattr(merge, "_recompute_primary_payload", _fake_primary_payload)
    output_dir = tmp_path / "merged"

    with pytest.raises(merge.FormalShardMergeError, match=message):
        merge.merge_formal_shards([left, right], output_dir=output_dir)

    assert not output_dir.exists()


def test_merge_rejects_requested_effective_concurrency_drift_within_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left = _write_shard(
        tmp_path / "inputs", model="hy3-ioa", score=60.0, workers=32
    )
    right = _write_shard(
        tmp_path / "inputs", model="stealth/ox-alpha", score=50.0, workers=4
    )
    _mutate_json(
        right / "RUN_MANIFEST.json",
        lambda manifest: manifest.update(max_workers_effective=3),
    )
    monkeypatch.setattr(merge, "_recompute_primary_payload", _fake_primary_payload)
    output_dir = tmp_path / "merged"

    with pytest.raises(
        merge.FormalShardMergeError,
        match="concurrency requested/effective mismatch",
    ):
        merge.merge_formal_shards([left, right], output_dir=output_dir)

    assert not output_dir.exists()


def test_merge_rejects_duplicate_model_and_noneligible_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _write_shard(tmp_path / "first", model="model-a", score=60.0)
    duplicate = _write_shard(tmp_path / "duplicate", model="model-a", score=60.0)
    other = _write_shard(tmp_path / "other", model="model-b", score=50.0)
    monkeypatch.setattr(merge, "_recompute_primary_payload", _fake_primary_payload)

    with pytest.raises(merge.FormalShardMergeError, match="duplicate model"):
        merge.merge_formal_shards(
            [first, duplicate], output_dir=tmp_path / "duplicate-output"
        )

    _mutate_json(
        other / "leaderboard.json",
        lambda value: value.update(leaderboard_eligible=False),
    )
    with pytest.raises(merge.FormalShardMergeError, match="leaderboard eligible"):
        merge.merge_formal_shards(
            [first, other], output_dir=tmp_path / "ineligible-output"
        )


def test_merge_rejects_extra_pass_row_and_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left = _write_shard(tmp_path / "inputs", model="model-a", score=60.0)
    right = _write_shard(tmp_path / "inputs", model="model-b", score=50.0)
    rows = [
        json.loads(line)
        for line in (right / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    extra = copy.deepcopy(rows[0])
    extra["pass_id"] = "pass-9"
    with (right / "episodes.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(extra) + "\n")
    monkeypatch.setattr(merge, "_recompute_primary_payload", _fake_primary_payload)

    with pytest.raises(merge.FormalShardMergeError, match="episode grid"):
        merge.merge_formal_shards(
            [left, right], output_dir=tmp_path / "extra-pass-output"
        )

    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("owned", encoding="utf-8")
    with pytest.raises(merge.FormalShardMergeError, match="output directory"):
        merge.merge_formal_shards([left, right], output_dir=output_dir)
    assert (output_dir / "keep.txt").read_text(encoding="utf-8") == "owned"


@pytest.mark.parametrize("field,value", [
    ("episode_checkpoint", False),
    ("provider_retry_max_attempts", 7),
    ("provider_retry_max_elapsed_s", 900.0),
])
def test_merge_rejects_different_recovery_contracts(tmp_path, monkeypatch, field, value):
    left = _write_shard(tmp_path / "inputs", model="model-a", score=60.0)
    right = _write_shard(tmp_path / "inputs", model="model-b", score=50.0)
    recovery = {
        "episode_checkpoint": True, "provider_retry_max_attempts": 6,
        "provider_retry_max_elapsed_s": 1800.0,
    }
    for shard in (left, right):
        _mutate_json(shard / "RUN_MANIFEST.json", lambda m: m.update(recovery))
    _mutate_json(right / "RUN_MANIFEST.json", lambda m: m.update({field: value}))
    monkeypatch.setattr(merge, "_recompute_primary_payload", _fake_primary_payload)
    with pytest.raises(merge.FormalShardMergeError, match="profile"):
        merge.merge_formal_shards([left, right], output_dir=tmp_path / "merged")
    assert not (tmp_path / "merged").exists()


def test_legacy_family_projection_does_not_invent_recovery_settings(tmp_path):
    shard = _write_shard(tmp_path, model="model-a", score=60.0)
    manifest = json.loads((shard / "RUN_MANIFEST.json").read_text())
    leaderboard = json.loads((shard / "leaderboard.json").read_text())
    projection = merge._formal_treatment_family_projection(manifest, leaderboard)
    assert projection["persistent_agent_profile"] == {
        key: manifest[key] for key in merge._PROFILE_FIELDS if key != "max_tokens"
    }


def test_portable_shards_resolve_only_bound_artifact_paths_for_merge(tmp_path, monkeypatch):
    left = _write_shard(tmp_path / "inputs", model="model-a", score=60.0)
    right = _write_shard(tmp_path / "inputs", model="model-b", score=50.0)
    original_bytes = {}
    for shard in (left, right):
        path = shard / "episodes.jsonl"
        row = json.loads(path.read_text())
        row["model_output"] = {"path": "../not-an-artifact"}
        row["episode_log_path"] = "logs/cell.log"
        row["trajectory_summary"]["evidence_path"] = "evidence/cell.jsonl"
        row["checkpoint_progress"] = {"path": ".episode_checkpoints/cell.jsonl"}
        row = merge.batch._portable_formal_result_paths([row], batch_root=shard)[0]
        path.write_text(json.dumps(row) + "\n")
        original_bytes[path] = path.read_bytes()

    source_roots = []
    source_gate = merge.batch._formal_row_eligibility

    def gate(row, **kwargs):
        if kwargs.get("required_suite_hash"):
            source_roots.append(kwargs.get("batch_root"))
        return source_gate(row, **kwargs)

    def recompute(rows):
        for row in rows:
            summary = row["trajectory_summary"]
            assert Path(summary["trajectory_path"]).is_absolute()
            assert Path(summary["provider_audit_artifact"]["path"]).is_absolute()
            assert Path(row["episode_log_path"]).is_absolute()
            assert Path(summary["evidence_path"]).is_absolute()
            assert Path(row["checkpoint_progress"]["path"]).is_absolute()
            assert row["model_output"]["path"] == "../not-an-artifact"
            assert source_gate(row, verify_artifact_bytes=True)[0]
        return _fake_primary_payload(rows)

    monkeypatch.setattr(merge.batch, "_formal_row_eligibility", gate)
    monkeypatch.setattr(merge, "_recompute_primary_payload", recompute)
    merged = merge.merge_formal_shards([left, right], output_dir=tmp_path / "merged")
    assert merged["n_episodes_ok"] == 2
    assert source_roots == [left, right]
    assert all(path.read_bytes() == data for path, data in original_bytes.items())


def _portable_recovered_shard(root):
    from runner.recovery_audit import build_recovery_audit
    from tests.test_recovery_audit import _row

    shard = _write_shard(root, model="hy3-ioa", score=60.0)
    row = json.loads((shard / "episodes.jsonl").read_text())
    row["run_semantics_fingerprint"] = "fixture-recovery-contract"
    prefix = Path(row["trajectory_summary"]["trajectory_path"])
    directory = prefix.parent
    old = _row(directory.with_name(directory.name + ".stale-old"), "old", ["success", "failed"], 0)
    current = _row(directory, "new", ["success", "success"], 1)
    identity_fields = (
        "scenario_slug", "scenario_signature", "model", "seed", "pass_id",
        "agent_treatment_sha256", "implementation_tree_sha256", "run_semantics_fingerprint",
        "suite_manifest_sha256", "suite_eligibility_sha256",
    )
    old.update({key: row[key] for key in identity_fields})
    current.update({key: row[key] for key in identity_fields})
    artifact = current["provider_audit_artifact"]
    audit_path = Path(f"{prefix}.provider_audit.jsonl")
    Path(artifact["path"]).replace(audit_path)
    artifact["path"] = str(audit_path)
    for key in ("execution_attempt_id", "status", "error_type", "error_cause_type", "checkpoint_progress"):
        row[key] = current[key]
    row["checkpoint_progress"]["replayed_boundaries"] = 1
    row["provider_audit_artifact"] = copy.deepcopy(artifact)
    row["trajectory_summary"]["provider_audit_artifact"] = copy.deepcopy(artifact)
    records = [json.loads(line) for line in audit_path.read_text().splitlines()]
    responses = [record for record in records if record["record_kind"] == "provider_response"]
    row["trajectory_summary"]["llm"].update(
        provider_request_count=2, provider_response_count=2,
        provider_model_identity_request_count=2, provider_model_identity_closed_count=2,
        provider_model_identity_exact_count=2,
        provider_model_identity_records=[record["response"]["model_identity_closure"] for record in responses],
    )
    recovery = build_recovery_audit([], old, row, shard, "hy3-ioa")
    assert recovery["closed"] and recovery["eligible"], recovery["reasons"]
    row["recovery_audit"] = recovery
    row["trajectory_summary"]["recovery_audit"] = copy.deepcopy(recovery)
    portable = merge.batch._portable_formal_result_paths([row], batch_root=shard)[0]
    (shard / "episodes.jsonl").write_text(json.dumps(portable) + "\n")
    return shard, Path(old["provider_audit_artifact"]["path"])


def test_merge_portable_recovery_history_is_verified_again_during_inference(tmp_path, monkeypatch):
    left, _ = _portable_recovered_shard(tmp_path / "inputs")
    right = _write_shard(tmp_path / "inputs", model="model-b", score=50.0)
    saved_bytes = (left / "episodes.jsonl").read_bytes()

    def recompute(rows):
        for row in rows:
            eligible, reasons = merge.batch._formal_row_eligibility(row, verify_artifact_bytes=True)
            assert eligible, reasons
        return _fake_primary_payload(rows)

    monkeypatch.setattr(merge, "_recompute_primary_payload", recompute)
    output = merge.merge_formal_shards([left, right], output_dir=tmp_path / "merged")
    assert output["n_episodes_ok"] == 2
    assert (left / "episodes.jsonl").read_bytes() == saved_bytes


def test_merge_rejects_tampered_recovery_history_bytes(tmp_path, monkeypatch):
    left, historical_audit = _portable_recovered_shard(tmp_path / "inputs")
    right = _write_shard(tmp_path / "inputs", model="model-b", score=50.0)
    historical_audit.write_bytes(b"tampered\n")
    monkeypatch.setattr(merge, "_recompute_primary_payload", _fake_primary_payload)
    with pytest.raises(merge.FormalShardMergeError, match="recovery_audit"):
        merge.merge_formal_shards([left, right], output_dir=tmp_path / "merged")
    assert not (tmp_path / "merged").exists()


@pytest.mark.parametrize("location", ["artifact", "history"])
def test_merge_rejects_artifact_paths_escaping_the_source_shard(tmp_path, location):
    shard, _ = _portable_recovered_shard(tmp_path / "inputs")
    row = json.loads((shard / "episodes.jsonl").read_text())
    if location == "artifact":
        row["trajectory_summary"]["provider_audit_artifact"]["path"] = "../outside.provider_audit.jsonl"
    else:
        row["recovery_audit"]["attempts"][0]["provider_audit_artifact"]["path"] = "../outside.provider_audit.jsonl"
    (shard / "episodes.jsonl").write_text(json.dumps(row) + "\n")
    manifest = json.loads((shard / "RUN_MANIFEST.json").read_text())
    with pytest.raises(merge.FormalShardMergeError, match="escapes shard"):
        merge._load_episode_rows(shard, manifest=manifest, model="hy3-ioa")


def test_merge_rejects_recovery_audit_from_another_cell_inside_shard(tmp_path):
    shard, historical_audit = _portable_recovered_shard(tmp_path / "inputs")
    unrelated = shard / "unrelated-cell"
    unrelated.mkdir()
    moved = unrelated / historical_audit.name
    moved.write_bytes(historical_audit.read_bytes())
    row = json.loads((shard / "episodes.jsonl").read_text())
    for container in (row, row["trajectory_summary"]):
        container["recovery_audit"]["attempts"][0]["provider_audit_artifact"]["path"] = moved.relative_to(shard).as_posix()
    (shard / "episodes.jsonl").write_text(json.dumps(row) + "\n")
    manifest = json.loads((shard / "RUN_MANIFEST.json").read_text())
    with pytest.raises(merge.FormalShardMergeError, match="recovery_audit"):
        merge._load_episode_rows(shard, manifest=manifest, model="hy3-ioa")
