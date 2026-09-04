"""Tests for trajectory logging summaries."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.trajectory_logger import EpisodeHeader, TrajectoryLogger
from run import _public_agent_config


def test_long_episode_id_keeps_all_sidecar_basenames_portable(
    tmp_path: Path,
) -> None:
    logger = TrajectoryLogger(
        episode_id="llm_agent_" + "inventory_replenishment_" * 16 + "s5201",
        output_dir=tmp_path,
    )

    evidence_path = logger.write_evidence(
        [{"evidence_id": "ev-long", "tick": 0}]
    )
    semantic = logger.write_semantic_ledger(
        [{"role": "system", "content": "mission"}]
    )
    provider = logger.write_provider_audit(
        [{"record_kind": "provider_request", "sequence": 1}]
    )
    logger.log_step(0, {"tick": 0}, {"actions": []}, 0.0)
    logger.finalize(final_score=0.0)

    name_max = os.pathconf(tmp_path, "PC_NAME_MAX")
    artifacts = list(tmp_path.iterdir())
    assert evidence_path.is_file()
    assert Path(semantic["path"]).is_file()
    assert Path(provider["path"]).is_file()
    assert artifacts
    assert all(len(path.name.encode("utf-8")) <= name_max for path in artifacts)
    assert not any(path.name.endswith(".tmp") for path in artifacts)


def test_atomic_sidecar_temp_name_does_not_extend_target_basename(
    tmp_path: Path,
) -> None:
    # The final evidence basename fits on macOS (NAME_MAX=255), while the old
    # ``.<target>.<uuid>.tmp`` temporary basename did not.
    logger = TrajectoryLogger(episode_id="e" * 225, output_dir=tmp_path)

    path = logger.write_evidence([{"evidence_id": "ev-boundary"}])

    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8"))["evidence_id"] == (
        "ev-boundary"
    )
    assert not any(item.name.endswith(".tmp") for item in tmp_path.iterdir())


def test_long_episode_ids_keep_distinct_deterministic_artifact_names(
    tmp_path: Path,
) -> None:
    shared_prefix = "long_episode_" * 32
    first = TrajectoryLogger(
        episode_id=f"{shared_prefix}alpha", output_dir=tmp_path
    )
    second = TrajectoryLogger(
        episode_id=f"{shared_prefix}beta", output_dir=tmp_path
    )
    repeated = TrajectoryLogger(
        episode_id=f"{shared_prefix}alpha", output_dir=tmp_path
    )

    assert first.episode_id != second.episode_id
    assert first.episode_id == repeated.episode_id
    assert len(first.episode_id.encode("utf-8")) <= 180


def test_driving_header_serializes_seconds_clock() -> None:
    header = EpisodeHeader(
        episode_id="driving",
        scenario_id="cut_in",
        scenario_signature="sig",
        domain="autonomous_driving",
        family="cut_in_prevention_and_emergency",
        difficulty_mode="time_pressure",
        difficulty_level="basic",
        backend_kind="sumo_ego",
        horizon_ticks=24,
        tick_minutes=None,
        agent_name="wait_only",
        agent_config=None,
        seed=42,
        start_time_utc="2026-08-13T00:00:00+00:00",
        tick_seconds=5.0,
        clock_contract={"schema_version": "driving_clock_v1"},
    )

    payload = header.to_dict()

    assert payload["tick_minutes"] is None
    assert payload["tick_seconds"] == 5.0
    assert payload["clock_contract"] == {"schema_version": "driving_clock_v1"}


def test_header_serialization_redacts_agent_extra_header_values(tmp_path: Path) -> None:
    from baselines.llm_agent import LLMConfig

    logger = TrajectoryLogger(episode_id="secret_header", output_dir=tmp_path)
    logger.set_header(
        EpisodeHeader(
            episode_id="secret_header",
            scenario_id="s1",
            scenario_signature="abc",
            domain="power_grid",
            family="daily_ops_24h",
            difficulty_mode="time_pressure",
            difficulty_level="basic",
            backend_kind="pglib_uc_synthetic",
            horizon_ticks=1,
            tick_minutes=60,
            agent_name="llm_agent",
            agent_config=_public_agent_config(
                {
                    "config": LLMConfig(
                        provider="azure",
                        model="gpt-test",
                        base_url=(
                            "https://user:secret-value@example.com/v1"
                            "?api_key=secret-value#fragment"
                        ),
                        responses_base_url=(
                            "https://example.com/responses?token=secret-value"
                        ),
                        extra_headers={"Ocp-Apim-Subscription-Key": "secret-value"},
                    )
                }
            ),
            seed=42,
            start_time_utc="2026-05-22T00:00:00+00:00",
        )
    )
    logger.log_step(0, {}, {"dominant_action": "wait", "actions": []}, 0.0)
    logger.finalize(final_score=0.0)

    raw = (tmp_path / "secret_header.header.json").read_text(encoding="utf-8")
    assert "secret-value" not in raw
    payload = json.loads(raw)
    assert payload["agent_config"]["config"]["extra_headers"] == {
        "Ocp-Apim-Subscription-Key": "[redacted]"
    }
    assert payload["agent_config"]["config"]["base_url"] == (
        "https://example.com/v1"
    )
    assert payload["agent_config"]["config"]["responses_base_url"] == (
        "https://example.com/responses"
    )


def test_finalize_counts_entries_flushed_before_close(tmp_path: Path) -> None:
    logger = TrajectoryLogger(
        episode_id="flush_case",
        output_dir=tmp_path,
        buffer_size=1,
    )
    logger.set_header(
        EpisodeHeader(
            episode_id="flush_case",
            scenario_id="s1",
            scenario_signature="abc",
            domain="power_grid",
            family="daily_ops_24h",
            difficulty_mode="time_pressure",
            difficulty_level="basic",
            backend_kind="pglib_uc_synthetic",
            horizon_ticks=2,
            tick_minutes=60,
            agent_name="wait_only",
            agent_config=None,
            seed=42,
            start_time_utc="2026-05-22T00:00:00+00:00",
        )
    )

    logger.log_step(0, {}, {"dominant_action": "wait", "actions": []}, 1.0)
    logger.log_step(1, {}, {"dominant_action": "wait", "actions": []}, 1.0)
    summary = logger.finalize(final_score=10.0)

    assert summary["total_ticks"] == 2
    assert logger.header is not None
    assert logger.header.total_ticks == 2


def test_default_logger_checkpoints_each_tick(tmp_path: Path) -> None:
    logger = TrajectoryLogger(
        episode_id="tick_checkpoint_case",
        output_dir=tmp_path,
    )

    logger.log_step(
        0,
        {"tick": 1},
        {"dominant_action": "wait", "actions": []},
        0.0,
    )

    checkpoint = tmp_path / "tick_checkpoint_case.trajectory.jsonl"
    assert checkpoint.is_file()
    assert json.loads(checkpoint.read_text(encoding="utf-8"))["tick"] == 0


def test_second_logger_cannot_append_to_existing_episode(tmp_path: Path) -> None:
    first = TrajectoryLogger(episode_id="collision", output_dir=tmp_path)
    first.log_step(
        0,
        {"tick": 1},
        {"dominant_action": "wait", "actions": []},
        1.0,
    )
    original = (tmp_path / "collision.trajectory.jsonl").read_text(
        encoding="utf-8"
    )

    second = TrajectoryLogger(episode_id="collision", output_dir=tmp_path)
    try:
        second.log_step(
            0,
            {"tick": 999},
            {"dominant_action": "wait", "actions": []},
            999.0,
        )
    except FileExistsError as exc:
        assert "collision" in str(exc)
    else:  # pragma: no cover - explicit assertion message
        raise AssertionError("a second logger must not reuse an episode id")

    assert (tmp_path / "collision.trajectory.jsonl").read_text(
        encoding="utf-8"
    ) == original


def test_write_evidence_persists_resolvable_items(tmp_path: Path) -> None:
    logger = TrajectoryLogger(episode_id="evidence_case", output_dir=tmp_path)

    path = logger.write_evidence(
        [
            {
                "evidence_id": "ev_1",
                "tick": 1,
                "kind": "tool_call",
                "payload": {"ok": True},
                "source": "tool",
            }
        ]
    )

    assert path == tmp_path / "evidence_case.evidence.jsonl"
    assert json.loads(path.read_text(encoding="utf-8"))["evidence_id"] == "ev_1"


def test_summary_counts_within_tick_investigation_requests() -> None:
    logger = TrajectoryLogger(episode_id="two_stage")
    logger.log_step(
        0,
        {},
        {
            "dominant_action": "dispatch",
            "actions": [{"name": "dispatch", "args": {}}],
        },
        0.0,
        tool_results=[
            {"name": "inspect_queue", "ok": True, "state_changing": False},
            {"name": "dispatch", "ok": True, "state_changing": True},
        ],
        info={
            "extra": {
                "within_tick_investigation": {
                    "investigation_action": {
                        "actions": [{"name": "inspect_queue", "args": {}}]
                    }
                }
            }
        },
    )

    summary = logger.finalize()

    assert summary["total_tool_calls"] == 2
    assert summary["state_changing_tool_calls"] == 1
    assert summary["state_changing_action_rate"] == 0.5


def test_finalize_persists_runner_trajectory_summary(tmp_path: Path) -> None:
    logger = TrajectoryLogger(
        episode_id="autonomy_sidecar",
        output_dir=tmp_path,
    )
    logger.write_evidence([{"evidence_id": "ev-1", "tick": 0}])
    logger.log_step(0, {"tick": 0}, {"actions": []}, 0.0)
    trajectory_summary = {
        "event_adaptive_autonomy": {
            "records": [{"kind": "autonomous_hold", "tick": 1}]
        },
        "terminal_integrity": {"release_ready": False},
    }
    logger.finalize(
        final_score=1.0,
        trajectory_summary=trajectory_summary,
    )

    payload = json.loads(
        (tmp_path / "autonomy_sidecar.summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["trajectory_summary"]["terminal_integrity"] == {
        "release_ready": False
    }
    trajectory = payload["trajectory_summary"]["trajectory_artifact"]
    evidence = payload["trajectory_summary"]["evidence_ledger_artifact"]
    for artifact in (trajectory, evidence):
        artifact_path = Path(artifact["path"])
        artifact_bytes = artifact_path.read_bytes()
        assert artifact["sha256"] == hashlib.sha256(artifact_bytes).hexdigest()
        assert artifact["byte_count"] == len(artifact_bytes)
        assert artifact["event_count"] == len(artifact_bytes.splitlines())
    assert trajectory["schema_version"] == "episode_trajectory_jsonl_v1"
    assert evidence["schema_version"] == "evidence_ledger_jsonl_v1"
    assert trajectory_summary["trajectory_artifact"] == trajectory
    assert trajectory_summary["evidence_ledger_artifact"] == evidence


def test_semantic_ledger_is_checksummed_and_bound_to_header(tmp_path: Path) -> None:
    logger = TrajectoryLogger(episode_id="ledger_case", output_dir=tmp_path)
    logger.set_header(
        EpisodeHeader(
            episode_id="ledger_case",
            scenario_id="s1",
            scenario_signature="abc",
            domain="power_grid",
            family="daily_ops_24h",
            difficulty_mode="time_pressure",
            difficulty_level="basic",
            backend_kind="pglib_uc_synthetic",
            horizon_ticks=2,
            tick_minutes=60,
            agent_name="llm_agent",
            agent_config=None,
            seed=42,
            start_time_utc="2026-08-23T00:00:00+00:00",
        )
    )
    artifact = logger.write_semantic_ledger(
        [
            {"role": "system", "content": "mission"},
            {"role": "user", "content": "typed alarm"},
        ]
    )
    logger.log_step(0, {"tick": 0}, {"actions": []}, 0.0)
    logger.finalize(final_score=1.0)

    path = tmp_path / "ledger_case.semantic_ledger.jsonl"
    assert artifact["path"] == str(path)
    assert artifact["event_count"] == 2
    assert len(artifact["sha256"]) == 64
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
    header = json.loads(
        (tmp_path / "ledger_case.header.json").read_text(encoding="utf-8")
    )
    assert header["semantic_ledger_sha256"] == artifact["sha256"]
    assert header["semantic_ledger_events"] == 2


def test_provider_audit_is_checksummed_and_bound_to_header(tmp_path: Path) -> None:
    logger = TrajectoryLogger(episode_id="provider_case", output_dir=tmp_path)
    logger.set_header(
        EpisodeHeader(
            episode_id="provider_case",
            scenario_id="s1",
            scenario_signature="abc",
            domain="traffic",
            family="corridor_control",
            difficulty_mode="time_pressure",
            difficulty_level="basic",
            backend_kind="sumo",
            horizon_ticks=1,
            tick_minutes=1,
            agent_name="llm_agent",
            agent_config=None,
            seed=42,
            start_time_utc="2026-08-23T00:00:00+00:00",
        )
    )

    artifact = logger.write_provider_audit(
        [
            {"record_kind": "provider_request", "sequence": 1},
            {
                "record_kind": "provider_response",
                "sequence": 2,
                "request_sequence": 1,
            },
        ]
    )
    logger.log_step(0, {"tick": 0}, {"actions": []}, 0.0)
    logger.finalize()

    header = json.loads(
        (tmp_path / "provider_case.header.json").read_text(encoding="utf-8")
    )
    assert artifact["schema_version"] == "provider_interaction_audit_v1"
    assert header["provider_audit_sha256"] == artifact["sha256"]
    assert header["provider_audit_events"] == 2
