from pathlib import Path
from types import SimpleNamespace

from runner.recovery_audit import build_recovery_audit, validate_recovery_audit
from scripts import batch_llm_eval as batch
from tests.test_recovery_audit import _row


def test_three_checkpoint_attempts_keep_archived_audit_bindings_and_counts(
    tmp_path, monkeypatch
):
    directory = tmp_path / "cell"
    first = _row(directory, "first", ["success", "failed"], 0)
    first["temperature"] = 0.0
    first["recovery_audit"] = build_recovery_audit([], None, first, tmp_path, "hy3-ioa")
    job = {
        key: first[key]
        for key in (
            "scenario_slug",
            "model",
            "seed",
            "pass_id",
            "scenario_signature",
            "temperature",
            "implementation_tree_sha256",
            "agent_treatment_sha256",
            "run_semantics_fingerprint",
            "suite_manifest_sha256",
            "suite_eligibility_sha256",
        )
    }
    job.update(
        trajectory_dir=str(directory),
        batch_output_dir=str(tmp_path),
        episode_checkpoint=True,
        llm_config={},
    )
    monkeypatch.setattr(
        batch, "_llm_config_from_dict", lambda _: SimpleNamespace(temperature=0.0)
    )
    monkeypatch.setattr(
        batch,
        "implementation_identity",
        lambda _: {"implementation_tree_sha256": "bound"},
    )
    outcomes = [(["success", "success", "failed"], 1), (["success"] * 4, 2)]

    def execute(args):
        statuses, reused = outcomes.pop(0)
        assert args[-1]["checkpoint_identity"]["agent_treatment_sha256"] == "bound"
        row = _row(directory, "replaced-by-worker-proof", statuses, reused)
        row["trajectory_summary"] = {"trajectory_path": str(directory / "cell")}
        return row

    monkeypatch.setattr(batch, "_run_one_safe", execute)
    previous = first
    for _ in range(2):
        pending = batch._filter_pending_jobs(
            [job], [previous], batch_root=tmp_path, resume_policy="retry-infrastructure"
        )
        assert len(pending) == 1
        previous = batch._run_llm_episode_job(pending[0])
        assert previous["recovery_audit"]["closed"]
        assert previous["recovery_audit"]["eligible"]
        assert validate_recovery_audit(previous, tmp_path, directory) == []
    audit = previous["recovery_audit"]
    assert audit["totals"]["request_attempts"] == 6
    assert audit["totals"]["failed_requests"] == 2
    assert audit["logical_provider_request_count"] == 4
    paths = [
        Path(attempt["provider_audit_artifact"]["path"])
        for attempt in audit["attempts"]
    ]
    assert len({path.parent for path in paths}) == 3
    assert all(path.is_file() for path in paths)
    assert (
        previous["provider_request_accounting_scope"] == "retained_logical_trajectory"
    )
    assert (
        "recovery_audit_unclosed"
        not in batch._formal_row_eligibility(previous, batch_root=tmp_path)[1]
    )
    paths[0].write_bytes(b"tampered\n")
    assert validate_recovery_audit(previous, tmp_path, directory)
    assert (
        "recovery_audit_unclosed"
        in batch._formal_row_eligibility(previous, batch_root=tmp_path)[1]
    )


def test_replayed_episode_without_history_cannot_bypass_formal_audit(tmp_path):
    row = _row(tmp_path / "cell", "orphan", ["success"] * 2, 1)
    row["checkpoint_progress"]["replayed_boundaries"] = 1
    assert (
        "recovery_audit_unclosed"
        in batch._formal_row_eligibility(row, batch_root=tmp_path)[1]
    )


def test_recovery_artifact_paths_survive_batch_relocation(tmp_path):
    import shutil

    root = tmp_path / "original"
    old = _row(root / "cell.stale-old", "old", ["success", "failed"], 0)
    current = _row(root / "cell", "new", ["success", "success"], 1)
    current["trajectory_summary"] = {"trajectory_path": str(root / "cell" / "cell")}
    current["recovery_audit"] = build_recovery_audit([], old, current, root, "hy3-ioa")
    portable = batch._portable_formal_result_paths([current], batch_root=root)[0]
    assert all(
        not Path(entry["provider_audit_artifact"]["path"]).is_absolute()
        for entry in portable["recovery_audit"]["attempts"]
    )
    assert not Path(portable["provider_audit_artifact"]["path"]).is_absolute()
    moved = tmp_path / "moved"
    shutil.copytree(root, moved)
    shutil.rmtree(root)
    assert validate_recovery_audit(portable, moved, moved / "cell") == []


def test_absolute_recovery_bindings_validate_in_combined_inference(tmp_path):
    old = _row(tmp_path / "cell.stale-old", "old", ["success", "failed"], 0)
    row = _row(tmp_path / "cell", "new", ["success", "success"], 1)
    row["recovery_audit"] = build_recovery_audit([], old, row, tmp_path, "hy3-ioa")
    assert validate_recovery_audit(row, None, tmp_path / "cell") == []
