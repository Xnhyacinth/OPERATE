from types import SimpleNamespace
from scripts import batch_llm_eval as batch
from tests.test_batch_lite_execution import _job


def test_provenance_resume_skips_completed_cells_from_previous_code():
    job = {**_job("case"), "implementation_policy": "provenance"}
    old = {**job, "status": "ok", "implementation_tree_sha256": "older-code"}
    assert (
        batch._filter_pending_jobs([job], [old], resume_policy="retry-infrastructure")
        == []
    )
    strict = {**job, "implementation_policy": "strict"}
    assert batch._filter_pending_jobs(
        [strict], [old], resume_policy="retry-infrastructure"
    ) == [strict]


def test_provenance_worker_records_actual_code_without_invalidating_result(monkeypatch):
    job = {**_job("case"), "implementation_policy": "provenance", "llm_config": {}}
    monkeypatch.setattr(
        batch, "_llm_config_from_dict", lambda _: SimpleNamespace(temperature=0)
    )
    monkeypatch.setattr(
        batch,
        "implementation_identity",
        lambda _: {"implementation_tree_sha256": "current-code"},
    )
    monkeypatch.setattr(batch, "_run_one_safe", lambda _: {"status": "ok"})
    row = batch._run_llm_episode_job(job)
    assert row["status"] == "ok"
    assert row["implementation_tree_sha256"] == "current-code"
    assert row["implementation_policy"] == "provenance"


def test_provenance_cli_continues_after_code_update_without_git(monkeypatch, tmp_path):
    import sys
    import json
    from tests.test_batch_lite_execution import _bind_saved_artifacts

    monkeypatch.setattr(batch, "_resolve_patterns", lambda *_: ["test/a", "test/b"])
    monkeypatch.setattr(batch, "_expand_scenarios", lambda x: x)
    monkeypatch.setattr(
        batch,
        "load_scenario_yaml",
        lambda slug: {
            "seed": 42,
            "scenario_signature": slug,
            "horizon_ticks": 1,
            "backend_kind": "mock",
        },
    )
    monkeypatch.setattr(
        batch, "_suite_eligibility_binding", lambda _: {"suite_blocked": False}
    )
    monkeypatch.setattr(batch, "_load_zhsrc_exports", lambda: {})
    monkeypatch.setattr(batch, "_load_named_zshrc_export", lambda _: None)
    monkeypatch.setattr(
        batch,
        "_git_metadata",
        lambda: {
            "git_metadata_available": False,
            "git_commit": None,
            "git_dirty": None,
        },
    )
    trees = ["first-code"]
    monkeypatch.setattr(
        batch,
        "implementation_identity",
        lambda _: {"implementation_tree_sha256": trees[0]},
    )
    monkeypatch.setenv("FIXTURE_KEY", "offline")
    dispatched = []

    def execute(jobs, path, mode, workers):
        for job in jobs:
            dispatched.append(job["scenario_slug"])
            row = batch._apply_llm_job_metadata(
                job, {"status": "ok", "implementation_tree_sha256": trees[0]}
            )
            batch._append_jsonl_atomic(path, _bind_saved_artifacts(job, row))

    monkeypatch.setattr(batch, "_run_global_jobs", execute)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch",
            "--output-dir",
            str(tmp_path),
            "--models",
            "fixture",
            "--api-key-env",
            "FIXTURE_KEY",
            "--implementation-policy",
            "provenance",
            "--max-jobs",
            "1",
            "--max-workers",
            "1",
            "--no-finalize",
            "--save-trajectories",
            "--interaction-mode",
            "logical_stateless",
        ],
    )
    assert batch.main() == 0
    trees[0] = "second-code"
    assert batch.main() == 0
    assert batch.main() == 0
    assert dispatched == ["test/a", "test/b"]
    rows = [
        json.loads(line)
        for line in (tmp_path / "episodes.jsonl").read_text().splitlines()
    ]
    assert {r["implementation_tree_sha256"] for r in rows} == {
        "first-code",
        "second-code",
    }
