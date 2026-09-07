from __future__ import annotations

import json
from pathlib import Path

import pytest

from runner.episode import EVALUATION_IMPLEMENTATION_FINGERPRINT
from scripts import run_protocol21_core_pipeline as pipeline
from scripts.run_protocol21_core_pipeline import (
    STAGE_ORDER,
    build_pipeline_plan,
    main,
)


def _source_suite(tmp_path: Path, n: int = 304) -> Path:
    path = tmp_path / "source.json"
    scenarios = []
    for index in range(n):
        scenario_path = tmp_path / f"scenario-{index}.yaml"
        scenario_path.write_text(
            f"scenario_id: scenario-{index}\nseed: {index}\n",
            encoding="utf-8",
        )
        scenarios.append(
            {
                "scenario_id": f"scenario-{index}",
                "scenario_signature": f"sig-{index}",
                "path": str(scenario_path),
            }
        )
    path.write_text(
        json.dumps(
            {
                "status": "working_set",
                "leaderboard_eligible": False,
                "scenarios": scenarios,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_plan_binds_every_scenario_yaml_path_and_sha256(
    tmp_path: Path,
) -> None:
    source = _source_suite(tmp_path, n=2)

    plan = build_pipeline_plan(
        source_suite=source,
        release_dir=tmp_path / "release",
        workers=1,
        sample_timeout_seconds=900,
    )

    payload = json.loads(source.read_text(encoding="utf-8"))
    assert (
        plan["source_suite_sha256"]
        == pipeline.hashlib.sha256(source.read_bytes()).hexdigest()
    )
    assert plan["scenario_yaml_graph"] == [
        {
            "scenario_id": row["scenario_id"],
            "path": row["path"],
            "sha256": pipeline.hashlib.sha256(
                Path(row["path"]).read_bytes()
            ).hexdigest(),
        }
        for row in payload["scenarios"]
    ]


def test_plan_rejects_a_scenario_without_a_yaml_path(tmp_path: Path) -> None:
    source = _source_suite(tmp_path, n=1)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["scenarios"][0].pop("path")
    source.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="scenario YAML path missing"):
        build_pipeline_plan(
            source_suite=source,
            release_dir=tmp_path / "release",
            workers=1,
            sample_timeout_seconds=900,
        )


@pytest.mark.parametrize("drift_target", ["source_suite", "scenario_yaml"])
def test_pipeline_fails_closed_when_planned_source_graph_drifts_mid_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    drift_target: str,
) -> None:
    source = _source_suite(tmp_path, n=1)
    source_payload = json.loads(source.read_text(encoding="utf-8"))
    scenario_path = Path(source_payload["scenarios"][0]["path"])
    output = tmp_path / "preflight.json"
    manifest_path = tmp_path / "protocol2_v21_pipeline_manifest.json"
    graph = [
        {
            "scenario_id": "scenario-0",
            "path": str(scenario_path),
            "sha256": pipeline.hashlib.sha256(scenario_path.read_bytes()).hexdigest(),
        }
    ]
    plan = {
        "source_suite": str(source),
        "runtime_source_suite": str(source),
        "source_suite_sha256": pipeline.hashlib.sha256(source.read_bytes()).hexdigest(),
        "scenario_yaml_graph": graph,
        "implementation_tree_sha256": "tree-sha",
        "core_release_pipeline_sha256": "pipeline-sha",
        "runtime_binding": {},
        "pipeline_manifest": str(manifest_path),
        "stages": [
            {
                "name": "preflight",
                "argv": ["preflight"],
                "core_release_pipeline_sha256": "pipeline-sha",
                "expected_output": str(output),
            }
        ],
    }
    monkeypatch.setattr(pipeline, "build_pipeline_plan", lambda **_: plan)
    monkeypatch.setattr(pipeline, "_validate_stage_output", lambda _: None)
    monkeypatch.setattr(
        pipeline,
        "implementation_identity",
        lambda: {
            "implementation_tree_sha256": "tree-sha",
            "core_release_pipeline_sha256": "pipeline-sha",
        },
    )

    def run_stage(*_args, **_kwargs):
        if drift_target == "source_suite":
            source.write_text("{}\n", encoding="utf-8")
        else:
            scenario_path.write_text("scenario_id: changed\n", encoding="utf-8")
        output.write_text("{}\n", encoding="utf-8")
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(pipeline.subprocess, "run", run_stage)

    rc = main(["--execute", "--release-dir", str(tmp_path / "fresh-run")])

    assert rc == 3
    assert "planned source graph drift" in capsys.readouterr().err
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "blocked"
    assert manifest["formal_run_blockers"] == ["source_input_drift"]
    assert manifest["source_suite"] == str(source)
    assert manifest["scenario_yaml_graph"] == graph


def test_dry_run_plan_has_fixed_stage_order_and_v21_cache(
    tmp_path: Path,
) -> None:
    source = _source_suite(tmp_path)
    plan = build_pipeline_plan(
        source_suite=source,
        release_dir=tmp_path,
        workers=4,
        sample_timeout_seconds=900,
    )

    assert [stage["name"] for stage in plan["stages"]] == list(STAGE_ORDER)
    assert plan["execution_profile"] == "full_release"
    assert plan["stop_after"] == "readiness"
    assert all(
        EVALUATION_IMPLEMENTATION_FINGERPRINT in " ".join(stage["argv"])
        or stage["name"] not in {"behavioral", "complexity"}
        for stage in plan["stages"]
    )
    source_stage = next(
        stage for stage in plan["stages"] if stage["name"] == "source_grounded"
    )
    assert {
        "--behavioral",
        "--task-contracts",
        "--complexity",
        "--strategy-depth",
    }.issubset(source_stage["argv"])
    complexity_stage = next(
        stage for stage in plan["stages"] if stage["name"] == "complexity"
    )
    exact_calls_index = complexity_stage["argv"].index("--exact-max-calls")
    assert complexity_stage["argv"][exact_calls_index + 1] == "6"


def test_candidate_replay_stops_after_materialize_core(tmp_path: Path) -> None:
    source = _source_suite(tmp_path, n=1)

    plan = build_pipeline_plan(
        source_suite=source,
        release_dir=tmp_path,
        workers=1,
        sample_timeout_seconds=900,
        stop_after="materialize_core",
    )

    assert plan["execution_profile"] == "candidate_replay"
    assert plan["stop_after"] == "materialize_core"
    assert [stage["name"] for stage in plan["stages"]] == list(
        STAGE_ORDER[: STAGE_ORDER.index("materialize_core") + 1]
    )
    assert "release_coverage" not in {stage["name"] for stage in plan["stages"]}
    assert "readiness" not in {stage["name"] for stage in plan["stages"]}


def test_quality_core_profile_uses_bounded_diagnostic_complexity(
    tmp_path: Path,
) -> None:
    source = _source_suite(tmp_path, n=1)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["constraints"] = {"core_admission_profile": "quality_core_v2"}
    source.write_text(json.dumps(payload), encoding="utf-8")

    plan = build_pipeline_plan(
        source_suite=source,
        release_dir=tmp_path,
        workers=1,
        sample_timeout_seconds=900,
        max_replays=64,
        exact_max_calls=6,
        exact_max_replays=4096,
        per_action_cap=-1,
    )

    complexity = next(
        stage for stage in plan["stages"] if stage["name"] == "complexity"
    )
    argv = complexity["argv"]

    assert plan["admission_profile"] == "quality_core_v2"
    assert plan["complexity_mode"] == "bounded_diagnostic"
    assert argv[argv.index("--max-replays") + 1] == "1"
    assert argv[argv.index("--exact-max-calls") + 1] == "0"
    assert argv[argv.index("--exact-max-replays") + 1] == "0"
    assert argv[argv.index("--per-action-cap") + 1] == "0"


def test_pipeline_plan_and_cache_are_bound_to_core_release_pipeline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _source_suite(tmp_path)
    monkeypatch.setattr(
        pipeline,
        "implementation_identity",
        lambda: {
            "implementation_tree_sha256": "tree-one",
            "core_release_pipeline_sha256": "pipeline-one",
            "release_tooling_sha256": "tooling-one",
        },
    )
    first = build_pipeline_plan(
        source_suite=source,
        release_dir=tmp_path,
        workers=4,
        sample_timeout_seconds=900,
    )
    monkeypatch.setattr(
        pipeline,
        "implementation_identity",
        lambda: {
            "implementation_tree_sha256": "tree-two",
            "core_release_pipeline_sha256": "pipeline-two",
            "release_tooling_sha256": "tooling-two",
        },
    )
    second = build_pipeline_plan(
        source_suite=source,
        release_dir=tmp_path,
        workers=4,
        sample_timeout_seconds=900,
    )

    def behavioral_cache(plan: dict[str, object]) -> str:
        stage = next(
            item
            for item in plan["stages"]  # type: ignore[index]
            if item["name"] == "behavioral"  # type: ignore[index]
        )
        argv = stage["argv"]  # type: ignore[index]
        return argv[argv.index("--cache-dir") + 1]  # type: ignore[index]

    first_cache = behavioral_cache(first)
    second_cache = behavioral_cache(second)
    assert first["core_release_pipeline_sha256"] == "pipeline-one"
    assert first["release_tooling_sha256"] == "tooling-one"
    assert all(
        stage["core_release_pipeline_sha256"] == "pipeline-one"
        for stage in first["stages"]
    )
    assert "pipeline-one" in first_cache
    assert "pipeline-two" in second_cache
    assert first_cache != second_cache


def test_pipeline_cache_reuses_unchanged_rows_when_suite_expands(
    tmp_path: Path,
) -> None:
    small = _source_suite(tmp_path, n=3)
    first = build_pipeline_plan(
        source_suite=small,
        release_dir=tmp_path,
        workers=4,
        sample_timeout_seconds=900,
    )
    expanded = _source_suite(tmp_path, n=4)
    second = build_pipeline_plan(
        source_suite=expanded,
        release_dir=tmp_path,
        workers=4,
        sample_timeout_seconds=900,
    )

    def behavioral_cache(plan: dict[str, object]) -> str:
        stage = next(
            item
            for item in plan["stages"]  # type: ignore[index]
            if item["name"] == "behavioral"  # type: ignore[index]
        )
        argv = stage["argv"]  # type: ignore[index]
        return argv[argv.index("--cache-dir") + 1]  # type: ignore[index]

    assert behavioral_cache(first) == behavioral_cache(second)


def test_default_cli_is_dry_run_and_executes_no_commands(
    tmp_path: Path,
    capsys,
) -> None:
    source = _source_suite(tmp_path)

    rc = main(
        [
            "--source-suite",
            str(source),
            "--release-dir",
            str(tmp_path),
            "--workers",
            "4",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 0
    assert "NO_COMMANDS_EXECUTED=true" in captured.out
    for stage in STAGE_ORDER:
        assert stage in captured.out


def test_pipeline_infers_dynamic_working_set_count(tmp_path: Path) -> None:
    source = _source_suite(tmp_path, n=303)

    plan = build_pipeline_plan(
        source_suite=source,
        release_dir=tmp_path,
        workers=4,
        sample_timeout_seconds=900,
    )

    assert plan["n_source_scenarios"] == 303
    assert all(
        "core304" not in str(stage["expected_output"]) for stage in plan["stages"]
    )


def test_pipeline_records_portable_argv_but_retains_runtime_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        pipeline,
        "implementation_identity",
        lambda: {
            "implementation_tree_sha256": "tree",
            "core_release_pipeline_sha256": "pipeline",
            "release_tooling_sha256": "tooling",
        },
    )
    source = _source_suite(tmp_path, n=1)
    release_dir = tmp_path / "release" / "candidate"

    plan = build_pipeline_plan(
        source_suite=source,
        release_dir=release_dir,
        workers=1,
        sample_timeout_seconds=900,
    )

    for stage in plan["stages"]:
        assert all(not Path(value).is_absolute() for value in stage["argv"])
        assert any(Path(value).is_absolute() for value in stage["runtime_argv"])
    assert plan["source_suite"] == "source.json"
    assert plan["pipeline_manifest"] == (
        "release/candidate/protocol2_v21_pipeline_manifest.json"
    )


def test_pipeline_requires_real_sumo_runtime_for_live_traffic(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _source_suite(tmp_path, n=1)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["scenarios"][0]["backend_kind"] = "sumo"
    source.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.delenv("OPERATE_TRAFFIC_BACKEND_REAL", raising=False)

    with pytest.raises(ValueError, match="OPERATE_TRAFFIC_BACKEND_REAL=1"):
        build_pipeline_plan(
            source_suite=source,
            release_dir=tmp_path,
            workers=1,
            sample_timeout_seconds=900,
        )

    monkeypatch.setenv("OPERATE_TRAFFIC_BACKEND_REAL", "1")
    plan = build_pipeline_plan(
        source_suite=source,
        release_dir=tmp_path,
        workers=1,
        sample_timeout_seconds=900,
    )
    assert plan["runtime_binding"] == {
        "requires_real_sumo": True,
        "OPERATE_TRAFFIC_BACKEND_REAL": "1",
        "requires_real_autonomous_driving_sumo": False,
        "OPERATE_AUTONOMOUS_DRIVING_SUMO_REAL": None,
    }


def test_pipeline_requires_real_sumo_runtime_for_autonomous_driving(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _source_suite(tmp_path, n=1)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["scenarios"][0]["backend_kind"] = "sumo_ego"
    source.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.delenv("OPERATE_AUTONOMOUS_DRIVING_SUMO_REAL", raising=False)

    with pytest.raises(
        ValueError, match="OPERATE_AUTONOMOUS_DRIVING_SUMO_REAL=1"
    ):
        build_pipeline_plan(
            source_suite=source,
            release_dir=tmp_path,
            workers=1,
            sample_timeout_seconds=900,
        )

    monkeypatch.setenv("OPERATE_AUTONOMOUS_DRIVING_SUMO_REAL", "1")
    plan = build_pipeline_plan(
        source_suite=source,
        release_dir=tmp_path,
        workers=1,
        sample_timeout_seconds=900,
    )
    assert plan["runtime_binding"] == {
        "requires_real_sumo": False,
        "OPERATE_TRAFFIC_BACKEND_REAL": None,
        "requires_real_autonomous_driving_sumo": True,
        "OPERATE_AUTONOMOUS_DRIVING_SUMO_REAL": "1",
    }


def test_pipeline_rejects_explicit_count_mismatch(tmp_path: Path) -> None:
    source = _source_suite(tmp_path, n=303)

    try:
        build_pipeline_plan(
            source_suite=source,
            release_dir=tmp_path,
            workers=4,
            sample_timeout_seconds=900,
            expected_count=304,
        )
    except ValueError as exc:
        assert "exactly 304" in str(exc)
    else:
        raise AssertionError("explicit working-set count mismatch must fail")


def test_materialize_and_readiness_outputs_are_hash_and_semantics_bound(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tree = "tree-current"
    monkeypatch.setattr(
        pipeline,
        "implementation_identity",
        lambda: {
            "implementation_tree_sha256": tree,
            "core_release_pipeline_sha256": "pipeline-current",
        },
    )
    dependency = tmp_path / "dependency.json"
    dependency.write_text("{}\n", encoding="utf-8")
    from core.protocol21_evidence import artifact_binding

    binding = artifact_binding(dependency, implementation_tree_sha256=tree)
    semantics = pipeline.required_semantics()
    materialize = tmp_path / "materialize.json"
    materialize.write_text(
        json.dumps(
            {
                "status": "protocol21_core_candidate",
                "implementation_tree_sha256": tree,
                "evaluation_protocol": {
                    "version": semantics["protocol_version"],
                    "implementation_fingerprint": semantics[
                        "implementation_fingerprint"
                    ],
                },
                "scoring_version": semantics["scoring_version"],
                "input_bindings": {"source_suite": binding},
            }
        ),
        encoding="utf-8",
    )
    pipeline._validate_stage_output(
        {"name": "materialize_core", "expected_output": str(materialize)}
    )

    readiness = tmp_path / "readiness.json"
    readiness.write_text(
        json.dumps(
            {
                "status": "blocked",
                "formal_evaluation_ready": False,
                "implementation_tree_sha256": tree,
                "evaluation_semantics": semantics,
                "artifact_bindings": {"source_suite": binding},
            }
        ),
        encoding="utf-8",
    )
    pipeline._validate_stage_output(
        {"name": "readiness", "expected_output": str(readiness)}
    )


def test_materialize_output_rejects_stale_tree(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        pipeline,
        "implementation_identity",
        lambda: {
            "implementation_tree_sha256": "tree-current",
            "core_release_pipeline_sha256": "pipeline-current",
        },
    )
    output = tmp_path / "materialize.json"
    output.write_text(
        json.dumps(
            {
                "status": "protocol21_core_candidate",
                "implementation_tree_sha256": "tree-stale",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        RuntimeError, match="materialize_core: implementation tree drift"
    ):
        pipeline._validate_stage_output(
            {"name": "materialize_core", "expected_output": str(output)}
        )


def test_behavioral_complete_rejects_failed_row_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = "tree-current"
    monkeypatch.setattr(
        pipeline,
        "implementation_identity",
        lambda: {
            "implementation_tree_sha256": tree,
            "core_release_pipeline_sha256": "pipeline-current",
        },
    )
    dependency = tmp_path / "source.json"
    dependency.write_text("{}\n", encoding="utf-8")
    from core.protocol21_evidence import artifact_binding

    output = tmp_path / "behavioral.json"
    output.write_text(
        json.dumps(
            {
                "status": "complete",
                "n_expected": 2,
                "n_completed": 2,
                "status_counts": {"failed": 1, "passed": 1},
                "implementation_tree_sha256": tree,
                "evaluation_semantics": pipeline.required_semantics(),
                "input_bindings": {
                    "source_suite": artifact_binding(
                        dependency,
                        implementation_tree_sha256=tree,
                    )
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="behavioral: non-passing rows remain"):
        pipeline._validate_stage_output(
            {"name": "behavioral", "expected_output": str(output)}
        )


def test_preflight_rejects_expected_count_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tree = "tree-current"
    monkeypatch.setattr(
        pipeline,
        "implementation_identity",
        lambda: {
            "implementation_tree_sha256": tree,
            "core_release_pipeline_sha256": "pipeline-current",
        },
    )
    source_suite = tmp_path / "source_suite.json"
    source_suite.write_text(json.dumps({"rows": []}) + "\n", encoding="utf-8")
    from core.protocol21_evidence import artifact_binding

    output = tmp_path / "preflight.json"
    output.write_text(
        json.dumps(
            {
                "status": "passed",
                "n_expected": 2,
                "n_completed": 1,
                "n_fatal": 0,
                "implementation_tree_sha256": tree,
                "input_bindings": {
                    "source_suite": artifact_binding(
                        source_suite,
                        implementation_tree_sha256=tree,
                    )
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="expected/completed mismatch"):
        pipeline._validate_stage_output(
            {
                "name": "preflight",
                "argv": [
                    "preflight",
                    "--source-suite",
                    str(source_suite),
                    "--expected-count",
                    "2",
                ],
                "expected_output": str(output),
            }
        )


@pytest.mark.parametrize("tooling_changed", [False, True])
def test_resume_reuses_a_verified_completed_stage(
    tmp_path: Path,
    monkeypatch,
    capsys,
    tooling_changed,
) -> None:
    output = tmp_path / "preflight.json"
    output.write_text("{}\n", encoding="utf-8")
    manifest_path = tmp_path / "protocol2_v21_pipeline_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "running",
                "source_suite": "source.json",
                "source_suite_sha256": "source-sha",
                "scenario_yaml_graph": [],
                "implementation_tree_sha256": "tree-sha",
                "core_release_pipeline_sha256": "pipeline-sha",
                "release_tooling_sha256": "tooling-old",
                "runtime_binding": {},
                "stages": [
                    {
                        "name": "preflight",
                        "argv": ["verified-preflight"],
                        "core_release_pipeline_sha256": "pipeline-sha",
                        "return_code": 0,
                        "output_sha256": pipeline.hashlib.sha256(
                            output.read_bytes()
                        ).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = {
        "source_suite": "source.json",
        "source_suite_sha256": "source-sha",
        "scenario_yaml_graph": [],
        "implementation_tree_sha256": "tree-sha",
        "core_release_pipeline_sha256": "pipeline-sha",
        "runtime_binding": {},
        "pipeline_manifest": str(manifest_path),
        "stages": [
            {
                "name": "preflight",
                "argv": ["verified-preflight"],
                "core_release_pipeline_sha256": "pipeline-sha",
                "expected_output": str(output),
            }
        ],
    }
    monkeypatch.setattr(pipeline, "build_pipeline_plan", lambda **_: plan)
    monkeypatch.setattr(pipeline, "_verify_planned_source_graph", lambda _: None)
    monkeypatch.setattr(pipeline, "_validate_stage_output", lambda _: None)
    monkeypatch.setattr(
        pipeline,
        "implementation_identity",
        lambda: {
            "implementation_tree_sha256": "tree-sha",
            "core_release_pipeline_sha256": "pipeline-sha",
        },
    )
    monkeypatch.setattr(
        pipeline.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("a verified stage must not be rerun")
        ),
    )

    plan["release_tooling_sha256"] = "tooling-new" if tooling_changed else "tooling-old"
    rc = main(["--execute", "--resume"])

    assert rc == 0
    stdout = capsys.readouterr().out
    assert "RESUMED_STAGE=preflight" in stdout
    assert "--formal-manifest '<VERSIONED_RELEASE_MANIFEST>'" in stdout
    assert "--finalize" in stdout
    resumed = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert resumed["stages"][0]["reused"] is True
    assert resumed["resume"]["prior_release_tooling_sha256"] == "tooling-old"


def test_fresh_execute_rejects_nonempty_release_dir_before_any_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    release_dir = tmp_path / "existing-run"
    release_dir.mkdir()
    stale_checkpoint = release_dir / "behavioral.json.checkpoint.json"
    stale_checkpoint.write_text("{}\n", encoding="utf-8")
    plan = {
        "source_suite": "source.json",
        "source_suite_sha256": "source-sha",
        "scenario_yaml_graph": [],
        "implementation_tree_sha256": "tree-sha",
        "core_release_pipeline_sha256": "pipeline-sha",
        "runtime_binding": {},
        "pipeline_manifest": str(release_dir / "protocol2_v21_pipeline_manifest.json"),
        "stages": [
            {
                "name": "behavioral",
                "argv": ["behavioral"],
                "core_release_pipeline_sha256": "pipeline-sha",
                "expected_output": str(release_dir / "behavioral.json"),
            }
        ],
    }
    monkeypatch.setattr(pipeline, "build_pipeline_plan", lambda **_: plan)
    monkeypatch.setattr(
        pipeline.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("fresh execution must fail before running a stage")
        ),
    )

    rc = main(["--execute", "--release-dir", str(release_dir)])

    assert rc == 3
    assert "fresh run release directory is not empty" in capsys.readouterr().err
    assert stale_checkpoint.read_text(encoding="utf-8") == "{}\n"


def test_readiness_false_marks_pipeline_manifest_blocked(
    tmp_path: Path,
    monkeypatch,
) -> None:
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text(
        json.dumps(
            {
                "formal_evaluation_ready": False,
                "formal_run_blockers": ["held_rows"],
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "protocol2_v21_pipeline_manifest.json"
    plan = {
        "source_suite": "source.json",
        "source_suite_sha256": "source-sha",
        "scenario_yaml_graph": [],
        "implementation_tree_sha256": "tree-sha",
        "core_release_pipeline_sha256": "pipeline-sha",
        "runtime_binding": {},
        "pipeline_manifest": str(manifest_path),
        "stages": [
            {
                "name": "readiness",
                "argv": ["readiness"],
                "core_release_pipeline_sha256": "pipeline-sha",
                "expected_output": str(readiness_path),
            }
        ],
    }
    monkeypatch.setattr(pipeline, "build_pipeline_plan", lambda **_: plan)
    monkeypatch.setattr(pipeline, "_verify_planned_source_graph", lambda _: None)
    monkeypatch.setattr(pipeline, "_validate_stage_output", lambda _: None)
    monkeypatch.setattr(
        pipeline,
        "implementation_identity",
        lambda: {
            "implementation_tree_sha256": "tree-sha",
            "core_release_pipeline_sha256": "pipeline-sha",
        },
    )
    monkeypatch.setattr(
        pipeline.subprocess,
        "run",
        lambda *_args, **_kwargs: type("Completed", (), {"returncode": 0})(),
    )

    rc = main(["--execute", "--release-dir", str(tmp_path / "fresh-run")])

    assert rc == 4
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "blocked"
    assert manifest["formal_run_blockers"] == ["held_rows"]
    assert (
        json.loads(readiness_path.read_text(encoding="utf-8"))[
            "core_release_pipeline_sha256"
        ]
        == "pipeline-sha"
    )


def test_candidate_replay_manifest_is_a_terminal_artifact_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    materialized = tmp_path / "materialized.json"
    manifest_path = tmp_path / "protocol2_v21_pipeline_manifest.json"
    plan = {
        "source_suite": "source.json",
        "source_suite_sha256": "source-sha",
        "scenario_yaml_graph": [],
        "implementation_tree_sha256": "tree-sha",
        "core_release_pipeline_sha256": "pipeline-sha",
        "runtime_binding": {},
        "execution_profile": "candidate_replay",
        "stop_after": "materialize_core",
        "pipeline_manifest": str(manifest_path),
        "stages": [
            {
                "name": "materialize_core",
                "argv": ["materialize"],
                "core_release_pipeline_sha256": "pipeline-sha",
                "expected_output": str(materialized),
            }
        ],
    }
    monkeypatch.setattr(pipeline, "build_pipeline_plan", lambda **_: plan)
    monkeypatch.setattr(pipeline, "_verify_planned_source_graph", lambda _: None)
    monkeypatch.setattr(pipeline, "_validate_stage_output", lambda _: None)
    monkeypatch.setattr(
        pipeline,
        "implementation_identity",
        lambda: {
            "implementation_tree_sha256": "tree-sha",
            "core_release_pipeline_sha256": "pipeline-sha",
        },
    )

    def run_stage(*_args, **_kwargs):
        materialized.write_text(
            json.dumps({"status": "protocol21_core_candidate"}),
            encoding="utf-8",
        )
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(pipeline.subprocess, "run", run_stage)

    assert (
        main(
            [
                "--execute",
                "--release-dir",
                str(tmp_path / "fresh-run"),
                "--stop-after",
                "materialize_core",
            ]
        )
        == 0
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "candidate_replay_complete"
    assert manifest["completed_stage"] == "materialize_core"
    assert manifest["terminal_stage_artifact"] == {
        "path": str(materialized),
        "sha256": pipeline.hashlib.sha256(materialized.read_bytes()).hexdigest(),
    }
    stdout = capsys.readouterr().out
    assert "PIPELINE_STATUS=candidate_replay_complete" in stdout
    assert f"REPLAY_LEDGER={manifest_path}" in stdout


def test_resume_rejects_a_manifest_from_a_different_source_suite(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    manifest_path = tmp_path / "protocol2_v21_pipeline_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "running",
                "source_suite": "source.json",
                "source_suite_sha256": "different-source-sha",
                "scenario_yaml_graph": [],
                "implementation_tree_sha256": "tree-sha",
                "core_release_pipeline_sha256": "pipeline-sha",
                "runtime_binding": {},
                "stages": [],
            }
        ),
        encoding="utf-8",
    )
    plan = {
        "source_suite": "source.json",
        "source_suite_sha256": "source-sha",
        "scenario_yaml_graph": [],
        "implementation_tree_sha256": "tree-sha",
        "core_release_pipeline_sha256": "pipeline-sha",
        "runtime_binding": {},
        "pipeline_manifest": str(manifest_path),
        "stages": [],
    }
    monkeypatch.setattr(pipeline, "build_pipeline_plan", lambda **_: plan)

    rc = main(["--execute", "--resume"])

    assert rc == 3
    assert "resume manifest source suite mismatch" in capsys.readouterr().err


def test_resume_rejects_a_missing_pipeline_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path = tmp_path / "protocol2_v21_pipeline_manifest.json"
    plan = {
        "source_suite": "source.json",
        "source_suite_sha256": "source-sha",
        "scenario_yaml_graph": [],
        "implementation_tree_sha256": "tree-sha",
        "core_release_pipeline_sha256": "pipeline-sha",
        "runtime_binding": {},
        "pipeline_manifest": str(manifest_path),
        "stages": [],
    }
    monkeypatch.setattr(pipeline, "build_pipeline_plan", lambda **_: plan)

    rc = main(["--execute", "--resume"])

    assert rc == 3
    assert "resume manifest missing" in capsys.readouterr().err
    assert not manifest_path.exists()


def test_resume_rejects_a_stage_when_its_output_hash_changed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "preflight.json"
    output.write_text("first\n", encoding="utf-8")
    stage = {
        "name": "preflight",
        "argv": ["verified-preflight"],
        "expected_output": str(output),
    }
    prior_stage = {
        "argv": ["verified-preflight"],
        "return_code": 0,
        "output_sha256": pipeline.hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.write_text("tampered\n", encoding="utf-8")
    monkeypatch.setattr(pipeline, "_validate_stage_output", lambda _: None)

    assert pipeline._resume_stage_is_reusable(stage, prior_stage) is False


def test_resume_rejects_a_manifest_from_a_different_core_pipeline(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "protocol2_v21_pipeline_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_suite": "source.json",
                "source_suite_sha256": "source-sha",
                "scenario_yaml_graph": [],
                "implementation_tree_sha256": "tree-sha",
                "core_release_pipeline_sha256": "pipeline-old",
                "runtime_binding": {},
                "stages": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="core release pipeline mismatch"):
        pipeline._load_resume_manifest(
            manifest_path,
            source_suite="source.json",
            source_suite_sha256="source-sha",
            scenario_yaml_graph=[],
            implementation_tree_sha256="tree-sha",
            core_release_pipeline_sha256="pipeline-current",
            runtime_binding={},
        )


def test_resume_preserves_historical_release_tooling(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "protocol2_v21_pipeline_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_suite": "source.json",
                "source_suite_sha256": "source-sha",
                "scenario_yaml_graph": [],
                "implementation_tree_sha256": "tree-sha",
                "core_release_pipeline_sha256": "pipeline-sha",
                "release_tooling_sha256": "tooling-old",
                "runtime_binding": {},
                "stages": [],
            }
        ),
        encoding="utf-8",
    )

    original_bytes = manifest_path.read_bytes()
    loaded = pipeline._load_resume_manifest(
        manifest_path,
        source_suite="source.json",
        source_suite_sha256="source-sha",
        scenario_yaml_graph=[],
        implementation_tree_sha256="tree-sha",
        core_release_pipeline_sha256="pipeline-sha",
        runtime_binding={},
    )
    assert loaded["release_tooling_sha256"] == "tooling-old"
    assert manifest_path.read_bytes() == original_bytes


def test_resume_rejects_a_manifest_from_a_different_scenario_yaml_graph(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "protocol2_v21_pipeline_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_suite": "source.json",
                "source_suite_sha256": "source-sha",
                "scenario_yaml_graph": [
                    {
                        "scenario_id": "scenario-0",
                        "path": "scenario.yaml",
                        "sha256": "old-yaml-sha",
                    }
                ],
                "implementation_tree_sha256": "tree-sha",
                "core_release_pipeline_sha256": "pipeline-sha",
                "runtime_binding": {},
                "stages": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="scenario YAML graph mismatch"):
        pipeline._load_resume_manifest(
            manifest_path,
            source_suite="source.json",
            source_suite_sha256="source-sha",
            scenario_yaml_graph=[
                {
                    "scenario_id": "scenario-0",
                    "path": "scenario.yaml",
                    "sha256": "current-yaml-sha",
                }
            ],
            implementation_tree_sha256="tree-sha",
            core_release_pipeline_sha256="pipeline-sha",
            runtime_binding={},
        )


def test_pipeline_fails_closed_when_core_pipeline_drifts_mid_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "preflight.json"
    manifest_path = tmp_path / "protocol2_v21_pipeline_manifest.json"
    plan = {
        "source_suite": "source.json",
        "source_suite_sha256": "source-sha",
        "scenario_yaml_graph": [],
        "implementation_tree_sha256": "tree-sha",
        "core_release_pipeline_sha256": "pipeline-start",
        "runtime_binding": {},
        "pipeline_manifest": str(manifest_path),
        "stages": [
            {
                "name": "preflight",
                "argv": ["preflight"],
                "core_release_pipeline_sha256": "pipeline-start",
                "expected_output": str(output),
            }
        ],
    }
    identities = iter(
        [
            {
                "implementation_tree_sha256": "tree-sha",
                "core_release_pipeline_sha256": "pipeline-start",
            },
            {
                "implementation_tree_sha256": "tree-sha",
                "core_release_pipeline_sha256": "pipeline-changed",
            },
        ]
    )

    def run_stage(*_args, **_kwargs):
        output.write_text(json.dumps({"status": "complete"}), encoding="utf-8")
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(pipeline, "build_pipeline_plan", lambda **_: plan)
    monkeypatch.setattr(pipeline, "_verify_planned_source_graph", lambda _: None)
    monkeypatch.setattr(pipeline, "implementation_identity", lambda: next(identities))
    monkeypatch.setattr(pipeline.subprocess, "run", run_stage)

    rc = main(["--execute", "--release-dir", str(tmp_path / "fresh-run")])

    assert rc == 3
    assert "core release pipeline drift" in capsys.readouterr().err
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "blocked"
    assert manifest["core_release_pipeline_sha256"] == "pipeline-start"
    assert manifest["stages"][0]["core_release_pipeline_sha256"] == ("pipeline-start")
    assert manifest["stages"][0]["core_release_pipeline_sha256_end"] == (
        "pipeline-changed"
    )
