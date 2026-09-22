import hashlib

import pytest

from evaluation.native_reference import read_verified


def test_reference_artifact_bytes_and_location_are_verified(tmp_path):
    p = tmp_path / "episode.json"
    p.write_bytes(b"{}")
    digest = hashlib.sha256(b"{}").hexdigest()
    assert read_verified(tmp_path, "episode.json", digest)[0] == b"{}"
    p.write_bytes(b'{"changed":true}')
    with pytest.raises(ValueError, match="hash mismatch"):
        read_verified(tmp_path, "episode.json", digest)
    with pytest.raises(ValueError, match="escapes"):
        read_verified(tmp_path, "../outside.json")


def test_full_bundle_requires_terminal_determinism(tmp_path):
    import json
    from evaluation.native_quality import REFERENCE_POLICIES
    from evaluation.native_reference import load_reference_report

    def write(name, payload):
        raw = json.dumps(payload).encode()
        (tmp_path / name).write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    episodes = []
    for policy, cost in zip(REFERENCE_POLICIES, (1000, 700, 500, 900)):
        for repeat in (0, 1):
            name = f"{policy}_{repeat}.json"
            row = {
                "agent_name": policy,
                "scenario_signature": "case",
                "seed": 1,
                "status": "ok",
                "ground_truth_summary": {
                    "cost_components": {"energy_cost": cost},
                    "chose_fatal_option": False,
                },
                "task_completion": {
                    "contract": "building_energy.citylearn.storage_dispatch.v1",
                    "applicable": True,
                    "completed": True,
                    "evidence": {"cost": cost},
                },
                "trajectory_summary": {
                    "operational_agency_valid_evidence_ids": ["c", "s"]
                },
                "score": {
                    "dimensions": [
                        {
                            "name": "economic_cost",
                            "applicable": True,
                            "evidence_ids": ["c"],
                        },
                        {
                            "name": "system_survival",
                            "applicable": True,
                            "floor_violation": False,
                            "evidence_ids": ["s"],
                        },
                    ]
                },
            }
            digest = write(name, row)
            episodes.append(
                {
                    "policy": policy,
                    "repetition": repeat,
                    "status": "ok",
                    "episode_path": name,
                    "episode_sha256": digest,
                    "artifacts": [{"path": name, "sha256": digest}],
                }
            )
    write(
        "manifest.json",
        {
            "implementation_identity": {"evaluation_runtime_sha256": "tree"},
            "policies": list(REFERENCE_POLICIES),
            "repetitions": 2,
            "rows": [
                {
                    "scenario_signature": "case",
                    "seed": 1,
                    "domain": "building_energy",
                    "backend_kind": "citylearn",
                }
            ],
        },
    )
    result = {
        "scenario_signature": "case",
        "seed": 1,
        "episodes": episodes,
        "determinism": {p: {"deterministic": True} for p in REFERENCE_POLICIES},
    }
    report = {
        "manifest": "manifest.json",
        "identity_unchanged": True,
        "end_implementation_identity": {"evaluation_runtime_sha256": "tree"},
        "results": [result],
    }
    write("report.json", report)
    assert (
        load_reference_report(tmp_path, "report.json")["contracts"][0]["status"]
        == "ready"
    )
    result["determinism"]["random"]["deterministic"] = False
    write("report.json", report)
    assert (
        load_reference_report(tmp_path, "report.json")["contracts"][0]["status"]
        == "unavailable"
    )
    result["episodes"][-1] = {
        "policy": "random",
        "repetition": 1,
        "status": "error",
        "error": "TimeoutError: reference exceeded 300.0s",
    }
    result["determinism"]["random"]["reason"] = "incomplete_replay"
    write("report.json", report)
    contract = load_reference_report(tmp_path, "report.json")["contracts"][0]
    assert contract["status"] == "unavailable"
    assert contract["reason"] == "reference_measurement_unavailable"
    assert contract["reference_failures"] == [
        {
            "policy": "random",
            "repetition": 1,
            "reason": "TimeoutError: reference exceeded 300.0s",
        }
    ]
    # A fixed expert target does not consume weak/no-action policy artifacts.
    for policy in ("wait_only", "random"):
        for repeat in (0, 1):
            (tmp_path / f"{policy}_{repeat}.json").unlink()
    selected = load_reference_report(
        tmp_path, "report.json", policies=("greedy_heuristic", "oracle_offline")
    )["contracts"][0]
    assert set(selected["policy_measurements"]) == {
        "greedy_heuristic",
        "oracle_offline",
    }
    assert selected["status"] == "policy_measurements_verified"
    assert "weak_cost" not in selected
    assert not selected.get("reference_failures")


@pytest.mark.parametrize(
    "fault", [None, "mismatch", "missing", "invalid", "incomplete"]
)
def test_fixed_fjsp_reference_checks_terminal_operations(tmp_path, fault):
    import json
    from evaluation.native_reference import load_reference_report

    def write(name, value):
        raw = json.dumps(value).encode()
        (tmp_path / name).write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    counts = dict(
        operations_total=3,
        operations_cancelled=0,
        operations_completed=3,
        operations_scheduled=3,
    )
    if fault == "incomplete":
        counts["operations_completed"] = 2
    terminal = dict(jobs_total=1, jobs_arrived=1, **counts)
    if fault == "mismatch":
        terminal["operations_completed"] = 2
    elif fault == "missing":
        terminal.pop("operations_scheduled")
    elif fault == "invalid":
        terminal["operations_completed"] = True
    episodes = []
    for policy in ("greedy_heuristic", "oracle_offline"):
        for repeat in (0, 1):
            name = f"{policy}_{repeat}.json"
            trace = f"{policy}_{repeat}.trajectory.jsonl"
            episode = dict(
                agent_name=policy,
                scenario_signature="case",
                seed=1,
                status="ok",
                ground_truth_summary=dict(
                    cost_components={"production_cost": 10}, chose_fatal_option=False
                ),
                task_completion=dict(
                    contract="logistics.job_shop.all_operations_scheduled.v1",
                    applicable=True,
                    completed=fault != "incomplete",
                    evidence=dict(operations_required=3, **counts),
                ),
                trajectory_summary=dict(
                    operational_agency_valid_evidence_ids=["c", "s"]
                ),
                score=dict(
                    dimensions=[
                        dict(name="economic_cost", applicable=True, evidence_ids=["c"]),
                        dict(
                            name="system_survival",
                            applicable=True,
                            floor_violation=False,
                            evidence_ids=["s"],
                        ),
                    ]
                ),
            )
            episodes.append(
                dict(
                    policy=policy,
                    repetition=repeat,
                    status="ok",
                    episode_path=name,
                    episode_sha256=write(name, episode),
                    artifacts=[
                        dict(path=trace, sha256=write(trace, {"observation": terminal}))
                    ],
                )
            )
    write(
        "manifest.json",
        dict(
            implementation_identity={"evaluation_runtime_sha256": "tree"},
            policies=["greedy_heuristic", "oracle_offline"],
            repetitions=2,
            rows=[
                dict(
                    scenario_signature="case",
                    seed=1,
                    domain="logistics",
                    backend_kind="dynasched_flexible_job_shop",
                )
            ],
        ),
    )
    write(
        "report.json",
        dict(
            manifest="manifest.json",
            identity_unchanged=True,
            end_implementation_identity={"evaluation_runtime_sha256": "tree"},
            results=[
                dict(
                    scenario_signature="case",
                    seed=1,
                    episodes=episodes,
                    determinism={
                        p: {"deterministic": True}
                        for p in ("greedy_heuristic", "oracle_offline")
                    },
                )
            ],
        ),
    )
    contract = load_reference_report(
        tmp_path, "report.json", policies=("greedy_heuristic", "oracle_offline")
    )["contracts"][0]
    measurement = contract["policy_measurements"]["greedy_heuristic"][0]
    if fault in {"mismatch", "missing", "invalid"}:
        assert measurement["applicable"] is False
        assert measurement["reason"] == "reference_terminal_operations_unproven"
    else:
        assert measurement["applicable"] is True
        assert measurement["feasible"] is (fault is None)
