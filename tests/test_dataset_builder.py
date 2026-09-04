from __future__ import annotations

import hashlib
import json
from pathlib import Path

from data.dataset_builder import build_dataset


def _write_header(directory: Path, episode_id: str, total_ticks: int = 3) -> None:
    (directory / f"{episode_id}.header.json").write_text(
        json.dumps(
            {
                "scenario_id": "scenario",
                "scenario_signature": "signature",
                "agent_name": "agent",
                "seed": 42,
                "total_ticks": total_ticks,
            }
        ),
        encoding="utf-8",
    )


def _write_summary(directory: Path, episode_id: str) -> None:
    (directory / f"{episode_id}.summary.json").write_text(
        json.dumps(
            {"state_changing_action_rate": 1.0, "tool_failure_rate": 0.0}
        ),
        encoding="utf-8",
    )


def _write_missing_evidence_trajectory(directory: Path, episode_id: str) -> None:
    rows = []
    for name in ("shed_load", "redispatch", "switch_line"):
        rows.append(
            {
                "reward": 0.0,
                "action": {
                    "actions": [{"name": name, "args": {}}],
                    "dominant_action": name,
                },
                "tool_results": [
                    {"name": name, "ok": True, "state_changing": True}
                ],
                "evidence_ids": [],
            }
        )
    (directory / f"{episode_id}.trajectory.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_valid_trajectory(directory: Path, episode_id: str) -> None:
    rows = []
    for idx, name in enumerate(("shed_load", "redispatch", "switch_line")):
        rows.append(
            {
                "reward": 0.0,
                "action": {
                    "actions": [{"name": name, "args": {}}],
                    "dominant_action": name,
                },
                "tool_results": [
                    {"name": name, "ok": True, "state_changing": True}
                ],
                "evidence_ids": [f"ev-{idx}"],
            }
        )
    (directory / f"{episode_id}.trajectory.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_dataset_builder_rejects_high_score_that_fails_critical_gate(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_header(source, "missing-evidence")
    _write_summary(source, "missing-evidence")
    _write_missing_evidence_trajectory(source, "missing-evidence")

    manifest = build_dataset(source, tmp_path / "output")

    assert manifest["n_accepted"] == 0
    assert manifest["n_rejected"] == 1
    assert manifest["rejected"][0]["episode_id"] == "missing-evidence"
    assert any(
        "state-changing steps lack evidence_ids" in issue
        for issue in manifest["rejected"][0]["issues"]
    )


def test_dataset_builder_counts_missing_companion_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_header(source, "missing-trajectory")
    _write_summary(source, "missing-trajectory")
    _write_header(source, "missing-summary")
    _write_missing_evidence_trajectory(source, "missing-summary")

    manifest = build_dataset(source, tmp_path / "output")

    assert manifest["n_accepted"] == 0
    assert manifest["n_rejected"] == 2
    issues = {
        row["episode_id"]: row["issues"] for row in manifest["rejected"]
    }
    assert issues["missing-trajectory"] == ["missing trajectory artifact"]
    assert issues["missing-summary"] == ["missing summary artifact"]


def test_dataset_builder_rejects_tampered_declared_provider_audit(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    episode_id = "provider-tamper"
    provider_payload = b'{"record_kind":"provider_request"}\n'
    provider_hash = hashlib.sha256(provider_payload).hexdigest()
    _write_header(source, episode_id)
    header_path = source / f"{episode_id}.header.json"
    header = json.loads(header_path.read_text(encoding="utf-8"))
    header.update(
        {
            "provider_audit_sha256": provider_hash,
            "provider_audit_events": 1,
        }
    )
    header_path.write_text(json.dumps(header), encoding="utf-8")
    _write_summary(source, episode_id)
    summary_path = source / f"{episode_id}.summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["trajectory_summary"] = {
        "provider_audit_artifact": {
            "sha256": provider_hash,
            "event_count": 1,
        }
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    _write_valid_trajectory(source, episode_id)
    (source / f"{episode_id}.provider_audit.jsonl").write_bytes(
        provider_payload + b'{"tampered":true}\n'
    )

    manifest = build_dataset(source, tmp_path / "output")

    assert manifest["n_accepted"] == 0
    assert manifest["rejected"] == [
        {
            "episode_id": episode_id,
            "issues": [
                "provider_audit sha256 mismatch",
                "provider_audit event count mismatch",
                "provider_audit summary binding mismatch",
            ],
        }
    ]


def test_dataset_builder_rejects_deleted_and_stripped_provider_audit(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    episode_id = "provider-deleted"
    provider_payload = b'{"record_kind":"provider_request"}\n'
    provider_hash = hashlib.sha256(provider_payload).hexdigest()
    _write_header(source, episode_id)
    header_path = source / f"{episode_id}.header.json"
    header = json.loads(header_path.read_text(encoding="utf-8"))
    header["provider_audit_sha256"] = provider_hash
    header["provider_audit_events"] = 1
    header_path.write_text(json.dumps(header), encoding="utf-8")
    _write_summary(source, episode_id)
    summary_path = source / f"{episode_id}.summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["trajectory_summary"] = {
        "provider_audit_artifact": {
            "sha256": provider_hash,
            "event_count": 1,
        }
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    _write_valid_trajectory(source, episode_id)

    manifest = build_dataset(source, tmp_path / "output")

    assert manifest["n_accepted"] == 0
    assert "missing declared provider_audit artifact" in manifest["rejected"][0][
        "issues"
    ]

    header.pop("provider_audit_sha256")
    header.pop("provider_audit_events")
    header_path.write_text(json.dumps(header), encoding="utf-8")
    manifest = build_dataset(source, tmp_path / "output-stripped")
    assert "provider_audit header binding missing" in manifest["rejected"][0][
        "issues"
    ]


def test_dataset_builder_requires_llm_semantic_and_provider_artifacts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    episode_id = "llm-unbound"
    _write_header(source, episode_id)
    header_path = source / f"{episode_id}.header.json"
    header = json.loads(header_path.read_text(encoding="utf-8"))
    header["agent_name"] = "llm_agent/gpt-test"
    header_path.write_text(json.dumps(header), encoding="utf-8")
    _write_summary(source, episode_id)
    _write_valid_trajectory(source, episode_id)

    manifest = build_dataset(source, tmp_path / "output")

    assert manifest["n_accepted"] == 0
    assert manifest["rejected"][0]["issues"] == [
        "required semantic_ledger artifact binding missing",
        "required provider_audit artifact binding missing",
    ]
