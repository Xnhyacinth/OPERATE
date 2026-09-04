"""
data.dataset_builder — Pack episodes into a canonical dataset.

Reads ``*.trajectory.jsonl`` + ``*.header.json`` + ``*.summary.json`` from
a directory, filters by quality threshold, and emits ``train/test/val``
splits as a single JSON manifest plus per-episode files.

Forked from ``dispatch-benchmark/data/dataset_builder.py``; simplified for
v0.1 (no domain-specific filters beyond quality).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .quality_validator import validate_trajectory


@dataclass
class DatasetSplit:
    name: str
    episodes: list[str]


def _validate_declared_sidecar(
    *,
    trajectory_dir: Path,
    episode_id: str,
    header: dict[str, Any],
    summary: dict[str, Any],
    stem: str,
    header_hash_key: str,
    header_count_key: str,
    summary_key: str,
    required: bool,
) -> tuple[list[str], dict[str, Any] | None]:
    expected_hash = header.get(header_hash_key)
    expected_count = header.get(header_count_key)
    artifact = ((summary.get("trajectory_summary") or {}).get(summary_key) or {})
    path = trajectory_dir / f"{episode_id}.{stem}.jsonl"
    declared = (
        expected_hash not in (None, "")
        or expected_count not in (None, "")
        or bool(artifact)
        or path.exists()
    )
    if not declared and not required:
        return [], None
    if not declared:
        return [f"required {stem} artifact binding missing"], None
    issues: list[str] = []
    if expected_hash in (None, "") or expected_count in (None, ""):
        issues.append(f"{stem} header binding missing")
    if not path.is_file():
        issues.append(f"missing declared {stem} artifact")
        return issues, None
    payload = path.read_bytes()
    actual_hash = hashlib.sha256(payload).hexdigest()
    actual_count = len(payload.splitlines())
    if str(expected_hash or "") != actual_hash:
        issues.append(f"{stem} sha256 mismatch")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count != actual_count
    ):
        issues.append(f"{stem} event count mismatch")
    if not isinstance(artifact, dict):
        issues.append(f"{stem} summary binding missing")
    elif (
        artifact.get("sha256") != actual_hash
        or artifact.get("event_count") != actual_count
    ):
        issues.append(f"{stem} summary binding mismatch")
    return issues, {
        "path": path.name,
        "sha256": actual_hash,
        "event_count": actual_count,
    }


def build_dataset(
    trajectory_dir: Path | str,
    output_dir: Path | str,
    *,
    quality_threshold: float = 0.7,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 0,
) -> dict[str, Any]:
    trajectory_dir = Path(trajectory_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    headers = sorted(trajectory_dir.glob("*.header.json"))
    accepted: list[str] = []
    artifact_bindings: dict[str, dict[str, Any]] = {}
    rejected: list[tuple[str, list[str]]] = []
    for hp in headers:
        episode_id = hp.stem.replace(".header", "")
        traj_p = trajectory_dir / f"{episode_id}.trajectory.jsonl"
        summary_p = trajectory_dir / f"{episode_id}.summary.json"
        if not traj_p.exists():
            rejected.append((episode_id, ["missing trajectory artifact"]))
            continue
        if not summary_p.exists():
            rejected.append((episode_id, ["missing summary artifact"]))
            continue
        with open(hp, encoding="utf-8") as f:
            header = json.load(f)
        entries = []
        with open(traj_p, encoding="utf-8") as f:
            for line in f:
                entries.append(json.loads(line))
        summary = json.loads(summary_p.read_text(encoding="utf-8"))
        agent_name = str(header.get("agent_name") or "")
        llm_artifacts_required = agent_name in {
            "llm_agent",
            "react_llm",
            "reflexion_llm",
        } or agent_name.startswith("llm_agent/")
        sidecar_issues: list[str] = []
        episode_bindings: dict[str, Any] = {}
        for spec in (
            (
                "semantic_ledger",
                "semantic_ledger_sha256",
                "semantic_ledger_events",
                "semantic_ledger_artifact",
            ),
            (
                "provider_audit",
                "provider_audit_sha256",
                "provider_audit_events",
                "provider_audit_artifact",
            ),
        ):
            issues, binding = _validate_declared_sidecar(
                trajectory_dir=trajectory_dir,
                episode_id=episode_id,
                header=header,
                summary=summary,
                stem=spec[0],
                header_hash_key=spec[1],
                header_count_key=spec[2],
                summary_key=spec[3],
                required=llm_artifacts_required,
            )
            sidecar_issues.extend(issues)
            if binding is not None:
                episode_bindings[spec[0]] = binding
        if sidecar_issues:
            rejected.append((episode_id, sidecar_issues))
            continue
        report = validate_trajectory(header, entries, summary)
        if report.passed and report.quality_score >= quality_threshold:
            accepted.append(episode_id)
            if episode_bindings:
                artifact_bindings[episode_id] = episode_bindings
        else:
            rejected.append((episode_id, report.issues))

    rng = random.Random(seed)
    rng.shuffle(accepted)
    n = len(accepted)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    splits = {
        "train": accepted[:n_train],
        "val": accepted[n_train : n_train + n_val],
        "test": accepted[n_train + n_val :],
    }
    manifest = {
        "n_accepted": n,
        "n_rejected": len(rejected),
        "quality_threshold": quality_threshold,
        "splits": splits,
        "rejected": [{"episode_id": e, "issues": i} for e, i in rejected],
        "artifact_bindings": artifact_bindings,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest
