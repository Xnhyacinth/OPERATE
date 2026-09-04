#!/usr/bin/env python3
"""Build a non-release structural probe for the CityLearn baeda_3dem graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lock_citylearn_source import build as build_source_lock  # noqa: E402

DEFAULT_SOURCE_ROOT = (
    REPO_ROOT / "works" / "CityLearn" / "data" / "datasets" / "baeda_3dem"
)
DEFAULT_SOURCE_LOCK_OUTPUT = (
    REPO_ROOT / ".hl" / "artifacts" / "citylearn_baeda_3dem_source_lock.json"
)
DEFAULT_REPORT_OUTPUT = (
    REPO_ROOT / ".hl" / "artifacts" / "citylearn_baeda_3dem_structural_probe.json"
)


def _stable_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _candidate(
    *,
    environment_status: str,
    physical_source_key: str | None = None,
) -> dict[str, object]:
    return {
        "candidate_id": "citylearn/baeda_3dem/independent_graph_probe",
        "candidate_only": True,
        "release_admission": False,
        "domain": "building_energy",
        "backend_kind": "citylearn",
        "dataset_id": "baeda_3dem",
        "physical_source_key": physical_source_key,
        "quality": {
            "behavioral_headroom": "unknown",
            "environment_status": environment_status,
            "safety": "unknown",
            "source_consumption": "unknown",
        },
    }


def _physical_source_key(source_lock: Mapping[str, Any]) -> str:
    runtime_files = source_lock.get("runtime_files")
    derivation_files = source_lock.get("derivation_files", {})
    if not isinstance(runtime_files, Mapping) or not runtime_files:
        raise ValueError("CityLearn lock has no runtime asset graph")
    if not isinstance(derivation_files, Mapping):
        raise ValueError("CityLearn lock has an invalid derivation asset graph")
    required_assets: list[dict[str, str]] = []
    for role, rows in (
        ("runtime", runtime_files),
        ("derivation", derivation_files),
    ):
        for declared_path, raw_row in sorted(rows.items()):
            if not isinstance(raw_row, Mapping):
                raise ValueError(f"invalid CityLearn {role} row: {declared_path}")
            digest = raw_row.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"invalid CityLearn {role} hash: {declared_path}")
            required_assets.append(
                {
                    "asset_role": role,
                    "declared_path": str(declared_path),
                    "sha256": digest,
                }
            )
    key = {
        "backend_kind": "citylearn",
        "required_source_assets": required_assets,
        "schema_version": "source_asset_graph_v1",
    }
    return json.dumps(key, sort_keys=True, separators=(",", ":"))


def build_structural_probe(
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    lock_builder: Callable[..., dict[str, object]] = build_source_lock,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    """Build source structure evidence without running scientific gates."""

    base_report: dict[str, object] = {
        "schema_version": "operate_citylearn_structural_probe_v1",
        "candidate_only": True,
        "release_admission": False,
        "simulator_executed": False,
        "scientific_rejection": False,
        "disposition": "held_repair",
        "next_stage": "bounded_runtime_source_consumption_and_headroom",
    }
    try:
        source_lock = lock_builder(source_root, dataset_id="baeda_3dem")
        if source_lock.get("source_id") != "baeda_3dem":
            raise ValueError("CityLearn lock dataset identity mismatch")
        physical_source_key = _physical_source_key(source_lock)
        dataset_identity = source_lock.get("dataset_identity")
        if not isinstance(dataset_identity, Mapping):
            raise ValueError("CityLearn lock has no dataset identity")
    except (ImportError, subprocess.SubprocessError, RuntimeError) as exc:
        report = {
            **base_report,
            "status": "held_environment_repair",
            "candidate": _candidate(environment_status="held_repair"),
            "environment_error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
        return None, report
    except (OSError, ValueError) as exc:
        report = {
            **base_report,
            "status": "held_source_repair",
            "candidate": _candidate(environment_status="held_repair"),
            "source_closure_error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
        return None, report

    report = {
        **base_report,
        "status": "structural_prefilter_complete",
        "candidate": _candidate(
            environment_status="ready",
            physical_source_key=physical_source_key,
        ),
        "source_lock_payload_sha256": _stable_sha256(source_lock),
        "dataset_identity": dict(dataset_identity),
    }
    return source_lock, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--source-lock-output", type=Path, default=DEFAULT_SOURCE_LOCK_OUTPUT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    source_lock, report = build_structural_probe(
        source_root=args.source_root,
    )
    if not args.execute:
        print(json.dumps(report, indent=2))
        return
    if source_lock is not None:
        args.source_lock_output.parent.mkdir(parents=True, exist_ok=True)
        args.source_lock_output.write_text(
            json.dumps(source_lock, indent=2) + "\n", encoding="utf-8"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
