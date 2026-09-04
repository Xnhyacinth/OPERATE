#!/usr/bin/env python3
"""Immutably rematerialize a locked driving catalog with current converters."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from domains.autonomous_driving.data.ngsim import (  # noqa: E402
    materialize_bundle,
    verify_bundle,
)
from scripts.build_autonomous_driving_catalog import build_catalog  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("autonomous_driving_rematerialize_json_object_required")
    return value


def _resolve(path_value: Any) -> Path:
    path = Path(str(path_value or ""))
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _output_name(candidate_id: str) -> str:
    name = candidate_id.replace(":", "_")
    if not name or Path(name).name != name:
        raise ValueError("autonomous_driving_rematerialize_candidate_id_invalid")
    return name


def _source_inputs(bundle: Path) -> dict[str, Path]:
    lock_path = bundle / "source/source.lock.json"
    source_lock = _load(lock_path)
    raw_name = str(source_lock.get("raw_file_name") or "")
    if not raw_name or Path(raw_name).name != raw_name:
        raise ValueError("autonomous_driving_rematerialize_raw_name_invalid")
    inputs = {
        "raw_path": bundle / "source" / raw_name,
        "source_lock_path": lock_path,
        "database_path": bundle / "normalized/trajectories.sqlite3",
        "normalization_lock_path": bundle / "normalized/normalization.lock.json",
        "mining_report_path": bundle / "mining/candidates.json",
    }
    if any(not path.is_file() for path in inputs.values()):
        raise ValueError("autonomous_driving_rematerialize_source_input_missing")
    return inputs


def _resume_candidate(bundle: Path, candidate_id: str) -> None:
    verify_bundle(bundle)
    fixture = _load(bundle / "runtime/fixture.json")
    observed = str((fixture.get("derivation") or {}).get("candidate_id") or "")
    if observed != candidate_id:
        raise ValueError("autonomous_driving_rematerialize_resume_candidate_mismatch")


def rematerialize_catalog(
    *,
    catalog: dict[str, Any],
    output_root: Path,
    output_catalog: Path,
    resume: bool = False,
) -> dict[str, Any]:
    if catalog.get("schema_version") != "autonomous_driving_candidate_catalog_v1":
        raise ValueError("autonomous_driving_rematerialize_catalog_schema_invalid")
    rows = catalog.get("bundles")
    if not isinstance(rows, list) or not rows:
        raise ValueError("autonomous_driving_rematerialize_catalog_empty")
    if output_catalog.exists():
        raise FileExistsError("autonomous_driving_rematerialize_catalog_output_exists")
    if output_root.exists() and not resume:
        raise FileExistsError("autonomous_driving_rematerialize_root_exists")
    output_root.mkdir(parents=True, exist_ok=True)

    candidate_ids: set[str] = set()
    results: list[dict[str, Any]] = []
    materialized = 0
    resumed = 0
    for value in rows:
        if not isinstance(value, dict):
            raise ValueError("autonomous_driving_rematerialize_catalog_row_invalid")
        candidate_id = str(value.get("candidate_id") or "")
        if not candidate_id or candidate_id in candidate_ids:
            raise ValueError("autonomous_driving_rematerialize_candidate_identity_invalid")
        candidate_ids.add(candidate_id)
        source_bundle = _resolve(value.get("bundle_path"))
        output_bundle = output_root / _output_name(candidate_id)
        if output_bundle.exists():
            if not resume:
                raise FileExistsError("autonomous_driving_rematerialize_bundle_exists")
            _resume_candidate(output_bundle, candidate_id)
            disposition = "resumed_verified"
            resumed += 1
        else:
            manifest = materialize_bundle(
                **_source_inputs(source_bundle),
                output_dir=output_bundle,
                candidate_id=candidate_id,
            )
            verification = verify_bundle(output_bundle)
            disposition = "materialized_verified"
            materialized += 1
            if verification.get("status") != "verified":
                raise ValueError("autonomous_driving_rematerialize_verification_failed")
            if str(manifest.get("selected_candidate_id") or "") != candidate_id:
                raise ValueError("autonomous_driving_rematerialize_candidate_mismatch")
        results.append(
            {
                "candidate_id": candidate_id,
                "source_bundle": str(source_bundle),
                "output_bundle": str(output_bundle.resolve()),
                "disposition": disposition,
            }
        )

    built = build_catalog(
        output_root,
        output_catalog,
        candidate_ids=candidate_ids,
    )
    if int(built.get("bundle_count") or 0) != len(candidate_ids):
        raise ValueError("autonomous_driving_rematerialize_catalog_count_mismatch")
    return {
        "schema_version": "autonomous_driving_rematerialization_v1",
        "status": "verified",
        "source_candidates": len(candidate_ids),
        "materialized": materialized,
        "resumed": resumed,
        "output_root": str(output_root.resolve()),
        "output_catalog": str(output_catalog.resolve()),
        "bundles": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-catalog", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    catalog_path = _resolve(args.catalog)
    output_root = _resolve(args.output_root)
    output_catalog = _resolve(args.output_catalog)
    report_path = _resolve(args.report)
    try:
        if report_path.exists():
            raise FileExistsError("autonomous_driving_rematerialize_report_exists")
        report = rematerialize_catalog(
            catalog=_load(catalog_path),
            output_root=output_root,
            output_catalog=output_catalog,
            resume=args.resume,
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "failed", "reason": str(error)}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "materialized": report["materialized"],
                "resumed": report["resumed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
