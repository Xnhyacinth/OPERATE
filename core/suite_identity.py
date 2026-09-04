"""Canonical scenario and suite identity helpers shared by batch/readiness."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from core.protocol21_evidence import portable_repo_path
from runner.resume import recompute_signature_with_seed


def canonical_scenario_slug(raw_path: str) -> str:
    slug = raw_path.replace("\\", "/")
    if slug.startswith("scenarios/"):
        slug = slug[len("scenarios/") :]
    if slug.endswith(".yaml"):
        slug = slug[:-5]
    return slug


def canonical_suite_manifest_sha256(
    scenarios: list[str],
    scenario_bodies: dict[str, dict[str, Any]],
) -> str:
    payload = {
        "schema_version": "1.0",
        "scenarios": [
            {
                "scenario_slug": slug,
                "scenario_signature": scenario_bodies[slug].get(
                    "scenario_signature"
                ),
                "seed": scenario_bodies[slug].get("seed"),
                "horizon_ticks": scenario_bodies[slug].get("horizon_ticks"),
            }
            for slug in scenarios
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def scenario_yaml_binding(
    path: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    resolved = path.resolve()
    body = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ValueError(f"scenario YAML must contain a mapping: {resolved}")
    return {
        "path": portable_repo_path(resolved, repo_root=repo_root),
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "body": body,
    }


def verify_scenario_row_against_yaml(
    row: dict[str, Any],
    *,
    path: Path,
) -> list[str]:
    binding = scenario_yaml_binding(path)
    body = binding["body"]
    errors: list[str] = []
    seed = int(row.get("seed", body.get("seed", 42)))
    try:
        signature = recompute_signature_with_seed(body, seed)
    except Exception:
        return ["scenario_signature_recompute_failed"]
    checks = {
        "scenario_id": body.get("scenario_id") or body.get("seed_id"),
        "scenario_signature": signature,
        "path": str(row.get("path") or ""),
        "seed": body.get("seed"),
        "domain": body.get("domain"),
        "backend_kind": body.get("backend_kind"),
        "family": body.get("family"),
        "difficulty_level": body.get("difficulty_level"),
        "difficulty_mode": body.get("difficulty_mode"),
        "horizon_ticks": body.get("horizon_ticks"),
    }
    for field, actual in checks.items():
        if field == "path":
            if not str(actual):
                errors.append("scenario_path_missing")
            continue
        expected = row.get(field)
        if expected is None and field in {"seed", "horizon_ticks"}:
            expected = actual
        if str(expected) != str(actual):
            errors.append(f"scenario_{field}_mismatch")
    return errors
