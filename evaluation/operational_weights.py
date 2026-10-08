"""Freeze domain, task-family and physical-source-aware task weights."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import yaml

from evaluation.leaderboard import _physical_source_identity
from evaluation.mission_contracts import _digest, _read_locked

VERSION = "operational_task_weights.v2"
LEGACY_VERSION = "operational_task_weights.v1"


def _key(row: dict) -> tuple[str, int]:
    signature, seed = row.get("scenario_signature"), row.get("seed")
    if not isinstance(signature, str) or not signature.strip() or type(seed) is not int:
        raise ValueError("invalid_weight_case_identity")
    return signature, seed


def _unique(rows: list[dict]) -> dict[tuple[str, int], dict]:
    if not rows:
        raise ValueError("empty_weight_suite")
    result = {}
    for row in rows:
        key = _key(row)
        if key in result:
            raise ValueError("duplicate_weight_case")
        result[key] = row
    return result


def compile_task_weights(suite_path: str | Path, *, root: str | Path = ".") -> dict:
    """Compile source-authenticated weights without reading any model outcome."""
    root = Path(root)
    path = Path(suite_path)
    if not path.is_absolute():
        path = root / path
    raw = path.read_bytes()
    rows = json.loads(raw)["scenarios"]
    _unique(rows)
    for row in rows:
        if not isinstance(row.get("domain"), str) or not row["domain"].strip():
            raise ValueError("missing_weight_domain")
    # A backend is an implementation detail, not a unit of scoring weight.
    # Missing physical identities remain separate cases and are disclosed.
    sources = {}
    for row in rows:
        family = row.get("family")
        if not isinstance(family, str) or not family.strip():
            raise ValueError("missing_weight_task_family")
        sources[_key(row)] = _physical_source_identity(row)
    domains = {row["domain"] for row in rows}
    families = {(row["domain"], row["family"]) for row in rows}
    source_groups = {}
    for row in rows:
        key = _key(row)
        source = (
            sources[key] if sources[key] is not None else ("unidentified_case", *key)
        )
        source_groups[key] = (row["domain"], row["family"], source)
    family_counts = Counter(domain for domain, _ in families)
    source_counts = Counter(group[:2] for group in set(source_groups.values()))
    case_counts = Counter(source_groups.values())
    cases = []
    for row in sorted(rows, key=_key):
        scenario = yaml.safe_load(
            _read_locked(
                root,
                {
                    "path": row["path"],
                    "sha256": row["yaml_sha256"],
                },
                label="scenario",
            )
        )
        for field in ("scenario_signature", "seed", "domain", "backend_kind"):
            if scenario.get(field) != row.get(field):
                raise ValueError("weight_scenario_identity_mismatch")
        if scenario.get("family") != row.get("family"):
            raise ValueError("weight_task_family_mismatch")
        cases.append(
            {
                "scenario_signature": row["scenario_signature"],
                "seed": row["seed"],
                "scenario_sha256": row["yaml_sha256"],
                "scenario_path": row["path"],
                "domain": row["domain"],
                "backend_kind": row["backend_kind"],
                "task_family": row.get("family", scenario.get("family")),
                "physical_source_cluster": sources[_key(row)],
                "score_weight": 1
                / (
                    len(domains)
                    * family_counts[row["domain"]]
                    * source_counts[(row["domain"], row["family"])]
                    * case_counts[source_groups[_key(row)]]
                ),
                "weight_rationale": "equal_domain_family_physical_source_then_case",
            }
        )
    weights = [row["score_weight"] for row in cases]
    result = {
        "schema_version": VERSION,
        "evaluation_version": "0.25.0",
        "suite_sha256": hashlib.sha256(raw).hexdigest(),
        "cases": cases,
        "weight_policy": "equal_domain_family_physical_source_then_case",
        "model_outcomes_used": False,
        "concentration": {
            "maximum_weight": max(weights),
            "maximum_single_case_score_contribution": 100 * max(weights),
            "weight_equivalent_case_count": 1 / math.fsum(w * w for w in weights),
            "interpretation": "weight_concentration_not_independent_sample_size",
            "cases_missing_physical_source_identity": sum(
                row["physical_source_cluster"] is None for row in cases
            ),
        },
    }
    result["manifest_sha256"] = _digest(result)
    return result


def validate_task_weights(
    manifest: dict, suite_contracts: dict
) -> dict[tuple[str, int], float]:
    """Validate frozen weights against authenticated suite contracts.

    ``suite_contracts`` is the output of ``compile_operational_suite``. The
    manifest digest provides integrity, not authority: callers must freeze and
    record the chosen manifest hash before comparing model results.
    """
    if manifest.get("schema_version") not in (VERSION, LEGACY_VERSION):
        raise ValueError("unsupported_weight_schema")
    if manifest.get("manifest_sha256") != _digest(
        {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    ):
        raise ValueError("weight_manifest_hash_mismatch")
    if (
        not suite_contracts.get("suite_sha256")
        or manifest.get("suite_sha256") != suite_contracts["suite_sha256"]
    ):
        raise ValueError("weight_suite_hash_mismatch")
    cases = _unique(manifest["cases"])
    expected = _unique(suite_contracts["contracts"])
    if cases.keys() != expected.keys():
        raise ValueError("weight_case_coverage_mismatch")
    default_expected = {}
    if (
        manifest["schema_version"] == VERSION
        and manifest.get("weight_policy")
        == "equal_domain_family_physical_source_then_case"
    ):
        domains = {row["domain"] for row in cases.values()}
        families = {(row["domain"], row["task_family"]) for row in cases.values()}
        groups = {}
        for key, row in cases.items():
            source = row["physical_source_cluster"]
            if source is None:
                source = ("unidentified_case", *key)
            groups[key] = (row["domain"], row["task_family"], source)
        family_counts = Counter(domain for domain, _ in families)
        source_counts = Counter(group[:2] for group in set(groups.values()))
        case_counts = Counter(groups.values())
        default_expected = {
            key: 1
            / (
                len(domains)
                * family_counts[row["domain"]]
                * source_counts[(row["domain"], row["task_family"])]
                * case_counts[groups[key]]
            )
            for key, row in cases.items()
        }
    weights = {}
    for key, row in cases.items():
        contract = expected[key]
        if (
            not contract.get("scenario_sha256")
            or row.get("scenario_sha256") != contract["scenario_sha256"]
        ):
            raise ValueError("weight_scenario_hash_mismatch")
        if row.get("domain") != contract.get("domain"):
            raise ValueError("weight_domain_mismatch")
        if manifest["schema_version"] == VERSION and (
            row.get("task_family") != contract.get("task_family")
            or row.get("physical_source_cluster")
            != contract.get("physical_source_cluster")
        ):
            raise ValueError("weight_source_lineage_mismatch")
        value = row.get("score_weight")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("invalid_score_weight")
        weight = float(value)
        if not math.isfinite(weight) or not 0 < weight <= 1:
            raise ValueError("invalid_score_weight")
        if default_expected and not math.isclose(
            weight, default_expected[key], rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError("default_weight_formula_mismatch")
        weights[key] = weight
    if not math.isclose(math.fsum(weights.values()), 1.0, abs_tol=1e-12, rel_tol=0):
        raise ValueError("score_weights_must_sum_to_one")
    return weights
