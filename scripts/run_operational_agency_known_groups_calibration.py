#!/usr/bin/env python3
"""Build a fail-closed runtime known-groups calibration artifact.

The artifact is diagnostic-only.  Scores are recomputed from authoritative
episode evidence and complete, uncapped per-action and action-group replays.
Missing experimental cells or per-domain positive controls produce a held
artifact with explicit blockers; this script never synthesizes scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.implementation_identity import implementation_identity  # noqa: E402
from evaluation.operational_agency import (  # noqa: E402
    DIMENSIONS,
    PROFILE_VERSION,
    operational_agency_profile_is_consistent,
)

DEFAULT_SLICE = REPO_ROOT / (
    "release/operate_v0_58_0_candidate/operate_v058_formal/agency/"
    "jsplib_known_groups/slice.json"
)
DEFAULT_EPISODES = (
    REPO_ROOT
    / "release/operate_v0_58_0_candidate/operate_v058_formal/agency/"
    "jsplib_known_groups/episodes.jsonl",
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "release/operate_v0_58_0_candidate/operate_v058_formal/agency/"
    "known_groups.json"
)

SCHEMA_VERSION = "operational-agency-known-groups-v1"
KNOWN_CELLS = frozenset(
    {
        "adaptive",
        "reactive",
        "adaptive_plan",
        "open_loop",
        "full_observation",
        "partial_observation",
        "wait_only",
        "random",
    }
)
COMPARISON_CELLS = {
    "adaptive_gt_reactive": ("adaptive", "reactive", "gt"),
    "adaptive_plan_gt_open_loop": ("adaptive_plan", "open_loop", "gt"),
    "full_observation_gte_partial": (
        "full_observation",
        "partial_observation",
        "gte",
    ),
}
WAIT_RANDOM_COMPARISON = "wait_random_near_zero"
NEAR_ZERO_EPSILON = 1e-9


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bound_path(repo_root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _resolve(repo_root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("bound path must be a non-empty string")
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _load_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSON object required: {path}:{line_number}")
        rows.append((line_number, value))
    return rows


def _int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _full_uncapped_attribution(counterfactual: object) -> bool:
    if not isinstance(counterfactual, Mapping):
        return False
    if counterfactual.get("per_action_capped") is not False:
        return False
    for prefix, entries_key in (
        ("per_action", "per_action"),
        ("per_action_group", "per_action_groups"),
    ):
        expected = _int(counterfactual.get(f"{prefix}_expected"))
        attempted = _int(counterfactual.get(f"{prefix}_attempted"))
        completed = _int(counterfactual.get(f"{prefix}_completed"))
        entries = counterfactual.get(entries_key)
        if (
            counterfactual.get(f"{prefix}_status") != "complete"
            or expected is None
            or expected < 0
            or attempted != expected
            or completed != expected
            or not isinstance(entries, list)
            or len(entries) != expected
            or counterfactual.get(f"{prefix}_failures") != []
        ):
            return False
    return True


def _agency_index(profile: Mapping[str, Any]) -> float:
    dimensions = profile["dimensions"]
    scores = [
        float(dimensions[name]["score"])
        if dimensions[name]["applicable"] is True
        else 0.0
        for name in DIMENSIONS
    ]
    return round(statistics.fmean(scores), 6)


def _known_group_contract(row: Mapping[str, Any]) -> tuple[list[str], dict[str, str], bool]:
    raw = row.get("operational_agency_known_groups")
    if isinstance(raw, Mapping):
        cells_value = raw.get("cells")
        cells = (
            list(dict.fromkeys(str(value) for value in cells_value))
            if isinstance(cells_value, list)
            else []
        )
        raw_pairs = raw.get("pair_ids")
        pair_ids = (
            {
                str(key): str(value)
                for key, value in raw_pairs.items()
                if str(key) and str(value)
            }
            if isinstance(raw_pairs, Mapping)
            else {}
        )
        return cells, pair_ids, raw.get("positive_control") is True
    agent_name = str(row.get("agent_name") or "")
    if agent_name in {"wait_only", "random"}:
        return [agent_name], {}, False
    return [], {}, False


def _physical_source_identity(row: Mapping[str, Any]) -> str:
    ledger = row.get("case_ledger")
    if not isinstance(ledger, Mapping):
        ledger = {}
    value = ledger.get("physical_source_lock") or row.get("physical_source_key")
    if value in (None, "", {}, []):
        return ""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _slice_identity_map(
    slice_payload: Mapping[str, Any],
) -> tuple[dict[str, dict[str, str]], list[str]]:
    rows = slice_payload.get("scenarios")
    if not isinstance(rows, list):
        raise ValueError("diagnostic slice requires a scenarios list")
    identities_by_signature: dict[str, dict[str, str]] = {}
    errors: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            errors.append("slice_non_object_row")
            continue
        signature = str(row.get("scenario_signature") or "")
        domain = str(row.get("domain") or "")
        physical_source = _physical_source_identity(row)
        if not signature or not domain or not physical_source:
            errors.append("slice_identity_domain_or_physical_source_missing")
            continue
        identity = {
            "domain": domain,
            "physical_source_identity": physical_source,
        }
        previous = identities_by_signature.get(signature)
        if previous is not None and previous != identity:
            errors.append("slice_signature_identity_collision")
            continue
        identities_by_signature[signature] = identity
    return identities_by_signature, sorted(set(errors))


def _block(
    blockers: list[dict[str, Any]], code: str, detail: str
) -> None:
    blockers.append({"code": code, "detail": detail})


def _comparison(
    *,
    name: str,
    left_cell: str,
    right_cell: str,
    relation: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    sides: dict[
        str, dict[tuple[str, str, str, int], list[Mapping[str, Any]]]
    ] = {
        left_cell: defaultdict(list),
        right_cell: defaultdict(list),
    }
    for row in rows:
        cells = row["cells"]
        if left_cell not in cells and right_cell not in cells:
            continue
        pair_id = row["pair_ids"].get(name)
        if not pair_id:
            continue
        key = (
            str(row["domain"]),
            str(row["physical_source_identity"]),
            str(pair_id),
            int(row["seed"]),
        )
        if left_cell in cells and right_cell in cells:
            sides[left_cell][key].append(row)
            sides[right_cell][key].append(row)
            continue
        if left_cell in cells:
            sides[left_cell][key].append(row)
        if right_cell in cells:
            sides[right_cell][key].append(row)
    all_keys = set(sides[left_cell]) | set(sides[right_cell])
    paired_deltas: list[dict[str, Any]] = []
    ambiguous_or_unmatched: list[str] = []
    for key in sorted(all_keys):
        left = sides[left_cell].get(key, [])
        right = sides[right_cell].get(key, [])
        if (
            len(left) != 1
            or len(right) != 1
            or left[0] is right[0]
            or left[0]["scenario_signature"] == right[0]["scenario_signature"]
        ):
            ambiguous_or_unmatched.append("|".join(str(value) for value in key))
            continue
        delta = float(left[0]["agency_index"]) - float(right[0]["agency_index"])
        paired_deltas.append(
            {
                "domain": key[0],
                "physical_source_identity": key[1],
                "pair_id": key[2],
                "seed": key[3],
                "left": left[0]["agency_index"],
                "right": right[0]["agency_index"],
                "delta": round(delta, 6),
            }
        )
    mean_delta = (
        round(statistics.fmean(row["delta"] for row in paired_deltas), 6)
        if paired_deltas
        else None
    )
    passed = bool(
        paired_deltas
        and not ambiguous_or_unmatched
        and mean_delta is not None
        and (mean_delta > 0.0 if relation == "gt" else mean_delta >= 0.0)
    )
    return {
        "passed": passed,
        "status": "passed" if passed else "failed_or_not_evaluable",
        "left_cell": left_cell,
        "right_cell": right_cell,
        "relation": relation,
        "statistic": "paired_fixed_six_dimension_zero_imputed_mean_v1",
        "n_pairs": len(paired_deltas),
        "mean_delta": mean_delta,
        "pairs": paired_deltas,
        "ambiguous_or_unmatched_pair_ids": ambiguous_or_unmatched,
    }


def build_known_groups_report(
    *,
    repo_root: Path,
    slice_path: Path,
    episode_paths: Sequence[Path],
    implementation_tree_sha256: str,
    required_domains: set[str] | None = None,
) -> dict[str, Any]:
    """Recompute a known-groups report from bound runtime episode files."""
    root = repo_root.resolve()
    slice_path = slice_path.resolve()
    normalized_episode_paths = [path.resolve() for path in episode_paths]
    slice_payload = _load_object(slice_path)
    identities_by_signature, slice_errors = _slice_identity_map(slice_payload)
    domains = (
        {str(value) for value in required_domains if str(value)}
        if required_domains is not None
        else {
            identity["domain"] for identity in identities_by_signature.values()
        }
    )
    blockers: list[dict[str, Any]] = []
    for code in slice_errors:
        _block(blockers, code, "diagnostic slice domain mapping is invalid")
    if not domains:
        _block(blockers, "required_domains_missing", "no calibration domains declared")

    selected: list[dict[str, Any]] = []
    selected_identities: Counter[tuple[str, str, int]] = Counter()
    profile_failures = 0
    attribution_failures = 0
    for episode_path in normalized_episode_paths:
        for line_number, row in _load_jsonl(episode_path):
            cells, pair_ids, positive_control = _known_group_contract(row)
            if not cells:
                continue
            signature = str(row.get("scenario_signature") or "")
            agent_name = str(row.get("agent_name") or "")
            seed = _int(row.get("seed"))
            slice_identity = identities_by_signature.get(signature, {})
            domain = slice_identity.get("domain", "")
            physical_source_identity = slice_identity.get(
                "physical_source_identity", ""
            )
            cell_contract_ok = bool(cells) and all(cell in KNOWN_CELLS for cell in cells)
            trajectory = row.get("trajectory_summary")
            counterfactual = row.get("counterfactual")
            profile_consistent = bool(
                isinstance(trajectory, Mapping)
                and operational_agency_profile_is_consistent(
                    trajectory,
                    counterfactual=counterfactual
                    if isinstance(counterfactual, Mapping)
                    else None,
                )
            )
            attribution_complete = _full_uncapped_attribution(counterfactual)
            profile_failures += int(not profile_consistent)
            attribution_failures += int(not attribution_complete)
            identity_valid = bool(
                signature
                and agent_name
                and seed is not None
                and domain
                and physical_source_identity
            )
            valid = bool(
                row.get("status") == "ok"
                and cell_contract_ok
                and identity_valid
                and profile_consistent
                and attribution_complete
            )
            profile = (
                trajectory.get("operational_agency_profile")
                if isinstance(trajectory, Mapping)
                else None
            )
            identity = (signature, agent_name, seed if seed is not None else -1)
            selected_identities[identity] += 1
            selected.append(
                {
                    "source": {
                        "path": _bound_path(root, episode_path),
                        "line": line_number,
                    },
                    "scenario_id": str(row.get("scenario_id") or ""),
                    "scenario_signature": signature,
                    "seed": seed,
                    "agent_name": agent_name,
                    "domain": domain,
                    "physical_source_identity": physical_source_identity,
                    "cells": cells,
                    "pair_ids": pair_ids,
                    "positive_control": positive_control,
                    "task_completed": bool(
                        isinstance(row.get("task_completion"), Mapping)
                        and row["task_completion"].get("completed") is True
                    ),
                    "profile_consistent": profile_consistent,
                    "full_uncapped_attribution": attribution_complete,
                    "causal_record_count": (
                        int(profile.get("causal_record_count") or 0)
                        if isinstance(profile, Mapping)
                        else 0
                    ),
                    "agency_index": (
                        _agency_index(profile)
                        if valid and isinstance(profile, Mapping)
                        else None
                    ),
                    "valid": valid,
                }
            )

    duplicate_count = sum(count - 1 for count in selected_identities.values() if count > 1)
    invalid_count = sum(not row["valid"] for row in selected)
    if duplicate_count:
        _block(
            blockers,
            "duplicate_episode_identity",
            f"{duplicate_count} duplicate selected episode identity binding(s)",
        )
    if invalid_count:
        _block(
            blockers,
            "invalid_episode_contract",
            f"{invalid_count} selected episode(s) lack authoritative evidence, full attribution, or identity binding",
        )
    valid_rows = [row for row in selected if row["valid"]]

    comparisons = {
        name: _comparison(
            name=name,
            left_cell=left,
            right_cell=right,
            relation=relation,
            rows=valid_rows,
        )
        for name, (left, right, relation) in COMPARISON_CELLS.items()
    }
    wait_rows = [row for row in valid_rows if "wait_only" in row["cells"]]
    random_rows = [row for row in valid_rows if "random" in row["cells"]]
    wait_random_rows = wait_rows + random_rows
    near_zero = bool(
        wait_rows
        and random_rows
        and all(
            float(row["agency_index"]) <= NEAR_ZERO_EPSILON
            and row["causal_record_count"] == 0
            for row in wait_random_rows
        )
    )
    comparisons[WAIT_RANDOM_COMPARISON] = {
        "passed": near_zero,
        "status": "passed" if near_zero else "failed_or_not_evaluable",
        "statistic": "maximum_fixed_six_dimension_zero_imputed_mean_v1",
        "epsilon": NEAR_ZERO_EPSILON,
        "n_wait": len(wait_rows),
        "n_random": len(random_rows),
        "max_agency_index": (
            max(float(row["agency_index"]) for row in wait_random_rows)
            if wait_random_rows
            else None
        ),
        "causal_record_count": sum(
            int(row["causal_record_count"]) for row in wait_random_rows
        ),
    }
    for name, comparison in comparisons.items():
        if comparison["passed"] is not True:
            _block(
                blockers,
                "known_groups_comparison_failed",
                f"{name} is failed or not evaluable from complete paired runtime episodes",
            )

    domain_controls: dict[str, dict[str, Any]] = {}
    for domain in sorted(domains):
        controls = [
            row
            for row in valid_rows
            if row["domain"] == domain
            and row["positive_control"]
            and row["task_completed"]
            and row["causal_record_count"] > 0
        ]
        causal_count = sum(int(row["causal_record_count"]) for row in controls)
        passed = bool(controls and causal_count > 0)
        domain_controls[domain] = {
            "passed": passed,
            "n_episodes": len(controls),
            "causal_record_count": causal_count,
            "authoritative_evidence_verified": bool(
                controls and all(row["profile_consistent"] for row in controls)
            ),
            "full_uncapped_attribution_verified": bool(
                controls and all(row["full_uncapped_attribution"] for row in controls)
            ),
            "native_task_completion_verified": bool(
                controls and all(row["task_completed"] for row in controls)
            ),
        }
        if not passed:
            _block(
                blockers,
                "domain_positive_control_missing",
                f"{domain} has no complete natural runtime causal positive control",
            )

    evidence_validation = {
        "authoritative_evidence_verified": bool(selected and profile_failures == 0),
        "full_uncapped_attribution_verified": bool(
            selected and attribution_failures == 0
        ),
        "n_invalid_selected_episodes": invalid_count,
        "n_selected_episodes": len(selected),
    }
    blockers = sorted(
        blockers,
        key=lambda row: (str(row["code"]), str(row["detail"])),
    )
    passed = bool(
        not blockers
        and evidence_validation["authoritative_evidence_verified"]
        and evidence_validation["full_uncapped_attribution_verified"]
        and all(row["passed"] is True for row in comparisons.values())
        and all(row["passed"] is True for row in domain_controls.values())
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "held_fail_closed",
        "diagnostic_only": True,
        "release_admission": False,
        "headline_score_included": False,
        "implementation_tree_sha256": implementation_tree_sha256,
        "profile_version": PROFILE_VERSION,
        "required_domains": sorted(domains),
        "input_bindings": {
            "slice": {
                "path": _bound_path(root, slice_path),
                "sha256": _sha256(slice_path),
            },
            "episodes": [
                {
                    "path": _bound_path(root, path),
                    "sha256": _sha256(path),
                    "n_rows": len(_load_jsonl(path)),
                }
                for path in normalized_episode_paths
            ],
        },
        "calibration_contract": {
            "agency_statistic": "fixed_six_dimension_zero_imputed_mean_v1",
            "pairing": (
                "explicit_domain_physical_source_seed_and_comparison_pair_id_v1"
            ),
            "near_zero_epsilon": NEAR_ZERO_EPSILON,
            "authoritative_profile_recomputation_required": True,
            "per_action_attribution_cap": None,
            "per_action_group_attribution_cap": None,
        },
        "evidence_validation": evidence_validation,
        "cell_counts": {
            cell: sum(cell in row["cells"] for row in valid_rows)
            for cell in sorted(KNOWN_CELLS)
        },
        "comparisons": comparisons,
        "domain_positive_controls": domain_controls,
        "selected_episode_summaries": selected,
        "blockers": blockers,
    }


def audit_known_groups_artifact(
    *,
    repo_root: Path,
    payload: Mapping[str, Any] | None,
    live_implementation_tree_sha256: str,
    required_domains: set[str],
) -> list[str]:
    """Recompute a bound artifact and return stable fail-closed error codes."""
    if not isinstance(payload, Mapping):
        return ["artifact_missing"]
    errors: list[str] = []
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("diagnostic_only") is not True
        or payload.get("release_admission") is not False
        or payload.get("headline_score_included") is not False
        or payload.get("profile_version") != PROFILE_VERSION
    ):
        errors.append("artifact_contract_invalid")
    if payload.get("implementation_tree_sha256") != live_implementation_tree_sha256:
        errors.append("implementation_binding_stale")
    if set(payload.get("required_domains") or []) != set(required_domains):
        errors.append("required_domain_coverage_mismatch")
    bindings = payload.get("input_bindings")
    try:
        if not isinstance(bindings, Mapping):
            raise ValueError("input bindings missing")
        slice_binding = bindings.get("slice")
        episode_bindings = bindings.get("episodes")
        if not isinstance(slice_binding, Mapping) or not isinstance(
            episode_bindings, list
        ):
            raise ValueError("input bindings malformed")
        root = repo_root.resolve()
        slice_path = _resolve(root, slice_binding.get("path")).resolve()
        if not slice_path.is_file() or _sha256(slice_path) != slice_binding.get(
            "sha256"
        ):
            raise ValueError("slice binding mismatch")
        episode_paths: list[Path] = []
        for binding in episode_bindings:
            if not isinstance(binding, Mapping):
                raise ValueError("episode binding malformed")
            path = _resolve(root, binding.get("path")).resolve()
            if (
                not path.is_file()
                or _sha256(path) != binding.get("sha256")
                or len(_load_jsonl(path)) != _int(binding.get("n_rows"))
            ):
                raise ValueError("episode binding mismatch")
            episode_paths.append(path)
        expected = build_known_groups_report(
            repo_root=root,
            slice_path=slice_path,
            episode_paths=episode_paths,
            implementation_tree_sha256=live_implementation_tree_sha256,
            required_domains=set(required_domains),
        )
        if dict(payload) != expected:
            errors.append("artifact_recomputation_mismatch")
    except (OSError, ValueError, json.JSONDecodeError):
        errors.append("input_binding_invalid")
    if payload.get("status") != "passed":
        errors.append("calibration_not_passed")
    return list(dict.fromkeys(errors))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", type=Path, default=DEFAULT_SLICE)
    parser.add_argument(
        "--episodes", nargs="+", type=Path, default=list(DEFAULT_EPISODES)
    )
    parser.add_argument("--required-domains", nargs="*")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = build_known_groups_report(
        repo_root=REPO_ROOT,
        slice_path=args.slice,
        episode_paths=args.episodes,
        implementation_tree_sha256=implementation_identity(REPO_ROOT)[
            "implementation_tree_sha256"
        ],
        required_domains=(
            set(args.required_domains) if args.required_domains is not None else None
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "blocker_codes": sorted(
                    {row["code"] for row in report["blockers"]}
                ),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
