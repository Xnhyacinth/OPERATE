#!/usr/bin/env python3
"""Materialize identity-safe Protocol-2.1 difficulty relabel candidates.

Replay calibration may prove that a scenario's declared difficulty is wrong.
The old identity-bound evidence cannot simply be reused under a new label. This
utility writes a new staging YAML, recomputes its domain-native signature and
working-set fingerprints, clears stale admission evidence, and emits a
migration ledger that requires a full Protocol-2.1 replay.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.difficulty_contract import (  # noqa: E402
    DIFFICULTY_CONTRACT_VERSION,
    DIFFICULTY_REQUIREMENTS,
)
from core.scenario_validator import validate_scenario_yaml  # noqa: E402
from core.suite_identity import verify_scenario_row_against_yaml  # noqa: E402
from runner.resume import recompute_signature_with_seed  # noqa: E402
from scripts.assemble_protocol21_repaired_working_set import (  # noqa: E402
    _build_row,
    _load_body,
)

_STALE_ROW_FIELDS = {
    "admission_fingerprint",
    "agentic_contract",
    "native_behavioral_validation",
    "observed_depth_validation",
    "protocol21_admission_status",
    "source_grounded_validation",
    "strategy_depth_validation",
    "task_contract_validation",
}
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")
_FIXED_POINT_BLOCKER = "difficulty_relabel_two_cycle_detected"
_FIXED_POINT_DISPOSITION = "held_difficulty_fixed_point_unresolved"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _rows(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    rows = payload.get(key)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"artifact must contain a {key} list")
    return rows


def _identity(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("scenario_id") or ""),
        str(row.get("scenario_signature") or ""),
    )


def _safe_leaf(value: str) -> str:
    cleaned = _SAFE_COMPONENT.sub("_", value).strip("_.-")
    return cleaned or "scenario"


def _normalized_physical_lock(
    lock: Any, *, repo_root: Path
) -> dict[str, Any]:
    if not isinstance(lock, dict):
        return {}
    normalized = copy.deepcopy(lock)
    assets = []
    for asset in lock.get("required_source_assets") or []:
        if not isinstance(asset, dict):
            continue
        raw_path = Path(str(asset.get("declared_path") or ""))
        resolved = raw_path if raw_path.is_absolute() else repo_root / raw_path
        try:
            declared = resolved.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            declared = resolved.resolve().as_posix()
        normalized_asset = copy.deepcopy(asset)
        normalized_asset["declared_path"] = declared
        assets.append(normalized_asset)
    normalized["required_source_assets"] = sorted(
        assets,
        key=lambda row: json.dumps(
            row, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ),
    )
    return normalized


def _physical_locks_match(
    old_lock: Any, new_lock: Any, *, repo_root: Path
) -> bool:
    """Compare current asset graphs with the legacy single-SHA trace lock."""
    old_normalized = _normalized_physical_lock(old_lock, repo_root=repo_root)
    new_normalized = _normalized_physical_lock(new_lock, repo_root=repo_root)
    if old_normalized and old_normalized == new_normalized:
        return True
    if not isinstance(old_lock, str) or len(old_lock) != 64:
        return False
    try:
        int(old_lock, 16)
    except ValueError:
        return False
    assets = new_normalized.get("required_source_assets") or []
    return len(assets) == 1 and assets[0].get("sha256") == old_lock.lower()


def _new_identity(
    row: dict[str, Any], *, calibrated_level: str
) -> tuple[str, str]:
    old_id = str(row.get("scenario_id") or "")
    old_signature = str(row.get("scenario_signature") or "")
    digest = hashlib.sha256(
        f"{old_id}\0{old_signature}".encode()
    ).hexdigest()[:8]
    leaf = _safe_leaf(Path(str(row.get("path") or old_id)).stem)
    new_leaf = f"{leaf}__{digest}__relabel_v1"
    new_id = "/".join(
        (
            str(row.get("domain") or ""),
            str(row.get("family") or ""),
            str(row.get("difficulty_mode") or ""),
            calibrated_level,
            new_leaf,
        )
    )
    return new_id, new_leaf


def _new_fixed_point_identity(
    row: dict[str, Any], *, calibrated_level: str
) -> tuple[str, str]:
    """Return one canonical identity for a terminal relabel replay.

    Ordinary relabel identities preserve the full migration chain.  That is
    useful provenance for normal one-way corrections, but it makes an
    evidence-dependent two-cycle grow an unbounded ``__relabel_v1`` suffix.
    A terminal fixed-point attempt instead binds one clean identity to the
    current row/evidence pair and explicitly forbids a second relabel.
    """

    old_id = str(row.get("scenario_id") or "")
    old_signature = str(row.get("scenario_signature") or "")
    digest = hashlib.sha256(
        f"terminal_single_replay_v1\0{old_id}\0{old_signature}\0{calibrated_level}".encode()
    ).hexdigest()[:12]
    leaf = _safe_leaf(Path(str(row.get("path") or old_id)).stem)
    leaf = re.sub(r"(?:__[0-9a-f]{8}__relabel_v1)+$", "", leaf)
    leaf = re.sub(r"__fixedpoint_[0-9a-f]{12}_v1$", "", leaf)
    new_leaf = f"{leaf}__fixedpoint_{digest}_v1"
    new_id = "/".join(
        (
            str(row.get("domain") or ""),
            str(row.get("family") or ""),
            str(row.get("difficulty_mode") or ""),
            calibrated_level,
            new_leaf,
        )
    )
    return new_id, new_leaf


def _clear_stale_admission(row: dict[str, Any]) -> None:
    for field in _STALE_ROW_FIELDS:
        row.pop(field, None)
    row["status"] = "working_set"
    row["leaderboard_eligible"] = False


def _mark_terminal_fixed_point_hold(
    row: dict[str, Any],
    *,
    declared_level: str,
    calibrated_level: str,
    cycle_levels: list[str],
) -> dict[str, Any]:
    """Return a fail-closed row for a replay-proven difficulty two-cycle."""

    held = copy.deepcopy(row)
    reasons = sorted(
        {
            str(reason)
            for reason in held.get("reason_codes") or []
            if str(reason)
        }
        | {_FIXED_POINT_BLOCKER}
    )
    held.update(
        {
            "status": "held_repair",
            "core_disposition": "held_repair",
            "leaderboard_eligible": False,
            "reason_codes": reasons,
        }
    )
    lineage = copy.deepcopy(held.get("protocol21_lineage") or {})
    lineage_reasons = sorted(
        {
            str(reason)
            for reason in lineage.get("reason_codes") or []
            if str(reason)
        }
        | {_FIXED_POINT_BLOCKER}
    )
    lineage.update(
        {
            "ready": False,
            "status": "held_repair",
            "reason_codes": lineage_reasons,
            "difficulty_fixed_point": {
                "status": _FIXED_POINT_DISPOSITION,
                "declared_difficulty_level": declared_level,
                "calibrated_difficulty_level": calibrated_level,
                "cycle_levels": list(cycle_levels),
                "terminal_after_replay": True,
            },
        }
    )
    held["protocol21_lineage"] = lineage
    case_ledger = copy.deepcopy(held.get("case_ledger") or {})
    case_ledger["behavioral_validation"] = _FIXED_POINT_DISPOSITION
    constraints = copy.deepcopy(case_ledger.get("selection_constraints") or {})
    constraint_reasons = sorted(
        {
            str(reason)
            for reason in constraints.get("reason_codes") or []
            if str(reason)
        }
        | {_FIXED_POINT_BLOCKER}
    )
    constraints.update(
        {
            "core_eligible": False,
            "leaderboard_eligible": False,
            "reason_codes": constraint_reasons,
        }
    )
    case_ledger["selection_constraints"] = constraints
    case_ledger["complexity_tags"] = sorted(
        {
            str(tag)
            for tag in case_ledger.get("complexity_tags") or []
            if str(tag) and str(tag) != "pending_full_protocol21_gates"
        }
        | {"difficulty_fixed_point_terminal_hold"}
    )
    case_ledger["keep_rationale"] = (
        "Replay proved a terminal Basic/Medium difficulty two-cycle; the row "
        "is held and no further identity relabel is permitted."
    )
    held["case_ledger"] = case_ledger
    return held


def _write_yaml(path: Path, body: dict[str, Any]) -> None:
    encoded = yaml.safe_dump(body, sort_keys=False, allow_unicode=True)
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        existing = yaml.safe_load(path.read_text(encoding="utf-8"))
        old_lineage = (
            existing.get("difficulty_calibration_lineage")
            if isinstance(existing, dict)
            else None
        ) or {}
        new_lineage = body.get("difficulty_calibration_lineage") or {}
        identity_fields = (
            "source_scenario_id",
            "source_scenario_signature",
            "contract_version",
            "declared_difficulty_level",
            "calibrated_difficulty_level",
        )
        if any(
            old_lineage.get(field) != new_lineage.get(field)
            for field in identity_fields
        ):
            raise ValueError(
                f"refusing to overwrite unrelated staging YAML: {path}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")


def materialize_relabels(
    *,
    source_suite: dict[str, Any],
    strategy_depth: dict[str, Any],
    strategy_artifact_path: Path,
    staging_root: Path,
    repo_root: Path = REPO_ROOT,
    only_status: str | None = None,
    terminal_after_replay: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if source_suite.get("status") != "working_set":
        raise ValueError("source suite must have status=working_set")
    if source_suite.get("leaderboard_eligible") is not False:
        raise ValueError("source suite must not be leaderboard eligible")
    source_rows = _rows(source_suite, "scenarios")
    strategy_rows = _rows(strategy_depth, "samples")
    by_identity = {_identity(row): row for row in strategy_rows}
    if len(by_identity) != len(strategy_rows):
        raise ValueError("strategy-depth identities must be unique")
    strategy_sha256 = _sha256(strategy_artifact_path)
    output_rows: list[dict[str, Any]] = []
    migrations: list[dict[str, Any]] = []
    fixed_point_holds: list[dict[str, Any]] = []
    n_filtered_out = 0
    for source_row in source_rows:
        if only_status is not None and source_row.get("status") != only_status:
            output_rows.append(copy.deepcopy(source_row))
            n_filtered_out += 1
            continue
        old_identity = _identity(source_row)
        strategy = by_identity.get(old_identity)
        if strategy is None:
            raise ValueError(f"strategy-depth row missing: {old_identity[0]}")
        calibration = strategy.get("difficulty_calibration") or {}
        calibrated_level = str(
            calibration.get("calibrated_difficulty_level") or ""
        )
        declared_level = str(source_row.get("difficulty_level") or "")
        relabel = (
            calibration.get("version") == DIFFICULTY_CONTRACT_VERSION
            and calibration.get("status") == "passed"
            and str(calibration.get("declared_difficulty_level") or "")
            == declared_level
            and calibration.get("declared_level_matches_evidence") is False
            and calibrated_level in DIFFICULTY_REQUIREMENTS
            and calibrated_level != declared_level
        )
        if not relabel:
            output_rows.append(copy.deepcopy(source_row))
            continue

        old_path = Path(str(source_row.get("path") or ""))
        if not old_path.is_absolute():
            old_path = repo_root / old_path
        source_verification_errors = verify_scenario_row_against_yaml(
            source_row, path=old_path
        )
        if source_verification_errors:
            raise ValueError(
                "source row/YAML mismatch: "
                f"{old_identity[0]}: {source_verification_errors}"
            )
        body = _load_body(old_path)
        body.pop("_physical_source_lock", None)
        fixed_point_lineage = body.get("difficulty_fixed_point_lineage") or {}
        if fixed_point_lineage.get("terminal_after_replay") is True:
            input_level = str(
                fixed_point_lineage.get("input_declared_difficulty_level") or ""
            )
            cycle_levels = [input_level, declared_level, calibrated_level]
            fixed_point_holds.append(
                {
                    "scenario_id": old_identity[0],
                    "scenario_signature": old_identity[1],
                    "declared_difficulty_level": declared_level,
                    "calibrated_difficulty_level": calibrated_level,
                    "disposition": _FIXED_POINT_DISPOSITION,
                    "blocker": _FIXED_POINT_BLOCKER,
                    "cycle_levels": cycle_levels,
                }
            )
            output_rows.append(
                _mark_terminal_fixed_point_hold(
                    source_row,
                    declared_level=declared_level,
                    calibrated_level=calibrated_level,
                    cycle_levels=cycle_levels,
                )
            )
            continue
        if terminal_after_replay:
            new_id, new_leaf = _new_fixed_point_identity(
                source_row, calibrated_level=calibrated_level
            )
        else:
            new_id, new_leaf = _new_identity(
                source_row, calibrated_level=calibrated_level
            )
        body["scenario_id"] = new_id
        body["seed_id"] = new_id
        body["difficulty_level"] = calibrated_level
        config = dict(body.get("backend_config") or {})
        config.setdefault(
            "source_denominator_key",
            str(
                source_row.get("source_denominator_key")
                or (source_row.get("case_ledger") or {}).get(
                    "source_denominator_key"
                )
                or ""
            ),
        )
        body["backend_config"] = config
        body["difficulty_calibration_lineage"] = {
            "contract_version": DIFFICULTY_CONTRACT_VERSION,
            "declared_difficulty_level": declared_level,
            "calibrated_difficulty_level": calibrated_level,
            "source_scenario_id": old_identity[0],
            "source_scenario_signature": old_identity[1],
            "strategy_artifact_sha256": strategy_sha256,
            "requires_full_protocol21_replay": True,
        }
        if terminal_after_replay:
            body["difficulty_fixed_point_lineage"] = {
                "policy_version": "terminal_single_replay_v1",
                "canonical_source_scenario_id": old_identity[0],
                "canonical_source_scenario_signature": old_identity[1],
                "input_declared_difficulty_level": declared_level,
                "evidence_selected_difficulty_level": calibrated_level,
                "terminal_after_replay": True,
                "mismatch_disposition": (
                    "held_difficulty_fixed_point_unresolved"
                ),
            }
        body.pop("scenario_signature", None)
        body["scenario_signature"] = recompute_signature_with_seed(
            body, int(body.get("seed") or 0)
        )
        new_path = (
            staging_root
            / str(source_row.get("domain") or "unknown")
            / str(source_row.get("family") or "unknown")
            / str(source_row.get("difficulty_mode") or "unknown")
            / calibrated_level
            / f"{new_leaf}.yaml"
        )
        validation_errors = validate_scenario_yaml(body, new_path)
        if validation_errors:
            raise ValueError(
                f"relabelled scenario invalid: {new_id}: {validation_errors}"
            )
        _write_yaml(new_path, body)
        verified_body = _load_body(new_path)
        old_lock = (source_row.get("case_ledger") or {}).get(
            "physical_source_lock"
        )
        new_lock = verified_body.get("_physical_source_lock")
        if not _physical_locks_match(
            old_lock, new_lock, repo_root=repo_root
        ):
            raise ValueError(
                f"relabelled physical source lock changed: {new_id}"
            )
        base_for_build = copy.deepcopy(source_row)
        base_ledger = copy.deepcopy(base_for_build.get("case_ledger") or {})
        base_ledger["physical_source_lock"] = copy.deepcopy(new_lock)
        base_for_build["case_ledger"] = base_ledger
        new_row = _build_row(
            body=verified_body,
            path=new_path,
            base=base_for_build,
            repo_root=repo_root,
        )
        _clear_stale_admission(new_row)
        verification_errors = verify_scenario_row_against_yaml(
            new_row, path=new_path
        )
        if verification_errors:
            raise ValueError(
                f"relabelled row/YAML mismatch: {new_id}: {verification_errors}"
            )
        output_rows.append(new_row)
        migrations.append(
            {
                "old_scenario_id": old_identity[0],
                "old_scenario_signature": old_identity[1],
                "old_path": _relative(old_path, repo_root),
                "old_yaml_sha256": _sha256(old_path),
                "declared_difficulty_level": declared_level,
                "new_scenario_id": new_row["scenario_id"],
                "new_scenario_signature": new_row["scenario_signature"],
                "new_path": new_row["path"],
                "new_yaml_sha256": _sha256(new_path),
                "calibrated_difficulty_level": calibrated_level,
                "strategy_artifact_sha256": strategy_sha256,
                "requires_full_protocol21_replay": True,
                "terminal_after_replay": terminal_after_replay,
            }
        )

    identities = [_identity(row) for row in output_rows]
    if len(identities) != len(set(identities)):
        raise ValueError("relabelled working-set identities must be unique")
    working_set = copy.deepcopy(source_suite)
    fixed_point_reason_codes = (
        [_FIXED_POINT_BLOCKER] if fixed_point_holds else []
    )
    requires_full_replay = bool(migrations)
    working_set.update(
        {
            "status": "working_set",
            "leaderboard_eligible": False,
            "release_ready": False,
            "n_scenarios": len(output_rows),
            "scenarios": output_rows,
            "difficulty_relabel_migration": {
                "version": "identity_safe_relabel_v1",
                "strategy_artifact_sha256": strategy_sha256,
                "n_relabelled": len(migrations),
                "n_fixed_point_held": len(fixed_point_holds),
                "status": (
                    "held_repair" if fixed_point_holds else "complete"
                ),
                "selection_eligible": not fixed_point_holds,
                "reason_codes": fixed_point_reason_codes,
                "requires_full_protocol21_replay": requires_full_replay,
            },
        }
    )
    ledger = {
        "schema_version": "1.0",
        "status": "held_repair" if fixed_point_holds else "complete",
        "pipeline": "protocol21_identity_safe_difficulty_relabel_v1",
        "difficulty_contract_version": DIFFICULTY_CONTRACT_VERSION,
        "strategy_artifact": _relative(strategy_artifact_path, repo_root),
        "strategy_artifact_sha256": strategy_sha256,
        "n_source": len(source_rows),
        "n_relabelled": len(migrations),
        "n_unchanged": len(source_rows) - len(migrations),
        "n_filtered_out": n_filtered_out,
        "n_fixed_point_held": len(fixed_point_holds),
        "only_status": only_status,
        "terminal_after_replay": terminal_after_replay,
        "selection_eligible": not fixed_point_holds,
        "reason_codes": fixed_point_reason_codes,
        "requires_full_protocol21_replay": requires_full_replay,
        "migrations": migrations,
        "fixed_point_holds": fixed_point_holds,
    }
    return working_set, ledger


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-suite", type=Path, required=True)
    parser.add_argument("--strategy-depth", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--only-status")
    parser.add_argument("--terminal-after-replay", action="store_true")
    args = parser.parse_args()
    try:
        working_set, ledger = materialize_relabels(
            source_suite=json.loads(args.source_suite.read_text(encoding="utf-8")),
            strategy_depth=json.loads(
                args.strategy_depth.read_text(encoding="utf-8")
            ),
            strategy_artifact_path=args.strategy_depth.resolve(),
            staging_root=args.staging_root.resolve(),
            only_status=args.only_status,
            terminal_after_replay=args.terminal_after_replay,
        )
        for path, payload in ((args.output, working_set), (args.ledger, ledger)):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": ledger["status"],
                "n_source": ledger["n_source"],
                "n_relabelled": ledger["n_relabelled"],
                "n_unchanged": ledger["n_unchanged"],
                "n_fixed_point_held": ledger["n_fixed_point_held"],
                "selection_eligible": ledger["selection_eligible"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
