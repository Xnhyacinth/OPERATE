#!/usr/bin/env python3
"""Audit the recoverability of the fifteen canonical NGSIM candidates."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from domains.autonomous_driving.data.archives import find_member_in_archive  # noqa: E402
from domains.autonomous_driving.data.canonical_recovery import (  # noqa: E402
    CANONICAL_CANDIDATES,
    HISTORICAL_RECIPE_COMMIT,
    OFFICIAL_ARCHIVE_REQUIREMENTS,
    NGSIMArchiveRequirement,
    NGSIMCanonicalCandidate,
)
from domains.autonomous_driving.data.contracts import file_sha256  # noqa: E402
from domains.autonomous_driving.data.ngsim import verify_bundle  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / ".hl/artifacts/ngsim_canonical_recovery_ledger_v1.json"
DEFAULT_ASSET_ROOT = REPO_ROOT / "works/autonomous_driving/ngsim/recovery/raw"
DEFAULT_BUNDLE_ROOT = REPO_ROOT / "works/autonomous_driving/ngsim/recovery/bundles"
DEFAULT_REPLAY_ROOT = REPO_ROOT / "works/autonomous_driving/ngsim/recovery/replay"


def verify_archive(
    requirement: NGSIMArchiveRequirement,
    asset_root: Path,
) -> dict[str, Any]:
    """Verify the local archive and locate its authoritative member.

    Uses :func:`find_member_in_archive` which recurses one level into nested
    zips, so members inside a zip-within-zip are found by suffix or basename
    matching without requiring the outer namelist to carry the logical path.
    """
    archive_path = asset_root / requirement.archive_name
    blockers: list[str] = []
    result: dict[str, Any] = {
        "path": str(archive_path),
        "archive_present": archive_path.is_file(),
        "archive_hash_verified": False,
        "member_present": False,
        "member_hash_verified": False,
        "source_bytes_verified": False,
    }
    if requirement.archive_sha256 is None:
        blockers.append("expected_archive_sha256_missing")
    if requirement.authoritative_member_sha256 is None:
        blockers.append("expected_authoritative_member_sha256_missing")
    if not archive_path.is_file():
        blockers.append("official_archive_bytes_missing")
        result["blockers"] = sorted(blockers)
        return result
    observed_archive_sha = file_sha256(archive_path)
    result["observed_archive_sha256"] = observed_archive_sha
    if requirement.archive_sha256 is not None:
        if observed_archive_sha == requirement.archive_sha256:
            result["archive_hash_verified"] = True
        else:
            blockers.append("official_archive_sha256_mismatch")
    try:
        # Suffix-only pass first; hash compared separately so blockers stay distinct.
        member_info = find_member_in_archive(
            archive_path,
            suffixes=requirement.authoritative_members,
        )
        result.update(
            {
                "member_present": True,
                "observed_member": member_info["logical_nested_path"],
                "observed_member_sha256": member_info["sha256"],
                "observed_member_byte_size": member_info["byte_size"],
            }
        )
        if requirement.authoritative_member_sha256 is not None:
            if member_info["sha256"] == requirement.authoritative_member_sha256:
                result["member_hash_verified"] = True
            else:
                blockers.append("authoritative_member_sha256_mismatch")
    except (OSError, zipfile.BadZipFile):
        blockers.append("official_archive_invalid_zip")
    except ValueError:
        blockers.append("authoritative_member_missing_or_ambiguous")
    result["source_bytes_verified"] = bool(
        result["archive_hash_verified"] and result["member_hash_verified"]
    )
    result["blockers"] = sorted(set(blockers))
    return result


def probe_remote(requirement: NGSIMArchiveRequirement) -> dict[str, Any]:
    request = urllib.request.Request(
        requirement.archive_url,
        headers={"Range": "bytes=0-0", "User-Agent": "OPERATE-NGSIM-Recovery/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return {
                "status": "reachable",
                "http_status": int(getattr(response, "status", 200)),
                "content_range": str(response.headers.get("Content-Range") or ""),
            }
    except urllib.error.HTTPError as error:
        return {"status": "blocked", "http_status": error.code}
    except OSError as error:
        return {"status": "unreachable", "error": f"{type(error).__name__}: {error}"}


def _candidate_slug(candidate_id: str) -> str:
    return candidate_id.replace(":", "_")


def _verify_candidate_bundle(
    candidate: NGSIMCanonicalCandidate,
    bundle_root: Path,
    verifier: Callable[[Path], Mapping[str, Any]],
) -> dict[str, Any]:
    path = bundle_root / _candidate_slug(candidate.candidate_id)
    if not path.is_dir():
        return {"status": "missing", "path": str(path), "verified": False}
    try:
        evidence = dict(verifier(path))
        fixture = json.loads(
            (path / "runtime/fixture.json").read_text(encoding="utf-8")
        )
        derivation = dict(fixture.get("derivation") or {})
        identity_verified = (
            str(derivation.get("candidate_id") or "") == candidate.candidate_id
            and str(derivation.get("source_window_sha256") or "")
            == candidate.source_window_sha256
            and str((fixture.get("ego") or {}).get("vehicle_id") or "")
            == candidate.ego_actor_id
        )
        return {
            "status": "verified" if identity_verified else "identity_mismatch",
            "path": str(path),
            "verified": identity_verified,
            "bundle_evidence": evidence,
        }
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {
            "status": "invalid",
            "path": str(path),
            "verified": False,
            "error": f"{type(error).__name__}: {error}",
        }


def _replay_index(replay_root: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not replay_root.is_dir():
        return rows
    for path in sorted(replay_root.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict):
            continue
        candidate_id = str(value.get("candidate_id") or "")
        if candidate_id and candidate_id not in rows:
            rows[candidate_id] = {**value, "evidence_path": str(path)}
    return rows


def build_recovery_ledger(
    *,
    asset_root: Path,
    bundle_root: Path,
    replay_root: Path,
    candidates: Sequence[NGSIMCanonicalCandidate] = CANONICAL_CANDIDATES,
    requirements: Mapping[str, NGSIMArchiveRequirement] = OFFICIAL_ARCHIVE_REQUIREMENTS,
    remote_probes: Mapping[str, Mapping[str, Any]] | None = None,
    bundle_verifier: Callable[[Path], Mapping[str, Any]] = verify_bundle,
) -> dict[str, Any]:
    archive_rows = {
        recording_id: {
            **requirement.to_dict(),
            "local_verification": verify_archive(requirement, asset_root),
            **(
                {"remote_probe": dict(remote_probes[recording_id])}
                if remote_probes and recording_id in remote_probes
                else {}
            ),
        }
        for recording_id, requirement in sorted(requirements.items())
    }
    replays = _replay_index(replay_root)
    candidate_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        requirement = requirements[candidate.recording_id]
        archive = archive_rows[candidate.recording_id]
        local = dict(archive["local_verification"])
        bundle = _verify_candidate_bundle(candidate, bundle_root, bundle_verifier)
        replay = replays.get(candidate.candidate_id)
        replay_verified = bool(
            replay
            and replay.get("status") == "verified"
            and replay.get("deterministic_semantic_replay") is True
            and replay.get("candidate_id") == candidate.candidate_id
        )
        blockers: list[str] = []
        if not candidate.metadata_recipe_complete:
            blockers.append("metadata_recipe_incomplete")
        if requirement.archive_sha256 is None:
            blockers.append("archive_hash_lock_missing")
        if requirement.authoritative_member_sha256 is None:
            blockers.append("member_hash_lock_missing")
        if not local.get("source_bytes_verified"):
            blockers.append("canonical_source_bytes_unverified")
        if not bundle.get("verified"):
            blockers.append("canonical_bundle_not_rebuilt")
        if not replay_verified:
            blockers.append("fresh_native_replay_missing")
        blockers.append("license_metadata_discrepancy_pending")
        next_actions = []
        if not candidate.metadata_recipe_complete:
            next_actions.append("rederive_candidate_recipe_from_verified_source_bytes")
        if not local.get("source_bytes_verified"):
            next_actions.append("obtain_and_verify_official_archive_and_member_bytes")
        if not bundle.get("verified"):
            next_actions.append(
                "rebuild_candidate_bundle_from_current_deterministic_pipeline"
            )
        if not replay_verified:
            next_actions.append("run_fresh_native_calibration_and_semantic_replay")
        next_actions.append("resolve_dataset_license_metadata_discrepancy")
        candidate_rows.append(
            {
                **candidate.to_dict(),
                "evidence_layers": {
                    "metadata": candidate.metadata_recipe_complete,
                    "source_bytes": bool(local.get("source_bytes_verified")),
                    "bundle": bool(bundle.get("verified")),
                    "fresh_native_replay": replay_verified,
                },
                "bundle_verification": bundle,
                "replay_verification": (
                    {
                        "status": "verified" if replay_verified else "invalid",
                        "evidence_path": str((replay or {}).get("evidence_path") or ""),
                    }
                    if replay
                    else {"status": "missing"}
                ),
                "final_disposition": "held_repair",
                "formal_core_allowed": False,
                "blockers": sorted(set(blockers)),
                "next_actions": next_actions,
            }
        )
    layer_counts = Counter()
    for row in candidate_rows:
        for name, passed in row["evidence_layers"].items():
            layer_counts[f"{name}_{'verified' if passed else 'missing'}"] += 1
    return {
        "schema_version": "ngsim_canonical_recovery_ledger_v1",
        "status": "held_repair",
        "formal_core_allowed": False,
        "admission_evidence_policy": (
            "historical metadata is identity/recipe evidence only; provider results are excluded; "
            "fresh source-byte, bundle, and native replay verification is mandatory"
        ),
        "historical_recipe_commit": HISTORICAL_RECIPE_COMMIT,
        "asset_root": str(asset_root),
        "bundle_root": str(bundle_root),
        "replay_root": str(replay_root),
        "summary": {
            "candidate_count": len(candidate_rows),
            "terminal_count": len(candidate_rows),
            "recording_counts": dict(
                sorted(Counter(row["recording_id"] for row in candidate_rows).items())
            ),
            "layer_counts": dict(sorted(layer_counts.items())),
            "unresolved_count": 0,
        },
        "official_archive_plan": list(archive_rows.values()),
        "candidates": candidate_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument("--replay-root", type=Path, default=DEFAULT_REPLAY_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--probe-remote", action="store_true")
    args = parser.parse_args()
    remote = (
        {
            recording_id: probe_remote(requirement)
            for recording_id, requirement in OFFICIAL_ARCHIVE_REQUIREMENTS.items()
        }
        if args.probe_remote
        else None
    )
    report = build_recovery_ledger(
        asset_root=args.asset_root,
        bundle_root=args.bundle_root,
        replay_root=args.replay_root,
        remote_probes=remote,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], sort_keys=True))
    return 2 if report["summary"]["unresolved_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
