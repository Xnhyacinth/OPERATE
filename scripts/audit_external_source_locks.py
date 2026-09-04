#!/usr/bin/env python3
"""Verify local source locks for requested upstream benchmark conversions."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "release"
    / "dt_sched_bench_v0_52_0_candidate"
    / "external_source_locks.json"
)

SPECS = {
    "cityflow_official": {
        "root": "works/CityFlow",
        "url": "https://github.com/cityflow-project/CityFlow",
        "asset_globs": ("examples/**/*.json",),
    },
    "flatland_official": {
        "root": "works/flatland-rl",
        "url": "https://github.com/flatland-association/flatland-rl",
        "asset_globs": ("env_data/**/*.pkl",),
    },
    "citylearn_official": {
        "root": "works/CityLearn",
        "url": "https://github.com/intelligent-environments-lab/CityLearn",
        "asset_globs": (
            "data/datasets/citylearn_challenge_2022_phase_3/schema.json",
            "data/datasets/citylearn_challenge_2022_phase_3/*.csv",
        ),
    },
    "alibaba_cluster_trace_official": {
        "root": "works/clusterdata",
        "url": "https://github.com/alibaba/clusterdata",
        "license_globs": ("cluster-trace-gpu-v2020/LICENSE",),
        "asset_globs": ("cluster-trace-gpu-v2020/data/**/*",),
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _assets(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    return sorted(
        {path for pattern in patterns for path in root.glob(pattern) if path.is_file()}
    )


def build(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    rows = []
    for source_id, spec in SPECS.items():
        root = repo_root / spec["root"]
        commit = _git(root, "rev-parse", "HEAD") if root.is_dir() else None
        remote = _git(root, "remote", "get-url", "origin") if commit else None
        license_globs = spec.get("license_globs", ("LICENSE*",))
        license_paths = (
            sorted(
                {
                    path
                    for pattern in license_globs
                    for path in root.glob(pattern)
                    if path.is_file()
                }
            )
            if root.is_dir()
            else []
        )
        assets = _assets(root, spec["asset_globs"]) if root.is_dir() else []
        clean = _git(root, "status", "--porcelain") == "" if commit else False
        verified = bool(commit and license_paths and assets and clean)
        rows.append(
            {
                "source_id": source_id,
                "status": "source_lock_verified" if verified else "source_lock_blocked",
                "source_url": spec["url"],
                "local_root": spec["root"],
                "git_commit": commit,
                "git_remote": remote,
                "git_worktree_clean": clean,
                "license_files": [
                    {
                        "path": path.relative_to(repo_root).as_posix(),
                        "sha256": _sha256(path),
                    }
                    for path in license_paths
                    if path.is_file()
                ],
                "asset_count": len(assets),
                "assets": [
                    {
                        "path": path.relative_to(repo_root).as_posix(),
                        "sha256": _sha256(path),
                        "size_bytes": path.stat().st_size,
                    }
                    for path in assets
                ],
                "source_lock_verified": verified,
            }
        )
    return {
        "schema_version": "0.1",
        "scope": "requested_external_source_locks",
        "status": (
            "verified" if all(row["source_lock_verified"] for row in rows) else "blocked"
        ),
        "sources": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "n_verified": sum(
                    row["source_lock_verified"] for row in report["sources"]
                ),
                "n_sources": len(report["sources"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
