#!/usr/bin/env python3
"""Audit source locks and SUMO preflight for proposed RESCO replacements."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "works" / "RESCO"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "release"
    / "dt_sched_bench_v0_52_0_candidate"
    / "resco_replacement_sources.json"
)
SOURCE_URL = "https://github.com/Pi-Star-Lab/RESCO.git"
SOURCE_COMMIT = "f1ed9a174f8de41fc9d8689373b836bc882570dc"
ENVIRONMENTS = (
    "cologne1",
    "cologne3",
    "cologne8",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(source: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _asset_paths(env_dir: Path, name: str) -> list[Path]:
    config = env_dir / f"{name}.sumocfg"
    root = ET.parse(config).getroot()
    values = []
    for tag in ("net-file", "route-files"):
        node = root.find(f".//{tag}")
        if node is not None:
            for value in str(node.get("value") or "").split(","):
                if value.strip():
                    values.append(env_dir / value.strip())
    return [config, *values, env_dir / "LICENSE"]


def _demand_entities(route_path: Path) -> int:
    root = ET.parse(route_path).getroot()
    return sum(
        len(root.findall(f".//{tag}"))
        for tag in ("vehicle", "flow", "trip", "person", "personFlow")
    )


def audit(
    source: Path,
    *,
    run_sumo: bool = True,
    environments: tuple[str, ...] = ENVIRONMENTS,
) -> dict[str, Any]:
    # CLI callers commonly pass a repository-relative source path.  Resolve it
    # once so every asset path can be safely serialized relative to REPO_ROOT
    # (and so the audit does not fail before it can report a missing/invalid
    # source lock).
    source = source.expanduser().resolve()
    selected_environments = tuple(
        dict.fromkeys(
            name.strip()
            for name in environments
            if isinstance(name, str) and name.strip()
        )
    )
    if not selected_environments:
        raise ValueError("at least one RESCO environment is required")
    commit = _git_commit(source)
    environments = []
    for name in selected_environments:
        env_dir = source / "resco_benchmark" / "environments" / name
        assets = _asset_paths(env_dir, name)
        missing = [str(path) for path in assets if not path.is_file()]
        net = env_dir / f"{name}.net.xml"
        route_assets = [path for path in assets if path.suffix in {".xml", ".gz"} and "net.xml" not in path.name and "sumocfg" not in path.name and path.name != "LICENSE"]
        demand_entities = sum(_demand_entities(path) for path in route_assets)
        tls_count = len(ET.parse(net).getroot().findall(".//tlLogic")) if net.is_file() else 0
        preflight = {"status": "not_run", "returncode": None, "stderr": ""}
        if run_sumo and not missing:
            proc = subprocess.run(
                [
                    str(REPO_ROOT / ".venv" / "bin" / "sumo"),
                    "-c",
                    str(env_dir / f"{name}.sumocfg"),
                    "--begin",
                    "0",
                    "--end",
                    "30",
                    "--no-step-log",
                    "true",
                    "--no-warnings",
                    "true",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            preflight = {
                "status": "passed" if proc.returncode == 0 else "failed",
                "returncode": proc.returncode,
                "stderr": proc.stderr[-1000:],
            }
        locked = (
            not missing
            and commit == SOURCE_COMMIT
            and tls_count > 0
            and demand_entities > 0
        )
        status = "blocked"
        if not missing and commit == SOURCE_COMMIT and tls_count > 0 and demand_entities == 0:
            status = "blocked_missing_demand_source"
        elif locked and preflight["status"] == "passed":
            status = "source_and_runtime_validated_protocol_mapping_pending"
        environments.append(
            {
                "source_id": f"resco:{name}",
                "environment": name,
                "status": status,
                "source_lock": {
                    "url": SOURCE_URL,
                    "commit": commit,
                    "expected_commit": SOURCE_COMMIT,
                    "license_file": str((env_dir / "LICENSE").relative_to(REPO_ROOT)),
                    "files": [
                        {
                            "path": str(path.relative_to(REPO_ROOT)),
                            "sha256": _sha256(path),
                        }
                        for path in assets
                        if path.is_file()
                    ],
                    "lock_strategy": "git_commit+per_file_sha256+environment_license",
                    "provenance_complete": locked,
                    "missing_files": missing,
                },
                "sumo_preflight": preflight,
                "n_traffic_lights": tls_count,
                "n_demand_entities": demand_entities,
                "remaining_gates": [
                    "benchmark_native_corridor_and_tls_mapping",
                    "native_control_action_sensitivity",
                    "four_agent_behavioral_calibration",
                    "four_level_difficulty_calibration",
                    "semantic_duplicate_detection",
                    "multi_model_repeat_trial_calibration",
                ],
            }
        )
    all_ready = all(
        row["status"] == "source_and_runtime_validated_protocol_mapping_pending"
        for row in environments
    )
    return {
        "schema_version": "0.1",
        "source": {"url": SOURCE_URL, "commit": commit, "expected_commit": SOURCE_COMMIT},
        "status": (
            "source_and_runtime_validated_protocol_mapping_pending"
            if all_ready
            else "partial"
        ),
        "environments": environments,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-run-sumo", action="store_true")
    parser.add_argument(
        "--environment",
        action="append",
        dest="environments",
        metavar="NAME",
        help="audit one named RESCO environment; repeat to select several",
    )
    args = parser.parse_args()
    report = audit(
        args.source,
        run_sumo=not args.no_run_sumo,
        environments=tuple(args.environments or ENVIRONMENTS),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "n_environments": len(report["environments"])}, indent=2))
    return 0 if report["status"] != "blocked" else 2


if __name__ == "__main__":
    raise SystemExit(main())
