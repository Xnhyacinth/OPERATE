#!/usr/bin/env python3
"""Lock a CityLearn dataset and every runtime asset referenced by its schema."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_citylearn_sources import (  # noqa: E402
    DEFAULT_SOURCE_ROOT,
    PACKAGE_VERSION_POLICY,
)

UPSTREAM_ROOT = REPO_ROOT / "works" / "CityLearn"
DEFAULT_OUTPUT = REPO_ROOT / ".hl" / "artifacts" / "citylearn_source_lock.json"
_RUNTIME_ASSET_SUFFIXES = {".csv", ".epw", ".json", ".pkl", ".pth"}
_DERIVATION_ASSETS = (
    "lbl-tracking_the_sun-res-pv.csv",
    "battery_choices.yaml",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_sha256(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _stable_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _implementation_tree_sha256(root: Path) -> str:
    graph = {
        path.relative_to(root / "citylearn").as_posix(): _sha256(path)
        for path in sorted((root / "citylearn").rglob("*.py"))
    }
    return _stable_sha256(graph)


def _reported_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _environment_identity(implementation_root: Path) -> dict[str, str]:
    commit = subprocess.run(
        ["git", "-C", str(implementation_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        [
            "git",
            "-C",
            str(implementation_root),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("CityLearn implementation checkout must be clean")
    return {
        "citylearn_version": importlib.metadata.version("citylearn"),
        "torch_version": importlib.metadata.version("torch"),
        "implementation_commit": commit,
        "implementation_tree_sha256": _implementation_tree_sha256(implementation_root),
    }


def _runtime_asset_graph(source_root: Path) -> dict[str, dict[str, Any]]:
    source_root = source_root.resolve()
    schema_path = source_root / "schema.json"
    if not schema_path.is_file():
        raise FileNotFoundError(f"missing CityLearn schema: {schema_path}")

    graph: dict[str, dict[str, Any]] = {
        "schema.json": {
            "path": schema_path,
            "schema_references": {"$document"},
        }
    }
    visited_json: set[Path] = set()

    def register(reference: str, pointer: str, document: Path) -> None:
        candidate = (document.parent / reference).resolve()
        try:
            relative = candidate.relative_to(source_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"CityLearn schema asset escapes dataset root: {reference}"
            ) from exc
        if not candidate.is_file():
            raise FileNotFoundError(
                f"schema-referenced CityLearn asset missing: {relative}"
            )
        row = graph.setdefault(
            relative,
            {"path": candidate, "schema_references": set()},
        )
        row["schema_references"].add(pointer)
        if candidate.suffix.lower() == ".json":
            walk_document(candidate, f"{pointer}::$document")

    def visit(value: object, pointer: str, document: Path) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                visit(child, f"{pointer}.{key}", document)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{pointer}[{index}]", document)
        elif (
            isinstance(value, str)
            and Path(value).suffix.lower() in _RUNTIME_ASSET_SUFFIXES
        ):
            register(value, pointer, document)

    def walk_document(document: Path, pointer: str = "$") -> None:
        document = document.resolve()
        if document in visited_json:
            return
        visited_json.add(document)
        payload = json.loads(document.read_text(encoding="utf-8"))
        visit(payload, pointer, document)

    walk_document(schema_path)
    return graph


def _derivation_asset_graph(source_root: Path) -> dict[str, Path]:
    misc_root = source_root.resolve().parent.parent / "misc"
    graph = {name: misc_root / name for name in _DERIVATION_ASSETS}
    missing = [name for name, path in graph.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing CityLearn derivation assets: {missing}")
    return graph


def _prefixed_digest(path: Path | None) -> str | None:
    return None if path is None else f"sha256:{_sha256(path)}"


def build(
    source_root: Path = DEFAULT_SOURCE_ROOT,
    *,
    dataset_id: str | None = None,
    repo_root: Path = REPO_ROOT,
    implementation_root: Path = UPSTREAM_ROOT,
    environment_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build a lock; it does not establish runtime consumption or headroom."""

    source_root = source_root.resolve()
    dataset_id = dataset_id or source_root.name
    runtime_graph = _runtime_asset_graph(source_root)
    derivation_graph = _derivation_asset_graph(source_root)
    identity = dict(
        environment_identity
        if environment_identity is not None
        else _environment_identity(implementation_root)
    )
    required_identity = {
        "citylearn_version",
        "torch_version",
        "implementation_commit",
        "implementation_tree_sha256",
    }
    missing_identity = sorted(required_identity - identity.keys())
    if missing_identity:
        raise ValueError(f"missing CityLearn environment identity: {missing_identity}")

    runtime_files = {
        relative: {
            "path": _reported_path(row["path"], repo_root),
            "sha256": _sha256(row["path"]),
            "schema_references": sorted(row["schema_references"]),
        }
        for relative, row in sorted(runtime_graph.items())
    }
    derivation_files = {
        name: {
            "path": _reported_path(path, repo_root),
            "sha256": _sha256(path),
        }
        for name, path in sorted(derivation_graph.items())
    }
    files = {
        row["path"]: row["sha256"]
        for row in [*runtime_files.values(), *derivation_files.values()]
    }
    runtime_hashes = {
        relative: row["sha256"] for relative, row in runtime_files.items()
    }
    derivation_hashes = {
        relative: row["sha256"] for relative, row in derivation_files.items()
    }
    schema = json.loads((source_root / "schema.json").read_text(encoding="utf-8"))
    buildings = schema.get("buildings", {})
    included_buildings = sorted(
        name
        for name, row in buildings.items()
        if not isinstance(row, Mapping) or row.get("include", True)
    )
    by_name = {Path(name).name: row["path"] for name, row in runtime_graph.items()}
    simulation_paths = [
        row["path"]
        for name, row in runtime_graph.items()
        if Path(name).name.startswith("Building_")
        and Path(name).suffix.lower() == ".csv"
    ]
    model_paths = [
        row["path"]
        for name, row in runtime_graph.items()
        if Path(name).suffix.lower() == ".pth"
    ]
    optional_absent = [
        name
        for name in ("carbon_intensity.csv", "pricing.csv")
        if name not in by_name
    ]
    runtime_asset_graph_sha256 = _stable_sha256(runtime_hashes)
    complete_asset_graph_sha256 = _stable_sha256(
        {"derivation": derivation_hashes, "runtime": runtime_hashes}
    )
    citylearn_version = str(identity["citylearn_version"])
    torch_version = str(identity["torch_version"])
    implementation_commit = str(identity["implementation_commit"])
    return {
        "source_id": dataset_id,
        "source_url": "https://github.com/intelligent-environments-lab/CityLearn",
        "dataset_source_url": (
            "https://github.com/intelligent-environments-lab/CityLearn"
        ),
        "license": "MIT",
        "license_verified": True,
        "terms_verified": True,
        "lock_strategy": "git_commit+recursive_schema_runtime_graph_sha256",
        "package_version": f"citylearn=={citylearn_version}",
        "package_version_policy": PACKAGE_VERSION_POLICY,
        "torch_version": f"torch=={torch_version}",
        "torch_version_policy": PACKAGE_VERSION_POLICY,
        "git_commit_or_release_tag": (f"v{citylearn_version}@{implementation_commit}"),
        "implementation_tree_sha256": str(identity["implementation_tree_sha256"]),
        "dataset_release_or_challenge_version": dataset_id,
        "schema_or_dataset_name": f"{dataset_id}/schema.json",
        "dataset_identity": {
            "dataset_id": dataset_id,
            "dataset_root": _reported_path(source_root, repo_root),
            "schema_path": runtime_files["schema.json"]["path"],
            "runtime_asset_count": len(runtime_files),
            "runtime_asset_graph_sha256": runtime_asset_graph_sha256,
            "complete_asset_graph_sha256": complete_asset_graph_sha256,
        },
        "schema_sha256": _prefixed_digest(source_root / "schema.json"),
        "weather_file_or_timeseries_lock": _prefixed_digest(
            Path(by_name["weather.csv"]) if "weather.csv" in by_name else None
        ),
        "simulation_file_sha256": (
            f"sha256:{_bundle_sha256(simulation_paths)}" if simulation_paths else None
        ),
        "dynamics_model_file_sha256": (
            f"sha256:{_bundle_sha256(model_paths)}" if model_paths else None
        ),
        "pricing_file_sha256": _prefixed_digest(
            Path(by_name["pricing.csv"]) if "pricing.csv" in by_name else None
        ),
        "carbon_intensity_file_sha256": _prefixed_digest(
            Path(by_name["carbon_intensity.csv"])
            if "carbon_intensity.csv" in by_name
            else None
        ),
        "pv_sizing_file_sha256": _prefixed_digest(
            derivation_graph["lbl-tracking_the_sun-res-pv.csv"]
        ),
        "battery_sizing_file_sha256": _prefixed_digest(
            derivation_graph["battery_choices.yaml"]
        ),
        "building_cluster": included_buildings,
        "episode_window": [
            schema.get("simulation_start_time_step"),
            schema.get("simulation_end_time_step"),
        ],
        "citylearn_offline": True,
        "random_episode_split": False,
        "rolling_episode_split": False,
        "simulator_seed": 2022,
        "runtime_files": runtime_files,
        "derivation_files": derivation_files,
        "optional_runtime_assets_absent": optional_absent,
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--dataset-id")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    payload = build(args.source_root, dataset_id=args.dataset_id)
    if not args.execute:
        print(json.dumps(payload, indent=2))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
