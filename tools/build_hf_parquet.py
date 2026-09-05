#!/usr/bin/env python3
"""Export the OPERATE Full and Lite suites as reversible Parquet datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any


EXPORT_SCHEMA_VERSION = "operate-hf-parquet-v1"
PUBLIC_EXPORT_SCHEMA_VERSION = "operate-hf-parquet-v2"
_PUBLIC_OMITTED_COLUMNS = {
    "release_id", "suite_id", "track", "is_official_full_denominator",
    "declared_leaderboard_eligible", "selection_algorithm", "status",
    "core_disposition", "construct_contract",
    "suite_template_json",
}
DEFAULT_RELEASE_DIR = Path("release/operate_v0_61_0")
PARQUET_NAME = "test-00000-of-00001.parquet"
_SCENARIOS_SENTINEL = {"__operate_scenarios__": True}
_ROW_FIELDS = (
    "scenario_id",
    "path",
    "domain",
    "backend_kind",
    "family",
    "difficulty_level",
    "difficulty_mode",
    "horizon_ticks",
    "seed",
    "physical_source_key",
    "source_denominator_key",
    "structural_fingerprint",
    "semantic_fingerprint",
    "scenario_signature",
    "yaml_sha256",
    "status",
    "core_disposition",
    "construct_contract",
)


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise RuntimeError(
            "Parquet export requires pyarrow; install the publication/HF extra "
            "or run with `uv run --with pyarrow`."
        ) from exc
    return pa, pq


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _pretty_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _safe_scenario_path(repo_root: Path, declared_path: str) -> Path:
    relative = PurePosixPath(declared_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe scenario path: {declared_path!r}")
    resolved_root = repo_root.resolve()
    resolved = (resolved_root / Path(*relative.parts)).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"scenario path escapes repository: {declared_path!r}") from exc
    return resolved


def _suite_template(suite: dict[str, Any]) -> list[list[Any]]:
    if not isinstance(suite.get("scenarios"), list):
        raise ValueError("suite scenarios must be a list")
    return [
        [key, _SCENARIOS_SENTINEL if key == "scenarios" else value]
        for key, value in suite.items()
    ]


def _validate_rows(rows: list[dict[str, Any]], *, suite_path: Path) -> None:
    if not rows:
        raise ValueError(f"suite contains no scenarios: {suite_path}")
    for field in _ROW_FIELDS:
        missing = [index for index, row in enumerate(rows) if field not in row]
        if missing:
            raise ValueError(
                f"suite row field {field!r} missing at indices {missing[:5]}"
            )
    for field in ("scenario_id", "path"):
        values = [str(row[field]) for row in rows]
        if len(values) != len(set(values)):
            raise ValueError(f"suite contains duplicate {field}")


def _load_suite(suite_path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = suite_path.read_bytes()
    suite = json.loads(raw)
    if not isinstance(suite, dict):
        raise ValueError(f"suite must be a JSON object: {suite_path}")
    _validate_rows(suite.get("scenarios", []), suite_path=suite_path)
    if _pretty_json_bytes(suite) != raw:
        raise ValueError(
            f"suite is not in the reproducible two-space JSON format: {suite_path}"
        )
    return raw, suite


def _arrow_schema(pa: Any) -> Any:
    string = pa.string()
    return pa.schema(
        [
            pa.field("export_schema_version", string, nullable=False),
            pa.field("subset", string, nullable=False),
            pa.field("row_index", pa.int32(), nullable=False),
            pa.field("release_id", string, nullable=False),
            pa.field("suite_id", string, nullable=False),
            pa.field("track", string, nullable=False),
            pa.field("is_official_full_denominator", pa.bool_(), nullable=False),
            pa.field("declared_leaderboard_eligible", pa.bool_(), nullable=False),
            pa.field("selection_algorithm", string, nullable=False),
            pa.field("suite_file_sha256", string, nullable=False),
            pa.field("parent_core_suite_sha256", string, nullable=False),
            pa.field("suite_template_json", string, nullable=False),
            pa.field("scenario_metadata_json", string, nullable=False),
            pa.field("scenario_yaml", string, nullable=False),
            pa.field("scenario_id", string, nullable=False),
            pa.field("path", string, nullable=False),
            pa.field("domain", string, nullable=False),
            pa.field("backend_kind", string, nullable=False),
            pa.field("family", string, nullable=False),
            pa.field("difficulty_level", string, nullable=False),
            pa.field("difficulty_mode", string, nullable=False),
            pa.field("horizon_ticks", pa.int64(), nullable=False),
            pa.field("seed", pa.int64(), nullable=False),
            pa.field("physical_source_key", string, nullable=False),
            pa.field("source_denominator_key", string, nullable=False),
            pa.field("structural_fingerprint", string, nullable=False),
            pa.field("semantic_fingerprint", string, nullable=False),
            pa.field("scenario_signature", string, nullable=False),
            pa.field("yaml_sha256", string, nullable=False),
            pa.field("status", string, nullable=False),
            pa.field("core_disposition", string, nullable=False),
            pa.field("construct_contract", string, nullable=False),
        ]
    )


def _suite_identity(
    suite: dict[str, Any], *, subset: str, suite_sha256: str
) -> dict[str, Any]:
    if subset == "full":
        release_id = str(suite["release_id"])
        return {
            "release_id": release_id,
            "suite_id": release_id,
            "track": "full",
            "is_official_full_denominator": True,
            "declared_leaderboard_eligible": bool(suite["leaderboard_eligible"]),
            "selection_algorithm": "",
            "parent_core_suite_sha256": suite_sha256,
        }
    if subset == "lite":
        return {
            "release_id": str(suite["parent_release_id"]),
            "suite_id": str(suite["suite_id"]),
            "track": str(suite["track"]),
            "is_official_full_denominator": False,
            "declared_leaderboard_eligible": bool(
                suite["formal_full_leaderboard_eligible"]
            ),
            "selection_algorithm": str(suite["selection_algorithm"]),
            "parent_core_suite_sha256": str(suite["parent_core_suite_sha256"]),
        }
    raise ValueError(f"unknown subset: {subset!r}")


def _records_for_suite(
    repo_root: Path,
    suite_path: Path,
    *,
    subset: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    suite_raw, suite = _load_suite(suite_path)
    suite_sha256 = _sha256_bytes(suite_raw)
    identity = _suite_identity(suite, subset=subset, suite_sha256=suite_sha256)
    template_json = _compact_json(_suite_template(suite))
    records: list[dict[str, Any]] = []
    for row_index, row in enumerate(suite["scenarios"]):
        scenario_path = _safe_scenario_path(repo_root, str(row["path"]))
        yaml_bytes = scenario_path.read_bytes()
        yaml_sha256 = _sha256_bytes(yaml_bytes)
        if yaml_sha256 != row["yaml_sha256"]:
            raise ValueError(
                f"scenario YAML hash mismatch for {row['path']}: "
                f"declared {row['yaml_sha256']}, actual {yaml_sha256}"
            )
        record = {
            "export_schema_version": EXPORT_SCHEMA_VERSION,
            "subset": subset,
            "row_index": row_index,
            **identity,
            "suite_file_sha256": suite_sha256,
            "suite_template_json": template_json,
            "scenario_metadata_json": _compact_json(row),
            "scenario_yaml": yaml_bytes.decode("utf-8"),
        }
        record.update({field: row[field] for field in _ROW_FIELDS})
        record["horizon_ticks"] = int(record["horizon_ticks"])
        record["seed"] = int(record["seed"])
        records.append(record)
    return records, suite


def _write_parquet(
    records: list[dict[str, Any]],
    output_path: Path,
    *,
    subset: str,
    public_metadata: bool = False,
) -> None:
    pa, pq = _require_pyarrow()
    version = PUBLIC_EXPORT_SCHEMA_VERSION if public_metadata else EXPORT_SCHEMA_VERSION
    schema = _arrow_schema(pa)
    if public_metadata:
        schema = pa.schema([field for field in schema if field.name not in _PUBLIC_OMITTED_COLUMNS])
        records = [{**record, "export_schema_version": version} for record in records]
    schema = schema.with_metadata(
        {
            b"operate.export_schema_version": version.encode(),
            b"operate.subset": subset.encode(),
            b"operate.rebuild_contract": b"suite-template+ordered-row-json+yaml-bytes",
        }
    )
    if public_metadata:
        schema = schema.with_metadata({
            **(schema.metadata or {}),
            b"operate.suite_template_json": _single_value(records, "suite_template_json").encode("utf-8"),
        })
    table = pa.Table.from_pylist(records, schema=schema)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        output_path,
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
        row_group_size=1024,
        data_page_version="2.0",
        version="2.6",
    )


def _validate_lite_membership(
    full: dict[str, Any], lite: dict[str, Any], *, core_sha256: str
) -> None:
    if lite.get("parent_core_suite_sha256") != core_sha256:
        raise ValueError("Lite parent_core_suite_sha256 does not match Full bytes")
    full_by_id = {row["scenario_id"]: row for row in full["scenarios"]}
    mismatches = [
        row["scenario_id"]
        for row in lite["scenarios"]
        if full_by_id.get(row["scenario_id"]) != row
    ]
    if mismatches:
        raise ValueError(
            "Lite contains rows that are absent from or differ from Full: "
            + ", ".join(mismatches[:5])
        )


def build_exports(
    repo_root: Path,
    output_dir: Path,
    *,
    release_dir: Path = DEFAULT_RELEASE_DIR,
    public_metadata: bool = False,
) -> dict[str, Any]:
    """Build deterministic Full/Lite Parquet files and their integrity manifest."""

    release_path = release_dir if release_dir.is_absolute() else repo_root / release_dir
    core_path = release_path / "core_suite.json"
    lite_path = release_path / "lite_suite.json"
    core_raw, full = _load_suite(core_path)
    _, lite = _load_suite(lite_path)
    core_sha256 = _sha256_bytes(core_raw)
    _validate_lite_membership(full, lite, core_sha256=core_sha256)

    artifacts: dict[str, Any] = {}
    for subset, suite_path in (("full", core_path), ("lite", lite_path)):
        records, suite = _records_for_suite(repo_root, suite_path, subset=subset)
        output_path = output_dir / subset / PARQUET_NAME
        _write_parquet(records, output_path, subset=subset, public_metadata=public_metadata)
        artifacts[subset] = {
            "path": output_path.relative_to(output_dir).as_posix(),
            "sha256": _sha256_bytes(output_path.read_bytes()),
            "size_bytes": output_path.stat().st_size,
            "n_rows": len(records),
            "suite_file_sha256": records[0]["suite_file_sha256"],
            "suite_id": records[0]["suite_id"],
            "n_physical_sources": len(
                {row["physical_source_key"] for row in suite["scenarios"]}
            ),
        }
        if public_metadata:
            artifacts[subset].pop("suite_id")

    manifest = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "release_id": str(full["release_id"]),
        "artifacts": artifacts,
    }
    if public_metadata:
        manifest["schema_version"] = PUBLIC_EXPORT_SCHEMA_VERSION
        manifest.pop("release_id")
    manifest_path = output_dir / "parquet_manifest.json"
    manifest_path.write_bytes(_pretty_json_bytes(manifest))
    return manifest


def _single_value(records: list[dict[str, Any]], field: str) -> Any:
    values = {record[field] for record in records}
    if len(values) != 1:
        raise ValueError(f"Parquet contains multiple {field} values")
    return next(iter(values))


def _payload_from_template(
    template_json: str, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    template = json.loads(template_json)
    if not isinstance(template, list):
        raise ValueError("suite template must be a list")
    payload: dict[str, Any] = {}
    scenario_slots = 0
    for item in template:
        if not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], str):
            raise ValueError("invalid suite template entry")
        key, value = item
        if key in payload:
            raise ValueError(f"duplicate suite template key: {key}")
        if value == _SCENARIOS_SENTINEL:
            scenario_slots += 1
            payload[key] = rows
        else:
            payload[key] = value
    if scenario_slots != 1 or "scenarios" not in payload:
        raise ValueError("suite template must contain exactly one scenarios slot")
    return payload


def _write_rebuilt_file(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"refusing to overwrite different file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def rebuild_parquet(parquet_path: Path, output_dir: Path) -> Path:
    """Rebuild the exact suite JSON and scenario YAML tree from one Parquet file."""

    _, pq = _require_pyarrow()
    table = pq.read_table(parquet_path)
    records = table.to_pylist()
    if not records:
        raise ValueError("Parquet contains no rows")
    version = _single_value(records, "export_schema_version")
    if version not in {EXPORT_SCHEMA_VERSION, PUBLIC_EXPORT_SCHEMA_VERSION}:
        raise ValueError("unsupported Parquet export schema")
    subset = _single_value(records, "subset")
    if subset not in {"full", "lite"}:
        raise ValueError(f"unsupported subset: {subset!r}")
    if version == PUBLIC_EXPORT_SCHEMA_VERSION:
        template_bytes = (table.schema.metadata or {}).get(b"operate.suite_template_json")
        if not template_bytes:
            raise ValueError("public suite template metadata missing")
        template_json = template_bytes.decode("utf-8")
    else:
        template_json = _single_value(records, "suite_template_json")
    expected_suite_sha256 = _single_value(records, "suite_file_sha256")

    indices = [int(record["row_index"]) for record in records]
    if sorted(indices) != list(range(len(records))) or len(indices) != len(set(indices)):
        raise ValueError("row_index must be unique and contiguous from zero")
    records.sort(key=lambda record: int(record["row_index"]))

    scenario_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    yaml_outputs: list[tuple[Path, bytes]] = []
    for record in records:
        row = json.loads(record["scenario_metadata_json"])
        if not isinstance(row, dict):
            raise ValueError("scenario_metadata_json must decode to an object")
        for field in _ROW_FIELDS:
            if version == PUBLIC_EXPORT_SCHEMA_VERSION and field in _PUBLIC_OMITTED_COLUMNS:
                continue
            expected = int(record[field]) if field in {"horizon_ticks", "seed"} else record[field]
            if row.get(field) != expected:
                raise ValueError(
                    f"flat column {field} differs from scenario metadata for "
                    f"{record['scenario_id']}"
                )
        scenario_id = str(row["scenario_id"])
        declared_path = str(row["path"])
        if scenario_id in seen_ids or declared_path in seen_paths:
            raise ValueError("Parquet contains duplicate scenario_id or path")
        seen_ids.add(scenario_id)
        seen_paths.add(declared_path)
        yaml_bytes = str(record["scenario_yaml"]).encode("utf-8")
        if _sha256_bytes(yaml_bytes) != row["yaml_sha256"]:
            raise ValueError(f"scenario YAML hash mismatch for {declared_path}")
        yaml_outputs.append(
            (_safe_scenario_path(output_dir, declared_path), yaml_bytes)
        )
        scenario_rows.append(row)

    payload = _payload_from_template(template_json, scenario_rows)
    suite_bytes = _pretty_json_bytes(payload)
    if _sha256_bytes(suite_bytes) != expected_suite_sha256:
        raise ValueError("rebuilt suite JSON does not match its declared SHA-256")

    suite_name = "core_suite.json" if subset == "full" else "lite_suite.json"
    suite_output = output_dir / suite_name
    for path, content in yaml_outputs:
        _write_rebuilt_file(path, content)
    _write_rebuilt_file(suite_output, suite_bytes)
    return suite_output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export", help="build Full and Lite Parquet")
    export.add_argument("--repo-root", type=Path, default=Path("."))
    export.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--public-metadata", action="store_true",
                        help="omit redundant internal display columns; preserve reversible payloads")

    rebuild = subparsers.add_parser("rebuild", help="restore suite JSON and YAML")
    rebuild.add_argument("--parquet", type=Path, required=True)
    rebuild.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "export":
        manifest = build_exports(
            args.repo_root.resolve(),
            args.output_dir,
            release_dir=args.release_dir,
            public_metadata=args.public_metadata,
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0
    output = rebuild_parquet(args.parquet, args.output_dir)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
