#!/usr/bin/env python3
"""Build a manifest-backed OPERATE runtime bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.implementation_identity import implementation_identity  # noqa: E402
from core.protocol21_evidence import canonicalize_repo_owned_paths  # noqa: E402

DEFAULT_RELEASE = REPO_ROOT / "release" / "operate_v0_61_0"
DEFAULT_OUTPUT = REPO_ROOT / "data_operate_v061"
DEFAULT_REPO_ID = "Xnhyacinth/OPERATE"
FORMAL_RESULT_TREE_INDEX_NAME = "FORMAL_RESULT_TREE_INDEX.json"
FORMAL_RESULT_TREE_INDEX_SCHEMA = "operate-formal-result-tree-index-v1"
BACKEND_LINKS = {
    "CityLearn": "citylearn",
    "JSPLIB-Instances": "jsplib",
    "M5": "m5",
    "OR-Gym": "or_gym",
    "OpenDSS-IEEE13": "opendss_ieee13",
    "PGLib-OPF": "pglib_opf",
    "PyVRP-Instances": "pyvrp_instances",
    "RESCO": "resco",
    "RTS-GMLC": "rts_gmlc",
    "VRPLIB": "vrplib",
    "nrel-microgrid": "nrel_microgrid",
    "sumo_ingolstadt": "sumo_ingolstadt",
    "DynaSchedBench": "dynasched",
}
_EXCLUDED_PARTS = frozenset({".git", "__pycache__", ".pytest_cache"})
_EXACT_SOURCE_SPECS = {
    "m5": {
        "backend_kind": "orgym_invmgmt",
        "contract_roles": ("derivation_input", "runtime_input"),
        "dataset": "m5_forecasting_accuracy",
        "delivery": "bundle",
        "paths": {
            "works/M5/calendar.csv",
            "works/M5/sales_train_evaluation.csv",
            "works/M5/sell_prices.csv",
        },
        "metadata": {
            "works/M5/source_lock.json": (
                "271c94965d27bf74b0d66ba89e71b5bc239ddc5192ce99305bbac256a848a9b3"
            ),
        },
        "provenance_variants": (
            {
                "license": "Kaggle competition rules + OR-Gym MIT",
                "lock_strategy": (
                    "kaggle_competition_terms+file_sha256+orgym_git_commit+"
                    "env_config_hash"
                ),
                "upstream_commit": (
                    "m5-forecasting-accuracy 2020-06-01 files;"
                    "orgym:0b18d16e569e2db70e83f09e867b53bdb4b87298"
                ),
                "url": "https://www.kaggle.com/competitions/m5-forecasting-accuracy",
            },
            {
                "license": "Kaggle competition rules + OR-Gym MIT",
                "lock_strategy": (
                    "kaggle_competition_terms+file_sha256+orgym_git_commit+"
                    "env_config_hash"
                ),
                "upstream_commit": (
                    "kaggle:m5-forecasting-accuracy:2020-06-01-files;"
                    "orgym:0b18d16e569e2db70e83f09e867b53bdb4b87298"
                ),
                "url": "https://www.kaggle.com/competitions/m5-forecasting-accuracy",
            },
        ),
        "redistribution": {
            "dataset": "m5_forecasting_accuracy",
            "license": "M5 competition terms; redistribution permission confirmed",
            "lock_strategy": "competition_release+raw_file_sha256",
            "notice": (
                "Exact release-referenced M5 inputs redistributed with dataset "
                "attribution under confirmed permission."
            ),
            "upstream_commit": "kaggle:m5-forecasting-accuracy:2020-06-01-files",
            "url": "https://www.kaggle.com/competitions/m5-forecasting-accuracy",
        },
    },
    "alibaba_openb_v2023": {
        "backend_kind": "alibaba_openb_gpu_placement",
        "dataset": "alibaba_cluster_trace_gpu_v2023_openb",
        "delivery": "upstream_fetch",
        "paths": {
            "works/clusterdata/cluster-trace-gpu-v2023/csv/"
            "openb_node_list_gpu_node.csv",
            "works/clusterdata/cluster-trace-gpu-v2023/csv/"
            "openb_pod_list_gpuspec33.csv",
        },
        "upstream_commit": "0d0f3f1efdbf1add6a7bcc63676eafbd1eb11f71",
        "license": "research trace terms; upstream repository license applies",
        "lock_strategy": "upstream_git_commit_raw_sha256_and_explicit_row_graph",
        "url": (
            "https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2023"
        ),
        "redistribution_notice": (
            "Trace bytes are not redistributed; the downloader fetches the "
            "exact files from the pinned upstream commit and verifies hashes."
        ),
    },
    "realm_j2": {
        "backend_kind": "jsplib_job_shop",
        "dataset": "realm_bench_j2_ccby",
        "delivery": "bundle",
        "paths": {
            "works/REALM-Bench-direct-pilot/datasets/clean/JSSP/J2.json",
        },
        "source_denominator_prefix": "realm_j2_ccby:",
        "upstream_commit": "9c3aa2ae97d65198f6ee29fe942d99f9b3a9c6eb",
        "license": "CC-BY-4.0 (REALM-Bench README, selected J2 JSON instance)",
        "lock_strategy": (
            "git_commit+file_sha256+selected_row_id+cc-by-runtime-source"
        ),
        "url": (
            "https://github.com/genglongling/REALM-Bench/tree/"
            "9c3aa2ae97d65198f6ee29fe942d99f9b3a9c6eb"
        ),
        "redistribution_notice": (
            "Exact release-referenced CC-BY-4.0 runtime source with upstream "
            "attribution preserved in the bundle manifest."
        ),
    },
}
_NGSIM_US101_SOURCE_SPEC = {
    "backend_kind": "sumo_ego",
    "delivery": "bundle",
    "root": (
        "works/autonomous_driving/ngsim/recovery/us101-v60-seven/bundles"
    ),
    "redistribution": {
        "dataset_id": "8ect-6jqj",
        "source_release": "doi:10.21949/1504477",
        "recording_id": "us-101",
        "license_id": "CC-BY-SA-4.0",
        "notice": (
            "Redistributed source-grounded NGSIM US-101 runtime bundles; "
            "shared bytes are stored once and restored to every hash-bound "
            "scenario install path."
        ),
    },
}
_NREL_MICROGRID_CITIES = (
    "albuquerque_nm",
    "atlanta_ga",
    "boston_ma",
    "chicago_il",
    "columbus_oh",
    "denver_co",
    "las_vegas_nv",
    "miami_fl",
    "minneapolis_mn",
    "nashville_tn",
    "phoenix_az",
    "portland_or",
    "sacramento_ca",
    "salt_lake_city_ut",
    "seattle_wa",
    "tucson_az",
)
_NREL_MICROGRID_SOURCE_SPEC = {
    "backend_kinds": {"pandapower_lv", "pymgrid_economic_dispatch"},
    "delivery": "bundle",
    "paths": {
        f"works/nrel-microgrid/{city}.npz" for city in _NREL_MICROGRID_CITIES
    },
    "redistribution": {
        "dataset": "nrel_oedi_derived_microgrid_profiles",
        "license": "mixed NREL/OEDI/NSRDB/OpenEI terms; attribution required",
        "lock_strategy": "release_runtime_closure_path_sha256",
        "notice": (
            "The bundle redistributes only the 16 deterministic derived NPZ "
            "profiles required by the Core; upstream attribution and source "
            "identifiers remain recorded in scenario provenance."
        ),
        "upstream_commit": "OEDI ComStock AMY2018 2021 release 1",
        "url": (
            "https://data.openei.org/submissions/4520 + https://nsrdb.nrel.gov "
            "+ https://developer.nrel.gov/docs/solar/pvwatts/v8/ + "
            "https://apps.openei.org/IURDB/"
        ),
    },
    "roles": ["derivation_input"],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _tar_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = Path(info.name).parts
    if (
        any(part in _EXCLUDED_PARTS for part in parts)
        or parts[-1:] == (".DS_Store",)
        or info.issym()
        or info.islnk()
    ):
        return None
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.mode = 0o755 if info.isdir() else 0o644
    info.pax_headers = {}
    return info


def _archive_file_contract(raw: Any, *, label: str) -> dict[str, str]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{label}_file_contract_invalid")
    files: dict[str, str] = {}
    for name, digest in raw.items():
        relative = PurePosixPath(str(name))
        if not (
            isinstance(name, str)
            and name
            and "\\" not in name
            and not relative.is_absolute()
            and ".." not in relative.parts
            and relative.as_posix() == name
            and isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{label}_file_contract_invalid:{name}")
        files[name] = digest
    return files


def _validate_tar_zst_files(
    archive_path: Path,
    expected: dict[str, str],
    *,
    label: str,
) -> None:
    zstd = shutil.which("zstd")
    if zstd is None:
        raise RuntimeError("zstd_not_found")
    process = subprocess.Popen(
        [zstd, "-dc", str(archive_path)],
        stdout=subprocess.PIPE,
    )
    assert process.stdout is not None
    observed: set[str] = set()
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                if member.name not in expected:
                    raise ValueError(f"{label}_archive_member_unexpected:{member.name}")
                if not member.isfile() or member.name in observed:
                    raise ValueError(f"{label}_archive_member_invalid:{member.name}")
                if not (
                    member.uid == 0
                    and member.gid == 0
                    and member.uname == ""
                    and member.gname == ""
                    and member.mtime == 0
                    and member.mode & 0o777 == 0o644
                ):
                    raise ValueError(
                        f"{label}_archive_member_metadata_invalid:{member.name}"
                    )
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"{label}_archive_member_unreadable:{member.name}")
                digest = hashlib.sha256()
                with stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != expected[member.name]:
                    raise ValueError(
                        f"{label}_archive_member_hash_mismatch:{member.name}"
                    )
                observed.add(member.name)
    except (OSError, tarfile.TarError, ValueError):
        process.terminate()
        process.wait()
        raise
    finally:
        process.stdout.close()
    if process.wait() != 0:
        raise RuntimeError(f"{label}_archive_decompression_failed")
    missing = sorted(set(expected) - observed)
    if missing:
        raise ValueError(f"{label}_archive_member_missing:{missing}")


def _bundle_archive_path(data_dir: Path, raw: str, *, label: str) -> Path:
    relative = PurePosixPath(raw)
    if not (
        raw
        and "\\" not in raw
        and not relative.is_absolute()
        and ".." not in relative.parts
        and relative.as_posix() == raw
    ):
        raise ValueError(f"{label}_archive_invalid")
    root = data_dir.resolve()
    archive = data_dir / raw
    try:
        archive.resolve(strict=True).relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(f"{label}_archive_invalid") from exc
    if not archive.is_file() or archive.is_symlink():
        raise ValueError(f"{label}_archive_invalid")
    return archive


def validate_bundle_archives(data_dir: Path, manifest: dict[str, Any]) -> None:
    """Verify exact archive membership and member bytes without extracting."""
    backend_archive = manifest.get("backend_archive")
    if backend_archive is not None:
        if not isinstance(backend_archive, str):
            raise ValueError("backend_archive_invalid")
        _validate_tar_zst_files(
            _bundle_archive_path(data_dir, backend_archive, label="backend"),
            _archive_file_contract(
                manifest.get("backend_archive_files"),
                label="backend",
            ),
            label="backend",
        )
    formal_archive = manifest.get("formal_evidence_archive")
    if formal_archive is not None:
        install_root = str(manifest.get("formal_evidence_install_root") or "")
        install_path = PurePosixPath(install_root)
        if not (
            isinstance(formal_archive, str)
            and formal_archive
            and install_root
            and not install_path.is_absolute()
            and ".." not in install_path.parts
            and install_path.as_posix() == install_root
        ):
            raise ValueError("formal_evidence_archive_contract_invalid")
        relative_files = _archive_file_contract(
            manifest.get("formal_evidence_files"),
            label="formal_evidence",
        )
        _validate_tar_zst_files(
            _bundle_archive_path(
                data_dir,
                formal_archive,
                label="formal_evidence",
            ),
            {
                (install_path / relative).as_posix(): digest
                for relative, digest in relative_files.items()
            },
            label="formal_evidence",
        )
    candidate_archive = manifest.get("candidate_evidence_archive")
    if candidate_archive is not None:
        install_root = str(manifest.get("candidate_evidence_install_root") or "")
        install_path = PurePosixPath(install_root)
        if not (
            isinstance(candidate_archive, str)
            and candidate_archive
            and install_root == "candidate_evidence"
            and install_path.as_posix() == install_root
        ):
            raise ValueError("candidate_evidence_archive_contract_invalid")
        relative_files = _archive_file_contract(
            manifest.get("candidate_evidence_files"),
            label="candidate_evidence",
        )
        _validate_tar_zst_files(
            _bundle_archive_path(
                data_dir,
                candidate_archive,
                label="candidate_evidence",
            ),
            {
                (install_path / relative).as_posix(): digest
                for relative, digest in relative_files.items()
            },
            label="candidate_evidence",
        )


def _add_archive_file(
    archive: tarfile.TarFile,
    *,
    source: Path,
    archive_name: str,
    files: dict[str, str],
) -> None:
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"backend_archive_source_invalid:{source}")
    if archive_name in files:
        if files[archive_name] != _sha256(source):
            raise ValueError(f"source_asset_archive_path_collision:{archive_name}")
        return
    archive.add(
        source,
        arcname=archive_name,
        recursive=False,
        filter=_tar_filter,
    )
    files[archive_name] = _sha256(source)


def _add_dynasched_runtime(
    archive: tarfile.TarFile,
    runtime: Path,
    files: dict[str, str],
) -> None:
    required = [runtime / name for name in ("pyproject.toml", "README.md", "LICENSE")]
    required.extend(sorted((runtime / "src" / "dsbx").rglob("*.py")))
    if any(not path.is_file() or path.is_symlink() for path in required):
        raise FileNotFoundError("DynaSched runtime package is incomplete")
    for path in required:
        relative = path.relative_to(runtime)
        _add_archive_file(
            archive,
            source=path,
            archive_name=(Path("backends") / "dynasched" / relative).as_posix(),
            files=files,
        )


def _repo_relative_path(repo_root: Path, raw: Any, *, label: str) -> tuple[str, Path]:
    value = str(raw or "")
    relative = Path(value)
    if (
        not value
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != value
    ):
        raise ValueError(f"{label}_path_invalid:{value}")
    resolved = (repo_root / relative).resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"{label}_path_invalid:{value}") from exc
    return value, resolved


def _is_trusted_dynasched_asset_path(path: Path) -> bool:
    return (
        path.parts[:2] == ("sources", "dynasched")
        or path.parts[:3] == ("works", "DynaSchedBench", "data")
        or path.parts == ("works", "DynaSchedBench", "LICENSE")
    )


def _dynasched_archive_path(install_path: str) -> str:
    path = Path(install_path)
    if path.parts[:2] == ("works", "DynaSchedBench"):
        return (Path("backends") / "dynasched" / Path(*path.parts[2:])).as_posix()
    return (Path("backends") / "dynasched_source_assets" / install_path).as_posix()


def _collect_dynasched_source_assets(
    repo_root: Path,
    source_suite: dict[str, Any],
) -> dict[str, Any]:
    """Bind the exact DynaSched files referenced by the final source suite."""
    raw_rows = source_suite.get("scenarios")
    if not isinstance(raw_rows, list):
        raise ValueError("release_source_suite_scenarios_invalid")
    files: dict[str, dict[str, Any]] = {}
    scenario_ids: list[str] = []
    license_paths: set[str] = set()
    for row in raw_rows:
        if not isinstance(row, dict) or row.get("backend_kind") != (
            "dynasched_flexible_job_shop"
        ):
            continue
        scenario_id = str(row.get("scenario_id") or "")
        scenario_relative, scenario_path = _repo_relative_path(
            repo_root,
            row.get("path"),
            label="dynasched_scenario",
        )
        if not scenario_path.is_file():
            raise FileNotFoundError(scenario_path)
        body = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
        if not isinstance(body, dict) or (
            body.get("scenario_id") != scenario_id
            or body.get("backend_kind") != "dynasched_flexible_job_shop"
        ):
            raise ValueError(
                f"dynasched_source_suite_scenario_mismatch:{scenario_relative}"
            )
        source_assets = (body.get("backend_config") or {}).get("source_assets")
        if not isinstance(source_assets, dict) or not source_assets:
            raise ValueError(f"dynasched_source_assets_missing:{scenario_id}")
        source_contract = body.get("source_contract")
        if not isinstance(source_contract, dict):
            raise ValueError(f"dynasched_source_contract_missing:{scenario_id}")
        contract_paths = {
            str(value)
            for field in (
                "runtime_input",
                "derivation_input",
                "implementation_asset",
                "metadata",
                "license",
            )
            for value in (source_contract.get(field) or [])
        }
        declared_paths: set[str] = set()
        for name, asset in source_assets.items():
            if not isinstance(name, str) or not isinstance(asset, dict):
                raise ValueError(f"dynasched_source_asset_invalid:{scenario_id}")
            install_path, source_path = _repo_relative_path(
                repo_root,
                asset.get("path"),
                label="dynasched_source_asset",
            )
            if not _is_trusted_dynasched_asset_path(Path(install_path)):
                raise ValueError(f"dynasched_source_asset_path_invalid:{install_path}")
            expected = str(asset.get("sha256") or "").removeprefix("sha256:")
            role = str(asset.get("role") or "")
            if len(expected) != 64 or not role:
                raise ValueError(f"dynasched_source_asset_invalid:{install_path}")
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            actual = _sha256(source_path)
            if actual != expected:
                raise ValueError(f"dynasched_source_asset_hash_mismatch:{install_path}")
            declared_paths.add(install_path)
            if role == "license":
                license_paths.add(install_path)
            current = files.get(install_path)
            if current is not None and current["sha256"] != expected:
                raise ValueError(
                    f"dynasched_source_asset_binding_conflict:{install_path}"
                )
            if current is None:
                current = {
                    "archive_path": _dynasched_archive_path(install_path),
                    "sha256": expected,
                    "roles": [],
                    "scenario_ids": [],
                }
                files[install_path] = current
            if role not in current["roles"]:
                current["roles"].append(role)
            if scenario_id not in current["scenario_ids"]:
                current["scenario_ids"].append(scenario_id)
        if declared_paths != contract_paths:
            raise ValueError(f"dynasched_source_contract_asset_mismatch:{scenario_id}")
        physical_lock = body.get("physical_source_lock")
        if physical_lock is not None:
            if not isinstance(physical_lock, dict) or (
                physical_lock.get("schema_version") != "source_asset_graph_v1"
                or physical_lock.get("backend_kind") != "dynasched_flexible_job_shop"
            ):
                raise ValueError(
                    f"dynasched_physical_source_lock_invalid:{scenario_id}"
                )
            locked_assets = physical_lock.get("required_source_assets")
            if not isinstance(locked_assets, list) or not locked_assets:
                raise ValueError(
                    f"dynasched_physical_source_lock_invalid:{scenario_id}"
                )
            locked_paths: set[str] = set()
            for locked in locked_assets:
                if not isinstance(locked, dict):
                    raise ValueError(
                        f"dynasched_physical_source_lock_invalid:{scenario_id}"
                    )
                locked_path = str(locked.get("declared_path") or "")
                locked_sha = str(locked.get("sha256") or "").removeprefix("sha256:")
                locked_paths.add(locked_path)
                asset_row = files.get(locked_path)
                if asset_row is None or asset_row["sha256"] != locked_sha:
                    raise ValueError(
                        f"dynasched_physical_source_lock_mismatch:{locked_path}"
                    )
            if locked_paths != set(source_contract.get("runtime_input") or []):
                raise ValueError(
                    f"dynasched_physical_source_lock_runtime_mismatch:{scenario_id}"
                )
        scenario_ids.append(scenario_id)
    if scenario_ids and not license_paths:
        raise ValueError("dynasched_redistribution_license_missing")
    for row in files.values():
        row["roles"].sort()
        row["scenario_ids"].sort()
    return {
        "n_scenarios": len(scenario_ids),
        "n_files": len(files),
        "scenario_ids": sorted(scenario_ids),
        "redistribution_license_paths": sorted(license_paths),
        "files": dict(sorted(files.items())),
    }


def _matches_exact_source(row: dict[str, Any], spec: dict[str, Any]) -> bool:
    if row.get("backend_kind") != spec["backend_kind"]:
        return False
    prefix = spec.get("source_denominator_prefix")
    return prefix is None or str(row.get("source_denominator_key") or "").startswith(
        str(prefix)
    )


def _suite_physical_source_lock(
    row: dict[str, Any],
    *,
    source_id: str,
) -> dict[str, Any]:
    ledger = row.get("case_ledger")
    lock = ledger.get("physical_source_lock") if isinstance(ledger, dict) else None
    if not isinstance(lock, dict) or (
        lock.get("schema_version") != "source_asset_graph_v1"
        or lock.get("backend_kind") != row.get("backend_kind")
    ):
        raise ValueError(f"{source_id}_physical_source_lock_invalid")
    return lock


def _exact_source_archive_path(source_id: str, install_path: str) -> str:
    return (
        Path("backends") / "release_source_assets" / source_id / install_path
    ).as_posix()


def _collect_exact_source_assets(
    repo_root: Path,
    source_suite: dict[str, Any],
    *,
    source_id: str,
) -> dict[str, Any]:
    """Bind one exact-file source family from suite locks to archive bytes."""
    spec = _EXACT_SOURCE_SPECS[source_id]
    raw_rows = source_suite.get("scenarios")
    if not isinstance(raw_rows, list):
        raise ValueError("release_source_suite_scenarios_invalid")
    files: dict[str, dict[str, Any]] = {}
    scenario_ids: list[str] = []
    redistribution: dict[str, str] | None = None
    for row in raw_rows:
        if not isinstance(row, dict) or not _matches_exact_source(row, spec):
            continue
        scenario_id = str(row.get("scenario_id") or "")
        scenario_relative, scenario_path = _repo_relative_path(
            repo_root,
            row.get("path"),
            label=f"{source_id}_scenario",
        )
        if not scenario_path.is_file():
            raise FileNotFoundError(scenario_path)
        body = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
        if not isinstance(body, dict) or (
            body.get("scenario_id") != scenario_id
            or body.get("backend_kind") != spec["backend_kind"]
        ):
            raise ValueError(
                f"{source_id}_source_suite_scenario_mismatch:{scenario_relative}"
            )
        source_contract = body.get("source_contract")
        if not isinstance(source_contract, dict):
            raise ValueError(f"{source_id}_source_contract_missing:{scenario_id}")
        contract_roles = tuple(spec.get("contract_roles") or ("runtime_input",))
        matching_roles = [
            str(role)
            for role in contract_roles
            if isinstance(source_contract.get(str(role)), list)
            and {str(value) for value in source_contract[str(role)]}
            == set(spec["paths"])
        ]
        if len(matching_roles) != 1:
            raise ValueError(
                f"{source_id}_source_asset_role_invalid:{scenario_id}"
            )
        contract_role = matching_roles[0]
        contract_paths = set(spec["paths"])

        lock = _suite_physical_source_lock(row, source_id=source_id)
        locked_assets = lock.get("required_source_assets")
        if not isinstance(locked_assets, list) or not locked_assets:
            raise ValueError(f"{source_id}_physical_source_lock_invalid")
        locked: dict[str, str] = {}
        for asset in locked_assets:
            if not isinstance(asset, dict):
                raise ValueError(f"{source_id}_physical_source_lock_invalid")
            install_path = str(asset.get("declared_path") or "")
            expected = str(asset.get("sha256") or "").removeprefix("sha256:")
            if install_path in locked or len(expected) != 64:
                raise ValueError(f"{source_id}_physical_source_lock_invalid")
            locked[install_path] = expected
        if set(locked) != contract_paths:
            raise ValueError(
                f"{source_id}_source_contract_asset_mismatch:{scenario_id}"
            )
        file_sha256s = source_contract.get("file_sha256s")
        if file_sha256s is not None and file_sha256s != locked:
            raise ValueError(f"{source_id}_source_contract_hash_mismatch:{scenario_id}")
        metadata = dict(spec.get("metadata") or {})
        if not set(metadata).issubset(set(source_contract.get("metadata") or [])):
            raise ValueError(f"{source_id}_source_metadata_missing:{scenario_id}")

        provenance = body.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError(f"{source_id}_provenance_missing:{scenario_id}")
        provenance_variants = spec.get("provenance_variants") or (
            spec.get("provenance") or spec,
        )
        provenance_fields = (
            ("license", "license"),
            ("lock_strategy", "lock_strategy"),
            ("commit", "upstream_commit"),
            ("url", "url"),
        )
        if not any(
            all(
                str(provenance.get(source_field) or "")
                == str(expected[spec_field])
                for source_field, spec_field in provenance_fields
            )
            for expected in provenance_variants
        ):
            raise ValueError(f"{source_id}_provenance_metadata_invalid")
        declared_redistribution = spec.get("redistribution")
        current_redistribution = (
            {str(key): str(value) for key, value in declared_redistribution.items()}
            if isinstance(declared_redistribution, dict)
            else {
                "dataset": str(spec["dataset"]),
                "license": str(provenance.get("license") or ""),
                "lock_strategy": str(provenance.get("lock_strategy") or ""),
                "notice": str(spec["redistribution_notice"]),
                "upstream_commit": str(provenance.get("commit") or ""),
                "url": str(provenance.get("url") or ""),
            }
        )
        if set(current_redistribution) != {
            "dataset",
            "license",
            "lock_strategy",
            "notice",
            "upstream_commit",
            "url",
        } or any(not value for value in current_redistribution.values()):
            raise ValueError(f"{source_id}_redistribution_metadata_invalid")
        if redistribution is None:
            redistribution = current_redistribution
        elif redistribution != current_redistribution:
            raise ValueError(f"{source_id}_redistribution_metadata_conflict")

        for install_path, expected in {**locked, **metadata}.items():
            role = "metadata" if install_path in metadata else contract_role
            _, source_path = _repo_relative_path(
                repo_root,
                install_path,
                label=f"{source_id}_source_asset",
            )
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            if _sha256(source_path) != expected:
                raise ValueError(
                    f"{source_id}_source_asset_hash_mismatch:{install_path}"
                )
            current = files.get(install_path)
            if current is not None and current["sha256"] != expected:
                raise ValueError(
                    f"{source_id}_source_asset_binding_conflict:{install_path}"
                )
            if current is None:
                current = {
                    "archive_path": _exact_source_archive_path(source_id, install_path),
                    "delivery": str(spec["delivery"]),
                    "sha256": expected,
                    "roles": [role],
                    "scenario_ids": [],
                }
                files[install_path] = current
            elif role not in current["roles"]:
                current["roles"].append(role)
            if scenario_id not in current["scenario_ids"]:
                current["scenario_ids"].append(scenario_id)
        scenario_ids.append(scenario_id)
    for file_row in files.values():
        file_row["roles"].sort()
        file_row["scenario_ids"].sort()
    return {
        "n_scenarios": len(scenario_ids),
        "n_files": len(files),
        "delivery": str(spec["delivery"]),
        "scenario_ids": sorted(scenario_ids),
        "redistribution": redistribution or {},
        "files": dict(sorted(files.items())),
    }


def _ngsim_us101_archive_path(digest: str) -> str:
    return (
        Path("backends")
        / "release_source_assets"
        / "ngsim_us101"
        / "blobs"
        / digest
    ).as_posix()


def _collect_ngsim_us101_source_assets(
    repo_root: Path,
    source_suite: dict[str, Any],
) -> dict[str, Any]:
    """Bind NGSIM install paths to a de-duplicated, hash-addressed blob set."""
    raw_rows = source_suite.get("scenarios")
    if not isinstance(raw_rows, list):
        raise ValueError("release_source_suite_scenarios_invalid")
    spec = _NGSIM_US101_SOURCE_SPEC
    root = str(spec["root"])
    files: dict[str, dict[str, Any]] = {}
    scenario_ids: list[str] = []
    for row in sorted(
        (
            value
            for value in raw_rows
            if isinstance(value, dict)
            and value.get("backend_kind") == spec["backend_kind"]
        ),
        key=lambda value: str(value.get("scenario_id") or ""),
    ):
        scenario_id = str(row.get("scenario_id") or "")
        scenario_relative, scenario_path = _repo_relative_path(
            repo_root,
            row.get("path"),
            label="ngsim_us101_scenario",
        )
        if not scenario_id or not scenario_path.is_file():
            raise ValueError(f"ngsim_us101_scenario_invalid:{scenario_relative}")
        body = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
        if not isinstance(body, dict) or (
            body.get("scenario_id") != scenario_id
            or body.get("backend_kind") != spec["backend_kind"]
        ):
            raise ValueError(
                f"ngsim_us101_source_suite_scenario_mismatch:{scenario_relative}"
            )
        provenance = body.get("provenance")
        if not isinstance(provenance, dict) or any(
            provenance.get(field) != expected
            for field, expected in spec["redistribution"].items()
            if field != "notice"
        ):
            raise ValueError(f"ngsim_us101_provenance_invalid:{scenario_id}")
        source_contract = body.get("source_contract")
        if not isinstance(source_contract, dict):
            raise ValueError(f"ngsim_us101_source_contract_missing:{scenario_id}")
        roles_by_path: dict[str, set[str]] = {}
        for role in (
            "runtime_input",
            "derivation_input",
            "implementation_asset",
            "metadata",
            "license",
        ):
            values = source_contract.get(role)
            if not isinstance(values, list):
                raise ValueError(
                    f"ngsim_us101_source_contract_invalid:{scenario_id}:{role}"
                )
            for raw_path in values:
                install_path, _source_path = _repo_relative_path(
                    repo_root,
                    raw_path,
                    label="ngsim_us101_source_asset",
                )
                if not install_path.startswith(f"{root}/"):
                    raise ValueError(
                        f"ngsim_us101_source_asset_path_invalid:{install_path}"
                    )
                roles_by_path.setdefault(install_path, set()).add(role)
        if not roles_by_path or not any(
            "license" in roles for roles in roles_by_path.values()
        ):
            raise ValueError(f"ngsim_us101_redistribution_license_missing:{scenario_id}")
        raw_hashes = source_contract.get("file_sha256s")
        consumed_paths = {
            path
            for path, roles in roles_by_path.items()
            if roles.intersection({"runtime_input", "derivation_input"})
        }
        if not isinstance(raw_hashes, dict) or set(raw_hashes) != consumed_paths:
            raise ValueError(f"ngsim_us101_source_contract_hashes_invalid:{scenario_id}")
        locked_assets = _suite_physical_source_lock(
            row,
            source_id="ngsim_us101",
        ).get("required_source_assets")
        if not isinstance(locked_assets, list) or not locked_assets:
            raise ValueError("ngsim_us101_physical_source_lock_invalid")
        locked: dict[str, str] = {}
        for asset in locked_assets:
            if not isinstance(asset, dict):
                raise ValueError("ngsim_us101_physical_source_lock_invalid")
            install_path = str(asset.get("declared_path") or "")
            digest = str(asset.get("sha256") or "").removeprefix("sha256:")
            if (
                install_path in locked
                or install_path not in roles_by_path
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                raise ValueError("ngsim_us101_physical_source_lock_invalid")
            locked[install_path] = digest
        if set(locked) != consumed_paths or raw_hashes != locked:
            raise ValueError(
                f"ngsim_us101_source_contract_asset_mismatch:{scenario_id}"
            )
        for install_path in sorted(roles_by_path):
            _, source_path = _repo_relative_path(
                repo_root,
                install_path,
                label="ngsim_us101_source_asset",
            )
            if not source_path.is_file() or source_path.is_symlink():
                raise FileNotFoundError(source_path)
            digest = _sha256(source_path)
            if install_path in locked and locked[install_path] != digest:
                raise ValueError(
                    f"ngsim_us101_source_asset_hash_mismatch:{install_path}"
                )
            current = files.get(install_path)
            if current is not None and current["sha256"] != digest:
                raise ValueError(
                    f"ngsim_us101_source_asset_binding_conflict:{install_path}"
                )
            if current is None:
                current = {
                    "archive_path": _ngsim_us101_archive_path(digest),
                    "delivery": "bundle",
                    "sha256": digest,
                    "roles": sorted(roles_by_path[install_path]),
                    "scenario_ids": [],
                }
                files[install_path] = current
            if scenario_id not in current["scenario_ids"]:
                current["scenario_ids"].append(scenario_id)

        bundle_roots = {
            Path(root) / Path(path).relative_to(root).parts[0]
            for path in roles_by_path
        }
        if len(bundle_roots) != 1:
            raise ValueError(f"ngsim_us101_checksums_bundle_mismatch:{scenario_id}")
        bundle_root = bundle_roots.pop()
        checksum_relative = (bundle_root / "checksums.sha256").as_posix()
        _, checksum_path = _repo_relative_path(
            repo_root, checksum_relative, label="ngsim_us101_checksums"
        )
        if not checksum_path.is_file() or checksum_path.is_symlink():
            raise FileNotFoundError(checksum_path)
        expected_checksums = "".join(
            f"{files[path]['sha256']}  {Path(path).relative_to(bundle_root).as_posix()}\n"
            for path in sorted(roles_by_path)
        ).encode("ascii")
        if checksum_path.read_bytes() != expected_checksums:
            raise ValueError(f"ngsim_us101_checksums_mismatch:{checksum_relative}")
        checksum_digest = hashlib.sha256(expected_checksums).hexdigest()
        checksum_row = files.setdefault(checksum_relative, {
            "archive_path": _ngsim_us101_archive_path(checksum_digest),
            "delivery": "bundle",
            "sha256": checksum_digest,
            "roles": ["metadata"],
            "scenario_ids": [],
        })
        if checksum_row["sha256"] != checksum_digest:
            raise ValueError(f"ngsim_us101_checksums_conflict:{checksum_relative}")
        if scenario_id not in checksum_row["scenario_ids"]:
            checksum_row["scenario_ids"].append(scenario_id)
        scenario_ids.append(scenario_id)
    for row in files.values():
        row["scenario_ids"].sort()
    blobs: dict[str, dict[str, Any]] = {}
    for install_path, row in sorted(files.items()):
        archive_path = str(row["archive_path"])
        blob = blobs.setdefault(
            archive_path,
            {
                "archive_path": archive_path,
                "sha256": str(row["sha256"]),
                "source_path": install_path,
                "install_paths": [],
            },
        )
        if blob["sha256"] != row["sha256"]:
            raise ValueError(f"ngsim_us101_blob_hash_conflict:{archive_path}")
        blob["install_paths"].append(install_path)
    return {
        "n_scenarios": len(scenario_ids),
        "n_files": len(files),
        "n_blobs": len(blobs),
        "delivery": "bundle",
        "scenario_ids": sorted(scenario_ids),
        "redistribution": dict(spec["redistribution"]),
        "files": dict(sorted(files.items())),
        "blobs": dict(sorted(blobs.items())),
    }


def _ngsim_us101_blob_contract(contract: Any) -> dict[str, str]:
    if not isinstance(contract, dict) or not isinstance(contract.get("files"), dict):
        raise ValueError("ngsim_us101_blob_contract_invalid")
    expected: dict[str, dict[str, Any]] = {}
    for install_path, row in contract["files"].items():
        if not isinstance(row, dict):
            raise ValueError("ngsim_us101_blob_contract_invalid")
        archive_path = str(row.get("archive_path") or "")
        digest = str(row.get("sha256") or "")
        if (
            re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or archive_path != _ngsim_us101_archive_path(digest)
        ):
            raise ValueError("ngsim_us101_blob_contract_invalid")
        blob = expected.setdefault(
            archive_path,
            {
                "archive_path": archive_path,
                "sha256": digest,
                "source_path": install_path,
                "install_paths": [],
            },
        )
        if blob["sha256"] != digest:
            raise ValueError("ngsim_us101_blob_contract_invalid")
        blob["install_paths"].append(install_path)
    if not (
        contract.get("n_blobs") == len(expected)
        and contract.get("blobs") == dict(sorted(expected.items()))
    ):
        raise ValueError("ngsim_us101_blob_contract_invalid")
    return {
        archive_path: str(row["sha256"])
        for archive_path, row in sorted(expected.items())
    }


def _collect_nrel_microgrid_source_assets(
    repo_root: Path,
    source_suite: dict[str, Any],
    external_contract: Any,
) -> dict[str, Any]:
    """Bind the exact derived NREL/OEDI profiles needed by released rows."""
    raw_rows = source_suite.get("scenarios")
    if not isinstance(raw_rows, list):
        raise ValueError("release_source_suite_scenarios_invalid")
    scenario_rows = [
        row
        for row in raw_rows
        if isinstance(row, dict)
        and row.get("backend_kind")
        in _NREL_MICROGRID_SOURCE_SPEC["backend_kinds"]
    ]
    if not scenario_rows:
        return {
            "n_scenarios": 0,
            "n_files": 0,
            "delivery": "bundle",
            "scenario_ids": [],
            "redistribution": {},
            "files": {},
        }
    if not isinstance(external_contract, dict):
        raise ValueError("nrel_microgrid_external_contract_missing")
    required_files = external_contract.get("required_files")
    metadata = external_contract.get("metadata")
    spec = _NREL_MICROGRID_SOURCE_SPEC
    if not (
        external_contract.get("delivery") == "user_provided"
        and external_contract.get("revision")
        == spec["redistribution"]["upstream_commit"]
        and external_contract.get("url") == spec["redistribution"]["url"]
        and isinstance(required_files, dict)
        and set(required_files) == spec["paths"]
        and isinstance(metadata, dict)
        and metadata.get("root") == "works/nrel-microgrid"
        and set(metadata.get("backend_kinds") or []) == spec["backend_kinds"]
        and metadata.get("roles")
        == {path: spec["roles"] for path in sorted(spec["paths"])}
    ):
        raise ValueError("nrel_microgrid_external_contract_invalid")

    scenario_ids: list[str] = []
    scenarios_by_path = {path: [] for path in sorted(spec["paths"])}
    for row in scenario_rows:
        scenario_id = str(row.get("scenario_id") or "")
        lock = _suite_physical_source_lock(row, source_id="nrel_microgrid")
        assets = lock.get("required_source_assets")
        if not isinstance(assets, list) or not assets:
            raise ValueError("nrel_microgrid_physical_source_lock_invalid")
        for asset in assets:
            if not isinstance(asset, dict):
                raise ValueError("nrel_microgrid_physical_source_lock_invalid")
            install_path = str(asset.get("declared_path") or "")
            expected = str(asset.get("sha256") or "").removeprefix("sha256:")
            if required_files.get(install_path) != expected:
                raise ValueError("nrel_microgrid_physical_source_lock_mismatch")
            scenarios_by_path[install_path].append(scenario_id)
        scenario_ids.append(scenario_id)
    if any(not bound for bound in scenarios_by_path.values()):
        raise ValueError("nrel_microgrid_unreferenced_profile")

    files: dict[str, dict[str, Any]] = {}
    for install_path, expected in sorted(required_files.items()):
        _, source_path = _repo_relative_path(
            repo_root,
            install_path,
            label="nrel_microgrid_source_asset",
        )
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        if _sha256(source_path) != expected:
            raise ValueError(
                f"nrel_microgrid_source_asset_hash_mismatch:{install_path}"
            )
        files[install_path] = {
            "archive_path": _exact_source_archive_path(
                "nrel_microgrid", install_path
            ),
            "delivery": "bundle",
            "sha256": expected,
            "roles": list(spec["roles"]),
            "scenario_ids": sorted(scenarios_by_path[install_path]),
        }
    return {
        "n_scenarios": len(scenario_ids),
        "n_files": len(files),
        "delivery": "bundle",
        "scenario_ids": sorted(scenario_ids),
        "redistribution": dict(spec["redistribution"]),
        "files": files,
    }


def _release_source_suite(
    repo_root: Path,
    release_dir: Path,
    release_manifest: dict[str, Any],
) -> tuple[Path, str, dict[str, Any]]:
    replay = release_manifest.get("protocol21_replay")
    if not isinstance(replay, dict):
        raise ValueError("release_source_suite_binding_missing")
    relative, path = _repo_relative_path(
        repo_root,
        replay.get("source_suite"),
        label="release_source_suite",
    )
    if path != release_dir / "protocol21_source_suite.json" or not path.is_file():
        raise ValueError(f"release_source_suite_path_mismatch:{relative}")
    expected = str(replay.get("source_suite_sha256") or "")
    actual = _sha256(path)
    if expected != actual:
        raise ValueError("release_source_suite_hash_mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("release_source_suite_invalid")
    rows = payload.get("scenarios")
    if not isinstance(rows, list) or payload.get("n_scenarios") != len(rows):
        raise ValueError("release_source_suite_scenarios_invalid")
    return path, actual, payload


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _release_relative_file(
    release_dir: Path,
    raw: Any,
    *,
    label: str,
) -> tuple[str, Path]:
    value = raw if isinstance(raw, str) else ""
    relative = PurePosixPath(value)
    if not (
        value
        and "\\" not in value
        and "\x00" not in value
        and not relative.is_absolute()
        and len(relative.parts) == 1
        and relative.as_posix() == value
    ):
        raise ValueError(f"{label}_path_invalid")
    return value, release_dir / value


def _release_backend_runtime_closure(
    repo_root: Path,
    release_dir: Path,
    release_manifest: dict[str, Any],
    *,
    additional_archive_roots: set[str],
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    binding = release_manifest.get("backend_runtime_closure")
    if not isinstance(binding, dict) or set(binding) != {
        "path",
        "sha256",
        "schema_version",
        "n_archived_files",
        "n_external_sources",
        "n_backend_links",
        "n_runtime_packages",
        "identity_sha256",
    }:
        raise ValueError("backend_runtime_closure_binding_missing")
    relative, path = _release_relative_file(
        release_dir,
        binding.get("path"),
        label="backend_runtime_closure",
    )
    expected = str(binding.get("sha256") or "")
    count_fields = (
        "n_archived_files",
        "n_external_sources",
        "n_backend_links",
        "n_runtime_packages",
    )
    if not (
        binding.get("schema_version") == "operate-backend-runtime-closure-v1"
        and path.is_file()
        and not path.is_symlink()
        and len(expected) == 64
        and _sha256(path) == expected
        and all(
            type(binding[field]) is int and binding[field] >= 0
            for field in count_fields
        )
        and isinstance(binding.get("identity_sha256"), str)
        and len(binding["identity_sha256"]) == 64
        and all(
            character in "0123456789abcdef" for character in binding["identity_sha256"]
        )
    ):
        raise ValueError("backend_runtime_closure_binding_invalid")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _validated_backend_runtime_closure(
        repo_root,
        payload,
        additional_archive_roots=additional_archive_roots,
        expected_release_id=str(release_manifest.get("release_id") or ""),
    )
    replay = release_manifest.get("protocol21_replay")
    expected_suite = (
        replay.get("source_suite_sha256") if isinstance(replay, dict) else None
    )
    if not (
        payload.get("release_id") == release_manifest.get("release_id")
        and payload.get("source_suite_sha256") == expected_suite
        and binding["n_archived_files"] == payload["summary"]["n_archived_files"]
        and binding["n_external_sources"] == payload["summary"]["n_external_sources"]
        and binding["n_backend_links"] == payload["summary"]["n_backend_links"]
        and binding["n_runtime_packages"] == payload["summary"]["n_runtime_packages"]
        and binding["identity_sha256"] == payload["identity_sha256"]
    ):
        raise ValueError("backend_runtime_closure_binding_invalid")
    return (
        path,
        {
            "schema_version": str(binding["schema_version"]),
            "path": relative,
            "sha256": expected,
            "n_archived_files": int(binding["n_archived_files"]),
            "n_external_sources": int(binding["n_external_sources"]),
            "n_backend_links": int(binding["n_backend_links"]),
            "n_runtime_packages": int(binding["n_runtime_packages"]),
            "identity_sha256": str(binding["identity_sha256"]),
        },
        payload,
    )


def _copy_optional_candidate_closure(
    *,
    repo_root: Path,
    release_dir: Path,
    release_manifest: dict[str, Any],
    output_dir: Path,
    tracked: dict[str, str],
) -> dict[str, Any] | None:
    """Include the compact terminal candidate ledger once materialized."""
    binding = release_manifest.get("candidate_closure")
    if binding is None:
        return None
    binding_fields = {
        "path",
        "sha256",
        "schema_version",
        "status",
        "n_independent_candidates",
        "n_terminal_candidates",
        "n_unresolved_candidates",
        "identity_set_sha256",
    }
    if not isinstance(binding, dict) or set(binding) != binding_fields:
        raise ValueError("candidate_closure_binding_invalid")
    _relative, source = _release_relative_file(
        release_dir,
        binding.get("path"),
        label="candidate_closure",
    )
    expected = str(binding.get("sha256") or "")
    if not (
        binding.get("schema_version") == "operate-candidate-closure-compact-v1"
        and source.is_file()
        and not source.is_symlink()
        and len(expected) == 64
        and _sha256(source) == expected
    ):
        raise ValueError("candidate_closure_binding_invalid")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("candidate_closure_binding_invalid")
    from scripts.finalize_operate_candidate_pool import (  # noqa: PLC0415
        validate_compact_candidate_closure,
    )

    validate_compact_candidate_closure(payload)
    summary = payload["summary"]
    if not (
        binding["status"] == payload["status"]
        and binding["n_independent_candidates"] == summary["n_independent_candidates"]
        and binding["n_terminal_candidates"] == summary["n_terminal_candidates"]
        and binding["n_unresolved_candidates"] == summary["n_unresolved_candidates"]
        and binding["identity_set_sha256"] == payload["identity_set_sha256"]
    ):
        raise ValueError("candidate_closure_binding_invalid")
    destination_name = "candidate_closure.json"
    destination = output_dir / destination_name
    shutil.copyfile(source, destination)
    tracked[destination_name] = _sha256(destination)
    return {
        **binding,
        "path": destination_name,
        "sha256": tracked[destination_name],
    }


def _candidate_evidence_files(
    repo_root: Path,
    candidate_closure: dict[str, Any],
) -> dict[str, str]:
    inputs = candidate_closure.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("candidate_evidence_inputs_missing")
    files: dict[str, str] = {}
    for raw_bindings in inputs.values():
        bindings = raw_bindings if isinstance(raw_bindings, list) else [raw_bindings]
        for binding in bindings:
            if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
                raise ValueError("candidate_evidence_binding_invalid")
            relative, path = _repo_relative_path(
                repo_root,
                binding.get("path"),
                label="candidate_evidence",
            )
            relative_path = PurePosixPath(relative)
            expected = str(binding.get("sha256") or "")
            if not (
                relative_path.parts[:2] == (".hl", "artifacts")
                and len(expected) == 64
                and all(character in "0123456789abcdef" for character in expected)
                and relative not in files
                and path.is_file()
                and not path.is_symlink()
                and _sha256(path) == expected
            ):
                raise ValueError(f"candidate_evidence_binding_invalid:{relative}")
            payloads = (
                [json.loads(path.read_text(encoding="utf-8"))]
                if path.suffix == ".json"
                else [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            )
            if canonicalize_repo_owned_paths(payloads, repo_root=repo_root) != payloads:
                raise ValueError(f"candidate_evidence_nonportable_path:{relative}")
            files[relative] = expected
    return dict(sorted(files.items()))


def _string_list(raw: Any, *, label: str) -> list[str]:
    if not (
        isinstance(raw, list)
        and raw
        and all(isinstance(value, str) and value for value in raw)
        and len(raw) == len(set(raw))
    ):
        raise ValueError(f"backend_runtime_closure_{label}_invalid")
    return raw


def _portable_archive_path(raw: Any, *, label: str) -> str:
    value = raw if isinstance(raw, str) else ""
    relative = PurePosixPath(value)
    if not (
        value
        and "\\" not in value
        and "\x00" not in value
        and not relative.is_absolute()
        and ".." not in relative.parts
        and relative.as_posix() == value
    ):
        raise ValueError(f"backend_runtime_closure_{label}_invalid:{value}")
    return value


def _expected_backend_archive_path(source_path: str) -> str:
    relative = PurePosixPath(source_path)
    if any(part in _EXCLUDED_PARTS for part in relative.parts) or (
        relative.name == ".DS_Store"
    ):
        raise ValueError(f"backend_runtime_closure_source_path_invalid:{source_path}")
    if relative.parts[:2] == ("sources", "dynasched") and len(relative.parts) > 2:
        return (
            PurePosixPath("backends") / "dynasched_source_assets" / relative
        ).as_posix()
    if relative.parts[:3] == ("sources", "resco", "arterial4x4") and len(
        relative.parts
    ) > 3:
        return (
            PurePosixPath("backends")
            / "resco"
            / "resco_benchmark"
            / "environments"
            / "arterial4x4"
            / PurePosixPath(*relative.parts[3:])
        ).as_posix()
    if len(relative.parts) < 3 or relative.parts[0] != "works":
        raise ValueError(f"backend_runtime_closure_source_path_invalid:{source_path}")
    works_name = relative.parts[1]
    target = BACKEND_LINKS.get(works_name)
    if target is None:
        raise ValueError(f"backend_runtime_closure_source_path_invalid:{source_path}")
    return (
        PurePosixPath("backends") / target / PurePosixPath(*relative.parts[2:])
    ).as_posix()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_external_sources(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict):
        raise ValueError("backend_runtime_closure_external_sources_invalid")
    for source_id, row in raw.items():
        if not (
            isinstance(source_id, str)
            and source_id
            and isinstance(row, dict)
            and set(row)
            == {"delivery", "url", "revision", "required_files", "metadata"}
            and row.get("delivery")
            in {"git_checkout", "upstream_fetch", "user_provided"}
            and isinstance(row.get("url"), str)
            and str(row["url"]).startswith("https://")
            and not any(character in "\x00\r\n" for character in str(row["url"]))
            and isinstance(row.get("revision"), str)
            and bool(row["revision"])
            and not any(character in "\x00\r\n" for character in str(row["revision"]))
            and isinstance(row.get("required_files"), dict)
            and bool(row["required_files"])
            and isinstance(row.get("metadata"), dict)
        ):
            raise ValueError(
                f"backend_runtime_closure_external_source_invalid:{source_id}"
            )
        metadata = row["metadata"]
        if not (
            set(metadata)
            == {
                "backend_kinds",
                "license_status",
                "redistributed",
                "roles",
                "root",
            }
            and metadata.get("redistributed") is False
            and isinstance(metadata.get("license_status"), str)
            and bool(metadata["license_status"])
            and isinstance(metadata.get("roles"), dict)
        ):
            raise ValueError(
                f"backend_runtime_closure_external_source_invalid:{source_id}"
            )
        root = _portable_archive_path(
            metadata.get("root"),
            label=f"external_source_root:{source_id}",
        )
        _string_list(
            metadata.get("backend_kinds"),
            label=f"external_source_backend_kinds:{source_id}",
        )
        if set(metadata["roles"]) != set(row["required_files"]):
            raise ValueError(
                f"backend_runtime_closure_external_source_invalid:{source_id}"
            )
        for source_path, digest in row["required_files"].items():
            portable_path = _portable_archive_path(
                source_path,
                label=f"external_source_path:{source_id}",
            )
            if not (
                (portable_path == root or portable_path.startswith(f"{root}/"))
                and _is_sha256(digest)
            ):
                raise ValueError(
                    f"backend_runtime_closure_external_source_invalid:{source_id}"
                )
            _string_list(
                metadata["roles"][source_path],
                label=f"external_source_roles:{source_id}",
            )
    return raw


def _validate_runtime_packages(repo_root: Path, raw: Any) -> int:
    if not isinstance(raw, dict):
        raise ValueError("backend_runtime_closure_runtime_packages_invalid")
    uv_lock = repo_root / "uv.lock"
    live_uv_lock_sha256 = _sha256(uv_lock) if uv_lock.is_file() else None
    virtual_sources: set[str] = set()
    for package, row in raw.items():
        if not (
            isinstance(package, str)
            and package
            and isinstance(row, dict)
            and set(row)
            in (
                {
                    "backend_kinds",
                    "lock_entries",
                    "lock_entries_sha256",
                    "uv_lock_sha256",
                },
                {
                    "backend_kinds",
                    "lock_entries",
                    "lock_entries_sha256",
                    "uv_lock_sha256",
                    "virtual_sources",
                },
            )
            and isinstance(row.get("lock_entries"), list)
            and bool(row["lock_entries"])
            and _is_sha256(row.get("lock_entries_sha256"))
            and row["lock_entries_sha256"] == _canonical_sha256(row["lock_entries"])
            and _is_sha256(row.get("uv_lock_sha256"))
            and row["uv_lock_sha256"] == live_uv_lock_sha256
        ):
            raise ValueError(
                f"backend_runtime_closure_runtime_package_invalid:{package}"
            )
        _string_list(
            row.get("backend_kinds"),
            label=f"runtime_package_backend_kinds:{package}",
        )
        for entry in row["lock_entries"]:
            if not (
                isinstance(entry, dict)
                and set(entry)
                in (
                    {"version", "source", "artifacts", "identity_sha256"},
                    {
                        "version",
                        "source",
                        "artifacts",
                        "resolution_markers",
                        "identity_sha256",
                    },
                )
                and isinstance(entry.get("version"), str)
                and bool(entry["version"])
                and isinstance(entry.get("source"), dict)
                and bool(entry["source"])
                and isinstance(entry.get("artifacts"), list)
                and entry.get("identity_sha256")
                == _canonical_sha256(
                    {
                        key: value
                        for key, value in entry.items()
                        if key != "identity_sha256"
                    }
                )
            ):
                raise ValueError(
                    f"backend_runtime_closure_runtime_package_invalid:{package}"
                )
            markers = entry.get("resolution_markers")
            if markers is not None and not (
                isinstance(markers, list)
                and markers
                and all(isinstance(marker, str) and marker for marker in markers)
                and len(markers) == len(set(markers))
            ):
                raise ValueError(
                    f"backend_runtime_closure_runtime_package_invalid:{package}"
                )
            for artifact in entry["artifacts"]:
                if not (
                    isinstance(artifact, dict)
                    and artifact.get("kind") in {"sdist", "wheel"}
                    and isinstance(artifact.get("url"), str)
                    and str(artifact["url"]).startswith("https://")
                    and isinstance(artifact.get("hash"), str)
                    and str(artifact["hash"]).startswith("sha256:")
                    and _is_sha256(str(artifact["hash"]).removeprefix("sha256:"))
                ):
                    raise ValueError(
                        f"backend_runtime_closure_runtime_package_invalid:{package}"
                    )
        raw_virtual_sources = row.get("virtual_sources", {})
        if not isinstance(raw_virtual_sources, dict):
            raise ValueError(
                f"backend_runtime_closure_runtime_package_invalid:{package}"
            )
        for source_ref, digest in raw_virtual_sources.items():
            if not (
                isinstance(source_ref, str)
                and "://" in source_ref
                and source_ref not in virtual_sources
                and _is_sha256(digest)
            ):
                raise ValueError(
                    f"backend_runtime_closure_runtime_package_invalid:{package}"
                )
            virtual_sources.add(source_ref)
    return len(virtual_sources)


def _validate_bound_runtime_files(
    repo_root: Path,
    raw: Any,
    *,
    inventory_name: str,
    declared_sources: set[str],
) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict):
        raise ValueError(
            f"backend_runtime_closure_{inventory_name}_files_invalid"
        )
    for raw_path, row in raw.items():
        source_path = _portable_archive_path(
            raw_path,
            label=f"{inventory_name}_file_path",
        )
        roles = row.get("roles") if isinstance(row, dict) else None
        backend_kinds = row.get("backend_kinds") if isinstance(row, dict) else None
        if not (
            isinstance(row, dict)
            and set(row) == {"sha256", "roles", "backend_kinds"}
            and source_path not in declared_sources
            and _is_sha256(row.get("sha256"))
            and _string_list(roles, label=f"{inventory_name}_file_roles")
            == sorted(roles)
            and _string_list(
                backend_kinds,
                label=f"{inventory_name}_file_backend_kinds",
            )
            == sorted(backend_kinds)
        ):
            raise ValueError(
                f"backend_runtime_closure_{inventory_name}_file_invalid:{raw_path}"
            )
        source = repo_root / source_path
        try:
            source.resolve(strict=True).relative_to(repo_root)
        except (FileNotFoundError, ValueError) as exc:
            raise FileNotFoundError(
                f"backend_runtime_closure_bound_file_missing:{source_path}"
            ) from exc
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(
                f"backend_runtime_closure_bound_file_missing:{source_path}"
            )
        if _sha256(source) != row["sha256"]:
            raise ValueError(
                f"backend_runtime_closure_bound_file_hash_mismatch:{source_path}"
            )
        declared_sources.add(source_path)
    return raw


def _validated_backend_runtime_closure(
    repo_root: Path,
    raw: Any,
    *,
    additional_archive_roots: set[str] | None = None,
    expected_release_id: str = "operate_v0_58_0",
) -> list[tuple[str, Path, str]]:
    top_level = {
        "schema_version",
        "release_id",
        "status",
        "terminal",
        "portable",
        "source_suite_sha256",
        "archived_files",
        "repo_tracked_files",
        "separately_bundled_files",
        "external_sources",
        "backend_links",
        "runtime_packages",
        "summary",
        "identity_sha256",
    }
    if not (
        isinstance(raw, dict)
        and set(raw) == top_level
        and raw.get("schema_version") == "operate-backend-runtime-closure-v1"
        and re.fullmatch(r"operate_v\d+_\d+_\d+", expected_release_id) is not None
        and raw.get("release_id") == expected_release_id
        and raw.get("status") == "backend_runtime_closure_complete"
        and raw.get("terminal") is True
        and raw.get("portable") is True
        and isinstance(raw.get("source_suite_sha256"), str)
        and len(raw["source_suite_sha256"]) == 64
        and isinstance(raw.get("archived_files"), dict)
        and isinstance(raw.get("backend_links"), dict)
        and isinstance(raw.get("runtime_packages"), dict)
    ):
        raise ValueError("backend_runtime_closure_invalid")
    archived_files = raw["archived_files"]
    external_sources = _validate_external_sources(raw["external_sources"])
    backend_links = raw["backend_links"]
    if any(
        not (
            isinstance(name, str)
            and name in BACKEND_LINKS
            and isinstance(target, str)
            and BACKEND_LINKS[name] == target
        )
        for name, target in backend_links.items()
    ):
        raise ValueError("backend_runtime_closure_backend_links_invalid")
    declared_sources: set[str] = set()
    entries: list[tuple[str, Path, str]] = []
    archived_roots: set[str] = set()
    for archive_name, row in archived_files.items():
        archive_path = PurePosixPath(str(archive_name))
        if not (
            isinstance(archive_name, str)
            and archive_name
            and "\\" not in archive_name
            and not archive_path.is_absolute()
            and ".." not in archive_path.parts
            and archive_path.parts[:1] == ("backends",)
            and len(archive_path.parts) > 2
            and archive_path.as_posix() == archive_name
            and isinstance(row, dict)
            and set(row) == {"source_path", "sha256", "roles", "backend_kinds"}
        ):
            raise ValueError("backend_runtime_closure_archived_file_invalid")
        source_path = _portable_archive_path(
            row["source_path"],
            label="source_path",
        )
        digest = str(row["sha256"])
        expected_archive = _expected_backend_archive_path(source_path)
        _string_list(row["roles"], label="roles")
        _string_list(row["backend_kinds"], label="backend_kinds")
        if not (
            expected_archive == archive_name
            and source_path not in declared_sources
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                f"backend_runtime_closure_archived_file_invalid:{archive_name}"
            )
        source = repo_root / source_path
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(
                f"backend_runtime_closure_source_missing:{source_path}"
            )
        try:
            source = source.resolve(strict=True)
            source.relative_to(repo_root.resolve())
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(
                f"backend_runtime_closure_source_outside_repo:{source_path}"
            ) from exc
        if _sha256(source) != digest:
            raise ValueError(f"backend_runtime_closure_hash_mismatch:{source_path}")
        declared_sources.add(source_path)
        archived_roots.add(archive_path.parts[1])
        entries.append((archive_name, source, digest))
    available_archive_roots = archived_roots | (additional_archive_roots or set())
    if not set(backend_links.values()).issubset(available_archive_roots):
        raise ValueError("backend_runtime_closure_backend_links_invalid")
    for source_path in declared_sources:
        parts = PurePosixPath(source_path).parts
        if parts[0] == "works":
            works_name = parts[1]
            if backend_links.get(works_name) != BACKEND_LINKS[works_name]:
                raise ValueError("backend_runtime_closure_backend_links_invalid")
    repo_tracked_files = _validate_bound_runtime_files(
        repo_root,
        raw["repo_tracked_files"],
        inventory_name="repo_tracked",
        declared_sources=declared_sources,
    )
    separately_bundled_files = _validate_bound_runtime_files(
        repo_root,
        raw["separately_bundled_files"],
        inventory_name="separately_bundled",
        declared_sources=declared_sources,
    )
    n_virtual_sources = _validate_runtime_packages(repo_root, raw["runtime_packages"])
    summary = raw.get("summary")
    n_external_files = sum(
        len(record["required_files"]) for record in external_sources.values()
    )
    expected_summary = {
        "n_archived_files": len(archived_files),
        "n_backend_links": len(backend_links),
        "n_external_sources": len(external_sources),
        "n_repo_tracked_files": len(repo_tracked_files),
        "n_runtime_packages": len(raw["runtime_packages"]),
        "n_separately_bundled_files": len(separately_bundled_files),
        "n_source_assets": None,
        "n_unresolved": 0,
        "n_virtual_sources": n_virtual_sources,
    }
    if not (
        isinstance(summary, dict)
        and set(summary) == set(expected_summary)
        and all(type(value) is int and value >= 0 for value in summary.values())
        and all(
            expected is None or summary[field] == expected
            for field, expected in expected_summary.items()
        )
        and (
            summary["n_archived_files"]
            + summary["n_repo_tracked_files"]
            + summary["n_separately_bundled_files"]
            + summary["n_virtual_sources"]
            <= summary["n_source_assets"]
            <= summary["n_archived_files"]
            + n_external_files
            + summary["n_repo_tracked_files"]
            + summary["n_separately_bundled_files"]
            + summary["n_virtual_sources"]
        )
        and raw.get("identity_sha256")
        == _canonical_sha256(
            {key: value for key, value in raw.items() if key != "identity_sha256"}
        )
    ):
        raise ValueError("backend_runtime_closure_identity_invalid")
    return sorted(entries)


def _build_backend_archive(
    repo_root: Path,
    archive_path: Path,
    source_assets: dict[str, dict[str, Any]],
    *,
    backend_source_closure: dict[str, Any],
) -> dict[str, str]:
    zstd = shutil.which("zstd")
    if zstd is None:
        raise RuntimeError("zstd_not_found")
    ngsim_shared_blobs = (
        _ngsim_us101_blob_contract(source_assets["ngsim_us101"])
        if "ngsim_us101" in source_assets
        else {}
    )
    additional_archive_roots = {
        PurePosixPath(str(row["archive_path"])).parts[1]
        for contract in source_assets.values()
        for row in contract["files"].values()
        if PurePosixPath(str(row["archive_path"])).parts[:1] == ("backends",)
        and len(PurePosixPath(str(row["archive_path"])).parts) > 2
    }
    closure = _validated_backend_runtime_closure(
        repo_root,
        backend_source_closure,
        additional_archive_roots=additional_archive_roots,
        expected_release_id=str(backend_source_closure.get("release_id") or ""),
    )
    bundled_source_files: dict[str, str] = {}
    for contract in source_assets.values():
        contract_delivery = contract.get("delivery")
        for install_path, row in contract["files"].items():
            delivery = row.get("delivery", contract_delivery)
            if delivery == "upstream_fetch":
                continue
            digest = str(row.get("sha256") or "")
            existing = bundled_source_files.setdefault(install_path, digest)
            if existing != digest:
                raise ValueError(
                    "backend_runtime_closure_separately_bundled_file_conflict:"
                    f"{install_path}"
                )
    for install_path, row in backend_source_closure[
        "separately_bundled_files"
    ].items():
        if bundled_source_files.get(install_path) != row["sha256"]:
            raise ValueError(
                "backend_runtime_closure_separately_bundled_file_unbound:"
                f"{install_path}"
            )
    expected_files = {name: digest for name, _source, digest in closure}
    closure_archive_paths = set(expected_files)
    process = subprocess.Popen(
        [zstd, "-T0", "-10", "-q", "-o", str(archive_path)],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    files: dict[str, str] = {}
    try:
        with tarfile.open(fileobj=process.stdin, mode="w|") as archive:
            for archive_name, source, _digest in closure:
                _add_archive_file(
                    archive,
                    source=source,
                    archive_name=archive_name,
                    files=files,
                )
            for source_id, contract in source_assets.items():
                contract_delivery = contract.get("delivery")
                for install_path, row in contract["files"].items():
                    archive_path_value = str(row["archive_path"])
                    if row.get("delivery", contract_delivery) == "upstream_fetch":
                        continue
                    digest = str(row["sha256"])
                    if archive_path_value in closure_archive_paths:
                        raise ValueError(
                            "backend_runtime_closure_source_asset_collision:"
                            f"{archive_path_value}"
                        )
                    expected = expected_files.setdefault(archive_path_value, digest)
                    if expected != digest:
                        raise ValueError(
                            "backend_runtime_closure_source_asset_collision:"
                            f"{archive_path_value}"
                        )
                    if archive_path_value in files:
                        if (
                            source_id != "ngsim_us101"
                            or ngsim_shared_blobs.get(archive_path_value) != digest
                            or files[archive_path_value] != digest
                        ):
                            raise ValueError(
                                "backend_runtime_closure_source_asset_collision:"
                                f"{archive_path_value}"
                            )
                        continue
                    _add_archive_file(
                        archive,
                        source=repo_root / install_path,
                        archive_name=archive_path_value,
                        files=files,
                    )
    except Exception:
        process.stdin.close()
        process.terminate()
        process.wait()
        archive_path.unlink(missing_ok=True)
        raise
    process.stdin.close()
    if process.wait() != 0:
        archive_path.unlink(missing_ok=True)
        raise RuntimeError("backend_archive_compression_failed")
    if files != expected_files:
        archive_path.unlink(missing_ok=True)
        raise ValueError("backend_runtime_closure_archive_hash_mismatch")
    try:
        _validate_tar_zst_files(archive_path, expected_files, label="backend")
    except Exception:
        archive_path.unlink(missing_ok=True)
        raise
    return dict(sorted(expected_files.items()))


def _backend_license_members(
    backend_runtime_closure: dict[str, Any],
    source_assets: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    by_root: dict[str, set[str]] = {}
    for archive_name, row in backend_runtime_closure["archived_files"].items():
        if "redistribution_license" not in row["roles"]:
            continue
        root = PurePosixPath(archive_name).parts[1]
        by_root.setdefault(root, set()).add(archive_name)
    for contract in source_assets.values():
        for row in contract["files"].values():
            if not set(row.get("roles") or []) & {"license", "redistribution_license"}:
                continue
            archive_name = str(row["archive_path"])
            archive_path = PurePosixPath(archive_name)
            if archive_path.parts[:1] != ("backends",) or len(archive_path.parts) < 3:
                raise ValueError("backend_license_archive_path_invalid")
            by_root.setdefault(archive_path.parts[1], set()).add(archive_name)
    required_roots = set(backend_runtime_closure["backend_links"].values())
    if not required_roots.issubset(by_root):
        raise ValueError(
            f"backend_redistribution_license_missing:{sorted(required_roots - by_root)}"
        )
    return {root: sorted(by_root[root]) for root in sorted(required_roots)}


def _formal_result_tree_files(
    *,
    repo_root: Path,
    release_root: Path,
    release_manifest: dict[str, Any],
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """Return the exact two finalizer-owned formal result trees, if published."""

    evidence = release_manifest.get("formal_evidence")
    if not isinstance(evidence, dict):
        return {}, {}
    names = {
        "logical_batch_manifest": "logical_persistent",
        "realtime_batch_manifest": "realtime_persistent",
    }
    present = {name for name in names if name in evidence}
    if not present:
        return {}, {}
    if present != set(names):
        raise ValueError("formal_result_tree_pair_incomplete")

    result_root = (release_root / "formal_results").resolve()
    files: dict[str, str] = {}
    contracts: dict[str, dict[str, Any]] = {}
    models: set[str] = set()
    for name, mode in names.items():
        binding = evidence[name]
        if not isinstance(binding, dict):
            raise ValueError(f"formal_result_tree_binding_invalid:{name}")
        path_value = binding.get("path")
        index_value = binding.get("tree_index_path")
        if not isinstance(path_value, str) or not isinstance(index_value, str):
            raise ValueError(f"formal_result_tree_binding_invalid:{name}")
        _, manifest_path = _repo_relative_path(
            repo_root, path_value, label=f"formal_result_tree_{name}"
        )
        _, index_path = _repo_relative_path(
            repo_root, index_value, label=f"formal_result_tree_index_{name}"
        )
        try:
            relative_manifest = manifest_path.relative_to(result_root)
        except ValueError as exc:
            raise ValueError(f"formal_result_tree_path_invalid:{name}") from exc
        tree_root = manifest_path.parent
        if not (
            len(relative_manifest.parts) == 4
            and relative_manifest.name == "RUN_MANIFEST.json"
            and manifest_path.is_file()
            and not manifest_path.is_symlink()
            and index_path == tree_root / FORMAL_RESULT_TREE_INDEX_NAME
            and index_path.is_file()
            and not index_path.is_symlink()
            and binding.get("interaction_mode") == mode
            and isinstance(binding.get("model"), str)
            and binding.get("model")
            and isinstance(binding.get("treatment_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", binding["treatment_sha256"])
            and relative_manifest.parts[1] == binding["treatment_sha256"]
            and binding.get("sha256") == _sha256(manifest_path)
            and binding.get("tree_index_sha256") == _sha256(index_path)
        ):
            raise ValueError(f"formal_result_tree_binding_invalid:{name}")
        models.add(str(binding["model"]))
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"formal_result_tree_index_invalid:{name}") from exc
        indexed = index.get("files") if isinstance(index, dict) else None
        canonical_index = {
            "schema_version": FORMAL_RESULT_TREE_INDEX_SCHEMA,
            "files": indexed,
        }
        if not (
            isinstance(indexed, list)
            and indexed
            and all(isinstance(row, dict) for row in indexed)
            and index.get("schema_version") == FORMAL_RESULT_TREE_INDEX_SCHEMA
            and index.get("root_sha256") == _canonical_sha256(canonical_index)
            and tree_root.name == index.get("root_sha256")
            and binding.get("tree_root_sha256") == index.get("root_sha256")
        ):
            raise ValueError(f"formal_result_tree_index_invalid:{name}")
        expected_names: set[str] = set()
        for row in indexed:
            relative_value = row.get("path")
            relative = Path(str(relative_value or ""))
            if not (
                isinstance(relative_value, str)
                and relative_value
                and not relative.is_absolute()
                and ".." not in relative.parts
                and relative.as_posix() == relative_value
                and relative_value != FORMAL_RESULT_TREE_INDEX_NAME
                and relative_value not in expected_names
                and isinstance(row.get("size_bytes"), int)
                and row["size_bytes"] >= 0
                and isinstance(row.get("sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
            ):
                raise ValueError(f"formal_result_tree_index_invalid:{name}")
            artifact = tree_root / relative
            if not (
                artifact.is_file()
                and not artifact.is_symlink()
                and artifact.stat().st_size == row["size_bytes"]
                and _sha256(artifact) == row["sha256"]
            ):
                raise ValueError(f"formal_result_tree_artifact_invalid:{name}")
            expected_names.add(relative_value)
        observed_names = {
            path.relative_to(tree_root).as_posix()
            for path in tree_root.rglob("*")
            if path.is_file()
        }
        if observed_names != expected_names | {FORMAL_RESULT_TREE_INDEX_NAME}:
            raise ValueError(f"formal_result_tree_membership_invalid:{name}")
        tree_files: dict[str, str] = {}
        for artifact in (tree_root / relative for relative in expected_names):
            release_relative = artifact.relative_to(release_root).as_posix()
            digest = _sha256(artifact)
            files[release_relative] = digest
            tree_files[release_relative] = digest
        index_relative = index_path.relative_to(release_root).as_posix()
        index_digest = _sha256(index_path)
        files[index_relative] = index_digest
        tree_files[index_relative] = index_digest
        contracts[mode] = {
            "binding": binding,
            "files": dict(sorted(tree_files.items())),
        }
    if len(models) != 1:
        raise ValueError("formal_result_tree_model_mismatch")
    return dict(sorted(files.items())), contracts


def _resolve_compact_formal_evidence(
    *,
    repo_root: Path,
    release_root: Path,
    release_manifest: dict[str, Any],
    formal_contract: dict[str, Any],
    formal_evidence: dict[str, Any],
) -> tuple[Path, str, dict[str, str], dict[str, dict[str, Any]]] | None:
    binding = release_manifest.get("formal_runtime_bundle")
    if binding is None:
        return None
    if not isinstance(binding, dict):
        raise ValueError("formal_runtime_bundle_binding_invalid")

    release_root = release_root.resolve()
    install_root = release_root.relative_to(repo_root).as_posix()
    if install_root != str(
        formal_contract.get("runtime_evidence_root") or ""
    ) or install_root != str(formal_evidence.get("runtime_root") or ""):
        raise ValueError("formal_evidence_root_binding_mismatch")

    relative, runtime_path = _release_relative_file(
        release_root,
        binding.get("path"),
        label="formal_runtime_bundle",
    )
    expected_runtime_sha256 = str(binding.get("sha256") or "")
    expected_scenarios = release_manifest.get("n_scenarios")
    expected_ordered_identity = binding.get("ordered_scenario_identity_sha256")
    expected_binding = {
        "path": relative,
        "sha256": expected_runtime_sha256,
        "size_bytes": runtime_path.stat().st_size if runtime_path.is_file() else -1,
        "schema_version": "operate-formal-runtime-bundle-v1",
        "n_scenarios": expected_scenarios,
        "ordered_scenario_identity_sha256": expected_ordered_identity,
    }
    if not (
        binding == expected_binding
        and runtime_path.is_file()
        and not runtime_path.is_symlink()
        and isinstance(expected_scenarios, int)
        and expected_scenarios > 0
        and isinstance(expected_ordered_identity, str)
        and re.fullmatch(r"[0-9a-f]{64}", expected_ordered_identity) is not None
        and re.fullmatch(r"[0-9a-f]{64}", expected_runtime_sha256) is not None
        and _sha256(runtime_path) == expected_runtime_sha256
    ):
        raise ValueError("formal_runtime_bundle_binding_invalid")
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if not (
        isinstance(runtime, dict)
        and runtime.get("schema_version") == binding["schema_version"]
        and runtime.get("release_id") == release_manifest.get("release_id")
        and runtime.get("formal_evaluation_ready") is True
        and runtime.get("formal_run_blockers") == []
        and runtime.get("implementation_tree_sha256")
        == release_manifest.get("implementation_tree_sha256")
        and runtime.get("core_release_pipeline_sha256")
        == release_manifest.get("core_release_pipeline_sha256")
        and runtime.get("n_scenarios") == expected_scenarios
        and runtime.get("ordered_scenario_identity_sha256")
        == expected_ordered_identity
    ):
        raise ValueError("formal_runtime_bundle_contract_invalid")
    core_binding = release_manifest.get("core_suite")
    if not isinstance(core_binding, dict):
        raise ValueError("formal_runtime_bundle_artifact_invalid:core_suite")
    _, core_path = _release_relative_file(
        release_root,
        core_binding.get("path"),
        label="formal_runtime_bundle_core_suite",
    )
    core = json.loads(core_path.read_text(encoding="utf-8"))
    core_rows = core.get("scenarios") if isinstance(core, dict) else None
    if not isinstance(core_rows, list) or not all(
        isinstance(row, dict) for row in core_rows
    ):
        raise ValueError("formal_runtime_bundle_artifact_invalid:core_suite")
    from scripts.verify_release_integrity import (  # noqa: PLC0415
        _formal_runtime_bundle_valid,
    )

    if not _formal_runtime_bundle_valid(
        release_root,
        release_manifest,
        core,
        list(core_rows),
    ):
        raise ValueError("formal_runtime_bundle_integrity_invalid")

    replay = release_manifest.get("protocol21_replay")
    if not isinstance(replay, dict):
        raise ValueError("formal_runtime_bundle_contract_invalid")
    expected_source_path = f"{install_root}/protocol21_source_suite.json"
    if replay.get("source_suite") != expected_source_path:
        raise ValueError("formal_runtime_bundle_source_suite_invalid")
    manifest_bindings: dict[str, Any] = {
        "core_suite": release_manifest.get("core_suite"),
        "source_suite": {
            "path": "protocol21_source_suite.json",
            "sha256": replay.get("source_suite_sha256"),
        },
        "public_evidence": {
            "path": replay.get("evidence_bundle"),
            "sha256": replay.get("evidence_bundle_sha256"),
        },
        "backend_runtime_closure": release_manifest.get(
            "backend_runtime_closure"
        ),
    }
    runtime_has_candidate = isinstance(runtime.get("candidate_closure"), dict)
    manifest_has_candidate = isinstance(
        release_manifest.get("candidate_closure"), dict
    )
    if not runtime_has_candidate or not manifest_has_candidate:
        raise ValueError("formal_runtime_bundle_candidate_closure_mismatch")
    manifest_bindings["candidate_closure"] = release_manifest["candidate_closure"]

    files = {relative: expected_runtime_sha256}
    for label, manifest_binding in manifest_bindings.items():
        runtime_binding = runtime.get(label)
        if not isinstance(runtime_binding, dict) or not isinstance(
            manifest_binding, dict
        ):
            raise ValueError(f"formal_runtime_bundle_artifact_invalid:{label}")
        artifact_relative, artifact_path = _release_relative_file(
            release_root,
            runtime_binding.get("path"),
            label=f"formal_runtime_bundle_{label}",
        )
        expected = str(runtime_binding.get("sha256") or "")
        if not (
            manifest_binding.get("path") == artifact_relative
            and manifest_binding.get("sha256") == expected
            and artifact_relative not in files
            and len(expected) == 64
            and artifact_path.is_file()
            and not artifact_path.is_symlink()
            and _sha256(artifact_path) == expected
        ):
            raise ValueError(f"formal_runtime_bundle_artifact_invalid:{label}")
        if label in {"core_suite", "source_suite"} and not (
            runtime_binding.get("n_scenarios") == expected_scenarios
            and runtime_binding.get("ordered_scenario_identity_sha256")
            == expected_ordered_identity
        ):
            raise ValueError(f"formal_runtime_bundle_artifact_invalid:{label}")
        if label == "backend_runtime_closure" and (
            runtime_binding.get("identity_sha256")
            != manifest_binding.get("identity_sha256")
        ):
            raise ValueError(f"formal_runtime_bundle_artifact_invalid:{label}")
        if label == "candidate_closure" and (
            runtime_binding.get("identity_set_sha256")
            != manifest_binding.get("identity_set_sha256")
        ):
            raise ValueError(f"formal_runtime_bundle_artifact_invalid:{label}")
        files[artifact_relative] = expected

    formal_result_files, formal_result_trees = _formal_result_tree_files(
        repo_root=repo_root,
        release_root=release_root,
        release_manifest=release_manifest,
    )
    if set(files) & set(formal_result_files):
        raise ValueError("formal_result_tree_artifact_collision")
    files.update(formal_result_files)

    runtime_selection = f"{install_root}/{relative}#scenarios"
    realtime_contract = release_manifest.get("formal_realtime_batch_contract")
    if not (
        formal_contract.get("selection_source") == runtime_selection
        and isinstance(realtime_contract, dict)
        and realtime_contract.get("selection_source") == runtime_selection
        and formal_evidence.get("readiness") == f"{install_root}/{relative}"
    ):
        raise ValueError("formal_runtime_bundle_selection_source_invalid")

    for relative_name in files:
        path = release_root / relative_name
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if canonicalize_repo_owned_paths([payload], repo_root=repo_root) != [
                payload
            ]:
                raise ValueError(
                    f"formal_evidence_nonportable_path:{relative_name}"
                )
    return (
        release_root,
        install_root,
        dict(sorted(files.items())),
        formal_result_trees,
    )


def _resolve_formal_evidence(
    *,
    repo_root: Path,
    release_root: Path,
    release_manifest: dict[str, Any],
) -> tuple[Path, str, dict[str, str], dict[str, dict[str, Any]]]:
    live_identity = implementation_identity(repo_root)
    live_tree = live_identity["implementation_tree_sha256"]
    compact_proof = isinstance(release_manifest.get("formal_runtime_bundle"), dict)
    if not _is_sha256(release_manifest.get("implementation_tree_sha256")):
        raise ValueError("release_implementation_tree_mismatch")
    if not compact_proof and release_manifest.get("implementation_tree_sha256") != live_tree:
        raise ValueError("release_implementation_tree_mismatch")
    live_pipeline_hash = live_identity.get("core_release_pipeline_sha256")
    if not (
        isinstance(live_pipeline_hash, str)
        and len(live_pipeline_hash) == 64
        and _is_sha256(release_manifest.get("core_release_pipeline_sha256"))
        and (compact_proof or release_manifest.get("core_release_pipeline_sha256") == live_pipeline_hash)
    ):
        raise ValueError("release_core_pipeline_mismatch")
    if compact_proof and (
        release_manifest["implementation_tree_sha256"] != live_tree
        or release_manifest["core_release_pipeline_sha256"] != live_pipeline_hash
    ):
        print(
            "INFO: packaging immutable admission evidence with newer code; "
            "current evaluation runs bind their own implementation identity",
            file=sys.stderr,
        )

    formal_contract = release_manifest.get("formal_batch_contract")
    formal_evidence = release_manifest.get("formal_evidence")
    pipeline_artifacts = release_manifest.get("pipeline_artifacts")
    if not all(
        isinstance(value, dict)
        for value in (formal_contract, formal_evidence, pipeline_artifacts)
    ):
        raise ValueError("formal_evidence_contract_missing")
    assert isinstance(formal_contract, dict)
    assert isinstance(formal_evidence, dict)
    assert isinstance(pipeline_artifacts, dict)

    compact = _resolve_compact_formal_evidence(
        repo_root=repo_root,
        release_root=release_root,
        release_manifest=release_manifest,
        formal_contract=formal_contract,
        formal_evidence=formal_evidence,
    )
    if compact is not None:
        return compact

    runtime_roots = {
        str(release_manifest.get("pipeline_dir") or ""),
        str(formal_contract.get("runtime_evidence_root") or ""),
        str(formal_evidence.get("runtime_root") or ""),
        str(pipeline_artifacts.get("path") or ""),
    }
    if "" in runtime_roots or len(runtime_roots) != 1:
        raise ValueError("formal_evidence_root_binding_mismatch")
    install_root = runtime_roots.pop()
    install_relative = Path(install_root)
    if (
        install_relative.is_absolute()
        or ".." in install_relative.parts
        or len(install_relative.parts) < 3
        or install_relative.parts[0] != "release"
    ):
        raise ValueError("formal_evidence_install_root_invalid")
    evidence_root = (repo_root / install_relative).resolve()
    try:
        evidence_root.relative_to((repo_root / "release").resolve())
    except ValueError as exc:
        raise ValueError("formal_evidence_install_root_invalid") from exc
    if not evidence_root.is_dir():
        raise ValueError("formal_evidence_root_missing")

    bindings: dict[str, Any] = {
        "protocol2_v21_pipeline_manifest.json": pipeline_artifacts.get(
            "pipeline_manifest_sha256"
        ),
    }
    stage_artifacts = pipeline_artifacts.get("stage_artifacts")
    if not isinstance(stage_artifacts, dict) or not stage_artifacts:
        raise ValueError("formal_evidence_stage_bindings_missing")
    for stage, raw_binding in stage_artifacts.items():
        if not isinstance(raw_binding, dict):
            raise ValueError(f"formal_evidence_stage_binding_invalid:{stage}")
        relative = str(raw_binding.get("relative_path") or "")
        expected = raw_binding.get("sha256")
        if not relative:
            raise ValueError(f"formal_evidence_stage_binding_invalid:{stage}")
        if relative in bindings:
            if bindings[relative] != expected:
                raise ValueError(f"formal_evidence_stage_binding_conflict:{stage}")
            continue
        bindings[relative] = expected
    for relative, expected in bindings.items():
        path = Path(relative)
        resolved = (evidence_root / path).resolve()
        try:
            resolved.relative_to(evidence_root)
        except ValueError as exc:
            raise ValueError(
                f"formal_evidence_binding_path_invalid:{relative}"
            ) from exc
        if (
            path.is_absolute()
            or ".." in path.parts
            or not resolved.is_file()
            or not isinstance(expected, str)
            or len(expected) != 64
            or _sha256(resolved) != expected
        ):
            raise ValueError(f"formal_evidence_binding_invalid:{relative}")

    files: dict[str, str] = {}
    for relative, expected in bindings.items():
        path = evidence_root / relative
        if path.is_symlink():
            raise ValueError(f"formal_evidence_symlink_forbidden:{relative}")
        if path.suffix in {".json", ".jsonl"}:
            payloads = (
                [json.loads(path.read_text(encoding="utf-8"))]
                if path.suffix == ".json"
                else [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            )
            if canonicalize_repo_owned_paths(payloads, repo_root=repo_root) != payloads:
                raise ValueError(f"formal_evidence_nonportable_path:{relative}")
        files[relative] = expected
    return evidence_root, install_root, files, {}


def _build_formal_evidence_archive(
    *,
    evidence_root: Path,
    install_root: str,
    files: dict[str, str],
    archive_path: Path,
) -> None:
    zstd = shutil.which("zstd")
    if zstd is None:
        raise RuntimeError("zstd_not_found")
    process = subprocess.Popen(
        [zstd, "-T0", "-10", "-q", "-o", str(archive_path)],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    try:
        with tarfile.open(fileobj=process.stdin, mode="w|") as archive:
            for relative in files:
                member = (Path(install_root) / relative).as_posix()
                archive.add(
                    evidence_root / relative,
                    arcname=member,
                    recursive=False,
                    filter=_tar_filter,
                )
    except Exception:
        process.stdin.close()
        process.terminate()
        process.wait()
        archive_path.unlink(missing_ok=True)
        raise
    process.stdin.close()
    if process.wait() != 0:
        archive_path.unlink(missing_ok=True)
        raise RuntimeError("formal_evidence_archive_compression_failed")


def build_operate_bundle(
    *,
    repo_root: Path,
    release_dir: Path,
    output_dir: Path,
    repo_id: str = DEFAULT_REPO_ID,
    include_backends: bool = True,
    release_manifest_path: Path | None = None,
    versionless_filenames: bool = False,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    release_dir = release_dir.resolve()
    output_dir = output_dir.resolve()
    release_dir.relative_to(repo_root)
    output_dir.relative_to(repo_root)
    release_id = release_dir.name
    if (
        re.fullmatch(r"operate_v\d+_\d+_\d+", release_id) is None
        or release_dir != repo_root / "release" / release_id
    ):
        raise ValueError("release_directory_mismatch")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"bundle_output_not_empty:{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    release_manifest_path = (
        release_manifest_path.resolve()
        if release_manifest_path is not None
        else release_dir / "manifest.json"
    )
    try:
        release_manifest_path.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError("release_manifest_path_outside_repo") from exc
    if release_manifest_path.is_symlink() or not release_manifest_path.is_file():
        raise ValueError("release_manifest_path_invalid")
    release_manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))
    if release_manifest.get("release_id") != release_id:
        raise ValueError("release_id_mismatch")
    release_version = ".".join(release_id.removeprefix("operate_v").split("_"))
    if release_manifest.get("release_version") not in {None, release_version}:
        raise ValueError("release_version_mismatch")
    if release_manifest.get("formal_evaluation_ready") is not True:
        raise ValueError("release_not_formal_evaluation_ready")

    source_assets: dict[str, dict[str, Any]] = {}
    backend_runtime_closure: dict[str, Any] = {}
    backend_closure_path: Path | None = None
    backend_closure_binding: dict[str, str] | None = None
    if include_backends:
        source_suite_path, source_suite_sha256, source_suite = _release_source_suite(
            repo_root,
            release_dir,
            release_manifest,
        )
        dynasched_source_assets = _collect_dynasched_source_assets(
            repo_root,
            source_suite,
        )
        closure_preview = json.loads(
            (release_dir / "backend_runtime_closure.json").read_text(encoding="utf-8")
        )
        preview_external_sources = (
            closure_preview.get("external_sources")
            if isinstance(closure_preview, dict)
            else None
        )
        nrel_source_assets = _collect_nrel_microgrid_source_assets(
            repo_root,
            source_suite,
            (preview_external_sources or {}).get("nrel_microgrid"),
        )
        ngsim_source_assets = _collect_ngsim_us101_source_assets(
            repo_root,
            source_suite,
        )
        contracts = {
            "dynasched": dynasched_source_assets,
            "ngsim_us101": ngsim_source_assets,
            "nrel_microgrid": nrel_source_assets,
            **{
                source_id: _collect_exact_source_assets(
                    repo_root,
                    source_suite,
                    source_id=source_id,
                )
                for source_id in _EXACT_SOURCE_SPECS
            },
        }
        for source_id, contract in contracts.items():
            if not contract["n_scenarios"]:
                continue
            contract["source_suite"] = {
                "path": source_suite_path.relative_to(repo_root).as_posix(),
                "sha256": source_suite_sha256,
            }
            source_assets[source_id] = contract
        source_asset_archive_roots = {
            PurePosixPath(str(row["archive_path"])).parts[1]
            for contract in source_assets.values()
            for row in contract["files"].values()
            if PurePosixPath(str(row["archive_path"])).parts[:1] == ("backends",)
            and len(PurePosixPath(str(row["archive_path"])).parts) > 2
        }
        (
            backend_closure_path,
            backend_closure_binding,
            backend_runtime_closure,
        ) = _release_backend_runtime_closure(
            repo_root,
            release_dir,
            release_manifest,
            additional_archive_roots=source_asset_archive_roots,
        )

    (
        evidence_root,
        evidence_install_root,
        evidence_files,
        formal_result_trees,
    ) = _resolve_formal_evidence(
        repo_root=repo_root,
        release_root=release_dir,
        release_manifest=release_manifest,
    )
    copied_manifest = output_dir / "release_manifest.json"
    shutil.copy2(release_manifest_path, copied_manifest)
    readme = output_dir / "README.md"
    shutil.copy2(repo_root / "docs" / "hf" / "OPERATE_DATASET_CARD.md", readme)
    tracked = {
        "README.md": _sha256(readme),
        "release_manifest.json": _sha256(copied_manifest),
    }
    candidate_closure_binding = _copy_optional_candidate_closure(
        repo_root=repo_root,
        release_dir=release_dir,
        release_manifest=release_manifest,
        output_dir=output_dir,
        tracked=tracked,
    )
    candidate_evidence_archive: str | None = None
    candidate_evidence_files: dict[str, str] = {}
    compact_formal_release = release_manifest.get("formal_runtime_bundle") is not None
    if candidate_closure_binding is not None and not compact_formal_release:
        candidate_closure_payload = json.loads(
            (output_dir / "candidate_closure.json").read_text(encoding="utf-8")
        )
        candidate_evidence_files = _candidate_evidence_files(
            repo_root,
            candidate_closure_payload,
        )
        candidate_evidence_archive = (
            "candidate_evidence.tar.zst"
            if versionless_filenames
            else f"{release_id}_candidate_evidence.tar.zst"
        )
        _build_formal_evidence_archive(
            evidence_root=repo_root,
            install_root="candidate_evidence",
            files=candidate_evidence_files,
            archive_path=output_dir / candidate_evidence_archive,
        )
        tracked[candidate_evidence_archive] = _sha256(
            output_dir / candidate_evidence_archive
        )
    bundled_backend_closure: dict[str, Any] | None = None
    if backend_closure_path is not None and backend_closure_binding is not None:
        closure_name = backend_closure_path.name
        copied_closure = output_dir / closure_name
        shutil.copyfile(backend_closure_path, copied_closure)
        tracked[closure_name] = _sha256(copied_closure)
        bundled_backend_closure = {
            **backend_closure_binding,
            "path": closure_name,
            "sha256": tracked[closure_name],
        }
    evidence_archive_name = (
        "formal_evidence.tar.zst"
        if versionless_filenames
        else f"{release_id}_formal_evidence.tar.zst"
    )
    evidence_archive_path = output_dir / evidence_archive_name
    _build_formal_evidence_archive(
        evidence_root=evidence_root,
        install_root=evidence_install_root,
        files=evidence_files,
        archive_path=evidence_archive_path,
    )
    tracked[evidence_archive_name] = _sha256(evidence_archive_path)
    backend_files: dict[str, str] = {}
    backend_licenses: dict[str, list[str]] = {}
    archive_name = (
        "backends.tar.zst"
        if versionless_filenames
        else f"{release_id}_backends.tar.zst"
    )
    if include_backends:
        archive_path = output_dir / archive_name
        backend_files = _build_backend_archive(
            repo_root,
            archive_path,
            source_assets,
            backend_source_closure=backend_runtime_closure,
        )
        backend_licenses = _backend_license_members(
            backend_runtime_closure,
            source_assets,
        )
        if any(
            member not in backend_files
            for members in backend_licenses.values()
            for member in members
        ):
            raise ValueError("backend_redistribution_license_archive_mismatch")
        tracked[archive_name] = _sha256(archive_path)

    n_scenarios = int(
        release_manifest.get("n_scenarios")
        or release_manifest.get("n_core_scenarios")
        or 0
    )
    manifest: dict[str, Any] = {
        "schema_version": "operate-runtime-bundle-v2",
        "bundle_kind": (
            "public_runtime_companion"
            if versionless_filenames
            else "private_runtime_companion"
        ),
        "release_id": release_id,
        "release_version": release_version,
        "implementation_tree_sha256": release_manifest.get(
            "implementation_tree_sha256"
        ),
        "core_release_pipeline_sha256": release_manifest.get(
            "core_release_pipeline_sha256"
        ),
        "hf_repo_id": repo_id,
        "visibility": "public" if versionless_filenames else "private",
        "n_scenarios": n_scenarios,
        "n_files": len(tracked),
        "n_release_artifacts": 1,
        "n_backend_files": len(backend_files),
        "files": tracked,
        "release_manifest_sha256": tracked["release_manifest.json"],
        "formal_evidence_archive": evidence_archive_name,
        "formal_evidence_install_root": evidence_install_root,
        "formal_evidence_required_files": sorted(evidence_files),
        "formal_evidence_files": evidence_files,
        **(
            {"formal_result_trees": formal_result_trees}
            if formal_result_trees
            else {}
        ),
        "backend_links": backend_runtime_closure.get("backend_links", {})
        if include_backends
        else {},
        "source_assets": source_assets if include_backends else {},
    }
    if versionless_filenames:
        manifest.pop("release_version")
        manifest["distribution_profile"] = "public_runtime_companion_v1"
    if include_backends:
        manifest["backend_archive"] = archive_name
        manifest["backend_archive_files"] = backend_files
        manifest["backend_runtime_closure"] = bundled_backend_closure
        manifest["external_sources"] = backend_runtime_closure["external_sources"]
        manifest["runtime_packages"] = backend_runtime_closure["runtime_packages"]
        manifest["backend_licenses"] = backend_licenses
    if candidate_closure_binding is not None:
        manifest["candidate_closure"] = candidate_closure_binding
        if candidate_evidence_archive is not None:
            manifest["candidate_evidence_archive"] = candidate_evidence_archive
            manifest["candidate_evidence_install_root"] = "candidate_evidence"
            manifest["candidate_evidence_required_files"] = sorted(
                candidate_evidence_files
            )
            manifest["candidate_evidence_files"] = candidate_evidence_files
    _write_json(output_dir / "MANIFEST.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--release-manifest", type=Path)
    parser.add_argument(
        "--versionless-filenames",
        action="store_true",
        help="use stable public archive names; release identity remains in MANIFEST.json",
    )
    args = parser.parse_args()
    manifest = build_operate_bundle(
        repo_root=REPO_ROOT,
        release_dir=args.release_dir,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        include_backends=True,
        release_manifest_path=args.release_manifest,
        versionless_filenames=args.versionless_filenames,
    )
    print(
        json.dumps(
            {
                "release_id": manifest["release_id"],
                "n_scenarios": manifest["n_scenarios"],
                "n_files": manifest["n_files"],
                "output": str(args.output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
