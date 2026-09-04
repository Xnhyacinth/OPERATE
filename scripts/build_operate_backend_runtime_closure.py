#!/usr/bin/env python3
"""Compile the minimal OPERATE backend runtime closure from frozen source locks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import tomllib
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_SUITE = REPO_ROOT / "release/operate_v0_61_0/protocol21_source_suite.json"
DEFAULT_RUNTIME_SOURCE_LOCK = (
    REPO_ROOT / "sources/locks/operate_v0_61_0/backend_runtime_sources.json"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_RELEASE_ID_RE = re.compile(r"operate_v\d+_\d+_\d+")

_ARCHIVE_ROOTS = {
    "works/OpenDSS-IEEE13": ("OpenDSS-IEEE13", "opendss_ieee13"),
    "works/PGLib-OPF": ("PGLib-OPF", "pglib_opf"),
    "works/PyVRP-Instances": ("PyVRP-Instances", "pyvrp_instances"),
    "works/RESCO": ("RESCO", "resco"),
    "works/VRPLIB": ("VRPLIB", "vrplib"),
    "works/sumo_ingolstadt": ("sumo_ingolstadt", "sumo_ingolstadt"),
}

_EXTERNAL_POLICIES = {
    "works/CityLearn": {
        "source_id": "citylearn",
        "delivery": "git_checkout",
        "url": "https://github.com/intelligent-environments-lab/CityLearn.git",
        "revision": "29062af6d077409e1c37a3e53a6cac30fd4d02bc",
        "license_status": "verified_mit",
    },
    "works/JSPLIB-Instances": {
        "source_id": "jsplib",
        "delivery": "git_checkout",
        "url": "https://github.com/tamy0612/JSPLIB.git",
        "revision": "eea2b60dd7e2f5c907ff7302662c61812eb7efdf",
        "license_status": "unspecified_upstream",
    },
    "works/M5": {
        "source_id": "m5",
        "delivery": "user_provided",
        "url": "https://www.kaggle.com/competitions/m5-forecasting-accuracy",
        "revision": "m5-forecasting-accuracy 2020-06-01 files",
        "license_status": "kaggle_competition_terms",
    },
    "works/pglib-uc": {
        "source_id": "pglib_uc",
        "delivery": "git_checkout",
        "url": "https://github.com/power-grid-lib/pglib-uc.git",
        "revision": "39a7f38cf4703de92f0291f0c873c2e98c789301",
        "license_status": "verified_cc_by_4_data_and_mit_software",
    },
    "works/sumo_ingolstadt_upstream": {
        "source_id": "sumo_ingolstadt_upstream",
        "delivery": "git_checkout",
        "url": "https://github.com/TUM-VT/sumo_ingolstadt.git",
        "revision": "e0a95deebe200ff81b6705044d66310d6266d42b",
        "license_status": "verified_apache_2_0",
    },
    "works/clusterdata": {
        "source_id": "alibaba_clusterdata",
        "delivery": "upstream_fetch",
        "url": "https://github.com/alibaba/clusterdata.git",
        "revision": "0d0f3f1efdbf1add6a7bcc63676eafbd1eb11f71",
        "license_status": "upstream_research_terms",
    },
    "works/nrel-microgrid": {
        "source_id": "nrel_microgrid",
        "delivery": "user_provided",
        "url": (
            "https://data.openei.org/submissions/4520 + https://nsrdb.nrel.gov "
            "+ https://developer.nrel.gov/docs/solar/pvwatts/v8/ "
            "+ https://apps.openei.org/IURDB/"
        ),
        "revision": "OEDI ComStock AMY2018 2021 release 1",
        "license_status": "nrel_oedi_attribution_required",
    },
}

_REPO_TRACKED_ROOTS = ("sources/alibaba", "sources/resco")
_SEPARATELY_BUNDLED_ROOTS = (
    "sources/dynasched",
    "works/DynaSchedBench",
    "works/REALM-Bench-direct-pilot",
    "works/autonomous_driving/ngsim/recovery/us101-v60-seven/bundles",
)
_VIRTUAL_SCHEMES = {"pandapower-simbench"}

_PACKAGE_BACKENDS = {
    "citylearn": {"citylearn"},
    "dss-python": {"opendss_fresh_feeders", "opendss_ieee13"},
    "dsbx": {"dynasched_flexible_job_shop"},
    "eclipse-sumo": {"sumo", "sumo_ego"},
    "or-gym": {"orgym_invmgmt"},
    "pandapower": {"cigre_distribution", "pandapower_acopf", "pandapower_lv"},
    "pymgrid": {"pymgrid_economic_dispatch"},
    "pyvrp": {"pyvrp_cvrp", "pyvrp_vrptw"},
    "simbench": {"cigre_distribution"},
}

_SUPPLEMENTAL_SOURCE_BACKENDS = {"orgym": {"orgym_invmgmt"}}
_OPENDSS_BACKENDS = {"opendss_fresh_feeders", "opendss_ieee13"}


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json_bytes(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{label}_missing:{path}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label}_invalid_json:{path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label}_not_object:{path}")
    return payload, raw


def _portable_path(raw: object, *, label: str) -> str:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise ValueError(f"{label}_not_portable:{raw}")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != raw:
        raise ValueError(f"{label}_not_portable:{raw}")
    return raw


def _source_reference(raw: object, *, label: str) -> str:
    if isinstance(raw, str) and "://" in raw:
        scheme = raw.split("://", 1)[0]
        if scheme in _VIRTUAL_SCHEMES and raw.split("://", 1)[1]:
            return raw
        raise ValueError(f"{label}_virtual_scheme_unsupported:{raw}")
    return _portable_path(raw, label=label)


def _locked_sha256(raw: object, *, label: str) -> str:
    if not isinstance(raw, str) or _SHA256_RE.fullmatch(raw) is None:
        raise ValueError(f"{label}_invalid")
    return raw


def _verified_repo_file(repo_root: Path, raw_path: str, digest: str) -> Path:
    path = repo_root / raw_path
    try:
        path.resolve(strict=True).relative_to(repo_root.resolve())
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(f"source_asset_missing_or_outside_repo:{raw_path}") from exc
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"source_asset_not_regular_file:{raw_path}")
    if _file_sha256(path) != digest:
        raise ValueError(f"source_asset_hash_mismatch:{raw_path}")
    return path


def _root_match(path: str, roots: Mapping[str, object] | tuple[str, ...]) -> str | None:
    for root in sorted(roots, key=len, reverse=True):
        if path == root or path.startswith(f"{root}/"):
            return root
    return None


def _release_id(source_suite: Mapping[str, Any], source_suite_path: Path) -> str:
    value = source_suite.get("release_id")
    if isinstance(value, str) and _RELEASE_ID_RE.fullmatch(value):
        return value
    matches = sorted(set(_RELEASE_ID_RE.findall(source_suite_path.as_posix())))
    if len(matches) == 1:
        return matches[0]
    raise ValueError("source_suite_release_id_missing")


def _scenario_contract(
    repo_root: Path,
    row: Mapping[str, Any],
) -> tuple[dict[str, set[str]], dict[str, str]]:
    scenario_path = _portable_path(row.get("path"), label="scenario_path")
    path = repo_root / scenario_path
    try:
        path.resolve(strict=True).relative_to(repo_root.resolve())
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(
            f"scenario_path_missing_or_outside_repo:{scenario_path}"
        ) from exc
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict) or not isinstance(body.get("source_contract"), dict):
        raise ValueError(f"scenario_source_contract_missing:{scenario_path}")
    contract = body["source_contract"]
    roles: dict[str, set[str]] = defaultdict(set)
    for role, values in contract.items():
        if role in {"derived_window", "file_sha256s"} or not isinstance(values, list):
            continue
        for raw in values:
            source_path = _source_reference(raw, label="source_contract_path")
            roles[source_path].add(str(role))
    raw_hashes = contract.get("file_sha256s") or {}
    if not isinstance(raw_hashes, dict):
        raise ValueError(f"scenario_source_contract_hashes_invalid:{scenario_path}")
    hashes = {
        _source_reference(source_path, label="source_contract_path"): _locked_sha256(
            digest,
            label="source_contract_sha256",
        )
        for source_path, digest in raw_hashes.items()
    }
    return roles, hashes


def _opendss_lock_assets(row: Mapping[str, Any], *, row_index: int) -> dict[str, str]:
    ledger = row.get("case_ledger")
    physical_lock = (
        ledger.get("physical_source_lock") if isinstance(ledger, dict) else None
    )
    required = (
        physical_lock.get("required_source_assets")
        if isinstance(physical_lock, dict)
        else None
    )
    if not isinstance(required, list) or not required:
        raise ValueError(f"opendss_physical_source_lock_invalid:{row_index}")
    assets: dict[str, str] = {}
    for raw_asset in required:
        if not isinstance(raw_asset, dict):
            raise ValueError(f"opendss_physical_source_lock_invalid:{row_index}")
        path = _portable_path(
            raw_asset.get("declared_path"),
            label="opendss_physical_source_path",
        )
        digest = _locked_sha256(
            raw_asset.get("sha256"),
            label="opendss_physical_source_sha256",
        )
        if path in assets:
            raise ValueError(f"opendss_physical_source_lock_duplicate:{path}")
        assets[path] = digest
    return assets


def _opendss_closure_runtime_assets(
    closure: Mapping[str, Any],
) -> dict[str, str]:
    assets: dict[str, str] = {}

    def add(
        path: object,
        digest: object,
        roles: object,
        backend_kinds: object,
    ) -> None:
        if not (
            isinstance(roles, list)
            and "runtime_input" in roles
            and isinstance(backend_kinds, list)
            and _OPENDSS_BACKENDS.intersection(backend_kinds)
        ):
            return
        source_path = _portable_path(path, label="opendss_closure_source_path")
        source_digest = _locked_sha256(
            digest,
            label="opendss_closure_source_sha256",
        )
        if source_path in assets:
            raise ValueError(f"opendss_closure_source_duplicate:{source_path}")
        assets[source_path] = source_digest

    archived = closure.get("archived_files")
    tracked = closure.get("repo_tracked_files")
    separate = closure.get("separately_bundled_files")
    external = closure.get("external_sources")
    if not all(
        isinstance(value, dict)
        for value in (archived, tracked, separate, external)
    ):
        raise ValueError("opendss_closure_inventory_invalid")
    for row in archived.values():
        if isinstance(row, dict):
            add(
                row.get("source_path"),
                row.get("sha256"),
                row.get("roles"),
                row.get("backend_kinds"),
            )
    for inventory in (tracked, separate):
        for path, row in inventory.items():
            if isinstance(row, dict):
                add(
                    path,
                    row.get("sha256"),
                    row.get("roles"),
                    row.get("backend_kinds"),
                )
    for source in external.values():
        if not isinstance(source, dict):
            continue
        required_files = source.get("required_files")
        metadata = source.get("metadata")
        if not isinstance(required_files, dict) or not isinstance(metadata, dict):
            continue
        roles = metadata.get("roles")
        backend_kinds = metadata.get("backend_kinds")
        if not isinstance(roles, dict):
            continue
        for path, digest in required_files.items():
            add(path, digest, roles.get(path), backend_kinds)
    return assets


def validate_opendss_runtime_asset_closure(
    *,
    repo_root: Path,
    source_suite: Mapping[str, Any],
    closure: Mapping[str, Any] | None = None,
    require_live_contract: bool = True,
) -> dict[str, str]:
    """Bind each OpenDSS native include graph to YAML, lock, and closure."""

    rows = source_suite.get("scenarios")
    if not isinstance(rows, list):
        raise ValueError("source_suite_scenarios_missing")
    expected_closure_assets: dict[str, str] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("backend_kind") not in _OPENDSS_BACKENDS:
            continue
        backend_kind = str(row["backend_kind"])
        locked_assets = _opendss_lock_assets(row, row_index=index)
        scenario_id = str(row.get("scenario_id") or index)
        if require_live_contract:
            scenario_path = _portable_path(row.get("path"), label="scenario_path")
            yaml_path = repo_root / scenario_path
            try:
                yaml_path.resolve(strict=True).relative_to(repo_root.resolve())
            except (FileNotFoundError, ValueError) as exc:
                raise ValueError(
                    f"scenario_path_missing_or_outside_repo:{scenario_path}"
                ) from exc
            try:
                scenario = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, yaml.YAMLError) as exc:
                raise ValueError(f"scenario_yaml_invalid:{scenario_path}") from exc
            if not isinstance(scenario, dict):
                raise ValueError(f"scenario_yaml_invalid:{scenario_path}")
            from domains.registry import (  # noqa: PLC0415
                get_backend_capability,
                resolve_backend_source_contract_builder,
            )

            if scenario.get("backend_kind") != backend_kind:
                raise ValueError(f"opendss_backend_kind_mismatch:{scenario_id}")
            capability = get_backend_capability(backend_kind)
            builder = resolve_backend_source_contract_builder(capability)
            live_contract = builder(scenario, repo_root)
            if not isinstance(live_contract, dict):
                raise ValueError(f"opendss_runtime_contract_invalid:{scenario_id}")
            live_runtime = live_contract.get("runtime_input")
            source_contract = scenario.get("source_contract")
            yaml_runtime = (
                source_contract.get("runtime_input")
                if isinstance(source_contract, dict)
                else None
            )
            yaml_hashes = (
                source_contract.get("file_sha256s")
                if isinstance(source_contract, dict)
                else None
            )
            if not (
                isinstance(live_runtime, list)
                and live_runtime
                and isinstance(yaml_runtime, list)
                and yaml_runtime
                and isinstance(yaml_hashes, dict)
            ):
                raise ValueError(f"opendss_runtime_contract_invalid:{scenario_id}")
            live_paths = [
                _portable_path(path, label="opendss_live_runtime_input")
                for path in live_runtime
            ]
            yaml_paths = [
                _portable_path(path, label="opendss_yaml_runtime_input")
                for path in yaml_runtime
            ]
            if (
                len(live_paths) != len(set(live_paths))
                or len(yaml_paths) != len(set(yaml_paths))
                or set(live_paths) != set(yaml_paths)
                or set(live_paths) != set(locked_assets)
            ):
                raise ValueError(f"opendss_runtime_input_mismatch:{scenario_id}")
            for path in live_paths:
                _verified_repo_file(repo_root, path, locked_assets[path])
            declared_hashes = {
                _portable_path(path, label="opendss_yaml_hash_path"): _locked_sha256(
                    digest,
                    label="opendss_yaml_hash_sha256",
                )
                for path, digest in yaml_hashes.items()
            }
            if declared_hashes != locked_assets:
                raise ValueError(f"opendss_runtime_hash_mismatch:{scenario_id}")
        for path, digest in locked_assets.items():
            existing = expected_closure_assets.setdefault(path, digest)
            if existing != digest:
                raise ValueError(f"opendss_source_hash_conflict:{path}")
    if closure is not None:
        carried_assets = _opendss_closure_runtime_assets(closure)
        if carried_assets != expected_closure_assets:
            raise ValueError("opendss_backend_runtime_closure_mismatch")
    return dict(sorted(expected_closure_assets.items()))


def _collect_locked_assets(
    repo_root: Path,
    source_suite: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    rows = source_suite.get("scenarios")
    if not isinstance(rows, list) or not rows:
        raise ValueError("source_suite_scenarios_missing")
    if source_suite.get("n_scenarios") not in {None, len(rows)}:
        raise ValueError("source_suite_scenario_count_mismatch")
    assets: dict[str, dict[str, Any]] = {}
    backend_kinds: set[str] = set()
    verified_files: set[tuple[str, str]] = set()
    for index, raw_row in enumerate(rows):
        if not isinstance(raw_row, dict):
            raise ValueError(f"source_suite_row_invalid:{index}")
        backend_kind = raw_row.get("backend_kind")
        if not isinstance(backend_kind, str) or not backend_kind:
            raise ValueError(f"source_suite_backend_kind_invalid:{index}")
        backend_kinds.add(backend_kind)
        roles, contract_hashes = _scenario_contract(repo_root, raw_row)
        ledger = raw_row.get("case_ledger")
        physical_lock = (
            ledger.get("physical_source_lock") if isinstance(ledger, dict) else None
        )
        required = (
            physical_lock.get("required_source_assets")
            if isinstance(physical_lock, dict)
            else None
        )
        if not isinstance(required, list) or not required:
            raise ValueError(f"source_asset_lock_invalid:{index}")
        for raw_asset in required:
            if not isinstance(raw_asset, dict):
                raise ValueError(f"source_asset_lock_invalid:{index}")
            path = raw_asset.get("declared_path")
            digest = raw_asset.get("sha256")
            if not isinstance(path, str) or not path or not isinstance(digest, str):
                raise ValueError(f"source_asset_lock_invalid:{index}")
            digest = _locked_sha256(digest, label="source_asset_lock")
            if "://" in path:
                scheme = path.split("://", 1)[0]
                if scheme not in _VIRTUAL_SCHEMES:
                    raise ValueError(f"source_asset_virtual_scheme_unsupported:{path}")
            else:
                path = _portable_path(path, label="source_asset_path")
                if (path, digest) not in verified_files:
                    _verified_repo_file(repo_root, path, digest)
                    verified_files.add((path, digest))
                if path not in roles:
                    raise ValueError(
                        f"source_asset_missing_from_source_contract:{path}"
                    )
                contract_digest = contract_hashes.get(path)
                if contract_digest is not None and contract_digest != digest:
                    raise ValueError(f"source_asset_contract_hash_mismatch:{path}")
            current = assets.setdefault(
                path,
                {
                    "sha256": digest,
                    "roles": set(),
                    "backend_kinds": set(),
                },
            )
            if current["sha256"] != digest:
                raise ValueError(f"source_asset_lock_conflict:{path}")
            current["roles"].update(roles.get(path) or {"runtime_input"})
            current["backend_kinds"].add(backend_kind)
    return assets, backend_kinds


def _external_record(policy: Mapping[str, str]) -> dict[str, Any]:
    return {
        "delivery": policy["delivery"],
        "url": policy["url"],
        "revision": policy["revision"],
        "required_files": {},
        "metadata": {
            "backend_kinds": [],
            "license_status": policy["license_status"],
            "redistributed": False,
            "roles": {},
            "root": "",
        },
    }


def _merge_external_file(
    record: dict[str, Any],
    *,
    path: str,
    sha256: str,
    roles: set[str],
    backend_kinds: set[str],
) -> None:
    existing = record["required_files"].get(path)
    if existing is not None and existing != sha256:
        raise ValueError(f"external_source_hash_conflict:{path}")
    record["required_files"][path] = sha256
    record["metadata"]["roles"][path] = sorted(roles)
    record["metadata"]["backend_kinds"] = sorted(
        set(record["metadata"]["backend_kinds"]) | backend_kinds
    )


def _verify_checkout(repo_root: Path, *, root: str, revision: str) -> None:
    checkout = repo_root / root
    if not (checkout / ".git").exists():
        raise ValueError(f"external_checkout_missing:{root}")
    result = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or result.stdout.strip() != revision:
        raise ValueError(f"external_checkout_revision_mismatch:{root}")


def _load_supplemental_sources(
    repo_root: Path,
    runtime_source_lock_path: Path,
    *,
    release_id: str,
    backend_kinds: set[str],
    external_sources: dict[str, dict[str, Any]],
    verify_checkout_revisions: bool,
) -> None:
    lock, _ = _load_json_bytes(runtime_source_lock_path, label="runtime_source_lock")
    if (
        lock.get("schema_version") != "operate-backend-runtime-source-lock-v1"
        or lock.get("release_id") != release_id
        or not isinstance(lock.get("sources"), dict)
    ):
        raise ValueError("runtime_source_lock_contract_invalid")
    for source_id, raw in sorted(lock["sources"].items()):
        if not isinstance(source_id, str) or not isinstance(raw, dict):
            raise ValueError("runtime_source_lock_source_invalid")
        required_backends_raw = raw.get("backend_kinds")
        if required_backends_raw is None:
            required_backends = _SUPPLEMENTAL_SOURCE_BACKENDS.get(source_id)
        elif isinstance(required_backends_raw, list) and all(
            isinstance(value, str) and value for value in required_backends_raw
        ):
            required_backends = set(required_backends_raw)
        else:
            raise ValueError(f"runtime_source_lock_backend_kinds_invalid:{source_id}")
        if not required_backends:
            raise ValueError(f"runtime_source_lock_backend_kinds_missing:{source_id}")
        active_backends = required_backends & backend_kinds
        if not active_backends:
            continue
        delivery = raw.get("delivery")
        url = raw.get("url")
        revision = raw.get("revision")
        root = _portable_path(raw.get("root"), label="runtime_source_root")
        if (
            delivery not in {"git_checkout", "upstream_fetch", "user_provided"}
            or not isinstance(url, str)
            or not url
            or not isinstance(revision, str)
            or not revision
            or raw.get("redistributed") is not False
            or not isinstance(raw.get("license_status"), str)
            or not raw["license_status"]
        ):
            raise ValueError(f"runtime_source_lock_metadata_invalid:{source_id}")
        files = raw.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError(f"runtime_source_lock_files_missing:{source_id}")
        record = external_sources.setdefault(
            source_id,
            {
                "delivery": delivery,
                "url": url,
                "revision": revision,
                "required_files": {},
                "metadata": {
                    "backend_kinds": [],
                    "license_status": raw["license_status"],
                    "redistributed": False,
                    "roles": {},
                    "root": root,
                },
            },
        )
        if any(
            record[field] != raw[field] for field in ("delivery", "url", "revision")
        ):
            raise ValueError(f"runtime_source_lock_external_conflict:{source_id}")
        if record["metadata"]["root"] not in {"", root}:
            raise ValueError(f"runtime_source_lock_root_conflict:{source_id}")
        record["metadata"]["root"] = root
        for path, file_lock in sorted(files.items()):
            path = _portable_path(path, label="runtime_source_file")
            if not (path == root or path.startswith(f"{root}/")):
                raise ValueError(f"runtime_source_file_outside_root:{path}")
            if not isinstance(file_lock, dict):
                raise ValueError(f"runtime_source_file_lock_invalid:{path}")
            digest = _locked_sha256(
                file_lock.get("sha256"),
                label="runtime_source_file_sha256",
            )
            roles_raw = file_lock.get("roles")
            if (
                not isinstance(roles_raw, list)
                or not roles_raw
                or not all(isinstance(role, str) and role for role in roles_raw)
            ):
                raise ValueError(f"runtime_source_file_roles_invalid:{path}")
            _verified_repo_file(repo_root, path, digest)
            _merge_external_file(
                record,
                path=path,
                sha256=digest,
                roles=set(roles_raw),
                backend_kinds=active_backends,
            )
        if verify_checkout_revisions and delivery == "git_checkout":
            _verify_checkout(repo_root, root=root, revision=revision)


def _load_archive_licenses(
    repo_root: Path,
    runtime_source_lock_path: Path,
    *,
    release_id: str,
    active_roots: Mapping[str, set[str]],
    archived_files: dict[str, dict[str, Any]],
) -> None:
    lock, _ = _load_json_bytes(runtime_source_lock_path, label="runtime_source_lock")
    raw_licenses = lock.get("archive_licenses")
    if (
        lock.get("schema_version") != "operate-backend-runtime-source-lock-v1"
        or lock.get("release_id") != release_id
        or not isinstance(raw_licenses, dict)
    ):
        raise ValueError("runtime_source_lock_contract_invalid")
    missing = sorted(set(active_roots) - set(raw_licenses))
    if missing:
        raise ValueError(f"archive_redistribution_license_missing:{','.join(missing)}")
    for root, backend_kinds in sorted(active_roots.items()):
        raw = raw_licenses[root]
        base_fields = {"backend_kinds", "path", "sha256"}
        if not isinstance(raw, dict) or frozenset(raw) not in {
            frozenset(base_fields),
            frozenset({*base_fields, "additional_licenses"}),
        }:
            raise ValueError(f"archive_redistribution_license_invalid:{root}")
        path = _portable_path(raw.get("path"), label="archive_license_path")
        digest = _locked_sha256(
            raw.get("sha256"),
            label="archive_license_sha256",
        )
        locked_backend_kinds = raw.get("backend_kinds")
        if (
            root not in _ARCHIVE_ROOTS
            or not path.startswith(f"{root}/")
            or not isinstance(locked_backend_kinds, list)
            or not locked_backend_kinds
            or not all(
                isinstance(backend_kind, str) and backend_kind
                for backend_kind in locked_backend_kinds
            )
            or set(locked_backend_kinds) != backend_kinds
        ):
            raise ValueError(f"archive_redistribution_license_invalid:{root}")
        _verified_repo_file(repo_root, path, digest)
        archive_root_name = _ARCHIVE_ROOTS[root][1]
        relative = PurePosixPath(path).relative_to(PurePosixPath(root))
        archive_path = (
            PurePosixPath("backends") / archive_root_name / relative
        ).as_posix()
        licenses = [(path, digest, archive_path)]
        additional = raw.get("additional_licenses", [])
        if not isinstance(additional, list):
            raise ValueError(f"archive_redistribution_license_invalid:{root}")
        for extra in additional:
            if not isinstance(extra, dict) or set(extra) != {
                "archive_path",
                "path",
                "sha256",
            }:
                raise ValueError(f"archive_redistribution_license_invalid:{root}")
            extra_path = _portable_path(
                extra.get("path"),
                label="archive_license_path",
            )
            extra_digest = _locked_sha256(
                extra.get("sha256"),
                label="archive_license_sha256",
            )
            extra_archive_path = _portable_path(
                extra.get("archive_path"),
                label="archive_license_archive_path",
            )
            if not extra_archive_path.startswith(f"backends/{archive_root_name}/"):
                raise ValueError(f"archive_redistribution_license_invalid:{root}")
            _verified_repo_file(repo_root, extra_path, extra_digest)
            licenses.append((extra_path, extra_digest, extra_archive_path))
        for source_path, source_digest, target_path in licenses:
            if target_path in archived_files:
                raise ValueError(
                    f"archive_redistribution_license_collision:{target_path}"
                )
            archived_files[target_path] = {
                "source_path": source_path,
                "sha256": source_digest,
                "roles": ["redistribution_license"],
                "backend_kinds": sorted(backend_kinds),
            }


def _runtime_package_entry(raw: Mapping[str, Any]) -> dict[str, Any]:
    version = raw.get("version")
    source = raw.get("source")
    if (
        not isinstance(version, str)
        or not version
        or not isinstance(source, dict)
        or not source
    ):
        raise ValueError(f"uv_lock_package_metadata_invalid:{raw.get('name')}")
    entry: dict[str, Any] = {
        "version": version,
        "source": source,
    }
    if raw.get("resolution-markers") is not None:
        entry["resolution_markers"] = raw["resolution-markers"]
    artifacts = []
    if isinstance(raw.get("sdist"), dict):
        artifacts.append({"kind": "sdist", **raw["sdist"]})
    for wheel in raw.get("wheels") or []:
        if not isinstance(wheel, dict):
            raise ValueError(f"uv_lock_wheel_invalid:{raw.get('name')}")
        artifacts.append({"kind": "wheel", **wheel})
    for artifact in artifacts:
        artifact_hash = artifact.get("hash")
        if (
            not isinstance(artifact.get("url"), str)
            or not artifact["url"]
            or not isinstance(artifact_hash, str)
            or not artifact_hash.startswith("sha256:")
            or _SHA256_RE.fullmatch(artifact_hash.removeprefix("sha256:")) is None
        ):
            raise ValueError(f"uv_lock_package_artifact_invalid:{raw.get('name')}")
    entry["artifacts"] = sorted(
        artifacts,
        key=lambda item: (str(item.get("kind")), str(item.get("url"))),
    )
    entry["identity_sha256"] = _canonical_sha256(entry)
    return entry


def _runtime_packages(
    repo_root: Path,
    *,
    backend_kinds: set[str],
    virtual_sources: dict[str, str],
) -> dict[str, dict[str, Any]]:
    uv_lock_path = repo_root / "uv.lock"
    try:
        uv_lock_bytes = uv_lock_path.read_bytes()
        uv_lock = tomllib.loads(uv_lock_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError("uv_lock_invalid_or_missing") from exc
    raw_packages = uv_lock.get("package")
    if not isinstance(raw_packages, list):
        raise ValueError("uv_lock_packages_missing")
    by_name: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for package in raw_packages:
        if isinstance(package, dict) and isinstance(package.get("name"), str):
            by_name[package["name"]].append(package)
    uv_lock_sha256 = hashlib.sha256(uv_lock_bytes).hexdigest()
    result: dict[str, dict[str, Any]] = {}
    for name, supported_backends in sorted(_PACKAGE_BACKENDS.items()):
        active_backends = supported_backends & backend_kinds
        if not active_backends:
            continue
        raw_entries = by_name.get(name)
        if not raw_entries:
            raise ValueError(f"runtime_package_not_uv_locked:{name}")
        entries = sorted(
            (_runtime_package_entry(entry) for entry in raw_entries),
            key=lambda entry: _canonical_sha256(entry),
        )
        package = {
            "backend_kinds": sorted(active_backends),
            "lock_entries": entries,
            "lock_entries_sha256": _canonical_sha256(entries),
            "uv_lock_sha256": uv_lock_sha256,
        }
        if name == "simbench" and virtual_sources:
            package["virtual_sources"] = dict(sorted(virtual_sources.items()))
        result[name] = package
    return result


def build_backend_runtime_closure(
    *,
    repo_root: Path,
    source_suite_path: Path,
    runtime_source_lock_path: Path | None = None,
    verify_checkout_revisions: bool = True,
) -> dict[str, Any]:
    """Build a deterministic, fail-closed runtime closure for one frozen suite."""
    repo_root = repo_root.resolve()
    source_suite, source_suite_bytes = _load_json_bytes(
        source_suite_path,
        label="source_suite",
    )
    release_id = _release_id(source_suite, source_suite_path)
    assets, backend_kinds = _collect_locked_assets(repo_root, source_suite)
    archived_files: dict[str, dict[str, Any]] = {}
    external_sources: dict[str, dict[str, Any]] = {}
    backend_links: dict[str, str] = {}
    repo_tracked_files: dict[str, dict[str, Any]] = {}
    separately_bundled_files: dict[str, dict[str, Any]] = {}
    virtual_sources: dict[str, str] = {}
    active_archive_roots: dict[str, set[str]] = defaultdict(set)

    for path, asset in sorted(assets.items()):
        if "://" in path:
            virtual_sources[path] = asset["sha256"]
            continue
        archive_root = _root_match(path, _ARCHIVE_ROOTS)
        if archive_root is not None:
            works_name, archive_root_name = _ARCHIVE_ROOTS[archive_root]
            relative = PurePosixPath(path).relative_to(PurePosixPath(archive_root))
            archive_path = (
                PurePosixPath("backends") / archive_root_name / relative
            ).as_posix()
            archived_files[archive_path] = {
                "source_path": path,
                "sha256": asset["sha256"],
                "roles": sorted(asset["roles"]),
                "backend_kinds": sorted(asset["backend_kinds"]),
            }
            active_archive_roots[archive_root].update(asset["backend_kinds"])
            backend_links[works_name] = archive_root_name
            continue
        external_root = _root_match(path, _EXTERNAL_POLICIES)
        if external_root is not None:
            policy = _EXTERNAL_POLICIES[external_root]
            source_id = policy["source_id"]
            record = external_sources.setdefault(source_id, _external_record(policy))
            record["metadata"]["root"] = external_root
            _merge_external_file(
                record,
                path=path,
                sha256=asset["sha256"],
                roles=asset["roles"],
                backend_kinds=asset["backend_kinds"],
            )
            continue
        if _root_match(path, _REPO_TRACKED_ROOTS) is not None:
            repo_tracked_files[path] = {
                "sha256": asset["sha256"],
                "roles": sorted(asset["roles"]),
                "backend_kinds": sorted(asset["backend_kinds"]),
            }
            continue
        if _root_match(path, _SEPARATELY_BUNDLED_ROOTS) is not None:
            separately_bundled_files[path] = {
                "sha256": asset["sha256"],
                "roles": sorted(asset["roles"]),
                "backend_kinds": sorted(asset["backend_kinds"]),
            }
            continue
        raise ValueError(f"source_asset_delivery_policy_missing:{path}")

    if "dynasched_flexible_job_shop" in backend_kinds:
        # build_operate_bundle's dedicated source_assets contract writes both
        # layouts and materializes this archive root; do not duplicate bytes.
        backend_links["DynaSchedBench"] = "dynasched"

    source_lock_path = runtime_source_lock_path or (
        repo_root / "sources/locks/operate_v0_61_0/backend_runtime_sources.json"
    )
    _load_archive_licenses(
        repo_root,
        source_lock_path,
        release_id=release_id,
        active_roots=active_archive_roots,
        archived_files=archived_files,
    )

    for record in external_sources.values():
        root = record["metadata"]["root"]
        if verify_checkout_revisions and record["delivery"] == "git_checkout":
            _verify_checkout(repo_root, root=root, revision=record["revision"])

    _load_supplemental_sources(
        repo_root,
        source_lock_path,
        release_id=release_id,
        backend_kinds=backend_kinds,
        external_sources=external_sources,
        verify_checkout_revisions=verify_checkout_revisions,
    )
    runtime_packages = _runtime_packages(
        repo_root,
        backend_kinds=backend_kinds,
        virtual_sources=virtual_sources,
    )
    summary = {
        "n_archived_files": len(archived_files),
        "n_backend_links": len(backend_links),
        "n_external_sources": len(external_sources),
        "n_repo_tracked_files": len(repo_tracked_files),
        "n_runtime_packages": len(runtime_packages),
        "n_separately_bundled_files": len(separately_bundled_files),
        "n_source_assets": len(assets),
        "n_unresolved": 0,
        "n_virtual_sources": len(virtual_sources),
    }
    closure: dict[str, Any] = {
        "schema_version": "operate-backend-runtime-closure-v1",
        "release_id": release_id,
        "status": "backend_runtime_closure_complete",
        "terminal": True,
        "portable": True,
        "source_suite_sha256": hashlib.sha256(source_suite_bytes).hexdigest(),
        "archived_files": dict(sorted(archived_files.items())),
        "external_sources": dict(sorted(external_sources.items())),
        "backend_links": dict(sorted(backend_links.items())),
        "repo_tracked_files": dict(sorted(repo_tracked_files.items())),
        "runtime_packages": dict(sorted(runtime_packages.items())),
        "separately_bundled_files": dict(sorted(separately_bundled_files.items())),
        "summary": summary,
    }
    validate_opendss_runtime_asset_closure(
        repo_root=repo_root,
        source_suite=source_suite,
        closure=closure,
    )
    closure["identity_sha256"] = _canonical_sha256(closure)
    return closure


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-suite", type=Path, default=DEFAULT_SOURCE_SUITE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--runtime-source-lock",
        type=Path,
        default=DEFAULT_RUNTIME_SOURCE_LOCK,
    )
    args = parser.parse_args()
    closure = build_backend_runtime_closure(
        repo_root=REPO_ROOT,
        source_suite_path=args.source_suite,
        runtime_source_lock_path=args.runtime_source_lock,
    )
    _write_json_atomic(args.output, closure)
    print(json.dumps(closure["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
