"""Role-aware source asset resolution for protocol-2.1.

Only runtime and derivation inputs participate in source-consumption gates.
Implementation, metadata, and licence files remain provenance-locked without
being incorrectly treated as simulator inputs.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from audit.provenance import _provenance_candidates

SourceAssetRole = Literal[
    "runtime_input",
    "derivation_input",
    "implementation_asset",
    "metadata",
    "license",
]

SOURCE_ASSET_ROLES: tuple[SourceAssetRole, ...] = (
    "runtime_input",
    "derivation_input",
    "implementation_asset",
    "metadata",
    "license",
)

# A constructor-backed public network is not a file, but it can still have a
# stable source identity when the backend proves the exact constructor,
# package version, and resulting native state were consumed. Keep this list
# explicit so arbitrary URLs or scenario-declared strings never become source
# evidence merely because they contain a URI scheme.
VIRTUAL_SOURCE_PREFIXES: tuple[str, ...] = (
    "pandapower-cigre-mv://",
    "pandapower-mv-oberrhein://",
    "pandapower-simbench://",
)


_VIRTUAL_SOURCE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "pandapower-cigre-mv",
        re.compile(
            r"^pandapower-cigre-mv://create_cigre_network_mv\(with_der='all'\)@"
            r"(?P<version>unknown|\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.-]+)?)$"
        ),
    ),
    (
        "pandapower-mv-oberrhein",
        re.compile(
            r"^pandapower-mv-oberrhein://mv_oberrhein\(\)@"
            r"(?P<version>unknown|\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.-]+)?)$"
        ),
    ),
    (
        "pandapower-simbench",
        re.compile(
            r"^pandapower-simbench://(?P<dataset>[A-Za-z0-9_.-]+)@"
            r"(?P<version>unknown|\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.-]+)?)$"
        ),
    ),
)


def virtual_source_reference_info(raw: str) -> dict[str, str] | None:
    """Parse a whitelisted constructor identity, rejecting arbitrary URIs."""
    value = str(raw).strip()
    for scheme, pattern in _VIRTUAL_SOURCE_PATTERNS:
        match = pattern.fullmatch(value)
        if match is None:
            continue
        info = {"scheme": scheme, **match.groupdict()}
        if scheme == "pandapower-cigre-mv":
            info["network"] = "cigre_mv_with_der_all"
        elif scheme == "pandapower-mv-oberrhein":
            info["network"] = "mv_oberrhein"
        elif scheme == "pandapower-simbench":
            info["network"] = f"simbench:{info['dataset']}"
        return {key: value for key, value in info.items() if value is not None}
    return None


def is_virtual_source_reference(raw: str) -> bool:
    return virtual_source_reference_info(raw) is not None


def virtual_source_identity_sha256(raw: str) -> str | None:
    if not is_virtual_source_reference(raw):
        return None
    return hashlib.sha256(
        f"protocol21-virtual-source:{raw}".encode()
    ).hexdigest()


def _normalized_physical_asset_value(value: Any, *, field: str = "") -> Any:
    if isinstance(value, dict):
        normalized = {
            str(key): _normalized_physical_asset_value(item, field=str(key))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
        assets = normalized.get("required_source_assets")
        if isinstance(assets, list):
            normalized["required_source_assets"] = sorted(
                assets,
                key=lambda item: json.dumps(
                    item, sort_keys=True, separators=(",", ":")
                ),
            )
        return normalized
    if isinstance(value, list):
        return [_normalized_physical_asset_value(item, field=field) for item in value]
    if isinstance(value, str) and ("file" in field.lower() or "path" in field.lower()):
        return Path(value).name
    return value


def canonical_physical_source_asset_key(lock: Any) -> str:
    """Identify the underlying source graph without counting derived windows.

    A derived window remains part of row lineage and admission identity, but two
    windows over the same locked assets are not two physical sources.
    """
    if not isinstance(lock, (dict, list)):
        return str(lock)
    if isinstance(lock, dict):
        lock = {key: value for key, value in lock.items() if key != "derived_window"}
    normalized = _normalized_physical_asset_value(lock)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


@dataclass(frozen=True)
class SourceAssetContract:
    required_runtime_source_files: tuple[str, ...]
    required_derivation_source_files: tuple[str, ...]
    implementation_assets: tuple[str, ...]
    metadata_files: tuple[str, ...]
    license_files: tuple[str, ...]
    locked_source_hashes: dict[str, str]
    resolved_source_paths: dict[str, str]
    missing_required_files: tuple[str, ...]
    derived_window_sha256: str | None
    recipe_version: str | None
    contract_errors: tuple[str, ...]


def physical_source_lock_from_contract(
    contract: SourceAssetContract,
    *,
    backend_kind: str,
) -> dict[str, Any] | None:
    """Build an exact physical-input identity from a verified source graph."""
    required = (
        *contract.required_runtime_source_files,
        *contract.required_derivation_source_files,
    )
    if (
        not required
        or contract.contract_errors
        or contract.missing_required_files
        or any(path not in contract.locked_source_hashes for path in required)
    ):
        return None
    lock: dict[str, Any] = {
        "schema_version": "source_asset_graph_v1",
        "backend_kind": backend_kind,
        "required_source_assets": [
            {
                "declared_path": path,
                "sha256": contract.locked_source_hashes[path],
            }
            for path in dict.fromkeys(required)
        ],
    }
    if contract.derived_window_sha256 and contract.recipe_version:
        lock["derived_window"] = {
            "sha256": contract.derived_window_sha256,
            "recipe_version": contract.recipe_version,
        }
    return lock


def _role_paths(raw_contract: dict[str, Any], role: SourceAssetRole) -> tuple[str, ...]:
    value = raw_contract.get(role) or []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value if str(item).strip())


def _resolve_path(raw: str, *, repo_root: Path) -> Path | None:
    if "://" in raw:
        return None
    return next(
        (
            candidate.resolve()
            for candidate in _provenance_candidates(raw, repo_root=repo_root)
            if candidate.is_file()
        ),
        None,
    )


def resolve_source_asset_contract(
    scenario: dict[str, Any],
    *,
    repo_root: Path,
) -> SourceAssetContract:
    raw = scenario.get("source_contract")
    errors: list[str] = []
    if not isinstance(raw, dict):
        raw = {}
        errors.append("source_contract_missing")

    by_role = {role: _role_paths(raw, role) for role in SOURCE_ASSET_ROLES}
    expected_hashes_raw = raw.get("file_sha256s")
    expected_hashes: dict[str, str] = {}
    if expected_hashes_raw is not None:
        if not isinstance(expected_hashes_raw, dict):
            errors.append("source_file_sha256s_invalid")
        else:
            expected_hashes = {
                str(path): str(digest).strip().lower()
                for path, digest in expected_hashes_raw.items()
            }
            required_paths = set(
                by_role["runtime_input"] + by_role["derivation_input"]
            )
            if set(expected_hashes) != required_paths:
                errors.append("source_file_sha256_binding_incomplete")
            for digest in expected_hashes.values():
                if len(digest) != 64 or any(
                    char not in "0123456789abcdef" for char in digest
                ):
                    errors.append("source_file_sha256_invalid")
    derived = raw.get("derived_window") or {}
    if not isinstance(derived, dict):
        derived = {}
        errors.append("derived_window_invalid")
    digest = str(derived.get("sha256") or "").strip() or None
    recipe = str(derived.get("recipe_version") or "").strip() or None
    if bool(digest) != bool(recipe):
        errors.append("derived_window_digest_recipe_incomplete")
    if digest is not None and (
        len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest.lower())
    ):
        errors.append("derived_window_sha256_invalid")

    locked: dict[str, str] = {}
    resolved: dict[str, str] = {}
    missing: list[str] = []
    required = (
        *by_role["runtime_input"],
        *by_role["derivation_input"],
    )
    for raw_path in required:
        virtual_hash = virtual_source_identity_sha256(raw_path)
        if virtual_hash is not None:
            locked[raw_path] = virtual_hash
            resolved[raw_path] = raw_path
            expected_hash = expected_hashes.get(raw_path)
            if expected_hash is not None and expected_hash != virtual_hash:
                errors.append("source_file_sha256_mismatch")
            continue
        path = _resolve_path(raw_path, repo_root=repo_root)
        if path is None:
            missing.append(raw_path)
            continue
        resolved[raw_path] = str(path)
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        locked[raw_path] = actual_hash
        expected_hash = expected_hashes.get(raw_path)
        if expected_hash is not None and expected_hash != actual_hash:
            errors.append("source_file_sha256_mismatch")

    if not by_role["runtime_input"] and not by_role["derivation_input"]:
        errors.append("source_contract_has_no_required_input")

    return SourceAssetContract(
        required_runtime_source_files=by_role["runtime_input"],
        required_derivation_source_files=by_role["derivation_input"],
        implementation_assets=by_role["implementation_asset"],
        metadata_files=by_role["metadata"],
        license_files=by_role["license"],
        locked_source_hashes=locked,
        resolved_source_paths=resolved,
        missing_required_files=tuple(missing),
        derived_window_sha256=digest,
        recipe_version=recipe,
        contract_errors=tuple(sorted(set(errors))),
    )
