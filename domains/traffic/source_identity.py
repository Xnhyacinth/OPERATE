"""Canonical source identity for native SUMO scenarios."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TRANSPORT = "traci_tcp"
_VERSION_RE = re.compile(r"(?i)(?:sumo\s+)?(\d+\.\d+\.\d+)")


class SumoSourceIdentityError(ValueError):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


@dataclass(frozen=True)
class SourceAsset:
    role: str
    path: str
    sha256: str
    order: int
    parent_asset: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "path": self.path,
            "sha256": self.sha256,
            "order": self.order,
            "parent_asset": self.parent_asset,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_sumo_version(raw: Any) -> str:
    """Normalize binary/TraCI representations to one semantic version."""
    if isinstance(raw, (tuple, list)) and len(raw) >= 2:
        raw = raw[1]
    match = _VERSION_RE.search(str(raw or ""))
    if match is None:
        raise SumoSourceIdentityError(
            "traffic_source_identity_version_unparseable",
            f"cannot parse SUMO version from {raw!r}",
        )
    return f"Eclipse SUMO sumo {match.group(1)}"


def build_sumo_source_identity_payload(
    graph: Mapping[str, Any],
    *,
    service_date: str,
    sumo_version: Any,
    transport: str,
) -> dict[str, Any]:
    try:
        payload = {
            "sumocfg_sha256": str(graph["sumocfg"]["sha256"]),
            "network_sha256": str(graph["network"]["sha256"]),
            "ordered_route_file_sha256s": [
                str(row["sha256"]) for row in graph["route_files"]
            ],
            "ordered_additional_file_sha256s": [
                str(row["sha256"]) for row in graph["additional_files"]
            ],
            "ordered_recursive_include_sha256s": [
                str(row["sha256"]) for row in graph["recursive_inputs"]
            ],
            "service_date": str(service_date),
            "sumo_version": normalize_sumo_version(sumo_version),
            "transport": str(transport),
        }
    except (KeyError, TypeError) as exc:
        raise SumoSourceIdentityError(
            "traffic_source_identity_payload_invalid",
            "SUMO source graph is incomplete",
        ) from exc
    if not payload["service_date"] or not payload["transport"]:
        raise SumoSourceIdentityError(
            "traffic_source_identity_payload_invalid",
            "service date and transport are required",
        )
    return payload


def compute_sumo_source_identity(payload: Mapping[str, Any]) -> str:
    required = {
        "sumocfg_sha256",
        "network_sha256",
        "ordered_route_file_sha256s",
        "ordered_additional_file_sha256s",
        "ordered_recursive_include_sha256s",
        "service_date",
        "sumo_version",
        "transport",
    }
    if set(payload) != required:
        raise SumoSourceIdentityError(
            "traffic_source_identity_payload_invalid",
            "source identity payload fields do not match the contract",
            fields=sorted(payload),
        )
    canonical = dict(payload)
    canonical["sumo_version"] = normalize_sumo_version(
        canonical["sumo_version"]
    )
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _open_xml(path: Path) -> ET.Element:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        return ET.parse(handle).getroot()


def _split_values(value: str | None) -> list[str]:
    return [
        item.strip()
        for item in str(value or "").split(",")
        if item.strip()
    ]


def resolve_sumo_input_graph(sumocfg: Path) -> dict[str, Any]:
    """Resolve the ordered input graph actually named by a SUMO config."""
    cfg = sumocfg.resolve()
    if not cfg.is_file():
        raise SumoSourceIdentityError(
            "traffic_source_identity_payload_invalid",
            f"SUMO config is missing: {cfg}",
        )
    root = _open_xml(cfg)
    input_node = root.find("input")
    if input_node is None:
        raise SumoSourceIdentityError(
            "traffic_source_identity_payload_invalid",
            "SUMO config has no input section",
        )
    assets: dict[str, list[SourceAsset]] = {
        "route_files": [],
        "additional_files": [],
        "recursive_inputs": [],
    }
    network: SourceAsset | None = None
    visited: set[Path] = set()

    def walk(path: Path, parent: Path) -> None:
        resolved = path.resolve()
        if resolved in visited:
            return
        visited.add(resolved)
        if not resolved.is_file():
            raise SumoSourceIdentityError(
                "traffic_source_identity_payload_invalid",
                f"recursive runtime input is missing: {resolved}",
                parent_asset=str(parent),
            )
        xml = _open_xml(resolved)
        for include in xml.iter("include"):
            raw = include.get("href") or include.get("file")
            if not raw:
                continue
            child = (resolved.parent / raw).resolve()
            if not child.is_file():
                raise SumoSourceIdentityError(
                    "traffic_source_identity_payload_invalid",
                    f"recursive runtime input is missing: {child}",
                    parent_asset=str(resolved),
                )
            assets["recursive_inputs"].append(
                SourceAsset(
                    role="recursive_include",
                    path=str(child),
                    sha256=sha256_file(child),
                    order=len(assets["recursive_inputs"]),
                    parent_asset=str(resolved),
                )
            )
            walk(child, resolved)

    role_by_tag = {
        "net-file": "network",
        "route-files": "route_files",
        "additional-files": "additional_files",
    }
    for child in input_node:
        role = role_by_tag.get(child.tag)
        if role is None:
            continue
        for order, raw in enumerate(_split_values(child.get("value"))):
            path = (cfg.parent / raw).resolve()
            if not path.is_file():
                raise SumoSourceIdentityError(
                    "traffic_source_identity_payload_invalid",
                    f"runtime input is missing: {path}",
                    role=role,
                )
            asset = SourceAsset(
                role=role,
                path=str(path),
                sha256=sha256_file(path),
                order=order,
                parent_asset=str(cfg),
            )
            if role == "network":
                network = asset
            else:
                assets[role].append(asset)
            walk(path, cfg)
    if network is None:
        raise SumoSourceIdentityError(
            "traffic_source_identity_payload_invalid",
            "SUMO config has no network input",
        )
    return {
        "sumocfg": SourceAsset(
            role="sumocfg",
            path=str(cfg),
            sha256=sha256_file(cfg),
            order=0,
        ).to_dict(),
        "network": network.to_dict(),
        **{
            key: [asset.to_dict() for asset in value]
            for key, value in assets.items()
        },
    }


def source_identity_from_graph(
    graph: Mapping[str, Any],
    *,
    service_date: str,
    sumo_version: Any,
    transport: str = TRANSPORT,
) -> tuple[dict[str, Any], str]:
    payload = build_sumo_source_identity_payload(
        graph,
        service_date=service_date,
        sumo_version=sumo_version,
        transport=transport,
    )
    return payload, compute_sumo_source_identity(payload)
