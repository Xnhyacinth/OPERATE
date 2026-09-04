#!/usr/bin/env python3
"""Preflight SUMO-RL / SUMO traffic sources before any release work.

This report is deliberately read-only and non-release. It checks local runtime
availability, the optional ``works/SUMO`` source clone, and the Traffic seed
anchor files already named by ``domains.traffic``. It does not install Python
packages, download SUMO scenarios, launch SUMO, write scenario YAMLs, or modify
release artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import Action, ToolCall  # noqa: E402
from core.sidecar.sumo_sidecar import probe_sumo_transport, sumo_available  # noqa: E402
from domains.traffic.adapter import TrafficEnvironment  # noqa: E402
from domains.traffic.seeds.from_lust import (  # noqa: E402
    _NET_FOR_FAMILY,
    _load_ingolstadt_binding,
    build_traffic_seed,
)
from evaluation.scorer import ScoringInputs, score_episode  # noqa: E402

REPORT_SCOPE = "sumo_rl_traffic_source_preflight"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "sumo_rl_traffic_source_preflight.json"
DEFAULT_LIVE_PROBE_OUTPUT = REPO_ROOT / "reports" / "sumo_live_adapter_probe.json"
DEFAULT_LIVE_HEADROOM_OUTPUT = REPO_ROOT / "reports" / "sumo_live_headroom_probe.json"
LIVE_ADAPTER_GENERATE_COMMAND = (
    "OPERATE_TRAFFIC_BACKEND_REAL=1 .venv/bin/python "
    "scripts/audit_sumo_rl_traffic_sources.py --run-live-probe"
)
# Live decision-headroom thresholds. The claim is *causal action-dependence* of
# the realized physical outcome, not the mock's engineered "relief always helps"
# semantics (which the binding caveat explicitly disclaims for the live net): a
# benchmark signal-plan action must measurably move the realized per-corridor
# delay vector (L1, in delay-minutes) across more than one corridor.
MIN_HEADROOM_L1_MINUTES = 1.0
MIN_HEADROOM_CORRIDORS_CHANGED = 2
LIVE_HEADROOM_GENERATE_COMMAND = (
    "OPERATE_TRAFFIC_BACKEND_REAL=1 .venv/bin/python "
    "scripts/audit_sumo_rl_traffic_sources.py --run-live-headroom-probe"
)
DEFAULT_BEHAVIORAL_GATE_OUTPUT = REPO_ROOT / "reports" / "traffic_behavioral_gate.json"
BEHAVIORAL_GATE_GENERATE_COMMAND = (
    ".venv/bin/python scripts/traffic_behavioral_gate.py "
    "--output reports/traffic_behavioral_gate.json --require-gate-passed"
)
DEFAULT_MATERIALIZER_DRAFT_OUTPUT = (
    REPO_ROOT
    / "reports"
    / "traffic_release_materializer_draft"
    / "materializer_draft.json"
)
MATERIALIZER_DRAFT_GENERATE_COMMAND = (
    ".venv/bin/python scripts/traffic_release_materializer_draft.py --require-ready"
)
DEFAULT_WRITTEN_DRAFT_BEHAVIORAL_AUDIT_OUTPUT = (
    REPO_ROOT / "reports" / "traffic_written_draft_behavioral_audit.json"
)
WRITTEN_DRAFT_BEHAVIORAL_AUDIT_GENERATE_COMMAND = (
    ".venv/bin/python scripts/traffic_written_draft_behavioral_audit.py --require-ready"
)
# Foundational (runtime / source-lock / live-execution / evidence) blockers vs
# downstream release-packaging blockers (behavioral gate + release materializer).
# The top-level status is only "missing sumo runtime / source locks" when a
# foundational rung is actually open; once the live backend has executed and the
# source is locked, remaining gates are honestly reported as packaging gates.
FOUNDATIONAL_BLOCKER_CODES = frozenset(
    {
        "sumo_runtime_not_available",
        "sumo_source_tree_missing",
        "sumo_runtime_license_not_verified",
        "traffic_seed_anchor_files_missing",
        "traffic_source_locks_incomplete",
        "selected_source_file_lock_incomplete",
        "native_tool_mapping_unavailable",
        "missing_live_backend_execution_probe",
        "missing_evidence_wiring_probe",
        "missing_score_consumption_probe",
    }
)
DEFAULT_SUMO_SOURCE_ROOT = REPO_ROOT / "works" / "SUMO"
SAFE_COMMANDS_NOW = [
    (
        ".venv/bin/python scripts/audit_sumo_rl_traffic_sources.py "
        "--output reports/sumo_rl_traffic_source_preflight.json"
    ),
    (
        ".venv/bin/python scripts/data_expansion_readiness.py "
        "--output reports/data_expansion_readiness.json"
    ),
]
TRAFFIC_REQUIRED_MODULES = ("sumo_rl", "traci", "libsumo", "sumolib")
SUMO_BINARIES = ("sumo", "sumo-gui", "netconvert", "duarouter")
SELECTED_SOURCE_CANDIDATE_ID = "sumo_ingolstadt"
SOURCE_DELIVERY_REQUIRED_BEFORE_LIVE_ADAPTER_PROBE = [
    "sumo_sidecar_runtime_version_locked",
    "selected_source_file_lock_closed",
    "all_family_anchor_files_present",
    "all_family_source_locks_complete",
    "network_route_and_config_sha256_recorded",
]
SOURCE_DELIVERY_VALIDATION_COMMANDS = [
    (
        ".venv/bin/python scripts/audit_sumo_rl_traffic_sources.py "
        "--output reports/sumo_rl_traffic_source_preflight.json"
    ),
    ".venv/bin/python -m pytest tests/test_v0_7_traffic.py -q",
]
SELECTED_SOURCE_LOCK_CLOSURE_REQUIRED_FIELDS = [
    "source_url",
    "license",
    "git_commit_or_release_tag",
    "lock_strategy",
    "sumo_runtime_version",
    "network_file_sha256",
    "route_or_demand_trace_sha256",
    "simulation_config_hash",
]
SELECTED_SOURCE_READY_WHEN = [
    "source_root_present",
    "source_git_commit_recorded",
    "source_license_verified",
    "required_files_present",
    "required_file_sha256_recorded",
    "sumo_sidecar_runtime_version_locked",
]
SELECTED_SOURCE_DECISION_PRESSURE_AXES = [
    "hidden_incident_replanning",
    "emergency_priority_vs_general_delay_tradeoff",
    "vip_vs_ems_ethical_conflict",
]
LIVE_ADAPTER_REQUIRED_BEFORE_RELEASE = [
    "OPERATE_TRAFFIC_BACKEND_REAL=1 explicit env gate",
    "sumo_runtime_available",
    "selected_source_file_lock_closed",
    "native_tool_mapping_available",
    "backend_reset_starts_sumo_sidecar",
    "backend_tick_advances_live_sumo",
    "native_tool_call_mutates_sumo_state",
    "evidence_rows_link_tool_to_state_change",
    "score_dimensions_consume_live_tool_evidence",
]
LIVE_NATIVE_TOOL_MAPPING_AVAILABLE = True
LIVE_NATIVE_TOOL_MAPPING_PROOFS = [
    (
        "SumoSidecar.set_traffic_light_program applies SUMO TLS program and "
        "reads it back through the public sidecar API"
    ),
    "SumoBackend.change_signal_plan maps corridor -> SUMO TLS program",
    "TrafficEnvironment executes change_signal_plan through core.ToolRegistry",
    (
        "tests/test_v0_7_traffic.py covers state mutation, SUMO program "
        "readback, observation delta, and control evidence"
    ),
]
SOURCE_DELIVERY_SPECS: tuple[dict[str, Any], ...] = (
    {
        "source_id": "lust",
        "root": "works/LuSTScenario",
        "url": "https://github.com/lcodeca/LuSTScenario",
        "license": "MIT (repo); OSM geometry ODbL provenance caveat",
        "license_posture": "usable_with_odbl_provenance_caveat",
        "families": ["daily_peak_commute"],
        "required_files": [
            "works/LuSTScenario/scenario/lust.net.xml",
            "works/LuSTScenario/scenario/DUARoutes/local.0.rou.xml",
        ],
    },
    {
        "source_id": "sumo_ingolstadt",
        "root": "works/sumo_ingolstadt",
        "url": "https://github.com/TUM-VT/sumo_ingolstadt",
        "license": "Apache-2.0",
        "license_posture": "permissive_source_lock_required",
        "families": ["incident_response", "vip_priority_dilemma"],
        "required_files": [
            "works/sumo_ingolstadt/simulation/24h_sim.sumocfg",
            "works/sumo_ingolstadt/simulation/ingolstadt_24h.net.xml.gz",
            (
                "works/sumo_ingolstadt/simulation/"
                "motorized_routes_2020-09-16_24h.rou.xml.gz"
            ),
            "works/sumo_ingolstadt/simulation/tlLogics_WAUT_2020-09-16_24h.add.xml",
        ],
    },
    {
        "source_id": "tapas_cologne",
        "root": "works/TAPASCologne",
        "url": "https://sumo.dlr.de/docs/Data/Scenarios.html",
        "license": "DLR research data redistribution terms need verification",
        "license_posture": "research_data_redistribution_terms_need_verification",
        "families": ["weather_capacity_drop"],
        "required_files": [
            "works/TAPASCologne/cologne.net.xml",
            "works/TAPASCologne/cologne.rou.xml",
        ],
    },
    {
        "source_id": "osm_slice",
        "root": "works/osm_region",
        "url": "https://sumo.dlr.de/docs/Tutorials/OSMWebWizard.html",
        "license": "ODbL-1.0",
        "license_posture": "share_alike_attribution_required",
        "requires_attribution": True,
        "requires_share_alike_notice": True,
        "families": ["event_egress"],
        "required_files": [
            "works/osm_region/osm.net.xml",
            "works/osm_region/osm.rou.xml",
        ],
    },
)
# The released traffic carrier is intentionally narrowed to the source-locked
# `sumo_ingolstadt` families. The other source specs are dev-only frontier tracks
# (genuinely un-source-locked, anchor data not vendored) and are machine-readably
# excluded from the RELEASED denominator — never silently dropped — exactly like
# the dev-only microgrid/disaster domains contribute 0 released scenarios. Their
# unlocked status stays fully visible as a diagnostic; it is just not counted as a
# release blocker for a carrier that does not include them.
RELEASED_SCOPE_SOURCE_ID = SELECTED_SOURCE_CANDIDATE_ID
RELEASED_SCOPE_FAMILIES = tuple(
    sorted(
        family
        for spec in SOURCE_DELIVERY_SPECS
        if spec["source_id"] == RELEASED_SCOPE_SOURCE_ID
        for family in spec.get("families", ())
    )
)
FRONTIER_FAMILY_SPECS = tuple(
    {
        "family": family,
        "source_id": spec["source_id"],
        "license_posture": spec.get("license_posture"),
        "exclusion_reason": "dev_only_frontier_source_not_yet_source_locked",
    }
    for spec in SOURCE_DELIVERY_SPECS
    if spec["source_id"] != RELEASED_SCOPE_SOURCE_ID
    for family in spec.get("families", ())
)
FORBIDDEN_SOURCE_SPECS = [
    {
        "source_id": "intas_legacy",
        "url": "https://github.com/silaslobo/InTAS",
        "license": "GPL-3.0",
        "reason": (
            "GPL traffic net is anchor/offline-bake only and must not enter a "
            "released artifact source path."
        ),
    }
]


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _path_fields(path: Path) -> dict[str, str]:
    try:
        rel = path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        return {
            "path": rel,
            "path_base": "repo_root",
            "absolute_path_at_build": str(path),
        }
    except ValueError:
        return {
            "path": str(path),
            "path_base": "absolute",
            "absolute_path_at_build": str(path),
        }


def _file_fingerprint(path: Path) -> dict[str, Any]:
    digest = _sha256(path)
    exists = path.exists()
    return {
        **_path_fields(path),
        "exists": exists,
        "sha256": digest,
        "matches_current_file": (
            _sha256(path) == digest if exists and path.is_file() else not exists
        ),
    }


def _git_output(args: list[str], cwd: Path) -> str | None:
    if not cwd.exists():
        return None
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=str(cwd),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _git_status_short(cwd: Path) -> list[str]:
    output = _git_output(["status", "--short", "--ignored=no"], cwd)
    if not output:
        return []
    return [line for line in output.splitlines() if line.strip()]


def _module_report(module: str) -> dict[str, Any]:
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ModuleNotFoundError, ValueError):
        spec = None
    return {
        "module": module,
        "importable": spec is not None,
        "origin": getattr(spec, "origin", None) if spec is not None else None,
    }


def _distribution_report(package: str) -> dict[str, Any]:
    try:
        metadata = importlib.metadata.metadata(package)
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return {"package": package, "installed": False}
    return {
        "package": package,
        "installed": True,
        "version": version,
        "name": metadata.get("Name"),
        "license": metadata.get("License") or metadata.get("Classifier"),
        "summary": metadata.get("Summary"),
        "home_page": metadata.get("Home-page") or metadata.get("Project-URL"),
    }


_SUMO_VERSION_RE = re.compile(r"\bVersion\s+([0-9]+(?:\.[0-9A-Za-z_-]+)+)")


def _sumo_runtime_version_report(sumo_binary: str | None) -> dict[str, Any]:
    if not sumo_binary:
        return {
            "sumo_runtime_version_verified": False,
            "sumo_runtime_version": None,
            "sumo_runtime_version_source": None,
            "sumo_runtime_version_raw": None,
        }
    try:
        raw = subprocess.check_output(
            [sumo_binary, "--version"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
    except Exception as exc:
        return {
            "sumo_runtime_version_verified": False,
            "sumo_runtime_version": None,
            "sumo_runtime_version_source": "sumo --version",
            "sumo_runtime_version_raw": None,
            "sumo_runtime_version_error": type(exc).__name__,
        }
    first_line = raw.splitlines()[0] if raw.splitlines() else raw.strip()
    match = _SUMO_VERSION_RE.search(raw)
    version = match.group(1) if match else None
    return {
        "sumo_runtime_version_verified": version is not None,
        "sumo_runtime_version": version,
        "sumo_runtime_version_source": "sumo --version",
        "sumo_runtime_version_raw": first_line,
    }


def _python_sumo_runtime_version_report(
    selected_transport: str | None,
) -> dict[str, Any]:
    if selected_transport != "libsumo":
        return {
            "sumo_runtime_version_verified": False,
            "sumo_runtime_version": None,
            "sumo_runtime_version_source": None,
            "sumo_runtime_version_raw": None,
        }
    for package in ("libsumo", "sumolib"):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
        return {
            "sumo_runtime_version_verified": True,
            "sumo_runtime_version": version,
            "sumo_runtime_version_source": f"python-package:{package}",
            "sumo_runtime_version_raw": version,
        }
    return {
        "sumo_runtime_version_verified": False,
        "sumo_runtime_version": None,
        "sumo_runtime_version_source": "python-package:libsumo|sumolib",
        "sumo_runtime_version_raw": None,
        "sumo_runtime_version_error": "PackageNotFoundError",
    }


def _runtime_preflight() -> dict[str, Any]:
    modules = {module: _module_report(module) for module in TRAFFIC_REQUIRED_MODULES}
    binaries = {
        binary: {"binary": binary, "path": shutil.which(binary)}
        for binary in SUMO_BINARIES
    }
    selected_transport = probe_sumo_transport()
    sumo_rl_distribution = _distribution_report("sumo-rl")
    runtime_version = _sumo_runtime_version_report(binaries["sumo"]["path"])
    if runtime_version.get("sumo_runtime_version_verified") is not True:
        runtime_version = _python_sumo_runtime_version_report(selected_transport)
    return {
        "selected_transport": selected_transport,
        "sumo_transport_available": selected_transport is not None,
        "modules": modules,
        "packages": {"sumo-rl": sumo_rl_distribution},
        "binaries": binaries,
        "environment": {
            "SUMO_HOME": os.environ.get("SUMO_HOME"),
            "OPERATE_TRAFFIC_FORCE_TRANSPORT": os.environ.get("OPERATE_TRAFFIC_FORCE_TRANSPORT"),
            "OPERATE_TRAFFIC_ALLOW_DOCKER": os.environ.get("OPERATE_TRAFFIC_ALLOW_DOCKER"),
            "OPERATE_TRAFFIC_BACKEND_REAL": os.environ.get("OPERATE_TRAFFIC_BACKEND_REAL"),
        },
        "runtime_lock_status": {
            **runtime_version,
            "sumo_rl_package_version_verified": (
                sumo_rl_distribution.get("installed") is True
            ),
            "transport_selection_is_pinned": bool(
                os.environ.get("OPERATE_TRAFFIC_FORCE_TRANSPORT")
            ),
        },
    }


def _detect_sumo_license(license_path: Path) -> str | None:
    if not license_path.exists():
        return None
    text = license_path.read_text(encoding="utf-8", errors="ignore")
    if "Apache License" in text and "Version 2.0" in text:
        return "Apache-2.0"
    if "Eclipse Public License" in text and "2.0" in text:
        return "EPL-2.0"
    return None


def _first_existing_license(root: Path) -> Path:
    for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING"):
        path = root / name
        if path.exists():
            return path
    return root / "LICENSE"


def _sumo_source_tree(source_root: Path) -> dict[str, Any]:
    license_path = _first_existing_license(source_root)
    readme_path = source_root / "README.md"
    source_root_fields = _path_fields(source_root)
    return {
        "path": source_root_fields["path"],
        "path_base": source_root_fields["path_base"],
        "absolute_path_at_build": str(source_root),
        "exists": source_root.exists(),
        "git": {
            "remote_url": _git_output(
                ["config", "--get", "remote.origin.url"], source_root
            ),
            "commit": _git_output(["rev-parse", "HEAD"], source_root),
            "dirty_paths": _git_status_short(source_root),
        },
        "license": {
            "path": _path_fields(license_path)["path"],
            "exists": license_path.exists(),
            "sha256": _sha256(license_path),
            "detected_license": _detect_sumo_license(license_path),
        },
        "readme": {
            "path": _path_fields(readme_path)["path"],
            "exists": readme_path.exists(),
            "sha256": _sha256(readme_path),
        },
    }


def _anchor_file_report(ref: str) -> dict[str, Any]:
    path = REPO_ROOT / ref
    return {
        **_file_fingerprint(path),
        "ref": ref,
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
    }


def _seed_anchor_row(family: str) -> dict[str, Any]:
    seed = build_traffic_seed(
        seed_id=f"{family}_source_preflight_s42",
        family=family,
        seed=42,
        difficulty_level="basic",
        difficulty_mode="time_pressure",
    )
    net_ref = str(seed.net_ref)
    route_ref = str(seed.route_ref)
    files = {
        "network": _anchor_file_report(net_ref),
        "route_or_demand": _anchor_file_report(route_ref),
    }
    missing_files = [
        name for name, item in files.items() if item.get("exists") is not True
    ]
    source_lock_missing_fields = []
    if not seed.provenance.url:
        source_lock_missing_fields.append("url")
    if not seed.provenance.commit:
        source_lock_missing_fields.append("commit_or_release_tag")
    if not seed.provenance.lock_strategy:
        source_lock_missing_fields.append("lock_strategy")
    if files["network"].get("sha256") is None:
        source_lock_missing_fields.append("network_file_sha256")
    if files["route_or_demand"].get("sha256") is None:
        source_lock_missing_fields.append("route_or_demand_trace_sha256")
    source_locked = seed.provenance.source_locked and not source_lock_missing_fields

    return {
        "family": family,
        "data_source": seed.provenance.data_source,
        "released_scope": family in RELEASED_SCOPE_FAMILIES,
        "backend_kind": seed.backend_kind,
        "sumo_mode": seed.sumo_mode,
        "horizon_ticks": seed.horizon_ticks,
        "tick_minutes": seed.tick_minutes,
        "net_ref": net_ref,
        "route_ref": route_ref,
        "files": files,
        "missing_files": missing_files,
        "source_locked": source_locked,
        "provenance": {
            "data_source": seed.provenance.data_source,
            "files": list(seed.provenance.files),
            "commit": seed.provenance.commit,
            "url": seed.provenance.url,
            "lock_strategy": seed.provenance.lock_strategy,
            "license": seed.provenance.license,
            "time_window": dict(seed.provenance.time_window),
            "source_locked": source_locked,
            "notes": seed.provenance.notes,
        },
        "source_lock_missing_fields": source_lock_missing_fields,
        "source_denominator_key_candidate": (
            f"traffic:{seed.provenance.data_source}:{net_ref}:{route_ref}:"
            f"seed={seed.seed}"
        ),
    }


def _seed_anchor_preflight() -> dict[str, Any]:
    families = [_seed_anchor_row(family) for family in sorted(_NET_FOR_FAMILY)]
    missing_anchor_families = [
        row["family"] for row in families if row.get("missing_files")
    ]
    source_lock_incomplete_families = [
        row["family"] for row in families if row.get("source_locked") is not True
    ]
    released = [row for row in families if row.get("released_scope")]
    released_missing_anchor_families = [
        row["family"] for row in released if row.get("missing_files")
    ]
    released_source_lock_incomplete_families = [
        row["family"] for row in released if row.get("source_locked") is not True
    ]
    return {
        "family_count_basis": "traffic_seed_factory_families",
        "released_scope_source_id": RELEASED_SCOPE_SOURCE_ID,
        "released_scope_families": list(RELEASED_SCOPE_FAMILIES),
        "frontier_dev_families": [dict(spec) for spec in FRONTIER_FAMILY_SPECS],
        "families": families,
        "missing_anchor_families": missing_anchor_families,
        "source_lock_incomplete_families": source_lock_incomplete_families,
        "released_scope_missing_anchor_families": released_missing_anchor_families,
        "released_scope_source_lock_incomplete_families": (
            released_source_lock_incomplete_families
        ),
    }


def _seed_anchor_summary(anchor_preflight: dict[str, Any]) -> dict[str, Any]:
    families = list(anchor_preflight.get("families") or [])
    return {
        "n_families": len(families),
        "data_sources": sorted({str(row.get("data_source")) for row in families}),
        "families_missing_anchor_files": list(
            anchor_preflight.get("missing_anchor_families") or []
        ),
        "families_source_lock_incomplete": list(
            anchor_preflight.get("source_lock_incomplete_families") or []
        ),
        "all_anchor_files_present": not anchor_preflight.get("missing_anchor_families"),
        "all_source_locked": not anchor_preflight.get(
            "source_lock_incomplete_families"
        ),
        # Released-carrier scope (sumo_ingolstadt). The release gates key off
        # these; the corpus-wide flags above stay as a frontier diagnostic.
        "released_scope_source_id": anchor_preflight.get("released_scope_source_id"),
        "released_scope_families": list(
            anchor_preflight.get("released_scope_families") or []
        ),
        "frontier_dev_families": list(
            anchor_preflight.get("frontier_dev_families") or []
        ),
        "released_scope_all_anchor_files_present": not anchor_preflight.get(
            "released_scope_missing_anchor_families"
        ),
        "released_scope_all_source_locked": not anchor_preflight.get(
            "released_scope_source_lock_incomplete_families"
        ),
    }


def _source_delivery_row(
    spec: dict[str, Any],
    family_rows_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    required_files = list(spec.get("required_files") or [])
    file_reports = [_anchor_file_report(path) for path in required_files]
    missing_required_files = [
        item["ref"] for item in file_reports if item.get("exists") is not True
    ]
    families = list(spec.get("families") or [])
    families_missing_anchor_files = [
        family
        for family in families
        if family_rows_by_name.get(family, {}).get("missing_files")
    ]
    families_source_lock_incomplete = [
        family
        for family in families
        if family_rows_by_name.get(family, {}).get("source_locked") is not True
    ]
    root = REPO_ROOT / str(spec["root"])
    license_path = _first_existing_license(root)
    detected_license = _detect_sumo_license(license_path)
    return {
        **dict(spec),
        "root_report": {
            **_path_fields(root),
            "exists": root.exists(),
            "is_dir": root.is_dir(),
            "git_commit": _git_output(["rev-parse", "HEAD"], root),
            "git_remote_url": _git_output(
                ["config", "--get", "remote.origin.url"], root
            ),
            "dirty_paths": _git_status_short(root),
        },
        "license_report": {
            **_path_fields(license_path),
            "exists": license_path.exists(),
            "sha256": _sha256(license_path),
            "detected_license": detected_license,
            "expected_license": spec.get("license"),
            "license_verified": detected_license == spec.get("license"),
        },
        "required_file_reports": file_reports,
        "missing_required_files": missing_required_files,
        "required_files_present": not missing_required_files,
        "families_missing_anchor_files": families_missing_anchor_files,
        "families_source_lock_incomplete": families_source_lock_incomplete,
        "families_source_locked": not families_source_lock_incomplete,
    }


def _selected_source_current_blockers(
    *,
    source: dict[str, Any],
    runtime: dict[str, Any],
) -> list[str]:
    groups = _selected_source_blocker_groups(source=source, runtime=runtime)
    return [blocker for group in groups.values() for blocker in group]


def _selected_source_blocker_groups(
    *,
    source: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, list[str]]:
    runtime_lock = runtime.get("runtime_lock_status") or {}
    groups = {
        "selected_source_file_lock": [],
        "family_provenance": [],
        "runtime": [],
        "package": [],
    }
    if source.get("root_report", {}).get("exists") is not True:
        groups["selected_source_file_lock"].append("source_root_missing")
    if not source.get("root_report", {}).get("git_commit"):
        groups["selected_source_file_lock"].append("source_git_commit_missing")
    if (source.get("license_report") or {}).get("license_verified") is not True:
        groups["selected_source_file_lock"].append("source_license_not_verified")
    if source.get("required_files_present") is not True:
        groups["selected_source_file_lock"].append("required_files_missing")
    if not source.get("required_file_reports") or any(
        item.get("sha256") is None for item in source.get("required_file_reports") or []
    ):
        groups["selected_source_file_lock"].append("required_file_sha256_missing")
    if source.get("families_source_locked") is not True:
        groups["family_provenance"].append("family_source_locks_incomplete")
    if runtime.get("sumo_transport_available") is not True:
        groups["runtime"].append("sumo_runtime_not_available")
    if runtime_lock.get("sumo_runtime_version_verified") is not True:
        groups["runtime"].append("sumo_runtime_version_not_verified")
    return groups


def _selected_source_file_lock(source: dict[str, Any]) -> dict[str, Any]:
    required_file_reports = list(source.get("required_file_reports") or [])
    checks = {
        "source_root_present": source.get("root_report", {}).get("exists") is True,
        "source_git_commit_recorded": bool(
            source.get("root_report", {}).get("git_commit")
        ),
        "source_license_verified": (
            (source.get("license_report") or {}).get("license_verified") is True
        ),
        "required_files_present": source.get("required_files_present") is True,
        "required_file_sha256_recorded": bool(required_file_reports)
        and all(item.get("sha256") is not None for item in required_file_reports),
    }
    missing_checks = [
        name for name, satisfied in checks.items() if satisfied is not True
    ]
    closed = not missing_checks
    return {
        "schema_version": "0.1",
        "source_id": source.get("source_id"),
        "non_release_artifact": True,
        "closure_gate": "selected_source_root_commit_license_required_files_sha256",
        "status": (
            "selected_source_file_lock_closed"
            if closed
            else "blocked_selected_source_file_lock_incomplete"
        ),
        "closed": closed,
        "checks": checks,
        "missing_checks": missing_checks,
        "source_url": source.get("url"),
        "root": source.get("root"),
        "git_commit": source.get("root_report", {}).get("git_commit"),
        "git_remote_url": source.get("root_report", {}).get("git_remote_url"),
        "license": source.get("license"),
        "license_verified": checks["source_license_verified"],
        "license_report": source.get("license_report") or {},
        "required_file_count": len(required_file_reports),
        "required_file_sha256_count": sum(
            1 for item in required_file_reports if item.get("sha256") is not None
        ),
        "required_file_hashes": required_file_reports,
    }


def _selected_source_candidate(
    *,
    sources: list[dict[str, Any]],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    source_by_id = {str(source.get("source_id")): source for source in sources}
    selected = source_by_id[SELECTED_SOURCE_CANDIDATE_ID]
    blocker_groups = _selected_source_blocker_groups(
        source=selected,
        runtime=runtime,
    )
    return {
        "source_id": selected["source_id"],
        "root": selected["root"],
        "url": selected["url"],
        "license": selected["license"],
        "license_posture": selected["license_posture"],
        "families": list(selected.get("families") or []),
        "selection_reason_code": "first_live_adapter_carrier",
        "why_first": (
            "Apache-2.0 source posture, two Traffic families, and the clearest "
            "incident/priority decision-pressure path make this the narrowest "
            "high-value live-adapter carrier."
        ),
        "decision_pressure_axes": list(SELECTED_SOURCE_DECISION_PRESSURE_AXES),
        "lock_strategy": (
            "git_commit_or_release_tag+sumo_sidecar_runtime_version+"
            "network_route_config_sha256"
        ),
        "lock_closure_required_fields": list(
            SELECTED_SOURCE_LOCK_CLOSURE_REQUIRED_FIELDS
        ),
        "ready_when": list(SELECTED_SOURCE_READY_WHEN),
        "current_blockers": [
            blocker for group in blocker_groups.values() for blocker in group
        ],
        "blocker_groups": blocker_groups,
    }


def _source_delivery_contract(
    runtime: dict[str, Any],
    seed_anchor_preflight: dict[str, Any],
    seed_summary: dict[str, Any],
) -> dict[str, Any]:
    family_rows_by_name = {
        str(row.get("family")): row
        for row in seed_anchor_preflight.get("families") or []
    }
    sources = [
        _source_delivery_row(spec, family_rows_by_name)
        for spec in SOURCE_DELIVERY_SPECS
    ]
    selected_source = _selected_source_candidate(sources=sources, runtime=runtime)
    selected_source_row = {str(source.get("source_id")): source for source in sources}[
        SELECTED_SOURCE_CANDIDATE_ID
    ]
    selected_source_file_lock = _selected_source_file_lock(selected_source_row)
    runtime_lock = runtime.get("runtime_lock_status") or {}
    requirement_status = {
        "sumo_sidecar_runtime_version_locked": (
            runtime.get("sumo_transport_available") is True
            and runtime_lock.get("sumo_runtime_version_verified") is True
        ),
        "selected_source_file_lock_closed": selected_source_file_lock["closed"],
        "all_family_anchor_files_present": (
            seed_summary.get("all_anchor_files_present") is True
        ),
        "all_family_source_locks_complete": (
            seed_summary.get("all_source_locked") is True
        ),
        "network_route_and_config_sha256_recorded": (
            all(
                item.get("sha256") is not None
                for source in sources
                for item in source.get("required_file_reports") or []
            )
            and bool(sources)
        ),
    }
    return {
        "schema_version": "0.1",
        "status": (
            "ready_for_non_release_live_adapter_probe"
            if all(requirement_status.values())
            else "blocked_waiting_for_source_packages"
        ),
        "root_count_basis": "traffic_source_package_roots",
        "n_source_roots": len(sources),
        "required_before_live_adapter_probe": list(
            SOURCE_DELIVERY_REQUIRED_BEFORE_LIVE_ADAPTER_PROBE
        ),
        "requirement_status": requirement_status,
        "selected_source_id": selected_source["source_id"],
        "selected_source_candidate": selected_source,
        "selected_source_file_lock": selected_source_file_lock,
        "next_source_lock_action": (
            (
                "Selected sumo_ingolstadt source root, family provenance locks, "
                "commit, license, and required-file SHA-256s are closed; next "
                "close remaining non-selected Traffic source packages, SUMO "
                "sidecar runtime/version, and the non-release live adapter proof."
            )
            if selected_source_file_lock["closed"]
            and selected_source_row.get("families_source_locked") is True
            else (
                "Selected sumo_ingolstadt source root, commit, license, and "
                "required-file SHA-256s are closed; next close Traffic family "
                "provenance/source locks, SUMO sidecar runtime/version, and the "
                "non-release live adapter proof."
            )
            if selected_source_file_lock["closed"]
            else (
                "Acquire/verify works/sumo_ingolstadt, record the upstream commit "
                "or release tag plus Apache-2.0 license evidence, then hash the "
                "network, route, and simulation config files before any live "
                "adapter probe."
            )
        ),
        "sources": sources,
        "forbidden_sources": list(FORBIDDEN_SOURCE_SPECS),
        "validation_commands": list(SOURCE_DELIVERY_VALIDATION_COMMANDS),
    }


def _input_fingerprints(
    seed_anchor_preflight: dict[str, Any],
    *,
    source_root: Path,
) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {
        "script": _file_fingerprint(Path(__file__).resolve()),
        "traffic_seed_factory": _file_fingerprint(
            REPO_ROOT / "domains" / "traffic" / "seeds" / "from_lust.py"
        ),
        "traffic_seed_schema": _file_fingerprint(
            REPO_ROOT / "domains" / "traffic" / "seeds" / "schema.py"
        ),
        "sumo_sidecar": _file_fingerprint(
            REPO_ROOT / "core" / "sidecar" / "sumo_sidecar.py"
        ),
        "sumo_backend": _file_fingerprint(
            REPO_ROOT / "domains" / "traffic" / "backends" / "sumo_backend.py"
        ),
        "sumo_source_license": _file_fingerprint(source_root / "LICENSE"),
        "sumo_source_readme": _file_fingerprint(source_root / "README.md"),
    }
    for row in seed_anchor_preflight.get("families") or []:
        family = str(row.get("family"))
        for kind, item in (row.get("files") or {}).items():
            path = REPO_ROOT / str(item.get("ref"))
            files[f"{family}.{kind}"] = _file_fingerprint(path)
    for spec in SOURCE_DELIVERY_SPECS:
        source_id = str(spec.get("source_id"))
        root = REPO_ROOT / str(spec.get("root"))
        files[f"source_delivery.{source_id}.license"] = _file_fingerprint(
            _first_existing_license(root)
        )
        readme = root / "README.md"
        files[f"source_delivery.{source_id}.readme"] = _file_fingerprint(readme)
        for idx, file_ref in enumerate(spec.get("required_files") or []):
            files[f"source_delivery.{source_id}.required_file.{idx}"] = (
                _file_fingerprint(REPO_ROOT / str(file_ref))
            )

    return {
        "schema_version": "0.1",
        "files": files,
        "all_present": all(item["exists"] is True for item in files.values()),
        "all_file_states_match_current_files": all(
            item["matches_current_file"] is True for item in files.values()
        ),
    }


def _release_blockers(
    runtime: dict[str, Any],
    source_tree: dict[str, Any],
    seed_summary: dict[str, Any],
    live_adapter_probe: dict[str, Any],
    *,
    behavioral_gate_passed: bool = False,
    materializer_draft_ready: bool = False,
    written_draft_behavioral_audit_passed: bool = False,
) -> list[str]:
    blockers: list[str] = []
    # The released sumo_ingolstadt carrier is driven entirely by the raw SUMO
    # sidecar (libsumo/traci transport — captured by `sumo_transport_available`),
    # NOT by the `sumo-rl` gym package: nothing under domains/traffic imports
    # `sumo_rl`. So the gym package import / source-lock are not carrier release
    # gates. The `sumo-rl` package stays tracked as a dev-only FRONTIER
    # requirement for the `sumo_rl_traffic_signal` candidate (daily_peak_commute
    # etc.) under the data-expansion frontier inventory, not here.
    if runtime.get("sumo_transport_available") is not True:
        blockers.append("sumo_runtime_not_available")
    if source_tree.get("exists") is not True:
        blockers.append("sumo_source_tree_missing")
    if source_tree.get("license", {}).get("detected_license") != "EPL-2.0":
        blockers.append("sumo_runtime_license_not_verified")
    # Release gates reflect the released carrier scope (sumo_ingolstadt). The
    # dev-only frontier families (LuST/TAPAS/OSM) stay visible as a diagnostic
    # under seed_anchor_summary.frontier_dev_families but never block a carrier
    # that does not include them.
    if seed_summary.get("released_scope_all_anchor_files_present") is not True:
        blockers.append("traffic_seed_anchor_files_missing")
    if seed_summary.get("released_scope_all_source_locked") is not True:
        blockers.append("traffic_source_locks_incomplete")
    # A recorded, scope-locked, passing behavioral gate clears the behavioral
    # gate rung; it can only *clear* this blocker, never add release readiness
    # (the release materializer rung below stays open).
    if not behavioral_gate_passed:
        blockers.append("missing_behavioral_gates")
    # A recorded, scope-locked, internally-validated release-materializer draft
    # *advances* this rung honestly: it proves a materializer exists and produces
    # validated registry/primary/core/manifest artifacts, but those artifacts are
    # written to a non-release draft dir and carry no release-level behavioral
    # audit + the live source-lock ladder is not closed on a release host — so the
    # carrier stays non-release with the stricter successor blocker rather than
    # losing the rung entirely.
    # Materializer rung, in increasing closure order:
    #   (a) no recorded draft               → missing_release_materializer
    #   (b) draft present but not audited    → ..._not_audited_into_release
    #   (c) draft audited on written rows    → real materializer still not run
    #       into a release/ wrapper (the honest stricter successor).
    # A passing written-draft behavioral audit can only *advance* this rung; it
    # never adds release readiness (the publishable descriptor + release-wrapper
    # write stay open).
    if not materializer_draft_ready:
        blockers.append("missing_release_materializer")
    elif not written_draft_behavioral_audit_passed:
        blockers.append("release_materializer_draft_not_audited_into_release")
    else:
        blockers.append("real_release_materializer_not_run_into_release_wrapper")
    blockers.extend(live_adapter_probe.get("blocker_codes") or [])
    return sorted(set(blockers))


def _one_live_signal_plan_run(*, corridor: str, program_slot: str) -> dict[str, Any]:
    """Run one real-SUMO ``change_signal_plan`` through the public env path.

    Returns the auditable facts of a single live run: transport, the resolved
    real SUMO program, the tool-result + evidence wiring, and an independent
    live read-back of the TLS program. Caller compares two runs for
    determinism. Assumes ``OPERATE_TRAFFIC_BACKEND_REAL=1`` and ``sumo_available()``.
    """
    seed = build_traffic_seed(
        seed_id="live_probe/incident_response",
        family="incident_response",
        seed=42,
        difficulty_level="basic",
        difficulty_mode="time_pressure",
    )
    seed.backend_kind = "sumo"
    seed.backend_config = {
        **seed.backend_config,
        "backend_kind": "sumo",
        # Bound the live window to a busy daytime hour + small substeps so the
        # probe is fast and meaningful (the released sim window is a later gate).
        "sumo_extra_args": ("--begin", "25200", "--end", "25320"),
        "sumo_substeps_per_tick": 3,
    }
    tls_id = seed.backend_config["corridor_tls_map"][corridor]
    real_program = seed.backend_config["sumo_corridor_program_map"][corridor][
        program_slot
    ]

    env = TrafficEnvironment()
    env.reset(seed.to_dict(), seed=seed.seed)
    backend = env._backend
    transport = getattr(backend, "_transport", None)
    ret = env.step(
        Action(
            tool_calls=[
                ToolCall(
                    name="change_signal_plan",
                    args={"corridor": corridor, "program": program_slot},
                )
            ],
            dominant="change_signal_plan",
        )
    )
    result = ret.tool_results[0]
    # Independent live read-back straight from the SUMO transport.
    live_program = backend._sidecar._conn.trafficlight.getProgram(tls_id)
    tick_advanced = len(getattr(backend, "_tick_records", [])) > 0
    control_rows = env.evidence.items_by_kind("control") if env.evidence else []
    trust_event_ids = [
        row.evidence_id
        for row in (env.evidence.items_by_kind("trust_event") if env.evidence else [])
        if row.payload.get("source_tool") == "change_signal_plan"
        and row.payload.get("corridor") == corridor
    ]
    evidence_trust_ok = any(
        row.payload.get("sumo_state_mutated") is True
        and row.payload.get("sumo_program_readback_matches") is True
        and row.payload.get("trust_event")
        for row in control_rows
    )
    score_consumption = _score_live_tool_evidence_consumption(
        env=env,
        seed=seed,
        trust_event_ids=trust_event_ids,
    )
    facts = {
        "selected_transport": transport,
        "backend_reset_starts_sumo_sidecar": transport is not None
        and backend._sidecar is not None,
        "backend_tick_advances_live_sumo": tick_advanced,
        "tool_ok": bool(result.ok),
        "tool_state_changing": bool(result.state_changing),
        "sumo_state_mutated": bool(result.payload.get("sumo_state_mutated")),
        "sumo_program_readback_matches": bool(
            result.payload.get("sumo_program_readback_matches")
        ),
        "resolved_real_program": str(result.payload.get("sumo_program_id")),
        "live_readback_program": str(live_program),
        "live_readback_matches_resolved": str(live_program) == str(real_program),
        "evidence_id": result.evidence_id,
        "evidence_id_in_info": result.evidence_id in ret.info.evidence_ids,
        "evidence_control_row_with_trust": evidence_trust_ok,
        "score_consumption_probe_passed": score_consumption[
            "score_consumption_probe_passed"
        ],
        "score_consumed_evidence_ids": score_consumption["score_consumed_evidence_ids"],
        "score_consuming_dimensions": score_consumption["score_consuming_dimensions"],
        "score_consumption_blocker_codes": score_consumption[
            "score_consumption_blocker_codes"
        ],
        "sumo_tls_id": tls_id,
    }
    backend.close()
    return facts


def _score_live_tool_evidence_consumption(
    *,
    env: TrafficEnvironment,
    seed: Any,
    trust_event_ids: list[str],
) -> dict[str, Any]:
    """Prove live-control evidence is consumed by scorer dimensions.

    This is a non-release probe: it does not change scoring semantics. It runs
    the public scorer over the just-executed live episode and checks whether the
    trust-event evidence emitted by ``change_signal_plan`` is actually cited by
    an applicable score dimension. The minimal contract is
    ``stakeholder_management`` because traffic native controls update trust via
    ``trust_event`` rows.
    """

    blockers: list[str] = []
    if not trust_event_ids:
        blockers.append("missing_live_tool_trust_event")
    if env.evidence is None:
        blockers.append("missing_evidence_logger")

    try:
        gt = env.ground_truth()
        backend_records = (
            env._backend.scoring_records()
            if hasattr(env._backend, "scoring_records")
            else []
        )
        score = score_episode(
            ScoringInputs(
                backend_tick_records=list(backend_records),
                realized_events=[
                    {**dict(item.payload), "tick": item.tick}
                    for item in (
                        env.evidence.items_by_kind("realized_event")
                        if env.evidence
                        else []
                    )
                ],
                cost_components=dict(gt.get("cost_components") or {}),
                per_load_shed_mwh=dict(gt.get("per_corridor_delay_minutes") or {}),
                load_classes={},
                evidence_logger=env.evidence,
                stakeholder_mgr=env.stakeholders,
                dilemma_mgr=env.dilemmas,
                chose_fatal_option=bool(gt.get("chose_fatal_option", False)),
                counterfactual_report=None,
                foresight_summary=None,
                lp_optimum=None,
                difficulty_level=str(getattr(seed, "difficulty_level", "basic")),
                scenario_signature=str(seed.signature()),
            )
        )
    except Exception as exc:
        return {
            "score_consumption_probe_passed": False,
            "score_consumed_evidence_ids": [],
            "score_consuming_dimensions": [],
            "score_consumption_blocker_codes": [
                *blockers,
                f"score_episode_failed:{type(exc).__name__}",
            ],
        }

    consumed_ids: set[str] = set()
    consuming_dimensions: list[str] = []
    trust_set = set(trust_event_ids)
    for dim in score.dimensions:
        if not dim.applicable:
            continue
        overlap = sorted(trust_set & set(dim.evidence_ids))
        if not overlap:
            continue
        consumed_ids.update(overlap)
        consuming_dimensions.append(dim.name)

    if "stakeholder_management" not in consuming_dimensions:
        blockers.append("stakeholder_management_missing_live_tool_evidence")
    if not consumed_ids:
        blockers.append("score_dimensions_did_not_consume_live_tool_evidence")

    return {
        "score_consumption_probe_passed": not blockers,
        "score_consumed_evidence_ids": sorted(consumed_ids),
        "score_consuming_dimensions": sorted(set(consuming_dimensions)),
        "score_consumption_blocker_codes": sorted(set(blockers)),
    }


def run_live_adapter_probe(
    *, corridor: str = "hospital_access", program_slot: str = "incident_relief"
) -> dict[str, Any]:
    """Opt-in, non-release live-SUMO adapter probe.

    Launches SUMO (only when ``OPERATE_TRAFFIC_BACKEND_REAL=1`` and a transport is
    reachable), runs the public ``change_signal_plan`` path twice, and records
    whether the live backend executed, mutated a real TLS program (read back),
    wired auditable evidence, and replayed deterministically. The result is
    locked to the binding net sha256 so a stale artifact cannot silently flip a
    preflight gate. It never modifies release artifacts; ``release_ready`` stays
    ``False`` (other gates — behavioral gating, materializer — remain open).
    """

    binding = _load_ingolstadt_binding()
    net_sha = str(binding.get("net_sha256") or "")
    base: dict[str, Any] = {
        "schema_version": "0.1",
        "scope": "sumo_live_adapter_probe",
        "non_release_artifact": True,
        "release_ready": False,
        "generated_at_utc": datetime.now(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "env_gate": "OPERATE_TRAFFIC_BACKEND_REAL=1",
        "binding_net_sha256": net_sha,
        "corridor": corridor,
        "program_slot": program_slot,
    }
    if os.environ.get("OPERATE_TRAFFIC_BACKEND_REAL") != "1":
        return {
            **base,
            "executed_with_live_backend": False,
            "status": "skipped_env_gate_unset",
            "reason": "OPERATE_TRAFFIC_BACKEND_REAL != 1",
        }
    if not sumo_available():
        return {
            **base,
            "executed_with_live_backend": False,
            "status": "skipped_no_sumo_transport",
            "reason": "no reachable SUMO transport (libsumo/traci/docker)",
        }

    run_a = _one_live_signal_plan_run(corridor=corridor, program_slot=program_slot)
    run_b = _one_live_signal_plan_run(corridor=corridor, program_slot=program_slot)
    deterministic = (
        run_a["resolved_real_program"] == run_b["resolved_real_program"]
        and run_a["live_readback_program"] == run_b["live_readback_program"]
        and run_a["live_readback_matches_resolved"] is True
        and run_b["live_readback_matches_resolved"] is True
    )
    tool_probe_passed = (
        run_a["tool_ok"]
        and run_a["tool_state_changing"]
        and run_a["sumo_state_mutated"]
        and run_a["sumo_program_readback_matches"]
        and run_a["live_readback_matches_resolved"]
    )
    evidence_probe_passed = (
        run_a["evidence_id"] is not None
        and run_a["evidence_id_in_info"]
        and run_a["evidence_control_row_with_trust"]
    )
    score_consumption_probe_passed = run_a["score_consumption_probe_passed"]
    executed = (
        run_a["backend_reset_starts_sumo_sidecar"]
        and run_a["backend_tick_advances_live_sumo"]
    )
    passed = bool(
        executed
        and tool_probe_passed
        and evidence_probe_passed
        and score_consumption_probe_passed
        and deterministic
    )
    return {
        **base,
        "selected_transport": run_a["selected_transport"],
        "sumo_tls_id": run_a["sumo_tls_id"],
        "resolved_real_program": run_a["resolved_real_program"],
        "executed_with_live_backend": bool(executed),
        "backend_reset_starts_sumo_sidecar": run_a["backend_reset_starts_sumo_sidecar"],
        "backend_tick_advances_live_sumo": run_a["backend_tick_advances_live_sumo"],
        "tool_protocol_effect_probe_passed": bool(tool_probe_passed),
        "evidence_wiring_probe_passed": bool(evidence_probe_passed),
        "score_consumption_probe_passed": bool(score_consumption_probe_passed),
        "score_consumed_evidence_ids": list(run_a["score_consumed_evidence_ids"]),
        "score_consuming_dimensions": list(run_a["score_consuming_dimensions"]),
        "score_consumption_blocker_codes": list(
            run_a["score_consumption_blocker_codes"]
        ),
        "deterministic_replay_passed": bool(deterministic),
        "all_probes_passed": passed,
        "status": "executed_with_live_backend" if passed else "live_probe_incomplete",
        "run_a": run_a,
        "run_b": run_b,
    }


def _one_live_headroom_episode(
    *, acting: bool, n_ticks: int, program_slot: str
) -> dict[str, Any]:
    """Run one real-SUMO episode under either the wait floor or an acting policy
    and return its realized physical outcome.

    The wait floor never touches the signals (the net's default programs run);
    the acting policy switches *every* bound corridor to ``program_slot`` on the
    first tick and then holds. Both run from the identical seed/window, so any
    difference in the realized per-corridor delay vector is the *causal* effect
    of the signal-plan action — live decision headroom, not RNG. Assumes
    ``OPERATE_TRAFFIC_BACKEND_REAL=1`` and ``sumo_available()``.
    """
    seed = build_traffic_seed(
        seed_id="live_headroom/incident_response",
        family="incident_response",
        seed=42,
        difficulty_level="basic",
        difficulty_mode="time_pressure",
    )
    seed.backend_kind = "sumo"
    seed.backend_config = {
        **seed.backend_config,
        "backend_kind": "sumo",
        # Morning-peak window so vehicles populate the bound TLS lanes and the
        # signal action has real traffic to act on (the released sim window is a
        # later gate); large substeps so a few ticks integrate enough physics.
        "sumo_extra_args": ("--begin", "28800", "--end", "32400"),
        "sumo_substeps_per_tick": 120,
    }
    corridors = [c.corridor_id for c in seed.corridors]

    env = TrafficEnvironment()
    env.reset(seed.to_dict(), seed=seed.seed)
    backend = env._backend
    for tick in range(int(n_ticks)):
        if acting and tick == 0:
            action = Action(
                tool_calls=[
                    ToolCall(
                        name="change_signal_plan",
                        args={"corridor": corridor, "program": program_slot},
                    )
                    for corridor in corridors
                ],
                dominant="change_signal_plan",
            )
        else:
            action = Action(tool_calls=[ToolCall(name="wait")], dominant="wait")
        env.step(action)

    per_corridor = backend.per_corridor_delay_minutes()
    travel_cost = float(backend.ground_truth_costs().get("travel_time_cost", 0.0))
    control_rows = env.evidence.items_by_kind("control") if env.evidence else []
    evidence_trust_ok = any(
        row.payload.get("sumo_state_mutated") is True
        and row.payload.get("sumo_program_readback_matches") is True
        and row.payload.get("trust_event")
        for row in control_rows
    )
    backend.close()
    return {
        "per_corridor_delay_minutes": {
            k: round(float(v), 3) for k, v in per_corridor.items()
        },
        "total_delay_minutes": round(sum(float(v) for v in per_corridor.values()), 3),
        "travel_time_cost": round(travel_cost, 3),
        "state_changes_observed": len(control_rows),
        "evidence_linked": bool(evidence_trust_ok),
    }


def run_live_headroom_probe(
    *, n_ticks: int = 6, program_slot: str = "incident_relief"
) -> dict[str, Any]:
    """Opt-in, non-release live-SUMO *decision-headroom* probe.

    Runs two real-SUMO episodes from the identical seed/window — a wait floor and
    an acting signal-plan policy — and measures whether the action causally moves
    the realized per-corridor delay vector. A third (repeated wait) episode pins
    metric determinism. This is the live counterpart of the deterministic-mock
    behavioral gate: it proves the agent's action changes the *real physical
    future*, evidence-linked and reproducible.

    Honest scope: the live net does **not** inherit the mock's engineered
    "relief always helps" throughput bonus (see the binding caveat). The probe
    therefore asserts *causal, differentiated action-dependence* of the realized
    outcome, and records the aggregate-delay delta as an observation (which may
    be a regression on a given window) rather than gating on improvement. Locked
    to the binding net sha so a stale artifact cannot flip a gate. Never writes
    release artifacts; ``release_ready`` stays ``False``.
    """
    binding = _load_ingolstadt_binding()
    net_sha = str(binding.get("net_sha256") or "")
    base: dict[str, Any] = {
        "schema_version": "0.1",
        "scope": "sumo_live_headroom_probe",
        "non_release_artifact": True,
        "release_ready": False,
        "generated_at_utc": datetime.now(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "env_gate": "OPERATE_TRAFFIC_BACKEND_REAL=1",
        "binding_net_sha256": net_sha,
        "program_slot": program_slot,
        "n_ticks": int(n_ticks),
        "headroom_metric": "l1_per_corridor_delay_minutes_acting_vs_wait",
        "min_headroom_l1_minutes": MIN_HEADROOM_L1_MINUTES,
        "min_corridors_changed": MIN_HEADROOM_CORRIDORS_CHANGED,
    }
    if os.environ.get("OPERATE_TRAFFIC_BACKEND_REAL") != "1":
        return {
            **base,
            "executed_with_live_backend": False,
            "status": "skipped_env_gate_unset",
            "reason": "OPERATE_TRAFFIC_BACKEND_REAL != 1",
        }
    if not sumo_available():
        return {
            **base,
            "executed_with_live_backend": False,
            "status": "skipped_no_sumo_transport",
            "reason": "no reachable SUMO transport (libsumo/traci/docker)",
        }

    wait = _one_live_headroom_episode(
        acting=False, n_ticks=n_ticks, program_slot=program_slot
    )
    acting = _one_live_headroom_episode(
        acting=True, n_ticks=n_ticks, program_slot=program_slot
    )
    wait_repeat = _one_live_headroom_episode(
        acting=False, n_ticks=n_ticks, program_slot=program_slot
    )

    corridors = sorted(
        set(wait["per_corridor_delay_minutes"])
        | set(acting["per_corridor_delay_minutes"])
    )
    per_corridor_delta = {
        c: round(
            acting["per_corridor_delay_minutes"].get(c, 0.0)
            - wait["per_corridor_delay_minutes"].get(c, 0.0),
            3,
        )
        for c in corridors
    }
    l1_delta = round(sum(abs(v) for v in per_corridor_delta.values()), 3)
    corridors_changed = sorted(c for c, v in per_corridor_delta.items() if abs(v) > 0.0)
    max_abs_corridor_delta = round(
        max((abs(v) for v in per_corridor_delta.values()), default=0.0), 3
    )
    aggregate_delay_delta = round(
        acting["total_delay_minutes"] - wait["total_delay_minutes"], 3
    )
    travel_cost_delta = round(acting["travel_time_cost"] - wait["travel_time_cost"], 3)

    metric_deterministic = (
        wait["per_corridor_delay_minutes"] == wait_repeat["per_corridor_delay_minutes"]
        and wait["travel_time_cost"] == wait_repeat["travel_time_cost"]
    )
    causal_outcome_change = l1_delta >= MIN_HEADROOM_L1_MINUTES
    differentiated = len(corridors_changed) >= MIN_HEADROOM_CORRIDORS_CHANGED
    state_changed = acting["state_changes_observed"] > 0
    evidence_linked = acting["evidence_linked"] is True
    executed = bool(wait["per_corridor_delay_minutes"]) and bool(
        acting["per_corridor_delay_minutes"]
    )

    passed = bool(
        executed
        and causal_outcome_change
        and differentiated
        and state_changed
        and evidence_linked
        and metric_deterministic
    )
    return {
        **base,
        "executed_with_live_backend": bool(executed),
        "headroom_l1_minutes": l1_delta,
        "corridors_changed": corridors_changed,
        "n_corridors_changed": len(corridors_changed),
        "max_abs_corridor_delta_minutes": max_abs_corridor_delta,
        "per_corridor_delta_minutes": per_corridor_delta,
        "aggregate_delay_delta_minutes": aggregate_delay_delta,
        "travel_time_cost_delta": travel_cost_delta,
        "acting_reduces_aggregate_delay": aggregate_delay_delta < 0.0,
        "causal_outcome_change_probe_passed": bool(causal_outcome_change),
        "differentiated_across_corridors": bool(differentiated),
        "state_change_probe_passed": bool(state_changed),
        "evidence_wiring_probe_passed": bool(evidence_linked),
        "metric_deterministic_replay_passed": bool(metric_deterministic),
        "all_probes_passed": passed,
        "status": (
            "live_decision_headroom_proven"
            if passed
            else "live_headroom_probe_incomplete"
        ),
        "live_aggregate_semantics_caveat": (
            "Causal, differentiated action-dependence of the realized per-corridor "
            "delay vector is proven on the live net. The mock backend's engineered "
            "throughput bonus (e.g. incident_relief=+30%) is NOT claimed live: the "
            "aggregate-delay delta on a given window may be a regression because a "
            "blanket relief switch is not the live optimum. Whether a *good* policy "
            "improves aggregate delay is what the full scorer measures."
        ),
        "wait_floor": wait,
        "acting": acting,
        "wait_repeat": wait_repeat,
    }


def _load_recorded_live_probe(
    path: Path, *, expected_net_sha: str
) -> dict[str, Any] | None:
    """Load a recorded live-probe artifact if it is present, valid, passing, and
    locked to the current binding net sha (else ``None`` — preflight stays
    honestly blocked)."""
    if not path.exists():
        return None
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(rec, dict):
        return None
    if rec.get("scope") != "sumo_live_adapter_probe":
        return None
    if str(rec.get("binding_net_sha256") or "") != str(expected_net_sha or ""):
        return None
    if rec.get("all_probes_passed") is not True:
        return None
    return rec


def _load_recorded_live_headroom(
    path: Path, *, expected_net_sha: str
) -> dict[str, Any] | None:
    """Load a recorded live-headroom artifact if it is present, valid, passing,
    and locked to the current binding net sha (else ``None`` — the carrier
    reports live decision headroom as not-yet-proven).

    Like the adapter probe, the headroom probe launches real SUMO and is
    net-sha-locked, so a stale artifact recorded against a different network
    cannot falsely assert live headroom."""
    if not path.exists():
        return None
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(rec, dict):
        return None
    if rec.get("scope") != "sumo_live_headroom_probe":
        return None
    if str(rec.get("binding_net_sha256") or "") != str(expected_net_sha or ""):
        return None
    if rec.get("all_probes_passed") is not True:
        return None
    return rec


def _load_recorded_behavioral_gate(path: Path) -> dict[str, Any] | None:
    """Load a recorded Traffic behavioral-gate artifact if it is present, valid,
    passing, and locked to the current released carrier scope (else ``None`` —
    the preflight stays honestly blocked on ``missing_behavioral_gates``).

    The behavioral gate runs on the deterministic mock backend, so unlike the
    live-probe it carries no net-sha lock; instead it is pinned to the released
    carrier scope (source id + family set) so a gate recorded against a
    different scope cannot falsely clear this carrier's gate."""
    if not path.exists():
        return None
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(rec, dict):
        return None
    if rec.get("scope") != "traffic_behavioral_gate":
        return None
    if rec.get("gate_passed") is not True:
        return None
    if str(rec.get("released_scope_source_id") or "") != RELEASED_SCOPE_SOURCE_ID:
        return None
    if list(rec.get("released_scope_families") or []) != list(RELEASED_SCOPE_FAMILIES):
        return None
    return rec


def _load_recorded_materializer_draft(path: Path) -> dict[str, Any] | None:
    """Load a recorded Traffic release-materializer-draft summary if present,
    valid, internally ready, and scope-locked to the released carrier (else
    ``None`` — the preflight stays on ``missing_release_materializer``).

    Like the behavioral gate, the draft is produced on the deterministic mock
    backend, so it is pinned to the released carrier scope rather than a net-sha
    lock. A ready draft can only *advance* the materializer rung; it never adds
    release readiness."""
    if not path.exists():
        return None
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(rec, dict):
        return None
    if rec.get("scope") != "traffic_release_materializer_draft":
        return None
    if rec.get("status") != "candidate_release_materializer_draft_ready":
        return None
    if (rec.get("validation") or {}).get("all_passed") is not True:
        return None
    if str(rec.get("released_scope_source_id") or "") != RELEASED_SCOPE_SOURCE_ID:
        return None
    return rec


def _load_recorded_written_draft_behavioral_audit(
    path: Path,
) -> dict[str, Any] | None:
    """Load a recorded Traffic written-draft behavioral audit if present, valid,
    internally ready, and scope-locked to the released carrier (else ``None`` —
    the materializer rung stays on ``..._not_audited_into_release``).

    Like the behavioral gate + draft, this audit runs on the deterministic mock
    backend over the written draft artifacts, so it is pinned to the released
    carrier scope rather than a net-sha lock. A ready audit can only *advance*
    the materializer rung; it never adds release readiness."""
    if not path.exists():
        return None
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(rec, dict):
        return None
    if rec.get("scope") != "traffic_written_draft_behavioral_audit":
        return None
    if rec.get("status") != "candidate_written_draft_behavioral_audit_ready":
        return None
    if (rec.get("written_draft_checks") or {}).get("all_passed") is not True:
        return None
    if str(rec.get("released_scope_source_id") or "") != RELEASED_SCOPE_SOURCE_ID:
        return None
    return rec


def _live_adapter_probe(
    runtime: dict[str, Any],
    source_delivery_contract: dict[str, Any],
    recorded: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected_file_lock_closed = (
        source_delivery_contract.get("selected_source_file_lock") or {}
    ).get("closed") is True
    runtime_available = runtime.get("sumo_transport_available") is True
    # A valid, net-sha-locked, passing recorded live probe converts the two
    # execution/evidence gates from synthetic to real. It can only *clear*
    # blockers — it never adds release readiness (behavioral gates + release
    # materializer remain open below in ``_release_blockers``).
    live_executed = bool(recorded and recorded.get("executed_with_live_backend"))
    tool_probe_passed = bool(
        recorded and recorded.get("tool_protocol_effect_probe_passed")
    )
    evidence_probe_passed = bool(
        recorded and recorded.get("evidence_wiring_probe_passed")
    )
    score_consumption_probe_passed = bool(
        recorded and recorded.get("score_consumption_probe_passed")
    )
    replay_passed = bool(recorded and recorded.get("deterministic_replay_passed"))
    score_consumption_blockers = list(
        (recorded or {}).get("score_consumption_blocker_codes") or []
    )
    if recorded and "score_consumption_probe_passed" not in recorded:
        score_consumption_blockers.append(
            "recorded_live_probe_missing_score_consumption_fields"
        )

    blockers: list[str] = []
    if not runtime_available:
        blockers.append("sumo_runtime_not_available")
    if not selected_file_lock_closed:
        blockers.append("selected_source_file_lock_incomplete")
    if not LIVE_NATIVE_TOOL_MAPPING_AVAILABLE:
        blockers.append("native_tool_mapping_unavailable")
    if not live_executed:
        blockers.append("missing_live_backend_execution_probe")
    if not evidence_probe_passed:
        blockers.append("missing_evidence_wiring_probe")
    if not score_consumption_probe_passed:
        blockers.append("missing_score_consumption_probe")

    live_execution_rung_passed = (
        live_executed and tool_probe_passed and evidence_probe_passed and replay_passed
    )
    if live_execution_rung_passed:
        rung = "executed_with_live_backend"
        status = (
            "executed_with_live_backend"
            if score_consumption_probe_passed
            else "blocked_missing_score_consumption_probe"
        )
    elif not runtime_available:
        rung = "adapted_from_mock_only"
        status = "blocked_missing_sumo_runtime"
    else:
        rung = "adapted_from_mock_only"
        status = "blocked_waiting_for_live_adapter_probe"

    return {
        "schema_version": "0.1",
        "scope": "sumo_live_adapter_probe",
        "non_release_artifact": True,
        "release_ready": False,
        "release_reentry_ready": False,
        "proceed_commands": [],
        "backend_kind": "sumo",
        "env_gate": "OPERATE_TRAFFIC_BACKEND_REAL=1",
        "selected_transport": (
            (recorded or {}).get("selected_transport")
            or runtime.get("selected_transport")
        ),
        "sumo_runtime_available": runtime_available,
        "selected_source_file_lock_closed": selected_file_lock_closed,
        "native_tool_mapping_available": LIVE_NATIVE_TOOL_MAPPING_AVAILABLE,
        "native_tool_mapping_scope": [
            "change_signal_plan",
        ],
        "native_tool_mapping_proofs": list(LIVE_NATIVE_TOOL_MAPPING_PROOFS),
        "recorded_live_probe_present": recorded is not None,
        "recorded_live_probe_generated_at_utc": (recorded or {}).get(
            "generated_at_utc"
        ),
        "recorded_live_probe_binding_net_sha256": (recorded or {}).get(
            "binding_net_sha256"
        ),
        "executed_with_live_backend": live_executed,
        "backend_reset_starts_sumo_sidecar": bool(
            recorded and recorded.get("backend_reset_starts_sumo_sidecar")
        ),
        "backend_tick_advances_live_sumo": bool(
            recorded and recorded.get("backend_tick_advances_live_sumo")
        ),
        "tool_protocol_effect_probe_passed": tool_probe_passed,
        "evidence_wiring_probe_passed": evidence_probe_passed,
        "score_consumption_probe_passed": score_consumption_probe_passed,
        "score_consumed_evidence_ids": list(
            (recorded or {}).get("score_consumed_evidence_ids") or []
        ),
        "score_consuming_dimensions": list(
            (recorded or {}).get("score_consuming_dimensions") or []
        ),
        "score_consumption_blocker_codes": sorted(set(score_consumption_blockers)),
        "deterministic_replay_passed": replay_passed,
        "source_integration_rung": rung,
        "status": status,
        "blocker_codes": sorted(set(blockers)),
        "required_before_release": list(LIVE_ADAPTER_REQUIRED_BEFORE_RELEASE),
        "next_probe": (
            LIVE_ADAPTER_GENERATE_COMMAND
            if score_consumption_blockers
            else (
                "Run a non-release TrafficEnvironment scenario with "
                "backend_kind='sumo' and OPERATE_TRAFFIC_BACKEND_REAL=1 on a host with "
                "SUMO, then verify sidecar reset/tick, native tool mutation, "
                "evidence rows, and deterministic replay before release packaging."
            )
        ),
    }


def build_sumo_traffic_source_preflight_report(
    *,
    source_root: Path = DEFAULT_SUMO_SOURCE_ROOT,
    live_probe_path: Path = DEFAULT_LIVE_PROBE_OUTPUT,
    live_headroom_path: Path = DEFAULT_LIVE_HEADROOM_OUTPUT,
    behavioral_gate_path: Path = DEFAULT_BEHAVIORAL_GATE_OUTPUT,
    materializer_draft_path: Path = DEFAULT_MATERIALIZER_DRAFT_OUTPUT,
    written_draft_behavioral_audit_path: Path = (
        DEFAULT_WRITTEN_DRAFT_BEHAVIORAL_AUDIT_OUTPUT
    ),
) -> dict[str, Any]:
    runtime = _runtime_preflight()
    source_tree = _sumo_source_tree(source_root)
    seed_anchor_preflight = _seed_anchor_preflight()
    seed_summary = _seed_anchor_summary(seed_anchor_preflight)
    source_delivery_contract = _source_delivery_contract(
        runtime, seed_anchor_preflight, seed_summary
    )
    # Read-only ingestion of any recorded live-probe artifact (this build never
    # launches SUMO; ``--run-live-probe`` produces the artifact separately).
    expected_net_sha = str(_load_ingolstadt_binding().get("net_sha256") or "")
    recorded_live_probe = _load_recorded_live_probe(
        live_probe_path, expected_net_sha=expected_net_sha
    )
    live_adapter_probe = _live_adapter_probe(
        runtime, source_delivery_contract, recorded_live_probe
    )
    # Read-only ingestion of any recorded live-headroom artifact (this build
    # never launches SUMO; ``--run-live-headroom-probe`` produces it). It is a
    # release-quality evidence signal — it makes live decision headroom
    # audit-visible — and never adds release readiness on its own.
    recorded_live_headroom = _load_recorded_live_headroom(
        live_headroom_path, expected_net_sha=expected_net_sha
    )
    live_headroom_proven = recorded_live_headroom is not None
    # Read-only ingestion of any recorded behavioral-gate artifact (this build
    # never runs the gate; ``scripts/traffic_behavioral_gate.py`` produces it).
    recorded_behavioral_gate = _load_recorded_behavioral_gate(behavioral_gate_path)
    behavioral_gate_passed = recorded_behavioral_gate is not None
    behavioral_gate_summary = (recorded_behavioral_gate or {}).get("summary") or {}
    # Read-only ingestion of any recorded release-materializer-draft summary
    # (this build never runs the materializer; the draft script produces it).
    recorded_materializer_draft = _load_recorded_materializer_draft(
        materializer_draft_path
    )
    materializer_draft_ready = recorded_materializer_draft is not None
    materializer_draft_summary = (recorded_materializer_draft or {}).get(
        "summary"
    ) or {}
    # Read-only ingestion of any recorded written-draft behavioral audit (this
    # build never runs it; the audit script produces it). A ready audit advances
    # the materializer rung from "draft not audited" to "real materializer not
    # run into a release wrapper"; it never adds release readiness.
    recorded_written_draft_audit = _load_recorded_written_draft_behavioral_audit(
        written_draft_behavioral_audit_path
    )
    written_draft_behavioral_audit_passed = recorded_written_draft_audit is not None
    blockers = _release_blockers(
        runtime,
        source_tree,
        seed_summary,
        live_adapter_probe,
        behavioral_gate_passed=behavioral_gate_passed,
        materializer_draft_ready=materializer_draft_ready,
        written_draft_behavioral_audit_passed=written_draft_behavioral_audit_passed,
    )
    foundational_blockers = sorted(
        b for b in blockers if b in FOUNDATIONAL_BLOCKER_CODES
    )
    if not blockers:
        status = "ready_for_non_release_live_adapter_probe"
    elif foundational_blockers:
        status = "blocked_missing_sumo_runtime_or_source_locks"
    else:
        # Runtime + source locks + live execution/evidence are satisfied; only
        # downstream release-packaging gates (behavioral / materializer) remain.
        status = "blocked_pending_release_packaging_gates"
    # Reflect the actual live-adapter rung honestly: once the recorded probe has
    # executed the live SUMO backend (sub-report rung `executed_with_live_backend`),
    # the carrier is at the executed-with-live-backend *probe* rung, not mock-only.
    source_integration_rung = (
        "executed_with_live_backend_probe"
        if live_adapter_probe.get("source_integration_rung")
        == "executed_with_live_backend"
        else "adapted_from_mock_only"
    )
    return {
        "schema_version": "0.1",
        "scope": REPORT_SCOPE,
        "non_release_artifact": True,
        "release_ready": False,
        "release_reentry_ready": False,
        "proceed_commands": [],
        "status": status,
        "source_integration_rung": source_integration_rung,
        "foundational_blocker_codes": foundational_blockers,
        "live_decision_headroom_proven": live_headroom_proven,
        "runtime_preflight": runtime,
        "sumo_source_tree": source_tree,
        "seed_anchor_preflight": seed_anchor_preflight,
        "seed_anchor_summary": seed_summary,
        "source_delivery_contract": source_delivery_contract,
        "live_adapter_probe": live_adapter_probe,
        "behavioral_gate": {
            "scope": "traffic_behavioral_gate",
            "recorded": behavioral_gate_passed,
            "gate_passed": behavioral_gate_passed,
            "released_scope_source_id": (recorded_behavioral_gate or {}).get(
                "released_scope_source_id"
            ),
            "n_cells": behavioral_gate_summary.get("n_cells"),
            "cells_gate_passed": behavioral_gate_summary.get("cells_gate_passed"),
            "all_signatures_unique": behavioral_gate_summary.get(
                "all_signatures_unique"
            ),
            "generate_command": BEHAVIORAL_GATE_GENERATE_COMMAND,
        },
        "release_materializer": {
            "scope": "traffic_release_materializer_draft",
            "recorded": materializer_draft_ready,
            "draft_ready": materializer_draft_ready,
            "released_scope_source_id": (recorded_materializer_draft or {}).get(
                "released_scope_source_id"
            ),
            "n_registry_rows": materializer_draft_summary.get("n_registry_rows"),
            "n_primary_rows": materializer_draft_summary.get("n_primary_rows"),
            "n_core_rows": materializer_draft_summary.get("n_core_rows"),
            "n_diagnostic_excluded_rows": materializer_draft_summary.get(
                "n_diagnostic_excluded_rows"
            ),
            "advances_blocker_to": (
                "real_release_materializer_not_run_into_release_wrapper"
                if materializer_draft_ready and written_draft_behavioral_audit_passed
                else "release_materializer_draft_not_audited_into_release"
                if materializer_draft_ready
                else "missing_release_materializer"
            ),
            "generate_command": MATERIALIZER_DRAFT_GENERATE_COMMAND,
        },
        "written_draft_behavioral_audit": {
            "scope": "traffic_written_draft_behavioral_audit",
            "recorded": written_draft_behavioral_audit_passed,
            "audited_on_written_artifacts": written_draft_behavioral_audit_passed,
            "released_scope_source_id": (recorded_written_draft_audit or {}).get(
                "released_scope_source_id"
            ),
            "backend": (recorded_written_draft_audit or {}).get("backend"),
            "audited_rows": (recorded_written_draft_audit or {}).get("audited_rows"),
            "behavioral_checks": (recorded_written_draft_audit or {}).get(
                "behavioral_checks"
            ),
            "advances_blocker_to": (
                "real_release_materializer_not_run_into_release_wrapper"
                if written_draft_behavioral_audit_passed
                else "release_materializer_draft_not_audited_into_release"
            ),
            "generate_command": WRITTEN_DRAFT_BEHAVIORAL_AUDIT_GENERATE_COMMAND,
        },
        "live_headroom_probe": {
            "scope": "sumo_live_headroom_probe",
            "recorded": live_headroom_proven,
            "live_decision_headroom_proven": live_headroom_proven,
            "binding_net_sha256": (recorded_live_headroom or {}).get(
                "binding_net_sha256"
            ),
            "generated_at_utc": (recorded_live_headroom or {}).get("generated_at_utc"),
            "headroom_metric": (recorded_live_headroom or {}).get("headroom_metric"),
            "headroom_l1_minutes": (recorded_live_headroom or {}).get(
                "headroom_l1_minutes"
            ),
            "n_corridors_changed": (recorded_live_headroom or {}).get(
                "n_corridors_changed"
            ),
            "max_abs_corridor_delta_minutes": (recorded_live_headroom or {}).get(
                "max_abs_corridor_delta_minutes"
            ),
            "metric_deterministic_replay_passed": (recorded_live_headroom or {}).get(
                "metric_deterministic_replay_passed"
            ),
            "evidence_wiring_probe_passed": (recorded_live_headroom or {}).get(
                "evidence_wiring_probe_passed"
            ),
            # The aggregate-delay sign is recorded for transparency only; it is
            # never a gate (a blanket relief switch is not the live optimum).
            "acting_reduces_aggregate_delay": (recorded_live_headroom or {}).get(
                "acting_reduces_aggregate_delay"
            ),
            "generate_command": LIVE_HEADROOM_GENERATE_COMMAND,
        },
        "release_blocker_codes": blockers,
        "safe_commands_now": list(SAFE_COMMANDS_NOW),
        "next_required_proof": (
            "Verify SUMO-RL package/license, SUMO runtime version, source-locked "
            "network and route files, and a non-release live adapter tool/evidence "
            "probe before any Traffic release packaging."
        ),
        "input_fingerprints": _input_fingerprints(
            seed_anchor_preflight,
            source_root=source_root,
        ),
        "policy": {
            "no_install": True,
            "no_download": True,
            "no_sumo_launch": True,
            "release_artifacts_modified": False,
            "traffic_remains_dev_only": True,
            "live_adapter_probe_recorded": recorded_live_probe is not None,
            "live_headroom_probe_recorded": recorded_live_headroom is not None,
            "materializer_draft_recorded": materializer_draft_ready,
            "written_draft_behavioral_audit_recorded": (
                written_draft_behavioral_audit_passed
            ),
        },
    }


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def exit_code_for_report(report: dict[str, Any]) -> int:
    return 0 if report.get("status") else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SUMO_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--live-probe-output", type=Path, default=DEFAULT_LIVE_PROBE_OUTPUT
    )
    parser.add_argument(
        "--run-live-probe",
        action="store_true",
        help=(
            "Opt-in: launch SUMO (requires OPERATE_TRAFFIC_BACKEND_REAL=1 + a "
            "reachable transport), run the live change_signal_plan adapter "
            "probe, and write the net-sha-locked probe artifact. The default "
            "report build never launches SUMO; it only ingests this artifact."
        ),
    )
    parser.add_argument(
        "--live-headroom-output", type=Path, default=DEFAULT_LIVE_HEADROOM_OUTPUT
    )
    parser.add_argument(
        "--run-live-headroom-probe",
        action="store_true",
        help=(
            "Opt-in: launch SUMO (requires OPERATE_TRAFFIC_BACKEND_REAL=1 + a "
            "reachable transport), run the live decision-headroom probe "
            "(wait vs acting signal-plan causal delta), and write the "
            "net-sha-locked headroom artifact. The default report build never "
            "launches SUMO."
        ),
    )
    parser.add_argument(
        "--written-draft-behavioral-audit-output",
        type=Path,
        default=DEFAULT_WRITTEN_DRAFT_BEHAVIORAL_AUDIT_OUTPUT,
        help=(
            "Path to the recorded written-draft behavioral audit artifact "
            "(produced by scripts/traffic_written_draft_behavioral_audit.py). "
            "This build only ingests it read-only."
        ),
    )
    args = parser.parse_args(argv)

    if args.run_live_probe:
        probe = run_live_adapter_probe()
        write_report(probe, args.live_probe_output)
        print(json.dumps(probe, indent=2, sort_keys=True))

    if args.run_live_headroom_probe:
        headroom = run_live_headroom_probe()
        write_report(headroom, args.live_headroom_output)
        print(json.dumps(headroom, indent=2, sort_keys=True))

    report = build_sumo_traffic_source_preflight_report(
        source_root=args.source_root,
        live_probe_path=args.live_probe_output,
        live_headroom_path=args.live_headroom_output,
        written_draft_behavioral_audit_path=(
            args.written_draft_behavioral_audit_output
        ),
    )
    write_report(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code_for_report(report)


if __name__ == "__main__":
    raise SystemExit(main())
