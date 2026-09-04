#!/usr/bin/env python3
"""Exhaustively refine local infrastructure source units without admitting them.

The output is a terminal candidate ledger.  It deliberately separates local
raw source units from independent candidate decisions, attempts repairable
environment/source/control-axis fixes before disposition, and never mutates a
release or treats provider-LLM output as admission evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from contextlib import suppress
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.opendss_candidate_inventory import (  # noqa: E402
    discover_opendss_entrypoints,
)
from core.protocol21_evidence import canonicalize_repo_owned_paths  # noqa: E402
from scripts.mine_sumo365_native_traffic import (  # noqa: E402
    _demand_change_window,
    parse_route_departures,
)

DEFAULT_SUITE = ROOT / "release/operate_v0_61_0/protocol21_source_suite.json"
DEFAULT_OUTPUT = (
    ROOT / ".hl/artifacts/operate_v058_infrastructure_candidate_refinement.json"
)
FINAL_DISPOSITIONS = {"ready_for_full_admission", "held_repair", "redesign", "secondary", "rejected"}
ARCHIVED_NGSIM_CANDIDATES = (
    "ngsim:1113433176100:a1a032494881bc29",
    "ngsim:1113433267500:2526d6e599205e7a",
    "ngsim:1118846999200:e08996e3fddf68d4",
    "ngsim:1118847062700:364428e7e12e2fee",
    "ngsim:1118847070800:abc2ea840a747b78",
    "ngsim:1118847132300:fc9b160cb3ccb957",
    "ngsim:1118847187100:3b6793cb928cf7fd",
    "ngsim:1118847260700:5aea66c4b9a5ba06",
    "ngsim:1118847360400:99e4d9e9718737e1",
    "ngsim:1118847482500:626fc1b70a91943d",
    "ngsim:1118847551400:ccdc6d3703d5ad43",
    "ngsim:1118847616700:5cf1d4d7a4c571a4",
    "ngsim:1118847677100:adc3ed02f831ff5e",
    "ngsim:1163040000:f0916a903b071474",
    "ngsim:1163335200:b70b5e2d16d97895",
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _graph_sha(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix().lower()):
        digest.update(path.as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _active_tokens(payload: Any) -> set[str]:
    tokens: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, str):
            tokens.add(value)
            tokens.update(re.findall(r"[A-Za-z0-9_.-]+", value))
            if value.startswith("{"):
                try:
                    visit(json.loads(value))
                except ValueError:
                    pass
        elif isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(payload)
    return tokens


def _attempt(code: str, status: str, detail: str) -> dict[str, str]:
    phase = "proposed" if status == "design_specified" else "executed"
    return {"code": code, "phase": phase, "status": status, "detail": detail}


def _row(
    *,
    source_family: str,
    source_unit: str,
    domain: str,
    final_disposition: str,
    reason_codes: list[str],
    repair_attempts: list[dict[str, str]],
    evidence: dict[str, Any],
    source_id: str | None = None,
    candidate_id: str | None = None,
) -> dict[str, Any]:
    if final_disposition not in FINAL_DISPOSITIONS:
        raise ValueError(f"invalid disposition: {final_disposition}")
    identity = candidate_id or (
        f"infrastructure/{source_family}/{_slug(source_unit) or 'source-unavailable'}"
    )
    return {
        "candidate_id": identity,
        "source_id": source_id or source_family,
        "source_family": source_family,
        "source_unit": source_unit,
        "domain": domain,
        "classification_scope": "candidate",
        "entity_kind": "canonical_source_unit",
        "final_disposition": final_disposition,
        "reason_codes": sorted(set(reason_codes)),
        "repair_attempts": repair_attempts,
        "evidence": evidence,
    }


def _missing(source_family: str, domain: str, path: Path | None) -> dict[str, Any]:
    return _row(
        source_family=source_family,
        source_unit="source_unavailable",
        domain=domain,
        final_disposition="held_repair",
        reason_codes=["external_source_unavailable"],
        repair_attempts=[
            _attempt(
                "local_source_discovery",
                "blocked_external",
                f"No local source was found at {path}",
            )
        ],
        evidence={"expected_path": None if path is None else str(path)},
    )


def _present_external(source_family: str, domain: str, path: Path) -> dict[str, Any]:
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    runtime_probe: dict[str, Any] | None = None
    source_unit = path.name
    if source_family == "grid2op_cache":
        dataset = path / "rte_case14_realistic"
        source_unit = dataset.name
        try:
            import grid2op

            env = grid2op.make(str(dataset))
            try:
                observation = env.reset()
                runtime_probe = {
                    "status": "passed",
                    "dataset": str(dataset),
                    "n_substations": int(env.n_sub),
                    "n_lines": int(env.n_line),
                    "n_generators": int(env.n_gen),
                    "observation_type": type(observation).__name__,
                    "chronics_path": str(env.chronics_handler.path),
                }
            finally:
                env.close()
        except Exception as exc:
            runtime_probe = {
                "status": "failed",
                "dataset": str(dataset),
                "error": f"{type(exc).__name__}: {exc}",
            }
    return _row(
        source_family=source_family,
        source_unit=source_unit,
        domain=domain,
        final_disposition="redesign",
        reason_codes=["source_present_executable_adapter_not_bound"],
        repair_attempts=[
            _attempt(
                "local_source_discovery",
                "passed",
                f"Canonical source root is locally available at {path}.",
            ),
            _attempt(
                "executable_backend_contract",
                "design_specified",
                "Retain as source/method-transfer evidence until a domain-native adapter, "
                "typed task contract and action-effect replay consume these exact values.",
            ),
        ],
        evidence={
            "source_root": str(path),
            "n_files": len(files),
            "representative_files": [str(candidate) for candidate in files[:25]],
            **({"bounded_runtime_probe": runtime_probe} if runtime_probe else {}),
        },
    )


def _pglib_opf(root: Path, active: set[str]) -> list[dict[str, Any]]:
    if not root.is_dir():
        return [_missing("pglib_opf", "power_grid", root)]
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("pglib_opf_case*.m")):
        unit = path.stem
        match = re.search(r"case(\d+)", unit)
        buses = int(match.group(1)) if match else 0
        source_text = path.read_text(encoding="utf-8", errors="ignore")
        native_matrices = all(
            f"mpc.{name}" in source_text for name in ("bus", "gen", "branch", "gencost")
        )
        attempts = [
            _attempt(
                "remove_unrelated_timeseries_requirement",
                "passed",
                "Static OPF source values define base topology and limits; typed seeded "
                "cross-tick reserve/ramp/recovery pressure is an allowed procedural stress.",
            ),
            _attempt(
                "source_native_control_axis",
                "passed",
                "Bind redispatch, reserve commitment, load shedding, generator limits and "
                "ramp/recovery ordering to the MATPOWER case.",
            ),
            _attempt(
                "matpower_native_matrix_preflight",
                "passed" if native_matrices else "failed",
                "Require source-native bus, generator, branch and generator-cost matrices; "
                "topology size is not an admission quota.",
            ),
        ]
        if unit in active:
            disposition = "secondary"
            reasons = ["physical_source_already_in_core"]
        elif native_matrices:
            disposition = "redesign"
            reasons = [
                "independent_topology",
                "source_bound_temporal_task_reference_contract_missing",
            ]
            attempts.append(
                _attempt(
                    "executable_temporal_task_materializer",
                    "design_specified",
                    "Bind a non-fabricated temporal axis, ordered native dispatch task and "
                    "bounded reference policy before candidate materialization.",
                )
            )
            if buses > 500:
                reasons.append("large_sparse_topology_hard_axis")
        else:
            disposition = "rejected"
            reasons = ["matpower_control_matrices_missing"]
        rows.append(
            _row(
                source_family="pglib_opf",
                source_unit=unit,
                domain="power_grid",
                final_disposition=disposition,
                reason_codes=reasons,
                repair_attempts=attempts,
                evidence={"path": str(path), "sha256": _sha256(path), "buses": buses},
            )
        )
    return rows or [_missing("pglib_opf", "power_grid", root)]


def _pglib_uc(root: Path, active: set[str]) -> list[dict[str, Any]]:
    if not root.is_dir():
        return [_missing("pglib_uc", "power_grid", root)]
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/*.json")):
        relative = path.relative_to(root).as_posix()
        attempts = [
            _attempt(
                "native_multiperiod_contract",
                "passed",
                "Demand, reserves, renewable availability and generator constraints remain "
                "source-native across the full horizon.",
            )
        ]
        try:
            payload = _load(path)
            required = {
                "time_periods",
                "demand",
                "reserves",
                "thermal_generators",
                "renewable_generators",
            }
            valid = isinstance(payload, dict) and required.issubset(payload)
            periods = int(payload.get("time_periods") or 0) if valid else 0
            valid = valid and periods > 1
        except (OSError, ValueError, TypeError):
            valid = False
            periods = 0
        if relative in active or path.name in active:
            disposition = "secondary"
            reasons = ["physical_source_already_in_core"]
        elif valid:
            disposition = "ready_for_full_admission"
            reasons = [
                "source_native_multiperiod_case",
                "bounded_delta_replay_required",
            ]
        else:
            disposition = "rejected"
            reasons = ["invalid_or_non_multiperiod_source_case"]
        rows.append(
            _row(
                source_family="pglib_uc",
                source_unit=relative,
                domain="power_grid",
                final_disposition=disposition,
                reason_codes=reasons,
                repair_attempts=attempts,
                evidence={"path": str(path), "sha256": _sha256(path), "time_periods": periods},
            )
        )
    return rows or [_missing("pglib_uc", "power_grid", root)]


def _referenced_local_files(value: Any, root: Path, output: set[Path]) -> None:
    if isinstance(value, dict):
        for nested in value.values():
            _referenced_local_files(nested, root, output)
    elif isinstance(value, list):
        for nested in value:
            _referenced_local_files(nested, root, output)
    elif isinstance(value, str) and Path(value).suffix.lower() in {".csv", ".json", ".pth"}:
        output.add(root / value)


def _citylearn_native_probe(schema_path: Path) -> dict[str, Any]:
    try:
        from citylearn.citylearn import CityLearnEnv

        env = CityLearnEnv(schema=str(schema_path), central_agent=True)
        env.reset()
        action_names = [
            str(name)
            for group in (env.action_names or [])
            for name in (group if isinstance(group, list) else [group])
        ]
        actions = [
            [0.0] * int(space.shape[0])
            for space in env.action_space
        ]
        env.step(actions)
        close = getattr(env, "close", None)
        if callable(close):
            close()
        return {
            "status": "passed",
            "n_buildings": len(env.buildings),
            "action_names": action_names,
            "native_step_executed": True,
        }
    except Exception as exc:  # candidate probe must terminally record runtime failures
        return {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:1000],
            "native_step_executed": False,
        }


def _citylearn(
    root: Path,
    active: set[str],
    *,
    execute_native_probes: bool,
) -> list[dict[str, Any]]:
    if not root.is_dir():
        return [_missing("citylearn", "building_energy", root)]
    rows: list[dict[str, Any]] = []
    for schema_path in sorted(root.glob("*/schema.json")):
        dataset = schema_path.parent
        unit = dataset.name
        attempts: list[dict[str, str]] = []
        try:
            schema = _load(schema_path)
        except (OSError, ValueError):
            rows.append(
                _row(
                    source_family="citylearn",
                    source_unit=unit,
                    domain="building_energy",
                    final_disposition="rejected",
                    reason_codes=["invalid_schema"],
                    repair_attempts=[_attempt("schema_parse", "failed", "schema.json is invalid")],
                    evidence={"path": str(schema_path)},
                )
            )
            continue
        referenced = {schema_path}
        _referenced_local_files(schema.get("buildings") or {}, dataset, referenced)
        missing = sorted(str(path.relative_to(dataset)) for path in referenced if not path.is_file())
        actions = schema.get("actions") if isinstance(schema, dict) else {}
        electrical = (actions or {}).get("electrical_storage") or {}
        buildings = [
            body
            for body in (schema.get("buildings") or {}).values()
            if isinstance(body, dict) and body.get("include", True)
        ]
        storage_active = bool(electrical.get("active")) and any(
            "electrical_storage" not in (building.get("inactive_actions") or [])
            for building in buildings
        )
        carbon_declared = any(building.get("carbon_intensity") for building in buildings)
        carbon_observed = ((schema.get("observations") or {}).get("carbon_intensity") or {}).get(
            "active"
        )
        optional_carbon = not carbon_declared and not carbon_observed
        attempts.append(
            _attempt(
                "optional_carbon_channel",
                "passed" if optional_carbon or carbon_declared else "not_applicable",
                "An absent carbon channel is marked inapplicable when the source schema does "
                "not declare it; it is not replaced with pricing or synthetic values.",
            )
        )
        attempts.append(
            _attempt(
                "native_storage_control_surface",
                "passed" if storage_active else "design_specified",
                "Use native electrical storage actions; otherwise add a domain-native action "
                "adapter for the dataset's declared controllable assets.",
            )
        )
        native_probe = (
            _citylearn_native_probe(schema_path)
            if execute_native_probes and not missing
            else {"status": "not_run"}
        )
        native_actions = list(native_probe.get("action_names") or [])
        native_control_available = bool(native_actions)
        attempts.append(
            _attempt(
                "citylearn_native_reset_step",
                str(native_probe["status"]),
                "Construct the source schema, reset the native environment and execute one "
                "zero-action step to distinguish data quality from runtime compatibility.",
            )
        )
        if unit in active:
            disposition = "secondary"
            reasons = ["dataset_already_contributes_core_windows"]
        elif missing:
            disposition = "held_repair"
            reasons = ["referenced_source_assets_unavailable"]
        elif execute_native_probes and native_probe["status"] != "passed":
            disposition = "held_repair"
            reasons = ["citylearn_runtime_compatibility_repair_required"]
            attempts.append(
                _attempt(
                    "runtime_dependency_alignment",
                    "held_repair",
                    "Align the runtime that deserializes source-native model artifacts; package "
                    "compatibility does not reject the source task.",
                )
            )
        elif (
            execute_native_probes
            and native_control_available
            and storage_active
            and "electrical_storage" in native_actions
        ):
            disposition = "ready_for_full_admission"
            reasons = [
                "native_citylearn_control_axis",
                "bounded_delta_replay_required",
            ]
        elif not storage_active:
            disposition = "redesign"
            reasons = ["electrical_storage_control_axis_absent"]
        elif not execute_native_probes:
            disposition = "ready_for_full_admission"
            reasons = ["independent_building_graph", "bounded_delta_replay_required"]
        else:
            disposition = "redesign"
            reasons = ["benchmark_native_electrical_storage_axis_unavailable"]
        present = [path for path in referenced if path.is_file()]
        rows.append(
            _row(
                source_family="citylearn",
                source_unit=unit,
                domain="building_energy",
                final_disposition=disposition,
                reason_codes=reasons,
                repair_attempts=attempts,
                evidence={
                    "schema_path": str(schema_path),
                    "asset_graph_sha256": _graph_sha(present),
                    "n_buildings": len(buildings),
                    "missing_assets": missing,
                    "optional_carbon": optional_carbon,
                    "native_probe": native_probe,
                },
            )
        )
    return rows or [_missing("citylearn", "building_energy", root)]


def _sumo_references(config: Path) -> tuple[list[Path], list[str], list[str]]:
    try:
        tree = ET.parse(config)
    except (ET.ParseError, OSError):
        return [], ["invalid_sumocfg"], []
    paths: list[Path] = [config]
    missing: list[str] = []
    for tag in ("net-file", "route-files", "additional-files"):
        for node in tree.findall(f".//{tag}"):
            for value in str(node.attrib.get("value") or "").split(","):
                if not value.strip():
                    continue
                path = (config.parent / value.strip()).resolve()
                if path.is_file():
                    paths.append(path)
                else:
                    missing.append(value.strip())
    archive_members: list[str] = []
    unresolved: list[str] = []
    archives = sorted(config.parent.glob("*.zip"))
    for value in sorted(set(missing)):
        resolved = False
        for archive in archives:
            try:
                with zipfile.ZipFile(archive) as bundle:
                    members = bundle.namelist()
            except (OSError, zipfile.BadZipFile):
                continue
            matches = [name for name in members if Path(name).name == Path(value).name]
            if len(matches) == 1:
                paths.append(archive)
                archive_members.append(f"{archive.name}!{matches[0]}")
                resolved = True
                break
        if not resolved:
            unresolved.append(value)
    return paths, unresolved, archive_members


def _sumo_declared_route_paths(config: Path) -> list[Path]:
    """Return only demand inputs declared by ``route-files``.

    Additional files can themselves contain public-transport routes.  They are
    part of the executable graph but are not the candidate's primary demand
    channel, so counting every ``*.rou.xml`` in the closure makes a valid
    single-demand configuration appear ambiguous.
    """

    tree = ET.parse(config)
    paths: list[Path] = []
    for node in tree.findall(".//route-files"):
        for value in str(node.attrib.get("value") or "").split(","):
            if value.strip():
                paths.append((config.parent / value.strip()).resolve())
    return [path for path in paths if path.is_file()]


def _sumo_native_probe(
    config: Path,
    graph: list[Path],
    archive_members: list[str],
) -> dict[str, Any]:
    binary = ROOT / ".venv/bin/sumo"
    if not binary.is_file():
        discovered = shutil.which("sumo")
        if discovered is None:
            return {"status": "runtime_unavailable", "error": "sumo binary missing"}
        binary = Path(discovered)
    try:
        with tempfile.TemporaryDirectory(prefix="operate-sumo-probe-") as temporary:
            probe_root = Path(temporary)
            for path in graph:
                if path.suffix.lower() == ".zip":
                    continue
                try:
                    relative = path.resolve().relative_to(config.parent.resolve())
                except ValueError:
                    relative = Path(path.name)
                target = probe_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
            for binding in archive_members:
                archive_name, member = binding.split("!", 1)
                archive = config.parent / archive_name
                with zipfile.ZipFile(archive) as bundle:
                    (probe_root / Path(member).name).write_bytes(bundle.read(member))
            command = [
                str(binary),
                "-c",
                str(probe_root / config.name),
                "--begin",
                "0",
                "--end",
                "1",
                "--no-step-log",
                "true",
                "--duration-log.disable",
                "true",
                "--xml-validation",
                "never",
            ]
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        return {
            "status": "passed" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "stderr": completed.stderr[-2000:],
            "stdout": completed.stdout[-1000:],
            "command": [binary.name, "-c", config.name, *command[3:]],
        }
    except (OSError, subprocess.SubprocessError, zipfile.BadZipFile) as exc:
        return {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:1000],
        }


def _sumo_family(
    root: Path,
    *,
    family: str,
    active: set[str],
    glob: str,
    execute_native_probes: bool,
) -> list[dict[str, Any]]:
    if not root.is_dir():
        return [_missing(family, "traffic", root)]
    rows: list[dict[str, Any]] = []
    for config in sorted(root.glob(glob)):
        unit = config.parent.name if family == "resco" else config.stem
        graph, missing, archive_members = _sumo_references(config)
        attempts = [
            _attempt(
                "sumocfg_reference_closure",
                "passed" if not missing else "design_specified",
                "Resolve the native network, route and additional files relative to the "
                "SUMO configuration without fabricating vehicle demand.",
            )
        ]
        if archive_members:
            attempts.append(
                _attempt(
                    "archive_member_runtime_materialization",
                    "passed",
                    "The referenced route member exists exactly once in the source archive and "
                    "can be deterministically extracted before native replay.",
                )
            )
        native_probe = (
            _sumo_native_probe(config, graph, archive_members)
            if execute_native_probes and not missing
            else {"status": "not_run"}
        )
        attempts.append(
            _attempt(
                "sumo_native_one_step",
                str(native_probe["status"]),
                "Load the exact SUMO configuration graph and execute a bounded native step.",
            )
        )
        try:
            route_paths = _sumo_declared_route_paths(config)
        except (OSError, ET.ParseError):
            route_paths = []
        source_transition: dict[str, Any]
        if len(route_paths) == 1:
            try:
                source_transition = _demand_change_window(
                    parse_route_departures(route_paths[0])
                )
            except (OSError, ET.ParseError, ValueError) as exc:
                source_transition = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        else:
            source_transition = {
                "status": "failed",
                "error": "exactly one material route file is required",
            }
        attempts.append(
            _attempt(
                "source_route_departure_transition",
                str(source_transition.get("status") or "failed"),
                "Identify one source-native departure-rate transition that can be "
                "registered as a typed task event without adding synthetic demand.",
            )
        )
        if unit in active:
            disposition = "secondary"
            reasons = ["traffic_network_or_service_date_already_in_core"]
        elif missing:
            disposition = "redesign"
            reasons = ["sumocfg_runtime_graph_incomplete"]
        elif execute_native_probes and native_probe["status"] == "runtime_unavailable":
            disposition = "held_repair"
            reasons = ["sumo_runtime_environment_repair_required"]
        elif execute_native_probes and native_probe["status"] != "passed":
            disposition = "redesign"
            reasons = ["sumo_native_configuration_probe_failed"]
        elif source_transition.get("status") == "passed":
            disposition = "ready_for_full_admission"
            reasons = [
                "source_transition_causal_response_binding_available",
                "bounded_delta_replay_required",
            ]
            attempts.append(
                _attempt(
                    "executable_source_window_event_and_tls_effect_binding",
                    "passed",
                    "Require the selected material route-departure transition to emit a "
                    "typed actionable runtime event and bind the later native TLS receipt "
                    "to that exact event before candidate replay.",
                )
            )
        else:
            disposition = "redesign"
            reasons = ["source_native_material_transition_unavailable"]
        rows.append(
            _row(
                source_family=family,
                source_unit=unit,
                domain="traffic",
                final_disposition=disposition,
                reason_codes=reasons,
                repair_attempts=attempts,
                evidence={
                    "config_path": str(config),
                    "asset_graph_sha256": _graph_sha(graph),
                    "missing_assets": missing,
                    "archive_members": archive_members,
                    "native_probe": native_probe,
                    "source_transition": source_transition,
                },
            )
        )
    return rows or [_missing(family, "traffic", root)]


def _nrel(root: Path, active: set[str]) -> list[dict[str, Any]]:
    if not root.is_dir():
        return [_missing("nrel_microgrid", "microgrid", root)]
    rows: list[dict[str, Any]] = []
    for source in sorted(root.glob("*.npz")):
        unit = source.stem
        provenance = source.with_suffix(".provenance.json")
        attempts = [
            _attempt(
                "profile_provenance_pair",
                "passed" if provenance.is_file() else "blocked_external",
                "Bind each site profile to its source-native provenance sidecar.",
            )
        ]
        if unit in active or source.name in active:
            disposition = "secondary"
            reasons = ["site_profile_already_contributes_core"]
        elif not provenance.is_file():
            disposition = "held_repair"
            reasons = ["source_provenance_sidecar_unavailable"]
        else:
            disposition = "redesign"
            reasons = [
                "independent_site_profile",
                "typed_task_reference_materializer_missing",
            ]
            attempts.append(
                _attempt(
                    "executable_microgrid_task_materializer",
                    "design_specified",
                    "Bind a source-native event window, ordered controls and reference policy "
                    "through the released microgrid backend before materialization.",
                )
            )
        rows.append(
            _row(
                source_family="nrel_microgrid",
                source_unit=unit,
                domain="microgrid",
                final_disposition=disposition,
                reason_codes=reasons,
                repair_attempts=attempts,
                evidence={
                    "profile_path": str(source),
                    "profile_sha256": _sha256(source),
                    "provenance_path": str(provenance),
                    "provenance_sha256": _sha256(provenance) if provenance.is_file() else None,
                },
            )
        )
    return rows or [_missing("nrel_microgrid", "microgrid", root)]


def _rts(root: Path, active: set[str]) -> list[dict[str, Any]]:
    if not root.is_dir():
        return []
    relative = (
        "RTS_Data/FormattedData/pandapower/pandapower_net.json",
        "RTS_Data/timeseries_data_files/Load/REAL_TIME_regional_Load.csv",
        "RTS_Data/timeseries_data_files/Load/DAY_AHEAD_regional_Load.csv",
        "RTS_Data/timeseries_data_files/PV/REAL_TIME_pv.csv",
        "RTS_Data/timeseries_data_files/WIND/REAL_TIME_wind.csv",
        "RTS_Data/SourceData/reserves.csv",
    )
    paths = [root / name for name in relative]
    missing = [name for name, path in zip(relative, paths, strict=True) if not path.is_file()]
    active_source = "rts_gmlc" in active
    return [
        _row(
            source_family="rts_gmlc",
            source_unit="coherent_grid_forecast_reserve_graph",
            domain="power_grid",
            final_disposition=(
                "secondary" if active_source else "held_repair" if missing else "redesign"
            ),
            reason_codes=(
                ["coherent_source_graph_already_in_core"]
                if active_source
                else ["coherent_source_graph_incomplete"]
                if missing
                else [
                    "coherent_topology_and_timeseries",
                    "exact_graph_loader_task_reference_contract_missing",
                ]
            ),
            repair_attempts=[
                _attempt(
                    "coherent_graph_binding",
                    "blocked_external" if missing else "passed",
                    "Bind RTS topology, real-time/day-ahead load, PV, wind and reserve files "
                    "as one source graph.",
                ),
                _attempt(
                    "executable_rts_graph_task_materializer",
                    "design_specified",
                    "Consume the coherent topology/forecast/reserve graph in one backend and "
                    "prove ordered native action effects before materialization.",
                ),
            ],
            evidence={
                "asset_graph_sha256": _graph_sha([path for path in paths if path.is_file()]),
                "missing_assets": missing,
                "paths": [str(path) for path in paths],
            },
        )
    ]


_DSS_INCLUDE = re.compile(r"^\s*(?:redirect|compile)\s*[\[=(]?\s*['\"]?([^)'\"\]\s]+)", re.I)
_OPENDSS_TEMP_ROOT = re.compile(
    r'(?:/[^/\s"\]]+)+/operate-opendss-[^/\s"\]]+'
)


def _dss_graph(root: Path, entry: Path) -> tuple[list[Path], list[str]]:
    visited: set[Path] = set()
    missing: set[str] = set()

    def visit(path: Path) -> None:
        path = path.resolve()
        if path in visited or not path.is_file():
            return
        visited.add(path)
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            match = _DSS_INCLUDE.match(line)
            if not match:
                continue
            nested = (path.parent / match.group(1)).resolve()
            if nested.is_file() and nested.is_relative_to(root.resolve()):
                visit(nested)
            else:
                missing.add(match.group(1))

    visit(entry)
    return sorted(visited), sorted(missing)


def _sanitize_probe_error(error: str, runtime_root: Path | None) -> str:
    if runtime_root is not None:
        prefixes = {str(runtime_root), str(runtime_root.resolve())}
        for prefix in sorted(prefixes, key=len, reverse=True):
            error = error.replace(prefix, "<opendss_runtime>")
    return _OPENDSS_TEMP_ROOT.sub("<opendss_runtime>", error)


def _opendss_native_probe_direct(root: Path, entry: Path) -> dict[str, Any]:
    backend: Any = None
    runtime_root: Path | None = None
    try:
        from domains.power_grid.backends.opendss_fresh_feeders import (
            OpenDssFreshFeederProbeBackend,
        )

        backend = OpenDssFreshFeederProbeBackend(
            source_root=root,
            feeder=f"candidate-{_slug(entry.relative_to(root).as_posix())}",
            master_file=entry.relative_to(root).as_posix(),
        )
        summary = backend.reset()
        snapshot = backend.snapshot()
        controls = {
            "capacitors": len(snapshot.get("capacitors") or []),
            "regcontrols": len(snapshot.get("regcontrols") or []),
            "switchable_lines": len(snapshot.get("lines") or []),
        }
        return {
            "status": (
                "passed" if summary.converged and any(controls.values()) else "failed"
            ),
            "summary": summary.to_dict(),
            "native_controls": controls,
            "source_trace_status": backend.protocol21_source_trace().get("status"),
        }
    except Exception as exc:  # candidate probe must preserve exact failure evidence
        runtime_directory = getattr(backend, "_runtime_directory", None)
        if runtime_directory is not None:
            runtime_root = Path(runtime_directory.name)
        return {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": _sanitize_probe_error(str(exc), runtime_root)[:2000],
        }
    finally:
        if backend is not None:
            with suppress(Exception):
                backend.close()


def _opendss_native_probe(root: Path, entry: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-X",
        "faulthandler",
        str(Path(__file__).resolve()),
        "--_opendss-native-probe-root",
        str(root.resolve()),
        "--_opendss-native-probe-entry",
        str(entry.resolve()),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "failed",
            "error_type": "NativeProbeTimeout",
            "error": f"OpenDSS candidate probe exceeded {exc.timeout} seconds",
        }
    if completed.returncode != 0:
        return {
            "status": "failed",
            "error_type": "NativeProbeProcessExit",
            "return_code": completed.returncode,
            "error": completed.stderr[-2000:],
        }
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        return {
            "status": "failed",
            "error_type": "NativeProbeProtocolError",
            "error": "OpenDSS candidate probe returned no JSON payload",
        }
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        return {
            "status": "failed",
            "error_type": "NativeProbeProtocolError",
            "error": f"invalid OpenDSS probe JSON: {exc}",
        }
    if not isinstance(payload, dict):
        return {
            "status": "failed",
            "error_type": "NativeProbeProtocolError",
            "error": "OpenDSS candidate probe payload must be an object",
        }
    return payload


def _opendss(
    root: Path,
    active: set[str],
    *,
    execute_native_probes: bool,
) -> list[dict[str, Any]]:
    if not root.is_dir():
        return [_missing("opendss_distribution", "power_grid", root)]
    resolved_root = root.resolve()
    entries = discover_opendss_entrypoints(resolved_root)
    rows: list[dict[str, Any]] = []
    seen_graphs: set[str] = set()
    for entry in entries:
        graph, missing = _dss_graph(resolved_root, entry)
        digest = _graph_sha(graph)
        unit = entry.relative_to(resolved_root).as_posix()
        active_source = unit in active or entry.name in active
        duplicate = digest in seen_graphs
        seen_graphs.add(digest)
        native_probe = (
            _opendss_native_probe(root, entry)
            if execute_native_probes and not missing and not duplicate and not active_source
            else {"status": "not_run"}
        )
        attempts = [
            _attempt(
                "coherent_dss_include_graph",
                "design_specified" if missing else "passed",
                "Canonicalize one feeder entry point and its recursive Redirect/Compile graph; "
                "never count copied files as independent cases.",
            ),
            _attempt(
                "explicit_master_feeder_adapter",
                "passed" if not missing else "blocked_source",
                "Bind an arbitrary entry point only through an explicit master file; this "
                "probe does not materialize release scenarios.",
            ),
            _attempt(
                "dss_python_compile_solve_control_probe",
                str(native_probe["status"]),
                "Compile and solve the exact include graph, then inventory native capacitor, "
                "regulator and switchable-line controls.",
            ),
        ]
        if active_source:
            disposition = "secondary"
            reasons = ["physical_source_already_in_core"]
        elif duplicate:
            disposition = "secondary"
            reasons = ["duplicate_coherent_feeder_graph"]
        elif native_probe["status"] == "passed":
            disposition = "ready_for_full_admission"
            reasons = [
                "native_compile_solve_control_axis",
                "bounded_delta_replay_required",
            ]
        else:
            disposition = "redesign"
            controls = native_probe.get("native_controls") or {}
            summary = native_probe.get("summary") or {}
            if missing:
                reasons = ["dss_include_graph_incomplete"]
            elif native_probe.get("error"):
                reasons = ["native_compile_solve_failed"]
            elif not summary.get("converged", False):
                reasons = ["native_solve_nonconverged"]
            elif not any(controls.values()):
                reasons = ["native_controllable_axis_absent"]
            else:
                reasons = ["native_probe_failed"]
        rows.append(
            _row(
                source_family="opendss_distribution",
                source_unit=unit,
                domain="power_grid",
                final_disposition=disposition,
                reason_codes=reasons,
                repair_attempts=attempts,
                evidence={
                    "entry_path": str(entry),
                    "asset_graph_sha256": digest,
                    "n_graph_files": len(graph),
                    "missing_includes": missing,
                    "native_probe": native_probe,
                    "repository_dss_file_count": len(list(root.rglob("*.dss")))
                    + len(list(root.rglob("*.DSS"))),
                },
            )
        )
    return rows or [_missing("opendss_distribution", "power_grid", root)]


def _metadata_candidates(
    path: Path | None,
    *,
    family: str,
    domain: str,
) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return [_missing(family, domain, path)]
    payload = _load(path)
    candidates = payload.get("candidates") if isinstance(payload, dict) else None
    rows: list[dict[str, Any]] = []
    for candidate in candidates or []:
        candidate_id = str(candidate.get("candidate_id") or "")
        if not candidate_id:
            continue
        rows.append(
            _row(
                source_family=family,
                source_unit=candidate_id,
                domain=domain,
                final_disposition="redesign",
                reason_codes=[
                    "locked_structural_candidate",
                    "metadata_only_executable_contract_unproven",
                ],
                repair_attempts=[
                    _attempt(
                        "source_contract_binding",
                        "passed",
                        "Use the candidate's locked physical/effective source identity without "
                        "package-version admission pins.",
                    ),
                    _attempt(
                        "executable_scenario_reset_and_native_effect",
                        "design_specified",
                        "A metadata row is not materializable until a complete strict scenario "
                        "resets and a native tool produces an audited state effect.",
                    ),
                ],
                evidence={"metadata_path": str(path), "metadata_sha256": _sha256(path)},
                candidate_id=candidate_id,
            )
        )
    return rows or [_missing(family, domain, path)]


def _datacenter(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    if not path.is_file():
        return [_missing("datacenter_spot_gpu", "datacenter", path)]
    payload = _load(path)
    candidates = payload.get("candidates") if isinstance(payload, dict) else []

    def pressure(candidate: dict[str, Any]) -> float:
        evidence = candidate.get("evidence") or {}
        return (
            float(evidence.get("gpu_demand_capacity_ratio") or 0.0)
            + math.log1p(float(evidence.get("duration_ratio") or 0.0))
            + float(evidence.get("arrival_epoch_count") or 0.0) / 10.0
            + float(evidence.get("organization_count") or 0.0) / 10.0
        )

    winners: dict[str, str] = {}
    for candidate in candidates or []:
        model = str((candidate.get("evidence") or {}).get("gpu_model") or "unknown")
        current = winners.get(model)
        if current is None:
            winners[model] = str(candidate.get("candidate_id") or "")
            continue
        prior = next(row for row in candidates if row.get("candidate_id") == current)
        if (pressure(candidate), str(candidate.get("candidate_id"))) > (
            pressure(prior),
            current,
        ):
            winners[model] = str(candidate.get("candidate_id") or "")
    rows = []
    for candidate in candidates or []:
        candidate_id = str(candidate.get("candidate_id") or "")
        if not candidate_id:
            continue
        evidence = dict(candidate.get("evidence") or {})
        model = str(evidence.get("gpu_model") or "unknown")
        winner = winners.get(model) == candidate_id
        rows.append(
            _row(
                source_family="datacenter_spot_gpu",
                source_unit=candidate_id,
                domain="datacenter",
                final_disposition="ready_for_full_admission" if winner else "secondary",
                reason_codes=(
                    ["hardest_independent_gpu_model_axis", "bounded_delta_replay_required"]
                    if winner
                    else ["same_gpu_model_lower_pressure_variant"]
                ),
                repair_attempts=[
                    _attempt(
                        "non_padding_hard_axis_selection",
                        "passed",
                        "Select the highest contention/duration/arrival pressure per GPU model; "
                        "retain other source windows as secondary variants.",
                    )
                ],
                evidence={
                    **evidence,
                    "pressure_score": pressure(candidate),
                    "source_window_sha256": candidate.get("source_window_sha256"),
                    "independent_decision_axes": candidate.get("independent_decision_axes") or [],
                },
                candidate_id=candidate_id,
            )
        )
    return rows or [_missing("datacenter_spot_gpu", "datacenter", path)]


def _autonomous(
    paths: list[Path], archived_candidates: tuple[str, ...]
) -> list[dict[str, Any]]:
    evidence_by_id: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            continue
        payload = _load(path)
        for candidate in payload.get("candidates") or []:
            candidate_id = str(candidate.get("candidate_id") or "")
            if candidate_id:
                evidence_by_id[candidate_id] = {**candidate, "evidence_path": str(path)}
    ids = sorted(set(archived_candidates) | set(evidence_by_id))
    if not ids:
        return [_missing("autonomous_driving_ngsim", "autonomous_driving", None)]
    rows = []
    for candidate_id in ids:
        evidence = evidence_by_id.get(candidate_id, {})
        file_fields = ("scenario_yaml", "calibration_path", "replay_path")
        files = [Path(str(evidence[field])) for field in file_fields if evidence.get(field)]
        files_present = len(files) == len(file_fields) and all(path.is_file() for path in files)
        runtime_verified = evidence.get("status") == "verified"
        rows.append(
            _row(
                source_family="autonomous_driving_ngsim",
                source_unit=candidate_id,
                domain="autonomous_driving",
                final_disposition=(
                    "ready_for_full_admission" if files_present and runtime_verified else "held_repair"
                ),
                reason_codes=(
                    ["native_runtime_and_candidate_evidence_available", "delta_replay_required"]
                    if files_present and runtime_verified
                    else ["canonical_source_and_portable_bundle_unavailable"]
                ),
                repair_attempts=[
                    _attempt(
                        "runtime_environment_closure",
                        "passed" if runtime_verified else "not_evidenced",
                        "The independent SUMO/CommonRoad runtime is repairable and is not an "
                        "intrinsic candidate-quality gate.",
                    ),
                    _attempt(
                        "canonical_source_bundle_recovery",
                        "passed" if files_present else "blocked_external",
                        "Recover one canonical bundle per NGSIM window from locked source bytes; "
                        "difficulty copies are not independent candidates.",
                    ),
                ],
                evidence={
                    **evidence,
                    "archive_ref": "ecd2a4fc^",
                    "files_present": files_present,
                },
                candidate_id=candidate_id,
            )
        )
    return rows


def build_refinement(
    *,
    source_suite_path: Path,
    pglib_opf_root: Path,
    pglib_uc_root: Path,
    citylearn_root: Path,
    sumo365_root: Path,
    resco_root: Path,
    nrel_microgrid_root: Path,
    opendss_root: Path,
    simbench_metadata_path: Path | None,
    datacenter_ledger_path: Path | None,
    autonomous_evidence_paths: list[Path],
    rts_root: Path | None = None,
    archived_autonomous_candidates: tuple[str, ...] = (),
    external_source_roots: dict[str, tuple[str, Path]] | None = None,
    execute_native_probes: bool = False,
) -> dict[str, Any]:
    suite = _load(source_suite_path)
    active = _active_tokens(suite)
    rows: list[dict[str, Any]] = []
    rows.extend(_pglib_opf(pglib_opf_root, active))
    rows.extend(_pglib_uc(pglib_uc_root, active))
    rows.extend(
        _citylearn(
            citylearn_root,
            active,
            execute_native_probes=execute_native_probes,
        )
    )
    rows.extend(
        _sumo_family(
            sumo365_root,
            family="sumo365_ingolstadt",
            active=active,
            glob="*.sumocfg",
            execute_native_probes=execute_native_probes,
        )
    )
    rows.extend(
        _sumo_family(
            resco_root,
            family="resco",
            active=active,
            glob="*/*.sumocfg",
            execute_native_probes=execute_native_probes,
        )
    )
    rows.extend(_nrel(nrel_microgrid_root, active))
    rows.extend(
        _opendss(
            opendss_root,
            active,
            execute_native_probes=execute_native_probes,
        )
    )
    rows.extend(
        _metadata_candidates(
            simbench_metadata_path,
            family="simbench_commercial",
            domain="power_grid",
        )
    )
    rows.extend(_datacenter(datacenter_ledger_path))
    rows.extend(_autonomous(autonomous_evidence_paths, archived_autonomous_candidates))
    if rts_root is not None:
        rows.extend(_rts(rts_root, active) or [_missing("rts_gmlc", "power_grid", rts_root)])
    for family, (domain, path) in sorted((external_source_roots or {}).items()):
        rows.append(
            _present_external(family, domain, path)
            if path.exists()
            else _missing(family, domain, path)
        )
    rows.sort(key=lambda row: (row["source_family"], row["source_unit"], row["candidate_id"]))
    keys = [(row["source_family"], row["source_unit"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("candidate refinement contains duplicate canonical source units")
    if any(
        not row["candidate_id"]
        or not row["source_id"]
        or not row["reason_codes"]
        or row["final_disposition"] not in FINAL_DISPOSITIONS
        for row in rows
    ):
        raise ValueError("every discovered unit requires a terminal disposition and evidence")
    dispositions = Counter(str(row["final_disposition"]) for row in rows)
    ready = [row for row in rows if row["final_disposition"] == "ready_for_full_admission"]
    summary = {
        "n_discovered": len(rows),
        "n_terminal": len(rows),
        "n_unresolved": 0,
        "n_raw_units": len(rows),
        "n_independent_candidates": len(ready),
        "dispositions": dict(sorted(dispositions.items())),
        "by_domain": dict(sorted(Counter(row["domain"] for row in rows).items())),
        "by_source_family": dict(
            sorted(Counter(row["source_family"] for row in rows).items())
        ),
    }
    report = {
        "schema_version": "operate-infrastructure-candidate-refinement-v1",
        "status": "terminal_candidate_refinement_complete",
        "candidate_only": True,
        "release_admission": False,
        "provider_llm_admission": False,
        "policy": {
            "environment_and_dependency_failures_do_not_reduce_core_denominator": True,
            "difficulty_and_family_quotas_are_diagnostic": True,
            "package_versions_are_environment_evidence_not_candidate_quality": True,
            "essential_delta_replay_before_materialization": [
                "source_values_drive_state",
                "deterministic_seeded_replay",
                "reference_beats_no_action",
                "native_control_changes_backend",
                "no_safety_regression",
            ],
        },
        "inputs": {
            "source_suite": {
                "path": str(source_suite_path),
                "sha256": _sha256(source_suite_path),
            }
        },
        "rows": rows,
        "raw_units": rows,
        "independent_candidates": ready,
        "summary": summary,
    }
    return canonicalize_repo_owned_paths(report, repo_root=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--_opendss-native-probe-root",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_opendss-native-probe-entry",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--source-suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args._opendss_native_probe_root or args._opendss_native_probe_entry:
        if not (
            args._opendss_native_probe_root and args._opendss_native_probe_entry
        ):
            parser.error("both internal OpenDSS probe paths are required")
        print(
            json.dumps(
                _opendss_native_probe_direct(
                    args._opendss_native_probe_root,
                    args._opendss_native_probe_entry,
                ),
                sort_keys=True,
            )
        )
        return 0
    external = {
        "building_data_genome_2": (
            "building_energy",
            ROOT / "works/building-data-genome-2",
        ),
        "cityflow_examples": ("traffic", ROOT / "works/CityFlow"),
        "flatland": ("rail", ROOT / "works/flatland-rl"),
        "grid2op_cache": ("power_grid", ROOT / "works/Grid2Op_cache"),
    }
    payload = build_refinement(
        source_suite_path=args.source_suite,
        pglib_opf_root=ROOT / "works/PGLib-OPF",
        pglib_uc_root=ROOT / "works/pglib-uc",
        citylearn_root=ROOT / "works/CityLearn/data/datasets",
        sumo365_root=ROOT / "works/sumo_ingolstadt/simulation/Ingolstadt SUMO 365",
        resco_root=ROOT / "works/RESCO/resco_benchmark/environments",
        nrel_microgrid_root=ROOT / "works/nrel-microgrid",
        opendss_root=ROOT / "works/OpenDSS-IEEE13",
        simbench_metadata_path=(
            ROOT / ".hl/artifacts/operate_v058_simbench_commercial_p19680_candidate_metadata.json"
        ),
        datacenter_ledger_path=None,
        autonomous_evidence_paths=[
            ROOT
            / ".hl/artifacts/core_supplement_20260815/autonomous_driving_calibration_batch_2_remaining.json"
        ],
        rts_root=ROOT / "works/RTS-GMLC",
        archived_autonomous_candidates=ARCHIVED_NGSIM_CANDIDATES,
        external_source_roots=external,
        execute_native_probes=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
