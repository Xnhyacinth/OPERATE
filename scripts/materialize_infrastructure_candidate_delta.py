#!/usr/bin/env python3
"""Materialize executable infrastructure candidates without mutating Core.

The input refinement ledger is structural evidence, not release admission.
Only rows explicitly scoped as ``candidate`` and disposed ``ready_for_full_admission`` are
considered.  A row is materialized only when an existing current-tree builder
can produce a source-locked scenario with a stable signature, typed backend
events, registered native tools, and an ordered task contract.  Every other
eligible row is retained as a machine-readable blocker.
"""

from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.scenario_validator import validate_scenario_yaml  # noqa: E402
from core.source_asset_contract import resolve_source_asset_contract  # noqa: E402
from core.suite_identity import recompute_signature_with_seed  # noqa: E402
from domains.registry import get_backend_capability  # noqa: E402
from scripts.build_pglib_bulk_protocol21_candidates import (  # noqa: E402
    build as build_pglib_bulk,
)
from scripts.build_protocol21_candidate_source_suite import (  # noqa: E402
    build_suite,
)
from scripts.refine_infrastructure_candidate_pool import _dss_graph  # noqa: E402
from scripts.lock_citylearn_source import build as build_citylearn_lock  # noqa: E402
from scripts.refine_citylearn_long_horizon_candidates import (  # noqa: E402
    WindowPlan,
    _episode_perturbations as citylearn_episode_perturbations,
    _scenario_for_window as build_citylearn_window_scenario,
    _source_window_payload as build_citylearn_window_payload,
)
from scripts.mine_sumo365_native_traffic import (  # noqa: E402
    _demand_change_window,
    parse_route_departures,
)
from domains.traffic.source_identity import resolve_sumo_input_graph  # noqa: E402
from domains.traffic.runtime_control_contract import (  # noqa: E402
    parsed_program_ids_by_tls,
)

DEFAULT_REFINEMENT_LEDGER = (
    ROOT / ".hl/artifacts/operate_v058_infrastructure_candidate_refinement.json"
)
DEFAULT_ACTIVE_SUITE = ROOT / "release/operate_v0_59_0/protocol21_source_suite.json"
DEFAULT_OUTPUT_ROOT = ROOT / ".hl/artifacts/operate_v058_infrastructure_delta"
DEFAULT_OPENDSS_TEMPLATE = (
    ROOT / "scenarios/operate_v0_58_0/power_grid/opendss_fresh_feeders_volt_var/"
    "deep_planning/medium/"
    "opendss_fresh_ieee34_volt_var_basic_s42_native_response_medium_two_tap_windows.yaml"
)
DEFAULT_CITYLEARN_TEMPLATE = (
    ROOT / "scenarios/operate_v0_58_0/building_energy/citylearn_der_storage_control/"
    "source_locked_long_horizon/extreme/"
    "citylearn_challenge_2022_phase_1_w216_287.yaml"
)
DEFAULT_SUMO_TEMPLATE = (
    ROOT / "scenarios/operate_v0_58_0/traffic/signal_coordination/deep_planning/basic/"
    "ingolstadt_24h_evening17_phase_s9616.yaml"
)

_PGLIB_UC_TYPED_PERTURBATIONS = frozenset(
    {
        "planned_maintenance",
        "generator_forced_outage",
        "fuel_supply_delay",
        "wind_dropout",
        "load_surge",
        "line_outage",
        "opponent_attack",
        "storm_window",
    }
)
_PGLIB_UC_SILENT_SOURCE_MODIFIERS = frozenset({"forecast_bias"})

_DSS_RUNTIME_FILE_REFERENCE = re.compile(
    r"""\b(?:csvfile|sngfile|dblfile|file)\s*=\s*[\[(]?\s*
    (?:"([^"]+)"|'([^']+)'|([^,\s\)\]]+))""",
    re.IGNORECASE | re.VERBOSE,
)
_DSS_AUXILIARY_FILE_REFERENCE = re.compile(
    r"""^\s*(?:buscoords|latlongcoords|guids)\s+
    (?:file\s*=\s*)?[\[(]?\s*
    (?:"([^"]+)"|'([^']+)'|([^\s\)!\]]+))""",
    re.IGNORECASE | re.VERBOSE,
)

_FAMILY_BLOCKERS: dict[str, tuple[str, tuple[str, ...], str]] = {
    "pglib_opf": (
        "pglib_opf_bounded_task_materializer_missing",
        (
            "source_bound_temporal_axis",
            "ordered_native_task_contract",
            "reference_policy",
        ),
        "domains.power_grid.seeds.from_pglib_opf",
    ),
    "citylearn": (
        "citylearn_source_window_materializer_missing",
        (
            "source_bound_event_window",
            "ordered_storage_task_contract",
            "reference_policy",
        ),
        "scripts.refine_citylearn_long_horizon_candidates",
    ),
    "opendss_distribution": (
        "opendss_response_window_materializer_missing",
        (
            "source_bound_response_window",
            "asset_specific_task_contract",
            "reference_policy",
        ),
        "domains.power_grid.backends.opendss_fresh_feeders",
    ),
    "sumo365_ingolstadt": (
        "sumo365_operational_window_materializer_missing",
        (
            "source_window_selection",
            "typed_actionable_event_schedule",
            "native_tool_headroom",
            "reference_policy",
        ),
        "scripts.mine_sumo365_native_traffic",
    ),
    "resco": (
        "resco_operational_window_materializer_missing",
        (
            "source_window_selection",
            "typed_actionable_event_schedule",
            "native_tool_headroom",
            "reference_policy",
        ),
        "scripts.build_traffic_resco_network_candidates",
    ),
    "rts_gmlc": (
        "rts_coherent_graph_materializer_missing",
        (
            "exact_da_rt_renewable_graph_loader",
            "ordered_native_task_contract",
            "reference_policy",
        ),
        "domains.power_grid.seeds.from_rts_gmlc",
    ),
    "simbench_commercial": (
        "simbench_current_tree_prefilter_missing",
        (
            "current_tree_native_prefilter",
            "source_window_scenario_body",
            "reference_policy",
        ),
        "scripts.build_simbench_protocol21_long_horizon_candidates",
    ),
    "nrel_microgrid": (
        "nrel_source_window_materializer_missing",
        (
            "source_window_selection",
            "typed_event_schedule",
            "ordered_native_task_contract",
            "reference_policy",
        ),
        "domains.microgrid.candidate_pipeline",
    ),
}


@dataclass(frozen=True)
class DeltaBuild:
    report: dict[str, Any]
    candidate_report: dict[str, Any]
    suite_preview: dict[str, Any]
    files: dict[Path, dict[str, Any]]
    json_files: dict[Path, dict[str, Any]]
    output_root: Path


def _load_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _opendss_source_ref(source_root: Path, path: Path | None = None) -> str:
    """Keep the canonical logical source root when it is a symlink.

    Runtime graph discovery resolves files for byte checks.  Re-emitting those
    resolved cache paths would make an otherwise portable candidate depend on
    the local cache layout instead of ``works/OpenDSS-IEEE13``.
    """
    logical = source_root
    if path is not None:
        relative = path.resolve().relative_to(source_root.resolve())
        logical = source_root / relative
    absolute = logical if logical.is_absolute() else ROOT / logical
    try:
        return absolute.absolute().relative_to(ROOT).as_posix()
    except ValueError:
        return str(absolute.absolute())


def _source_unit_from_body(body: dict[str, Any], uc_root: Path) -> str | None:
    case_file = str((body.get("backend_config") or {}).get("case_file") or "")
    if not case_file:
        return None
    path = Path(case_file)
    if not path.is_absolute():
        path = ROOT / path
    try:
        return path.resolve().relative_to(uc_root.resolve()).as_posix()
    except ValueError:
        return None


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _strip_dss_comment(line: str) -> str:
    quote: str | None = None
    index = 0
    while index < len(line):
        char = line[index]
        if quote is not None:
            if char == quote:
                quote = None
        elif char in {'"', "'"}:
            quote = char
        elif char == "!" or line[index : index + 2] == "//":
            return line[:index]
        index += 1
    return line


def _opendss_runtime_file_assets(source_root: Path, graph: list[Path]) -> list[Path]:
    root = source_root.resolve()
    assets: set[Path] = set()
    for source in graph:
        if source.suffix.lower() != ".dss":
            continue
        for raw_line in source.read_text(
            encoding="utf-8", errors="strict"
        ).splitlines():
            line = _strip_dss_comment(raw_line)
            if line.lstrip().lower().startswith("export "):
                continue
            declared_paths = [
                next(value for value in match.groups() if value).strip()
                for match in _DSS_RUNTIME_FILE_REFERENCE.finditer(line)
            ]
            auxiliary = _DSS_AUXILIARY_FILE_REFERENCE.match(line)
            if auxiliary is not None:
                declared_paths.append(
                    next(
                        value for value in auxiliary.groups() if value is not None
                    ).strip()
                )
            for raw in declared_paths:
                normalized = raw.replace("\\", "/")
                if re.match(r"^[a-z]:/", normalized, re.IGNORECASE):
                    raise ValueError(f"OpenDSS runtime file escapes source root: {raw}")
                path = Path(normalized)
                resolved = (
                    path.resolve()
                    if path.is_absolute()
                    else (source.parent / path).resolve()
                )
                if not resolved.is_relative_to(root):
                    raise ValueError(f"OpenDSS runtime file escapes source root: {raw}")
                if not resolved.is_file():
                    raise FileNotFoundError(
                        f"missing OpenDSS runtime file input: {resolved}"
                    )
                assets.add(resolved)
    return sorted(assets)


def _opendss_control(row: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    controls = ((row.get("evidence") or {}).get("native_probe") or {}).get(
        "native_controls"
    ) or {}
    if int(controls.get("regcontrols") or 0) > 0:
        return "set_transformer_tap", {"trafo_id": 0, "tap_pos": 1}
    if int(controls.get("capacitors") or 0) > 0:
        return "switch_capacitor", {"cap_id": 0, "status": False}
    return None


def _is_ckt5_manual_takeover(source_unit: str) -> bool:
    normalized = source_unit.replace("\\", "/").lower()
    return normalized.endswith(
        "/epritestcircuits/ckt5/master_ckt5.dss"
    ) or normalized == "epritestcircuits/ckt5/master_ckt5.dss"


def _opendss_source_axes(
    row: dict[str, Any],
    *,
    candidate_slug: str,
    source_root: Path,
    source_unit: str,
    source_refs: list[str],
    master_sha256: str,
    source_denominator_key: str,
) -> dict[str, Any]:
    native_probe = (row.get("evidence") or {}).get("native_probe") or {}
    native_controls = native_probe.get("native_controls") or {}
    summary = native_probe.get("summary") or {}
    controllable_assets = [
        asset
        for count_key, asset in (
            ("capacitors", "capacitor_banks"),
            ("regcontrols", "voltage_regulators"),
            ("switchable_lines", "switchable_lines"),
        )
        if int(native_controls.get(count_key) or 0) > 0
    ]
    axes: dict[str, Any] = {
        "backend": "opendss_fresh_feeders",
        "preview_backend": "opendss_fresh_feeder_probe",
        "source": "dss_extensions_electricdss_tst",
        "feeder": candidate_slug,
        "master_file": source_unit,
        "topology": "unbalanced_three_phase_distribution",
        "decision_axis": "fresh_feeder_volt_var_control",
        "source_denominator_key": source_denominator_key,
        "controllable_assets": controllable_assets,
        "source_root": _opendss_source_ref(source_root),
        "master_sha256": master_sha256,
        "runtime_include_graph": source_refs,
    }
    for key in (
        "n_buses",
        "n_lines",
        "n_loads",
        "n_capacitors",
        "n_regcontrols",
    ):
        value = summary.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            axes[key] = value
    return axes


def _build_opendss_body(
    row: dict[str, Any],
    *,
    output_root: Path,
    template_path: Path,
) -> tuple[Path, dict[str, Any]]:
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    if not isinstance(template, dict):
        raise ValueError(f"OpenDSS template is not a mapping: {template_path}")
    entry = Path(str((row.get("evidence") or {}).get("entry_path") or ""))
    if not entry.is_file():
        raise FileNotFoundError(entry)
    source_root = entry
    while source_root.parent != source_root and not (source_root / ".git").exists():
        source_root = source_root.parent
        if source_root.name == "OpenDSS-IEEE13":
            break
    if source_root.name != "OpenDSS-IEEE13":
        common = entry.parent
        source_root = common
        source_unit = str(row.get("source_unit") or "")
        for _ in Path(source_unit).parts[:-1]:
            source_root = source_root.parent
    graph, missing = _dss_graph(source_root, entry)
    if missing:
        raise ValueError(f"OpenDSS include graph incomplete: {missing}")
    runtime_files = _opendss_runtime_file_assets(source_root, graph)
    control = _opendss_control(row)
    if control is None:
        raise ValueError("OpenDSS candidate has no audited native control")
    tool, args = control
    source_unit = entry.relative_to(source_root).as_posix()
    manual_takeover = _is_ckt5_manual_takeover(source_unit)
    runtime_graph = sorted({*graph, *runtime_files})
    source_refs = [
        _opendss_source_ref(source_root, path) for path in runtime_graph
    ]
    hashes = {
        _opendss_source_ref(source_root, path): _sha256(path)
        for path in runtime_graph
    }
    candidate_slug = _slug(str(row["candidate_id"]))
    master_ref = _opendss_source_ref(source_root, entry)
    master_sha256 = _sha256(entry)
    source_denominator_key = (
        f"opendss:{(row.get('evidence') or {}).get('asset_graph_sha256')}:{tool}"
    )
    difficulty = "basic" if manual_takeover else "high"
    scenario_id = (
        "power_grid/opendss_infrastructure_feeder_control/deep_planning/"
        f"{difficulty}/{candidate_slug}"
    )
    perturbations = [] if manual_takeover else [
        {
            "kind": "load_surge",
            "trigger_tick": 1,
            "duration_ticks": 2,
            "target": {"load_fraction": 0.05},
            "intensity": 1.0,
            "hidden": False,
            "notes": "Deterministic procedural response-window stress overlay.",
        }
    ]
    body = copy.deepcopy(template)
    body.update(
        {
            "seed_id": scenario_id,
            "scenario_id": scenario_id,
            "family": "opendss_infrastructure_feeder_control",
            "difficulty_mode": "deep_planning",
            "difficulty_level": difficulty,
            "horizon_ticks": 12 if manual_takeover else 6,
            "seed": 42,
            "source_contract": {
                "runtime_input": source_refs,
                "derivation_input": [],
                "file_sha256s": hashes,
            },
            "perturbations": perturbations,
            "provenance": {
                "data_source": "dss_extensions_electricdss_tst",
                "files": source_refs,
                "url": "https://github.com/dss-extensions/electricdss-tst",
                "commit": "3b208397160213cae4a9e2d0a7d1aa3528ce26e1",
                "lock_strategy": "git_commit+sha256_include_graph",
                "license": "BSD-3-Clause",
                "time_window": {
                    "master_file": source_unit,
                    "decision_axis": tool,
                },
                "notes": (
                    "Locked native feeder graph and source-declared yearly load programs; "
                    "the agent replaces the source CapControl automation under an explicit "
                    "manual supervisory takeover treatment."
                    if manual_takeover
                    else "Locked native feeder graph with deterministic typed load-surge "
                    "response windows; the stress overlay does not create source independence."
                ),
            },
            "candidate_only": True,
            "release_admission": False,
        }
    )
    config = body.setdefault("backend_config", {})
    config.update(
        {
            "feeder": candidate_slug,
            "source_root": _opendss_source_ref(source_root),
            "master_file": source_unit,
            "source_denominator_key": source_denominator_key,
            "source_axes": _opendss_source_axes(
                row,
                candidate_slug=candidate_slug,
                source_root=source_root,
                source_unit=source_unit,
                source_refs=source_refs,
                master_sha256=master_sha256,
                source_denominator_key=source_denominator_key,
            ),
            "controllable_assets": [tool],
            "preventive_stabilization_recipe": {
                "version": "opendss_preventive_stabilization_v1",
                "source_identity": master_ref,
                "runtime_consumption_required": True,
                "startup_controls_are_not_agent_actions": True,
            },
            "initial_native_controls": [],
            "reference_policy_contract": (
                "opendss.live_voltage_feedback.v2"
                if manual_takeover
                else "opendss.live_voltage_feedback.v1"
            ),
            "task_contract": {
                "contract": (
                    "power_grid.reliability_loss_mitigation.v2"
                    if manual_takeover
                    else "power_grid.opendss.preventive_stabilization.v1"
                ),
                "control_window": {
                    "first_tick": 0 if manual_takeover else 1,
                    "last_tick": 4,
                },
            },
            "task_requirements": {
                "min_distinct_control_ticks": 1,
                "min_distinct_physical_tools": 1,
                "ordered_tool_milestones": [
                    {
                        "tool": tool,
                        "not_before_tick": 0 if manual_takeover else 1,
                        "not_after_tick": 4,
                    },
                ],
            },
            "runtime_source_lock": {
                "master_file": master_ref,
                "master_sha256": master_sha256,
                "include_graph_paths": source_refs,
                "file_sha256s": hashes,
            },
        }
    )
    if manual_takeover:
        config["native_yearly_program"] = {
            "start_hour": 0,
            "step_hours": 1,
        }
        config["manual_takeover_contract"] = {
            "contract": "opendss.supervisory_manual_takeover.v1",
            "source_capcontrols_present": True,
            "control_mode": "off_for_agent_supervision",
            "counterfactual_control_mode": "same_as_treatment",
        }
    config.pop("response_window_recipe", None)
    config.pop("reference_action_schedule", None)
    body.pop("source_lock", None)
    body["scenario_signature"] = recompute_signature_with_seed(body, 42)
    output = (
        output_root / "scenarios/power_grid/opendss_infrastructure_feeder_control/"
        f"deep_planning/{difficulty}" / f"{candidate_slug}.yaml"
    )
    return output, body


def _probe_opendss_body(body: dict[str, Any]) -> dict[str, Any]:
    from domains.power_grid.backends.opendss_fresh_feeders import (
        OpenDssFreshFeedersBackend,
    )

    seed = SimpleNamespace(
        horizon_ticks=body["horizon_ticks"],
        perturbations=body.get("perturbations") or [],
        backend_config=body["backend_config"],
    )
    backend = OpenDssFreshFeedersBackend()
    summary = backend.reset(seed)
    tool = str(body["backend_config"]["controllable_assets"][0])
    before = backend.snapshot()
    manual_takeover = isinstance(
        body["backend_config"].get("manual_takeover_contract"), dict
    )
    if manual_takeover and tool == "switch_capacitor":
        reference_actions: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        for control in before.get("capacitors") or []:
            if not any(int(state) for state in control.get("states") or []):
                continue
            args = {"cap_id": int(control["cap_id"]), "status": False}
            tool_results.append(backend.apply_tool_effect(tool, args))
            reference_actions.append({"tick": 0, "tool": tool, "args": args})
        after = backend.snapshot()
        state_changing = before != after and any(
            result.get("_status") == "applied" for result in tool_results
        )
        native_improvement = (
            int(after.get("n_voltage_violations") or 0)
            < int(before.get("n_voltage_violations") or 0)
            or float(after.get("voltage_band_error") or 0.0)
            < float(before.get("voltage_band_error") or 0.0)
        )
        return {
            "status": (
                "passed"
                if summary.converged and state_changing and native_improvement
                else "failed"
            ),
            "reset": bool(summary.converged),
            "native_tool": tool,
            "tool_results": tool_results,
            "state_changing": state_changing,
            "native_improvement": native_improvement,
            "before": {
                "n_voltage_violations": before.get("n_voltage_violations"),
                "voltage_min_pu": before.get("voltage_min_pu"),
                "voltage_max_pu": before.get("voltage_max_pu"),
            },
            "after": {
                "n_voltage_violations": after.get("n_voltage_violations"),
                "voltage_min_pu": after.get("voltage_min_pu"),
                "voltage_max_pu": after.get("voltage_max_pu"),
            },
            "reference_actions": reference_actions,
        }
    if tool == "set_transformer_tap":
        control = before["regcontrols"][0]
        current = int(control["tap_number"])
        first_target = current + 1 if current < int(control["tap_max"]) else current - 1
        first_args = {"trafo_id": 0, "tap_pos": first_target}
    elif tool == "switch_capacitor":
        control = before["capacitors"][0]
        initial_status = bool((control.get("states") or [0])[0])
        first_args = {"cap_id": 0, "status": not initial_status}
    else:
        raise ValueError(f"unsupported OpenDSS response tool: {tool}")
    first_result = backend.apply_tool_effect(tool, first_args)
    after_first = backend.snapshot()
    first_changed = before != after_first
    reference_actions = [
        {"tick": 1, "tool": tool, "args": first_args},
    ]
    return {
        "status": (
            "passed"
            if (
                summary.converged
                and first_result.get("_status") == "applied"
                and first_changed
            )
            else "failed"
        ),
        "reset": bool(summary.converged),
        "native_tool": tool,
        "tool_results": [first_result],
        "state_changing": first_changed,
        "reference_actions": reference_actions,
    }


def _citylearn_load_solar_window(
    *,
    template: dict[str, Any],
    plan: WindowPlan,
    source_root: Path,
    lock_path: Path,
    lock: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    schema = json.loads((source_root / "schema.json").read_text(encoding="utf-8"))
    building_rows = schema.get("buildings") or {}
    first = next(iter(building_rows.values()), None)
    energy_file = str((first or {}).get("energy_simulation") or "")
    source_asset = source_root / energy_file
    if not source_asset.is_file():
        raise FileNotFoundError(
            f"CityLearn building timeseries missing: {source_asset}"
        )
    with source_asset.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    start = plan.start
    end = start + plan.horizon - 1
    if len(rows) <= end:
        raise ValueError(
            "CityLearn source is shorter than the 72-tick candidate window"
        )
    locked_digest = next(
        (
            str(digest)
            for ref, digest in (lock.get("files") or {}).items()
            if Path(str(ref)).resolve() == source_asset.resolve()
        ),
        "",
    )
    if len(locked_digest) != 64:
        raise ValueError("CityLearn building timeseries is absent from the source lock")
    candidates: list[tuple[float, int, str, str]] = []
    for column, kind in (
        ("non_shiftable_load", "load_change"),
        ("solar_generation", "generation_change"),
    ):
        values = [float(rows[tick][column]) for tick in range(start, end + 1)]
        candidates.extend(
            (abs(values[local] - values[local - 1]), local, column, kind)
            for local in range(1, len(values))
            if abs(values[local] - values[local - 1]) > 1e-8
        )
    selected: list[tuple[float, int, str, str]] = []
    per_column: Counter[str] = Counter()
    for candidate in sorted(candidates, reverse=True):
        _, local, column, _ = candidate
        if per_column[column] >= 2 or any(
            abs(local - prior[1]) < 4 for prior in selected
        ):
            continue
        selected.append(candidate)
        per_column[column] += 1
        if len(selected) == 4:
            break
    if len(selected) < 4 or set(per_column) != {
        "non_shiftable_load",
        "solar_generation",
    }:
        raise ValueError("four separated native load/solar transitions are unavailable")
    events = []
    response_windows = []
    for index, (delta, local, column, kind) in enumerate(
        sorted(selected, key=lambda x: x[1])
    ):
        source_tick = start + local
        before = float(rows[source_tick - 1][column])
        after = float(rows[source_tick][column])
        event_id = f"{plan.dataset_id}_{column}_t{source_tick}"
        events.append(
            {
                "event_id": event_id,
                "kind": kind,
                "trigger_tick": source_tick,
                "duration_ticks": 4,
                "hidden": index >= 2,
                "channel": f"building_timeseries.{column}",
                "source_asset": _relative(source_asset),
                "source_asset_sha256": locked_digest,
                "source_row_before": source_tick - 1,
                "source_row_after": source_tick,
                "source_value_before": before,
                "source_value_after": after,
                "materiality_metric": "source_value_absolute_delta",
                "materiality_threshold": max(delta * 0.5, 1e-9),
                "source_observed": True,
                "procedural_overlay": False,
            }
        )
        policy = (
            "discharge"
            if (column == "non_shiftable_load" and after > before)
            or (column == "solar_generation" and after < before)
            else "charge"
        )
        response_windows.append(
            {
                "event_id": event_id,
                "first_tick": local,
                "last_tick": min(plan.horizon - 1, local + 3),
                "native_control": "set_storage_dispatch",
                "expected_control_policy": policy,
            }
        )
    runtime_files = {
        str(ref): str(digest)
        for ref, digest in (lock.get("files") or {}).items()
        if Path(str(ref)).resolve().is_relative_to(source_root)
    }
    window_hash = hashlib.sha256(
        json.dumps(events, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    body = copy.deepcopy(template)
    scenario_id = (
        "building_energy/citylearn_native_load_solar_storage_control/"
        f"source_locked_long_horizon/high/{plan.dataset_id}_w{start}_{end}"
    )
    body.update(
        {
            "seed_id": scenario_id,
            "scenario_id": scenario_id,
            "family": "citylearn_native_load_solar_storage_control",
            "source_root": _relative(source_root),
            "source_lock": _relative(lock_path),
            "horizon_ticks": plan.horizon,
            "difficulty_level": "high",
            "difficulty_mode": "source_locked_long_horizon",
            "perturbations": citylearn_episode_perturbations(
                events, source_window_start=start
            ),
            "source_contract": {
                "runtime_input": sorted(runtime_files),
                "derivation_input": [],
                "file_sha256s": runtime_files,
                "derived_window": {
                    "sha256": window_hash,
                    "recipe_version": "citylearn_native_load_solar_window_v1",
                },
            },
        }
    )
    config = body["backend_config"]
    config.update(
        {
            "simulation_start_time_step": start,
            "simulation_end_time_step": end + 24,
            "episode_time_steps": plan.horizon + 24,
            "native_source_events": events,
            "source_window": {
                "first_time_step": start,
                "last_time_step": end,
                "runtime_context_last_time_step": end + 24,
                "source_window_sha256": window_hash,
            },
            "task_contract": {
                "contract": "building_energy.citylearn.load_solar_storage_dispatch.v1",
                "standing_plan_required": True,
                "milestone_ticks": [
                    0,
                    *[int(row["first_tick"]) for row in response_windows],
                ],
                "response_windows": response_windows,
            },
        }
    )
    return body, {"window_sha256": window_hash}


def _build_citylearn_body(
    row: dict[str, Any],
    *,
    output_root: Path,
    template_path: Path,
    lock_builder: Callable[..., dict[str, Any]],
    window_start: int | None = None,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    if not isinstance(template, dict):
        raise ValueError(f"CityLearn template is not a mapping: {template_path}")
    schema_path = Path(str((row.get("evidence") or {}).get("schema_path") or ""))
    if not schema_path.is_file():
        raise FileNotFoundError(schema_path)
    source_root = schema_path.parent.resolve()
    missing_channels = [
        name
        for name in ("pricing.csv", "carbon_intensity.csv")
        if not (source_root / name).is_file()
    ]
    dataset_id = str(row.get("source_unit") or source_root.name)
    lock = lock_builder(source_root, dataset_id=dataset_id)
    lock_path = output_root / "source_locks" / f"{_slug(dataset_id)}.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    start = (
        int(window_start)
        if window_start is not None
        else int(schema.get("simulation_start_time_step") or 0)
    )
    plan = WindowPlan(dataset_id=dataset_id, start=start, horizon=72)
    template["source_root"] = _relative(source_root)
    if missing_channels:
        body, window = _citylearn_load_solar_window(
            template=template,
            plan=plan,
            source_root=source_root,
            lock_path=lock_path,
            lock=lock,
        )
    else:
        try:
            window = build_citylearn_window_payload(
                plan=plan, scenario=template, lock=lock
            )
            body = build_citylearn_window_scenario(
                plan=plan,
                base=template,
                window=window,
                lock_path=lock_path,
                lock=lock,
            )
        except ValueError as exc:
            if "transition" not in str(exc):
                raise
            body, window = _citylearn_load_solar_window(
                template=template,
                plan=plan,
                source_root=source_root,
                lock_path=lock_path,
                lock=lock,
            )
            missing_channels = ["natural_tariff_carbon_transition_window"]
    buildings = sorted((schema.get("buildings") or {}).keys())
    if not buildings:
        raise ValueError("CityLearn source has no enabled building")
    config = body["backend_config"]
    response_windows = list(config["task_contract"]["response_windows"])
    schedule = []
    milestones = []
    for index, response in enumerate(response_windows):
        tick = int(response["first_tick"])
        rate = 0.2 if response["expected_control_policy"] == "charge" else -0.2
        args = {"building_id": buildings[0], "rate": rate}
        schedule.append({"tick": tick, "tool": "set_storage_dispatch", "args": args})
        milestones.append(
            {
                "tool": "set_storage_dispatch",
                "args": args,
                "not_before_tick": tick,
                "not_after_tick": int(response["last_tick"]),
            }
        )
        if index >= 2:
            break
    config.update(
        {
            "source_denominator_key": (
                f"citylearn:{dataset_id}:{window['window_sha256']}:storage_dispatch"
            ),
            "reference_policy_contract": (
                "citylearn.native_load_solar_storage_response_window.v1"
                if missing_channels
                else "citylearn.native_storage_response_window.v1"
            ),
            "reference_action_schedule": schedule,
        }
    )
    config["task_requirements"].update(
        {
            "min_distinct_control_ticks": len(milestones),
            "min_distinct_physical_tools": 1,
            "ordered_tool_milestones": milestones,
        }
    )
    body["candidate_only"] = True
    body["release_admission"] = False
    body.setdefault("provenance", {}).update(
        {
            "data_source": dataset_id,
            "files": sorted(body["source_contract"]["runtime_input"]),
            "source_window": [start, start + 71],
            "notes": (
                "Exact locked CityLearn native transitions with no procedural event overlay; "
                + (
                    "the task uses load/solar storage balancing because tariff or carbon is "
                    "not present in this source."
                    if missing_channels
                    else "the task uses tariff, carbon, building-load and solar channels."
                )
            ),
        }
    )
    body["scenario_signature"] = recompute_signature_with_seed(body, int(body["seed"]))
    output = (
        output_root / "scenarios/building_energy/citylearn_der_storage_control/"
        "source_locked_long_horizon/extreme"
        / f"{_slug(dataset_id)}_w{start}_{start + 71}.yaml"
    )
    return output, body, lock_path, lock


def _probe_citylearn_body(
    body: dict[str, Any], *, source_lock_payload: dict[str, Any]
) -> dict[str, Any]:
    from core import Action, ToolCall
    from domains.building_energy.adapter import BuildingEnergyEnvironment

    with tempfile.TemporaryDirectory(prefix="operate-citylearn-lock-") as temp_dir:
        lock_path = Path(temp_dir) / "source_lock.json"
        lock_path.write_text(
            json.dumps(source_lock_payload, sort_keys=True) + "\n", encoding="utf-8"
        )
        probe_body = copy.deepcopy(body)
        probe_body["source_lock"] = str(lock_path)
        env = BuildingEnergyEnvironment()
        try:
            observation = env.reset(probe_body, seed=int(probe_body["seed"]))
            buildings = sorted((observation.get("buildings") or {}).keys())
            if not buildings:
                raise RuntimeError("CityLearn reset returned no buildings")
            pending = env.step(
                Action(
                    tool_calls=[
                        ToolCall(
                            name="set_storage_dispatch",
                            args={"building_id": buildings[0], "rate": 0.2},
                            call_id="candidate-native-effect",
                        )
                    ]
                )
            )
            realized = env.step(Action(tool_calls=[ToolCall(name="wait")]))
            results = [*pending.tool_results, *realized.tool_results]
            state_changing = any(
                result.name == "set_storage_dispatch"
                and result.ok
                and result.state_changing
                for result in results
            )
            delayed = any(
                result.name == "set_storage_dispatch"
                and (result.payload or {}).get("_status") == "pending"
                for result in results
            )
            return {
                "status": "passed" if state_changing and delayed else "failed",
                "reset": True,
                "native_tool": "set_storage_dispatch",
                "delayed_ack": delayed,
                "state_changing": state_changing,
            }
        finally:
            env.close()


def _sumo_graph_rows(graph: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        graph["sumocfg"],
        graph["network"],
        *graph.get("route_files", []),
        *graph.get("additional_files", []),
        *graph.get("recursive_inputs", []),
    ]


def _build_sumo365_body(
    row: dict[str, Any], *, family: str, output_root: Path, template_path: Path
) -> tuple[Path, dict[str, Any]]:
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    if not isinstance(template, dict):
        raise ValueError(f"SUMO template is not a mapping: {template_path}")
    config_path = Path(str((row.get("evidence") or {}).get("config_path") or ""))
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    graph = resolve_sumo_input_graph(config_path)
    native_tls_ids = sorted(parsed_program_ids_by_tls(graph))
    if not native_tls_ids:
        raise ValueError("SUMO source graph exposes no native TLS programs")
    route_rows = list(graph.get("route_files") or [])
    if len(route_rows) != 1:
        raise ValueError("SUMO365 candidate requires one source demand route file")
    route_path = Path(str(route_rows[0]["path"]))
    event = _demand_change_window(parse_route_departures(route_path))
    if event.get("status") != "passed":
        raise ValueError(f"source-native demand transition unavailable: {event}")
    begin = int(event["begin"])
    end = int(event["end"])
    horizon = max(2, (end - begin) // 60)
    trigger_tick = max(1, (int(event["event_time"]) - begin) // 60)
    service_date = str(row.get("source_unit") or config_path.stem)
    candidate_slug = _slug(str(row["candidate_id"]))
    scenario_id = (
        f"traffic/signal_coordination/source_locked_native_window/high/{candidate_slug}"
    )
    assets = _sumo_graph_rows(graph)
    refs = [_relative(Path(str(asset["path"]))) for asset in assets]
    hashes = {
        _relative(Path(str(asset["path"]))): str(asset["sha256"]) for asset in assets
    }
    license_ref: str | None = None
    if family == "resco":
        license_path = config_path.parent / "LICENSE"
        if not license_path.is_file():
            raise FileNotFoundError(
                f"RESCO environment license is missing: {license_path}"
            )
        license_ref = _relative(license_path)
    body = copy.deepcopy(template)
    body.update(
        {
            "seed_id": scenario_id,
            "scenario_id": scenario_id,
            "difficulty_level": "high",
            "difficulty_mode": "deep_planning",
            "horizon_ticks": horizon,
            "seed": 9616,
            "net_ref": _relative(Path(str(graph["network"]["path"]))),
            "route_ref": _relative(route_path),
            "corridors": [],
            "perturbations": [],
            "source_contract": {
                "runtime_input": refs,
                "derivation_input": [],
                "license": [license_ref] if license_ref is not None else [],
                "file_sha256s": hashes,
                "derived_window": {
                    "sha256": hashlib.sha256(
                        json.dumps(event, sort_keys=True).encode()
                    ).hexdigest(),
                    "recipe_version": "sumo365_route_departure_change_v1",
                },
            },
            "candidate_only": True,
            "release_admission": False,
        }
    )
    config = body["backend_config"]
    config.update(
        {
            "sumo_net_path": _relative(Path(str(graph["network"]["path"]))),
            "sumo_route_path": _relative(route_path),
            "sumo_config_path": _relative(config_path),
            # This legacy backend field is also the allow-list for model-visible
            # native TLS state.  Self-map exact source IDs without synthesizing
            # corridor entities or exposing backend-private state.
            "corridor_tls_map": {tls_id: tls_id for tls_id in native_tls_ids},
            "sumo_tls_binding_net_sha256": str(graph["network"]["sha256"]),
            "source_denominator_key": (
                f"sumo365:{service_date}:{event['event_time']}:native_phase_duration"
            ),
            "service_date": service_date,
            "sumo_extra_args": [
                "--begin",
                str(begin),
                "--end",
                str(end),
            ],
            "reference_policy_contract": "sumo.native_phase_duration_window.v1",
            "reference_phase_duration_seconds": 20.0,
            "source_event_registry": {
                "traffic_demand_change": {
                    "event_class": "task",
                    "actionable_ticks": [trigger_tick],
                    "materiality_metric": "interval_vehicle_flow",
                    "materiality_threshold": 1,
                    "response_window_ticks": 4,
                }
            },
            "task_contract": {
                "contract": "traffic.travel_delay_mitigation.v1",
                "objective_component": "travel_time_cost",
                "reference_policy": "deterministic_wait_only",
                "source_response_windows": [
                    {
                        "event_type": "traffic_demand_change",
                        "event_tick": trigger_tick,
                        "first_tick": trigger_tick,
                        "last_tick": min(horizon - 1, trigger_tick + 4),
                    }
                ],
            },
            "task_requirements": {
                "min_distinct_control_ticks": 1,
                "min_distinct_physical_tools": 1,
                "ordered_tool_milestones": [
                    {
                        "tool": "set_signal_phase_duration",
                        "any_state_changing": True,
                        "not_before_tick": trigger_tick,
                        "not_after_tick": min(horizon - 1, trigger_tick + 4),
                    }
                ],
            },
        }
    )
    body["world_evolution_contract"] = {
        "source_channel": "sumocfg_route_departures",
        "runtime_event_type": "traffic_demand_change",
        "origin": "source_schedule",
        "source_event_window": event,
        "source_route_sha256": str(route_rows[0]["sha256"]),
        "proof_required": (
            "Runtime departures must change live vehicle state before native signal control."
        ),
    }
    body.setdefault("provenance", {}).update(
        {
            "data_source": f"{family}:{service_date}",
            "files": [*refs, *([license_ref] if license_ref is not None else [])],
            "url": (
                "https://github.com/Pi-Star-Lab/RESCO"
                if family == "resco"
                else "https://github.com/TUM-VT/sumo_ingolstadt"
            ),
            "commit": (
                "f1ed9a174f8de41fc9d8689373b836bc882570dc"
                if family == "resco"
                else "e0a95deebe200ff81b6705044d66310d6266d42b"
            ),
            "time_window": {
                "service_date": service_date,
                "window_begin_seconds": begin,
                "event_time_seconds": int(event["event_time"]),
                "window_end_seconds": end,
            },
            "notes": (
                "The event is a material change in locked route departures; no synthetic "
                "traffic perturbation is added."
            ),
        }
    )
    if family == "resco":
        body["provenance"].update(
            {
                "license": (
                    f"CC-BY-NC-SA-4.0 (RESCO {service_date} environment assets); "
                    "GPL-3.0-only (RESCO repository software)"
                ),
                "lock_strategy": (
                    "git_commit+per_file_sha256+environment_license_sha256+"
                    "derived_window"
                ),
            }
        )
    body["scenario_signature"] = recompute_signature_with_seed(body, 9616)
    output = (
        output_root
        / "scenarios/traffic/signal_coordination/source_locked_native_window/high"
        / f"{candidate_slug}.yaml"
    )
    return output, body


def _probe_sumo_body(body: dict[str, Any]) -> dict[str, Any]:
    from core import Action, ToolCall
    from domains.traffic.adapter import TrafficEnvironment

    previous = os.environ.get("OPERATE_TRAFFIC_BACKEND_REAL")
    os.environ["OPERATE_TRAFFIC_BACKEND_REAL"] = "1"
    env = TrafficEnvironment()
    try:
        env.reset(body, seed=int(body["seed"]))
        backend = env._backend
        source_event: dict[str, Any] | None = None
        source_evidence_id = ""
        for _ in range(int(body["horizon_ticks"])):
            step = env.step(Action())
            source_event = next(
                (
                    event
                    for event in step.info.realized_events
                    if isinstance(event, dict)
                    and event.get("type") == "traffic_demand_change"
                    and event.get("decision_required") is True
                ),
                None,
            )
            if source_event is None:
                continue
            source_event_id = str(source_event.get("event_id") or "")
            source_evidence_id = next(
                (
                    evidence_id
                    for evidence_id, row in (
                        env._visible_source_events_by_evidence_id.items()
                    )
                    if str(row.get("event_id") or "") == source_event_id
                ),
                "",
            )
            if source_evidence_id:
                break
        if source_event is None or not source_evidence_id:
            raise RuntimeError("SUMO source transition emitted no actionable evidence")
        contract = getattr(backend, "_runtime_control_contract", None) or {}
        tls_ids = sorted(contract.get("tls") or {})
        if not tls_ids:
            raise RuntimeError("SUMO runtime exposed no TLS control")
        selected_tls = ""
        runtime: dict[str, Any] = {}
        duration: float | None = None
        opportunity_tick = int(
            source_event.get("response_opportunity_tick") or 0
        )
        deadline_tick = int(source_event.get("response_deadline_tick") or 0)
        max_waits = max(0, deadline_tick - opportunity_tick)
        decision_wait_ticks = 0
        for wait_ticks in range(max_waits + 1):
            for tls_id in tls_ids:
                query = backend.apply_tool_effect(
                    "query_signal_control", {"tls_id": tls_id}
                )
                candidate = (query.get("tls") or {}).get(tls_id) or {}
                state = str(candidate.get("current_state") or "")
                if not any(signal in {"g", "G"} for signal in state) or any(
                    signal in {"y", "Y"} for signal in state
                ):
                    continue
                bounds = candidate.get("current_phase_bounds") or {}
                current = float(candidate.get("remaining_duration") or 0.0)
                choices = [
                    float(bounds.get("min_duration") or 0.0),
                    float(bounds.get("max_duration") or 0.0),
                ]
                duration = next(
                    (
                        value
                        for value in choices
                        if value > 0 and abs(value - current) >= 1.0
                    ),
                    None,
                )
                if duration is not None:
                    selected_tls = tls_id
                    runtime = candidate
                    decision_wait_ticks = wait_ticks
                    break
            if selected_tls or wait_ticks == max_waits:
                break
            env.step(Action())
        if not selected_tls or duration is None:
            raise RuntimeError("SUMO TLS has no live green bounded duration action")
        call_id = "sumo-source-response-probe"
        response = env.step(
            Action(
                tool_calls=[
                    ToolCall(
                        name="set_signal_phase_duration",
                        args={
                            "tls_id": selected_tls,
                            "observed_program": runtime["current_program"],
                            "observed_phase": runtime["current_phase"],
                            "remaining_duration_seconds": duration,
                        },
                        call_id=call_id,
                        consumes_evidence_ids=[source_evidence_id],
                    )
                ]
            )
        )
        result = next(
            (
                item
                for item in response.tool_results
                if item.call_id == call_id
            ),
            None,
        )
        effect = next(
            (
                event
                for event in response.info.realized_events
                if isinstance(event, dict)
                and event.get("origin") == "agent_caused"
                and event.get("call_id") == call_id
            ),
            None,
        )
        changed = bool(
            result is not None
            and result.ok
            and result.state_changing
            and effect is not None
            and effect.get("before_state_digest")
            != effect.get("after_state_digest")
        )
        causal_action_binding = bool(
            changed
            and effect is not None
            and effect.get("causal_parent_event_id")
            == source_event.get("event_id")
        )
        return {
            "status": "passed" if causal_action_binding else "failed",
            "reset": True,
            "native_tool": "set_signal_phase_duration",
            "tls_id": selected_tls,
            "decision_wait_ticks": decision_wait_ticks,
            "state_changing": changed,
            "tool_status": (
                (result.payload or {}).get("_status")
                if result is not None
                else None
            ),
            "source_event_id": source_event.get("event_id"),
            "effect_event_id": effect.get("event_id") if effect else None,
            "causal_action_binding": causal_action_binding,
        }
    finally:
        env.close()
        if previous is None:
            os.environ.pop("OPERATE_TRAFFIC_BACKEND_REAL", None)
        else:
            os.environ["OPERATE_TRAFFIC_BACKEND_REAL"] = previous


def _body_contract_blockers(body: dict[str, Any], *, repo_root: Path) -> list[str]:
    blockers = [f"scenario_schema:{error}" for error in validate_scenario_yaml(body)]
    seed = body.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        blockers.append("scenario_seed_invalid")
    else:
        expected = recompute_signature_with_seed(body, seed)
        if body.get("scenario_signature") != expected:
            blockers.append("scenario_signature_mismatch")

    source_contract = resolve_source_asset_contract(body, repo_root=repo_root)
    blockers.extend(
        f"source_contract:{code}" for code in source_contract.contract_errors
    )
    blockers.extend(
        f"source_file_missing:{path}" for path in source_contract.missing_required_files
    )

    backend_kind = str(body.get("backend_kind") or "")
    try:
        capability = get_backend_capability(backend_kind)
    except KeyError:
        blockers.append("backend_capability_unregistered")
        return blockers
    if not capability.formal_core_allowed:
        blockers.append("backend_not_formal_core_capable")
    if not capability.control_tools:
        blockers.append("native_control_tools_unregistered")
    if not capability.source_scheduled_event_types:
        blockers.append("typed_backend_event_contract_missing")

    requirements = (body.get("backend_config") or {}).get("task_requirements") or {}
    milestones = requirements.get("ordered_tool_milestones") or []
    if not isinstance(milestones, list) or not milestones:
        blockers.append("ordered_native_task_contract_missing")
    else:
        registered = set(capability.control_tools)
        for milestone in milestones:
            tool = str((milestone or {}).get("tool") or "")
            if tool not in registered:
                blockers.append(f"task_tool_unregistered:{tool or 'missing'}")

    if backend_kind == "pglib_uc_synthetic":
        for event in body.get("perturbations") or []:
            kind = str((event or {}).get("kind") or "")
            if kind not in (
                _PGLIB_UC_TYPED_PERTURBATIONS | _PGLIB_UC_SILENT_SOURCE_MODIFIERS
            ):
                blockers.append(f"typed_event_route_missing:{kind or 'missing'}")
    return sorted(set(blockers))


def _blocker(
    row: dict[str, Any],
    *,
    code: str,
    missing: tuple[str, ...] | list[str],
    builder_route: str,
    detail: str,
) -> dict[str, Any]:
    return {
        "candidate_id": str(row["candidate_id"]),
        "source_family": str(row.get("source_family") or row.get("source_id") or ""),
        "source_unit": str(row.get("source_unit") or ""),
        "domain": str(row.get("domain") or ""),
        "status": "blocked",
        "blocker_code": code,
        "missing_contract_components": list(missing),
        "builder_route": builder_route,
        "detail": detail,
        "refinement_evidence": row.get("evidence") or {},
    }


def _static_family_blocker(row: dict[str, Any]) -> dict[str, Any]:
    family = str(row.get("source_family") or row.get("source_id") or "")
    code, missing, route = _FAMILY_BLOCKERS.get(
        family,
        (
            "current_tree_materializer_unavailable",
            (
                "scenario_builder",
                "source_contract",
                "task_contract",
                "reference_policy",
            ),
            "unregistered",
        ),
    )
    return _blocker(
        row,
        code=code,
        missing=missing,
        builder_route=route,
        detail=(
            "The refinement row proves structural candidacy only. Current-tree code does not "
            "yet produce all executable candidate contracts, so no YAML is emitted."
        ),
    )


def build_delta(
    *,
    refinement_ledger: Path,
    output_root: Path,
    active_suite: Path,
    pglib_uc_root: Path,
    pglib_opf_root: Path,
    repo_root: Path = ROOT,
    native_probe: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    citylearn_probe: Callable[..., dict[str, Any]] | None = None,
    citylearn_lock_builder: Callable[..., dict[str, Any]] = build_citylearn_lock,
    sumo_probe: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> DeltaBuild:
    ledger = _load_object(refinement_ledger, "refinement ledger")
    rows = ledger.get("rows")
    if not isinstance(rows, list):
        raise ValueError("refinement ledger must contain a rows list")
    candidate_ids = [
        str(row.get("candidate_id") or "") for row in rows if isinstance(row, dict)
    ]
    if any(not candidate_id for candidate_id in candidate_ids):
        raise ValueError("every refinement row requires candidate_id")
    duplicates = sorted(
        candidate_id
        for candidate_id, count in Counter(candidate_ids).items()
        if count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate candidate_id values: {duplicates}")
    if not active_suite.is_file():
        raise FileNotFoundError(active_suite)

    eligible = sorted(
        (
            row
            for row in rows
            if isinstance(row, dict)
            and row.get("classification_scope") == "candidate"
            and row.get("final_disposition") == "ready_for_full_admission"
        ),
        key=lambda row: str(row["candidate_id"]),
    )
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in eligible:
        family = str(row.get("source_family") or row.get("source_id") or "")
        by_family.setdefault(family, []).append(row)

    files: dict[Path, dict[str, Any]] = {}
    json_files: dict[Path, dict[str, Any]] = {}
    materialized: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    uc_rows = {
        str(row.get("source_unit") or ""): row for row in by_family.pop("pglib_uc", [])
    }
    if uc_rows:
        try:
            _, _, built_files = build_pglib_bulk(
                uc_root=pglib_uc_root,
                opf_root=pglib_opf_root,
                base_core=active_suite,
                staging_root=output_root / "scenarios",
                report_path=output_root / "_pglib_uc_builder_report.json",
                suite_path=output_root / "_pglib_uc_builder_suite.json",
            )
        except Exception as exc:
            for row in uc_rows.values():
                blockers.append(
                    _blocker(
                        row,
                        code="pglib_uc_builder_failed",
                        missing=("scenario_builder_output",),
                        builder_route="scripts.build_pglib_bulk_protocol21_candidates",
                        detail=f"{type(exc).__name__}: {exc}",
                    )
                )
        else:
            seen_units: set[str] = set()
            for path, body in built_files.items():
                source_unit = _source_unit_from_body(body, pglib_uc_root)
                if source_unit not in uc_rows:
                    continue
                row = uc_rows[source_unit]
                contract_blockers = _body_contract_blockers(body, repo_root=repo_root)
                if contract_blockers:
                    blockers.append(
                        _blocker(
                            row,
                            code="pglib_uc_executable_contract_incomplete",
                            missing=contract_blockers,
                            builder_route="scripts.build_pglib_bulk_protocol21_candidates",
                            detail="Existing builder output failed current-tree executable contract checks.",
                        )
                    )
                    seen_units.add(source_unit)
                    continue
                files[path] = body
                materialized.append(
                    {
                        "candidate_id": str(row["candidate_id"]),
                        "source_family": "pglib_uc",
                        "source_unit": source_unit,
                        "domain": str(row.get("domain") or "power_grid"),
                        "status": "materialized_candidate",
                        "scenario_id": str(body["scenario_id"]),
                        "scenario_signature": str(body["scenario_signature"]),
                        "path": _relative(path),
                        "builder_route": "scripts.build_pglib_bulk_protocol21_candidates",
                        "prompt_mode": "strict",
                        "release_admission": False,
                    }
                )
                seen_units.add(source_unit)
            for source_unit, row in uc_rows.items():
                if source_unit in seen_units:
                    continue
                source_path = pglib_uc_root / source_unit
                blockers.append(
                    _blocker(
                        row,
                        code=(
                            "pglib_uc_source_file_missing"
                            if not source_path.is_file()
                            else "pglib_uc_builder_produced_no_candidate"
                        ),
                        missing=("source_file",)
                        if not source_path.is_file()
                        else ("builder_output",),
                        builder_route="scripts.build_pglib_bulk_protocol21_candidates",
                        detail=(
                            f"Source file unavailable: {source_path}"
                            if not source_path.is_file()
                            else "The source is absent from the builder output, including when already "
                            "bound by the active suite."
                        ),
                    )
                )

    opendss_rows = by_family.pop("opendss_distribution", [])
    opendss_probe = native_probe or _probe_opendss_body
    for row in opendss_rows:
        try:
            path, body = _build_opendss_body(
                row,
                output_root=output_root,
                template_path=DEFAULT_OPENDSS_TEMPLATE,
            )
            contract_blockers = _body_contract_blockers(body, repo_root=repo_root)
            if contract_blockers:
                raise ValueError(f"contract blockers: {contract_blockers}")
            probe = opendss_probe(body)
            if (
                probe.get("status") != "passed"
                or probe.get("reset") is not True
                or probe.get("state_changing") is not True
            ):
                raise RuntimeError(f"native reset/tool-effect probe failed: {probe}")
            reference_actions = probe.get("reference_actions")
            manual_takeover = isinstance(
                body["backend_config"].get("manual_takeover_contract"), dict
            )
            causal_actions_valid = isinstance(reference_actions, list) and bool(
                reference_actions
            )
            if causal_actions_valid:
                causal_actions_valid = all(
                    isinstance(action, dict)
                    and isinstance(action.get("args"), dict)
                    and action.get("tool") == probe.get("native_tool")
                    for action in reference_actions
                )
            if causal_actions_valid and manual_takeover:
                causal_actions_valid = all(
                    int(action.get("tick", -1)) == 0 for action in reference_actions
                )
            elif causal_actions_valid:
                causal_actions_valid = (
                    len(reference_actions) == 1
                    and [action.get("tick") for action in reference_actions] == [1]
                )
            if not causal_actions_valid:
                raise RuntimeError(
                    f"native probe lacks a causal response action: {probe}"
                )
            for milestone, action in zip(
                body["backend_config"]["task_requirements"]["ordered_tool_milestones"],
                reference_actions,
            ):
                milestone["tool"] = action["tool"]
            body["scenario_signature"] = recompute_signature_with_seed(
                body, int(body["seed"])
            )
        except Exception as exc:
            blockers.append(
                _blocker(
                    row,
                    code="opendss_executable_probe_failed",
                    missing=("scenario_reset", "native_tool_state_effect"),
                    builder_route=(
                        "domains.power_grid.backends.opendss_fresh_feeders:"
                        "OpenDssFreshFeedersBackend"
                    ),
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        files[path] = body
        materialized.append(
            {
                "candidate_id": str(row["candidate_id"]),
                "source_family": "opendss_distribution",
                "source_unit": str(row["source_unit"]),
                "domain": str(row.get("domain") or "power_grid"),
                "status": "materialized_candidate",
                "scenario_id": str(body["scenario_id"]),
                "scenario_signature": str(body["scenario_signature"]),
                "path": _relative(path),
                "builder_route": (
                    "domains.power_grid.backends.opendss_fresh_feeders:"
                    "OpenDssFreshFeedersBackend"
                ),
                "prompt_mode": "strict",
                "release_admission": False,
                "native_probe": probe,
            }
        )

    sumo_native_probe = sumo_probe or _probe_sumo_body
    for family in ("sumo365_ingolstadt", "resco"):
        for row in by_family.pop(family, []):
            try:
                path, body = _build_sumo365_body(
                    row,
                    family=family,
                    output_root=output_root,
                    template_path=DEFAULT_SUMO_TEMPLATE,
                )
                contract_blockers = _body_contract_blockers(
                    body, repo_root=repo_root
                )
                if contract_blockers:
                    raise ValueError(f"contract blockers: {contract_blockers}")
                probe = sumo_native_probe(body)
                if (
                    probe.get("status") != "passed"
                    or probe.get("reset") is not True
                    or probe.get("state_changing") is not True
                    or probe.get("causal_action_binding") is not True
                ):
                    raise RuntimeError(
                        f"native source-to-TLS causal probe failed: {probe}"
                    )
            except Exception as exc:
                blockers.append(
                    _blocker(
                        row,
                        code="sumo_source_transition_causal_probe_failed",
                        missing=(
                            "typed_actionable_source_event",
                            "causal_action_receipt_binding",
                            "native_tls_effect_attribution",
                        ),
                        builder_route=(
                            "scripts.mine_sumo365_native_traffic+"
                            "domains.traffic.backends.sumo_backend:SumoTrafficBackend"
                        ),
                        detail=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            files[path] = body
            materialized.append(
                {
                    "candidate_id": str(row["candidate_id"]),
                    "source_family": family,
                    "source_unit": str(row["source_unit"]),
                    "domain": str(row.get("domain") or "traffic"),
                    "status": "materialized_candidate",
                    "scenario_id": str(body["scenario_id"]),
                    "scenario_signature": str(body["scenario_signature"]),
                    "path": _relative(path),
                    "builder_route": (
                        "scripts.mine_sumo365_native_traffic+"
                        "domains.traffic.backends.sumo_backend:SumoTrafficBackend"
                    ),
                    "prompt_mode": "strict",
                    "release_admission": False,
                    "native_probe": probe,
                }
            )

    citylearn_rows = by_family.pop("citylearn", [])
    citylearn_native_probe = citylearn_probe or _probe_citylearn_body
    for row in citylearn_rows:
        try:
            last_error: Exception | None = None
            for window_start in (None, 72, 144, 216, 288, 360, 432):
                try:
                    path, body, lock_path, lock = _build_citylearn_body(
                        row,
                        output_root=output_root,
                        template_path=DEFAULT_CITYLEARN_TEMPLATE,
                        lock_builder=citylearn_lock_builder,
                        window_start=window_start,
                    )
                    contract_blockers = _body_contract_blockers(
                        body, repo_root=repo_root
                    )
                    if contract_blockers:
                        raise ValueError(f"contract blockers: {contract_blockers}")
                    probe = citylearn_native_probe(body, source_lock_payload=lock)
                    if (
                        probe.get("status") != "passed"
                        or probe.get("reset") is not True
                        or probe.get("state_changing") is not True
                    ):
                        raise RuntimeError(
                            f"native reset/tool-effect probe failed: {probe}"
                        )
                    break
                except Exception as exc:
                    last_error = exc
                    if "demand is greater than" not in str(exc):
                        raise
            else:
                raise RuntimeError(
                    f"bounded CityLearn source-window repair exhausted: {last_error}"
                )
        except Exception as exc:
            blockers.append(
                _blocker(
                    row,
                    code="citylearn_executable_probe_failed",
                    missing=(
                        "complete_native_decision_channels",
                        "scenario_reset",
                        "native_tool_state_effect",
                    ),
                    builder_route=(
                        "scripts.refine_citylearn_long_horizon_candidates+"
                        "domains.building_energy.adapter:BuildingEnergyEnvironment"
                    ),
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        files[path] = body
        json_files[lock_path] = lock
        materialized.append(
            {
                "candidate_id": str(row["candidate_id"]),
                "source_family": "citylearn",
                "source_unit": str(row["source_unit"]),
                "domain": str(row.get("domain") or "building_energy"),
                "status": "materialized_candidate",
                "scenario_id": str(body["scenario_id"]),
                "scenario_signature": str(body["scenario_signature"]),
                "path": _relative(path),
                "builder_route": (
                    "scripts.refine_citylearn_long_horizon_candidates+"
                    "domains.building_energy.adapter:BuildingEnergyEnvironment"
                ),
                "prompt_mode": "strict",
                "release_admission": False,
                "native_probe": probe,
            }
        )

    for family_rows in by_family.values():
        blockers.extend(_static_family_blocker(row) for row in family_rows)

    materialized.sort(key=lambda row: row["candidate_id"])
    blockers.sort(key=lambda row: row["candidate_id"])
    if set(row["candidate_id"] for row in materialized) & set(
        row["candidate_id"] for row in blockers
    ):
        raise RuntimeError("candidate cannot be both materialized and blocked")
    n_terminal = len(materialized) + len(blockers)
    if n_terminal != len(eligible):
        raise RuntimeError(
            "eligible infrastructure candidates were not terminally accounted"
        )

    candidate_report = {
        "schema_version": "infrastructure-candidate-report-v1",
        "status": "staging_candidates_pending_full_admission",
        "candidate_only": True,
        "release_admission": False,
        "active_core_modified": False,
        "constraints": {
            "prompt_mode": "strict",
            "full_protocol21_required": True,
            "model_outcomes_used_for_materialization": False,
        },
        "scenarios": [
            {
                "scenario_id": row["scenario_id"],
                "scenario_signature": row["scenario_signature"],
                "path": row["path"],
                "candidate_id": row["candidate_id"],
                "source_unit": row["source_unit"],
                "status": "pending_protocol21_full_admission",
            }
            for row in materialized
        ],
    }
    report = {
        "schema_version": "infrastructure-candidate-delta-v1",
        "status": "complete_candidate_delta",
        "candidate_only": True,
        "release_admission": False,
        "active_core_modified": False,
        "input_bindings": {
            "refinement_ledger": {
                "path": _relative(refinement_ledger),
                "sha256": _sha256(refinement_ledger),
            },
            "active_suite": {
                "path": _relative(active_suite),
                "sha256": _sha256(active_suite),
                "read_only": True,
            },
        },
        "policy": {
            "eligible_scope": "classification_scope=candidate AND final_disposition=ready_for_full_admission",
            "strict_prompt_required": True,
            "complete_executable_contract_required": True,
            "structural_probe_is_not_materialization": True,
            "blocked_rows_are_never_dropped": True,
        },
        "materialized": materialized,
        "blockers": blockers,
        "summary": {
            "n_input_rows": len(rows),
            "n_ready_for_full_admission": len(eligible),
            "n_materialized": len(materialized),
            "n_blocked": len(blockers),
            "n_terminal": n_terminal,
            "n_unresolved": 0,
        },
    }
    suite_preview = {
        "schema_version": "protocol2.1-working-set-v1",
        "status": "pending_execute",
        "leaderboard_eligible": False,
        "release_ready": False,
        "n_scenarios": len(materialized),
    }
    return DeltaBuild(
        report=report,
        candidate_report=candidate_report,
        suite_preview=suite_preview,
        files=files,
        json_files=json_files,
        output_root=output_root,
    )


def _write_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite candidate artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def execute_delta(result: DeltaBuild) -> dict[str, Any]:
    for path, body in result.json_files.items():
        _write_new(path, json.dumps(body, indent=2, sort_keys=True) + "\n")
    for path, body in result.files.items():
        _write_new(path, yaml.safe_dump(body, sort_keys=False))
    report_path = result.output_root / "candidate_report.json"
    _write_new(
        report_path,
        json.dumps(result.candidate_report, indent=2, sort_keys=True) + "\n",
    )
    suite = build_suite(report_path)
    suite["constraints"] = {
        **(suite.get("constraints") or {}),
        "prompt_mode": "strict",
        "active_core_modified": False,
        "complete_executable_contract_required": True,
    }
    suite["input_bindings"] = result.report["input_bindings"]
    _write_new(
        result.output_root / "source_suite.json",
        json.dumps(suite, indent=2, sort_keys=True) + "\n",
    )
    _write_new(
        result.output_root / "materialization_ledger.json",
        json.dumps(result.report, indent=2, sort_keys=True) + "\n",
    )
    return suite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refinement-ledger", type=Path, default=DEFAULT_REFINEMENT_LEDGER
    )
    parser.add_argument("--active-suite", type=Path, default=DEFAULT_ACTIVE_SUITE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--pglib-uc-root", type=Path, default=ROOT / "works/pglib-uc")
    parser.add_argument("--pglib-opf-root", type=Path, default=ROOT / "works/PGLib-OPF")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = build_delta(
        refinement_ledger=args.refinement_ledger.resolve(),
        output_root=args.output_root.resolve(),
        active_suite=args.active_suite.resolve(),
        pglib_uc_root=args.pglib_uc_root.resolve(),
        pglib_opf_root=args.pglib_opf_root.resolve(),
    )
    if args.execute:
        execute_delta(result)
    print(json.dumps(result.report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
