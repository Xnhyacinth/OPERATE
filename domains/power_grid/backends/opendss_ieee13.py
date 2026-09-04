"""OpenDSS IEEE 13-node non-release live probe backend.

This module is intentionally smaller than the formal OPERATE backend surface. It
loads the SHA-256-pinned ``works/OpenDSS-IEEE13`` anchor through
``dss-python`` when that optional runtime is installed, solves the real
OpenDSS power flow, and exposes a tiny Volt-Var control surface for source
integration probes. It does not materialize scenarios or modify any release
suite.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from types import MappingProxyType
from typing import Any

from core import (
    Action,
    EvidenceLogger,
    POMDPEnvironment,
    StepInfo,
    StepReturn,
    TickBudget,
    ToolContext,
    ToolRegistry,
    ToolSpec,
)
from core.protocol21_evidence import canonicalize_repo_owned_paths

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "works" / "OpenDSS-IEEE13"
MASTER_FILE = "13Bus/IEEE13Nodeckt.dss"
# The electricdss-tst clone nests IEEE test cases under
# ``Version8/Distrib/IEEETestCases/`` (the manifest's locked ``upstream_path``).
# Some checkouts flatten this; keep both candidates so the backend resolves
# regardless of which layout ``works/OpenDSS-IEEE13`` has.
_NESTED_PREFIX = "Version8/Distrib/IEEETestCases"
VOLTAGE_LOWER_PU = 0.95
VOLTAGE_UPPER_PU = 1.05
OPENDSS_IEEE13_EVENT_CLASS_REGISTRY = MappingProxyType(
    {
        "load_surge": "alarm",
        "load_surge_cleared": "lifecycle",
    }
)
_INCLUDE_RE = re.compile(
    r"""^\s*(redirect|compile)\s+
        (?:\(([^)]+)\)|\[([^]]+)\]|"([^"]+)"|'([^']+)'|([^\s!]+))""",
    re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)
_NEW_OBJECT_RE = re.compile(
    r"""^\s*new\s+(?:object\s*=\s*)?(?:[\"'])?
    ([a-z][a-z0-9_]*)\.([^\s\"']+)(?:[\"'])?""",
    re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)
_DATA_FILE_RE = re.compile(
    r"""\bfile\s*=\s*(?:\(\s*)?
        (?:"([^"]+)"|'([^']+)'|([^,\s\)!\]]+))""",
    re.IGNORECASE | re.VERBOSE,
)
_AUXILIARY_INPUT_RE = re.compile(
    r"""^\s*(?:buscoords|latlongcoords|guids)\s+
    (?:file\s*=\s*)?(?:\(\s*)?
    (?:"([^"]+)"|'([^']+)'|([^\s\)!]+))""",
    re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)


def _semantic_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_native_include_graph(
    master_file: Path,
    *,
    windows_compile_aliases: dict[str, Path] | None = None,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Resolve the exact Compile/Redirect graph that OpenDSS will execute.

    This is intentionally independent of scenario provenance declarations.
    Missing recursive inputs fail before launching the solver.
    """
    master = master_file.resolve()
    queued: list[tuple[Path, str, str | None, str | None]] = [
        (master, "compile_master", None, None)
    ]
    visited: set[Path] = set()
    assets: list[dict[str, str]] = []
    inventory: Counter[str] = Counter()
    while queued:
        path, role, included_from, declared_source_path = queued.pop(0)
        path = path.resolve()
        if path in visited:
            continue
        if not path.is_file():
            raise FileNotFoundError(
                f"missing OpenDSS Compile/Redirect input: {path}"
            )
        visited.add(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        asset = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "role": role,
        }
        if included_from is not None:
            asset["included_from"] = included_from
        if declared_source_path is not None:
            asset["declared_source_path"] = declared_source_path
        assets.append(asset)
        inventory.update(
            match.group(1).lower() for match in _NEW_OBJECT_RE.finditer(text)
        )
        for match in _INCLUDE_RE.finditer(text):
            directive = match.group(1).lower()
            raw = next(value for value in match.groups()[1:] if value)
            declared_path = raw.strip()
            include_path = (path.parent / declared_path).resolve()
            source_path_alias: str | None = None
            if (
                directive == "compile"
                and re.match(r"^[A-Za-z]:[\\\\/]", declared_path)
                and windows_compile_aliases is not None
            ):
                alias = windows_compile_aliases.get(
                    PureWindowsPath(declared_path).name.lower()
                )
                if alias is not None:
                    include_path = Path(alias).resolve()
                    source_path_alias = declared_path
            queued.append(
                (
                    include_path,
                    f"{directive}_input",
                    str(path),
                    source_path_alias,
                )
            )
        for match in _AUXILIARY_INPUT_RE.finditer(text):
            raw = next(value for value in match.groups() if value)
            queued.append(
                (
                    (path.parent / raw.strip()).resolve(),
                    "runtime_auxiliary_input",
                    str(path),
                    None,
                )
            )
        for match in _DATA_FILE_RE.finditer(text):
            raw = next(value for value in match.groups() if value)
            queued.append(
                (
                    (path.parent / raw.strip()).resolve(),
                    "runtime_data_input",
                    str(path),
                    None,
                )
            )
    return assets, dict(sorted(inventory.items()))


def _source_hash_keys(path: str, digest: str) -> dict[str, str]:
    actual = Path(path)
    keys = {str(actual): digest}
    if actual.is_relative_to(REPO_ROOT):
        relative = str(actual.relative_to(REPO_ROOT))
        keys[relative] = digest
        nested_prefix = (
            "works/OpenDSS-IEEE13/Version8/Distrib/IEEETestCases/"
        )
        if relative.startswith(nested_prefix):
            suffix = relative.removeprefix(nested_prefix)
            if suffix.startswith("13Bus/"):
                keys[f"works/OpenDSS-IEEE13/{suffix}"] = digest
            if suffix.startswith(("34Bus/", "123Bus/")):
                keys[f"works/OpenDSS-IEEE34-IEEE123/{suffix}"] = digest
            if suffix == "IEEELineCodes.DSS":
                keys["works/OpenDSS-IEEE13/IEEELineCodes.DSS"] = digest
    return keys


def _native_protocol21_trace(
    *,
    assets: list[dict[str, str]],
    inventory: dict[str, int],
    dss_version: str | None,
    circuit: Any,
    summary: dict[str, Any],
    source_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime_assets = assets
    assets = canonicalize_repo_owned_paths(assets, repo_root=REPO_ROOT)
    profile = canonicalize_repo_owned_paths(
        dict(source_profile or {}), repo_root=REPO_ROOT
    )
    native_state = {
        "circuit_name": str(circuit.Name),
        **summary,
    }
    parser_output_digest = _semantic_digest(
        {
            "assets": assets,
            "parsed_source_inventory": inventory,
        }
    )
    observed_state_digest = _semantic_digest(native_state)
    initial_state_digest = str(
        profile.get("initial_state_digest") or observed_state_digest
    )
    count_fields = {
        "line": "n_lines",
        "load": "n_loads",
        "transformer": "n_transformers",
        "capacitor": "n_capacitors",
        "regcontrol": "n_regcontrols",
    }
    parsed_native_counts = {
        object_kind: {
            "parsed": int(inventory.get(object_kind, 0)),
            "native": int(native_state.get(state_field) or 0),
        }
        for object_kind, state_field in count_fields.items()
    }
    count_match = all(
        counts["parsed"] == counts["native"]
        for counts in parsed_native_counts.values()
    )
    base_state_effect = bool(
        native_state["circuit_name"]
        and native_state.get("converged") is True
        and int(native_state.get("n_buses") or 0) > 0
        and inventory.get("circuit", 0) > 0
        and count_match
    )
    profile_state_effect = (
        bool(profile.get("runtime_values_match")) if profile else True
    )
    state_effect = base_state_effect and profile_state_effect
    consumed_hashes: dict[str, str] = {}
    for asset in runtime_assets:
        consumed_hashes.update(_source_hash_keys(asset["path"], asset["sha256"]))
    semantic_payload = {
        "dss_version": dss_version,
        "parser_output_digest": parser_output_digest,
        "parsed_native_element_counts": parsed_native_counts,
        "native_solver_state": native_state,
        "source_profile": profile,
    }
    opened_hashes = {asset["path"]: asset["sha256"] for asset in assets}
    consumed_channels = [
        "master_dss",
        "redirected_dss_assets",
        "native_element_definitions",
    ]
    derived_state_fields = [
        "circuit_name",
        "bus_voltage_pu",
        "native_element_counts",
        "solver_converged",
    ]
    source_field_to_state_field_map = {
        "circuit": ["circuit_name", "n_buses", "n_nodes"],
        "line": ["n_lines", "bus_voltage_pu"],
        "load": ["n_loads", "bus_voltage_pu"],
        "transformer": ["n_transformers", "bus_voltage_pu"],
        "capacitor": ["n_capacitors", "bus_voltage_pu"],
        "regcontrol": ["n_regcontrols", "bus_voltage_pu"],
    }
    if profile.get("profile_kind") == "native_duty_program":
        consumed_channels.extend(
            [
                "source_dss_program",
                "loadshape_definition",
                "loadshape_multiplier",
                "generator_duty_schedule",
            ]
        )
        derived_state_fields.extend(
            [
                "aggregate_generation_mw",
                "net_grid_import_mw",
                "voltage_min_pu",
                "voltage_max_pu",
            ]
        )
        source_field_to_state_field_map["generator_duty_schedule"] = [
            "aggregate_generation_mw",
            "net_grid_import_mw",
            "voltage_min_pu",
            "voltage_max_pu",
        ]
    elif profile:
        consumed_channels.extend(
            ["loadshape_definition", "loadshape_multiplier"]
        )
        derived_state_fields.extend(
            ["aggregate_demand_mw", "aggregate_reactive_demand_mvar"]
        )
        source_field_to_state_field_map["loadshape_multiplier"] = [
            "aggregate_demand_mw",
            "aggregate_reactive_demand_mvar",
            "bus_voltage_pu",
        ]
    post_source_state_digests = list(
        profile.get("post_source_state_digests") or []
    )
    if not post_source_state_digests and state_effect:
        post_source_state_digests = [initial_state_digest]
    consumption_ticks = list(profile.get("consumption_ticks") or [])
    if not consumption_ticks and state_effect:
        consumption_ticks = [0]
    runtime_source_events = list(profile.get("runtime_source_events") or [])
    return canonicalize_repo_owned_paths({
        "status": "passed" if state_effect else "held",
        "proof_kind": "native_include_graph",
        "runtime_opened_assets": assets,
        "opened_source_paths": [asset["path"] for asset in assets],
        "opened_source_sha256": opened_hashes,
        "consumed_source_hashes": consumed_hashes,
        "lineage_source_hashes": opened_hashes,
        "consumed_window_sha256": parser_output_digest,
        "recipe_version": "opendss_native_include_graph_v1",
        "parser_output_digest": parser_output_digest,
        "parsed_source_inventory": inventory,
        "parsed_native_element_counts": parsed_native_counts,
        "parsed_native_element_count_match": count_match,
        "native_solver_state": native_state,
        "consumed_channels": consumed_channels,
        "derived_backend_state_fields": derived_state_fields,
        "consumption_ticks": consumption_ticks if state_effect else [],
        "initial_state_digest": initial_state_digest,
        "post_source_state_digests": post_source_state_digests if state_effect else [],
        "source_field_to_state_field_map": source_field_to_state_field_map,
        "source_state_effect_observed": state_effect,
        "state_effect_observed": state_effect,
        "deterministic_source_trace": True,
        "trace_semantic_digest": _semantic_digest(semantic_payload),
        "runtime_trace_observed": state_effect,
        "evidence_from_scenario_config_only": False,
        "source_time_variation_claimed": bool(profile),
        "runtime_source_events": runtime_source_events,
        "blockers": [] if state_effect else ["initial_solver_state_unproven"],
    }, repo_root=REPO_ROOT)


def _resolve_master_file(source_root: Path, master_rel: str) -> Path:
    """Resolve the OpenDSS master file under ``source_root``, checking both
    the flat layout (``<root>/13Bus/IEEE13Nodeckt.dss``) and the nested
    electricdss-tst layout (``<root>/Version8/Distrib/IEEETestCases/13Bus/...``).
    Returns the first existing candidate; falls back to the flat path so the
    downstream ``FileNotFoundError`` carries the expected (flat) message."""
    flat = source_root / master_rel
    if flat.exists():
        return flat
    nested = source_root / _NESTED_PREFIX / master_rel
    if nested.exists():
        return nested
    return flat  # let the caller's existence check raise on the canonical path


class OpenDssRuntimeUnavailable(RuntimeError):
    """Raised when ``dss-python`` is not importable."""


@dataclass
class OpenDssSolveSummary:
    converged: bool
    n_buses: int
    n_nodes: int
    n_loads: int
    n_lines: int
    n_transformers: int
    n_capacitors: int
    n_regcontrols: int
    voltage_min_pu: float | None
    voltage_max_pu: float | None
    n_voltage_violations: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "converged": self.converged,
            "n_buses": self.n_buses,
            "n_nodes": self.n_nodes,
            "n_loads": self.n_loads,
            "n_lines": self.n_lines,
            "n_transformers": self.n_transformers,
            "n_capacitors": self.n_capacitors,
            "n_regcontrols": self.n_regcontrols,
            "voltage_min_pu": self.voltage_min_pu,
            "voltage_max_pu": self.voltage_max_pu,
            "n_voltage_violations": self.n_voltage_violations,
        }


class OpenDssIeee13ProbeBackend:
    """Minimal OpenDSS IEEE13 live adapter for non-release probes."""

    backend_kind = "opendss_ieee13_probe"

    def __init__(self, *, source_root: Path = DEFAULT_SOURCE_ROOT) -> None:
        self.source_root = Path(source_root)
        self.master_file = _resolve_master_file(self.source_root, MASTER_FILE)
        self._dss: Any = None
        self._circuit: Any = None
        self._dss_version: str | None = None
        self._runtime_source_assets: list[dict[str, str]] = []
        self._parsed_source_inventory: dict[str, int] = {}
        self._protocol21_source_evidence: dict[str, Any] | None = None

    @property
    def dss_version(self) -> str | None:
        return self._dss_version

    def reset(self) -> OpenDssSolveSummary:
        self._protocol21_source_evidence = None
        self._runtime_source_assets = []
        self._parsed_source_inventory = {}
        self._circuit = None
        old_cwd = Path.cwd()
        try:
            dss_module = _import_dss()
            self._dss_version = str(getattr(dss_module, "__version__", "unknown"))
            if not self.master_file.exists():
                raise FileNotFoundError(
                    f"missing OpenDSS master file: {self.master_file}"
                )
            (
                self._runtime_source_assets,
                self._parsed_source_inventory,
            ) = _resolve_native_include_graph(self.master_file)
            self._dss = dss_module.DSS.NewContext()
            self._dss.Text.Command = "Clear"
            self._dss.Text.Command = f'Compile "{self.master_file}"'
            self._dss.Text.Command = "Set ControlMode=OFF"
            self._circuit = self._dss.ActiveCircuit
            summary = self.solve()
            self._protocol21_source_evidence = _native_protocol21_trace(
                assets=self._runtime_source_assets,
                inventory=self._parsed_source_inventory,
                dss_version=self._dss_version,
                circuit=self._circuit,
                summary=summary.to_dict(),
            )
            return summary
        finally:
            os.chdir(old_cwd)

    def solve(self) -> OpenDssSolveSummary:
        circuit = self._require_circuit()
        circuit.Solution.Solve()
        voltages = [float(v) for v in circuit.AllBusVmagPu]
        return OpenDssSolveSummary(
            converged=bool(circuit.Solution.Converged),
            n_buses=int(circuit.NumBuses),
            n_nodes=int(circuit.NumNodes),
            n_loads=int(circuit.Loads.Count),
            n_lines=int(circuit.Lines.Count),
            n_transformers=int(circuit.Transformers.Count),
            n_capacitors=int(circuit.Capacitors.Count),
            n_regcontrols=int(circuit.RegControls.Count),
            voltage_min_pu=min(voltages) if voltages else None,
            voltage_max_pu=max(voltages) if voltages else None,
            n_voltage_violations=sum(
                1 for v in voltages if v < VOLTAGE_LOWER_PU or v > VOLTAGE_UPPER_PU
            ),
        )

    def snapshot(self) -> dict[str, Any]:
        circuit = self._require_circuit()
        summary = self.solve().to_dict()
        return {
            "backend_kind": self.backend_kind,
            "dss_version": self._dss_version,
            "master_file": str(self.master_file),
            "circuit_name": str(circuit.Name),
            **summary,
            "capacitors": self._capacitor_states(),
            "regcontrols": self._regcontrol_states(),
            "line_current_max_a": self._line_current_max_a(),
        }

    def protocol21_source_trace(self) -> dict[str, Any]:
        """Return solver-linked evidence for the actual compiled DSS graph."""
        if self._protocol21_source_evidence is None:
            return {
                "status": "held",
                "proof_kind": "native_include_graph",
                "runtime_trace_observed": False,
                "evidence_from_scenario_config_only": False,
                "source_state_effect_observed": False,
                "state_effect_observed": False,
                "blockers": ["backend_not_reset"],
            }
        return dict(self._protocol21_source_evidence)

    def apply_tool_effect(
        self, name: str, args: dict[str, Any], ctx: Any | None = None
    ) -> dict[str, Any]:
        if name == "switch_capacitor":
            return self._with_tool_evidence(name, self._switch_capacitor(args), ctx)
        if name == "set_transformer_tap":
            return self._with_tool_evidence(name, self._set_regulator_tap(args), ctx)
        if name in {"set_der_reactive_power", "set_battery_dispatch"}:
            return self._with_tool_evidence(
                name,
                {
                    "_status": "unsupported_on_ieee13_probe",
                    "tool": name,
                    "reason": (
                        "IEEE13 source probe does not expose DER/storage assets yet"
                    ),
                },
                ctx,
            )
        return self._with_tool_evidence(
            name, {"_status": "error", "error": "unknown_tool", "tool": name}, ctx
        )

    def _switch_capacitor(self, args: dict[str, Any]) -> dict[str, Any]:
        circuit = self._require_circuit()
        cap_id = int(args.get("cap_id", -1))
        status = bool(args.get("status"))
        names = [str(n) for n in circuit.Capacitors.AllNames]
        if cap_id < 0 or cap_id >= len(names):
            return {
                "_status": "error",
                "error": "unknown_controllable_asset",
                "asset": "cap_id",
                "index": cap_id,
                "n_available": len(names),
            }
        name = names[cap_id]
        circuit.Capacitors.Name = name
        before = [int(s) for s in circuit.Capacitors.States]
        circuit.Capacitors.States = [1 if status else 0 for _ in before]
        summary = self.solve()
        after = [int(s) for s in circuit.Capacitors.States]
        return {
            "_status": "applied" if before != after else "no_effect",
            "cap_id": cap_id,
            "capacitor": name,
            "status": status,
            "states_before": before,
            "states_after": after,
            "converged_after_solve": summary.converged,
            "voltage_min_pu": summary.voltage_min_pu,
            "voltage_max_pu": summary.voltage_max_pu,
            "n_voltage_violations": summary.n_voltage_violations,
        }

    def _set_regulator_tap(self, args: dict[str, Any]) -> dict[str, Any]:
        circuit = self._require_circuit()
        reg_id = int(args.get("trafo_id", -1))
        tap_pos = int(args.get("tap_pos", 0))
        names = [str(n) for n in circuit.RegControls.AllNames]
        if reg_id < 0 or reg_id >= len(names):
            return {
                "_status": "error",
                "error": "unknown_controllable_asset",
                "asset": "trafo_id",
                "index": reg_id,
                "n_available": len(names),
            }
        name = names[reg_id]
        circuit.RegControls.Name = name
        before = int(circuit.RegControls.TapNumber)
        circuit.RegControls.TapNumber = tap_pos
        summary = self.solve()
        after = int(circuit.RegControls.TapNumber)
        return {
            "_status": "applied" if before != after else "no_effect",
            "trafo_id": reg_id,
            "regcontrol": name,
            "tap_before": before,
            "tap_after": after,
            "converged_after_solve": summary.converged,
            "voltage_min_pu": summary.voltage_min_pu,
            "voltage_max_pu": summary.voltage_max_pu,
            "n_voltage_violations": summary.n_voltage_violations,
        }

    def _capacitor_states(self) -> list[dict[str, Any]]:
        circuit = self._require_circuit()
        out: list[dict[str, Any]] = []
        idx = circuit.Capacitors.First
        while idx:
            out.append(
                {
                    "cap_id": len(out),
                    "name": str(circuit.Capacitors.Name),
                    "states": [int(s) for s in circuit.Capacitors.States],
                }
            )
            idx = circuit.Capacitors.Next
        return out

    def _regcontrol_states(self) -> list[dict[str, Any]]:
        circuit = self._require_circuit()
        out: list[dict[str, Any]] = []
        idx = circuit.RegControls.First
        while idx:
            out.append(
                {
                    "trafo_id": len(out),
                    "name": str(circuit.RegControls.Name),
                    "transformer": str(circuit.RegControls.Transformer),
                    "tap_number": int(circuit.RegControls.TapNumber),
                }
            )
            idx = circuit.RegControls.Next
        return out

    def _line_current_max_a(self) -> float | None:
        circuit = self._require_circuit()
        maxima: list[float] = []
        idx = circuit.Lines.First
        while idx:
            circuit.SetActiveElement(f"Line.{circuit.Lines.Name}")
            currents = [float(v) for v in circuit.ActiveCktElement.CurrentsMagAng]
            maxima.extend(currents[0::2])
            idx = circuit.Lines.Next
        return max(maxima) if maxima else None

    def _require_circuit(self) -> Any:
        if self._circuit is None:
            raise RuntimeError("OpenDSS IEEE13 backend has not been reset")
        return self._circuit

    def _with_tool_evidence(
        self, tool_name: str, payload: dict[str, Any], ctx: Any | None
    ) -> dict[str, Any]:
        evidence = (
            getattr(ctx, "extra", {}).get("evidence") if ctx is not None else None
        )
        if evidence is None or not hasattr(evidence, "log"):
            return payload
        tick = int(getattr(ctx, "tick", 0) or 0)
        out = dict(payload)
        out["evidence_id"] = evidence.log(
            kind="opendss_tool_effect",
            tick=tick,
            payload={
                "tool": tool_name,
                "ok": out.get("_status") == "applied",
                "state_changing": out.get("_status") == "applied",
                "result": {k: v for k, v in out.items() if k != "evidence_id"},
            },
            source="tool",
        )
        return out


class OpenDssIeee13Backend(OpenDssIeee13ProbeBackend):
    """Released IEEE13 backend facade for ``PowerGridEnvironment``."""

    backend_kind = "opendss_ieee13"

    def __init__(self, *, source_root: Path = DEFAULT_SOURCE_ROOT) -> None:
        super().__init__(source_root=source_root)
        self._seed_obj: Any | None = None
        self._tick = 0
        self._horizon = 3
        self._tick_records: list[dict[str, Any]] = []
        self._cumulative_cost_components: dict[str, float] = {}
        self._perturbations: list[dict[str, Any]] = []
        self._base_load_values: dict[str, tuple[float, float]] = {}
        self._active_load_surge_indices: set[int] = set()

    def reset(self, scenario_seed: Any | None = None) -> OpenDssSolveSummary:
        self._seed_obj = scenario_seed
        self._tick = 0
        self._tick_records = []
        self._cumulative_cost_components = {}
        self._perturbations = []
        self._base_load_values = {}
        self._active_load_surge_indices = set()
        if scenario_seed is not None:
            self._horizon = int(getattr(scenario_seed, "horizon_ticks", 3) or 3)
            self._perturbations = [
                self._perturbation_dict(event)
                for event in (getattr(scenario_seed, "perturbations", []) or [])
                if isinstance(event, dict) or hasattr(event, "kind")
            ]
            self._validate_perturbations()
            source_root = (getattr(scenario_seed, "backend_config", {}) or {}).get(
                "source_root"
            )
            if source_root:
                self.source_root = Path(source_root)
                self.master_file = _resolve_master_file(self.source_root, MASTER_FILE)
        summary = super().reset()
        self._capture_base_load_values()
        return summary

    def _validate_perturbations(self) -> None:
        for event in self._perturbations:
            kind = str(event.get("kind") or "")
            expected_class = OPENDSS_IEEE13_EVENT_CLASS_REGISTRY.get(kind)
            if expected_class is None or kind == "load_surge_cleared":
                raise ValueError(
                    f"unsupported OpenDSS IEEE13 event kind: {kind or '<missing>'}"
                )
            declared_class = event.get("event_class")
            if declared_class is not None and str(declared_class) != expected_class:
                raise ValueError(
                    "OpenDSS IEEE13 event class does not match registry: "
                    f"{kind!r} declares {declared_class!r}, "
                    f"expected {expected_class!r}"
                )

    def tick(self, current_tick: int) -> Any:
        realized_events = self._apply_procedural_perturbations(current_tick)
        snapshot = self.snapshot()
        record = self._backend_record(
            current_tick,
            snapshot,
            realized_events=realized_events,
        )
        self._tick_records.append(record)
        self._add_cost_components(self._cost_components(snapshot))
        self._tick = int(current_tick) + 1
        return _OpenDssTickRecord(record)

    def snapshot(self) -> dict[str, Any]:
        raw = super().snapshot()
        raw["backend_kind"] = self.backend_kind
        raw["tick"] = self._tick
        raw["horizon"] = self._horizon
        raw.update(self._native_load_state())
        raw["entities"] = self._entities(raw)
        raw["totals"] = {
            "n_voltage_violations": raw.get("n_voltage_violations"),
            "voltage_min_pu": raw.get("voltage_min_pu"),
            "voltage_max_pu": raw.get("voltage_max_pu"),
            "line_current_max_a": raw.get("line_current_max_a"),
            "rho_max": 0.0,
            "n_overloads": 0,
            "n_disconnected_lines": 0,
        }
        return raw

    def ground_truth_costs(self) -> dict[str, float]:
        if not self._cumulative_cost_components:
            return {
                "production_cost": 0.0,
                "voltage_violation_cost": 0.0,
                "voltage_band_deviation_cost": 0.0,
            }
        return {k: round(v, 2) for k, v in self._cumulative_cost_components.items()}

    def per_load_shed_mwh(self) -> dict[str, float]:
        return {}

    def scoring_records(self) -> list[dict[str, Any]]:
        return list(self._tick_records)

    def forecast_for(self, horizon_ticks: int) -> list[dict[str, Any]]:
        snap = self.snapshot()
        return [
            {
                "tick": self._tick + i + 1,
                "voltage_min_pu": snap.get("voltage_min_pu"),
                "voltage_max_pu": snap.get("voltage_max_pu"),
                "n_voltage_violations": snap.get("n_voltage_violations"),
            }
            for i in range(max(0, int(horizon_ticks)))
        ]

    def _entities(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        entities: dict[str, Any] = {}
        for cap in snapshot.get("capacitors") or []:
            entities[f"capacitor_{cap['cap_id']}"] = {
                "kind": "capacitor",
                "status": any(int(s) for s in cap.get("states") or []),
                **cap,
            }
        for reg in snapshot.get("regcontrols") or []:
            entities[f"regulator_{reg['trafo_id']}"] = {
                "kind": "voltage_regulator",
                **reg,
            }
        return entities

    def _backend_record(
        self,
        tick: int,
        snapshot: dict[str, Any],
        *,
        realized_events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "tick": int(tick),
            "backend_kind": self.backend_kind,
            "aggregate_demand_mw": float(
                snapshot.get("aggregate_demand_mw") or 0.0
            ),
            "aggregate_reactive_demand_mvar": float(
                snapshot.get("aggregate_reactive_demand_mvar") or 0.0
            ),
            "aggregate_generation_mw": 0.0,
            "balance_error_mw": 0.0,
            "reserves_required_mw": 0.0,
            "reserves_procured_mw": 0.0,
            "production_cost": 0.0,
            "startup_cost": 0.0,
            "shed_penalty": 0.0,
            "rho_max": 0.0,
            "n_overloads": 0,
            "n_voltage_violations": int(snapshot.get("n_voltage_violations") or 0),
            "n_disconnected_lines": 0,
            "done": False,
            "converged": bool(snapshot.get("converged")),
            "voltage_min_pu": snapshot.get("voltage_min_pu"),
            "voltage_max_pu": snapshot.get("voltage_max_pu"),
            "voltage_band_error": self._voltage_band_error(snapshot),
            "line_current_max_a": snapshot.get("line_current_max_a"),
            "realized_events": realized_events,
        }

    @staticmethod
    def _perturbation_dict(event: Any) -> dict[str, Any]:
        if isinstance(event, dict):
            return dict(event)
        return {
            "kind": getattr(event, "kind", ""),
            "trigger_tick": getattr(event, "trigger_tick", 0),
            "duration_ticks": getattr(event, "duration_ticks", 1),
            "hidden": getattr(event, "hidden", False),
            "target": dict(getattr(event, "target", {}) or {}),
            "intensity": getattr(event, "intensity", 1.0),
            "notes": getattr(event, "notes", ""),
        }

    def _capture_base_load_values(self) -> None:
        circuit = self._require_circuit()
        values: dict[str, tuple[float, float]] = {}
        idx = circuit.Loads.First
        while idx:
            values[str(circuit.Loads.Name)] = (
                float(circuit.Loads.kW),
                float(circuit.Loads.kvar),
            )
            idx = circuit.Loads.Next
        self._base_load_values = values

    def _set_all_loads_multiplier(self, multiplier: float) -> None:
        circuit = self._require_circuit()
        idx = circuit.Loads.First
        while idx:
            name = str(circuit.Loads.Name)
            base_kw, base_kvar = self._base_load_values[name]
            circuit.Loads.kW = base_kw * multiplier
            circuit.Loads.kvar = base_kvar * multiplier
            idx = circuit.Loads.Next
        circuit.Solution.Solve()

    def _native_load_state(self) -> dict[str, float | None]:
        circuit = self._require_circuit()
        real_kw, reactive_kvar = (float(value) for value in circuit.TotalPower)
        return {
            "aggregate_demand_mw": max(0.0, -real_kw / 1000.0),
            "aggregate_reactive_demand_mvar": max(0.0, -reactive_kvar / 1000.0),
        }

    def _apply_procedural_perturbations(
        self, current_tick: int
    ) -> list[dict[str, Any]]:
        """Apply deterministic load events to the compiled IEEE13 circuit.

        These declared procedural overlays change native OpenDSS load setpoints;
        they are never reported as source-consumption evidence for the locked
        feeder graph.
        """
        if not self._perturbations or not self._base_load_values:
            return []
        events: list[dict[str, Any]] = []
        for index, raw_event in enumerate(self._perturbations):
            event = self._perturbation_dict(raw_event)
            if str(event.get("kind")) != "load_surge":
                continue
            trigger = int(event.get("trigger_tick", 0) or 0)
            duration = max(1, int(event.get("duration_ticks", 1) or 1))
            end_tick = trigger + duration
            if (
                index in self._active_load_surge_indices
                and current_tick >= end_tick
            ):
                self._set_all_loads_multiplier(1.0)
                self._active_load_surge_indices.remove(index)
                events.append(
                    {
                        "event_id": f"opendss-ieee13-load-surge-clear:{index}:{current_tick}",
                        "type": "load_surge_cleared",
                        "event_class": OPENDSS_IEEE13_EVENT_CLASS_REGISTRY[
                            "load_surge_cleared"
                        ],
                        "origin": "procedural_perturbation",
                        "tick": int(current_tick),
                        "changed_state_fields": [
                            "aggregate_demand_mw",
                            "aggregate_reactive_demand_mvar",
                            "bus_voltage_pu",
                        ],
                        "decision_required": False,
                        "actionable": False,
                    }
                )
            if current_tick != trigger or index in self._active_load_surge_indices:
                continue
            before = self._native_load_state()
            target = dict(event.get("target") or {})
            fraction = float(
                target.get("load_fraction", event.get("intensity", 0.0))
            )
            if fraction <= 0.0:
                continue
            self._set_all_loads_multiplier(1.0 + fraction)
            after = self._native_load_state()
            relative_change = abs(
                float(after["aggregate_demand_mw"] or 0.0)
                - float(before["aggregate_demand_mw"] or 0.0)
            ) / max(abs(float(before["aggregate_demand_mw"] or 0.0)), 1e-9)
            material = relative_change >= 0.01
            actionable = (
                material
                and not bool(event.get("hidden"))
                and int(current_tick) + 1 < self._horizon
            )
            self._active_load_surge_indices.add(index)
            events.append(
                {
                    "event_id": f"opendss-ieee13-load-surge:{index}:{trigger}",
                    "type": "load_surge",
                    "event_class": OPENDSS_IEEE13_EVENT_CLASS_REGISTRY[
                        "load_surge"
                    ],
                    "origin": "procedural_perturbation",
                    "declared_perturbation": True,
                    "hidden": bool(event.get("hidden")),
                    "tick": int(current_tick),
                    "source_asset": [str(self.master_file.resolve())],
                    "target": target,
                    "intensity": fraction,
                    "duration_ticks": duration,
                    "changed_state_fields": [
                        "aggregate_demand_mw",
                        "aggregate_reactive_demand_mvar",
                        "bus_voltage_pu",
                    ],
                    "materiality_metric": "aggregate_demand_relative_change",
                    "materiality_value": relative_change,
                    "materiality_threshold": 0.01,
                    "materiality_passed": material,
                    "decision_required": actionable,
                    "actionable": actionable,
                    "response_window_required": actionable,
                    "response_opportunity_tick": (
                        int(current_tick) + 1 if actionable else None
                    ),
                    "response_window_end_tick": max(
                        int(current_tick) + 1,
                        min(self._horizon - 1, end_tick),
                    ),
                    "before_state": before,
                    "after_state": after,
                }
            )
        return events

    def _cost_components(self, snapshot: dict[str, Any]) -> dict[str, float]:
        violations = float(snapshot.get("n_voltage_violations") or 0.0)
        return {
            "production_cost": 0.0,
            "voltage_violation_cost": 100.0 * violations,
            "voltage_band_deviation_cost": 1000.0 * self._voltage_band_error(snapshot),
        }

    def _add_cost_components(self, components: dict[str, float]) -> None:
        for key, value in components.items():
            self._cumulative_cost_components[key] = (
                self._cumulative_cost_components.get(key, 0.0) + float(value)
            )

    def _voltage_band_error(self, snapshot: dict[str, Any]) -> float:
        v_min = snapshot.get("voltage_min_pu")
        v_max = snapshot.get("voltage_max_pu")
        band_error = 0.0
        if isinstance(v_min, (int, float)):
            band_error += max(0.0, VOLTAGE_LOWER_PU - float(v_min))
        if isinstance(v_max, (int, float)):
            band_error += max(0.0, float(v_max) - VOLTAGE_UPPER_PU)
        return band_error


class _OpenDssTickRecord:
    def __init__(self, record: dict[str, Any]) -> None:
        self.realized_events: list[dict[str, Any]] = []
        for key, value in record.items():
            setattr(self, key, value)


class OpenDssIeee13DraftEnvironment(POMDPEnvironment):
    """Non-release StepReturn-compatible OpenDSS IEEE13 draft environment.

    This class is a promotion rung between the live source probe and a released
    backend. It exercises the normal POMDP and tool-protocol shape, but it is
    intentionally excluded from release registries and suite materializers.
    """

    domain = "power_grid"

    def __init__(self, *, source_root: Path = DEFAULT_SOURCE_ROOT) -> None:
        self._source_root = Path(source_root)
        self._backend = OpenDssIeee13ProbeBackend(source_root=self._source_root)
        self._tools: ToolRegistry | None = None
        self._evidence: EvidenceLogger | None = None
        self._scenario_config: dict[str, Any] = {}
        self._tick = 0
        self._horizon = 3
        self._seed = 0
        self._episode_id = "opendss_ieee13_draft"
        self._backend_tick_records: list[dict[str, Any]] = []
        self._cumulative_cost_components: dict[str, float] = {}

    def reset(self, scenario_config: dict[str, Any], seed: int) -> dict[str, Any]:
        if scenario_config.get("release_ready") is True:
            raise ValueError("OpenDSS IEEE13 draft environment is non-release only")
        if scenario_config.get("non_release_artifact") is not True:
            raise ValueError("OpenDSS IEEE13 draft scenarios must be non-release")
        self._scenario_config = dict(scenario_config)
        self._seed = int(seed)
        self._tick = 0
        self._horizon = int(scenario_config.get("horizon_ticks", 3))
        self._backend_tick_records = []
        self._cumulative_cost_components = {}
        self._episode_id = (
            f"opendss_ieee13_draft:"
            f"{scenario_config.get('scenario_id', 'unnamed')}:s{seed}"
        )
        self._backend = OpenDssIeee13ProbeBackend(source_root=self._source_root)
        self._backend.reset()
        self._tools = ToolRegistry(budget=self.budget, seed=seed)
        self._tools.reset(seed=seed)
        register_opendss_ieee13_probe_tools(self._tools, self._backend)
        self._evidence = EvidenceLogger(self._episode_id)
        obs = self.snapshot()
        self._log_state_evidence(label="initial", snapshot=obs)
        return obs

    def step(self, action: Action) -> StepReturn:
        if self._tools is None or self._evidence is None:
            raise RuntimeError("OpenDSS IEEE13 draft environment has not been reset")

        ctx = ToolContext(
            tick=self._tick,
            seed=self._seed,
            backend=self._backend,
            extra={"evidence": self._evidence, "env": self},
        )
        executable_action = Action(
            tool_calls=[
                call for call in action.tool_calls if call.name not in {"wait", "noop"}
            ],
            dominant=action.dominant,
            assistant_text=action.assistant_text,
        )
        tool_results = self._tools.execute_action(executable_action, ctx)
        for result in tool_results:
            self._evidence.log(
                kind="tool_call",
                tick=self._tick,
                payload={
                    "name": result.name,
                    "ok": result.ok,
                    "error_code": result.error_code,
                    "cost_units": result.cost_units,
                    "payload": result.payload,
                },
                source="tool",
            )
        obs = self.snapshot()
        state_evidence_id = self._log_state_evidence(label="after_step", snapshot=obs)
        backend_record = self._backend_record(obs)
        self._backend_tick_records.append(backend_record)
        self._add_cost_components(self._cost_components(obs))
        backend_evidence_id = self._evidence.log(
            kind="backend_tick",
            tick=self._tick,
            payload=backend_record,
            source="engine",
        )
        evidence_ids = [
            item.evidence_id
            for item in self._evidence.items()
            if item.tick == self._tick
        ]
        if state_evidence_id not in evidence_ids:
            evidence_ids.append(state_evidence_id)
        if backend_evidence_id not in evidence_ids:
            evidence_ids.append(backend_evidence_id)

        self._tick += 1
        obs["tick"] = self._tick
        done = self._tick >= self._horizon
        return StepReturn(
            observation=obs,
            tool_results=tool_results,
            reward=self._reward_signal(obs),
            done=done,
            info=StepInfo(
                evidence_ids=evidence_ids,
                extra={
                    "source_integration_rung": "adapted_from_probe",
                    "non_release_artifact": True,
                    "release_ready": False,
                    "scenario_id": self._scenario_config.get("scenario_id"),
                },
            ),
        )

    def snapshot(self) -> dict[str, Any]:
        raw = self._backend.snapshot()
        raw["tick"] = self._tick
        raw["source_integration_rung"] = "adapted_from_probe"
        raw["non_release_artifact"] = True
        raw["release_ready"] = False
        raw["scenario_id"] = self._scenario_config.get("scenario_id")
        return raw

    def ground_truth(self) -> dict[str, Any]:
        raw = self.snapshot()
        raw["evidence_count"] = len(self._evidence.items()) if self._evidence else 0
        raw["cost_components"] = dict(self._cumulative_cost_components)
        raw["backend_tick_records"] = list(self._backend_tick_records)
        return raw

    @property
    def tick(self) -> int:
        return self._tick

    @property
    def horizon(self) -> int:
        return self._horizon

    @property
    def budget(self) -> TickBudget:
        return TickBudget(
            max_tool_calls_per_tick=4,
            max_total_tool_calls=8,
            max_cost_units_per_tick=2.0,
        )

    @property
    def evidence(self) -> EvidenceLogger:
        if self._evidence is None:
            raise RuntimeError("OpenDSS IEEE13 draft environment has not been reset")
        return self._evidence

    def get_tool_specs(self) -> list[dict[str, Any]]:
        return self._tools.openai_schemas() if self._tools else []

    def supports_counterfactual(self) -> bool:
        return True

    def close(self) -> None:
        pass

    def _log_state_evidence(self, *, label: str, snapshot: dict[str, Any]) -> str:
        if self._evidence is None:
            raise RuntimeError("OpenDSS IEEE13 draft environment has not been reset")
        return self._evidence.log(
            kind="opendss_state",
            tick=self._tick,
            payload={
                "label": label,
                "converged": snapshot.get("converged"),
                "voltage_min_pu": snapshot.get("voltage_min_pu"),
                "voltage_max_pu": snapshot.get("voltage_max_pu"),
                "n_voltage_violations": snapshot.get("n_voltage_violations"),
                "capacitors": snapshot.get("capacitors") or [],
                "regcontrols": snapshot.get("regcontrols") or [],
                "line_current_max_a": snapshot.get("line_current_max_a"),
            },
            source="engine",
        )

    def _reward_signal(self, snapshot: dict[str, Any]) -> float:
        violations = float(snapshot.get("n_voltage_violations") or 0.0)
        v_min = snapshot.get("voltage_min_pu")
        v_max = snapshot.get("voltage_max_pu")
        band_error = 0.0
        if isinstance(v_min, (int, float)):
            band_error += max(0.0, VOLTAGE_LOWER_PU - float(v_min))
        if isinstance(v_max, (int, float)):
            band_error += max(0.0, float(v_max) - VOLTAGE_UPPER_PU)
        return -(violations + band_error)

    def _cost_components(self, snapshot: dict[str, Any]) -> dict[str, float]:
        violations = float(snapshot.get("n_voltage_violations") or 0.0)
        band_error = self._voltage_band_error(snapshot)
        return {
            "production_cost": 0.0,
            "voltage_violation_cost": 100.0 * violations,
            "voltage_band_deviation_cost": 1000.0 * band_error,
        }

    def _add_cost_components(self, components: dict[str, float]) -> None:
        for key, value in components.items():
            self._cumulative_cost_components[key] = (
                self._cumulative_cost_components.get(key, 0.0) + float(value)
            )

    def _backend_record(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        return {
            "tick": self._tick,
            "backend_kind": OpenDssIeee13ProbeBackend.backend_kind,
            "done": False,
            "balance_error_mw": 0.0,
            "reserves_required_mw": 0.0,
            "reserves_procured_mw": 0.0,
            "n_overloads": 0,
            "n_disconnected_lines": 0,
            "rho_max": 0.0,
            "n_voltage_violations": int(snapshot.get("n_voltage_violations") or 0),
            "voltage_min_pu": snapshot.get("voltage_min_pu"),
            "voltage_max_pu": snapshot.get("voltage_max_pu"),
            "voltage_band_error": self._voltage_band_error(snapshot),
            "line_current_max_a": snapshot.get("line_current_max_a"),
            "production_cost": 0.0,
        }

    def _voltage_band_error(self, snapshot: dict[str, Any]) -> float:
        v_min = snapshot.get("voltage_min_pu")
        v_max = snapshot.get("voltage_max_pu")
        band_error = 0.0
        if isinstance(v_min, (int, float)):
            band_error += max(0.0, VOLTAGE_LOWER_PU - float(v_min))
        if isinstance(v_max, (int, float)):
            band_error += max(0.0, float(v_max) - VOLTAGE_UPPER_PU)
        return band_error


def build_opendss_ieee13_non_release_scenario_drafts(
    *, source_root: Path = DEFAULT_SOURCE_ROOT
) -> list[dict[str, Any]]:
    """Return tiny non-release draft scenarios for the IEEE13 probe env."""

    source_root = Path(source_root)
    common = {
        "domain": "power_grid",
        "family": "opendss_ieee13_non_release_probe",
        "backend_kind": OpenDssIeee13ProbeBackend.backend_kind,
        "source_root": str(source_root),
        "source_master_file": MASTER_FILE,
        "source_integration_rung": "adapted_from_probe",
        "non_release_artifact": True,
        "release_ready": False,
        "release_reentry_ready": False,
        "proceed_commands": [],
        "horizon_ticks": 3,
        "source_axes": {
            "feeder": "ieee13_node_test_feeder",
            "topology": "unbalanced_three_phase_distribution",
        },
    }
    return [
        {
            **common,
            "scenario_id": "opendss_ieee13_capacitor_voltage_probe",
            "difficulty_mode": "time_pressure",
            "difficulty_level": "draft",
            "decision_axis": "capacitor_bank_switching",
            "draft_action": {
                "tool": "switch_capacitor",
                "args": {"cap_id": 0, "status": False},
            },
        },
        {
            **common,
            "scenario_id": "opendss_ieee13_regulator_tap_probe",
            "difficulty_mode": "deep_planning",
            "difficulty_level": "draft",
            "decision_axis": "regulator_tap_adjustment",
            "draft_action": {
                "tool": "set_transformer_tap",
                "args": {"trafo_id": 0, "tap_pos": 7},
            },
        },
    ]


def register_opendss_ieee13_probe_tools(
    reg: ToolRegistry, backend: OpenDssIeee13ProbeBackend
) -> None:
    """Register the tiny non-release OpenDSS probe control surface."""

    reg.register(
        ToolSpec(
            name="switch_capacitor",
            description="Switch an IEEE13 capacitor bank in the OpenDSS probe.",
            parameters={
                "type": "object",
                "properties": {
                    "cap_id": {"type": "integer"},
                    "status": {"type": "boolean"},
                },
                "required": ["cap_id", "status"],
            },
            handler=lambda args, ctx: backend.apply_tool_effect(
                "switch_capacitor", args, ctx
            ),
            state_changing=True,
            semantic_role="control",
            native_target_kind="capacitor_bank",
            actuator_family="shunt_switching",
            cost_units=1.0,
        )
    )
    reg.register(
        ToolSpec(
            name="set_transformer_tap",
            description="Set an IEEE13 regulator tap in the OpenDSS probe.",
            parameters={
                "type": "object",
                "properties": {
                    "trafo_id": {"type": "integer"},
                    "tap_pos": {"type": "integer"},
                },
                "required": ["trafo_id", "tap_pos"],
            },
            handler=lambda args, ctx: backend.apply_tool_effect(
                "set_transformer_tap", args, ctx
            ),
            state_changing=True,
            semantic_role="control",
            native_target_kind="voltage_regulator",
            actuator_family="tap_changer",
            cost_units=1.0,
        )
    )


def _import_dss() -> Any:
    try:
        import dss  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - exercised via probe flag
        raise OpenDssRuntimeUnavailable(
            "dss-python is not installed; install dss-python to run the "
            "OpenDSS IEEE13 live probe"
        ) from exc
    return dss
