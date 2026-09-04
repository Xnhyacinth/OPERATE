"""
domains.power_grid.backends.cigre_distribution — pandapower CIGRE MV backend.

pandapower CIGRE MV distribution feeder solving AC power flow each tick.
Voltage limits, line loading, DER injections and load shedding all act
on the actual feeder model. The canonical machine-readable description lives
in the active OPERATE release manifest's
``backend_descriptors.cigre_distribution`` block.

Built on the public **CIGRE European MV benchmark network with DER**
shipped inside pandapower. Provides:

- 15-bus medium-voltage radial network with two HV feeders, 18 loads, 13
  DER generators (PV + wind), and 2 storage units.
- Per-tick AC power flow via ``pandapower.runpp``.
- Native distribution observables: bus voltage magnitude (pu), line
  loading percentage, transformer tap position, switch state.
- Native distribution tools: tap-changer adjustment, capacitor on/off,
  DER curtailment, load shed, switch open/close.
- A chronic that pulses load (residential evening peak), drops wind in a
  storm window, and trips a DG mid-horizon — the kind of stresses real
  feeder operators face.

The backend exposes the SAME shape as ``PglibUcSyntheticBackend`` and
``Grid2OpBackend`` (``reset/tick/snapshot/apply_tool_effect/ground_truth_costs``)
so the adapter and scorer are reusable without modification.

The backend extends OPERATE beyond transmission scheduling into native
distribution physics.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import pandapower as pp  # type: ignore[import-untyped]

from core.source_asset_contract import (
    virtual_source_identity_sha256,
    virtual_source_reference_info,
)

from ..seeds.schema import ScenarioSeed


CIGRE_PERTURBATION_EVENT_REGISTRY = MappingProxyType(
    {
        "line_outage": ("line_outage", "alarm"),
        "generator_forced_outage": ("generator_outage", "alarm"),
        "load_surge": ("load_surge", "alarm"),
        "wind_dropout": ("wind_dropout", "alarm"),
        "renewable_output_error": ("renewable_output_error", "alarm"),
        "forecast_bias": ("forecast_bias", "forecast"),
        "storm_window": ("storm_window", "alarm"),
    }
)


def _build_distribution_net(network: str) -> Any:
    """Construct a pandapower distribution net by ``network`` tag.

    All nets here are radial/meshed MV/LV feeders that ``pp.runpp``
    solves and whose loads/sgen/lines the topology-agnostic backend tick
    loop already handles. New v0.4 topologies (mv_oberrhein real German MV
    grid; a balanced synthetic Volt-Var LV feeder) are added alongside the
    original CIGRE MV without changing ``backend_kind``.
    """
    import pandapower.networks as pn  # type: ignore[import-untyped]

    if network == "cigre_mv_with_der_all":
        return pn.create_cigre_network_mv(with_der="all")
    if network == "mv_oberrhein":
        return pn.mv_oberrhein()
    if network == "synthetic_volt_control_lv":
        return pn.create_synthetic_voltage_control_lv_network()
    # v0.6: SimBench MV feeders (BSD-3-Clause simbench package; bundled data,
    # no large download). Imported lazily so the backend stays importable on
    # hosts without simbench. Tag "1-MV-rural--0" maps to the switched grid
    # code "1-MV-rural--0--sw".
    if network.startswith("simbench:"):
        import simbench as sb  # type: ignore[import-untyped]

        code = network.split("simbench:", 1)[1]
        if "--sw" not in code and "--no_sw" not in code:
            code = f"{code}--sw"
        return sb.get_simbench_net(code)
    # Unknown tag → fail closed on the canonical default rather than
    # silently mis-modelling.
    return pn.create_cigre_network_mv(with_der="all")


@dataclass
class CigreTickRecord:
    tick: int
    aggregate_demand_mw: float
    aggregate_generation_mw: float
    balance_error_mw: float
    reserves_required_mw: float
    reserves_procured_mw: float
    production_cost: float
    startup_cost: float
    shed_penalty: float
    rho_max: float
    n_overloads: int
    n_voltage_violations: int  # buses with |V-1.0|>0.05
    n_disconnected_lines: int
    converged: bool = True
    done: bool = False
    realized_events: list[dict[str, Any]] = field(default_factory=list)


class CigreDistributionBackend:
    """CIGRE MV distribution backend (pandapower)."""

    # Stable registry key used by the adapter to attach the Protocol-2.1
    # hybrid supervisory cadence to agent-facing snapshots.
    backend_kind = "cigre_distribution"

    # Tariffs roughly aligned with transmission backends.
    SHED_PENALTY_PER_MWH_BY_CLASS: dict[str, float] = {
        "residential": 200.0,
        "commercial": 400.0,
        "industrial": 600.0,
        "data_center": 1500.0,
        "transit": 1500.0,
        "water": 2500.0,
        "hospital": 5000.0,
    }
    SHED_PENALTY_DEFAULT = 1000.0
    PRODUCTION_COST_PER_MWH = 50.0
    OVERLOAD_COST_PER_TICK = 200.0
    # SAIDI/CAIDI-style penalty: each bus held outside [0.95, 1.05] pu
    # for one tick costs roughly the same as a sustained ~3 MWh
    # commercial customer interruption — far above marginal generation
    # cost so the agent has a real reason to maintain voltage by either
    # shedding load or curtailing DER on radial feeders. Calibrated so a
    # 1-tick voltage violation costs more than the shed penalty needed
    # to resolve it.
    VOLTAGE_VIOLATION_COST_PER_TICK = 1200.0
    DISCONNECTION_COST_PER_LINE_TICK = 500.0
    RESERVE_TARGET_FRACTION_OF_DEMAND = 0.10
    VOLTAGE_LOWER_PU = 0.95
    VOLTAGE_UPPER_PU = 1.05

    def __init__(self) -> None:
        self._net: Any = None
        self._seed_obj: ScenarioSeed | None = None
        self._tick: int = 0
        self._horizon: int = 24
        self._tick_records: list[CigreTickRecord] = []
        self._loads: dict[str, dict[str, Any]] = {}
        self._base_load_p_mw: dict[int, float] = {}
        self._base_load_q_mvar: dict[int, float] = {}
        self._base_sgen_p_mw: dict[int, float] = {}
        self._simbench_profile_values: dict[tuple[str, str], Any] | None = None
        self._profile_start_index = 0
        self._profile_step = 1
        self._simbench_source_window_sha256: str | None = None
        self._simbench_source_recipe_version: str | None = None
        self._simbench_profile_applied_ticks: set[int] = set()
        self._cumulative_shed_mwh: dict[str, float] = {}
        self._pending_curtail: dict[int, float] = {}  # sgen index → MW cap
        self._tap_offsets: dict[int, int] = {}  # trafo index → tap offset
        self._line_status_overrides: dict[int, bool] = {}
        self._capacitor_on: dict[int, bool] = {}
        # v0.6 distribution-native Volt-Var controls. Opt-in via
        # backend_config["volt_var_controls"]; OFF by default so every
        # v0.4.1 / v0.5 distribution scenario is byte-behaviour-identical
        # (no augmentation, all new tools no-op).
        self._volt_var_enabled: bool = False
        self._tap_targets: dict[int, int] = {}  # trafo idx → absolute tap_pos
        self._tap_base: dict[int, int] = {}  # trafo idx → base tap_pos
        self._der_q_targets: dict[int, float] = {}  # sgen idx → q_mvar setpoint
        self._der_q_base: dict[int, float] = {}  # sgen idx → base q_mvar
        self._storage_p_targets: dict[int, float] = {}  # storage idx → p_mw
        self._cap_shunt_idx: list[int] = []  # indices of augmented cap-bank shunts
        self._wind_factor_this_tick: float = 1.0
        self._renewable_output_factor_this_tick: float = 1.0
        self._load_surge_factor_this_tick: float = 0.0
        # Per-stakeholder-class surge factor (DC-2): load_surge now
        # targets specific load classes instead of being applied globally.
        self._class_surge_factors: dict[str, float] = {}
        self._forecast_bias: float = 0.0
        self._pending_reserve_extra_mw: float = 0.0
        self._last_balance_error: float = 0.0
        self._done: bool = False
        # v0.2.2 F-01: per-episode delayed-effect queue for request_mutual_aid
        # entries are (due_tick, mw) and drained at the start of tick().
        self._pending_mutual_aid: list[tuple[int, float]] = []
        # BUG-2: tracked per-sgen outage windows. Must be cleared in reset()
        # or DER outages from episode N persist into episode N+1.
        self._sgen_outage_until: dict[int, int] = {}
        # NIT-2: O(1) index→load_id reverse lookup (built in reset()).
        self._idx_to_load_id: dict[int, str] = {}
        self._source_constructor_uri: str | None = None
        self._source_constructor_hash: str | None = None
        self._source_constructor_state_digest: str | None = None
        self._source_solver_state_digest: str | None = None
        self._source_constructor_blockers: list[str] = []
        self._telemetry_confidence: float = 1.0

    # ── Reset ──────────────────────────────────────────────────────────

    def reset(self, scenario_seed: ScenarioSeed) -> None:
        self._seed_obj = scenario_seed
        self._tick = 0
        self._horizon = scenario_seed.horizon_ticks
        self._tick_records.clear()
        self._cumulative_shed_mwh.clear()
        self._pending_curtail.clear()
        self._tap_offsets.clear()
        self._line_status_overrides.clear()
        self._capacitor_on.clear()
        self._tap_targets.clear()
        self._tap_base.clear()
        self._der_q_targets.clear()
        self._der_q_base.clear()
        self._storage_p_targets.clear()
        self._cap_shunt_idx.clear()
        self._volt_var_enabled = False
        self._wind_factor_this_tick = 1.0
        self._renewable_output_factor_this_tick = 1.0
        self._load_surge_factor_this_tick = 0.0
        self._class_surge_factors.clear()
        self._forecast_bias = 0.0
        self._pending_reserve_extra_mw = 0.0
        self._last_balance_error = 0.0
        self._done = False
        # v0.2.2 F-01: clear any mutual-aid carryover between episodes.
        self._pending_mutual_aid = []
        # BUG-2: clear forced-outage state between episodes.
        self._sgen_outage_until.clear()
        # NIT-2: rebuild O(1) lookup
        self._idx_to_load_id.clear()
        self._source_constructor_uri = None
        self._source_constructor_hash = None
        self._source_constructor_state_digest = None
        self._source_solver_state_digest = None
        self._source_constructor_blockers = []
        self._telemetry_confidence = 1.0

        for perturbation in scenario_seed.perturbations:
            self._validate_perturbation_event(perturbation)

        # v0.4 Bucket B (D1): the backend is topology-agnostic (it operates
        # on whatever net.load / net.sgen / net.line / net.bus exist), so a
        # ``network`` switch lets the same backend drive additional real
        # distribution feeders. Default = CIGRE MV (byte-identical to v0.3),
        # so existing distribution_volt_var hashes are preserved.
        network = "cigre_mv_with_der_all"
        if self._seed_obj is not None:
            network = str(
                (self._seed_obj.backend_config or {}).get(
                    "network", "cigre_mv_with_der_all"
                )
            )
        self._net = _build_distribution_net(network)
        source_files = list(
            (self._seed_obj.provenance.files if self._seed_obj else ()) or ()
        )
        constructor_refs = [
            (str(value), virtual_source_reference_info(str(value)))
            for value in source_files
        ]
        constructor_refs = [
            (uri, info) for uri, info in constructor_refs if info is not None
        ]
        if not constructor_refs:
            self._source_constructor_blockers = [
                "constructor_reference_missing"
            ]
        else:
            matching_refs = [
                (uri, info)
                for uri, info in constructor_refs
                if info.get("network") == network
            ]
            if not matching_refs:
                self._source_constructor_blockers = [
                    "constructor_network_mismatch"
                ]
            else:
                constructor_uri, constructor_info = matching_refs[0]
                recorded_version = str(
                    constructor_info.get("version") or "unknown"
                )
                actual_version = str(getattr(pp, "__version__", "unknown"))
                if constructor_info.get("scheme") == "pandapower-simbench":
                    import simbench as sb  # type: ignore[import-untyped]

                    actual_version = str(getattr(sb, "__version__", "unknown"))
                if recorded_version not in {"unknown", actual_version}:
                    self._source_constructor_blockers = [
                        "constructor_version_mismatch"
                    ]
                else:
                    self._source_constructor_uri = constructor_uri
                    self._source_constructor_hash = (
                        virtual_source_identity_sha256(constructor_uri)
                    )
                    self._source_constructor_state_digest = (
                        self._constructor_state_digest(network)
                    )
        # Cache base profiles so we can rebuild each tick deterministically.
        self._base_load_p_mw = {
            int(i): float(self._net.load.p_mw.iloc[i])
            for i in range(len(self._net.load))
        }
        self._base_load_q_mvar = {
            int(i): float(self._net.load.q_mvar.iloc[i])
            for i in range(len(self._net.load))
        }
        self._base_sgen_p_mw = {
            int(i): float(self._net.sgen.p_mw.iloc[i])
            for i in range(len(self._net.sgen))
        }
        bc = (self._seed_obj.backend_config or {}) if self._seed_obj else {}
        self._simbench_profile_values = None
        self._profile_start_index = 0
        self._profile_step = 1
        self._simbench_source_window_sha256 = None
        self._simbench_source_recipe_version = None
        self._simbench_profile_applied_ticks.clear()
        if network.startswith("simbench:"):
            import simbench as sb  # type: ignore[import-untyped]

            self._simbench_profile_values = sb.get_absolute_values(
                self._net, profiles_instead_of_study_cases=True
            )
            self._profile_start_index = int(bc.get("profile_start_index", 0))
            self._profile_step = max(1, int(bc.get("profile_step", 1)))
            self._simbench_source_window_sha256 = self._simbench_window_digest()
            recipe = bc.get("long_horizon_candidate") or {}
            self._simbench_source_recipe_version = str(recipe.get("pipeline_version") or "") or None

        # v0.6 distribution-native Volt-Var controls (opt-in). When the seed
        # does not request them the net is untouched, so existing
        # distribution_volt_var scenarios remain byte-behaviour-identical.
        self._volt_var_enabled = bool(bc.get("volt_var_controls", False))
        if self._volt_var_enabled:
            self._setup_volt_var_controls()

        # Map load assignments → pandapower load indices in seed order
        self._loads = {}
        for idx, assignment in enumerate(scenario_seed.load_assignments):
            if idx >= len(self._net.load):
                break
            self._loads[assignment.load_id] = {
                "load_index": idx,
                "bus_id": int(self._net.load.bus.iloc[idx]),
                "stakeholder_class": assignment.stakeholder_class,
                "criticality": assignment.criticality,
                "shed_this_tick_mw": 0.0,
            }
            self._cumulative_shed_mwh[assignment.load_id] = 0.0
            self._idx_to_load_id[idx] = assignment.load_id

    # ── v0.6 distribution-native Volt-Var controls ──────────────────────

    def _constructor_state_digest(self, network: str) -> str:
        """Digest the native network returned by the locked constructor."""
        net = self._net
        payload = {
            "network": network,
            "pandapower_version": str(getattr(pp, "__version__", "")),
            "bus_count": len(net.bus),
            "line_count": len(net.line),
            "load_count": len(net.load),
            "sgen_count": len(net.sgen),
            "storage_count": len(net.storage),
            "load_bus": [int(value) for value in net.load.bus.tolist()],
            "load_p_mw": [round(float(value), 9) for value in net.load.p_mw.tolist()],
            "load_q_mvar": [
                round(float(value), 9) for value in net.load.q_mvar.tolist()
            ],
            "sgen_bus": [int(value) for value in net.sgen.bus.tolist()],
            "sgen_p_mw": [round(float(value), 9) for value in net.sgen.p_mw.tolist()],
            "line_from_bus": [int(value) for value in net.line.from_bus.tolist()],
            "line_to_bus": [int(value) for value in net.line.to_bus.tolist()],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
            ).hexdigest()

    def _simbench_window_digest(self) -> str:
        """Recompute the source-window digest from the profiles actually loaded."""
        assert self._simbench_profile_values is not None
        load_frame = self._simbench_profile_values[("load", "p_mw")]
        renewable_indices = [
            int(index)
            for index, row in self._net.sgen.iterrows()
            if any(token in str(row.get("type") or "").lower() for token in ("pv", "solar", "res"))
        ]
        generation_frame = self._simbench_profile_values[("sgen", "p_mw")]
        indices = [
            (self._profile_start_index + tick * self._profile_step) % len(load_frame)
            for tick in range(self._horizon)
        ]
        payload = {
            "load_mw": [round(float(load_frame.iloc[index].sum()), 9) for index in indices],
            "sgen_mw": [
                round(float(generation_frame.iloc[index][renewable_indices].sum()), 9)
                for index in indices
            ],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _solved_state_digest(self) -> str | None:
        """Digest solved native observables when a power flow has converged."""
        net = self._net
        if net is None or not hasattr(net, "res_bus") or not hasattr(net, "res_line"):
            return None
        if len(net.res_bus) == 0 or len(net.res_line) == 0:
            return None
        try:
            payload = {
                "bus_vm_pu": [
                    round(float(value), 9)
                    for value in net.res_bus.vm_pu.tolist()
                ],
                "line_loading_percent": [
                    round(float(value), 9)
                    for value in net.res_line.loading_percent.tolist()
                ],
            }
        except (AttributeError, TypeError, ValueError):
            return None
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()

    def protocol21_source_trace(self) -> dict[str, Any]:
        """Prove consumption of the locked pandapower constructor identity."""
        if self._source_constructor_blockers:
            return {
                "status": "held",
                "proof_kind": "derived_source_window",
                "runtime_trace_observed": False,
                "evidence_from_scenario_config_only": False,
                "source_state_effect_observed": False,
                "state_effect_observed": False,
                "blockers": list(self._source_constructor_blockers),
            }
        uri = self._source_constructor_uri
        source_hash = self._source_constructor_hash
        state_digest = self._source_constructor_state_digest
        if not uri or not source_hash or not state_digest:
            return {
                "status": "held",
                "proof_kind": "derived_source_window",
                "runtime_trace_observed": False,
                "evidence_from_scenario_config_only": False,
                "source_state_effect_observed": False,
                "state_effect_observed": False,
                "blockers": ["constructor_runtime_trace_unavailable"],
            }
        solved_digest = self._source_solver_state_digest
        source_profile_consumed = bool(self._simbench_profile_applied_ticks)
        consumed_window_sha256 = (
            self._simbench_source_window_sha256
            if source_profile_consumed and self._simbench_source_window_sha256
            else source_hash
        )
        recipe_version = (
            self._simbench_source_recipe_version
            if source_profile_consumed and self._simbench_source_recipe_version
            else "pandapower_constructor_runtime_v1"
        )
        state_digest_kind = (
            "native_solver_state"
            if solved_digest is not None
            else "constructor_materialization"
        )
        semantic = {
            "constructor_uri": uri,
            "constructor_identity_sha256": source_hash,
            "constructor_state_digest": state_digest,
            "solved_state_digest": solved_digest,
            "state_digest_kind": state_digest_kind,
        }
        return {
            "status": "passed",
            "proof_kind": "derived_source_window",
            "runtime_opened_assets": [
                {
                    "path": uri,
                    "sha256": source_hash,
                    "role": "derivation_input",
                    "kind": "virtual_constructor",
                }
            ],
            "opened_source_paths": [uri],
            "opened_source_sha256": {uri: source_hash},
            "consumed_source_hashes": {},
            "lineage_source_hashes": {uri: source_hash},
            "consumed_window_sha256": consumed_window_sha256,
            "recipe_version": recipe_version,
            "consumed_channels": [
                "pandapower_network_constructor",
                *(["simbench_profile_window"] if source_profile_consumed else []),
            ],
            "derived_backend_state_fields": [
                "network_topology",
                "load_parameters",
                "der_capabilities",
                "bus_voltage_pu",
                "line_loading_percent",
            ],
            "consumption_ticks": sorted({0, *self._simbench_profile_applied_ticks}),
            "initial_state_digest": state_digest,
            "constructor_materialization_digest": state_digest,
            "solved_state_digest": solved_digest,
            "state_digest_kind": state_digest_kind,
            "post_source_state_digests": [solved_digest or state_digest],
            "source_state_effect_observed": True,
            "state_effect_observed": True,
            "state_effect_kind": state_digest_kind,
            "deterministic_source_trace": True,
            "trace_semantic_digest": hashlib.sha256(
                json.dumps(
                    semantic, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest(),
            "runtime_trace_observed": True,
            "evidence_from_scenario_config_only": False,
            "source_time_variation_claimed": source_profile_consumed,
            "blockers": [],
        }

    def _setup_volt_var_controls(self) -> None:
        """Cache control bases and augment capacitor banks (OFF by default).

        Adds switchable capacitor-bank shunts only when the feeder ships
        none (CIGRE/SimBench MV nets usually have no shunts). The banks are
        created ``in_service=False`` so the baseline power flow is identical
        to the un-augmented net until ``switch_capacitor`` turns one on.
        """
        net = self._net
        if len(net.trafo) > 0 and "tap_pos" in net.trafo:
            for i in range(len(net.trafo)):
                tp = net.trafo.tap_pos.iloc[i]
                if tp == tp:  # not NaN
                    self._tap_base[int(i)] = int(tp)
        for i in range(len(net.sgen)):
            self._der_q_base[int(i)] = (
                float(net.sgen.q_mvar.iloc[i]) if "q_mvar" in net.sgen else 0.0
            )
        existing = (
            [int(i) for i in net.shunt.index]
            if hasattr(net, "shunt") and len(net.shunt)
            else []
        )
        self._cap_shunt_idx = list(existing)
        if not existing:
            feeder_mw = sum(self._base_load_p_mw.values())
            # Capacitive Mvar bank; bounded so a large feeder does not get a
            # destabilising bank that pushes the solve to over-voltage.
            q_bank = max(0.3, round(min(5.0, 0.12 * feeder_mw), 3))
            buses_seen: list[int] = []
            for li, _mw in sorted(self._base_load_p_mw.items(), key=lambda kv: -kv[1]):
                if li >= len(net.load):
                    continue
                bus = int(net.load.bus.iloc[li])
                if bus in buses_seen:
                    continue
                buses_seen.append(bus)
                idx = pp.create_shunt(
                    net,
                    bus=bus,
                    q_mvar=-q_bank,
                    p_mw=0.0,
                    name=f"capbank_{bus}",
                    in_service=False,
                )
                self._cap_shunt_idx.append(int(idx))
                if len(self._cap_shunt_idx) >= 3:
                    break

    def _apply_volt_var_controls(self) -> None:
        """Push queued tap / capacitor / DER-Q / storage setpoints into the net."""
        net = self._net
        for tid, pos in self._tap_targets.items():
            if 0 <= tid < len(net.trafo) and "tap_pos" in net.trafo:
                p = pos
                try:
                    lo, hi = net.trafo.tap_min.iloc[tid], net.trafo.tap_max.iloc[tid]
                    if lo == lo and hi == hi:
                        p = int(max(int(lo), min(int(hi), pos)))
                except Exception:
                    pass
                net.trafo.at[tid, "tap_pos"] = p
        for did, q in self._der_q_targets.items():
            if 0 <= did < len(net.sgen):
                net.sgen.at[did, "q_mvar"] = q
        for cid, on in self._capacitor_on.items():
            if 0 <= cid < len(net.shunt):
                net.shunt.at[cid, "in_service"] = bool(on)
        for sid, p in self._storage_p_targets.items():
            if 0 <= sid < len(net.storage):
                net.storage.at[sid, "p_mw"] = p

    # ── Tool effects ────────────────────────────────────────────────────

    @staticmethod
    def _invalid_asset_payload(
        tool: str, asset: str, idx: int, count: int
    ) -> dict[str, Any]:
        return {
            "_status": "error",
            "error": "unknown_controllable_asset",
            "tool": tool,
            "asset": asset,
            "index": idx,
            "n_available": count,
        }

    def apply_tool_effect(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "shed_load":
            load_id = str(args.get("load_id", ""))
            mw = float(args.get("mw", 0.0))
            entry = self._loads.get(load_id)
            if entry is None:
                return {"_status": "error", "error": "unknown_load", "load_id": load_id}
            entry["shed_this_tick_mw"] += mw
            tick_h = float(self._seed_obj.tick_minutes if self._seed_obj else 60) / 60.0
            self._cumulative_shed_mwh[load_id] += mw * tick_h
            return {
                "load_id": load_id,
                "shed_mw": mw,
                "stakeholder_class": entry["stakeholder_class"],
                "criticality": entry["criticality"],
            }
        if name == "switch_branch":
            line_index = int(args.get("line_index", 0))
            connect = bool(args.get("connect", True))
            self._line_status_overrides[line_index] = connect
            return {"line_id": line_index, "connect": connect, "queued": True}
        if name == "redispatch_generation":
            # Distribution analogue: cap a single DER (sgen) at target_mw.
            try:
                sgen_idx = int(args.get("generator_index", 0))
            except (TypeError, ValueError):
                # Try to find by gen_id name
                gen_id = str(args.get("generator_id", ""))
                sgen_idx = next(
                    (i for i, n in enumerate(self._net.sgen.name) if str(n) == gen_id),
                    0,
                )
            target = float(args.get("target_mw", args.get("delta_mw", 0.0)))
            self._pending_curtail[sgen_idx] = max(0.0, target)
            return {
                "generator_index": sgen_idx,
                "target_mw": round(target, 3),
                "queued": True,
            }
        if name == "commit_reserve":
            mw = float(args.get("mw", 0.0))
            self._pending_reserve_extra_mw += mw
            return {
                "reserve_pending_mw": self._pending_reserve_extra_mw,
                "info": f"{name} queued (CIGRE reserve channel synthetic)",
            }
        if name == "request_mutual_aid":
            # v0.2.2 F-01: dedicated delayed-effect path; never mutate
            # reserves from this code path.
            return {
                "_status": "ack",
                "info": (
                    "mutual-aid uses the dedicated delayed-effect path; "
                    "this code path no longer mutates reserves"
                ),
            }
        if name == "topology_action":
            # On distribution, treat topology_action as bulk feeder switch.
            # DC-11: include `_status` to mirror the other noop returns.
            sub_id = int(args.get("substation_id", 0))
            return {
                "_status": "noop",
                "substation_id": sub_id,
                "info": "noop on CIGRE backend",
            }
        if name in (
            "set_transformer_tap",
            "switch_capacitor",
            "set_der_reactive_power",
            "set_battery_dispatch",
        ):
            if not self._volt_var_enabled:
                return {
                    "_status": "unsupported",
                    "info": (
                        f"{name} requires backend_config['volt_var_controls']=true "
                        "(v0.6 distribution-native Volt-Var family)"
                    ),
                }
            if name == "set_transformer_tap":
                tid = int(args.get("trafo_id", args.get("trafo_index", 0)))
                if not (0 <= tid < len(self._net.trafo)):
                    return self._invalid_asset_payload(
                        name, "trafo_id", tid, len(self._net.trafo)
                    )
                if "tap_pos" not in self._net.trafo:
                    return {
                        "_status": "error",
                        "error": "tap_control_unavailable",
                        "tool": name,
                        "asset": "trafo_id",
                        "index": tid,
                    }
                pos = int(args.get("tap_pos", 0))
                self._tap_targets[tid] = pos
                return {"trafo_id": tid, "tap_pos": pos, "queued": True}
            if name == "switch_capacitor":
                cid = int(args.get("cap_id", args.get("capacitor_id", 0)))
                if not (0 <= cid < len(self._net.shunt)):
                    return self._invalid_asset_payload(
                        name, "cap_id", cid, len(self._net.shunt)
                    )
                status = bool(args.get("status", True))
                self._capacitor_on[cid] = status
                return {"cap_id": cid, "status": status, "queued": True}
            if name == "set_der_reactive_power":
                did = int(args.get("der_id", args.get("sgen_index", 0)))
                if not (0 <= did < len(self._net.sgen)):
                    return self._invalid_asset_payload(
                        name, "der_id", did, len(self._net.sgen)
                    )
                q = float(args.get("q_mvar", 0.0))
                self._der_q_targets[did] = q
                return {"der_id": did, "q_mvar": round(q, 3), "queued": True}
            # set_battery_dispatch
            sid = int(args.get("storage_id", args.get("storage_index", 0)))
            if not (0 <= sid < len(self._net.storage)):
                return self._invalid_asset_payload(
                    name, "storage_id", sid, len(self._net.storage)
                )
            p = float(args.get("p_mw", 0.0))
            self._storage_p_targets[sid] = p
            return {"storage_id": sid, "p_mw": round(p, 3), "queued": True}
        return {"_status": "noop"}

    # ── v0.2.2 F-01: unified delayed-effect API for request_mutual_aid ──

    def queue_mutual_aid_effect(self, *, due_tick: int, mw: float) -> None:
        """Queue a mutual-aid reserve injection to land at ``due_tick``.

        Drained at the START of ``tick(due_tick)`` so the reserve appears
        in that tick's record and not earlier.
        """
        self._pending_mutual_aid.append((int(due_tick), float(mw)))

    def _drain_mutual_aid(self, current_tick: int) -> float:
        """Drain matured mutual-aid entries; return total MW added this tick."""
        added = 0.0
        kept: list[tuple[int, float]] = []
        for due_tick, mw in self._pending_mutual_aid:
            if due_tick <= current_tick:
                added += mw
            else:
                kept.append((due_tick, mw))
        self._pending_mutual_aid = kept
        return added

    # ── Tick ────────────────────────────────────────────────────────────

    def tick(self, current_tick: int) -> CigreTickRecord:
        assert self._net is not None
        self._tick = current_tick
        # v0.2.2 F-01: drain matured mutual-aid effects BEFORE perturbations
        # so the reserve increment is visible exactly in this tick's record.
        matured_aid_mw = self._drain_mutual_aid(current_tick)
        if matured_aid_mw > 0.0:
            self._pending_reserve_extra_mw += matured_aid_mw
        realized_events = self._apply_perturbations_at_tick(current_tick)
        if self._simbench_profile_values is not None and current_tick == 0:
            realized_events.insert(
                0,
                {
                    "type": "simbench_profile_window_started",
                    "event_class": "telemetry",
                    "decision_required": False,
                    "actionable": False,
                    "tick": 0,
                    "profile_start_index": self._profile_start_index,
                    "profile_step": self._profile_step,
                },
            )
        if self._simbench_profile_values is not None:
            self._simbench_profile_applied_ticks.add(current_tick)
        if matured_aid_mw > 0.0:
            realized_events.append(
                {
                    "kind": "mutual_aid_arrived",
                    "event_class": "agent_outcome",
                    "origin": "agent_caused",
                    "decision_required": False,
                    "actionable": False,
                    "tick": current_tick,
                    "mw": round(matured_aid_mw, 3),
                }
            )

        # Compose chronic + perturbations into pandapower DataFrame state.
        tick_h = float(self._seed_obj.tick_minutes if self._seed_obj else 60) / 60.0

        # Legacy feeders use the original synthetic diurnal. SimBench feeders
        # consume their source-bundled full-year absolute profiles instead.
        peak_tick = max(1, int(self._horizon * 0.7))
        diurnal = 0.85 + 0.30 * math.sin(math.pi * current_tick / max(1, peak_tick))
        diurnal = max(0.6, min(1.20, diurnal))
        surge_factor = 1.0 + self._load_surge_factor_this_tick
        profile_index = self._profile_start_index + current_tick * self._profile_step
        profile_load_p = self._profile_frame("load", "p_mw", profile_index)
        profile_load_q = self._profile_frame("load", "q_mvar", profile_index)

        for idx, base in self._base_load_p_mw.items():
            # Pull this-tick shed quantity directly from the load entry.
            # v0.1.2 fix: `apply_tool_effect("shed_load")` wrote
            # `entry["shed_this_tick_mw"]` but tick was reading the
            # unrelated `_pending_shed_mw` dict, so shed actions had
            # zero physical effect — the oracle ended up paying both
            # the shed penalty AND the voltage-violation cost.
            load_id = self._load_id_by_index(idx)
            entry = self._loads.get(load_id, {})
            shed_mw = float(entry.get("shed_this_tick_mw", 0.0))
            source_p = self._profile_value(profile_load_p, idx, base * diurnal)
            source_q = self._profile_value(
                profile_load_q,
                idx,
                self._base_load_q_mvar.get(idx, 0.0) * diurnal,
            )
            new_p = max(0.0, source_p * surge_factor - shed_mw)
            self._net.load.at[idx, "p_mw"] = new_p
            # Q scales with P (constant power factor), and shed reduces
            # both P and Q proportionally.
            p_factor = new_p / max(source_p * surge_factor, 1e-09) if source_p > 0 else 1.0
            self._net.load.at[idx, "q_mvar"] = source_q * surge_factor * p_factor

        # Static gen (DER) profile: PV solar follows daylight curve, wind
        # follows the wind factor + the curtail cap.
        profile_sgen_p = self._profile_frame("sgen", "p_mw", profile_index)
        for idx, base in self._base_sgen_p_mw.items():
            kind = str(self._net.sgen.type.iloc[idx])
            klow = kind.lower()
            # pandapower CIGRE MV uses "WP" for Wind Power; also accept
            # "wind" / "wt" for compatibility with other networks.
            source_p = self._profile_value(profile_sgen_p, idx, -1.0)
            if source_p >= 0.0:
                factor = self._wind_factor_this_tick if (
                    "wind" in klow or klow in {"wt", "wp"}
                ) else 1.0
                variable_renewable_factor = (
                    self._renewable_output_factor_this_tick
                    if any(token in klow for token in ("pv", "solar", "res"))
                    else 1.0
                )
                new_p = (
                    source_p
                    * factor
                    * variable_renewable_factor
                )
            elif "wind" in klow or klow == "wt" or klow == "wp":
                new_p = base * self._wind_factor_this_tick
            else:
                # PV: solar daylight curve, no output at "night" (tick 0/end)
                solar = max(
                    0.0, math.sin(math.pi * current_tick / max(1, self._horizon))
                )
                new_p = base * solar
            cap = self._pending_curtail.get(idx)
            if cap is not None and cap < new_p:
                new_p = cap
            self._net.sgen.at[idx, "p_mw"] = new_p

        # Apply line status overrides
        for line_idx, connect in self._line_status_overrides.items():
            if 0 <= line_idx < len(self._net.line):
                self._net.line.at[line_idx, "in_service"] = connect

        # v0.6: push queued distribution-native Volt-Var setpoints (tap /
        # capacitor / DER reactive / storage) before the power flow solve.
        if self._volt_var_enabled:
            self._apply_volt_var_controls()

        # Power flow
        converged = True
        try:
            pp.runpp(self._net, numba=False)
            if self._source_constructor_state_digest is not None:
                self._source_solver_state_digest = self._solved_state_digest()
        except pp.LoadflowNotConverged:
            converged = False
        except Exception:
            converged = False

        # Build the tick record
        if converged:
            demand_mw = float(self._net.load.p_mw.sum())
            generation_mw = float(self._net.sgen.p_mw.sum()) + float(
                self._net.res_ext_grid.p_mw.sum()
                if "p_mw" in self._net.res_ext_grid.columns
                else 0.0
            )
            rho_values = self._net.res_line.loading_percent.to_numpy() / 100.0
            rho_max = float(rho_values.max()) if len(rho_values) > 0 else 0.0
            n_overload = int((rho_values > 1.0).sum())
            v_pu = self._net.res_bus.vm_pu.to_numpy()
            n_voltage_viol = int(
                ((v_pu < self.VOLTAGE_LOWER_PU) | (v_pu > self.VOLTAGE_UPPER_PU)).sum()
            )
            n_disc = int((~self._net.line.in_service.to_numpy()).sum())
        else:
            # BUG-6: non-convergence at tick 0 has no previous record →
            # using last=None silently zeroed every violation count.
            # Default to worst-case sentinels so the safety dimensions
            # see the failure instead of scoring 100% on a crashed solve.
            last = self._tick_records[-1] if self._tick_records else None
            demand_mw = float(self._net.load.p_mw.sum())
            generation_mw = last.aggregate_generation_mw if last else 0.0
            rho_max = last.rho_max if last else 2.0
            n_overload = last.n_overloads if last else len(self._net.line)
            n_voltage_viol = last.n_voltage_violations if last else len(self._net.bus)
            n_disc = last.n_disconnected_lines if last else 0

        # Per-class shed penalty
        shed_mw = 0.0
        shed_penalty_this_tick = 0.0
        for entry in self._loads.values():
            sh = float(entry.get("shed_this_tick_mw", 0.0))
            if sh <= 0:
                continue
            tariff = self.SHED_PENALTY_PER_MWH_BY_CLASS.get(
                str(entry.get("stakeholder_class", "")), self.SHED_PENALTY_DEFAULT
            )
            shed_penalty_this_tick += sh * tariff * tick_h
            shed_mw += sh

        # Production cost from external slack + DER (heuristic)
        prod_cost = self.PRODUCTION_COST_PER_MWH * demand_mw * tick_h
        # Add voltage-violation cost (distribution-specific)
        prod_cost += n_voltage_viol * self.VOLTAGE_VIOLATION_COST_PER_TICK
        prod_cost += n_overload * self.OVERLOAD_COST_PER_TICK
        prod_cost += n_disc * self.DISCONNECTION_COST_PER_LINE_TICK

        # Synthetic reserves (X% of demand). CIGRE MV has limited reserve
        # channels; we treat external-grid headroom as procured.
        reserves_required = self.RESERVE_TARGET_FRACTION_OF_DEMAND * demand_mw
        sgen_max_total = sum(self._base_sgen_p_mw.values())
        reserves_procured = (
            max(0.0, sgen_max_total - generation_mw) + self._pending_reserve_extra_mw
        )

        balance_error = generation_mw - demand_mw
        self._last_balance_error = balance_error

        record = CigreTickRecord(
            tick=current_tick,
            aggregate_demand_mw=round(demand_mw, 2),
            aggregate_generation_mw=round(generation_mw, 2),
            balance_error_mw=round(balance_error, 2),
            reserves_required_mw=round(reserves_required, 2),
            reserves_procured_mw=round(reserves_procured, 2),
            production_cost=round(prod_cost, 2),
            startup_cost=0.0,
            shed_penalty=round(shed_penalty_this_tick, 2),
            rho_max=rho_max,
            n_overloads=n_overload,
            n_voltage_violations=n_voltage_viol,
            n_disconnected_lines=n_disc,
            converged=converged,
            done=current_tick >= self._horizon - 1,
            realized_events=realized_events,
        )
        self._tick_records.append(record)

        # Reset per-tick counters
        for entry in self._loads.values():
            entry["shed_this_tick_mw"] = 0.0
        # `_pending_curtail` persists until the agent revisits it
        self._wind_factor_this_tick = 1.0
        self._renewable_output_factor_this_tick = 1.0
        self._load_surge_factor_this_tick = 0.0
        # DC-2: per-class surge factors are per-tick too
        self._class_surge_factors.clear()
        return record

    def _profile_frame(self, element: str, variable: str, index: int) -> Any:
        if self._simbench_profile_values is None:
            return None
        frame = self._simbench_profile_values.get((element, variable))
        if frame is None or len(frame) == 0:
            return None
        return frame.iloc[index % len(frame)]

    @staticmethod
    def _profile_value(row: Any, column: int, fallback: float) -> float:
        if row is None or column not in row.index:
            return fallback
        try:
            return float(row[column])
        except (TypeError, ValueError):
            return fallback

    def _load_id_by_index(self, idx: int) -> str:
        # NIT-2: O(1) lookup; the map is built in reset().
        return self._idx_to_load_id.get(idx, f"unknown_load_{idx}")

    # ── Perturbations ───────────────────────────────────────────────────

    def _declared_perturbation_event(
        self,
        *,
        perturbation_index: int,
        perturbation: Any,
        event_type: str,
        tick: int,
        changed_state_fields: list[str],
        materiality_metric: str,
        materiality_value: float,
        materiality_threshold: float,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Describe a procedural change after native state mutation.

        The seed/provenance remains source-locked, while these deterministic
        overlays are explicitly marked as declared perturbations.  A later
        decision opportunity is machine-readable so terminal-tick shocks
        cannot be mistaken for adaptive-response evidence.
        """
        expected_type, event_class = self._validate_perturbation_event(
            perturbation
        )
        if event_type != expected_type:
            raise ValueError(
                "CIGRE event type does not match registry: "
                f"{perturbation.kind!s} declares {event_type!r}, "
                f"expected {expected_type!r}"
            )
        start = int(perturbation.trigger_tick)
        response_tick = tick + 1
        has_response = response_tick < self._horizon
        actionable = has_response and not bool(perturbation.hidden)
        end = start + max(1, int(perturbation.duration_ticks))
        return {
            "event_id": f"cigre-procedural:{perturbation_index}:{event_type}:{start}",
            "type": event_type,
            "tick": tick,
            "origin": "declared_perturbation",
            "event_class": event_class,
            "declared_perturbation": True,
            "declared_event": {
                "kind": str(perturbation.kind),
                "trigger_tick": start,
                "duration_ticks": int(perturbation.duration_ticks),
                "target": dict(perturbation.target or {}),
                "procedural_variant": True,
            },
            "hidden": bool(perturbation.hidden),
            "actionable": actionable,
            "decision_required": actionable,
            "changed_state_fields": list(dict.fromkeys(changed_state_fields)),
            "materiality_metric": materiality_metric,
            "materiality_value": float(materiality_value),
            "materiality_threshold": float(materiality_threshold),
            "materiality_passed": abs(float(materiality_value))
            >= abs(float(materiality_threshold)),
            "response_window_required": True,
            "response_opportunity_tick": response_tick if has_response else None,
            "response_window_end_tick": min(end - 1, self._horizon - 1)
            if has_response
            else None,
            "terminal_response_window_missing": not has_response,
            **(payload or {}),
        }

    @staticmethod
    def _validate_perturbation_event(perturbation: Any) -> tuple[str, str]:
        kind = str(getattr(perturbation, "kind", "") or "")
        try:
            event_type, event_class = CIGRE_PERTURBATION_EVENT_REGISTRY[kind]
        except KeyError as exc:
            raise ValueError(
                f"unsupported CIGRE event kind: {kind or '<missing>'}"
            ) from exc
        declared_class = getattr(perturbation, "event_class", None)
        if declared_class is not None and str(declared_class) != event_class:
            raise ValueError(
                "CIGRE event class does not match registry: "
                f"{kind!r} declares {declared_class!r}, expected {event_class!r}"
            )
        return event_type, event_class

    def _apply_perturbations_at_tick(self, tick: int) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if self._seed_obj is None:
            return events
        # Telemetry confidence is a native observable, not merely a fog
        # configuration hint.  It recovers when a storm window expires.
        self._telemetry_confidence = 1.0
        # _sgen_outage_until is initialised in __init__ / reset() (BUG-2).
        for perturbation_index, p in enumerate(self._seed_obj.perturbations):
            start = int(p.trigger_tick)
            end = start + max(1, int(p.duration_ticks))
            if not (start <= tick < end):
                if tick == end and p.kind == "line_outage":
                    line_id = int(p.target.get("line_index", 0))
                    self._line_status_overrides[line_id] = True
                    events.append(
                        {
                            "type": "line_restored",
                            "event_class": "lifecycle",
                            "tick": tick,
                            "line_id": line_id,
                            "origin": "endogenous_completion",
                            "decision_required": False,
                            "actionable": False,
                            "changed_state_fields": ["line_in_service"],
                        }
                    )
                continue
            if p.kind == "line_outage":
                if tick == start:
                    line_id = int(p.target.get("line_index", 0))
                    self._line_status_overrides[line_id] = False
                    events.append(
                        self._declared_perturbation_event(
                            perturbation_index=perturbation_index,
                            perturbation=p,
                            event_type="line_outage",
                            tick=tick,
                            changed_state_fields=[
                                "line_in_service",
                                "line_loading_percent",
                                "bus_voltage_pu",
                            ],
                            materiality_metric="outaged_line_count",
                            materiality_value=1.0,
                            materiality_threshold=1.0,
                            payload={"line_id": line_id, "intensity": p.intensity},
                        )
                    )
            elif p.kind == "generator_forced_outage":
                # CIGRE v0.1.2: trip the indexed sgen by forcing its
                # output cap to 0 for the window AND emit a realized_event
                # so the scorer has evidence to attach.
                sgen_idx = int(p.target.get("index", 0)) % max(len(self._net.sgen), 1)
                self._sgen_outage_until[sgen_idx] = end
                self._pending_curtail[sgen_idx] = 0.0
                if tick == start:
                    events.append(
                        self._declared_perturbation_event(
                            perturbation_index=perturbation_index,
                            perturbation=p,
                            # Keep the canonical runtime type stable across
                            # power-grid backends; the source perturbation
                            # kind remains generator_forced_outage below.
                            event_type="generator_outage",
                            tick=tick,
                            changed_state_fields=[
                                "generator_availability",
                                "aggregate_generation_mw",
                                "reserves_procured_mw",
                            ],
                            materiality_metric="outaged_generator_count",
                            materiality_value=1.0,
                            materiality_threshold=1.0,
                            payload={
                                "generator_id": f"sgen_{sgen_idx}",
                                "intensity": p.intensity,
                                "perturbation_kind": "generator_forced_outage",
                            },
                        )
                    )
            elif p.kind == "load_surge":
                # DC-2: target the named stakeholder class only; fall
                # back to a global surge when no class is specified.
                target_class = p.target.get("stakeholder_class")
                if target_class:
                    self._class_surge_factors[str(target_class)] = float(p.intensity)
                else:
                    self._load_surge_factor_this_tick = float(p.intensity)
                if tick == start:
                    events.append(
                        self._declared_perturbation_event(
                            perturbation_index=perturbation_index,
                            perturbation=p,
                            event_type="load_surge",
                            tick=tick,
                            changed_state_fields=[
                                "aggregate_demand_mw",
                                "line_loading_percent",
                                "reserves_procured_mw",
                            ],
                            materiality_metric="demand_surge_fraction",
                            materiality_value=abs(float(p.intensity)),
                            materiality_threshold=0.01,
                            payload={
                                "intensity": p.intensity,
                                "stakeholder_class": target_class,
                            },
                        )
                    )
            elif p.kind == "wind_dropout":
                self._wind_factor_this_tick = max(0.05, 1.0 - float(p.intensity))
                if tick == start:
                    events.append(
                        self._declared_perturbation_event(
                            perturbation_index=perturbation_index,
                            perturbation=p,
                            event_type="wind_dropout",
                            tick=tick,
                            changed_state_fields=[
                                "wind_generation_mw",
                                "aggregate_generation_mw",
                            ],
                            materiality_metric="wind_output_multiplier_delta",
                            materiality_value=abs(float(p.intensity)),
                            materiality_threshold=0.01,
                            payload={"factor": self._wind_factor_this_tick},
                        )
                    )
            elif p.kind == "renewable_output_error":
                self._renewable_output_factor_this_tick = max(
                    0.0,
                    1.0 + float(p.intensity),
                )
                if tick == start:
                    events.append(
                        self._declared_perturbation_event(
                            perturbation_index=perturbation_index,
                            perturbation=p,
                            event_type="renewable_output_error",
                            tick=tick,
                            changed_state_fields=[
                                "renewable_generation_mw",
                                "aggregate_generation_mw",
                            ],
                            materiality_metric="renewable_output_multiplier_delta",
                            materiality_value=abs(float(p.intensity)),
                            materiality_threshold=0.01,
                            payload={
                                "factor": self._renewable_output_factor_this_tick
                            },
                        )
                    )
            elif p.kind == "forecast_bias":
                direction = p.target.get("bias_direction", "under-forecast")
                sign = 1.0 if direction == "under-forecast" else -1.0
                self._forecast_bias = sign * float(p.intensity)
                if tick == start:
                    events.append(
                        self._declared_perturbation_event(
                            perturbation_index=perturbation_index,
                            perturbation=p,
                            event_type="forecast_bias",
                            tick=tick,
                            changed_state_fields=["demand_forecast_mw"],
                            materiality_metric="forecast_bias_fraction",
                            materiality_value=abs(float(self._forecast_bias)),
                            materiality_threshold=0.01,
                            payload={"bias_direction": direction},
                        )
                    )
            elif p.kind == "storm_window":
                intensity = max(0.0, min(1.0, float(p.intensity)))
                self._telemetry_confidence = min(
                    self._telemetry_confidence,
                    max(0.05, 1.0 - intensity),
                )
                if tick == start:
                    events.append(
                        self._declared_perturbation_event(
                            perturbation_index=perturbation_index,
                            perturbation=p,
                            event_type="storm_window",
                            tick=tick,
                            changed_state_fields=[
                                "telemetry_confidence",
                            ],
                            materiality_metric="telemetry_confidence_loss",
                            materiality_value=intensity,
                            materiality_threshold=0.01,
                            payload={
                                "intensity": p.intensity,
                                "telemetry_confidence": self._telemetry_confidence,
                            },
                        )
                    )
        # Clear expired sgen outages (allows the sgen to ramp back).
        expired = [
            idx for idx, until in self._sgen_outage_until.items() if tick >= until
        ]
        for idx in expired:
            self._sgen_outage_until.pop(idx, None)
            # Don't auto-restore _pending_curtail — that requires an
            # explicit agent redispatch call (semantically correct).
        return events

    # ── Snapshot ────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        if self._net is None or self._seed_obj is None:
            return {"entities": {}, "totals": {}}
        entities: dict[str, dict[str, Any]] = {}
        # Loads
        for lid, entry in self._loads.items():
            idx = int(entry["load_index"])
            entities[lid] = {
                "kind": "load",
                "bus_id": int(self._net.load.bus.iloc[idx]),
                "current_demand_mw": float(self._net.load.p_mw.iloc[idx]),
                "stakeholder_class": entry["stakeholder_class"],
                "criticality": entry["criticality"],
                "cumulative_shed_mwh": round(
                    self._cumulative_shed_mwh.get(lid, 0.0), 3
                ),
            }
        # DER generators
        for i in range(len(self._net.sgen)):
            kind = str(self._net.sgen.type.iloc[i] or "")
            klow = kind.lower()
            sgen_kind = (
                "renewable"
                if any(k in klow for k in ("wind", "pv", "solar"))
                or klow in ("wp", "wt")
                else "generator"
            )
            # DC-1: expose the effective output cap so agents can tell
            # "physically outaged" (effective_cap_mw == 0) from
            # "available but pending redispatch".
            base_max = float(self._base_sgen_p_mw.get(i, 0.0))
            effective_cap = float(self._pending_curtail.get(i, base_max))
            entities[f"sgen_{i}"] = {
                "kind": sgen_kind,
                "bus_id": int(self._net.sgen.bus.iloc[i]),
                "output_mw": float(self._net.sgen.p_mw.iloc[i]),
                "q_mvar": float(self._net.sgen.q_mvar.iloc[i])
                if "q_mvar" in self._net.sgen
                else 0.0,
                "power_max": base_max,
                "effective_cap_mw": effective_cap,
                "type": kind,
            }
        # Lines (basic visibility — fog can hide flow values)
        for i in range(len(self._net.line)):
            entities[f"line_{i}"] = {
                "kind": "line",
                "from_bus": int(self._net.line.from_bus.iloc[i]),
                "to_bus": int(self._net.line.to_bus.iloc[i]),
                "in_service": bool(self._net.line.in_service.iloc[i]),
                "loading_percent": float(
                    self._net.res_line.loading_percent.iloc[i]
                    if hasattr(self._net, "res_line") and len(self._net.res_line) > i
                    else 0.0
                ),
            }
        # v0.6: expose per-bus voltages + controllable Volt-Var assets so an
        # agent/oracle can observe and steer them (only when enabled).
        if self._volt_var_enabled:
            if hasattr(self._net, "res_bus") and len(self._net.res_bus):
                for b in range(len(self._net.bus)):
                    try:
                        vm = float(self._net.res_bus.vm_pu.iloc[b])
                    except Exception:
                        continue
                    if not math.isfinite(vm):  # isolated / OOS bus
                        continue
                    entities[f"bus_{b}"] = {"kind": "bus", "vm_pu": round(vm, 4)}
            for i in range(len(self._net.trafo)):
                try:
                    tp = self._net.trafo.tap_pos.iloc[i]
                    entities[f"trafo_{i}"] = {
                        "kind": "transformer",
                        "tap_pos": int(tp) if tp == tp else 0,
                    }
                except Exception:
                    pass
            for ci in self._cap_shunt_idx:
                if ci < len(self._net.shunt):
                    entities[f"capacitor_{ci}"] = {
                        "kind": "capacitor",
                        "bus_id": int(self._net.shunt.bus.iloc[ci]),
                        "on": bool(self._net.shunt.in_service.iloc[ci]),
                        "q_mvar": float(self._net.shunt.q_mvar.iloc[ci]),
                    }
            for s in range(len(self._net.storage)):
                entities[f"storage_{s}"] = {
                    "kind": "battery",
                    "bus_id": int(self._net.storage.bus.iloc[s]),
                    "p_mw": float(self._net.storage.p_mw.iloc[s]),
                }

        last = self._tick_records[-1] if self._tick_records else None
        totals = {
            "demand_mw": float(self._net.load.p_mw.sum()),
            "der_generation_mw": float(self._net.sgen.p_mw.sum()),
            "telemetry_confidence": round(self._telemetry_confidence, 4),
            "pending_reserve_mw": round(self._pending_reserve_extra_mw, 4),
            "balance_error_mw": last.balance_error_mw if last else 0.0,
            "reserves_required_mw": last.reserves_required_mw if last else 0.0,
            "reserves_procured_mw": last.reserves_procured_mw if last else 0.0,
            "rho_max": last.rho_max if last else 0.0,
            "n_voltage_violations": last.n_voltage_violations if last else 0,
            "n_overloads": last.n_overloads if last else 0,
            "n_disconnected_lines": last.n_disconnected_lines if last else 0,
        }
        return {
            "tick": self._tick,
            "horizon": self._horizon,
            "telemetry_confidence": round(self._telemetry_confidence, 4),
            "entities": entities,
            "totals": totals,
        }

    def scoring_records(self) -> list[dict[str, Any]]:
        """Per-tick rows for ``evaluation.scorer`` (v0.3.1 P0 fix).

        Pre-v0.3.1 this backend had NO ``scoring_records()`` method, so the
        live ``run.py`` path (``hasattr(backend, "scoring_records")``) fell
        through to an empty list and ``system_survival`` /
        ``safety_violation`` / ``adaptive_replanning`` were silently dropped
        (``applicable=False``) for every ``distribution_volt_var`` scenario —
        even though this AC feeder is the only backend that natively measures
        bus-voltage collapse. The ``CigreTickRecord`` already carries every
        field (see the dataclass above); this method just exposes them under
        the canonical scorer keys that ``score_safety_violation`` /
        ``score_system_survival`` read.

        ``done`` is early-guarded (``r.tick < horizon - 1``) so a normal
        horizon-end tick is never miscounted as a catastrophic blackout by
        ``score_system_survival`` (which has no horizon guard of its own).
        """
        return [
            {
                "tick": r.tick,
                "aggregate_demand_mw": r.aggregate_demand_mw,
                "aggregate_generation_mw": r.aggregate_generation_mw,
                "balance_error_mw": r.balance_error_mw,
                "reserves_required_mw": r.reserves_required_mw,
                "reserves_procured_mw": r.reserves_procured_mw,
                "production_cost": r.production_cost,
                "startup_cost": r.startup_cost,
                "shed_penalty": r.shed_penalty,
                "rho_max": float(r.rho_max),
                "n_overloads": int(r.n_overloads),
                "n_voltage_violations": int(r.n_voltage_violations),
                "n_disconnected_lines": int(r.n_disconnected_lines),
                "done": bool(r.done and r.tick < self._horizon - 1),
                "converged": bool(r.converged),
            }
            for r in self._tick_records
        ]

    def per_load_shed_mwh(self) -> dict[str, float]:
        return dict(self._cumulative_shed_mwh)

    def ground_truth_costs(self) -> dict[str, float]:
        """Roll up the per-tick costs into a single dict for the scorer."""
        voltage = sum(
            r.n_voltage_violations * self.VOLTAGE_VIOLATION_COST_PER_TICK
            for r in self._tick_records
        )
        overload = sum(
            r.n_overloads * self.OVERLOAD_COST_PER_TICK
            for r in self._tick_records
        )
        disconnection = sum(
            r.n_disconnected_lines * self.DISCONNECTION_COST_PER_LINE_TICK
            for r in self._tick_records
        )
        # Tick records keep the historical all-in operating cost. Split its
        # native reliability penalties here so task completion can prove
        # state-changing mitigation without changing the summed episode cost.
        production = (
            sum(r.production_cost for r in self._tick_records)
            - voltage
            - overload
            - disconnection
        )
        startup = sum(r.startup_cost for r in self._tick_records)
        shed = sum(r.shed_penalty for r in self._tick_records)
        balance = sum(
            abs(r.balance_error_mw) * 100.0
            for r in self._tick_records
            if r.balance_error_mw < -10.0
        )
        return {
            "production_cost": round(production, 2),
            "startup_cost": round(startup, 2),
            "shed_penalty": round(shed, 2),
            "voltage_violation_cost": round(voltage, 2),
            "overload_cost": round(overload, 2),
            "disconnection_cost": round(disconnection, 2),
            "balance_error_cost": round(balance, 2),
            "reserve_violation_cost": 0.0,
        }

    def forecast_for(self, horizon_ticks: int) -> list[dict[str, Any]]:
        """Synthetic forecast that mirrors the underlying diurnal curve
        but applies the active ``forecast_bias`` so under-forecast biases
        propagate to the agent's `forecast_query` call.
        """
        out: list[dict[str, Any]] = []
        if self._seed_obj is None:
            return out
        peak_tick = max(1, int(self._horizon * 0.7))
        base_total = sum(self._base_load_p_mw.values())
        for offset in range(1, max(1, horizon_ticks) + 1):
            t = self._tick + offset
            diurnal = 0.85 + 0.30 * math.sin(math.pi * t / max(1, peak_tick))
            diurnal = max(0.6, min(1.20, diurnal))
            true_demand = base_total * diurnal
            # DC-10: no artificial floor — match pglib backend semantics.
            biased = true_demand * max(0.0, 1.0 - self._forecast_bias)
            out.append(
                {
                    "tick": t,
                    "demand_mw_forecast": round(biased, 2),
                    "wind_factor_forecast": round(0.9, 3),
                    "solar_factor_forecast": round(
                        max(0.0, math.sin(math.pi * t / max(1, self._horizon))), 3
                    ),
                }
            )
        return out
