"""
baselines.greedy_heuristic — Priority-list shed + simple redispatch
heuristic with backend-aware overload handling.

Rules (in priority order):

1. **Dilemma**: on the first active dilemma, immediately pick the first
   non-fatal option with a rationale.
2. **Line overload** (Grid2Op or other backend exposing ``rho``):
   if any line has rho > 1.05, shed a residential load to relieve.
3. **Disconnected line** (rho == 0 + status False):
   no recovery — but bring up an idle generator if balance dips.
4. **Balance error** (synthetic UC backend):
   under-gen → bring up largest idle gen; over-gen → trim largest
   committed gen.
5. **Reserve shortfall** > 5 MW: ``commit_reserve``.
6. **Periodic refresh**: every 4 ticks ``query_grid_state``.

This is a strong-enough baseline that LLM agents must beat it to be
considered "useful as a dispatch center".
"""

from __future__ import annotations

from typing import Any

from core import Action, ToolCall
from core.pomdp_env import POMDPEnvironment

from .base import BaselineAgent

_SOURCE_EVENT_ORIGINS = {
    "source_schedule",
    "source_trace",
    "declared_perturbation",
}


def _visible_source_evidence_ids(observation: dict[str, Any]) -> list[str]:
    """Return evidence attached to source events visible in this observation."""

    evidence_ids: list[str] = []
    for event in observation.get("__last_realized_events__") or []:
        if (
            not isinstance(event, dict)
            or str(event.get("origin") or "") not in _SOURCE_EVENT_ORIGINS
            or event.get("hidden") is True
            or event.get("actionable") is not True
            or event.get("decision_required") is not True
            or not str(event.get("event_id") or "").strip()
            or event.get("materiality_passed") is False
        ):
            continue
        for evidence_id in event.get("evidence_ids") or []:
            value = str(evidence_id).strip()
            if value and value not in evidence_ids:
                evidence_ids.append(value)
    return evidence_ids


class GreedyHeuristicAgent(BaselineAgent):
    name = "greedy_heuristic"

    # Default thresholds suit transmission (RTS-GMLC / pglib / Grid2Op).
    # Re-scaled at reset() against the observed network size so the agent
    # also triggers on distribution-scale networks (CIGRE MV ~ 45 MW).
    BALANCE_DEAD_BAND_MW = 50.0
    LOAD_SHED_MIN_MW = 5.0

    def __init__(self) -> None:
        self._tick = 0
        self._gen_ids: list[str] = []
        self._chose_for: set[str] = set()
        self._scenario_config: dict[str, Any] = {}
        # Effective thresholds after network-scale autocalibration.
        self._eff_balance_band = self.BALANCE_DEAD_BAND_MW
        self._eff_load_shed_min = self.LOAD_SHED_MIN_MW

    def reset(
        self, env: POMDPEnvironment, scenario_config: dict[str, Any], seed: int
    ) -> None:
        self._tick = 0
        self._reset_idem_seq()
        self._scenario_config = dict(scenario_config or {})
        self._chose_for.clear()
        self._rho_history = []
        self._mutual_aid_requested = False
        # BUG-3: hysteresis bookkeeping must restart per-episode or the
        # voltage-shed branch is suppressed for ~10 ticks of every
        # batch run after the first.
        self._last_volt_shed_tick = -9999
        obs = env.snapshot()
        self._gen_ids = [
            eid
            for eid, e in obs.get("entities", {}).items()
            if e.get("kind") == "generator"
        ]
        # Auto-scale: if total demand is much smaller than the default
        # threshold (e.g. CIGRE MV ~ 45 MW vs default 50 MW band), drop
        # both dead-band and minimum shed to ~1% of aggregate demand.
        total_demand = float(
            obs.get("totals", {}).get(
                "aggregate_demand_mw",
                obs.get("totals", {}).get("demand_mw", 0.0),
            )
            or 0.0
        )
        if total_demand > 0 and total_demand < 500.0:
            self._eff_balance_band = max(0.5, total_demand * 0.02)
            self._eff_load_shed_min = max(0.1, total_demand * 0.01)
        else:
            self._eff_balance_band = self.BALANCE_DEAD_BAND_MW
            self._eff_load_shed_min = self.LOAD_SHED_MIN_MW

    def act(
        self, observation: dict[str, Any], tool_specs: list[dict[str, Any]]
    ) -> Action:
        self._tick += 1
        calls: list[ToolCall] = []
        avail_tools = {
            name
            for spec in (tool_specs or [])
            if isinstance(spec, dict)
            for name in (
                spec.get("name"),
                (spec.get("function") or {}).get("name")
                if isinstance(spec.get("function"), dict)
                else None,
            )
            if name
        }

        # Traffic corridor control (v0.8): reactive single-corridor relief.
        # Mirrors ``scripts/traffic_behavioral_gate._policy_greedy`` — grant the
        # max-throughput relief program to the most critical high-demand
        # corridor each tick. A state-reactive heuristic that an LLM dispatch
        # center must beat, sitting between wait_only and the all-corridor
        # oracle on the released traffic families.
        if str(self._scenario_config.get("domain", "power_grid")) == "traffic":
            return self._traffic_greedy_action(observation, avail_tools)
        if (
            str(self._scenario_config.get("domain", "power_grid"))
            == "autonomous_driving"
        ):
            from .autonomous_driving_policy import greedy_action

            return greedy_action(observation, tool_specs)

        # 1. Resolve any active dilemma (pick the non-fatal option once)
        for dilemma in observation.get("active_dilemmas", []) or []:
            did = dilemma.get("dilemma_id", "")
            if did in self._chose_for:
                continue
            non_fatal = next(
                (o for o in dilemma.get("options", []) if not o.get("fatal")),
                None,
            )
            if non_fatal is None:
                continue
            calls.append(
                ToolCall(
                    name="moral_choice",
                    args={
                        "dilemma_id": did,
                        "option_id": non_fatal["option_id"],
                        "rationale": (
                            "Greedy baseline: pick the first non-fatal option. "
                            "Avoids critical-load shedding while accepting "
                            "reversible economic / fairness costs."
                        ),
                    },
                    idempotency_key=f"mc_{did}",
                )
            )
            self._chose_for.add(did)

        # 2. Line-overload response (Grid2Op exposes rho per line).
        # Strategy: escalate gradually. Mild overload (rho>1.05) → only
        # commit extra reserves and request mutual aid. Sustained severe
        # overload (rho>1.5 for 2+ ticks) → shed a non-critical bus by a
        # larger fraction (60%) so synthetic relief actually offsets the
        # shed tariff. A single-tick spike often clears on its own
        # (chronics-driven), and shedding for it pays without benefit.
        totals = observation.get("totals", {}) or {}
        rho_max = float(totals.get("rho_max", 0.0))
        self._rho_history.append(rho_max)
        if len(self._rho_history) > 4:
            self._rho_history.pop(0)
        prev_rho = self._rho_history[-2] if len(self._rho_history) >= 2 else 0.0

        if rho_max > 1.05:
            calls.append(
                ToolCall(
                    name="commit_reserve",
                    args={"mw": round(max(10.0, rho_max * 5), 1)},
                    idempotency_key=f"cr_overload_{self._tick}",
                )
            )
            if rho_max > 2.0 and not self._mutual_aid_requested:
                calls.append(
                    ToolCall(
                        name="request_mutual_aid",
                        args={"neighbor": "iso_east", "mw": 30.0},
                        idempotency_key=f"ma_{self._tick}",
                    )
                )
                self._mutual_aid_requested = True

        if rho_max > 1.5 and prev_rho > 1.5:
            load_candidates = [
                (eid, e)
                for eid, e in observation.get("entities", {}).items()
                if e.get("kind") == "load"
                and float(e.get("current_demand_mw", 0)) > self._eff_load_shed_min
                and float(e.get("criticality", 1.0)) < 0.45
            ]
            if load_candidates:
                load_candidates.sort(
                    key=lambda kv: float(kv[1].get("criticality", 1.0))
                )
                lid, e = load_candidates[0]
                shed_mw = max(8.0, float(e.get("current_demand_mw", 10.0)) * 0.6)
                calls.append(
                    ToolCall(
                        name="shed_load",
                        args={
                            "load_id": lid,
                            "mw": round(shed_mw, 1),
                            "reason": (
                                f"sustained severe overload rho_max={rho_max:.2f}"
                                f" (prev={prev_rho:.2f})"
                            ),
                        },
                        idempotency_key=f"shed_sustained_{lid}_{self._tick}",
                    )
                )

        # 2b. Voltage violation response (distribution backends).
        # Same calibrated thresholds as oracle: act only when ≥3 buses
        # are outside [0.95, 1.05] pu and apply a small (30%) shed with
        # a 3-tick cooldown so we don't over-pay the shed tariff.
        n_volt_viol = int(totals.get("n_voltage_violations", 0))
        last_volt_shed_tick = getattr(self, "_last_volt_shed_tick", -10)
        if n_volt_viol >= 3 and (self._tick - last_volt_shed_tick) >= 3:
            load_candidates = [
                (eid, e)
                for eid, e in observation.get("entities", {}).items()
                if e.get("kind") == "load"
                and float(e.get("current_demand_mw", 0)) > self._eff_load_shed_min
                and float(e.get("criticality", 1.0)) < 0.45
            ]
            if load_candidates:
                load_candidates.sort(
                    key=lambda kv: float(kv[1].get("current_demand_mw", 0.0)),
                    reverse=True,
                )
                lid, e = load_candidates[0]
                shed_mw = max(0.5, float(e.get("current_demand_mw", 1.0)) * 0.30)
                calls.append(
                    ToolCall(
                        name="shed_load",
                        args={
                            "load_id": lid,
                            "mw": round(shed_mw, 2),
                            "reason": (
                                f"voltage violations on {n_volt_viol} buses; "
                                f"shedding lowest-criticality load on radial feeder"
                            ),
                        },
                        idempotency_key=f"shed_volt_{lid}_{self._tick}",
                    )
                )
                self._last_volt_shed_tick = self._tick

        # 3. Balance check (synthetic backend's main signal)
        balance_err = float(totals.get("balance_error_mw", 0.0))
        if balance_err < -self._eff_balance_band:
            # under-generation — bring up the largest non-committed gen
            candidates = [
                (eid, e)
                for eid, e in observation.get("entities", {}).items()
                if e.get("kind") == "generator"
                and not e.get("committed", True)
                and e.get("hours_down", 0) >= 1
            ]
            if candidates:
                candidates.sort(
                    key=lambda kv: float(kv[1].get("power_max", 0.0)), reverse=True
                )
                top_gid = candidates[0][0]
                calls.append(
                    ToolCall(
                        name="redispatch_generation",
                        args={
                            "generator_id": top_gid,
                            "target_mw": float(candidates[0][1].get("power_max", 100.0))
                            * 0.7,
                            "commit": True,
                        },
                        idempotency_key=f"rd_{top_gid}_{self._tick}",
                    )
                )
        elif balance_err > self._eff_balance_band:
            committed = [
                (eid, e)
                for eid, e in observation.get("entities", {}).items()
                if e.get("kind") == "generator" and e.get("committed", False)
            ]
            if committed:
                committed.sort(
                    key=lambda kv: float(kv[1].get("output_mw", 0.0)), reverse=True
                )
                top_gid = committed[0][0]
                calls.append(
                    ToolCall(
                        name="redispatch_generation",
                        args={
                            "generator_id": top_gid,
                            "target_mw": float(committed[0][1].get("output_mw", 50.0))
                            * 0.95,
                            "commit": True,
                        },
                        idempotency_key=f"rd_trim_{top_gid}_{self._tick}",
                    )
                )

        # 3. Reserve shortfall
        shortfall = max(
            0.0,
            float(totals.get("reserves_required_mw", 0.0))
            - float(totals.get("reserves_procured_mw", 0.0)),
        )
        if shortfall > 5.0:
            calls.append(
                ToolCall(
                    name="commit_reserve",
                    args={"mw": round(shortfall, 1)},
                    idempotency_key=f"cr_{self._tick}",
                )
            )

        # 4. Periodic state refresh
        if self._tick % 4 == 0:
            calls.append(
                ToolCall(
                    name="query_grid_state",
                    idempotency_key=self._next_idem_key("qgs"),
                )
            )

        # This baseline is shared across native domains. Filter its
        # power-grid heuristics against the registry so unavailable tools do
        # not leak into trajectories as unknown calls.
        calls = [call for call in calls if call.name in avail_tools]
        if not calls and "wait" in avail_tools:
            calls.append(
                ToolCall(name="wait", idempotency_key=self._next_idem_key("w"))
            )

        dominant = calls[0].name if calls else "wait"
        return Action(tool_calls=calls[:4], dominant=dominant)

    def _traffic_greedy_action(
        self,
        observation: dict[str, Any],
        avail_tools: set[str] | None = None,
    ) -> Action:
        corridors = {
            cid: attrs
            for cid, attrs in (observation.get("entities") or {}).items()
            if isinstance(attrs, dict) and attrs.get("kind") == "corridor"
        }
        backend_config = self._scenario_config.get("backend_config") or {}
        if backend_config.get("live_phase_control") is True:
            runtime_tls = (
                (observation.get("runtime_signal_control") or {}).get("tls")
                or {}
            )
            capture_rows = (
                (observation.get("vehicle_control_capture") or {}).get(
                    "records"
                )
                or []
            )
            pressure: dict[str, dict[str, int]] = {}
            for row in capture_rows:
                tls_id = str(
                    (row.get("tls_context") or {}).get("tls_id") or ""
                )
                signal = str(
                    (row.get("phase_context") or {}).get(
                        "link_signal_state"
                    )
                    or ""
                )
                if not tls_id:
                    continue
                bucket = pressure.setdefault(tls_id, {"green": 0, "red": 0})
                if signal in {"g", "G"}:
                    bucket["green"] += 1
                elif signal in {"r", "R"}:
                    bucket["red"] += 1
            candidates = [
                tls_id
                for tls_id, row in runtime_tls.items()
                if any(
                    signal in {"g", "G"}
                    for signal in str(row.get("current_state") or "")
                )
            ]
            if not candidates:
                return Action(
                    tool_calls=[
                        ToolCall(
                            name="wait",
                            idempotency_key=self._next_idem_key("w"),
                        )
                    ],
                    dominant="wait",
                )
            tls_id = max(
                candidates,
                key=lambda value: (
                    sum(pressure.get(value, {}).values()),
                    value,
                ),
            )
            runtime = runtime_tls[tls_id]
            bounds = runtime.get("current_phase_bounds") or {}
            minimum = float(bounds.get("min_duration") or 0.0)
            maximum = float(bounds.get("max_duration") or 0.0)
            if minimum <= 0.0 or maximum < minimum:
                return Action(
                    tool_calls=[
                        ToolCall(
                            name="wait",
                            idempotency_key=self._next_idem_key("w"),
                        )
                    ],
                    dominant="wait",
                )
            tls_pressure = pressure.get(tls_id, {})
            desired = (
                min(maximum, max(minimum, 30.0))
                if tls_pressure.get("green", 0)
                >= tls_pressure.get("red", 0)
                else minimum
            )
            calls = [
                ToolCall(
                    name="set_signal_phase_duration",
                    args={
                        "tls_id": tls_id,
                        "observed_program": str(
                            runtime.get("current_program") or ""
                        ),
                        "observed_phase": int(
                            runtime.get("current_phase") or 0
                        ),
                        "remaining_duration_seconds": desired,
                    },
                    idempotency_key=(
                        f"greedy_tr_phase_{tls_id}_{self._tick}"
                    ),
                    consumes_evidence_ids=(
                        _visible_source_evidence_ids(observation) or None
                    ),
                )
            ]
            plan_marker = "traffic_live_sumo_phase_supervision_plan"
            if (
                str(self._scenario_config.get("domain", "")) == "traffic"
                and str(self._scenario_config.get("backend_kind", "")) == "sumo"
                and "commit_to_plan" in (avail_tools or set())
                and plan_marker not in self._chose_for
            ):
                calls.append(
                    ToolCall(
                        name="commit_to_plan",
                        args={
                            "plan_id": "greedy-live-sumo-phase-supervision-v1",
                            "horizon_ticks": max(
                                2,
                                int(
                                    self._scenario_config.get("horizon_ticks")
                                    or 2
                                ),
                            ),
                            "review_after_ticks": 2,
                            "rationale": (
                                "Apply observed queue-pressure phase control "
                                "while native SUMO advances, then reassess the "
                                "signal at each supervisory decision."
                            ),
                        },
                        idempotency_key=self._next_idem_key("greedy_tr_plan"),
                    )
                )
                self._chose_for.add(plan_marker)
            return Action(
                tool_calls=calls,
                dominant="set_signal_phase_duration",
            )
        if not corridors:
            return Action(
                tool_calls=[
                    ToolCall(name="wait", idempotency_key=self._next_idem_key("w"))
                ],
                dominant="wait",
            )

        def impact(cid: str) -> tuple[float, str]:
            attrs = corridors[cid]
            criticality = float(attrs.get("criticality") or 0.0)
            demand = float(attrs.get("demand_veh") or 0.0)
            return (-(criticality * demand), cid)

        worst = min(corridors, key=impact)
        return Action(
            tool_calls=[
                ToolCall(
                    name="change_signal_plan",
                    args={"corridor": worst, "program": "incident_relief"},
                    idempotency_key=f"greedy_tr_relief_{worst}_{self._tick}",
                )
            ],
            dominant="change_signal_plan",
        )
