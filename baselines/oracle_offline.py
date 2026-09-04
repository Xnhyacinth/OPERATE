"""
baselines.oracle_offline — Knows-all offline upper bound.

This agent peeks at the env's ground_truth (which is normally hidden from
the agent — that's why it's an *oracle*) to make perfectly-informed
decisions. It is intentionally not LLM-based; it's a deterministic
priority-list dispatcher that uses perfect chronics knowledge.

Use as the leaderboard's upper-bound row. The gap from `oracle_offline`
to `greedy_heuristic` measures how much heads-up perfect info is worth;
the gap from `oracle_offline` to LLM models measures how close LLMs get
to the omniscient bound under the same physics.
"""

from __future__ import annotations

import math
from typing import Any

from core import Action, ToolCall
from core.difficulty_levels import canonical_difficulty_level
from core.pomdp_env import POMDPEnvironment

from .base import BaselineAgent
from .greedy_heuristic import _visible_source_evidence_ids


class OracleOfflineAgent(BaselineAgent):
    name = "oracle_offline"

    LOAD_SHED_MIN_MW_DEFAULT = 5.0

    def __init__(self) -> None:
        self._env: POMDPEnvironment | None = None
        self._tick = 0
        self._chose_for: set[str] = set()
        self._eff_load_shed_min = self.LOAD_SHED_MIN_MW_DEFAULT
        self._scenario_config: dict[str, Any] = {}
        self._reviewed_datacenter_event_ids: set[str] = set()
        self._opendss_solar_ramp_transient_source_event: tuple[str, int] | None = None

    def reset(
        self, env: POMDPEnvironment, scenario_config: dict[str, Any], seed: int
    ) -> None:
        self._env = env
        self._scenario_config = dict(scenario_config or {})
        self._tick = 0
        self._reset_idem_seq()
        self._chose_for.clear()
        self._rho_history = []
        # BUG-3: see greedy_heuristic — restart hysteresis per-episode.
        self._last_volt_shed_tick = -9999
        # v0.6: previous-tick ground-truth voltage-violation count, used to
        # require a *sustained* violation before the oracle steers reactive
        # power (see the Volt-Var branch below).
        self._prev_n_volt_viol = 0
        # v0.6: reactive devices the oracle has energised for voltage support.
        # Tracked so that once the violation clears the oracle RELEASES them
        # (caps OFF, DER q→0) instead of leaving them latched — a latched
        # capacitor/DER setpoint overshoots on a later load swing and creates
        # a fresh, oracle-induced violation (observed on mv_oberrhein medium).
        self._engaged_caps: set[int] = set()
        self._engaged_ders: set[int] = set()
        self._lv_source_prepositioned = False
        self._native_state_loss_milestones_done: set[str] = set()
        self._microgrid_battery_retry_state: dict[str, Any] | None = None
        self._reviewed_datacenter_event_ids.clear()
        self._opendss_solar_ramp_transient_source_event = None
        # Network-scale auto-calibration (same logic as the greedy
        # baseline): for distribution-scale networks, lower the load
        # shed minimum so the oracle's preemptive shed branch can fire.
        obs = env.snapshot()
        total_demand = float(
            obs.get("totals", {}).get(
                "aggregate_demand_mw",
                obs.get("totals", {}).get("demand_mw", 0.0),
            )
            or 0.0
        )
        if total_demand > 0 and total_demand < 500.0:
            self._eff_load_shed_min = max(0.1, total_demand * 0.01)
        else:
            self._eff_load_shed_min = self.LOAD_SHED_MIN_MW_DEFAULT

    def _pglib_native_reference_operations(
        self,
        *,
        avail_tools: set[str],
    ) -> list[dict[str, Any]]:
        assert self._env is not None
        native_backend = getattr(self._env, "_backend", None)
        reference_dispatch = getattr(
            native_backend,
            "native_oracle_reference_dispatch",
            None,
        )
        if not callable(reference_dispatch):
            raise RuntimeError("pglib_uc_native_reference_unavailable")
        difficulty = canonical_difficulty_level(
            str(self._scenario_config.get("difficulty_level", "basic"))
        )
        max_calls = {
            "basic": 3,
            "medium": 4,
            "high": 5,
            "extreme": 6,
        }[difficulty]
        operations = list(reference_dispatch(max_calls=max_calls) or [])
        diagnostics_fn = getattr(
            native_backend,
            "native_oracle_reference_diagnostics",
            None,
        )
        diagnostics = diagnostics_fn() if callable(diagnostics_fn) else {}
        status = str((diagnostics or {}).get("status") or "")
        if not operations and status != "optimal":
            raise RuntimeError(
                "pglib_uc_native_reference_failed: "
                f"status={status or 'missing'}"
            )

        supported: list[dict[str, Any]] = []
        for operation in operations:
            tool_name = str(operation.get("tool") or "")
            args = dict(operation.get("args") or {})
            if tool_name == "dispatch_generation_portfolio" and not (
                isinstance(args.get("dispatches"), list)
                and args["dispatches"]
            ):
                raise RuntimeError("pglib_uc_native_reference_empty_portfolio")
            if tool_name in avail_tools:
                supported.append({"tool": tool_name, "args": args})
        if operations and not supported:
            raise RuntimeError("pglib_uc_native_reference_tool_unavailable")
        return supported

    def reconcile_control_receipts(
        self,
        observation: dict[str, Any],
        tool_specs: list[dict[str, Any]],
    ) -> Action:
        """Recompute one PGLib control after a same-tick injected failure."""
        if str(self._scenario_config.get("backend_kind") or "") != (
            "pglib_uc_synthetic"
        ):
            return Action()
        avail_tools = {
            name
            for spec in tool_specs or []
            if isinstance(spec, dict)
            for name in [
                str(
                    spec.get("name")
                    or (spec.get("function") or {}).get("name")
                    or ""
                )
            ]
            if name
        }
        calls_by_id = {
            str(call.get("call_id") or ""): call
            for call in observation.get("__control_calls__") or []
            if isinstance(call, dict) and call.get("call_id")
        }
        receipt = next(
            (
                row
                for row in observation.get("__control_receipts__") or []
                if isinstance(row, dict)
                and row.get("ok") is False
                and row.get("error_code") == "INJECTED_FAILURE"
                and row.get("state_changing") is True
                and row.get("call_id")
            ),
            None,
        )
        if receipt is None:
            return Action()
        parent_id = str(receipt["call_id"])
        original = calls_by_id.get(parent_id)
        if original is None:
            return Action()
        failed_tool = str(receipt.get("name") or "")
        if failed_tool == "dispatch_generation_portfolio":
            operations = self._pglib_native_reference_operations(
                avail_tools=avail_tools
            )
            operation = next(
                (
                    row
                    for row in operations
                    if row["tool"] == failed_tool
                ),
                None,
            )
            if operation is None:
                raise RuntimeError(
                    "pglib_uc_native_reference_missing_retry_portfolio"
                )
            args = dict(operation["args"])
        else:
            args = dict(original.get("args") or {})
        retry = ToolCall(
            name=failed_tool,
            args=args,
            idempotency_key=self._next_idem_key(
                f"orc_pglib_uc_retry_{failed_tool}_{self._tick}"
            ),
            depends_on_call_ids=[parent_id],
        )
        return Action(tool_calls=[retry], dominant=retry.name)

    def act(
        self, observation: dict[str, Any], tool_specs: list[dict[str, Any]]
    ) -> Action:
        assert self._env is not None
        self._tick += 1
        # Peek at the ground truth — this is the "oracle" privilege
        gt = self._env.ground_truth()
        gt_entities = gt.get("entities", {})
        avail_tools = {
            n
            for spec in (tool_specs or [])
            if isinstance(spec, dict)
            for n in (
                spec.get("name"),
                (spec.get("function") or {}).get("name")
                if isinstance(spec.get("function"), dict)
                else None,
            )
            if n
        }
        gens = [
            (eid, e) for eid, e in gt_entities.items() if e.get("kind") == "generator"
        ]

        # Traffic corridor-control oracle (v0.8). Unlike the power-grid levers
        # below, signal control acts every tick (a relief program persists but
        # is overridden by a later ``signal_failure`` shock, so the oracle
        # re-grants it) and is governed by the traffic env's own per-tick tool
        # budget rather than the 4-call power-grid cap — so it mirrors the
        # decision headroom already proven by ``scripts/traffic_behavioral_gate``
        # instead of being throttled to a power-grid budget.
        if str(self._scenario_config.get("domain", "power_grid")) == "traffic":
            traffic_calls = self._traffic_oracle_calls(observation, avail_tools)
            if traffic_calls:
                return Action(tool_calls=traffic_calls, dominant=traffic_calls[0].name)
            return Action(
                tool_calls=[
                    ToolCall(name="wait", idempotency_key=self._next_idem_key("orc_w"))
                ],
                dominant="wait",
            )
        if (
            str(self._scenario_config.get("domain", "power_grid"))
            == "autonomous_driving"
        ):
            from .autonomous_driving_policy import oracle_action

            return oracle_action(
                observation,
                tool_specs,
                self._scenario_config,
            )

        calls: list[ToolCall] = []

        # v0.6 (scoped, frozen-safe): on a per-tick AC-OPF backend the
        # ``wait_only`` baseline already solves a full optimal power flow
        # (``pp.runopp``) every tick, so any oracle dispatch/shed/commit lever
        # only PINS setpoints and degrades that optimum (observed: acopf
        # case30_ieee medium oracle=402k vs wait=174k). Cells that are
        # structurally OPF-saturated (no commitment / foresight headroom the
        # oracle can exploit) set ``backend_config.oracle_opf_inert`` so the
        # oracle mirrors ``wait_only`` exactly — it is then classified as a
        # clean ``score_headroom`` diagnostic instead of an "oracle worse than
        # wait" failure. The flag lives only on the specific v0.6 cells that
        # need it; frozen releases (v0.4.1 / v0.5.0) never carry it, so oracle
        # behaviour there is byte-identical.
        if bool(gt.get("oracle_opf_inert", False)):
            return Action(
                tool_calls=[
                    ToolCall(name="wait", idempotency_key=self._next_idem_key("orc_w"))
                ],
                dominant="wait",
            )

        acopf_reserve_calls = self._acopf_reserve_lever_oracle_calls(avail_tools)
        acopf_ordered_calls = self._acopf_extreme_ordered_recovery_oracle_calls(
            gt_entities=gt_entities,
            avail_tools=avail_tools,
        )
        if acopf_ordered_calls:
            return Action(
                tool_calls=acopf_ordered_calls[:4],
                dominant=acopf_ordered_calls[0].name,
            )
        if acopf_reserve_calls:
            return Action(
                tool_calls=acopf_reserve_calls[:4],
                dominant=acopf_reserve_calls[0].name,
            )

        if str(self._scenario_config.get("backend_kind", "")) == (
            "pglib_uc_synthetic"
        ):
            operations = self._pglib_native_reference_operations(
                avail_tools=avail_tools
            )
            native_calls = [
                ToolCall(
                    name=str(operation["tool"]),
                    args=dict(operation["args"]),
                    idempotency_key=self._next_idem_key(
                        f"orc_pglib_uc_{operation['tool']}_{self._tick}"
                    ),
                )
                for operation in operations
            ]
            if native_calls:
                return Action(
                    tool_calls=native_calls,
                    dominant=native_calls[0].name,
                )
            return Action(
                tool_calls=[
                    ToolCall(
                        name="wait",
                        idempotency_key=self._next_idem_key("orc_pglib_uc_wait"),
                    )
                ],
                dominant="wait",
            )

        backend_config = self._scenario_config.get("backend_config") or {}
        battery_retry_calls = self._microgrid_battery_retry_calls(
            observation=observation,
            avail_tools=avail_tools,
        )
        if battery_retry_calls:
            return Action(
                tool_calls=battery_retry_calls,
                dominant=battery_retry_calls[0].name,
            )
        if str(backend_config.get("task_contract", {}).get("contract")) in {
            "microgrid.lv_voltage.staged_recovery.v2",
            "microgrid.lv_voltage.cross_tick_recovery.v2",
        }:
            recovery_calls = self._source_grounded_lv_oracle_calls(
                gt_entities=gt_entities,
                avail_tools=avail_tools,
                ground_truth=gt,
            )
            if recovery_calls:
                return Action(
                    tool_calls=recovery_calls,
                    dominant=recovery_calls[0].name,
                )
            return Action(
                tool_calls=[
                    ToolCall(
                        name="wait",
                        idempotency_key=self._next_idem_key("orc_mg_recovery_wait"),
                    )
                ],
                dominant="wait",
            )

        new_domain_calls = self._new_domain_oracle_calls(
            gt_entities=gt_entities,
            avail_tools=avail_tools,
            ground_truth=gt,
            observation=observation,
        )
        if new_domain_calls:
            backend_kind = str(self._scenario_config.get("backend_kind", ""))
            if backend_kind == "citylearn":
                call_cap = 12
            elif backend_kind in {"jsplib_job_shop", "co_bench_job_shop"}:
                difficulty = canonical_difficulty_level(
                    str(self._scenario_config.get("difficulty_level", "basic"))
                )
                call_cap = {
                    "basic": 6,
                    "medium": 8,
                    "high": 10,
                    "extreme": 12,
                }[difficulty]
            else:
                call_cap = 4
            return Action(
                tool_calls=new_domain_calls[:call_cap],
                dominant=new_domain_calls[0].name,
            )

        opendss_calls = self._opendss_ieee13_oracle_calls(
            gt_entities=gt_entities,
            avail_tools=avail_tools,
            observation=observation,
        )
        if opendss_calls:
            return Action(
                tool_calls=opendss_calls[:4],
                dominant=opendss_calls[0].name,
            )

        opendss_fresh_calls = self._opendss_fresh_feeders_oracle_calls(
            avail_tools=avail_tools,
            observation=observation,
        )
        if opendss_fresh_calls:
            return Action(
                tool_calls=opendss_fresh_calls[:4],
                dominant=opendss_fresh_calls[0].name,
            )

        # Resolve dilemmas: always pick the non-fatal option, with a rich
        # rationale citing the consequence delta from ground truth.
        for dilemma in observation.get("active_dilemmas", []) or []:
            did = dilemma["dilemma_id"]
            if did in self._chose_for:
                continue
            non_fatal = next(
                (o for o in dilemma["options"] if not o.get("fatal")),
                dilemma["options"][0] if dilemma["options"] else None,
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
                            "Oracle baseline: with full chronics knowledge, "
                            "this option minimizes long-horizon hospital risk "
                            "and keeps water/transit / ICU stakeholders fully "
                            "served while shifting load to reversible classes."
                        ),
                    },
                    idempotency_key=f"mc_{did}",
                )
            )
            self._chose_for.add(did)

        # Overload response with ground-truth rho info (Grid2Op).
        # Oracle uses two tiers:
        #   - rho > 1.05: top up reserves proactively (cheap, no shed)
        #   - rho > 1.5:  shed the lowest-criticality bus aggressively
        # The hand-off at 1.5 avoids paying shed tariffs for transient
        # overloads that clear on their own; the larger shed magnitude
        # (70% of bus demand) means synthetic relief actually offsets
        # the tariff.
        totals_gt = gt.get("totals", {}) or {}
        rho_max = float(totals_gt.get("rho_max", 0.0))
        has_acopf_reserve_lever = isinstance(
            (self._scenario_config.get("backend_config") or {}).get(
                "acopf_reserve_decision_lever"
            ),
            dict,
        )
        if (
            rho_max > 1.05
            and not has_acopf_reserve_lever
            and "commit_reserve" in avail_tools
        ):
            calls.append(
                ToolCall(
                    name="commit_reserve",
                    args={"mw": round(max(12.0, rho_max * 6), 1)},
                    idempotency_key=f"orc_cr_overload_{self._tick}",
                )
            )
        if rho_max > 1.5:
            load_candidates = [
                (eid, e)
                for eid, e in gt_entities.items()
                if e.get("kind") == "load"
                and float(e.get("current_demand_mw", 0)) > self._eff_load_shed_min
                and float(e.get("criticality", 1.0)) < 0.45
            ]
            if load_candidates:
                load_candidates.sort(
                    key=lambda kv: float(kv[1].get("criticality", 1.0))
                )
                lid, e = load_candidates[0]
                shed_mw = max(10.0, float(e.get("current_demand_mw", 10.0)) * 0.7)
                calls.append(
                    ToolCall(
                        name="shed_load",
                        args={
                            "load_id": lid,
                            "mw": round(shed_mw, 1),
                            "reason": (
                                f"oracle: ground-truth rho_max={rho_max:.2f} > 1.5, "
                                f"larger preemptive shed of lowest-criticality bus"
                            ),
                        },
                        idempotency_key=f"orc_shed_{lid}_{self._tick}",
                    )
                )

        # Compute capacity headroom from full state
        total_committed_cap = sum(
            float(e.get("power_max", 0.0))
            for _, e in gens
            if e.get("committed", False)
            and self._tick >= int(e.get("forced_outage_until", -1))
        )
        total_demand = float(gt.get("totals", {}).get("aggregate_demand_mw", 0.0))

        if total_demand > total_committed_cap and gens:
            # bring up the largest non-committed generator
            non_committed = [
                (eid, e) for eid, e in gens if not e.get("committed", False)
            ]
            if non_committed:
                non_committed.sort(
                    key=lambda kv: float(kv[1].get("power_max", 0.0)), reverse=True
                )
                gid, e = non_committed[0]
                calls.append(
                    ToolCall(
                        name="redispatch_generation",
                        args={
                            "generator_id": gid,
                            "target_mw": float(e.get("power_max", 50.0)) * 0.85,
                            "commit": True,
                        },
                        idempotency_key=f"orc_rd_{gid}_{self._tick}",
                    )
                )

        # Voltage-violation response (distribution backends).
        #
        # v0.6: when the feeder exposes distribution-native Volt-Var controls
        # (``backend_config['volt_var_controls']=true``), the oracle steers
        # voltage with *reactive* levers instead of shedding real load:
        #   - undervoltage  → switch capacitor banks ON + inject capacitive
        #     reactive power on DERs (pandapower sgen ``q_mvar > 0`` raises V).
        #   - overvoltage   → switch banks OFF + absorb reactive on DERs
        #     (``q_mvar < 0`` lowers V).
        # These cost nothing in shed penalty, so the oracle strictly dominates
        # both ``wait_only`` (which eats the voltage-violation cost) and the
        # legacy shed-only policy — restoring positive score_headroom on the
        # distribution_volt_var families. The shed-only branch below is kept
        # ONLY as a fallback for feeders without Volt-Var controls.
        bus_vms = [
            float(e["vm_pu"])
            for e in gt_entities.values()
            if e.get("kind") == "bus" and "vm_pu" in e
        ]
        volt_var = bool(bus_vms) and bool(
            {"switch_capacitor", "set_der_reactive_power"} & avail_tools
        )
        n_volt_viol = int(totals_gt.get("n_voltage_violations", 0))
        # v0.6: only steer voltage on a *sustained* violation — present at
        # BOTH the previous and the current tick. Rationale:
        #   - A reactive controller cannot pre-empt a single-tick spike (it
        #     observes the violation in the same tick it fires), so acting on
        #     a transient that self-clears next tick buys nothing.
        #   - Worse, the injected capacitor / DER reactive setpoints persist;
        #     when the underlying load shifts a few ticks later they overshoot
        #     and CREATE fresh violations (observed on mv_oberrhein basic: a
        #     lone 23-bus spike at t6 triggered a 37-bus overshoot at t9),
        #     making the oracle worse than wait_only.
        #   - Sustained violations (e.g. CIGRE basic: 9 buses for 6 ticks) are
        #     exactly where multi-tick reactive support nets positive, so the
        #     oracle still strictly dominates wait_only there.
        # This restores oracle ≥ wait_only headroom on transient-only feeders
        # while preserving dominance on real, persistent voltage stress.
        sustained_viol = n_volt_viol > 0 and self._prev_n_volt_viol > 0
        self._prev_n_volt_viol = n_volt_viol
        if volt_var and sustained_viol:
            min_vm = min(bus_vms)
            max_vm = max(bus_vms)
            caps = [
                (eid, e)
                for eid, e in gt_entities.items()
                if e.get("kind") == "capacitor"
            ]
            # Largest DERs first — strongest reactive levers within the
            # per-tick call budget (oracle trims to 4 calls/tick).
            ders = sorted(
                (
                    (eid, e)
                    for eid, e in gt_entities.items()
                    if e.get("kind") in ("renewable", "generator")
                    and str(eid).startswith("sgen_")
                ),
                key=lambda kv: float(kv[1].get("power_max", 0.0)),
                reverse=True,
            )
            if min_vm < 0.95 and "switch_capacitor" in avail_tools:
                depth = 0.95 - min_vm
                for cid_key, ce in caps[:2]:
                    if ce.get("on", False):
                        continue
                    cid = int(str(cid_key).split("_")[-1])
                    calls.append(
                        ToolCall(
                            name="switch_capacitor",
                            args={
                                "cap_id": cid,
                                "status": True,
                            },
                            idempotency_key=f"orc_cap_on_{cid}_{self._tick}",
                        )
                    )
                    self._engaged_caps.add(cid)
                if "set_der_reactive_power" in avail_tools:
                    for did_key, de in ders[:3]:
                        did = int(str(did_key).split("_")[-1])
                        pmax = float(de.get("power_max", 0.0)) or 0.5
                        q = min(0.4 * max(pmax, 0.5), depth * 8.0)
                        calls.append(
                            ToolCall(
                                name="set_der_reactive_power",
                                args={
                                    "der_id": did,
                                    "q_mvar": round(max(0.1, q), 3),
                                },
                                idempotency_key=f"orc_derq_{did}_{self._tick}",
                            )
                        )
                        self._engaged_ders.add(did)
            elif max_vm > 1.05:
                rise = max_vm - 1.05
                for cid_key, ce in caps[:2]:
                    if not ce.get("on", False):
                        continue
                    if "switch_capacitor" not in avail_tools:
                        break
                    cid = int(str(cid_key).split("_")[-1])
                    calls.append(
                        ToolCall(
                            name="switch_capacitor",
                            args={
                                "cap_id": cid,
                                "status": False,
                            },
                            idempotency_key=f"orc_cap_off_{cid}_{self._tick}",
                        )
                    )
                if "set_der_reactive_power" in avail_tools:
                    for did_key, de in ders[:3]:
                        did = int(str(did_key).split("_")[-1])
                        pmax = float(de.get("power_max", 0.0)) or 0.5
                        q = min(0.4 * max(pmax, 0.5), rise * 8.0)
                        calls.append(
                            ToolCall(
                                name="set_der_reactive_power",
                                args={
                                    "der_id": did,
                                    "q_mvar": -round(max(0.1, q), 3),
                                },
                                idempotency_key=f"orc_derq_{did}_{self._tick}",
                            )
                        )
        elif volt_var and (self._engaged_caps or self._engaged_ders):
            # v0.6: the violation has cleared but the oracle still holds latched
            # reactive support from an earlier tick. RELEASE it now (caps OFF,
            # DER q→0) so the support cannot overshoot on a later load swing and
            # manufacture a fresh, oracle-induced violation. Without this the
            # oracle is *worse* than wait_only on feeders with a second load
            # peak (mv_oberrhein medium: a t10 undervoltage support that stayed
            # latched produced an 112-bus overshoot at t17).
            for cid in sorted(self._engaged_caps):
                if "switch_capacitor" not in avail_tools:
                    break
                calls.append(
                    ToolCall(
                        name="switch_capacitor",
                        args={
                            "cap_id": cid,
                            "status": False,
                        },
                        idempotency_key=f"orc_cap_rel_{cid}_{self._tick}",
                    )
                )
            for did in sorted(self._engaged_ders):
                if "set_der_reactive_power" not in avail_tools:
                    break
                calls.append(
                    ToolCall(
                        name="set_der_reactive_power",
                        args={
                            "der_id": did,
                            "q_mvar": 0.0,
                        },
                        idempotency_key=f"orc_derq_rel_{did}_{self._tick}",
                    )
                )
            self._engaged_caps.clear()
            self._engaged_ders.clear()
        elif not volt_var:
            # Legacy fallback (radial feeders WITHOUT Volt-Var controls):
            # minimal-shed. Gated on ``not volt_var`` so that a feeder which
            # DOES expose reactive levers never falls through to load shedding
            # on a transient (non-sustained) spike — it simply waits, matching
            # wait_only on unpreventable single-tick events.
            # - Only act when *severe* (≥3 buses outside [0.95, 1.05] pu).
            # - Shed a SMALL share (20%) of the largest low-criticality load.
            # - Hysteresis: at most one voltage-driven shed every 3 ticks.
            last_volt_shed_tick = getattr(self, "_last_volt_shed_tick", -10)
            if n_volt_viol >= 3 and (self._tick - last_volt_shed_tick) >= 3:
                load_candidates = [
                    (eid, e)
                    for eid, e in gt_entities.items()
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
                    shed_mw = max(0.5, float(e.get("current_demand_mw", 1.0)) * 0.20)
                    calls.append(
                        ToolCall(
                            name="shed_load",
                            args={
                                "load_id": lid,
                                "mw": round(shed_mw, 2),
                                "reason": (
                                    f"oracle: GT n_voltage_violations={n_volt_viol} "
                                    f"(≥3 buses); minimal-shed strategy on radial feeder"
                                ),
                            },
                            idempotency_key=f"orc_shed_volt_{lid}_{self._tick}",
                        )
                    )
                    self._last_volt_shed_tick = self._tick

        # Proactive reserves
        shortfall = max(
            0.0,
            float(gt.get("totals", {}).get("reserves_required_mw", 0.0))
            - float(gt.get("totals", {}).get("reserves_procured_mw", 0.0)),
        )
        backend_config = self._scenario_config.get("backend_config") or {}
        if (
            shortfall > 0.0
            and "commit_reserve" in avail_tools
            and not isinstance(backend_config.get("acopf_reserve_decision_lever"), dict)
        ):
            calls.append(
                ToolCall(
                    name="commit_reserve",
                    args={"mw": round(shortfall * 1.2, 1)},
                    idempotency_key=f"orc_cr_{self._tick}",
                )
            )

        if not calls:
            calls.append(
                ToolCall(name="wait", idempotency_key=self._next_idem_key("orc_w"))
            )
        return Action(tool_calls=calls[:4], dominant=calls[0].name)

    def _acopf_reserve_lever_oracle_calls(
        self, avail_tools: set[str]
    ) -> list[ToolCall]:
        """Exploit opt-in AC-OPF reserve windows before they become shortfalls."""
        if str(self._scenario_config.get("domain", "power_grid")) != "power_grid":
            return []
        if str(self._scenario_config.get("backend_kind", "")) != "pandapower_acopf":
            return []
        if "commit_reserve" not in avail_tools:
            return []
        marker = "acopf_reserve_decision_lever"
        if marker in self._chose_for:
            return []
        backend_config = self._scenario_config.get("backend_config") or {}
        lever = backend_config.get(marker)
        if not isinstance(lever, dict) or not lever:
            return []
        window_start = int(lever.get("window_start_tick", 0))
        # The first agent decision has self._tick == 1 while the backend is
        # still about to simulate tick 0, so pre-commit at least one decision
        # before the reserve window begins.
        if self._tick > max(1, window_start):
            return []
        required_mw = float(lever.get("required_mw", 0.0) or 0.0)
        if required_mw <= 0:
            return []
        self._chose_for.add(marker)
        return [
            ToolCall(
                name="commit_reserve",
                args={"mw": round(required_mw, 3)},
                idempotency_key=f"orc_acopf_reserve_{self._tick}",
            )
        ]

    def _acopf_extreme_ordered_recovery_oracle_calls(
        self,
        *,
        gt_entities: dict[str, Any],
        avail_tools: set[str],
    ) -> list[ToolCall]:
        """Replay the explicit three-stage AC-OPF extreme contract.

        The extreme row intentionally requires two genuine policy switches:
        reserve procurement -> low-criticality protection -> redispatch.  The
        baseline must therefore follow the declared native milestones rather
        than relying on an incidental rho threshold to decide whether a shed
        call happens.  Each stage is still applied through the real backend
        tool and is checked by the ordinary task/counterfactual gates.
        """
        if (
            str(self._scenario_config.get("domain", "")) != "power_grid"
            or str(self._scenario_config.get("backend_kind", "")) != "pandapower_acopf"
            or str(self._scenario_config.get("difficulty_level", "")) != "extreme"
        ):
            return []
        backend_config = self._scenario_config.get("backend_config") or {}
        profile = backend_config.get("task_contract_profile") or {}
        if profile.get("contract") != "power_grid.acopf.extreme_ordered_recovery.v1":
            return []
        milestones = (backend_config.get("task_requirements") or {}).get(
            "ordered_tool_milestones"
        ) or []
        lever = backend_config.get("acopf_reserve_decision_lever") or {}
        current_tick = int(getattr(self._env, "tick", max(0, self._tick - 1)))
        for ordinal, milestone in enumerate(milestones):
            if not isinstance(milestone, dict):
                continue
            marker = f"acopf_extreme_milestone_{ordinal}"
            if marker in self._chose_for:
                continue
            not_before = int(milestone.get("not_before_tick", 0) or 0)
            not_after = int(milestone.get("not_after_tick", 10**9) or 10**9)
            if current_tick < not_before:
                return []
            if current_tick > not_after:
                continue
            tool = str(milestone.get("tool") or "")
            if tool not in avail_tools:
                return []
            if tool == "commit_reserve":
                required_mw = float(lever.get("required_mw", 0.0) or 0.0)
                if required_mw <= 0.0:
                    return []
                args = {"mw": round(required_mw, 3)}
            elif tool == "shed_load":
                candidates = [
                    (entity_id, entity)
                    for entity_id, entity in gt_entities.items()
                    if isinstance(entity, dict)
                    and entity.get("kind") == "load"
                    and float(entity.get("current_demand_mw", 0.0) or 0.0) > 0.0
                    and str(entity.get("stakeholder_class") or "")
                    not in {"hospital", "water"}
                ]
                if not candidates:
                    return []
                candidates.sort(
                    key=lambda item: (
                        float(item[1].get("criticality", 1.0) or 1.0),
                        str(item[0]),
                    )
                )
                load_id, entity = candidates[0]
                demand_mw = float(entity.get("current_demand_mw", 0.0) or 0.0)
                args = {
                    "load_id": str(load_id),
                    "mw": round(max(0.1, demand_mw * 0.05), 3),
                    "reason": (
                        "Extreme contract protection stage: shed a bounded share "
                        "of the lowest-criticality non-protected load."
                    ),
                }
            elif tool == "redispatch_generation":
                generators = [
                    (entity_id, entity)
                    for entity_id, entity in gt_entities.items()
                    if isinstance(entity, dict)
                    and entity.get("kind") == "generator"
                    and self._acopf_generator_dispatchable(
                        entity, current_tick=current_tick
                    )
                ]
                if not generators:
                    return []
                generators.sort(
                    key=lambda item: float(item[1].get("power_max", 0.0) or 0.0),
                    reverse=True,
                )
                generator_id, entity = generators[0]
                power_max = float(entity.get("power_max", 0.0) or 0.0)
                args = {
                    "generator_id": str(generator_id),
                    "target_mw": round(max(0.1, power_max * 0.85), 3),
                    "commit": True,
                }
            else:
                return []
            self._chose_for.add(marker)
            return [
                ToolCall(
                    name=tool,
                    args=args,
                    idempotency_key=f"orc_acopf_extreme_{ordinal}_{current_tick}",
                )
            ]
        return []

    @staticmethod
    def _acopf_generator_dispatchable(
        entity: dict[str, Any], *, current_tick: int
    ) -> bool:
        """Reject generator pins while the native outage is still active."""
        if "forced_outage_until" in entity:
            try:
                outage_until = float(entity["forced_outage_until"])
            except (TypeError, ValueError):
                return False
            return math.isfinite(outage_until) and outage_until <= current_tick
        # AC-OPF snapshots normally expose forced_outage_until.  If a custom
        # snapshot omits it, require the native committed state rather than a
        # non-native ``available`` hint that cannot prove control leverage.
        return entity.get("committed") is True

    def _traffic_oracle_calls(
        self,
        observation: dict[str, Any],
        avail_tools: set[str],
    ) -> list[ToolCall]:
        """Offline-best myopic corridor-control policy (the headroom upper bound).

        Mirrors ``scripts/traffic_behavioral_gate._policy_oracle``: grant the
        max-throughput ``incident_relief`` program to every corridor (worst
        first, by the observable ``criticality x demand_veh`` product, so the
        highest-impact corridors survive the env's per-tick tool budget), plus
        an explicit EMS/VIP priority grant on protected routes (exploiting the
        priority capacity bonus that the single-corridor greedy and the
        state-blind fixed-time plans both miss). ``criticality`` / ``demand_veh``
        / ``carries_*`` are static corridor attributes that survive the fog
        layer, so this is the same policy the deterministic-backend behavioral
        gate proved clears the oracle-vs-wait headroom threshold.
        """
        corridors = {
            cid: attrs
            for cid, attrs in (observation.get("entities") or {}).items()
            if isinstance(attrs, dict) and attrs.get("kind") == "corridor"
        }
        backend_config = self._scenario_config.get("backend_config") or {}
        if (
            backend_config.get("live_phase_control") is True
            and "set_signal_phase_duration" in avail_tools
        ):
            runtime_tls = (observation.get("runtime_signal_control") or {}).get(
                "tls"
            ) or {}
            pressure: dict[str, int] = {}
            for row in (observation.get("vehicle_control_capture") or {}).get(
                "records"
            ) or []:
                tls_id = str((row.get("tls_context") or {}).get("tls_id") or "")
                if tls_id:
                    pressure[tls_id] = pressure.get(tls_id, 0) + 1
            candidates = [
                (tls_id, runtime)
                for tls_id, runtime in runtime_tls.items()
                if any(
                    signal in {"g", "G"}
                    for signal in str(runtime.get("current_state") or "")
                )
            ]
            if not candidates:
                return []
            tls_id, runtime = max(
                candidates,
                key=lambda item: (pressure.get(item[0], 0), item[0]),
            )
            bounds = runtime.get("current_phase_bounds") or {}
            minimum = float(bounds.get("min_duration") or 0.0)
            maximum = float(bounds.get("max_duration") or 0.0)
            requested = float(
                backend_config.get("reference_phase_duration_seconds") or 0.0
            )
            schedule = backend_config.get("reference_phase_duration_schedule")
            scheduled_value: Any | None = None
            if isinstance(schedule, dict):
                scheduled_value = schedule.get(str(self._tick))
            elif isinstance(schedule, list) and schedule:
                index = min(len(schedule) - 1, max(0, self._tick - 1))
                scheduled_value = schedule[index]
            if scheduled_value is not None:
                try:
                    requested = float(scheduled_value)
                except (TypeError, ValueError):
                    requested = 0.0
            if minimum <= 0.0 or maximum < minimum or requested <= 0.0:
                return []
            duration = max(minimum, min(maximum, requested))
            calls = [
                ToolCall(
                    name="set_signal_phase_duration",
                    args={
                        "tls_id": tls_id,
                        "observed_program": str(runtime.get("current_program") or ""),
                        "observed_phase": int(runtime.get("current_phase") or 0),
                        "remaining_duration_seconds": duration,
                    },
                    idempotency_key=(f"orc_tr_phase_{tls_id}_{self._tick}"),
                    consumes_evidence_ids=(
                        _visible_source_evidence_ids(observation) or None
                    ),
                )
            ]
            plan_marker = "traffic_live_sumo_phase_supervision_plan"
            if (
                str(self._scenario_config.get("domain", "")) == "traffic"
                and str(self._scenario_config.get("backend_kind", "")) == "sumo"
                and "commit_to_plan" in avail_tools
                and plan_marker not in self._chose_for
            ):
                calls.append(
                    ToolCall(
                        name="commit_to_plan",
                        args={
                            "plan_id": "oracle-live-sumo-phase-supervision-v1",
                            "horizon_ticks": max(
                                2,
                                int(self._scenario_config.get("horizon_ticks") or 2),
                            ),
                            "review_after_ticks": 2,
                            "rationale": (
                                "Apply the source-bound phase-duration policy "
                                "while native SUMO advances, then review signal "
                                "pressure at the next supervisory decision."
                            ),
                        },
                        idempotency_key=self._next_idem_key("orc_tr_plan"),
                    )
                )
                self._chose_for.add(plan_marker)
            return calls
        if not corridors:
            return []
        if "change_signal_plan" not in avail_tools:
            return []
        if (
            backend_config.get("live_phase_control") is True
            and "extend_current_green_phase" in avail_tools
        ):
            ordered = sorted(
                corridors,
                key=lambda cid: (
                    -float(corridors[cid].get("queue") or 0.0),
                    -float(corridors[cid].get("demand_veh") or 0.0),
                    cid,
                ),
            )
            return [
                ToolCall(
                    name="extend_current_green_phase",
                    args={"corridor": cid, "duration_s": 60.0},
                    idempotency_key=f"orc_tr_extend_{cid}_{self._tick}",
                )
                for cid in ordered
            ]

        def impact(cid: str) -> tuple[float, str]:
            attrs = corridors[cid]
            criticality = float(attrs.get("criticality") or 0.0)
            demand = float(attrs.get("demand_veh") or 0.0)
            return (-(criticality * demand), cid)

        calls: list[ToolCall] = []
        for cid in sorted(corridors, key=impact):
            attrs = corridors[cid]
            calls.append(
                ToolCall(
                    name="change_signal_plan",
                    args={"corridor": cid, "program": "incident_relief"},
                    idempotency_key=f"orc_tr_relief_{cid}_{self._tick}",
                )
            )
            if "dispatch_emergency_priority" not in avail_tools:
                continue
            if attrs.get("carries_ems_corridor"):
                calls.append(
                    ToolCall(
                        name="dispatch_emergency_priority",
                        args={"corridor": cid, "mode": "ems"},
                        idempotency_key=f"orc_tr_ems_{cid}_{self._tick}",
                    )
                )
            elif attrs.get("carries_vip_route"):
                calls.append(
                    ToolCall(
                        name="dispatch_emergency_priority",
                        args={"corridor": cid, "mode": "vip"},
                        idempotency_key=f"orc_tr_vip_{cid}_{self._tick}",
                    )
                )
        return calls

    def _new_domain_oracle_calls(
        self,
        *,
        gt_entities: dict[str, Any],
        avail_tools: set[str],
        ground_truth: dict[str, Any],
        observation: dict[str, Any],
    ) -> list[ToolCall]:
        """Minimal domain-native branches for v0.7 release-packaged domains."""
        domain = str(self._scenario_config.get("domain", "power_grid"))
        backend_kind = str(self._scenario_config.get("backend_kind", ""))

        if domain == "building_energy" and backend_kind == "citylearn":
            backend_config = self._scenario_config.get("backend_config") or {}
            if backend_config.get("oracle_policy_contract") not in {
                "citylearn.locked_future_tariff_storage_arbitrage.v1",
                "citylearn.locked_native_peak_response.v1",
            }:
                return []
            native_backend = getattr(self._env, "_backend", None)
            native_env = getattr(native_backend, "_env", None)
            native_buildings = list(getattr(native_env, "buildings", []) or [])
            if not native_buildings:
                return []
            tick = max(0, self._tick - 1)
            task_contract = backend_config.get("task_contract") or {}
            response_windows = list(task_contract.get("response_windows") or [])
            if response_windows:
                calls: list[ToolCall] = []
                building_ids = sorted(ground_truth.get("buildings") or {})
                if tick == 0 and "commit_to_plan" in avail_tools:
                    calls.append(
                        ToolCall(
                            name="commit_to_plan",
                            args={
                                "plan_id": "citylearn-oracle-plan-v1",
                                "review_after_ticks": 4,
                            },
                            idempotency_key="orc_citylearn_plan_v1",
                        )
                    )
                source_start = int(
                    backend_config.get("simulation_start_time_step") or 0
                )
                hidden_ticks = sorted(
                    int(event.get("trigger_tick") or -1) - source_start
                    for event in backend_config.get("native_source_events") or []
                    if event.get("hidden") is True
                )
                if tick in hidden_ticks:
                    if "inspect_building_state" in avail_tools and building_ids:
                        calls.append(
                            ToolCall(
                                name="inspect_building_state",
                                args={"building_id": str(building_ids[0])},
                                idempotency_key=f"orc_citylearn_inspect_{tick}",
                            )
                        )
                    if "commit_to_plan" in avail_tools:
                        version = hidden_ticks.index(tick) + 2
                        calls.append(
                            ToolCall(
                                name="commit_to_plan",
                                args={
                                    "plan_id": f"citylearn-oracle-plan-v{version}",
                                    "replaces_plan_id": (
                                        f"citylearn-oracle-plan-v{version - 1}"
                                    ),
                                    "revision_reason": (
                                        "hidden_source_transition_investigated"
                                    ),
                                    "review_after_ticks": 4,
                                },
                                idempotency_key=(f"orc_citylearn_plan_v{version}"),
                            )
                        )
                effect_tick = tick + int(
                    backend_config.get("storage_control_delay_ticks") or 0
                )
                for window_index, window in enumerate(response_windows):
                    first_tick = int(window.get("first_tick") or 0)
                    last_tick = int(window.get("last_tick") or -1)
                    if not first_tick <= effect_tick <= last_tick:
                        continue
                    sign = (
                        1.0
                        if window.get("expected_control_policy") == "charge"
                        else -1.0
                    )
                    rate = sign * (
                        0.18 + 0.02 * ((effect_tick - first_tick + window_index) % 3)
                    )
                    if "set_storage_dispatch" in avail_tools and building_ids:
                        calls.append(
                            ToolCall(
                                name="set_storage_dispatch",
                                args={
                                    "dispatches": [
                                        {
                                            "building_id": str(building_id),
                                            "rate": rate,
                                        }
                                        for building_id in building_ids
                                    ]
                                },
                                idempotency_key=f"orc_citylearn_storage_{tick}_batch",
                            )
                        )
                    break
                return calls
            horizon = int(self._scenario_config.get("horizon_ticks") or 0)
            prices = [
                float(value)
                for value in native_buildings[0].pricing.electricity_pricing[:horizon]
            ]
            if len(prices) != horizon or not prices:
                return []
            peak_price = max(prices)
            peak_ticks = [
                index
                for index, price in enumerate(prices)
                if abs(price - peak_price) <= 1e-12
            ]
            peak_start = min(peak_ticks)
            peak_end = peak_start
            while (
                peak_end < len(prices) and abs(prices[peak_end] - peak_price) <= 1e-12
            ):
                peak_end += 1
            if tick == 0 and "commit_to_plan" in avail_tools:
                return [
                    ToolCall(
                        name="commit_to_plan",
                        args={
                            "plan_id": "citylearn-oracle-plan-v1",
                            "review_after_ticks": 2,
                        },
                        idempotency_key="orc_citylearn_plan_v1",
                    )
                ]
            if tick == peak_start and "commit_to_plan" in avail_tools:
                return [
                    ToolCall(
                        name="commit_to_plan",
                        args={
                            "plan_id": "citylearn-oracle-plan-v2",
                            "replaces_plan_id": "citylearn-oracle-plan-v1",
                            "revision_reason": "locked_future_peak_window_started",
                            "review_after_ticks": 2,
                        },
                        idempotency_key="orc_citylearn_plan_v2",
                    )
                ]
            active_rate = (
                0.25
                if max(0, peak_start - 6) <= tick < peak_start
                else -0.25
                if peak_start < tick < peak_end
                else None
            )
            if active_rate is None or "set_storage_dispatch" not in avail_tools:
                return []
            buildings = ground_truth.get("buildings") or {}
            return [
                ToolCall(
                    name="set_storage_dispatch",
                    args={
                        "dispatches": [
                            {
                                "building_id": str(building_id),
                                "rate": active_rate,
                            }
                            for building_id in sorted(buildings)
                        ]
                    },
                    idempotency_key=f"orc_citylearn_storage_{tick}_batch",
                )
            ]

        if (
            domain == "microgrid"
            and backend_kind == "pymgrid_economic_dispatch"
            and str(
                (self._scenario_config.get("backend_config") or {})
                .get("native_state_loss_task", {})
                .get("contract", "")
            )
            == "microgrid.native_state_loss.v1"
        ):
            return self._native_state_loss_oracle_calls(
                gt_entities=gt_entities,
                avail_tools=avail_tools,
            )

        if (
            domain == "logistics"
            and backend_kind == "dynasched_flexible_job_shop"
            and "dispatch_flexible_operations" in avail_tools
        ):
            native_backend = getattr(self._env, "_backend", None)
            reference_dispatch = getattr(
                native_backend, "native_oracle_reference_dispatch", None
            )
            if not callable(reference_dispatch):
                return []
            operations = reference_dispatch()
            if not operations:
                return []
            return [
                ToolCall(
                    name="dispatch_flexible_operations",
                    args={"operations": operations},
                    idempotency_key=self._next_idem_key(
                        f"orc_dynasched_native_tick_{self._tick}"
                    ),
                )
            ]

        if domain == "logistics" and backend_kind in {
            "jsplib_job_shop",
            "co_bench_job_shop",
        }:
            return self._jsplib_job_shop_oracle_calls(
                ground_truth=ground_truth,
                avail_tools=avail_tools,
                observation=observation,
            )

        if domain == "logistics" and backend_kind == "orgym_invmgmt":
            calls = self._orgym_inventory_oracle_calls(
                ground_truth=ground_truth,
                avail_tools=avail_tools,
                observation=observation,
            )
            if calls:
                return calls
            if "wait" in avail_tools:
                return [
                    ToolCall(
                        name="wait",
                        idempotency_key=f"orc_orgym_wait_{self._tick}",
                    )
                ]
            return []

        if domain == "logistics" and backend_kind in {
            "pyvrp_cvrp",
            "pyvrp_vrptw",
            "pyvrp_lastmile",
        }:
            return self._routing_oracle_calls(
                gt_entities=gt_entities,
                avail_tools=avail_tools,
            )

        if domain == "datacenter" and backend_kind == "alibaba_openb_gpu_placement":
            if self._tick == 1 and "set_placement_policy" in avail_tools:
                return [
                    ToolCall(
                        name="set_placement_policy",
                        args={"policy": "fragmentation_aware"},
                        idempotency_key="orc_dc_openb_fragmentation_policy_v1",
                    )
                ]
            if "place_pod" not in avail_tools:
                return []
            queued = (ground_truth.get("placement") or {}).get("queued_pods") or []
            feasible = [
                pod
                for pod in queued
                if isinstance(pod, dict) and pod.get("feasible_node_ids")
            ]
            if not feasible:
                return []
            pod = min(
                feasible,
                key=lambda row: (
                    int(row.get("due_tick") or 0),
                    -int(row.get("wait_ticks") or 0),
                    str(row.get("pod_id") or ""),
                ),
            )
            return [
                ToolCall(
                    name="place_pod",
                    args={
                        "pod_id": str(pod["pod_id"]),
                        "node_id": str(pod["feasible_node_ids"][0]),
                    },
                    idempotency_key=self._next_idem_key(
                        f"orc_dc_openb_place_{pod['pod_id']}_{self._tick}"
                    ),
                )
            ]

        if domain == "datacenter" and backend_kind == "alibaba_trace_sim":
            calls: list[ToolCall] = []
            backend_config = self._scenario_config.get("backend_config") or {}
            source_schema = str(
                (backend_config.get("source_transform") or {}).get("source_schema")
                or ""
            )
            runtime_source_only = source_schema == "alibaba-spot-gpu-v2026-v1"
            policy_tick = 1
            if self._tick == policy_tick and "set_queue_policy" in avail_tools:
                calls.append(
                    ToolCall(
                        name="set_queue_policy",
                        args={
                            "policy": (
                                "deadline_criticality_first"
                                if runtime_source_only
                                else "shortest_job_first"
                            )
                        },
                        idempotency_key="orc_dc_source_policy_1",
                    )
                )
            policy_review_calls = self._datacenter_policy_review_calls(
                observation=observation,
                avail_tools=avail_tools,
                policy_call_will_change=(self._tick == policy_tick),
            )
            calls.extend(policy_review_calls)
            if "reserve_gpu_capacity" in avail_tools:
                for perturbation in self._scenario_config.get("perturbations") or []:
                    if (
                        not bool(perturbation.get("hidden"))
                        and int(perturbation.get("trigger_tick") or -1)
                        == self._tick + 1
                    ):
                        projected_factor = max(
                            0.1,
                            1.0 - float(perturbation.get("intensity") or 0.0),
                        )
                        base_capacity = float(
                            backend_config.get("gpu_capacity_units") or 0.0
                        )
                        if runtime_source_only:
                            units = self._datacenter_observed_reservation_shortfall(
                                ground_truth,
                                available_gpu_override=(
                                    base_capacity * projected_factor
                                    if base_capacity > 0.0
                                    else None
                                ),
                            )
                            units = min(
                                units,
                                self._datacenter_source_reservation_headroom(
                                    ground_truth,
                                    backend_config=backend_config,
                                    available_gpu_override=(
                                        base_capacity * projected_factor
                                        if base_capacity > 0.0
                                        else None
                                    ),
                                ),
                            )
                        else:
                            units = self._datacenter_reservation_shortfall(
                                backend_config,
                                trigger_tick=int(perturbation["trigger_tick"]),
                                duration_ticks=max(
                                    1,
                                    int(perturbation.get("duration_ticks") or 1),
                                ),
                                capacity_factor=projected_factor,
                            )
                        if units <= 0:
                            continue
                        duration = max(
                            1,
                            int(perturbation.get("duration_ticks") or 1),
                        )
                        calls.append(
                            ToolCall(
                                name="reserve_gpu_capacity",
                                args={
                                    "gpu_units": round(max(1.0, units), 3),
                                    "duration_ticks": min(4, duration),
                                },
                                idempotency_key=(
                                    f"orc_dc_visible_reserve_{self._tick}"
                                ),
                            )
                        )
                        return calls
                capacity = ground_truth.get("capacity") or {}
                capacity_factor = float(capacity.get("capacity_factor", 1.0) or 1.0)
                reserved = float(
                    capacity.get("reserved_gpu_units", 0.0) or 0.0
                ) + float(capacity.get("pending_reserved_gpu_units", 0.0) or 0.0)
                if capacity_factor < 1.0 - 1e-9 and reserved <= 1e-9:
                    if runtime_source_only:
                        units = self._datacenter_observed_reservation_shortfall(
                            ground_truth,
                        )
                        units = min(
                            units,
                            self._datacenter_source_reservation_headroom(
                                ground_truth,
                                backend_config=backend_config,
                            ),
                        )
                    else:
                        units = self._datacenter_reservation_shortfall(
                            backend_config,
                            trigger_tick=self._tick,
                            duration_ticks=2,
                            capacity_factor=capacity_factor,
                        )
                    if units <= 1e-9:
                        return calls
                    calls.append(
                        ToolCall(
                            name="reserve_gpu_capacity",
                            args={
                                "gpu_units": round(max(1.0, units), 3),
                                "duration_ticks": 2,
                            },
                            idempotency_key=(f"orc_dc_reactive_reserve_{self._tick}"),
                        )
                    )
            return calls

        backend_config = self._scenario_config.get("backend_config") or {}
        if (
            domain == "microgrid"
            and backend_kind == "pandapower_lv"
            and backend_config.get("source_profile_applied") is True
        ):
            return self._source_grounded_lv_oracle_calls(
                gt_entities=gt_entities,
                avail_tools=avail_tools,
                ground_truth=ground_truth,
            )

        if self._tick != 1:
            return []

        if domain == "microgrid":
            if backend_kind == "pandapower_lv" and "curtail_der" in avail_tools:
                ders = [
                    did
                    for did, ent in gt_entities.items()
                    if ent.get("kind") in {"pv", "renewable"}
                ]
                return [
                    ToolCall(
                        name="curtail_der",
                        args={"der_id": did, "target_mw": 0.0},
                        idempotency_key=f"orc_mg_curtail_{did}_{self._tick}",
                    )
                    for did in sorted(ders)[:4]
                ]

            if (
                backend_kind == "pymgrid_economic_dispatch"
                and "set_battery_dispatch" in avail_tools
            ):
                battery = gt_entities.get("batt0") or next(
                    (
                        ent
                        for ent in gt_entities.values()
                        if ent.get("kind") == "battery"
                    ),
                    None,
                )
                if isinstance(battery, dict):
                    max_discharge = float(battery.get("max_discharge_mw", 0.0) or 0.0)
                    soc = float(battery.get("soc_mwh", 0.0) or 0.0)
                    discharge = min(max_discharge, soc)
                    if discharge > 0:
                        return [
                            ToolCall(
                                name="set_battery_dispatch",
                                args={
                                    "battery_id": "batt0",
                                    "p_mw": -round(discharge, 3),
                                },
                                idempotency_key=f"orc_mg_batt_{self._tick}",
                            )
                        ]

        return []

    def _native_state_loss_oracle_calls(
        self,
        *,
        gt_entities: dict[str, Any],
        avail_tools: set[str],
    ) -> list[ToolCall]:
        """Schedule requests so their physical effects land inside each milestone window.

        This is intentionally scoped to the staging EMS task contract.  The
        oracle has perfect source knowledge, but still uses the runtime's
        native dispatch, battery, and PCC tools so task completion is proved
        by physical state and replay evidence rather than acknowledgements.
        """
        backend_config = self._scenario_config.get("backend_config") or {}
        requirements = backend_config.get("task_requirements") or {}
        milestones = requirements.get("ordered_tool_milestones") or []
        current_tick = int(getattr(self._env, "tick", max(0, self._tick - 1)))
        for milestone in milestones:
            if not isinstance(milestone, dict):
                continue
            tool = str(milestone.get("tool") or "")
            if not tool or tool in self._native_state_loss_milestones_done:
                continue
            if tool not in avail_tools:
                return []
            effect_not_before = int(milestone.get("not_before_tick", current_tick))
            effect_not_after = int(milestone.get("not_after_tick", effect_not_before))
            registry = getattr(self._env, "_tools", None)
            resolve_imperfection = getattr(registry, "resolve_imperfection", None)
            delay_ticks = 0
            if callable(resolve_imperfection):
                try:
                    imperfection = resolve_imperfection(tool) or {}
                    delay_ticks = max(0, int(imperfection.get("delay_ticks", 0)))
                except (TypeError, ValueError):
                    delay_ticks = 0
            # Preserve the declared lower-bound request tick when its delayed
            # effect is still legal; only pull the request earlier when needed
            # to meet a narrow physical-effect window.
            not_before = max(0, effect_not_before)
            not_after = effect_not_after - delay_ticks
            if not_before > not_after:
                not_before = max(0, effect_not_before - delay_ticks)
            if not_after < 0 or not_before > not_after:
                return []
            if current_tick < not_before:
                return []
            if current_tick > not_after:
                continue
            if tool == "dispatch_genset":
                genset = next(
                    (
                        entity
                        for entity in gt_entities.values()
                        if entity.get("kind") == "genset"
                        and entity.get("available", True)
                    ),
                    {},
                )
                p_mw = max(
                    0.1,
                    float(genset.get("max_mw", 0.0) or 0.0),
                )
                args = {"genset_id": "genset0", "p_mw": round(p_mw, 3)}
            elif tool == "set_battery_dispatch":
                battery = next(
                    (
                        entity
                        for entity in gt_entities.values()
                        if entity.get("kind") == "battery"
                    ),
                    {},
                )
                discharge = min(
                    float(battery.get("max_discharge_mw", 0.0) or 0.0),
                    float(battery.get("soc_mwh", 0.0) or 0.0),
                )
                args = {
                    "battery_id": "batt0",
                    "p_mw": -round(max(0.1, discharge), 3),
                }
            elif tool == "connect_pcc":
                args = {"connect": True}
            else:
                return []
            self._native_state_loss_milestones_done.add(tool)
            if tool == "set_battery_dispatch":
                return [
                    self._microgrid_battery_dispatch_call(
                        p_mw=float(args["p_mw"]),
                        key_prefix="orc_native_state_loss_battery",
                    )
                ]
            return [
                ToolCall(
                    name=tool,
                    args=args,
                    idempotency_key=f"orc_native_state_loss_{tool}_{current_tick}",
                )
            ]
        return []

    def _microgrid_battery_dispatch_call(
        self,
        *,
        p_mw: float,
        key_prefix: str,
        retry_count: int = 0,
    ) -> ToolCall:
        idempotency_key = self._next_idem_key(key_prefix)
        self._microgrid_battery_retry_state = {
            "idempotency_key": idempotency_key,
            "p_mw": float(p_mw),
            "retry_count": int(retry_count),
        }
        return ToolCall(
            name="set_battery_dispatch",
            args={"battery_id": "batt0", "p_mw": float(p_mw)},
            idempotency_key=idempotency_key,
        )

    def _microgrid_battery_retry_calls(
        self,
        *,
        observation: dict[str, Any],
        avail_tools: set[str],
    ) -> list[ToolCall]:
        state = self._microgrid_battery_retry_state
        backend_config = self._scenario_config.get("backend_config") or {}
        native_contract = str(
            (backend_config.get("native_state_loss_task") or {}).get("contract") or ""
        )
        task_contract = str(
            (backend_config.get("task_contract") or {}).get("contract") or ""
        )
        if (
            str(self._scenario_config.get("domain", "")) != "microgrid"
            or "set_battery_dispatch" not in avail_tools
            or not isinstance(state, dict)
            or (
                native_contract != "microgrid.native_state_loss.v1"
                and task_contract
                not in {
                    "microgrid.lv_voltage.staged_recovery.v2",
                    "microgrid.lv_voltage.cross_tick_recovery.v2",
                }
            )
        ):
            return []
        matching_result = next(
            (
                result
                for result in reversed(observation.get("__last_tool_results__") or [])
                if isinstance(result, dict)
                and result.get("name") == "set_battery_dispatch"
                and result.get("idempotency_key") in {None, state["idempotency_key"]}
            ),
            None,
        )
        if matching_result is None:
            return []
        retry_count = int(state["retry_count"])
        if (
            matching_result.get("ok") is not False
            or matching_result.get("error_code") != "INJECTED_FAILURE"
            or retry_count >= 2
        ):
            self._microgrid_battery_retry_state = None
            return []
        return [
            self._microgrid_battery_dispatch_call(
                p_mw=float(state["p_mw"]) / 2.0,
                key_prefix="orc_mg_battery_retry",
                retry_count=retry_count + 1,
            )
        ]

    def _source_grounded_lv_oracle_calls(
        self,
        *,
        gt_entities: dict[str, Any],
        avail_tools: set[str],
        ground_truth: dict[str, Any],
    ) -> list[ToolCall]:
        """Reference control for locked-profile LV candidate calibration.

        The first decision pre-positions storage for the known upcoming PV
        ramp. Later decisions use current AC-power-flow violations to combine
        Volt-Var support with incremental curtailment. This branch is opt-in
        so historical LV scenarios retain their original one-shot oracle.
        """
        task_contract = dict(
            (self._scenario_config.get("backend_config") or {}).get("task_contract")
            or {}
        )
        if task_contract.get("contract") == "microgrid.lv_voltage.staged_recovery.v2":
            target_q = {
                "der0": 0.01,
                "der1": 0.03,
                "der3": 0.03,
                "der4": 0.05,
            }
            if self._tick in {1, 4}:
                calls: list[ToolCall] = []
                battery = gt_entities.get("batt0") or {}
                if "set_battery_dispatch" in avail_tools:
                    rate_key = (
                        "max_charge_mw" if self._tick == 1 else "max_discharge_mw"
                    )
                    rate = float(battery.get(rate_key) or 0.0)
                    if rate > 0:
                        calls.append(
                            self._microgrid_battery_dispatch_call(
                                p_mw=(
                                    round(rate, 6)
                                    if self._tick == 1
                                    else -round(rate, 6)
                                ),
                                key_prefix=(
                                    "orc_mg_staged_charge"
                                    if self._tick == 1
                                    else "orc_mg_staged_discharge"
                                ),
                            )
                        )
                if "set_der_reactive_power" in avail_tools:
                    sign = -1.0 if self._tick == 1 else 1.0
                    phase = "absorb" if self._tick == 1 else "support"
                    for der_id, requested_q in target_q.items():
                        entity = gt_entities.get(der_id) or {}
                        q_limit = float(entity.get("max_abs_q_mvar") or 0.0)
                        if q_limit <= 0:
                            continue
                        calls.append(
                            ToolCall(
                                name="set_der_reactive_power",
                                args={
                                    "der_id": der_id,
                                    "q_mvar": round(
                                        sign * min(requested_q, q_limit),
                                        6,
                                    ),
                                },
                                idempotency_key=(f"orc_mg_staged_{phase}_{der_id}"),
                            )
                        )
                return calls
            return []

        if (
            task_contract.get("contract")
            == "microgrid.lv_voltage.cross_tick_recovery.v2"
        ):
            if self._tick == 1 and "set_battery_dispatch" in avail_tools:
                battery = gt_entities.get("batt0") or {}
                charge = float(battery.get("max_charge_mw") or 0.0)
                if charge > 0:
                    return [
                        self._microgrid_battery_dispatch_call(
                            p_mw=round(charge, 6),
                            key_prefix="orc_mg_recovery_precharge",
                        )
                    ]
            if self._tick == 6 and "set_der_reactive_power" in avail_tools:
                target_q = {
                    "der0": 0.01,
                    "der1": 0.03,
                    "der3": 0.03,
                    "der4": 0.05,
                }
                calls = []
                for der_id, requested_q in target_q.items():
                    entity = gt_entities.get(der_id) or {}
                    q_limit = float(entity.get("max_abs_q_mvar") or 0.0)
                    if q_limit <= 0:
                        continue
                    calls.append(
                        ToolCall(
                            name="set_der_reactive_power",
                            args={
                                "der_id": der_id,
                                "q_mvar": round(min(requested_q, q_limit), 6),
                            },
                            idempotency_key=f"orc_mg_recovery_q_{der_id}",
                        )
                    )
                return calls
            if self._tick == 8 and "set_battery_dispatch" in avail_tools:
                battery = gt_entities.get("batt0") or {}
                discharge = float(battery.get("max_discharge_mw") or 0.0)
                if discharge > 0:
                    return [
                        self._microgrid_battery_dispatch_call(
                            p_mw=-round(discharge, 6),
                            key_prefix="orc_mg_recovery_discharge",
                        )
                    ]
            return []

        if not self._lv_source_prepositioned and "set_battery_dispatch" in avail_tools:
            battery = gt_entities.get("batt0") or {}
            capacity = float(battery.get("max_e_mwh") or 0.0)
            energy = float(battery.get("soc_mwh") or 0.0)
            efficiency = float(battery.get("efficiency") or 1.0)
            max_charge = float(battery.get("max_charge_mw") or 0.0)
            tick_hours = (
                float(self._scenario_config.get("tick_minutes", 60) or 60) / 60.0
            )
            feasible = max(0.0, capacity - energy) / max(1e-9, efficiency * tick_hours)
            charge = min(max_charge, feasible)
            self._lv_source_prepositioned = True
            if charge > 1e-9:
                return [
                    self._microgrid_battery_dispatch_call(
                        p_mw=round(charge, 6),
                        key_prefix="orc_mg_source_precharge",
                    )
                ]

        totals = ground_truth.get("totals") or {}
        n_violations = int(totals.get("n_voltage_violations") or 0)
        if n_violations <= 0:
            return []

        ders = sorted(
            (
                (did, ent)
                for did, ent in gt_entities.items()
                if ent.get("kind") in {"pv", "renewable"}
                and float(ent.get("output_mw") or 0.0) > 1e-9
            ),
            key=lambda item: (-float(item[1].get("output_mw") or 0.0), item[0]),
        )
        calls: list[ToolCall] = []
        if "set_battery_dispatch" in avail_tools:
            calls.append(
                ToolCall(
                    name="set_battery_dispatch",
                    args={"battery_id": "batt0", "p_mw": 0.0},
                    idempotency_key=f"orc_mg_source_hold_{self._tick}",
                )
            )
        if "set_der_reactive_power" in avail_tools:
            for did, entity in ders[:2]:
                output = float(entity.get("output_mw") or 0.0)
                calls.append(
                    ToolCall(
                        name="set_der_reactive_power",
                        args={
                            "der_id": did,
                            "q_mvar": round(-min(0.01, output * 0.25), 6),
                        },
                        idempotency_key=f"orc_mg_source_q_{did}_{self._tick}",
                    )
                )
        if "curtail_der" in avail_tools and ders:
            did, entity = ders[0]
            output = float(entity.get("output_mw") or 0.0)
            calls.append(
                ToolCall(
                    name="curtail_der",
                    args={"der_id": did, "target_mw": round(output * 0.6, 6)},
                    idempotency_key=f"orc_mg_source_curtail_{did}_{self._tick}",
                )
            )
        return calls

    def _datacenter_policy_review_calls(
        self,
        *,
        observation: dict[str, Any],
        avail_tools: set[str],
        policy_call_will_change: bool,
    ) -> list[ToolCall]:
        """Review visible source events under the active queue policy.

        The oracle may use ground truth for its upper-bound decisions, but it
        must cite the same runtime event ids that a normal agent could observe.
        Hidden perturbations and future job identities never enter this list.
        The review is deliberately read-only; downstream policy value is
        established later by the masked-policy replay gate.
        """
        if "review_persistent_policy" not in avail_tools:
            return []
        queue = observation.get("queue") or {}
        try:
            policy_generation = int(queue.get("policy_generation") or 1)
        except (TypeError, ValueError):
            return []
        if policy_call_will_change:
            current_policy = str(queue.get("queue_policy") or "")
            if current_policy != "shortest_job_first":
                policy_generation += 1
        try:
            current_runtime_tick = int(getattr(self._env, "tick", self._tick - 1))
        except (TypeError, ValueError):
            current_runtime_tick = self._tick - 1

        reviewable_types = {
            "job_arrival",
            "capacity_reduction",
            "queue_burst",
            "sla_deadline_pressure",
        }
        event_ids: list[str] = []
        for event in observation.get("runtime_events") or []:
            if not isinstance(event, dict):
                continue
            event_id = str(event.get("event_id") or "").strip()
            if not event_id or event_id in self._reviewed_datacenter_event_ids:
                continue
            if str(event.get("type") or "") not in reviewable_types:
                continue
            if event.get("materiality_passed") is False:
                continue
            try:
                event_tick = int(event.get("tick"))
            except (TypeError, ValueError):
                continue
            # Review on the first decision after a source event.  Waiting two
            # full runtime ticks can push a late arrival (for example W059 at
            # tick 6 of an 8-tick episode) onto the terminal decision, where
            # the review is accepted but no simulator tick remains to record
            # a native policy outcome.
            if current_runtime_tick < event_tick + 1:
                continue
            event_ids.append(event_id)
        if not event_ids:
            return []
        event_ids = sorted(event_ids)[:8]
        self._reviewed_datacenter_event_ids.update(event_ids)
        return [
            ToolCall(
                name="review_persistent_policy",
                args={
                    "event_ids": event_ids,
                    "policy_generation": policy_generation,
                    "rationale": (
                        "Review the active queue policy against visible arrivals "
                        "and source pressure; retain it only while the native "
                        "dispatch order continues to improve bounded queue cost."
                    ),
                },
                idempotency_key=self._next_idem_key("orc_dc_policy_review"),
            )
        ]

    def _datacenter_reservation_shortfall(
        self,
        backend_config: dict[str, Any],
        *,
        trigger_tick: int,
        duration_ticks: int,
        capacity_factor: float,
    ) -> float:
        """Minimum trace-backed capacity needed to admit an active job."""
        jobs = list(backend_config.get("jobs") or [])
        if not jobs:
            return 0.0
        horizon = max(1, int(self._scenario_config.get("horizon_ticks") or 1))
        tick_seconds = max(
            1.0,
            float(self._scenario_config.get("tick_minutes") or 1) * 60.0,
        )
        starts = [float(job.get("start_time") or 0.0) for job in jobs]
        minimum_start = min(starts)
        span = max(1.0, max(starts) - minimum_start)
        active_windows = []
        for job in jobs:
            submit_tick = min(
                horizon - 1,
                max(
                    0,
                    int(
                        round(
                            (
                                float(job.get("start_time") or minimum_start)
                                - minimum_start
                            )
                            / span
                            * max(1, horizon - 2)
                        )
                    ),
                ),
            )
            job_duration = max(
                1,
                int(
                    math.ceil(
                        max(1.0, float(job.get("duration_seconds") or 1.0))
                        / tick_seconds
                    )
                ),
            )
            gpu_units = max(
                0.0,
                float(job.get("requested_gpu_units") or 0.0)
                * float(job.get("instance_count") or 1.0),
            )
            active_windows.append((submit_tick, submit_tick + job_duration, gpu_units))
        interval_end = min(horizon, trigger_tick + duration_ticks)
        largest_active_job = max(
            (
                max(
                    (gpu for start, end, gpu in active_windows if start <= tick < end),
                    default=0.0,
                )
                for tick in range(trigger_tick, interval_end)
            ),
            default=0.0,
        )
        base_gpu = float(backend_config.get("gpu_capacity_units") or 0.0)
        return max(0.0, largest_active_job - base_gpu * capacity_factor)

    @staticmethod
    def _datacenter_observed_reservation_shortfall(
        ground_truth: dict[str, Any],
        *,
        available_gpu_override: float | None = None,
    ) -> float:
        """Infer the minimum reserve from native runtime queue/capacity state.

        Fetch/build-only traces intentionally omit source rows from the public
        scenario.  The oracle may inspect the environment's ground truth, but
        must not depend on a redistributed copy in ``scenario_config.jobs``.
        """

        capacity = ground_truth.get("capacity") or {}
        queue = ground_truth.get("queue") or {}
        jobs = queue.get("queued_jobs") or []
        if not isinstance(jobs, list) or not jobs:
            return 0.0
        largest_job = max(
            (
                max(0.0, float(job.get("gpu_units") or 0.0))
                for job in jobs
                if isinstance(job, dict)
            ),
            default=0.0,
        )
        available_gpu = (
            max(0.0, float(available_gpu_override))
            if available_gpu_override is not None
            else max(0.0, float(capacity.get("gpu_capacity_units") or 0.0))
        )
        allocated_gpu = max(0.0, float(capacity.get("gpu_allocated_units") or 0.0))
        free_gpu = max(0.0, available_gpu - allocated_gpu)
        return max(0.0, largest_job - free_gpu)

    @staticmethod
    def _datacenter_source_reservation_headroom(
        ground_truth: dict[str, Any],
        *,
        backend_config: dict[str, Any],
        available_gpu_override: float | None = None,
    ) -> float:
        """Return source-bounded reserve capacity not already active or queued."""

        capacity = ground_truth.get("capacity") or {}
        source_ceiling = max(
            0.0, float(backend_config.get("gpu_capacity_units") or 0.0)
        )
        available_gpu = (
            max(0.0, float(available_gpu_override))
            if available_gpu_override is not None
            else max(0.0, float(capacity.get("gpu_capacity_units") or 0.0))
        )
        pending_gpu = max(
            0.0,
            float(capacity.get("pending_reserved_gpu_units") or 0.0),
        )
        return max(0.0, source_ceiling - available_gpu - pending_gpu)

    def _routing_oracle_calls(
        self,
        *,
        gt_entities: dict[str, Any],
        avail_tools: set[str],
    ) -> list[ToolCall]:
        """Prioritize feasible deliveries without using destructive drops."""
        vehicles = [
            (vehicle_id, entity)
            for vehicle_id, entity in gt_entities.items()
            if entity.get("kind") == "vehicle"
        ]
        active = [
            (vehicle_id, entity)
            for vehicle_id, entity in vehicles
            if entity.get("active") and not entity.get("broken")
        ]
        calls: list[ToolCall] = []
        if "dispatch_vehicle" in avail_tools and any(
            entity.get("active") and entity.get("broken") for _, entity in vehicles
        ):
            standby = next(
                (
                    vehicle_id
                    for vehicle_id, entity in vehicles
                    if entity.get("is_standby") and not entity.get("active")
                ),
                None,
            )
            if standby is not None:
                calls.append(
                    ToolCall(
                        name="dispatch_vehicle",
                        args={"vehicle_id": standby},
                        idempotency_key=f"orc_log_dispatch_{standby}_{self._tick}",
                    )
                )
        if "assign_stop" not in avail_tools or not active:
            return calls

        customers = [
            (customer_id, entity)
            for customer_id, entity in gt_entities.items()
            if entity.get("kind") == "customer"
            and not entity.get("served")
            and not entity.get("dropped")
            and not entity.get("blocked")
        ]
        customers.sort(
            key=lambda item: (
                int(item[1].get("due_tick", 10**9) or 10**9),
                -float(item[1].get("criticality", 0.0) or 0.0),
                str(item[0]),
            )
        )
        remaining = {
            vehicle_id: float(entity.get("remaining_capacity", 0.0) or 0.0)
            for vehicle_id, entity in active
        }
        for customer_id, customer in customers:
            demand = float(customer.get("demand", 0.0) or 0.0)
            feasible = [
                vehicle_id
                for vehicle_id in sorted(remaining)
                if remaining[vehicle_id] + 1e-9 >= demand
            ]
            if not feasible:
                continue
            vehicle_id = max(feasible, key=lambda value: (remaining[value], value))
            remaining[vehicle_id] -= demand
            calls.append(
                ToolCall(
                    name="assign_stop",
                    args={
                        "vehicle_id": vehicle_id,
                        "customer_id": customer_id,
                    },
                    idempotency_key=(
                        f"orc_log_assign_{vehicle_id}_{customer_id}_{self._tick}"
                    ),
                )
            )
            if len(calls) >= 4:
                break
        return calls

    def _orgym_inventory_oracle_calls(
        self,
        *,
        ground_truth: dict[str, Any],
        avail_tools: set[str],
        observation: dict[str, Any],
    ) -> list[ToolCall]:
        if "place_replenishment_order" not in avail_tools:
            return []
        backend_config = self._scenario_config.get("backend_config") or {}
        policy = dict(backend_config.get("reference_control_policy") or {})
        if policy.get("kind") == "m5_source_windowed_replenishment_v1":
            return self._m5_source_windowed_replenishment_oracle_calls(
                backend_config=backend_config,
                policy=policy,
                decision_tick=int(observation.get("tick") or 0),
            )
        if self._tick < 1:
            return []
        snapshot_period = int((ground_truth.get("period") or self._tick - 1) or 0)
        cfg = dict(backend_config.get("orgym_env_config") or {})
        if backend_config.get("orgym_env_config_redacted") or "user_D" not in cfg:
            from domains.logistics.seeds.from_m5_orgym import (
                resolve_m5_orgym_env_config,
            )

            cfg = resolve_m5_orgym_env_config(backend_config)
        demand = [int(x) for x in (cfg.get("user_D") or [])]
        lead_times = [int(x) for x in (cfg.get("L") or [1])]
        lead = max(0, lead_times[0] if lead_times else 1)
        target_period = snapshot_period + lead
        if target_period >= len(demand):
            return []
        qty = int(demand[target_period])
        if qty <= 0:
            return []
        return [
            ToolCall(
                name="place_replenishment_order",
                args={"quantity": qty, "stage": 0},
                idempotency_key=f"orc_orgym_order_{target_period}_{self._tick}",
            )
        ]

    def _m5_source_windowed_replenishment_oracle_calls(
        self,
        *,
        backend_config: dict[str, Any],
        policy: dict[str, Any],
        decision_tick: int,
    ) -> list[ToolCall]:
        """Derive a staged order from a locked M5 demand window at runtime.

        The release scenario carries only phase timing and a window length.  It
        never carries an embedded raw M5 demand vector or a precomputed order
        quantity.  The oracle's privileged source view is reconstructed from
        the same SHA-locked M5 source contract consumed by the native OR-Gym
        backend.
        """
        phases = policy.get("phases")
        if not isinstance(phases, list) or not phases:
            raise ValueError("m5_source_windowed_replenishment_policy_missing_phases")
        matching = [
            phase
            for phase in phases
            if isinstance(phase, dict)
            and int(phase.get("decision_tick") or -1) == decision_tick
        ]
        if not matching:
            return []
        if len(matching) != 1:
            raise ValueError("m5_source_windowed_replenishment_policy_duplicate_tick")
        phase = matching[0]
        try:
            window_days = int(phase["arrival_window_days"])
            offset_days = int(phase.get("arrival_window_offset_days") or 0)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "m5_source_windowed_replenishment_policy_invalid_window"
            ) from exc
        if window_days <= 0 or offset_days < 0:
            raise ValueError("m5_source_windowed_replenishment_policy_invalid_window")

        cfg = dict(backend_config.get("orgym_env_config") or {})
        if backend_config.get("orgym_env_config_redacted") or "user_D" not in cfg:
            from domains.logistics.seeds.from_m5_orgym import (
                resolve_m5_orgym_env_config,
            )

            cfg = resolve_m5_orgym_env_config(backend_config)
        demand = [int(value) for value in (cfg.get("user_D") or [])]
        lead_times = [int(value) for value in (cfg.get("L") or [1])]
        capacities = [int(value) for value in (cfg.get("c") or [])]
        if not demand or not capacities:
            raise ValueError(
                "m5_source_windowed_replenishment_policy_source_unavailable"
            )
        lead = max(0, lead_times[0] if lead_times else 1)
        source_start = decision_tick + lead + offset_days
        source_end = source_start + window_days
        quantity = min(capacities[0], sum(demand[source_start:source_end]))
        if quantity <= 0:
            return []
        return [
            ToolCall(
                name="place_replenishment_order",
                args={"quantity": quantity, "stage": 0},
                idempotency_key=(
                    "orc_orgym_source_window_"
                    f"{decision_tick}_{source_start}_{source_end}"
                ),
            )
        ]

    def _opendss_ieee13_oracle_calls(
        self,
        *,
        gt_entities: dict[str, Any],
        avail_tools: set[str],
        observation: dict[str, Any] | None = None,
    ) -> list[ToolCall]:
        if str(self._scenario_config.get("domain", "power_grid")) != "power_grid":
            return []
        if str(self._scenario_config.get("backend_kind", "")) != "opendss_ieee13":
            return []
        if "set_transformer_tap" not in avail_tools:
            return []
        marker = "opendss_ieee13_release_volt_var"
        regulators = sorted(
            (
                (eid, ent)
                for eid, ent in gt_entities.items()
                if ent.get("kind") == "voltage_regulator"
            ),
            key=lambda item: str(item[0]),
        )
        if not regulators:
            return []
        # The IEEE13 release rows are source-locked to the published feeder.
        # A one-step candidate sweep in the non-release gate showed that
        # lowering regulator 3 from tap 9 to 8 removes one voltage violation
        # without worsening the band error. Keep the release oracle simple and
        # deterministic rather than embedding a full OpenDSS search loop here.
        preferred = next(
            (
                (eid, ent)
                for eid, ent in regulators
                if int(ent.get("trafo_id", -1)) == 2
            ),
            regulators[-1],
        )
        trafo_id = int(preferred[1].get("trafo_id", len(regulators) - 1))
        current_tap = int(preferred[1].get("tap_number", 9))
        backend_config = self._scenario_config.get("backend_config") or {}
        dynamic_ieee13 = (
            str(backend_config.get("feeder") or "") == "ieee13_node_test_feeder"
            and str(
                (backend_config.get("world_change_contract") or {}).get("kind") or ""
            )
            == "deterministic_procedural_overlay"
        )
        observed_material_change = any(
            str(event.get("type") or event.get("event_type") or "")
            in {"load_surge", "load_surge_cleared"}
            for event in (observation or {}).get("__last_realized_events__") or []
            if isinstance(event, dict)
        )
        review_marker = "opendss_ieee13_post_event_voltage_review"
        if (
            self._tick > 1
            and dynamic_ieee13
            and observed_material_change
            and review_marker not in self._chose_for
        ):
            voltage_max_pu = (observation or {}).get("voltage_max_pu")
            if (
                isinstance(voltage_max_pu, (int, float))
                and float(voltage_max_pu) > 1.05
                and current_tap > int(preferred[1].get("tap_min", -16))
            ):
                self._chose_for.add(review_marker)
                return [
                    ToolCall(
                        name="set_transformer_tap",
                        args={
                            "trafo_id": trafo_id,
                            "tap_pos": current_tap - 1,
                        },
                        idempotency_key=(
                            f"orc_opendss_post_event_review_{trafo_id}_{self._tick}"
                        ),
                    )
                ]
            self._chose_for.add(review_marker)
            return []
        if self._tick != 1 or marker in self._chose_for:
            return []
        # The one-step tap 8 probe improves the initial state but crosses the
        # catastrophic five-bus threshold later in both dynamic IEEE13 rows.
        # The native eight-tick sweep identifies tap 12 as the smallest shared
        # horizon-safe setting; keep static/frozen probes on the legacy rule.
        tap_pos = 12 if dynamic_ieee13 else min(current_tap, 8)
        self._chose_for.add(marker)
        calls = [
            ToolCall(
                name="set_transformer_tap",
                args={"trafo_id": trafo_id, "tap_pos": tap_pos},
                idempotency_key=f"orc_opendss_tap_{trafo_id}_{self._tick}",
            )
        ]
        if dynamic_ieee13 and "commit_to_plan" in avail_tools:
            calls.append(
                ToolCall(
                    name="commit_to_plan",
                    args={
                        "plan_id": "oracle-opendss-ieee13-voltage-supervision-v1",
                        "horizon_ticks": max(
                            2,
                            int(self._scenario_config.get("horizon_ticks") or 2),
                        ),
                        "review_after_ticks": 2,
                        "rationale": (
                            "Hold the source-bound horizon-safe regulator tap "
                            "through native load evolution, then review its "
                            "observed voltage response."
                        ),
                    },
                    idempotency_key="orc_opendss_ieee13_supervision_plan_v1",
                )
            )
        return calls

    def _opendss_fresh_feeders_oracle_calls(
        self,
        *,
        avail_tools: set[str],
        observation: dict[str, Any] | None = None,
    ) -> list[ToolCall]:
        """Apply recognized source-bound fresh-feeder reference policies.

        Task milestones and control probes describe what a row measures; they
        are not an oracle policy.  In particular, their physical tap asset and
        value never coach the baseline.  Source-program rows use an exact
        source-bound policy; ordinary feeders use current native voltage and
        line state only.
        """
        if str(self._scenario_config.get("domain", "power_grid")) != "power_grid":
            return []
        if (
            str(self._scenario_config.get("backend_kind", ""))
            != "opendss_fresh_feeders"
        ):
            return []
        backend_config = self._scenario_config.get("backend_config") or {}
        if not isinstance(backend_config.get("native_duty_program"), dict):
            return self._opendss_fresh_feedback_calls(
                avail_tools=avail_tools,
                observation=observation or {},
            )
        # Static fresh-feeder probes intentionally act on the initial decision.
        # A native duty profile instead advances independently during that first
        # simulator tick, so its reference controller must wait for the first
        # source-derived response opportunity rather than pre-positioning from
        # privileged future information.
        target_decision = (
            2 if isinstance(backend_config.get("native_duty_program"), dict) else 1
        )
        probe = backend_config.get("control_action_probe")
        if not isinstance(probe, dict):
            probe = {}
        requirements = backend_config.get("task_requirements") or {}
        milestones = requirements.get("ordered_tool_milestones") or []
        follow_up = backend_config.get("control_action_follow_up") or {}
        duty_program = backend_config.get("native_duty_program")
        solar_ramp_ieee123_source = (
            isinstance(duty_program, dict)
            and str(backend_config.get("feeder") or "") == "ieee123"
            and str(backend_config.get("master_file") or "")
            == "123Bus/IEEE123Master.dss"
            and str(backend_config.get("source_denominator_key") or "")
            == "opendss_fresh_feeders:ieee123:SolarRamp:step60"
            and int(self._scenario_config.get("horizon_ticks") or 0) == 6
            and str(duty_program.get("scenario_file") or "")
            == (
                "works/OpenDSS-IEEE13/Version8/Distrib/IEEETestCases/"
                "123Bus/SolarRamp.DSS"
            )
            and str(duty_program.get("data_file") or "")
            == (
                "works/OpenDSS-IEEE13/Version8/Distrib/IEEETestCases/"
                "123Bus/SolarRamp.csv"
            )
            and str(duty_program.get("profile_name") or "").lower() == "solarramp"
            and int(duty_program.get("start_step") or -1) == 60
            and int(duty_program.get("substeps_per_tick") or 0) == 60
        )
        source_solar_ramp_events: list[tuple[str, int]] = []
        for event in (observation or {}).get("__last_realized_events__") or []:
            if (
                not isinstance(event, dict)
                or str(event.get("type") or event.get("event_type") or "")
                != "generation_ramp"
                or str(event.get("origin") or "") != "source_schedule"
            ):
                continue
            event_id = event.get("event_id")
            event_tick = event.get("tick")
            if (
                not isinstance(event_id, str)
                or not event_id
                or not isinstance(event_tick, int)
                or isinstance(event_tick, bool)
            ):
                continue
            source_solar_ramp_events.append((event_id, event_tick))
        source_solar_ramp_events.sort(key=lambda event: (event[1], event[0]))
        if milestones:
            # Complete each milestone at the earliest legal decision tick.  A
            # marker is kept per ordinal so a repeated tool name (e.g. a later
            # tap adjustment) remains an independently auditable stage.
            for ordinal, milestone in enumerate(milestones):
                marker = f"opendss_fresh_milestone_{ordinal}"
                if marker in self._chose_for:
                    settling_marker = "opendss_ieee123_solarramp_settled"
                    transient_source_event = (
                        self._opendss_solar_ramp_transient_source_event
                    )
                    observed_newer_source_event = bool(
                        transient_source_event
                        and any(
                            event_id != transient_source_event[0]
                            and event_tick > transient_source_event[1]
                            for event_id, event_tick in source_solar_ramp_events
                        )
                    )
                    if (
                        ordinal == 0
                        and len(milestones) == 1
                        and solar_ramp_ieee123_source
                        and str(milestone.get("tool") or "") == "set_transformer_tap"
                        and int(milestone.get("not_before_tick", 0) or 0) == 1
                        and int(milestone.get("not_after_tick", 10**9) or 10**9) == 3
                        and self._tick <= 3
                        and observed_newer_source_event
                        and "set_transformer_tap" in avail_tools
                        and settling_marker not in self._chose_for
                    ):
                        # The first source-bound -12 intervention is deliberately
                        # transient: native RegControl advances during the same
                        # duty tick.  The exact-source sweep therefore settles
                        # the same actuator at 5 on the next source wake; that
                        # position persists through the following native solve.
                        self._chose_for.add(settling_marker)
                        return [
                            ToolCall(
                                name="set_transformer_tap",
                                args={"trafo_id": 0, "tap_pos": 5},
                                idempotency_key=(
                                    f"orc_opendss_ieee123_solarramp_settle_{self._tick}"
                                ),
                            )
                        ]
                    continue
                earliest = int(milestone.get("not_before_tick", 0) or 0)
                if isinstance(backend_config.get("native_duty_program"), dict):
                    # The source profile advances once before its first
                    # actionable response window.  Keep the reference policy
                    # from acting on the initial observation even when the
                    # task contract's lower bound is tick 1.
                    earliest = max(earliest, target_decision)
                latest = int(milestone.get("not_after_tick", 10**9) or 10**9)
                if self._tick < earliest or self._tick > latest:
                    return []
                tool = str(milestone.get("tool") or "")
                if tool not in avail_tools:
                    return []
                solar_ramp_ieee123 = (
                    tool == "set_transformer_tap" and solar_ramp_ieee123_source
                )
                if tool == "set_transformer_tap" and not solar_ramp_ieee123:
                    return []
                milestone_args = milestone.get("args")
                if solar_ramp_ieee123:
                    if not source_solar_ramp_events:
                        return []
                    # The static probe's tap 6 is safe only before the locked
                    # SolarRamp advances.  At the first source-derived review,
                    # the offline reference instead applies the native sweep's
                    # horizon-safe tap: -12 keeps every one of the six live
                    # OpenDSS solves below the catastrophic five-bus floor.
                    # This is scoped to the exact source program/window; all
                    # unrecognized tap policies remain fail-closed.
                    args = {"trafo_id": 0, "tap_pos": -12}
                elif isinstance(milestone_args, dict):
                    # A repeated native tool can still target a different
                    # physical asset/position; milestone-local arguments take
                    # precedence over the legacy single-action probe.
                    args = milestone_args
                elif tool == "set_transformer_tap":
                    args = probe.get("args")
                elif tool == str(follow_up.get("tool") or ""):
                    args = follow_up.get("args")
                else:
                    args = milestone.get("args")
                if not isinstance(args, dict):
                    return []
                self._chose_for.add(marker)
                if solar_ramp_ieee123:
                    self._opendss_solar_ramp_transient_source_event = (
                        source_solar_ramp_events[-1]
                    )
                calls = [
                    ToolCall(
                        name=tool,
                        args=dict(args),
                        idempotency_key=(
                            f"orc_opendss_fresh_milestone_{ordinal}_{self._tick}"
                        ),
                    )
                ]
                if (
                    solar_ramp_ieee123
                    and source_solar_ramp_events
                    and "commit_to_plan" in avail_tools
                ):
                    calls.append(
                        ToolCall(
                            name="commit_to_plan",
                            args={
                                "plan_id": (
                                    "oracle-opendss-ieee123-solarramp-supervision-v1"
                                ),
                                "rationale": (
                                    "Apply the source-bound transient tap, then "
                                    "review the next SolarRamp solve and settle "
                                    "the native regulator response."
                                ),
                                "predicted_events": [
                                    {
                                        "event_type": "generation_ramp",
                                        "tick_offset": 1,
                                    }
                                ],
                                "horizon_ticks": 2,
                                "review_after_ticks": 2,
                            },
                            idempotency_key=(
                                f"orc_opendss_ieee123_solarramp_plan_{self._tick}"
                            ),
                        )
                    )
                return calls
            return []

        if self._tick != target_decision:
            return []
        if "set_transformer_tap" not in avail_tools:
            return []
        # A task probe is an evaluation target, not a policy contract.  Static
        # fresh-feeder tap probes therefore remain held until an observation-
        # derived or independently source-bound oracle policy is implemented.
        return []

    def _opendss_fresh_feedback_calls(
        self,
        *,
        avail_tools: set[str],
        observation: dict[str, Any],
    ) -> list[ToolCall]:
        """Respond to native fresh-feeder state without replaying YAML answers."""
        ground_truth = self._env.ground_truth() if self._env is not None else {}
        line_rows = [
            row for row in ground_truth.get("lines") or [] if isinstance(row, dict)
        ]
        disconnected = sorted(
            (
                row
                for row in line_rows
                if not bool(row.get("in_service"))
                and bool(
                    row.get(
                        "unexpectedly_disconnected",
                        not bool(row.get("in_service")),
                    )
                )
            ),
            key=lambda row: int(row.get("line_index", -1)),
        )
        for row in line_rows:
            if bool(row.get("in_service")):
                self._chose_for.discard(
                    f"opendss_fresh_reclose_{int(row.get('line_index', -1))}"
                )
        if disconnected and "switch_branch" in avail_tools:
            line_index = int(disconnected[0].get("line_index", -1))
            marker = f"opendss_fresh_reclose_{line_index}"
            if line_index >= 0 and marker not in self._chose_for:
                self._chose_for.add(marker)
                return [
                    ToolCall(
                        name="switch_branch",
                        args={"line_index": line_index, "connect": True},
                        idempotency_key=(
                            f"orc_opendss_fresh_reclose_{line_index}_{self._tick}"
                        ),
                    )
                ]

        n_violations = int(observation.get("n_voltage_violations") or 0)
        voltage_min = float(observation.get("voltage_min_pu") or 0.0)
        voltage_max = float(observation.get("voltage_max_pu") or 0.0)
        if n_violations > 0 and voltage_max > 1.05:
            enabled_capacitors = sorted(
                (
                    row
                    for row in observation.get("capacitors") or []
                    if isinstance(row, dict)
                    and any(int(state) for state in row.get("states") or [])
                ),
                key=lambda row: int(row.get("cap_id", -1)),
            )
            if enabled_capacitors and "switch_capacitor" in avail_tools:
                return [
                    ToolCall(
                        name="switch_capacitor",
                        args={
                            "cap_id": int(capacitor.get("cap_id", 0)),
                            "status": False,
                        },
                        idempotency_key=(
                            "orc_opendss_fresh_feedback_cap_off_"
                            f"{int(capacitor.get('cap_id', 0))}_{self._tick}"
                        ),
                    )
                    for capacitor in enabled_capacitors
                ]
            if voltage_min >= 0.955:
                return []
        # A narrow source-native safety margin supports preventive control on
        # an already near-limit feeder.  This is computed from the live solve,
        # not from task probe arguments or a milestone deadline.
        if n_violations <= 0 and (voltage_min <= 0.0 or voltage_min >= 0.955):
            return []

        regulators = sorted(
            (
                row
                for row in observation.get("regcontrols") or []
                if isinstance(row, dict)
            ),
            key=lambda row: int(row.get("trafo_id", -1)),
        )
        if regulators and "set_transformer_tap" in avail_tools:
            regulator = regulators[0]
            current = int(regulator.get("tap_number", 0))
            maximum = int(regulator.get("tap_max", current))
            backend_config = self._scenario_config.get("backend_config") or {}
            repeated_ieee34_feedback = (
                str(backend_config.get("feeder") or "") == "ieee34"
                and str(backend_config.get("master_file") or "")
                == "34Bus/ieee34Mod1.dss"
            )
            target = min(maximum, current + (4 if repeated_ieee34_feedback else 2))
            tap_marker = "opendss_fresh_feedback_tap"
            if repeated_ieee34_feedback:
                tap_marker += f"_{int(regulator.get('trafo_id', 0))}_{current}_{target}"
            if target != current and tap_marker not in self._chose_for:
                self._chose_for.add(tap_marker)
                # Give the persistent native tap four decision cycles before
                # escalating to a capacitor.  This is a feedback review delay,
                # not a task-milestone tick copied from scenario metadata.
                self._opendss_fresh_cap_review_tick = self._tick + 4
                calls = [
                    ToolCall(
                        name="set_transformer_tap",
                        args={
                            "trafo_id": int(regulator.get("trafo_id", 0)),
                            "tap_pos": target,
                        },
                        idempotency_key=(
                            f"orc_opendss_fresh_feedback_tap_{self._tick}"
                        ),
                    )
                ]
                if "commit_to_plan" in avail_tools:
                    calls.append(
                        ToolCall(
                            name="commit_to_plan",
                            args={
                                "plan_id": ("oracle-opendss-fresh-voltage-feedback-v1"),
                                "horizon_ticks": 4,
                                "review_after_ticks": 2,
                                "rationale": (
                                    "Hold the observed undervoltage tap "
                                    "correction, monitor native line state, "
                                    "then escalate only if voltage stress "
                                    "persists."
                                ),
                            },
                            idempotency_key=("orc_opendss_fresh_feedback_plan_v1"),
                        )
                    )
                return calls

        if self._tick < int(
            getattr(self, "_opendss_fresh_cap_review_tick", self._tick)
        ):
            return []
        capacitors = sorted(
            (
                row
                for row in observation.get("capacitors") or []
                if isinstance(row, dict)
                and not any(int(state) for state in row.get("states") or [])
            ),
            key=lambda row: int(row.get("cap_id", -1)),
        )
        if capacitors and "switch_capacitor" in avail_tools:
            capacitor = capacitors[0]
            self._opendss_fresh_cap_review_tick = 10**9
            return [
                ToolCall(
                    name="switch_capacitor",
                    args={
                        "cap_id": int(capacitor.get("cap_id", 0)),
                        "status": True,
                    },
                    idempotency_key=(f"orc_opendss_fresh_feedback_cap_{self._tick}"),
                )
            ]
        return []

    def _jsplib_job_shop_oracle_calls(
        self,
        *,
        ground_truth: dict[str, Any],
        avail_tools: set[str],
        observation: dict[str, Any],
    ) -> list[ToolCall]:
        active_disruptions = ground_truth.get("active_machine_disruptions") or {}
        if "repair_machine" in avail_tools and isinstance(active_disruptions, dict):
            # A delayed repair can materialize in the same supervisory tick
            # that produces the next observation.  Do not immediately submit
            # the same repair again while its clearance is still reflected in
            # ``active_machine_disruptions``; prefer a different active outage
            # so Extreme rows can prove both recovery milestones.
            recently_repaired: set[int] = set()
            for result in observation.get("__last_tool_results__") or []:
                if not isinstance(result, dict):
                    continue
                if result.get("name") != "repair_machine" or not result.get("ok"):
                    continue
                payload = result.get("payload") or {}
                if payload.get("_status") == "machine_repair_requested":
                    try:
                        recently_repaired.add(int(payload["machine_id"]))
                    except (KeyError, TypeError, ValueError):
                        continue
            active_machine_ids = sorted(
                int(machine_id)
                for machine_id, until_tick in active_disruptions.items()
                if int(until_tick) > self._tick
            )
            repairable_machine_ids = [
                machine_id
                for machine_id in active_machine_ids
                if machine_id not in recently_repaired
            ]
            if repairable_machine_ids:
                machine_id = repairable_machine_ids[0]
                return [
                    ToolCall(
                        name="repair_machine",
                        args={"machine_id": machine_id},
                        idempotency_key=self._next_idem_key(
                            f"orc_jobshop_repair_{machine_id}_{self._tick}"
                        ),
                    )
                ]
        if not {
            "dispatch_ready_operations",
            "dispatch_job_operation",
        }.intersection(avail_tools):
            return []
        ready = ground_truth.get("ready_operations") or {}
        if not isinstance(ready, dict) or not ready:
            return []
        job_available = ground_truth.get("job_available_at") or {}
        machine_available = ground_truth.get("machine_available_at") or {}

        def key(item: tuple[str, Any]) -> tuple[int, int, int, str]:
            job_id, op = item
            if not isinstance(op, dict):
                return (10**12, 10**12, 10**12, str(job_id))
            machine_id = int(op.get("machine_id", 0) or 0)
            duration = int(op.get("duration", 0) or 0)
            start = max(
                int(job_available.get(job_id, 0) or 0),
                int(machine_available.get(machine_id, 0) or 0),
            )
            return (start + duration, duration, machine_id, str(job_id))

        ordered: list[dict[str, Any]] = []
        for job_id, op in sorted(ready.items(), key=key):
            if not isinstance(op, dict):
                continue
            ordered.append(
                {
                    "job_id": str(job_id),
                    "operation_index": int(op.get("operation_index", 0) or 0),
                }
            )
        if "dispatch_ready_operations" in avail_tools and ordered:
            dynamic_config = (self._scenario_config.get("backend_config") or {}).get(
                "dynamic_job_shop"
            ) or {}
            try:
                batch_limit = int(dynamic_config.get("max_dispatch_batch_size", 50))
            except (TypeError, ValueError):
                batch_limit = 50
            batch = ordered[: max(1, min(50, batch_limit))]
            return [
                ToolCall(
                    name="dispatch_ready_operations",
                    args={"operations": batch},
                    idempotency_key=self._next_idem_key(
                        f"orc_jobshop_batch_tick_{self._tick}"
                    ),
                )
            ]
        return [
            ToolCall(
                name="dispatch_job_operation",
                args=item,
                idempotency_key=self._next_idem_key(
                    f"orc_jobshop_{item['job_id']}_{item['operation_index']}"
                ),
            )
            for item in ordered
        ]
