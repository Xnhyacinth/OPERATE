"""
domains.power_grid.backends.grid2op_backend — Optional Grid2Op wrapper.

Provides a real AC/DC power-flow simulator with:

- IEEE n-bus chronics (l2rpn_case14_sandbox is the v0.1 default)
- Seeded storm/opponent perturbations injected as real Grid2Op actions
- Native rho (line loading) and overload information
- Topology actions (substation reconfigurations) and line switching

Import is lazy: ``from .grid2op_backend import Grid2OpBackend`` does not
itself ``import grid2op`` until ``Grid2OpBackend()`` is instantiated. This
keeps the rest of OPERATE operable on Python installs that do not
have Grid2Op available.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from typing import Any

from ..seeds.schema import ScenarioSeed

LOGGER = logging.getLogger(__name__)


GRID2OP_AVAILABLE = False
try:  # pragma: no cover - exercised in CI when grid2op is installed
    import grid2op  # type: ignore[import]
    from grid2op.Parameters import Parameters  # type: ignore[import]
    from grid2op.Reward import EconomicReward  # type: ignore[import]

    GRID2OP_AVAILABLE = True
except ImportError:  # pragma: no cover
    grid2op = None  # type: ignore[assignment]


def _resolve_local_env_name(
    env_name: str,
    test_env: bool,
    *,
    available_local_envs: list[str],
) -> tuple[str, bool]:
    """Resolve legacy Grid2Op aliases without network-capable probing."""
    available = set(available_local_envs)
    if env_name in available or not env_name.endswith("_small"):
        return env_name, test_env
    canonical = env_name.removesuffix("_small")
    if canonical in available:
        return canonical, True
    return env_name, test_env


# grid2op's MakeEnv fetches its dataset index from api.github.com anonymously,
# which hits the 60/hour unauthenticated rate limit fast (each grid2op.make()
# call re-checks). If GITHUB_TOKEN is set, inject it as a Bearer header on
# grid2op's requests.Session so the limit rises to 5000/hour. This is a
# best-effort patch: if grid2op's internals move, it silently no-ops.
import os as _os  # noqa: E402


def _patch_grid2op_github_auth() -> None:
    token = _os.environ.get("GITHUB_TOKEN") or _os.environ.get("GH_TOKEN")
    if not token or grid2op is None:
        return
    try:
        from grid2op.MakeEnv import Make as _g2op_make  # type: ignore[import]
        if getattr(_g2op_make, "_dt_sched_auth_patched", False):
            return
        # Wrap _send_request_retry so each call uses an authed session.
        _orig_send = _g2op_make._send_request_retry

        def _authed_send(url, nb_retry=3, gh_session=None):  # type: ignore[no-untyped-def]
            if gh_session is None:
                import requests  # type: ignore[import]
                gh_session = requests.Session()
                gh_session.headers["Authorization"] = f"Bearer {token}"
            gh_session.headers["Authorization"] = f"Bearer {token}"
            return _orig_send(url, nb_retry=nb_retry, gh_session=gh_session)

        _g2op_make._send_request_retry = _authed_send
        _g2op_make._dt_sched_auth_patched = True
        LOGGER.info("GITHUB_TOKEN detected: patched grid2op to authenticate GitHub API calls (5000/hr)")
    except Exception as exc:  # noqa: BLE001  pragma: no cover
        LOGGER.debug("grid2op github-auth patch skipped: %s", exc)


_patch_grid2op_github_auth()


class Grid2OpUnavailable(RuntimeError):
    """Raised when grid2op is required but not importable."""


@dataclass
class _Grid2OpTickRecord:
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
    n_disconnected_lines: int
    economic_reward: float
    done: bool = False
    realized_events: list[dict[str, Any]] = field(default_factory=list)


class Grid2OpBackend:
    """Thin wrapper around grid2op.make() exposing the same surface as
    PglibUcSyntheticBackend so the adapter can route to either uniformly."""

    # Per-class VoLL tariff (matches the synthetic backend).
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
    SHED_PENALTY_PER_MWH = SHED_PENALTY_DEFAULT  # legacy alias
    # Per-MWh production-cost proxy used when mapping Grid2Op EconomicReward
    # (which lives in [0, 1] with higher = lower marginal cost) into a
    # positive cost component the OPERATE scorer can consume.
    # Calibrated against typical pglib-uc thermal marginal costs.
    PRODUCTION_COST_PER_MWH = 50.0
    OVERLOAD_COST_PER_TICK = 200.0
    DISCONNECTION_COST_PER_LINE_TICK = 500.0
    # Reserve targets at the Grid2Op backend level (we don't have a native
    # spinning-reserve quantity in L2RPN envs, so we synthesize one as a
    # fraction of demand and credit non-loading capacity as procured).
    RESERVE_TARGET_FRACTION_OF_DEMAND = 0.10

    def __init__(self, *, frame_skip: int = 3) -> None:
        """Construct a Grid2Op backend wrapper.

        Args:
            frame_skip: Number of native L2RPN sub-steps the backend
                advances per benchmark tick. The L2RPN sandbox is
                natively a 5-min step ("288 steps/day", see Grid2Op
                ``l2rpn_case14_sandbox`` chronics). OPERATE
                operates at supervisory-decision cadence (default
                15 min), so each agent tick performs the agent's
                composed action on the FIRST sub-step and ``noop``
                on the remaining ``frame_skip - 1`` sub-steps. With
                ``frame_skip=3``, 1 LLM tick = 15 min of physical
                time. ``backend_config["frame_skip"]`` may override
                this per scenario.
        """
        if not GRID2OP_AVAILABLE:
            raise Grid2OpUnavailable(
                "grid2op is not installed. Install via `pip install grid2op` "
                "(Python 3.10–3.14 supported). OPERATE will fall "
                "back to the pglib_uc_synthetic backend otherwise."
            )
        self._env = None
        self._seed_obj: ScenarioSeed | None = None
        self._tick: int = 0
        self._horizon: int = 24
        self._tick_records: list[_Grid2OpTickRecord] = []
        self._loads: dict[str, dict[str, Any]] = {}
        self._pending_action_payloads: list[dict[str, Any]] = []
        self._last_obs: Any | None = None
        self._last_reward: float = 0.0
        self._done: bool = False
        # L2RPN frame-skip: agent acts on sub-step 1; noop on the rest.
        self._frame_skip: int = max(1, int(frame_skip))
        # accumulators
        self._cumulative_shed_mwh: dict[str, float] = {}
        # v0.2.2 F-01: per-episode delayed-effect queue for request_mutual_aid.
        self._pending_mutual_aid: list[tuple[int, float]] = []

    # ── Reset ───────────────────────────────────────────────────────────

    def reset(
        self, scenario_seed: ScenarioSeed
    ) -> None:  # pragma: no cover - needs grid2op
        self._seed_obj = scenario_seed
        self._tick = 0
        self._horizon = scenario_seed.horizon_ticks
        self._tick_records.clear()
        self._cumulative_shed_mwh.clear()
        self._pending_action_payloads.clear()
        self._last_obs = None
        self._last_reward = 0.0
        self._done = False
        # v0.2.2 F-01: clear any mutual-aid carryover between episodes.
        self._pending_mutual_aid = []
        self._pending_reserve_extra_mw = 0.0

        env_name = scenario_seed.backend_config.get("env_name", "l2rpn_case14_sandbox")
        test_env = bool(scenario_seed.backend_config.get("test", True))
        # Never probe an environment with ``grid2op.make``: a missing name can
        # trigger an interactive remote download. Prefer the exact local name,
        # and use the suffix-free alias only when it is already installed.
        env_name, test_env = _resolve_local_env_name(
            env_name,
            test_env,
            available_local_envs=grid2op.list_available_local_env(),
        )
        # Allow per-scenario frame_skip override (default falls back to the
        # constructor value). Stored on backend_config rather than on
        # ScenarioSeed itself so existing scenario hashes stay stable.
        cfg_skip = scenario_seed.backend_config.get("frame_skip")
        if cfg_skip is not None:
            self._frame_skip = max(1, int(cfg_skip))
        params = Parameters()
        # Whether cascading line trips from sustained overloads
        # auto-terminate the episode. Default True (overload damage is
        # surfaced through the safety_violation dimension, NOT by a hard
        # game-over). Extreme-difficulty scenarios can flip this back to
        # False via backend_config to expose cascading-failure dynamics.
        params.NO_OVERFLOW_DISCONNECTION = bool(
            scenario_seed.backend_config.get("no_overflow_disconnection", True)
        )
        env = grid2op.make(
            env_name,
            param=params,
            reward_class=EconomicReward,
            test=test_env,
        )
        env.seed(int(scenario_seed.seed))
        chronics_id = int(scenario_seed.backend_config.get("chronics_id", 0))
        with contextlib.suppress(Exception):
            env.set_id(chronics_id)
        start_step = int(scenario_seed.backend_config.get("start_step", 0))
        self._last_obs = env.reset()
        for _ in range(start_step):
            self._last_obs, self._last_reward, self._done, _ = env.step(
                env.action_space({})
            )
        self._env = env

        # map load ids → stakeholder assignment
        self._loads = {}
        for assignment in scenario_seed.load_assignments:
            self._loads[assignment.load_id] = {
                "bus_id": assignment.bus_id,
                "stakeholder_class": assignment.stakeholder_class,
                "criticality": assignment.criticality,
                "shed_this_tick_mw": 0.0,
            }
            self._cumulative_shed_mwh[assignment.load_id] = 0.0

    # ── Tool effects ────────────────────────────────────────────────────

    def apply_tool_effect(
        self, name: str, args: dict[str, Any]
    ) -> dict[str, Any]:  # pragma: no cover - needs grid2op
        assert self._env is not None
        if name == "switch_branch":
            line_id = int(args.get("line_index", 0))
            connect = bool(args.get("connect", True))
            try:
                self._queue_action(
                    {"set_line_status": [(line_id, 1 if connect else -1)]}
                )
                return {"line_id": line_id, "connect": connect, "queued": True}
            except Exception as exc:
                return {"_status": "error", "error": str(exc)}
        if name == "topology_action":
            sub_id = int(args.get("substation_id", 0))
            config = args.get("bus_config", [])
            try:
                self._queue_action({"set_bus": {"substations_id": [(sub_id, config)]}})
                return {"substation_id": sub_id, "queued": True}
            except Exception as exc:
                return {"_status": "error", "error": str(exc)}
        if name == "redispatch_generation":
            try:
                gid = self._generator_index(args)
                target = float(args.get("target_mw", args.get("delta_mw", 0.0)))
                current = self._current_generator_mw(gid)
                delta = (
                    float(args["delta_mw"]) if "delta_mw" in args else target - current
                )
                self._queue_action({"redispatch": [(gid, delta)]})
                return {
                    "generator_index": gid,
                    "generator_id": args.get("generator_id", str(gid)),
                    "target_mw": target,
                    "delta_mw": round(delta, 3),
                    "queued": True,
                }
            except Exception as exc:
                return {"_status": "error", "error": str(exc)}
        if name == "shed_load":
            target = str(args.get("load_id", ""))
            mw = float(args.get("mw", 0.0))
            entry = self._loads.get(target)
            if entry is None:
                return {"_status": "error", "error": "unknown_load", "load_id": target}
            entry["shed_this_tick_mw"] += mw
            tick_h = float(self._seed_obj.tick_minutes if self._seed_obj else 60) / 60.0
            self._cumulative_shed_mwh[target] += mw * tick_h
            # v0.1.3 (per architect review): also queue an offsetting
            # redispatch DOWN on the most-loaded generator so the net
            # load reduction propagates to Grid2Op physics. L2RPN
            # sandbox has no native load curtailment API; this is the
            # closest physical analogue (less demand → less generation
            # → less line loading). Without this the agent's shed only
            # affects the score-side synthetic relief.
            try:
                obs = self._env.get_obs() if self._env is not None else None
                if obs is not None:
                    gen_p = list(getattr(obs, "gen_p", []) or [])
                    if gen_p:
                        # Pick the highest-output dispatchable gen.
                        # Only redispatchable units accept negative deltas.
                        redispatchable = getattr(self._env, "gen_redispatchable", None)
                        candidates = [
                            (i, gen_p[i])
                            for i in range(len(gen_p))
                            if (redispatchable is None or bool(redispatchable[i]))
                        ]
                        if candidates:
                            candidates.sort(key=lambda kv: kv[1], reverse=True)
                            top_gid = candidates[0][0]
                            # Cap the delta to the gen's down ramp limit.
                            ramp_down = getattr(self._env, "gen_max_ramp_down", None)
                            max_step = (
                                float(ramp_down[top_gid])
                                if ramp_down is not None
                                else mw
                            )
                            delta = -min(mw, max_step)
                            self._queue_action({"redispatch": [(top_gid, delta)]})
            except Exception:
                # Best-effort propagation; never fail the shed itself.
                pass
            return {
                "load_id": target,
                "shed_mw": mw,
                "stakeholder_class": entry["stakeholder_class"],
                "criticality": entry["criticality"],
                "info": (
                    "shed recorded; offsetting redispatch queued to "
                    "approximate physical curtailment (L2RPN sandbox "
                    "has no native load_p API)"
                ),
            }
        if name == "commit_reserve":
            # L2RPN envs have no native spinning-reserve channel; we credit
            # the agent's intent so foresight/audit see the call, but the
            # numerical effect lives in the synthetic reserve tracker below.
            mw = float(args.get("mw", 0.0))
            self._pending_reserve_extra_mw = (
                getattr(self, "_pending_reserve_extra_mw", 0.0) + mw
            )
            return {
                "reserve_pending_mw": self._pending_reserve_extra_mw,
                "info": f"{name} queued (Grid2Op reserve channel synthetic)",
            }
        if name == "request_mutual_aid":
            # v0.2.2 F-01: handled by the dedicated delayed-effect path
            # (queue_mutual_aid_effect / _drain_mutual_aid); never mutate
            # reserves immediately from this code path.
            return {
                "_status": "ack",
                "info": (
                    "mutual-aid uses the dedicated delayed-effect path; "
                    "this code path no longer mutates reserves"
                ),
            }
        return {"_status": "noop"}

    # ── v0.2.2 F-01: unified delayed-effect API for request_mutual_aid ──

    def queue_mutual_aid_effect(
        self, *, due_tick: int, mw: float
    ) -> None:  # pragma: no cover - needs grid2op for end-to-end runs
        """Queue a mutual-aid reserve injection to land at ``due_tick``.

        Drained at the START of ``tick(due_tick)`` so the reserve only
        appears at exactly that tick.
        """
        self._pending_mutual_aid.append((int(due_tick), float(mw)))

    def _drain_mutual_aid(
        self, current_tick: int
    ) -> float:  # pragma: no cover - exercised via end-to-end runs
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

    def _queue_action(self, payload: dict[str, Any]) -> None:
        self._pending_action_payloads.append(payload)

    def _generator_index(self, args: dict[str, Any]) -> int:
        if "generator_index" in args:
            return int(args["generator_index"])
        raw = str(args.get("generator_id", "0"))
        return int(raw.removeprefix("gen_").removeprefix("generator_"))

    def _current_generator_mw(self, generator_index: int) -> float:
        assert self._env is not None
        obs = self._env.get_obs()
        if obs is None:
            return 0.0
        for attr in ("gen_p", "prod_p"):
            values = getattr(obs, attr, None)
            if values is not None and generator_index < len(values):
                return float(values[generator_index])
        return 0.0

    def _record_step(
        self,
        obs: Any,
        reward: float | None,
        done: bool,
        info: dict[str, Any] | None,
        realized_events: list[dict[str, Any]],
    ) -> _Grid2OpTickRecord:  # pragma: no cover
        # On game-over Grid2Op zeroes obs.rho / obs.line_status / obs.load_p
        # — counting them as "all 20 lines tripped this tick" would inflate
        # disconnection_cost by ~$10k regardless of agent behaviour. Preserve
        # the last-known values when the env reports done=True with zeroed obs.
        is_terminal_zeroed = (
            bool(done)
            and obs is not None
            and (
                (
                    hasattr(obs, "rho")
                    and len(obs.rho) > 0
                    and float(max(obs.rho)) == 0.0
                )
                and (hasattr(obs, "line_status") and not any(obs.line_status))
            )
        )
        last = self._tick_records[-1] if self._tick_records else None
        if is_terminal_zeroed and last is not None:
            rho_max = last.rho_max
            n_overload = last.n_overloads
            n_disc = last.n_disconnected_lines
            demand_mw = last.aggregate_demand_mw
            generation_mw = last.aggregate_generation_mw
        else:
            rho = obs.rho if obs is not None else []
            rho_max = float(max(rho)) if len(rho) > 0 else 0.0
            n_overload = int(sum(1 for r in rho if r > 1.0))
            n_disc = (
                int(sum(1 for s in obs.line_status if not s)) if obs is not None else 0
            )
            demand_mw = self._sum_obs(obs, "load_p")
            generation_mw = self._sum_obs(obs, "gen_p", "prod_p")
        economic_reward = float(reward) if reward is not None else 0.0
        # Per-class shed penalty using the VoLL tariff table.
        shed_mw = 0.0
        shed_penalty_this_tick = 0.0
        for entry in self._loads.values():
            sh = float(entry.get("shed_this_tick_mw", 0.0))
            if sh <= 0:
                continue
            tariff = self.SHED_PENALTY_PER_MWH_BY_CLASS.get(
                str(entry.get("stakeholder_class", "")), self.SHED_PENALTY_DEFAULT
            )
            # tick is 15 min for L2RPN sandbox; convert MW × tick → MWh
            tick_h = float(self._seed_obj.tick_minutes if self._seed_obj else 15) / 60.0
            shed_penalty_this_tick += sh * tariff * tick_h
            shed_mw += sh
        # Synthetic shed-relief: Grid2Op IEEE-14 sandbox doesn't support
        # native load curtailment, but shedding a fraction f of demand on
        # an overloaded path reduces line loading by roughly 1.5f under
        # standard power-flow sensitivities. Apply this so the agent's
        # shed actions actually move the overload signal — otherwise the
        # safety_violation dimension never responds and the shed_penalty
        # is unfairly net-negative for any active agent.
        if shed_mw > 0 and demand_mw > 0:
            relief_factor = min(0.6, 1.5 * shed_mw / demand_mw)
            rho_max = max(0.0, rho_max * (1.0 - relief_factor))
            # Also reduce overload count proportionally
            if relief_factor >= 0.3 and n_overload > 0:
                n_overload = max(
                    0, n_overload - max(1, int(n_overload * relief_factor))
                )
        opponent_events = self._events_from_info(info)
        # EconomicReward returns values in [0, 1] where higher = better. Map
        # (1 - r) × $/MWh × demand_MWh to a positive production-cost proxy
        # the OPERATE scorer can accumulate. Zero on terminal ticks.
        tick_h = float(self._seed_obj.tick_minutes if self._seed_obj else 60) / 60.0
        if is_terminal_zeroed:
            prod_cost = 0.0
        else:
            efficiency_gap = 1.0 - max(0.0, min(1.0, economic_reward))
            prod_cost = (
                efficiency_gap * self.PRODUCTION_COST_PER_MWH * demand_mw * tick_h
            )
        # Synthetic reserves: required = X% of demand, procured = unused
        # gen-headroom + any pending mutual-aid / committed reserve.
        reserves_required = self.RESERVE_TARGET_FRACTION_OF_DEMAND * demand_mw
        gen_max = self._sum_obs(obs, "gen_pmax")
        if gen_max <= 0:
            gen_max = generation_mw * 1.3
        reserves_procured = max(0.0, gen_max - generation_mw) + float(
            getattr(self, "_pending_reserve_extra_mw", 0.0)
        )
        record = _Grid2OpTickRecord(
            tick=self._tick,
            aggregate_demand_mw=round(demand_mw, 2),
            aggregate_generation_mw=round(generation_mw, 2),
            balance_error_mw=round(generation_mw - demand_mw, 2),
            reserves_required_mw=round(reserves_required, 2),
            reserves_procured_mw=round(reserves_procured, 2),
            production_cost=round(prod_cost, 2),
            startup_cost=0.0,
            shed_penalty=round(shed_penalty_this_tick, 2),
            rho_max=rho_max,
            n_overloads=n_overload,
            n_disconnected_lines=n_disc,
            economic_reward=economic_reward,
            done=bool(done),
            realized_events=[*realized_events, *opponent_events],
        )
        self._tick_records.append(record)
        self._last_obs = obs
        self._last_reward = economic_reward
        self._done = bool(done)
        return record

    def _sum_obs(self, obs: Any, *attrs: str) -> float:
        if obs is None:
            return 0.0
        for attr in attrs:
            values = getattr(obs, attr, None)
            if values is not None:
                return (
                    float(values.sum())
                    if hasattr(values, "sum")
                    else float(sum(values))
                )
        return 0.0

    def _events_from_info(self, info: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not isinstance(info, dict):
            return []
        attack = info.get("opponent_attack_line")
        if attack is None:
            return []
        if isinstance(attack, list):
            return [
                {"type": "opponent_attack", "tick": self._tick, "line_id": line_id}
                for line_id in attack
            ]
        return [{"type": "opponent_attack", "tick": self._tick, "line_id": attack}]

    # ── Tick (driven by adapter — perturbations apply here) ─────────────

    def tick(self, current_tick: int) -> _Grid2OpTickRecord:  # pragma: no cover
        assert self._env is not None
        self._tick = current_tick
        # v0.2.2 F-01: drain matured mutual-aid effects BEFORE composing
        # the env action so the reserve increment is visible in THIS tick's
        # record and not earlier or later.
        matured_aid_mw = self._drain_mutual_aid(current_tick)
        if matured_aid_mw > 0.0:
            self._pending_reserve_extra_mw = (
                getattr(self, "_pending_reserve_extra_mw", 0.0) + matured_aid_mw
            )
        realized_events = self._queue_perturbations(current_tick)
        if matured_aid_mw > 0.0:
            realized_events.append(
                {
                    "type": "mutual_aid_arrived",
                    "tick": current_tick,
                    "mw": round(matured_aid_mw, 3),
                }
            )
        # L2RPN sandbox is natively 5 min/step (see Grid2Op
        # ``l2rpn_case14_sandbox`` chronics: 288 steps/day). OPERATE
        # operates at supervisory-decision cadence (default 15 min ≡
        # ``frame_skip=3``). Each agent tick performs the agent's
        # composed action on the FIRST sub-step and ``noop`` on the
        # remaining ``frame_skip - 1`` sub-steps. We collect per-sub-step
        # opponent-attack info events into the realized-event UNION,
        # use the LAST sub-step's observation for physical signals, and
        # average the per-sub-step economic reward so the resulting
        # production_cost reflects ``tick_minutes`` of dispatch (not
        # ``tick_minutes / frame_skip``). Game-over short-circuits the
        # loop and uses the terminal observation directly.
        action = self._compose_pending_action()
        rewards: list[float] = []
        info_events: list[dict[str, Any]] = []
        last_obs: Any = None
        last_done: bool = False
        for _sub in range(self._frame_skip):
            obs, reward, done, info = self._env.step(action)
            rewards.append(float(reward) if reward is not None else 0.0)
            last_obs, last_done = obs, bool(done)
            info_events.extend(self._events_from_info(info))
            # Subsequent sub-steps are pure noop (action consumed once).
            action = self._env.action_space({})
            if done:
                break
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        # Pass ``info=None`` so ``_record_step`` does NOT redo
        # ``_events_from_info`` on the last sub-step's info — we have
        # already unioned info_events across all sub-steps above.
        record = self._record_step(
            last_obs,
            avg_reward,
            last_done,
            None,
            [*realized_events, *info_events],
        )
        # reset per-tick shed counter
        for entry in self._loads.values():
            entry["shed_this_tick_mw"] = 0.0
        return record

    def _compose_pending_action(self) -> Any:
        assert self._env is not None
        action = self._env.action_space({})
        for payload in self._pending_action_payloads:
            action += self._env.action_space(payload)
        self._pending_action_payloads.clear()
        return action

    def _queue_perturbations(self, current_tick: int) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if self._seed_obj is None:
            return events
        for perturbation in self._seed_obj.perturbations:
            start = int(perturbation.trigger_tick)
            end = start + max(1, int(perturbation.duration_ticks))
            if perturbation.kind == "line_outage":
                line_id = int(perturbation.target.get("line_index", 0))
                if current_tick == start:
                    self._queue_action({"set_line_status": [(line_id, -1)]})
                    events.append(
                        {
                            "type": "line_outage",
                            "tick": current_tick,
                            "line_id": line_id,
                            "cause": perturbation.target.get("cause"),
                            "hidden": perturbation.hidden,
                            "intensity": perturbation.intensity,
                        }
                    )
                elif current_tick == end:
                    self._queue_action({"set_line_status": [(line_id, 1)]})
                    events.append(
                        {
                            "type": "line_restored",
                            "tick": current_tick,
                            "line_id": line_id,
                            "hidden": perturbation.hidden,
                        }
                    )
            elif perturbation.kind == "opponent_attack" and start <= current_tick < end:
                line_id = self._opponent_attack_line(current_tick, perturbation.target)
                self._queue_action({"set_line_status": [(line_id, -1)]})
                events.append(
                    {
                        "type": "opponent_attack",
                        "tick": current_tick,
                        "line_id": line_id,
                        "strategy": perturbation.target.get(
                            "strategy", "seeded_line_attack"
                        ),
                        "hidden": perturbation.hidden,
                        "intensity": perturbation.intensity,
                    }
                )
            elif perturbation.kind == "storm_window" and current_tick == start:
                events.append(
                    {
                        "type": "storm_window",
                        "tick": current_tick,
                        "intensity": perturbation.intensity,
                        "hidden": perturbation.hidden,
                    }
                )
        return events

    def _opponent_attack_line(self, current_tick: int, target: dict[str, Any]) -> int:
        assert self._env is not None
        obs = self._env.get_obs()
        n_line = int(getattr(obs, "n_line", 1) or 1)
        if "line_index" in target:
            return int(target["line_index"]) % n_line
        seed = int(self._seed_obj.seed if self._seed_obj else 0)
        return (seed + current_tick * 7) % n_line

    def snapshot(self) -> dict[str, Any]:  # pragma: no cover
        if self._env is None:
            return {"entities": {}, "tick": self._tick, "horizon": self._horizon}
        obs = self._last_obs
        if obs is None:
            obs = self._env.get_obs()
        entities: dict[str, dict[str, Any]] = {}
        for i in range(obs.n_line):
            entities[f"line_{i}"] = {
                "kind": "line",
                "status": bool(obs.line_status[i]),
                "rho": round(float(obs.rho[i]), 3),
            }
        gen_p = getattr(obs, "gen_p", getattr(obs, "prod_p", []))
        gen_pmax = getattr(obs, "gen_pmax", None)
        gen_pmin = getattr(obs, "gen_pmin", None)
        gen_redispatchable = getattr(obs, "gen_redispatchable", None)
        # Grid2Op exposes maintenance/outage cooldowns via
        # ``time_before_cooldown_line`` arrays; surface them so fog can hide.
        cooldown = getattr(obs, "time_before_cooldown_line", None)
        for i in range(getattr(obs, "n_gen", len(gen_p))):
            output = float(gen_p[i]) if i < len(gen_p) else 0.0
            pmax = (
                float(gen_pmax[i])
                if gen_pmax is not None and i < len(gen_pmax)
                else max(output * 1.5, 50.0)
            )
            pmin = (
                float(gen_pmin[i])
                if gen_pmin is not None and i < len(gen_pmin)
                else 0.0
            )
            committed = output > 0.01
            redisp = (
                bool(gen_redispatchable[i])
                if gen_redispatchable is not None and i < len(gen_redispatchable)
                else True
            )
            # Surface the same fields the synthetic backend produces so
            # heuristic agents (greedy/oracle) and the LLM tool spec see a
            # uniform generator schema across backends.
            entities[f"gen_{i}"] = {
                "kind": "generator",
                "output_mw": round(output, 2),
                "power_max": round(pmax, 2),
                "power_min": round(pmin, 2),
                "committed": committed,
                "redispatchable": redisp,
                "hours_up": 1 if committed else 0,
                "hours_down": 0 if committed else 1,
                "must_run": not redisp and committed,
                "fuel_supply_factor": 1.0,
                "forced_outage_until": -1,  # exposed; fog hides it
            }
        # Surface line cooldowns so fog policy can hide them under storm windows
        if cooldown is not None:
            for i in range(len(cooldown)):
                cd = int(cooldown[i])
                if cd > 0 and f"line_{i}" in entities:
                    entities[f"line_{i}"]["cooldown_ticks"] = cd
        # Surface per-Grid2Op-load demand (load_p[i]) onto the named load
        # entities we previously created from the seed's load_assignments.
        # Grid2Op loads are addressed by integer index; we map them to our
        # named loads in order. If counts differ (typical: 11 Grid2Op
        # loads ↔ 11 stakeholder buckets), this works one-to-one.
        load_p_arr = getattr(obs, "load_p", [])
        for i, (lid, entry) in enumerate(self._loads.items()):
            demand = float(load_p_arr[i]) if i < len(load_p_arr) else 0.0
            entities[lid] = {
                "kind": "load",
                "stakeholder_class": entry["stakeholder_class"],
                "criticality": entry["criticality"],
                "bus_id": entry["bus_id"],
                "current_demand_mw": round(demand, 2),
                "cumulative_shed_mwh": round(
                    self._cumulative_shed_mwh.get(lid, 0.0), 3
                ),
            }
        last = self._tick_records[-1] if self._tick_records else None
        return {
            "entities": entities,
            "totals": {
                "aggregate_demand_mw": last.aggregate_demand_mw if last else 0.0,
                "aggregate_generation_mw": last.aggregate_generation_mw
                if last
                else 0.0,
                "balance_error_mw": last.balance_error_mw if last else 0.0,
                "rho_max": last.rho_max if last else 0.0,
                "n_overloads": last.n_overloads if last else 0,
                "n_disconnected_lines": last.n_disconnected_lines if last else 0,
                "economic_reward": last.economic_reward if last else 0.0,
            },
            "tick": self._tick,
            "horizon": self._horizon,
        }

    def forecast_for(self, horizon: int) -> list[dict[str, Any]]:  # pragma: no cover
        # Use Grid2Op's native ``obs.simulate`` to look ahead one tick at a
        # time. For >1 tick we extrapolate by walking forecasted load+gen;
        # this is honest about what the backend can actually see ahead
        # without committing chronics to disk.
        if self._env is None:
            return []
        obs = self._last_obs or self._env.get_obs()
        if obs is None:
            return []
        out: list[dict[str, Any]] = []
        # First tick: use simulate for an accurate one-step lookahead
        try:
            sim_obs, _r, _d, _info = obs.simulate(self._env.action_space({}))
            sim_demand = float(sim_obs.load_p.sum())
            sim_gen = float(getattr(sim_obs, "gen_p", sim_obs.prod_p).sum())
            out.append(
                {
                    "tick": self._tick + 1,
                    "demand_mw_forecast": round(sim_demand, 2),
                    "gen_mw_forecast": round(sim_gen, 2),
                    "source": "grid2op_simulate",
                }
            )
        except Exception:
            sim_demand = float(obs.load_p.sum())
            out.append(
                {
                    "tick": self._tick + 1,
                    "demand_mw_forecast": round(sim_demand, 2),
                    "source": "fallback_repeat_current",
                }
            )
        # Remaining ticks: flat repeat of last observable demand (we don't
        # ship multi-tick chronics for grid2op envs in v0.1).
        repeat = float(obs.load_p.sum())
        for k in range(2, horizon + 1):
            out.append(
                {
                    "tick": self._tick + k,
                    "demand_mw_forecast": round(repeat, 2),
                    "source": "fallback_repeat_current",
                }
            )
        return out

    def ground_truth_costs(self) -> dict[str, float]:  # pragma: no cover
        # Vocabulary matches the synthetic backend's
        # ``ground_truth_costs`` keys so ``evaluation/counterfactual.py``
        # doesn't have to alias overload/disconnection at scoring time.
        if not self._tick_records:
            return {
                "production_cost": 0.0,
                "startup_cost": 0.0,
                "shed_penalty": 0.0,
                "balance_error_cost": 0.0,
                "reserve_violation_cost": 0.0,
                "safety_violation_cost": 0.0,
            }
        overload_ticks = sum(r.n_overloads for r in self._tick_records)
        disc_ticks = sum(r.n_disconnected_lines for r in self._tick_records)
        production_cost = sum(r.production_cost for r in self._tick_records)
        shed_penalty = sum(r.shed_penalty for r in self._tick_records)
        balance_err = sum(abs(r.balance_error_mw) for r in self._tick_records) * 50.0
        reserve_short = (
            sum(
                max(0.0, r.reserves_required_mw - r.reserves_procured_mw)
                for r in self._tick_records
            )
            * 50.0
        )
        return {
            "production_cost": round(float(production_cost), 2),
            "startup_cost": 0.0,
            "shed_penalty": round(float(shed_penalty), 2),
            "balance_error_cost": round(float(balance_err), 2),
            "reserve_violation_cost": round(float(reserve_short), 2),
            "safety_violation_cost": round(
                float(
                    overload_ticks * self.OVERLOAD_COST_PER_TICK
                    + disc_ticks * self.DISCONNECTION_COST_PER_LINE_TICK
                ),
                2,
            ),
        }

    def scoring_records(self) -> list[dict[str, Any]]:  # pragma: no cover
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
                "rho_max": r.rho_max,
                "n_overloads": r.n_overloads,
                "n_disconnected_lines": r.n_disconnected_lines,
                # v0.3.1 P2 fix: emit the canonical scorer keys that were
                # previously dropped. ``done`` is early-guarded
                # (``r.tick < horizon - 1``) so it flags only a real
                # cascading game-over (Grid2Op terminates the episode mid-
                # horizon; our horizon << chronic length) and never the
                # natural horizon-end tick. Without this, a blackout episode
                # scored ~100 on system_survival. ``n_voltage_violations`` is
                # honestly 0 — the L2RPN backend does not model bus-voltage
                # violations — but the key is now explicit so the scorer
                # reads it deliberately rather than silently defaulting.
                "n_voltage_violations": int(getattr(r, "n_voltage_violations", 0) or 0),
                "done": bool(r.done and r.tick < self._horizon - 1),
                "economic_reward": r.economic_reward,
            }
            for r in self._tick_records
        ]

    def per_load_shed_mwh(self) -> dict[str, float]:  # pragma: no cover
        return dict(self._cumulative_shed_mwh)
