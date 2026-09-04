"""Experimental multi-domain episode runner.

This module deliberately stays outside the formal single-domain runner and
scorer.  It provides a small diagnostic harness for advancing two or more
domain-native environments on one clock while they exchange typed events over
one :class:`core.cascade_bus.CascadeBus`.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from core import (
    Action,
    CascadeBus,
    CascadeEvent,
    POMDPEnvironment,
    StepReturn,
    ToolCall,
    ToolResult,
)

COUPLED_CONSTRUCT_ID = "operational_agency.coupled_diagnostic.v1"
_PARTICIPANT_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_TOOL_SEPARATOR = "__"

EnvironmentFactory = Callable[[CascadeBus], POMDPEnvironment]
EnvironmentInput = POMDPEnvironment | EnvironmentFactory


@dataclass(frozen=True)
class CoupledBudget:
    """Episode-wide budget shared by all participating environments."""

    max_model_turns: int | None = None
    max_tool_calls: int | None = None
    max_cost_units: float | None = None

    def __post_init__(self) -> None:
        for name in ("max_model_turns", "max_tool_calls"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.max_cost_units is not None and self.max_cost_units < 0:
            raise ValueError("max_cost_units must be non-negative")


@dataclass
class CoupledStepReturn:
    """A diagnostic joint step; never consumed by the formal scorer."""

    master_tick: int
    observations: dict[str, dict[str, Any]]
    tool_results: list[ToolResult]
    rewards: dict[str, float]
    done: bool
    environment_results: dict[str, StepReturn]


class CoupledEpisodeRunner:
    """Advance multiple native environments using a shared clock and bus.

    Participant factories receive the runner's ``CascadeBus``.  Already-built
    environments are accepted only when they are all bound to that exact bus;
    replacing an adapter's bus after construction would lose subscriptions and
    is therefore intentionally unsupported.

    Tool names exposed to an agent use ``<participant>__<native_tool>``.  The
    runner validates and removes the namespace before calling an adapter, then
    restores it on returned results.  Participant order is sorted and fixed so
    same-tick cascade delivery is deterministic.
    """

    diagnostic_only = True

    def __init__(
        self,
        participants: Mapping[str, EnvironmentInput],
        *,
        cascade_bus: CascadeBus | None = None,
        budget: CoupledBudget | None = None,
    ) -> None:
        if len(participants) < 2:
            raise ValueError("a coupled episode requires at least two participants")
        self._participant_ids = sorted(participants)
        for participant_id in self._participant_ids:
            if not _PARTICIPANT_RE.fullmatch(participant_id) or _TOOL_SEPARATOR in participant_id:
                raise ValueError(
                    "participant IDs must contain only letters, numbers, '_' or '-' "
                    "and must not contain '__'"
                )

        instantiated = {
            participant_id: value
            for participant_id, value in participants.items()
            if isinstance(value, POMDPEnvironment)
        }
        inferred_buses = [self._environment_bus(env) for env in instantiated.values()]
        if cascade_bus is None and inferred_buses:
            cascade_bus = inferred_buses[0]
        self._cascade_bus = cascade_bus or CascadeBus()
        if any(bus is not self._cascade_bus for bus in inferred_buses):
            raise ValueError("all instantiated environments must use the same CascadeBus")

        self._environments: dict[str, POMDPEnvironment] = {}
        for participant_id in self._participant_ids:
            value = participants[participant_id]
            env = value if isinstance(value, POMDPEnvironment) else value(self._cascade_bus)
            if not isinstance(env, POMDPEnvironment):
                raise TypeError(
                    f"participant factory {participant_id!r} did not return a POMDPEnvironment"
                )
            if self._environment_bus(env) is not self._cascade_bus:
                raise ValueError("all instantiated environments must use the same CascadeBus")
            self._environments[participant_id] = env

        self._budget = budget or CoupledBudget()
        self._master_tick = 0
        self._reset_called = False
        self._failed = False
        self._done_by_participant = dict.fromkeys(self._participant_ids, False)
        self._terminal_ticks: dict[str, int] = {}
        self._model_turns_used = 0
        self._tool_calls_used = 0
        self._cost_units_used = 0.0
        self._reward_by_participant = dict.fromkeys(self._participant_ids, 0.0)
        self._cascade_deliveries: list[dict[str, Any]] = []
        self._clock_alignment: list[dict[str, Any]] = []
        self._observed_events: list[CascadeEvent] = []
        self._cascade_bus.subscribe("*", self._observe_cascade_event)

    @staticmethod
    def _environment_bus(env: POMDPEnvironment) -> CascadeBus:
        bus = getattr(env, "cascade_bus", None)
        if bus is None:
            bus = getattr(env, "_cascade_bus", None)
        if not isinstance(bus, CascadeBus):
            raise ValueError("coupled environments must expose their CascadeBus")
        return bus

    def _observe_cascade_event(self, event: CascadeEvent) -> None:
        self._observed_events.append(event)

    @property
    def cascade_bus(self) -> CascadeBus:
        return self._cascade_bus

    @property
    def environments(self) -> dict[str, POMDPEnvironment]:
        return dict(self._environments)

    @property
    def master_tick(self) -> int:
        return self._master_tick

    def reset(
        self,
        scenarios: Mapping[str, dict[str, Any]],
        seeds: Mapping[str, int] | int,
    ) -> dict[str, dict[str, Any]]:
        """Reset all participants without clearing their bus subscriptions."""
        self._require_exact_participants(scenarios, label="scenario")
        if isinstance(seeds, Mapping):
            self._require_exact_participants(seeds, label="seed")
            seed_by_participant = {key: int(value) for key, value in seeds.items()}
        else:
            seed_by_participant = {
                participant_id: int(seeds) for participant_id in self._participant_ids
            }

        self._master_tick = 0
        self._failed = False
        self._done_by_participant = dict.fromkeys(self._participant_ids, False)
        self._terminal_ticks = {}
        self._model_turns_used = 0
        self._tool_calls_used = 0
        self._cost_units_used = 0.0
        self._reward_by_participant = dict.fromkeys(self._participant_ids, 0.0)
        self._cascade_deliveries = []
        self._clock_alignment = []
        self._observed_events = []

        observations: dict[str, dict[str, Any]] = {}
        for participant_id in self._participant_ids:
            observations[participant_id] = self._environments[participant_id].reset(
                copy.deepcopy(scenarios[participant_id]),
                seed_by_participant[participant_id],
            )
        ticks = self._ticks()
        if any(tick != 0 for tick in ticks.values()):
            raise RuntimeError(f"coupled reset must leave every environment at tick 0: {ticks}")
        self._reset_called = True
        return observations

    def _require_exact_participants(self, values: Mapping[str, Any], *, label: str) -> None:
        received = set(values)
        expected = set(self._participant_ids)
        if received != expected:
            missing = sorted(expected - received)
            extra = sorted(received - expected)
            raise ValueError(
                f"{label} participants do not match runner; missing={missing}, extra={extra}"
            )

    def get_tool_specs(self) -> list[dict[str, Any]]:
        """Return detached OpenAI schemas with deterministic namespaces."""
        schemas: list[dict[str, Any]] = []
        seen: set[str] = set()
        for participant_id in self._participant_ids:
            for native_schema in self._environments[participant_id].get_tool_specs():
                schema = copy.deepcopy(native_schema)
                function = schema.get("function")
                if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                    raise ValueError(f"invalid tool schema from participant {participant_id!r}")
                native_name = function["name"]
                coupled_name = self._coupled_tool_name(participant_id, native_name)
                if coupled_name in seen:
                    raise ValueError(f"duplicate coupled tool name: {coupled_name}")
                seen.add(coupled_name)
                function["name"] = coupled_name
                schema["x-coupled-participant"] = participant_id
                schema["x-native-tool-name"] = native_name
                schemas.append(schema)
        return schemas

    @staticmethod
    def _coupled_tool_name(participant_id: str, native_name: str) -> str:
        name = f"{participant_id}{_TOOL_SEPARATOR}{native_name}"
        if len(name) > 64:
            raise ValueError(f"coupled tool name exceeds 64 characters: {name}")
        return name

    def step(self, action: Action, *, model_turns: int = 1) -> CoupledStepReturn:
        """Advance every non-terminal participant exactly one native tick."""
        if not self._reset_called:
            raise RuntimeError("reset must be called before step")
        if self._failed:
            raise RuntimeError("coupled runner is unusable after a clock or budget failure")
        if all(self._done_by_participant.values()):
            raise RuntimeError("coupled episode is already terminal")
        if model_turns < 0:
            raise ValueError("model_turns must be non-negative")

        actions_by_participant = self._route_action(action)
        attempted_calls = len(action.tool_calls)
        attempted_cost = self._declared_action_cost(action)
        self._preflight_budget(
            model_turns=model_turns,
            tool_calls=attempted_calls,
            cost_units=attempted_cost,
        )

        active = [
            participant_id
            for participant_id in self._participant_ids
            if not self._done_by_participant[participant_id]
        ]
        ticks_before = self._ticks()
        if any(ticks_before[key] != self._master_tick for key in active):
            self._record_clock_alignment(
                ticks_before=ticks_before,
                ticks_after=ticks_before,
                active=active,
                aligned=False,
            )
            self._failed = True
            raise RuntimeError(f"coupled clock drift before step: {ticks_before}")

        environment_results: dict[str, StepReturn] = {}
        aggregate_results: list[ToolResult] = []
        rewards: dict[str, float] = {}
        observations: dict[str, dict[str, Any]] = {}
        observed_before = len(self._observed_events)
        failed_before = self._cascade_bus.failed_deliveries
        self._cascade_bus.begin_delivery_batch()
        try:
            for participant_id in active:
                result = self._environments[participant_id].step(
                    actions_by_participant[participant_id]
                )
                environment_results[participant_id] = result
                observations[participant_id] = result.observation
                rewards[participant_id] = float(result.reward)
                self._reward_by_participant[participant_id] += float(result.reward)
                aggregate_results.extend(
                    replace(
                        tool_result,
                        name=self._coupled_tool_name(participant_id, tool_result.name),
                    )
                    for tool_result in result.tool_results
                )
                if result.done:
                    self._done_by_participant[participant_id] = True
                    self._terminal_ticks[participant_id] = int(
                        self._environments[participant_id].tick
                    )
        finally:
            self._cascade_bus.end_delivery_batch()
        self._record_cascade_deliveries(
            participant_id="joint_tick_barrier",
            observed_before=observed_before,
            subscriber_failures=(self._cascade_bus.failed_deliveries - failed_before),
        )

        ticks_after = self._ticks()
        expected_tick = self._master_tick + 1
        aligned = all(ticks_after[key] == expected_tick for key in active)
        self._record_clock_alignment(
            ticks_before=ticks_before,
            ticks_after=ticks_after,
            active=active,
            aligned=aligned,
        )
        if not aligned:
            self._failed = True
            raise RuntimeError(f"coupled clock drift after step: {ticks_after}")

        actual_cost = sum(float(row.cost_units) for row in aggregate_results)
        next_cost_total = self._cost_units_used + actual_cost
        if (
            self._budget.max_cost_units is not None
            and next_cost_total > self._budget.max_cost_units
        ):
            # Declared tool costs are reserved before any backend advances.
            # A larger runtime charge is therefore a contract violation, not
            # permission to commit an over-budget accounting state.
            self._failed = True
            raise RuntimeError("actual joint cost-unit budget exceeded declared reservation")

        self._master_tick = expected_tick
        self._model_turns_used += model_turns
        self._tool_calls_used += attempted_calls
        self._cost_units_used = next_cost_total
        done = all(self._done_by_participant.values())
        return CoupledStepReturn(
            master_tick=self._master_tick,
            observations=observations,
            tool_results=aggregate_results,
            rewards=rewards,
            done=done,
            environment_results=environment_results,
        )

    def _route_action(self, action: Action) -> dict[str, Action]:
        known_tools = {schema["function"]["name"] for schema in self.get_tool_specs()}
        calls: dict[str, list[ToolCall]] = {
            participant_id: [] for participant_id in self._participant_ids
        }
        for call in action.tool_calls:
            if call.name not in known_tools:
                raise ValueError(f"unknown coupled tool: {call.name}")
            participant_id, native_name = call.name.split(_TOOL_SEPARATOR, 1)
            if self._done_by_participant[participant_id]:
                raise ValueError(f"cannot call a tool on terminal participant: {participant_id}")
            calls[participant_id].append(replace(call, name=native_name))
        return {
            participant_id: Action(
                tool_calls=participant_calls,
                dominant=action.dominant,
                assistant_text=action.assistant_text,
                rationale=action.rationale,
            )
            for participant_id, participant_calls in calls.items()
        }

    def _declared_action_cost(self, action: Action) -> float:
        costs = {
            schema["function"]["name"]: schema.get("x-cost-units")
            for schema in self.get_tool_specs()
        }
        total = 0.0
        for call in action.tool_calls:
            value = costs.get(call.name)
            if not isinstance(value, (int, float)):
                raise ValueError(f"coupled tool lacks declared cost units: {call.name}")
            total += float(value)
        return total

    def _preflight_budget(self, *, model_turns: int, tool_calls: int, cost_units: float) -> None:
        if (
            self._budget.max_model_turns is not None
            and self._model_turns_used + model_turns > self._budget.max_model_turns
        ):
            raise ValueError("joint model-turn budget exceeded")
        if (
            self._budget.max_tool_calls is not None
            and self._tool_calls_used + tool_calls > self._budget.max_tool_calls
        ):
            raise ValueError("joint tool-call budget exceeded")
        if (
            self._budget.max_cost_units is not None
            and self._cost_units_used + cost_units > self._budget.max_cost_units
        ):
            raise ValueError("joint cost-unit budget exceeded")

    def _record_cascade_deliveries(
        self,
        *,
        participant_id: str,
        observed_before: int,
        subscriber_failures: int,
    ) -> None:
        observed = self._observed_events[observed_before:]
        for event in observed:
            row = event.to_dict()
            row.update(
                {
                    "master_tick": self._master_tick,
                    "published_during_participant": participant_id,
                    "audit_subscriber_received": True,
                    "subscriber_failures_during_step": subscriber_failures,
                }
            )
            self._cascade_deliveries.append(row)

    def _ticks(self) -> dict[str, int]:
        return {participant_id: int(env.tick) for participant_id, env in self._environments.items()}

    def _record_clock_alignment(
        self,
        *,
        ticks_before: dict[str, int],
        ticks_after: dict[str, int],
        active: list[str],
        aligned: bool,
    ) -> None:
        self._clock_alignment.append(
            {
                "master_tick_before": self._master_tick,
                "master_tick_after": self._master_tick + 1,
                "active_participants": list(active),
                "ticks_before": dict(sorted(ticks_before.items())),
                "ticks_after": dict(sorted(ticks_after.items())),
                "aligned": aligned,
            }
        )

    def diagnostic_report(self) -> dict[str, Any]:
        """Return JSON-safe evidence; it has no formal leaderboard status."""
        exhausted = any(
            (
                limit is not None and used >= limit
                for used, limit in (
                    (self._model_turns_used, self._budget.max_model_turns),
                    (self._tool_calls_used, self._budget.max_tool_calls),
                    (self._cost_units_used, self._budget.max_cost_units),
                )
            )
        )
        return {
            "construct_id": COUPLED_CONSTRUCT_ID,
            "diagnostic_only": True,
            "formal_leaderboard_eligible": False,
            "participant_order": list(self._participant_ids),
            "master_tick": self._master_tick,
            "budget": {
                "model_turns_used": self._model_turns_used,
                "tool_calls_used": self._tool_calls_used,
                "cost_units_used": self._cost_units_used,
                "max_model_turns": self._budget.max_model_turns,
                "max_tool_calls": self._budget.max_tool_calls,
                "max_cost_units": self._budget.max_cost_units,
                "exhausted": exhausted,
            },
            "cascade_deliveries": copy.deepcopy(self._cascade_deliveries),
            "cascade_failed_deliveries": self._cascade_bus.failed_deliveries,
            "clock_alignment": copy.deepcopy(self._clock_alignment),
            "joint_outcome": {
                "cumulative_reward": sum(self._reward_by_participant.values()),
                "reward_by_participant": dict(sorted(self._reward_by_participant.items())),
                "done_by_participant": dict(sorted(self._done_by_participant.items())),
            },
            "terminal_parity": {
                "terminal_ticks": dict(sorted(self._terminal_ticks.items())),
                "all_terminal": all(self._done_by_participant.values()),
                "same_terminal_tick": (
                    len(self._terminal_ticks) == len(self._participant_ids)
                    and len(set(self._terminal_ticks.values())) == 1
                ),
            },
        }

    def close(self) -> None:
        """Close every native backend, attempting all participants."""
        errors: list[str] = []
        for participant_id in self._participant_ids:
            try:
                self._environments[participant_id].close()
            except Exception as exc:  # pragma: no cover - backend-specific
                errors.append(f"{participant_id}: {type(exc).__name__}: {exc}")
        if errors:
            raise RuntimeError("failed to close coupled environments: " + "; ".join(errors))


__all__ = [
    "COUPLED_CONSTRUCT_ID",
    "CoupledBudget",
    "CoupledEpisodeRunner",
    "CoupledStepReturn",
]
