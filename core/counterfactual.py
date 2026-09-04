"""
core.counterfactual — Counterfactual replay engine.

Computes ``prevented_loss = replay_no_action_cost − actual_cost`` by
re-running the exact same seeded scenario with the agent's actions removed
(or selectively masked) and measuring how much worse the outcome gets.

This is the single biggest scoring upgrade OPERATE introduces over
DispatchBench: instead of inferring foresight indirectly from prediction
accuracy, we *directly measure* what would have happened if the agent had
not acted.

Contract requirements for the host environment:

- ``POMDPEnvironment.reset(scenario_config, seed)`` must be deterministic.
- All RNG sources must derive from ``seed`` (or be saved/restored across
  ``reset`` calls).
- ``ground_truth()`` must include the numeric ``cost_components`` the
  scorer aggregates into a single counterfactual_loss.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from .pomdp import Action, ToolCall
from .pomdp_env import POMDPEnvironment


@dataclass
class CounterfactualReport:
    actual_cost: float
    counterfactual_cost: float
    prevented_loss: float  # cf_cost - actual_cost; positive ⇒ agent helped
    actual_components: dict[str, float] = field(default_factory=dict)
    counterfactual_components: dict[str, float] = field(default_factory=dict)
    masking_policy: str = "wait_only"
    notes: str = ""
    # v0.2.1: True iff the counterfactual baseline produced an
    # interpretable cost number. When the wait_only replay itself
    # crashes or accumulates near-zero cost (e.g. game-over at tick 0),
    # `normalized_prevention` becomes meaningless and downstream
    # scoring marks the dimension applicable=False.
    applicable: bool = True
    # Machine-readable classification of WHY `applicable=False`, mirroring
    # the `leaderboard_eligibility.diagnostic_cells[].reason.code` vocabulary
    # style used at the release-manifest level (e.g.
    # "baseline_gap_zero_agent_differentiation"). Prior to this field the
    # only opt-out signal was the free-text `notes`, which a downstream
    # consumer cannot branch on programmatically. Empty when `applicable`
    # is True. See `COUNTERFACTUAL_REASON_CODES` for the closed set of
    # values this module assigns.
    reason_code: str = ""
    # P1-3: per-action prevented_loss attribution. Each entry is
    # ``{tick, call_index, call_id, idempotency_key, tool_name,
    # args_signature, marginal_prevented_loss}`` — the outcome delta from
    # masking JUST that one state-changing tool call while leaving all sibling
    # calls and other ticks actual. Empty by default; populated only when
    # ``run_counterfactual(..., per_action=True)``. The per-episode
    # ``prevented_loss`` (all-masked baseline) is unchanged.
    per_action: list[dict[str, Any]] = field(default_factory=list)
    # True iff ``per_action`` was truncated at the requested attribution cap.
    # Lets a
    # researcher detect incomplete attribution without comparing against the
    # trajectory action count.
    per_action_capped: bool = False
    # Machine-readable coverage for the optional per-call attribution pass.
    # Episode-level counterfactual applicability remains independent: a failed
    # single-call mask invalidates only the attribution claim, never the
    # all-masked wait baseline above.
    per_action_status: str = "not_requested"
    per_action_expected: int = 0
    per_action_attempted: int = 0
    per_action_completed: int = 0
    per_action_failures: list[dict[str, Any]] = field(default_factory=list)
    # Optional interaction attribution for repeated, substitutable controls.
    # Each row masks every state-changing call with the same native tool and
    # argument signature.  This detects response strategies whose individual
    # calls have zero leave-one-out effect because a sibling call compensates.
    per_action_groups: list[dict[str, Any]] = field(default_factory=list)
    per_action_group_status: str = "not_requested"
    per_action_group_expected: int = 0
    per_action_group_attempted: int = 0
    per_action_group_completed: int = 0
    per_action_group_failures: list[dict[str, Any]] = field(default_factory=list)
    # Internal replay state used by offline calibration to apply the same
    # task-success contract as the live runner. It is intentionally omitted
    # from ``to_dict`` so hidden backend state is never published.
    actual_ground_truth: dict[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    counterfactual_ground_truth: dict[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    @property
    def normalized_prevention(self) -> float:
        """0 = agent matched the do-nothing baseline; 1 = agent fully averted
        all marginal loss the do-nothing baseline incurred. Returns 0.0
        (not 1.0) when the counterfactual itself produced no meaningful
        cost — a "perfect prevention" of nothing is not credit-worthy.
        """
        if not self.applicable:
            return 0.0
        if not all(
            math.isfinite(value)
            for value in (
                self.actual_cost,
                self.counterfactual_cost,
                self.prevented_loss,
            )
        ):
            return 0.0
        if self.actual_cost < 0 or self.counterfactual_cost < 0:
            return 0.0
        if self.counterfactual_cost <= 0:
            return 0.0
        ratio = self.prevented_loss / self.counterfactual_cost
        return max(0.0, min(1.0, ratio))

    def to_dict(self) -> dict[str, Any]:
        return {
            "actual_cost": _finite_json_cost(self.actual_cost),
            "counterfactual_cost": _finite_json_cost(self.counterfactual_cost),
            "prevented_loss": _finite_json_cost(self.prevented_loss),
            "actual_components": _finite_json_components(self.actual_components),
            "counterfactual_components": _finite_json_components(
                self.counterfactual_components
            ),
            "masking_policy": self.masking_policy,
            "normalized_prevention": round(self.normalized_prevention, 4),
            "applicable": bool(self.applicable),
            "reason_code": self.reason_code,
            "notes": self.notes,
            "per_action": list(self.per_action),
            "per_action_capped": bool(self.per_action_capped),
            "per_action_status": self.per_action_status,
            "per_action_expected": int(self.per_action_expected),
            "per_action_attempted": int(self.per_action_attempted),
            "per_action_completed": int(self.per_action_completed),
            "per_action_failures": list(self.per_action_failures),
            "per_action_groups": list(self.per_action_groups),
            "per_action_group_status": self.per_action_group_status,
            "per_action_group_expected": int(self.per_action_group_expected),
            "per_action_group_attempted": int(self.per_action_group_attempted),
            "per_action_group_completed": int(self.per_action_group_completed),
            "per_action_group_failures": list(self.per_action_group_failures),
        }


# Closed vocabulary of `CounterfactualReport.reason_code` values. Extend this
# set (and document the new member) rather than inventing ad-hoc strings at
# call sites — downstream consumers (audit reports, leaderboard eligibility)
# branch on exact string match.
REASON_CODE_CF_BASELINE_UNUSABLE = "cf_baseline_produced_no_usable_cost"
REASON_CODE_BACKEND_OPTED_OUT = "backend_declared_supports_counterfactual_false"
REASON_CODE_REPLAY_SCHEDULE_UNPROVEN = "counterfactual_replay_schedule_unproven"
COUNTERFACTUAL_REASON_CODES: frozenset[str] = frozenset(
    {
        REASON_CODE_CF_BASELINE_UNUSABLE,
        REASON_CODE_BACKEND_OPTED_OUT,
        REASON_CODE_REPLAY_SCHEDULE_UNPROVEN,
    }
)


def backend_opt_out_report(*, masking_policy: str, backend_domain: str = "") -> CounterfactualReport:
    """Build the `applicable=False` report a caller returns when
    `env.supports_counterfactual()` is False, instead of running (and
    wasting compute on) a replay the backend has declared unsupported.

    This is the machine-readable opt-out red line #5 requires: a scenario
    that cannot support masked-action replay must say so via a structured
    field, not just skip silently or bury the reason in prose.
    """
    return CounterfactualReport(
        actual_cost=0.0,
        counterfactual_cost=0.0,
        prevented_loss=0.0,
        masking_policy=masking_policy,
        applicable=False,
        reason_code=REASON_CODE_BACKEND_OPTED_OUT,
        notes=(
            f"{backend_domain or 'backend'}.supports_counterfactual() is False; "
            "counterfactual replay skipped rather than run against an "
            "unsupported environment"
        ),
    )


MaskingPolicy = Callable[[Action, int], Action]


@dataclass(frozen=True)
class _RecordedScheduleReplay:
    """Private outcome for a replay pinned to the live action trace length."""

    ground_truth: dict[str, Any]
    notes: list[str]
    steps_executed: int
    recorded_steps: int
    horizon: int
    final_step_done: bool

    @property
    def completed_recorded_schedule(self) -> bool:
        return self.steps_executed == self.recorded_steps

    @property
    def actual_schedule_is_proven(self) -> bool:
        """The live trace ended at a terminal step or the finite horizon."""
        return bool(
            self.completed_recorded_schedule
            and self.recorded_steps > 0
            and (
                self.final_step_done
                or self.recorded_steps >= self.horizon
            )
        )


def wait_only_policy(action: Action, _tick: int) -> Action:
    """Replace every action with a single ``wait`` call.

    The strictest counterfactual baseline: the agent does **nothing** at
    every tick. Used to quantify how much value a state-changing agent
    actually adds versus the ``wait_only`` baseline. Note that with this
    policy ``information_efficiency`` collapses to whatever investigation
    evidence the env auto-emits at reset (typically zero), so any
    cross-policy comparison of that dimension must use
    ``keep_investigations_policy`` instead — see below.
    """
    if action.is_noop:
        return action
    return Action(tool_calls=[ToolCall(name="wait")], dominant="wait")


# P1-3: per-action prevented_loss attribution cap. Masking one tool call at a
# time is O(k) replays over the episode; on job_shop episodes with hundreds
# of ``dispatch_job_operation`` calls this would dominate runtime. The cap
# bounds attribution to the first 20 state-changing calls by default. A
# researcher who needs full coverage can pass ``per_action_cap=None`` — the
# per-episode ``prevented_loss`` (all-masked baseline) is unaffected.
_PER_ACTION_CAP: int = 20


def make_mask_single_action_policy(
    target_tick: int,
    target_call_index: int | None = None,
) -> MaskingPolicy:
    """Return a policy masking one recorded action or one of its tool calls.

    Used by :func:`run_counterfactual` (with ``per_action=True``) for
    per-call prevented-loss attribution. When ``target_call_index`` is given,
    only that call is removed and sibling calls at the same tick remain. The
    legacy one-argument form still replaces the whole target-tick action with
    ``wait``.

    Implementation note: the replay engine invokes the policy as
    ``masking_policy(base_action, tick_idx)`` where ``tick_idx`` is the
    per-tick loop counter — i.e. the position of the action in
    ``actual_actions``. So matching on ``tick_idx == target_tick`` is
    sufficient and avoids the closure-captured counter the original brief
    sketch proposed (which would be fragile across multiple replay
    invocations).
    """
    def policy(action: Action, tick: int) -> Action:
        if tick != target_tick:
            return action
        if target_call_index is None:
            return wait_only_policy(action, tick)
        if not 0 <= target_call_index < len(action.tool_calls):
            return action
        remaining_calls = [
            call
            for index, call in enumerate(action.tool_calls)
            if index != target_call_index
        ]
        if not remaining_calls:
            remaining_calls = [ToolCall(name="wait")]
        return Action(
            tool_calls=remaining_calls,
            dominant=remaining_calls[0].name,
            assistant_text=action.assistant_text,
            rationale=action.rationale,
        )

    return policy


def make_mask_action_group_policy(
    targets: Iterable[tuple[int, int]],
) -> MaskingPolicy:
    """Return a policy masking an exact set of ``(tick, call_index)`` calls."""
    target_set = {(int(tick), int(call_index)) for tick, call_index in targets}

    def policy(action: Action, tick: int) -> Action:
        remaining_calls = [
            call
            for call_index, call in enumerate(action.tool_calls)
            if (tick, call_index) not in target_set
        ]
        if not remaining_calls:
            remaining_calls = [ToolCall(name="wait")]
        return Action(
            tool_calls=remaining_calls,
            dominant=remaining_calls[0].name,
            assistant_text=action.assistant_text,
            rationale=action.rationale,
        )

    return policy


# Legacy power-grid read-only tool set. Retained ONLY as a fallback for
# callers that cannot introspect the env's tool registry (e.g. a direct call
# to :func:`keep_investigations_policy` without an env). New code should use
# :func:`make_keep_investigations_policy` / :func:`keep_investigations_policy_for_env`
# so the kept set is derived generically from each domain's registered
# non-state-changing tools (Hard Red Line: ``core/`` stays backend-agnostic).
_LEGACY_KEEP_INVESTIGATIONS_TOOLS: frozenset[str] = frozenset(
    {
        "wait",
        "noop",
        "query_grid_state",
        "query_chronics_window",
        "forecast_query",
        "investigate_substation",
        "stakeholder_query",
    }
)


def make_keep_investigations_policy(
    readonly_names: Iterable[str] | None = None,
) -> MaskingPolicy:
    """Build a keep-investigations masking policy.

    The policy keeps only read-only (investigative) tool calls and drops
    state-changing actions, isolating the value of *acting* from the value of
    *knowing*. ``readonly_names`` should be the env's registered
    non-state-changing tool names (see
    :meth:`core.pomdp_env.POMDPEnvironment.readonly_tool_names`); ``wait`` and
    ``noop`` are always kept. When ``readonly_names`` is ``None`` the policy
    falls back to the legacy power-grid read-only set for backward
    compatibility.
    """
    if readonly_names is None:
        keep = set(_LEGACY_KEEP_INVESTIGATIONS_TOOLS)
    else:
        keep = set(readonly_names) | {"wait", "noop"}

    def _policy(action: Action, _tick: int) -> Action:
        filtered = [c for c in action.tool_calls if c.name in keep]
        if not filtered:
            filtered = [ToolCall(name="wait")]
        return Action(tool_calls=filtered, dominant=filtered[0].name)

    return _policy


def keep_investigations_policy_for_env(env: POMDPEnvironment) -> MaskingPolicy:
    """Domain-aware keep-investigations policy derived from an env's registry.

    ``env`` should already be ``reset`` so its tool registry is populated.
    Falls back to the legacy set when the env cannot report its read-only
    tools.
    """
    names: set[str] | None = None
    getter = getattr(env, "readonly_tool_names", None)
    if callable(getter):
        try:
            names = getter()
        except Exception:
            names = None
    return make_keep_investigations_policy(names)


def keep_investigations_policy(action: Action, _tick: int) -> Action:
    """Keep only read-only investigative tools, drop state-changing ones.

    A weaker baseline than :func:`wait_only_policy`: the agent is
    allowed to **observe** the world (``query_grid_state``,
    ``forecast_query``, etc.) but cannot affect it. This isolates the
    economic / safety value of *acting* from the value of *knowing*.

    Interaction note (v0.2.2 P2-6): :func:`wait_only_policy` replaces
    *every* tool call with a bare ``wait``, which means the
    ``information_efficiency`` dimension under that policy reflects
    only whatever investigation evidence the env emits unconditionally
    at reset. If the audit needs to attribute information_efficiency
    fairly to the agent's investigation behaviour (rather than to its
    inaction), pair the actual run with :func:`keep_investigations_policy`
    so the counterfactual preserves the agent's investigations and
    only its state-changing actions are masked. Mixing the two policies
    in a single comparison cell will produce systematically biased
    information_efficiency deltas — always pick one per cell.

    .. note::
        This bare function is the **legacy** entry point and uses the
        power-grid read-only set. Domain-agnostic callers should prefer
        :func:`keep_investigations_policy_for_env` (or
        :func:`make_keep_investigations_policy` with env-derived names) so the
        kept tool set matches the actual domain's registered read-only tools.
    """
    return make_keep_investigations_policy(None)(action, _tick)


def _replay_env_with_actions(
    env: POMDPEnvironment,
    actions: list[Action],
    min_steps: int,
    *,
    masking_policy: MaskingPolicy | None = None,
    catch_exceptions: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    """Step an already-reset ``env`` through ``actions``, then return its
    ``ground_truth()`` snapshot.

    Runs for ``max(min_steps, len(actions))`` ticks, padding with a bare
    ``wait`` action beyond ``len(actions)`` — every ``masking_policy`` must
    accept this padded input. When ``masking_policy`` is given, each tick's
    base action (real or padded) is passed through it before stepping; this
    is how :func:`run_counterfactual`'s masked replay works. When
    ``catch_exceptions`` is True, an exception from ``env.step`` is caught,
    recorded in the returned notes, and treated as end-of-episode (this is
    the defensive behaviour ``run_counterfactual`` relies on for masked
    baselines; the plain action-replay passes don't opt into it since a
    crash there should propagate like it always has).

    Returns ``(ground_truth, notes)``.
    """
    n_steps = max(min_steps, len(actions))
    notes: list[str] = []
    for tick_idx in range(n_steps):
        if tick_idx < len(actions):
            base_action = actions[tick_idx]
        else:
            base_action = Action(tool_calls=[ToolCall(name="wait")], dominant="wait")
        step_action = (
            masking_policy(base_action, tick_idx) if masking_policy else base_action
        )
        if catch_exceptions:
            try:
                result = env.step(step_action)
            except Exception as exc:  # pragma: no cover - defensive
                notes.append(
                    f"cf step at tick {tick_idx} raised {type(exc).__name__}: {exc}"
                )
                break
        else:
            result = env.step(step_action)
        if result.done:
            break
    return env.ground_truth(), notes


def _replay_recorded_schedule(
    env: POMDPEnvironment,
    actions: list[Action],
    *,
    masking_policy: MaskingPolicy | None = None,
) -> _RecordedScheduleReplay:
    """Replay exactly the finite tick window recorded by the live runner.

    The live runner may deliberately continue after ``done=True`` to give an
    agent one response tick for a newly visible interrupt or to drain a
    delayed action through its due tick.  Therefore ``done`` is evidence that
    the recorded schedule may end, not an unconditional instruction to drop
    later actions already present in the trajectory.

    Both actual and masked branches use this identical recorded window.  No
    waits are synthesized beyond it: if the actual trace neither reaches the
    environment horizon nor ends on ``done=True``, its terminal schedule is
    unproven and the caller must fail closed.
    """
    notes: list[str] = []
    steps_executed = 0
    final_step_done = False
    for tick_idx, base_action in enumerate(actions):
        step_action = (
            masking_policy(base_action, tick_idx)
            if masking_policy is not None
            else base_action
        )
        try:
            result = env.step(step_action)
        except Exception as exc:  # pragma: no cover - backend-specific guard
            notes.append(
                "recorded schedule step "
                f"{tick_idx} raised {type(exc).__name__}: {exc}"
            )
            break
        steps_executed += 1
        final_step_done = bool(result.done)
    return _RecordedScheduleReplay(
        ground_truth=env.ground_truth(),
        notes=notes,
        steps_executed=steps_executed,
        recorded_steps=len(actions),
        horizon=max(1, int(getattr(env, "horizon", len(actions) or 1))),
        final_step_done=final_step_done,
    )


def _replay_recorded_and_extract_costs_and_truth(
    env_factory: Callable[[], POMDPEnvironment],
    scenario_config: dict[str, Any],
    seed: int,
    actions: list[Action],
    cost_extractor: Callable[[dict[str, Any]], dict[str, float]],
    *,
    masking_policy: MaskingPolicy | None = None,
) -> tuple[dict[str, float], _RecordedScheduleReplay]:
    """Run one exact recorded-window replay and always close its backend."""
    env = env_factory()
    try:
        env.reset(copy.deepcopy(scenario_config), seed)
        replay = _replay_recorded_schedule(
            env,
            actions,
            masking_policy=masking_policy,
        )
        return cost_extractor(replay.ground_truth), replay
    finally:
        env.close()


def _replay_and_extract_costs(
    env_factory: Callable[[], POMDPEnvironment],
    scenario_config: dict[str, Any],
    seed: int,
    actions: list[Action],
    cost_extractor: Callable[[dict[str, Any]], dict[str, float]],
    *,
    horizon_floor: int | None = None,
    masking_policy: MaskingPolicy | None = None,
    catch_exceptions: bool = False,
) -> tuple[dict[str, float], list[str]]:
    components, notes, _ground_truth = _replay_and_extract_costs_and_truth(
        env_factory,
        scenario_config,
        seed,
        actions,
        cost_extractor,
        horizon_floor=horizon_floor,
        masking_policy=masking_policy,
        catch_exceptions=catch_exceptions,
    )
    return components, notes


def _replay_and_extract_costs_and_truth(
    env_factory: Callable[[], POMDPEnvironment],
    scenario_config: dict[str, Any],
    seed: int,
    actions: list[Action],
    cost_extractor: Callable[[dict[str, Any]], dict[str, float]],
    *,
    horizon_floor: int | None = None,
    masking_policy: MaskingPolicy | None = None,
    catch_exceptions: bool = False,
) -> tuple[dict[str, float], list[str], dict[str, Any]]:
    """Replay once, extract costs, and retain private task-contract state.

    ``horizon_floor``, when given, makes the replay run for at least
    ``max(env.horizon, horizon_floor, len(actions))`` ticks (v0.2.1 fix: the
    counterfactual baseline must cover the FULL scenario horizon, not just
    ``len(actions)``, or a crashed agent gets falsely credited with
    preventing the ticks it never lived to see). When omitted, the replay
    runs for exactly ``len(actions)`` ticks — the plain actual-action
    replay pass both public functions start with.
    """
    env = env_factory()
    try:
        env.reset(copy.deepcopy(scenario_config), seed)
        if horizon_floor is None:
            min_steps = len(actions)
        else:
            full_horizon = int(getattr(env, "horizon", horizon_floor or 1))
            min_steps = max(full_horizon, horizon_floor, len(actions))
        ground_truth, notes = _replay_env_with_actions(
            env,
            actions,
            min_steps,
            masking_policy=masking_policy,
            catch_exceptions=catch_exceptions,
        )
        return cost_extractor(ground_truth), notes, ground_truth
    finally:
        env.close()


def run_counterfactual(
    env_factory: Callable[[], POMDPEnvironment],
    scenario_config: dict[str, Any],
    seed: int,
    actual_actions: list[Action],
    cost_extractor: Callable[[dict[str, Any]], dict[str, float]],
    masking_policy: MaskingPolicy = wait_only_policy,
    masking_label: str = "wait_only",
    per_action: bool = False,
    per_action_cap: int | None = _PER_ACTION_CAP,
    per_action_groups: bool = False,
    per_action_group_cap: int | None = _PER_ACTION_CAP,
    readonly_tool_names: set[str] | None = None,
) -> CounterfactualReport:
    """Run a masked-action replay and produce a CounterfactualReport.

    Parameters
    ----------
    env_factory:
        A zero-arg callable that returns a *fresh* environment instance.
        Re-instantiating (rather than re-using) avoids RNG-state leakage
        between actual and counterfactual runs.
    scenario_config:
        The same scenario config used for the actual run.
    seed:
        The same seed used for the actual run; ensures the same fault
        injections, chronics, opponent attacks, and forecast noise.
    actual_actions:
        Sequence of Action objects the agent executed in the actual run.
        Length must equal the env's per-tick action count for the realized
        episode (i.e., one entry per tick that ran).
    cost_extractor:
        Function that maps an env ``ground_truth()`` snapshot to a dict of
        cost components (e.g., ``{"shed_energy_mwh": 12.3, "violation_ticks":
        4, "casualty_proxy": 2}``). The scorer sums positive numeric values
        in this dict to derive a single scalar cost.
    per_action:
        When True, populate ``CounterfactualReport.per_action`` with one
        entry per state-changing tool call (capped at 20 by default) giving
        ``{tick, call_index, call_id, idempotency_key, tool_name, args_signature,
        marginal_prevented_loss}`` — the outcome delta from masking JUST
        that one call while leaving its sibling calls and all other ticks
        actual. Off by default for backward compatibility; the per-episode
        ``prevented_loss`` is unchanged either way (it is the all-masked
        baseline, computed in Pass 2 below).
    readonly_tool_names:
        Registered non-state-changing tool names.  When supplied, per-action
        attribution skips investigation and planning-only actions instead of
        treating every non-wait tool as a physical control.
    per_action_cap:
        Maximum state-changing actions to replay. The default bounds online
        cost at 20; pass ``None`` for complete offline calibration.
    per_action_groups:
        When True, additionally mask repeated state-changing calls as groups
        keyed by exact native tool and argument signature.  This captures
        substitutable-control interaction effects without changing per-call
        attribution.
    per_action_group_cap:
        Maximum repeated action groups to replay. ``None`` requests all groups.
    """
    # ── Pass 1: replay the actual actions to capture the actual cost ──
    # Pin both branches to the finite tick sequence the live runner recorded.
    # This preserves terminal response ticks and pending-action drains while
    # avoiding fabricated horizon-padding ticks after an incomplete trace.
    actual_components, actual_replay = _replay_recorded_and_extract_costs_and_truth(
        env_factory,
        scenario_config,
        seed,
        actual_actions,
        cost_extractor,
    )

    # ── Pass 2: same scenario, same seed, but masked actions ──
    # Apply the intervention exactly once. If it leaves the complete action
    # trace structurally unchanged, this is the identity intervention
    # ``do(A := A)``: another native replay adds no causal information and can
    # introduce sidecar startup jitter. The outer determinism gate still runs
    # independent episodes.
    masked_actions = [
        masking_policy(copy.deepcopy(action), tick)
        for tick, action in enumerate(actual_actions)
    ]
    identity_mask = masked_actions == actual_actions
    if identity_mask:
        cf_components = dict(actual_components)
        counterfactual_replay = actual_replay
    else:
        cf_components, counterfactual_replay = (
            _replay_recorded_and_extract_costs_and_truth(
                env_factory,
                scenario_config,
                seed,
                masked_actions,
                cost_extractor,
            )
        )

    actual_cost = _sum_costs(actual_components)
    cf_cost = _sum_costs(cf_components)
    notes_parts = [
        *(f"actual {note}" for note in actual_replay.notes),
        *(f"cf {note}" for note in counterfactual_replay.notes),
    ]
    if identity_mask:
        notes_parts.append("identity_mask_reused_actual_replay")
    schedule_proven = bool(
        actual_replay.actual_schedule_is_proven
        and counterfactual_replay.completed_recorded_schedule
        and actual_replay.steps_executed == counterfactual_replay.steps_executed
    )
    if not schedule_proven:
        notes_parts.append(
            "recorded replay schedule could not be proven complete and equal "
            "for actual/masked branches "
            f"(actual={actual_replay.steps_executed}/"
            f"{actual_replay.recorded_steps}, "
            f"masked={counterfactual_replay.steps_executed}/"
            f"{counterfactual_replay.recorded_steps}, "
            f"horizon={actual_replay.horizon}, "
            f"actual_final_done={actual_replay.final_step_done})"
        )

    # When the counterfactual itself produced no usable
    # baseline (cf crashed early / produced near-zero cost), the
    # normalised score is meaningless. Mark applicable=False rather
    # than handing out a free 1.0 (perfect prevention of nothing).
    costs_finite = math.isfinite(actual_cost) and math.isfinite(cf_cost)
    components_usable = _cost_components_are_usable(
        actual_components
    ) and _cost_components_are_usable(cf_components)
    applicable = (
        schedule_proven
        and components_usable
        and costs_finite
        and cf_cost > 0
    )
    if schedule_proven and (
        not components_usable or not costs_finite or cf_cost <= 0
    ):
        notes_parts.append(
            f"cf baseline produced cf_cost={cf_cost:.2f}; "
            "normalized_prevention unavailable (cost components must be "
            "finite and non-negative)"
        )
    notes = "; ".join(notes_parts)
    if applicable:
        reason_code = ""
    elif not schedule_proven:
        reason_code = REASON_CODE_REPLAY_SCHEDULE_UNPROVEN
    else:
        reason_code = REASON_CODE_CF_BASELINE_UNUSABLE

    # P1-3: per-action prevented_loss attribution. Opt-in; the per-episode
    # prevented_loss above is the all-masked baseline and is NEVER touched
    # by this loop. Each entry masks ONE state-changing action to ``wait``
    # and records the marginal outcome delta. Capped at 20 by default to
    # bound the O(k) replay cost on long
    # episodes (e.g. job_shop).
    per_action_entries: list[dict[str, Any]] = []
    per_action_capped: bool = False
    per_action_status = (
        "unavailable" if per_action and not applicable else "not_requested"
    )
    per_action_expected = 0
    per_action_attempted = 0
    per_action_failures: list[dict[str, Any]] = []
    if per_action and applicable:
        attribution = _attribute_per_action_prevented_loss(
            env_factory=env_factory,
            scenario_config=scenario_config,
            seed=seed,
            actual_actions=actual_actions,
            cost_extractor=cost_extractor,
            actual_cost=actual_cost,
            readonly_tool_names=readonly_tool_names,
            max_actions=per_action_cap,
        )
        per_action_entries = attribution.entries
        per_action_capped = attribution.capped
        per_action_status = attribution.status
        per_action_expected = attribution.expected
        per_action_attempted = attribution.attempted
        per_action_failures = attribution.failures
        if per_action_failures:
            failure_ids = ",".join(
                str(row.get("call_id") or row.get("tool_name") or "unknown")
                for row in per_action_failures
            )
            notes = "; ".join(
                part
                for part in (
                    notes,
                    "per-action attribution incomplete for " + failure_ids,
                )
                if part
            )

    per_action_group_entries: list[dict[str, Any]] = []
    per_action_group_status = (
        "unavailable"
        if per_action_groups and not applicable
        else "not_requested"
    )
    per_action_group_expected = 0
    per_action_group_attempted = 0
    per_action_group_failures: list[dict[str, Any]] = []
    if per_action_groups and applicable:
        group_attribution = _attribute_repeated_action_groups_prevented_loss(
            env_factory=env_factory,
            scenario_config=scenario_config,
            seed=seed,
            actual_actions=actual_actions,
            cost_extractor=cost_extractor,
            actual_cost=actual_cost,
            readonly_tool_names=readonly_tool_names,
            max_groups=per_action_group_cap,
        )
        per_action_group_entries = group_attribution.entries
        per_action_group_status = group_attribution.status
        per_action_group_expected = group_attribution.expected
        per_action_group_attempted = group_attribution.attempted
        per_action_group_failures = group_attribution.failures
        if per_action_group_failures:
            failure_ids = ",".join(
                str(row.get("group_id") or "unknown")
                for row in per_action_group_failures
            )
            notes = "; ".join(
                part
                for part in (
                    notes,
                    "per-action-group attribution incomplete for " + failure_ids,
                )
                if part
            )

    return CounterfactualReport(
        actual_cost=actual_cost,
        counterfactual_cost=cf_cost,
        prevented_loss=cf_cost - actual_cost,
        actual_components=actual_components,
        counterfactual_components=cf_components,
        masking_policy=masking_label,
        applicable=applicable,
        reason_code=reason_code,
        notes=notes,
        per_action=per_action_entries,
        per_action_capped=per_action_capped,
        per_action_status=per_action_status,
        per_action_expected=per_action_expected,
        per_action_attempted=per_action_attempted,
        per_action_completed=len(per_action_entries),
        per_action_failures=per_action_failures,
        per_action_groups=per_action_group_entries,
        per_action_group_status=per_action_group_status,
        per_action_group_expected=per_action_group_expected,
        per_action_group_attempted=per_action_group_attempted,
        per_action_group_completed=len(per_action_group_entries),
        per_action_group_failures=per_action_group_failures,
        actual_ground_truth=actual_replay.ground_truth,
        counterfactual_ground_truth=counterfactual_replay.ground_truth,
    )


def _sum_costs(components: dict[str, float]) -> float:
    """Aggregate non-negative finite numeric cost components fail-closed.

    Cost components are penalties, not credits.  A negative or non-finite
    component is therefore an invalid replay result, even when another
    component would make the aggregate positive.  Returning zero keeps the
    report JSON-safe; callers separately check
    :func:`_cost_components_are_usable` before marking a replay applicable.
    """
    total = 0.0
    for value in components.values():
        if not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0:
            return 0.0
        total += numeric
        if not math.isfinite(total):
            return 0.0
    return total


def _cost_components_are_usable(components: dict[str, float]) -> bool:
    """Return whether all numeric cost components are finite and non-negative."""
    total = 0.0
    for value in components.values():
        if not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0:
            return False
        total += numeric
        if not math.isfinite(total):
            return False
    return True


def _finite_json_cost(value: float) -> float:
    """Keep invalid replay costs from escaping into JSON as NaN/Infinity."""
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return numeric if math.isfinite(numeric) else 0.0


def _finite_json_components(components: dict[str, float]) -> dict[str, Any]:
    """Serialize component values without emitting NaN/Infinity."""
    return {
        key: _finite_json_cost(value)
        if isinstance(value, (int, float))
        else value
        for key, value in components.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# P1-3: per-action prevented_loss attribution helpers
# ─────────────────────────────────────────────────────────────────────────────


def _state_changing_call(
    action: Action, readonly_tool_names: set[str] | None = None
) -> ToolCall | None:
    """Return the first registered state-changing call in an action.

    Read-only investigative actions
    (``query_grid_state``, ``forecast_query``, …) are NOT state-changing —
    masking them to ``wait`` has no outcome effect, so they are skipped to
    avoid spending a replay on a zero-marginal attribution entry.

    Mixed actions are handled call-by-call: a read-only query followed by a
    control is attributed to the control rather than being skipped because
    the query happened to be first.
    """
    calls = _state_changing_calls(action, readonly_tool_names)
    return calls[0][1] if calls else None


def _state_changing_calls(
    action: Action,
    readonly_tool_names: set[str] | None = None,
) -> list[tuple[int, ToolCall]]:
    """Return ``(call_index, call)`` for every physical call in an action."""
    readonly = set(readonly_tool_names or ()) | {"wait", "noop"}
    return [
        (call_index, call)
        for call_index, call in enumerate(action.tool_calls)
        if call.name not in readonly
    ]


def _is_state_changing_action(
    action: Action, readonly_tool_names: set[str] | None = None
) -> bool:
    return _state_changing_call(action, readonly_tool_names) is not None


def _args_signature(action: Action, call: ToolCall | None = None) -> str:
    """Stable string signature of an action's first (dominant) tool call
    args, for attribution diagnostics. The full args dict is serialised
    with sorted keys so identical calls produce identical signatures
    regardless of insertion order."""
    if not action.tool_calls:
        return ""
    selected = call or action.tool_calls[0]
    args = selected.args or {}
    # sorted by key for deterministic output; repr for type stability
    items = sorted(args.items(), key=lambda kv: kv[0])
    return ",".join(f"{k}={v!r}" for k, v in items)


@dataclass
class _PerActionAttribution:
    entries: list[dict[str, Any]] = field(default_factory=list)
    capped: bool = False
    expected: int = 0
    attempted: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.failures:
            return "incomplete"
        if self.capped:
            return "capped"
        return "complete"


@dataclass
class _PerActionGroupAttribution:
    entries: list[dict[str, Any]] = field(default_factory=list)
    capped: bool = False
    expected: int = 0
    attempted: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.failures:
            return "incomplete"
        if self.capped:
            return "capped"
        return "complete"


def _attribute_per_action_prevented_loss(
    *,
    env_factory: Callable[[], POMDPEnvironment],
    scenario_config: dict[str, Any],
    seed: int,
    actual_actions: list[Action],
    cost_extractor: Callable[[dict[str, Any]], dict[str, float]],
    actual_cost: float,
    readonly_tool_names: set[str] | None,
    max_actions: int | None,
) -> _PerActionAttribution:
    """For each selected state-changing action, replay
    the episode with ONLY that action masked to ``wait`` and record the
    marginal prevented loss = ``masked_one_cost - actual_cost``.

    The result records expected, attempted, completed, and failed calls. A
    failed single-call replay is never silently omitted from the attribution
    coverage claim. The per-episode ``prevented_loss`` (all-masked baseline)
    is unaffected.
    """
    # Attribute every state-changing ToolCall independently. A single Action
    # may contain several parallel controls plus read-only investigations;
    # masking the whole tick would conflate their marginal effects.
    state_changing_calls: list[tuple[int, int, ToolCall]] = [
        (tick, call_index, call)
        for tick, action in enumerate(actual_actions)
        for call_index, call in _state_changing_calls(
            action,
            readonly_tool_names,
        )
    ]

    # Cap: bound the O(k) replay cost. When the number of state-changing
    # actions exceeds the cap, only the first selected actions are attributed.
    # A researcher needing full coverage can pass None; the
    # per-episode ``prevented_loss`` (all-masked baseline) is unaffected.
    if max_actions is None:
        capped_calls = state_changing_calls
        was_capped = False
    else:
        cap = max(0, int(max_actions))
        capped_calls = state_changing_calls[:cap]
        was_capped = len(state_changing_calls) > cap

    entries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for tick, call_index, state_call in capped_calls:
        masked_components, replay = _replay_recorded_and_extract_costs_and_truth(
            env_factory,
            scenario_config,
            seed,
            actual_actions,
            cost_extractor,
            masking_policy=make_mask_single_action_policy(
                tick,
                call_index,
            ),
        )
        if not replay.completed_recorded_schedule:
            failures.append(
                {
                    "tick": int(tick),
                    "call_index": int(call_index),
                    "call_id": state_call.call_id,
                    "idempotency_key": state_call.idempotency_key,
                    "tool_name": str(state_call.name),
                    "reason_code": REASON_CODE_REPLAY_SCHEDULE_UNPROVEN,
                    "notes": list(replay.notes),
                }
            )
            continue
        if not _cost_components_are_usable(masked_components):
            failures.append(
                {
                    "tick": int(tick),
                    "call_index": int(call_index),
                    "call_id": state_call.call_id,
                    "idempotency_key": state_call.idempotency_key,
                    "tool_name": str(state_call.name),
                    "reason_code": REASON_CODE_CF_BASELINE_UNUSABLE,
                    "notes": [
                        "masked replay produced non-finite or negative "
                        "cost components"
                    ],
                }
            )
            continue
        masked_cost = _sum_costs(masked_components)
        marginal = masked_cost - actual_cost
        entries.append(
            {
                "tick": int(tick),
                "call_index": int(call_index),
                "call_id": state_call.call_id,
                "idempotency_key": state_call.idempotency_key,
                "tool_name": str(state_call.name),
                "args_signature": _args_signature(
                    actual_actions[tick],
                    state_call,
                ),
                "marginal_prevented_loss": float(marginal),
            }
        )
    return _PerActionAttribution(
        entries=entries,
        capped=was_capped,
        expected=len(state_changing_calls),
        attempted=len(capped_calls),
        failures=failures,
    )


def _attribute_repeated_action_groups_prevented_loss(
    *,
    env_factory: Callable[[], POMDPEnvironment],
    scenario_config: dict[str, Any],
    seed: int,
    actual_actions: list[Action],
    cost_extractor: Callable[[dict[str, Any]], dict[str, float]],
    actual_cost: float,
    readonly_tool_names: set[str] | None,
    max_groups: int | None,
) -> _PerActionGroupAttribution:
    """Attribute interaction effects for exact repeated native controls.

    Groups are deliberately narrow: calls must share both tool name and the
    canonical argument signature.  Seed, prompt, timing, or vague semantic
    similarity never creates a group.  At least two calls and complete call
    IDs are required so downstream event-response evidence can bind the replay.
    """
    grouped: dict[
        tuple[str, str],
        list[tuple[int, int, ToolCall]],
    ] = {}
    for tick, action in enumerate(actual_actions):
        for call_index, call in _state_changing_calls(
            action,
            readonly_tool_names,
        ):
            key = (str(call.name), _args_signature(action, call))
            grouped.setdefault(key, []).append((tick, call_index, call))

    repeated = [
        (key, calls)
        for key, calls in grouped.items()
        if len(calls) >= 2
    ]
    if max_groups is None:
        selected = repeated
        was_capped = False
    else:
        cap = max(0, int(max_groups))
        selected = repeated[:cap]
        was_capped = len(repeated) > cap

    entries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for (tool_name, args_signature), calls in selected:
        group_id = f"{tool_name}:{args_signature}"
        call_ids = [str(call.call_id or "") for _, _, call in calls]
        if any(not call_id for call_id in call_ids):
            failures.append(
                {
                    "group_id": group_id,
                    "reason_code": "action_group_call_id_missing",
                }
            )
            continue
        masked_components, replay = _replay_recorded_and_extract_costs_and_truth(
            env_factory,
            scenario_config,
            seed,
            actual_actions,
            cost_extractor,
            masking_policy=make_mask_action_group_policy(
                (tick, call_index) for tick, call_index, _call in calls
            ),
        )
        if not replay.completed_recorded_schedule:
            failures.append(
                {
                    "group_id": group_id,
                    "call_ids": call_ids,
                    "reason_code": REASON_CODE_REPLAY_SCHEDULE_UNPROVEN,
                    "notes": list(replay.notes),
                }
            )
            continue
        if not _cost_components_are_usable(masked_components):
            failures.append(
                {
                    "group_id": group_id,
                    "call_ids": call_ids,
                    "reason_code": REASON_CODE_CF_BASELINE_UNUSABLE,
                    "notes": [
                        "masked group replay produced non-finite or negative "
                        "cost components"
                    ],
                }
            )
            continue
        entries.append(
            {
                "group_id": group_id,
                "call_ids": call_ids,
                "ticks": [int(tick) for tick, _, _ in calls],
                "tool_name": tool_name,
                "args_signature": args_signature,
                "masked_action_group_delta": float(
                    _sum_costs(masked_components) - actual_cost
                ),
            }
        )
    return _PerActionGroupAttribution(
        entries=entries,
        capped=was_capped,
        expected=len(repeated),
        attempted=len(selected),
        failures=failures,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Multi-policy counterfactual (Phase 3, Part A)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CounterfactualRegretReport:
    """Aggregate regret report comparing an agent against multiple baselines.

    Each entry in ``per_baseline`` corresponds to one masking policy
    (e.g. wait_only, random, greedy_heuristic).  The report also stores
    the best (lowest-cost) baseline cost so downstream scoring can compute
    a ``max_regret`` relative to the strongest baseline.
    """

    actual_cost: float
    per_baseline: list[CounterfactualReport] = field(default_factory=list)
    best_baseline_cost: float = 0.0

    @property
    def max_regret(self) -> float:
        """How much worse the agent performed vs. the best baseline.

        Positive means the agent outperformed every baseline (actual
        cost < best baseline cost).  Negative means at least one baseline
        outperformed the agent.
        """
        applicable_costs = [
            report.counterfactual_cost
            for report in self.per_baseline
            if report.applicable
        ]
        if not applicable_costs:
            return 0.0
        return min(applicable_costs) - self.actual_cost

    def to_dict(self) -> dict[str, Any]:
        return {
            "actual_cost": float(self.actual_cost),
            "best_baseline_cost": float(self.best_baseline_cost),
            "max_regret": float(self.max_regret),
            "per_baseline": [r.to_dict() for r in self.per_baseline],
        }


def greedy_heuristic_masking_policy(action: Action, tick: int) -> Action:
    """Placeholder: keep only state-changing tool calls.

    This approximates a greedy heuristic by dropping meta-tools
    (wait, noop) and investigation tools.
    """
    _greedy_keep = {"wait", "noop", "query_grid_state", "shed_load"}
    filtered = [c for c in action.tool_calls if c.name in _greedy_keep]
    if not filtered:
        filtered = [ToolCall(name="wait")]
    return Action(tool_calls=filtered, dominant=filtered[0].name)


def multi_policy_counterfactual(
    env_factory: Callable[[], POMDPEnvironment],
    scenario_config: dict[str, Any],
    seed: int,
    actual_actions: list[Action],
    cost_extractor: Callable[[dict[str, Any]], dict[str, float]],
    baseline_action_providers: (dict[str, Callable[[], list[Action]]] | None) = None,
) -> CounterfactualRegretReport:
    """Run counterfactual replay against multiple baseline policies.

    Parameters
    ----------
    env_factory, scenario_config, seed, actual_actions, cost_extractor:
        Same as :func:`run_counterfactual`.
    baseline_action_providers:
        Optional dict mapping policy name to a zero-arg callable that returns
        the full action sequence that policy would execute.  When provided,
        the multi-policy report runs each baseline through the full seeded
        environment.  When ``None``, the function defaults to three built-in
        masking policies: ``wait_only``, ``keep_investigations``, and
        ``greedy_heuristic``.

    Returns
    -------
    CounterfactualRegretReport
        Aggregated report with per-baseline detail.
    """
    # ── Pass 1: capture actual cost ──
    env_actual = env_factory()
    try:
        env_actual.reset(copy.deepcopy(scenario_config), seed)
        # Derive the domain's read-only tool set once (env is freshly reset, so its
        # registry is populated). Reused by the keep_investigations masking policy
        # so it stays domain-agnostic rather than hardcoding power-grid tools.
        keep_investigations = keep_investigations_policy_for_env(env_actual)
        actual_replay = _replay_recorded_schedule(
            env_actual,
            actual_actions,
        )
        actual_components = cost_extractor(actual_replay.ground_truth)
        actual_cost = _sum_costs(actual_components)
    finally:
        env_actual.close()

    # ── Pass 2: run baselines ──
    if baseline_action_providers is not None:
        policies: list[tuple[str, list[Action] | None]] = []
        for name, provider in baseline_action_providers.items():
            try:
                actions = provider()
            except Exception:
                actions = []
            if not isinstance(actions, list):
                actions = []
            policies.append((name, actions))
    else:
        policies = [
            ("wait_only", None),
            ("keep_investigations", None),
            ("greedy_heuristic", None),
        ]

    per_baseline: list[CounterfactualReport] = []
    best_cf_cost = float("inf")

    for label, baseline_actions in policies:
        if baseline_actions is not None:
            report = _baseline_report_from_action_replay(
                env_factory,
                scenario_config,
                seed,
                actual_actions,
                actual_cost,
                actual_components,
                actual_replay,
                baseline_actions,
                cost_extractor,
                label,
            )
        else:
            report = _baseline_report_from_masking_policy(
                env_factory,
                scenario_config,
                seed,
                actual_actions,
                cost_extractor,
                keep_investigations,
                label,
            )
        per_baseline.append(report)
        if report.applicable and report.counterfactual_cost < best_cf_cost:
            best_cf_cost = report.counterfactual_cost

    if best_cf_cost == float("inf"):
        best_cf_cost = 0.0

    return CounterfactualRegretReport(
        actual_cost=actual_cost,
        per_baseline=per_baseline,
        best_baseline_cost=best_cf_cost,
    )


def _baseline_report_from_action_replay(
    env_factory: Callable[[], POMDPEnvironment],
    scenario_config: dict[str, Any],
    seed: int,
    actual_actions: list[Action],
    actual_cost: float,
    actual_components: dict[str, float],
    actual_replay: _RecordedScheduleReplay,
    baseline_actions: list[Action],
    cost_extractor: Callable[[dict[str, Any]], dict[str, float]],
    label: str,
) -> CounterfactualReport:
    """Replay a full baseline action sequence (e.g. a real ``RandomAgent``
    trajectory supplied via ``baseline_action_providers``) and score it
    against ``actual_cost``.
    """
    cf_components, baseline_replay = _replay_recorded_and_extract_costs_and_truth(
        env_factory,
        scenario_config,
        seed,
        list(baseline_actions),
        cost_extractor,
    )
    cf_cost = _sum_costs(cf_components)
    schedule_proven = bool(
        actual_replay.actual_schedule_is_proven
        and len(baseline_actions) == len(actual_actions)
        and baseline_replay.completed_recorded_schedule
    )
    costs_finite = math.isfinite(actual_cost) and math.isfinite(cf_cost)
    components_usable = _cost_components_are_usable(
        actual_components
    ) and _cost_components_are_usable(cf_components)
    applicable = (
        schedule_proven
        and components_usable
        and costs_finite
        and cf_cost > 0
    )
    if not schedule_proven:
        reason_code = REASON_CODE_REPLAY_SCHEDULE_UNPROVEN
        notes = (
            "explicit baseline did not prove the same recorded replay "
            f"schedule (actual_steps={len(actual_actions)}, "
            f"baseline_steps={len(baseline_actions)}, "
            f"baseline_executed={baseline_replay.steps_executed})"
        )
    elif not components_usable or not costs_finite or cf_cost <= 0:
        reason_code = REASON_CODE_CF_BASELINE_UNUSABLE
        notes = (
            f"cf baseline produced cf_cost={cf_cost:.2f}; "
            "cost components must be finite and non-negative"
        )
    else:
        reason_code = ""
        notes = ""
    return CounterfactualReport(
        actual_cost=actual_cost,
        counterfactual_cost=cf_cost,
        prevented_loss=cf_cost - actual_cost,
        actual_components=actual_components,
        counterfactual_components=cf_components,
        masking_policy=label,
        applicable=applicable,
        reason_code=reason_code,
        notes=notes,
    )


def _baseline_report_from_masking_policy(
    env_factory: Callable[[], POMDPEnvironment],
    scenario_config: dict[str, Any],
    seed: int,
    actual_actions: list[Action],
    cost_extractor: Callable[[dict[str, Any]], dict[str, float]],
    keep_investigations: MaskingPolicy,
    label: str,
) -> CounterfactualReport:
    """Replay the agent's own ``actual_actions`` through one of the
    built-in masking policies (``wait_only`` / ``keep_investigations`` /
    ``greedy_heuristic``) via :func:`run_counterfactual`.
    """
    masking_policy: MaskingPolicy
    if label == "wait_only":
        masking_policy = wait_only_policy
    elif label == "keep_investigations":
        masking_policy = keep_investigations
    elif label == "greedy_heuristic":
        masking_policy = greedy_heuristic_masking_policy
    else:
        masking_policy = wait_only_policy

    return run_counterfactual(
        env_factory=env_factory,
        scenario_config=scenario_config,
        seed=seed,
        actual_actions=actual_actions,
        cost_extractor=cost_extractor,
        masking_policy=masking_policy,
        masking_label=label,
    )
