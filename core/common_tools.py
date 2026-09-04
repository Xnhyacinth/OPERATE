"""
core.common_tools — Domain-agnostic tool-handler + serialization helpers.

Every domain adapter (power_grid, logistics, traffic, microgrid, disaster)
re-implemented the same handful of pieces verbatim or near-verbatim:

- ``moral_choice`` / ``commit_to_plan`` tool handlers (operate purely on
  ``core.ethical_dilemma`` / ``core.evidence`` concepts via ``env.dilemmas``
  / ``env.evidence``, so they never needed to be domain-specific).
- ``wait`` / ``noop`` meta tool specs.
- A best-effort dataclass/dict serializer for ``StepInfo.extra`` blobs.
- Arming a domain's ``EthicalDilemmaManager`` from a seed's ``dilemmas``
  list (the seed's dilemma objects are duck-typed the same way across all
  five domains: ``dilemma_id``, ``trigger_tick``, ``description``,
  ``options``, ``resolution_deadline_ticks``, ``default_option_id``,
  ``expected_tradeoff_tokens``, ``expected_stakeholder_tokens``).

This module knows nothing about Grid2Op, pandapower, SUMO, PyVRP, or RCRS —
only the core ``ToolContext`` / ``EthicalDilemmaManager`` / ``EvidenceLogger``
contracts — so it is safe to live in ``core`` per the ``core`` <-> ``domains``
boundary (``.hl/policy.md`` Hard Red Line, ``CLAUDE.md``).

Small, real per-domain behavior deltas are preserved via keyword flags
(``verbose_errors``, ``events_key``, ``include_horizon_ticks``,
``dict_fallback``) rather than force-unified — see call sites in each
domain's ``native_tools.py`` / ``adapter.py`` for which flag values
reproduce that domain's pre-extraction behavior byte-for-byte.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from .ethical_dilemma import Dilemma, EthicalDilemmaManager, MoralChoice, MoralOption
from .tool_protocol import ToolContext, ToolSpec

__all__ = [
    "arm_dilemmas",
    "commit_to_plan_handler",
    "moral_choice_handler",
    "noop_tool_spec",
    "plan_autonomy_properties",
    "safe_dataclass_to_dict",
    "wait_tool_spec",
]


# ─────────────────────────────────────────────────────────────────────────────
# Serialization
# ─────────────────────────────────────────────────────────────────────────────


def safe_dataclass_to_dict(obj: Any, *, dict_fallback: bool = False) -> Any:
    """Best-effort dataclass-to-dict conversion for the ``StepInfo.extra`` blob.

    ``dict_fallback=True`` additionally projects plain objects that expose
    ``__dict__`` (but are not dataclasses) into a dict of their public
    attributes before falling back to ``str(obj)``. The power_grid adapter
    opts into this (its backend records occasionally nest non-dataclass
    helper objects); the other four domains never needed it, so they keep
    the narrower default to stay behavior-identical to their pre-extraction
    code.
    """
    if hasattr(obj, "__dataclass_fields__"):
        return {
            k: safe_dataclass_to_dict(getattr(obj, k), dict_fallback=dict_fallback)
            for k in obj.__dataclass_fields__
        }
    if isinstance(obj, dict):
        return {
            k: safe_dataclass_to_dict(v, dict_fallback=dict_fallback)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [safe_dataclass_to_dict(v, dict_fallback=dict_fallback) for v in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if dict_fallback and hasattr(obj, "__dict__"):
        return {
            k: safe_dataclass_to_dict(v, dict_fallback=dict_fallback)
            for k, v in vars(obj).items()
            if not k.startswith("_")
        }
    return str(obj)


# ─────────────────────────────────────────────────────────────────────────────
# Dilemma arming
# ─────────────────────────────────────────────────────────────────────────────


def arm_dilemmas(mgr: EthicalDilemmaManager, dilemma_seeds: Iterable[Any]) -> None:
    """Register every seed dilemma on ``mgr`` as a ``core.Dilemma``.

    ``dilemma_seeds`` is any iterable of duck-typed seed objects exposing
    ``dilemma_id``, ``trigger_tick``, ``description``, ``options`` (a list
    of dicts with ``option_id`` / ``label`` / ``fatal`` /
    ``expected_consequences``), ``resolution_deadline_ticks``,
    ``default_option_id``, ``expected_tradeoff_tokens``, and
    ``expected_stakeholder_tokens`` — i.e. each domain's local
    ``DilemmaSeed`` (or ``TrafficDilemmaSeed``). Deliberately parameterized
    by attribute access rather than by class name so no domain schema type
    needs to be imported here.
    """
    for d_seed in dilemma_seeds:
        dilemma = Dilemma(
            dilemma_id=d_seed.dilemma_id,
            trigger_tick=d_seed.trigger_tick,
            description=d_seed.description,
            options=[
                MoralOption(
                    option_id=o.get("option_id", "?"),
                    label=o.get("label", o.get("option_id", "?")),
                    fatal=bool(o.get("fatal", False)),
                    expected_consequences=dict(o.get("expected_consequences", {})),
                )
                for o in d_seed.options
            ],
            resolution_deadline_ticks=d_seed.resolution_deadline_ticks,
            default_option_id=d_seed.default_option_id,
            expected_tradeoff_tokens=list(d_seed.expected_tradeoff_tokens),
            expected_stakeholder_tokens=list(d_seed.expected_stakeholder_tokens),
        )
        mgr.register_dilemma(dilemma)


# ─────────────────────────────────────────────────────────────────────────────
# Ethics / record tool handlers
# ─────────────────────────────────────────────────────────────────────────────


def moral_choice_handler(
    env: Any, *, verbose_errors: bool = True
) -> Callable[[dict[str, Any], ToolContext], dict[str, Any]]:
    """Factory for the ``moral_choice`` tool handler shared by every domain.

    ``env`` must expose ``.dilemmas`` (an ``EthicalDilemmaManager | None``)
    and ``.evidence`` (an ``EvidenceLogger | None``) — the same
    ``PowerGridEnvironment`` / ``LogisticsEnvironment`` / ... accessor
    pattern every domain adapter already implements.

    ``verbose_errors=True`` (power_grid / disaster / traffic's original
    behavior) includes ``previous_option`` on ``already_resolved`` and
    ``option_id`` on ``unknown_option`` error payloads. Logistics never
    exposed those extra fields; pass ``verbose_errors=False`` there to stay
    byte-identical to its pre-extraction handler.
    """

    def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        mgr = env.dilemmas
        if mgr is None:
            return {"_status": "error", "error": "no_dilemma_manager"}
        did = str(args.get("dilemma_id", ""))
        # Reject moral_choice for dilemmas the agent hasn't been informed
        # about yet (not in triggered list), or that are already resolved,
        # or that don't exist.
        triggered = {d.dilemma_id: d for d in mgr.record.dilemmas_triggered}
        if did not in triggered:
            return {
                "_status": "error",
                "error": "dilemma_not_active",
                "dilemma_id": did,
                "active": list(triggered.keys()),
            }
        if did in mgr.record.choices:
            result: dict[str, Any] = {
                "_status": "error",
                "error": "already_resolved",
                "dilemma_id": did,
            }
            if verbose_errors:
                result["previous_option"] = mgr.record.choices[did].chosen_option_id
            return result
        dilemma_def = triggered[did]
        option_id = str(args.get("option_id", ""))
        valid_options = {o.option_id for o in dilemma_def.options}
        if option_id not in valid_options:
            result = {
                "_status": "error",
                "error": "unknown_option",
                "dilemma_id": did,
            }
            if verbose_errors:
                result["option_id"] = option_id
            result["valid_options"] = sorted(valid_options)
            return result
        choice = MoralChoice(
            dilemma_id=did,
            chosen_option_id=option_id,
            rationale=str(args.get("rationale", "")),
            tick_chosen=ctx.tick,
            tradeoffs_considered=[
                str(value) for value in args.get("tradeoffs_considered", [])
            ],
            affected_stakeholders=[
                str(value) for value in args.get("affected_stakeholders", [])
            ],
            reversibility_assessment=str(
                args.get("reversibility_assessment", "")
            ),
        )
        mgr.record_choice(choice)
        if env.evidence is not None:
            env.evidence.log(
                "moral_choice",
                ctx.tick,
                payload={
                    "dilemma_id": choice.dilemma_id,
                    "option_id": choice.chosen_option_id,
                    "rationale": choice.rationale,
                    "tradeoffs_considered": choice.tradeoffs_considered,
                    "affected_stakeholders": choice.affected_stakeholders,
                    "reversibility_assessment": choice.reversibility_assessment,
                },
                source="agent",
            )
        return {
            "dilemma_id": choice.dilemma_id,
            "chosen_option_id": choice.chosen_option_id,
            "ack": True,
        }

    return handler


def commit_to_plan_handler(
    env: Any,
    *,
    events_key: str = "predicted_events",
    include_horizon_ticks: bool = True,
) -> Callable[[dict[str, Any], ToolContext], dict[str, Any]]:
    """Factory for the ``commit_to_plan`` tool handler shared by every domain.

    ``env`` must expose ``.evidence`` (an ``EvidenceLogger | None``).

    ``events_key`` is both the incoming-args key and the evidence-payload
    key for the predicted-events list: power_grid / disaster / traffic /
    logistics all use ``"predicted_events"``; microgrid uses
    ``"predictions"``. ``include_horizon_ticks=False`` (logistics,
    microgrid) omits the ``horizon_ticks`` field entirely rather than
    defaulting it to 0, matching each domain's pre-extraction payload
    shape exactly.
    """

    def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        review_after = args.get("review_after_ticks")
        if review_after is not None:
            if (
                isinstance(review_after, bool)
                or not isinstance(review_after, int)
                or review_after <= 0
            ):
                return {
                    "_status": "error",
                    "error": "review_interval_must_be_positive_integer",
                }
        expiry = args.get("plan_expires_at_tick")
        if expiry is not None:
            expiry_tick = int(expiry)
            if expiry_tick <= int(ctx.tick):
                return {
                    "_status": "error",
                    "error": "plan_expiry_not_in_future",
                    "plan_expires_at_tick": expiry_tick,
                }
            horizon = int(
                getattr(env, "horizon", getattr(env, "_horizon", 0)) or 0
            )
            if horizon > 0 and expiry_tick >= horizon:
                return {
                    "_status": "error",
                    "error": "plan_expiry_has_no_response_window",
                    "plan_expires_at_tick": expiry_tick,
                    "horizon_ticks": horizon,
                }
        if env.evidence is not None:
            payload: dict[str, Any] = {
                "plan_id": str(args.get("plan_id", "")),
                "rationale": str(args.get("rationale", "")),
            }
            if include_horizon_ticks:
                payload["horizon_ticks"] = int(args.get("horizon_ticks", 0))
            payload[events_key] = list(args.get(events_key, []))
            if args.get("replaces_plan_id"):
                payload["replaces_plan_id"] = str(args["replaces_plan_id"])
            if args.get("revision_reason"):
                payload["revision_reason"] = str(args["revision_reason"])
            if args.get("trigger_evidence_ids"):
                payload["trigger_evidence_ids"] = [
                    str(value) for value in args["trigger_evidence_ids"]
                ]
            if args.get("review_after_ticks") is not None:
                payload["review_after_ticks"] = int(args["review_after_ticks"])
            if args.get("wake_if") is not None:
                payload["wake_if"] = [str(value) for value in args["wake_if"]]
            if args.get("plan_expires_at_tick") is not None:
                payload["plan_expires_at_tick"] = int(
                    args["plan_expires_at_tick"]
                )
            env.evidence.log(
                "commit_to_plan",
                ctx.tick,
                payload=payload,
                source="agent",
            )
        return {"plan_id": args.get("plan_id"), "ack": True}

    return handler


def plan_autonomy_properties() -> dict[str, dict[str, Any]]:
    """Optional plan fields for event-adaptive long-horizon supervision."""
    return {
        "review_after_ticks": {
            "type": "integer",
            "minimum": 1,
            "description": (
                "Request the next model review after this many simulator ticks "
                "while current controls remain in force. The runner advances "
                "the backend autonomously and wakes early for visible events, "
                "tool failures, safety warnings, or active dilemmas."
            ),
        },
        "wake_if": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "visible_event",
                    "forecast_update",
                    "delayed_tool",
                ],
            },
            "uniqueItems": True,
            "description": (
                "Optional event classes that may wake the plan before its "
                "scheduled review. Mandatory safety, task, dilemma, terminal, "
                "and failed-action interrupts cannot be disabled."
            ),
        },
        "plan_expires_at_tick": {
            "type": "integer",
            "minimum": 1,
            "description": (
                "Optional absolute simulator tick at which the standing plan "
                "must be reviewed."
            ),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Meta tools
# ─────────────────────────────────────────────────────────────────────────────


def wait_tool_spec(
    description: str = (
        "Decline intervention for this control interval; simulator time "
        "advances independently."
    ),
) -> ToolSpec:
    """Shared ``wait`` meta-tool spec. ``description`` varies per domain."""
    return ToolSpec(
        name="wait",
        description=description,
        parameters={"type": "object", "properties": {}, "required": []},
        handler=lambda args, ctx: {"_status": "waited"},
        semantic_role="meta",
        native_target_kind="simulation_clock",
        fail_rate=0.0,
        delay_ticks=0,
    )


def noop_tool_spec(description: str = "Alias for wait.") -> ToolSpec:
    """Shared ``noop`` meta-tool spec. ``description`` varies per domain."""
    return ToolSpec(
        name="noop",
        description=description,
        parameters={"type": "object", "properties": {}, "required": []},
        handler=lambda args, ctx: {"_status": "noop"},
        semantic_role="meta",
        native_target_kind="simulation_clock",
        fail_rate=0.0,
        delay_ticks=0,
    )
