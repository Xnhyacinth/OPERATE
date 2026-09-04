"""
evaluation.foresight — Foresight / predictive-capability evaluator.

Forked-and-refactored from
``dispatch-benchmark/evaluation/foresight_evaluator.py`` to align with
OPERATE's domain-native event vocabularies.

The evaluator consumes the agent's ``commit_to_plan`` evidence items
(which carry ``predicted_events``) and verifies them against the realized
event log in ``EvidenceLogger.items_by_kind("realized_event")``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from core import EvidenceLogger


@dataclass(frozen=True, slots=True)
class ForecastEventSchema:
    """Typed matching contract for one forecastable native event."""

    domains: tuple[str, ...]
    target_keys: tuple[str, ...]


FORECAST_EVENT_REGISTRY = MappingProxyType(
    {
        # Traffic
        "incident": ForecastEventSchema(("traffic",), ("corridor", "edge")),
        "lane_blockage": ForecastEventSchema(("traffic",), ("corridor", "edge")),
        "traffic_lane_blockage": ForecastEventSchema(
            ("traffic",), ("corridor", "edge", "lane_id")
        ),
        "signal_failure": ForecastEventSchema(("traffic",), ("corridor", "tls_id")),
        "detector_dropout": ForecastEventSchema(("traffic",), ("corridor", "edge")),
        "weather_capacity_drop": ForecastEventSchema(
            ("traffic",), ("corridor", "edge")
        ),
        "vip_arrival": ForecastEventSchema(("traffic",), ("corridor",)),
        "ems_corridor_request": ForecastEventSchema(("traffic",), ("corridor",)),
        "traffic_demand_surge": ForecastEventSchema(
            ("traffic",), ("corridor", "route_id", "tls_id")
        ),
        # Logistics (routing and dynamic job shop)
        "vehicle_breakdown": ForecastEventSchema(
            ("logistics",), ("vehicle_id", "region")
        ),
        "blocked_arc": ForecastEventSchema(
            ("logistics",), ("customer_id", "arc_id", "region")
        ),
        "traffic_delay": ForecastEventSchema(
            ("logistics",), ("customer_id", "region")
        ),
        "urgent_order": ForecastEventSchema(
            ("logistics",), ("customer_id", "job_id", "order_id")
        ),
        "job_arrival": ForecastEventSchema(("logistics",), ("job_id",)),
        "due_date_set": ForecastEventSchema(("logistics",), ("job_id",)),
        "machine_breakdown": ForecastEventSchema(
            ("logistics",), ("machine_id", "machine")
        ),
        "process_time_change": ForecastEventSchema(("logistics",), ("job_id",)),
        "priority_change": ForecastEventSchema(("logistics",), ("job_id",)),
        "order_cancellation": ForecastEventSchema(("logistics",), ("job_id",)),
        "preventive_maintenance": ForecastEventSchema(
            ("logistics",), ("machine_id", "machine")
        ),
        "route_change": ForecastEventSchema(("logistics",), ("job_id",)),
        "due_date_change": ForecastEventSchema(("logistics",), ("job_id",)),
        # Microgrid
        "grid_outage": ForecastEventSchema(("microgrid",), ("asset_id", "bus_id")),
        "price_spike": ForecastEventSchema(("microgrid",), ("asset_id", "bus_id")),
        "pv_ramp": ForecastEventSchema(("microgrid",), ("der_id", "bus_id")),
        "der_failure": ForecastEventSchema(("microgrid",), ("der_id", "bus_id")),
        "load_spike": ForecastEventSchema(("microgrid",), ("load_id", "bus_id")),
        # Power grid. ``load_surge`` is also native to logistics and traffic.
        "line_outage": ForecastEventSchema(("power_grid",), ("line_id",)),
        "generator_outage": ForecastEventSchema(
            ("power_grid",), ("generator_id",)
        ),
        "fuel_supply_delay": ForecastEventSchema(
            ("power_grid",), ("generator_id",)
        ),
        "wind_dropout": ForecastEventSchema(("power_grid",), ("generator_id",)),
        "opponent_attack": ForecastEventSchema(
            ("power_grid",), ("line_id", "generator_id")
        ),
        "storm_window": ForecastEventSchema(
            ("power_grid",), ("line_id", "region")
        ),
        "planned_maintenance": ForecastEventSchema(
            ("power_grid",), ("line_id", "generator_id")
        ),
        "planned_maintenance_window": ForecastEventSchema(
            ("power_grid",), ("line_id", "generator_id")
        ),
        "generation_ramp": ForecastEventSchema(
            ("power_grid",), ("generator_id",)
        ),
        "renewable_output_error": ForecastEventSchema(
            ("power_grid",), ("generator_id",)
        ),
        "forecast_bias": ForecastEventSchema(
            ("power_grid", "microgrid"), ("generator_id", "der_id", "bus_id")
        ),
        "load_surge": ForecastEventSchema(
            ("power_grid", "traffic", "logistics"),
            ("stakeholder_class", "load_id", "corridor", "job_id", "region"),
        ),
        "demand_surge": ForecastEventSchema(
            ("traffic", "logistics"), ("corridor", "job_id", "region")
        ),
    }
)


def is_forecastable_event(payload: dict[str, Any]) -> bool:
    """Return whether an engine event belongs to the closed forecast registry."""

    event_type = str(payload.get("type") or payload.get("kind") or "")
    if event_type not in FORECAST_EVENT_REGISTRY:
        return False
    return not (
        str(payload.get("origin") or "") == "agent_caused"
        or str(payload.get("event_class") or "")
        in {"agent_outcome", "lifecycle", "telemetry"}
    )


def _target_for_event(payload: dict[str, Any], schema: ForecastEventSchema) -> str | None:
    candidates = [payload]
    declared = payload.get("declared_event")
    if isinstance(declared, dict):
        target = declared.get("target")
        if isinstance(target, dict):
            candidates.append(target)
    target = payload.get("target")
    if isinstance(target, dict):
        candidates.append(target)
    for candidate in candidates:
        for key in (*schema.target_keys, "target_id", "entity_id"):
            value = candidate.get(key)
            if value is not None:
                try:
                    return str(value)
                except Exception:
                    return None
    return None


@dataclass
class PredictionEvent:
    event_type: str
    target_id: str | None
    predicted_tick: int
    issued_tick: int
    source_evidence_id: str
    actual_occurred: bool = False
    actual_tick: int | None = None
    confidence: float = 0.5


@dataclass
class ProactiveAction:
    action_type: str
    target_id: str | None
    tick_taken: int
    predicted_event_type: str
    prevented_or_mitigated: bool = False
    mitigation_score: float = 0.0


@dataclass
class ForesightMetrics:
    total_predictions: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    forecastable_realized_events: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    mean_tick_error: float = 0.0
    tick_accuracy_within_tol: float = 0.0
    proactive_actions_count: int = 0
    successful_preventions: int = 0
    mitigation_rate: float = 0.0
    foresight_score: float = 0.0
    prediction_events: list[PredictionEvent] = field(default_factory=list)
    proactive_actions: list[ProactiveAction] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_predictions": self.total_predictions,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "forecastable_realized_events": self.forecastable_realized_events,
            "precision": round(self.precision, 3),
            "recall": round(self.recall, 3),
            "f1_score": round(self.f1_score, 3),
            "mean_tick_error": round(self.mean_tick_error, 3),
            "tick_accuracy_within_tol": round(self.tick_accuracy_within_tol, 3),
            "proactive_actions_count": self.proactive_actions_count,
            "successful_preventions": self.successful_preventions,
            "mitigation_rate": round(self.mitigation_rate, 3),
            "foresight_score": round(self.foresight_score, 2),
        }


def evaluate_foresight(
    evidence_logger: EvidenceLogger,
    *,
    tick_tolerance: int = 3,
) -> ForesightMetrics:
    """Compute foresight metrics from the evidence log."""
    metrics = ForesightMetrics()

    # ── Pull realized events ────────────────────────────────────────────
    realized_items = evidence_logger.items_by_kind("realized_event")
    realized: list[tuple[str, str | None, int]] = []  # (event_type, target_id, tick)
    for item in realized_items:
        payload = item.payload or {}
        if not is_forecastable_event(payload):
            continue
        etype = str(payload.get("type") or payload.get("kind") or "")
        realized.append(
            (etype, _target_for_event(payload, FORECAST_EVENT_REGISTRY[etype]), item.tick)
        )
    metrics.forecastable_realized_events = len(realized)

    # ── Pull agent predictions from commit_to_plan items ───────────────
    plan_items = evidence_logger.items_by_kind("commit_to_plan")
    predictions: list[PredictionEvent] = []
    for item in plan_items:
        plan_tick = item.tick
        raw_predictions = item.payload.get("predicted_events")
        if raw_predictions is None:
            raw_predictions = item.payload.get("predictions")
        for pred in raw_predictions or []:
            if not isinstance(pred, dict):
                continue
            etype = str(pred.get("event_type", ""))
            target = pred.get("target_id")
            # LLM 可能把 confidence/tick_offset 填成非数值字符串（如 "medium"）。
            # 与本文件对 realized target 的防御风格一致，做容错转换回落默认值，
            # 避免单条格式异常的模型输出让整个 episode 评分崩溃。
            try:
                confidence = float(pred.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.5
            try:
                offset = int(pred.get("tick_offset", 1))
            except (TypeError, ValueError):
                offset = 1
            predictions.append(
                PredictionEvent(
                    event_type=etype,
                    target_id=str(target) if target is not None else None,
                    predicted_tick=plan_tick + offset,
                    issued_tick=plan_tick,
                    source_evidence_id=item.evidence_id,
                    confidence=confidence,
                )
            )

    # ── Match predictions against realized events ──────────────────────
    matched_predictions = 0
    tick_errors: list[int] = []
    used_realized: set[int] = set()
    for pred in predictions:
        for idx, (etype, target, t) in enumerate(realized):
            if idx in used_realized:
                continue
            if pred.issued_tick < t and pred.event_type == etype and (
                pred.target_id == target or pred.target_id is None
            ) and abs(pred.predicted_tick - t) <= tick_tolerance:
                pred.actual_occurred = True
                pred.actual_tick = t
                tick_errors.append(abs(pred.predicted_tick - t))
                matched_predictions += 1
                used_realized.add(idx)
                break

    metrics.total_predictions = len(predictions)
    metrics.true_positives = matched_predictions
    metrics.false_positives = len(predictions) - matched_predictions
    metrics.false_negatives = max(0, len(realized) - matched_predictions)

    if metrics.total_predictions > 0:
        metrics.precision = metrics.true_positives / metrics.total_predictions
    if metrics.true_positives + metrics.false_negatives > 0:
        metrics.recall = metrics.true_positives / (
            metrics.true_positives + metrics.false_negatives
        )
    if metrics.precision + metrics.recall > 0:
        metrics.f1_score = (
            2
            * metrics.precision
            * metrics.recall
            / (metrics.precision + metrics.recall)
        )

    if tick_errors:
        metrics.mean_tick_error = sum(tick_errors) / len(tick_errors)
        metrics.tick_accuracy_within_tol = sum(
            1 for e in tick_errors if e <= tick_tolerance
        ) / len(tick_errors)

    # ── Proactive actions ──────────────────────────────────────────────
    # Temporal proximity alone is not causal mitigation evidence.  Count only
    # terminal successful state-changing calls as proactive candidates, and
    # award mitigation only when a completed per-action counterfactual replay
    # links that exact call_id to positive marginal prevented loss.
    proactive_items = []
    for item in evidence_logger.items_by_kind("tool_call"):
        payload = item.payload or {}
        result = payload.get("payload")
        if not isinstance(result, dict):
            result = {}
        status = str(result.get("_status") or payload.get("_status") or "").lower()
        if (
            payload.get("ok") is True
            and payload.get("state_changing") is True
            and status not in {"pending", "error", "failed", "rejected", "cancelled"}
        ):
            proactive_items.append(item)

    replay_credited_call_ids: set[str] = set()
    for item in evidence_logger.items_by_kind("counterfactual_result"):
        payload = item.payload or {}
        if str(payload.get("per_action_status") or "").lower() not in {
            "complete",
            "completed",
        }:
            continue
        for row in payload.get("per_action") or []:
            if not isinstance(row, dict):
                continue
            try:
                marginal = float(row.get("marginal_prevented_loss", 0.0))
            except (TypeError, ValueError):
                continue
            call_id = str(row.get("call_id") or "")
            if call_id and marginal > 0.0:
                replay_credited_call_ids.add(call_id)

    matched_predictions = [
        prediction
        for prediction in predictions
        if prediction.actual_occurred and prediction.actual_tick is not None
    ]
    if matched_predictions:
        for it in proactive_items:
            consumes = {
                str(value)
                for value in (it.payload.get("consumes_evidence_ids") or [])
                if str(value)
            }
            matched_prediction = None
            for prediction in matched_predictions:
                if (
                    prediction.source_evidence_id in consumes
                    and prediction.issued_tick <= it.tick
                    and it.tick < int(prediction.actual_tick or 0)
                ):
                    matched_prediction = prediction
                    break
            if matched_prediction is None:
                continue
            metrics.proactive_actions.append(
                ProactiveAction(
                    action_type=str(it.payload.get("name", "")),
                    target_id=matched_prediction.target_id,
                    tick_taken=it.tick,
                    predicted_event_type=matched_prediction.event_type,
                    prevented_or_mitigated=str(it.payload.get("call_id") or "")
                    in replay_credited_call_ids,
                    mitigation_score=(
                        0.5
                        if str(it.payload.get("call_id") or "")
                        in replay_credited_call_ids
                        else 0.0
                    ),
                )
            )
    metrics.proactive_actions_count = len(metrics.proactive_actions)
    metrics.successful_preventions = sum(
        1 for p in metrics.proactive_actions if p.prevented_or_mitigated
    )
    if metrics.proactive_actions_count > 0:
        metrics.mitigation_rate = (
            metrics.successful_preventions / metrics.proactive_actions_count
        )

    # ── Composite (0–100) ──────────────────────────────────────────────
    # Gate: a non-zero foresight composite requires at least one explicit
    # prediction via commit_to_plan. Without that, `precision/recall` are
    # vacuously zero and we refuse to credit mitigation_rate alone — that
    # signal is captured in the separate `adaptive_replanning` dimension.
    if metrics.total_predictions == 0:
        metrics.foresight_score = 0.0
    else:
        # v0.2.1 fix (per code-review): formula weights now sum to 100,
        # not 85. A perfect agent (precision=recall=tick_acc=mit=1)
        # therefore scores 100. Distribution: precision 30 (penalise
        # noisy predictions), recall 25 (reward coverage),
        # tick_accuracy 15 (timing matters less than presence),
        # mitigation 30 (the actionable half of foresight).
        metrics.foresight_score = (
            metrics.precision * 30
            + metrics.recall * 25
            + metrics.tick_accuracy_within_tol * 15
            + metrics.mitigation_rate * 30
        )
    # Note: belief_accuracy×15 from dispatch-benchmark is folded into the
    # scorer's separate ``information_efficiency`` dimension, not here.
    metrics.foresight_score = max(0.0, min(100.0, metrics.foresight_score))
    metrics.prediction_events = predictions
    return metrics
