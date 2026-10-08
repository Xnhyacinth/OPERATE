"""Acceptance-based review delivery diagnostics; never a task score."""
from collections import Counter
from typing import Any


class PlanReviewLedger:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.latest: dict[str, Any] | None = None

    def replace(self, *, tick: int, plan_id: str | None, call_id: str | None,
                review_tick: int | None, expiry_tick: int | None) -> None:
        if self.latest is not None and self.latest["status"] == "scheduled":
            self.latest.update(status="superseded", resolved_tick=tick,
                               replacement_call_id=call_id)
        deadlines = [value for value in (review_tick, expiry_tick) if value is not None]
        self.latest = None
        if not deadlines:
            return
        self.latest = {
            "review_id": f"review-{len(self.records) + 1}",
            "plan_id": plan_id, "call_id": call_id, "accepted_tick": tick,
            "requested_review_tick": review_tick, "plan_expires_at_tick": expiry_tick,
            "due_tick": min(deadlines), "status": "scheduled",
        }
        self.records.append(self.latest)

    def visible(self, tick: int) -> dict[str, Any] | None:
        if self.latest is None:
            return None
        result = dict(self.latest)
        if result["status"] == "scheduled" and tick >= result["due_tick"]:
            result["status"] = "due"
        return result

    def resolve_due(self, *, tick: int, delivered: bool, reasons: list[str],
                    budget_blocked: bool = False) -> None:
        if (self.latest is not None and self.latest["status"] == "scheduled"
                and tick >= self.latest["due_tick"]):
            self.latest.update(
                status=("due_served" if delivered else
                        "budget_blocked" if budget_blocked else "harness_missed"),
                resolved_tick=tick, decision_reasons=list(reasons),
            )

    def finish(self, tick: int) -> None:
        if self.latest is not None and self.latest["status"] == "scheduled":
            self.latest.update(status="terminal_censored", resolved_tick=tick)

    def summary(self) -> dict[str, Any]:
        counts = Counter(row["status"] for row in self.records)
        eligible = counts["due_served"] + counts["harness_missed"]
        return {
            "schema_version": "plan_review_lifecycle.v1",
            "scheduled": len(self.records), "outcomes": dict(counts),
            "delivery_eligible": eligible,
            "served_rate": counts["due_served"] / eligible if eligible else None,
            "delivery_semantics": "decision_invoked_not_model_success",
            "records": self.records,
        }
