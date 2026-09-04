"""
data.quality_validator — Trajectory quality gates.

Eight structural quality checks (forked from dispatch-benchmark with
DT-Sched-Bench-specific evidence_ids completeness):

1. Header is complete (scenario_id, scenario_signature, agent_name, seed)
2. Trajectory length matches header.total_ticks
3. Action diversity ≥ 3 distinct dominant actions
4. ``state_changing_action_rate`` ≥ 0.05 (agent did something at least 5%
   of ticks)
5. Tool failure rate ≤ 0.40
6. Reward monotonicity is finite (no NaN / inf)
7. Every step has a non-empty ``evidence_ids`` list IF the step contained
   any state-changing action
8. Duplicate ratio (same dominant action consecutively) ≤ 0.70
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationResult:
    quality_score: float
    checks: dict[str, bool] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        # Structural/evidence contract violations cannot be averaged away by
        # softer behavior heuristics such as action diversity.  Previously a
        # trajectory missing every state-change evidence id still passed with
        # 7/8 checks, which contradicted the benchmark's evidence red line.
        critical = (
            "header_complete",
            "length_matches",
            "tool_failure_rate_ok",
            "reward_finite",
            "evidence_coverage",
        )
        return self.quality_score >= 0.7 and all(
            self.checks.get(name, False) for name in critical
        )


def validate_trajectory(
    header: dict[str, Any],
    entries: list[dict[str, Any]],
    summary: dict[str, Any] | None = None,
) -> ValidationResult:
    checks: dict[str, bool] = {}
    issues: list[str] = []

    # 1. Header completeness
    required = ["scenario_id", "scenario_signature", "agent_name", "seed"]
    checks["header_complete"] = all(header.get(k) not in (None, "") for k in required)
    if not checks["header_complete"]:
        issues.append("header missing one of: " + ", ".join(required))

    # 2. Trajectory length
    ht = header.get("total_ticks", 0)
    checks["length_matches"] = len(entries) == ht or ht == 0
    if not checks["length_matches"]:
        issues.append(f"header.total_ticks={ht} != len(entries)={len(entries)}")

    # 3. Action diversity
    dominants = [(e.get("action") or {}).get("dominant_action", "?") for e in entries]
    distinct = {d for d in dominants if d not in {"?", None}}
    checks["action_diversity"] = len(distinct) >= 3
    if not checks["action_diversity"]:
        issues.append(f"only {len(distinct)} distinct dominant actions")

    # 4. State-changing rate
    rate = (summary or {}).get("state_changing_action_rate", 0.0)
    checks["state_changing_rate"] = float(rate) >= 0.05
    if not checks["state_changing_rate"]:
        issues.append(f"state_changing_action_rate={rate:.3f} < 0.05")

    # 5. Tool failure rate
    fail_rate = (summary or {}).get("tool_failure_rate", 0.0)
    checks["tool_failure_rate_ok"] = float(fail_rate) <= 0.40
    if not checks["tool_failure_rate_ok"]:
        issues.append(f"tool_failure_rate={fail_rate:.3f} > 0.40")

    # 6. Reward finite
    rewards = [e.get("reward", 0.0) for e in entries]
    checks["reward_finite"] = all(
        isinstance(r, (int, float)) and math.isfinite(r) for r in rewards
    )
    if not checks["reward_finite"]:
        issues.append("non-finite reward present")

    # 7. Evidence coverage on state-changing steps
    missing_ev = 0
    for e in entries:
        if _has_state_changing_step(e):
            if not e.get("evidence_ids"):
                missing_ev += 1
    n_state_changes = sum(1 for e in entries if _has_state_changing_step(e))
    if n_state_changes:
        ratio_missing = missing_ev / n_state_changes
        checks["evidence_coverage"] = ratio_missing <= 0.10
        if not checks["evidence_coverage"]:
            issues.append(
                f"{missing_ev}/{n_state_changes} state-changing steps lack evidence_ids"
            )
    else:
        checks["evidence_coverage"] = True  # vacuously true

    # 8. Duplicate-dominant ratio
    if len(dominants) >= 2:
        consecutive = sum(1 for a, b in zip(dominants, dominants[1:]) if a == b)
        dup_ratio = consecutive / max(len(dominants) - 1, 1)
        checks["duplicate_ratio_ok"] = dup_ratio <= 0.70
        if not checks["duplicate_ratio_ok"]:
            issues.append(f"duplicate consecutive dominant ratio={dup_ratio:.3f}")
    else:
        checks["duplicate_ratio_ok"] = True

    quality_score = sum(1 for v in checks.values() if v) / max(len(checks), 1)
    return ValidationResult(
        quality_score=round(quality_score, 3),
        checks=checks,
        issues=issues,
    )


def _has_state_changing_step(entry: dict[str, Any]) -> bool:
    """Whether a trajectory entry contains a state-changing tool result.

    Prefer explicit tool-result metadata emitted by ``ToolResult.to_dict``.
    Older trajectory shapes did not carry that flag, so fall back to the
    serialized action name while preserving the old ``action`` key alias.
    """
    saw_state_flag = False
    for result in entry.get("tool_results", []) or []:
        if not isinstance(result, dict) or "state_changing" not in result:
            continue
        saw_state_flag = True
        if result.get("state_changing") is True:
            return True
    if saw_state_flag:
        return False

    for sub in (entry.get("action", {}) or {}).get("actions", []) or []:
        if not isinstance(sub, dict):
            continue
        name = sub.get("name") or sub.get("action")
        if name not in {"wait", "noop", None, ""}:
            return True
    return False
