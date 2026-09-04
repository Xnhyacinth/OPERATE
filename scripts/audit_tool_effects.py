#!/usr/bin/env python3
"""Audit whether batch tool use exposes protocol, evidence, and outcome effects."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluation.action_taxonomy import CONTROL_TOOL_NAMES  # noqa: E402
from scripts.analyze_decision_impact import (  # noqa: E402
    _backend_from_row,  # noqa: E402
    _domain_from_row,  # noqa: E402
)

EXAMPLE_LIMIT = 20

PROTOCOL_ERROR_BUCKETS = {
    "TICK_BUDGET_EXHAUSTED": "budget",
    "TICK_COST_BUDGET_EXHAUSTED": "budget",
    "EPISODE_BUDGET_EXHAUSTED": "budget",
    "DUPLICATE_SUPPRESSED": "duplicate_suppression",
    "COOLDOWN": "cooldown",
    "INJECTED_FAILURE": "injected_failure",
    "VALIDATION_ERROR": "validation",
    "DOMAIN_REJECTED": "domain_rejected",
    "IDEMPOTENCY_KEY_CONFLICT": "idempotency_conflict",
    "UNSUPPORTED_TOOL_EFFECT": "no_effect_or_unsupported",
    "OUT_OF_RANGE_TOOL_EFFECT": "no_effect_or_unsupported",
    "NO_EFFECT": "no_effect_or_unsupported",
    "HANDLER_EXCEPTION": "handler_exception",
    "UNKNOWN_TOOL": "unknown_tool",
}

TOOL_SCORE_DIMENSION_CANDIDATES = {
    "place_replenishment_order": {"adaptive_replanning", "tool_use_efficiency"},
    "dispatch_job_operation": {"adaptive_replanning", "tool_use_efficiency"},
    "dispatch_ready_operations": {"adaptive_replanning", "tool_use_efficiency"},
    "commit_reserve": {"adaptive_replanning", "stakeholder_management"},
    "request_mutual_aid": {"adaptive_replanning", "stakeholder_management"},
    "change_signal_plan": {"adaptive_replanning", "stakeholder_management"},
}


def classify_idempotency_conflict(
    previous: dict[str, Any], current: dict[str, Any]
) -> str:
    # Mirrors ``core.tool_protocol._idempotency_conflict``, which compares
    # the logical call signature ``(name, args)`` -- NOT tick. A re-dispatch
    # of the SAME logical call on a later tick (e.g. an oracle re-attempting
    # an operation after INJECTED_FAILURE/COOLDOWN, or any agent retrying
    # across ticks) is a benign duplicate_retry, not a conflict. Including
    # ``tick`` here previously mis-flagged every cross-tick retry as a
    # ``true_conflict``, producing tens of thousands of false BLOCKING
    # issues (one per re-dispatched operation per tick -- observed on the
    # oracle's ``orc_jobshop_{job_id}_{op_index}`` key, which intentionally
    # omits a tick discriminator because each operation is a unique logical
    # call that may legitimately be re-attempted).
    same_name = previous.get("name") == current.get("name")
    same_args = previous.get("args") == current.get("args")
    if same_name and same_args:
        return "duplicate_retry"
    return "true_conflict"


def classify_state_changing_score_evidence(
    tool_name: str, dimensions: list[dict[str, Any]], result: dict[str, Any]
) -> str:
    evidence_id = result.get("evidence_id")
    if not evidence_id:
        return "diagnostic_state_change_not_scored"
    for dim in dimensions:
        if evidence_id in set(dim.get("evidence_ids") or []):
            return "score_evidence_present"
    candidate_dims = TOOL_SCORE_DIMENSION_CANDIDATES.get(tool_name, set())
    for dim in dimensions:
        if dim.get("applicable") and dim.get("name") in candidate_dims:
            return "expected_score_evidence_missing"
    return "not_expected_for_score"


def _empty_bucket() -> dict[str, int]:
    return {
        "episodes": 0,
        "episodes_with_trajectory": 0,
        "trajectory_missing": 0,
        "action_tool_calls": 0,
        "control_action_calls": 0,
        "tool_results": 0,
        "tool_results_ok": 0,
        "tool_results_failed": 0,
        "state_changing_results": 0,
        "state_changing_ok": 0,
        "state_changing_ok_with_evidence": 0,
        "state_changing_ok_without_evidence": 0,
        "state_changing_evidence_not_in_step": 0,
        "delayed_state_changing_evidence_from_prior_step": 0,
        "state_changing_evidence_not_in_score": 0,
        "state_changing_outcome_changed": 0,
        "state_changing_cost_matched": 0,
        "missing_idempotency_key": 0,
        "idempotency_key_conflicts": 0,
        "duplicate_retry": 0,
        "rationale_missing": 0,
        "protocol_metadata_missing": 0,
        "budget_rejections": 0,
        "duplicate_suppressed": 0,
        "cooldowns": 0,
        "validation_errors": 0,
        "domain_rejections": 0,
        "no_effect_or_unsupported": 0,
        "delayed_or_pending_results": 0,
        "expected_score_evidence_missing": 0,
        "diagnostic_state_change_not_scored": 0,
        "not_expected_for_score": 0,
        "score_evidence_present": 0,
        "blocking_issues": 0,
        "warnings": 0,
    }


def _bump(bucket: dict[str, int], key: str, amount: int = 1) -> None:
    bucket[key] = int(bucket.get(key, 0)) + amount


def _add_example(
    examples: dict[str, list[dict[str, Any]]],
    kind: str,
    row: dict[str, Any],
    *,
    domain: str,
    backend: str,
    detail: dict[str, Any] | None = None,
) -> None:
    bucket = examples.setdefault(kind, [])
    if len(bucket) >= EXAMPLE_LIMIT:
        return
    item: dict[str, Any] = {
        "scenario_id": row.get("scenario_id"),
        "scenario_slug": row.get("scenario_slug"),
        "model": row.get("model", row.get("agent_name")),
        "seed": row.get("seed"),
        "family": row.get("family"),
        "domain": domain,
        "backend_kind": backend,
    }
    if detail:
        item.update(detail)
    bucket.append(item)


def _dimensions(score: dict[str, Any]) -> list[dict[str, Any]]:
    dims = score.get("dimensions")
    if isinstance(dims, list):
        return [d for d in dims if isinstance(d, dict)]
    if isinstance(dims, dict):
        out: list[dict[str, Any]] = []
        for name, body in dims.items():
            if isinstance(body, dict):
                item = dict(body)
                item.setdefault("name", name)
                out.append(item)
        return out
    return []


def _score_evidence_ids(row: dict[str, Any]) -> set[str]:
    score = row.get("score") or {}
    if not isinstance(score, dict):
        return set()
    ids: set[str] = set()
    for dim in _dimensions(score):
        for ev in dim.get("evidence_ids") or []:
            if isinstance(ev, str) and ev:
                ids.add(ev)
    return ids


def _trajectory_file(row: dict[str, Any]) -> Path | None:
    traj = row.get("trajectory_summary") or {}
    raw = traj.get("trajectory_path") if isinstance(traj, dict) else None
    if not raw:
        return None
    base = Path(str(raw))
    candidates = [base]
    if not str(base).endswith(".trajectory.jsonl"):
        candidates.append(Path(str(base) + ".trajectory.jsonl"))
    if base.suffix:
        candidates.append(base.with_suffix(".trajectory.jsonl"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[1] if len(candidates) > 1 else base


def _load_trajectory(path: Path) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    rows: list[dict[str, Any]] = []
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for idx, line in enumerate(raw_lines):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            if idx == len(raw_lines) - 1:
                break
            raise
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _iter_actions(entry: dict[str, Any]) -> list[dict[str, Any]]:
    action = entry.get("action") or {}
    if not isinstance(action, dict):
        return []
    actions = action.get("actions") or []
    return [a for a in actions if isinstance(a, dict)]


def _iter_results(entry: dict[str, Any]) -> list[dict[str, Any]]:
    results = entry.get("tool_results") or []
    return [r for r in results if isinstance(r, dict)]


def _issue_all(
    buckets: list[dict[str, int]],
    key: str,
    *,
    blocking: bool = False,
    warning: bool = False,
) -> None:
    for bucket in buckets:
        _bump(bucket, key)
        if blocking:
            _bump(bucket, "blocking_issues")
        if warning:
            _bump(bucket, "warnings")


def _error_bucket(error_code: str | None) -> str | None:
    if not error_code:
        return None
    return PROTOCOL_ERROR_BUCKETS.get(str(error_code))


def _apply_error_rollup(buckets: list[dict[str, int]], error_code: str | None) -> None:
    kind = _error_bucket(error_code)
    if kind == "budget":
        _issue_all(buckets, "budget_rejections", warning=True)
    elif kind == "duplicate_suppression":
        _issue_all(buckets, "duplicate_suppressed", warning=True)
    elif kind == "idempotency_conflict":
        _issue_all(buckets, "idempotency_key_conflicts", blocking=True)
    elif kind == "cooldown":
        _issue_all(buckets, "cooldowns", warning=True)
    elif kind == "validation":
        _issue_all(buckets, "validation_errors", warning=True)
    elif kind == "domain_rejected":
        _issue_all(buckets, "domain_rejections", warning=True)
    elif kind == "no_effect_or_unsupported":
        _issue_all(buckets, "no_effect_or_unsupported", warning=True)
    elif kind in {"injected_failure", "handler_exception", "unknown_tool"}:
        for bucket in buckets:
            _bump(bucket, "warnings")


def _audit_row(
    row: dict[str, Any],
    *,
    by_domain: dict[str, dict[str, int]],
    by_domain_backend: dict[str, dict[str, dict[str, int]]],
    by_family: dict[str, dict[str, int]],
    examples: dict[str, list[dict[str, Any]]],
) -> Counter[str]:
    rollup: Counter[str] = Counter()
    domain = _domain_from_row(row)
    backend = _backend_from_row(row)
    family = str(row.get("family") or "unknown")
    buckets = [by_domain[domain], by_domain_backend[domain][backend], by_family[family]]
    for bucket in buckets:
        _bump(bucket, "episodes")

    score = row.get("score") or {}
    dimensions = _dimensions(score) if isinstance(score, dict) else []
    traj_path = _trajectory_file(row)
    entries = _load_trajectory(traj_path) if traj_path is not None else None
    if entries is None:
        rollup["trajectory_missing"] += 1
        _issue_all(buckets, "trajectory_missing", warning=True)
        _add_example(
            examples,
            "trajectory_missing",
            row,
            domain=domain,
            backend=backend,
            detail={"trajectory_path": str(traj_path) if traj_path else None},
        )
        return rollup

    for bucket in buckets:
        _bump(bucket, "episodes_with_trajectory")

    idem_seen: dict[str, dict[str, Any]] = {}
    row_has_state_change = False
    prior_step_evidence: set[str] = set()
    row_outcome_changed = bool(
        (row.get("decision_impact") or {}).get("outcome_changed")
    )

    for entry in entries:
        step_evidence = {
            ev for ev in (entry.get("evidence_ids") or []) if isinstance(ev, str) and ev
        }
        for action in _iter_actions(entry):
            name = str(action.get("name") or "")
            _issue_all(buckets, "action_tool_calls")
            if name in CONTROL_TOOL_NAMES:
                _issue_all(buckets, "control_action_calls")
            idem = action.get("idempotency_key")
            if not isinstance(idem, str) or not idem:
                rollup["missing_idempotency_key"] += 1
                _issue_all(buckets, "missing_idempotency_key", blocking=True)
                _add_example(
                    examples,
                    "missing_idempotency_key",
                    row,
                    domain=domain,
                    backend=backend,
                    detail={"tick": entry.get("tick"), "tool": name},
                )
            else:
                prior = idem_seen.get(idem)
                current_call = {
                    "tick": entry.get("tick"),
                    "idempotency_key": idem,
                    "name": name,
                    "args": action.get("args") or {},
                }
                if prior is not None:
                    classification = classify_idempotency_conflict(prior, current_call)
                    if classification == "true_conflict":
                        rollup["idempotency_key_conflicts"] += 1
                        _issue_all(buckets, "idempotency_key_conflicts", blocking=True)
                        _add_example(
                            examples,
                            "idempotency_key_conflicts",
                            row,
                            domain=domain,
                            backend=backend,
                            detail={
                                "idempotency_key": idem,
                                "prior": {
                                    "name": prior.get("name"),
                                    "args": prior.get("args"),
                                    "tick": prior.get("tick"),
                                },
                                "current": {
                                    "name": current_call.get("name"),
                                    "args": current_call.get("args"),
                                    "tick": current_call.get("tick"),
                                },
                            },
                        )
                    else:
                        rollup["duplicate_retry"] += 1
                        _issue_all(buckets, "duplicate_retry")
                else:
                    idem_seen[idem] = current_call

            # P1-2: non-blocking warning when a non-wait action carries no
            # rationale. ``rationale`` lives on the ``Action`` (entry-level
            # ``entry["action"]["rationale"]``), NOT on the per-tool-call
            # dicts iterated here (``ToolCall.rationale`` is always None by
            # design — see core/pomdp.py). Check it once per entry, outside
            # the per-tool-call loop, so the signal fires when the agent
            # omitted a rationale for the action rather than on every call.
            # (Handled below, after the action loop, via the entry action.)

        # P1-2 (entry-level): a non-wait Action with no rationale means
        # failure-recipe mining cannot reason about *why* the agent acted.
        # Non-blocking — metadata only.
        entry_action = entry.get("action") or {}
        if isinstance(entry_action, dict):
            dominant = entry_action.get("dominant_action") or ""
            if dominant not in ("wait", "noop", "") and entry_action.get("rationale") is None:
                rollup["rationale_missing"] += 1
                _issue_all(buckets, "rationale_missing")

        for result in _iter_results(entry):
            _issue_all(buckets, "tool_results")
            if "state_changing" not in result or "idempotency_key" not in result:
                rollup["protocol_metadata_missing"] += 1
                _issue_all(buckets, "protocol_metadata_missing", blocking=True)
                _add_example(
                    examples,
                    "protocol_metadata_missing",
                    row,
                    domain=domain,
                    backend=backend,
                    detail={"tick": entry.get("tick"), "tool": result.get("name")},
                )
            ok = bool(result.get("ok"))
            if ok:
                _issue_all(buckets, "tool_results_ok")
            else:
                _issue_all(buckets, "tool_results_failed", warning=True)
                _apply_error_rollup(buckets, result.get("error_code"))
            payload = result.get("payload") or {}
            if int(result.get("latency_ticks") or 0) > 0 or (
                isinstance(payload, dict) and payload.get("_status") == "pending"
            ):
                _issue_all(buckets, "delayed_or_pending_results", warning=True)
            if bool(result.get("state_changing")):
                row_has_state_change = True
                _issue_all(buckets, "state_changing_results")
                if ok:
                    _issue_all(buckets, "state_changing_ok")
                    evidence_id = result.get("evidence_id")
                    if isinstance(evidence_id, str) and evidence_id:
                        _issue_all(buckets, "state_changing_ok_with_evidence")
                        if (
                            evidence_id not in step_evidence
                            and int(result.get("latency_ticks") or 0) > 0
                            and evidence_id in prior_step_evidence
                        ):
                            _issue_all(
                                buckets,
                                "delayed_state_changing_evidence_from_prior_step",
                            )
                        elif evidence_id not in step_evidence:
                            rollup["state_changing_evidence_not_in_step"] += 1
                            _issue_all(
                                buckets,
                                "state_changing_evidence_not_in_step",
                                blocking=True,
                            )
                            _add_example(
                                examples,
                                "state_changing_evidence_not_in_step",
                                row,
                                domain=domain,
                                backend=backend,
                                detail={
                                    "tick": entry.get("tick"),
                                    "tool": result.get("name"),
                                    "evidence_id": evidence_id,
                                },
                            )
                        classification = classify_state_changing_score_evidence(
                            str(result.get("name") or ""), dimensions, result
                        )
                        for bucket in buckets:
                            _bump(bucket, classification)
                        if classification == "expected_score_evidence_missing":
                            rollup["state_changing_evidence_not_in_score"] += 1
                            _issue_all(
                                buckets,
                                "state_changing_evidence_not_in_score",
                                warning=True,
                            )
                            _add_example(
                                examples,
                                "state_changing_evidence_not_in_score",
                                row,
                                domain=domain,
                                backend=backend,
                                detail={
                                    "tick": entry.get("tick"),
                                    "tool": result.get("name"),
                                    "evidence_id": evidence_id,
                                },
                            )
                    else:
                        classification = classify_state_changing_score_evidence(
                            str(result.get("name") or ""), dimensions, result
                        )
                        for bucket in buckets:
                            _bump(bucket, classification)
                        rollup["state_changing_ok_without_evidence"] += 1
                        _issue_all(
                            buckets,
                            "state_changing_ok_without_evidence",
                            blocking=True,
                        )
                        _add_example(
                            examples,
                            "state_changing_ok_without_evidence",
                            row,
                            domain=domain,
                            backend=backend,
                            detail={
                                "tick": entry.get("tick"),
                                "tool": result.get("name"),
                            },
                        )
        prior_step_evidence.update(step_evidence)

    if row_has_state_change:
        if row_outcome_changed:
            _issue_all(buckets, "state_changing_outcome_changed")
        else:
            _issue_all(buckets, "state_changing_cost_matched", warning=True)
            _add_example(
                examples,
                "state_changing_cost_matched",
                row,
                domain=domain,
                backend=backend,
                detail={
                    "prevented_loss": (row.get("counterfactual") or {}).get(
                        "prevented_loss"
                    )
                },
            )
    return rollup


def build_report(
    rows: list[dict[str, Any]], *, expected_domains: list[str] | None = None
) -> dict[str, Any]:
    ok_rows = [r for r in rows if r.get("status", "ok") == "ok"]
    by_domain: dict[str, dict[str, int]] = defaultdict(_empty_bucket)
    by_domain_backend: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(_empty_bucket)
    )
    by_family: dict[str, dict[str, int]] = defaultdict(_empty_bucket)
    examples: dict[str, list[dict[str, Any]]] = {}

    for row in ok_rows:
        _audit_row(
            row,
            by_domain=by_domain,
            by_domain_backend=by_domain_backend,
            by_family=by_family,
            examples=examples,
        )

    totals = _empty_bucket()
    for bucket in by_domain.values():
        for key, value in bucket.items():
            totals[key] = int(totals.get(key, 0)) + int(value)

    missing_expected_domains: list[str] = []
    blockers: list[str] = []
    warnings: list[str] = []
    if expected_domains:
        seen = set(by_domain)
        missing_expected_domains = sorted(d for d in expected_domains if d not in seen)
        if missing_expected_domains:
            blockers.append(
                "missing expected domain coverage: "
                + ", ".join(missing_expected_domains)
            )

    if totals["trajectory_missing"]:
        warnings.append(
            f"{totals['trajectory_missing']} ok episode(s) lack trajectory files"
        )
    if totals["state_changing_cost_matched"]:
        warnings.append(
            f"{totals['state_changing_cost_matched']} state-changing episode bucket(s) did not change counterfactual outcome"
        )
    if totals["tool_results_failed"]:
        warnings.append(
            f"{totals['tool_results_failed']} failed tool result(s) observed"
        )
    if totals["state_changing_evidence_not_in_score"]:
        warnings.append(
            f"{totals['state_changing_evidence_not_in_score']} state-changing evidence id(s) were not consumed by score dimensions"
        )

    n_blocking = int(totals["blocking_issues"]) + len(blockers)
    n_warnings = int(totals["warnings"]) + len(warnings)
    status = "blocking" if n_blocking else ("warning" if n_warnings else "pass")

    audit = {
        "schema_version": "0.1",
        "status": status,
        "n_total": len(rows),
        "n_ok": len(ok_rows),
        "totals": totals,
        "expected_domains": list(expected_domains or []),
        "missing_expected_domains": missing_expected_domains,
        "blockers": blockers,
        "warnings": warnings,
        "examples": examples,
        "by_domain": dict(sorted(by_domain.items())),
        "by_domain_backend": {
            domain: dict(sorted(backends.items()))
            for domain, backends in sorted(by_domain_backend.items())
        },
        "by_family": dict(sorted(by_family.items())),
    }
    return {
        "tool_effect_audit": audit,
        "notes": [
            "This audit is report-only; it does not change scorer semantics or release artifacts.",
            "Trajectory files prove the tool-protocol surface: idempotency_key, state_changing, error_code, latency_ticks, and evidence_id.",
            "Immediate state-changing results must appear in the same step evidence_ids list; delayed materializations may link evidence from the original prior step.",
            "Evidence ids not consumed by score dimensions are warnings because some tool evidence is diagnostic/contextual rather than scorer-facing.",
        ],
    }


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for idx, line in enumerate(raw_lines):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            if idx == len(raw_lines) - 1:
                break
            raise
        if isinstance(item, dict):
            rows.append(item)
    return rows


def write_markdown(report: dict[str, Any], out_path: Path) -> None:
    audit = report.get("tool_effect_audit") or {}
    totals = audit.get("totals") or {}
    lines = [
        "# Tool Effect Audit",
        "",
        "Checks whether batch trajectories expose tool-protocol metadata, evidence links, and outcome effects.",
        "",
        f"- Status: **{audit.get('status', 'unknown')}**",
        f"- Episodes OK: **{audit.get('n_ok', 0)}** / {audit.get('n_total', 0)}",
        f"- Episodes with trajectory: **{totals.get('episodes_with_trajectory', 0)}**",
        f"- Trajectory missing: **{totals.get('trajectory_missing', 0)}**",
        f"- Action tool calls: **{totals.get('action_tool_calls', 0)}**",
        f"- Tool results OK / failed: **{totals.get('tool_results_ok', 0)}** / **{totals.get('tool_results_failed', 0)}**",
        f"- State-changing OK with evidence: **{totals.get('state_changing_ok_with_evidence', 0)}**",
        f"- Blocking issues: **{totals.get('blocking_issues', 0)}**",
        f"- Warnings: **{totals.get('warnings', 0)}**",
        "",
        "## Issue Counts",
        "",
        f"- Missing idempotency keys: **{totals.get('missing_idempotency_key', 0)}**",
        f"- Idempotency key conflicts: **{totals.get('idempotency_key_conflicts', 0)}**",
        f"- Protocol metadata missing: **{totals.get('protocol_metadata_missing', 0)}**",
        f"- State-changing OK without evidence: **{totals.get('state_changing_ok_without_evidence', 0)}**",
        f"- State-changing evidence not in step: **{totals.get('state_changing_evidence_not_in_step', 0)}**",
        f"- Delayed evidence linked from prior step: **{totals.get('delayed_state_changing_evidence_from_prior_step', 0)}**",
        f"- State-changing evidence not in score: **{totals.get('state_changing_evidence_not_in_score', 0)}**",
        f"- Budget rejections: **{totals.get('budget_rejections', 0)}**",
        f"- Duplicate suppressed: **{totals.get('duplicate_suppressed', 0)}**",
        f"- No-effect / unsupported: **{totals.get('no_effect_or_unsupported', 0)}**",
        f"- Delayed / pending results: **{totals.get('delayed_or_pending_results', 0)}**",
    ]
    if audit.get("missing_expected_domains"):
        lines.append(
            f"- Missing expected domains: {', '.join(audit['missing_expected_domains'])}"
        )
    by_domain = audit.get("by_domain") or {}
    if by_domain:
        lines.extend(["", "## By Domain", ""])
        for domain, bucket in sorted(by_domain.items()):
            lines.append(
                f"- `{domain}`: episodes={bucket.get('episodes', 0)}, "
                f"state_changing_ok={bucket.get('state_changing_ok', 0)}, "
                f"blocking={bucket.get('blocking_issues', 0)}, "
                f"warnings={bucket.get('warnings', 0)}"
            )
    examples = audit.get("examples") or {}
    if examples:
        lines.extend(["", "## Examples", ""])
        for kind, rows in sorted(examples.items()):
            lines.append(f"### {kind}")
            for row in rows[:5]:
                lines.append(
                    f"- `{row.get('scenario_id')}` ({row.get('domain')}/{row.get('backend_kind')})"
                )
            lines.append("")
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--expected-domains",
        default="",
        help="Optional comma-separated domain coverage list for audit status.",
    )
    args = parser.parse_args()
    out_dir = args.output_dir.resolve()
    ep_path = out_dir / "episodes.jsonl"
    if not ep_path.is_file():
        print(f"missing {ep_path}", file=sys.stderr)
        return 1
    expected_domains = [
        item.strip() for item in args.expected_domains.split(",") if item.strip()
    ]
    report = build_report(
        _load_jsonl_rows(ep_path), expected_domains=expected_domains or None
    )
    (out_dir / "tool_effect_audit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_markdown(report, out_dir / "TOOL_EFFECT_AUDIT.md")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
