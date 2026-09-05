#!/usr/bin/env python3
"""Mine cross-domain agent failure recipes from batch episodes and trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluation.trajectory_paths import trajectory_file as _trajectory_file  # noqa: E402
from evaluation.action_taxonomy import CONTROL_TOOL_NAMES  # noqa: E402
from evaluation.score_evidence import native_effect_is_scored  # noqa: E402
from scripts.analyze_decision_impact import (  # noqa: E402
    _backend_from_row,  # noqa: E402
    _domain_from_row,  # noqa: E402
    _episode_impact,  # noqa: E402
)

EXAMPLE_LIMIT = 20
INFORMATION_TOOL_PREFIXES = (
    "investigate",
    "query",
    "forecast",
    "inspect",
    "measure",
    "diagnose",
)
BLOCKING_PROTOCOL_CODES = {
    "IDEMPOTENCY_KEY_CONFLICT",
    "HANDLER_EXCEPTION",
    "UNKNOWN_TOOL",
}

RECIPE_KEYS = (
    "episodes",
    "episodes_with_trajectory",
    "legacy_missing_trajectory",
    "harmful_control_action",
    "cost_matched_control",
    "missing_decision_impact",
    "counterfactual_missing_or_not_applicable",
    "tool_protocol_failure",
    "tool_result_failed",
    "state_changing_without_evidence",
    "state_changing_evidence_not_in_score",
    "stale_info_unconsumed",
    "stale_info_missing_score_evidence",
    "blocking_issues",
    "warnings",
)

DRILLDOWN_BUCKETS = (
    "by_tool",
    "by_family",
    "by_domain",
    "by_backend",
    "by_model",
    "by_likely_dimension",
)


def _empty_bucket() -> dict[str, int]:
    return {key: 0 for key in RECIPE_KEYS}


def _empty_gap_bucket() -> dict[str, int]:
    return {"events": 0, "missing_score_evidence_ids": 0}


def _bump(bucket: dict[str, int], key: str, amount: int = 1) -> None:
    bucket[key] = int(bucket.get(key, 0)) + amount


def _bump_all(buckets: list[dict[str, int]], key: str, amount: int = 1) -> None:
    for bucket in buckets:
        _bump(bucket, key, amount)


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


def _information_evidence_ids(row: dict[str, Any]) -> set[str]:
    score = row.get("score") or {}
    if not isinstance(score, dict):
        return set()
    ids: set[str] = set()
    for dim in _dimensions(score):
        if str(dim.get("name") or "") != "information_efficiency":
            continue
        for ev in dim.get("evidence_ids") or []:
            if isinstance(ev, str) and ev:
                ids.add(ev)
    return ids


def _evidence_ids_from_result(result: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for key in ("evidence_id", "evidence_ids"):
        value = result.get(key)
        if isinstance(value, str) and value:
            ids.add(value)
        elif isinstance(value, list):
            ids.update(item for item in value if isinstance(item, str) and item)
    payload = result.get("payload")
    if isinstance(payload, dict):
        value = payload.get("evidence_id")
        if isinstance(value, str) and value:
            ids.add(value)
        values = payload.get("evidence_ids")
        if isinstance(values, list):
            ids.update(item for item in values if isinstance(item, str) and item)
    return ids


def _safe_json_blob(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False).lower()
    except TypeError:
        return str(value).lower()


def _is_information_tool(name: Any) -> bool:
    text = str(name or "").lower()
    return text.startswith(INFORMATION_TOOL_PREFIXES)


def _stale_attrs(entry: dict[str, Any], entry_index: int) -> list[dict[str, Any]]:
    observation = entry.get("observation") or {}
    entities = observation.get("entities") if isinstance(observation, dict) else None
    if not isinstance(entities, dict):
        return []
    out: list[dict[str, Any]] = []
    for entity_id, entity in entities.items():
        if not isinstance(entity, dict):
            continue
        stale = entity.get("_stale_attrs")
        if not isinstance(stale, dict):
            continue
        for attr, ticks in sorted(stale.items()):
            try:
                staleness_ticks = int(ticks)
            except (TypeError, ValueError):
                staleness_ticks = None
            out.append(
                {
                    "entry_index": entry_index,
                    "tick": entry.get("tick"),
                    "entity_id": str(entity_id),
                    "attr": str(attr),
                    "staleness_ticks": staleness_ticks,
                }
            )
    return out


def _stale_consumption(
    stale: dict[str, Any],
    entries: list[dict[str, Any]],
    info_score_evidence: set[str],
) -> tuple[bool, bool, dict[str, Any] | None]:
    entity = str(stale.get("entity_id") or "").lower()
    attr = str(stale.get("attr") or "").lower()
    start = int(stale.get("entry_index") or 0)
    for entry in entries[start:]:
        for action in _iter_actions(entry):
            if not _is_information_tool(action.get("name")):
                continue
            blob = _safe_json_blob(action)
            if entity in blob and attr in blob:
                return (
                    True,
                    False,
                    {"tick": entry.get("tick"), "tool": action.get("name")},
                )
        step_evidence_ids = {
            ev for ev in (entry.get("evidence_ids") or []) if isinstance(ev, str) and ev
        }
        for result in _iter_results(entry):
            if not _is_information_tool(result.get("name")):
                continue
            blob = _safe_json_blob(result)
            if entity not in blob or attr not in blob:
                continue
            evidence_ids = _evidence_ids_from_result(result) | step_evidence_ids
            return (
                True,
                bool(info_score_evidence.intersection(evidence_ids)),
                {
                    "tick": entry.get("tick"),
                    "tool": result.get("name"),
                    "evidence_ids": sorted(evidence_ids),
                },
            )
    return False, False, None


def _example_base(row: dict[str, Any], *, domain: str, backend: str) -> dict[str, Any]:
    return {
        "scenario_id": row.get("scenario_id"),
        "scenario_slug": row.get("scenario_slug"),
        "model": row.get("model", row.get("agent_name")),
        "seed": row.get("seed"),
        "family": row.get("family"),
        "domain": domain,
        "backend_kind": backend,
    }


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
    item = _example_base(row, domain=domain, backend=backend)
    item["classification"] = kind
    if detail:
        item.update(detail)
    bucket.append(item)


def _record_recipe(
    *,
    kind: str,
    row: dict[str, Any],
    domain: str,
    backend: str,
    buckets: list[dict[str, int]],
    examples: dict[str, list[dict[str, Any]]],
    detail: dict[str, Any] | None = None,
    blocking: bool = False,
    warning: bool = True,
) -> None:
    _bump_all(buckets, kind)
    if blocking:
        _bump_all(buckets, "blocking_issues")
    if warning:
        _bump_all(buckets, "warnings")
    _add_example(examples, kind, row, domain=domain, backend=backend, detail=detail)


def _likely_dimension_for_evidence_gap(
    *, tool: str, evidence_id: str, result: dict[str, Any]
) -> str:
    text = " ".join(
        [
            tool,
            evidence_id,
            _safe_json_blob(result.get("payload") or {}),
        ]
    ).lower()
    if "moral" in text or "ethical" in text or "ethic" in text:
        return "ethical_quality"
    if "stakeholder" in text or "trust" in text or "priority" in text:
        return "stakeholder_management"
    if "stale" in text or _is_information_tool(tool):
        return "information_efficiency"
    return "adaptive_replanning"


def _record_gap_bucket(
    bucket: dict[str, dict[str, int]], key: Any, *, events: int, evidence_ids: int
) -> None:
    label = str(key or "unknown")
    item = bucket.setdefault(label, _empty_gap_bucket())
    item["events"] += events
    item["missing_score_evidence_ids"] += evidence_ids


def _record_evidence_gap_drilldown(
    drilldown: dict[str, Any],
    row: dict[str, Any],
    *,
    domain: str,
    backend: str,
    tick: Any,
    tool: str,
    missing_score_evidence_ids: list[str],
    result: dict[str, Any],
) -> None:
    family = str(row.get("family") or "unknown")
    model = str(row.get("model", row.get("agent_name", "unknown")))
    n_missing = len(missing_score_evidence_ids)
    drilldown["n_events"] += 1
    drilldown["n_missing_score_evidence_ids"] += n_missing
    _record_gap_bucket(drilldown["by_tool"], tool, events=1, evidence_ids=n_missing)
    _record_gap_bucket(drilldown["by_family"], family, events=1, evidence_ids=n_missing)
    _record_gap_bucket(drilldown["by_domain"], domain, events=1, evidence_ids=n_missing)
    _record_gap_bucket(
        drilldown["by_backend"], backend, events=1, evidence_ids=n_missing
    )
    _record_gap_bucket(drilldown["by_model"], model, events=1, evidence_ids=n_missing)

    by_dimension: dict[str, list[str]] = defaultdict(list)
    for evidence_id in missing_score_evidence_ids:
        by_dimension[
            _likely_dimension_for_evidence_gap(
                tool=tool, evidence_id=evidence_id, result=result
            )
        ].append(evidence_id)
    for dimension, evidence_ids in by_dimension.items():
        _record_gap_bucket(
            drilldown["by_likely_dimension"],
            dimension,
            events=1,
            evidence_ids=len(evidence_ids),
        )

    examples = drilldown["examples"]
    for evidence_id in missing_score_evidence_ids:
        if len(examples) >= EXAMPLE_LIMIT:
            break
        likely_dimension = _likely_dimension_for_evidence_gap(
            tool=tool, evidence_id=evidence_id, result=result
        )
        item = _example_base(row, domain=domain, backend=backend)
        item.update(
            {
                "classification": "state_changing_evidence_not_in_score",
                "tick": tick,
                "tool": tool,
                "evidence_id": evidence_id,
                "likely_dimension": likely_dimension,
                "repair_target": "evidence_wiring",
            }
        )
        examples.append(item)


def _audit_decision_impact(
    row: dict[str, Any],
    *,
    domain: str,
    backend: str,
    buckets: list[dict[str, int]],
    examples: dict[str, list[dict[str, Any]]],
) -> None:
    had_decision_impact = bool(row.get("decision_impact"))
    if not had_decision_impact:
        _record_recipe(
            kind="missing_decision_impact",
            row=row,
            domain=domain,
            backend=backend,
            buckets=buckets,
            examples=examples,
        )
    impact = _episode_impact(row)
    n_control = int(impact.get("n_control_calls", 0) or 0)
    prevented_loss = float(impact.get("prevented_loss", 0.0) or 0.0)
    if n_control > 0 and bool(impact.get("agent_hurt")):
        _record_recipe(
            kind="harmful_control_action",
            row=row,
            domain=domain,
            backend=backend,
            buckets=buckets,
            examples=examples,
            detail={
                "prevented_loss": prevented_loss,
                "interpretation": impact.get("interpretation"),
            },
        )
    if n_control > 0 and not bool(impact.get("outcome_changed")):
        _record_recipe(
            kind="cost_matched_control",
            row=row,
            domain=domain,
            backend=backend,
            buckets=buckets,
            examples=examples,
            detail={
                "prevented_loss": prevented_loss,
                "interpretation": impact.get("interpretation"),
            },
        )
    counterfactual = row.get("counterfactual")
    if (
        not isinstance(counterfactual, dict)
        or counterfactual.get("applicable") is False
    ):
        _record_recipe(
            kind="counterfactual_missing_or_not_applicable",
            row=row,
            domain=domain,
            backend=backend,
            buckets=buckets,
            examples=examples,
            detail={
                "reason": (
                    counterfactual.get("reason")
                    if isinstance(counterfactual, dict)
                    else "missing_counterfactual"
                )
            },
        )


def _audit_trajectory(
    row: dict[str, Any],
    *,
    batch_root: Path | None = None,
    domain: str,
    backend: str,
    buckets: list[dict[str, int]],
    examples: dict[str, list[dict[str, Any]]],
    evidence_gap_drilldown: dict[str, Any],
) -> None:
    traj_path = _trajectory_file(row, batch_root=batch_root)
    entries = _load_trajectory(traj_path) if traj_path is not None else None
    if entries is None:
        _record_recipe(
            kind="legacy_missing_trajectory",
            row=row,
            domain=domain,
            backend=backend,
            buckets=buckets,
            examples=examples,
            detail={"trajectory_path": str(traj_path) if traj_path else None},
            warning=True,
        )
        return
    _bump_all(buckets, "episodes_with_trajectory")

    score_evidence = _score_evidence_ids(row)
    info_score_evidence = _information_evidence_ids(row)
    stale_markers: list[dict[str, Any]] = []

    for idx, entry in enumerate(entries):
        stale_markers.extend(_stale_attrs(entry, idx))
        action_names = {str(a.get("name") or "") for a in _iter_actions(entry)}
        for result in _iter_results(entry):
            name = str(result.get("name") or "")
            ok = bool(result.get("ok"))
            error_code = str(result.get("error_code") or "")
            is_control = name in CONTROL_TOOL_NAMES or bool(
                action_names.intersection(CONTROL_TOOL_NAMES)
            )
            if not ok:
                _record_recipe(
                    kind="tool_result_failed",
                    row=row,
                    domain=domain,
                    backend=backend,
                    buckets=buckets,
                    examples=examples,
                    detail={
                        "tick": entry.get("tick"),
                        "tool": name,
                        "error_code": error_code or None,
                    },
                )
                if error_code in BLOCKING_PROTOCOL_CODES:
                    _record_recipe(
                        kind="tool_protocol_failure",
                        row=row,
                        domain=domain,
                        backend=backend,
                        buckets=buckets,
                        examples=examples,
                        detail={
                            "tick": entry.get("tick"),
                            "tool": name,
                            "error_code": error_code,
                        },
                        blocking=True,
                    )
            if not bool(result.get("state_changing")):
                continue
            if not ok:
                continue
            evidence_ids = _evidence_ids_from_result(result)
            if is_control and not evidence_ids:
                _record_recipe(
                    kind="state_changing_without_evidence",
                    row=row,
                    domain=domain,
                    backend=backend,
                    buckets=buckets,
                    examples=examples,
                    detail={"tick": entry.get("tick"), "tool": name},
                    blocking=True,
                )
                continue
            missing_score = sorted(
                ev for ev in evidence_ids if ev not in score_evidence
            )
            if missing_score and native_effect_is_scored(result, entry, score_evidence):
                missing_score = []
            if is_control and missing_score:
                _record_evidence_gap_drilldown(
                    evidence_gap_drilldown,
                    row,
                    domain=domain,
                    backend=backend,
                    tick=entry.get("tick"),
                    tool=name,
                    missing_score_evidence_ids=missing_score,
                    result=result,
                )
                _record_recipe(
                    kind="state_changing_evidence_not_in_score",
                    row=row,
                    domain=domain,
                    backend=backend,
                    buckets=buckets,
                    examples=examples,
                    detail={
                        "tick": entry.get("tick"),
                        "tool": name,
                        "evidence_ids": missing_score,
                    },
                )

    for stale in stale_markers:
        consumed, score_consumed, detail = _stale_consumption(
            stale, entries, info_score_evidence
        )
        if not consumed:
            _record_recipe(
                kind="stale_info_unconsumed",
                row=row,
                domain=domain,
                backend=backend,
                buckets=buckets,
                examples=examples,
                detail={
                    "tick": stale.get("tick"),
                    "entity_id": stale.get("entity_id"),
                    "attr": stale.get("attr"),
                    "staleness_ticks": stale.get("staleness_ticks"),
                },
            )
        elif not score_consumed:
            _record_recipe(
                kind="stale_info_missing_score_evidence",
                row=row,
                domain=domain,
                backend=backend,
                buckets=buckets,
                examples=examples,
                detail={
                    "tick": stale.get("tick"),
                    "entity_id": stale.get("entity_id"),
                    "attr": stale.get("attr"),
                    "consumption_detail": detail,
                },
            )


def _merge_total(totals: dict[str, int], buckets: dict[str, dict[str, int]]) -> None:
    for bucket in buckets.values():
        for key, value in bucket.items():
            totals[key] = int(totals.get(key, 0)) + int(value)


def _new_evidence_gap_drilldown() -> dict[str, Any]:
    out: dict[str, Any] = {
        "n_events": 0,
        "n_missing_score_evidence_ids": 0,
        "examples": [],
    }
    for name in DRILLDOWN_BUCKETS:
        out[name] = {}
    return out


def _sort_gap_bucket(bucket: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    return {
        key: bucket[key]
        for key in sorted(
            bucket,
            key=lambda k: (
                -int(bucket[k].get("missing_score_evidence_ids", 0)),
                -int(bucket[k].get("events", 0)),
                k,
            ),
        )
    }


def _finalize_evidence_gap_drilldown(drilldown: dict[str, Any]) -> dict[str, Any]:
    out = {
        "n_events": int(drilldown.get("n_events", 0)),
        "n_missing_score_evidence_ids": int(
            drilldown.get("n_missing_score_evidence_ids", 0)
        ),
        "examples": list(drilldown.get("examples") or []),
    }
    for name in DRILLDOWN_BUCKETS:
        out[name] = _sort_gap_bucket(drilldown.get(name) or {})
    return out


def _next_repair_targets(totals: dict[str, int]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    if totals.get("harmful_control_action", 0):
        targets.append(
            {
                "target_code": "agent_failure_recipe_review",
                "reason": "Control actions made the counterfactual outcome worse.",
                "next_action": "Mine the example trajectories before changing prompts; classify whether the issue is tool semantics, observation fog, or agent policy.",
            }
        )
    if totals.get("cost_matched_control", 0):
        targets.append(
            {
                "target_code": "scenario_headroom_or_oracle_gate",
                "reason": "Control actions executed but matched wait-only cost.",
                "next_action": "Check whether the cell has enough oracle-vs-wait headroom or whether the tool should remain diagnostic.",
            }
        )
    if totals.get("state_changing_evidence_not_in_score", 0) or totals.get(
        "state_changing_without_evidence", 0
    ):
        targets.append(
            {
                "target_code": "evidence_wiring",
                "reason": "State-changing tool evidence is missing or not consumed by scorer dimensions.",
                "next_action": "Wire tool-call evidence ids through the relevant dimension or mark the dimension non-applicable with a reason.",
            }
        )
    if totals.get("stale_info_missing_score_evidence", 0) or totals.get(
        "stale_info_unconsumed", 0
    ):
        targets.append(
            {
                "target_code": "staleness_information_efficiency",
                "reason": "Stale observations were not converted into scorer-visible information use.",
                "next_action": "Decide whether to replay/read stale attributes into information_efficiency or keep them as diagnostic warnings.",
            }
        )
    if totals.get("tool_protocol_failure", 0):
        targets.append(
            {
                "target_code": "tool_protocol_runtime",
                "reason": "Tool protocol failures can invalidate action-causality evidence.",
                "next_action": "Fix runtime idempotency/validation/handler failures before interpreting model capability.",
            }
        )
    return targets


def _status(totals: dict[str, int]) -> tuple[str, list[str], list[str]]:
    blocking_codes: list[str] = []
    warning_codes: list[str] = []
    if totals.get("tool_protocol_failure", 0):
        blocking_codes.append("tool_protocol_failure")
    if totals.get("state_changing_without_evidence", 0):
        blocking_codes.append("state_changing_without_evidence")
    for key in (
        "legacy_missing_trajectory",
        "harmful_control_action",
        "cost_matched_control",
        "missing_decision_impact",
        "counterfactual_missing_or_not_applicable",
        "tool_result_failed",
        "state_changing_evidence_not_in_score",
        "stale_info_unconsumed",
        "stale_info_missing_score_evidence",
    ):
        if totals.get(key, 0):
            warning_codes.append(key)
    if blocking_codes:
        return "blocking", sorted(blocking_codes), sorted(warning_codes)
    if warning_codes:
        return "warning", [], sorted(warning_codes)
    return "pass", [], []


def build_report(
    rows: list[dict[str, Any]], *, batch_root: Path | None = None
) -> dict[str, Any]:
    ok_rows = [r for r in rows if r.get("status", "ok") == "ok"]
    by_domain: dict[str, dict[str, int]] = defaultdict(_empty_bucket)
    by_domain_backend: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(_empty_bucket)
    )
    by_family: dict[str, dict[str, int]] = defaultdict(_empty_bucket)
    by_model: dict[str, dict[str, int]] = defaultdict(_empty_bucket)
    examples: dict[str, list[dict[str, Any]]] = {}
    evidence_gap_drilldown = _new_evidence_gap_drilldown()

    for row in ok_rows:
        domain = _domain_from_row(row)
        backend = _backend_from_row(row)
        family = str(row.get("family") or "unknown")
        model = str(row.get("model", row.get("agent_name", "unknown")))
        buckets = [
            by_domain[domain],
            by_domain_backend[domain][backend],
            by_family[family],
            by_model[model],
        ]
        _bump_all(buckets, "episodes")
        _audit_decision_impact(
            row,
            domain=domain,
            backend=backend,
            buckets=buckets,
            examples=examples,
        )
        _audit_trajectory(
            row,
            batch_root=batch_root,
            domain=domain,
            backend=backend,
            buckets=buckets,
            examples=examples,
            evidence_gap_drilldown=evidence_gap_drilldown,
        )

    totals = _empty_bucket()
    _merge_total(totals, by_domain)
    status, blocking_codes, warning_codes = _status(totals)
    audit = {
        "schema_version": "0.2",
        "status": status,
        "n_total": len(rows),
        "n_ok": len(ok_rows),
        "totals": totals,
        "blocking_codes": blocking_codes,
        "warning_codes": warning_codes,
        "examples": examples,
        "evidence_gap_drilldown": _finalize_evidence_gap_drilldown(
            evidence_gap_drilldown
        ),
        "by_domain": dict(sorted(by_domain.items())),
        "by_domain_backend": {
            domain: dict(sorted(backends.items()))
            for domain, backends in sorted(by_domain_backend.items())
        },
        "by_family": dict(sorted(by_family.items())),
        "by_model": dict(sorted(by_model.items())),
        "next_repair_targets": _next_repair_targets(totals),
    }
    return {
        "agent_failure_recipes": audit,
        "notes": [
            "This report is read-only and does not change scorer semantics, release artifacts, scenario YAMLs, or suite selection.",
            "Failure recipes are diagnostics for benchmark quality: they identify harmful actions, zero-headroom controls, protocol failures, stale-information gaps, and evidence wiring gaps.",
            "Legacy missing trajectories are warnings, not blockers; protocol conflicts and state-changing controls without evidence are blockers.",
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
    audit = report.get("agent_failure_recipes") or {}
    totals = audit.get("totals") or {}
    lines = [
        "# Agent Failure Recipes",
        "",
        "Mines clean batch episodes for action-level failure patterns that should drive benchmark repair or future scenario design.",
        "",
        f"- Status: **{audit.get('status', 'unknown')}**",
        f"- Episodes OK: **{audit.get('n_ok', 0)}** / {audit.get('n_total', 0)}",
        f"- Episodes with trajectory: **{totals.get('episodes_with_trajectory', 0)}**",
        f"- Harmful control actions: **{totals.get('harmful_control_action', 0)}**",
        f"- Cost-matched controls: **{totals.get('cost_matched_control', 0)}**",
        f"- Tool protocol failures: **{totals.get('tool_protocol_failure', 0)}**",
        "- State-changing evidence not in score: "
        f"**{totals.get('state_changing_evidence_not_in_score', 0)}**",
        "- Stale-info missing score evidence: "
        f"**{totals.get('stale_info_missing_score_evidence', 0)}**",
        f"- Legacy missing trajectories: **{totals.get('legacy_missing_trajectory', 0)}**",
    ]
    if audit.get("blocking_codes"):
        lines.append(f"- Blocking codes: {', '.join(audit['blocking_codes'])}")
    if audit.get("warning_codes"):
        lines.append(f"- Warning codes: {', '.join(audit['warning_codes'])}")
    by_domain = audit.get("by_domain") or {}
    if by_domain:
        lines.extend(["", "## By Domain", ""])
        for domain, bucket in sorted(by_domain.items()):
            lines.append(
                f"- `{domain}`: episodes={bucket.get('episodes', 0)}, "
                f"harmful={bucket.get('harmful_control_action', 0)}, "
                f"cost_matched={bucket.get('cost_matched_control', 0)}, "
                f"protocol={bucket.get('tool_protocol_failure', 0)}, "
                f"evidence_gap={bucket.get('state_changing_evidence_not_in_score', 0)}, "
                f"stale_gap={bucket.get('stale_info_missing_score_evidence', 0)}"
            )
    drilldown = audit.get("evidence_gap_drilldown") or {}
    if drilldown.get("n_events"):
        lines.extend(["", "## Evidence Gap Drilldown", ""])
        lines.append(
            "- Missing score evidence IDs: "
            f"**{drilldown.get('n_missing_score_evidence_ids', 0)}** across "
            f"**{drilldown.get('n_events', 0)}** state-changing tool events."
        )
        for title, key in (
            ("Top Tools", "by_tool"),
            ("Top Families", "by_family"),
            ("Top Domains", "by_domain"),
            ("Top Likely Dimensions", "by_likely_dimension"),
        ):
            bucket = drilldown.get(key) or {}
            if not bucket:
                continue
            lines.extend(["", f"### {title}", ""])
            for name, stats in list(bucket.items())[:5]:
                lines.append(
                    f"- `{name}`: events={stats.get('events', 0)}, "
                    "missing_score_evidence_ids="
                    f"{stats.get('missing_score_evidence_ids', 0)}"
                )
    targets = audit.get("next_repair_targets") or []
    if targets:
        lines.extend(["", "## Next Repair Targets", ""])
        for target in targets:
            lines.append(
                f"- `{target.get('target_code')}`: {target.get('reason')} "
                f"Next: {target.get('next_action')}"
            )
    examples = audit.get("examples") or {}
    if examples:
        lines.extend(["", "## Examples", ""])
        for kind, rows in sorted(examples.items()):
            lines.append(f"### {kind}")
            for row in rows[:5]:
                lines.append(
                    f"- `{row.get('scenario_id')}` ({row.get('domain')}/{row.get('backend_kind')}, "
                    f"model=`{row.get('model')}`)"
                )
            lines.append("")
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    out_dir = args.output_dir.resolve()
    episodes_path = out_dir / "episodes.jsonl"
    if not episodes_path.is_file():
        print(f"missing {episodes_path}", file=sys.stderr)
        return 1
    report = build_report(_load_jsonl_rows(episodes_path), batch_root=out_dir)
    (out_dir / "agent_failure_recipes_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_markdown(report, out_dir / "AGENT_FAILURE_RECIPES.md")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
