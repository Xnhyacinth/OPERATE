#!/usr/bin/env python3
"""Audit whether stale observations are consumed by agent information actions."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.analyze_decision_impact import _backend_from_row  # noqa: E402
from scripts.analyze_decision_impact import _domain_from_row as _base_domain_from_row  # noqa: E402

EXAMPLE_LIMIT = 20
BACKEND_TO_DOMAIN = {
    "mock_sumo": "traffic",
    "mock_sumo_deterministic": "traffic",
    "sumo": "traffic",
    "grid2op": "power_grid",
    "pglib_uc_synthetic": "power_grid",
    "pandapower_acopf": "power_grid",
    "cigre_distribution": "power_grid",
    "opendss_ieee13": "power_grid",
    "opendss_fresh_feeders": "power_grid",
    "jsplib_job_shop": "logistics",
    "pyvrp_cvrp": "logistics",
    "pyvrp_vrptw": "logistics",
    "orgym_invmgmt": "logistics",
    "pandapower_lv": "microgrid",
    "pymgrid_economic_dispatch": "microgrid",
}
FAMILY_TO_DOMAIN = {
    "daily_ops_24h": "power_grid",
    "daily_ops_real_forecast_24h": "power_grid",
    "critical_winter_peak": "power_grid",
    "reserve_stress_24h": "power_grid",
    "wind_uncertainty_24h": "power_grid",
    "storm_emergency_6h": "power_grid",
    "storm_emergency_6h_idf2023": "power_grid",
    "storm_wcci_2022": "power_grid",
    "storm_l2rpn_sandbox": "power_grid",
    "storm_l2rpn_neurips2020_track1": "power_grid",
    "storm_l2rpn_neurips2020_track2": "power_grid",
    "storm_l2rpn_icaps2021": "power_grid",
    "distribution_volt_var": "power_grid",
    "distribution_volt_var_oberrhein": "power_grid",
    "acopf_dispatch_24h": "power_grid",
    "opendss_ieee13_volt_var": "power_grid",
    "opendss_fresh_feeders_volt_var": "power_grid",
    "signal_coordination": "traffic",
    "route_assignment": "traffic",
    "incident_response": "traffic",
    "vip_priority_dilemma": "traffic",
    "job_shop_dispatch": "logistics",
    "inventory_replenishment": "logistics",
    "cvrp_dispatch": "logistics",
    "vrptw_dispatch": "logistics",
    "microgrid_economic_dispatch_24h": "microgrid",
    "microgrid_lv_voltage_6h": "microgrid",
    "microgrid_lv_voltage_staged_6h": "microgrid",
    "microgrid_lv_voltage_recovery_10h": "microgrid",
}
INFORMATION_TOOL_PREFIXES = (
    "investigate",
    "query",
    "forecast",
    "inspect",
    "measure",
    "diagnose",
)


@dataclass(frozen=True)
class StaleAttr:
    entry_index: int
    tick: Any
    entity_id: str
    attr: str
    staleness_ticks: int | None


def _empty_bucket() -> dict[str, int]:
    return {
        "episodes": 0,
        "episodes_with_trajectory": 0,
        "trajectory_missing": 0,
        "stale_observation_entries": 0,
        "stale_entity_attrs": 0,
        "stale_entity_attrs_consumed": 0,
        "stale_entity_attrs_unconsumed": 0,
        "information_efficiency_applicable_with_stale": 0,
        "information_efficiency_consumed_stale_evidence": 0,
        "information_efficiency_missing_consumption_evidence": 0,
        "warnings": 0,
        "blocking_issues": 0,
    }


def _bump(bucket: dict[str, int], key: str, amount: int = 1) -> None:
    bucket[key] = int(bucket.get(key, 0)) + amount


def _issue_all(
    buckets: list[dict[str, int]],
    key: str,
    *,
    amount: int = 1,
    blocking: bool = False,
    warning: bool = False,
) -> None:
    for bucket in buckets:
        _bump(bucket, key, amount)
        if blocking:
            _bump(bucket, "blocking_issues")
        if warning:
            _bump(bucket, "warnings")


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


def _domain_from_row(
    row: dict[str, Any], trajectory: dict[str, Any] | None = None
) -> str:
    domain = str(row.get("domain") or "").strip()
    if domain:
        return domain
    backend = str(row.get("backend_kind") or row.get("backend") or "").strip()
    if backend in BACKEND_TO_DOMAIN:
        return BACKEND_TO_DOMAIN[backend]
    family = str(row.get("family") or "").strip()
    if family in FAMILY_TO_DOMAIN:
        return FAMILY_TO_DOMAIN[family]
    if isinstance(trajectory, dict):
        meta = trajectory.get("scenario") or trajectory.get("scenario_config") or {}
        if isinstance(meta, dict):
            merged = dict(meta)
            for key, value in row.items():
                if isinstance(value, str):
                    if value.strip():
                        merged[key] = value
                    continue
                if value is not None:
                    merged[key] = value
            inferred = _domain_from_row(merged, None)
            if inferred != "unknown":
                return inferred
    return _base_domain_from_row(row)


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


def _information_applicable(row: dict[str, Any]) -> bool:
    score = row.get("score") or {}
    if not isinstance(score, dict):
        return False
    for dim in _dimensions(score):
        if str(dim.get("name") or "") == "information_efficiency":
            return bool(dim.get("applicable", True))
    return False


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


def _safe_json_blob(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False).lower()
    except TypeError:
        return str(value).lower()


def _is_information_tool(name: Any) -> bool:
    text = str(name or "").lower()
    return text.startswith(INFORMATION_TOOL_PREFIXES)


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


def _extract_stale_attrs(entry: dict[str, Any], entry_index: int) -> list[StaleAttr]:
    observation = entry.get("observation") or {}
    entities = observation.get("entities") if isinstance(observation, dict) else None
    if not isinstance(entities, dict):
        return []
    out: list[StaleAttr] = []
    for entity_id, entity in entities.items():
        if not isinstance(entity, dict):
            continue
        stale_attrs = entity.get("_stale_attrs")
        if not isinstance(stale_attrs, dict):
            continue
        for attr, ticks in sorted(stale_attrs.items()):
            try:
                staleness_ticks = int(ticks)
            except (TypeError, ValueError):
                staleness_ticks = None
            out.append(
                StaleAttr(
                    entry_index=entry_index,
                    tick=entry.get("tick"),
                    entity_id=str(entity_id),
                    attr=str(attr),
                    staleness_ticks=staleness_ticks,
                )
            )
    return out


def _stale_consumption(
    stale: StaleAttr,
    entries: list[dict[str, Any]],
    info_evidence_ids: set[str],
) -> tuple[bool, bool, dict[str, Any] | None]:
    entity = stale.entity_id.lower()
    attr = stale.attr.lower()
    for entry in entries[stale.entry_index :]:
        action_detail: dict[str, Any] | None = None
        for action in _iter_actions(entry):
            if not _is_information_tool(action.get("name")):
                continue
            blob = _safe_json_blob(action)
            if entity in blob and attr in blob:
                action_detail = {
                    "tick": entry.get("tick"),
                    "tool": action.get("name"),
                    "source": "action",
                }
        for result in _iter_results(entry):
            if not _is_information_tool(result.get("name")):
                continue
            blob = _safe_json_blob(result)
            if entity not in blob or attr not in blob:
                continue
            result_evidence = _evidence_ids_from_result(result)
            score_consumed = bool(info_evidence_ids.intersection(result_evidence))
            return (
                True,
                score_consumed,
                {
                    "tick": entry.get("tick"),
                    "tool": result.get("name"),
                    "source": "tool_result",
                    "evidence_ids": sorted(result_evidence),
                },
            )
        if action_detail is not None:
            return True, False, action_detail
    return False, False, None


def _audit_row(
    row: dict[str, Any],
    *,
    by_domain: dict[str, dict[str, int]],
    by_domain_backend: dict[str, dict[str, dict[str, int]]],
    by_family: dict[str, dict[str, int]],
    examples: dict[str, list[dict[str, Any]]],
) -> Counter[str]:
    rollup: Counter[str] = Counter()
    trajectory_summary = row.get("trajectory_summary")
    domain = _domain_from_row(
        row, trajectory_summary if isinstance(trajectory_summary, dict) else None
    )
    backend = _backend_from_row(row)
    family = str(row.get("family") or "unknown")
    buckets = [by_domain[domain], by_domain_backend[domain][backend], by_family[family]]
    for bucket in buckets:
        _bump(bucket, "episodes")

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

    stale_attrs: list[StaleAttr] = []
    stale_entries: set[int] = set()
    for idx, entry in enumerate(entries):
        current = _extract_stale_attrs(entry, idx)
        if current:
            stale_entries.add(idx)
            stale_attrs.extend(current)
    if not stale_attrs:
        return rollup

    _issue_all(buckets, "stale_observation_entries", amount=len(stale_entries))
    _issue_all(buckets, "stale_entity_attrs", amount=len(stale_attrs))
    if _information_applicable(row):
        _issue_all(buckets, "information_efficiency_applicable_with_stale")

    info_evidence_ids = _information_evidence_ids(row)
    for stale in stale_attrs:
        consumed, score_consumed, detail = _stale_consumption(
            stale, entries, info_evidence_ids
        )
        if consumed:
            _issue_all(buckets, "stale_entity_attrs_consumed")
        else:
            rollup["stale_entity_attrs_unconsumed"] += 1
            _issue_all(buckets, "stale_entity_attrs_unconsumed", warning=True)
            _add_example(
                examples,
                "stale_entity_attrs_unconsumed",
                row,
                domain=domain,
                backend=backend,
                detail={
                    "tick": stale.tick,
                    "entity_id": stale.entity_id,
                    "attr": stale.attr,
                    "staleness_ticks": stale.staleness_ticks,
                },
            )
        if score_consumed:
            _issue_all(buckets, "information_efficiency_consumed_stale_evidence")
        else:
            rollup["information_efficiency_missing_consumption_evidence"] += 1
            _issue_all(
                buckets,
                "information_efficiency_missing_consumption_evidence",
                warning=True,
            )
            _add_example(
                examples,
                "information_efficiency_missing_consumption_evidence",
                row,
                domain=domain,
                backend=backend,
                detail={
                    "tick": stale.tick,
                    "entity_id": stale.entity_id,
                    "attr": stale.attr,
                    "consumed_by_tool": consumed,
                    "consumption_detail": detail,
                },
            )
    return rollup


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
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

    warnings: list[str] = []
    if totals["trajectory_missing"]:
        warnings.append(
            f"{totals['trajectory_missing']} ok episode(s) lack trajectory files"
        )
    if totals["stale_entity_attrs_unconsumed"]:
        warnings.append(
            f"{totals['stale_entity_attrs_unconsumed']} stale entity attr(s) were not consumed by later information actions"
        )
    if totals["information_efficiency_missing_consumption_evidence"]:
        warnings.append(
            f"{totals['information_efficiency_missing_consumption_evidence']} stale entity attr(s) lack information_efficiency evidence consumption"
        )

    n_blocking = int(totals["blocking_issues"])
    n_warnings = int(totals["warnings"]) + len(warnings)
    status = "blocking" if n_blocking else ("warning" if n_warnings else "pass")

    audit = {
        "schema_version": "0.1",
        "status": status,
        "n_total": len(rows),
        "n_ok": len(ok_rows),
        "totals": totals,
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
        "staleness_consumption_audit": audit,
        "notes": [
            "This audit is report-only and never mutates release artifacts.",
            "For live scorer versions >=0.6.2, fresh runs should also expose stale_observation evidence to information_efficiency.",
            "For older/legacy rows, this audit detects stale observations from trajectory files and checks whether the score consumed matching information evidence.",
            "Stale attrs are detected from observation.entities[*]._stale_attrs in trajectory files.",
            "Consumption requires a later information-style tool action/result to mention the same entity and attr.",
            "information_efficiency evidence consumption is counted only when matching tool evidence ids appear in the score dimension.",
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
    audit = report.get("staleness_consumption_audit") or {}
    totals = audit.get("totals") or {}
    lines = [
        "# Staleness Consumption Audit",
        "",
        "Checks whether stale observation markers are later consumed by information-gathering tools and scorer-facing evidence.",
        "",
        "Fresh scorer outputs at `SCORING_VERSION >= 0.6.2` are expected to expose stale-observation evidence directly through `information_efficiency`; legacy rows are diagnosed from trajectory files.",
        "",
        f"- Status: **{audit.get('status', 'unknown')}**",
        f"- Episodes OK: **{audit.get('n_ok', 0)}** / {audit.get('n_total', 0)}",
        f"- Episodes with trajectory: **{totals.get('episodes_with_trajectory', 0)}**",
        f"- Trajectory missing: **{totals.get('trajectory_missing', 0)}**",
        f"- Stale observation entries: **{totals.get('stale_observation_entries', 0)}**",
        f"- Stale entity attrs: **{totals.get('stale_entity_attrs', 0)}**",
        f"- Consumed stale attrs: **{totals.get('stale_entity_attrs_consumed', 0)}**",
        f"- Unconsumed stale attrs: **{totals.get('stale_entity_attrs_unconsumed', 0)}**",
        "- information_efficiency consumed stale evidence: "
        f"**{totals.get('information_efficiency_consumed_stale_evidence', 0)}**",
        "- information_efficiency missing stale evidence: "
        f"**{totals.get('information_efficiency_missing_consumption_evidence', 0)}**",
        f"- Blocking issues: **{totals.get('blocking_issues', 0)}**",
        f"- Warnings: **{totals.get('warnings', 0)}**",
    ]
    by_domain = audit.get("by_domain") or {}
    if by_domain:
        lines.extend(["", "## By Domain", ""])
        for domain, bucket in sorted(by_domain.items()):
            lines.append(
                f"- `{domain}`: episodes={bucket.get('episodes', 0)}, "
                f"stale_attrs={bucket.get('stale_entity_attrs', 0)}, "
                f"consumed={bucket.get('stale_entity_attrs_consumed', 0)}, "
                f"unconsumed={bucket.get('stale_entity_attrs_unconsumed', 0)}, "
                f"warnings={bucket.get('warnings', 0)}"
            )
    examples = audit.get("examples") or {}
    if examples:
        lines.extend(["", "## Examples", ""])
        for kind, rows in sorted(examples.items()):
            lines.append(f"### {kind}")
            for row in rows[:5]:
                lines.append(
                    f"- `{row.get('scenario_id')}` ({row.get('domain')}/{row.get('backend_kind')}) "
                    f"entity=`{row.get('entity_id')}` attr=`{row.get('attr')}`"
                )
            lines.append("")
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    out_dir = args.output_dir.resolve()
    ep_path = out_dir / "episodes.jsonl"
    if not ep_path.is_file():
        print(f"missing {ep_path}", file=sys.stderr)
        return 1
    report = build_report(_load_jsonl_rows(ep_path))
    (out_dir / "staleness_consumption_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_markdown(report, out_dir / "STALENESS_CONSUMPTION.md")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
