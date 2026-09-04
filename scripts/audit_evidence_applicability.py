#!/usr/bin/env python3
"""Audit score dimensions for evidence/applicability contract violations."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

EXAMPLE_LIMIT = 20


FAMILY_TO_BACKEND = {
    "daily_ops_24h": "pglib_uc_synthetic",
    "daily_ops_real_forecast_24h": "pglib_uc_synthetic",
    "critical_winter_peak": "pglib_uc_synthetic",
    "reserve_stress_24h": "pglib_uc_synthetic",
    "wind_uncertainty_24h": "pglib_uc_synthetic",
    "storm_emergency_6h": "grid2op",
    "storm_l2rpn_sandbox": "grid2op",
    "storm_l2rpn_neurips2020_track1": "grid2op",
    "storm_l2rpn_icaps2021": "grid2op",
    "distribution_volt_var": "cigre_distribution",
    "distribution_volt_var_oberrhein": "cigre_distribution",
    "acopf_dispatch_24h": "pandapower_acopf",
    "opendss_ieee13_volt_var": "opendss_ieee13",
    "cvrp_dispatch": "pyvrp_cvrp",
    "vrptw_dispatch": "pyvrp_vrptw",
    "job_shop_dispatch": "jsplib_job_shop",
    "incident_response": "mock_sumo",
    "vip_priority_dilemma": "mock_sumo",
}

FAMILY_TO_DOMAIN = {
    "daily_ops_24h": "power_grid",
    "daily_ops_real_forecast_24h": "power_grid",
    "critical_winter_peak": "power_grid",
    "reserve_stress_24h": "power_grid",
    "wind_uncertainty_24h": "power_grid",
    "storm_emergency_6h": "power_grid",
    "storm_l2rpn_sandbox": "power_grid",
    "storm_l2rpn_neurips2020_track1": "power_grid",
    "storm_l2rpn_icaps2021": "power_grid",
    "distribution_volt_var": "power_grid",
    "distribution_volt_var_oberrhein": "power_grid",
    "acopf_dispatch_24h": "power_grid",
    "opendss_ieee13_volt_var": "power_grid",
    "cvrp_dispatch": "logistics",
    "vrptw_dispatch": "logistics",
    "job_shop_dispatch": "logistics",
    "incident_response": "traffic",
    "vip_priority_dilemma": "traffic",
}


def _domain_from_row(row: dict[str, Any]) -> str:
    domain = str(row.get("domain") or "").strip()
    if domain:
        return domain
    family = str(row.get("family") or "").strip()
    if family in FAMILY_TO_DOMAIN:
        return FAMILY_TO_DOMAIN[family]
    slug = str(row.get("scenario_slug") or "").strip()
    if slug:
        parts = [p for p in slug.split("/") if p]
        if parts[:1] == ["releases"] and len(parts) > 2:
            return parts[2]
        if parts and parts[0] in {"power_grid", "logistics", "traffic"}:
            return parts[0]
    return "unknown"


def _backend_from_row(row: dict[str, Any]) -> str:
    backend = str(row.get("backend_kind") or row.get("backend") or "").strip()
    if backend:
        return backend
    return FAMILY_TO_BACKEND.get(str(row.get("family") or ""), "unknown")


def _dimensions_from_score(score: dict[str, Any]) -> list[dict[str, Any]]:
    dims = score.get("dimensions")
    if isinstance(dims, list):
        return [d for d in dims if isinstance(d, dict)]
    if isinstance(dims, dict):
        out: list[dict[str, Any]] = []
        for name, body in dims.items():
            if not isinstance(body, dict):
                continue
            item = dict(body)
            item.setdefault("name", name)
            out.append(item)
        return out
    return []


def _empty_bucket() -> dict[str, int]:
    return {
        "episodes": 0,
        "dimensions": 0,
        "applicable": 0,
        "non_applicable": 0,
        "applicable_missing_evidence": 0,
        "applicable_missing_reason": 0,
        "applicable_zero_support": 0,
        "non_applicable_missing_reason": 0,
        "malformed_dimensions": 0,
        "missing_dimensions": 0,
        "blocking_issues": 0,
        "warnings": 0,
    }


def _add_issue(
    *,
    examples: dict[str, list[dict[str, Any]]],
    kind: str,
    row: dict[str, Any],
    dim: dict[str, Any] | None,
    domain: str,
    backend: str,
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
    if dim is not None:
        item.update(
            {
                "dimension": dim.get("name"),
                "applicable": dim.get("applicable"),
                "support_count": dim.get("support_count"),
                "reason": dim.get("reason"),
                "n_evidence_ids": len(dim.get("evidence_ids") or []),
            }
        )
    bucket.append(item)


def _bump_bucket(bucket: dict[str, int], key: str, amount: int = 1) -> None:
    bucket[key] = int(bucket.get(key, 0)) + amount


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [r for r in rows if r.get("status", "ok") == "ok"]
    by_domain: dict[str, dict[str, int]] = defaultdict(_empty_bucket)
    by_domain_backend: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(_empty_bucket)
    )
    by_family: dict[str, dict[str, int]] = defaultdict(_empty_bucket)
    by_dimension: dict[str, dict[str, int]] = defaultdict(_empty_bucket)
    examples: dict[str, list[dict[str, Any]]] = {}
    rollup: Counter[str] = Counter()

    for row in ok_rows:
        domain = _domain_from_row(row)
        backend = _backend_from_row(row)
        family = str(row.get("family") or "unknown")
        score = row.get("score") or {}
        dims = _dimensions_from_score(score) if isinstance(score, dict) else []

        buckets = [
            by_domain[domain],
            by_domain_backend[domain][backend],
            by_family[family],
        ]
        for bucket in buckets:
            _bump_bucket(bucket, "episodes")

        if not dims:
            rollup["missing_dimensions"] += 1
            for bucket in buckets:
                _bump_bucket(bucket, "missing_dimensions")
                _bump_bucket(bucket, "blocking_issues")
            _add_issue(
                examples=examples,
                kind="missing_dimensions",
                row=row,
                dim=None,
                domain=domain,
                backend=backend,
            )
            continue

        for dim in dims:
            dim_name = str(dim.get("name") or "unknown")
            dim_bucket = by_dimension[dim_name]
            all_buckets = [*buckets, dim_bucket]
            for bucket in all_buckets:
                _bump_bucket(bucket, "dimensions")

            applicable = bool(dim.get("applicable", True))
            reason = str(dim.get("reason") or "").strip()
            evidence_ids = dim.get("evidence_ids") or []
            if not isinstance(evidence_ids, list):
                evidence_ids = []
            support_count_raw = dim.get("support_count")
            try:
                support_count = int(support_count_raw)
            except (TypeError, ValueError):
                support_count = 0

            if applicable:
                rollup["applicable"] += 1
                for bucket in all_buckets:
                    _bump_bucket(bucket, "applicable")
                if not evidence_ids:
                    rollup["applicable_missing_evidence"] += 1
                    for bucket in all_buckets:
                        _bump_bucket(bucket, "applicable_missing_evidence")
                        _bump_bucket(bucket, "blocking_issues")
                    _add_issue(
                        examples=examples,
                        kind="applicable_missing_evidence",
                        row=row,
                        dim=dim,
                        domain=domain,
                        backend=backend,
                    )
                if not reason:
                    rollup["applicable_missing_reason"] += 1
                    for bucket in all_buckets:
                        _bump_bucket(bucket, "applicable_missing_reason")
                        _bump_bucket(bucket, "blocking_issues")
                    _add_issue(
                        examples=examples,
                        kind="applicable_missing_reason",
                        row=row,
                        dim=dim,
                        domain=domain,
                        backend=backend,
                    )
                if support_count <= 0:
                    rollup["applicable_zero_support"] += 1
                    for bucket in all_buckets:
                        _bump_bucket(bucket, "applicable_zero_support")
                        _bump_bucket(bucket, "warnings")
                    _add_issue(
                        examples=examples,
                        kind="applicable_zero_support",
                        row=row,
                        dim=dim,
                        domain=domain,
                        backend=backend,
                    )
            else:
                rollup["non_applicable"] += 1
                for bucket in all_buckets:
                    _bump_bucket(bucket, "non_applicable")
                if not reason:
                    rollup["non_applicable_missing_reason"] += 1
                    for bucket in all_buckets:
                        _bump_bucket(bucket, "non_applicable_missing_reason")
                        _bump_bucket(bucket, "blocking_issues")
                    _add_issue(
                        examples=examples,
                        kind="non_applicable_missing_reason",
                        row=row,
                        dim=dim,
                        domain=domain,
                        backend=backend,
                    )

    n_blocking = sum(
        int(rollup.get(key, 0))
        for key in (
            "missing_dimensions",
            "applicable_missing_evidence",
            "applicable_missing_reason",
            "non_applicable_missing_reason",
        )
    )
    n_warnings = int(rollup.get("applicable_zero_support", 0))
    status = "blocking" if n_blocking else ("warning" if n_warnings else "pass")
    audit = {
        "schema_version": "0.1",
        "status": status,
        "n_total": len(rows),
        "n_ok": len(ok_rows),
        "n_dimensions": int(rollup["applicable"] + rollup["non_applicable"]),
        "n_applicable": int(rollup["applicable"]),
        "n_non_applicable": int(rollup["non_applicable"]),
        "n_missing_dimensions": int(rollup["missing_dimensions"]),
        "n_applicable_missing_evidence": int(rollup["applicable_missing_evidence"]),
        "n_applicable_missing_reason": int(rollup["applicable_missing_reason"]),
        "n_applicable_zero_support": int(rollup["applicable_zero_support"]),
        "n_non_applicable_missing_reason": int(rollup["non_applicable_missing_reason"]),
        "n_blocking_issues": int(n_blocking),
        "n_warnings": n_warnings,
        "by_domain": dict(sorted(by_domain.items())),
        "by_domain_backend": {
            domain: dict(sorted(backends.items()))
            for domain, backends in sorted(by_domain_backend.items())
        },
        "by_family": dict(sorted(by_family.items())),
        "by_dimension": dict(sorted(by_dimension.items())),
        "examples": examples,
    }
    return {
        "evidence_applicability_audit": audit,
        "notes": [
            "Applicable dimensions must have evidence_ids and a non-empty reason.",
            "Applicable dimensions with support_count <= 0 are warnings because some legacy/cross-domain artifacts may still carry evidence ids with weak support accounting.",
            "Non-applicable dimensions must have a non-empty machine-readable/human-readable reason.",
            "This audit is report-only; it does not change scorer semantics or release artifacts.",
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
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if idx == len(raw_lines) - 1:
                break
            raise
    return rows


def write_markdown(report: dict[str, Any], out_path: Path) -> None:
    audit = report.get("evidence_applicability_audit") or {}
    lines = [
        "# Evidence Applicability Audit",
        "",
        "Checks that scored dimensions obey the evidence contract.",
        "",
        f"- Status: **{audit.get('status', 'unknown')}**",
        f"- Episodes OK: **{audit.get('n_ok', 0)}** / {audit.get('n_total', 0)}",
        f"- Dimensions audited: **{audit.get('n_dimensions', 0)}**",
        f"- Blocking issues: **{audit.get('n_blocking_issues', 0)}**",
        "",
        "## Issue Counts",
        "",
        f"- Applicable missing evidence: **{audit.get('n_applicable_missing_evidence', 0)}**",
        f"- Applicable missing reason: **{audit.get('n_applicable_missing_reason', 0)}**",
        f"- Applicable zero support warnings: **{audit.get('n_applicable_zero_support', 0)}**",
        "- Non-applicable missing reason: "
        f"**{audit.get('n_non_applicable_missing_reason', 0)}**",
        f"- Missing dimensions: **{audit.get('n_missing_dimensions', 0)}**",
    ]
    by_dimension = audit.get("by_dimension") or {}
    if by_dimension:
        lines.extend(["", "## By Dimension", ""])
        for name, bucket in sorted(
            by_dimension.items(),
            key=lambda item: (-int(item[1].get("blocking_issues", 0)), item[0]),
        )[:20]:
            lines.append(
                f"- `{name}`: blocking={bucket.get('blocking_issues', 0)}, "
                f"applicable={bucket.get('applicable', 0)}, "
                f"non_applicable={bucket.get('non_applicable', 0)}"
            )
    examples = audit.get("examples") or {}
    if examples:
        lines.extend(["", "## Examples", ""])
        for kind, rows in sorted(examples.items()):
            lines.append(f"### {kind}")
            for row in rows[:5]:
                lines.append(
                    f"- `{row.get('scenario_id')}` / `{row.get('dimension', 'score')}` "
                    f"({row.get('domain')}/{row.get('backend_kind')})"
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
    report = build_report(_load_jsonl_rows(episodes_path))
    (out_dir / "evidence_applicability_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(report, out_dir / "EVIDENCE_APPLICABILITY.md")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
