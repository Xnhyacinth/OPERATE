#!/usr/bin/env python3
"""Analyze whether LLM tool use changed simulator outcomes vs wait-only CF."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluation.action_taxonomy import summarize_decision_impact  # noqa: E402

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


def _backend_from_row(row: dict) -> str:
    backend_kind = str(row.get("backend_kind") or "").strip()
    if backend_kind:
        return backend_kind
    backend = str(row.get("backend") or "").strip()
    if backend:
        return backend
    family = str(row.get("family") or "").strip()
    if family in FAMILY_TO_BACKEND:
        return FAMILY_TO_BACKEND[family]
    slug = row.get("scenario_slug")
    if not slug:
        return "unknown"
    base = slug.split("/")[-1] if "/" in slug else slug
    if base.startswith(("st_", "wp_")):
        return "grid2op"
    if base.startswith("do_"):
        return "pglib"
    return "other"


def _domain_from_row(row: dict) -> str:
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


def _episode_impact(row: dict) -> dict:
    existing = row.get("decision_impact")
    if existing and "interpretation" in existing:
        return existing
    traj = row.get("trajectory_summary") or {}
    return summarize_decision_impact(
        traj.get("tool_histogram") or {},
        row.get("counterfactual"),
        tool_results_ok=int(traj.get("tool_results_ok", 0)),
        tool_results_failed=int(traj.get("tool_results_failed", 0)),
    )


def _empty_impact_bucket() -> dict:
    return {
        "n": 0,
        "control_episodes": 0,
        "control_calls": 0,
        "control_outcome_changed": 0,
        "control_cost_matched": 0,
        "outcome_changed": 0,
        "helped": 0,
        "hurt": 0,
        "investigation_only": 0,
        "meta_only": 0,
        "tool_exec_failed": 0,
        "counterfactual_applicable": 0,
        "positive_prevented_loss": 0,
        "missing_decision_impact": 0,
        "mean_score": 0.0,
    }


def _add_to_impact_bucket(
    bucket: dict,
    *,
    row: dict,
    impact: dict,
    missing_decision_impact: bool,
) -> None:
    score = row.get("score") or {}
    control_calls = int(impact.get("n_control_calls", 0) or 0)
    outcome_changed = bool(impact.get("outcome_changed", False))
    interpretation = str(impact.get("interpretation") or "")
    prevented_loss = float(impact.get("prevented_loss", 0.0) or 0.0)
    counterfactual = row.get("counterfactual") or {}
    counterfactual_applicable = bool(counterfactual.get("applicable", True))

    bucket["n"] += 1
    bucket["control_calls"] += control_calls
    bucket["mean_score"] += float(score.get("total_score", 0.0) or 0.0)
    bucket["tool_exec_failed"] += int(impact.get("tool_results_failed", 0) or 0)
    if missing_decision_impact:
        bucket["missing_decision_impact"] += 1
    if control_calls > 0:
        bucket["control_episodes"] += 1
    if control_calls > 0 and outcome_changed:
        bucket["control_outcome_changed"] += 1
    if control_calls > 0 and not outcome_changed:
        bucket["control_cost_matched"] += 1
    if outcome_changed:
        bucket["outcome_changed"] += 1
    if bool(impact.get("agent_helped", False)):
        bucket["helped"] += 1
    if bool(impact.get("agent_hurt", False)):
        bucket["hurt"] += 1
    if bool(impact.get("investigation_only_episode", False)):
        bucket["investigation_only"] += 1
    if interpretation == "no_action_taken_meta_only":
        bucket["meta_only"] += 1
    if counterfactual_applicable:
        bucket["counterfactual_applicable"] += 1
    if prevented_loss > 1.0:
        bucket["positive_prevented_loss"] += 1


def _finalize_impact_bucket(bucket: dict) -> dict:
    n = bucket["n"] or 1
    out = dict(bucket)
    out["mean_score"] = round(float(out["mean_score"]) / n, 2)
    out["control_calls_per_episode"] = round(float(out["control_calls"]) / n, 2)
    return out


def _impact_examples(
    examples: dict[str, list[dict[str, object]]], category: str, row: dict, impact: dict
) -> None:
    bucket = examples.setdefault(category, [])
    if len(bucket) >= 5:
        return
    bucket.append(
        {
            "model": row.get("model", row.get("agent_name")),
            "scenario_id": row.get("scenario_id"),
            "scenario_slug": row.get("scenario_slug"),
            "family": row.get("family"),
            "backend_kind": _backend_from_row(row),
            "domain": _domain_from_row(row),
            "interpretation": impact.get("interpretation"),
            "n_control_calls": impact.get("n_control_calls", 0),
            "prevented_loss": impact.get("prevented_loss", 0.0),
        }
    )


def build_report(
    rows: list[dict], *, expected_domains: list[str] | None = None
) -> dict:
    ok = [r for r in rows if r.get("status") == "ok"]
    by_model: dict[str, dict] = {}
    by_model_backend: dict[str, dict[str, dict]] = defaultdict(dict)
    by_domain: dict[str, dict] = defaultdict(_empty_impact_bucket)
    by_domain_backend: dict[str, dict[str, dict]] = defaultdict(
        lambda: defaultdict(_empty_impact_bucket)
    )
    by_family: dict[str, dict] = defaultdict(_empty_impact_bucket)
    interpretations: Counter[str] = Counter()
    flags: list[str] = []
    warnings: list[str] = []
    blockers: list[str] = []
    audit_examples: dict[str, list[dict[str, object]]] = {}
    missing_decision_impact = 0
    control_cost_matched = 0
    control_outcome_changed = 0
    positive_prevented_loss = 0
    counterfactual_applicable = 0
    control_episodes = 0

    for r in ok:
        model = str(r.get("model", r.get("agent_name", "?")))
        domain = _domain_from_row(r)
        backend = _backend_from_row(r)
        family = str(r.get("family") or "unknown")
        had_decision_impact = bool(r.get("decision_impact"))
        impact = _episode_impact(r)
        if not had_decision_impact:
            missing_decision_impact += 1
            _impact_examples(audit_examples, "missing_decision_impact", r, impact)
        interpretations[str(impact.get("interpretation") or "unknown")] += 1

        bucket = by_model.setdefault(
            model,
            {
                "n": 0,
                "outcome_changed": 0,
                "helped": 0,
                "hurt": 0,
                "investigation_only": 0,
                "control_episodes": 0,
                "control_calls": 0,
                "tool_exec_failed": 0,
                "llm_failed_eps": 0,
                "scores": [],
            },
        )
        control_calls = int(impact.get("n_control_calls", 0) or 0)
        b = bucket
        b["n"] += 1
        b["scores"].append(float(r["score"]["total_score"]))
        if bool(impact.get("outcome_changed", False)):
            b["outcome_changed"] += 1
        if bool(impact.get("agent_helped", False)):
            b["helped"] += 1
        if bool(impact.get("agent_hurt", False)):
            b["hurt"] += 1
        if bool(impact.get("investigation_only_episode", False)):
            b["investigation_only"] += 1
        if control_calls > 0:
            b["control_episodes"] += 1
            control_episodes += 1
        b["control_calls"] += control_calls
        b["tool_exec_failed"] += int(impact.get("tool_results_failed", 0) or 0)
        llm = (r.get("trajectory_summary") or {}).get("llm") or {}
        if int(llm.get("llm_calls_failed", 0)) > 0:
            b["llm_failed_eps"] += 1

        mb = by_model_backend[model].setdefault(
            backend,
            {"n": 0, "mean_score": 0.0, "outcome_changed": 0},
        )
        mb["n"] += 1
        mb["mean_score"] += float(r["score"]["total_score"])
        if bool(impact.get("outcome_changed", False)):
            mb["outcome_changed"] += 1

        _add_to_impact_bucket(
            by_domain[domain],
            row=r,
            impact=impact,
            missing_decision_impact=not had_decision_impact,
        )
        _add_to_impact_bucket(
            by_domain_backend[domain][backend],
            row=r,
            impact=impact,
            missing_decision_impact=not had_decision_impact,
        )
        _add_to_impact_bucket(
            by_family[family],
            row=r,
            impact=impact,
            missing_decision_impact=not had_decision_impact,
        )
        by_family[family]["backend_kind"] = backend
        by_family[family]["domain"] = domain

        cf = r.get("counterfactual") or {}
        if bool(cf.get("applicable", True)):
            counterfactual_applicable += 1
        if float(impact.get("prevented_loss", 0.0) or 0.0) > 1.0:
            positive_prevented_loss += 1
            _impact_examples(audit_examples, "positive_prevented_loss", r, impact)
        if control_calls > 0 and bool(impact.get("outcome_changed", False)):
            control_outcome_changed += 1
            _impact_examples(audit_examples, "control_outcome_changed", r, impact)
        if control_calls > 0 and not bool(impact.get("outcome_changed", False)):
            control_cost_matched += 1
            _impact_examples(audit_examples, "control_cost_matched", r, impact)
            flags.append(
                f"{model} {r.get('scenario_id')} s{r.get('seed')}: "
                "control tools but |prevented_loss|<=1"
            )
        traj = r.get("trajectory_summary") or {}
        if (
            int(traj.get("n_tool_calls", 0) or 0) > 0
            and int(impact.get("tool_results_failed", 0) or 0) > 5
        ):
            flags.append(
                f"{model} {r.get('scenario_id')}: high tool_results_failed="
                f"{impact.get('tool_results_failed', 0)}"
            )

    for _model, b in by_model.items():
        n = b["n"] or 1
        b["mean_score"] = sum(b["scores"]) / n
        b.pop("scores", None)
        b["control_calls_per_episode"] = round(b["control_calls"] / n, 2)

    for _model, backends in by_model_backend.items():
        for _backend, mb in backends.items():
            n = mb["n"] or 1
            mb["mean_score"] = round(mb["mean_score"] / n, 2)

    by_domain_final = {
        domain: _finalize_impact_bucket(bucket)
        for domain, bucket in sorted(by_domain.items())
    }
    by_domain_backend_final = {
        domain: {
            backend: _finalize_impact_bucket(bucket)
            for backend, bucket in sorted(backends.items())
        }
        for domain, backends in sorted(by_domain_backend.items())
    }
    by_family_final = {
        family: _finalize_impact_bucket(bucket)
        for family, bucket in sorted(by_family.items())
    }

    missing_expected_domains: list[str] = []
    if expected_domains:
        seen_domains = set(by_domain_final)
        missing_expected_domains = sorted(
            domain for domain in expected_domains if domain not in seen_domains
        )
        if missing_expected_domains:
            blockers.append(
                "missing expected domain coverage: "
                + ", ".join(missing_expected_domains)
            )
    if missing_decision_impact:
        warnings.append(
            f"{missing_decision_impact} ok episode(s) lacked precomputed decision_impact"
        )
    if control_cost_matched:
        warnings.append(
            f"{control_cost_matched} control episode(s) matched wait-only cost"
        )

    audit_status = "pass"
    if blockers:
        audit_status = "blocking"
    elif warnings:
        audit_status = "warning"

    return {
        "n_total": len(rows),
        "n_ok": len(ok),
        "interpretation_counts": dict(interpretations),
        "by_model": by_model,
        "by_model_backend": dict(by_model_backend),
        "by_domain": by_domain_final,
        "by_domain_backend": by_domain_backend_final,
        "by_family": by_family_final,
        "action_impact_audit": {
            "schema_version": "0.1",
            "status": audit_status,
            "n_ok": len(ok),
            "n_control_episodes": control_episodes,
            "n_control_outcome_changed_episodes": control_outcome_changed,
            "n_control_cost_matched_episodes": control_cost_matched,
            "n_counterfactual_applicable_episodes": counterfactual_applicable,
            "n_positive_prevented_loss_episodes": positive_prevented_loss,
            "n_missing_decision_impact": missing_decision_impact,
            "expected_domains": list(expected_domains or []),
            "missing_expected_domains": missing_expected_domains,
            "warnings": warnings,
            "blockers": blockers,
            "examples": audit_examples,
        },
        "warnings": warnings,
        "blockers": blockers,
        "flags": flags[:100],
        "notes": [
            "prevented_loss = counterfactual_cost - actual_cost (wait_only masked replay).",
            "investigation_only_episode: zero control tools; cost often equals wait-only CF.",
            "action_impact_audit is report-only unless expected domains are supplied by the caller.",
            "outcome_changed: |prevented_loss| > 1.0 USD-scale cost aggregate.",
        ],
    }


def _load_jsonl_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
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


def write_markdown(report: dict, out_path: Path) -> None:
    lines = [
        "# Decision impact analysis",
        "",
        "Measures whether agent tool calls changed final simulator cost vs a "
        "**wait-only counterfactual replay** (same seed, masked actions).",
        "",
        f"- Episodes OK: **{report['n_ok']}** / {report['n_total']}",
        "",
        "## Interpretation mix",
        "",
    ]
    for k, v in sorted(
        (report.get("interpretation_counts") or {}).items(), key=lambda x: -x[1]
    ):
        lines.append(f"- `{k}`: {v}")
    audit = report.get("action_impact_audit") or {}
    if audit:
        lines.extend(
            [
                "",
                "## Action-impact audit",
                "",
                f"- Status: **{audit.get('status', 'unknown')}**",
                f"- Control episodes: **{audit.get('n_control_episodes', 0)}**",
                "- Control outcome changed / cost-matched: "
                f"**{audit.get('n_control_outcome_changed_episodes', 0)}** / "
                f"**{audit.get('n_control_cost_matched_episodes', 0)}**",
                "- Positive prevented-loss episodes: "
                f"**{audit.get('n_positive_prevented_loss_episodes', 0)}**",
                "- Missing precomputed `decision_impact`: "
                f"**{audit.get('n_missing_decision_impact', 0)}**",
            ]
        )
        if audit.get("missing_expected_domains"):
            lines.append(
                "- Missing expected domains: "
                + ", ".join(f"`{d}`" for d in audit["missing_expected_domains"])
            )
    lines.extend(["", "## By model", ""])
    for model, b in sorted((report.get("by_model") or {}).items()):
        lines.append(f"### {model}")
        lines.append(f"- Mean score: **{b.get('mean_score', 0):.2f}** (n={b.get('n')})")
        lines.append(
            f"- Outcome changed vs wait-only: **{b.get('outcome_changed')}/{b.get('n')}**"
        )
        lines.append(f"- Helped / hurt: **{b.get('helped')}** / **{b.get('hurt')}**")
        lines.append(
            f"- Investigation-only episodes: **{b.get('investigation_only')}**"
        )
        lines.append(
            f"- Episodes with control tools: **{b.get('control_episodes')}**; "
            f"avg control calls/ep: **{b.get('control_calls_per_episode', 0)}**"
        )
        lines.append(f"- Episodes with LLM API failures: **{b.get('llm_failed_eps')}**")
        lines.append("")
    if report.get("flags"):
        lines.extend(["## Flags (sample)", ""])
        for f in report["flags"][:20]:
            lines.append(f"- {f}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output_dir", type=Path)
    p.add_argument(
        "--expected-domains",
        default="",
        help="Optional comma-separated domain coverage list for audit status.",
    )
    args = p.parse_args()
    out_dir = args.output_dir.resolve()
    ep_path = out_dir / "episodes.jsonl"
    if not ep_path.is_file():
        print(f"missing {ep_path}", file=sys.stderr)
        return 1
    rows = _load_jsonl_rows(ep_path)
    expected_domains = [
        d.strip() for d in args.expected_domains.split(",") if d.strip()
    ]
    report = build_report(rows, expected_domains=expected_domains or None)
    (out_dir / "decision_impact_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_markdown(report, out_dir / "DECISION_IMPACT.md")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
