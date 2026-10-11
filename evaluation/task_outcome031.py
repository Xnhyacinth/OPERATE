"""Reference-anchored 0.31 outcomes from authenticated 0.30 case evidence.

Each case is placed on a commensurate scale where the source-seeded do-nothing
controller scores 0 and closing the do-nothing-to-reference cost gap scores
100; the joint score reaches 100 only with full service. Native cost and
service fulfillment are anchored separately and combined as a bottleneck. The 0.30 report supplies authenticated C, F and hard gates;
the original 0.30 case scores remain alongside, unchanged.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math

import hashlib
import json
from pathlib import Path

from evaluation.operational_weights import (
    compile_task_weights,
    hierarchical_case_weights,
    validate_task_weights,
)
from evaluation.reference_anchors031 import validate_reference_anchors
from evaluation.trajectory_outcome027 import canonical_digest

VERSION = "0.31.0"
REVISION = "reference_anchored_operational_outcome.v1"
VERSION_0311 = "0.31.1"
REVISION_0311 = "reference_anchored_operational_outcome.v2_continuous"
VERSION_0312 = "0.31.2"
REVISION_0312 = "reference_anchored_operational_outcome.v3_service_bounded"
COMBINATION_V1 = (
    "min(cost_gap_closed,service_shortfall_closed);full_service_non_binding"
)
COMBINATION_V2 = (
    "continuous_v2:min(axis_caps);service_cap_lifts_to_ceiling_over_last_ramp_points;"
    "regression_guards_fall_over_ramp_points"
)
COMBINATION_V3 = (
    "continuous_v3:min(cost_cap,service_cap);joint_ceiling_100;no_service_uplift"
)
SCORE_BOUNDS_V3 = (-100.0, 100.0)
RAMP_WIDTH = 5.0
POLICY_SCHEMA = "operate_anchored_scoring_policy031.v1"
HEADLINE_WEIGHT_POLICY = "equal_domain_then_case"
HARD_FAILURE_SCORE = -100.0
SCORE_BOUNDS = (-100.0, 150.0)
DRIVING_DOMAIN = "autonomous_driving"


def _repo_ref(path: Path, root: Path) -> dict:
    raw = Path(path).read_bytes()
    return dict(
        path=str(Path(path).resolve().relative_to(Path(root).resolve())),
        sha256=hashlib.sha256(raw).hexdigest(),
        byte_count=len(raw),
    )


def _read_repo_ref(ref: dict, root: Path) -> bytes:
    raw = (Path(root) / ref["path"]).read_bytes()
    if (
        hashlib.sha256(raw).hexdigest() != ref["sha256"]
        or len(raw) != ref["byte_count"]
    ):
        raise ValueError("policy031_bound_artifact_mismatch")
    return raw


def compile_policy031(
    anchors_path: Path,
    *,
    root: Path,
    base_policy_path: Path,
    suite_path: Path,
    evaluation_version: str = VERSION_0312,
) -> dict:
    """Bind anchors, base policy, fixed weights and rules; reads no model data."""
    if evaluation_version not in (VERSION, VERSION_0311, VERSION_0312):
        raise ValueError("unsupported_policy031_evaluation_version")
    continuous = evaluation_version != VERSION
    bounded = evaluation_version == VERSION_0312
    anchors = json.loads(Path(anchors_path).read_bytes())
    policy = dict(
        schema_version=POLICY_SCHEMA,
        evaluation_version=evaluation_version,
        protocol_revision=REVISION_0312
        if bounded
        else REVISION_0311
        if continuous
        else REVISION,
        base_policy=_repo_ref(base_policy_path, root),
        reference_anchors=dict(
            _repo_ref(anchors_path, root), anchors_sha256=anchors["anchors_sha256"]
        ),
        weight_manifest=compile_task_weights(
            suite_path, root=root, weight_policy=HEADLINE_WEIGHT_POLICY
        ),
        rules=dict(
            h_min=anchors["h_min"],
            service_delta=anchors["service_delta"],
            score_bounds=list(SCORE_BOUNDS_V3 if bounded else SCORE_BOUNDS),
            hard_failure_score=HARD_FAILURE_SCORE,
            lower_anchor=anchors["lower_policy"],
            upper_anchor="lowest_cost_non_regressing_of:"
            + ",".join(anchors["upper_policies"]),
            cost_objective="settled_native_objective_C",
            combination=COMBINATION_V3
            if bounded
            else COMBINATION_V2
            if continuous
            else COMBINATION_V1,
            headline_weights="nominal_weights_of_eligible_cases_renormalized_globally",
        ),
    )
    if continuous:
        policy["rules"]["ramp_width"] = RAMP_WIDTH
    policy["policy_sha256"] = canonical_digest(policy)
    return policy


def load_policy031(path: Path, *, root: Path) -> dict:
    """Verify the policy self-hash and every bound artifact before use."""
    policy = json.loads(Path(path).read_bytes())
    if policy.get("schema_version") != POLICY_SCHEMA:
        raise ValueError("unsupported_policy031_schema")
    if policy.get("policy_sha256") != canonical_digest(
        {k: v for k, v in policy.items() if k != "policy_sha256"}
    ):
        raise ValueError("policy031_hash_mismatch")
    expected = {
        VERSION: (REVISION, COMBINATION_V1, SCORE_BOUNDS),
        VERSION_0311: (REVISION_0311, COMBINATION_V2, SCORE_BOUNDS),
        VERSION_0312: (REVISION_0312, COMBINATION_V3, SCORE_BOUNDS_V3),
    }.get(policy.get("evaluation_version"))
    if (
        expected is None
        or (
            policy.get("protocol_revision"),
            policy["rules"].get("combination"),
            tuple(policy["rules"].get("score_bounds", [])),
        )
        != expected
    ):
        raise ValueError("policy031_version_rules_mismatch")
    if policy.get("evaluation_version") == VERSION_0312 and (
        policy["rules"].get("hard_failure_score") != HARD_FAILURE_SCORE
        or not _finite(policy["rules"].get("ramp_width"))
        or policy["rules"]["ramp_width"] <= 0
    ):
        raise ValueError("policy031_invalid_failure_or_ramp_rule")
    _read_repo_ref(policy["base_policy"], root)
    anchors = json.loads(_read_repo_ref(policy["reference_anchors"], root))
    if (
        anchors.get("anchors_sha256") != policy["reference_anchors"]["anchors_sha256"]
        or anchors["h_min"] != policy["rules"]["h_min"]
        or anchors["service_delta"] != policy["rules"]["service_delta"]
        or policy["weight_manifest"].get("weight_policy") != HEADLINE_WEIGHT_POLICY
    ):
        raise ValueError("policy031_rules_anchor_mismatch")
    return {**policy, "reference_anchors": anchors}


def _finite(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _clip(value: float) -> float:
    return min(SCORE_BOUNDS[1], max(SCORE_BOUNDS[0], value))


def cost_axis(cost: float, anchor: dict, *, h_min: float) -> dict:
    """Fraction of the do-nothing-to-reference cost gap closed, in percent.

    Without reference headroom the cost axis only guards against regression
    beyond the do-nothing cost; it cannot reward an unanchored improvement.
    """
    wait, ref, scale = anchor["C_wait"], anchor["C_ref"], anchor["cost_scale"]
    if anchor["cost_axis_informative"]:
        return dict(informative=True, value=100.0 * (wait - cost) / (wait - ref))
    if cost <= wait + h_min * scale:
        return dict(informative=False, value=None)
    return dict(informative=False, value=-100.0 * (cost - wait) / scale)


def service_axis(fulfillment: float | None, anchor: dict, *, delta: float) -> dict:
    """Shortfall closed toward the full source obligation, in percent.

    Meeting the full obligation never binds, so it cannot cap superior cost.
    Without do-nothing shortfall, service guards against regression only.
    """
    if not anchor["service_applicable"]:
        return dict(informative=False, value=None)
    wait = anchor["F_wait"]
    if fulfillment >= 100.0 - delta:
        return dict(informative=anchor["service_axis_informative"], value=None)
    if anchor["service_axis_informative"]:
        return dict(
            informative=True, value=100.0 * (fulfillment - wait) / (100.0 - wait)
        )
    if fulfillment >= wait - delta:
        return dict(informative=False, value=None)
    return dict(
        informative=False, value=-100.0 * (wait - fulfillment) / max(wait, delta)
    )


def _lift(cap: float, position: float, *, ceiling: float = SCORE_BOUNDS[1]) -> float:
    """Raise a cap toward the score ceiling; position 1 removes the cap."""
    position = min(1.0, max(0.0, position))
    return cap + (ceiling - cap) * position


def combine_continuous(
    cost: float,
    fulfillment,
    anchor: dict,
    *,
    h_min: float,
    delta: float,
    ramp: float,
    service_bounded: bool = False,
) -> tuple[float, float | None, float | None, str]:
    """Continuous bottleneck for the historical V2 and service-bounded V3.

    The score is the minimum over axis caps. The service cap equals f until the
    last ``ramp`` points of the do-nothing shortfall, then rises linearly to the
    score ceiling at full fulfillment. Regression guards on uninformative axes
    impose no cap inside their tolerance and fall linearly to the 0.31.0
    regression value over the next ``ramp`` points. Outside these bands cost
    savings cannot offset unserved obligations. V3 removes the service uplift,
    caps already-fulfilled service by its measured F, and uses ceiling 100.
    Regression guards retain their explicit tolerance and transition bands.
    """
    wait, ref, scale = anchor["C_wait"], anchor["C_ref"], anchor["cost_scale"]
    ceiling = 100.0 if service_bounded else SCORE_BOUNDS[1]
    caps, g, f = {}, None, None
    if anchor["cost_axis_informative"]:
        g = caps["cost"] = 100.0 * (wait - cost) / (wait - ref)
    else:
        loss = 100.0 * (cost - wait) / scale
        caps["cost_regression"] = _lift(
            -loss, 1.0 - (loss - 100.0 * h_min) / ramp, ceiling=ceiling
        )
    if anchor["service_applicable"]:
        f_wait = anchor["F_wait"]
        if anchor["service_axis_informative"]:
            f = 100.0 * (fulfillment - f_wait) / (100.0 - f_wait)
            caps["service"] = (
                f if service_bounded else _lift(f, (f - (100.0 - ramp)) / ramp)
            )
        else:
            shortfall = f_wait - fulfillment
            caps["service_regression"] = _lift(
                -100.0 * shortfall / max(f_wait, delta),
                1.0 - (shortfall - delta) / ramp,
                ceiling=ceiling,
            )
            if service_bounded:
                # An already-fulfilled baseline has no shortfall to normalize.
                # Keep its regression guard, but never call incomplete service 100.
                caps["service_completion"] = fulfillment
    if g is None and f is None:
        caps["none"] = 100.0 if anchor["headline_eligible"] else 0.0
    binding = min(caps, key=lambda k: (caps[k], k))
    return caps[binding], g, f, binding


def score_case031(
    cost,
    fulfillment,
    hard_failure,
    anchor: dict,
    *,
    h_min: float,
    delta: float,
    combination: str = COMBINATION_V1,
    ramp: float = 5.0,
) -> dict:
    """One trajectory's anchored score; unknown evidence is never imputed."""
    if combination not in (COMBINATION_V1, COMBINATION_V2, COMBINATION_V3):
        raise ValueError("unsupported_combination_rule")
    bounds = SCORE_BOUNDS_V3 if combination == COMBINATION_V3 else SCORE_BOUNDS
    result = dict(
        S=None,
        S_unclipped=None,
        g=None,
        f=None,
        binding_axis=None,
        determined=False,
        hard_failure=hard_failure,
        missing_bounds=list(bounds),
        reason=None,
    )
    if hard_failure is True:
        result.update(
            S=HARD_FAILURE_SCORE,
            S_unclipped=HARD_FAILURE_SCORE,
            determined=True,
            binding_axis="hard_failure",
            missing_bounds=[HARD_FAILURE_SCORE] * 2,
            reason="verified_native_hard_failure",
        )
        return result
    if hard_failure is not False:
        result["reason"] = "hard_gate_state_unknown"
        return result
    if not _finite(cost):
        result["reason"] = "native_cost_missing"
        return result
    if anchor["service_applicable"] and not _finite(fulfillment):
        result["reason"] = "fulfillment_missing"
        return result
    reason = (
        "reference_anchored_joint_outcome"
        if anchor["service_applicable"]
        else "reference_anchored_cost_only"
    )
    if combination in (COMBINATION_V2, COMBINATION_V3):
        raw, g_value, f_value, binding = combine_continuous(
            cost,
            fulfillment,
            anchor,
            h_min=h_min,
            delta=delta,
            ramp=ramp,
            service_bounded=combination == COMBINATION_V3,
        )
        score = min(bounds[1], max(bounds[0], raw))
        result.update(
            S=score,
            S_unclipped=raw,
            g=g_value,
            f=f_value,
            binding_axis=binding,
            determined=True,
            missing_bounds=[score, score],
            reason=reason,
        )
        return result
    g = cost_axis(cost, anchor, h_min=h_min)
    f = service_axis(fulfillment, anchor, delta=delta)
    axes = {
        name: axis["value"]
        for name, axis in (("cost", g), ("service", f))
        if axis["value"] is not None
    }
    if axes:
        binding = min(axes, key=lambda k: (axes[k], k))
        raw = axes[binding]
    else:
        binding, raw = "none_binding", 100.0 if anchor["headline_eligible"] else 0.0
    score = _clip(raw)
    result.update(
        S=score,
        S_unclipped=raw,
        g=g["value"],
        f=f["value"],
        binding_axis=binding,
        determined=True,
        missing_bounds=[score, score],
        reason="reference_anchored_joint_outcome"
        if anchor["service_applicable"]
        else "reference_anchored_cost_only",
    )
    return result


def headline_weights(nominal: dict, anchors: dict, *, exclude_domains=()) -> dict:
    """Drop ineligible cases and renormalize globally, never within a domain.

    A domain with few informative cases loses mass instead of letting one case
    stand for the whole domain.
    """
    eligible = {
        k
        for k, a in anchors.items()
        if a["headline_eligible"] and a["domain"] not in exclude_domains
    }
    mass = math.fsum(nominal[k] for k in eligible)
    if not eligible or mass <= 0:
        raise ValueError("no_headline_eligible_cases")
    return {k: nominal[k] / mass for k in sorted(eligible)}


def aggregate031(
    cases: dict,
    weights: dict,
    models: list[str],
    anchors: dict,
    *,
    score_bounds: tuple = SCORE_BOUNDS,
) -> dict:
    """Fixed-weight index; incomplete models keep bounds and receive no rank."""
    if not math.isclose(math.fsum(weights.values()), 1.0, abs_tol=1e-12):
        raise ValueError("headline_weights_must_sum_to_one")
    table = []
    for model in models:
        values = {k: cases.get((model, *k)) for k in weights}
        known = {k: v["S"] for k, v in values.items() if v and v["determined"]}
        complete = len(known) == len(weights)
        bounds = [
            math.fsum(
                weights[k]
                * (values[k]["missing_bounds"][j] if values[k] else score_bounds[j])
                for k in weights
            )
            for j in (0, 1)
        ]
        domains = {}
        for domain in sorted({anchors[k]["domain"] for k in weights}):
            keys = [k for k in weights if anchors[k]["domain"] == domain]
            mass = math.fsum(weights[k] for k in keys)
            measured = [k for k in keys if k in known]
            domains[domain] = dict(
                weight_mass=mass,
                score=math.fsum(weights[k] * known[k] for k in keys) / mass
                if len(measured) == len(keys)
                else None,
                measured_cases=len(measured),
                expected_cases=len(keys),
            )
        hard = sum(bool(v and v["hard_failure"] is True) for v in values.values())
        binding = Counter(
            v["binding_axis"] for v in values.values() if v and v["determined"]
        )
        table.append(
            dict(
                model=model,
                primary_score=math.fsum(weights[k] * known[k] for k in weights)
                if complete
                else None,
                primary_rank=None,
                complete=complete,
                n_expected=len(weights),
                n_scored=len(known),
                missing_measurement_bounds=bounds,
                domains=domains,
                hard_failure_cases=hard,
                binding_axis_counts=dict(sorted(binding.items())),
                reason_counts=dict(
                    Counter(
                        (v["reason"] if v else "missing_selected_case")
                        for v in values.values()
                    )
                ),
            )
        )
    ordered = sorted(
        (r for r in table if r["complete"]),
        key=lambda r: (-r["primary_score"], r["model"]),
    )
    previous = rank = None
    for position, row in enumerate(ordered, 1):
        if row["primary_score"] != previous:
            rank = position
        row["primary_rank"], previous = rank, row["primary_score"]
    return dict(models=table, leaderboard=[r["model"] for r in ordered])


def _row_inputs(row: dict) -> tuple:
    outcome = row.get("outcome030") or {}
    if outcome.get("evidence_qualified") is not True:
        return None, None, None
    return row.get("C"), row.get("F"), outcome.get("hard_failure")


def _row_loss030(row: dict):
    """0.30 population-normalized voltage loss (LV/CIGRE 0.30, DSS 0.28)."""
    for name in ("normalization030", "normalization028"):
        loss = (row.get(name) or {}).get("dimensionless_loss")
        if _finite(loss):
            return loss
    return None


def loss_anchor(anchor: dict, *, h_min: float) -> dict | None:
    wait, ref = anchor.get("loss030_wait"), anchor.get("loss030_ref")
    if not (_finite(wait) and _finite(ref)):
        return None
    scale = max(abs(wait), abs(ref), 1e-9)
    return dict(
        anchor,
        C_wait=wait,
        C_ref=ref,
        cost_scale=scale,
        cost_axis_informative=wait - ref >= h_min * scale,
    )


def substitute_domain_rows(base: dict, supplements: list[dict], *, domain: str) -> dict:
    """Replace one domain's rows with explicitly mapped supplement rows.

    ``supplements`` items: {"report": 0.30 report, "sha256": str, "label": str,
    "models": {base_model: supplement_model}}. Base models without a mapped
    supplement row become unknown for that domain; no historical row is kept.
    """
    contracts = {
        (c["scenario_signature"], c["seed"]): c
        for c in base["source_contracts"]["contracts"]
    }
    keys = {k for k, c in contracts.items() if c["domain"] == domain}
    report = deepcopy(base)
    provided = {}
    for supplement in supplements:
        other = supplement["report"]
        if (
            other.get("evaluation_version") != "0.30.0"
            or other["source_contracts"] != base["source_contracts"]
        ):
            raise ValueError("supplement_source_contract_mismatch")
        rows = {
            (r["model"], r["scenario_signature"], r["seed"]): r
            for r in other["graded_episodes"]
        }
        for base_model, model in supplement["models"].items():
            for key in keys:
                row = rows.get((model, *key))
                if row is None:
                    continue
                if (base_model, *key) in provided:
                    raise ValueError("supplement_case_provided_twice")
                provided[(base_model, *key)] = dict(
                    deepcopy(row),
                    model=base_model,
                    substitution=dict(
                        source_report_sha256=supplement["sha256"],
                        label=supplement["label"],
                        supplement_model=model,
                    ),
                )
    selected = []
    for row in report["graded_episodes"]:
        key = (row["scenario_signature"], row["seed"])
        if key not in keys:
            selected.append(row)
            continue
        replacement = provided.pop((row["model"], *key), None)
        if replacement is None:
            replacement = dict(
                model=row["model"],
                scenario_signature=key[0],
                seed=key[1],
                C=None,
                F=None,
                Q=None,
                outcome030=dict(
                    evidence_qualified=False,
                    hard_failure=None,
                    reason="historical_domain_superseded_without_corrected_retest",
                ),
                substitution=dict(
                    source_report_sha256=None,
                    label="no_corrected_retest",
                    superseded_Q030=row.get("Q"),
                ),
            )
        selected.append(replacement)
    if provided:
        raise ValueError("supplement_row_for_undeclared_model")
    report["graded_episodes"] = selected
    report["domain_substitution"] = dict(
        domain=domain,
        supplements=[
            {k: s[k] for k in ("sha256", "label", "models")} for s in supplements
        ],
        interpretation="descriptive_cross_implementation_substitution_execution_identities_retained",
    )
    return report


def build_anchored_outcomes(
    base: dict, policy: dict, *, sensitivity: bool = True
) -> dict:
    """Score every graded row of an authenticated 0.30 report under 0.31."""
    if base.get("evaluation_version") != "0.30.0":
        raise ValueError("outcome031_requires_outcome030")
    rules = policy["rules"]
    anchors = validate_reference_anchors(
        policy["reference_anchors"], base["source_contracts"]
    )
    base_policy = policy.get("base_policy")
    if base_policy is not None and (base.get("policy") or {}).get(
        "sha256"
    ) != base_policy.get("sha256"):
        raise ValueError("outcome031_base_policy_mismatch")
    nominal = validate_task_weights(policy["weight_manifest"], base["source_contracts"])
    if nominal.keys() != anchors.keys():
        raise ValueError("outcome031_weight_coverage_mismatch")
    models = [m["model"] for m in base["models"]]
    cases, rows = {}, []
    for row in base["graded_episodes"]:
        key = (row["scenario_signature"], row["seed"])
        anchor = anchors[key]
        cost, fulfillment, hard = _row_inputs(row)
        case = score_case031(
            cost,
            fulfillment,
            hard,
            anchor,
            h_min=rules["h_min"],
            delta=rules["service_delta"],
            combination=rules["combination"],
            ramp=rules.get("ramp_width", RAMP_WIDTH),
        )
        case.update(
            model=row["model"],
            scenario_signature=key[0],
            seed=key[1],
            domain=anchor["domain"],
            task_family=anchor["task_family"],
            backend_kind=anchor["backend_kind"],
            C=cost,
            F=fulfillment,
            C_wait=anchor["C_wait"],
            C_ref=anchor["C_ref"],
            F_wait=anchor["F_wait"],
            ref_policy=anchor["ref_policy"],
            headline_eligible=anchor["headline_eligible"],
            Q030=row.get("Q"),
            N030=row.get("N"),
            loss030=_row_loss030(row),
            canonical_episode_sha256=(row.get("origin") or {}).get(
                "canonical_episode_sha256"
            ),
            evidence_ids=list((row.get("outcome030") or {}).get("evidence_ids") or []),
            anchor_ref=dict(
                anchors_sha256=policy["reference_anchors"]["anchors_sha256"]
            ),
            substitution=row.get("substitution"),
        )
        if (row["model"], *key) in cases:
            raise ValueError("duplicate_graded_case")
        cases[(row["model"], *key)] = case
        rows.append(case)
    weights = headline_weights(nominal, anchors)
    aggregate = aggregate031(
        cases, weights, models, anchors, score_bounds=tuple(rules["score_bounds"])
    )
    base_models = {m["model"]: m for m in base["models"]}
    for model in aggregate["models"]:
        model["Q030"] = base_models[model["model"]].get("primary_score")
    report = dict(
        schema_version="operate_reference_anchored_outcome031_report.v1",
        evaluation_version=policy.get("evaluation_version", VERSION),
        protocol_revision=policy.get("protocol_revision", REVISION),
        base_evaluation_version="0.30.0",
        policy_sha256=policy["policy_sha256"],
        rules=deepcopy(rules),
        anchors_sha256=policy["reference_anchors"]["anchors_sha256"],
        weight_policy=policy["weight_manifest"]["weight_policy"],
        weight_manifest_sha256=policy["weight_manifest"]["manifest_sha256"],
        headline_weights=[
            dict(scenario_signature=k[0], seed=k[1], weight=w)
            for k, w in weights.items()
        ],
        ineligible_cases=[
            dict(
                scenario_signature=k[0],
                seed=k[1],
                domain=a["domain"],
                task_family=a["task_family"],
                **a["ineligibility"],
            )
            for k, a in sorted(anchors.items())
            if not a["headline_eligible"]
        ],
        domain_weight_mass={
            d: math.fsum(w for k, w in weights.items() if anchors[k]["domain"] == d)
            for d in sorted({a["domain"] for a in anchors.values()})
        },
        weight_concentration=dict(
            maximum_weight=max(weights.values()),
            weight_equivalent_case_count=1 / math.fsum(w * w for w in weights.values()),
        ),
        cases=rows,
        **aggregate,
        domain_substitution=base.get("domain_substitution"),
        primary_is_binary_task_attainment=False,
        interpretation="percent_of_do_nothing_to_best_reference_gap_closed_with_service_bottleneck",
    )
    report["anchor_semantics"] = anchor_semantics(anchors, rules)
    if sensitivity:
        report["sensitivity"] = sensitivities(cases, nominal, anchors, models, policy)
    report["headline_weights_sha256"] = canonical_digest(report["headline_weights"])
    return report


def _privileged(policy: str) -> bool:
    return policy == "oracle_offline" or policy.startswith(
        "citylearn_tariff_reference@"
    )


def anchor_semantics(anchors: dict, rules: dict) -> dict:
    """What 0 and 100 mean on this panel, measured on the references themselves."""
    eligible = {k: a for k, a in anchors.items() if a["headline_eligible"]}
    joint = []
    for a in eligible.values():
        ref = a["references"][a["ref_policy"]]
        joint.append(
            score_case031(
                ref["C"],
                ref["F"],
                ref["hard_failure"],
                a,
                h_min=rules["h_min"],
                delta=rules["service_delta"],
                combination=rules["combination"],
                ramp=rules.get("ramp_width", RAMP_WIDTH),
            )["S"]
        )
    policies = Counter(a["ref_policy"] for a in eligible.values())
    return dict(
        zero="seeded_do_nothing_controller_on_every_case",
        hundred=(
            "reference_cost_or_better_on_informative_cost_axis;"
            "otherwise_cost_regression_guard_satisfied;full_service_where_applicable"
            if rules["combination"] == COMBINATION_V3
            else "closing_the_do_nothing_to_reference_cost_gap;joint_100_requires_full_service"
        ),
        eligible_cases=len(eligible),
        reference_joint_score_100_cases=sum(abs(s - 100.0) < 1e-9 for s in joint),
        reference_joint_score_below_100_cases=sum(s < 100.0 - 1e-9 for s in joint),
        upper_anchor_policy_counts=dict(sorted(policies.items())),
        privileged_upper_anchor_cases=sum(
            n for p, n in policies.items() if _privileged(p)
        ),
        interpretation="privileged_heuristic_reference_not_an_optimum;scores_are_not_percent_of_optimal_capability",
    )


def sensitivities(cases, nominal, anchors, models, policy) -> dict:
    """Alternative fixed choices; none is selected by model outcomes."""

    def scores(weights, case_map=cases):
        out = aggregate031(
            case_map,
            weights,
            models,
            anchors,
            score_bounds=tuple(policy["rules"]["score_bounds"]),
        )["models"]
        return {r["model"]: r["primary_score"] for r in out}

    weight_rows = {
        (c["scenario_signature"], c["seed"]): c
        for c in policy["weight_manifest"]["cases"]
    }
    result = {}
    for name in (
        "equal_domain_family_physical_source_then_case",
        "equal_domain_physical_source_then_case",
        "uniform_case",
    ):
        alt = hierarchical_case_weights(weight_rows, name)
        result[f"weights:{name}"] = scores(headline_weights(alt, anchors))
    result["exclude_domain:autonomous_driving"] = scores(
        headline_weights(nominal, anchors, exclude_domains=(DRIVING_DOMAIN,))
    )
    for domain in sorted({a["domain"] for a in anchors.values()} - {DRIVING_DOMAIN}):
        result[f"exclude_domain:{domain}"] = scores(
            headline_weights(nominal, anchors, exclude_domains=(domain,))
        )
    bounds = policy["rules"]["score_bounds"]

    def clip_cost(value):
        return min(bounds[1], max(bounds[0], value))

    cost_only = {
        k: dict(
            v,
            S=clip_cost(v["g"]) if v["g"] is not None else v["S"],
            missing_bounds=[clip_cost(v["g"])] * 2
            if v["g"] is not None
            else v["missing_bounds"],
        )
        for k, v in cases.items()
    }
    result["cost_axis_only"] = scores(headline_weights(nominal, anchors), cost_only)
    rules = policy["rules"]
    voltage = {}
    for k, v in cases.items():
        anchor = loss_anchor(anchors[k[1:]], h_min=rules["h_min"])
        if anchor is None or v["hard_failure"] is True:
            voltage[k] = v
            continue
        loss = v["loss030"] if v["determined"] else None
        voltage[k] = score_case031(
            loss,
            v["F"],
            v["hard_failure"] if loss is not None else None,
            anchor,
            h_min=rules["h_min"],
            delta=rules["service_delta"],
            combination=rules["combination"],
            ramp=rules.get("ramp_width", RAMP_WIDTH),
        )
    result["voltage_population_loss030"] = scores(
        headline_weights(nominal, anchors), voltage
    )
    headline = headline_weights(nominal, anchors)
    for low, high in ((-100.0, 200.0), (-50.0, 125.0), (-100.0, 100.0)):
        bounded = {
            k: dict(
                v,
                S=low
                if v["hard_failure"] is True
                else min(high, max(low, v["S_unclipped"])),
            )
            if v["determined"]
            else v
            for k, v in cases.items()
        }
        result[f"clip_bounds:[{low:g},{high:g}]"] = scores(headline, bounded)

    def rescored(**override):
        settings = (
            dict(
                h_min=rules["h_min"],
                delta=rules["service_delta"],
                combination=rules["combination"],
                ramp=rules.get("ramp_width", RAMP_WIDTH),
            )
            | override
        )
        return {
            k: score_case031(
                v["C"], v["F"], v["hard_failure"], anchors[k[1:]], **settings
            )
            if v["determined"]
            else v
            for k, v in cases.items()
        }

    for delta in (0.5, 2.0):
        result[f"service_delta:{delta:g}"] = scores(headline, rescored(delta=delta))
    if rules["combination"] in (COMBINATION_V2, COMBINATION_V3):
        for ramp in (2.0, 10.0):
            result[f"ramp_width:{ramp:g}"] = scores(headline, rescored(ramp=ramp))
        result["combination:v1_threshold"] = scores(
            headline, rescored(combination=COMBINATION_V1)
        )
    if rules["combination"] == COMBINATION_V3:
        result["combination:v2_service_uplift"] = scores(
            headline, rescored(combination=COMBINATION_V2)
        )
    return result
