"""Fixed source opportunities for six operational capability diagnostics.

Hashes establish input identity, not construct validity. Native predicate
fulfillment plus an evidence-linked action is association, not a verified causal
improvement. Independent calibration is required before capability ranking.

Outcome predicates cover start_tick..end_tick. The optional effect_start_tick
(default observation_start_tick) independently authorizes preventative effects
before that outcome window; source recovery contracts can require later effects.
"""

from __future__ import annotations

from copy import deepcopy
import math

from evaluation.operational_opportunities import _effect_links, _satisfied
from evaluation.trajectory_outcome027 import canonical_digest
from evaluation.scorer import _completed_successful_tool_payload

AXES = (
    "scheduling",
    "global_coordination",
    "resource_preservation",
    "adaptation",
    "proactivity",
    "timeliness",
)
SCHEMA = "operational_capability_contract029.v1"
ORIGINS = {
    "autonomous_query",
    "agent_scheduled_review",
    "initial_mission",
    "mandatory_delivery",
    "not_observed",
}
AUTONOMOUS = {"autonomous_query", "agent_scheduled_review", "initial_mission"}


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _ids(value):
    return (
        isinstance(value, list)
        and bool(value)
        and all(_text(v) for v in value)
        and len(set(value)) == len(value)
    )


def _tick(value):
    return type(value) is int and value >= 0


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _unknown(reason):
    return {
        "applicable": None,
        "score": None,
        "bounds": None,
        "expected_opportunities": None,
        "measured_opportunities": 0,
        "reason": reason,
        "opportunities": [],
        "evidence_ids": [],
    }


def _contracts(contract, identity, expected):
    if not isinstance(contract, dict) or contract.get("schema_version") != SCHEMA:
        raise ValueError("capability_contract_schema_invalid")
    if (
        any(contract.get(k) != identity.get(k) for k in ("scenario_signature", "seed"))
        or type(contract.get("seed")) is not int
    ):
        raise ValueError("capability_contract_identity_mismatch")
    if not expected or canonical_digest(contract) != expected:
        raise ValueError("capability_contract_hash_mismatch")
    horizon, axes = contract.get("horizon_ticks"), contract.get("axes")
    if (
        not _tick(horizon)
        or horizon == 0
        or not isinstance(axes, dict)
        or not set(axes) <= set(AXES)
    ):
        raise ValueError("capability_contract_axes_or_horizon_invalid")
    specs = {}
    for axis, definition in axes.items():
        if (
            not isinstance(definition, dict)
            or type(definition.get("applicable")) is not bool
        ):
            raise ValueError("capability_axis_applicability_invalid")
        if definition["applicable"] is False:
            if not _text(definition.get("reason")) or definition.get("opportunities"):
                raise ValueError("capability_inapplicable_axis_invalid")
            continue
        opportunities = definition.get("opportunities")
        if not isinstance(opportunities, list) or not opportunities:
            raise ValueError("capability_opportunities_missing")
        for original in opportunities:
            spec = deepcopy(original)
            key = spec.get("opportunity_id")
            start, end = spec.get("start_tick"), spec.get("end_tick")
            obs_start = spec.setdefault("observation_start_tick", start)
            effect_start = spec.setdefault("effect_start_tick", obs_start)
            if (
                not _text(key)
                or key in specs
                or not all(_tick(t) for t in (obs_start, start, end, effect_start))
                or not obs_start <= effect_start <= end
                or not obs_start <= start <= end < horizon
                or spec.get("mode") not in {"action_required", "quiet"}
                or spec.get("scope") not in {"terminal", "throughout"}
            ):
                raise ValueError("capability_opportunity_invalid")
            discovery, feasible = spec.get("discoverability"), spec.get("feasibility")
            if (
                not isinstance(discovery, dict)
                or discovery.get("kind")
                not in {"query_reachable", "initial_mission", "mandatory_delivery"}
                or not _ids(discovery.get("source_evidence_ids"))
                or not isinstance(feasible, dict)
                or not _ids(feasible.get("source_evidence_ids"))
            ):
                raise ValueError("capability_fixed_source_proof_missing")
            predicates = spec.get("predicates")
            if (
                not isinstance(predicates, list)
                or not predicates
                or any(
                    not isinstance(p, dict)
                    or not _text(p.get("metric"))
                    or p.get("operator") not in {"eq", "ge", "le"}
                    or not _number(p.get("value"))
                    for p in predicates
                )
            ):
                raise ValueError("capability_predicate_invalid")
            if axis == "resource_preservation" and not {"resource", "service"} <= {
                p.get("role") for p in predicates
            }:
                raise ValueError("capability_resource_requires_service_and_resource")
            if (
                axis == "global_coordination"
                and len({p["entity"] for p in predicates if _text(p.get("entity"))}) < 2
            ):
                raise ValueError("capability_global_requires_multiple_entities")
            if axis == "proactivity" and spec["mode"] == "action_required":
                alert = spec.get("mandatory_alert_tick")
                if (
                    not _tick(alert)
                    or not start < alert <= horizon
                    or discovery["kind"] == "mandatory_delivery"
                ):
                    raise ValueError("capability_proactivity_prealert_window_invalid")
            specs[key] = {**spec, "axis": axis}
    for quiet in (s for s in specs.values() if s["mode"] == "quiet"):
        for positive in (s for s in specs.values() if s["mode"] == "action_required"):
            if max(quiet["start_tick"], positive["observation_start_tick"]) <= min(
                quiet["end_tick"], positive["end_tick"]
            ):
                raise ValueError("capability_quiet_overlaps_required_action_window")
    return specs


def _ledger(items, specs, horizon):
    if not isinstance(items, list):
        raise ValueError("capability_ledger_invalid")
    seen, records = set(), {k: [] for k in specs}
    for item in items:
        if (
            not isinstance(item, dict)
            or not _text(item.get("evidence_id"))
            or item["evidence_id"] in seen
            or not _tick(item.get("tick"))
            or item["tick"] > horizon
            or not isinstance(item.get("payload"), dict)
        ):
            raise ValueError("capability_evidence_invalid_or_duplicate")
        seen.add(item["evidence_id"])
        if item.get("kind") != "capability_opportunity":
            continue
        p = item["payload"]
        if (
            item.get("source") != "engine"
            or p.get("opportunity_id") not in specs
            or p.get("phase") not in {"sample", "close"}
        ):
            raise ValueError("capability_native_record_invalid")
        if p["phase"] == "sample" and (
            not isinstance(p.get("metrics"), dict)
            or any(not _number(v) for v in p["metrics"].values())
            or (p.get("visible") is not None and type(p["visible"]) is not bool)
            or p.get("observation_origin") not in ORIGINS
            or (p.get("visible") is True and p["observation_origin"] == "not_observed")
        ):
            raise ValueError("capability_sample_invalid")
        records[p["opportunity_id"]].append(item)
    return records


def _window(spec, records):
    samples, close = {}, None
    for item in records:
        tick, p = item["tick"], item["payload"]
        if p["phase"] == "sample":
            if (
                close is not None
                or tick in samples
                or not spec["observation_start_tick"] <= tick <= spec["end_tick"]
            ):
                raise ValueError("capability_sample_duplicate_or_outside_window")
            samples[tick] = item
        else:
            if close is not None or tick < spec["end_tick"]:
                raise ValueError("capability_close_duplicate_or_early")
            for field in ("control_census_complete", "action_lifecycle_complete"):
                if p.get(field) is not None and type(p[field]) is not bool:
                    raise ValueError("capability_closure_flag_invalid")
            attempts = p.get("attempted_control_call_ids")
            if attempts is not None and (
                not isinstance(attempts, list)
                or any(not _text(v) for v in attempts)
                or len(set(attempts)) != len(attempts)
            ):
                raise ValueError("capability_control_census_invalid")
            close = item
    complete = (
        len(samples) == spec["end_tick"] - spec["observation_start_tick"] + 1
        and close is not None
    )
    return samples, close, complete


def _score_opportunity(spec, records, items):
    samples, close, complete = _window(spec, records)
    row = {
        "opportunity_id": spec["opportunity_id"],
        "mode": spec["mode"],
        "start_tick": spec["start_tick"],
        "end_tick": spec["end_tick"],
        "score": None,
        "status": "unknown",
        "reason": "window_evidence_incomplete",
        "evidence_ids": [i["evidence_id"] for i in records],
        "causal_improvement_verified": False,
        "evidence_level": "native_predicate_outcome",
        "effect_links": [],
    }
    if not complete:
        return row
    by_id = {item["evidence_id"]: item for item in items}
    proof_ids = []
    for kind in ("discoverability", "feasibility"):
        deadline = (
            spec["observation_start_tick"]
            if kind == "discoverability"
            else spec["start_tick"]
        )
        for evidence_id in spec[kind]["source_evidence_ids"]:
            proof = by_id.get(evidence_id)
            payload = proof["payload"] if proof else {}
            if (
                proof is None
                or proof.get("source") != "engine"
                or proof.get("kind") != "capability_source_fact"
                or proof["tick"] > deadline
                or payload.get("fact_type") != kind
                or not _ids(payload.get("opportunity_ids"))
                or spec["opportunity_id"] not in payload["opportunity_ids"]
                or not _text(payload.get("source_fact"))
            ):
                row["reason"] = "source_proof_evidence_missing"
                return row
            proof_ids.append(evidence_id)
    row["evidence_ids"] += proof_ids
    if close["payload"].get("action_lifecycle_complete") is not True:
        row["reason"] = "action_lifecycle_census_incomplete"
        return row
    tested = (
        [samples[t] for t in range(spec["start_tick"], spec["end_tick"] + 1)]
        if spec["scope"] == "throughout"
        else [samples[spec["end_tick"]]]
    )
    states = [
        _satisfied(p, i["payload"]["metrics"])
        for i in tested
        for p in spec["predicates"]
    ]
    if None in states:
        row["reason"] = "native_metric_missing"
        return row
    success = all(states)
    if spec["mode"] == "quiet":
        census = close["payload"]
        if (
            census.get("control_census_complete") is not True
            or "attempted_control_call_ids" not in census
        ):
            row["reason"] = "control_census_incomplete"
            return row
        # A completion receipt/effect can settle a control submitted before the
        # quiet window. Authenticate that lifecycle before treating it as standing.
        link_spec = {
            "opportunity_id": spec["opportunity_id"],
            "start_tick": spec["observation_start_tick"],
            "end_tick": spec["end_tick"],
            "sample_evidence_ids": [i["evidence_id"] for i in samples.values()],
        }
        links = _effect_links(items, [link_spec])
        calls = {}
        for item in items:
            if item.get("kind") == "tool_call" and item.get("source") in {
                "tool",
                "engine",
            }:
                calls.setdefault(item["payload"].get("call_id"), []).append(item)
        standing = {
            link["call_id"]
            for link in links
            if min(i["tick"] for i in calls[link["call_id"]]) < spec["start_tick"]
        }
        linked_effects = {link["effect_evidence_id"] for link in links}
        attempts = bool(census["attempted_control_call_ids"])
        for item in items:
            if not spec["start_tick"] <= item["tick"] <= spec["end_tick"]:
                continue
            p = item["payload"]
            if item.get("kind") == "tool_call" and (
                p.get("state_changing") is True or p.get("control_attempt") is True
            ):
                if p.get("call_id") not in standing:
                    attempts = True
            if (
                item.get("source") == "engine"
                and item.get("kind") == "realized_event"
                and p.get("origin") == "agent_caused"
            ):
                if item["evidence_id"] not in linked_effects or items.index(
                    item
                ) > items.index(close):
                    row["reason"] = "quiet_effect_attribution_unknown"
                    return row
                if p.get("call_id") not in standing:
                    attempts = True
        row["effect_links"] = links
        row["evidence_ids"] += [link["effect_evidence_id"] for link in links]
        success = success and not attempts
    elif spec["axis"] in {"adaptation", "timeliness", "proactivity"}:
        # Native effects reference action records: a missing referenced receipt
        # or observation is evidence loss, not a confirmed agent omission.
        by_id = {i["evidence_id"]: i for i in items}
        for effect in items:
            payload = effect["payload"]
            if not (
                effect.get("source") == "engine"
                and effect.get("kind") == "realized_event"
                and payload.get("origin") == "agent_caused"
                and payload.get("opportunity_id") == spec["opportunity_id"]
                and spec["observation_start_tick"] <= effect["tick"] <= spec["end_tick"]
            ):
                continue
            calls = [
                i
                for i in items
                if i.get("kind") == "tool_call"
                and i.get("source") in {"tool", "engine"}
                and i["payload"].get("call_id") == payload.get("call_id")
            ]
            successful = [
                i for i in calls if _completed_successful_tool_payload(i["payload"])
            ]
            receipt = successful[0]["payload"] if len(successful) == 1 else {}
            consumed = receipt.get("consumes_evidence_ids")
            requested = payload.get("requested_action")
            if (
                len(successful) != 1
                or not isinstance(consumed, list)
                or any(not isinstance(e, str) or e not in by_id for e in consumed)
                or not isinstance(requested, dict)
                or any(requested.get(k) != receipt.get(k) for k in ("name", "args"))
            ):
                row["reason"] = "action_effect_evidence_unresolved"
                return row
        eligible_samples = list(samples.values())
        if any(i["payload"].get("visible") is None for i in eligible_samples):
            row["reason"] = "observation_delivery_evidence_missing"
            return row
        if spec["axis"] == "proactivity":
            eligible_samples = [
                i
                for i in eligible_samples
                if i["payload"]["observation_origin"] in AUTONOMOUS
            ]
        link_spec = {
            "opportunity_id": spec["opportunity_id"],
            "start_tick": spec["observation_start_tick"],
            "end_tick": spec["end_tick"],
            "sample_evidence_ids": [i["evidence_id"] for i in eligible_samples],
        }
        links = [
            link
            for link in _effect_links(items, [link_spec])
            if link["effect_tick"] >= spec["effect_start_tick"]
            and next(
                n
                for n, i in enumerate(items)
                if i["evidence_id"] == link["effect_evidence_id"]
            )
            < items.index(close)
        ]
        if spec["axis"] == "proactivity":
            links = [
                link
                for link in links
                if link["effect_tick"] < spec["mandatory_alert_tick"]
            ]
        row.update(
            effect_links=links,
            evidence_level="native_outcome_with_action_effect_association",
        )
        row["evidence_ids"] += [link["effect_evidence_id"] for link in links]
        success = success and bool(links)
    row.update(
        score=100.0 if success else 0.0,
        status="fulfilled" if success else "unfulfilled",
        reason=None,
    )
    return row


def _rate(rows):
    known = [r for r in rows if r["score"] is not None]
    return {
        "score": sum(r["score"] for r in known) / len(rows)
        if rows and len(known) == len(rows)
        else None,
        "bounds": [
            sum(r["score"] for r in known) / len(rows),
            (sum(r["score"] for r in known) + 100 * (len(rows) - len(known)))
            / len(rows),
        ]
        if rows
        else None,
        "expected_opportunities": len(rows),
        "measured_opportunities": len(known),
    }


def score_capabilities(
    contract: dict | None,
    *,
    evidence_ledger: list | None,
    identity: dict,
    expected_contract_sha256: str | None,
    evidence_verified: bool = False,
) -> dict:
    """Score a source-anchored fixed census; missing obligations never disappear."""
    if (
        not isinstance(identity, dict)
        or not _text(identity.get("scenario_signature"))
        or type(identity.get("seed")) is not int
        or type(evidence_verified) is not bool
    ):
        raise ValueError("capability_identity_or_qualification_invalid")
    result = {
        "schema_version": "operational_capability_report029.v1",
        "evaluation_version": "0.29.0",
        "identity": deepcopy(identity),
        "contract_sha256": expected_contract_sha256 if contract else None,
        "ranking_ready": False,
        "causal_improvement_verified": False,
        "interpretation": "source_opportunity_diagnostics_require_independent_construct_calibration",
        "axes": {a: _unknown("missing_source_contract") for a in AXES},
    }
    if contract is None:
        return result
    specs = _contracts(contract, identity, expected_contract_sha256)
    records = (
        _ledger(evidence_ledger, specs, contract["horizon_ticks"])
        if evidence_ledger is not None
        else None
    )
    for axis in AXES:
        definition = contract["axes"].get(axis)
        if definition is None:
            result["axes"][axis] = _unknown("missing_axis_contract")
            continue
        if not definition["applicable"]:
            result["axes"][axis] = {
                **_unknown(definition["reason"]),
                "applicable": False,
                "expected_opportunities": 0,
            }
            continue
        rows = []
        for key, spec in specs.items():
            if spec["axis"] != axis:
                continue
            if records is None or not evidence_verified:
                rows.append(
                    {
                        "opportunity_id": key,
                        "mode": spec["mode"],
                        "score": None,
                        "status": "unknown",
                        "reason": "native_evidence_unverified_or_missing",
                        "evidence_ids": [],
                    }
                )
            else:
                rows.append(_score_opportunity(spec, records[key], evidence_ledger))
        summary = {
            "applicable": True,
            **_rate(rows),
            "reason": None,
            "opportunities": rows,
            "evidence_ids": sorted({e for r in rows for e in r["evidence_ids"]}),
        }
        if axis == "proactivity":
            positive, quiet = (
                _rate([r for r in rows if r["mode"] == mode])
                for mode in ("action_required", "quiet")
            )
            summary.update(positive=positive, quiet=quiet, score=None, bounds=None)
            if positive["expected_opportunities"] and quiet["expected_opportunities"]:
                summary["bounds"] = [
                    (positive["bounds"][i] + quiet["bounds"][i]) / 2 for i in (0, 1)
                ]
                if positive["score"] is not None and quiet["score"] is not None:
                    summary["score"] = (positive["score"] + quiet["score"]) / 2
            else:
                summary["reason"] = "positive_and_quiet_contrast_required"
        if summary["score"] is None and summary["reason"] is None:
            summary["reason"] = "incomplete_fixed_opportunity_measurement"
        result["axes"][axis] = summary
    return result
