"""0.28 intensive DSS quality; original native loss and 0.27 remain intact."""

from copy import deepcopy
import math

from evaluation.operational_outcome027 import aggregate_outcome027, native_quality027

VERSION = "0.28.0"
REVISION = "source_population_component_quality.v1"
DSS_BACKENDS = {"opendss_ieee13", "opendss_fresh_feeders"}


def verify_dss_component_records(
    components: dict, records: list, contract: dict
) -> None:
    """Bind component meanings to the original complete native tick records."""
    nodes, horizon = contract["source_node_count"], contract["horizon_ticks"]
    if not isinstance(records, list) or len(records) != horizon:
        raise ValueError("native_dss_component_window_incomplete")
    counts, extrema = [], []
    for tick, record in enumerate(records):
        if (
            not isinstance(record, dict)
            or type(record.get("tick")) is not int
            or record.get("tick") != tick
            or type(record.get("n_voltage_violations")) is not int
            or not 0 <= record["n_voltage_violations"] <= nodes
        ):
            raise ValueError("native_dss_component_record_invalid")
        low, high = record.get("voltage_min_pu"), record.get("voltage_max_pu")
        if (
            any(
                type(v) not in (int, float) or not math.isfinite(v) for v in (low, high)
            )
            or low > high
        ):
            raise ValueError("native_dss_component_extrema_invalid")
        counts.append(100.0 * record["n_voltage_violations"])
        extrema.append(1000.0 * (max(0.0, 0.95 - low) + max(0.0, high - 1.05)))
    # Match the producer's ordered cumulative addition and terminal rounding.
    expected = {
        "production_cost": 0.0,
        "voltage_violation_cost": round(sum(counts), 2),
        "voltage_band_deviation_cost": round(sum(extrema), 2),
    }
    if components != expected:
        raise ValueError("native_dss_components_record_mismatch")


def normalize_dss_components(components: dict, cost: float, contract: dict) -> dict:
    """Normalize recorded point components, retaining their producer rounding."""
    nodes, horizon = contract.get("source_node_count"), contract.get("horizon_ticks")
    if any(type(v) is not int or v <= 0 for v in (nodes, horizon)):
        raise ValueError("invalid_source_node_time_exposure")
    names = {"production_cost", "voltage_violation_cost", "voltage_band_deviation_cost"}
    if not isinstance(components, dict) or set(components) != names:
        raise ValueError("native_dss_cost_components_missing_or_unsupported")
    if any(
        type(v) not in (int, float) or not math.isfinite(v) or v < 0
        for v in [cost, *components.values()]
    ):
        raise ValueError("invalid_native_dss_cost_component")
    count, extrema = (
        components["voltage_violation_cost"],
        components["voltage_band_deviation_cost"],
    )
    if (
        components["production_cost"] != 0
        or not (count / 100).is_integer()
        or count > 100 * nodes * horizon
    ):
        raise ValueError("native_dss_component_not_source_authorized")
    if not math.isclose(math.fsum(components.values()), cost, rel_tol=0, abs_tol=1e-8):
        raise ValueError("native_dss_component_total_mismatch")
    count_scale, extrema_scale = 100 * horizon * nodes, 100 * horizon
    count_loss = count / count_scale
    loss = math.fsum([count_loss, extrema / extrema_scale])
    quality = native_quality027(loss, 1.0, signed=False)
    # Count costs are exact integer multiples of100. The producer rounds the
    # cumulative global-extrema component to two decimals, not each tick.
    interval = [
        native_quality027(count_loss + v / extrema_scale, 1.0, signed=False)
        for v in (extrema + 0.005, max(0.0, extrema - 0.005))
    ]
    return {
        "verified": True,
        "native_cost": cost,
        "native_cost_components": deepcopy(components),
        "source_node_count": nodes,
        "horizon_ticks": horizon,
        "count_scale": count_scale,
        "extrema_scale": extrema_scale,
        "dimensionless_loss": loss,
        "native_quality": quality,
        "quality_interval": interval,
        "interpretation": "population_normalized_voltage_exposure_plus_global_extrema_severity",
    }


def outcome028(previous: dict, normalization: dict | None) -> dict:
    """Use authenticated 0.27 qualification/F/gates with the new DSS quality."""
    result = deepcopy(previous)
    result.update(evaluation_version=VERSION, protocol_revision=REVISION)
    if not result["evidence_qualified"] or result["hard_failure"] is True:
        return result
    quality = normalization["native_quality"] if normalization is not None else None
    if normalization is not None:
        interval = normalization.get("quality_interval")
        if (
            type(quality) not in (int, float)
            or not math.isfinite(quality)
            or not 0 <= quality <= 100
            or not isinstance(interval, list)
            or len(interval) != 2
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 100
                for v in interval
            )
            or not interval[0] <= quality <= interval[1]
        ):
            raise ValueError("invalid_normalized_dss_quality_interval")
    result.update(
        native_quality_if_no_hard_failure=quality,
        native_quality=None,
        score=None,
        score_interval=None,
        numeric_determined=False,
        publish_eligible=False,
    )
    fraction = result["completion_score"] if result["completion_measured"] else None
    result["missing_measurement_bounds"] = [
        0.0,
        min(
            100.0,
            quality if quality is not None else 100.0,
            fraction if fraction is not None else 100.0,
        ),
    ]
    if result["hard_failure"] is None:
        return result
    result["native_quality"] = quality
    if quality is None:
        result["reason"] = "native_dss_component_measurement_missing"
        return result
    if result["completion_applicable"] and fraction is None:
        return result
    point = min(quality, fraction) if fraction is not None else quality
    interval = normalization["quality_interval"]
    result.update(
        score=point,
        score_interval=[
            min(v, fraction) if fraction is not None else v for v in interval
        ],
        numeric_determined=True,
        publish_eligible=True,
        missing_measurement_bounds=[point, point],
    )
    return result


def aggregate_outcome028(rows, contracts, weights, *, models):
    # Reuse the unchanged denominator, missingness, F, safety and rank rules.
    adapted = [{**r, "outcome027": r["outcome028"]} for r in rows]
    result = aggregate_outcome027(adapted, contracts, weights, models=models)
    result.update(evaluation_version=VERSION, protocol_revision=REVISION)
    return result
