"""Versioned source-only DSS normalization after original evidence validation."""

from copy import deepcopy
import json
from pathlib import Path

from evaluation.mission_contracts import _digest
from evaluation.native_scorecard import _read
from evaluation.operational_outcome028 import (
    DSS_BACKENDS,
    REVISION,
    VERSION,
    normalize_dss_components,
    outcome028,
    verify_dss_component_records,
)
from evaluation.trajectory_outcome027 import (
    ROOT,
    canonical_digest,
    grade_episode027,
    load_policy as load_policy027,
    read_bound,
)

DEFAULT_POLICY = ROOT / "evaluation/policies/lite141_028.json"


def compile_normalizations(base: dict) -> list[dict]:
    populations = {
        (p["scenario_signature"], p["seed"]): p
        for p in base["voltage_population_contracts"]
    }
    result = []
    for c in base["source_contracts"]["contracts"]:
        if c["backend_kind"] not in DSS_BACKENDS:
            continue
        p = populations[c["scenario_signature"], c["seed"]]
        nodes = p["required_node_ids"]
        row = dict(
            scenario_signature=c["scenario_signature"],
            seed=c["seed"],
            source_contract_sha256=c["contract_sha256"],
            population_contract_sha256=p["contract_sha256"],
            source_node_count=len(nodes),
            horizon_ticks=len(p["required_ticks"]),
            count_formula="native_voltage_violation_cost/(100*H*M)",
            extrema_formula="native_global_voltage_band_deviation_cost/(100*H)",
            component_rounding_decimals=2,
        )
        row["contract_sha256"] = canonical_digest(row)
        result.append(row)
    return result


def load_policy(path: Path = DEFAULT_POLICY, *, root: Path = ROOT) -> dict:
    policy = json.loads(Path(path).read_bytes())
    if policy.get("schema_version") != "operate_trajectory_scoring_policy028.v1":
        if policy.get("evaluation_version") != "0.27.0":
            raise ValueError("trajectory_policy_schema_version_mismatch")
        return load_policy027(path, root=root)
    # The extension may never override independently authenticated base facts.
    # A valid self-hash authenticates bytes, not authority to change obligations.
    allowed = {
        "schema_version",
        "evaluation_version",
        "protocol_revision",
        "base_policy",
        "normalization_contracts",
        "scoring_entry",
        "primary_interpretation",
        "policy_sha256",
    }
    if set(policy) != allowed:
        raise ValueError("trajectory028_policy_fields_not_authorized")
    if (
        policy.get("policy_sha256")
        != _digest({k: v for k, v in policy.items() if k != "policy_sha256"})
        or policy.get("evaluation_version") != VERSION
        or policy.get("protocol_revision") != REVISION
    ):
        raise ValueError("trajectory028_policy_hash_or_version_mismatch")
    ref = policy["base_policy"]
    base_raw, bound = read_bound({**ref, "path": str(Path(root) / ref["path"])})
    base = load_policy027(Path(bound["path"]), root=root)
    if policy.get("normalization_contracts") != compile_normalizations(base):
        raise ValueError("trajectory028_source_normalization_mismatch")
    if Path(bound["path"]).read_bytes() != base_raw:
        raise ValueError("trajectory028_base_policy_changed")
    return {**base, **{k: policy[k] for k in allowed}, "base_policy_artifact": bound}


def grade_episode028(
    episode,
    config,
    journal,
    contract,
    scenario,
    population,
    *,
    normalization_contract=None,
    **kwargs,
):
    row = grade_episode027(
        episode, config, journal, contract, scenario, population, **kwargs
    )
    legacy = row["outcome027"]
    normalization = None
    if contract["backend_kind"] in DSS_BACKENDS and legacy["hard_failure"] is not True:
        if normalization_contract is None:
            raise ValueError("source_dss_normalization_contract_missing")
        expected = compile_normalizations(
            {
                "source_contracts": {"contracts": [contract]},
                "voltage_population_contracts": [population],
            }
        )[0]
        if normalization_contract != expected:
            raise ValueError("source_dss_normalization_contract_mismatch")
        inputs = _read(row["artifact_binding"], "scoring_inputs_artifact")["payload"][
            "inputs"
        ]
        components = inputs.get("cost_components")
        if isinstance(components, dict) and row["C"] is not None:
            verify_dss_component_records(
                components, inputs.get("backend_tick_records"), normalization_contract
            )
            normalization = normalize_dss_components(
                components, row["C"], normalization_contract
            )
        result = outcome028(legacy, normalization)
    else:
        result = deepcopy(legacy)
        result.update(evaluation_version=VERSION, protocol_revision=REVISION)
    row.update(
        outcome028=result,
        normalization028=normalization,
        legacy_N027=row["N"],
        legacy_Q027=row["Q"],
        N=result["native_quality"],
        Q=result["score"],
    )
    return row
