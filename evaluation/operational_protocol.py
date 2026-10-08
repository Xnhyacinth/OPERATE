"""Load frozen 0.25 task weights and acceptance definitions, never fit targets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evaluation.native_objectives import NATIVE_OBJECTIVES
from evaluation.operational_acceptance import validate_acceptance_contract
from evaluation.operational_weights import validate_task_weights


def objective_identity(contract):
    backend = contract["backend_kind"]
    return dict(
        objective_id="fjsp.source_obligation_settled_loss.v1"
        if backend == "dynasched_flexible_job_shop"
        else f"{backend}.{NATIVE_OBJECTIVES[backend][1]}.v1",
        unit="native_time_plus_unfinished_penalty"
        if backend == "dynasched_flexible_job_shop"
        else "native_cost_units",
    )


def read_frozen(descriptor, *, root):
    if (
        not isinstance(descriptor, dict)
        or not descriptor.get("path")
        or not descriptor.get("sha256")
    ):
        raise ValueError("frozen_descriptor_requires_path_and_sha256")
    path = Path(descriptor["path"])
    if not path.is_absolute():
        path = Path(root) / path
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != descriptor["sha256"]:
        raise ValueError("frozen_scoring_artifact_hash_mismatch")
    return raw


def load_operational_protocol(config, compiled, *, root):
    if not config.get("task_weights"):
        raise ValueError(
            "acceptance mode requires a frozen task_weights manifest; prepare it before scoring"
        )
    manifest = json.loads(read_frozen(config["task_weights"], root=root))
    weights = validate_task_weights(manifest, compiled)
    descriptor = config.get("acceptance_contracts")
    targets = {}
    mode = None
    if descriptor:
        suite = json.loads(read_frozen(descriptor, root=root))
        if (
            suite.get("schema_version") != "operational_acceptance_suite.v1"
            or suite.get("suite_sha256") != compiled["suite_sha256"]
        ):
            raise ValueError("acceptance_suite_identity_mismatch")
        mode = suite.get("quality_mode")
        if mode not in (
            "fixed_threshold",
            "uniform_threshold_interval",
            "mixed_declared",
        ):
            raise ValueError("acceptance_suite_quality_mode_missing")
        cases = {(c["scenario_signature"], c["seed"]): c for c in compiled["contracts"]}
        if not isinstance(suite.get("contracts"), list):
            raise ValueError("acceptance_contract_inventory_missing")
        for contract in suite["contracts"]:
            identity = contract.get("identity") or {}
            key = (identity.get("scenario_signature"), identity.get("seed"))
            if key not in cases or key in targets:
                raise ValueError("foreign_or_duplicate_acceptance_case")
            source = {**cases[key], **objective_identity(cases[key])}
            checked = validate_acceptance_contract(contract, source)
            if checked["status"] != "ready":
                raise ValueError("invalid_acceptance_contract:" + checked["reason"])
            if mode != "mixed_declared" and contract["quality"]["mode"] != mode:
                raise ValueError("mixed_acceptance_modes_not_declared")
            provenance = contract["provenance"]
            hashes = set()
            for artifact in provenance["artifacts"]:
                read_frozen(artifact, root=root)
                hashes.add(artifact["sha256"])
            if provenance["kind"] == "same_information_reference":
                for pair in provenance["information_parity_evidence"].values():
                    if (
                        pair["reference_sha256"] not in hashes
                        or pair["task_sha256"] not in hashes
                    ):
                        raise ValueError("information_parity_artifact_not_bound")
            targets[key] = contract
    return dict(
        task_weights=manifest,
        weights=weights,
        acceptance_contracts=targets,
        quality_mode=mode,
        calibration_coverage=dict(
            expected_cases=len(compiled["contracts"]),
            declared_contract_cases=len(targets),
            missing_contract_cases=len(compiled["contracts"]) - len(targets),
            complete=len(targets) == len(compiled["contracts"]),
            interpretation="frozen_declarations_and_artifact_integrity_not_scientific_certification",
        ),
    )
