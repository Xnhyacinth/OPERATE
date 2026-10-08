"""Hash-bound recovery of missing private voltage measurements.

Recovery is an explicit, provider-free operation. Ordinary scoring verifies a
previous sidecar without importing a native engine or changing original bytes.
Derived evidence has its own IDs and retains the original execution identity.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from evaluation.source_voltage_population import measure_voltage_completion027

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "native_measurement_recovery.v1"
PP_QUANTIZED_EQUIVALENCE = {
    "schema_version": "pandapower_lv_consumer_quantization_equivalence.v1",
    "backend_kind": "pandapower_lv",
    "source_module": "domains/microgrid/backends/pandapower_lv.py",
    "source_module_sha256": "ca6dc9accef75cefffd440dacb06d2fe9c796bfd992874a736070209a10d1b72",
    "allowed_raw_fields": ["grid_exchange_mw", "network_loss_mw"],
    "production_cost_per_mwh": 50.0,
    "voltage_violation_cost_per_tick": 1200.0,
    "overload_cost_per_tick": 200.0,
    "disconnection_cost_per_line_tick": 500.0,
    "production_round_decimal_places": 3,
    "production_quantum": 0.001,
    "balance_round_decimal_places": 4,
    "balance_quantum_mw": 0.0001,
    "required_other_consumers": "exact_records_observations_effects_tool_receipts_and_cost_components",
    "interpretation": "score_preserving_consumer_quantization_not_raw_state_bit_equality",
}


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _read(ref: dict, *, lines: bool = False, artifact_relocations: dict | None = None):
    path = Path((artifact_relocations or {}).get(ref["sha256"], ref["path"]))
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise ValueError("native_recovery_artifact_hash_mismatch")
    if ref.get("byte_count", len(raw)) != len(raw):
        raise ValueError("native_recovery_artifact_size_mismatch")
    return (
        [json.loads(line) for line in raw.splitlines() if line.strip()]
        if lines
        else json.loads(raw)
    )


def _inputs(
    scenario,
    source_contract,
    population,
    snapshot_ref,
    trace_ref,
    ledger_ref,
    *,
    artifact_relocations=None,
):
    snapshot = _read(snapshot_ref, artifact_relocations=artifact_relocations)
    trace, ledger = (
        _read(trace_ref, lines=True, artifact_relocations=artifact_relocations),
        _read(ledger_ref, lines=True, artifact_relocations=artifact_relocations),
    )
    if (
        snapshot.get("schema_version") != "episode_scoring_snapshot_v1"
        or snapshot.get("kind") != "scoring_inputs"
    ):
        raise ValueError("native_recovery_snapshot_schema_invalid")
    identity = snapshot["payload"]["identity"]
    inputs = snapshot["payload"]["inputs"]
    if any(
        identity.get(key) != scenario.get(key) for key in ("scenario_signature", "seed")
    ):
        raise ValueError("native_recovery_original_identity_mismatch")
    records, horizon = inputs.get("backend_tick_records"), scenario["horizon_ticks"]
    if (
        not isinstance(records, list)
        or len(records) != horizon
        or len(trace) != horizon
        or any(type(row.get("tick")) is not int for row in records + trace)
        or [row["tick"] for row in records] != list(range(horizon))
        or [row["tick"] for row in trace] != list(range(1, horizon + 1))
    ):
        raise ValueError("native_recovery_incomplete_source_horizon")
    if any(row.get("observation", {}).get("tick") != row["tick"] for row in trace):
        raise ValueError("native_recovery_observation_clock_mismatch")
    if len({row["evidence_id"] for row in ledger}) != len(ledger):
        raise ValueError("native_recovery_duplicate_original_evidence")
    states = [
        row
        for row in ledger
        if row.get("source") == "engine" and row.get("kind") == "backend_tick"
    ]
    if (
        len(states) != horizon
        or any(type(row.get("tick")) is not int for row in states)
        or [row["tick"] for row in states] != list(range(horizon))
    ):
        raise ValueError("native_recovery_original_engine_window_invalid")
    for old, state in zip(records, states, strict=True):
        payload = deepcopy(state["payload"])
        if (
            old.get("done") is False
            and payload.get("done") is True
            and old["tick"] == horizon - 1
        ):
            payload["done"] = (
                False  # Established scoring-record terminal representation.
            )
        if (
            "catastrophic_failure" in old
            and "catastrophic_failure" not in payload
            and scenario["backend_kind"] == "pandapower_lv"
        ):
            payload["catastrophic_failure"] = bool(
                not payload.get("converged")
                or (
                    payload.get("aggregate_demand_mw", float("inf")) <= 1e-9
                    and payload.get("unserved_energy_mwh", 0) > 1e-9
                )
                or old.get("done") is True
            )
        if _differences(old, payload):
            raise ValueError("native_recovery_original_engine_record_mismatch")
    # This pure check also validates the population hash and source identity.
    completion = measure_voltage_completion027(
        scenario,
        source_contract=source_contract,
        population=population,
        snapshot_inputs=inputs,
        trace=trace,
    )
    return snapshot, inputs, trace, ledger, completion


def _differences(
    old, new, path="", *, locked_paths=(), symmetric=False, allowed_new_keys=()
):
    """Compare recorded fields exactly; ignore evidence annotations only."""
    diffs = []
    if isinstance(old, dict) and isinstance(new, dict):
        for key, value in old.items():
            if key in {"evidence_ids", "evidence_id"}:
                continue
            if key not in new:
                diffs.append(
                    {"path": path + "/" + key, "old": value, "new_missing": True}
                )
            else:
                diffs.extend(
                    _differences(
                        value,
                        new[key],
                        path + "/" + key,
                        locked_paths=locked_paths,
                        symmetric=symmetric,
                    )
                )
        if symmetric:
            for key in sorted(
                set(new)
                - set(old)
                - {"evidence_ids", "evidence_id"}
                - set(allowed_new_keys)
            ):
                diffs.append(
                    {"path": path + "/" + key, "old_missing": True, "new": new[key]}
                )
        return diffs
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return [{"path": path, "old_length": len(old), "new_length": len(new)}]
        for index, (left, right) in enumerate(zip(old, new, strict=True)):
            diffs.extend(
                _differences(
                    left,
                    right,
                    path + "/" + str(index),
                    locked_paths=locked_paths,
                    symmetric=symmetric,
                )
            )
        return diffs
    if old == new and (type(old) is bool) == (type(new) is bool):
        return []
    if isinstance(old, str) and isinstance(new, str):
        # A path relocation is accepted only for an independently locked input.
        for relative in locked_paths:
            suffix = "/" + relative.lstrip("/")
            if old.endswith(suffix) and new.endswith(suffix):
                return []
    diff = {"path": path, "old": old, "new": new}
    if (
        type(old) in (int, float)
        and type(new) in (int, float)
        and math.isfinite(old)
        and math.isfinite(new)
    ):
        diff["absolute_difference"] = abs(new - old)
        diff["ulps"] = abs(new - old) / max(math.ulp(old), math.ulp(new))
    return [diff]


def _runtime_fence(scenario, population):
    """Verify source bytes and installed constructor; construct no circuit here."""
    from core.source_asset_contract import resolve_source_asset_contract

    lock = resolve_source_asset_contract(scenario, repo_root=ROOT)
    if lock.contract_errors or lock.missing_required_files:
        raise ValueError("native_recovery_source_asset_lock_failed")
    if scenario["backend_kind"].startswith("opendss"):
        from evaluation.source_voltage_population import opendss_runtime_asset_lock

        actual = opendss_runtime_asset_lock()
    else:
        from domains.power_grid.backends.cigre_distribution import (
            _constructor_runtime_asset,
        )

        actual = _constructor_runtime_asset(population["network"])
    if actual != population["constructor_runtime_asset"]:
        raise ValueError("native_recovery_constructor_bytes_mismatch")
    modules = {}
    for folder in ("core", "domains", "runner"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            modules[str(path.relative_to(ROOT))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    modules["evaluation/native_measurement_recovery.py"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    return {
        "constructor_runtime_asset": actual,
        "source_file_sha256s": lock.locked_source_hashes,
        "replay_module_sha256s": modules,
        "replay_module_manifest_sha256": _digest(modules),
    }


def _locked_paths(scenario, inputs, trace):
    locked_hashes = (scenario.get("source_contract") or {}).get("file_sha256s", {})
    locked_paths = set(locked_hashes)

    # Historical event metadata can spell the same locked input through a
    # repository alias. Authenticate the local alias before allowing relocation.
    def bind_aliases(value):
        if isinstance(value, dict):
            for item in value.values():
                bind_aliases(item)
        elif isinstance(value, list):
            for item in value:
                bind_aliases(item)
        elif isinstance(value, str) and value.startswith("/"):
            marker = "/OPERATE-Benchmark/"
            if marker in value:
                relative = value.split(marker, 1)[1]
                candidate = ROOT / relative
                if (
                    candidate.is_file()
                    and hashlib.sha256(candidate.read_bytes()).hexdigest()
                    in locked_hashes.values()
                ):
                    locked_paths.add(relative)

    bind_aliases(inputs["backend_tick_records"])
    bind_aliases(trace)
    return tuple(sorted(locked_paths))


def _replay(scenario, inputs, trace, ledger, population):
    from runner.checkpoint import _action_from_json

    if scenario["backend_kind"] == "pandapower_lv":
        from domains.microgrid.adapter import MicrogridEnvironment

        env = MicrogridEnvironment()
    else:
        from domains.power_grid.adapter import PowerGridEnvironment

        env = PowerGridEnvironment()
    locked_paths = _locked_paths(scenario, inputs, trace)
    records, diffs, observations, effects = [], [], [], []
    try:
        env.reset(scenario, scenario["seed"])
        env.snapshot()  # The original logical episode's initial boundary read.
        for row, old in zip(trace, inputs["backend_tick_records"], strict=True):
            extra = (row.get("info") or {}).get("extra") or {}
            investigation = extra.get("within_tick_investigation")
            if investigation:
                env.execute_investigation(
                    _action_from_json(investigation["investigation_action"])
                )
            reconciliation = extra.get("control_reconciliation")
            if reconciliation:
                env.stage_control(_action_from_json(reconciliation["initial_action"]))
                if reconciliation.get("retry_action"):
                    env.stage_control(_action_from_json(reconciliation["retry_action"]))
                result = env.advance_staged_control()
            else:
                result = env.step(_action_from_json(row["action"]))
            new = env._backend.scoring_records()[-1]
            records.append(new)
            observations.append(deepcopy(result.observation))
            effects.append(deepcopy(result.info.realized_events))
            diffs.extend(
                _differences(
                    old,
                    new,
                    f"/records/{old['tick']}",
                    locked_paths=locked_paths,
                    symmetric=True,
                    allowed_new_keys={
                        "native_node_meter"
                        if scenario["backend_kind"].startswith("opendss")
                        else "native_service_meter"
                    },
                )
            )
            diffs.extend(
                _differences(
                    row["observation"],
                    result.observation,
                    f"/observations/{old['tick']}",
                    locked_paths=locked_paths,
                    symmetric=True,
                )
            )
            diffs.extend(
                _differences(
                    (row.get("info") or {}).get("realized_events", []),
                    result.info.realized_events,
                    f"/effects/{old['tick']}",
                    locked_paths=locked_paths,
                    symmetric=True,
                )
            )
        replay_ledger = env.evidence.to_jsonable()
        meter_key = (
            "native_node_meter"
            if scenario["backend_kind"].startswith("opendss")
            else "native_service_meter"
        )
        replay_states = [
            row
            for row in replay_ledger
            if row.get("source") == "engine" and row.get("kind") == "backend_tick"
        ]
        if len(replay_states) != len(records) or any(
            type(state.get("tick")) is not int
            or row["tick"] != state["tick"]
            or row.get(meter_key) != state["payload"].get(meter_key)
            for row, state in zip(records, replay_states, strict=True)
        ):
            raise ValueError("native_recovery_replay_meter_engine_mismatch")
        original_tools = [
            {"tick": row["tick"], "payload": row["payload"]}
            for row in ledger
            if row.get("source") == "tool" and row.get("kind") == "tool_call"
        ]
        replay_tools = [
            {"tick": row["tick"], "payload": row["payload"]}
            for row in replay_ledger
            if row.get("source") == "tool" and row.get("kind") == "tool_call"
        ]
        diffs.extend(
            _differences(
                original_tools,
                replay_tools,
                "/tool_receipts",
                locked_paths=locked_paths,
                symmetric=True,
            )
        )
        costs = env._backend.ground_truth_costs()
        diffs.extend(
            _differences(
                inputs["cost_components"], costs, "/cost_components", symmetric=True
            )
        )
        return records, {
            "differences": diffs,
            "original_cost_components": inputs["cost_components"],
            "replay_cost_components": costs,
            "cost_components_exact": costs == inputs["cost_components"],
            "original_tool_receipts_sha256": _digest(original_tools),
            "replay_tool_receipts_sha256": _digest(replay_tools),
            "replayed_tool_receipt_count": len(replay_tools),
            "primary_record_fields_exact": not any(
                d["path"].startswith("/records/")
                and d["path"].split("/")[3]
                not in {"network_loss_mw", "grid_exchange_mw"}
                for d in diffs
            ),
            "original_records_sha256": _digest(inputs["backend_tick_records"]),
            "replayed_records_sha256": _digest(records),
            "replayed_records": records,
            "replayed_observations": observations,
            "replayed_effects": effects,
            "replayed_tool_receipts": replay_tools,
            "replayed_engine_states": replay_states,
        }
    finally:
        env.close()


def _proof_differences(scenario, inputs, trace, ledger, proof):
    """Reconstruct the complete comparison from original bytes and replay outputs."""
    records = proof.get("replayed_records")
    observations, effects = (
        proof.get("replayed_observations"),
        proof.get("replayed_effects"),
    )
    size = len(inputs["backend_tick_records"])
    if any(
        not isinstance(value, list) or len(value) != size
        for value in (records, observations, effects)
    ):
        raise ValueError("native_recovery_replay_proof_window_invalid")
    if any(
        type(row.get("tick")) is not int or row["tick"] != tick
        for tick, row in enumerate(records)
    ):
        raise ValueError("native_recovery_replay_proof_clock_invalid")
    paths = _locked_paths(scenario, inputs, trace)
    diffs = []
    for old, new, row, observation, event in zip(
        inputs["backend_tick_records"],
        records,
        trace,
        observations,
        effects,
        strict=True,
    ):
        tick = old["tick"]
        diffs.extend(
            _differences(
                old,
                new,
                f"/records/{tick}",
                locked_paths=paths,
                symmetric=True,
                allowed_new_keys={
                    "native_node_meter"
                    if scenario["backend_kind"].startswith("opendss")
                    else "native_service_meter"
                },
            )
        )
        diffs.extend(
            _differences(
                row["observation"],
                observation,
                f"/observations/{tick}",
                locked_paths=paths,
                symmetric=True,
            )
        )
        diffs.extend(
            _differences(
                (row.get("info") or {}).get("realized_events", []),
                event,
                f"/effects/{tick}",
                locked_paths=paths,
                symmetric=True,
            )
        )
    original_tools = [
        {"tick": row["tick"], "payload": row["payload"]}
        for row in ledger
        if row.get("source") == "tool" and row.get("kind") == "tool_call"
    ]
    replay_tools = proof.get("replayed_tool_receipts")
    if not isinstance(replay_tools, list):
        raise ValueError("native_recovery_replay_receipts_missing")
    diffs.extend(
        _differences(
            original_tools,
            replay_tools,
            "/tool_receipts",
            locked_paths=paths,
            symmetric=True,
        )
    )
    costs = proof.get("replay_cost_components")
    if not isinstance(costs, dict):
        raise ValueError("native_recovery_replay_cost_components_missing")
    diffs.extend(
        _differences(
            inputs["cost_components"], costs, "/cost_components", symmetric=True
        )
    )
    if (
        proof.get("original_records_sha256") != _digest(inputs["backend_tick_records"])
        or proof.get("replayed_records_sha256") != _digest(records)
        or proof.get("original_cost_components") != inputs["cost_components"]
        or proof.get("original_tool_receipts_sha256") != _digest(original_tools)
        or proof.get("replay_tool_receipts_sha256") != _digest(replay_tools)
        or proof.get("differences") != diffs
    ):
        raise ValueError("native_recovery_replay_proof_digest_or_difference_mismatch")
    states = proof.get("replayed_engine_states")
    key = (
        "native_node_meter"
        if scenario["backend_kind"].startswith("opendss")
        else "native_service_meter"
    )
    if (
        not isinstance(states, list)
        or len(states) != size
        or any(
            state.get("source") != "engine"
            or state.get("kind") != "backend_tick"
            or type(state.get("tick")) is not int
            or state.get("tick") != record["tick"]
            or state.get("payload", {}).get(key) != record.get(key)
            for state, record in zip(states, records, strict=True)
        )
    ):
        raise ValueError("native_recovery_replayed_engine_meter_mismatch")
    return diffs


def _equivalent(scenario, inputs, records, diffs, policy, runtime):
    if not diffs:
        return True
    if (
        policy != PP_QUANTIZED_EQUIVALENCE
        or scenario["backend_kind"] != "pandapower_lv"
    ):
        return False
    if (runtime or {}).get("replay_module_sha256s", {}).get(
        policy["source_module"]
    ) != policy["source_module_sha256"]:
        return False
    tick_h = scenario.get("tick_minutes", 60) / 60.0
    if type(tick_h) not in (int, float) or not math.isfinite(tick_h) or tick_h <= 0:
        return False
    originals = inputs["backend_tick_records"]
    for diff in diffs:
        parts = diff["path"].split("/")
        if (
            len(parts) != 4
            or parts[1] != "records"
            or not parts[2].isdigit()
            or parts[3] not in policy["allowed_raw_fields"]
        ):
            return False
        tick, field = int(parts[2]), parts[3]
        if tick >= len(originals):
            return False
        old, new = originals[tick].get(field), records[tick].get(field)
        if any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in (old, new)
        ):
            return False
        # Preserve sign, exact-zero and import/export tariff branches.
        if (old == 0) != (new == 0) or (old > 0) != (new > 0) or (old < 0) != (new < 0):
            return False
        if field == "grid_exchange_mw":
            gain = policy["production_cost_per_mwh"] * tick_h
            if abs(new - old) * gain >= policy["production_quantum"] / 2:
                return False
            record = originals[tick]
            penalties = (
                record["n_voltage_violations"]
                * policy["voltage_violation_cost_per_tick"]
                + record["n_overloads"] * policy["overload_cost_per_tick"]
                + record["n_disconnected_lines"]
                * policy["disconnection_cost_per_line_tick"]
            )
            if any(
                round(
                    max(0.0, value) * gain + penalties,
                    policy["production_round_decimal_places"],
                )
                != record["production_cost"]
                for value in (old, new)
            ):
                return False
        elif abs(new - old) >= policy["balance_quantum_mw"] / 2 or round(
            old, policy["balance_round_decimal_places"]
        ) != round(new, policy["balance_round_decimal_places"]):
            return False
    return True


def recover_voltage_measurement(
    scenario: dict,
    *,
    source_contract: dict,
    population: dict,
    snapshot_ref: dict,
    trace_ref: dict,
    ledger_ref: dict,
    allow_replay: bool = False,
    recovery_equivalence: dict | None = None,
    artifact_relocations: dict | None = None,
) -> dict:
    """Recover missing measurements only with explicit offline replay permission.

    The declared LV consumer quantization policy is the only supported nonexact
    rule. All other native, observation, effect, receipt and cost fields are exact.
    """
    snapshot, inputs, trace, ledger, completion = _inputs(
        scenario,
        source_contract,
        population,
        snapshot_ref,
        trace_ref,
        ledger_ref,
        artifact_relocations=artifact_relocations,
    )
    meter_key = (
        "native_node_meter"
        if scenario["backend_kind"].startswith("opendss")
        else "native_service_meter"
    )
    records = inputs["backend_tick_records"]
    proof, fence = {"method": "original_private_meter"}, None
    if allow_replay:
        import yaml

        scenario_path = Path(source_contract["scenario_path"])
        if not scenario_path.is_absolute():
            scenario_path = ROOT / scenario_path
        raw_scenario = scenario_path.read_bytes()
        if (
            hashlib.sha256(raw_scenario).hexdigest()
            != source_contract["scenario_sha256"]
            or yaml.safe_load(raw_scenario) != scenario
        ):
            raise ValueError("native_recovery_replay_scenario_bytes_mismatch")
    if completion.get("score") is None and allow_replay:
        fence = _runtime_fence(scenario, population)
        records, proof = _replay(scenario, inputs, trace, ledger, population)
        if _runtime_fence(scenario, population) != fence:
            raise ValueError("native_recovery_runtime_changed_during_replay")
        proof["method"] = (
            "deterministic_environment_replay_with_original_queries_and_protocol"
        )
        reconstructed = deepcopy(inputs)
        reconstructed["backend_tick_records"] = records
        completion = measure_voltage_completion027(
            scenario,
            source_contract=source_contract,
            population=population,
            snapshot_inputs=reconstructed,
            trace=trace,
        )
    elif completion.get("score") is not None:
        states = [
            row
            for row in ledger
            if row.get("source") == "engine" and row.get("kind") == "backend_tick"
        ]
        if any(
            state["payload"].get(meter_key) != row.get(meter_key)
            for state, row in zip(states, records, strict=True)
        ):
            raise ValueError("native_recovery_original_meter_ledger_mismatch")
    candidate = deepcopy(completion)
    if proof.get("replayed_records") is not None:
        diffs = _proof_differences(scenario, inputs, trace, ledger, proof)
    else:
        diffs = proof.get("differences") or []
    if not _equivalent(scenario, inputs, records, diffs, recovery_equivalence, fence):
        completion = {
            **completion,
            "score": None,
            "reason": "native_recovery_equivalence_unqualified",
        }
    derived = []
    for old, new in zip(inputs["backend_tick_records"], records, strict=True):
        if new.get(meter_key) is None:
            continue
        payload = {
            "tick": old["tick"],
            "original_record_sha256": _digest(old),
            "population_contract_sha256": population["contract_sha256"],
            "meter_key": meter_key,
            "meter": new[meter_key],
            "replayed_record_sha256": _digest(new),
        }
        derived.append(
            {
                "evidence_id": "derived-native-voltage:" + _digest(payload),
                "source": "native_recovery",
                "kind": "derived_native_voltage_measurement",
                "tick": old["tick"],
                "payload": payload,
            }
        )
    result = {
        "schema_version": SCHEMA,
        "status": "recovered" if completion.get("score") is not None else "unqualified",
        "original_artifacts": {
            "snapshot": deepcopy(snapshot_ref),
            "trace": deepcopy(trace_ref),
            "ledger": deepcopy(ledger_ref),
        },
        "original_identity": snapshot["payload"]["identity"],
        "scenario_signature": scenario["scenario_signature"],
        "seed": scenario["seed"],
        "source_contract_sha256": _digest(source_contract),
        "population_contract_sha256": population["contract_sha256"],
        "replay_runtime": fence,
        "replay_proof": proof,
        "recovery_equivalence": recovery_equivalence,
        "completion": completion,
        "candidate_completion": candidate,
        "derived_evidence": derived,
        "original_evidence_ids_sha256": _digest([row["evidence_id"] for row in ledger]),
    }
    result["completion"]["evidence_ids"] = [row["evidence_id"] for row in derived]
    # Reauthenticate original bytes after any native work; never mutate them.
    _inputs(
        scenario,
        source_contract,
        population,
        snapshot_ref,
        trace_ref,
        ledger_ref,
        artifact_relocations=artifact_relocations,
    )
    result["sidecar_sha256"] = _digest(result)
    return result


def verify_voltage_measurement_recovery(
    sidecar: dict,
    scenario: dict,
    *,
    source_contract: dict,
    population: dict,
    snapshot_ref: dict,
    trace_ref: dict,
    ledger_ref: dict,
    recovery_equivalence: dict | None = None,
    artifact_relocations: dict | None = None,
) -> dict:
    """Authenticate cached replay outputs without importing native engines."""
    if sidecar.get("schema_version") == CIGRE_PQ_CERTIFICATE_SCHEMA:
        if recovery_equivalence is not None:
            raise ValueError("cigre_pq_certificate_requires_no_replay_equivalence")
        return verify_cigre_original_completion_certificate(
            sidecar,
            scenario,
            source_contract=source_contract,
            population=population,
            snapshot_ref=snapshot_ref,
            trace_ref=trace_ref,
            ledger_ref=ledger_ref,
            artifact_relocations=artifact_relocations,
        )
    if sidecar.get("schema_version") != SCHEMA or sidecar.get(
        "sidecar_sha256"
    ) != _digest(
        {key: value for key, value in sidecar.items() if key != "sidecar_sha256"}
    ):
        raise ValueError("native_recovery_sidecar_hash_mismatch")
    snapshot, inputs, trace, ledger, _ = _inputs(
        scenario,
        source_contract,
        population,
        snapshot_ref,
        trace_ref,
        ledger_ref,
        artifact_relocations=artifact_relocations,
    )
    refs = {"snapshot": snapshot_ref, "trace": trace_ref, "ledger": ledger_ref}
    if (
        any(
            sidecar.get("original_artifacts", {}).get(name, {}).get("sha256")
            != ref["sha256"]
            or sidecar.get("original_artifacts", {}).get(name, {}).get("byte_count")
            != ref.get("byte_count")
            for name, ref in refs.items()
        )
        or sidecar.get("original_identity") != snapshot["payload"]["identity"]
        or sidecar.get("scenario_signature") != scenario["scenario_signature"]
        or sidecar.get("seed") != scenario["seed"]
        or sidecar.get("source_contract_sha256") != _digest(source_contract)
        or sidecar.get("population_contract_sha256") != population["contract_sha256"]
        or sidecar.get("original_evidence_ids_sha256")
        != _digest([row["evidence_id"] for row in ledger])
    ):
        raise ValueError("native_recovery_sidecar_identity_mismatch")
    proof = sidecar.get("replay_proof") or {}
    replayed = (
        proof.get("method")
        == "deterministic_environment_replay_with_original_queries_and_protocol"
    )
    if replayed:
        diffs = _proof_differences(scenario, inputs, trace, ledger, proof)
        records = proof["replayed_records"]
    else:
        records, diffs = inputs["backend_tick_records"], proof.get("differences") or []
        if proof.get("method") != "original_private_meter":
            raise ValueError("native_recovery_proof_method_invalid")
    if sidecar.get("recovery_equivalence") != recovery_equivalence:
        raise ValueError("native_recovery_equivalence_policy_mismatch")
    if not _equivalent(
        scenario,
        inputs,
        records,
        diffs,
        recovery_equivalence,
        sidecar.get("replay_runtime"),
    ):
        raise ValueError("native_recovery_equivalence_unqualified")
    reconstructed = deepcopy(inputs)
    reconstructed["backend_tick_records"] = deepcopy(records)
    derived = sidecar.get("derived_evidence") or []
    if (
        len(derived) != len(records)
        or any(type(row.get("tick")) is not int for row in derived)
        or [row.get("tick") for row in derived] != list(range(len(records)))
    ):
        raise ValueError("native_recovery_derived_window_incomplete")
    key = (
        "native_node_meter"
        if scenario["backend_kind"].startswith("opendss")
        else "native_service_meter"
    )
    for original, record, evidence in zip(
        inputs["backend_tick_records"], records, derived, strict=True
    ):
        payload = evidence.get("payload") or {}
        if (
            evidence.get("evidence_id") != "derived-native-voltage:" + _digest(payload)
            or evidence.get("source") != "native_recovery"
            or evidence.get("kind") != "derived_native_voltage_measurement"
            or payload.get("original_record_sha256") != _digest(original)
            or payload.get("population_contract_sha256")
            != population["contract_sha256"]
            or payload.get("tick") != original["tick"]
            or payload.get("meter_key") != key
            or payload.get("meter") != record.get(key)
            or payload.get("replayed_record_sha256") != _digest(record)
        ):
            raise ValueError("native_recovery_derived_evidence_invalid")
    measured = measure_voltage_completion027(
        scenario,
        source_contract=source_contract,
        population=population,
        snapshot_inputs=reconstructed,
        trace=trace,
    )
    if measured != sidecar.get("candidate_completion"):
        raise ValueError("native_recovery_completion_mismatch")
    if sidecar.get("status") != "recovered" or measured.get("score") is None:
        raise ValueError("native_recovery_equivalence_unqualified")
    expected = {**measured, "evidence_ids": [row["evidence_id"] for row in derived]}
    if sidecar.get("completion") != expected:
        raise ValueError("native_recovery_public_completion_mismatch")
    return deepcopy(expected)


CIGRE_PQ_CERTIFICATE_SCHEMA = "cigre_original_constant_pq_completion.v1"
CIGRE_CONSTANT_PQ_LAW = {
    "schema_version": "cigre_simbench_constant_pq_source_law.v1",
    "backend_kind": "cigre_distribution",
    "networks": {
        "simbench:1-MV-rural--0-sw": "MV1.101",
        "simbench:1-MV-semiurb--0-sw": "MV2.101",
    },
    "package_versions": {"pandapower": "3.5.3", "simbench": "1.6.2"},
    "package_files": {
        "pandapower/results_bus.py": "c09a5c168a45d8861a5d0979dc92060c8ed0c239d4c3ebd64daac10b5099e230",
        "pandapower/auxiliary.py": "8fb28ddf53eb813455f90760ec6d00b68d965c67fcecad3305e7ae37613df481",
        "simbench/converter/csv_pp_converter.py": "b88f796015d1259a2eb8b8e2a414549dcef8601d740dc3f51269bebb83891102",
        "simbench/converter/format_information.py": "355c60bf32e29435c0212b6d4ad809a590c2494b29ae9c62e5f46419d12dbfc5",
        "simbench/networks/extract_simbench_grids_from_csv.py": "96d94d608939a26c54d259cf52f518d6a19fac932d711bc069165fe93e6ec11e",
        "simbench/networks/profiles.py": "ac205d286296702dc45dfcc64bda1dbcaf07ce26f6087f746facb9f598510c70",
        "simbench/networks/simbench_code.py": "8af0ec8c22b98a2b11edb23497238f5c666fb67e4308910a2fa3b1b624ebe12a",
    },
    "repository_files": {
        "domains/power_grid/backends/cigre_distribution.py": "b64b5cd881ed1e3c2d59629e64278bd3f425aeb56124664af03edeb702a6c56a",
        "domains/power_grid/adapter.py": "c1017b1d6080fbea857fb8886502a515c1e092e6f2f5ab1c7b2b083d7c527ff8",
        "domains/power_grid/native_tools.py": "6ec4ee3f796e55bda51270d2f29c5939e297741509eeca489fbab2503b41ad10",
        "core/fog_of_war.py": "d1ea2999fbcd4f3400d946df08d552701e8875df5b43c18df7426483db8404ba",
        "core/stakeholder_trust.py": "c0d9d6b0ccb3cae8e6860740fed9c7f0ccf2ce423a09180fd862d9c50db3618b",
        "core/tool_protocol.py": "12d3b08a5390ae546934747388ae57e0becdb616612f17cc10ae02292963fc92",
        "core/world_evolution_contract.py": "773d58026c193a7bc622006ce45463a7131dbc47d386f438f733bb3772c94754",
        "core/common_tools.py": "daa68e288e379c530041f010f5ecbfb91133388f1f6cc58eec400d433751daac",
        "runner/episode.py": "263304b7a918e52bcd26130d6c0b3904ac1c1d9af7d51022eda3a7bfb6640973",
    },
    "load_defaults": {
        "in_service": True,
        "scaling": 1.0,
        "const_i_p_percent": 0.0,
        "const_z_p_percent": 0.0,
        "const_i_q_percent": 0.0,
        "const_z_q_percent": 0.0,
    },
    "result_law": "res_load.P=P_set*scaling*load_is*(1-ci-cz+ci*vm+cz*vm^2)",
    "allowed_nonload_perturbations": ["renewable_output_error", "forecast_bias"],
    "safe_materialized_tools": [
        "set_der_reactive_power",
        "set_transformer_tap",
        "switch_capacitor",
        "redispatch_generation",
        "set_battery_dispatch",
        "commit_reserve",
        "request_mutual_aid",
        "negotiate_with_stakeholder",
    ],
    "safe_nonphysical_tools": [
        "wait",
        "noop",
        "query_grid_state",
        "query_chronics_window",
        "forecast_query",
        "query_active_dilemmas",
        "escalate_to_human",
        "commit_to_plan",
        "cancel_action",
    ],
    "requested_volume": "original_unnoised_snapshot_totals.demand_mw_times_tick_hours",
    "interpretation": "symbolic_native_delivery_certificate_not_per_load_res_load_measurement",
}


def _verify_cigre_pq_source(scenario, source_contract, population):
    """Read pinned source code/data using metadata; import no simulator package."""
    import csv
    from importlib.metadata import distribution
    import yaml

    path = Path(source_contract["scenario_path"])
    if not path.is_absolute():
        path = ROOT / path
    raw = path.read_bytes()
    if (
        hashlib.sha256(raw).hexdigest() != source_contract["scenario_sha256"]
        or yaml.safe_load(raw) != scenario
    ):
        raise ValueError("cigre_pq_scenario_bytes_mismatch")
    law = CIGRE_CONSTANT_PQ_LAW
    config = scenario.get("backend_config") or {}
    network = config.get("network")
    if (
        network not in law["networks"]
        or population.get("network") != network
        or config.get("profile_source") != "simbench_bundled_full_year"
    ):
        raise ValueError("cigre_pq_source_network_or_profile_unsupported")
    distributions = {name: distribution(name) for name in law["package_versions"]}
    for name, expected in law["package_versions"].items():
        if distributions[name].version != expected:
            raise ValueError("cigre_pq_source_law_package_version_mismatch")
    for relative, expected in law["package_files"].items():
        actual = Path(distributions[relative.split("/")[0]].locate_file(relative))
        if hashlib.sha256(actual.read_bytes()).hexdigest() != expected:
            raise ValueError("cigre_pq_source_law_package_bytes_mismatch")
    for relative, expected in law["repository_files"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("cigre_pq_source_law_repository_bytes_mismatch")
    asset = population.get("constructor_runtime_asset") or {}
    if (
        asset.get("package") != "simbench"
        or asset.get("package_version") != law["package_versions"]["simbench"]
        or asset.get("module") != "networks/extract_simbench_grids_from_csv.py"
        or asset.get("module_sha256")
        != law["package_files"]["simbench/networks/extract_simbench_grids_from_csv.py"]
    ):
        raise ValueError("cigre_pq_constructor_asset_identity_mismatch")
    paths = {}
    for item in asset.get("data_assets") or []:
        relative = item["path"]
        if not relative.startswith("simbench/") or ".." in Path(relative).parts:
            raise ValueError("cigre_pq_constructor_asset_path_invalid")
        actual = Path(distributions["simbench"].locate_file(relative))
        if hashlib.sha256(actual.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("cigre_pq_constructor_data_bytes_mismatch")
        paths[Path(relative).name] = actual
    if not {"Load.csv", "LoadProfile.csv"} <= paths.keys():
        raise ValueError("cigre_pq_load_source_tables_missing")
    subnet = law["networks"][network]
    with paths["Load.csv"].open(encoding="utf-8", newline="") as stream:
        loads = [
            row
            for row in csv.DictReader(stream, delimiter=";")
            if row.get("subnet") == subnet
            or row.get("subnet", "").startswith(subnet + "_")
        ]
    assignments = scenario.get("load_assignments") or []
    inventory = population["native_source_population"]["loads"]
    if (
        len(loads) != len(inventory)
        or len(assignments) != len(inventory)
        or len({row["id"] for row in loads}) != len(loads)
        or len({row["load_id"] for row in assignments}) != len(assignments)
    ):
        raise ValueError("cigre_pq_source_load_identity_coverage_mismatch")
    selected = []
    for index, (row, assignment, native) in enumerate(
        zip(loads, assignments, inventory, strict=True)
    ):
        p, q = float(row["pLoad"]), float(row["qLoad"])
        if (
            not math.isfinite(p)
            or p < 0
            or not math.isfinite(q)
            or not row.get("profile")
            or row["profile"] == "NULL"
        ):
            raise ValueError("cigre_pq_source_base_load_invalid")
        if native["index"] != index or not str(assignment.get("bus_id", "")).endswith(
            "_" + str(native["bus"])
        ):
            raise ValueError("cigre_pq_source_assignment_native_mapping_mismatch")
        selected.append(
            {
                "native_load_index": index,
                "csv_source_load_id": row["id"],
                "assignment_load_id": assignment["load_id"],
                "native_bus_index": native["bus"],
                "profile": row["profile"],
                "base_p_mw": p,
                "base_q_mvar": q,
            }
        )
    start, step = config.get("profile_start_index"), config.get("profile_step")
    if type(start) is not int or start < 0 or type(step) is not int or step <= 0:
        raise ValueError("cigre_pq_source_profile_axis_invalid")
    columns = {
        row["profile"] + suffix for row in selected for suffix in ("_pload", "_qload")
    }
    with paths["LoadProfile.csv"].open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter=";")
        header = next(reader)
        if len(header) != len(set(header)) or not columns <= set(header):
            raise ValueError("cigre_pq_source_profile_columns_missing")
        positions = {name: header.index(name) for name in columns}
        frames = [
            {name: row[index] for name, index in positions.items()}
            for row in reader
            if row
        ]
    if not frames:
        raise ValueError("cigre_pq_source_profile_window_missing")
    indices = [
        (start + tick * step) % len(frames) for tick in range(scenario["horizon_ticks"])
    ]
    values = []
    for frame_index in indices:
        frame = frames[frame_index]
        snapshot = []
        for load in selected:
            relative_p = float(frame[load["profile"] + "_pload"])
            relative_q = float(frame[load["profile"] + "_qload"])
            active, reactive = (
                load["base_p_mw"] * relative_p,
                load["base_q_mvar"] * relative_q,
            )
            if (
                not math.isfinite(relative_p)
                or relative_p < 0
                or not math.isfinite(relative_q)
                or not math.isfinite(active)
                or active < 0
                or not math.isfinite(reactive)
            ):
                raise ValueError("cigre_pq_source_profile_power_invalid")
            snapshot.append([active, reactive])
        values.append(snapshot)
    return {
        "source_law_sha256": _digest(law),
        "constructor_asset_sha256": _digest(asset),
        "selected_source_loads": selected,
        "selected_source_loads_sha256": _digest(selected),
        "profile_start_index": start,
        "profile_step": step,
        "profile_row_count": len(frames),
        "consumed_profile_indices": indices,
        "source_profile_power_values_sha256": _digest(values),
        "active_source_profile_power_nonnegative": True,
        "reactive_source_profile_power_finite": True,
    }


def _cigre_pq_control_proof(scenario, inputs, trace, ledger):
    law = CIGRE_CONSTANT_PQ_LAW
    if any(
        p.get("kind") not in law["allowed_nonload_perturbations"]
        for p in scenario.get("perturbations", [])
    ):
        raise ValueError("cigre_pq_load_or_topology_perturbation_unsupported")
    for record in inputs["backend_tick_records"]:
        if (
            type(record.get("n_disconnected_lines")) is not int
            or record["n_disconnected_lines"] != 0
        ):
            raise ValueError("cigre_pq_native_disconnect_present_or_unknown")
        if record.get("shed_penalty") != 0:
            raise ValueError("cigre_pq_native_shed_penalty_present_or_unknown")
    per_load = inputs.get("per_load_shed_mwh")
    if (
        not isinstance(per_load, dict)
        or set(per_load)
        != {row["load_id"] for row in scenario.get("load_assignments", [])}
    ) or any(
        type(value) not in (int, float) or not math.isfinite(value) or value != 0
        for value in per_load.values()
    ):
        raise ValueError("cigre_pq_cumulative_shed_present_or_unknown")
    calls = {}
    for row in trace:
        extra = (row.get("info") or {}).get("extra") or {}
        actions = [row.get("action") or {}]
        if extra.get("within_tick_investigation"):
            actions.append(extra["within_tick_investigation"]["investigation_action"])
        if extra.get("control_reconciliation"):
            actions.extend(
                a
                for a in [
                    extra["control_reconciliation"].get("initial_action"),
                    extra["control_reconciliation"].get("retry_action"),
                ]
                if a
            )
        for action in actions:
            for call in action.get("actions", []):
                key = call.get("call_id")
                if key:
                    if key in calls and (
                        calls[key].get("name"),
                        calls[key].get("args"),
                    ) != (call.get("name"), call.get("args")):
                        raise ValueError("cigre_pq_conflicting_original_call_identity")
                    calls[key] = call
    receipts = [
        row
        for row in ledger
        if row.get("source") == "tool" and row.get("kind") == "tool_call"
    ]
    materialized = []
    blocked = {
        "pending",
        "unsupported",
        "noop",
        "error",
        "failed",
        "rejected",
        "canceled",
        "cancelled",
        "expired",
        "superseded",
    }
    for receipt in receipts:
        payload = receipt["payload"]
        result = payload.get("payload") or {}
        status = str(result.get("_status", result.get("status", ""))).lower()
        if (
            type(payload.get("ok")) is not bool
            or type(payload.get("state_changing")) is not bool
        ):
            raise ValueError("cigre_pq_tool_receipt_status_untyped")
        if (
            payload.get("ok") is True
            and status not in blocked
            and payload.get("name")
            not in (
                set(law["safe_materialized_tools"])
                | set(law["safe_nonphysical_tools"])
                | {"shed_load", "switch_branch"}
            )
        ):
            raise ValueError("cigre_pq_successful_unknown_callback")
        if (
            payload.get("ok") is not True
            or status in blocked
            or payload.get("state_changing") is not True
        ):
            if (
                status == "pending"
                and payload.get("name") not in law["safe_materialized_tools"]
                and payload.get("name") not in law["safe_nonphysical_tools"]
            ):
                raise ValueError("cigre_pq_pending_load_or_unknown_control")
            continue
        name = payload.get("name")
        request = calls.get(payload.get("call_id"))
        if request is None or request.get("name") != name:
            raise ValueError("cigre_pq_materialized_control_request_unbound")
        args = request.get("args") or {}
        if name == "shed_load":
            quantities = [result.get("shed_mw"), args.get("mw")]
            if any(
                type(quantity) not in (int, float)
                or not math.isfinite(quantity)
                or quantity != 0
                for quantity in quantities
            ):
                raise ValueError("cigre_pq_nonzero_materialized_shed")
        elif name == "switch_branch":
            if result.get("connect", args.get("connect")) is not True:
                raise ValueError("cigre_pq_materialized_disconnect")
        elif name not in law["safe_materialized_tools"]:
            raise ValueError("cigre_pq_materialized_load_or_unknown_control")
        materialized.append(
            {
                "evidence_id": receipt["evidence_id"],
                "tick": receipt["tick"],
                "call_id": payload.get("call_id"),
                "name": name,
                "payload_sha256": _digest(payload),
            }
        )
    # Close every actual effect channel, not just aggregate zero shed totals.
    from core.world_evolution_contract import realized_event_evidence_tick

    if any(
        entry.get("kind") == "cascade_effect" or entry.get("source") == "cascade_bus"
        for entry in ledger
    ):
        raise ValueError("cigre_pq_external_cascade_effect_present")
    entries = [
        entry
        for entry in ledger
        if entry.get("source") == "engine" and entry.get("kind") == "realized_event"
    ]
    events = [entry.get("payload") or {} for entry in entries]
    input_events = inputs.get("realized_events")
    trace_events = [
        event
        for row in trace
        for event in (row.get("info") or {}).get("realized_events", [])
    ]
    if (
        not isinstance(input_events, list)
        or _differences(
            input_events,
            [{**entry["payload"], "tick": entry["tick"]} for entry in entries],
            symmetric=True,
        )
        or _differences(trace_events, events, symmetric=True)
    ):
        raise ValueError("cigre_pq_authoritative_effect_inventory_mismatch")
    for record in inputs["backend_tick_records"]:
        if "realized_events" in record:
            expected = [
                event for event in events if event.get("tick") == record["tick"]
            ]
            if _differences(record["realized_events"], expected, symmetric=True):
                raise ValueError("cigre_pq_native_record_effect_inventory_mismatch")
    allowed_fields = {
        "demand_mw",
        "der_generation_mw",
        "reserves_procured_mw",
        "rho_max",
        "n_voltage_violations",
        "n_overloads",
        "n_disconnected_lines",
        "renewable_generation_mw",
        "aggregate_generation_mw",
        "demand_forecast_mw",
        "der_reactive_power_mvar",
        "capacitor_in_service",
        "transformer_tap_pos",
        "storage_dispatch_mw",
        "der_dispatch_cap_mw",
        "pending_reserve_mw",
    }
    supported_effect_tools = {
        "set_der_reactive_power",
        "set_transformer_tap",
        "switch_capacitor",
        "redispatch_generation",
        "set_battery_dispatch",
        "commit_reserve",
    }
    materialized_ids = {row["call_id"] for row in materialized}
    started = []
    for entry, event in zip(entries, events, strict=True):
        if (
            type(entry.get("tick")) is not int
            or type(event.get("tick")) is not int
            or not 0 <= event["tick"] < scenario["horizon_ticks"]
            or realized_event_evidence_tick(event, applied_tick=event["tick"])
            != entry["tick"]
        ):
            raise ValueError("cigre_pq_authoritative_effect_clock_invalid")
        kind = event.get("type", event.get("kind", ""))
        if not set(event.get("changed_state_fields", [])) <= allowed_fields:
            raise ValueError("cigre_pq_unknown_or_load_mutation_effect_field")
        if kind == "simbench_profile_window_started":
            config = scenario["backend_config"]
            if (
                event["tick"] != 0
                or event.get("profile_start_index") != config.get("profile_start_index")
                or event.get("profile_step") != config.get("profile_step")
            ):
                raise ValueError("cigre_pq_original_consumed_profile_axis_mismatch")
            started.append(event)
        elif kind in law["allowed_nonload_perturbations"]:
            if not any(
                p.get("kind") == kind and p.get("trigger_tick") == event["tick"]
                for p in scenario.get("perturbations", [])
            ):
                raise ValueError("cigre_pq_undeclared_original_perturbation_effect")
        elif kind.startswith("cigre_") and kind.endswith("_applied"):
            name = kind[6:-8]
            call_id = event.get("call_id")
            request = calls.get(call_id)
            if (
                name not in supported_effect_tools
                or call_id not in materialized_ids
                or request is None
                or event.get("tool_name") != name
                or _differences(
                    {"name": request["name"], "args": request.get("args", {})},
                    event.get("requested_action"),
                    symmetric=True,
                )
            ):
                raise ValueError("cigre_pq_native_control_effect_request_unbound")
        else:
            raise ValueError("cigre_pq_unknown_original_native_effect")
    if len(started) != 1:
        raise ValueError("cigre_pq_original_consumed_profile_start_unproven")
    return {
        "materialized_receipts": materialized,
        "all_receipts_sha256": _digest(receipts),
        "authoritative_effects_sha256": _digest(events),
        "load_request_factor_by_native_tick": [1.0] * scenario["horizon_ticks"],
        "nonzero_effective_shed_or_load_status_zip_mutation": False,
    }


def certify_cigre_original_completion(
    scenario: dict,
    *,
    source_contract: dict,
    population: dict,
    snapshot_ref: dict,
    trace_ref: dict,
    ledger_ref: dict,
    artifact_relocations: dict | None = None,
) -> dict:
    """Certify native delivery symbolically; record no invented private meter."""
    snapshot, inputs, trace, ledger, _ = _inputs(
        scenario,
        source_contract,
        population,
        snapshot_ref,
        trace_ref,
        ledger_ref,
        artifact_relocations=artifact_relocations,
    )
    if scenario["backend_kind"] != "cigre_distribution":
        raise ValueError("cigre_pq_certificate_backend_unsupported")
    if any(
        "native_service_meter" in record for record in inputs["backend_tick_records"]
    ) or any(
        "native_service_meter" in entry.get("payload", {})
        for entry in ledger
        if entry.get("source") == "engine" and entry.get("kind") == "backend_tick"
    ):
        raise ValueError(
            "cigre_pq_original_private_meter_present_requires_direct_measurement"
        )
    inventory = population["native_source_population"]
    buses, loads = inventory["buses"], inventory["loads"]
    required = [f"bus_{row['index']}" for row in buses]
    if (
        population.get("excluded_bus_ids") != {}
        or population.get("required_bus_ids") != required
        or not buses
        or not loads
        or [b["index"] for b in buses] != list(range(len(buses)))
        or any(b.get("in_service") is not True for b in buses)
        or [load["index"] for load in loads] != list(range(len(loads)))
        or any(
            load.get("in_service") is not True
            or load.get("scaling") != 1.0
            or load["bus"] not in range(len(buses))
            for load in loads
        )
    ):
        raise ValueError("cigre_pq_complete_active_source_population_unproven")
    source_proof = _verify_cigre_pq_source(scenario, source_contract, population)
    control_proof = _cigre_pq_control_proof(scenario, inputs, trace, ledger)
    violations, demands, engine_ids = [], [], []
    for record, row in zip(inputs["backend_tick_records"], trace, strict=True):
        if record.get("converged") is not True:
            raise ValueError("cigre_pq_original_native_convergence_unproven")
        count = record.get("n_voltage_violations")
        if type(count) is not int or not 0 <= count <= len(buses):
            raise ValueError("cigre_pq_original_native_voltage_count_invalid")
        entities = row["observation"].get("entities") or {}
        if {key for key in entities if key.startswith("bus_")} != set(required):
            raise ValueError("cigre_pq_original_source_bus_coverage_incomplete")
        if any(
            type(entities[key].get("vm_pu")) not in (int, float)
            or not math.isfinite(entities[key]["vm_pu"])
            or entities[key]["vm_pu"] <= 0
            for key in required
        ):
            raise ValueError("cigre_pq_original_source_bus_voltage_not_positive_finite")
        demand = (row["observation"].get("totals") or {}).get("demand_mw")
        if (
            type(demand) not in (int, float)
            or not math.isfinite(demand)
            or demand < 0
            or round(demand, 2) != record.get("aggregate_demand_mw")
        ):
            raise ValueError("cigre_pq_original_requested_aggregate_invalid")
        violations.append(count)
        demands.append(demand)
    hours = scenario.get("tick_minutes", 60) / 60
    volume = math.fsum(demands) * hours
    if (
        not math.isfinite(hours)
        or hours <= 0
        or not math.isfinite(volume)
        or volume <= 0
    ):
        raise ValueError("cigre_pq_original_requested_volume_not_positive")
    engine_ids = [
        row["evidence_id"]
        for row in ledger
        if row.get("source") == "engine" and row.get("kind") == "backend_tick"
    ]
    payload = {
        "proof_kind": "original_native_voltage_count_and_constant_pq_supply_law",
        "population_contract_sha256": population["contract_sha256"],
        "source_contract_sha256": _digest(source_contract),
        "source_law_sha256": _digest(CIGRE_CONSTANT_PQ_LAW),
        "source_proof": source_proof,
        "control_proof": control_proof,
        "original_native_evidence_ids": engine_ids,
        "original_native_records_sha256": _digest(inputs["backend_tick_records"]),
        "original_native_voltage_violation_counts": violations,
        "original_unnoised_requested_demand_mw": demands,
        "original_artifact_sha256s": [
            snapshot_ref["sha256"],
            trace_ref["sha256"],
            ledger_ref["sha256"],
        ],
        "load_fulfillment_symbolically_certified": 100.0,
        "requested_energy_mwh": volume,
        "supply_proof_interpretation": "constant_PQ_native_res_load_equals_preshed_request_under_verified_source_and_effect_conditions",
    }
    evidence_id = "derived-original-pq-completion:" + _digest(payload)
    compliant = len(buses) * len(violations) - sum(violations)
    completion = {
        "protocol_revision": "source_voltage_delivered_load.v1",
        "kind": "voltage_reliability",
        "applicable": True,
        "score": 100.0 * compliant / (len(buses) * len(violations)),
        "reason": "original_native_voltage_and_source_law_constant_pq_supply",
        "population_contract_sha256": population["contract_sha256"],
        "voltage_compliant_bus_time": compliant,
        "voltage_required_bus_time": len(buses) * len(violations),
        "load_fulfillment": 100.0,
        "requested_energy_mwh": volume,
        "unserved_energy_mwh": 0.0,
        "aggregation": "minimum_noncompensatory_required_services",
        "supply_proof_kind": "source_law_symbolic_certificate",
        "raw_per_load_delivery_measured": False,
        "evidence_ids": [evidence_id],
    }
    certificate = {
        "schema_version": CIGRE_PQ_CERTIFICATE_SCHEMA,
        "status": "qualified",
        "proof_kind": payload["proof_kind"],
        "scenario_signature": scenario["scenario_signature"],
        "seed": scenario["seed"],
        "original_identity": snapshot["payload"]["identity"],
        "original_artifacts": {
            "snapshot": deepcopy(snapshot_ref),
            "trace": deepcopy(trace_ref),
            "ledger": deepcopy(ledger_ref),
        },
        "source_contract_sha256": _digest(source_contract),
        "population_contract_sha256": population["contract_sha256"],
        "source_law": deepcopy(CIGRE_CONSTANT_PQ_LAW),
        "proof": payload,
        "completion": completion,
        "derived_evidence": [
            {
                "evidence_id": evidence_id,
                "source": "original_evidence_certificate",
                "kind": "source_law_completion_certificate",
                "tick": scenario["horizon_ticks"] - 1,
                "payload": payload,
            }
        ],
        "native_engine_calls": 0,
        "native_replay": False,
        "private_native_meter_fabricated": False,
    }
    _inputs(
        scenario,
        source_contract,
        population,
        snapshot_ref,
        trace_ref,
        ledger_ref,
        artifact_relocations=artifact_relocations,
    )
    certificate["sidecar_sha256"] = _digest(certificate)
    return certificate


def verify_cigre_original_completion_certificate(
    certificate: dict,
    scenario: dict,
    *,
    source_contract: dict,
    population: dict,
    snapshot_ref: dict,
    trace_ref: dict,
    ledger_ref: dict,
    artifact_relocations: dict | None = None,
) -> dict:
    """Re-derive a symbolic certificate from authenticated original bytes only."""
    if certificate.get(
        "schema_version"
    ) != CIGRE_PQ_CERTIFICATE_SCHEMA or certificate.get("sidecar_sha256") != _digest(
        {key: value for key, value in certificate.items() if key != "sidecar_sha256"}
    ):
        raise ValueError("cigre_pq_certificate_hash_mismatch")
    # Paths can move while the original descriptor remains authoritative.
    original_refs = certificate["original_artifacts"]
    incoming = {"snapshot": snapshot_ref, "trace": trace_ref, "ledger": ledger_ref}
    if any(
        original_refs[name].get("sha256") != value.get("sha256")
        or original_refs[name].get("byte_count") != value.get("byte_count")
        for name, value in incoming.items()
    ):
        raise ValueError("cigre_pq_certificate_original_binding_mismatch")
    relocations = dict(artifact_relocations or {})
    for name, value in incoming.items():
        if value["path"] != original_refs[name]["path"]:
            relocations.setdefault(value["sha256"], value["path"])
    expected = certify_cigre_original_completion(
        scenario,
        source_contract=source_contract,
        population=population,
        snapshot_ref=original_refs["snapshot"],
        trace_ref=original_refs["trace"],
        ledger_ref=original_refs["ledger"],
        artifact_relocations=relocations,
    )
    if expected != certificate:
        raise ValueError("cigre_pq_certificate_derivation_mismatch")
    return deepcopy(expected["completion"])
