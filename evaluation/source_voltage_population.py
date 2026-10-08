"""Candidate 0.27 source-fixed voltage and delivered-load fulfillment.

Source construction reads pinned constructor data without advancing a backend
or calling a solver for pandapower; DSS construction uses bounded initial
snapshot solves and records their explicit entry count. Measurement consumes
authenticated private native meters;
legacy requested-load or model-visible voltage fields cannot replace them.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import math

from domains.pandapower_service import capture_source_population, population_sha256


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


@lru_cache(maxsize=16)
def _source_inventory(network: str, asset_sha256: str) -> dict:
    # The caller verifies installed bytes before every invocation, including a
    # cache hit. The asset identity separates inventories across package locks.
    from domains.power_grid.backends.cigre_distribution import _build_distribution_net
    import pandapower.topology as topology

    net = _build_distribution_net(network)
    if list(net.bus.index) != list(range(len(net.bus))):
        raise ValueError("voltage_source_bus_entity_mapping_unsupported")
    active = {
        int(index) for index in net.bus.index if bool(net.bus.at[index, "in_service"])
    }
    if active & set(map(int, topology.unsupplied_buses(net))):
        raise ValueError("voltage_source_active_unsupplied_obligations_undefined")
    inventory = capture_source_population(net)
    population_sha256(inventory)  # Reject a nonfinite source scaling table.
    if any(not _number(row["scaling"]) for row in inventory["loads"]):
        raise ValueError("voltage_source_load_scaling_invalid")
    return inventory


def compile_source_voltage_population(
    scenario: dict,
    *,
    scenario_sha256: str,
    constructor_asset_lock: dict,
    constructor_lock_sha256: str,
) -> dict:
    """Freeze source bus/load identities, source exclusions and both tick axes.

    The caller authenticates scenario YAML bytes against ``scenario_sha256``.
    ``constructor_lock_sha256`` is the canonical JSON SHA-256 of the portable
    supplied lock (sort_keys=True, separators=(",", ":"), allow_nan=False).
    The network needs its own explicit lock entry, even when its loader/data
    bytes happen to equal another network's. No observed entities select IDs.
    """
    backend = scenario.get("backend_kind")
    if backend not in {
        "pandapower_lv",
        "cigre_distribution",
        "opendss_ieee13",
        "opendss_fresh_feeders",
    }:
        raise ValueError("voltage_source_backend_unsupported")
    horizon = scenario.get("horizon_ticks")
    if (
        not _sha256(scenario_sha256)
        or not isinstance(scenario.get("scenario_signature"), str)
        or not scenario["scenario_signature"]
        or type(scenario.get("seed")) is not int
        or type(horizon) is not int
        or horizon <= 0
        or not isinstance(constructor_asset_lock, dict)
        or not _sha256(constructor_lock_sha256)
    ):
        raise ValueError("voltage_source_identity_invalid")
    if population_sha256(constructor_asset_lock) != constructor_lock_sha256:
        raise ValueError("voltage_constructor_lock_hash_mismatch")
    if backend in {"opendss_ieee13", "opendss_fresh_feeders"}:
        return _compile_dss_source_population(
            scenario,
            scenario_sha256=scenario_sha256,
            constructor_asset_lock=constructor_asset_lock,
            constructor_lock_sha256=constructor_lock_sha256,
        )
    config = scenario.get("backend_config") or {}
    network = (
        "synthetic_volt_control_lv"
        if backend == "pandapower_lv"
        else config.get("network", "cigre_mv_with_der_all")
    )
    expected_asset = (constructor_asset_lock.get("constructors") or {}).get(network)
    if not isinstance(expected_asset, dict) or not expected_asset:
        raise ValueError("voltage_constructor_network_lock_missing")
    from domains.power_grid.backends.cigre_distribution import (
        _constructor_runtime_asset,
    )

    actual_asset = _constructor_runtime_asset(network)
    if not actual_asset or actual_asset != expected_asset:
        raise ValueError("voltage_constructor_installed_bytes_mismatch")
    inventory = deepcopy(_source_inventory(network, population_sha256(actual_asset)))
    required = [
        f"bus_{row['index']}" for row in inventory["buses"] if row["in_service"]
    ]
    excluded = {
        f"bus_{row['index']}": "source_bus_out_of_service_voltage_exclusion_only"
        for row in inventory["buses"]
        if not row["in_service"]
    }
    if not required or not any(row["in_service"] for row in inventory["loads"]):
        raise ValueError("voltage_source_population_empty")
    result = {
        "schema_version": "required_voltage_population.v1",
        "source_sha256": scenario_sha256,
        "scenario_signature": scenario["scenario_signature"],
        "seed": scenario["seed"],
        "backend_kind": backend,
        "network": network,
        "required_bus_ids": required,
        "excluded_bus_ids": excluded,
        "required_ticks": list(range(1, horizon + 1)),
        "required_native_ticks": list(range(horizon)),
        "tick_alignment": "post_observation_tick_equals_native_tick_plus_one",
        "exclusion_scope": "voltage_only_does_not_remove_source_load_obligations",
        "native_source_population": inventory,
        "native_source_population_sha256": population_sha256(inventory),
        "constructor_runtime_asset": actual_asset,
        "constructor_lock_sha256": constructor_lock_sha256,
        "source_population_policy": "all_source_in_service_buses_and_all_source_load_indices",
    }
    result["contract_sha256"] = population_sha256(result)
    return result


def _result(population: dict, reason: str, **details: object) -> dict:
    return {
        "protocol_revision": "source_voltage_delivered_load.v1",
        "kind": "voltage_reliability",
        "applicable": True,
        "score": None,
        "reason": reason,
        "population_contract_sha256": population.get("contract_sha256"),
        **details,
    }


def _indexed(rows: object, expected: set[int]) -> dict[int, dict] | None:
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) or type(row.get("index")) is not int for row in rows
    ):
        return None
    by_index = {row["index"]: row for row in rows}
    return (
        by_index if len(rows) == len(by_index) and set(by_index) == expected else None
    )


def measure_voltage_completion027(
    scenario: dict,
    *,
    source_contract: dict,
    population: dict,
    snapshot_inputs: dict,
    trace: list[dict],
) -> dict:
    """Require fixed-population voltage and native delivered-load evidence.

    Caller authenticates snapshots, trajectory and evidence IDs. Missing actual
    supply, full load coverage or convergence is N/A, never voltage-only F.
    Native violation IDs/counts are recorded from unrounded native results by
    the producer; model-visible rounded voltages do not redefine their limits.
    """
    body = {key: value for key, value in population.items() if key != "contract_sha256"}
    if population.get("contract_sha256") != population_sha256(body):
        raise ValueError("voltage_population_contract_hash_mismatch")
    if (
        population.get("schema_version") != "required_voltage_population.v1"
        or any(
            population.get(key) != scenario.get(key)
            for key in ("scenario_signature", "seed", "backend_kind")
        )
        or source_contract.get("scenario_sha256") != population.get("source_sha256")
        or any(
            source_contract.get(key) != scenario.get(key)
            for key in ("scenario_signature", "seed", "backend_kind")
        )
        or (source_contract.get("service") or {}).get("kind") != "voltage_reliability"
    ):
        raise ValueError("voltage_source_identity_mismatch")
    horizon = scenario["horizon_ticks"]
    native_ticks, observation_ticks = list(range(horizon)), list(range(1, horizon + 1))
    if (
        population.get("required_ticks") != observation_ticks
        or population.get("required_native_ticks") != native_ticks
    ):
        raise ValueError("voltage_source_tick_scope_mismatch")
    records = snapshot_inputs.get("backend_tick_records")
    if (
        not isinstance(records, list)
        or len(records) != horizon
        or not isinstance(trace, list)
        or len(trace) != horizon
        or any(not isinstance(row, dict) for row in records + trace)
        or any(type(row.get("tick")) is not int for row in records + trace)
        or [row.get("tick") for row in records] != native_ticks
        or [row.get("tick") for row in trace] != observation_ticks
        or any(
            not isinstance(row.get("observation"), dict)
            or type(row["observation"].get("tick")) is not int
            or row["observation"].get("tick") != row["tick"]
            for row in trace
        )
    ):
        return _result(population, "native_voltage_supply_window_incomplete")
    if scenario["backend_kind"] in {"opendss_ieee13", "opendss_fresh_feeders"}:
        return _measure_dss_voltage(population, records, horizon)
    inventory = population["native_source_population"]
    inventory_hash = population_sha256(inventory)
    if inventory_hash != population.get("native_source_population_sha256"):
        raise ValueError("voltage_source_inventory_hash_mismatch")
    source_buses = {row["index"]: row for row in inventory["buses"]}
    source_loads = {row["index"]: row for row in inventory["loads"]}
    required = {row["index"] for row in inventory["buses"] if row["in_service"]}
    if {f"bus_{index}" for index in required} != set(population["required_bus_ids"]):
        raise ValueError("voltage_source_required_population_mismatch")
    hours = float(scenario.get("tick_minutes", 60)) / 60
    if not _number(hours) or hours <= 0:
        return _result(population, "native_voltage_supply_tick_duration_invalid")
    demand, unserved, compliant = 0.0, 0.0, 0
    for record in records:
        meter = record.get("native_service_meter")
        if (
            not isinstance(meter, dict)
            or meter.get("schema_version") != "native_load_service.v1"
        ):
            return _result(population, "native_supply_meter_missing")
        if record.get("converged") is not True or meter.get("converged") is not True:
            return _result(population, "native_supply_convergence_unproven")
        if (
            meter.get("source_population_sha256") != inventory_hash
            or type(meter.get("tick")) is not int
            or meter.get("tick") != record["tick"]
            or not _number(meter.get("tick_hours"))
            or not math.isclose(meter["tick_hours"], hours, abs_tol=1e-12, rel_tol=0)
        ):
            return _result(population, "native_supply_meter_identity_invalid")
        buses = _indexed(meter.get("buses"), set(source_buses))
        loads = _indexed(meter.get("loads"), set(source_loads))
        if (
            buses is None
            or loads is None
            or meter.get("measurement_complete") is not True
        ):
            return _result(population, "native_supply_population_evidence_incomplete")
        known_unserved = set()
        for index in required:
            bus = buses[index]
            if bus.get("source_in_service") is not True:
                return _result(population, "native_voltage_source_status_mismatch")
            if bus.get("measurement_status") == "native_unserved" and bus.get(
                "native_proof"
            ) in {"native_topology_unsupplied", "native_bus_out_of_service"}:
                known_unserved.add(index)
            elif not (
                bus.get("measurement_status") == "native_voltage"
                and bus.get("native_proof") == "native_res_bus"
                and _number(bus.get("vm_pu"))
            ):
                return _result(population, "native_voltage_bus_state_unknown")
        violations = meter.get("native_voltage_violation_bus_indices")
        count = record.get("n_voltage_violations")
        if (
            not isinstance(violations, list)
            or any(
                type(index) is not int or index not in source_buses
                for index in violations
            )
            or len(violations) != len(set(violations))
            or type(count) is not int
            or count < 0
            or count > len(source_buses)
            or count != len(violations)
        ):
            return _result(population, "native_voltage_violation_count_unverified")
        requested_total, delivered_total, unserved_total = 0.0, 0.0, 0.0
        for index, source in source_loads.items():
            load = loads[index]
            request, delivery, shortfall = (
                load.get(key) for key in ("requested_mw", "delivered_mw", "unserved_mw")
            )
            if (
                load.get("bus") != source["bus"]
                or type(load.get("source_in_service")) is not bool
                or load["source_in_service"] != source["in_service"]
                or not all(_number(value) for value in (request, delivery, shortfall))
                or not math.isclose(
                    shortfall, max(0.0, request - delivery), abs_tol=1e-9, rel_tol=1e-9
                )
            ):
                return _result(population, "native_delivered_load_meter_invalid")
            status, proof = load.get("measurement_status"), load.get("native_proof")
            if not source["in_service"]:
                valid = (
                    status == "source_excluded"
                    and proof == "source_load_out_of_service"
                    and request == delivery == shortfall == 0
                )
            elif status == "native_res_load":
                valid = (
                    proof == "native_res_load"
                    and buses[source["bus"]].get("measurement_status")
                    == "native_voltage"
                )
            else:
                valid = (
                    status == "native_unserved"
                    and delivery == 0
                    and (
                        proof == "native_load_out_of_service"
                        or source["bus"] in known_unserved
                        and proof == buses[source["bus"]].get("native_proof")
                    )
                )
            if not valid:
                return _result(population, "native_delivered_load_state_unknown")
            requested_total += request
            delivered_total += delivery
            unserved_total += shortfall
        if any(
            not _number(meter.get(key))
            or not math.isclose(meter[key], value, abs_tol=1e-9, rel_tol=1e-9)
            for key, value in (
                ("requested_demand_mw", requested_total),
                ("delivered_demand_mw", delivered_total),
                ("unserved_demand_mw", unserved_total),
            )
        ):
            return _result(population, "native_supply_total_meter_conflict")
        compliant += len(required) - len((set(violations) & required) | known_unserved)
        demand += requested_total * hours
        unserved += unserved_total * hours
    if demand <= 0:
        return _result(population, "native_supply_denominator_zero")
    voltage = 100.0 * compliant / (len(required) * horizon)
    supplied = 100.0 * max(0.0, demand - unserved) / demand
    return _result(
        population,
        "source_fixed_voltage_and_native_delivered_supply",
        score=min(voltage, supplied),
        voltage_compliance=voltage,
        load_fulfillment=supplied,
        requested_energy_mwh=demand,
        unserved_energy_mwh=unserved,
        voltage_compliant_bus_time=compliant,
        voltage_required_bus_time=len(required) * horizon,
        aggregation="minimum_noncompensatory_required_services",
        strict_native_supply=True,
    )


def opendss_runtime_asset_lock() -> dict:
    """Portable identity of the installed DSS Python/native execution bytes."""
    from importlib.metadata import distribution
    from pathlib import Path
    import hashlib
    import dss

    packages = {}
    for name in ("dss-python", "dss-python-backend"):
        dist = distribution(name)
        hashes = {}
        for item in dist.files or []:
            if str(item).endswith((".py", ".so", ".dll", ".dylib", ".pyd")):
                hashes[str(item)] = hashlib.sha256(
                    Path(dist.locate_file(item)).read_bytes()
                ).hexdigest()
        if not hashes:
            raise ValueError("voltage_dss_runtime_package_bytes_missing")
        packages[name] = {"version": dist.version, "file_sha256s": hashes}
    return {"packages": packages, "native_version": str(dss.DSS.Version)}


def _compile_dss_source_population(
    scenario: dict,
    *,
    scenario_sha256: str,
    constructor_asset_lock: dict,
    constructor_lock_sha256: str,
) -> dict:
    """Construct the locked circuit once; never execute a tick or batch duty solve."""
    import hashlib
    from pathlib import Path
    from domains.power_grid.adapter import _rebuild_seed_from_dict
    from domains.power_grid.backends import opendss_fresh_feeders as fresh
    from domains.power_grid.backends.opendss_ieee13 import (
        _resolve_native_include_graph,
    )

    expected_runtime = constructor_asset_lock.get("dss_runtime")
    if not isinstance(expected_runtime, dict) or not expected_runtime:
        raise ValueError("voltage_dss_runtime_lock_missing")
    actual_runtime = opendss_runtime_asset_lock()
    if expected_runtime != actual_runtime:
        raise ValueError("voltage_dss_installed_bytes_mismatch")
    hashes = (scenario.get("source_contract") or {}).get("file_sha256s")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("voltage_dss_source_file_lock_missing")
    # Verify every supplied byte before native construction, including a selected
    # duty program/data file that is deliberately not redirected in its entirety.
    resolved_locks = {}
    for path, expected in hashes.items():
        actual = Path(path)
        if not actual.is_absolute():
            actual = fresh.REPO_ROOT / actual
        if (
            not _sha256(str(expected).removeprefix("sha256:"))
            or not actual.is_file()
            or hashlib.sha256(actual.read_bytes()).hexdigest()
            != str(expected).removeprefix("sha256:")
        ):
            raise ValueError("voltage_dss_source_file_bytes_mismatch")
        resolved_locks[actual.resolve()] = (
            str(path),
            str(expected).removeprefix("sha256:"),
        )

    config = scenario.get("backend_config") or {}
    if config.get("source_profile"):
        raise ValueError("voltage_dss_source_profile_scope_unsupported")
    feeder = config.get("feeder", "ieee13")
    master_name = config.get("master_file") or (
        "13Bus/IEEE13Nodeckt.dss"
        if scenario["backend_kind"] == "opendss_ieee13"
        else fresh.FEEDER_MASTER_FILES.get(feeder)
    )
    root = Path(config.get("source_root") or fresh.DEFAULT_SOURCE_ROOT)
    if not root.is_absolute():
        root = fresh.REPO_ROOT / root
    master = fresh._resolve_master_file(root, master_name)
    assets, _ = _resolve_native_include_graph(master)
    for asset in assets:
        locked = resolved_locks.get(Path(asset["path"]).resolve())
        if locked is None or locked[1] != asset["sha256"]:
            raise ValueError("voltage_dss_runtime_graph_not_source_locked")
    duty = config.get("native_duty_program") or {}
    for key in ("scenario_file", "data_file"):
        if key in duty and str(duty[key]) not in hashes:
            raise ValueError("voltage_dss_selected_program_not_source_locked")

    source_calc_commands = 0

    def count_source_initializations(path):
        nonlocal source_calc_commands
        for line in (
            Path(path).read_text(encoding="utf-8", errors="strict").splitlines()
        ):
            command = fresh._strip_dss_comment(line)
            name = fresh._dss_command(command)
            raw_name = command.split(maxsplit=1)[0].lower() if command.strip() else ""
            if len(raw_name) >= 5 and "calcvoltagebases".startswith(raw_name):
                source_calc_commands += 1
            if name in {"compile", "redirect"}:
                match = fresh._DSS_INCLUDE_ARGUMENT_RE.match(command)
                if match is None:
                    raise ValueError("voltage_dss_source_include_unparseable")
                raw = next(value for value in match.groups() if value)
                count_source_initializations(
                    (Path(path).parent / raw.replace("\\", "/")).resolve()
                )

    count_source_initializations(master)

    class SourceBackend(fresh.OpenDssFreshFeedersBackend):
        # Count all Python solve() entries, including configured startup controls.
        source_api_solver_calls = 0

        def solve(self):
            self.source_api_solver_calls += 1
            return super().solve()

    backend = SourceBackend(feeder=feeder, source_root=root, master_file=master_name)
    try:
        seed = _rebuild_seed_from_dict(scenario, override_seed=scenario["seed"])
        # Use the existing isolated source interpreter for both DSS families.
        # IEEE13's node population is identical; no episode actions/ticks run.
        backend.reset(seed)
        circuit = backend._require_circuit()
        nodes = [str(value) for value in circuit.AllNodeNames]
        if (
            not nodes
            or len(nodes) != len(set(nodes))
            or len(nodes) != int(circuit.NumNodes)
        ):
            raise ValueError("voltage_dss_source_node_population_invalid")
        policy = backend._runtime_execution_policy
        source_solve_commands = int(policy["preserved_solve_count"]) + int(
            policy["transformed_solve_count"]
        )
        profile_calls = int(
            bool(
                config.get("native_duty_program") or config.get("native_yearly_program")
            )
        )
        calls = (
            backend.source_api_solver_calls
            + source_solve_commands
            + source_calc_commands
            + profile_calls
        )
        source_assets = [
            {
                "path": resolved_locks[Path(row["path"]).resolve()][0],
                "sha256": row["sha256"],
                "role": row["role"],
            }
            for row in backend._runtime_source_assets
        ]
        result = {
            "schema_version": "required_voltage_population.v1",
            "source_sha256": scenario_sha256,
            "scenario_signature": scenario["scenario_signature"],
            "seed": scenario["seed"],
            "backend_kind": scenario["backend_kind"],
            "network": feeder,
            "required_node_ids": nodes,
            "excluded_node_ids": {},
            "required_ticks": list(range(1, scenario["horizon_ticks"] + 1)),
            "required_native_ticks": list(range(scenario["horizon_ticks"])),
            "tick_alignment": "post_observation_tick_equals_native_tick_plus_one",
            "native_source_population": {"node_ids": nodes},
            "native_source_population_sha256": population_sha256({"node_ids": nodes}),
            "constructor_runtime_asset": actual_runtime,
            "constructor_lock_sha256": constructor_lock_sha256,
            "source_population_policy": "all_locked_source_native_nodes_including_zero_voltage",
            "source_initialization": {
                "source_solver_calls": calls,
                "python_api_solver_calls": backend.source_api_solver_calls,
                "source_solve_commands": source_solve_commands,
                "source_calc_voltage_bases_commands": source_calc_commands,
                "selected_program_initial_snapshot_calls": profile_calls,
                "count_scope": "explicit_solver_entry_commands_excluding_internal_iterations",
                "native_episode_ticks": 0,
                "whole_duty_program_executed": False,
                "runtime_source_assets": source_assets,
            },
            "supply_measurement": "not_measured_by_source_service_contract",
        }
        result["contract_sha256"] = population_sha256(result)
        return result
    finally:
        backend.close()


def _measure_dss_voltage(population: dict, records: list[dict], horizon: int) -> dict:
    inventory = population.get("native_source_population") or {}
    nodes = inventory.get("node_ids")
    inventory_hash = population_sha256(inventory)
    if (
        not isinstance(nodes, list)
        or not nodes
        or len(nodes) != len(set(nodes))
        or nodes != population.get("required_node_ids")
        or inventory_hash != population.get("native_source_population_sha256")
    ):
        raise ValueError("voltage_source_required_population_mismatch")
    compliant = 0
    for record in records:
        meter = record.get("native_node_meter")
        if (
            not isinstance(meter, dict)
            or meter.get("schema_version") != "native_node_voltage.v1"
        ):
            return _result(population, "native_node_voltage_meter_missing")
        if record.get("converged") is not True or meter.get("converged") is not True:
            return _result(population, "native_node_voltage_convergence_unproven")
        if (
            type(meter.get("tick")) is not int
            or meter["tick"] != record["tick"]
            or meter.get("source_population_sha256") != inventory_hash
        ):
            return _result(population, "native_node_voltage_meter_identity_invalid")
        rows = meter.get("nodes")
        if (
            not isinstance(rows, list)
            or any(not isinstance(row, dict) for row in rows)
            or [row.get("node_id") for row in rows] != nodes
            or type(meter.get("native_node_count")) is not int
            or meter["native_node_count"] != len(nodes)
            or meter.get("measurement_complete") is not True
            or any(not _number(row.get("vm_pu")) for row in rows)
            or meter.get("nonfinite_voltage_node_ids") != []
        ):
            return _result(
                population, "native_node_voltage_population_evidence_incomplete"
            )
        violations = [
            row["node_id"] for row in rows if row["vm_pu"] < 0.95 or row["vm_pu"] > 1.05
        ]
        zeros = [row["node_id"] for row in rows if row["vm_pu"] == 0]
        if (
            meter.get("native_voltage_violation_node_ids") != violations
            or meter.get("zero_voltage_node_ids") != zeros
            or type(record.get("n_voltage_violations")) is not int
            or record["n_voltage_violations"] != len(violations)
        ):
            return _result(population, "native_node_voltage_violation_count_unverified")
        compliant += len(nodes) - len(violations)
    return _result(
        population,
        "source_fixed_native_node_time_voltage",
        score=100 * compliant / (len(nodes) * horizon),
        voltage_compliant_node_time=compliant,
        voltage_required_node_time=len(nodes) * horizon,
        aggregation="all_source_node_time_voltage_compliance",
        supply_measurement="not_measured_by_source_service_contract",
    )
