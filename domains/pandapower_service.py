"""Private native supply meters shared by the pandapower domain backends.

These functions read source tables and solved native results. They never run
a solver, change native state, or infer an outage from a nonfinite voltage.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def population_sha256(population: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            population, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def capture_source_population(net: Any) -> dict:
    """Freeze native bus/load indices and source status before control."""
    return {
        "schema_version": "pandapower_source_population.v1",
        "buses": [
            {"index": int(index), "in_service": bool(row.in_service)}
            for index, row in net.bus.iterrows()
        ],
        "loads": [
            {
                "index": int(index),
                "bus": int(row.bus),
                "in_service": bool(row.in_service),
                "scaling": float(row.scaling),
            }
            for index, row in net.load.iterrows()
        ],
    }


def _number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def build_native_service_meter(
    net: Any,
    *,
    source_population: dict,
    requested_mw: dict[int, float],
    tick: int,
    tick_hours: float,
    converged: bool,
    voltage_lower_pu: float,
    voltage_upper_pu: float,
) -> dict:
    """Meter every source load with independent native outage/solver proof.

    ``requested_mw`` is the exogenous source request before any model shedding,
    with source scaling applied. A failed solve yields unknown delivery, even
    if stale finite ``res_load`` values remain. A disconnected bus is known
    unserved only from native topology or an explicit native in-service flag.
    """
    import pandapower.topology as topology

    source_load_ids = {row["index"] for row in source_population["loads"]}
    source_bus_ids = {row["index"] for row in source_population["buses"]}
    if set(requested_mw) != source_load_ids:
        raise ValueError("native_service_source_request_population_mismatch")
    if (
        set(map(int, net.load.index)) != source_load_ids
        or set(map(int, net.bus.index)) != source_bus_ids
    ):
        raise ValueError("native_service_source_population_changed")
    unsupplied = set(map(int, topology.unsupplied_buses(net)))
    bus_rows, violations = [], []
    for source in source_population["buses"]:
        index = source["index"]
        vm = (
            float(net.res_bus.at[index, "vm_pu"])
            if converged and index in net.res_bus.index
            else None
        )
        if (
            converged
            and _number(vm)
            and (vm < voltage_lower_pu or vm > voltage_upper_pu)
        ):
            violations.append(index)
        if not converged:
            status, proof = "unknown_solver", "native_solver_not_converged"
            vm = None
        elif not bool(net.bus.at[index, "in_service"]):
            status, proof = "native_unserved", "native_bus_out_of_service"
        elif index in unsupplied:
            status, proof = "native_unserved", "native_topology_unsupplied"
        elif _number(vm):
            status, proof = "native_voltage", "native_res_bus"
        else:
            status, proof = (
                "unknown_native_result",
                "native_voltage_missing_or_nonfinite",
            )
        bus_rows.append(
            {
                "index": index,
                "source_in_service": source["in_service"],
                "vm_pu": vm if _number(vm) else None,
                "measurement_status": status,
                "native_proof": proof,
            }
        )
    by_bus = {row["index"]: row for row in bus_rows}
    load_rows = []
    for source in source_population["loads"]:
        index, bus = source["index"], source["bus"]
        if int(net.load.at[index, "bus"]) != bus:
            raise ValueError("native_service_source_load_bus_changed")
        request = requested_mw[index] if source["in_service"] else 0.0
        if not _number(request):
            raise ValueError("native_service_source_request_invalid")
        delivered, unserved = None, None
        if not source["in_service"]:
            status, proof = "source_excluded", "source_load_out_of_service"
            delivered, unserved = 0.0, 0.0
        elif not converged:
            status, proof = "unknown_solver", "native_solver_not_converged"
        elif not bool(net.load.at[index, "in_service"]):
            status, proof = "native_unserved", "native_load_out_of_service"
            delivered, unserved = 0.0, request
        elif by_bus[bus]["measurement_status"] == "native_unserved":
            status, proof = "native_unserved", by_bus[bus]["native_proof"]
            delivered, unserved = 0.0, request
        elif by_bus[bus]["measurement_status"] == "native_voltage":
            value = (
                float(net.res_load.at[index, "p_mw"])
                if index in net.res_load.index
                else None
            )
            if _number(value):
                status, proof = "native_res_load", "native_res_load"
                delivered, unserved = value, max(0.0, request - value)
            else:
                status, proof = (
                    "unknown_native_result",
                    "native_load_result_missing_or_nonfinite",
                )
        else:
            status, proof = "unknown_native_result", "native_bus_supply_unverified"
        load_rows.append(
            {
                "index": index,
                "bus": bus,
                "source_in_service": source["in_service"],
                "requested_mw": request,
                "delivered_mw": delivered,
                "unserved_mw": unserved,
                "measurement_status": status,
                "native_proof": proof,
            }
        )
    complete = all(row["delivered_mw"] is not None for row in load_rows)
    return {
        "schema_version": "native_load_service.v1",
        "tick": tick,
        "tick_hours": tick_hours,
        "converged": converged,
        "source_population_sha256": population_sha256(source_population),
        "measurement_complete": complete,
        "loads": load_rows,
        "buses": bus_rows,
        "native_voltage_violation_bus_indices": violations if converged else None,
        "requested_demand_mw": math.fsum(row["requested_mw"] for row in load_rows),
        "delivered_demand_mw": math.fsum(row["delivered_mw"] for row in load_rows)
        if complete
        else None,
        "unserved_demand_mw": math.fsum(row["unserved_mw"] for row in load_rows)
        if complete
        else None,
    }
