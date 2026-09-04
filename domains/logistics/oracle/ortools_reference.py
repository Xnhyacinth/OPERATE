"""
domains.logistics.oracle.ortools_reference — Offline cached routing oracle.

Computes a deterministic bounded routing reference and caches it into the
seed's ``backend_config['reference_optimum']``. This is the ``optimality_gap``
reference and the headroom-gate upper bound (spec §8/§9).

The original OR-Tools guided local search used a solution-count limit without
a search-work limit. Feasible instances could therefore run until the outer
wall-clock timeout. Formal replay now uses a bounded capacity partition followed
by nearest-neighbour plus single-pass 2-opt on every host. It does not vary with
optional dependencies and is content-cached across repeated episode and
counterfactual resets. The module name and availability flag remain for
compatibility; OR-Tools is not used by the formal reference path.
"""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from ..seeds.schema import LogisticsScenarioSeed

ORTOOLS_AVAILABLE = importlib.util.find_spec("ortools") is not None
_MAX_PARTITION_STATES = 100_000


@dataclass(frozen=True)
class _ReferenceSolution:
    cost: float | None
    method: str
    routes: tuple[tuple[int, ...], ...]
    route_loads: tuple[int, ...]
    feasible: bool | None
    comparable: bool
    reason_code: str | None
    partition_states: int


def _network(seed_obj: LogisticsScenarioSeed) -> dict[str, Any]:
    return dict(seed_obj.backend_config.get("network", {}) or {})


def _nodes(
    seed_obj: LogisticsScenarioSeed,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Return integer-rounded (coords, demands) with depot at index 0."""
    net = _network(seed_obj)
    depot = net.get("depot", {"x": 0.0, "y": 0.0})
    coords: list[tuple[int, int]] = [
        (int(round(float(depot["x"]))), int(round(float(depot["y"]))))
    ]
    demands: list[int] = [0]
    for c in net.get("customers", []):
        coords.append((int(round(float(c["x"]))), int(round(float(c["y"])))))
        demands.append(int(max(0, round(float(c["demand"])))))
    return coords, demands


def _distance_matrix(coords: list[tuple[int, int]]) -> list[list[int]]:
    n = len(coords)
    return [
        [
            int(
                round(
                    math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
                )
            )
            for j in range(n)
        ]
        for i in range(n)
    ]


def compute_reference_optimum(
    seed_obj: LogisticsScenarioSeed, *, cache: bool = True
) -> dict[str, Any]:
    """Compute + (optionally) cache the deterministic routing reference optimum.

    Returns a JSON-safe dict ``{reference_optimum, method, num_vehicles,
    deterministic_stop}``. Two calls with the same seed return identical
    results (replay-stable).
    """
    coords, demands = _nodes(seed_obj)
    net = _network(seed_obj)
    n_vehicles = max(1, int(net.get("n_vehicles", 2) or 2))
    capacity = int(max(1, round(float(net.get("capacity", 0.0) or 0.0))))
    solution = _cached_bounded_reference(
        tuple(coords), tuple(demands), n_vehicles, capacity
    )

    result = {
        "reference_optimum": (
            float(solution.cost) if solution.cost is not None else None
        ),
        "objective_component": "routing_operating_cost",
        "method": solution.method,
        "num_vehicles": n_vehicles,
        "capacity": capacity,
        "n_nodes": len(coords),
        "routes": [list(route) for route in solution.routes],
        "route_loads": list(solution.route_loads),
        "feasible": solution.feasible,
        "comparable": solution.comparable,
        "reason_code": solution.reason_code,
        "deterministic_stop": {
            "algorithm": "bounded_capacity_partition_nn_single_pass_2opt",
            "work_bound": "partition_state_limit_then_polynomial_route_ordering",
            "max_partition_states": _MAX_PARTITION_STATES,
            "partition_states": solution.partition_states,
            "max_2opt_passes": 1,
            "time_limit": None,
        },
    }
    if cache:
        seed_obj.backend_config["reference_optimum"] = dict(result)
    return result


@lru_cache(maxsize=None)
def _cached_bounded_reference(
    coords: tuple[tuple[int, int], ...],
    demands: tuple[int, ...],
    n_vehicles: int,
    capacity: int,
) -> _ReferenceSolution:
    matrix = _distance_matrix(list(coords))
    return _solve_greedy(matrix, list(demands), n_vehicles, capacity)


def _solve_greedy(
    matrix: list[list[int]], demands: list[int], n_vehicles: int, capacity: int
) -> _ReferenceSolution:
    """Deterministic capacity partition + nearest-neighbour route ordering.

    Best-fit decreasing handles easy instances. A memoized search with a hard
    state limit handles tight feasible partitions without ever emitting an
    overloaded route. Inputs proven infeasible, or not resolved within the
    declared bound, remain explicitly non-comparable.
    """
    customers = list(range(1, len(matrix)))
    total_demand = sum(demands[c] for c in customers)
    if total_demand > n_vehicles * capacity:
        return _non_comparable_reference("fleet_capacity_insufficient")
    if any(demands[c] > capacity for c in customers):
        return _non_comparable_reference("customer_demand_exceeds_vehicle_capacity")

    routes = _best_fit_decreasing(customers, demands, n_vehicles, capacity)
    partition_states = 0
    reason_code: str | None = None
    if routes is None:
        routes, partition_states, search_exhausted = _bounded_capacity_partition(
            customers,
            demands,
            n_vehicles,
            capacity,
        )
        if routes is None:
            reason_code = (
                "capacity_partition_search_limit_exceeded"
                if search_exhausted
                else "capacity_partition_infeasible"
            )
            return _non_comparable_reference(
                reason_code,
                feasible=None if search_exhausted else False,
                partition_states=partition_states,
            )

    total = 0.0
    ordered_routes: list[tuple[int, ...]] = []
    for route in routes:
        ordered = _nn_order(route, matrix)
        ordered = _two_opt(ordered, matrix)
        ordered_routes.append(tuple(ordered))
        total += _route_cost(ordered, matrix)
    route_loads = tuple(sum(demands[c] for c in route) for route in ordered_routes)
    return _ReferenceSolution(
        cost=float(int(round(total))),
        method="bounded_nn_2opt",
        routes=tuple(ordered_routes),
        route_loads=route_loads,
        feasible=True,
        comparable=True,
        reason_code=reason_code,
        partition_states=partition_states,
    )


def _non_comparable_reference(
    reason_code: str,
    *,
    feasible: bool | None = False,
    partition_states: int = 0,
) -> _ReferenceSolution:
    return _ReferenceSolution(
        cost=None,
        method="infeasible_capacity" if feasible is False else "non_comparable",
        routes=(),
        route_loads=(),
        feasible=feasible,
        comparable=False,
        reason_code=reason_code,
        partition_states=partition_states,
    )


def _best_fit_decreasing(
    customers: list[int],
    demands: list[int],
    n_vehicles: int,
    capacity: int,
) -> list[list[int]] | None:
    routes: list[list[int]] = [[] for _ in range(n_vehicles)]
    loads = [0] * n_vehicles
    for customer in sorted(customers, key=lambda item: (-demands[item], item)):
        demand = demands[customer]
        fitting = [
            vehicle
            for vehicle in range(n_vehicles)
            if loads[vehicle] + demand <= capacity
        ]
        if not fitting:
            return None
        vehicle = min(
            fitting,
            key=lambda item: (capacity - loads[item] - demand, item),
        )
        routes[vehicle].append(customer)
        loads[vehicle] += demand
    return routes


def _bounded_capacity_partition(
    customers: list[int],
    demands: list[int],
    n_vehicles: int,
    capacity: int,
) -> tuple[list[list[int]] | None, int, bool]:
    ordered = sorted(customers, key=lambda item: (-demands[item], item))
    remaining = [capacity] * n_vehicles
    routes: list[list[int]] = [[] for _ in range(n_vehicles)]
    rejected_states: set[tuple[int, tuple[int, ...]]] = set()
    states = 0
    search_exhausted = False

    def assign(position: int) -> bool:
        nonlocal search_exhausted, states
        if position == len(ordered):
            return True
        if states >= _MAX_PARTITION_STATES:
            search_exhausted = True
            return False
        states += 1
        state = (position, tuple(sorted(remaining, reverse=True)))
        if state in rejected_states:
            return False

        customer = ordered[position]
        demand = demands[customer]
        tried_remaining: set[int] = set()
        for vehicle in sorted(
            range(n_vehicles),
            key=lambda item: (remaining[item] - demand, item),
        ):
            if remaining[vehicle] < demand:
                continue
            if remaining[vehicle] in tried_remaining:
                continue
            tried_remaining.add(remaining[vehicle])
            remaining[vehicle] -= demand
            routes[vehicle].append(customer)
            if assign(position + 1):
                return True
            routes[vehicle].pop()
            remaining[vehicle] += demand
        rejected_states.add(state)
        return False

    if assign(0):
        return routes, states, False
    return None, states, search_exhausted


def _nn_order(route: list[int], matrix: list[list[int]]) -> list[int]:
    if not route:
        return []
    remaining = list(route)
    ordered: list[int] = []
    cur = 0  # depot
    while remaining:
        nxt = min(remaining, key=lambda c: (matrix[cur][c], c))
        ordered.append(nxt)
        cur = nxt
        remaining.remove(nxt)
    return ordered


def _route_cost(order: list[int], matrix: list[list[int]]) -> int:
    if not order:
        return 0
    cost = matrix[0][order[0]]
    for a, b in zip(order, order[1:], strict=False):
        cost += matrix[a][b]
    cost += matrix[order[-1]][0]
    return cost


def _two_opt(order: list[int], matrix: list[list[int]]) -> list[int]:
    if len(order) < 4:
        return order
    best = list(order)
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                candidate = best[:i] + best[i : j + 1][::-1] + best[j + 1 :]
                if _route_cost(candidate, matrix) < _route_cost(best, matrix):
                    best = candidate
                    improved = True
        # Single pass guard to keep it deterministic + bounded.
        break
    return best
