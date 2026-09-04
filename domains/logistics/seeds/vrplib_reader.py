"""
domains.logistics.seeds.vrplib_reader — VRPLIB instance I/O.

Normalizes a parsed VRPLIB ``.vrp`` instance to a plain-Python dict:

    {
        "name": str,
        "type": "CVRP" | "VRPTW" | ...,
        "capacity": int,
        "n_vehicles": int,
        "service_time": float,
        "depot_index": int,                      # 0-based index into nodes
        "nodes": [{"x": float, "y": float,
                   "demand": float,
                   "tw_early": float, "tw_late": float}],  # node 0 = depot
    }

Reader precedence:

1. The MIT ``vrplib`` package (``vrplib.read_instance``) when installed
   (returns numpy arrays which we coerce to floats).
2. A minimal **pure-Python** VRPLIB parser (the ``.vrp`` format is simple:
   a header of ``KEY : VALUE`` lines followed by ``*_SECTION`` blocks).
   This keeps the parser runnable on a host where ``vrplib`` is absent
   (spec §T1 graceful-skip).

Plus a handful of small **embedded synthetic** instances so the seed
builders, structural-std test, and reproducibility test run fully offline
even when no real instance is anchored under ``works/``. Embedded
instances are clearly labeled ``synthetic`` in their name and are never
released — Stage-1 ships 0 scenarios.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised only when vrplib is installed
    import vrplib as _vrplib  # type: ignore[import]

    VRPLIB_AVAILABLE = True
except Exception:  # pragma: no cover
    _vrplib = None  # type: ignore[assignment]
    VRPLIB_AVAILABLE = False


# Repo-root-relative anchors for source-locked instance sets.
REPO_ROOT = Path(__file__).resolve().parents[3]
WORKS_ROOT = REPO_ROOT / "works" / "PyVRP-Instances"
VRPLIB_PACKAGE_TEST_DATA_ROOT = REPO_ROOT / "works" / "VRPLIB" / "tests" / "data"


# ─────────────────────────────────────────────────────────────────────────────
# Embedded synthetic fallback instances (offline-runnable; never released)
# ─────────────────────────────────────────────────────────────────────────────


def _synthetic_instance(
    name: str, n_customers: int, *, with_tw: bool
) -> dict[str, Any]:
    """Deterministic synthetic VRPLIB-shaped instance keyed by ``name``.

    Used only when no real anchored instance is available (offline tests).
    Coordinates / demands are a fixed deterministic function of ``name`` so
    the seed signature is stable across runs.
    """
    h = _name_hash(name)
    nodes: list[dict[str, float]] = []
    # Depot at the centroid.
    nodes.append(
        {"x": 50.0, "y": 50.0, "demand": 0.0, "tw_early": 0.0, "tw_late": 1000.0}
    )
    for i in range(1, n_customers + 1):
        ang = (h + i * 137) % 360
        rad = 10.0 + ((h + i * 53) % 30)
        x = round(50.0 + rad * math.cos(math.radians(ang)), 3)
        y = round(50.0 + rad * math.sin(math.radians(ang)), 3)
        demand = float(1 + ((h + i * 7) % 9))
        if with_tw:
            early = float((h + i * 11) % 40)
            late = early + 40.0 + float((h + i * 3) % 40)
        else:
            early, late = 0.0, 1000.0
        nodes.append(
            {"x": x, "y": y, "demand": demand, "tw_early": early, "tw_late": late}
        )
    cap = max(20, int(sum(n["demand"] for n in nodes) / 3))
    return {
        "name": name,
        "type": "VRPTW" if with_tw else "CVRP",
        "capacity": cap,
        "n_vehicles": 4,
        "service_time": 10.0 if with_tw else 0.0,
        "depot_index": 0,
        "nodes": nodes,
        "synthetic": True,
    }


def _name_hash(name: str) -> int:
    import hashlib

    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")


# Canonical embedded instances (names mirror the spec's documented examples
# so ``build_*_seed(instance="A-n32-k5")`` works offline as a stand-in when
# the real Augerat set is not in the MIT mirror).
_EMBEDDED: dict[str, dict[str, Any]] = {
    "A-n32-k5": _synthetic_instance("A-n32-k5", 31, with_tw=False),
    "synthetic-cvrp-12": _synthetic_instance("synthetic-cvrp-12", 12, with_tw=False),
    "C101": _synthetic_instance("C101", 25, with_tw=True),
    "synthetic-vrptw-12": _synthetic_instance("synthetic-vrptw-12", 12, with_tw=True),
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def resolve_instance_path(
    instance: str,
    subdir: str,
    *,
    source_id: str = "vrplib",
) -> Path | None:
    """Resolve ``<instance>.vrp`` for a supported VRPLIB source."""
    if source_id == "vrplib":
        candidate = WORKS_ROOT / subdir / f"{instance}.vrp"
    elif source_id == "vrplib_package_test_data":
        candidate = VRPLIB_PACKAGE_TEST_DATA_ROOT / f"{instance}.vrp"
    elif source_id == "vrplib_package_solomon":
        candidate = (
            VRPLIB_PACKAGE_TEST_DATA_ROOT
            / "cvrplib"
            / "Vrp-Set-Solomon"
            / f"{instance}.txt"
        )
    elif source_id == "vrplib_package_x_set":
        candidate = (
            VRPLIB_PACKAGE_TEST_DATA_ROOT
            / "cvrplib"
            / "Vrp-Set-X"
            / "X"
            / f"{instance}.vrp"
        )
    elif source_id == "vrplib_package_lkh_cvrptw":
        candidate = (
            VRPLIB_PACKAGE_TEST_DATA_ROOT
            / "lkh-3"
            / "CVRPTW"
            / "INSTANCES"
            / f"{instance}.vrptw"
        )
    elif source_id == "vrplib_package_lkh_cvrp":
        candidate = (
            VRPLIB_PACKAGE_TEST_DATA_ROOT
            / "lkh-3"
            / "CVRP"
            / "INSTANCES"
            / f"{instance}.vrp"
        )
    elif source_id == "vrplib_package_cvrplib_root":
        candidate = (
            VRPLIB_PACKAGE_TEST_DATA_ROOT
            / "cvrplib"
            / f"{instance}.vrp"
        )
    else:
        raise KeyError(f"unknown VRPLIB source_id: {source_id!r}")
    return candidate if candidate.exists() else None


def provenance_file_for_instance(
    instance: str,
    subdir: str,
    *,
    source_id: str = "vrplib",
) -> str:
    if source_id == "vrplib":
        return f"works/PyVRP-Instances/{subdir}/{instance}.vrp"
    if source_id == "vrplib_package_test_data":
        return f"works/VRPLIB/tests/data/{instance}.vrp"
    if source_id == "vrplib_package_solomon":
        return f"works/VRPLIB/tests/data/cvrplib/Vrp-Set-Solomon/{instance}.txt"
    if source_id == "vrplib_package_x_set":
        return f"works/VRPLIB/tests/data/cvrplib/Vrp-Set-X/X/{instance}.vrp"
    if source_id == "vrplib_package_lkh_cvrptw":
        return (
            "works/VRPLIB/tests/data/lkh-3/CVRPTW/INSTANCES/"
            f"{instance}.vrptw"
        )
    if source_id == "vrplib_package_lkh_cvrp":
        return (
            "works/VRPLIB/tests/data/lkh-3/CVRP/INSTANCES/"
            f"{instance}.vrp"
        )
    if source_id == "vrplib_package_cvrplib_root":
        return f"works/VRPLIB/tests/data/cvrplib/{instance}.vrp"
    raise KeyError(f"unknown VRPLIB source_id: {source_id!r}")


def read_instance(path: str | Path) -> dict[str, Any]:
    """Read a VRPLIB ``.vrp`` file into the normalized dict.

    Prefers the ``vrplib`` package; falls back to the pure-Python parser.
    """
    path = Path(path)
    if path.suffix.lower() == ".txt":
        return _read_solomon_txt(path)
    if VRPLIB_AVAILABLE:
        try:
            return _normalize_vrplib(_vrplib.read_instance(str(path)))  # type: ignore[union-attr]
        except Exception:
            # Fall through to the pure-Python parser on any vrplib hiccup.
            pass
    return _read_vrplib_pure(path)


def load_instance(
    instance: str,
    subdir: str,
    *,
    source_id: str = "vrplib",
) -> dict[str, Any]:
    """Load ``instance`` by name: real anchored file first, else embedded.

    Returns the normalized dict with an extra ``anchored`` boolean flag.
    """
    p = resolve_instance_path(instance, subdir, source_id=source_id)
    if p is not None:
        out = read_instance(p)
        out["anchored"] = True
        out.setdefault("name", instance)
        return out
    if source_id != "vrplib":
        raise FileNotFoundError(
            f"instance {instance!r} not anchored under {provenance_file_for_instance(instance, subdir, source_id=source_id)}"
        )
    if instance in _EMBEDDED:
        out = dict(_EMBEDDED[instance])
        out["nodes"] = [dict(n) for n in out["nodes"]]
        out["anchored"] = False
        return out
    raise FileNotFoundError(
        f"instance {instance!r} not anchored under works/PyVRP-Instances/{subdir} "
        f"and no embedded synthetic fallback exists"
    )


def instance_is_anchored(
    instance: str,
    subdir: str,
    *,
    source_id: str = "vrplib",
) -> bool:
    return resolve_instance_path(instance, subdir, source_id=source_id) is not None


# ─────────────────────────────────────────────────────────────────────────────
# Normalization + pure-Python parser
# ─────────────────────────────────────────────────────────────────────────────


def _normalize_vrplib(inst: dict[str, Any]) -> dict[str, Any]:
    """Coerce a ``vrplib.read_instance`` dict (numpy arrays) to plain Python."""
    coords = inst.get("node_coord")
    demand = inst.get("demand")
    tw = inst.get("time_window")
    svc = inst.get("service_time")
    depot = inst.get("depot")
    n = len(coords)
    nodes: list[dict[str, float]] = []
    for i in range(n):
        x = float(coords[i][0])
        y = float(coords[i][1])
        d = float(demand[i]) if demand is not None else 0.0
        if tw is not None:
            early = float(tw[i][0])
            late = float(tw[i][1])
        else:
            early, late = 0.0, 1.0e9
        nodes.append({"x": x, "y": y, "demand": d, "tw_early": early, "tw_late": late})
    depot_index = 0
    try:
        if depot is not None:
            depot_index = int(depot[0])
    except Exception:
        depot_index = 0
    service_time = 0.0
    try:
        if svc is not None:
            service_time = float(svc[0]) if hasattr(svc, "__len__") else float(svc)
    except Exception:
        service_time = 0.0
    return {
        "name": str(inst.get("name", "")),
        "type": str(inst.get("type", "CVRP")),
        "capacity": int(inst.get("capacity", 0) or 0),
        "n_vehicles": int(inst.get("vehicles", 0) or 0),
        "service_time": service_time,
        "depot_index": depot_index,
        "nodes": nodes,
        "synthetic": False,
    }


def _read_vrplib_pure(path: Path) -> dict[str, Any]:
    """Minimal pure-Python VRPLIB ``.vrp`` parser (no numpy / no vrplib).

    Supports the header keys and sections used by CVRP / VRPTW instances:
    ``NAME``, ``TYPE``, ``DIMENSION``, ``CAPACITY``, ``VEHICLES``,
    ``SERVICE_TIME``, ``NODE_COORD_SECTION``, ``DEMAND_SECTION``,
    ``TIME_WINDOW_SECTION``, ``DEPOT_SECTION``.
    """
    header: dict[str, str] = {}
    coords: dict[int, tuple[float, float]] = {}
    demands: dict[int, float] = {}
    tws: dict[int, tuple[float, float]] = {}
    depot_ids: list[int] = []
    section: str | None = None

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            upper = line.upper()
            if upper in {"EOF"}:
                break
            if upper.endswith("_SECTION"):
                section = upper
                continue
            if ":" in line and section is None:
                key, _, val = line.partition(":")
                header[key.strip().upper()] = val.strip()
                continue
            parts = line.split()
            if section == "NODE_COORD_SECTION" and len(parts) >= 3:
                nid = int(float(parts[0]))
                coords[nid] = (float(parts[1]), float(parts[2]))
            elif section == "DEMAND_SECTION" and len(parts) >= 2:
                demands[int(float(parts[0]))] = float(parts[1])
            elif section == "TIME_WINDOW_SECTION" and len(parts) >= 3:
                tws[int(float(parts[0]))] = (float(parts[1]), float(parts[2]))
            elif section == "DEPOT_SECTION" and parts:
                v = int(float(parts[0]))
                if v >= 0:
                    depot_ids.append(v)

    ids = sorted(coords.keys())
    nodes: list[dict[str, float]] = []
    for nid in ids:
        x, y = coords[nid]
        early, late = tws.get(nid, (0.0, 1.0e9))
        nodes.append(
            {
                "x": x,
                "y": y,
                "demand": float(demands.get(nid, 0.0)),
                "tw_early": early,
                "tw_late": late,
            }
        )
    depot_index = 0
    if depot_ids and depot_ids[0] in ids:
        depot_index = ids.index(depot_ids[0])
    return {
        "name": header.get("NAME", path.stem),
        "type": header.get("TYPE", "CVRP"),
        "capacity": int(float(header.get("CAPACITY", "0") or 0)),
        "n_vehicles": int(float(header.get("VEHICLES", "0") or 0)),
        "service_time": float(header.get("SERVICE_TIME", "0") or 0),
        "depot_index": depot_index,
        "nodes": nodes,
        "synthetic": False,
    }


def _read_solomon_txt(path: Path) -> dict[str, Any]:
    """Parse classic Solomon VRPTW ``.txt`` instances.

    The format is table-based rather than VRPLIB-section-based:
    a vehicle block gives ``NUMBER CAPACITY`` and the customer block
    gives ``CUST NO., XCOORD., YCOORD., DEMAND, READY TIME, DUE DATE,
    SERVICE TIME``.
    """
    name = path.stem
    capacity = 0
    n_vehicles = 0
    nodes: list[dict[str, float]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for idx, raw in enumerate(lines):
        stripped = raw.strip()
        if idx == 0 and stripped:
            name = stripped
        parts = stripped.split()
        if len(parts) == 2 and all(
            part.replace(".", "", 1).isdigit() for part in parts
        ) and n_vehicles == 0 and capacity == 0:
            n_vehicles = int(float(parts[0]))
            capacity = int(float(parts[1]))
            continue
        if len(parts) >= 7 and parts[0].lstrip("-").isdigit():
            nodes.append(
                {
                    "x": float(parts[1]),
                    "y": float(parts[2]),
                    "demand": float(parts[3]),
                    "tw_early": float(parts[4]),
                    "tw_late": float(parts[5]),
                    "service_time": float(parts[6]),
                }
            )
    if not nodes:
        raise ValueError(f"no Solomon customer rows parsed from {path}")
    service_times = [node.get("service_time", 0.0) for node in nodes[1:]]
    service_time = float(service_times[0]) if service_times else 0.0
    return {
        "name": name,
        "type": "VRPTW",
        "capacity": capacity,
        "n_vehicles": n_vehicles,
        "service_time": service_time,
        "depot_index": 0,
        "nodes": [
            {
                "x": node["x"],
                "y": node["y"],
                "demand": node["demand"],
                "tw_early": node["tw_early"],
                "tw_late": node["tw_late"],
            }
            for node in nodes
        ],
        "synthetic": False,
    }
