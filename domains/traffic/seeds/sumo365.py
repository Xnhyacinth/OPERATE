"""SUMO Ingolstadt 365 multi-date Traffic seed support (v0.32 staging).

This module is intentionally source/data wiring only. It exposes the nine local
``works/sumo_ingolstadt/simulation/Ingolstadt SUMO 365`` dates as
source-locked Traffic seeds for host-independent mock filtering. The live SUMO
backend remains a later gate; these seeds default to ``mock_sumo`` and carry
``release_ready=False`` so they cannot silently enter a publishable denominator.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import replace
from functools import cache
from pathlib import Path
from typing import Any

from .from_lust import (
    INGOLSTADT_SOURCE_COMMIT,
    INGOLSTADT_SOURCE_URL,
    build_traffic_seed,
)
from .materializer_support import traffic_source_denominator_key
from .schema import Provenance, TrafficScenarioSeed

REPO_ROOT = Path(__file__).resolve().parents[3]
SUMO365_ROOT = Path("works/sumo_ingolstadt/simulation/Ingolstadt SUMO 365")
SUMO365_NET_REF = SUMO365_ROOT / "ingolstadt_net.net.xml"
SUMO365_SERVICE_DATES: tuple[str, ...] = (
    "2023-06-19",
    "2023-06-20",
    "2023-06-25",
    "2023-06-26",
    "2023-06-27",
    "2023-07-02",
    "2023-07-03",
    "2023-07-04",
    "2023-07-09",
)
SUMO365_LICENSE = "Apache-2.0"
SUMO365_LOCK_STRATEGY = "git_commit+per_date_file_sha256"

SUMO365_EXPECTED_ROUTE_VEHICLE_COUNTS: dict[str, int] = {
    "2023-06-19": 292327,
    "2023-06-20": 283372,
    "2023-06-25": 186585,
    "2023-06-26": 283143,
    "2023-06-27": 289703,
    "2023-07-02": 177223,
    "2023-07-03": 292118,
    "2023-07-04": 174082,
    "2023-07-09": 5983,
}

SUMO365_EXPECTED_FILE_SHA256S: dict[str, dict[str, str]] = {
    "2023-06-19": {
        "sumocfg": "b1e97bc86509b8ab414587897caad060d51c99c4ab5f3b94de27508ed0829874",
        "network": "ab043e0974eb3ccd8f76799c8687960f3ac42c155f6efe7b01405ebba34549f8",
        "route": "25607629c08de6ec47f8c02489fb906c4d34f56391e84ad9701154480907a029",
        "tl_logic": "d74238f4193a86b1f2a3023896bba018e4df8ac8401f76a19983d008ff67a825",
        "waut": "1673b335dc90a45392cdb6c1b657861d4008a3abd80bc6ee6d1ad3a0147cc9ba",
        "pt_stops": "6eaa43bf0b8cd80f53146b02fa26e6282cc9c184b2c16190f2d51974936ef03f",
        "pt_trips": "fe31154ba1c1212b3dc44f5c31cd9dcd902fec4808ba8210deb73a643001d297",
    },
    "2023-06-20": {
        "sumocfg": "a9e2b6cd8b3dd46599e4091ee1c336682589a23806995d33240322e50b92b8fe",
        "network": "ab043e0974eb3ccd8f76799c8687960f3ac42c155f6efe7b01405ebba34549f8",
        "route": "e727cde9ac23af0a60b8bcabb601d185084bc43b86a8d5e50d049600ec5b7bf2",
        "tl_logic": "8e12ecb9ea67ba63d4e6bcc32b2b2c0f9540e7da078858c59ebbb1bc123b7671",
        "waut": "bfc56a94ab24920d72502610cd7c5d78d5d01c66ae41c8ec179abe79ac1d6548",
        "pt_stops": "6eaa43bf0b8cd80f53146b02fa26e6282cc9c184b2c16190f2d51974936ef03f",
        "pt_trips": "172f4edb8e6e96868746ea0f131d8b44b49a704ba964310f8072f7e895179b93",
    },
    "2023-06-25": {
        "sumocfg": "a706bd2623ec3cf52f4c4dd4a2bc7851f7cef7ac59830f4e1840563ac54ae1f9",
        "network": "ab043e0974eb3ccd8f76799c8687960f3ac42c155f6efe7b01405ebba34549f8",
        "route": "d954854aa929ea9fb5c5ca6972dbf3a829cf57e065582d2994fd96bb485eff08",
        "tl_logic": "9dbf071c410192b586fb6d016572c286166e2333c4c2579ad5a317e304f8fb9f",
        "waut": "136aea20ea1c45be0339a774f1c94411590a5cab8c442b915cd052b97d080e2c",
        "pt_stops": "6eaa43bf0b8cd80f53146b02fa26e6282cc9c184b2c16190f2d51974936ef03f",
        "pt_trips": "1de15ad364a78470d6fca94762c72da788dc42b3131c8371c307ef9ad401736b",
    },
    "2023-06-26": {
        "sumocfg": "ace83ca4e6b98590f34e82c1238f54a30e659900712963a236586e9fe211711c",
        "network": "ab043e0974eb3ccd8f76799c8687960f3ac42c155f6efe7b01405ebba34549f8",
        "route": "95cacf99a1d95e93443d99f9fe719b7508c1bc865d3ae19b486d4ad8490cc668",
        "tl_logic": "6eed28be792995ad8cedd97761428f95e8d574229ed7fc1d2ff48f658424d4c9",
        "waut": "4b9b669434646b96b55a712a856a98519d60b203d3264178fd614b9aa727e8f7",
        "pt_stops": "6eaa43bf0b8cd80f53146b02fa26e6282cc9c184b2c16190f2d51974936ef03f",
        "pt_trips": "c4e7c3d61b7c8cf9e25c84a66c9d6db851a4072e377c0e53f73f342d4e367615",
    },
    "2023-06-27": {
        "sumocfg": "78c79ab99d8a24f9d5710a4ddb069f24a20dd9f8f26f8a160197af9e0c69b140",
        "network": "ab043e0974eb3ccd8f76799c8687960f3ac42c155f6efe7b01405ebba34549f8",
        "route": "98c5448e0bd7741a435ce3e98f765ff6a00c94c86b6f0d153ae0d137032eeb0f",
        "tl_logic": "8e12ecb9ea67ba63d4e6bcc32b2b2c0f9540e7da078858c59ebbb1bc123b7671",
        "waut": "bfc56a94ab24920d72502610cd7c5d78d5d01c66ae41c8ec179abe79ac1d6548",
        "pt_stops": "6eaa43bf0b8cd80f53146b02fa26e6282cc9c184b2c16190f2d51974936ef03f",
        "pt_trips": "2b80b11304adb1b63c205267c7a4ab3b688aab48b32a08dfee90d5c1c67f34f3",
    },
    "2023-07-02": {
        "sumocfg": "403b5e742172265f70eb6f7e0c4ac3ffc55c4cb0d8973e73e4f943943d16fd9a",
        "network": "ab043e0974eb3ccd8f76799c8687960f3ac42c155f6efe7b01405ebba34549f8",
        "route": "dd6b63e57abdd8c6a90385dc6ed476803b519ce555ad89edadd11bef33aaac3a",
        "tl_logic": "9dbf071c410192b586fb6d016572c286166e2333c4c2579ad5a317e304f8fb9f",
        "waut": "136aea20ea1c45be0339a774f1c94411590a5cab8c442b915cd052b97d080e2c",
        "pt_stops": "6eaa43bf0b8cd80f53146b02fa26e6282cc9c184b2c16190f2d51974936ef03f",
        "pt_trips": "fb34a42083e61057a74c255c60c2810753d3e88b6ffdffcae85e60cb5af186c3",
    },
    "2023-07-03": {
        "sumocfg": "6875818f518f72e9c25e8cf4369eb419a36d55fd9631f9777b8df25f7b89a9c2",
        "network": "ab043e0974eb3ccd8f76799c8687960f3ac42c155f6efe7b01405ebba34549f8",
        "route": "fa1d3ba41606ce1588219701dd2e316ff6c1b509717804a7b75e9fef1b3a6961",
        "tl_logic": "d74238f4193a86b1f2a3023896bba018e4df8ac8401f76a19983d008ff67a825",
        "waut": "1673b335dc90a45392cdb6c1b657861d4008a3abd80bc6ee6d1ad3a0147cc9ba",
        "pt_stops": "6eaa43bf0b8cd80f53146b02fa26e6282cc9c184b2c16190f2d51974936ef03f",
        "pt_trips": "f21d8665aae4ccefe68ba7ca9cec4c5619c087b21cbadd1ff9f91b7e54ae484d",
    },
    "2023-07-04": {
        "sumocfg": "38c55a2d232c9267c73df59bb4144644d06d30fe28c3bdc2e5ad0a304811b4da",
        "network": "ab043e0974eb3ccd8f76799c8687960f3ac42c155f6efe7b01405ebba34549f8",
        "route": "e0dc6115c40f575f3092717bea0d141de9da9d75beefbc2eb7a72cc2a2a2abfd",
        "tl_logic": "8e12ecb9ea67ba63d4e6bcc32b2b2c0f9540e7da078858c59ebbb1bc123b7671",
        "waut": "bfc56a94ab24920d72502610cd7c5d78d5d01c66ae41c8ec179abe79ac1d6548",
        "pt_stops": "6eaa43bf0b8cd80f53146b02fa26e6282cc9c184b2c16190f2d51974936ef03f",
        "pt_trips": "7f0ee8543c0c78d41e4749a7758dd319424e3825a8a2f708a6f052809981e530",
    },
    "2023-07-09": {
        "sumocfg": "773ac32ed4a203e93690d3d1e2da6cebb6395a04535ceb111efa4023efcfcd8d",
        "network": "ab043e0974eb3ccd8f76799c8687960f3ac42c155f6efe7b01405ebba34549f8",
        "route": "b2813f3ddd288ca8c91fdba9bb9a2b6f3c70d113f6c5af8fbc4f66e314b3162d",
        "tl_logic": "9dbf071c410192b586fb6d016572c286166e2333c4c2579ad5a317e304f8fb9f",
        "waut": "136aea20ea1c45be0339a774f1c94411590a5cab8c442b915cd052b97d080e2c",
        "pt_stops": "6eaa43bf0b8cd80f53146b02fa26e6282cc9c184b2c16190f2d51974936ef03f",
        "pt_trips": "ceb1294165deeebc1cfa9704fc98ba0d5a703177f6faf0bfc9484a2bc8d3d3fa",
    },
}


def _repo_rel(path: Path) -> str:
    return path.as_posix()


def _abs(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _sha256_file(path: Path) -> str | None:
    ap = _abs(path)
    if not ap.exists() or not ap.is_file():
        return None
    h = hashlib.sha256()
    with ap.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _route_ref(service_date: str) -> Path:
    _require_known_date(service_date)
    return SUMO365_ROOT / "Routes" / f"routes_{service_date}_24h_det_calib.rou.xml.gz"


def _sumocfg_ref(service_date: str) -> Path:
    _require_known_date(service_date)
    return SUMO365_ROOT / f"{service_date}.sumocfg"


def _tl_refs(service_date: str) -> tuple[Path, Path]:
    _require_known_date(service_date)
    return (
        SUMO365_ROOT / "TL" / f"{service_date}_tlLogics_24h.tll.xml",
        SUMO365_ROOT / "TL" / f"{service_date}_WAUT.xml",
    )


def _pt_refs(service_date: str) -> tuple[Path, Path]:
    _require_known_date(service_date)
    return (
        SUMO365_ROOT / "PT" / "pt_stops.add.xml",
        SUMO365_ROOT / "PT" / f"{service_date}_gtfs_trips.rou.xml",
    )


def _require_known_date(service_date: str) -> None:
    if service_date not in SUMO365_SERVICE_DATES:
        raise ValueError(
            f"unknown SUMO365 service_date {service_date!r}; "
            f"expected one of {', '.join(SUMO365_SERVICE_DATES)}"
        )


def sumo365_date_files(service_date: str) -> dict[str, str]:
    """Repo-relative file handles for a SUMO365 service date."""
    tl_logic, waut = _tl_refs(service_date)
    pt_stops, pt_trips = _pt_refs(service_date)
    return {
        "sumocfg": _repo_rel(_sumocfg_ref(service_date)),
        "network": _repo_rel(SUMO365_NET_REF),
        "route": _repo_rel(_route_ref(service_date)),
        "tl_logic": _repo_rel(tl_logic),
        "waut": _repo_rel(waut),
        "pt_stops": _repo_rel(pt_stops),
        "pt_trips": _repo_rel(pt_trips),
    }


def sumo365_file_sha256s(service_date: str) -> dict[str, str | None]:
    return {
        key: _sha256_file(Path(ref))
        for key, ref in sumo365_date_files(service_date).items()
    }


def sumo365_file_hash_mismatches(service_date: str) -> dict[str, dict[str, str | None]]:
    observed = sumo365_file_sha256s(service_date)
    expected = SUMO365_EXPECTED_FILE_SHA256S.get(service_date, {})
    return {
        key: {"expected": expected.get(key), "observed": observed.get(key)}
        for key in sorted(set(expected) | set(observed))
        if observed.get(key) != expected.get(key)
    }


@cache
def sumo365_route_vehicle_count(service_date: str) -> int:
    """Count vehicles in the gzipped route file without redistributing it."""
    expected_hash = SUMO365_EXPECTED_FILE_SHA256S.get(service_date, {}).get("route")
    observed_hash = _sha256_file(_route_ref(service_date))
    if observed_hash != expected_hash:
        return 0
    route = _abs(_route_ref(service_date))
    if not route.exists():
        return 0
    count = 0
    max_lines = 2_000_000
    with gzip.open(route, "rt", encoding="utf-8", errors="ignore") as fh:
        for line_no, line in enumerate(fh, start=1):
            if line_no > max_lines:
                raise ValueError(f"SUMO365 route file exceeds line cap: {service_date}")
            count += line.count("<vehicle ")
    expected_count = SUMO365_EXPECTED_ROUTE_VEHICLE_COUNTS.get(service_date)
    return count if count == expected_count else 0


def sumo365_day_type(service_date: str) -> str:
    # The 2023 dates are fixed; avoid extra dependencies for weekday parsing.
    return (
        "weekend"
        if service_date in {"2023-06-25", "2023-07-02", "2023-07-09"}
        else "weekday"
    )


def _date_seed(service_date: str, family: str) -> int:
    digest = hashlib.sha256(f"sumo365|{service_date}|{family}".encode()).digest()
    return 42 + int.from_bytes(digest[:2], "big") % 10_000


def _date_demand_scale(service_date: str) -> float:
    counts = [max(1, sumo365_route_vehicle_count(d)) for d in SUMO365_SERVICE_DATES]
    current = max(1, sumo365_route_vehicle_count(service_date))
    median = sorted(counts)[len(counts) // 2]
    # Keep the mock stable while preserving real cross-date demand differences.
    return max(0.35, min(1.35, current / max(1, median)))


def _apply_date_demand(seed: TrafficScenarioSeed, service_date: str) -> None:
    scale = _date_demand_scale(service_date)
    adjusted = []
    for idx, corridor in enumerate(seed.corridors):
        # Small deterministic corridor skew prevents a date from being only a
        # uniform multiplier while staying derived from the source-locked date.
        digest = hashlib.sha256(
            f"{service_date}|{corridor.corridor_id}|{idx}".encode()
        ).digest()
        skew = 0.92 + (int.from_bytes(digest[:2], "big") % 17) / 100.0
        adjusted.append(
            replace(
                corridor,
                demand_veh=max(1, int(round(corridor.demand_veh * scale * skew))),
            )
        )
    seed.corridors = adjusted


def sumo365_source_lock(service_date: str) -> dict[str, Any]:
    file_refs = sumo365_date_files(service_date)
    sha256s = sumo365_file_sha256s(service_date)
    mismatches = sumo365_file_hash_mismatches(service_date)
    expected_count = SUMO365_EXPECTED_ROUTE_VEHICLE_COUNTS.get(service_date)
    observed_count = sumo365_route_vehicle_count(service_date)
    return {
        "data_source": "sumo_ingolstadt_365",
        "service_date": service_date,
        "url": INGOLSTADT_SOURCE_URL,
        "commit": INGOLSTADT_SOURCE_COMMIT,
        "license": SUMO365_LICENSE,
        "lock_strategy": SUMO365_LOCK_STRATEGY,
        "source_locked": all(sha256s.values())
        and not mismatches
        and observed_count == expected_count,
        "files": list(file_refs.values()),
        "file_refs": file_refs,
        "file_sha256s": sha256s,
        "expected_file_sha256s": SUMO365_EXPECTED_FILE_SHA256S.get(service_date, {}),
        "file_hash_mismatches": mismatches,
        "route_vehicle_count": observed_count,
        "expected_route_vehicle_count": expected_count,
        "route_vehicle_count_matches_expected": observed_count == expected_count,
        "day_type": sumo365_day_type(service_date),
    }


def build_sumo365_traffic_seed(
    *,
    seed_id: str,
    service_date: str,
    family: str = "incident_response",
    difficulty_level: str = "basic",
    difficulty_mode: str = "time_pressure",
) -> TrafficScenarioSeed:
    """Build a host-independent SUMO365 date-conditioned Traffic seed."""
    _require_known_date(service_date)
    seed = build_traffic_seed(
        seed_id=seed_id,
        family=family,
        seed=_date_seed(service_date, family),
        difficulty_level=difficulty_level,
        difficulty_mode=difficulty_mode,
    )
    files = sumo365_date_files(service_date)
    lock = sumo365_source_lock(service_date)
    seed.net_ref = files["network"]
    seed.route_ref = files["route"]
    seed.provenance = Provenance(
        data_source="sumo_ingolstadt_365",
        files=list(files.values()),
        commit=INGOLSTADT_SOURCE_COMMIT,
        url=INGOLSTADT_SOURCE_URL,
        lock_strategy=SUMO365_LOCK_STRATEGY,
        time_window={
            "service_date": service_date,
            "day_type": lock["day_type"],
            "horizon_ticks": seed.horizon_ticks,
            "tick_minutes": seed.tick_minutes,
            "route_vehicle_count": lock["route_vehicle_count"],
        },
        license=SUMO365_LICENSE,
        source_locked=bool(lock["source_locked"]),
        notes=(
            "SUMO Ingolstadt 365 service-date seed: real per-date route, TLS, "
            "WAUT, PT, and shared net files are referenced by repo-relative "
            "works/ handles and SHA-256 locked. Mock scoring uses only a "
            "date-derived demand scaling; live SUMO headroom remains a release "
            "blocker before any publishable denominator entry."
        ),
    )
    _apply_date_demand(seed, service_date)
    seed.backend_kind = "mock_sumo"
    seed.backend_config = {
        **seed.backend_config,
        "backend_kind": "mock_sumo",
        "source_integration_rung": "parsed_from",
        "release_ready": False,
        "release_reentry_ready": False,
        "service_date": service_date,
        "day_type": lock["day_type"],
        "sumo365_files": files,
        "sumo365_file_sha256s": lock["file_sha256s"],
        "sumo365_route_vehicle_count": lock["route_vehicle_count"],
        "sumo365_demand_scale": round(_date_demand_scale(service_date), 6),
        "source_denominator_key": None,  # filled below after seed is complete
        "physics_step_seconds": 1,
        "decision_interval_seconds": 30,
        "runtime_derived_tls_control": True,
    }
    # ``build_traffic_seed`` carries the locked 2020 Ingolstadt binding.
    # SUMO365 uses a different 2023 physical network, so its TLS/program
    # bindings must be generated from the exact runtime source identity.
    seed.backend_config.pop("corridor_tls_map", None)
    seed.backend_config.pop("sumo_corridor_program_map", None)
    seed.backend_config.pop("sumo_tls_binding_net_sha256", None)
    seed.backend_config["source_denominator_key"] = sumo365_source_denominator_key(seed)
    return seed


def sumo365_source_denominator_key(seed: TrafficScenarioSeed) -> str:
    service_date = str(seed.backend_config.get("service_date") or "unknown_date")
    base = traffic_source_denominator_key(seed)
    # Replace the source prefix so core grouping sees per-date physical demand.
    _source, family, level = base.split(":", 2)
    return f"sumo_ingolstadt_365:{service_date}:{family}:{level}"


def sumo365_decision_blob(seed: TrafficScenarioSeed) -> str:
    return json.dumps(seed.to_dict(), sort_keys=True, ensure_ascii=False)
