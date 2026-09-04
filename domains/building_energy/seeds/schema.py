"""CityLearn-native pilot scenario identity."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class BuildingEnergyScenarioSeed:
    seed_id: str
    family: str = "citylearn_der_storage_control"
    domain: str = "building_energy"
    backend_kind: str = "citylearn"
    source_root: str = ""
    source_lock: str = ""
    backend_config: dict[str, Any] = field(default_factory=dict)
    horizon_ticks: int = 24
    tick_minutes: int = 60
    seed: int = 2022
    difficulty_level: str = "basic"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def signature(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def rebuild_seed_from_dict(
    data: dict[str, Any], override_seed: int
) -> BuildingEnergyScenarioSeed:
    return BuildingEnergyScenarioSeed(
        seed_id=str(data.get("seed_id") or "citylearn_pilot"),
        family=str(data.get("family") or "citylearn_der_storage_control"),
        domain=str(data.get("domain") or "building_energy"),
        backend_kind=str(data.get("backend_kind") or "citylearn"),
        source_root=str(data.get("source_root") or ""),
        source_lock=str(data.get("source_lock") or ""),
        backend_config=dict(data.get("backend_config") or {}),
        horizon_ticks=max(1, int(data.get("horizon_ticks") or 24)),
        tick_minutes=max(1, int(data.get("tick_minutes") or 60)),
        seed=int(override_seed),
        difficulty_level=str(data.get("difficulty_level") or "basic"),
    )
