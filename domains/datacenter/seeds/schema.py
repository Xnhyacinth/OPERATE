"""Source-locked datacenter scheduling scenario schema."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass
class DatacenterPerturbation:
    kind: Literal[
        "capacity_reduction",
        "queue_burst",
        "sla_deadline_pressure",
    ]
    trigger_tick: int
    duration_ticks: int = 1
    hidden: bool = False
    target: dict[str, Any] = field(default_factory=dict)
    intensity: float = 1.0
    notes: str = ""


@dataclass
class JobStakeholder:
    load_id: str
    stakeholder_class: str = "batch"
    criticality: float = 0.3
    demand: float = 0.0
    bus_id: str | None = None


@dataclass
class Provenance:
    data_source: str
    files: list[str] = field(default_factory=list)
    commit: str | None = None
    url: str | None = None
    lock_strategy: str | None = None
    time_window: dict[str, Any] = field(default_factory=dict)
    license: str = "Apache-2.0 upstream repository; trace terms apply"
    notes: str = ""


@dataclass
class DatacenterScenarioSeed:
    seed_id: str
    family: str = "gpu_cluster_scheduling"
    domain: str = "datacenter"
    backend_kind: str = "alibaba_trace_sim"
    backend_config: dict[str, Any] = field(default_factory=dict)
    horizon_ticks: int = 8
    tick_minutes: int = 30
    seed: int = 42
    load_assignments: list[JobStakeholder] = field(default_factory=list)
    perturbations: list[DatacenterPerturbation] = field(default_factory=list)
    dilemmas: list[dict[str, Any]] = field(default_factory=list)
    difficulty_mode: Literal["time_pressure", "deep_planning"] = "time_pressure"
    difficulty_level: Literal["basic", "medium", "high", "extreme"] = "basic"
    provenance: Provenance = field(
        default_factory=lambda: Provenance(data_source="unspecified")
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def signature(self) -> str:
        body = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(body.encode()).hexdigest()[:16]

    def complexity_metrics(self) -> dict[str, Any]:
        jobs = list(self.backend_config.get("jobs") or [])
        durations = [float(job.get("duration_seconds") or 0.0) for job in jobs]
        gpu_demands = [
            float(job.get("requested_gpu_units") or 0.0)
            * float(job.get("instance_count") or 1.0)
            for job in jobs
        ]
        capacity = float(self.backend_config.get("gpu_capacity_units") or 1.0)
        return {
            "horizon_minutes": self.horizon_ticks * self.tick_minutes,
            "n_perturbations": len(self.perturbations),
            "observability_burden": sum(p.hidden for p in self.perturbations),
            "decision_depth": max(
                1,
                len(self.perturbations)
                + int(self.difficulty_mode == "deep_planning"),
            ),
            "n_jobs": len(jobs),
            "duration_ratio": (
                max(durations) / max(1.0, min(durations)) if durations else 0.0
            ),
            "gpu_demand_to_capacity_ratio": sum(gpu_demands) / max(1.0, capacity),
            "n_distinct_users": len(
                {str(job.get("user") or "") for job in jobs}
            ),
        }
