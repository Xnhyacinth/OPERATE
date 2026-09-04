"""Datacenter-native stakeholder groups derived from trace users."""

from __future__ import annotations

from collections.abc import Iterable

from core import StakeholderGroup

from .seeds.schema import DatacenterScenarioSeed


def build_stakeholder_groups(
    seed_obj: DatacenterScenarioSeed,
    runtime_tenant_ids: Iterable[str] | None = None,
) -> list[StakeholderGroup]:
    """Build one auditable tenant group per anonymized trace user."""
    if runtime_tenant_ids is None:
        jobs = list(seed_obj.backend_config.get("jobs") or [])
        users = sorted({str(job.get("user") or "unknown") for job in jobs})
    else:
        users = sorted({str(user) for user in runtime_tenant_ids})
    return [
        StakeholderGroup(
            group_id=user,
            display_name=f"Trace tenant {user}",
            baseline_trust=0.6,
            volatility=0.06,
            positive_delta={"timely_response": 0.08},
            negative_delta={"delayed_response": -0.12},
            metadata={"class": "gpu_tenant", "domain": "datacenter"},
        )
        for user in users
    ]
