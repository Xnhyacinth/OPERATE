"""runner.resume — scenario-signature recomputation (P3-2).

Moved verbatim from ``run._recompute_signature_with_seed`` so the batch
LLM runner no longer reaches into ``run``'s ``_``-private namespace.
"""

from __future__ import annotations

from typing import Any

from domains.registry import DomainSpec, get_domain_spec


def recompute_signature_with_seed(
    scenario: dict[str, Any], seed: int, spec: DomainSpec | None = None
) -> str:
    """Like ``_recompute_signature`` but uses the actual run seed.

    Use this from ``run_one`` whenever a ``seed_override`` was applied so
    the reported signature reflects the realized episode rather than the
    YAML template seed. ``spec`` is resolved from ``scenario['domain']``
    when not supplied.
    """
    spec = spec or get_domain_spec(scenario.get("domain"))
    return spec.scenario_signature(scenario, int(seed))
