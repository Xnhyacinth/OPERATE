"""Canonical scenario-level dimension applicability contract."""

from __future__ import annotations

from collections.abc import Mapping

from evaluation.scorer import DISCRIMINATIVE_CORE_DIMENSIONS

DIMENSION_APPLICABILITY_DIMENSIONS = frozenset(DISCRIMINATIVE_CORE_DIMENSIONS) - {
    "task_completion"
}


def dimension_applicability_contract_issue(
    applicability: object,
) -> tuple[str, str | None] | None:
    """Return the first canonical contract issue, or ``None`` when valid."""

    if not isinstance(applicability, Mapping) or set(applicability) != (
        DIMENSION_APPLICABILITY_DIMENSIONS
    ):
        return ("incomplete", None)
    for dimension in sorted(DIMENSION_APPLICABILITY_DIMENSIONS):
        spec = applicability[dimension]
        if not isinstance(spec, Mapping) or not isinstance(
            spec.get("applicable"), bool
        ):
            return ("invalid", dimension)
        reason = spec.get("reason")
        if not isinstance(reason, str) or not reason:
            return ("reason_missing", dimension)
    return None


def dimension_applicability_contract_is_valid(applicability: object) -> bool:
    """Return whether a scenario declares all formal diagnostic dimensions."""

    return dimension_applicability_contract_issue(applicability) is None
