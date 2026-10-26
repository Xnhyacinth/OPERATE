"""Canonical four-level difficulty taxonomy.

Frozen releases may contain two legacy aliases. They are accepted only as
migration inputs and canonicalize to ``extreme`` for every live protocol
surface. Stronger physical mechanics belong in ``stress_profile`` metadata,
not in additional public difficulty levels.
"""

from __future__ import annotations

CANONICAL_DIFFICULTY_LEVELS: tuple[str, ...] = (
    "basic",
    "medium",
    "high",
    "extreme",
)

_RAW_TO_CANONICAL: dict[str, str] = {
    "basic": "basic",
    "medium": "medium",
    "high": "high",
    "extreme": "extreme",
    "extreme_plus": "extreme",
    "cascading": "extreme",
}

LEGACY_DIFFICULTY_ALIASES: frozenset[str] = frozenset(
    set(_RAW_TO_CANONICAL) - set(CANONICAL_DIFFICULTY_LEVELS)
)

RAW_LEVEL_RANK: dict[str, int] = {
    "basic": 0,
    "medium": 1,
    "high": 2,
    "extreme": 3,
    "extreme_plus": 3,
    "cascading": 3,
}


def canonical_difficulty_level(level: str | None) -> str:
    """Map a raw difficulty label to the public four-rung ladder."""
    return _RAW_TO_CANONICAL.get(str(level), str(level))


def raw_level_rank(level: str | None) -> int:
    """Monotonic public rank, including frozen-release aliases."""
    return RAW_LEVEL_RANK.get(str(level), -1)
