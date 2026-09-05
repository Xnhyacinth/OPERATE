"""Resolve portable batch trajectory references without depending on the cwd."""

from pathlib import Path
from typing import Any


def resolve_batch_path(value: object, *, batch_root: Path | None = None) -> Path:
    """Prefer portable paths; permit existing legacy paths only inside the run."""
    path = Path(str(value))
    if path.is_absolute() or batch_root is None:
        return path.resolve()
    root = batch_root.resolve()
    portable = (root / path).resolve()
    if portable.exists():
        return portable
    legacy = path.resolve()
    if legacy.is_relative_to(root) and legacy.exists():
        return legacy
    return portable


def trajectory_file(
    row: dict[str, Any], *, batch_root: Path | None = None
) -> Path | None:
    summary = row.get("trajectory_summary") or {}
    raw = summary.get("trajectory_path") if isinstance(summary, dict) else None
    if not raw:
        return None
    base = Path(str(raw))
    candidates = [base]
    if not str(base).endswith(".trajectory.jsonl"):
        candidates.append(Path(str(base) + ".trajectory.jsonl"))
    if base.suffix:
        candidates.append(base.with_suffix(".trajectory.jsonl"))
    for candidate in candidates:
        candidate = resolve_batch_path(candidate, batch_root=batch_root)
        if batch_root is not None and not candidate.is_relative_to(batch_root.resolve()):
            continue
        if candidate.is_file():
            return candidate
    return None
