"""Source data path helpers for power-grid backends and seed factories."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_WORKSPACE_ROOT = REPO_ROOT.parent


def workspace_roots() -> tuple[Path, Path]:
    """Prefer repo-local sources while accepting the older sibling layout."""
    return (REPO_ROOT, LEGACY_WORKSPACE_ROOT)


def resolve_source_ref(
    ref: str | Path,
    *,
    description: str = "source file",
) -> Path:
    """Resolve a release ``works/...`` ref against supported local layouts."""
    raw = Path(ref)
    if raw.is_absolute():
        path = raw
    else:
        path = next(
            (
                candidate
                for root in workspace_roots()
                if (candidate := root / raw).exists()
            ),
            REPO_ROOT / raw,
        )
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def source_root(*parts: str) -> Path:
    rel = Path("works").joinpath(*parts)
    return next(
        (candidate for root in workspace_roots() if (candidate := root / rel).exists()),
        REPO_ROOT / rel,
    )


def source_ref(path: Path) -> str:
    resolved = path.resolve()
    for root in workspace_roots():
        try:
            return str(resolved.relative_to(root.resolve()))
        except ValueError:
            continue
    return str(path)
