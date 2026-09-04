"""Provenance-file existence check."""

from __future__ import annotations

from pathlib import Path

from audit._common import (
    DEFAULT_REGISTRY_PATH,
    REPO_ROOT,
    _load_registry,
)


def _provenance_candidates(rel: str, repo_root: Path | None = None) -> list[Path]:
    """Return current and sanctioned legacy-layout locations for a source."""
    root = repo_root or REPO_ROOT
    rel_path = Path(rel)
    if rel_path.is_absolute():
        return [rel_path]
    candidates = [root / rel_path, root.parent / rel_path]
    legacy_prefixes = {
        "works/OpenDSS-IEEE34-IEEE123/": (
            "works/OpenDSS-IEEE13/Version8/Distrib/IEEETestCases/"
        ),
        "works/OpenDSS-IEEE13/13Bus/": (
            "works/OpenDSS-IEEE13/Version8/Distrib/IEEETestCases/13Bus/"
        ),
    }
    for old, new in legacy_prefixes.items():
        if rel.startswith(old):
            candidates.append(root / f"{new}{rel.removeprefix(old)}")
    if rel == "works/OpenDSS-IEEE13/IEEELineCodes.DSS":
        candidates.append(
            root
            / "works/OpenDSS-IEEE13/Version8/Distrib/IEEETestCases/IEEELineCodes.DSS"
        )
    return candidates


def check_provenance_files(
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> tuple[int, int, list[str]]:
    try:
        registry = _load_registry(registry_path)
    except FileNotFoundError as exc:
        return 0, 0, [str(exc)]
    issues: list[str] = []
    n_ok = 0
    n_total = 0
    seen_files: dict[str, bool] = {}
    for row in registry.get("scenarios", []):
        for rel in row.get("provenance_files", []):
            # v0.2.1: virtual handles (programmatic upstreams w/o an
            # on-disk file) are recognised by the URI scheme prefix.
            # `grid2op://` (Grid2Op chronics IDs), `pandapower-cigre-mv://`
            # (CIGRE MV network); v0.4 Bucket B adds the mv_oberrhein and
            # synthetic-LV pandapower distribution topologies.
            _virtual_schemes = (
                "grid2op://",
                "pandapower-cigre-mv://",
                "pandapower-mv-oberrhein://",
                "pandapower-synthetic-lv://",
                "pandapower-simbench://",
            )
            if rel.startswith(_virtual_schemes):
                continue
            n_total += 1
            if rel in seen_files:
                if seen_files[rel]:
                    n_ok += 1
                continue
            candidates = _provenance_candidates(rel, REPO_ROOT)
            exists = any(path.exists() for path in candidates)
            seen_files[rel] = exists
            if not exists:
                issues.append(f"missing provenance file: {rel}")
            else:
                n_ok += 1
    return n_ok, n_total, issues
