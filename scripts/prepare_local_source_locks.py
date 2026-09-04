#!/usr/bin/env python3
"""Create reproducible local lock manifests for pinned source checkouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
JSPLIB_COMMIT = "eea2b60dd7e2f5c907ff7302662c61812eb7efdf"
JSPLIB_REPO = "https://github.com/tamy0612/JSPLIB"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_jsplib_checksums(root: Path) -> str:
    metadata_path = root / "instances.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    paths = sorted(str(row["path"]) for row in metadata)
    missing = [rel for rel in paths if not (root / rel).is_file()]
    if missing:
        raise FileNotFoundError(f"missing JSPLIB instances: {missing[:5]}")
    lines = [
        f"# repo: {JSPLIB_REPO}",
        f"# commit: {JSPLIB_COMMIT}",
        f"{_sha256(metadata_path)}  instances.json",
    ]
    lines.extend(f"{_sha256(root / rel)}  {rel}" for rel in paths)
    return "\n".join(lines) + "\n"


def verify_checkout_commit(root: Path, expected: str = JSPLIB_COMMIT) -> None:
    actual = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected:
        raise ValueError(f"JSPLIB checkout is {actual}; expected {expected}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsplib-root", type=Path, default=REPO_ROOT / "works/JSPLIB-Instances"
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = args.jsplib_root.expanduser().resolve()
    verify_checkout_commit(root)
    expected = render_jsplib_checksums(root)
    target = root / "CHECKSUMS.txt"
    if args.check:
        if not target.is_file() or target.read_text(encoding="utf-8") != expected:
            raise SystemExit(f"stale or missing checksum manifest: {target}")
    else:
        target.write_text(expected, encoding="utf-8")
        print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
