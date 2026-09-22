#!/usr/bin/env python3
"""Recompute realtime evidence-closure for settled artifacts after a closure fix.

The coordinator stamps ``evidence_closure`` into each episode artifact at
write time. When a closure-keying defect is later fixed, already-settled
artifacts keep the stale (wrongly failing) closure even though their episode
data is sound. This tool recomputes the closure from the artifact's own
ledger and transitions with the *current* code, rewrites the artifact when it
now passes, and rewrites the batch journal (``episodes.jsonl``) rows that were
marked ineligible solely for closure reasons.

Fail-closed: an artifact whose recomputed closure still fails is left
untouched, and a row is only upgraded when its remaining eligibility reasons
are exactly the closure-derived set. Model-visible bytes and provider audit
are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runner.realtime_episode import _build_evidence_closure  # noqa: E402

CLOSURE_DERIVED_REASONS = {
    "evidence_closure_incomplete",
    "artifact_validation_failed",
    "episode_not_complete",
    "artifact_not_evaluation_ready",
}


def _canonical(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class _LedgerEnv:
    """Supply the artifact's own ledger the way env.evidence would."""

    def __init__(self, ledger: list[dict[str, Any]]) -> None:
        self._ledger = ledger

    class _Evidence:
        def __init__(self, ledger: list[dict[str, Any]]) -> None:
            self._ledger = ledger

        def to_jsonable(self) -> list[dict[str, Any]]:
            return self._ledger

    @property
    def evidence(self) -> "_LedgerEnv._Evidence":
        return self._Evidence(self._ledger)


def repair_treatment(treatment_dir: Path, *, write: bool) -> dict[str, Any]:
    episodes_path = treatment_dir / "episodes.jsonl"
    rows = [json.loads(line) for line in episodes_path.read_text().splitlines() if line]
    upgraded: list[dict[str, Any]] = []
    untouched_fail: list[str] = []

    for index, row in enumerate(rows):
        if row.get("status") != "ineligible":
            continue
        reasons = set(row.get("eligibility_reasons") or [])
        if not reasons or not reasons <= CLOSURE_DERIVED_REASONS:
            continue
        artifact_path = treatment_dir / str(row.get("artifact_path") or "")
        if not artifact_path.is_file():
            continue
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        ledger = (artifact.get("evidence_closure") or {}).get("ledger") or []
        if not ledger:
            continue
        closure = _build_evidence_closure(_LedgerEnv(ledger), artifact)
        if closure.get("closure_complete") is not True:
            untouched_fail.append(row.get("job_key"))
            continue
        if not write:
            upgraded.append(row)
            continue

        backup = artifact_path.with_suffix(artifact_path.suffix + ".closurefix.bak")
        if not backup.exists():
            shutil.copy2(artifact_path, backup)
        artifact["evidence_closure"] = closure
        validation = artifact.get("artifact_validation") or {}
        if validation.get("blocker_codes") == ["EVIDENCE_CLOSURE_INCOMPLETE"]:
            artifact["artifact_validation"] = {
                **validation,
                "blocker_codes": [],
                "valid": True,
            }
        artifact_path.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        rows[index] = {
            **row,
            "status": "ok",
            "eligibility_reasons": [],
            "artifact_sha256": _sha256(artifact_path),
        }
        upgraded.append(rows[index])

    if write and upgraded:
        episodes_path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
            encoding="utf-8",
        )
    return {
        "treatment": treatment_dir.name,
        "rows_total": len(rows),
        "upgraded": len(upgraded),
        "still_failing": len(untouched_fail),
        "still_failing_keys": untouched_fail[:10],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_root", type=Path, help="batch output root containing treatment-* dirs"
    )
    parser.add_argument(
        "--write", action="store_true", help="apply repairs (default: report only)"
    )
    args = parser.parse_args()

    treatments = sorted(args.output_root.glob("treatment-*"))
    if not treatments:
        print(f"no treatment dirs under {args.output_root}", file=sys.stderr)
        return 1
    for treatment in treatments:
        print(json.dumps(repair_treatment(treatment, write=args.write)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
