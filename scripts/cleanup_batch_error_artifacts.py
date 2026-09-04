#!/usr/bin/env python3
"""Delete superseded batch logs/trajectories. Dry-run by default.

Keeps trajectory dirs and logs belonging to the latest compact ``status=ok``
row per cell. Deletes error rows **and** superseded dirty-ok retries.

Resolves trajectory directories from ``trajectory_summary.trajectory_path``
(the episode prefix directory), not from top-level ``episode_log_path``.
Refuses to ``rmtree`` when ok rows exist but the keep-set of trajectory
dirs is empty — that was the pass1 data-loss failure mode.

Requires ``--apply`` to delete anything.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"episodes jsonl not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _cell_key(row: dict[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(row.get("model") or ""),
        str(row.get("scenario_slug") or row.get("scenario_id") or ""),
        int(row.get("seed", -1) or -1),
        str(row.get("pass_id") or "pass-0"),
    )


def compact_latest_terminal(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest_any: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    latest_terminal: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    order: list[tuple[str, str, int, str]] = []
    for row in rows:
        key = _cell_key(row)
        if key not in latest_any:
            order.append(key)
        latest_any[key] = row
        if str(row.get("status", "ok")) != "in_flight":
            latest_terminal[key] = row
    return [latest_terminal.get(key, latest_any[key]) for key in order]


def trajectory_dir_from_row(row: dict[str, Any]) -> Path | None:
    summary = row.get("trajectory_summary")
    if isinstance(summary, dict):
        raw = summary.get("trajectory_path")
        if raw:
            return Path(str(raw)).parent
    raw_dir = row.get("trajectory_dir")
    if raw_dir:
        return Path(str(raw_dir))
    return None


def plan_cleanup(rows: list[dict[str, Any]]) -> dict[str, Any]:
    compact = [
        row
        for row in compact_latest_terminal(rows)
        if str(row.get("status", "ok")) != "in_flight"
    ]
    ok_rows = [row for row in compact if str(row.get("status")) == "ok"]
    error_rows = [row for row in compact if str(row.get("status")) != "ok"]

    keep_traj: set[Path] = set()
    keep_logs: set[Path] = set()
    ok_missing_traj = 0
    for row in ok_rows:
        traj = trajectory_dir_from_row(row)
        if traj is None:
            ok_missing_traj += 1
        else:
            keep_traj.add(traj.resolve())
        log_raw = row.get("episode_log_path")
        if log_raw:
            keep_logs.add(Path(str(log_raw)).resolve())

    if ok_rows and not keep_traj:
        raise RuntimeError(
            "refusing to delete trajectories: "
            f"{len(ok_rows)} ok rows exist but keep-set of trajectory dirs is empty "
            "(trajectory_summary.trajectory_path missing). This is the pass1 "
            "data-loss failure mode."
        )

    delete_traj: list[Path] = []
    delete_logs: list[Path] = []
    seen_traj: set[Path] = set()
    seen_logs: set[Path] = set()
    for row in rows:
        traj = trajectory_dir_from_row(row)
        if traj is not None:
            resolved = traj.resolve()
            if (
                resolved not in keep_traj
                and resolved not in seen_traj
                and traj.exists()
            ):
                delete_traj.append(resolved)
                seen_traj.add(resolved)
        log_raw = row.get("episode_log_path")
        if log_raw:
            log_path = Path(str(log_raw))
            resolved_log = log_path.resolve() if log_path.exists() else log_path
            if (
                resolved_log not in keep_logs
                and resolved_log not in seen_logs
                and log_path.exists()
            ):
                delete_logs.append(log_path.resolve())
                seen_logs.add(resolved_log)

    return {
        "n_rows": len(rows),
        "n_compact": len(compact),
        "n_ok": len(ok_rows),
        "n_error": len(error_rows),
        "ok_missing_traj": ok_missing_traj,
        "keep_traj_dirs": sorted(str(path) for path in keep_traj),
        "keep_logs": sorted(str(path) for path in keep_logs),
        "delete_traj_dirs": sorted({str(path) for path in delete_traj}),
        "delete_logs": sorted({str(path) for path in delete_logs}),
    }


def apply_cleanup(plan: dict[str, Any]) -> dict[str, int]:
    deleted_dirs = 0
    deleted_logs = 0
    for raw in plan["delete_traj_dirs"]:
        path = Path(raw)
        if path.is_dir():
            shutil.rmtree(path)
            deleted_dirs += 1
    for raw in plan["delete_logs"]:
        path = Path(raw)
        if path.is_file():
            path.unlink()
            deleted_logs += 1
    return {"deleted_traj_dirs": deleted_dirs, "deleted_logs": deleted_logs}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--episodes",
        type=Path,
        default=None,
        help="Defaults to <output-dir>/episodes.jsonl",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete. Without this flag the script only prints a plan.",
    )
    args = parser.parse_args(argv)
    episodes_path = args.episodes or (args.output_dir / "episodes.jsonl")
    plan = plan_cleanup(_load_jsonl(episodes_path))
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not args.apply:
        print("dry-run; pass --apply to delete", file=sys.stderr)
        return 0
    result = apply_cleanup(plan)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
