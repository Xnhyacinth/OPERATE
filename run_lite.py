#!/usr/bin/env python3
"""Run the fixed current OPERATE-Lite efficiency/development suite."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts import batch_llm_eval


REPO_ROOT = Path(__file__).resolve().parent
LITE_SUITE = REPO_ROOT / "release/operate_v0_61_0/lite_suite.json"


def main() -> int:
    forwarded = sys.argv[1:]
    forbidden = {"--formal-run", "--scenario-slice", "--scenarios"}
    conflicts = sorted(forbidden.intersection(forwarded))
    if conflicts:
        joined = ", ".join(conflicts)
        raise SystemExit(
            f"OPERATE-Lite fixes its scenario set and is not a Full formal shard; "
            f"remove: {joined}"
        )

    payload = json.loads(LITE_SUITE.read_text(encoding="utf-8"))
    paths = []
    for row in payload["scenarios"]:
        path = str(row["path"])
        if path.startswith("scenarios/"):
            path = path.removeprefix("scenarios/")
        if path.endswith(".yaml"):
            path = path[:-5]
        paths.append(path)
    sys.argv = [
        "scripts/batch_llm_eval.py",
        "--scenario-slice",
        "custom",
        "--scenarios",
        *paths,
        *forwarded,
    ]
    return batch_llm_eval.main()


if __name__ == "__main__":
    raise SystemExit(main())
