#!/usr/bin/env python3
"""Run the current OPERATE Full / Core suite."""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

from scripts import batch_llm_eval


REPO_ROOT = Path(__file__).resolve().parent
CORE_SUITE = REPO_ROOT / "benchmark/core_suite.json"


def main() -> int:
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument("--core-suite", type=Path, default=CORE_SUITE)
    selected, forwarded = selector.parse_known_args(sys.argv[1:])
    core_suite = selected.core_suite.resolve()
    forbidden = {
        "--formal-run",
        "--formal-manifest",
        "--scenario-slice",
        "--scenarios",
        "--finalize-only",
        "--retry-cells",
        "--lite-lineage-suite",
        "--seeds",
    }
    conflicts = sorted({
        argument for argument in forwarded
        if argument.startswith("--")
        and argument.split("=", 1)[0] != "--finalize"
        and any(flag.startswith(argument.split("=", 1)[0]) for flag in forbidden)
    })
    if conflicts:
        joined = ", ".join(conflicts)
        raise SystemExit(
            "The public Full runner binds benchmark/core_suite.json. Private "
            f"qualification manifests are not shipped; remove: {joined}"
        )

    for index, argument in enumerate(forwarded):
        option, separator, value = argument.partition("=")
        if option.startswith("--") and "--seed-mode".startswith(option):
            mode = (
                value
                if separator
                else (forwarded[index + 1] if index + 1 < len(forwarded) else "")
            )
            if mode != "scenario":
                raise SystemExit("OPERATE Full requires --seed-mode scenario")

    payload = json.loads(core_suite.read_text(encoding="utf-8"))
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
        *forwarded,
        "--seed-mode",
        "scenario",
        "--scenario-slice",
        "custom",
        "--scenarios",
        *paths,
    ]
    return batch_llm_eval.main()


if __name__ == "__main__":
    raise SystemExit(main())
