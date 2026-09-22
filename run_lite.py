#!/usr/bin/env python3
"""Run the fixed current OPERATE-Lite efficiency/development suite."""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

from scripts import batch_llm_eval


REPO_ROOT = Path(__file__).resolve().parent


def _default_lite_suite() -> Path:
    """Maintenance layout first; the versionless public tree ships ``benchmark/``."""
    maintained = REPO_ROOT / "release/operate_v0_62_0/lite_suite.json"
    if maintained.is_file():
        return maintained
    return REPO_ROOT / "benchmark/lite_suite.json"


LITE_SUITE = _default_lite_suite()

# Documented OPERATE-Lite persistent working-context profile (64 messages /
# 512000 chars / 128 memory items). The batch applies these bounds only under
# `--formal-run`, which Lite forbids, so without them Lite silently runs the
# 32 / 48000 / 64 diagnostic defaults and is not the documented profile.
# Provider failures must abort: Lite is not `--formal-run`, so the batch
# diagnostic default would otherwise synthesize environment-advancing `wait`.
# Injected only when the caller did not name the flag itself.
LITE_PROFILE_FLAGS = (
    ("--persistent-history-max-messages", "64"),
    ("--persistent-context-max-chars", "512000"),
    ("--persistent-memory-max-items", "128"),
    ("--provider-timeout-s", "300"),
    ("--provider-failure-policy", "abort"),
    ("--max-consecutive-provider-failures", "1"),
)


def main() -> int:
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument(
        "--implementation-policy",
        choices=["provenance", "strict"],
        default="provenance",
    )
    selector.add_argument("--lite-suite", type=Path, default=LITE_SUITE)
    selected, forwarded = selector.parse_known_args(sys.argv[1:])
    lite_suite = selected.lite_suite.resolve()
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
    # The downstream argparse parser accepts both --flag=value and long-option
    # abbreviations. Neither may replace the fixed Lite scope or recover Full.
    conflicts = sorted(
        {
            argument
            for argument in forwarded
            if argument.startswith("--")
            and argument.split("=", 1)[0] != "--finalize"
            and any(flag.startswith(argument.split("=", 1)[0]) for flag in forbidden)
        }
    )
    if conflicts:
        joined = ", ".join(conflicts)
        raise SystemExit(
            f"OPERATE-Lite fixes its scenario set and is not a Full formal shard; "
            f"remove: {joined}"
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
                raise SystemExit("OPERATE-Lite lineage requires --seed-mode scenario")

    # Downstream argparse accepts long-option abbreviations, so a prefix match
    # in *either* direction counts as the caller having chosen the bound: an
    # explicit `--persistent-context 256000` (a legal unambiguous prefix of
    # `--persistent-context-max-chars`) must not be overridden by the injected
    # default, which argparse would apply last. An explicit choice is never
    # overridden, in either `--flag value` or `--flag=value` form.
    declared_options = {
        argument.split("=", 1)[0] for argument in forwarded if argument.startswith("--")
    }
    forwarded_profile = list(forwarded)
    profile_source: list[str] = []
    for flag, value in LITE_PROFILE_FLAGS:
        if any(
            flag.startswith(option) or option.startswith(flag)
            for option in declared_options
        ):
            profile_source.append(f"{flag}=caller")
            continue
        forwarded_profile.extend([flag, value])
        profile_source.append(f"{flag}=run_lite_default")

    # Record the resolved provenance on stderr so a run log shows which profile
    # the batch actually ran; run_lite.py has no run-metadata channel of its
    # own, and this stays a log line rather than new config plumbing.
    print(
        "[INFO] OPERATE-Lite persistent profile: " + ", ".join(profile_source),
        file=sys.stderr,
    )
    print(
        "[INFO] OPERATE-Lite is an efficiency/development track, not a Full "
        "leaderboard or long-horizon denominator; provider failures abort and "
        "are never converted to wait.",
        file=sys.stderr,
    )

    payload = json.loads(lite_suite.read_text(encoding="utf-8"))
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
        *forwarded_profile,
        "--implementation-policy",
        selected.implementation_policy,
        "--lite-lineage-suite",
        str(lite_suite),
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
