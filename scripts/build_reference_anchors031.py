#!/usr/bin/env python3
"""Compile the 0.31 reference anchors and scoring policy.

Reads only frozen source contracts and deterministic CPU reference episodes.
No model trajectory, provider call or native replay is involved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.reference_anchors031 import build_reference_anchors  # noqa: E402
from evaluation.task_outcome031 import compile_policy031  # noqa: E402

DEFAULT_H_MIN = 0.02
DEFAULT_SERVICE_DELTA = 1.0
POLICIES = ROOT / "evaluation/policies"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--extra-reference-dir",
        type=Path,
        action="append",
        default=[],
        help="additional fixed reference configurations for a subset of cases",
    )
    parser.add_argument(
        "--evaluation-version", choices=("0.31.0", "0.31.1", "0.31.2"), default="0.31.2"
    )
    parser.add_argument("--policy027", type=Path, default=POLICIES / "lite141_027.json")
    parser.add_argument("--policy028", type=Path, default=POLICIES / "lite141_028.json")
    parser.add_argument(
        "--suite", type=Path, default=ROOT / "release/operate_v0_62_0/lite_suite.json"
    )
    parser.add_argument("--h-min", type=float, default=DEFAULT_H_MIN)
    parser.add_argument("--service-delta", type=float, default=DEFAULT_SERVICE_DELTA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-output", type=Path)
    args = parser.parse_args()
    if args.output.exists() or (args.policy_output and args.policy_output.exists()):
        raise ValueError("anchor_and_policy_outputs_must_be_new")
    manifest = build_reference_anchors(
        [p.resolve() for p in args.reference_dir],
        policy027=json.loads(args.policy027.read_bytes()),
        policy028=json.loads(args.policy028.read_bytes()),
        root=ROOT,
        h_min=args.h_min,
        service_delta=args.service_delta,
        extra_reference_dirs=[p.resolve() for p in args.extra_reference_dir],
    )
    args.output.write_text(json.dumps(manifest, indent=1, allow_nan=False) + "\n")
    summary = {"anchors": str(args.output), **manifest["counts"]}
    if args.policy_output:
        policy = compile_policy031(
            args.output,
            root=ROOT,
            base_policy_path=args.policy028,
            suite_path=args.suite,
            evaluation_version=args.evaluation_version,
        )
        args.policy_output.write_text(
            json.dumps(policy, indent=1, allow_nan=False) + "\n"
        )
        summary.update(
            policy=str(args.policy_output), policy_sha256=policy["policy_sha256"]
        )
    print(json.dumps({**summary, "anchors_sha256": manifest["anchors_sha256"]}))


if __name__ == "__main__":
    main()
