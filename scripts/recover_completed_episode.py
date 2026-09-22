"""Recover completed-runtime scoring without provider calls or historical writes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runner.postprocessing import recover_completed_episode
from runner.worker_deadline import DEFAULT_POSTPROCESSING_TIMEOUT_S, run_with_deadline


def _recover(job):
    recover_completed_episode(Path(job["input"]), job["sha256"],
                              expected_identity=job["identity"], output=Path(job["output"]))
    return {"output": job["output"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--identity', type=Path, required=True,
                        help='JSON file containing the pinned original snapshot identity')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout-s', type=float, default=DEFAULT_POSTPROCESSING_TIMEOUT_S)
    args = parser.parse_args()
    run_with_deadline(_recover, {'input': str(args.input), 'sha256': args.sha256,
                                'identity': json.loads(args.identity.read_text()),
                                'output': str(args.output)},
                      episode_timeout_s=args.timeout_s, postprocessing_timeout_s=args.timeout_s)


if __name__ == '__main__':
    main()
