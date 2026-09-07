# Repository layout and data usage

This public repository ships one current benchmark: 769 Core rows, 193 Lite
rows, scoring `0.15.0`, and the evaluation code needed to run them.

```text
core/          backend-neutral contracts and evidence primitives
domains/       native operational backends and tools
runner/        logical and realtime coordinators
baselines/     baseline and LLM agent implementations
evaluation/    scoring, counterfactuals, and statistics
scenarios/     769 current contracts as `scenarios/<domain>/...`
sources/       compact source assets and immutable locks
benchmark/     current Core, Lite, and runtime-closure catalogs
scripts/       evaluation, install, and verification entrypoints
tests/         runtime and public-catalog tests
docs/          current design and runbooks
```

`works/`, provider outputs, trajectories, and reports are local/generated and
ignored by Git. The Hugging Face companion restores large redistributable
runtime assets into `operate_data/` and `works/`. It does not overlay
`benchmark/` or `scenarios/`.

`benchmark/core_suite.json` and `benchmark/manifest.json` are the evaluation
denominator. Clone this repository and download the public dataset; do not
look for historical `release/` trees or versioned scenario folders.

## Data flow

1. The catalog resolves a canonical scenario YAML.
2. The domain adapter loads lock-bound upstream assets.
3. The seeded executable backend advances independently of the model.
4. The agent observes only the partial observation and tool surface.
5. Typed events, actions, receipts, effects, and scores are appended to evidence.
6. Counterfactual replay uses the same seed with eligible actions removed.
7. Formal outputs bind artifacts by SHA-256 and treatment hash.

Write provider outputs under `batch_results/`. One model/treatment gets one
empty directory; incompatible treatment configs fail closed instead of sharing
checkpoints.

## Clean-clone path

```bash
python -m pip install uv==0.12.5
bash scripts/setup_eval_env.sh
.venv/bin/python scripts/verify_release_integrity.py benchmark
```

The setup script restores bundle-delivered assets and clones CityLearn,
JSPLIB, OR-Gym, PGLib-UC, SUMO Ingolstadt, and clusterdata at their
pinned commits. A `--download-only` bundle check verifies remote bytes, not
formal runtime readiness.

The public GitHub repository and public Hugging Face dataset each expose one
current state. Record the exact 40-character HF commit from
`operate_data/.operate-bundle-owner.json` for a pinned reproduction.
