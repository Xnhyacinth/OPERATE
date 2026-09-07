# Repository layout and data usage

The historical `data_operate_v058/` compatibility path remains available for
inherited runtime assets. It is not the active v0.62 release denominator.

The parent v0.61 admission ledger records 2,476 terminal candidate decisions
with zero unresolved. v0.62 introduces zero newly mined candidates; it preserves
that historical lineage while qualifying corrected contracts for the same
769-row, 502-source denominator. Historical admission evidence is not relabelled
as newly executed evidence.

The repository contains one active release line.

```text
core/          backend-neutral contracts and evidence primitives
domains/       native operational backends and tools
runner/        logical and realtime coordinators
baselines/     baseline and LLM agent implementations
evaluation/    scoring, counterfactuals, and statistics
scenarios/     769 current contracts as `scenarios/<domain>/...`
sources/       compact source assets and immutable locks
benchmark/     current Core, Lite, and runtime-closure catalogs
scripts/       replay, evaluation, and distribution entrypoints
tests/         current runtime and release-contract tests
docs/          current design and runbooks
```

`works/`, `.audit-cache/`, local rebuild caches, provider
outputs, trajectories, and reports are local/generated and ignored by Git.
Their required hashes and install locations are bound by the promoted manifest
and public HF bundle. The local runtime-companion install root is
`operate_data/`; `MANIFEST.json` records the installed runtime companion.
Compact companions restore binaries under `operate_data/` and `works/`; they
do not overlay `benchmark/`.

The frozen working set contains 769 rows across 502 physical sources;
the parent admission ledger has 2,476 terminal decisions and none remain unresolved. These
are candidate-closure facts, not a formal denominator. The matching promoted
`core_suite.json` and `manifest.json` remain authoritative for evaluation.

## Data flow

1. The source suite resolves a canonical scenario YAML.
2. The domain adapter loads manifest-locked upstream assets.
3. The seeded executable backend advances independently of the model.
4. The agent observes only the partial observation and tool surface.
5. Typed events, actions, receipts, effects, and scores are appended to evidence.
6. Counterfactual replay uses the same seed with eligible actions removed.
7. Formal outputs bind all artifacts by SHA-256 and treatment hash.

Do not write provider outputs into `release/`. One model/treatment gets one
empty directory under `batch_results/`; incompatible treatment configs fail
closed instead of sharing checkpoints.

## Clean-clone path

```bash
python -m pip install uv==0.12.5
bash scripts/setup_eval_env.sh
.venv/bin/python scripts/verify_release_integrity.py benchmark
```

The setup script restores bundle-delivered assets and clones CityLearn,
JSPLIB, OR-Gym, PGLib-UC, SUMO Ingolstadt, and clusterdata at their
manifest-pinned commits. A
`--download-only` bundle check verifies remote bytes, not formal runtime
readiness.

The public dataset exposes one current data state; commits may retain update
history, but superseded files and redundant version tags are not part of the
current public surface. The private maintainer archive is not an evaluation input.
