# Repository layout and data usage

The repository contains one active release line.

```text
core/          backend-neutral contracts and evidence primitives
domains/       native operational backends and tools
runner/        logical and realtime coordinators
baselines/     baseline and LLM agent implementations
evaluation/    scoring, counterfactuals, and statistics
scenarios/operate_v0_58_0/  743 inherited contracts selected by v0.61
scenarios/operate_v0_59_0/  8 inherited additions introduced by v0.59
scenarios/operate_v0_60_0/  13 scenario additions introduced by v0.60
scenarios/operate_v0_61_0/  5 scenario additions introduced by v0.61
sources/       compact source assets and immutable locks
release/operate_v0_61_0/    active source suite and promoted manifest
scripts/       replay, audit, evaluation, merge, and distribution entrypoints
tests/         current runtime and release-contract tests
docs/          current design and runbooks
```

`works/`, `.audit-cache/`, `.hl/release_rebuild/`, provider
outputs, trajectories, and reports are local/generated and ignored by Git.
Their required hashes and install locations are bound by the promoted manifest
and public HF bundle. The local runtime-companion install root is
`operate_data/`; `MANIFEST.json`
binds the installed bytes to `operate_v0_61_0`. When a bundle declares
`candidate_evidence_archive`, those inputs are restored under
`candidate_evidence/.hl/artifacts/`. The compact v0.61 companion must ship
`candidate_closure.json` instead of that archive. Either form is evidence for
the terminal 2,476-candidate partition, not additional evaluation rows.

The frozen working set contains 769 rows across 502 physical sources;
all 2,476 candidate decisions are terminal and none remain unresolved. These
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
.venv/bin/python scripts/verify_release_integrity.py release/operate_v0_61_0
```

The setup script restores bundle-delivered assets and clones CityLearn,
JSPLIB, OR-Gym, PGLib-UC, SUMO Ingolstadt, and clusterdata at their
manifest-pinned commits. A
`--download-only` bundle check verifies remote bytes, not formal runtime
readiness.

The public dataset exposes one current data state; commits may retain update
history, but superseded files and redundant version tags are not part of the
current public surface. The private maintainer archive is not an evaluation input.
