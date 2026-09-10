# Repository layout and data usage

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
scenarios/  48 inherited contracts selected by v0.62
scenarios/  8 inherited additions introduced by v0.59
scenarios/  11 selected inherited contracts from v0.60
scenarios/  1 selected inherited contract from v0.61
scenarios/  701 corrected contracts selected by v0.62
sources/       compact source assets and immutable locks
benchmark/    active source suite and promoted manifest
scripts/       replay, audit, evaluation, merge, and distribution entrypoints
tests/         current runtime and release-contract tests
docs/          current design and runbooks
```

`works/`, `.audit-cache/`, `.hl/release_rebuild/`, provider
outputs, trajectories, and reports are local/generated and ignored by Git.
Their required hashes and install locations are bound by the promoted manifest
and private HF bundle. The local runtime-companion install root is
`operate_data/`; that name is a compatibility path, and `MANIFEST.json`
binds the installed bytes to `operate`. When a bundle declares
`candidate_evidence_archive`, those inputs are restored under
`candidate_evidence/.hl/artifacts/`. The compact v0.62 companion must ship
`candidate_closure.json` instead of that archive. Either form is evidence for
the terminal 2,476-candidate partition, not additional evaluation rows.

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
: "${OPERATE_HF_REVISION:?set the immutable private HF commit from the publication receipt}"
bash scripts/setup_eval_env.sh
.venv/bin/python scripts/verify_release_integrity.py benchmark
```

The setup script restores bundle-delivered assets and clones CityLearn,
JSPLIB, OR-Gym, and clusterdata at their manifest-pinned commits. The
Kaggle-gated M5 source additionally requires either `M5_ZIP` or
`KAGGLE_TOKEN` after its competition terms have been accepted. A
`--download-only` bundle check verifies remote bytes, not formal runtime
readiness.

The private GitHub/HF archive is `Xnhyacinth/OPERATE`; the public
current-state distribution is `Xnhyacinth/OPERATE`. Historical maintenance
artifacts remain private and are not additional formal evaluation inputs.
