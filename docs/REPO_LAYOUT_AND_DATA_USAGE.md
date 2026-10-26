# Repository and downloaded data

The [root directory table](../README.md#repository) describes the maintained
code. `benchmark/` is the single runtime catalog; `scenarios/` contains its
769 tasks. Frozen scoring inputs are indexed under `evaluation/policies/` and
reuse runtime files when their bytes match.

`sources/` contains small redistributable inputs and upstream source locks.
Setup downloads larger assets into ignored `operate_data/` and `works/`.
The runtime companion's `MANIFEST.json` binds those assets and source licenses.
Do not copy local caches, model outputs or downloaded datasets into Git.

Use [data configuration](DATA_ACCESS.md) to select a dataset revision and
[provenance](DATA_PROVENANCE.md) to inspect its source obligations.
