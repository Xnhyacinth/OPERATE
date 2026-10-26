---
license: other
pretty_name: OPERATE
tags:
  - benchmark
  - llm-agents
  - agentic-evaluation
  - scheduling
  - simulation
  - long-horizon
task_categories:
  - other
configs:
  - config_name: full
    default: true
    data_files:
      - split: test
        path: full/test-00000-of-00001.parquet
  - config_name: lite
    data_files:
      - split: test
        path: lite/test-00000-of-00001.parquet
---

# OPERATE

OPERATE evaluates persistent operational agency using seeded simulator backends
and public source data. The LLM is the agent under test.

The current dataset contains 769 Full tasks across 502 physical sources and
141 Lite tasks across 82 sources. Lite is a development subset selected using
model outcomes; it is not a held-out test or a Full leaderboard substitute.

`full` and `lite` provide the corresponding Parquet configurations. Scenario
YAML, suite metadata and file hashes preserve executable contracts. See
[the schema](PARQUET_SCHEMA.md) for lossless reconstruction.

Set `OPERATE_HF_REPO_ID` to the dataset repository. Runtime setup verifies the
companion's `MANIFEST.json` and records its immutable revision. Large assets
install outside Git. Code and setup instructions are provided in the
[OPERATE repository](https://github.com/Xnhyacinth/OPERATE).

The primary treatment is `logical_persistent`; realtime supervision is separate.
The repository's benchmark manifest records certification status. This dataset
distributes tasks and runtime assets; it contains no model result tables.

## Licenses

The dataset-card value `other` reflects source-specific terms. Distributing this
companion is not relicensing upstream data under MIT. Original simulator,
source-data and attribution terms apply; consult the source locks, runtime
manifest and repository's third-party license index.
