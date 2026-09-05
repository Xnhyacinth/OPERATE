---
license: other
license_name: operate-mixed-upstream-licenses
license_link: https://github.com/Xnhyacinth/OPERATE/blob/main/THIRD_PARTY_LICENSES.md
pretty_name: OPERATE
tags:
  - benchmark
  - llm-agents
  - agentic-evaluation
  - scheduling
  - simulation
  - long-horizon
  - timeseries
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

**Benchmarking Persistent Operational Agency in Source-Grounded Executable
Systems**

OPERATE evaluates LLM agents as long-running operational decision centers in
seeded executable systems. The environment, not the model, produces state
transitions. Observations are partial, events and actions are typed, and scores
are linked to recorded evidence.

This public dataset is the runtime companion to the single current state of the
[OPERATE code repository](https://github.com/Xnhyacinth/OPERATE). It is public,
ungated, and intentionally has no selectable public version series. For a
formal run, record the exact 40-character HF commit SHA shown by the Hub.

Updates accumulate as commits; the current tree contains only the latest Full,
Lite and runtime artifacts. See the repository
[update log](https://github.com/Xnhyacinth/OPERATE/blob/main/CHANGELOG.md).

## Benchmark scope

The Core contains 769 scenarios over 502 physical sources and seven domains.
OPERATE-Lite contains 193 exact Core-locked rows from 122 physical sources.
It retains a 104-row coverage core over joint task classes, source families,
event/control mechanisms, native scale and declared source variation. Complete
rounds then increase independent-source support, adding 23, 14 and 13 rows until
the first complete round inside the 150–200-row development budget. All admitted
Autonomous Driving, Building Energy, Microgrid, Power Grid and Traffic rows are
then retained, adding 30 window/condition variants. Datacenter retains all
11 medium and 7 high cases plus its 9 selected basic cases, adding 9 rows;
Logistics remains at 66 selected rows. Every row has an inclusion/exclusion reason.
Selection does not use LLM scores. Core admission supplies the quality
requirement, not the size budget.
All 17 backends, 22 task families, four difficulty levels and six horizon buckets
remain covered. This is a development/ablation subset, not a statistical sample,
a mathematical minimum or the Full/Core leaderboard denominator.

Primary results use `logical_persistent`. `realtime_persistent` is a separate
supervision treatment for proactive monitoring, correct silence, latency,
cancellation, supersession, action lifecycle, and safety takeover.
`logical_stateless` is a compatibility ablation.

## Files

| File | Purpose |
| --- | --- |
| `MANIFEST.json` | Exact file hashes and required runtime/source bindings |
| `release_manifest.json` | Promoted Core and scientific readiness contract |
| `backend_runtime_closure.json` | Runtime packages, archives, links, and external sources |
| `candidate_closure.json` | Terminal disposition of the candidate inventory |
| `full/test-00000-of-00001.parquet` | Self-contained Full scenario contracts and suite metadata |
| `lite/test-00000-of-00001.parquet` | Self-contained Lite scenario contracts and suite metadata |
| `parquet_manifest.json` | Full/Lite Parquet hashes, row counts, and source-suite bindings |
| `backends.tar.zst` | Redistributable native runtime assets |
| `formal_evidence.tar.zst` | Compact qualification metadata with original identity |

Scenario contracts, Full/Lite definitions, source locks, evaluation code, and
install tooling live in GitHub. This companion restores the large redistributed
assets omitted from Git. Assets distributed through their upstream repositories
remain URL- and checksum-bound in the manifests.

The Parquet configurations are independently browsable and reversible. Each
row contains the exact scenario YAML and row metadata; shared suite metadata
is stored once in the original Parquet file, not repeated in the viewer. See the
[Parquet schema](https://github.com/Xnhyacinth/OPERATE/blob/main/docs/hf/PARQUET_SCHEMA.md).
The 22 public columns omit redundant release IDs, track labels, and admission
status fields. Full/Lite is identified by `subset`. Internal reference IDs remain
inside the reversible JSON/YAML payloads and runtime manifests where required
for exact reconstruction and cross-file checks; they are not selectable public
tags. The private archive retains the full maintenance metadata and history.

Historical qualification does not have to match the code of a new independent
evaluation. Each run records its actual implementation; data integrity,
same-run stability and compatible resume/merge remain mandatory. Maintenance
uses affected-scope tests rather than automatically repeating full calibration.

```python
from datasets import load_dataset

full = load_dataset("Xnhyacinth/OPERATE", "full", split="test")
lite = load_dataset("Xnhyacinth/OPERATE", "lite", split="test")
```

## Download and verify

Download the current public snapshot:

```bash
git clone https://github.com/Xnhyacinth/OPERATE.git
cd OPERATE
python -m pip install uv==0.12.5
uv sync --frozen --python 3.13 --extra dev --extra llm --extra hf \
  --extra released-backends --extra simulators

uv run python scripts/download_from_hf.py --download-only
```

Anonymous download is supported; no `HF_TOKEN` is required. The local
`operate_data/` directory is not a selectable benchmark version;
`MANIFEST.json` binds the installed bytes to the current Core, and the local
owner receipt records the resolved immutable HF commit. Pass
`--revision <HF-COMMIT-SHA>` for a pinned reproduction.

The runtime bundle includes only the three byte-exact M5 tables referenced by
Core. Their hashes and 81 scenario bindings are recorded in `MANIFEST.json`.
The original `source_lock.json` is carried separately as metadata, without
adding a fourth physical input. NGSIM bundles also retain their original
`checksums.sha256` lists for native verification.
On 2026-09-04, the dataset publisher confirmed that it holds permission to
redistribute these files. The M5 Competition Rules remain applicable; OR-Gym
code remains MIT-licensed. No `M5_ZIP` or `KAGGLE_TOKEN` is required for the
bundled snapshot.

The immutable release closure retains its admission-time external-acquisition
record. The top-level `source_assets.m5` entry in `MANIFEST.json` is the
authoritative distribution-time overlay for the now-bundled files.

For complete native installation and a baseline runtime check, run:

```bash
bash scripts/setup_eval_env.sh --smoke
```

The setup script acquires the remaining declared sources and runs one
`wait_only` episode per released backend. Required-source and smoke failures
are fatal. This validates runtime/evidence integrity, not model performance;
model evaluation additionally requires your model API credentials.
Unanswered passive-baseline alarms remain reported as policy warnings, not
successful interventions; default strict evaluation is unchanged.

See the GitHub [README](https://github.com/Xnhyacinth/OPERATE#readme) for a
baseline episode and Lite command, and the
[formal evaluation runbook](https://github.com/Xnhyacinth/OPERATE/blob/main/docs/FORMAL_EVALUATION.md)
for treatment-bound runs.

## Intended use and limitations

OPERATE is intended for evaluating persistent operational agency, tool use,
monitoring, and intervention under partial observability. It is not a training
corpus, a safety certification, or evidence that a model is suitable for live
infrastructure control.

- Domain row counts are uneven. Official aggregation is stratified; raw row
  counts do not measure physical-source diversity.
- Lite substantially undersamples Logistics sources and must be reported under
  its own track name.
- Procedural stressors are seeded and labelled; they do not become real source
  events merely because the base state is source-grounded.
- Public reproducibility and leaderboard eligibility are separate. Formal
  provider evaluations and result distribution are still pending, so official
  leaderboard submissions are not open.

## Provenance, licenses, and citation

OPERATE-authored code and metadata are MIT-licensed. This companion is a
mixed-license collection: every redistributed upstream asset retains its own
terms, notices, and per-root manifest binding. The `other` metadata value does
not relicense those assets under MIT. See the
[third-party license inventory](https://github.com/Xnhyacinth/OPERATE/blob/main/THIRD_PARTY_LICENSES.md)
and [data provenance](https://github.com/Xnhyacinth/OPERATE/blob/main/docs/DATA_PROVENANCE.md).

When reporting results, cite the repository plus the exact Git commit, HF
revision, release manifest, treatment hash, and model/provider binding. Citation
metadata is provided in
[`CITATION.cff`](https://github.com/Xnhyacinth/OPERATE/blob/main/CITATION.cff).
