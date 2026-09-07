# OPERATE data

The promoted `operate_v0_62_0` source suite contains 769 rows over 502 physical
sources. Its parent candidate ledger is terminal (2,476 independent candidates,
0 unresolved). The manifest selects 701 corrected contracts under
`scenarios/operate_v0_62_0/`, 48 inherited contracts under
`scenarios/operate_v0_58_0/`, 8 under `scenarios/operate_v0_59_0/`, 11 under
`scenarios/operate_v0_60_0/`, and 1 under `scenarios/operate_v0_61_0/`.
Compact
redistributable source assets live under `sources/`. Only an atomically
promoted `core_suite.json` plus its matching `manifest.json` defines the formal
release denominator.

The public Hugging Face repository is `Xnhyacinth/OPERATE`. A published bundle
is authoritative only when its `MANIFEST.json` passes integrity verification and
the release manifest reports `formal_evaluation_ready=true`.
The downloader resolves the public snapshot to an immutable HF commit and
records it in the local owner receipt. Use that recorded commit, not a mutable
branch name, as the formal runtime binding.

## Reproduce the environment

```bash
python -m pip install uv==0.12.5
uv sync --frozen --python 3.13 --extra released-backends --extra llm --extra hf
```

Python 3.10–3.14 is supported by the framework. Use Python 3.13 for the complete
Core because CityLearn 2.5.0 does not currently support Python 3.14.

## Download and verify the runtime companion

```bash
uv run python scripts/download_from_hf.py --download-only
```

`operate_data/` is the stable local install root. The directory name is not the
release ID; `MANIFEST.json` binds those bytes to
`operate_v0_62_0` and rejects cross-release reuse. The historical
`data_operate_v058` compatibility path remains available for inherited assets.

The downloader resolves the current public snapshot once and records its exact
commit in the owner receipt. Pass `--revision <HF-COMMIT-SHA>` to reproduce a
specific published snapshot.

The HF download is a manifest-bound runtime companion to the Git checkout, not
a standalone portable-Core tree. It restores formal replay evidence and the
declared redistributable runtime assets that are intentionally not tracked by
Git.

For a complete native installation, run `bash scripts/setup_eval_env.sh`.
Native execution also requires each row's declared backend runtime. Traffic
requires the official SUMO runtime and `OPERATE_TRAFFIC_BACKEND_REAL=1`.

## Provenance boundary

File presence is not provenance. Formal admission requires evidence that locked
source bytes drive backend state transitions. See
[`../docs/DATA_PROVENANCE.md`](../docs/DATA_PROVENANCE.md).

OPERATE code and benchmark-authored metadata are MIT-licensed. Upstream data and
simulators retain their own licenses and acquisition terms.
