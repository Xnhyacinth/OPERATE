<div align="center">
  <img src="assets/operate-banner.png" width="100%" alt="OPERATE — Persistent Operational Agency" />

  <p><strong>Benchmarking Persistent Operational Agency in Source-Grounded Executable Systems</strong></p>

  <p>
    <a href="https://github.com/Xnhyacinth/OPERATE/actions/workflows/ci.yml"><img src="https://github.com/Xnhyacinth/OPERATE/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
    <a href="https://huggingface.co/datasets/Xnhyacinth/OPERATE"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Full%20%7C%20Lite-FFD21E" alt="Hugging Face dataset: Full and Lite" /></a>
    <a href="https://huggingface.co/collections/Xnhyacinth/operate-6a9ed166a0c8cd671910ca1a"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Collection-OPERATE-FFD21E" alt="Hugging Face collection: OPERATE" /></a>
    <img src="https://img.shields.io/badge/Full-769%20scenarios-0F766E" alt="Full: 769 scenarios" />
    <img src="https://img.shields.io/badge/Lite-193%20scenarios-0EA5A4" alt="Lite: 193 scenarios" />
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%E2%80%933.14-3776AB?logo=python&logoColor=white" alt="Python 3.10 through 3.14" /></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/Code-MIT-blue" alt="Code license: MIT" /></a>
  </p>
</div>

OPERATE evaluates whether an LLM agent can supervise and control evolving,
partially observable operational systems over long horizons. State transitions
come from seeded executable backends; the model is the agent under test, never
the environment or the judge.

[Documentation](docs/README.md) ·
[Hugging Face data](https://huggingface.co/datasets/Xnhyacinth/OPERATE) ·
[Hugging Face collection](https://huggingface.co/collections/Xnhyacinth/operate-6a9ed166a0c8cd671910ca1a) ·
[Evaluation protocol](docs/FORMAL_EVALUATION.md) ·
[Data provenance](docs/DATA_PROVENANCE.md)

The public repository and Hugging Face dataset each expose one current
benchmark state. Scenario contracts live under `scenarios/<domain>/...`.
Manifests verify dataset and evidence bytes. Public evaluation does not require
a particular Git commit, a clean checkout, or the maintainer code-tree hash.
Code versions are recorded as provenance, not used as a public execution gate. There are no public release tags or
selectable historical datasets.

## Current qualification and provider status

The current Core qualifies at scoring `0.15.0` and scores live runs at `0.17.0`,
769 Core rows across 502 physical sources and 193 Lite rows across 122 sources.
Qualification records describe the code used for their original checks. Public
users can run their own checkout; ordinary resume preserves completed cells
when only code changes, while recording each attempt's actual implementation. Formal logical/realtime provider runs and leaderboard
result publication remain pending.

## What is evaluated

- persistent autonomy after one mission briefing;
- proactive investigation under partial observability;
- correct silence when no intervention is justified;
- typed alarms, scheduled reviews, and environment-driven wakeups;
- replanning, tool use, delayed effects, and counterfactual prevention;
- cancellation, steering, supersession, latency, and safety takeover;
- evidence-linked outcomes across seven operational domains.

`logical_persistent` is the primary leaderboard treatment. The simulator moves
in deterministic logical time while the agent maintains a bounded persistent
session. `realtime_persistent` is an independent supervision scorecard in which
the environment continues while provider requests and actions are in flight.
`logical_stateless` is only a compatibility ablation.

A tick is a causal coordinate for deadlines, delayed actions, attribution, and
replay. It is not a new natural-language prompt. After `session_start`, the
model is invoked only by typed actionable wakeups, scheduled reviews,
investigation results, or lifecycle receipts that require reconciliation.
The canonical `agent_scheduled_v1` policy leaves review timing to the agent,
does not create harness-periodic scans, and keeps unknown events non-actionable.

## Full and Lite

The official Full track contains 769 source-grounded scenarios across 502 physical
sources and seven domains. OPERATE-Lite selects a coverage core and enriches it
with independent physical-source support, then retains every admitted row in
the five smaller domains under the policy below. It
preserves all 17 backends, 22 task families, four difficulty levels, and six
horizon buckets, but it is not a substitute for the Full leaderboard denominator.

| Domain | Full rows | Lite rows | Full sources | Lite sources |
| --- | ---: | ---: | ---: | ---: |
| Autonomous Driving | 7 | 7 | 7 | 7 |
| Building Energy | 18 | 18 | 6 | 6 |
| Datacenter | 142 | 27 | 4 | 4 |
| Logistics | 527 | 66 | 443 | 63 |
| Microgrid | 37 | 37 | 21 | 21 |
| Power Grid | 19 | 19 | 11 | 11 |
| Traffic | 19 | 19 | 10 | 10 |
| **Total** | **769** | **193** | **502** | **122** |

Formal logical/realtime provider runs remain pending; public result release
and leaderboard eligibility remain false.

`benchmark/core_suite.json` and `benchmark/manifest.json` define the public
denominator. Clone this repository, install the Hugging Face runtime companion,
and evaluate against the current catalog. Ordinary resume verifies the dataset,
prompt, scoring protocol and model settings without pinning the code tree.
A checkpoint inside an unfinished episode still requires compatible replay.
See [current benchmark status](docs/CURRENT_RELEASE.md) and the
[formal evaluation runbook](docs/FORMAL_EVALUATION.md).

## Quick start

Python 3.10 through 3.14 is supported by the framework. Use Python 3.13 for a
single-environment full-Core run because of current upstream CityLearn pins.

```bash
git clone https://github.com/Xnhyacinth/OPERATE.git
cd OPERATE
python -m pip install uv==0.12.5
bash scripts/setup_eval_env.sh --smoke
```

This installs the locked environment, downloads and verifies the runtime
companion, restores declared sources, and runs one `wait_only` episode per
released backend. The smoke checks runtime and evidence integrity, not model
performance. Model evaluation additionally requires your model API credentials.
Unanswered alarms from the passive baseline remain visible as policy warnings;
the installation profile does not turn them into successful interventions.
Strict agent/action validation and safety checks remain unchanged.

The public HF download is content-addressed and does not require `HF_TOKEN`.
The installer resolves the current public snapshot once and records its exact
commit in `operate_data/.operate-bundle-owner.json`. For a pinned reproduction,
set `OPERATE_HF_REVISION=<HF-COMMIT-SHA>`. The companion includes the
permission-cleared, byte-exact M5 source tables used by the Full track; no
`M5_ZIP`, `KAGGLE_TOKEN`, or other manual data credential is required. Run
`bash scripts/setup_eval_env.sh` to install without the smoke. Traffic execution is
fail-closed unless `OPERATE_TRAFFIC_BACKEND_REAL=1`, and Autonomous Driving is
fail-closed unless
`OPERATE_AUTONOMOUS_DRIVING_SUMO_REAL=1`; both require a real TraCI/libsumo
transport.

For metadata analysis without installing simulators, install the Hugging Face
`datasets` package in that analysis environment and load either configuration
directly:

```python
from datasets import load_dataset

full = load_dataset("Xnhyacinth/OPERATE", "full", split="test")
lite = load_dataset("Xnhyacinth/OPERATE", "lite", split="test")
```

Each Parquet row contains viewer-friendly metadata and the exact scenario YAML;
the export can reconstruct the bound suite and scenario tree byte for byte.

## Verify and run

```bash
# Framework and persistent-agent contracts
uv run python -m pytest -q \
  tests/test_batch_llm_eval.py \
  tests/test_batch_realtime_llm_eval.py \
  tests/test_agentic_formal.py

# Verify the current public catalog
uv run python scripts/verify_release_integrity.py benchmark

# One baseline episode selected by the current catalog
uv run python run.py \
  --scenario datacenter/gpu_cluster_queue_control/deep_planning/high/alibaba_gpu_native_500_dfc0551ac1_c9da905bb4_high \
  --agent wait_only --seed 42
```

## Run Full / Core

The official Full track is the 769-row catalog in `benchmark/core_suite.json`.
Use `run_full.py`; it expands that suite and does not read a private
qualification manifest.

```bash
OPERATE_TRAFFIC_BACKEND_REAL=1 \
OPERATE_AUTONOMOUS_DRIVING_SUMO_REAL=1 \
uv run python run_full.py \
  --output-dir batch_results/full/my_model \
  --models my-model \
  --api-key-env API_KEY \
  --base-url-env BASE_URL \
  --api-mode chat_completions \
  --interaction-mode logical_persistent \
  --seed-mode scenario --prompt-mode strict \
  --model-context-window-tokens 131072 \
  --model-max-output-tokens 32768 \
  --temperature 0 --max-tokens 32768 \
  --save-trajectories --finalize
```

## Run OPERATE-Lite

`OPERATE-Lite` contains 193 exact Core-locked rows from 122 physical sources.
Core admission and verified YAML identities supply the quality requirement;
selection does not rank cases by any LLM's scores.

First, deterministic coverage selection retains joint task classes, source
families, native event/control mechanisms, scale shapes and declared source
variation: driving hazards/deadlines, microgrid site/forecast/supply conditions,
building event channels, power networks/feeders and controllable traffic topology.
This yields a 104-row coverage core. Then complete rounds increase independent
physical-source support for those features, stopping at the first complete round
inside the 150–200-row development budget. The current rounds add 23, 14 and 13
rows, retaining the entire coverage core. Finally, all admitted rows in
Autonomous Driving, Building Energy, Microgrid, Power Grid and Traffic are
retained, adding 30 window/condition variants. Datacenter additionally retains
all 11 medium and 7 high cases, alongside its 9 selected basic cases; this adds
9 rows. Logistics remains at 66 selected rows. This explicit retention policy
avoids overcompressing already small domains; it is not a quality exception.

The budget is an explicit cost/coverage tradeoff, not a quality threshold or a
reason to discard required coverage, small-domain variants or the scarce
medium/high datacenter cases.
All 17 backends, 22 task families, four difficulty levels and six horizon buckets
remain covered. The suite records inclusion, exclusion and coverage reasons;
it is neither a statistical sample nor a mathematical minimum. Full retains
the complete admitted source/window variation. Lite scores are not Full scores.
Request `--max-tokens` must equal `--model-max-output-tokens` for that route.

```bash
OPERATE_TRAFFIC_BACKEND_REAL=1 \
OPERATE_AUTONOMOUS_DRIVING_SUMO_REAL=1 \
uv run python run_lite.py \
  --output-dir batch_results/lite/my_model \
  --models my-model \
  --api-key-env API_KEY \
  --base-url-env BASE_URL \
  --api-mode chat_completions \
  --interaction-mode logical_persistent \
  --seed-mode scenario --prompt-mode strict \
  --model-context-window-tokens 131072 \
  --model-max-output-tokens 32768 \
  --temperature 0 --max-tokens 32768 \
  --save-trajectories --finalize
```

The exact selection and generator are
[`benchmark/lite_suite.json`](benchmark/lite_suite.json)
and [`tools/build_lite_suite.py`](tools/build_lite_suite.py).

## Formal persistent evaluation

Each exact model is a separate treatment-bound shard. Model ID, provider route,
harness, prompt and context compiler, advertised limits, generation settings,
and dataset define the evaluation treatment. Code hashes are provenance. Requested/effective
concurrency is an immutable run-scope field bound to the output directory and
formal manifest. Incompatible resume attempts fail closed.

Use the [formal evaluation runbook](docs/FORMAL_EVALUATION.md) for exact
per-model capability, provider route, quota, reasoning, concurrency, and dry-run
bindings. Provider failures, output truncation, route fallback, text-only
pseudo-tools, or identity drift remain explicit failures; they are never
converted to `wait`.

Maintenance changes use [affected-scope validation](docs/VALIDATION_POLICY.md).
Public `run_full.py` and `run_lite.py` default to `--implementation-policy provenance`:
no Git checkout or code hash is required to start, continue or finish an ordinary
evaluation. Completed results retain their original code records. Changed datasets,
model settings or scoring protocols still require a separate run. For opt-in
`--episode-checkpoint`, choose `--implementation-policy strict`; it verifies the
exact replay state before continuing inside an unfinished episode.
See [recovery and result validity](docs/FRAMEWORK_RECOVERY.md).

## Evidence and scoring

Every environment event, observation, tool call, action, safety decision,
receipt, effect, cancellation, and supersession is appended to authoritative
artifacts. The model sees a deterministic bounded projection plus structured
memory for unresolved alarms, obligations, facts, commitments, forecasts, and
numeric trends. Compaction never replaces authoritative history.

The scorer emits evidence-linked operational dimensions, including survival,
cost, safety, equity, adaptive replanning, information efficiency, foresight,
optimality gap, counterfactual prevention, and tool-use efficiency. Formal
reports also include domain/backend strata and separate proactive-monitoring,
correct-silence, latency, takeover, and realtime lifecycle diagnostics.

## Repository layout

```text
core/          backend-neutral event, evidence, tool, memory, and safety contracts
domains/       native backend adapters and domain tools
runner/        logical and realtime episode coordinators
baselines/     wait, random, heuristic, oracle, ReAct, Reflexion, and LLM agents
evaluation/    evidence-linked scoring and counterfactual analysis
scenarios/     active OPERATE scenario contracts
sources/       tracked source assets and locks
release/       active source suite and promoted release manifest
scripts/       replay, audit, formal batch, merge, and distribution tools
tests/         runtime and release-contract tests
```

Scenario membership is defined only by the current `core_suite.json`,
`lite_suite.json`, and `manifest.json`. Authoring-era fields retained inside
scenario contracts are provenance, not an additional admission decision.

## Scope and limitations

- Core domains are uneven by design and leaderboard aggregation is stratified;
  row counts must not be interpreted as physical-source diversity.
- Lite is diversity-weighted and substantially undersamples Logistics sources.
  Report it as `OPERATE-Lite`, never as a Full/Core score.
- The one-command installer restores bundled inputs and anonymously resolves
  manifest-pinned upstream runtimes. Every installed byte is hash-checked before
  a formal run; third-party terms remain separate from the code license.
- Official leaderboard eligibility is distinct from public reproducibility and
  remains pending until the bound provider runs and result distribution finish.

## Contributing and citation

See [CONTRIBUTING.md](CONTRIBUTING.md) for changes to code, scenarios, or
backends. Cite the repository and the exact Git commit, HF dataset revision,
release manifest, treatment hash, and model/provider binding used. Citation
metadata is available in [CITATION.cff](CITATION.cff).

Code and benchmark-authored metadata are MIT-licensed. Upstream simulators and
datasets retain their own licenses and acquisition terms.
