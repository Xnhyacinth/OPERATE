<div align="center">
  <img src="assets/operate-banner.png" width="100%" alt="OPERATE — Persistent Operational Agency" />

  <p><strong>Benchmarking Persistent Operational Agency in Source-Grounded Executable Systems</strong></p>

  <p>
    <a href="https://github.com/Xnhyacinth/OPERATE/actions/workflows/ci.yml"><img src="https://github.com/Xnhyacinth/OPERATE/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
    <a href="https://huggingface.co/datasets/Xnhyacinth/OPERATE"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Full%20%7C%20Lite-FFD21E" alt="Hugging Face dataset: Full and Lite" /></a>
    <img src="https://img.shields.io/badge/Full-769%20scenarios-0F766E" alt="Full: 769 scenarios" />
    <img src="https://img.shields.io/badge/Lite-159%20scenarios-0EA5A4" alt="Lite: 159 scenarios" />
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
[Evaluation protocol](docs/FORMAL_EVALUATION.md) ·
[Data provenance](docs/DATA_PROVENANCE.md)

The public repository and Hugging Face dataset each expose one versionless,
current benchmark state; the private maintainer archive retains development
history. There are no public release tags or selectable historical datasets.
Internal release IDs and content hashes remain in manifests solely to bind
code, data, prompts, treatments, and results reproducibly.

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
sources and seven domains. OPERATE-Lite retains every row from the five smaller
domains and applies diversity-aware caps only to Datacenter and Logistics. It
preserves all 17 backends, 22 task families, four difficulty levels, and six
horizon buckets, but it is not a substitute for the Full leaderboard denominator.

| Domain | Full rows | Lite rows | Full sources | Lite sources |
| --- | ---: | ---: | ---: | ---: |
| Autonomous Driving | 7 | 7 | 7 | 7 |
| Building Energy | 18 | 18 | 6 | 6 |
| Datacenter | 142 | 24 | 4 | 4 |
| Logistics | 527 | 35 | 443 | 29 |
| Microgrid | 37 | 37 | 21 | 21 |
| Power Grid | 19 | 19 | 11 | 11 |
| Traffic | 19 | 19 | 10 | 10 |
| **Total** | **769** | **159** | **502** | **88** |

Candidate closure is complete: all 2,476 independent candidates have a terminal
disposition and none remain unresolved. The manifest-bound twelve-stage replay
promoted all 769 source rows and marks the release formal-evaluation ready.
The public Hugging Face dataset exposes only the current snapshot. Formal runs
record its resolved immutable commit in the local owner receipt. Formal provider
runs are still in progress, so the published code/data are ready for independent
evaluation while official leaderboard eligibility remains false.

The promoted `core_suite.json` and `manifest.json` define the formal denominator.
The source suite remains an auditable replay and provenance input. No
trajectory produced under an earlier namespace, tree, prompt profile, or
treatment hash may be resumed or merged.
See [current release status](docs/CURRENT_RELEASE.md) and the
[formal evaluation runbook](docs/FORMAL_EVALUATION.md).

## Quick start

Python 3.10 through 3.14 is supported by the framework. Use Python 3.13 for a
single-environment full-Core run because of current upstream CityLearn pins.

```bash
git clone https://github.com/Xnhyacinth/OPERATE.git
cd OPERATE
python -m pip install uv==0.12.5
uv sync --frozen --python 3.13 --extra dev --extra llm --extra hf \
  --extra released-backends --extra simulators

uv run python scripts/download_from_hf.py --download-only
```

The public HF download is content-addressed and does not require `HF_TOKEN`.
The command resolves the current public snapshot once and records its exact
commit in `operate_data/.operate-bundle-owner.json`. For a pinned reproduction,
pass `--revision <HF-COMMIT-SHA>` explicitly. The companion includes the
permission-cleared, byte-exact M5 source tables used by the Full track; no
`M5_ZIP`, `KAGGLE_TOKEN`, or other manual data credential is required. Run
`bash scripts/setup_eval_env.sh` to download, verify, restore all source/runtime
assets, and install the locked environment. Traffic execution is
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

# Verify the promoted release
uv run python scripts/verify_release_integrity.py \
  release/operate_v0_61_0 --portable

# One baseline episode selected by the current manifest
uv run python run.py \
  --scenario operate_v0_58_0/datacenter/gpu_cluster_queue_control/deep_planning/high/alibaba_gpu_native_500_dfc0551ac1_c9da905bb4_high \
  --agent wait_only --seed 42
```

## Run OPERATE-Lite

`OPERATE-Lite` is a policy-derived 159-row efficiency/development track built
only from Core-locked rows. A domain is downsampled only when its row count is
greater than twice the median domain count; this identifies Datacenter and
Logistics without naming them in the selector. Every other Core row is retained.
Within each overrepresented domain, the selector keeps up to three diverse rows
per non-empty backend × family × difficulty × horizon stratum, prioritizing new
physical sources and semantic/structural fingerprints. The resulting suite
retains all 17 backends, 22 task families, four difficulty levels, and six
horizon buckets. Lite scores must not be reported as Full leaderboard scores.

```bash
OPERATE_TRAFFIC_BACKEND_REAL=1 \
OPERATE_AUTONOMOUS_DRIVING_SUMO_REAL=1 \
uv run python run_lite.py \
  --output-dir batch_results/lite/my_model \
  --models my-model \
  --api-key-env MY_API_KEY \
  --base-url-env MY_BASE_URL \
  --api-mode chat_completions \
  --interaction-mode logical_persistent \
  --seed-mode scenario --prompt-mode strict \
  --model-context-window-tokens 131072 \
  --model-max-output-tokens 32768 \
  --temperature 0 --max-tokens 32768 \
  --save-trajectories --finalize
```

The exact selection and generator are
[`release/operate_v0_61_0/lite_suite.json`](release/operate_v0_61_0/lite_suite.json)
and [`tools/build_lite_suite.py`](tools/build_lite_suite.py).

## Formal persistent evaluation

Each exact model is a separate treatment-bound shard. Model ID, provider route,
harness, prompt and context compiler, advertised limits, generation settings,
and release tree are bound by the agent treatment hash. Requested/effective
concurrency is an immutable run-scope field bound to the output directory and
formal manifest. Incompatible resume attempts fail closed.

Use the [formal evaluation runbook](docs/FORMAL_EVALUATION.md) for exact
per-model capability, provider route, quota, reasoning, concurrency, and dry-run
bindings. Provider failures, output truncation, route fallback, text-only
pseudo-tools, or identity drift remain explicit failures; they are never
converted to `wait`.

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
