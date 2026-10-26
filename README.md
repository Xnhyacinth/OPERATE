<div align="center">
  <img src="assets/operate-banner.png" width="100%" alt="OPERATE — Persistent Operational Agency" />
</div>

# OPERATE

OPERATE evaluates whether an LLM agent can monitor, investigate and control an
operational system over time. Seeded simulators and public source data determine
what happens; the model makes decisions through domain tools.

The benchmark contains **769 Full scenarios across seven domains** and a
**141-scenario Lite development subset**: driving, building energy, datacenter,
logistics, microgrid, power grid and traffic. Lite covers six domains and omits
traffic. It was selected using development-model outcomes and is not a held-out
test or a replacement for Full.

[Documentation](docs/README.md) · [Evaluation](docs/EVALUATION.md) ·
[Data and configuration](docs/DATA_ACCESS.md) · [License](LICENSE)

## Install

Use Python 3.13 for all backends. The framework supports Python 3.10–3.14;
individual simulators have narrower requirements.

```bash
git clone https://github.com/Xnhyacinth/OPERATE.git
cd OPERATE
python -m pip install uv==0.12.5
export OPERATE_HF_REPO_ID='Xnhyacinth/OPERATE'
bash scripts/setup_eval_env.sh --smoke
```

Setup installs the locked environment, downloads the runtime companion, restores
source assets and runs backend checks. Public downloads need no API token.
The resolved dataset revision is recorded in
`operate_data/.operate-bundle-owner.json`; set `OPERATE_HF_REVISION` to that
commit to reproduce the same inputs. See [data provenance](docs/DATA_PROVENANCE.md)
for source licenses and integrity checks.

## Evaluate a model

Supply your provider's credentials and advertised model limits locally:

```bash
export API_KEY='<your-api-key>'
export BASE_URL='<provider-base-url>'
export MODEL_ID='<model-id>'
export CONTEXT_TOKENS='<context-window-tokens>'
export OUTPUT_TOKENS='<maximum-output-tokens>'
export OPERATE_AUTONOMOUS_DRIVING_SUMO_REAL=1
export OPERATE_TRAFFIC_BACKEND_REAL=1

uv run --frozen --no-sync python run_lite.py \
  --models "$MODEL_ID" --output-dir outputs/lite \
  --model-context-window-tokens "$CONTEXT_TOKENS" \
  --model-max-output-tokens "$OUTPUT_TOKENS" \
  --max-tokens "$OUTPUT_TOKENS" \
  --interaction-mode logical_persistent --prompt-mode strict \
  --save-trajectories --resume --dry-run
```

Inspect the dry run, then remove `--dry-run` to execute. For Full evaluation,
provider settings, resume and scoring, follow the
[evaluation runbook](docs/FORMAL_EVALUATION.md).

`logical_persistent` is the primary treatment. The model receives one mission
briefing and subsequently responds to events, tool feedback and its scheduled
reviews. `realtime_persistent` evaluates wall-clock supervision separately.
Installation checks establish runtime availability, not model performance or
formal benchmark certification.

## Repository

| Path | Purpose |
| --- | --- |
| `benchmark/`, `scenarios/` | Current Full/Lite catalogs and executable tasks |
| `core/`, `domains/` | Shared contracts and simulator adapters |
| `runner/`, `baselines/` | Episode execution and reference agents |
| `evaluation/`, `audit/` | Scoring, frozen scoring inputs and evidence checks |
| `data/`, `sources/` | Data utilities, source locks and compact source assets |
| `scripts/`, `tools/` | Setup, evaluation, recovery and data export |
| `docs/`, `examples/`, `tests/` | Usage, examples and regression checks |

Large runtime assets are downloaded separately. Local configuration, caches,
trajectories and results are ignored by Git.

## Contribute and cite

See [contributing](CONTRIBUTING.md) for setup and tests and
[CITATION.cff](CITATION.cff) for citation metadata. Code uses the MIT license;
third-party data and simulators retain their
[upstream terms](THIRD_PARTY_LICENSES.md).
