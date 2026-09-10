# Contributing to OPERATE

## Current Core

New evaluation work targets the current Core: 769 source rows across 502
physical sources qualifying at `0.15.0` and scoring live runs at `0.17.0`. Scenario contracts live under
`scenarios/<domain>/...`. Catalogs live under `benchmark/`.

## Test tiers

Use focused contract tests during normal iteration:

```bash
uv run python -m pytest -q \
  tests/test_event_protocol.py tests/test_task_completion_contracts.py \
  tests/test_operational_agency.py tests/test_batch_llm_eval.py \
  tests/test_verify_release_integrity.py
```

## Setup

Python 3.10–3.14 is supported. Use the lockfile-managed environment:

```bash
python -m pip install uv==0.12.5
uv sync --frozen --python 3.13 --extra dev --extra llm --extra hf
uv run python -m pytest -q tests/test_tool_protocol.py tests/test_evaluation.py
```

Python 3.14 supports the framework and compatible backend slices. Use Python
3.13 for a one-process full-Core environment until CityLearn supports 3.14.

Full backend data lives in gitignored `works/` or a Hugging Face runtime bundle.
See [`data/README.md`](data/README.md) and
[`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md).

## Invariants

1. Simulators own state transitions. LLMs are agents under test.
2. Every Core row is derived from a catalog-declared public source.
3. Domain tools, entities, and stakeholders stay native.
4. Dimension scores need `evidence_ids` or an explicit `applicable=False` reason.
5. Counterfactual replay is a real no-action rerun, or an explicit opt-out.
6. Leaderboard-eligible prompts use `--prompt-mode strict`.
7. All tools go through `core/tool_protocol.py`.
8. Formal outputs stay bound to the exact treatment identity.

## Pull requests

- Keep diffs surgical. Do not reformat unrelated files.
- Runtime/scoring changes and data/corpus changes belong in separate commits.
- Add or update a test for any contract you touch (`tests/test_*.py`).
- Do not commit `works/`, `reports/`, `.audit-cache/`, `.venv/`, or API keys.
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Where to put new work

| Kind | Location |
| --- | --- |
| Backend-agnostic contracts | `core/` |
| Simulator adapters | `domains/<domain>/` |
| Scorer / statistics | `evaluation/` |
| Evaluation entrypoints | `scripts/` |
| Current scenario contracts | `scenarios/<domain>/...` |
| Current catalogs | `benchmark/` |
