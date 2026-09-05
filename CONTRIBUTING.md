# Contributing to OPERATE

## Current Core

New evaluation work targets the promoted `operate_v0_61_0` Core: 769 source
rows across 502 physical sources at scoring `0.14.0`. Historical release and
provider artifacts are not current inputs.

## Test tiers

Use focused contract tests during normal iteration:

```bash
uv run python -m pytest -q \
  tests/test_event_protocol.py tests/test_task_completion_contracts.py \
  tests/test_operational_agency.py tests/test_batch_llm_eval.py \
  tests/test_verify_release_integrity.py
```

Run the focused runtime, persistent-agent, replay, and release-contract suites
before a release or formal evaluation.

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

Changes to runtime, scoring, or release artifacts must preserve these project
invariants:

1. Simulators own state transitions. LLMs are agents under test.
2. Every Core row is derived from a manifest-declared public source.
3. Domain tools, entities, and stakeholders stay native. No emergency-schema leakage.
4. Dimension scores need `evidence_ids` or an explicit `applicable=False` reason.
5. Counterfactual replay is a real no-action rerun, or an explicit opt-out.
6. Leaderboard-eligible prompts use `--prompt-mode strict`.
7. All tools go through `core/tool_protocol.py`.
8. Formal outputs stay bound to the exact treatment and release identity.

## Pull requests

- Keep diffs surgical. Do not reformat unrelated files.
- Runtime/scoring changes and data/corpus changes belong in separate commits.
- Add or update a test for any contract you touch (`tests/test_*.py`).
- Do not commit `works/`, `reports/`, `.audit-cache/`, `.venv/`, or API keys.
- Do not describe `pglib_uc_synthetic` as a digital twin or as solving power flow.

## Where to put new work

| Kind | Location |
| --- | --- |
| Backend-agnostic contracts | `core/` |
| Simulator adapters | `domains/<domain>/` |
| Scorer / statistics | `evaluation/` |
| Replay and release tools | `scripts/` |
| New v0.61 scenario contracts | `scenarios/operate_v0_61_0/` |
| Inherited active contracts | `scenarios/operate_v0_58_0/` (743 manifest-selected paths) |
| Earlier active additions | `scenarios/operate_v0_59_0/` (8 paths), `scenarios/operate_v0_60_0/` (13 paths) |
| Active release | `release/operate_v0_61_0/` |
