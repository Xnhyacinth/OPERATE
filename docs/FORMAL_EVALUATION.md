# Evaluate a model

Install the runtime with [setup](../README.md#install), then verify the catalog:

```bash
uv run --frozen --no-sync python scripts/verify_release_integrity.py benchmark
```

Full uses `benchmark/core_suite.json` and `benchmark/manifest.json`.
Lite uses `benchmark/lite_suite.json` as a fixed development panel.
Select the same single model throughout each shard.

## Provider settings

Set `API_KEY`, `BASE_URL`, `MODEL_ID`, `CONTEXT_TOKENS` and `OUTPUT_TOKENS`
locally. Context and output limits must match your provider's advertised route.
Only enable reasoning, thinking or streaming controls supported by that route;
`--help` lists the available flags. The public tree contains no provider account
configuration or fixed model roster.

A provider quota remains unknown unless supplied for that account. Omit quota
flags when unknown. When known, bind `--provider-rpm-limit`,
`--provider-rpd-limit` and `--provider-rate-limit-scope` together; processes sharing
an account must share a non-secret scope. Never include credentials in a scope,
command argument or output file.

## Logical evaluation

```bash
export OPERATE_AUTONOMOUS_DRIVING_SUMO_REAL=1
export OPERATE_TRAFFIC_BACKEND_REAL=1

uv run --frozen --no-sync python run_full.py \
  --models "$MODEL_ID" --output-dir outputs/full \
  --api-key-env API_KEY --base-url-env BASE_URL \
  --model-context-window-tokens "$CONTEXT_TOKENS" \
  --model-max-output-tokens "$OUTPUT_TOKENS" \
  --max-tokens "$OUTPUT_TOKENS" \
  --interaction-mode logical_persistent \
  --pass-k 1 --seed-mode scenario --prompt-mode strict \
  --temperature 0 --scheduler-mode global --max-workers 1 \
  --persistent-history-max-messages 64 \
  --persistent-context-max-chars 512000 --persistent-memory-max-items 128 \
  --provider-failure-policy abort --max-consecutive-provider-failures 1 \
  --save-trajectories --resume --finalize --dry-run
```

Review the plan, then remove `--dry-run`. For Lite, use `run_lite.py` and a fresh
output directory with the same provider settings; do not add `--formal-run`.
Full and Lite set their own fixed scenario scope. Their default
`--implementation-policy provenance` records the actual code without requiring
a specific commit or clean checkout.

The persistent profile bounds history to 64 messages, visible context to
512,000 characters and memory to 128 items. The agent receives one mission
briefing and then typed actionable wakeups, feedback, receipts and its scheduled
reviews. `agent_scheduled_v1` disables harness-periodic scans. Unknown events
are non-actionable unless the domain registry declares otherwise.

The command above and the Lite defaults abort provider failures instead of
turning them into a `wait`.
Transport retries are recorded separately. Invalid protocol advances the world
and is counted as an invalid decision, separately from a deliberate `wait`.
Native driving and traffic execution require real SUMO transport; missing
backends must not be replaced with mocks.

## Resume and score

Keep `--resume` and the original output directory to continue a compatible run.
Completed cells retain their original provenance across code updates; unfinished
episode checkpoints still require compatible replay. Dataset, model, provider,
protocol or concurrency changes need a new run scope. See
[recovery](EVALUATION_RECOVERY.md) for interrupted postprocessing.

Use `ranking.primary_score` for live wait-relative ranking. The
[raw trajectory scorer](OFFLINE_SCORING.md) and
[anchored scorer](ANCHORED_SCORING.md) use separate versioned contracts and must
be identified separately. Incomplete evidence retains missing-value bounds;
it must not become a complete ranking.

Merge only compatible formal shards:

```bash
uv run --frozen --no-sync python scripts/merge_formal_llm_shards.py --help
```

The merger verifies dataset, treatment and implementation identities. Readiness
or a successful installation check does not certify completed model evaluation.

## Realtime supervision

`scripts/batch_realtime_llm_eval.py` runs a separate `realtime_persistent`
scorecard with `native_dt_v1`: the plant advances in wall time during requests
and actions. The default speed subset is
`benchmark/realtime_speed_suite.json`. Do not pool its scores into the logical
primary treatment.

Formal realtime execution requires a compatible `--formal-manifest`, which is
not distributed in this repository. Inspect its interface with:

```bash
uv run --frozen --no-sync python scripts/batch_realtime_llm_eval.py --help
```

Public users can evaluate the logical tracks independently with the commands
above. See [interaction](AGENTIC_INTERACTION.md) for realtime evidence semantics.
