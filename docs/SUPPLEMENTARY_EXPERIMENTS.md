# Supplementary behavior experiments

OPERATE reports three matched analyses outside the primary
`logical_persistent` score. They use the same executable task, model route,
generation settings, current observation, and scoring contract within each
comparison. A pilot or incomplete pair is engineering evidence, not a paper
result.

## E1: persistent versus stateless context

New E1 runs require `--context-ablation-mode matched_transcript_v1` in both
`logical_persistent` and `logical_stateless` arms. They use the same system
prompt, full current event projection, deterministic derived memory, plan and
decision ledger, agent-owned review cadence, and provider context budget. Only
provider-visible transcript retention differs: the stateless arm starts each
request with system plus the latest event; the persistent arm carries prior
messages while they fit. Both retain an append-only audit ledger. Under a
provider cap, historical messages are dropped before they can force different
current-state compaction between arms.

This measures the added value of transcript continuity **conditional on shared
structured memory**, not memory versus no memory. The default
`--context-ablation-mode none` preserves the ordinary treatment behavior and is
not accepted as this matched E1 design. Historical E1 executions retain their
original identities and must not be pooled as new matched results.

For a new single-task pilot, set the model/route/capability variables and a
fresh output root, then use the same command for both arms (this invokes the
provider and should be run only when that spend is authorized):

```bash
for arm in logical_persistent logical_stateless; do
  uv run python scripts/batch_llm_eval.py \
    --output-dir "${E1_OUTPUT_ROOT:?}/E1/${arm}" \
    --scenario-slice custom --scenarios "${E1_SCENARIO:?}" \
    --models "${E1_MODEL:?}" --api-key-env "${E1_API_KEY_ENV:?}" \
    --base-url-env "${E1_BASE_URL_ENV:?}" --api-mode chat_completions \
    --interaction-mode "$arm" --context-ablation-mode matched_transcript_v1 \
    --model-context-window-tokens "${E1_CONTEXT_TOKENS:?}" \
    --model-max-output-tokens "${E1_MAX_OUTPUT_TOKENS:?}" \
    --temperature 0 --max-tokens 8192 --protocol-repair-max-tokens 4096 \
    --persistent-context-max-chars 48000 --persistent-memory-max-items 64 \
    --persistent-history-max-messages 32 --provider-timeout-s 150 \
    --provider-failure-policy abort --prompt-mode strict --seed-mode scenario \
    --pass-k 1 --max-workers 1 --save-trajectories --no-resume
done
```

Use the same frozen scenario matrix for expansion; the pilot is not a full
matrix. Run artifact/identity acceptance before calculating paired effects.

Report paired changes in primary score, native cost, deadline misses, safety
violations, successful evidence-consuming controls, late-stage failures, model
calls, and tool calls. A pair is invalid when either arm has a provider,
artifact, or prompt-budget failure.

Report the canonical wait-relative primary score and the fixed all-dimension
composite separately; `fixed_score` is the composite, not the primary score.
Also report the applicable-dimension view within each matched task. Do not
compare that adaptive view across unmatched task mixes. Matched E1 is an
independent treatment and cannot be pooled into the main table.

## E3: fixed provider delay

E3 measures sensitivity to decision latency under realtime execution. Each
task uses the same model and task state with fixed 0 s, 1 s, and 5 s delivery
delay. The environment continues while the request is in flight. Report score,
action expiry, supersession, cancellation, takeover, deadline misses, useful
controls, and provider latency. Interpret a task only when all three delay arms
pass the artifact and provider-audit gates.

The producer is `run.py` or `scripts/batch_realtime_llm_eval.py` with
`--response-delivery-delay-s {0,1,5}`. Delay starts after a complete agent
decision, before delivery to the coordinator; it is not a token delay and does
not pause the environment. Cancellation interrupts the delay, and the normal
execution fence prevents stale/canceled responses from acting. Provider audit
records bind ready time, delivery-settlement time, configured delay and
cancellation. The analyzer checks this evidence rather than trusting a label.
Use identical native clocks, routes, generation settings and context budgets
for all three arms. Each arm has a distinct treatment hash and output directory.

A no-provider full-subset preflight is:

```bash
uv run --no-sync python scripts/batch_realtime_llm_eval.py \
  --suite benchmark/formal_runtime_bundle.json \
  --formal-manifest benchmark/manifest.json \
  --output-root .hl/supplementary/E3/preflight \
  --model deepseek-ai/deepseek-v4.1-flash \
  --model-context-window-tokens 1000000 --model-max-output-tokens 32768 \
  --response-delivery-delay-s 5 --dry-run
```

E3 has no primary aggregate. Its result is the matched delay-response curve:
response misses, effected or expired actions, discarded late responses,
superseded turns, takeovers, controlled holds, and protocol-valid responses at
0, 1, and 5 seconds. This keeps transport timing separate from logical task
quality.

## E4: information reveal ablation

New E4 runs use `native_prefix_information_v1`, produced by
`scripts/run_supplementary_e4.py`. This is an explicit, reproducible contract;
old private prefix trials are historical evidence, not interchangeable cells.

Both arms replay the same declared actions from the same Core scenario/seed.
Selected successful **read-only** receipts from that prefix are delivered to
the reveal arm at the first decision; they are withheld from the other arm.
Both arms pay the same prefix tool costs and reach the same native state.
The ordinary logical coordinator then runs the complete remaining episode.
The withhold agent can investigate normally. Authoritative evidence remains
unchanged; the coordinator records which evidence each agent actually saw.

The prefix file declares `prefix_id`, `scenario_path`, `seed`, `prefix_actions`
(serialized Action dictionaries with unique call IDs), `reveal_call_ids`,
`reveal_checks`, and `response_deadline_tick`. Each reveal check names a
`call_id` and a top-level `payload_field`. That field must exist in the native
receipt and be absent throughout the public snapshot. This rejects empty
interventions such as a capacity query whose fields are already public, and
queries that reveal the same fields through the backend's observation filter.
This structural check complements review of the selected field's meaning;
it does not establish decision relevance by itself. Freeze the selection and
its rationale before inspecting model outcomes.

Example native prefix (a source-trace arrival forecast, not synthesized data):

```json
{
  "prefix_id": "alibaba_forecast_t1",
  "scenario_path": "scenarios/datacenter/gpu_cluster_queue_control/time_pressure/basic/alibaba_gpu_w052_377_382_basic.yaml",
  "seed": 94,
  "prefix_actions": [{"actions": [{"name": "forecast_trace_arrivals", "args": {"horizon_ticks": 4}, "call_id": "forecast-1"}]}],
  "reveal_call_ids": ["forecast-1"],
  "reveal_checks": [{"call_id": "forecast-1", "payload_field": "expected_job_count"}],
  "response_deadline_tick": 4
}
```

Prepare a JSON `LLMConfig` using the same verified provider route/capabilities
as the other treatments, with strict prompts, `logical_persistent` and
`provider_failure_policy=abort`. Credentials are named by environment variable,
not embedded in the config. Preflight replays the native prefix twice, verifies
the release row/signature and deterministic state, and makes zero model calls:

```bash
uv run --no-sync python scripts/run_supplementary_e4.py \
  --prefix "$E4_PREFIX_JSON" --llm-config "$E4_CONFIG_JSON" \
  --output-root "$E4_OUTPUT_ROOT/E4/$E4_PREFIX_ID"
```

Only after preflight, repeat with `--execute` to call the model. Execution
rejects changed identities and already-started pairs. Failures retain their
provider/evidence records; partial pairs are not silently resampled. The
summary requires both arms, current schema, matching nonempty identities,
closed model-identity records and no provider failures.

There is no E4 aggregate score. Report investigation calls through the first
control or the end of the predeclared observation window, commit attempts,
visible evidence, provider/tool calls, and component-wise native outcomes over
the episode, including the identical prefix costs in both arms. `response_deadline_tick` is the **exclusive analysis
window boundary**, not a modified backend action deadline. The field
`window_exhausted_after_query` means a query was issued but no control was
attempted before that boundary; it does not prove that query latency caused
the missed opportunity. No control, delayed effects and exhausted windows stay
in the denominator. Ground-truth records are audit-only and never model input.

## Expansion rule

Run one complete model matrix before adding models. Expand to another model only
after every pair or delay group is complete and the shared harness has no open
validity defect. One run per condition is the initial analysis; repeated trials
are reserved for cells where stochastic provider behavior or a variance claim
requires them. Realtime supervision remains separate from the primary logical
leaderboard.

## Acceptance before expansion

Freeze the scientific matrix (scenarios, prefix/window selections, arms, model
route and capabilities) before execution. A summary's `complete` flag covers
only its declared cells; it is not proof that the scientific selection is
adequate. Never shrink the matrix after seeing failed or weak results.

Local fake-provider controls establish implementation behavior, not real
provider compatibility or ranking evidence. Run a bounded real-provider
conformance pilot, then one complete model matrix with strict artifact
acceptance, before expanding to other models. Native-clock results cannot be
replaced by compressed-clock engineering probes. Historical results keep their
original identities and cannot be resumed or pooled into these new treatments.
