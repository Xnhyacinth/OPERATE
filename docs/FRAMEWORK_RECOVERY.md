# Framework recovery and scoring repair

Recovery preserves the original sample. Completed model outcomes stay terminal;
only the actual terminal infrastructure cause can trigger another attempt. A
historical failed request inside an otherwise completed episode is not a reason
to sample the model again.

## Request recovery

The common provider loop recognizes both `httpx` and the locked OpenAI SDK's
`httpx2` transport exceptions. SDK retries remain disabled. A request retry keeps
its messages, tools, generation settings and logical environment unchanged.
Intermediate failures, the retry root and each request/response remain audited.
A partial stream without a valid terminal marker cannot execute an action. A
transport failure after a valid complete generation is recorded separately from
the returned generation, including missing usage information.

The logical batch and campaign support:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `--provider-retry-max-attempts` | 5 | Total wire attempts per pending request, including the first |
| `--provider-retry-max-elapsed-s` | 1800 | Recovery window for waits and starting/reading requests |
| `--episode-checkpoint` | off | Durable replay recovery for saved logical persistent episodes |

For example, a **new** campaign job may explicitly bind:

```json
{
  "provider_retry_max_attempts": 12,
  "provider_retry_max_elapsed_s": 1800,
  "episode_checkpoint": true
}
```

These fields are forwarded to Full and Lite logical commands. The two wire
budgets participate in the treatment hash; checkpoint enablement is bound to run semantics and is immutable
in the output run configuration. Keep the same budgets on every resume. This
example is a configuration choice, not a claim that twelve attempts guarantee
completion or are optimal for a provider.

Numeric and HTTP-date `Retry-After` values cannot be shortened by the local
backoff cap. If the next eligible request falls outside the recovery window,
`ProviderRetryBudgetExhaustedError` exposes `retry_at` and `budget_reason`. The
campaign holds that cell until the deadline and can run other cells. This still
consumes a transport attempt; hard quota failures use a separate pool cooldown.

The synchronous SDK timeout is a socket/request timeout, not an exact wall-clock
kill switch. Recovery checks the window at wait, request and stream-chunk
boundaries and reduces the next SDK timeout to the remaining window. A blocked
socket can settle later, within that finite timeout. No orphan provider thread
is created to simulate hard cancellation.

## Logical episode checkpoint

`--episode-checkpoint` requires `logical_persistent` and saved trajectories. It
uses `runner/checkpoint.py`, outside the realtime wall-clock treatment. It does
not serialize native runtimes or load pickle data.

The journal lives under the batch's `.episode_checkpoints/` directory and is
bound to scenario/seed/pass, suite, implementation, provider treatment and run
semantics. Every successful agent boundary is committed before its returned
action reaches the environment. The journal stores the action, input/schema
hashes, environment and evidence hashes, and explicit JSON agent state. State
changes use deltas; append-only audit/session prefixes are not rewritten for
every boundary. An exclusive cell lock, chained hashes and an atomic head bind
record count and bytes.

On restart the ordinary runner recreates the seeded backend, tool registry,
RNG, pending controls and scheduling state by replaying the same operations.
Recorded agent decisions restore the semantic session, bounded context, memory,
plans, idempotency sequence and provider audit without making provider calls.
Every boundary must match before a new model request is allowed. Model-visible
observations and authoritative evidence must agree, including delayed receipts.
The replay must consume the saved frontier before scoring.

Wrong identity, changed inputs, evidence divergence, a torn journal/head pair,
valid-line truncation or concurrent ownership raises `CheckpointIntegrityError`.
The cell needs repair; it is not restarted as a fresh model sample. The recovery
path is fail-closed for backends that cannot reproduce their saved boundaries.
Native replay still takes simulator time, but it does not repeat completed LLM
requests.

A failed provider boundary is not a successful journal entry. Its audit remains
in that attempt's archived trajectory directory; the continuation restores the
last successful boundary and makes the next pending request. Prior attempts are
retained, rather than silently folded into a successful provider transcript.
Checkpoint progress distinguishes replayed and new boundaries.

`recovery_audit` binds each attempt's immutable provider-audit file and verifies
that its reused request/response prefix is identical. Totals count only newly
attempted requests in each execution, including failed suffixes; simply summing
attempt files would double-count the successful prefix. The ordinary final
LLM statistics describe the retained logical trajectory, as declared by
`provider_request_accounting_scope`. They are not a cumulative billing ledger.
Local preflight/limiter rejections are distinguished from dispatched requests.

Missing parent audits, unmatched prefixes or unsettled request evidence leave
recovery accounting unclosed and block formal eligibility. This includes hard
process deaths whose failed suffix was never persisted. Such a checkpoint may
still reproduce the logical state, but that alone does not certify complete
provider accounting. Historical model, identity or harness failures cannot be
converted into eligible transport recovery by the checkpoint.

This feature cannot retrofit checkpoints into old trajectories. An old GLM or
Hy3 run does not become compatible after a code fix: neither its implementation
hash nor its treatment may be restamped. Use a new output namespace for the new
implementation. No live campaign is migrated automatically.

## Cell status and lifecycle budgets

| Terminal category | Scheduling | Interpretation |
| --- | --- | --- |
| Completed execution/model task failure | Terminal | Measured model outcome; do not resample |
| `provider_transient_error` | Bounded retry / `retry_at` hold | Interrupted provider request |
| `quota_deferred` / `quota_parked` | Pool cooldown | No transport-attempt charge, even if execution had started |
| `harness_error`, `needs_repair=true` | Hold the cell | No measured model outcome; batch remains incomplete |
| `provider_configuration_error` | Stop the route/job | Credential, model identity or provider configuration repair |

Legacy native `ValueError` and related contract errors remain visible as repair
cells. `n_episodes_harness_error` is a subset of the existing error counter;
`pending_after`, `held_count` and `runnable_pending` separate incomplete scope
from dispatchable work. `scope_attempts_closed` is false while repair cells
remain. Artifact-integrity failures under campaign scheduling are reported as
`repair_cells`; valid independent cells may proceed without relaxing a single
artifact check. The standalone strict resume check still raises for damaged
artifacts unless using campaign cell isolation.

The logical campaign supplies `--held-cells` with exact scenario/model/seed/pass
identifiers. Held cells remain in the promised denominator. Realtime campaigns
retain their existing whole-job repair stop and do not use logical replay.

`queue/<lane>/attempts.jsonl` is the append-only lifecycle ledger; `state.json`
is a projection. Reapplying the same execution-attempt ID does not charge it
twice. Initial migration imports archived invocation proof, including attempts
hidden by a prior manual counter reset. Unattributed old counter residuals are
recorded separately as `legacy_unattributed_budget` and held for reconciliation;
they are not invented transport charges or automatically discarded as quota.

A deliberate budget extension records the reason, prior charged count and
unattributed budget, without clearing history:

```bash
uv run --no-sync python scripts/run_eval_campaign.py \
  --config /path/to/campaign/config.json --lane tencent \
  --extend-attempts 2 --job-id JOB_ID \
  --cell-key '["SCENARIO_SLUG", "MODEL", 42, "pass-0"]' \
  --reason 'Transport fix verified; extend the existing lifetime budget'
```

The lane must be idle for an extension; the command uses its existing lock.
Extending a transport budget never clears a deterministic `needs_repair` hold.

## Durable scoring inputs

The episode writes an immutable `*.completed_runtime.json` before counterfactual
postprocessing, preserving final ground truth, evidence, actions, manager state
and provider/session audit. It then writes `*.scoring_inputs.json` before
scoring. Scoring failures retain these bindings in the error row and summary.
Nonfinite native values are preserved with typed JSON markers for diagnosis;
the scorer still rejects them.

CityLearn's `energy_cost` is explicitly signed because native net consumption
cost can include export revenue. Negative values are not clipped. With no
positive comparable reference, optimality gap is N/A with evidence retained;
other objectives keep their nonnegative contract. Existing valid positive-cost
scores and the scoring version remain unchanged.

Recompute only from the complete scoring-input snapshot and its recorded hash:

```bash
uv run --no-sync python -m evaluation.scoring_snapshot \
  --input /path/to/episode.scoring_inputs.json \
  --sha256 RECORDED_SNAPSHOT_SHA256 \
  --output /path/to/new-rescore.json
```

This invokes no backend or LLM. It creates a separate result carrying the source
identity/hash and current scoring implementation, with
`formal_completion_claimed=false`; it refuses to overwrite an existing output.
It does not update `episodes.jsonl` or make an old run leaderboard-eligible.
Snapshot bytes are not rewritten when other trajectory locators become
portable. A completed-runtime snapshot alone needs counterfactual repair first;
a historical failure with neither snapshot cannot use this scorer-only command.


## Public code provenance

Public Full/Lite entrypoints default to `--implementation-policy provenance`:
Git availability, clean checkout and code-tree equality are not execution gates.
Ordinary resume matches scenario/seed/pass, dataset, model and protocol, retains
completed rows after code changes, and records the actual tree on each new row.
Old rows are never restamped. This mode requires `retry-infrastructure`; it does
not repeatedly sample completed model failures. Checkpoints inside unfinished
episodes still use the explicit strict policy and verified replay. Maintainer
qualification artifacts keep their historical identity for reproducibility.

CityLearn storage inspection now reads the last completed native output index.
Its native clock advances to the next input row before that row's output exists;
reading that slot returned false zero SOC/net consumption after charging.
Historical model decisions made from those values require separate remeasurement;
rescoring alone cannot repair their information. The source input clock and
native scoring index are unchanged by this observation correction.
