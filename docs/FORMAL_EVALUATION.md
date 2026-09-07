# OPERATE formal evaluation

This runbook applies to the promoted 769-row `operate_v0_62_0` dataset and
records the actual implementation used by each new run. The `core_suite.json` and
`manifest.json`, not a pre-admission source suite, define a formal shard. Any
older artifacts and interrupted provider runs are not compatible checkpoints.

## Contract

- `logical_persistent` is the primary leaderboard treatment.
- `realtime_persistent` is a separate supervision scorecard and is not pooled
  into the thirteen-dimension primary score.
- One complete model shard is a valid formal unit. There is no fixed model
  roster or three-repeat gate; `pass_k` is an explicit reliability stratum.
- The simulator advances state. A tick is a causal coordinate, not a repeated
  natural-language prompt. After the mission briefing, the model is called
  only for typed wakeups, actionable feedback, scheduled reviews, or required
  receipt reconciliation.
- Every run is bound to the exact release manifest, model/provider route,
  harness, prompt and context profile, generation limits and implementation
  tree through the agent treatment hash. Concurrency is separately immutable
  in the run scope and output namespace.
- The canonical `agent_scheduled_v1` wakeup policy gives review scheduling to
  the agent, disables harness-periodic scans, delivers typed actionable events,
  and treats unknown events as non-actionable.

Formal startup verifies dataset/source integrity, compatible scoring contracts,
backend assets, provider settings and the run namespace. Historical qualification
proof is checked against its original snapshot, not required to match the code
of a new independent run. That run binds its actual current implementation;
resume and merge still reject incompatible execution identities. A maintenance
fix does not require repeating the full native qualification pipeline.
See [Validation policy](VALIDATION_POLICY.md) for affected-scope tests.

## Runtime bootstrap

Resolve the current public runtime once, then use the exact revision recorded
in `operate_data/.operate-bundle-owner.json`. A branch such as `main` is not a
formal binding because it is mutable.

```bash
python -m pip install uv==0.12.5
uv sync --frozen --python 3.13 --extra dev --extra llm --extra hf \
  --extra released-backends --extra simulators
uv run python scripts/download_from_hf.py --download-only
export OPERATE_HF_REVISION="$(uv run python - <<'PY'
import json
from pathlib import Path

receipt = json.loads(Path("operate_data/.operate-bundle-owner.json").read_text())
print(receipt["revision"])
PY
)"
```

The private maintainer archive and its post-provider CAS publication workflow
are not required to run an independent shard.

## Environment

Use Python 3.13 for a full seven-domain run because the current CityLearn pin is
not available on Python 3.14. The framework and compatible backend slices are
supported on Python 3.10 through 3.14.

```bash
: "${OPERATE_HF_REVISION:?set the resolved immutable HF revision}"
bash scripts/setup_eval_env.sh
export OPERATE_FORMAL_MANIFEST='release/operate_v0_62_0/manifest.json'
export OPERATE_FORMAL_READINESS="$(
  .venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

manifest = json.loads(Path(os.environ["OPERATE_FORMAL_MANIFEST"]).read_text())
print(manifest["formal_evidence"]["readiness"])
PY
)"
test -f "$OPERATE_FORMAL_READINESS"
.venv/bin/python scripts/verify_release_integrity.py \
  release/operate_v0_62_0
```

`setup_eval_env.sh` installs the locked environment, clones the six
manifest-declared external source repositories at pinned commits, and restores
the public runtime companion at `OPERATE_HF_REVISION`. The companion includes
the permission-cleared, byte-exact M5 tables used by Full; no separate Kaggle
credential is required. A `download_from_hf.py --download-only` check verifies
bundle bytes only; it does not establish a runnable formal environment.

Traffic rows additionally require the declared SUMO assets and
`OPERATE_TRAFFIC_BACKEND_REAL=1`. Autonomous-driving rows require the
manifest-bound NGSIM/SUMO source bundle and
`OPERATE_AUTONOMOUS_DRIVING_SUMO_REAL=1`. Do not replace a missing native
backend with a mock or emulated fallback in a formal shard.

## Provider shards

Start a provider shard only after the dataset/runtime integrity check passes.
This reads and verifies existing artifacts; it does not execute calibration.

The examples read the credential from `API_KEY` and the OpenAI-compatible
route from `BASE_URL`. Export both securely before executing a shard; no
maintainer shell configuration is required. Never place the credential value
in command arguments, config, trajectories, or logs. Request
`--max-tokens` equals that route's advertised maximum output, with a separate
8,192-token protocol-repair budget. Formal logical and realtime shards use
`provider_failure_policy=abort` with a one-failure circuit threshold: a failed
provider/tool-calling decision terminates the episode and is never converted to
an environment-advancing `wait`. Before that circuit policy applies, the harness
may retry the identical wire request at most four times for transport-only 429
or 5xx failures, using deterministic 5/10/20/40-second backoff and a bounded
provider `Retry-After` value. Every attempt independently reserves quota and has
one request/response audit record. Content, protocol, context-budget, hard-quota,
and model-identity failures are not transient retries.

| Model | Route context / max output | Request output | Workers | Effort / thinking |
| --- | ---: | ---: | ---: | --- |
| `hy3-ioa` | 192,000 / 64,000 | 64,000 | 16 Full | `native` / `high`, thinking `enabled` |
| `gpt-5.6-luna` | 272,000 / 128,000 | 128,000 | 12 Lite | `native` / `high`, thinking omitted |
| `deepseek-v4-flash-ioa` | 1,000,000 / 50,000 | 50,000 | 12 Lite | `native` / `high`, thinking `enabled` |
| `glm-5.3-flash-ioa` | 1,000,000 / 128,000 | 128,000 | 12 Lite | `native` / `high`, thinking `enabled` |

Keep this table in sync with `docs/provider_route_bindings.json`. That file is
the append-only record of each route's envelope, request `--max-tokens`,
thinking controls, and worker counts. The OpenRouter example below remains
`z-ai/glm-5.2:free` at 256,000 / 230,400 with 8 Full workers.

These are explicit example bindings, not auto-detected provider guarantees.
Verify your exact route's advertised limits and account quota before execution;
different bindings require a new treatment and output namespace.

The promoted v0.62 Core contains no Grid2Op-backed row, so the manifest-required
global scheduler may use the provider-specific worker counts above. If a later
Core contains any Grid2Op row, formal startup fails closed for more than one
worker; that release must use a new output namespace with `max-workers=1`.
A different worker count is always a different run scope and is not a
compatible formal checkpoint.

`tool_choice=auto` is the formal cross-provider profile. This does not permit a
model to evade an action-required decision: the protocol validator still
rejects a text-only/non-executable decision. Reasoning effort is never inferred
from the provider name; a shard either binds an advertised effort explicitly or
omits the field.

Declared provider quotas are part of the treatment. Free-model shards using
the same account must use a non-secret shared scope such as `provider-free-shared`
and its shared 20 RPM limiter. Bind the account's applicable free-tier daily
allowance explicitly: 50 RPD when the account has purchased fewer than 10
credits, or 1,000 RPD once that threshold is met. When an account-level limit is
unknown, the formal commands below omit RPM, RPD, and quota scope instead of
inventing a provider guarantee. The resulting `null` values are still
treatment-bound; structured 429 responses and retry outcomes remain auditable.
Scope names identify quota pools and never contain API-key bytes. When a limit
is declared, the limiter state defaults to `~/.cache/operate/provider-rate-limits`;
all processes reserve a slot under a file lock after request-budget preflight and
before provider transport.

Provider result publication does not change the private repository visibility.

An RPM reservation can add a short audited wait. Exhausting RPD raises a
`ProviderQuotaExhaustedError`, writes a resumable sentinel with the next UTC-day
reset, and parks remaining cells instead of blocking workers for hours. Changing
the RPM, RPD, or scope requires a fresh output directory because all three fields
are treatment- and run-config-bound.

The following commands describe separate example shards; there is no required
provider order or fixed model roster. Put the public OpenAI-compatible endpoint
in `BASE_URL` and read the credential only from `API_KEY`. Never merge limiter
scopes, trajectories, or treatment hashes across accounts.

Set the account-specific daily cap for a free-tier example; do not copy one of
the documented allowance strata without checking which one applies to the
account.

```bash
export BASE_URL='https://openrouter.ai/api/v1'
export OPERATE_TRAFFIC_BACKEND_REAL=1
export OPERATE_AUTONOMOUS_DRIVING_SUMO_REAL=1
: "${RPD_LIMIT:?set the applicable free-tier RPD limit}"

PYTHONPATH=. .venv/bin/python scripts/batch_llm_eval.py \
  --output-dir batch_results/operate_v0_62_0/formal/logical_persistent/glm_5_2_free_w8 \
  --formal-manifest "$OPERATE_FORMAL_MANIFEST" \
  --models z-ai/glm-5.2:free \
  --api-key-env API_KEY --base-url-env BASE_URL \
  --api-mode chat_completions --stream-chat-completions \
  --model-context-window-tokens 256000 \
  --model-max-output-tokens 230400 \
  --interaction-mode logical_persistent \
  --pass-k 1 --seed-mode scenario --prompt-mode strict \
  --temperature 0 --reasoning-effort high \
  --max-tokens 230400 --protocol-repair-max-tokens 8192 \
  --provider-timeout-s 300 \
  --provider-rpm-limit 20 \
  --provider-rpd-limit "$RPD_LIMIT" \
  --provider-rate-limit-scope provider-free-shared \
  --persistent-history-max-messages 64 \
  --persistent-context-max-chars 512000 \
  --persistent-memory-max-items 128 \
  --scheduler-mode global --max-workers 8 \
  --save-trajectories --resume --formal-run --finalize --dry-run
```

A second route has a separate complete command. Concurrency is fixed at 16.
Account quota remains unknown for that route, so the command omits RPM, RPD, and
quota scope. If the account later publishes or returns an account-specific
limit, start a new treatment namespace with all three quota arguments supplied
together.

```bash
export BASE_URL='https://copilot.tencent.com/v2'
export OPERATE_TRAFFIC_BACKEND_REAL=1
export OPERATE_AUTONOMOUS_DRIVING_SUMO_REAL=1

PYTHONPATH=. .venv/bin/python scripts/batch_llm_eval.py \
  --output-dir batch_results/operate_v0_62_0/formal/logical_persistent/hy3_ioa_w16 \
  --formal-manifest "$OPERATE_FORMAL_MANIFEST" \
  --models hy3-ioa \
  --api-key-env API_KEY --base-url-env BASE_URL \
  --api-mode chat_completions --stream-chat-completions \
  --model-context-window-tokens 192000 \
  --model-max-output-tokens 64000 \
  --interaction-mode logical_persistent \
  --pass-k 1 --seed-mode scenario --prompt-mode strict \
  --temperature 0 --reasoning-effort high \
  --reasoning-effort-format native --thinking-type enabled \
  --max-tokens 64000 --protocol-repair-max-tokens 8192 \
  --provider-timeout-s 300 \
  --persistent-history-max-messages 64 \
  --persistent-context-max-chars 512000 \
  --persistent-memory-max-items 128 \
  --scheduler-mode global --max-workers 16 \
  --save-trajectories --resume --formal-run --finalize --dry-run
```

Lite is not a Full formal shard. The native GLM-5.3-Flash Lite command uses
`run_lite.py`; do not add `--formal-run`.

```bash
export BASE_URL='https://copilot.tencent.com/v2'
export OPERATE_TRAFFIC_BACKEND_REAL=1
export OPERATE_AUTONOMOUS_DRIVING_SUMO_REAL=1

PYTHONPATH=. .venv/bin/python run_lite.py \
  --output-dir batch_results/operate_v0_62_0/lite/logical_persistent/glm_5_3_flash_ioa_w12 \
  --lite-suite release/operate_v0_62_0/lite_suite.json \
  --models glm-5.3-flash-ioa \
  --api-key-env API_KEY --base-url-env BASE_URL \
  --api-mode chat_completions --stream-chat-completions \
  --model-context-window-tokens 1000000 \
  --model-max-output-tokens 128000 \
  --interaction-mode logical_persistent \
  --pass-k 1 --seed-mode scenario --prompt-mode strict \
  --temperature 0 --reasoning-effort high \
  --reasoning-effort-format native --thinking-type enabled \
  --max-tokens 128000 --protocol-repair-max-tokens 8192 \
  --provider-timeout-s 300 \
  --persistent-history-max-messages 64 \
  --persistent-context-max-chars 512000 \
  --persistent-memory-max-items 128 \
  --scheduler-mode global --max-workers 12 \
  --save-trajectories --resume --finalize --dry-run
```

The logical dry-run is not filesystem-empty: it writes `run_config.json`,
`batch_run.log`, `logs/`, and, because trajectories are required,
`trajectories/` when creating a new preview namespace, under the normal output
lock. Inspecting an existing compatible namespace is read-only: it does not
overwrite configuration, repair journals, append logs or rewrite path maps. After
reviewing the resolved treatment hash, model identity, provider route,
capabilities and manifest-selected formal scope, the same output namespace may
be reused by removing only `--dry-run` while keeping every immutable run-scope
field identical and retaining `--resume`. Disabling resume refuses an
initialized namespace rather than truncating its journal. A capability, quota,
scope, scheduler, or concurrency change requires a new output directory; an
incompatible existing `run_config.json` fails closed. Rate limiting, route
fallback, truncated output, text-only tool responses, or provider identity
drift are recorded as failures; the runner does not silently convert them to
`wait`.

## Realtime supervision shards

Realtime uses the same 769 manifest-selected rows but writes a separate
supervision scorecard. The suite argument is the manifest-bound readiness
artifact restored by the runtime companion, not an independently selected
scenario list. Its dry-run validates the treatment and prints the derived hash
directory without creating it; remove only `--dry-run` to execute and retain
`--resume` for recovery.

OpenAI-compatible GLM:

```bash
export OPERATE_TRAFFIC_BACKEND_REAL=1
export OPERATE_AUTONOMOUS_DRIVING_SUMO_REAL=1
: "${RPD_LIMIT:?set the applicable free-tier RPD limit}"

PYTHONPATH=. .venv/bin/python scripts/batch_realtime_llm_eval.py \
  --suite "$OPERATE_FORMAL_READINESS" \
  --formal-manifest "$OPERATE_FORMAL_MANIFEST" \
  --output-root batch_results/operate_v0_62_0/formal/realtime_persistent/glm_5_2_free_w8 \
  --model z-ai/glm-5.2:free --provider openai_compatible \
  --base-url https://openrouter.ai/api/v1 --api-key-env API_KEY \
  --api-mode chat_completions \
  --model-context-window-tokens 256000 \
  --model-max-output-tokens 230400 \
  --max-tokens 230400 --protocol-repair-max-tokens 8192 \
  --persistent-history-max-messages 64 \
  --persistent-context-max-chars 512000 \
  --persistent-memory-max-items 128 \
  --provider-timeout-s 300 \
  --provider-rpm-limit 20 \
  --provider-rpd-limit "$RPD_LIMIT" \
  --provider-rate-limit-scope provider-free-shared \
  --tick-interval-s 5 --termination-grace-s 5 \
  --max-workers 8 --pass-k 1 --reasoning-effort high \
  --resume --dry-run
```

hy3-ioa:

```bash
export OPERATE_TRAFFIC_BACKEND_REAL=1
export OPERATE_AUTONOMOUS_DRIVING_SUMO_REAL=1

PYTHONPATH=. .venv/bin/python scripts/batch_realtime_llm_eval.py \
  --suite "$OPERATE_FORMAL_READINESS" \
  --formal-manifest "$OPERATE_FORMAL_MANIFEST" \
  --output-root batch_results/operate_v0_62_0/formal/realtime_persistent/hy3_ioa_w16 \
  --model hy3-ioa --provider openai_compatible \
  --base-url https://copilot.tencent.com/v2 --api-key-env API_KEY \
  --api-mode chat_completions \
  --model-context-window-tokens 192000 \
  --model-max-output-tokens 64000 \
  --max-tokens 64000 --protocol-repair-max-tokens 8192 \
  --reasoning-effort high --reasoning-effort-format native --thinking-type enabled \
  --persistent-history-max-messages 64 \
  --persistent-context-max-chars 512000 \
  --persistent-memory-max-items 128 \
  --provider-timeout-s 300 \
  --tick-interval-s 5 --termination-grace-s 5 \
  --max-workers 16 --pass-k 1 \
  --resume --dry-run
```

## Context and artifacts

Authoritative environment, tool, action, receipt, and effect evidence is
append-only. The model sees a deterministic bounded projection plus structured
memory for unresolved alarms, obligations, facts, commitments, forecasts, and
numeric trends. Compaction preserves semantic-ledger hashes and never rewrites
the authoritative trajectory with an untracked summarizer model.

Every terminal row must bind the main trajectory and semantic ledger by path,
schema, SHA-256, byte count, and record count. Resume accepts only a complete
compatible terminal row under the same treatment; stale `in_flight` records or
artifacts from another implementation tree fail closed.

## Publication gate

A shard may be reported only after every manifest-selected Core row is terminal,
derived reports are finalized, provider and artifact audits pass, and there are no
fatal, orphan, missing, duplicate, or treatment-mismatched rows. Domain/backend
strata and evidence support accompany the primary aggregate.

After one same single model has completed both formal treatments, prepare the
exact candidate manifest outside the canonical release directory:

```bash
mkdir -p output/release_finalize
PYTHONPATH=. .venv/bin/python scripts/finalize_operate_release.py \
  --release-manifest release/operate_v0_62_0/manifest.json \
  --logical-batch-manifest '<logical-treatment>/RUN_MANIFEST.json' \
  --realtime-batch-manifest '<realtime-treatment>/RUN_MANIFEST.json' \
  --output-manifest output/release_finalize/candidate_manifest.json \
  --prepare-distribution-candidate
```

Independent evaluation does not require uploading a distribution bundle. The
public checkout does not include maintainer publication tools.

The finalizer revalidates and content-addresses both result trees before it sets
`public_release_ready` and `leaderboard_eligible`. This scientific readiness
state is independent of the already-public source/runtime distribution.

See [`AGENTIC_INTERACTION.md`](AGENTIC_INTERACTION.md) for the event loop,
memory, cancellation, supersession, safety-supervisor, and realtime scorecard
contracts.
