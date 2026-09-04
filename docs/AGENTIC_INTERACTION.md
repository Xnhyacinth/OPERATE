# Agentic Interaction and Real-Time Supervision

This document defines how OPERATE tests a model as a long-running
supervisory decision center without making the model part of the environment.
It separates the deterministic primary leaderboard from the independent
real-time supervision scorecard and from historical replay treatments.

## Core interpretation

An LLM does not keep thinking after a response ends. Long-running autonomy is
therefore an agent-session property: a coordinator resumes the same semantic
session after an agent-scheduled review, a matching watch condition, an
environment alarm, a completed tool call, or an evaluator steer.

The simulator owns all state transitions. A simulator tick remains a required
causal coordinate for deadlines, delayed tools, idempotency, attribution and
replay. It is not, by itself, a reason to send the model a new natural-language
task prompt.

## Experimental treatments

| Treatment             | Clock                                        | Session                                   | Intended use                                                   | Leaderboard                      |
| --------------------- | -------------------------------------------- | ----------------------------------------- | -------------------------------------------------------------- | -------------------------------- |
| `logical_stateless`   | deterministic logical time                   | fresh bounded provider transcript per decision epoch | compatibility ablation and historical replay             | non-primary ablation             |
| `logical_persistent`  | deterministic logical time                   | one semantic episode session              | long-horizon memory, proactive review and event response       | primary formal leaderboard       |
| `realtime_persistent` | soft real-time monotonic single-writer clock | persistent, steerable/cancellable session | latency, controlled hold, stale action and interruption stress | independent formal scorecard     |

`logical_persistent` is an explicit `LLMConfig.interaction_mode` and is the
primary formal treatment. `logical_stateless` remains executable as the
current compatibility ablation and for reproduction of frozen historical
releases, but a stateless row cannot enter the primary leaderboard.
Here, stateless means that each provider call starts from a fresh `[system,
user]` transcript. It is not a cross-decision-memoryless ablation: the bounded
user payload deliberately carries the benchmark-managed `decision_ledger` and
`plan_state` projections used by the historical Protocol-2.1 control. Episode
metadata records this as `session-context-contract/1.0` with
`cross_decision_memoryless=false`; a future memoryless ablation would require a
new treatment name and hash rather than silently changing this compatibility
contract.
The system/mission/scenario briefing occurs once in the semantic ledger. Later
messages are typed continuation events. Stateless provider APIs may still need
to replay a bounded projection in each HTTP request; artifacts must call this
_semantic prompt once_, not claim that every provider transmits it once.
Completed decisions stay in the append-only semantic ledger and are projected
into the next typed event's `decision_ledger`. They are not replayed as plain
assistant JSON, which would train the next turn to imitate serialized audit
records instead of emitting native tool calls.
The qualification currently covers plain `llm_agent`; ReAct and Reflexion
remain separate stateless chat-completions treatments and fail closed if a
persistent or Responses-API treatment is requested.

Runner `multi_turn` drafts are also rejected for decision-epoch agents because
their executable action is already computed before legacy draft collection.
This prevents paid but causally unused model calls; a future deliberative
treatment must make the final provider request explicitly consume the drafts.

`realtime_persistent` is implemented as a separate scorecard runner, not a
branch inside `run_one`. The environment actor owns `env.step()` on a
monotonic clock while provider turns execute concurrently. Existing native
investigation tools mutate shared budgets, evidence and reveal ledgers, so
they remain serialized on that same actor and can stall the soft real-time
clock; every such duration and missed deadline is explicit in the artifact.
The treatment does not claim a hard real-time plant clock. A future hard-
independent cell requires backend-specific immutable investigation snapshots
and an audited commit boundary rather than calling the current mutable tool
path concurrently. The
coordinator uses native steer when a harness advertises it and otherwise uses
cancel-and-resume plus response supersession. Late responses are retained in
the turn ledger but cannot reach the environment actor. This treatment emits a
separate `realtime-diagnostics/1.6` scorecard. It is formal only within its own
release-bound clock, supervisor, provider and concurrency stratum and is never
merged with the thirteen-dimension logical primary leaderboard.

The promoted `operate_v0_61_0` release binds the complete realtime artifact stack:
`realtime-formal-batch/1.1`, `realtime-formal-scorecard/1.1`,
`realtime-episode/1.1`, `realtime-treatment/1.1`,
`realtime-provider-audit-contract/1.0`, diagnostics 1.6, and coordinator
`realtime_episode_v5`. Any missing or older component fails closed and cannot
be resumed into this treatment.

## Formal publication strata

The current logical contract runs one model per shard. Model roster size is
therefore not a release constant. Each shard declares `pass_k >= 1` and a
requested concurrency between 1 and 32; pass count and requested/effective
concurrency are immutable run-scope fields, not hidden launcher details. Pass count is a
cross-model reliability stratum; provider-specific concurrency is recorded per
model and need not match across shards. A shard may be combined with another
shard only after the publication step verifies the same suite, prompt/context
profile, pass stratum, harness, implementation tree and formal treatment-family
hash. Each shard's provider route is independently treatment-bound and audited,
but different models do not need the same route. Provider errors, throttling contamination, incomplete coverage, effective/requested
worker drift or mixed treatment hashes fail closed.

The realtime scorecard has its own strata. In addition to the fields above, its
identity binds simulator-tick duration, episode-timeout policy, cancellation and
steer capability, safety-supervisor identity and scorecard version. Changing a
clock or supervisor starts a new scorecard stratum; it is not a resume of an
existing run.

The synchronous direct-API adapter serializes calls that share one mutable
semantic ledger. If its in-flight HTTP call cannot be canceled, the alarm turn
queues behind it while the environment keeps advancing; this limitation and
its latency are part of the treatment result. A harness may overlap work only
when it implements the `RealtimeTurnDriver` native steer/cancel contract and
declares that capability in the artifact. Concurrent calls into one
`LLMAgent` instance are forbidden.

Each direct-API turn is a behavioral transaction. The driver snapshots the
semantic ledger, structured memory, active plan and recent-action state before
the call; only the arbitration winner commits those mutations. Superseded and
timed-out turns restore the snapshot, while their provider request/response
audit remains append-only because the paid call still occurred. A synchronous
SDK call has logical cancellation, not a hard wall-time guarantee. The artifact
therefore reports outstanding calls at return and never claims process-level or
provider-level hard timeout enforcement.

Provider-bound logical and realtime command templates are maintained in
`docs/FORMAL_EVALUATION.md`. A realtime command must name the credential
environment variable, provider route, model capability limits, canonical
context profile, clock stratum and treatment-bound trajectory directory.

The default CLI safety supervisor is the explicitly recorded
`domain_neutral_hold` policy. It is a controlled hold, not a domain-native
shield and not evidence of successful native takeover. A scorecard may claim
native shielding, minimum-risk behavior or takeover only when a domain runner
injects the corresponding native supervisor and its identity is treatment
bound. The ledger distinguishes model action, controlled hold and native
minimum-risk takeover. The descriptor-bound
`autonomous_driving_runtime_assurance_v1` treatment is the first concrete
native profile: it requires the `sumo_ego` backend and the native
`request_minimal_risk_maneuver` tool, rejects an unsupported domain/backend or
missing tool before a provider call, and produces a distinct treatment hash.
It does not claim provider-native steering. That separate capability is true
only when the selected harness implements `steer_realtime_turn`; the direct API
fallback remains cancellation/supersession plus the late-action fence. The
deterministic conformance suite uses a delayed stub provider and never consumes
a paid API.

## Decision-event protocol

The runner adds a provider-neutral `__decision_epoch__` envelope:

```json
{
  "decision_id": "decision-7",
  "model_decision_index": 7,
  "reasons": ["scheduled_review"],
  "state_version": 19,
  "simulator_tick": 19,
  "deadline_tick": 21
}
```

Persistent sessions compile it into one of these event kinds:

- `session_start`: initial mission and visible state;
- `environment_alarm`: a registry-declared visible disruption or new task;
- `safety_warning` and `forecast_update`: typed safety/forecast wakeups;
- `scheduled_review`: an agent-requested review or plan expiry;
- `tool_result`: same-epoch read-only investigation output;
- `tool_failure` and `delayed_tool`: failed or delayed tool feedback;
- `action_receipt`: stale, expired, rejected, canceled, failed or unexplained
  no-effect action feedback used to reconcile and replan;
- `provider_retry`: an explicit retry after a recorded provider failure.

Quiet, pending-action and standing-plan hold ticks advance the simulator using
an empty action and do not resume the LLM. Hidden events never steer the agent.
Successful state-changing control receipts do not create another turn. When
several typed triggers share a transition, the coordinator dispatches them in
descending priority and stable event-sequence order; lower-priority events are
queued with their full payload and later replayed, not silently discarded. If
a queued deadline passes, the original trigger is recorded as expired/missed
and a new current-state continuation preserves its original event identity and
timestamp. Repeated typed event keys coalesce to the latest pending update.
Read-only investigations preserve the actor's authoritative monotonic
simulator tick even if a backend snapshot contains a staged tick. An action
submitted in the final pre-terminal interval can execute only through an
immediate tool; delayed effects due at or after the terminal horizon are
rejected so no accepted pending result becomes orphaned. Once the actor
reports `done`, terminal evidence and receipts are ingested but no new
decision-required turn is opened.

The canonical cadence is `agent_scheduled_v1`: the agent owns review timing,
`harness_periodic_supervisory_scan=false`, typed actionable events wake the
session early, and unknown events remain non-actionable. A
`supervisory_scan` record is therefore legacy/diagnostic evidence, not a source
of model turns in the current formal treatment.

The existing `commit_to_plan(review_after_ticks, wake_if,
plan_expires_at_tick)` is the agent-owned proactive scheduling mechanism. It is
preferred over an unconditional benchmark-side polling loop. `wake_if` is a
typed subscription over `visible_event`, `forecast_update`, and `delayed_tool`;
an active standing plan suppresses unsubscribed optional wakeups with an
explicit audit reason. Registry-declared safety,
task, and failure events remain mandatory and cannot be unsubscribed. A revised
successful plan supersedes the prior scheduled review, while the turn ledger
retains the displaced review ticks.

`commit_to_plan` is never stripped or trusted by the coordinator. It executes
through the registered ToolProtocol like every other tool. A review is
scheduled only after a matching successful terminal ToolResult; failed and
delayed results resume the session for plan reconciliation. A pending delayed
acknowledgement updates lifecycle state without waking the model; the terminal
result is the actionable continuation. An alarm supersedes the active turn but
does not silently erase a future agent-owned review. If the review becomes due
with a higher-priority trigger, it is retained and queued with explicit causal
identity.

## Context and evidence layers

Long-running context is divided into three layers:

1. **Authoritative evidence** — append-only observations, environment events,
   tool requests/results, actions, scores and evidence IDs. Evaluation and
   replay read this layer only.
2. **Semantic session ledger** — one system mission plus typed user events and
   normalized assistant decisions. This is append-only for the episode.
3. **Model-visible context** — a bounded projection of the ledger. When either
   its configured message limit or character budget is exceeded, deterministic
   compaction retains the system message, latest event, active confirmed plan,
   compacted recent decisions and hashes of compacted messages. Bulky tool
   outcomes are replaced by identity/status stubs; the authoritative ledger is
   unchanged. Before each provider request, older typed events also drop stale
   copies of the rolling `structured_memory` and `decision_ledger` projections;
   the latest event carries their current values and the semantic ledger keeps
   every original copy. This prevents repeated state snapshots from turning a
   linear episode history into avoidable request growth.

Compaction count, covered hashes and tick are recorded. No hidden evaluator LLM
generates a privileged summary. The current visible prompt budget and the
session-history message budget are independent controls.
If the logical runner cannot ingest a transition into agent-maintained plan
state, it records a structured `agent_transition_ingest_failed` violation in
the trajectory. The event contract then fails closed, so the episode cannot be
admitted as a formal persistent measurement.

For a `tool_result` continuation, read-only schemas already used in the prior
investigation are removed from the next request; state-changing tools and
explicit `wait` remain. This prevents repeated investigation loops and avoids
resending irrelevant schemas.

The current formal profile fixes temperature 0, 32,768 output tokens, an 8,192-
token protocol-repair budget, 64 visible history messages, a 512,000-character
context projection, 128 items per structured-memory bucket, a 300-second provider timeout,
`tool_choice=auto`, streamed Chat Completions, and fail-closed provider handling
with a one-failure circuit threshold. A provider/tool-calling failure terminates
the formal episode before a tool-less retry can synthesize `wait`, after at most
four fully audited same-request retries for transport-only 429/5xx failures.
Those retries use deterministic bounded backoff, re-enter the shared limiter,
and remain cancelable while waiting; they do not advance the simulator or append
a duplicate semantic event. Action-required
epochs still fail closed unless the model emits an executable native tool call.
These values are budgets, not targets: a model is not rewarded for consuming them. Non-formal
persistent invocations retain their lighter local defaults and form a different
treatment. Reasoning effort is sent only when the model shard explicitly
declares a supported value. Reasoning settings and effective wire
streaming are always recorded and treatment bound.

The 512,000-character projection is a fixed working-context treatment, not a
claim that every provider accepts 512,000 tokens. The preflight separately
binds each model's context window and maximum output. The semantic ledger and
authoritative evidence remain complete even when the visible projection is
compacted. Overrides to history, projection, memory, timeout, repair or output
budgets create a different treatment and cannot enter the canonical
current-release stratum.
The repository carries a frozen, offline capability entry for
`stealth/ox-alpha` (1,048,576 context tokens and 131,072 maximum output tokens).
Persistent runs with any other model must pass both
`--model-context-window-tokens` and `--model-max-output-tokens`; the runner never
queries mutable provider metadata at run time. Before each main or repair HTTP
request, a deterministic UTF-8-byte upper bound covers the provider-compiled
messages, tool schemas, wire wrapper and requested output reserve. Requests that
cannot fit the treatment-bound limits fail before any provider call. The audit
records both configured streaming and effective wire streaming, since Responses,
Anthropic and Google transports do not inherit the chat-completions stream flag.
Because those two capability flags are scalar, batch runs accept them for only
one model at a time; heterogeneous models must be run separately unless every
model is covered by the frozen per-model registry.
These formal budgets leave output capacity after reasoning while keeping
actions bounded; they are not applied retroactively to frozen
`logical_stateless` treatments.
The current formal realtime clock is 5 wall seconds per simulator tick. Its
episode timeout is derived from the scenario horizon plus one provider-timeout
and one tick of teardown slack, so long-horizon rows are not silently cut off
by a fixed five-minute default; shorter intervals/timeouts are explicit
latency stress treatments.
Every effective parameter is treatment-bound. A length-terminated or text-only
response is invalid, never synthesized into `wait`. Text-only output may
receive one bounded protocol-repair request; its text is never executed
directly, and all repaired native calls are retained only while the shared
per-epoch call and cost budgets permit.

The model-visible context is bounded, but the reference implementation keeps
the authoritative semantic ledger and credential-free provider audit records
in memory until the episode artifact is finalized. This is appropriate for the
released finite-horizon cells; it is not a claim of 24/7 constant-memory
operation. A continuous deployment must add an append-only, fsynced chunk sink
with a hash manifest and retain only indexes/windows in RAM as a separate
harness treatment.

Provider request artifacts store a credential-free, provider-neutral
pre-compilation envelope rather than claiming to capture SDK wire bytes. The
implementation identity binds the OpenAI, Anthropic and Google compiler code;
provider-conformance tests require every compiler to preserve the complete
persistent user/assistant history.

The real-time artifact also embeds the provider request/response audit joined
to turn, decision, action, receipt and commit/rollback outcome. Treatment
identity includes header names and allowlisted behavioral routing values, but
never stores or hashes Authorization, Cookie or other credential values.

## Required real-time contract

`realtime_persistent` must not replace deterministic replay. It requires a
single environment actor that owns `env.step()` and state-changing tool access,
while provider turns run concurrently. Every submission and receipt must carry
the following identity and lifecycle:

```text
AgentEvent:
  event_id, event_seq, kind, priority, decision_id, state_version,
  simulator_tick, deadline_tick, deadline_monotonic_ns, evidence_ids

ActionSubmission:
  action_id, decision_id, turn_id, based_on_state_version,
  valid_from_tick, expires_at_tick, supersedes_action_id

ActionReceipt.status:
  accepted | queued | applied | effected | rejected | stale | expired |
  canceled | superseded | failed | no_effect
```

The initial fail-closed rule is compare-and-swap: a state-changing action is
accepted only against its observed state version and before its deadline. Read
tools execute through `execute_investigation` on the environment actor and
report the observation version they actually observed without advancing the
simulator tick. Arbitration is fixed and replayable: every candidate first
passes through the configured safety supervisor, then the latest valid
non-superseded command is applied. The default domain-neutral supervisor
records a pass but does not claim domain-native shielding; domain-specific
cells must inject their native shield. There is no anonymous empty-action
fallback: every tick without a valid model command carries a typed
`SafetyDecision`, supervisor identity, mode, and reason.

An alarm arriving during a provider turn is delivered using native steer when
available, otherwise cancel-and-resume, otherwise supersession. A late response
from a superseded turn is recorded but never executed. Interrupting a model
turn does not roll back a tool or environment side effect.

Action receipt callbacks only enqueue immutable receipt data. The coordinator
loop updates the turn ledger and emits typed reconciliation events; callbacks
never run coordinator logic while the environment actor's condition lock is
held. This keeps the single-writer actor and coordinator lock order acyclic.

For autonomous driving, the high-frequency shield/autopilot remains responsible
for immediate control. The LLM acts only at the supervisory cadence. A takeover
cell should distinguish notification, transition demand, acknowledgement,
assumption of control and minimum-risk fallback when the model is late.

## Harness policy

The benchmark reference remains a small provider-neutral direct-API harness.
Codex App Server, OpenAI Agents SDK, Claude Agent SDK/Managed Agents, LangGraph
and similar systems may be added as separate harness treatments after passing a
common conformance suite. They must not become the simulator supervisor.

Treatment identity must include model/provider snapshot, public route ID, API
mode/version, harness/version, prompt and context compiler hashes, tool-schema
hash, executable implementation-tree hash, sampling/reasoning parameters,
output limits, timeout/retry policy,
sandbox, skills and MCP configuration. Results from direct API and vendor
harnesses measure different agent systems and must not be merged into a single
model-only ranking.

## Independent real-time scorecard axes

`realtime-diagnostics/1.6` currently reports:

- actionable trigger counts split by typed kind, plus delivered,
  acknowledged, decided, acted and effected stages;
- correct detection without requiring an effect, false alarm, missed trigger
  and correct silence;
- trigger-to-decision and trigger-to-effect latency in simulation and wall
  time, with exact event/turn/action/lifecycle/proven-effect joins;
- proactive scheduled-review service, unnecessary polling and environment
  ticks per model turn;
- stale, expired, canceled and superseded action rates;
- controlled-hold, minimum-risk/native takeover and supervisor-failure counts.
- invalid model responses, including output truncation and missing native tool
  calls; these are missed decisions rather than correct silence.
- native-tool compliance before repair, repair attempts/successes/failures and
  the share of logical decisions that depended on repair.

No-event/no-op cells are mandatory: acting unnecessarily is a failure, not
evidence of proactivity.

`delivered` is only a harness-delivery stage. `detected` requires a completed,
valid transport acknowledgement, not a claim that the model semantically
recognized the hazard. `decided` additionally requires a decision epoch;
a hung provider is therefore delivered but not detected. The scorecard reports
delivery-, decision- and response-missed counts separately. A completed,
deliberate no-action decision counts as a response and can be correct silence;
it is not credited as an intervention or effect.

Alarm-to-investigation latency, watch-condition coverage, useful discoveries
per token/tool cost, takeover lead/acknowledgement/assumption stages and
cross-stage constraint-retention scores are planned axes, not current
claims. Agent-scheduled reviews, correct silence, action/effect latency,
stale/expired/superseded lifecycle outcomes and controlled-hold/takeover counts
are implemented independently. Compaction/context records remain preserved in
`llm_interaction_stats` rather than being collapsed into a single score;
provider native-tool and repair counters are also projected into the real-time
scorecard.

The episode embeds final observation, cumulative reward and action lifecycle
outcomes. It also embeds the complete environment evidence ledger, its hash,
reference-closure result, scenario signature and audit-only ground truth.
Ground truth is never placed in a model-visible observation. These scorecard
metrics remain independent of `evaluation.scorer.SCORING_VERSION`,
counterfactual replay and the frozen thirteen dimensions.

Artifact acceptance is fail closed. A completed simulator execution is not
evaluation-ready unless provider audit is complete, behavioral state has
settled, teardown is safe and every consumed/dependent/effect evidence edge
closes against the authoritative ledger. Delayed results are checked against
the visibility at their original submitted or safety-applied call, not the
later tick where the result materializes.

Queued events are never converted into new provider turns after the simulator
has terminated. They remain in the artifact with
`terminal_dispatch_suppressed=true` and
`dispatch_suppressed_reason=ENVIRONMENT_DONE`, so missed terminal work is
auditable without creating an unanswerable post-episode decision.

An in-flight streamed turn is execution-fenced and canceled when the episode
ends. The direct driver then allows a bounded two-second audit-settlement grace:
this cannot apply a late action, but lets the canceled worker record its
terminal provider response and roll back transactional memory before artifact
validation. An unresponsive worker still fails closed as unsettled.

## Python runtime tiers

Core and direct LLM clients support Python 3.10 through 3.14. CI installs the
project on 3.10, 3.12 and 3.14. Python 3.14 selects the first qualified package
lines for NumPy, pandas and SciPy. Full simulator support remains tiered:
CityLearn and the autonomous-driving pilot retain their isolated qualified
environments, while OR-Gym uses the documented `--no-deps` compatibility path.
This is not a claim that one monolithic backend environment is dependency-clean
on every Python version.

## External design references

- [SentinelBench](https://github.com/microsoft/sentinel_environments): evolving
  environments, no-op monitoring cells, reaction-time/resource metrics and
  condition-based waiting.
- [Robotouille](https://github.com/portal-cornell/robotouille): separate
  synchronous/asynchronous long-horizon planning results.
- [OSWorld 2.0](https://github.com/xlang-ai/OSWorld-V2): pinned component
  releases, state/checkpoint grading and stale-observation failures.
- [tau2 full-duplex orchestrator](https://github.com/sierra-research/tau2-bench/tree/main/src/tau2/orchestrator):
  simulation-time/wall-time traces and final-state replay grading.
- [VibeLifeBench](https://github.com/evolvent-ai/VibeLifeBench): silent world
  mutations versus explicit notification events and cross-stage assertions.
- [REALM-Bench](https://github.com/genglongling/REALM-Bench): framework ablations,
  parallel planning threads, dependencies and disruption frequency.
- [CARLA synchrony and time-step](https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/)
  and [nuPlan](https://github.com/motional/nuplan-devkit): deterministic versus
  asynchronous clocks and open-/closed-loop treatments.
- [UN Regulation No. 157](https://unece.org/sites/default/files/2023-12/R157a1e_0.pdf):
  transition demand, continued operation and minimum-risk manoeuvre.
- [Codex App Server](https://github.com/openai/codex/blob/main/codex-rs/app-server/README.md):
  persistent thread/turn/item, steer, interrupt and explicit compaction.
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/running_agents/):
  client/server continuation choices, sessions, tracing and deterministic tests.
