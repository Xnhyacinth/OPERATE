# Benchmark design and construct validity

OPERATE evaluates an LLM as a supervisory scheduling agent over a
seeded, code-driven environment. It does not evaluate whether a model can
continue a fictional narrative. The released task state comes from real source
windows consumed by native simulators; declared, deterministic stress overlays
create outages, surges, breakdowns and observation loss without making an LLM
part of the world transition.

The promoted `operate_v0_61_0` release uses scoring 0.14.0. Results from a different
implementation hash are diagnostic evidence only and cannot validate or resume
a current formal shard.

Its Core contains 769 rows across 502 physical sources: 7 Autonomous Driving,
18 Building Energy, 142 Datacenter, 527 Logistics, 37 Microgrid, 19 Power Grid,
and 19 Traffic. These counts describe corpus composition; hierarchical
source→backend→domain aggregation, rather than row count, defines the primary
weighting.

## Closed-loop episode

```text
initial mission + visible state
             |
             v
      typed decision event <-------------------------------+
             |                                             |
             v                                             |
  new provider request with bounded semantic session       |
             |                                             |
       investigate / plan / control / wait                 |
             |                                             |
             v                                             |
 ToolProtocol validates budget, delay, failure and identity |
             |                                             |
             v                                             |
 seeded backend advances and records native state/effects   |
             |                                             |
             +-- scheduled review / subscribed event ------+
             +-- actionable alarm / task / safety event ----+
             +-- quiet tick: advance without LLM request
```

The system mission is written once to the semantic ledger. A direct stateless
HTTP API still receives a new request at every decision event because no model
continues computing after its response ends. The harness reconstructs the same
semantic session from the system mission, typed events, structured working
memory and recent decisions. A provider-side thread or Codex-like harness is
therefore not required for the canonical treatment. Such a harness may be
evaluated only as a separately identified treatment because native steering,
server-side memory and cancellation can change behavior.

## Autonomy and interruption

The model owns its next intended review through
`commit_to_plan(review_after_ticks, wake_if, plan_expires_at_tick)`. The
coordinator owns only the clock, typed event delivery, action arbitration and
safety fallback. It invokes the model on:

- the initial session event;
- an agent-scheduled review or subscribed visible event;
- a registered actionable alarm, task, safety warning or failed/delayed tool.

The canonical `agent_scheduled_v1` metadata declares that the agent owns review
timing, `harness_periodic_supervisory_scan=false`, typed actionable events wake
the session, and unknown events remain non-actionable.

A simulator tick is a causal timestamp, not a new natural-language prompt.
Routine telemetry and already-consumed source intervals do not wake the model.
Unknown events fail closed as non-actionable and are audited. Hidden events do
not steer the agent; the model must use available observation tools or infer
their effects. This separates proactive investigation from passive alarm
response.

`logical_persistent` pauses logical time while a provider request is in flight,
which preserves deterministic replay. `realtime_persistent` advances a
single-writer environment actor while the provider turn is outstanding and
records cancellation, supersession, stale action, safety hold and eventual
effect. The two treatments answer different questions and are never pooled.
The default realtime hold is domain-neutral and must not be described as a
domain-native or autonomous-driving takeover policy.

## Context and memory

Three layers are deliberately separate:

1. Authoritative evidence is append-only and drives replay and scoring.
2. The semantic ledger stores the full mission, typed events and normalized
   decisions for audit.
3. Model-visible context is a deterministic bounded projection.

Structured memory contains unresolved alarms, obligations, confirmed facts,
active commitments, forecasts and state trends. It is benchmark-managed so
all models receive a reproducible memory compiler. This treatment measures
decision quality with standardized memory, not a model's unaided ability to
invent a memory system. Agent-managed memory should be published as a separate
treatment.

The latest event carries the current structured-memory and decision-ledger
projection. Older visible events retain their event-specific observations but
drop stale copies of those rolling projections; the full copies remain in the
semantic ledger. Deterministic compaction preserves the system mission, latest
event, active plan, bounded recent decisions and hashes of compacted content.
Formal artifacts bind the context/output limits, temperature, reasoning mode,
tool choice, timeout, streaming behavior and compaction implementation.

The 32K output value is a maximum, not a target. A length-terminated partial
tool call is rejected and may receive one bounded repair turn. Very small caps
such as 1K confound tool-use competence with truncation; very large raw history
without semantic projection confounds long-horizon reasoning with repeated
transport cost. Provider token, latency, truncation and repair statistics must
therefore be reported with model scores.

## Task and sample contract

Every publishable sample needs:

- a manifest-locked public source and proof that its values drive backend
  state transitions;
- native observations, entities, controls and task outcomes;
- a typed event registry distinguishing routine telemetry from actionable
  events;
- discoverable read tools and at least one effective control path when the task
  requires intervention;
- deterministic no-action replay or an explicit non-applicability reason;
- action-to-backend-effect evidence and treatment-bound trajectory artifacts;
- a source denominator key so same-source variants do not inflate the primary
  leaderboard.

The Core should be reported by task archetype as well as domain: continuous
dispatch, hidden-state proactive monitoring, event intervention, long-horizon
planning and realtime takeover/latency. Real source data plus a procedural
stress overlay must be described exactly that way; it is not an empirically
observed historical emergency. Paired variants should reuse the same source,
seed and event tape when isolating observability, alarm delivery or realtime
latency.

Core admission establishes source grounding, executable outcomes and task
quality; it does not establish that every admitted row positively calibrates
agent-owned review, proactive discovery, standing-plan interruption or another
agency construct. Formal outputs must therefore disclose task-archetype and
agency-coverage strata, including the number of applicable opportunities in
each stratum. Claims about persistent or proactive agency are limited to the
strata that contain those opportunities rather than extrapolated to the whole
Core.

Extra tools are legitimate distractors when their schemas are native and calls
have real cost. Gold tool sequences are diagnostic only. A correct alternative
strategy must not fail merely because it differs from a reference trajectory.

## Scoring

The primary score is environment-verifiable: task outcome, system outcome,
safety/responsibility, adaptation/foresight and action efficiency. Task outcome
is separated from process-capability checks so an equivalent safe solution is
not forced to imitate one authored tool path. The leaderboard macro-averages
within effective source, backend and domain instead of letting the 142-row
Datacenter family dominate by row count.

The thirteen evidence-linked dimensions and the six-dimensional operational
agency profile remain diagnostics. Operational-agency credit requires a native
event-to-action-to-effect chain plus a positive masked replay delta; model prose
or self-reported evidence is insufficient. Coverage of proactive opportunities,
correct silence, alarm delivery, semantic detection and response latency must
be reported separately and stratified by task archetype and agency coverage. A
transport acknowledgement is not semantic detection.

## Relationship to other benchmarks

The design follows the strongest reproducibility patterns without copying
their world model:

- [tau2-bench](https://github.com/sierra-research/tau2-bench) separates task,
  tools, environment and orchestration, and grades equivalent database end
  states rather than a single gold action path.
- [Meta Agents Research Environments](https://github.com/facebookresearch/meta-agents-research-environments)
  uses event-driven scenarios and app state; OPERATE keeps physical
  state transitions simulator-only instead of using an LLM user as the world.
- [OSWorld](https://github.com/xlang-ai/OSWorld) and
  [WebArena](https://github.com/web-arena-x/webarena) demonstrate executable
  environments and state-based verification; OPERATE applies that
  principle to scheduling backends and counterfactual replay.
- [SWE-agent history processors](https://github.com/SWE-agent/SWE-agent/blob/main/sweagent/agent/history_processors.py)
  make context policy explicit; OPERATE likewise treats memory and
  compaction as treatment-bound harness behavior.

The benchmark's distinctive contribution is the combination of real source
consumption, native simulator evolution, partial observability, agent-owned
review scheduling, asynchronous typed interruption and replay-backed causal
scoring in the task strata where those constructs are applicable. It should not
claim that every Core row is long-horizon, hidden-event, positively calibrated
for autonomy or realtime. Data/code readiness means the promoted manifest and
native replay close under the current implementation identity; it does not
claim a provider result. Public-release and leaderboard claims remain false
until a complete current-identity formal shard passes the publication gates.
