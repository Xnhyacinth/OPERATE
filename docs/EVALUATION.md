# Evaluation contract

OPERATE scores verified environment outcomes, not prose similarity or a gold
tool sequence. The promoted `operate` release binds qualification
artifacts to scoring `0.15.0`. The live headline is `SCORING_VERSION = 0.17.0`.
Frozen 0.15.0 identities are not rewritten; new evaluations use 0.17.0.

The 769-row, 502-physical-source Core is data/code ready for formal shards.
Provider runs are still pending, so this readiness does not make the release
public or leaderboard eligible.

## Formal treatments

- `logical_persistent` is the primary deterministic leaderboard treatment.
- `realtime_persistent` is an independent supervision treatment with its own
  clock, cancellation, supersession, safety and latency contract. Live formal
  runs use `realtime_persistent.v3` / `native_dt_v1`: wall ticks equal the
  source-converted plant quantum, thinking time is real wall time, and the
  default scorecard is the 37-row speed-critical subset. Do not pool it into
  the 0.17 primary.
- `logical_stateless` is a non-primary compatibility treatment.

Treatments are never pooled. A shard is comparable only when its suite,
implementation identity, prompt/context profile, provider route, pass stratum,
harness and formal treatment-family hash match.

## Primary aggregation

Each eligible episode emits a wait-relative 0–100 primary score:

- **Wait-relative outcome** is the ranking number. Prefer evidenced
  `counterfactual_prevention` (wait-parity = 0). If that dimension is not
  applicable, reanchor evidenced `economic_cost` so wait-parity is 0 rather
  than 50. Do not average the two, and do not mix `optimality_gap` into the
  headline.
- A **survival floor** zeros the primary. The pre-gate wait-relative value
  remains in `survivor_outcome`; catastrophe rate is reported separately
  rather than being the only grid ranking story.
- **Feasibility contracts** (job-shop: every remaining operation scheduled)
  also require binary task completion. An infeasible schedule scores 0;
  `schedule_coverage` stays a diagnostic. A feasible schedule is then ranked
  by the wait-relative outcome, not by a second completion weight.
- **Mitigation contracts** (datacenter, grids, traffic, CityLearn, routing)
  keep the 0/1 beat-wait flag as a diagnostic. It is not 30% of the headline.

The historical 30/25/20/15/10 five-group mix remains in
`legacy_five_group_total` and in `group_scores`, including the three-member
`system_outcome` mean. Adaptation/foresight and action efficiency stay
published; they no longer occupy primary weight. The primary does not claim
to measure proactivity or realtime supervision.

Within a group, the episode denominator contains the dimensions declared
applicable by that scenario and backed by the engine-authored applicability
contract. The support set may vary by task, but never by whether a model chose
to investigate or call a tool: an omitted applicable behavior receives an
evidence-linked zero. Formal outputs publish `group_support` beside the group
scores and aggregate support coverage across the evaluated strata.

Missing required **outcome** evidence never improves a primary score. Formal
ranking aggregates episodes within effective source, effective sources within
backend, backends within domain, and then equal-weights domains. This
prevents repeated variants or the largest domain from dominating the headline.

## Evidence-linked diagnostics

The scorer also reports thirteen diagnostic dimensions:

`system_survival`, `economic_cost`, `safety_violation`,
`weighted_equity_score`, `ethical_quality`, `stakeholder_management`,
`adaptive_replanning`, `information_efficiency`, `foresight_score`,
`optimality_gap`, `counterfactual_prevention`, `tool_use_efficiency`, and
`stakeholder_equity`.

Every applicable dimension carries `evidence_ids`. A dimension without valid
evidence is non-applicable or contributes zero according to the frozen scoring
contract. `robustness_to_fog` and `adaptive_decision_making` remain cross-batch
analyses and are not silently injected into a per-episode headline.

## Causal agency checks

Operational-agency credit requires a native event-to-action-to-effect chain and
a positive masked replay delta. Model prose, an emitted intent, or a transport
acknowledgement is insufficient. The diagnostic scorecard separately reports:

- proactive opportunity detection;
- correct silence during non-actionable intervals;
- semantic alarm detection;
- response and intervention latency;
- takeover/cancel/supersession outcomes;
- tool efficiency and duplicate suppression;
- context truncation, repair and provider failures.

Those diagnostics, and the realtime supervision table, stay out of the 0.17
primary. Long-horizon memory and proactive adaptation are not claimed by the
headline.

## Realtime clock and scorecard

Live `realtime_persistent` uses `native_dt_v1`. Bind `wall_tick_interval_s` to
the scenario's native plant quantum before the episode; do not slow the clock
for slow thinking. Default speed-scorecard membership requires native tick
`<= 60s` and `horizon * native <= 1800s`. That selects 37 Core rows: 7
autonomous driving (5s NGSIM supervisory ticks), 19 traffic, 6 power, and 5
datacenter (60s). CityLearn/microgrid hour ticks and long job-shop horizons
are not 1:1 operator-latency tests. Investigation still serializes on the
environment actor; `harness_periodic_supervisory_scan` remains false; hold is
not native takeover.

## Counterfactuals and determinism

Each formal scenario supplies deterministic no-action replay or a
machine-readable non-applicability reason. Source bytes, seed, scenario
signature, event tape, runtime identity and scoring version are release-bound.
Changing any of them creates a new treatment or release; it is never an in-place
resume.

## Publication gate

A result is formal only when coverage is complete, every row is bound to the
release manifest and treatment hash, provider capability audit passes, there
are no orphan trajectories, and the current-tree release integrity and
readiness checks pass. The published result must include `group_support` and
its aggregate coverage rather than only the five group scores. Partial or
incompatible runs remain diagnostic artifacts.

See [BENCHMARK_DESIGN.md](BENCHMARK_DESIGN.md) for construct validity,
[AGENTIC_INTERACTION.md](AGENTIC_INTERACTION.md) for the event/session model,
and [FORMAL_EVALUATION.md](FORMAL_EVALUATION.md) for executable commands.
