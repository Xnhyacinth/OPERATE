# Evaluation contract

OPERATE scores verified environment outcomes, not prose similarity or a gold
tool sequence. The promoted `operate` release binds qualification
artifacts to scoring `0.15.0`. The recorded episode scorer remains `SCORING_VERSION = 0.21.0`.
Frozen 0.15.0–0.20.0 identities are not rewritten; live episode artifacts retain 0.21.0.
The separate [offline evaluation 0.22](EVALUATION_022.md) reports
`evaluation_version=0.22.0`, native task quality and evidence-linked capability
diagnostics while preserving each source episode's scoring and execution identity.

For newly requested offline comparisons, use
[0.23.1 fixed expert-target attainment](EVALUATION_023_TARGETS.md): frozen native
targets, task-local constraints, and source/backend/domain macro attainment.
It requires no weakest-policy anchor. Native cost gaps and limited-scope A/L
profiles remain separate. Historical 0.22 and 0.23.0 scores are preserved;
their calculations below are not the new headline. Known framework defects
must remain visible when interpreting historical trajectories.

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
  the 0.21 wait-relative primary.
- `logical_stateless` is a non-primary compatibility treatment.

Treatments are never pooled. A shard is comparable only when its suite,
implementation identity, prompt/context profile, provider route, pass stratum,
harness and formal treatment-family hash match.

## Native operational-quality Index (`native_quality.v1`)

The new post-episode scoring entry point is
`scripts/evaluate_native_quality.py`. It measures the actual native task outcome
under evidenced hard constraints, not the number of plans or tool calls. The
reference protocol executes the fixed `wait_only`, `greedy_heuristic`,
`oracle_offline`, and seeded `random` policies twice per case. Among reproducible,
feasible reference policies, let W be the highest cost and S the lowest cost.
These policies are not claimed to be optimal, and LLM results never set W or S.

For native minimization cost C:

    z = (W - C) / (W - S)
    raw_reference_score = 100 * z
    native_quality = 100 * z / (1 + abs(z))

For these gap-normalized objectives, the weak reference scores 0 and the strong
reference 50 on the main scale.
Improvement beyond the strong heuristic continues to increase the score;
there is no deliberate clipping platform. Each case has a bounded contribution
with diminishing marginal utility. This is a declared cross-task utility
choice, not a claim that all possible utility functions induce the same rank.
Raw linear quality and native costs remain available for interpretation.

Completed DynaSched makespan has a natural zero and uses
`scale = max(W - S, S)` and `z = (W - C) / scale`. This prevents two nearly
identical dispatch heuristics from magnifying tiny elapsed-time differences.
Its strong reference need not score 50. Identical positive feasible reference
makespans still define this scale; their gap-normalized raw score is N/A with
an explicit reason. The original gap-normalized raw score is otherwise retained
only as a diagnostic. This makespan-specific floor must not be
applied to signed or translation-dependent monetary objectives.

An evidenced hard failure or infeasible job-shop schedule scores -100. Failure
to reach a mitigation threshold is not automatically infeasibility. Task success
and constraint failure rates are separate columns. Missing evidence, missing
matched references, unstable repetitions or a degenerate normalization scale produce
N/A. A full Index requires the complete fixed suite denominator; there is no
fallback to the historical primary or thirteen-dimension composite.

Native costs come from the actual execution's settled backend components;
pymgrid uses its existing native-state-loss task contract. A counterfactual
actual replay must not replace a different observed live cost. Replay mismatch
is retained as a diagnostic. Each measurement identifies its objective, units,
hard-gate scope and evidence source. Recorded task-contract summaries are not
mislabelled as independently reconstructed native tick records.

Reference episode and trajectory bytes are checked against their recorded
hashes, and their scenario, seed, backend and runtime are matched before scoring.
This verifies local runner artifacts, not arbitrary maliciously rewritten
manifests or provider authenticity. Runtime/prompt/treatment cohorts remain
separate. Main aggregation stays source → backend → domain macro; source-cluster
intervals use that same estimand. Index completeness and formal/public run
certification remain distinct. The frozen release is not promoted by this tool.

An optional `reference_compatibility` entry can reuse CPU reference episodes
across verified changes limited to unused LLM-provider methods and literal
provider maps. The code-file comparison and dependency-lock comparison produce
a hashed receipt; the model runtime identity remains unchanged in every output
and model cohorts are never merged. Ambiguous compatible reference sets fail
instead of selecting a scale by input order.

To score existing trajectories, supply a JSON manifest with `suite`, `runs`
(model labels mapped to episode JSONL paths), and `references` (objects with
the reference worktree `root` and its relative `report` path), then run:

```bash
uv run python scripts/evaluate_native_quality.py \
  --manifest .hl/native_index_inputs.json \
  --output .hl/native_index_results --bootstrap 2000
```

The output directory must be new. `report.json` retains native costs, raw
quality, task/constraint results, reference evidence, input hashes, comparison
cohorts and cluster intervals; `scores.csv` provides per-case old/new scores.
Repeated attempts require an explicit selection policy: `latest_started` uses
recorded timestamps rather than choosing the best-scoring attempt. Missing
timestamps or tied duplicates remain unresolved. Use `row_filters` only for
fixed execution identities, never for selecting favorable outcomes.

The older clipped linear prototype in
`tools/native_quality_index_candidate.py` is superseded by this production
protocol. Existing episode logs and legacy reporting commands retain their
original scoring identity. Full reference calibration is required before a new
native-quality leaderboard can be emitted; implementation alone is not evidence
that every Lite row is calibrated.

## Legacy primary aggregation (0.21)

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

## Two headline numbers

Two different per-episode numbers are both called a "total" and they are not
interchangeable. The 13-dimension composite is written to `score.total_score`
(the `fixed_all_dimensions` view): every emitted dimension counts toward a
fixed weight denominator, and non-applicable dimensions contribute zero rather
than leaving the denominator. Ranking uses the `discriminative_core` view
instead, whose value is the wait-relative primary: counterfactual prevention
when it is evidenced, otherwise `economic_cost` reanchored so wait-parity is 0.
Each completed episode also emits `ranking.primary_score` /
`ranking.wait_relative_score` with `aggregation=wait_relative_outcome_v1`.
On the same row these can differ by several points, and a row whose primary is
zero can still carry a positive `score.total_score`. Quote the primary and name
its source (`wait_relative_score` / `wait_relative_source`) whenever a number
is presented as a ranking; cite `score.total_score` only as the composite
diagnostic. `ANALYSIS.md`, `summary.csv` (`primary_score`), and score plots
use that ranking number.

An empty or illegal model protocol still calls `env.step`. Logical time is
the simulator clock; freezing it would give the model extra thinking time.
The plant outcome is wait-like, so wait-relative ranking remains the headline.
Process accounting records those ticks as `n_invalid_model_decisions`, not
`n_deliberate_wait_actions`. Runner holds (`native_idle_hold`, plan hold) are
neither. Formal Lite/Full abort provider failures instead of converting them
to `wait`. `robustness_to_fog` and `adaptive_decision_making` remain reserved
cross-batch analyses and are not emitted per episode. Hidden, stale, or delayed
events are surprise for `adaptive_replanning` when a reveal-to-effect chain is
evidenced; the scorer does not require a backend-authored `surprise` flag.
Realtime initiative, silence, delay, cancellation, supersession, and takeover
stay on the independent supervision scorecard and are never pooled into the
wait-relative primary.

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

## Scoring 0.18–0.20 repair contract

- OpenDSS and LV survival use native convergence/terminal failure flags.
  Ordinary voltage-band violations remain safety and economic costs; their
  node/phase count is not a count of collapsed buses.
- CityLearn declares signed `energy_cost`. Actual and counterfactual replays,
  including masked action replays, must agree on that declaration. Undeclared
  negative penalties and non-finite values remain invalid. A non-positive
  wait cost has no applicable ratio under this normalization.
- Missing outcome evidence returns no leaderboard value, not a model zero.
  `score_group_contracts` includes schedule coverage and `native_outcome`:
  actual/wait cost, prevented loss, unclipped wait-relative change, and a
  clipping indicator. Compare native costs only within matched tasks. The
  bounded primary still ties some completed schedules and clips negative
  improvement at zero; those ties do not imply equal native outcomes.
- Normal completions, telemetry and agent-caused effects do not create an
  adaptation opportunity. Query and plan credit requires successful consumed
  evidence, a real native effect and positive masked replay attribution.
  A revised plan must reference an earlier recorded plan; a claimed ID alone
  earns no revision credit. Native reveal/deadline/surprise metadata survives
  canonicalization, and consumed reveal evidence establishes observation time.
- Trade-off quality remains unavailable until structured, replay-bound
  alternative policies exist. It is not a measured model weakness or strength.

## Scoring 0.21 evidence repair

The wait-relative formula and source/backend/domain weights remain unchanged.
An unavailable episode primary is `null`, never a measured zero or a fallback
to the composite. An applicable dimension must reference evidence present in
that episode's authoritative ledger, including for composite diagnostics.
Missing survival-gate evidence makes the primary unavailable; it cannot remove
the catastrophe gate. Structural native non-applicability remains distinct.

A backend effect without an observed, consumed event chain is retained as
`backend_effect_only`. It can affect native costs but cannot earn observed
response or adaptation credit. Direct and resolved tool-result evidence aliases
remain recorded so a delayed receipt retains its original request tick.

Reports publish `native_outcome_diagnostics`: harmed/parity/improved counts,
measured/unavailable counts, and sample and source/backend/domain macro harm
rates. Costs are never summed across domains. Harm is measured only when a
usable replay report exists; an unusable/non-positive normalization baseline
remains unavailable, not no-harm. This companion distinguishes outcomes that
both clip to zero in the primary, without changing the ranking formula.

Current primary tables and flat/composite diagnostics are named separately.
Incomplete or mismatched support cannot be promoted by the reporting layer.

The default leaderboard summary also emits `capability_diagnostics`: thirteen
persisted diagnostic dimensions and five-group support coverage on the shared
clean, measured-primary `(scenario, seed, pass)` intersection. Dimension means
use that dimension's common evidenced support across models; missing and
non-applicable measurements remain distinct. These are diagnostic means, not a
new capability ranking or an independent artifact certification. Operational-
agency profiles report successful-chain support and conditional scores; their
per-dimension opportunity denominator and success rate remain unavailable.
Process/resource columns use the same shared cells and expose measured counts;
provider latency sums are not episode wall time or monetary cost.
For version selection and campaign gates, see
[the evaluation hardening decision](research/2026-09-19-evaluation-hardening.md).

## Causal agency checks

Operational-agency credit requires a native event-to-action-to-effect chain and
a positive masked replay delta. Model prose, an emitted intent, or a transport
acknowledgement is insufficient. The diagnostic scorecard separately reports:

- proactive opportunity detection;
- correct silence during non-actionable intervals;
- completed valid alarm acknowledgement (semantic detection is not measured);
- response and intervention latency;
- takeover/cancel/supersession outcomes;
- tool efficiency and duplicate suppression;
- context truncation, repair and provider failures.

Those diagnostics, and the realtime supervision table, stay out of the 0.21
wait-relative primary. Long-horizon memory and proactive adaptation are not
claimed by the headline.

## Realtime clock and scorecard

Live `realtime_persistent` uses `native_dt_v1`. Bind `wall_tick_interval_s` to
the scenario's native plant quantum before the episode; do not slow the clock
for slow thinking. Default speed-scorecard membership requires native tick
`<= 60s` and `horizon * native <= 1800s`. That selects 37 Core rows: 7
autonomous driving (5s NGSIM supervisory ticks), 19 traffic, 6 power, and 5
datacenter (60s). CityLearn/microgrid hour ticks and long job-shop horizons
are not 1:1 operator-latency tests. Investigation still serializes on the
environment actor; `harness_periodic_supervisory_scan` remains false; hold is
not native takeover. Environment completion uses the remaining episode wall
budget to settle both provider requests and serialized observation ingestion.
If that budget expires, unfinished behavioral state stays ineligible; it is
not read concurrently or converted into a model score.

## Horizon analysis and Lite scope

Lite v11 keeps 141 rows, including 298-tick continuous scheduling and
400-tick fault/priority recovery representatives. The withdrawn v10 rule
removed every row above 192 ticks; that cost-only cutoff erased the long-tick
stratum. The retained representatives are chosen by shortest horizon within
frozen backend/family/difficulty-mode strata, not by model score.

A horizon chart must show Core and Lite separately with their denominators.
Core has 54/769 rows above 192 ticks; Lite has 2/141. This reporting split is
not a scientific definition of persistent agency, and Lite is not a
frequency-preserving sample of Core. Two long-tick representatives cannot
support a broad long-horizon capability or significance claim on their own.
Use matched, eligible results from the long-task diagnostic slice for that
analysis; report missing coverage rather than filling it with zeros.

Keep configured tick budget, realized decision/model-call count, native source
time, and provider wall time separate. DynaSched advances to native event or
machine boundaries: its declared `tick_minutes: 1` does not establish that
every step spans one physical minute. CityLearn's 72 hourly ticks are a long
source-time window, but elapsed source time alone does not demonstrate deep
planning or memory retention. Those claims require cross-boundary plan,
observation, action, and effect evidence.

The 1023-tick dynamic Dyna case remains in Core and long-task diagnostics.
Its cancellation, route, processing-time, maintenance and due-date changes
have value that the smaller representatives do not fully retain. Restoring
204/204 old selection features must not be described as complete dynamic
mechanism coverage.

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
