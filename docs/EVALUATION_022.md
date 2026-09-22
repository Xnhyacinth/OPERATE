# Offline evaluation 0.22

Version 0.22 evaluates existing, authenticated trajectories with
`native_quality.v1` and a separate evidence-linked capability report. It does
not rewrite an episode's original scorer version, execution implementation,
provider profile or treatment. Legacy 0.21 rows remain 0.21 execution artifacts;
the new evaluation version identifies the offline evaluator.

## Outcome and scheduling quality

The native objective remains the declared simulator task objective. Fixed,
reproducible reference policies define the normalization scale; model outcomes
never determine that scale. Native cost, constraint failure and task success
remain separate. A feasible job-shop solution is compared by native completion
cost; an unfinished schedule cannot obtain a good score by stopping early.
The makespan scale uses its natural-zero floor, as specified in
[EVALUATION.md](EVALUATION.md). Native quality does not reward tool frequency,
plan prose, or similarity to a preferred tool sequence.

The full Index uses the fixed suite denominator and source → backend → domain
aggregation. Unavailable references or measurements remain N/A. A partial index
must not be labeled a full Lite 141 result. Cohorts retain execution, prompt,
provider and treatment identities; offline rescoring does not certify a formal
run. The default strict policy does not merge incompatible historical rows.

### Latest-framework comparison requested by the user

An explicit manifest policy, `comparison_policy: latest_framework_user_assumed`,
treats the selected latest-framework runs as one comparison population. It
reuses a uniquely selected reference for the same scenario signature and seed
without requiring the model and reference execution trees to match. It also
permits cross-run implementation/profile hashes within that declared population.
This is a user-declared comparability assumption, not a verified equivalence
receipt. Reports retain original execution/profile identities and remain
`formal_run_certified: false`; historical evidence is never relabeled.

Scenario membership, model identity, interaction-mode separation, attachment
binding, and per-episode execution start/end checks still apply. An actual
mid-run implementation drift or missing execution proof is distinct from a
stable model run using a different reference tree. Missing cases do not become
zero scores or reduce the fixed denominator. Multiple competing reference
scales for a case require an explicit selection rather than score-based choice.

## Behavior, initiative and temporal evidence

`evaluation.capability_report.build_capability_report(episode, scenario_spec=...)`
returns a companion report independently of whether native references are ready.
Its input artifact authentication uses
`bind_capability_evidence(episode, spec, source_path=None)` at the evaluation
entry point. The helper checks recorded attachment hashes, byte/event counts,
continuous realized trajectory ticks, the authoritative evidence inventory,
scoring snapshot identity, actual native costs and masked replay content.
Missing attachments withhold authenticated results; hashing a JSONL input alone
is not attachment validation. Headers, when present, must match scenario, seed,
backend and horizon. This is local artifact binding, not resistance to an actor
rewriting both manifests and artifacts. The capability profile itself checks the persisted operational-agency profile against episode-local evidence
identifiers and the actual individual/group masked replay deltas before
publishing conditional dimension scores. Every displayed temporal chain is
checked separately, preventing one valid chain from validating another.

The report exposes observed event/action/effect coordinates, observation-to-action
and action-to-effect spans, positive masked deltas, plan evidence and native
effect evidence. These describe verified causal responses, including the
existing pre-deadline initiative and temporal-planning dimensions. They do not
establish an opportunity success rate: the profile's support includes successful
chains, not an independently enumerated opportunity denominator. Missing or
inconsistent causal evidence produces N/A, not a model failure or an invented
zero. Trade-off quality remains unavailable without replay-bound alternatives.

The companion makes no composite capability score. Repeated tool calls and
repeated plan declarations do not improve native quality or create causal
credit. Equivalent effective strategies remain valid.

## Long-horizon interpretation

The matched suite supplies configured horizon; the episode supplies recorded
steps. Disagreeing recorded counts are explicitly unavailable. These metadata
are not an independently authenticated trace-coverage count. Source time is
not inferred from step count, especially for event-driven DynaSched. Provider
wall time, native source time and model decision count are different quantities.

The existing `>192` split is descriptive, not a threshold for memory competence.
Lite retains two long-tick representatives out of 141, whereas Core contains
54 out of 769. A plan-to-effect span alone cannot establish correct retention
across compaction or unseen future disturbances. The report therefore leaves
memory-retention scores unavailable; a defensible long-horizon study requires
matched long-task strata, cross-boundary obligation/observation/plan evidence,
and appropriate memory-treatment ablations. The benchmark-managed structured
memory treatment is not unaided model memory.

## Realtime separation

Logical episodes cannot measure provider-latency-dependent intervention or
correct realtime silence. Their realtime section is inapplicable. Realtime
results use the existing independent ledger-based `realtime-diagnostics/1.7`
scorecard with its action arbitration, cancellation, supersession, takeover and
latency joins. A realtime label or persisted summary alone is insufficient to
recompute those results. Neither the native Index nor the capability companion
pools realtime and logical treatments.

## Verification scope

Behavioral regressions cover missing causal evidence, changed masked replay
binding, tick-count disagreement, no mutation of source artifacts, and treatment
separation. Existing native-objective and agency tests retain feasibility,
actual-live-cost and positive causal-effect constraints. Reference calibration
and full-suite coverage are separate executable acceptance conditions, not
claims established merely by these unit tests.

The offline report also emits `long_task_diagnostic.v1` through
`build_long_task_report(graded_rows, suite_rows, model)`. Its denominator is the
predeclared suite rows above 192 configured ticks, not the observed successful
rows. Native costs remain visible with authenticated evidence even if matched
references are missing. The subset Index requires every declared long-task
case and the same source/backend/domain aggregation; the caller enforces
execution-cohort identity. This diagnostic never substitutes for the full Lite
Index or a memory-retention experiment.
