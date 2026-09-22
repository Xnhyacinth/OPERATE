# OPERATE offline evaluation 0.23

The recommended revised headline is now documented in
[EVALUATION_023_TARGETS.md](EVALUATION_023_TARGETS.md) (0.23.1 fixed expert-target
attainment). This document preserves the historical 0.23.0 / v2 calculation.

The executable offline protocol is `operate_quality.v2`. It rescored authenticated
stored trajectories; it does not rewrite historical 0.21/0.22 results or certify
new provider executions. All component scores have evidence links. The LLM is
never the environment or scorer.

## Measurements and scope

| Dimension | Measurement | Current fixed Lite population |
|---|---|---|
| R — operational result | Native objective quality, with declared hard constraints; FJSP additionally requires complete native obligations | All 141 tasks, 13 backends |
| A — timely supervision response | Observe a declared outage, execute a native intervention within its response window, and verify native loss mitigation | 23 original pymgrid response-window tasks |
| L — staged fulfillment | Satisfy the existing native voltage-recovery requirement at each declared stage | 5 original LV multi-stage tasks |

A reports autonomous, mandatory-prompted and unresolved response origin
separately. A successful response after an alarm is not called proactive.
Neither correct silence nor unaided memory retention is measured by these
retrospective adapters. Current A/L coverage is concentrated in microgrid tasks:
a three-component score on this declared population is not proof of broad
cross-domain autonomy or long-term memory. Reports expose domain/backend support.
Expanding these claims requires additional source-defined opportunities, not
new names for tool counts or model-written plans.

The source compiler reads only canonical suite/scenario bytes, before model
results. It retains every R case and all structurally supported A/L cases,
including missed windows and failed phases. Source ineligibility is recorded
explicitly and never inferred from a model doing nothing. Original scenario
hashes and source assets remain unchanged.

## R: original mission semantics

Twelve native optimization backends keep their continuous cost/loss objective.
They do not acquire an invented binary all-demand-completed goal. Verified
fixed-policy native quality is mapped from [-100,100] to [0,100]; equality with
the worst eligible reference maps to 50 (that policy need not be wait-only).
This normalization is not a success probability, an absolute
completion rate or a claim that the reference is optimal. Native costs and
original 0.22 values remain visible.

FJSP is a feasibility task. All source jobs must have arrived and all required
operations (total minus source cancellations) must be complete. Incomplete
native obligations score zero; completion fraction is reported when the native
denominator is known. The completed schedule then receives normalized native
quality. Model-authored completion assertions cannot replace terminal counts.
Earlier arbitrary 50/60-point completion bands have been removed.

Optional externally declared `task_acceptance.v1` contracts support additional
feasibility tasks. These must be frozen, hash-bound native predicates and quality
bounds, not targets chosen after seeing model results.

## A: original response windows

`legacy_agency_adapter.py` compiles the scenario's existing
`pymgrid_native_state_loss_response_v1` recipe and declared grid outage. Its
fixed denominator includes every eligible original window. Scoring joins:

1. engine-recorded native outage;
2. actual pre-action input showing islanded state;
3. recorded allowed control call and proven native state change;
4. effect within the original native response deadline;
5. native window loss `sum(abs(balance_error_mw) * 200 + shed_penalty)`
   meets the original source thresholds: wait loss at least 1000 and actual
   reduction at least 100.

An unclosed fully recorded window is zero. Missing physical evidence is N/A,
not zero. Native event ticks and post-transition observation/effect boundaries
are explicitly converted. Within-tick investigation can precede a later action;
observation and action timestamps need not be equal. Autonomous and passive
origins use recorded decision reasons and mandatory-alert boundaries. Unknown
origin remains unknown even if the response itself can be verified.

The task-native window improvement is a binary validation of timely mitigation,
not a second sum of economic gain. Economic per-action CF improvement alone
cannot prove service recovery and no longer gives credit. Total episode loss
stays in R. The observed action/effect chain and policy-window outcome do not
prove one call's independent native causal contribution. Mitigation does not
necessarily mean loss has fallen to zero.

This A instrument measures timely supervision response, not the earlier proposed
balanced positive/negative proactivity classification. No quiet-tick negative
examples are invented. Existing standing protection must not require redundant
interventions merely to earn credit; unresolved protection evidence is explicitly
unavailable rather than scored as a missed necessary action.

## L: original multi-stage obligations

`legacy_persistence_adapter.py` reads original LV `phase_ticks` and
`minimum_reduction_each_phase`. At each fixed stage it recomputes native voltage
violations from the authenticated actual and wait-reference tick records.
The score is the fraction of original stage requirements fulfilled. It does
not trust an unbound episode summary or require a particular control sequence.
Early termination does not remove later required stages. Missing reference
proof withholds measurement. The reference's original runtime and hash remain
visible under the selected compatibility policy.

This measures native staged fulfillment, not how many plans/calls occurred or
how many ticks elapsed. It is narrower than general long-horizon memory.

## Aggregation and safety

Each measurement has a unique `(scenario_signature, seed, dimension)` key.
One physical episode may support distinct R/A/L measurements; its economic gain
is not summed three times. Main aggregation preserves domain balance:

1. Freeze the domain × dimension applicability matrix from source contracts.
2. Within each cell, macro-average effective sources and then backends.
3. Within a domain, use the fixed R/A/L utility weights 0.50/0.25/0.25,
   normalized only over structurally declared dimensions.
4. Macro-average the six domains with equal weight.

Structural absence is decided from the suite before reading model outcomes.
Missing evidence in a declared cell never removes the cell or changes its
weight: the affected overall score remains N/A. The current suite gives only
microgrid all three dimensions and the other five domains R. Consequently the
main score's effective weights are R=91.67%, A=4.17%, L=4.17%. This is the explicit
tradeoff for domain fairness with the current evidence population; it is not a
claim that agency receives half of the global score. Expanding A/L to other
domains needs actual source-defined measurements.

The older axis-first `0.50 R + 0.25 A + 0.25 L` is retained only as
`axis_first_diagnostic_index`: here it gives microgrid approximately 58.33%
implicit influence and must not replace the domain-balanced main score.
Weights are explicit utility choices, not fitted or proven-optimal constants.
Reports include domain weights, effective global weights and sensitivity to
0.4/0.5/0.6 R weight plus equal within-domain axis weights. Component profiles
retain their own fixed source -> backend -> domain macro averages.

Safety is a task-local native hard gate before aggregation, and a separate
failure rate deduplicated by physical episode. A measured hard-failure task
scores zero. Missing safety proof cannot create a score. Native soft losses are
not separately charged as an extra weighted penalty. Modeled constraints are
not a general deployment-safety certificate.

Every dimension keeps its fixed denominator. Missing measurements do not become
zero or transfer weights to available components. A full comparison requires
complete same-scope measurements for the compared models. A separately labeled
complete-model ranking may compare the covered models while excluding explicitly
listed incomplete models; it is not a ranking of the entire requested cohort. Partial coverage is
reported separately. Source-cluster bootstrap keeps a shared physical source
paired across dimensions and models, with the same macro estimand per draw.
Small strata are labeled and cannot support statistical-separation claims.
Single provider runs do not establish repeat reliability.

## Existing thirteen metrics

- Survival and executable hard safety constraints feed the safety gate.
- Economic cost and justified optimality references feed R; explicit service,
  equity or stakeholder requirements belong to their native task contract.
- Useful discovery and response evidence feed A; raw call/token counts remain
  efficiency diagnostics.
- Actual staged fulfillment feeds L; replanning prose does not earn points.
- Counterfactual effects remain causal evidence rather than another weighted
  copy of the same economic benefit.
- Ethical/stakeholder process proxies remain diagnostics unless an executable
  mission requirement gives them a valid native outcome meaning.

Original 0.21 total/primary and 0.22 native results remain beside each new
measurement. Their thirteen weights are not averaged into 0.23.

## Reproducible use and compatibility

First freeze measurements from an existing authenticated 0.22 input manifest:

```bash
uv run --no-sync python scripts/compile_operate_measurements.py \
  --manifest INPUT.json --output .hl/compiled_measurements
uv run --no-sync python scripts/evaluate_operate_quality.py \
  --manifest .hl/compiled_measurements/manifest.json --output .hl/evaluation_023
```

The input manifest binds selected episode JSONL files by SHA-256. The generated
`operate_measurement_suite.v2` binds the canonical suite and every source-derived
measurement contract. The scorer recompiles the entire retrospective population from locked
scenario bytes and checks exact membership and content, so deleting difficult
windows or forging an external recipe cannot override source eligibility.
Prospectively recorded generic `agency_measurement_contract.v1` instruments
remain supported by the separate generic evaluator, including genuine declared
negative opportunities and exact engine recording markers. Retrospective
adapters never invent such historical markers.

Default `strict` comparison retains execution/reference identity requirements.
The user's explicit `latest_framework_user_assumed` mode allows code-tree
metadata differences while preserving original values and all physical artifact,
scenario, cost, provider and profile checks. It does not turn one provider model
into another. Explicit recovery projections retain original status and a hashed
proof chain; they do not rewrite batch files or pretend to be new executions.

Offline rescoring itself does not require rerunning the model. Installation
changes are limited to offline evaluation modules and their tests. Active
provider runs using `implementation_policy=provenance` record code-tree changes
without treating those metadata changes as model failures; strict runs retain
their original restrictions. This does not authorize changing simulator behavior
or live runner code during an episode.
