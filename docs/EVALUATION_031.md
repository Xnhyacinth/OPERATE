# Lite141 reference-anchored outcomes 0.31

Current: [0.31.2 service-bounded scoring](EVALUATION_0312.md).
The definitions and results below describe historical 0.31.0/0.31.1.

0.31 is an opt-in, separately versioned scoring protocol for the existing
Lite141 trajectories. It reads authenticated 0.30 case evidence (settled native
cost C, fulfillment F and hard gates) and changes how cases are put on a common
scale, weighted and reported. 0.28 and 0.30 reports, policies and evidence are
unchanged. No provider call, native episode replay or source construction is
made.

## 0.31.1 revision (historical)

0.31.1 is bound by `evaluation/policies/lite141_0311.json`, with anchors in
`evaluation/policies/reference_anchors0311.json`. It fixes definition flaws
found in a review of 0.31.0 and adds the CityLearn reference planned there.
Where this section differs from the 0.31.0 text below, 0.31.1 applies. The
0.31.0 policy still reproduces its published case and model scores exactly.

**Anchor meaning.** 0 is the seeded do-nothing controller on every case. 100
means closing the do-nothing-to-reference cost gap; the joint score reaches 100
only with full service. On this panel the selected reference itself scores 100
on 55 of 141 eligible cases. On the other 86 it leaves source obligations unmet.
Upper anchors are heuristics, and 131 of them are privileged: `oracle_offline`
or the CityLearn tariff reference read future or native information. Scores
are therefore not a percentage of optimal capability.

**Continuous, non-compensatory combination.** 0.31.0 dropped the service cap
once fulfillment reached 99%. With a cost axis of 150, raising fulfillment
from 98.999% to 99% moved the case score from about 98 to 150. Its regression
guards jumped the same way at their tolerance edges. 0.31.1 takes
`S = min(axis caps)`, where:

- the service cap equals `f` until the last `ρ = 5` points of the do-nothing
  shortfall are closed, then rises linearly to the score ceiling at full
  fulfillment;
- an uninformative axis imposes no cap inside its tolerance (`δ` fulfillment
  points or `h_min` of cost) and falls linearly to the 0.31.0 regression value
  over the next `ρ` points.

The score is continuous and monotone in C and F. Outside these narrow bands,
cost savings cannot offset unserved obligations: a trajectory with cost axis
225 and service axis −60 still scores −60. The 14 rows that sat in the old
jump zone change slightly. Rank Spearman against the 0.31.0 threshold rule on
the same anchors is 1.000; varying `ρ` (2, 10) or `δ` (0.5, 2) gives at least
0.998; clip bounds [−100, 200], [−50, 125] or [−100, 100] give at least 0.98.

**CityLearn reference.** On the nine tariff-arbitrage CityLearn cases,
`oracle_offline` charged before a peak and discharged during it at a fixed
rate. Under the `fixed_reset_deliverable_energy_v1` settlement it cost more
than doing nothing, so 0.31.0 excluded those cases. The new privileged
`citylearn_tariff_reference` (`baselines/citylearn_tariff_reference.py`)
works per building, reading the locked load, PV and price windows:

- it charges from the building's own PV surplus;
- it discharges only to cover the building's own demand when the price is at
  or above a window quantile;
- it leaves residual energy to the terminal settlement.

Eighteen configurations are fixed in code: rate cap {0.05, 0.1, 0.2, 0.3,
0.5, 1} × discharge quantile {0, 0.5, 0.8}. Each ran twice per case with
deterministic agreement (360 CPU episodes, no model calls). The anchor builder
keeps the lowest non-regressing cost. All nine cases gain headroom (5.6%–82%),
so 141 of 141 cases are eligible and building energy returns to its full 1/6
domain weight. Replaying a recorded HY3 storage schedule under the current
runtime gives 27.9 against its historical 26.7, so these gains are not a
runtime artifact. Every model episode's recorded no-action cost also matches
the current do-nothing anchor exactly, on all backends except pymgrid, where
the recorded counterfactual is economic cost rather than task loss.

Consequence: CityLearn headroom is narrow relative to the spread of model
costs. Building energy accounts for 37% of between-model variance, driving
for 25%, and 35 of 190 building-energy case scores sit at a clip bound.
Weights are not tuned on this. The building-excluded order has Spearman 0.933
with the full order, and the driving-excluded order 0.944.

**Uncertainty.** The primary interval is a domain-stratified source-cluster
bootstrap: whole physical source clusters are resampled within each domain,
with fixed domain mass. The case bootstrap is kept as a companion. Datacenter
has only 3 clusters and is flagged as coarse. A compact letter display replaces
tiers for comparisons: models sharing a letter are not resolved at P ≥ 0.95.
Letters may overlap, so an unresolved pair is never split across groups.
`heuristic_groups` keep the old chained rule, named as heuristic.

**Influence diagnostics.** Every report includes leave-one-case-out effects
(rank-shift distribution, worst cases, minimum Spearman), case and family
shares of between-model variance, clip counts by domain, and clip-bound,
`δ`, `ρ` and rule sensitivities. The driving-excluded index is a co-primary
column, shown next to the driving retest reliability.

**Process scorecard.** An absent `event_response_records` key, a record
without `event_id`/`response_status`, or a causal record without integer
ticks now makes event metrics unknown instead of zero. An empty record list
still means no response. Metrics with no variation across models are flagged
`non_discriminating_on_panel`. Currently these are the hidden-event response
rate and foresight mitigation. The flag is not evidence that a capability is
absent.

**Case substitution.** `--case-spec` substitutes explicitly listed whole
episodes from re-authenticated supplements, with provenance. It reproduces the
HY3 OpenDSS completion (three cases) without an ad hoc script.

Reproduce 0.31.1:

```
uv run --frozen --no-sync python scripts/calibrate_native_quality_references.py \
  --suite benchmark/lite_suite.json --citylearn-tariff-grid \
  --output PATH/TO/NEW_citylearn_tariff_refs

uv run --frozen --no-sync python scripts/build_reference_anchors031.py \
  --reference-dir PATH/TO/reference_episodes \
  --extra-reference-dir PATH/TO/NEW_citylearn_tariff_refs \
  --evaluation-version 0.31.1 \
  --output NEW_anchors.json --policy-output NEW_policy.json

uv run --frozen --no-sync python scripts/evaluate_task_outcomes031.py \
  --policy evaluation/policies/lite141_0311.json \
  --base-report PATH/TO/report030.json \
  --base-report-sha256 d95ed6a0c583425a4479908ebdb53a89b81a25cb1b6833b955253b914e4e482c \
  --domain-spec PATH/TO/driving_domain_spec.json \
  --case-spec PATH/TO/case_spec.json \
  --episodes-journal ... --output-dir PATH/TO/NEW_report0311
```

New exports: `influence.json` and `uncertainty_case_bootstrap.json`.

## Why a new protocol

Diagnosis of the 0.30 index on the same panel:

- `N = 100/(1+C/S)` uses a source-intrinsic scale S that is not tied to what is
  achievable. The seeded do-nothing controller scores 39.96 and the offline
  oracle 59.84, so 20 complete models span only 39.5–51.8.
- 23% of the weight sat on cases where no reference controller beats doing
  nothing, while routing costs sit 8–20× above S on the flat tail of the curve.
- `min(F, N)` compares incommensurate quantities (routing F≈98 against N≈10;
  driving F=0 forces Q=0), so seven near-binary driving cases produced 76% of
  between-model variance.
- Driving trajectories carried the historical recovery-token feedback defect.

[EVALUATION_029.md](EVALUATION_029.md) forbids new anchors added merely to widen
gaps. 0.31 is justified by construct validity instead: the policy already names
`wait_only` as the inaction lower bound and the offline/greedy controllers as
reference upper bounds, and each case score now states the fraction of an
achievable improvement that a trajectory realised. Anchors and weights use no
model outcome. Because the revision was designed after inspecting these
trajectories, describe results as retrospective, not preregistered.

## Per-trajectory score

For case i, the CPU reference controllers `wait_only`, `greedy_heuristic` and
`oracle_offline` are measured with the same native rules as model trajectories
(see [reference anchors](#reference-anchors)).

- `C_wait`, `F_wait`: do-nothing cost and fulfillment.
- `C_ref`: lowest cost among `oracle_offline` and `greedy_heuristic` that has no
  hard failure and does not reduce fulfillment below `F_wait - δ`.

Cost axis (scheduling/operating quality):

```
g = 100 · (C_wait − C) / (C_wait − C_ref)          if the cost axis is informative
```

The cost axis is informative when `C_wait − C_ref ≥ h_min · max(|C_wait|, |C_ref|)`.
Otherwise it only penalises regression beyond doing nothing:
`g = −100 · (C − C_wait) / scale` when `C > C_wait + h_min·scale`, else non-binding.

Service axis (completion of source obligations):

```
f = 100 · (F − F_wait) / (100 − F_wait)            if F_wait ≤ 100 − δ
```

Full fulfillment (`F ≥ 100 − δ`) never binds, so it cannot cap superior cost.
When doing nothing already meets the full obligation, service only penalises
regression: `f = −100 · (F_wait − F) / F_wait` when `F < F_wait − δ`.

Case score:

```
S = clip(min(binding axes), −100, 150)
S = −100                                            verified native hard failure
S = g                                               CityLearn (no fulfillment meter)
```

Do-nothing scores exactly 0 on every case. The selected reference closes 100%
of the cost gap; its joint score is 100 only where it also meets the full
obligation (37 of 132 eligible cases). On the other 95 the CPU reference itself
leaves source obligations unmet, and the service axis stays anchored to full
fulfillment rather than to a weak heuristic's shortfall. Unknown evidence keeps
its weight with bounds [−100, 150]; nothing is imputed.

Fixed rules: `h_min = 0.02`, `δ = 1.0` fulfillment point, bounds [−100, 150].
They are bound in `evaluation/policies/lite141_031.json`.

The cost objective is the settled native objective C for every backend. The
mission prompt describes objectives qualitatively ("maintain voltage limits …
control energy cost") and does not state numeric weights, so the backend's own
cost is the only numerically declared objective. Anchoring removes the scale
problem the 0.28/0.30 population normalizations addressed; the 0.30
population-normalized voltage loss remains a reported sensitivity.

## Reference anchors

`scripts/build_reference_anchors031.py` compiles
`evaluation/policies/reference_anchors031.json` from the deterministic CPU
reference executions under `PATH/TO/reference_episodes`:

- Every policy has two repetitions; both must be status-ok, hash-verified and
  agree on C, F, hard gate and voltage loss.
- Measurement reuses the reader's rules: settled native cost (pymgrid task
  loss, FJSP settlement), `measure_service`/`measure_completion`, fixed-population
  voltage completion, 0.30 driving recovery attainment and
  `extract_native_objective` hard gates. Applied to the local HY3 trajectories,
  this code reproduces all 141 authenticated 0.30 rows exactly (C, F and hard
  gate across 13 backends).
- Missing evidence, disagreement between repetitions or a foreign case fails
  closed.

Result: 132 of 141 cases are headline-eligible (126 cost-informative, 123
service-informative). Nine CityLearn cases are ineligible: neither reference
improves cost on them and do-nothing meets the full obligation. They appear in
every export with the machine-readable code `no_reference_headroom`. A stronger
CityLearn reference controller would restore them.

The anchor manifest also records, per case, the declared perturbations the
backend publishes as typed events in the do-nothing ledger. These IDs are
identical across all four reference policies on all 141 cases, so they are
exogenous and fix the process-scorecard denominators.

## Weights

Weight policy `equal_domain_then_case`: each domain 1/6, equal cases within a
domain. Ineligible cases are dropped and the remaining nominal weights are
renormalized globally, never within a domain; a domain with few informative
cases loses mass rather than letting one case stand for it. Building energy
therefore carries 2.0% instead of 16.7%; the other domains carry 19.6% each.
The largest case weight is 2.8% (5.6% under 0.30); the weight-equivalent case
count is 77 (57 under 0.30).

The previous hierarchy, domain → physical source → case, and uniform weighting
are reported as sensitivities. The choice uses manifest-only criteria
(effective case count, maximum weight). `evaluation/operational_weights.py`
implements all policies with one hierarchical rule; the default reproduces the
frozen 0.27 manifest byte-for-byte.

## Driving input

Historical driving trajectories carry the recovery-token feedback defect. The
optional domain specification replaces each model's driving rows with its
corrected-feedback retest, rebuilt from raw journals by the 0.30 reader:

- 16 models from the 2026-10-10 retest (`driving_retest_20261010` archives);
  all 112 rows match the remote grader's published C/F/Q exactly.
- hy4-preview, glm-5.3-flash-ioa and hy3-ioa from the local recovery-clock
  reruns.

The mapping is explicit. A model without a mapped retest becomes unknown in
driving, never keeps its historical row: claude-sonnet-5.5 (its local "retest"
is its historical journal) and iquest-m1. Execution identities stay recorded.
This is a descriptive cross-implementation substitution, not a same-run formal
merge.

## Uncertainty and diagnostics

- **Bootstrap**: domain-stratified case resampling with fixed domain mass,
  2,000 replicates and seed 0. Reports 95% task-sampling intervals and rank
  intervals; it does not estimate repeated-run variance.
- **Tiers**: a model starts a new tier when the top member of the current tier
  beats it in at least 90% of replicates. Ranks are meaningful only across tiers.
- **Domain variance share**: each domain's covariance share of between-model
  index variance. It is reported only and never used for weighting.
- **Test-retest**: case-level and model-level agreement between historical and
  corrected driving executions. The harness differs between the two runs.

## Process scorecard

`evaluation/process_scorecard031.py` reports process metrics beside the index,
with no headline weight. Event-response records list only answered events, so
they never set a denominator.

- **Response rate**: share of the case's published declared events that receive
  a causally linked control (one consuming the event evidence, with a backend
  effect) requested inside the event window. The window is the published
  deadline, else the declared perturbation duration.
- **Hidden-event response rate**: the same, over hidden declared events.
- **Proactive lead**: the remaining window fraction at the first credited
  request, averaged over all declared events (0 for unanswered events).
- **Intervention efficacy**: no-action counterfactual `normalized_prevention`
  where applicable.
- **Foresight mitigation**: successful preventions over forecastable realized
  events.
- **Invalid-decision share and tool-failure share**: lower is better.
- **Correct silence**: N/A, because Lite141 has no source-defined quiet windows.

Each metric reports measured/applicable coverage. A missing journal is unknown,
not zero.

## Reproduction

```
uv run --frozen --no-sync python scripts/build_reference_anchors031.py \
  --reference-dir PATH/TO/reference_episodes \
  --output NEW_anchors.json --policy-output NEW_policy.json

uv run --frozen --no-sync python scripts/evaluate_task_outcomes031.py \
  --manifest PATH/TO/raw_input_manifest.json \
  --domain-spec PATH/TO/driving_domain_spec.json \
  --output-dir PATH/TO/NEW_scoring031
```

`--manifest` rebuilds the 0.30 base from raw journals. `--base-report` with
`--base-report-sha256` accepts a hash-pinned 0.30 report instead; the output
then records `raw_reauthenticated: false`. A domain specification lists
supplements as `{label, manifest, models: {base_model: supplement_model}}`.
`--episodes-journal` adds journals for the process scorecard. Exports:
`report.json`, `report.md`, `table.csv`, `cases.csv`, `uncertainty.json`,
`sensitivity.json`, `process_scorecard.json`, `process_summary.csv` and
`checksums.json`.

## Interpretation

A 0.31 case score is the percentage of the do-nothing-to-reference improvement
a trajectory realised, limited by its weakest required axis. Negative values
mean worse than doing nothing. It is not binary task success, an optimality
gap, comprehensive safety or held-out generalization. Reference controllers
are heuristics, not optima, so scores above 100 are possible on the cost axis.
