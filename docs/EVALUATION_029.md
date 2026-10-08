# Operational outcome and capability scorecard (0.29)

The frozen retrospective reporting protocol is **0.29.1**, retaining the frozen
0.28 Q/N/F outcome definitions. Use `scripts/evaluate_operational_scorecard.py`
to reproduce the recorded fixed-panel results. The compatible low-level raw
reader `scripts/evaluate_trajectories.py` continues to produce 0.28 outcomes.
Historical policies and output directories remain immutable.

**Readiness update (2026-10-07).** Follow-up artifact and source-code review
confirmed a SUMO recovery-token public-clock mismatch and unresolved recovery
attainment versus terminal-dwell semantics. Voltage loss also uses population
fractions in DSS but absolute bus counts in LV/CIGRE. The historical numbers
remain reproducible; final decision-capability leaderboard validation is open.
Correcting a scorer cannot reconstruct decisions under corrected historical
feedback. These issues are separate from the optional six-axis extension.

The main experiment is the existing selected model trajectories on Lite141.
New model runs and the optional six-axis capability contracts below are **not
prerequisites** for this retrospective outcome protocol. The protocol evaluates
recorded operational performance on the selected panel. It does not require an
independent score for every capability in the benchmark's motivation.

The entry reads original journals and authenticated native evidence, with no
model call, native replay or source constructor. It produces the primary table,
component diagnostics, fixed-cohort ordering, measurement bounds, sensitivity,
and a protocol/checksum bundle. Missing evidence stays explicit.

## Frozen retrospective scoring and ranking protocol

The machine identifier is `lite_fixed_panel_retrospective_outcome029.v1`.
`retrospective_protocol.json` binds the source-suite and weight-manifest hashes.
The input manifest fixes selected whole episodes before scoring; do not select
the best score or splice attempts. Because the protocol was refined using these
existing trajectories, describe it as retrospective, not preregistered.

1. **Primary endpoint.** For a case, Q is zero on a verified native hard failure;
   otherwise Q=min(F,N) when service is applicable, or Q=N for the source-defined
   CityLearn structural exception. N retains the frozen 0.28 normalization,
   including DSS component exposure. The model score is the weighted sum of Q
   using the frozen domain → task family → physical source → case weights.
   It is a 0–100 operational outcome index, not binary success or percentage of
   cases solved.
2. **Utility choice.** The minimum implements a bottleneck utility: surplus in
   one component cannot raise the case score above its weaker component. It is
   monotone but not strictly discriminative in both components. Negative signed
   costs retain their native values while N saturates at100. These are explicit
   normative choices, not extraction errors. Do not replace them with a mean,
   product, new zero-cost anchor or model-population normalization simply to
   widen observed model gaps. Any later change needs a separately versioned
   protocol and recomputation for every compared model.
3. **Required companion results.** Publish N, applicable F and coverage, raw
   native costs/components, declared hard failures and safety coverage, domain
   and task-family summaries, same-case native C/F comparisons, measurement
   bounds, and whole-source/domain deletion sensitivity alongside Q. A Q tie
   does not establish equal native performance. Raw DSS C/F comparisons refer
   to the original objective and can differ from component-normalized N.
4. **Fixed denominator.** All141 Q values must be determined for a point score
   and rank. Incomplete models remain in the report with their combined missing
   evidence/numeric-precision bounds; do not impute zero or renormalize observed
   cases. Source-authorized structural F non-applicability is distinct from
   missing F. A verified hard zero can determine Q even if F is unavailable.
5. **Point order.** Rank complete eligible models by unrounded Q descending,
   using competition ties (1,1,3). State the complete cohort's members and size
   and the total number of declared models. Missing models have no point rank.
   Adding a model does not change existing scores; it can change ordinal ranks.
6. **Resolution of order.** A positive lower bound on paired Q difference
   supports a recorded-measurement ordering. Overlap means unresolved, not a
   tie; overlapping pairs must not be collapsed into transitive tie groups.
   Report interval-compatible competition-rank ranges within the complete
   cohort and separate hypothetical ranges over all declared models, including
   missing evidence. These bounds are not assigned ranks for incomplete models.
7. **Sensitivity.** Removing a whole physical source or domain and renormalizing
   retained original weights checks dependence on panel composition. This is
   distinct from measurement precision and does not estimate repeat-run noise.
   No repeat-run confidence interval or statistical superiority is claimed.
8. **Interpretation.** Results can support comparisons of operational performance
   on Lite141 and evidence-linked analyses of service, resource cost and observed
   recovery. A good Q alone does not prove a specific internal cognitive
   mechanism, causal benefit over inaction, or all six capabilities separately.
   Lite is a model-informed development panel, so held-out generalization is
   outside the claim. Task semantics below accompany the result table.

A complete model's retrospective ranking eligibility is independent of
`capability_ranking_ready`. The latter applies only to the optional six-axis
extension. Original execution provenance fields are retained without changing
what the retrospective score measures.

## Relating results to the benchmark motivation

The primary endpoint measures the consequences of operational decisions. The
following analyses can explain the motivation using existing records without
inventing six independent numerical scales:

| Intended ability | Existing evidence and defensible analysis |
|---|---|
| Scheduling decisions | Source-defined service/progress F together with native scheduling/operating costs; compare concrete decision sequences under the executed task semantics. |
| Global coordination | Joint service/cost outcomes and source-linked state changes across affected entities; a high Q alone does not identify global reasoning. |
| Resource preservation | Native cost components alongside service, so low expenditure caused by abandoned service is visible; keep original units and objectives. |
| Adaptation | Disturbance, observation, action/receipt/effect and subsequent native-state evidence in selected trajectories; distinguish a documented recovery from a causal improvement claim. |
| Proactivity | Recorded agent-scheduled reviews, observations and action sequences as behavioral case evidence; activity counts alone are not a proactive-capability score. |
| Timeliness | Action/effect timing against obligations actually executed by the backend; wall-clock supervision belongs to the separate realtime treatment. |

Only use a row's proposed analysis when its original records contain the stated
evidence. These analyses supplement the fixed-panel outcome comparison; they
are not extra weights in Q. The source-contracted six-axis extension below is
reserved for explicit independent opportunity-rate measurements.

## Paper reporting language

> We evaluate the recorded trajectories on the fixed OPERATE-Lite141 panel
> using the 0.29.1 retrospective reporting protocol and frozen 0.28 operational
> outcome index. We aggregate case scores with source-fixed hierarchical weights
> and report native quality, fulfillment, safety coverage, domain results, and
> component tradeoffs. Point ranks include only models with all141 case scores;
> incomplete results retain fixed-denominator measurement bounds. Measurement
> ranges and source-deletion sensitivity are reported separately. The results
> characterize this model-informed panel and do not estimate repeated-run
> variability or held-out generalization.

## Design and research basis

OPERATE tests a supervisory decision center in source-grounded executable
systems. The measurement targets are scheduling, global coordination, resource
preservation, adaptation, proactivity and timeliness. These are separately
specified constructs; six convenient proxies are not an interchangeable scale.

[HELM](https://arxiv.org/abs/2211.09110) motivates reporting multiple metrics and
coverage gaps. [MetricEval](https://aclanthology.org/2023.emnlp-main.676/)
distinguishes reliability from construct validity. Accordingly, deterministic
reproduction and source authentication do not certify that a capability has
been adequately measured. [CheckList](https://aclanthology.org/2020.acl-main.442/)
motivates capability-specific behavioral, invariance and directional tests.
The transfer of these principles to OPERATE is a design choice, not a claim
that those papers validated this scoring implementation.

The [OECD/JRC composite-indicator handbook](https://www.oecd.org/en/publications/handbook-on-constructing-composite-indicators-methodology-and-user-guide_9789264043466-en.html)
explains weighting, compensability and sensitivity. Consequently, this version
does not replace the declared outcome utility with an uncalibrated arithmetic
or geometric six-axis mean. Neither aggregation makes critical failures
impossible to offset automatically; recorded hard failures and unmodeled safety
coverage are reported separately.

[Agarwal et al.](https://arxiv.org/abs/2108.13264) discuss uncertainty from few
runs and performance profiles. With one selected trajectory per case, this
reader reports exact recorded differences and deterministic sensitivity,
not repeated-run confidence intervals. The repeated success measure in
[tau-bench](https://proceedings.iclr.cc/paper_files/paper/2025/file/1b126cc38b8638e07bef37e7b2bb72bf-Paper-Conference.pdf)
likewise requires repeated executions. Its discussion of end-state evaluation
also motivates checking process obligations separately from terminal outcomes.

## Three parts of the report

1. **Operational outcomes.** Frozen Q/N/F, raw case costs and objective IDs,
   cost components, safety coverage, fulfillment/native bottlenecks, signed-cost
   saturation, domain and task-family summaries. Missing values keep their
   fixed weight. F excludes only source-authorized structural non-applicability.
2. **Component comparisons and sensitivity.** For the same case and native
   objective, lower C and higher applicable F define a Pareto comparison;
   verified native catastrophic failure takes priority. Dominance, tradeoff,
   equality and unknown remain distinct. No opponent-dependent win share is
   promoted to an absolute capability score. Deleting each whole physical
   source or domain and renormalizing the remaining *original* weights shows
   changed-panel influence. This does not preserve the original target
   distribution or constitute a confidence interval.
3. **Capability opportunity diagnostics.** Source-defined opportunities and
   trusted native predicates establish fixed measurement denominators. The
   source contract, not a model-generated score or the set of successful
   actions, supplies the expected opportunities. Unmeasured capabilities stay
   N/A, with reasons, evidence support and coverage.

The paired Q bounds include recorded numerical precision and missing
measurement. Original 0.28 missing-measurement bounds remain separately
available and condition on recorded point costs. DSS N intervals propagate
component rounding; other native intervals preserve the authenticated source
reader's precision. No interval is relabeled statistical significance.

## Task semantics

`task_semantics.json` accompanies all results. Current routing tasks execute
source demand dispatch waves, not full original VRPTW travel/service windows.
Alibaba lateness is queued-job lateness, not a verified completion SLA. SUMO F
is conditional terminal supervisory recovery, not route arrival. LV/CIGRE F is
the minimum of marginal voltage compliance and delivered energy, not a joint
load-node-time meter. DSS measures voltage exposure; CityLearn F is structurally
inapplicable. The complete retained definitions are in
[the fulfillment contract](EVALUATION_026_MAIN_TABLE.md#task-semantic-acceptance).
These semantic boundaries are not silently changed by better reporting.

## Source capability contract

The optional `operational_capability_contract029` object must be embedded in
scenario YAML whose bytes are authenticated by the source policy. Identity and
horizon must match the source contract. An episode, sidecar or caller-provided
self-hash cannot authorize different thresholds or fewer obligations. Existing
frozen scenarios must not be edited to inject a new measurement; new source
contracts require their own materialization and policy boundary.

Contract schema: `operational_capability_contract029.v1` with
`scenario_signature`, integer `seed`, `horizon_ticks`, and `axes` keyed by:

| Axis | Required source construct |
|---|---|
| `scheduling` | Native service, feasibility, precedence and/or deadline predicates meaningful to the task |
| `global_coordination` | Predicates covering multiple affected entities and source-defined shared obligations |
| `resource_preservation` | Joint required service and resource predicates; conserving resources while abandoning service cannot pass |
| `adaptation` | A source disturbance/invalidated obligation, effective response and subsequent native outcome |
| `proactivity` | Fixed discoverable positive opportunities and safe quiet contrasts, with autonomous observation/effect before a mandatory alert |
| `timeliness` | Effective native outcome inside the task's permitted action/effect window |

An applicable axis declares `applicable: true` and nonempty `opportunities`.
Explicit `applicable: false` requires a source reason. An absent axis is unknown
scope, not structural non-applicability. The engine checks the resource/service
roles and multiple-entity shape, but these syntactic checks do not establish
physical coupling, valid service thresholds or construct calibration.

Each opportunity declares:

- A globally unique `opportunity_id`, `start_tick`, `end_tick`, and optional
  `observation_start_tick` for earlier legitimate observation/commitment.
- Optional `effect_start_tick` (default observation start), which sets the
  earliest permitted effect. Prevention may legitimately act before an adverse
  outcome window; recovery can explicitly require effect after the disturbance.
- `mode: action_required | quiet` and `scope: terminal | throughout`.
- Nonempty `predicates`: `{metric, operator: eq|ge|le, value}` over finite
  engine-native values. Resource predicates declare `role: service|resource`;
  global predicates identify `entity`. Thresholds must come from task semantics,
  not observed model ranks.
- `discoverability: {kind: query_reachable|initial_mission|mandatory_delivery,
  source_evidence_ids: [...]}` and `feasibility: {source_evidence_ids: [...]}`.
  These are source-fixed eligibility proofs, not a condition on whether the
  tested model actually queried or depleted resources.
- For proactive positive opportunities, integer `mandatory_alert_tick`; passive
  mandatory delivery cannot be used as the autonomous discovery condition.

The source proof IDs must resolve to timely engine `capability_source_fact`
records with matching `fact_type`, containing `opportunity_ids`, and a nonempty
`source_fact` explanation. Missing/mismatched proofs keep the opportunity unknown.
These facts must be produced from source/native logic; an engine assertion or
hash alone does not scientifically establish feasibility. Independent source
review and positive/negative controller calibration remain necessary.

All timing here is native logical ticks. Realtime wall-clock latency remains in
its [independent treatment](AGENTIC_INTERACTION.md); this module does not turn
logical provider-wait time into realtime evidence.

## Native records and attribution

Engine `capability_opportunity` records carry `opportunity_id`, `phase: sample`
and native `metrics` at every tick from observation start through end. They also
carry actual `visible` and `observation_origin` (`autonomous_query`,
`agent_scheduled_review`, `initial_mission`, `mandatory_delivery`, or
`not_observed`). Actual visibility determines evidence for a response, never
whether an independently discoverable opportunity enters the denominator.

A `phase: close` record closes the reconciled window and declares
`action_lifecycle_complete: true`. Quiet windows additionally require
`control_census_complete: true` and `attempted_control_call_ids`, including
failed or ineffective controls. Quiet windows cannot overlap necessary action
windows. Quiet credit requires the native predicate to hold and no new control
attempt, but does not force an artificial wait call. Authenticated standing
commitments submitted earlier can take effect during quiet windows; unlinked
effects remain unknown.

Adaptation, proactivity and timeliness require a consumed observation, a valid
successful tool lifecycle and an engine effect with matching call, action and
state transition. Canceled, stale, superseded or ineffective acknowledgements
cannot supply an effective action. Alternative valid tools are accepted; there
is no gold tool path. The output distinguishes native predicate outcomes from
outcomes associated with linked effects. **Association is not proof of causal
improvement**; `causal_improvement_verified` remains false. Stronger causal claims
require an independently bound appropriate counterfactual, outside this new
opportunity-rate implementation.

Complete evidence and an unmet predicate or missing valid response gives zero.
Missing samples, observation delivery state, source proof or lifecycle closure
is unknown and preserves the denominator. Proactivity averages the positive and
quiet rates equally only when both fixed classes are measured; action spamming
and universal inactivity cannot exploit class imbalance. Other axes report
complete source-defined opportunity fulfillment rates. No per-axis point is
emitted for incomplete required opportunity evidence.

## Optional six-axis extension readiness

The fixed-denominator, predicate, attribution and report machinery is
implemented and behavior-tested. Current released Lite141 scenarios do not
contain this new contract, so their six-axis coverage is unknown. Historical
trajectories are not backfilled with invented source facts or observations.
`capability_ranking_ready` remains false even for synthetic complete fixtures:
conformance tests are not empirical capability calibration.

For a future independent six-axis ranking (not this retrospective main table),
freeze meaningful source contracts,
verify native measurement producers and known-group controls (local greed,
resource depletion, passive-only monitoring, over-intervention and delay), and
establish fixed comparable support. Scheduling/global/resource fields must not
be renamed proof of a cognitive mechanism merely because constraints were met.
No all-capability composite is defined in this version.

## Commands and outputs

```bash
uv run --frozen --no-sync python scripts/evaluate_operational_scorecard.py \
  --run-dir '<original-completed-run-directory>' \
  --output-dir '<new-scorecard-directory>'

uv run --frozen --no-sync python scripts/evaluate_operational_scorecard.py \
  --manifest '<frozen-input-manifest.json>' \
  --artifact-root '<restored-sha256-artifact-directory>' \
  --output-dir '<new-scorecard-directory>'
```

The manifest must bind the 0.28 source policy. Outputs include authenticated
`outcome028.json`, `report.json`, `report.md`, `table.csv`, `domain_summary.csv`,
`capability_summary.csv`, `task_semantics.json` and the original input manifest.
Additional outputs are `retrospective_protocol.json`,
`retrospective_ranking.json`, `rank_sensitivity.json`, `pairwise_summary.csv`
(when at least two models are declared), and `checksums.json`. The checksum
manifest covers every other exported file; it is an integrity inventory, not a
digital signature. Archive it with the original evidence and scoring runtime.

CSV retains F, coverage, hard-failure/safety scope, missing/precision bounds,
capability reasons and qualification flags; a point rank is explicitly an
outcome-index rank. Original execution differences are retained as provenance,
but this extension does not make them the capability measurement target.

For an independent rebuild of a 0.29 report, add `--audit-report '<report.json>'`
and use a fresh output directory. Scoring runtime bytes must match; relocation
of authenticated bytes and Git bookkeeping do not change the semantic binding.
Historical 0.28 audits still require their own frozen scoring runtime. The new
reader does not silently relabel historical receipts as current executions.
