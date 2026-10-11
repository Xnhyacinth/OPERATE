# Lite141 reference-anchored outcomes 0.31.2

0.31.2 repairs the service-completion semantics of 0.31.1. It is an opt-in
fixed-panel retrospective scorer, not certification of a new formal provider
run. The default raw reader remains 0.28.0. Historical policies, anchors,
episodes and reports are preserved.

## Frozen scoring decision

Policy: `evaluation/policies/lite141_0312.json`, canonical SHA256
`911263b1248266416d5a206474d09a55f7f8b5beaf75904ec402a1b20cd08293`.
It reuses the exact `reference_anchors0311.json` and `equal_domain_then_case`
weights. All 141 cases remain eligible; each of six domains has mass 1/6.
No reference, case, denominator or domain weight was selected to improve a
model's rank or equalize observed between-model variance.

The headline range is **[-100, 100]**. Verified native hard failure is -100;
missing cost, service or hard-gate evidence stays unknown, with bounds and no
complete-panel rank. A complete panel receives a point rank, not a claim that
every adjacent pair is statistically separated.

For an informative cost axis, retain the diagnostic value
`g = 100 * (C_wait - C) / (C_wait - C_ref)`. It may exceed 100 and is exported
alongside native C, C_wait and C_ref. For an informative service axis, use
`f = 100 * (F - F_wait) / (100 - F_wait)` directly as the service cap, without
the 0.31.1 uplift. The main score is the minimum of applicable caps, clipped
to [-100, 100]. Thus cost savings cannot lift this service cap above f.

When service has no informative baseline shortfall, retain the continuous
regression guard and additionally cap by measured fulfillment F. When cost
has no informative improvement gap, retain its continuous regression guard.
The guards use ceiling 100, with the previous tolerances and ramp width.
Their transition bands tolerate some regression; this is not a rule that
every infinitesimal regression must receive a negative score. Cost-only
tasks have no fabricated fulfillment meter.

Where service applies, 100 requires full measured service. Where the cost
axis is informative, it also requires reaching or improving on reference
cost. On an uninformative cost axis, the non-regression guard applies instead.
100 does not mean optimal capability or binary task success. Floating-point
rounding at display precision is not a one-percentage-point completion waiver.

## Why the ceiling changed

0.31.1 could score incomplete service above 100: the recorded HY4 case
`d590cf4d8d345e54` has F=96.7741935483871 and S=115.66077003121755.
Under 0.31.2 it scores 96.7741935483871. This conflicts with the old report's
claim that joint 100 requires full service.

Continuity in fulfillment, a strict full-service threshold for 100, and a
joint score above 100 at full service cannot all coexist. 0.31.2 chooses
continuity and the completion threshold. Extra cost improvement stays visible
in g and native cost, rather than raising the main score ceiling.

## Evidence and comparison scope

The report exports `certification.json`: raw reauthentication, source formal-run
certification and same-run merge certification are separate. Rescoring does
not upgrade historical execution identities. Corrected driving and the three
HY3 OpenDSS substitutions are explicit whole-episode replacements, not merges
of partial totals. Missing corrected driving remains incomplete.

The result is a fixed Lite141 retrospective comparison of recorded model
deployments. Provider, runtime and historical configuration differences remain
part of those observations. It is not a held-out benchmark claim, a repeated-run
confidence interval, or a strict same-implementation formal model comparison.

## Bias and uncertainty reporting

- Keep source-cluster bootstrap intervals, pairwise descriptive letters and
  the driving-excluded companion. Shared letters mean insufficient separation
  under the chosen bootstrap threshold, not proven equivalence. There is no
  family-wise multiple-comparison correction.
- `rank_sensitivity.json` exposes each model's rank under all declared
  alternative weights and domain exclusions. Its specification range is not
  a confidence interval and exclusions change the target task scope.
- Keep raw cost/fulfillment, clipping, case influence and domain variance
  diagnostics. Small reference gaps can amplify cost differences; neither
  reference gaps nor weights are tuned to suppress this observation.
- 131 reference anchors use privileged future/native information. They are
  normalization references, not same-observation online competitors. The
  other 10 use the existing greedy controller.
- Source clustering accounts for shared sources in task resampling; it does
  not create more independent data. Datacenter has only three source clusters.
- Historical versus corrected driving agreement is cross-harness agreement,
  not same-condition test-retest reliability.

The policy is frozen for subsequent evaluations. Generalization, online
baseline parity and repeated-run uncertainty require new experimental evidence;
changing the scorer alone cannot establish them.

## Verification and reproduction

Use the repository environment: `uv run --frozen --no-sync`. Run the focused
task-outcome, reference-anchor and process-scorecard tests and Ruff. Historical
0.31.0 and 0.31.1 case fields must remain unchanged under their own policies.
Private finalization artifacts retain the exact original argument vector.
The public snapshot includes the scorer, tests and immutable policies;
supply your own authenticated raw trajectory artifacts to reproduce scoring.

Use `scripts/evaluate_task_outcomes031.py --policy
evaluation/policies/lite141_0312.json`, with the recorded raw manifest or
explicit hash-pinned report, corrected `--domain-spec`, HY3 `--case-spec`,
and a new `--output-dir`. The output binds the policy, anchors, evaluator
runtime and entrypoint. No provider calls or native replay occur during scoring.
