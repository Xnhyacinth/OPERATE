# Reference-anchored scoring

The optional anchored scorer uses policy `0.31.2` in
`evaluation/policies/lite141_0312.json`. The default raw trajectory reader
remains a separate `0.28.0` contract.

## Score

The main range is **[-100, 100]**. Verified native hard failure scores -100.
Missing cost, service or hard-gate evidence remains unknown, with bounds and
no complete-panel rank.

For an informative cost axis, the diagnostic gain is
`g = 100 * (C_wait - C) / (C_wait - C_ref)`. For an informative service axis,
`f = 100 * (F - F_wait) / (100 - F_wait)` is the service cap. The main score is
the minimum of applicable caps, clipped to [-100, 100]. Extra cost gain remains
a separate diagnostic and cannot compensate for incomplete service.

Where baseline service has no shortfall, a continuous non-regression guard and
measured fulfillment cap apply. An uninformative cost gap uses its own
non-regression guard. Cost-only tasks do not receive a fabricated service meter.
Where service applies, reaching 100 requires full measured service. A reference
cost is a normalization anchor, not a proven optimum.

All 141 cases keep fixed weights, with equal mass across six domains. References
include privileged offline controllers; they are not same-observation online
competitors. Source-cluster intervals, weight sensitivity and case influence
remain diagnostics, not certificates of a total ranking or generalization.

## Run

```bash
uv run --frozen --no-sync python scripts/evaluate_anchored_outcomes.py --help
```

Supply a recorded raw manifest or an explicitly hash-pinned report, the required
policy and a new output directory. Optional whole-episode substitutions must
be declared through the CLI's explicit specifications. Partial totals are never
silently combined.

Output binds the policy, references, evaluator implementation and entry point.
Raw evidence authentication, formal-run certification and compatible same-run
merge are separate checks. Rescoring does not upgrade the original execution.
The command makes no model calls, native replay or new source construction.
