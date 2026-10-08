# Default Lite141 trajectory scoring (0.28)

The default reader remains `scripts/evaluate_trajectories.py`. Its policy is
`evaluation/policies/lite141_028.json`, which binds the immutable 0.27 source
policy plus independently rebuilt DSS normalization contracts. The panel,
weights, raw execution authentication, fulfillment, hard gates, and historical
recovery requirements are unchanged. Ordinary scoring makes zero model calls,
native episode replays, or source constructor calls.

## Population-normalized DSS quality

For both `opendss_ieee13` and `opendss_fresh_feeders`, let H be the source-fixed
horizon and M the audited required phase-node population. Neither is inferred
from observed, compliant, energized, or selected nodes.

The native producer records separate cumulative components:

- `C_count = 100 × sum(native voltage violation counts)`.
- `C_extrema = 1000 × sum(max(0, 0.95-v_min) + max(0, v_max-1.05))`.
- Production cost is zero for these locked backends.

0.28 uses the dimensionless exposure

```text
R = C_count/(100 H M) + C_extrema/(100 H)
N = 100/(1+R)
Q = min(F,N), except verified native hard failure gives N=Q=0.
```

Count exposure measures a node-time fraction. Global extrema severity retains
its time exposure and is not divided by population. Replicating equal node
conditions preserves quality; a severe local extreme is not diluted by adding
other nodes. This defines relative voltage quality, not absolute economic harm,
binary task success, or an independently calibrated business acceptance budget.
The native 100/1000 coefficients are preserved; they were not fit to model scores.

Components must exactly reconstruct from the original complete native records,
match their source count limits and authenticated total C, and use the producer's
terminal two-decimal rounding. The numerical score interval propagates that
rounding for the extrema component. It is not a statistical confidence interval.
Missing component evidence stays unknown; the reader does not subtract a
guessed count loss from total C or reuse a cached 0.27 N as new quality.
Verified hard zero does not require inventing missing components or F.

Original C, component values, F and execution identities are never rewritten.
The S case column retains the frozen 0.27 total-cost exposure metadata. For DSS,
new N uses `normalization028.count_scale` and `extrema_scale`, not C/S.
`legacy_N027`, `legacy_Q027`, and `outcome027` retain the recomputed old-policy
values alongside `outcome028`. Other backends retain their original N/Q rules.

## Scoring and audit

```bash
uv run --frozen --no-sync python scripts/evaluate_trajectories.py \
  --run-dir '<completed-run-directory>' --output-dir '<new028-output-directory>'

uv run --frozen --no-sync python scripts/evaluate_trajectories.py \
  --manifest '<frozen-input-manifest.json>' \
  --audit-report '<028-output-directory>/report.json' \
  --output-dir '<new-audit-directory>'
```

An original manifest retains its own bound policy; the default never upgrades
an old manifest silently. To recompute 0.27 explicitly from raw runs:

```bash
uv run --frozen --no-sync python scripts/evaluate_trajectories.py \
  --policy evaluation/policies/lite141_027.json \
  --run-dir '<completed-run-directory>' --output-dir '<new027-output-directory>'
```

Historical reports keep their original scoring runtime. The audit still requires
that runtime's bytes; evaluating an old policy on new code creates a new report
with its actual runtime, rather than relabeling an old receipt.

Display aliases cannot manufacture a composite model: automatic manifest
creation rejects many-to-one actual-ID mappings, and scoring independently
rejects selected whole episodes whose display label combines distinct actual
model IDs. Model rows expose `selected_actual_model_ids`.

All 141 eligible Q values are required for a point score/rank. All 131 applicable
F values are required for the F column. Missing evidence keeps its fixed weight
and identification bounds. Historical replay-equivalence and source-law evidence
remain separately bound; a stronger normalized N never manufactures missing F.
The outcome index remains separate from proactive, response, memory, and realtime
mechanism diagnostics. Their definitions and the original evidence requirements
remain in [the frozen 0.27 protocol](EVALUATION_027.md).
