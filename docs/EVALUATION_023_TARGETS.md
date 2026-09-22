# OPERATE evaluation 0.23.1: fixed-target attainment

This revision replaces the recommended headline calculation for offline Lite
comparisons. Historical `operate_quality.v2` / 0.23.0 reports remain immutable.
The new score is **fixed expert-target attainment**, not a renamed original-task
success rate, a calibrated probability, or a universal measure of model ability.

## Measurement design

Three views answer separate questions:

1. **Operational target attainment (headline):** does the run satisfy its native
   hard constraints and reach the frozen native objective target? FJSP also
   requires every source job to arrive and all required operations to complete.
2. **Operational behavior profile:** timely responses and sustained phase
   fulfillment, each on its declared source opportunity population. Historical
   A/L measurements cover 23/5 microgrid cases and are reported separately. They
   do not establish broad proactivity or long-term memory, and do not receive
   invented global weights.
3. **Native performance and resource diagnostics:** actual objective, target,
   signed native gap, constraints, and execution problems. Units remain native;
   currency, makespan and physical loss are not added together. Token/call counts
   do not earn behavioral points.

For each task i, let C_i be the authenticated native objective and T_i its
frozen target. The episode value is 100 if native feasibility holds and
C_i <= T_i (within numerical accounting tolerance), otherwise 0. Missing proof
or missing target is N/A. Signed costs work directly: -20 meets a -10 target;
-5 does not. There is no weak-policy anchor, zero-cost division, 50-point
offset, or normalization based on other submitted models.

Binary target attainment has a deliberate interpretation and limitation: near
misses and large misses both fail the target. Native gaps must remain visible;
ties must not be broken using unrelated metrics or model reputation. Fixed
absolute acceptance budgets or certified solver bounds are valid future target
sources when their task, horizon, information and cost definitions are proven.
The implemented initial target compiler uses the following existing independent
controller evidence; it does not claim to have certified optimization bounds.

## Frozen target population

The candidate policies are fixed as `greedy_heuristic` and `oracle_offline`.
Both require two authenticated, deterministic measurements on the same source,
seed, horizon and native objective. A proven infeasible policy is excluded;
missing execution/evidence is not proof of infeasibility and makes the target
unavailable. The minimum cost among eligible candidates becomes the target.
At least one eligible candidate is required. FJSP reference traces must prove
all source jobs have arrived in addition to operation completion.

The compiler reads source contracts and controller evidence, never model runs.
It freezes target values, candidate audit evidence, source hashes and repeat
provenance into a hash-bound target suite. Adding/removing competitor models
does not change this suite. Changing a target requires a new target-suite hash
and rescores the entire comparison under that version.

`oracle_offline` may have future information or controller privileges absent
from the tested model. These are expert/clairvoyant targets, **not certified
optima and not a certificate of attainability under the model's information
and action constraints**. Do not relabel the score original mission completion.
Wait/random policies remain useful controls or causal diagnostics but their
artifacts and extrema are not required by this headline evaluator.

## Aggregation, coverage and uncertainty

Compute means within effective source, then backend, then domain; average the
six domains equally. This is domain-balanced target attainment, not simply
passed cases divided by 141. Publish the latter count and micro percentage too.
Missing rows retain their denominator and make the model's full-suite score
unavailable. Compare only complete models on the same frozen population; never
rank a partial mean as a full-suite result. Genuine hard failure gates a valid
measurement to zero; it cannot manufacture a score for a missing target.

Same scores share ranks. One selected trajectory per task does not estimate
repeat reliability. Historical best-of-run selection is prohibited; retain
the existing explicit, provenance-preserving latest-first / old-fill policy.
Rank differences alone do not establish statistical significance.

The current Lite141 suite explicitly used development model outcomes to select
hard cases. It is an efficiency/development set, not a model-blind sample of
Full. This protocol does not change that membership or promote it to a formal
public leaderboard. Execution/provider identities remain recorded under the
user's explicit comparison policy.

## Framework defects and rescoring

Stored outcomes remain observations of the execution that actually occurred.
A verified framework defect must be labeled; fixing code does not retroactively
repair a trajectory. The September 21 SUMO recovery-token defect can block a
valid issued token after normal state advancement. Affected cases require
re-execution to measure behavior under the fixed environment; they must not be
described as pure model failures or silently replaced with hypothetical success.

Offline rescoring cannot create missing native evidence or absolute acceptance
targets. In particular, CityLearn service continuity alone cannot replace its
economic objective, and safe indefinite driving stops cannot be relabeled
successful travel. The fixed-target protocol retains the original objectives.

## Commands

```bash
uv run --no-sync python scripts/compile_fixed_targets.py \
  --manifest INPUT_MANIFEST.json --output .hl/target_suite
uv run --no-sync python scripts/evaluate_fixed_targets.py \
  --manifest .hl/target_suite/manifest.json --output .hl/target_scores
```

For another trajectory cohort, reuse the exact `fixed_targets` descriptor in
its frozen input manifest. The evaluator verifies episode artifacts/identities
without loading the old four-policy reference calibration. Original 0.21,
0.22 and 0.23.0 reports remain diagnostic historical comparisons.

## Design references

SWE-bench's [% Resolved](https://www.swebench.com/) illustrates an explicit
binary attainment estimand; OPERATE's task targets remain native to its own
domains. [Spider 2.0](https://spider2-sql.github.io/) distinguishes methods with
special information settings, reinforcing the need to disclose oracle access.
[tau-bench](https://arxiv.org/abs/2406.12045) treats repeated-trial reliability
as a separate measurement; it cannot be inferred from our single trajectories.
