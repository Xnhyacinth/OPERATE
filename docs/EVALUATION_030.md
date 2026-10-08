# Lite141 task outcomes 0.30

0.30 is an opt-in retrospective scoring revision for the existing Lite141
trajectories. It preserves the released scenarios, original evidence, historical
0.28/0.29 results, fixed task weights, safety gates and missing-evidence rules.
It does not require new provider runs or six new capability contracts.

The score measures recorded operational outcomes on a model-informed development
panel. It is not a held-out generalization estimate or an isolated ability score.

## Task corrections

For pandapower LV and CIGRE distribution, let H be the source horizon and M the
fixed required bus population. The revised dimensionless loss is

```
R = voltage_violation_cost / (1200 * H * M)
    + sum(other_native_cost_components) / original_source_scale
N = 100 / (1 + R)
```

The source-authorized signed-loss saturation rule remains in force. The voltage
component is checked against original per-tick violating-bus counts. The fixed
population cannot shrink because the model disconnects or sheds service.
Operating, unserved-energy and terminal-asset terms retain their original scale;
they are not all divided by M. DSS retains its existing 0.28 population and
extrema normalization. This unifies the population interpretation without
pretending that the native backends expose identical physical measurements.

For driving, F measures the attained consecutive nominal dwell after the most
recent native minimal-risk episode, capped at the source-required dwell. Nominal
activity before the first MRM does not count toward recovery. A degraded tick
interrupts the current streak but does not erase a completed recovery; a renewed
MRM resets the recovery obligation. Terminal mode and terminal nominal dwell
remain separate diagnostics. With no MRM, conditional recovery remains not
required; F=100 is not a claim of perfect driving or nominal terminal health.
Native risk and hard-failure gates continue to affect the outcome.

The common composition remains Q=min(F,N) when service/recovery fulfillment
applies; the source-authorized CityLearn exception remains Q=N. A verified hard
failure yields zero. Unknown measurements retain their weight and bounds, never
an imputed zero or a renormalized partial point score.

## Aggregation and interpretation

Keep equal domains, equal families within domain, equal physical sources within
family, and equal cases within source/family. Only models with every required
case measured receive an unrounded-Q competition rank. Domain/family results,
safety coverage, precision/missingness bounds and source/domain deletion
sensitivity accompany the scalar index. These bounds are not repeated-run
confidence intervals.

Task-specific utility and a common aggregation rule serve different purposes.
Native obligations, physical units and feasibility differ across tasks, so a
single raw loss formula would erase relevant meaning. Fixed aggregation makes
those declared task utilities comparable as one benchmark-defined preference.
The chosen transforms and weights are explicit conventions; the implementation
does not establish their unique optimality. Native component tradeoffs remain
available alongside the scalar score.

## Historical driving feedback

The runtime now reports recovery token expiry in the same decision-tick
coordinate exposed to the agent. Internal token TTL, state binding, health
checks and single-use validation are unchanged. The previous interface used the
last completed native tick in this receipt. The correction applies to future
executions only. Existing trajectories retain that historical feedback defect;
rescoring cannot recover what a model would have done under corrected feedback.
Use the driving-excluded sensitivity result as a companion, not a replacement
main table or a counterfactual corrected-feedback result.

## Reproduction

```
uv run --frozen --no-sync python scripts/evaluate_task_outcomes.py \
  --manifest PATH/TO/original_input_manifest.json \
  --output-dir .hl/analysis/task_outcomes030
```

The command reconstructs 0.28 outcomes from original journals/configurations,
provider/native artifacts and bound recovery evidence, then applies the versioned
task corrections. It accepts no cached score report as a public scoring input.
It calls no provider, source constructor or native episode replay. The report
binds the actual evaluator implementation, source policies, fixed weights and
input manifest. New output directories prevent overwriting historical packages.

Exports include the preserved 0.28 base, 0.30 report, model table, domain table,
per-case score changes, ranking, deletion sensitivity and file checksums.

## Methodological comparisons

[Artificial Analysis](https://artificialanalysis.ai/methodology/intelligence-benchmarking)
combines evaluations with different scoring mechanisms through disclosed fixed
mappings and weights. [CityLearn Challenge 2020](https://www.citylearn.net/citylearn_challenge/2020.html)
normalizes multiple cost criteria against a specified rule-based controller.
[D4RL](https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/offline_env.py)
uses environment-specific fixed reference bounds. These support separating
within-task measurement from cross-task aggregation; they do not validate our
particular coefficients or justify fitting them to outside model rankings.

[Terminal-Bench task guidance](https://www.tbench.ai/news/writing-a-good-terminal-bench-task)
emphasizes explicit task goals and corresponding verification, allowing multiple
valid implementations. For continuous operational tasks, this supports checking
native outcomes against actual obligations rather than adding an undisclosed
terminal requirement or prescribing one action sequence.

The voltage change is a relative utility revision: the voltage-count term has
1/M of its former influence relative to the other native costs. It is not merely
a change of units. The intended interpretation is proportional source-population
exposure, with continued service guarded by F. Raw native C and its components
remain available; their Pareto comparison is a raw-cost diagnostic, not a claim
that the revised N has identical preferences.
