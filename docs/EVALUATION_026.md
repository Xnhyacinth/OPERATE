# Evaluation 0.26.1: source-obligation outcomes

0.26.1 scores existing authenticated Lite trajectories offline. It adds a
source-grounded measure of **what service was actually delivered**, while
retaining the 0.25 settled native-loss quality as a separate reported result.
No model, simulator, optimizer, wait-only reference or competing model result
is needed to compile a completion denominator or score a trajectory.

## What completion means

The denominator is frozen from the source and seeded scenario, including
initial work, work scheduled to arrive later, and exogenous demand or urgent
orders within the mission horizon. The agent cannot improve its completion
fraction by refusing, cancelling, hiding, or moving obligations beyond the
window. The numerator is *effective native delivery*, not a dispatch command,
plan, tool call or task acceptance. The original 0.25 evidence rules still
require a complete observation window, except for its explicitly validated
native mission-terminal cases. Missing or inconsistent evidence yields N/A.

| Task group | Fixed population | Delivered outcome |
|---|---|---|
| CVRP / VRPTW | Original customers and seeded urgent orders, weighted by source demand and declared criticality | Validly served customers; timely fraction separately reported |
| FJSP | All source operations, including future-arriving jobs | Operations actually completed; cancellation does not count. Complete-job fraction is a separate diagnostic |
| Alibaba GPU | Fixed source workload window, including future jobs | Jobs with native `done` status |
| Inventory | Complete locked demand stream | Demand units served by the backend |
| SUMO driving | Source task requires guarded recovery and six stable terminal ticks **if MRM occurs** | Without MRM the recovery obligation is met; after MRM, fraction of required consecutive terminal native `nominal` ticks. Route progress remains diagnostic |
| Pymgrid | Source load profile plus seeded load spikes | Demand minus metered voluntary and involuntary shed energy |
| PGLib UC | Locked case demand plus seeded surges | Demand minus metered shed or physical shortfall, counted once |
| Pandapower LV / CIGRE | Native load exposure and fixed source bus population | Minimum of delivered-load fraction and voltage-compliant bus-time fraction |
| OpenDSS feeders | Fixed source node population and episode ticks | Voltage-compliant node-time; separately modelled load shortfall is unavailable |
| CityLearn | Automatic building load service, no agent-controllable unmet-service meter | **No separate completion rate**; settled economic quality remains its case result |

For LV and CIGRE, the minimum is a **bottleneck of two marginal measures**.
It is **not** the fraction of load actually served at compliant voltage: the
two measures use different denominators, and the released evidence does not
certify a common load-to-compliant-bus-time allocation across both backends.
Neither multiplying the marginals nor
treating their minimum as a joint probability would recover that quantity. Native
overloads, reserves, fuel, energy, scheduling delay, route cost and terminal
resource settlement remain in the 0.25 native-quality column and native ledger.
OpenDSS node-time likewise measures reliability of the modelled voltage
condition, not a separately proven quantity of energy served.

The source-work measures are **progress at episode end**, not a judgment that
every unfinished job was feasibly finishable before the window closed. FJSP
counts completed operations, whereas Alibaba counts completed jobs; operation
progress must not be called finished-job delivery. The denominator still
includes late-arriving source work so the agent cannot improve the score by
refusing it. Original deadlines and native lateness/unfinished penalties remain
in the 0.25 quality result. Routing counts delivered original obligations even
when late; its separate timeliness diagnostic and native delay costs describe
that distinction. No new post-hoc due-date exemption changes an old score.

The earlier `source_obligation_outcome.v1` used SUMO route-position percentage
as its completion cap. The native meter tracks physical position, but the
seven source task contracts ask for risk mitigation, guarded recovery and six
terminal stable ticks, **not** arrival at the end of the roughly 2.5-km route.
None of 163 scored SUMO rows reached that end. The old cap contributed about
2.10 of Hy4's 2.12 cap points and 3.33 of Grok's 3.36, making this a material
construct mismatch. The old report remains an archived sensitivity result.

`source_obligation_outcome.v2` keeps casewise `min(F,N)` but defines SUMO
`F=100` when no MRM occurs; otherwise
`F = 100 * min(terminal_consecutive_nominal_ticks,
required_stable_dwell_ticks) / required_stable_dwell_ticks`. The denominator
comes from the original task (`6` in current Lite). Native tick records must
prove assurance state and MRM occurrence, with the MRM flag agreeing with the
native assurance mode on every tick. An agent that avoided MRM must not
be charged for failing to perform a recovery that was never required; after
MRM, only subsequent terminal `nominal` ticks count. Verified
catastrophe remains a hard zero. Route progress stays visible as a diagnostic,
and native delay costs remain in `N`. Partial dwell credit is a declared
progress measure, not a strict pass/fail rule. Full recovery credit, including
the no-MRM case, certifies only this conditional outcome requirement; it does
not certify all source prevention, supervision or recovery-process rules.
In particular, a no-MRM episode can end in the native `degraded` mode and
still receive `F=100` for the inapplicable recovery obligation. Its modeled
risk, intervention and delay costs remain in `N`; `F=100` alone must not be
reported as full safe-driving mission completion.

The six consecutive ticks are **supervisory tick-end** observations, not proof
that every 0.1-second SUMO physics substep remained nominal. A recovery can
transition from MRM to nominal within the first counted tick. The exact
six-tick criterion is frozen in the source scenario, but the strict model
briefing gives the general safety/progress objective and tool semantics rather
than that numeric scoring threshold. Treat this as retrospective outcome
measurement, not as a disclosed-budget mission pass rate. Whether the stronger
all-substep stability interpretation is appropriate requires a separately
declared task contract; it must not silently replace this score.

Each case now includes a hash-bound `source_completion_contract.v1` with
`completion_kind`, population scope, numerator definition, time-window rule,
late-arrival rule and evidence requirements. These fields distinguish job
completion, operation progress, demand fulfillment, stable recovery, node-time
compliance and the LV/CIGRE service-compliance bottleneck. CityLearn is
structurally inapplicable; an applicable case with missing completion evidence
is N/A rather than falling back to economic quality.

## Scores and interpretation

Let `F_i` be the source-obligation fraction for a task where it exists and
`N_i` the existing 0.25 settled native-quality score. The 0.26 case score is:

```
Q_i = 0     if a verified native hard failure occurred
Q_i = min(F_i, N_i) for 131 tasks with source-grounded outcome evidence
Q_i = N_i   for 10 CityLearn economic-only tasks
Index = sum(frozen_0.25_task_weight_i * Q_i)
```

The minimum is a non-compensatory requirement: high service delivery cannot
erase poor timing, risk or resource economics, and cheap operation cannot erase
under-delivery. It introduces no fitted coefficient, additive behavior bonus
or second penalty for the same shortfall. For example, a fully delivered order
set with native quality 8 receives 8, not 100; delivery 40 with quality 70 receives
40. The 0.25 cost-precision interval propagates through the minimum.

For a complete model, the exact 0.25-to-0.26 change is:

```
I_025 - I_026 = sum_{completion-applicable i} w_i * max(N_i - F_i, 0)
```

The report checks this identity before publication. It records the total
weighted cap reduction and the weight mass of cases where `F_i < N_i`; the
companion `score_audit_ledger.jsonl` contains each case's `C`, `S`, `N`, `F`,
`Q`, task weight, cap reduction, completion type, contract hash and artifact
hashes. SUMO rows additionally retain route progress, MRM occurrence and
terminal stable ticks. This gives a direct explanation of rank changes while preserving raw
native results. The 131-task completion-only mean has a different denominator
from the full Index and cannot be substituted into the casewise formula.

This is a **task-outcome index**, not a pure 141-task completion rate. The
report also provides a completion-only aggregate over the fixed 131 applicable
tasks, with their frozen weights renormalized within that declared slice, and
the full 0.25 native-quality score on all 141 tasks. The latter remains essential:
two agents can deliver the same work with very different lateness, risk, energy
use, resource depletion or operating cost. A completion-only number cannot
rank those decisions adequately. Conversely, low native cost alone can reflect
under-delivery; the new fulfillment cap makes that visible.

The 0.25 native quality is one **scalar index**, but its settled objective
usually combines several backend-native losses and constraints. It is not a
single physical dimension and is not task completion. 0.26 deliberately does
not add a free-form action-count, proactivity, plan-length or timeliness bonus.
Timeliness matters through delivered outcomes and native delay costs and is
reported explicitly where a source-grounded deadline measure exists.

Strict binary mission attainment remains N/A without independently calibrated
service, safety and cost thresholds. A partial completion fraction of 75 does
not mean the task passed, and a 0.26 Index of 75 does not mean the model passed
75% of 141 missions. The Index reflects the explicitly declared Lite task mix;
one trajectory per task does not establish repeated-run reliability.

## Reproduction and scope

The scorer reads the frozen selected-trajectory 0.25 report, checks its
protocol, re-compiles source contracts and verifies the frozen weight manifest.
For 86 tasks already measured by 0.25, it reuses their authenticated service
result after checking its fixed source denominator and arithmetic. Seven SUMO
tasks instead read authenticated native tick records for stable recovery.
For the newly measured electricity tasks it reads SHA-bound native snapshots,
evidence ledgers and trajectories. FJSP trajectories also supply an independent
complete-job diagnostic. Each 0.26 row retains completion, hard gate,
raw native quality, mode and provenance. Scoring never selects a different
trajectory based on its score. Full-suite ranking requires all 141 cases;
incomplete evidence does not shrink the denominator.

```bash
.venv/bin/python scripts/evaluate_completion026.py \
  --source-report .hl/analysis/latest_tables_20260925/current31_score025/report.json \
  --suite benchmark/lite_suite.json \
  --output-dir .hl/analysis/evaluation026/current31_stable_recovery_v5
```

The scorer is retrospective and does not rewrite a historical provider run or
itself perform public distribution. Lite141 is the declared offline main-table
denominator under the [0.26.1 release policy](EVALUATION_026_MAIN_TABLE.md);
Core Full is an optional extension. Lite remains a model-informed development
panel. For the OPERATE Lite main table, report the 0.26.1 Index together with
completion-only,
0.25 native quality, hard-failure rate, delivery timeliness and coverage. The
remaining measurement question is whether the backend-native objective and
service meters truly capture each mission's obligations; this requires
adversarial under-service, deferral, delayed-recovery and terminal-resource
audits, not another normalization curve or a model-dependent target.
