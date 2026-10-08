# 0.26.1 main-table protocol and release boundary

## Decision

Freeze the 0.26.1 **scoring rule** for the existing Lite141 trajectories. Lite141
is the declared offline **main-table release denominator**; Core Full is an
optional, separately scored extension. The main score is a source-obligation-
capped native operational-quality index, not a binary mission success rate.
The release policy is frozen in
`benchmark/lite_main_scoring_0261.json` and the current
Lite141 score candidate passes its offline scoring gate. Public distribution
has not been performed by this audit.

The panel is a model-informed efficiency/development selection, not an
unbiased held-out test set. Its historical `formal_full_leaderboard_eligible`
flag describes eligibility for the older **Full/Core** protocol. Likewise,
`formal_run_certified=false`, `leaderboard_eligible=false`, and
`retrospective_only=true` remain unchanged on the scored historical report.
They do not describe the newly declared Lite offline main-table policy and must
not be silently relabeled. Rankings are valid only for this fixed Lite panel,
its selected trajectories, and the stated offline scorer.

The offline evaluator version is `0.26.1` / `source_obligation_outcome.v2`.
The original live episode scorer remains `0.21.0`; the offline score does not
rewrite historical runtime or qualification identities. This protocol never
uses another participating model, an outcome-selected reference, or a fresh
replay to set a target. Wait-only, CPU controllers and clairvoyant oracles may
be reported separately as diagnostics, not as the denominator of this Index.

## Main score

For each case `i`, the bound native record supplies complete-horizon and
terminal-settled cost `C_i`, a positive source-derived scale `S_i`, an explicitly
supported signed-cost flag, and a verified native hard-failure indicator. The
0.25 quality component is

```
N_i = 0                    if a verified native hard failure occurred
N_i = 100 / (1 + C_i/S_i)  if C_i >= 0 and no hard failure
N_i = 100                  if C_i < 0, signed cost is declared, and no hard failure
```

`S_i` is a fixed operating exposure, not an optimum, acceptance target or
success budget. A negative cost in a non-signed objective is invalid, not
silently clipped. The raw cost, components, source scale and numerical
precision interval remain available for inspection.

The source-obligation measure `F_i` is compiled without model outcomes. For
131 cases it represents valid delivery, progress, service or conditional
recovery in that backend's native units; the denominator is fixed by the
source and scenario. For 10 CityLearn cases a separate controllable unmet-load
meter does not exist, so completion is structurally inapplicable. The case
score is

```
Q_i = 0             for verified native hard failure
Q_i = min(F_i, N_i) for completion-applicable cases
Q_i = N_i           for CityLearn economic-only cases
```

This is a non-compensatory cap, not a weighted behavior score. A missed order
may lower both `F_i` and the native cost quality; the minimum prevents adding
those effects as two independent penalties. The cap need not bind if native
cost has already reflected poor service. Agent tool count, plan length,
proactivity labels and process verbosity do not add points. Timely action,
resource stewardship and long-horizon decisions matter insofar as they change
the recorded native outcome or delivered obligation.

The Index uses the frozen case-weight manifest:

```
w_i = 1 / (domains × families_in_domain × sources_in_family × cases_in_source)
Index(model) = Σ_i w_i Q_i, over all 141 Lite cases
```

Every weight is positive and the sum is one. A full-model Index and rank need
141 valid bound case scores. A model failure with verified native consequences
is scored; absent or inconsistent evidence is N/A and never removed from the
fixed denominator. The separate 131-case `F` mean renormalizes weights within
that declared support and must not be interpreted as the 141-case Index.

The table must show the Index, 0.25 native quality, the applicable-support
`F` measure, weighted hard-failure rate and valid coverage. Strict binary
mission attainment is uncalibrated and remains N/A. The Index cannot be read
as a percentage of missions passed or a repeated-run reliability estimate.

## Task-semantic acceptance

The 141 completion contracts declare their metric type and denominator. They
do not assert that all types are the same final-delivery percentage. The
following narrower claims are part of the main-table contract:

| Cases | What the score can claim | What it cannot claim |
|---|---|---|
| Alibaba GPU (33) | Completed-job fraction, native queue wait and loss while a job is **queued past a procedural `due_tick`**, unfinished-work settlement | A source-native or independently verified job **completion** deadline SLA; running jobs past `due_tick` do not accrue that queue-SLA term |
| CVRP/VRPTW routing (38) | Valid delivery of the original and seeded urgent demand; procedural dispatch-wave lateness is a diagnostic and native cost | Full source VRPTW travel-time, service-duration or original time-window execution; the backend explicitly reports these as unavailable |
| SUMO ego (7) | Native MRM trigger and, when triggered, up to six consecutive terminal **supervisory tick-end** `nominal` states; native risk and delay in `N_i` | Full-route arrival, six wholly nominal 0.1-second physics ticks, causal credit to the agent for every recovery, or proof of every prevention/process requirement |
| FJSP (3) | Source-operation completion progress; complete-job fraction separately | Operation percentage as final job-delivery percentage |
| LV/CIGRE (9) | Bottleneck of delivered-load fraction and voltage-compliant node-time fraction | Joint load served at compliant voltage, which requires matched load-by-node-time evidence |
| OpenDSS (4) | Voltage-compliant node time | Independently metered energy delivered |
| CityLearn (10) | Settled economic quality | An invented agent-controllable service completion fraction |

The remaining inventory, Pymgrid and PGLib fractions use fixed demand and
metered fulfillment. Future arrivals and seeded shocks remain in their source
population; refusal or cancellation cannot shrink that population. Late
arrivals make a source-work progress fraction different from a claim that all
unfinished work was feasibly completable before the horizon.

SUMO's source scenarios prescribe six post-recovery stable supervisory ticks
conditional on MRM, but the strict generic model briefing does not enumerate
that numerical threshold. Therefore `F=100` is not a disclosed-budget task
pass. A no-MRM episode receives full credit for the **inapplicable recovery
obligation** even if its terminal mode is degraded; risk and delay still enter
`N_i`. The published count samples tick-end state. An all-physics-substep
stability interpretation changes some scores and is a separately labelled
sensitivity, not an undocumented replacement of 0.26.1.

## Reviewable artifact and gates

Reproduce the hash-bound Lite main-table candidate without any model or backend
run:

```bash
.venv/bin/python scripts/audit_offline_main_table026.py \
  --report .hl/analysis/evaluation026/current31_stable_recovery_v5/report.json \
  --suite benchmark/lite_suite.json \
  --policy benchmark/lite_main_scoring_0261.json \
  --output-dir .hl/analysis/evaluation026/formal_readiness_20260927/reproduction_new \
  --require-lite-main
```

The verifier checks the 0.25 report hash, scored report and ledger hashes,
scoring-code hashes, scenario hashes, recomputed completion contracts, frozen
weights, selected trajectory identities, native `C/S` scores, casewise `min`,
full-panel weighted sums, hard gates, score intervals and ties/ranks. It does
not rehash the approximately 15 GB of episode attachments already authenticated
upstream by the 0.25 scorer. The output is `audit.json`, `table.json` and
`table.md`, with no private attachment paths in the small table. Use a fresh
output directory for each reproduction; the verified current candidate is
`.hl/analysis/evaluation026/formal_readiness_20260927/lite_main_v6/`.

`--core` optionally records the separate Full scope in the audit. It does not
change the Lite gate or denominator. `--require-formal-core` cannot certify a
Core leaderboard from a Lite report; Core Full requires its own complete
trajectories, 17-backend scoring support and release evidence. None of those
are prerequisites for publishing the declared Lite141 main table. Publication
should bundle this policy, scorer, contracts, weights, selected-trajectory
manifest and authenticated evidence alongside the score table. The table must
retain the model-informed-panel, single-run and task-semantic limitations above.
