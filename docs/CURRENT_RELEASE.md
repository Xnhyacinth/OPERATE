# Current release

The promoted input namespace is `operate`. Qualification artifacts stay
bound to scoring `0.15.0`. Live episode artifacts use scoring `0.21.0`; the
default Lite141 offline main table uses `0.28.0` through
`scripts/evaluate_trajectories.py`; the frozen released `0.26.1` policy retains
its historical meaning. Lite has 141 rows across 82 physical
sources. The optional Full/Core extension has 769 rows across 502 sources.
Native qualification is bound to its recorded implementation identity. Later
verified maintenance changes bind their actual new execution identity; resume
and merge remain strict. The Lite141 offline main-table scoring gate has passed;
public code/data distribution and retrospective result companions are maintained
separately from formal Full certification. Optional Full logical/realtime
provider runs remain pending.


The active release namespace is `operate`. Lite141 is the offline
main-table denominator. The current reader policy is
`evaluation/policies/lite141_028.json`; the frozen release policy remains at
`benchmark/lite_main_scoring_0261.json`. This document also
describes the promoted Core as an optional Full extension; historical releases,
tags, and provider trajectories are not valid inputs to a current run.

## Promoted Core

- 769 source-grounded scenario contracts across 502 physical sources
- 141 Lite rows across 82 physical sources, selected by
  `reviewed_horizon_balanced_v11`
- seven domains: Autonomous Driving, Building Energy, Datacenter, Logistics,
  Microgrid, Power Grid, and Traffic
- tracked source locks and compact Alibaba/DynaSched assets under `sources/`
- 769 current scenario contracts under `scenarios/<domain>/...`
- Lite main suite at `benchmark/lite_suite.json` with frozen
  0.26.1 scoring policy at `benchmark/lite_main_scoring_0261.json`
- optional Full/Core suite at `benchmark/core_suite.json`
- matching Full manifest at `benchmark/manifest.json`
- replay and provenance suite at `benchmark/core_suite.json`
- promoted qualification scoring version `0.15.0`
- live evaluation scoring version `0.21.0` (`wait_relative_outcome_v1`)
- default offline scoring version `0.28.0`; direct raw artifact reader and
  separately verified historical measurement recovery, described in
  [the current scoring protocol](EVALUATION_028.md)
- live realtime contract `realtime_persistent.v3` / `native_dt_v1`; default
  supervision scorecard is the 37-row speed-critical subset in
  `benchmark/realtime_speed_suite.json`

The domain distribution is 7 Autonomous Driving, 18 Building Energy,
142 Datacenter, 527 Logistics, 37 Microgrid, 19 Power Grid, and 19 Traffic
rows. The primary hierarchical aggregation prevents row count alone from
determining domain weight.

The parent v0.61 admission ledger records 2,476 terminal candidate decisions
with zero unresolved. v0.62 introduces zero newly mined candidates; it preserves
that historical lineage while qualifying corrected contracts for the same
769-row, 502-source denominator. Historical admission evidence is not relabelled
as newly executed evidence.

`lite_suite.json` defines the offline main-table denominator;
`lite_main_scoring_0261.json` retains the frozen released scorer identity, while
the current reader uses `evaluation/policies/lite141_028.json`.
`core_suite.json` together with its matching
`manifest.json` defines the optional Full denominator.
`protocol21_source_suite.json` remains the bound replay input and provenance
ledger.

## Lite review scope

Lite v11 freezes the previous 144-row model-informed selection. A routine
192-tick budget retains the shortest longer candidate per backend/family/
difficulty-mode stratum: Dyna static moo (298 ticks) and breakdown recovery
(400 ticks), alongside Sweep (111 ticks). Three longer alternatives remain
in Core/long-horizon diagnostics. The interim v10 uniform cap is withdrawn.
The resulting 141 rows retain 204/204 parent features and six domains;
this does not establish complete Core or scientific coverage. Traffic remains
outside Lite. Native ticks do not bound provider latency or call count, and
192 ticks is a budget cutoff, not a definition of long-horizon agency.

Lite's long-tick bands are starved: 121/141 rows are at most 32 ticks, the
97–192 band holds 1 row and the 193+ band 2 rows, and only 2 rows exceed 128
ticks. Lite alone cannot support a broad long-horizon capability,
memory-retention or significance claim. Such claims need additional evidence,
for example the optional Core extension, where 54/769 rows run above 192 ticks,
and a matched long-task diagnostic slice. Report the missing coverage rather
than filling it with zeros.

Historical zero/ceiling scores affected by measurement defects are not new
exclusion evidence. Decisions are in
`benchmark/lite_review_dispositions.json`.

## Release status

- Current 0.28 raw-trajectory comparison (2026-10-06): 23 declared models,
  2,958 determined Q values out of 3,243 fixed targets and 20 complete models.
  It uses an explicit user-authorized descriptive cross-framework policy;
  3 historical native-meter gaps and 282 unavailable outcomes remain unranked.
  DSS quality now separates source node-time count exposure from global extrema
  severity; original C/F and all non-DSS N/Q remain unchanged.
  See [current results, evidence and reproduction](LATEST_RESULTS_028.md).
- Lite141 0.26.1 offline main-table scoring gate: passed for the existing
  authenticated trajectory panel; 20 models have 141/141 valid scores and 11
  incomplete models are unranked. See
  [the main-table audit](EVALUATION_026_MAIN_TABLE.md).
- Lite selection used development-model outcomes. The main table is a declared
  fixed-panel retrospective comparison, not an unbiased held-out test.
- Latest opt-in outcome revision: 0.30.0, with the same 23 declared models,
  20 complete scores and 2,958 determined outcomes. It corrects LV/CIGRE
  source-population exposure and driving recovery semantics; the default reader
  remains 0.28.0. See [latest results and historical comparison](LATEST_RESULTS_030.md).
- Public code/data distribution is available. The dated retrospective aggregate
  companion is separate from the pending Full formal workflow. Complete raw
  trajectories, recovery evidence, full reports and private audit archives remain
  in the private HF trajectory dataset. The [12-model offline analysis](LITE141_ANALYSIS.md)
  is a companion subset, not a change to the main-table denominator.

The following historical flags describe the optional Full/prospective formal
workflow, not the separately declared Lite offline main-table gate:

- the dataset's admission replay and atomic promotion are complete;
- `formal_evaluation_ready=true`;
- `formal_logical_persistent_evaluation_pending`;
- `formal_realtime_persistent_evaluation_pending`;
- `formal_runtime_evidence_distribution_pending`;
- `public_release_ready=false` and `leaderboard_eligible=false` for that Full
  workflow.

Agency positive controls and baseline smoke remain available as independent
diagnostics. They are not release-admission gates, formal provider inputs, or
members of the runtime evidence bundle.

Qualification records retain the code identity under which they were produced.
Later maintenance fixes use [affected-scope validation](VALIDATION_POLICY.md),
not automatic whole-suite requalification. New evaluations record their actual
current code; historical proof is not relabelled as a new execution. Integrity
diagnostics expose any difference between these two identities.
Portable frozen-input integrity checks pass. The current software has different
runtime/tooling hashes from historical formal qualification; those old flags
do not certify a new formal run on today's implementation.

Formal resume/merge excludes interrupted or completed results with different
release, implementation, prompt/context or provider bindings. The separately
declared offline descriptive comparison retains authenticated historical whole
episodes and their original identities under the policy in
[EVALUATION_028.md](EVALUATION_028.md).

## Formal treatments

`logical_persistent` is the primary leaderboard treatment. It begins with one
mission briefing and continues through typed wakeups, scheduled reviews, tool
feedback, and lifecycle receipts. Quiet environment ticks do not create model
requests.
Its `agent_scheduled_v1` cadence makes the agent responsible for choosing or
omitting its next review; the harness does not inject periodic scans.

`realtime_persistent` is a separate formal supervision scorecard. The
environment continues while provider calls and actions are in flight, and the
coordinator records latency, steering, cancellation, supersession, expiry,
safety arbitration, and takeover. The default controlled hold is not counted as
a domain-native takeover; that claim requires an explicitly bound domain shield.
Wall ticks follow each row's native plant quantum (`tick_seconds` or
`tick_minutes`), bound before the episode. The default speed card is the 37
Core rows whose native tick is at most 60s and whose plant wall is at most
30 minutes. Hour-scale and multi-hour plants stay on `logical_persistent`
outcomes; they are not 1:1 operator-latency tests. A uniform 5s overlay is a
labelled stress treatment, not the live formal clock.

`logical_stateless` is a compatibility ablation and is not a release gate.

## Context contract

Authoritative events and trajectories are append-only. A deterministic bounded
projection is sent to the model together with structured memory for unresolved
alarms, obligations, facts, commitments, forecasts, and numeric trends. The
default formal profile uses:

- temperature `0`
- per-route native context envelope and request output; `hy3-ioa` is 192,000 /
  64,000 and `gpt-5.6-luna` is 272,000 / 128,000
- protocol-repair budget `8192` tokens
- provider timeout `300` seconds
- projected history `64` messages
- projected context `512000` characters
- structured memory `128` items across semantic buckets
- strict prompt mode, streaming, `tool_choice=auto`, action-required protocol
  validation, and global scheduling

Advertised provider context/output caps are the request budgets for that
route. Do not copy one model's envelope onto another.
No hidden summarizer rewrites the authoritative history.

## Distribution

`Xnhyacinth/OPERATE` carries the current public code, catalogs, scenarios,
tests, and concise release documentation. Historical maintenance sources and
audit material are not part of this snapshot.
Public commits and the changelog record updates; superseded scenario files and
old tagged distributions are not retained in the public current tree. The
immutable original Lite141 suite and scenario bytes required by the offline
scoring policy are retained alongside the versionless runtime catalog; this
source-bound companion preserves policy hashes and is not an old run bundle.
The public setup installs the versionless runtime companion under `operate_data/`;
the maintenance checkout may retain its `operate_data/` compatibility root.
The manifest and recorded immutable HF revision, not that directory name, bind
the installed bytes to the current code and Core.

The v0.62 distribution receipt records the immutable runtime-companion
revision and verified Full/Lite artifact hashes.

See [FORMAL_EVALUATION.md](FORMAL_EVALUATION.md) for provider commands and
[AGENTIC_INTERACTION.md](AGENTIC_INTERACTION.md) for event-loop semantics.

Launch evaluation only from this `main` tree. `.hl/` worktrees, detached
snapshots, and historical branches are not current inputs.
