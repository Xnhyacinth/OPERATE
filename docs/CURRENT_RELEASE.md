# Current release

The promoted input namespace is `operate`. Qualification artifacts stay
bound to scoring `0.15.0`. Live evaluations use scoring `0.17.0`. Core remains
769 rows across 502 physical sources, with 193 Lite rows across 122 sources.
Native qualification is bound to its recorded implementation identity. Later
verified maintenance changes bind their actual new execution identity; resume
and merge remain strict. Formal logical/realtime provider runs and leaderboard
result publication remain pending.


The active release namespace is `operate`. This document describes the
current promoted Core; historical releases, tags, and provider trajectories are
not valid inputs to a current formal run.

## Promoted Core

- 769 source-grounded scenario contracts across 502 physical sources
- seven domains: Autonomous Driving, Building Energy, Datacenter, Logistics,
  Microgrid, Power Grid, and Traffic
- tracked source locks and compact Alibaba/DynaSched assets under `sources/`
- 769 current scenario contracts under `scenarios/<domain>/...`
- Core suite at `benchmark/core_suite.json`
- matching formal manifest at `benchmark/manifest.json`
- replay and provenance suite at `benchmark/core_suite.json`
- promoted qualification scoring version `0.15.0`
- live evaluation scoring version `0.17.0` (`wait_relative_outcome_v1`)
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

`core_suite.json` together with its matching `manifest.json` defines the formal
denominator. `protocol21_source_suite.json` remains the bound replay input and
provenance ledger.

## Release status

- the dataset's admission replay and atomic promotion are complete;
- `formal_evaluation_ready=true`;
- `formal_logical_persistent_evaluation_pending`;
- `formal_realtime_persistent_evaluation_pending`;
- `formal_runtime_evidence_distribution_pending`;
- `public_release_ready=false` and `leaderboard_eligible=false`.

Agency positive controls and baseline smoke remain available as independent
diagnostics. They are not release-admission gates, formal provider inputs, or
members of the runtime evidence bundle.

Qualification records retain the code identity under which they were produced.
Later maintenance fixes use [affected-scope validation](VALIDATION_POLICY.md),
not automatic whole-suite requalification. New evaluations record their actual
current code; historical proof is not relabelled as a new execution. Integrity
diagnostics expose any difference between these two identities.

Interrupted or completed results from an earlier package name, release ID,
implementation tree, prompt/context profile, or provider binding are excluded.

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
- structured memory `128` items per semantic bucket
- strict prompt mode, streaming, `tool_choice=auto`, action-required protocol
  validation, and global scheduling

Advertised provider context/output caps are the request budgets for that
route. Do not copy one model's envelope onto another.
No hidden summarizer rewrites the authoritative history.

## Distribution

`Xnhyacinth/OPERATE` is the private maintenance repository and retains
historical sources and audit material. The public `Xnhyacinth/OPERATE` GitHub
repository and Hugging Face dataset carry only the current complete snapshot.
Public commits and the changelog record updates; superseded scenario files and
old tagged distributions are not retained in the public current tree.
The public setup installs the versionless runtime companion under `operate_data/`;
the maintenance checkout may retain its `operate_data/` compatibility root.
The manifest and recorded immutable HF revision, not that directory name, bind
the installed bytes to the current code and Core.

The previously verified v0.61 private HF snapshot remains historical.
The v0.62 distribution receipt must record its own newly uploaded immutable
HF revision and verified Full/Lite artifact hashes.

See [FORMAL_EVALUATION.md](FORMAL_EVALUATION.md) for provider commands and
[AGENTIC_INTERACTION.md](AGENTIC_INTERACTION.md) for event-loop semantics.
