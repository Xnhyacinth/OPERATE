# Current release

The promoted input namespace is `operate_v0_62_0`, with scoring `0.15.0`,
769 Core rows across 502 physical sources and 193 Lite rows across 122 sources.
Native qualification is bound to its recorded implementation identity. Later
verified maintenance changes bind their actual new execution identity; resume
and merge remain strict. Formal logical/realtime provider runs and leaderboard
result publication remain pending.


The public distribution exposes one current benchmark state. Its internal
reproducibility namespace is `operate_v0_62_0`. This document describes the
current promoted Core; historical releases, tags, and provider trajectories are
not valid inputs to a current formal run.

## Promoted Core

- 769 source-grounded scenario contracts across 502 physical sources
- seven domains: Autonomous Driving, Building Energy, Datacenter, Logistics,
  Microgrid, Power Grid, and Traffic
- tracked source locks and compact Alibaba/DynaSched assets under `sources/`
- 701 corrected contracts under `scenarios/operate_v0_62_0/`, 48 inherited
  under `scenarios/operate_v0_58_0/`, 8 under `scenarios/operate_v0_59_0/`,
  11 under `scenarios/operate_v0_60_0/`, and 1 under
  `scenarios/operate_v0_61_0/`, exactly as selected by the v0.62 manifest
- Core suite at `release/operate_v0_62_0/core_suite.json`
- matching formal manifest at `release/operate_v0_62_0/manifest.json`
- replay and provenance suite at `release/operate_v0_62_0/protocol21_source_suite.json`
- current-tree scoring version `0.15.0`

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
- v0.62 native qualification: `formal_evaluation_ready=true`;
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

The public GitHub repository and `Xnhyacinth/OPERATE` Hugging Face dataset
expose the current code/data surface. Commits and the changelog record updates;
superseded scenario files and archives are absent from the current tree.
The downloader resolves the
current snapshot once and records its immutable HF commit in the local owner
receipt; no earlier revision is a valid substitute. The setup script restores
the runtime companion through the stable `operate_data/` install root; that
directory name is not a release identity.
Its `MANIFEST.json` and the dataset manifest bind the installed bytes and
qualification snapshot. Each formal shard separately records its actual current
implementation and Git commit; qualification code need not be identical.

`OPERATE-Lite` is a separate 193-row, 122-source efficiency/development track. It preserves
all released backends and task families, but its diversity-weighted scores are
not interchangeable with the 769-row Full leaderboard.

See [FORMAL_EVALUATION.md](FORMAL_EVALUATION.md) for provider commands and
[AGENTIC_INTERACTION.md](AGENTIC_INTERACTION.md) for event-loop semantics.
