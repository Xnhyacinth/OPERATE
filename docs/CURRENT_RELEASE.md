# Current release

The public distribution exposes one current benchmark state. Its internal
reproducibility namespace is `operate_v0_61_0`. This document describes the
current promoted Core; historical releases, tags, and provider trajectories are
not valid inputs to a current formal run.

## Promoted Core

- 769 source-grounded scenario contracts across 502 physical sources
- seven domains: Autonomous Driving, Building Energy, Datacenter, Logistics,
  Microgrid, Power Grid, and Traffic
- tracked source locks and compact Alibaba/DynaSched assets under `sources/`
- 747 inherited scenario contracts under `scenarios/operate_v0_58_0/`, 8
  additions under `scenarios/operate_v0_59_0/`, and 13
  additions under `scenarios/operate_v0_60_0/`, plus 1 under
  `scenarios/operate_v0_61_0/`, exactly as selected by the manifest
- Core suite at `release/operate_v0_61_0/core_suite.json`
- matching formal manifest at `release/operate_v0_61_0/manifest.json`
- replay and provenance suite at `release/operate_v0_61_0/protocol21_source_suite.json`
- current-tree scoring version `0.14.0`

The domain distribution is 7 Autonomous Driving, 18 Building Energy,
142 Datacenter, 527 Logistics, 37 Microgrid, 19 Power Grid, and 19 Traffic
rows. The primary hierarchical aggregation prevents row count alone from
determining domain weight.

The exhaustive candidate ledger is closed: 2,476 independent candidates have
terminal dispositions and zero remain unresolved. The final replay selected all
769 source rows, with no secondary, rejected, held-repair, or retired rows in the
release partition.

`core_suite.json` together with its matching `manifest.json` defines the formal
denominator. `protocol21_source_suite.json` remains the bound replay input and
provenance ledger.

## Release status

- twelve-stage replay and atomic promotion are complete;
- `formal_evaluation_ready=true`;
- `formal_logical_persistent_evaluation_pending`;
- `formal_realtime_persistent_evaluation_pending`;
- `formal_runtime_evidence_distribution_pending`;
- `public_release_ready=false` and `leaderboard_eligible=false`.

Agency positive controls and baseline smoke remain available as independent
diagnostics. They are not release-admission gates, formal provider inputs, or
members of the runtime evidence bundle.

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
- main output budget `32768` tokens
- protocol-repair budget `8192` tokens
- provider timeout `300` seconds
- projected history `64` messages
- projected context `512000` characters
- structured memory `128` items per semantic bucket
- strict prompt mode, streaming, `tool_choice=auto`, action-required protocol
  validation, and global scheduling

Advertised provider context/output caps remain model-specific treatment fields.
No hidden summarizer rewrites the authoritative history.

## Distribution

The public GitHub repository and history-squashed `Xnhyacinth/OPERATE` Hugging
Face dataset expose the current code/data surface. The downloader resolves the
current snapshot once and records its immutable HF commit in the local owner
receipt; no earlier revision is a valid substitute. The setup script restores
the runtime companion through the stable `operate_data/` install root; that
directory name is not a release identity.
Its `MANIFEST.json` and the current release manifest
must bind the installed bytes, implementation tree, and Git commit before a
formal shard starts.

`OPERATE-Lite` is a separate policy-derived 159-row efficiency/development track. It preserves
all released backends and task families, but its diversity-weighted scores are
not interchangeable with the 769-row Full leaderboard.

See [FORMAL_EVALUATION.md](FORMAL_EVALUATION.md) for provider commands and
[AGENTIC_INTERACTION.md](AGENTIC_INTERACTION.md) for event-loop semantics.
