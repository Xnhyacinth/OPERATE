# Frozen Lite141 trajectory scoring (0.27)

The current default is [0.28](EVALUATION_028.md). This document retains the
0.27 policy and measurement rules; select `evaluation/policies/lite141_027.json`
explicitly or use an original manifest bound to that policy.

`scripts/evaluate_trajectories.py` is the default offline main-table reader.
It reads original episode journals, original run configurations, provider audits,
native scoring snapshots, trajectories and evidence ledgers. Previous score
reports and their cached scores are not inputs. Ordinary scoring makes zero
provider calls, native episode replays or source constructor calls.

The source panel remains `benchmark/lite_suite.json`: 141 cases
from 82 physical sources. The current scorer policy is
`evaluation/policies/lite141_027.json`. The released `0.26.1` policy and its
qualification evidence remain immutable historical artifacts.

## Main score

For settled native cost C and independently compiled source scale S:

- Nonnegative C: `N = 100 / (1 + C/S)`.
- Source-authorized signed negative C: N=100, retaining the original signed C.
- Verified native hard failure: N=Q=0, without inventing missing F.
- Otherwise, where source fulfillment applies: `Q = min(F, N)`.
- The ten CityLearn automatic-load-service cases use Q=N; their independent F
  is structurally inapplicable.

C, S, F, N and Q remain separate case columns. Q is continuous operational
utility on this fixed panel, not binary mission success, an optimality gap or a
claim of comprehensive safety. Native constraints and failure coverage retain
their domain meanings. Behavior, opportunity and realtime diagnostics have no
headline weight.

Fixed domain/family/physical-source/case weights require all 141 eligible Q
values before a model receives a point Index and rank. The fulfillment-only
column separately requires all 131 applicable F values. Missing evidence keeps
its weight and produces identification bounds; it never becomes zero or changes
the denominator. These bounds are not statistical confidence intervals.

## Score raw runs

```bash
uv run --frozen --no-sync python scripts/evaluate_trajectories.py \
  --run-dir '<completed-run-directory>' \
  --output-dir '<new-output-directory>'
```

Repeat `--run-dir` for multiple original directories. Each must contain
`episodes.jsonl` and `run_config.json`. A subset run can contribute its available
cases; the report still contains 141 targets per declared model. Output
directories must be new and outside frozen release/source/scenario inputs.

The command writes `input_manifest.json`, `report.json`, `report.md`,
`case_rows.jsonl` and `table.csv`. It selects the latest status-ok whole episode
by recorded invocation time, journal line and path before reading its score.
It never combines ticks from different attempts or selects the highest score.
Original failures and source references remain in the report.
`n_observed` counts selected original records, including evidence-unqualified
candidates; it excludes padded missing targets. `n_scored` counts qualified Q
values. The fixed denominator remains 141 for every declared model.

Reproduce and audit a generated report with its frozen manifest:

```bash
uv run --frozen --no-sync python scripts/evaluate_trajectories.py \
  --manifest '<score-directory>/input_manifest.json' \
  --audit-report '<score-directory>/report.json' \
  --output-dir '<new-audit-directory>'
```

The auditor rebuilds every case and aggregate and requires the same scoring
runtime bytes. It writes a separate `audit_receipt.json` without changing the
original report or its execution identities.

For archived data, use a frozen input manifest:

```bash
uv run --frozen --no-sync python scripts/evaluate_trajectories.py \
  --manifest '<raw-input-manifest.json>' --output-dir '<new-output-directory>'
```

The manifest schema is `operate_trajectory_outcome027_inputs.v1`. It binds the
policy and every original journal/configuration by SHA-256 and byte count,
declares model labels, pass index, treatment and selection policy, and may bind
measurement recovery sidecars. `artifact_relocations` maps original artifact
SHA-256 values to restored local paths; identical bytes remain mandatory.
For a content-addressed backup, `--artifact-root '<restored-blob-directory>'`
resolves SHA-256-named files without rewriting the frozen input manifest.
Audits compare authenticated references independently of their storage paths;
all score, identity, count and hash fields still have to reproduce exactly.
`source_case_scope=project_frozen_Lite141` can project a larger original journal
onto the fixed panel and records every excluded outside-panel row.

`scripts/analyze_batch_results.py` exposes the same current reader under
`primary_evaluation` for logical batches wholly within Lite141. Its retained
`mean_by_model` is explicitly a legacy live-score diagnostic, not a current
ranking. Live runner scores retain their recorded `0.21.0` identity.

## Required evidence

The reader validates original profile/treatment hashes and public agent
configuration, raw request/response payload hashes, declared model aliases,
provider routes, generation settings and actual input-plus-output budgets.
Fallback-to-wait and tool-less fallback contamination are rejected. A bounded,
typed transient retry can be accepted only when its wire request is unchanged
and the chain closes successfully; intermediate failures remain evidence.
Invalid model decisions on successful authenticated calls remain measured model
behavior.

Source-only compilation independently rebuilds S, case strata and weights from
the locked suite and scenarios. Audited native population/constructor hashes
are frozen with the reader; rehashing a policy cannot remove an obligation.
It requires the locked source files declared in each contract, including the
CityLearn schema/CSV files under `works/`. The private source-assets archive
restores these exact repository-relative bytes for a fresh checkout.
Historical replay proofs additionally require the bound recovery-source-alias
archive: it restores exact canonical OpenDSS files and their historical
`operate_data/` aliases. Every alias is authenticated against its locked
canonical bytes; proof checks and original evidence remain unchanged.
The native ledger, scoring snapshot and complete source time window must agree.
A native terminal marker closes a shorter window only for an independently
verified source terminal condition.

Pandapower LV and CIGRE fulfillment is the minimum of source bus-time voltage
compliance and delivered-energy fulfillment. All required source buses and
loads remain obligations. Model-visible requested load and rounded voltages
alone are insufficient. OpenDSS measures source node-time voltage fulfillment;
its construct does not claim delivered-energy measurement. Missing nodes,
nonfinite readings and unproven disconnection remain unknown.

## Historical measurement recovery

A manifest `recoveries` entry binds a sidecar to an exact model/case and
`canonical_episode_sha256`, plus its own `sidecar` path/SHA-256/byte count.
An applicable explicit `recovery_equivalence` policy is part of that input.
`evaluation/native_measurement_recovery.py` independently verifies the original
artifact hashes, identity, source population, full window, proof and derived
evidence before the reader uses its F. The original C and all original artifacts
remain unchanged.

Explicit provider-free native replay can produce a sidecar through
`recover_voltage_measurement(..., allow_replay=True)`. It reproduces original
queries, tool receipts and controls. Exact comparison covers native records,
observations, effects, receipts and every cost component. The bounded LV
consumer-quantization policy additionally permits only two raw numerical fields
when all actual cost/rounding consumers and other fields remain unchanged.
This is a separately disclosed recovery phase; the default reader never starts it.

For the locked CIGRE SimBench constant-PQ cases, a separate source-law
certificate can establish delivery from original convergence, complete finite
source-bus coverage, native violation counts and the closed control/effect
inventory. It verifies pinned package/source bytes, load defaults and consumed
profiles; it rejects effective shedding, unsupported load/topology changes and
unknown effects. This is symbolic native delivery proof, with
`raw_per_load_delivery_measured=false`; it never fabricates private meter rows
or substitutes rounded public voltages for native violation counts.

A candidate whose proof fails equivalence cannot contribute a score. A complete
trajectory may still lack a historical measurement that cannot be reconstructed
from its recorded primitives. Such cases report the specific missing evidence
and preserve C/N where they are independently established. Future producers
record full private native meters so ordinary scoring does not need recovery.

## Execution identity and interpretation

The default comparison checks stable original episode and configuration trees.
An explicitly declared `comparison_policy=latest_framework_user_assumed`
supports user-authorized descriptive comparison of later-framework trajectories.
It retains original tree/start/end/configuration differences and missing original
fields under `execution_binding`; it does not replace them with today's identity.
Original profile, provider, source and raw evidence checks remain mandatory.

The explicit selection
`latest_status_ok_or_terminal_tree_drift_whole_episode_before_scoring` is
available only with that user-assumed policy. It additionally considers complete
native snapshots rejected solely by `ImplementationIdentityError:
implementation_tree_drift`. Before scoring, the completed-runtime identity,
end tree, actions, native records, effects, costs, provider binding and source
fields must authenticate against the original scoring snapshot and trace.
Source-window closure remains mandatory. Original error/status/repair flags and
tree differences stay recorded, and strict execution eligibility stays false.
Other harness, provider and incomplete-run errors remain excluded. The older
status-ok-only manifest and its results retain their original selection meaning.

`native_measurement_qualified` and `strict_execution_identity_qualified` are
distinct. Both descriptive modes report `formal_run_certified=false` and
`same_run_141_merge_certified=false`. A point rank describes the recorded
deployment outcomes under the stated comparison conditions; it does not prove
that all models were retested on one implementation or isolate framework effects.

Lite is a model-informed development panel. These results do not establish
held-out generalization, binary attainment, long-horizon memory, causal agency
or scientific optimality. Realtime supervision remains an independent report.

## Realtime evidence reader

`scripts/evaluate_realtime_trajectories.py --manifest '<bound-realtime-input.json>'
--output '<new-report.json>'` reauthenticates original journals, run settings,
ledgers and wire payloads, and reconstructs supervision diagnostics. Historical
path-projection recovery requires a preimage matching the original embedded
digest and exact forward projection, plus a separately bound derivation receipt.
Responses, original validation flags and execution identities remain unchanged.
Core37 runs retain their individual treatment/concurrency identities; cross-run
case availability is descriptive. E3 reports the fixed 11-case delay triplets.
Neither output supplies the logical-primary Index.
