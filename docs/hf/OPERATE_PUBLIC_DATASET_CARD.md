---
license: other
pretty_name: OPERATE
tags:
  - benchmark
  - llm-agents
  - agentic-evaluation
  - scheduling
  - simulation
  - long-horizon
task_categories:
  - other
configs:
  - config_name: full
    default: true
    data_files:
      - split: test
        path: full/test-00000-of-00001.parquet
  - config_name: lite
    data_files:
      - split: test
        path: lite/test-00000-of-00001.parquet
---

# OPERATE

**Benchmarking Persistent Operational Agency in Source-Grounded Executable
Systems**

This is the current public dataset for OPERATE. Filenames are versionless;
immutable Hugging Face commits identify each snapshot. Matching public code is supplied in the OPERATE repository.
Dataset and archive locations are supplied by the distributor; configure
`OPERATE_HF_REPO_ID` and `OPERATE_TRAJECTORIES_REPO_ID` locally.

The current catalog contains 769 Full scenarios across 502 physical sources
and 141 Lite scenarios across 82 sources. Qualification scoring is `0.15.0`.
Live evaluations use scoring `0.21.0` (`wait_relative_outcome_v1`). Live
realtime uses `native_dt_v1`; the default speed card is 37 Core rows whose
native tick is at most 60s and whose plant wall is at most 30 minutes.

`logical_persistent` is the primary treatment. `realtime_persistent` is a
separate supervision scorecard and is not pooled into the primary score.
Lite is an efficiency and development track, not a Full leaderboard
denominator. Lite v11 freezes the prior model-informed 144-row selection.
It retains shorter cases plus the shortest long candidate per backend/family/
difficulty-mode stratum: Dyna moo (298 ticks), breakdown (400), and Sweep (111)
remain; three longer alternatives stay in Core/diagnostics. The routine
192-tick cutoff is a budget, not a definition of long-horizon agency. The
141-row set retains 204/204 declared parent features across six domains;
that does not establish complete Core or scientific coverage. Legacy all-zero/
ceiling results await repaired-measurement retests, not automatic exclusion.

The GitHub tree flattens scenario contracts to `scenarios/<domain>/...`.
Parquet tables are reversible to that public catalog. Upstream simulators and
source datasets retain their own licenses; the dataset-card `other` value does
not relicense those assets under MIT.

Formal provider result publication remains pending.
`leaderboard_eligible` is false.

## Lite141 retrospective result companion

The default raw reader remains `0.28.0`. The current opt-in reference-anchored
scorer is `0.31.2`, also the default policy of its separate anchored entrypoint.
The 2026-10-11 Sonnet-complete snapshot declares 23 deployments: 20 have all
141 measurements, with 2,954 determined outcomes out of 3,243 fixed targets.
Sonnet scores 32.08 (point rank 9/20). Missing outcomes stay unknown and unranked.
Source-cluster intervals describe task sampling, not repeated-run variation;
complete coverage does not imply statistically distinct adjacent ranks.

The 2026-10-11 aggregate companion (`<aggregate-results-directory>`)
is published and verified at its immutable revision. It reports the fixed-panel retrospective
comparison after raw reauthentication. Cross-execution substitutions do not
certify a same-run formal cohort or held-out model-capability ranking.

The 2026-10-08 companion (`<aggregate-results-directory>` at revision `main`)
remains a historical snapshot: it contains the 0.25–0.30 comparison, domain
summaries, deletion sensitivity and the separate 12-model analysis figures
with Source Data. Its 2,958 measured Q values belong to those historical
protocols and inputs, not the current 0.31.2 snapshot.

These fixed-panel retrospective results are separate from pending Full formal
provider evaluation. Frozen catalog and runtime payloads are unchanged. Sanitized
trajectory/provider/native evidence and audit reports are available in the
public sanitized trajectory archive (`<trajectory-archive-directory>`).

Scientific records, actions, scorer outputs, numeric values and identifiers are
preserved. Infrastructure URLs and paths and credential values are redacted;
OS metadata exclusions are recorded in the publication manifest. Sanitized
files have new hashes and do not replace the original evidence bytes.
Original evidence is restricted to approved access. Manuscript drafts remain private.
