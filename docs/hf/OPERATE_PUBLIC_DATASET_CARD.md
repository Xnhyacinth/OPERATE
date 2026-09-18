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
immutable Hugging Face commits identify each snapshot. The matching public
code lives at [Xnhyacinth/OPERATE](https://github.com/Xnhyacinth/OPERATE).

The current catalog contains 769 Full scenarios across 502 physical sources
and 141 Lite scenarios across 82 sources. Qualification scoring is `0.15.0`.
Live evaluations use scoring `0.20.0` (`wait_relative_outcome_v1`). Live
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
