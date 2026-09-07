# Current release

The public repository ships one current benchmark: scoring `0.15.0`, 769 Core
rows across 502 physical sources, and 193 Lite rows across 122 sources. Formal
provider runs and leaderboard publication remain pending.

## Promoted Core

- 769 source-grounded scenario contracts across 502 physical sources
- seven domains: Autonomous Driving, Building Energy, Datacenter, Logistics,
  Microgrid, Power Grid, and Traffic
- tracked source locks and compact Alibaba/DynaSched assets under `sources/`
- current contracts under `scenarios/<domain>/...`
- Core suite at `benchmark/core_suite.json`
- matching catalog at `benchmark/manifest.json`
- Lite suite at `benchmark/lite_suite.json`

The domain distribution is 7 Autonomous Driving, 18 Building Energy,
142 Datacenter, 527 Logistics, 37 Microgrid, 19 Power Grid, and 19 Traffic
rows.

`core_suite.json` together with `manifest.json` defines the public denominator.

## Status

- `formal_evaluation_ready=true`
- provider logical/realtime shards pending
- `leaderboard_eligible=false`

`logical_persistent` is the primary leaderboard treatment. `realtime_persistent`
is a separate supervision scorecard. `logical_stateless` is a compatibility
ablation.
