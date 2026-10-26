# Benchmark status

`benchmark/core_suite.json` and `benchmark/manifest.json` define Full:
769 scenarios across 502 physical sources. `benchmark/lite_suite.json` defines
Lite: 141 scenarios across 82 sources. Runtime scenario contracts live under
`scenarios/<domain>/`. Both catalogs retain the source identities of the latest
maintained benchmark.

| Domain | Full | Lite |
| --- | ---: | ---: |
| Autonomous driving | 7 | 7 |
| Building energy | 18 | 10 |
| Datacenter | 142 | 33 |
| Logistics | 527 | 53 |
| Microgrid | 37 | 28 |
| Power grid | 19 | 10 |
| Traffic | 19 | 0 |
| **Total** | **769** | **141** |

Lite is a model-informed development panel. It omits traffic and has limited
long-horizon coverage: 121 cases have at most 32 ticks, and only two exceed
192 ticks. It cannot establish broad long-horizon capability on its own.
Its membership and source bindings are fixed; selection evidence is distinct
from new model evaluations.

Qualification records retain scoring version `0.15.0`. Live episode scoring
uses `0.21.0`; the default raw reader uses `0.28.0`, and the optional anchored
scorer uses `0.31.2`. These identify different scoring contracts, not alternative
runtime datasets. See [evaluation](EVALUATION.md).

The manifest records `formal_evaluation_ready=true`, `leaderboard_eligible=false`
and `public_release_ready=false`. Readiness describes the admitted task/runtime
snapshot. It does not certify completed Full provider runs or model results.
No result table is distributed here.

The independent realtime scorecard uses `native_dt_v1` and the 37-row
`benchmark/realtime_speed_suite.json`. Its scores are not pooled into the primary
logical treatment. Public logical runs use `run_full.py` and `run_lite.py`;
formal realtime execution additionally requires a compatible evaluation manifest.

Frozen offline scoring contracts resolve through
[evaluation/policies](../evaluation/policies/README.md). Their original hashes
remain distinct from the current runtime catalog. Code identity is recorded as
provenance; dataset and treatment integrity remain mandatory.

Two CityLearn runtime signatures differ from their original catalog signatures
because source-lock paths were flattened. Lite records both and verifies the
runtime signature against the hash-bound YAML at binding and worker entry.
Those runtime trajectories cannot replace the original frozen offline inputs.
Public source bindings carry no private qualification case ledger and do not
promote formal eligibility.
