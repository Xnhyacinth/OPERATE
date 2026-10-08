# Current Lite141 results: 0.27, 2026-10-05

The default raw-trajectory reader evaluated 23 declared models against the fixed
141-case development panel: 2,853 of 3,243 target Q values are determined,
and 19 models have complete point scores. Source scales and domain/family/
physical-source/case weights are unchanged. These are descriptive recorded
deployment outcomes under the explicitly user-authorized cross-framework
comparison policy. Original providers, profiles, trees and errors remain bound
in the case ledger; this is not a same-run formal comparison.

Q is continuous operational utility, not task pass rate or held-out accuracy.
N measures source-scaled native economic quality. F measures source-obligation
fulfillment over the 131 applicable cases; the ten CityLearn economic-only
cases do not enter F. See [the scoring protocol](EVALUATION_027.md).

| Rank | Model | Q /100 | N /100 | F /100 |
|---:|---|---:|---:|---:|
| 1 | hy4-preview | 40.3624 | 40.3815 | 85.4332 |
| 2 | deepseek-ai/deepseek-v4-pro-0813 | 38.7481 | 38.8056 | 83.6999 |
| 3 | deepseek-ai/deepseek-v4.1-flash | 38.7306 | 38.8592 | 80.3676 |
| 4 | moonshotai/kimi-k3 | 38.5229 | 39.5135 | 80.9871 |
| 5 | qwen/qwen3.8-max | 37.9899 | 39.9364 | 79.3913 |
| 6 | claude-sonnet-5.5 | 37.7404 | 38.3154 | 83.0135 |
| 7 | glm-5.2 | 36.5408 | 37.9254 | 78.2834 |
| 8 | glm-5.3-flash-ioa | 36.5137 | 37.3739 | 82.4075 |
| 9 | gpt-6.1-sol | 35.8955 | 38.2882 | 74.5329 |
| 10 | zai-org/glm-5.3 | 35.3156 | 37.5653 | 76.1586 |
| 11 | claude-opus-4-8 | 35.2325 | 36.6778 | 77.9034 |
| 12 | qwen/qwen3.8-27b | 34.7100 | 37.2002 | 74.6888 |
| 13 | deepseek-ai/deepseek-v4-flash-0731 | 34.5074 | 36.2658 | 76.9831 |
| 14 | gpt-6-sol | 33.8449 | 38.3340 | 67.9105 |
| 15 | Qwen/Qwen3.6-27B | 33.0500 | 36.5481 | 73.4473 |
| 16 | iquest-m1 | 32.5055 | 34.2833 | 73.2550 |
| 17 | gpt-5.6-sol | 31.1913 | 36.8968 | 64.8393 |
| 18 | gpt-6-luna | 29.0534 | 35.1844 | 63.8492 |
| 19 | gpt-5.6-luna | 28.2542 | 34.1964 | 62.7047 |

## Explicit gaps

| Model | Determined Q /141 | Missing evidence | Fixed-panel Q bounds |
|---|---:|---|---:|
| hy3-ioa | 138 | 3 historical OpenDSS cases lack recoverable full-node proof | [35.2744, 35.6772] |
| gpt-5.5 | 36 | 105 cases have no closed successful whole execution | [18.6651, 83.2154] |
| grok-4.6 | 0 | 141 cases have no closed successful whole execution | [0.0000, 100.0000] |
| grok-4.7 | 0 | 141 cases have no closed successful whole execution | [0.0000, 100.0000] |

These bounds retain the original unknown weights; they are identification
bounds, not confidence intervals or point ranks. HY3 retains independently
established N=38.8449, while F and Q remain unavailable. Its three missing
case signatures are `26b91b6632e18e64`, `4906475ce3e8295c` and
`bafda3192d001a15`, all at seed 42. Missing outcomes are not scored as zero.

## Evidence and verification

Scoring read original journals/configurations, native snapshots, ledgers and
provider audits; previous score reports were not inputs. The final reader
made zero provider calls, native episode replays or source constructor calls.
It selected whole episodes before scoring and preserved historical artifacts.
Selected original records total 2,856; padded missing targets do not count as
observed records. GPT-5.5 has 36 selected records and the Grok models have zero.

Of the 65 prior native-meter gaps, 62 have separately authenticated recovery
evidence: 46 replay-equivalence certificates and 16 locked CIGRE constant-PQ
source-law certificates. The original C and raw bytes are unchanged. The
remaining three OpenDSS cases lack reconstructable full-node voltage evidence.

Two complete native FJSP executions previously marked solely with
`implementation_tree_drift` now contribute under the explicit user-assumed
selection policy after completed-runtime, source-window, provider, actions,
costs and fatal-state authentication. Original error/repair flags and strict
identity ineligibility remain visible. The report contains 2,334 strictly
qualified execution identities and does not claim formal certification.

Independent arithmetic/cost review and a public `--audit-report` rebuild from
a separate source directory verified every case and aggregate. The software
review addressed reproduced selector and completed-runtime fatal-state bugs;
the final changed-scope review found no actionable regressions. Focused tests,
Ruff and diff checks passed. Frozen release/source/scenario/0.26.1 evidence
is unchanged; prospective formal-runtime binding is not newly certified.

Realtime remains separate: E3 has 21/22 complete fixed delay-triplet groups;
Core w16 has 35/37 valid HY3 cases and 34/37 valid HY4 cases. Historical
path-preimage derivations retain original SHA checks and failed flags.
Different concurrency cohorts are not merged into a formal run or this Q.

## Reproduce

[Raw input manifest](https://huggingface.co/datasets/Xnhyacinth/OPERATE-Private-Trajectories-v0.62/blob/main/outcome027_20261005/results/input_manifest.json),
[full result](https://huggingface.co/datasets/Xnhyacinth/OPERATE-Private-Trajectories-v0.62/blob/main/outcome027_20261005/results/report.json),
[case ledger](https://huggingface.co/datasets/Xnhyacinth/OPERATE-Private-Trajectories-v0.62/blob/main/outcome027_20261005/results/case_rows.jsonl),
[CSV](https://huggingface.co/datasets/Xnhyacinth/OPERATE-Private-Trajectories-v0.62/blob/main/outcome027_20261005/results/table.csv) and
[archive inventory and audit bindings](https://huggingface.co/datasets/Xnhyacinth/OPERATE-Private-Trajectories-v0.62/blob/main/outcome027_20261005/manifest.json)
are in the private trajectory dataset. Access requires its existing permission.

The inventory binds a content-addressed original-raw archive, exact locked
source assets, new-run/supplementary evidence, a software snapshot and final
review/audit receipts. Restore the software, source assets and bound historical
recovery-source aliases into one directory. Each of the six aliases and six
canonical OpenDSS files retains its independently locked bytes. Then
extract the raw archive into a separate directory, and install the pinned
repository environment using the [README](../README.md), then run:

```bash
uv run --frozen --no-sync python scripts/evaluate_trajectories.py \
  --manifest '<raw-directory>/raw_input_manifest.json' \
  --artifact-root '<raw-directory>/blobs' \
  --audit-report '<downloaded-report.json>' \
  --output-dir '<new-audit-directory>'
```

Scoring code: `63f41f7104ccf2b903e9ddae9256ea1bd2bda425`.
Runtime SHA-256: `ae7c9e8db43e98576ea464b7352c95b8470b0f23d37afe1b8d673addda3db145`.
Report SHA-256: `7b5a6f02d5219767300f6f3e66710400d7bd6d71e6c73b7e849cf8195aa81842`.
Original manifest SHA-256: `2043e76055233ab827c657b7c103325b69b7ccde7999f9382475999c93e91143`.
