# Latest Lite141 results: 0.28, 2026-10-06

All 23 declared models were rescored from original raw execution evidence under
`evaluation/policies/lite141_028.json`. The fixed panel has 3,243 targets:
2,958 determined Q values and 20 complete 141-case models. GPT-5.5 now has its
completed 141-case input. The three HY3 native-meter gaps and 282 unavailable
Grok targets retain their weights and have no point rank.

Q is the continuous operational outcome index; N is native quality and F is
fulfillment on the 131 applicable cases. DSS N uses source-normalized node-time
violation exposure plus global extrema severity. Other native quality rules,
original C, F, hard gates and source weights are unchanged. The policy was
fixed before computing this ranking; model scores do not set its scales.
See [the 0.28 protocol](EVALUATION_028.md).

| Rank | Model | Q /100 | N /100 | F /100 | Q coverage |
|---:|---|---:|---:|---:|---:|
| 1 | hy4-preview | 44.4805 | 44.4996 | 85.4332 | 141/141 |
| 2 | deepseek-ai/deepseek-v4-pro-0813 | 43.1890 | 43.2465 | 83.6999 | 141/141 |
| 3 | moonshotai/kimi-k3 | 43.0791 | 44.0697 | 80.9871 | 141/141 |
| 4 | deepseek-ai/deepseek-v4.1-flash | 43.0716 | 43.2001 | 80.3676 | 141/141 |
| 5 | claude-sonnet-5.5 | 42.3485 | 42.9235 | 83.0135 | 141/141 |
| 6 | qwen/qwen3.8-max | 42.1624 | 44.1089 | 79.3913 | 141/141 |
| 7 | glm-5.2 | 40.9570 | 42.3416 | 78.2834 | 141/141 |
| 8 | glm-5.3-flash-ioa | 40.7469 | 41.6071 | 82.4075 | 141/141 |
| 9 | gpt-6.1-sol | 40.2767 | 42.6694 | 74.5329 | 141/141 |
| 10 | deepseek-ai/deepseek-v4-flash-0731 | 39.9315 | 41.6899 | 76.9831 | 141/141 |
| 11 | zai-org/glm-5.3 | 39.8431 | 42.0928 | 76.1586 | 141/141 |
| 12 | claude-opus-4-8 | 39.8173 | 41.2626 | 77.9034 | 141/141 |
| 13 | qwen/qwen3.8-27b | 39.0799 | 41.5702 | 74.6888 | 141/141 |
| 14 | gpt-6-sol | 38.1657 | 42.6549 | 67.9105 | 141/141 |
| 15 | Qwen/Qwen3.6-27B | 38.0047 | 41.5028 | 73.4473 | 141/141 |
| 16 | iquest-m1 | 37.8897 | 39.6676 | 73.2550 | 141/141 |
| 17 | gpt-5.5 | 36.3688 | 39.9936 | 71.5816 | 141/141 |
| 18 | gpt-5.6-sol | 35.1750 | 40.8805 | 64.8393 | 141/141 |
| 19 | gpt-6-luna | 33.0836 | 39.2146 | 63.8492 | 141/141 |
| 20 | gpt-5.6-luna | 32.5788 | 38.5210 | 62.7047 | 141/141 |
| — | grok-4.6 | N/A | N/A | N/A | 0/141 |
| — | grok-4.7 | N/A | N/A | N/A | 0/141 |
| — | hy3-ioa | N/A | 43.9761 | N/A | 138/141 |

## Missing evidence

| Model | Fixed-panel Q bounds | Reason |
|---|---:|---|
| grok-4.6 | [0.0000, 100.0000] | No eligible closed whole execution |
| grok-4.7 | [0.0000, 100.0000] | No eligible closed whole execution |
| hy3-ioa | [36.2273, 40.8085] | 3 historical OpenDSS cases lack qualified full-node fulfillment proof |

Bounds reflect missing measurements, not repeated-run confidence intervals.
HY3 retains independently verified N, but incomplete F and Q remain N/A.
An unavailable Grok result is not measured zero quality.

## Recalculation and review

- 2,961 selected original whole episodes; 2,439 satisfy strict execution identity.
- 62 historical recovery sidecars retained: 46 replay-equivalence and 16 CIGRE source-law certificates.
- 84 DSS component records verified; 2,877 other selected cases retain exactly the same N/Q.
- All original C/S/F, canonical episode hashes, native/provider/execution bindings, recovery, weights and recomputed 0.27 N/Q match the prior audited inputs.
- A separate arithmetic comparison rebuilt new DSS N from original components; the public `--audit-report` entry rebuilt every case and aggregate successfully.
- 173 focused regression tests, Ruff and `git diff --check` passed. A review-discovered policy-overlay override defect was reproduced and fixed: caller self-hashes cannot authorize new base contracts, weights or populations.
- Final `codex review --uncommitted` completed with no actionable defects after the fix.
- Display aliases reject merging different actual model identities; selected actual IDs remain explicit.
- Ordinary scoring and audit each made zero provider calls, native episode replays and source constructor calls.

The table compares recorded deployment outcomes under the declared descriptive
whole-episode policy. Original execution differences remain visible in the ledger.
It is suitable as the fixed-panel outcome ranking. Proactivity and situational
response mechanisms require their separate opportunity/response diagnostics;
the outcome total alone does not identify why a model attained a result.
No binary task-attainment or comprehensive long-horizon certification is implied.

## Local artifacts and reproduction

The complete result archive is indexed by the private HF trajectory manifest;
the [2026-10-08 public companion](https://huggingface.co/datasets/Xnhyacinth/OPERATE/tree/main/results/lite141_20261008)
contains aggregate version comparisons. Generated artifacts remain outside Git under
`.hl/analysis/outcome028_population_normalization_20261006/`:

- `raw_latest23_028_manifest.json`: original journals/configs, policy and recovery bindings.
- `results028/report.json`, `case_rows.jsonl`, `table.csv`, `report.md`: complete result and ledger.
- `audit028/audit_receipt.json`: full reader reconstruction receipt.
- `scoring_source028.tar.gz` and `scoring_source_manifest.json`: exact scoring code/policies/lockfiles; source assets and raw bytes are restored separately.
- `comparison_receipt.json` and `compare_results.py`: independent lineage/arithmetic comparison.
- `native_component_review.md`, `code_review_final.md`, `scientific_review_final.md`: bounded implementation reviews.

```bash
uv run --frozen --no-sync python scripts/evaluate_trajectories.py \
  --manifest '<restored-results>/raw_latest23_028_manifest.json' \
  --artifact-root '<restored-original-evidence>/blobs' \
  --audit-report '<restored-results>/results028/report.json' \
  --output-dir '<new-audit-directory>'
```

Restore the exact bound original artifacts, source files and scoring runtime
before portable reproduction; the audit refuses a changed runtime. This run
uses the working implementation based on commit `5c3b1c61f9b873078df21d38209b85d115d003e2`,
with the new scoring bytes identified by the runtime hash below. Historical
0.27 policies, reports and publication records retain their original identity.

- Scoring runtime SHA-256: `a6134895e57fc5579180a641a4299ed010a87f1fa0ed6d91d4046468c9c3dfac`.
- Policy file SHA-256: `2afae5f8b0f8f6f52e4a45b23e0d7ad45cb93ee75613b984ec1cbd4d1cea4671`.
- Input manifest file SHA-256: `e3bb08856ad958c61255e310abfd28692ab8c40e0d662599c22e9e9b56bc4a1d`.
- Report file SHA-256: `4853cb63d2ce7d5b618b8afd12f9fef22339240550cb558662f6bb5361dce5ff`.
- Audit receipt file SHA-256: `a4d6ec569b1dec2877c59e1d2de64f3a842bef4d119e9f3580a818c90466d9f2`.
- Scoring source archive SHA-256: `93928cb0fe313881e00cf96070437054e953b915c5341faff306e14ac6595a5b`.
- Comparison receipt file SHA-256: `11f8ed2739cc9e5d248e348a059a697e67e7a7ffb67f0ff20f51dfa765c87334`.
