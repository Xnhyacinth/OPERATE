# Latest Lite141 retrospective results: 0.30

Snapshot: 2026-10-08. The default raw-trajectory evaluator remains **0.28.0**;
**0.30.0** is the latest opt-in task-outcome revision. The panel has 23 declared
models, 3,243 fixed targets, 2,958 determined Q values and 20 complete 141-case
models. HY3 has 138/141; Grok 4.6 and 4.7 have no selected outcomes. Missing
measurements retain fixed weights and bounds; they receive no point rank.

The table uses the completed historical comparison panel, including the GPT-5.5
105-case supplement. It preserves the published versions. Score changes across
versions describe changed measurement rules, not changed model execution.
0.25 atan and 0.25 v3 are distinct historical native-quality mappings. 0.26.1
and 0.27 apply per-case fulfillment bottlenecks; 0.28 corrects DSS exposure.
0.29.0/0.29.1 retain the 0.28 Q values and add operational reporting/ranking
companions. 0.30 additionally corrects LV/CIGRE population exposure and native
driving recovery. See [0.30 protocol](EVALUATION_030.md),
[0.28 protocol](EVALUATION_028.md) and [0.29 reporting](EVALUATION_029.md).

| Model | 0.25 atan | 0.25 v3 | 0.26.1 | 0.27 | 0.28 | 0.29.0 | 0.29.1 | 0.30 | 0.30 rank | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hy4-preview | 20.9565 | 40.3815 | 40.3624 | 40.3624 | 44.4805 | 44.4805 | 44.4805 | 51.6752 | 1 | 141/141 |
| deepseek-ai/deepseek-v4-pro-0813 | 20.0941 | 38.8056 | 38.7481 | 38.7481 | 43.1890 | 43.1890 | 43.1890 | 50.6491 | 2 | 141/141 |
| deepseek-ai/deepseek-v4.1-flash | 20.4680 | 38.8592 | 38.7306 | 38.7306 | 43.0716 | 43.0716 | 43.0716 | 49.9571 | 3 | 141/141 |
| moonshotai/kimi-k3 | 20.3898 | 39.5135 | 38.5229 | 38.5229 | 43.0791 | 43.0791 | 43.0791 | 49.8603 | 4 | 141/141 |
| claude-sonnet-5.5 | 20.1224 | 38.3154 | 37.7404 | 37.7404 | 42.3485 | 42.3485 | 42.3485 | 49.6839 | 5 | 141/141 |
| qwen/qwen3.8-max | 20.9435 | 39.9364 | 37.9899 | 37.9899 | 42.1624 | 42.1624 | 42.1624 | 49.3357 | 6 | 141/141 |
| glm-5.3-flash-ioa | 19.3439 | 37.3739 | 36.5137 | 36.5137 | 40.7469 | 40.7469 | 40.7469 | 48.4261 | 7 | 141/141 |
| claude-opus-4-8 | 19.1130 | 36.6778 | 35.2325 | 35.2325 | 39.8173 | 39.8173 | 39.8173 | 48.4236 | 8 | 141/141 |
| glm-5.2 | 19.6188 | 37.9254 | 36.5408 | 36.5408 | 40.9570 | 40.9570 | 40.9570 | 48.0812 | 9 | 141/141 |
| gpt-6.1-sol | 19.5302 | 38.2882 | 35.8955 | 35.8955 | 40.2767 | 40.2767 | 40.2767 | 47.4768 | 10 | 141/141 |
| deepseek-ai/deepseek-v4-flash-0731 | 18.8225 | 36.2658 | 34.5074 | 34.5074 | 39.9315 | 39.9315 | 39.9315 | 47.3159 | 11 | 141/141 |
| zai-org/glm-5.3 | 19.4888 | 37.5653 | 35.3156 | 35.3156 | 39.8431 | 39.8431 | 39.8431 | 47.2138 | 12 | 141/141 |
| qwen/qwen3.8-27b | 19.4197 | 37.2002 | 34.7100 | 34.7100 | 39.0799 | 39.0799 | 39.0799 | 46.6918 | 13 | 141/141 |
| iquest-m1 | 17.5239 | 34.2833 | 32.5055 | 32.5055 | 37.8897 | 37.8897 | 37.8897 | 45.5714 | 14 | 141/141 |
| Qwen/Qwen3.6-27B | 18.7451 | 36.5481 | 33.0500 | 33.0500 | 38.0047 | 38.0047 | 38.0047 | 45.2852 | 15 | 141/141 |
| gpt-6-sol | 19.6664 | 38.3340 | 33.8449 | 33.8449 | 38.1657 | 38.1657 | 38.1657 | 45.1471 | 16 | 141/141 |
| gpt-5.5 | 18.3044 | 35.6167 | 31.9919 | 31.9919 | 36.3688 | 36.3688 | 36.3688 | 43.7011 | 17 | 141/141 |
| gpt-5.6-sol | 18.9133 | 36.8968 | 31.1913 | 31.1913 | 35.1750 | 35.1750 | 35.1750 | 42.4976 | 18 | 141/141 |
| gpt-6-luna | 18.1244 | 35.1844 | 29.0534 | 29.0534 | 33.0836 | 33.0836 | 33.0836 | 41.5339 | 19 | 141/141 |
| gpt-5.6-luna | 17.5468 | 34.1964 | 28.2542 | 28.2542 | 32.5788 | 32.5788 | 32.5788 | 39.5334 | 20 | 141/141 |
| grok-4.6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0/141 |
| grok-4.7 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0/141 |
| hy3-ioa | 19.7836 | 38.8449 | 35.6772 | N/A | N/A | N/A | N/A | N/A | N/A | 138/141 |

## Calculation and accompanying evidence

For each applicable case, Q=min(F,N); source-authorized economic-only cases use
Q=N and verified hard failures score zero. The aggregate is sum(w_i Q_i), using
frozen equal-domain/family/source/case hierarchical weights. A minimum taken
after aggregation would be a different metric. Unknown cases keep their weight.
The 0.30 loss revisions affect nine LV/CIGRE and seven driving cases; original
cost components, 0.28 evidence and DSS measurement are preserved.

Full 0.30 report SHA256:
`d95ed6a0c583425a4479908ebdb53a89b81a25cb1b6833b955253b914e4e482c`.
Full 0.29.1 report SHA256:
`af1040ac9ec47946ed952e825fa6a9e601969cdf19346e0623986f46b02b388a`.
The private trajectory archive carries complete reports, original input
manifests, recovery evidence, checksums and audit receipts. Public aggregate
tables are available in the [dated HF result companion](https://huggingface.co/datasets/Xnhyacinth/OPERATE/tree/main/results/lite141_20261008).

Point ranks describe recorded outcomes on Lite-Dev. Precision and missingness
bounds are not repeated-run confidence intervals. Historical driving feedback
used the old token clock; rescoring does not repair those decisions. Source and
domain deletion sensitivity, including driving exclusion, accompanies the
private full report. Optional Full formal provider evaluation remains pending.

The [offline analysis layer](LITE141_ANALYSIS.md) provides a separate 12-model
behavior/usage subset and plots. Its subset is not the 23-model denominator.
