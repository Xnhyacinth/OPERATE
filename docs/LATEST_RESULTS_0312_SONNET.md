# Lite141 retrospective results: 0.31.2 — Sonnet completion

Snapshot: 2026-10-11. [Scoring protocol](EVALUATION_0312.md).

**Accepted scope: a reproducible, raw-reauthenticated fixed-panel retrospective point ranking.**
It is not a same-run formal cohort or held-out model-capability ranking.
Complete coverage qualifies a model for this point table; it does not establish
statistical separation from its neighbors. Seven new Sonnet driving episodes complete this snapshot.

## Results

| Point rank | Model deployment | Score | 95% task interval | 95% task rank interval | Specification rank range | Excl. driving |
|---:|---|---:|---|---|---|---:|
| 1 | hy4-preview | 49.04 | [34.5, 61.4] | 1–3 | 1–2 | 49.66 |
| 2 | deepseek-ai/deepseek-v4.1-flash | 47.32 | [34.5, 56.4] | 1–4 | 1–4 | 47.12 |
| 3 | moonshotai/kimi-k3 | 43.54 | [32.8, 52.0] | 1–5 | 2–4 | 40.63 |
| 4 | qwen/qwen3.8-max | 41.05 | [23.5, 53.8] | 2–13 | 3–7 | 45.21 |
| 5 | glm-5.2 | 34.93 | [18.6, 46.2] | 4–15 | 5–11 | 34.70 |
| 6 | hy3-ioa | 33.27 | [19.9, 45.1] | 4–13 | 3–9 | 31.02 |
| 7 | deepseek-ai/deepseek-v4-pro-0813 | 32.78 | [21.6, 42.5] | 4–14 | 5–13 | 28.51 |
| 8 | deepseek-ai/deepseek-v4-flash-0731 | 32.76 | [19.1, 44.0] | 4–13 | 5–12 | 28.89 |
| 9 | claude-sonnet-5.5 | 32.08 | [21.9, 40.9] | 4–14 | 7–10 | 30.30 |
| 10 | qwen/qwen3.8-27b | 31.18 | [18.3, 41.7] | 5–14 | 5–15 | 30.65 |
| 11 | glm-5.3-flash-ioa | 27.67 | [17.6, 38.6] | 4–16 | 11–14 | 27.85 |
| 12 | Qwen/Qwen3.6-27B | 27.40 | [17.3, 38.5] | 6–15 | 9–15 | 28.99 |
| 13 | gpt-6-sol | 26.68 | [13.3, 38.1] | 6–17 | 9–14 | 29.94 |
| 14 | gpt-6.1-sol | 26.56 | [14.6, 37.5] | 6–16 | 6–15 | 30.23 |
| 15 | zai-org/glm-5.3 | 25.86 | [12.7, 38.3] | 7–17 | 13–15 | 26.86 |
| 16 | gpt-5.6-sol | 17.27 | [6.4, 28.4] | 12–18 | 16–18 | 25.84 |
| 17 | gpt-5.5 | 17.21 | [1.3, 30.6] | 12–19 | 16–18 | 12.80 |
| 18 | claude-opus-4-8 | 14.79 | [0.6, 27.5] | 15–19 | 17–18 | 20.04 |
| 19 | gpt-6-luna | 7.89 | [-4.7, 20.6] | 17–20 | 19–20 | 10.70 |
| 20 | gpt-5.6-luna | 0.05 | [-14.6, 14.6] | 19–20 | 19–20 | 7.38 |

All ranked deployments have 141/141 measured cases. Specification ranges use the
same 20-model cohort across declared alternative weights and domain exclusions;
they are not confidence intervals. Bootstrap intervals resample physical-source
clusters within domains and do not measure repeated-run variability.

| Incomplete deployment | Coverage | Excl. driving score |
|---|---:|---:|
| deployment-16 | 134/141 | 10.93 |
| grok-4.6 | 0/141 | N/A |
| grok-4.7 | 0/141 | N/A |

## What changed and what remains uncertain

- The main score is now bounded by [-100, 100]. Applicable incomplete service cannot
  reach 100. Diagnostic cost gain g retains values above 100. Anchors and nominal weights
  are byte-bound to the same choices as 0.31.1; no model outcome selected new weights.
- All six domains retain mass 1/6. Between-model variance shares: autonomous_driving 29.0%, building_energy 32.6%, datacenter 11.1%, logistics 11.5%, microgrid 9.6%, power_grid 6.2%.
- Removing one case can move a model up to 4 ranks; minimum Spearman 0.962.
  The top 10 cases account for 46.9% of between-model variance.
- 34/141 selected references themselves attain joint 100. This is a cost reference plus
  service obligation scale, not percent of optimal ability. 131 upper anchors use privileged
  information and are not same-observation online competitors.
- Narrow cost headroom, clipping and the chosen domain/task distribution still affect
  comparisons. The sensitivity ranges expose this; they do not remove the dependence.
- Pairwise bootstrap letters are descriptive, without family-wise correction. Datacenter
  has only three source clusters. Historical/corrected driving agreement mixes harness
  conditions and is not pure repeated-run reliability.

## Evidence scope and reproduction

The source measurements were raw-reauthenticated. Formal-run, same-run
merge and held-out generalization certification remain false. Original
journals, provider/native evidence, full reports and recovery audits
remain in the private maintainer archive. This public document contains
aggregate results only.

Use [the 0.31.2 protocol](EVALUATION_0312.md) and the public
`scripts/evaluate_task_outcomes031.py` entrypoint with your own
authenticated trajectories. Its default policy is 0.31.2; the separate
default raw reader remains 0.28.0.

The [earlier 19-deployment snapshot](LATEST_RESULTS_0312.md) remains
historical; this snapshot adds seven whole Sonnet episodes.
