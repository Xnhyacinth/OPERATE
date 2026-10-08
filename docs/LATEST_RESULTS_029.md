# OPERATE-Lite141 retrospective results (0.29.1)

This table uses the frozen [0.29.1 retrospective protocol](EVALUATION_029.md)
and the frozen0.28 outcome index. It evaluates the selected existing trajectories;
the protocol refinement is retrospective, not preregistered. The panel is
model-informed and does not establish held-out generalization.

**Readiness update (2026-10-07).** A subsequent audit confirmed a SUMO public
recovery-token clock mismatch and unresolved recovery/normalization semantics.
This table preserves the historical results; it does not certify model
performance under corrected feedback or a fully validated final capability
ranking. See the protocol's readiness update before using the rankings.

All23 declared models remain visible.20 have complete141-case scores; HY3 has
138 determined scores and3 missing service measurements; the two Grok selections
have no authenticated score in this input manifest. Their trajectories are not
assigned zero and their existence is not disputed. This table describes the
bound manifest's evidence, not a search over every possible local archive.

Native costs/components, task-family/domain summaries, paired component
comparisons, and source/domain deletion sensitivity are part of the results.
Generate the full bundle with the command in the protocol; this summary is not
a substitute for those files or their underlying evidence.

Protocol SHA-256: `eae7248592bde90b770f32600e56669d728a58f59e546da102a0014a6961b32d`.

The complete 0.29.0/0.29.1 archive is indexed in the private trajectory dataset's
2026-10-08 delivery manifest. The [latest version comparison](LATEST_RESULTS_030.md)
includes these preserved scores and the opt-in 0.30 revision.

Scoring runtime SHA-256: `eba28acb250eda45d9a11012ecf4365134a7e4294869d75472070638b9072034`.

Q/N/F retain the 0.28 fixed-panel outcome definition. The point rank is a declared utility ordering, not a capability rank.
Missing source capability contracts or measurements remain N/A. No six-axis composite or repeated-run confidence interval is manufactured.

Protocol: `lite_fixed_panel_retrospective_outcome029.v1`. Outcome ranks compare 20 complete models out of 23 declared models; incomplete models are unranked. Optional six-axis calibration is not an outcome-ranking prerequisite.
Comparison policy: `latest_framework_user_assumed`. Original execution provenance remains available in the full report.

| Model | Q | N | F | Scored / expected | Hard failure % | Outcome rank | Measurement rank range | Delete-source rank range |
|---|---:|---:|---:|---:|---:|---:|---|---|
| hy4-preview | 44.4805 | 44.4996 | 85.4332 | 141/141 | 0.0000 | 1 | 1–1 | 1–1 |
| deepseek-ai/deepseek-v4-pro-0813 | 43.1890 | 43.2465 | 83.6999 | 141/141 | 0.0000 | 2 | 2–2 | 2–5 |
| moonshotai/kimi-k3 | 43.0791 | 44.0697 | 80.9871 | 141/141 | 0.0000 | 3 | 3–3 | 2–5 |
| deepseek-ai/deepseek-v4.1-flash | 43.0716 | 43.2001 | 80.3676 | 141/141 | 0.2315 | 4 | 4–4 | 2–4 |
| claude-sonnet-5.5 | 42.3485 | 42.9235 | 83.0135 | 141/141 | 0.6944 | 5 | 5–5 | 4–6 |
| qwen/qwen3.8-max | 42.1624 | 44.1089 | 79.3913 | 141/141 | 0.6944 | 6 | 6–6 | 2–6 |
| glm-5.2 | 40.9570 | 42.3416 | 78.2834 | 141/141 | 0.0000 | 7 | 7–7 | 7–9 |
| glm-5.3-flash-ioa | 40.7469 | 41.6071 | 82.4075 | 141/141 | 0.2315 | 8 | 8–8 | 7–10 |
| gpt-6.1-sol | 40.2767 | 42.6694 | 74.5329 | 141/141 | 0.2315 | 9 | 9–9 | 7–13 |
| deepseek-ai/deepseek-v4-flash-0731 | 39.9315 | 41.6899 | 76.9831 | 141/141 | 0.0000 | 10 | 10–10 | 8–12 |
| zai-org/glm-5.3 | 39.8431 | 42.0928 | 76.1586 | 141/141 | 0.0000 | 11 | 11–11 | 9–12 |
| claude-opus-4-8 | 39.8173 | 41.2626 | 77.9034 | 141/141 | 0.0000 | 12 | 12–12 | 8–13 |
| qwen/qwen3.8-27b | 39.0799 | 41.5702 | 74.6888 | 141/141 | 0.0000 | 13 | 13–13 | 10–14 |
| gpt-6-sol | 38.1657 | 42.6549 | 67.9105 | 141/141 | 0.1543 | 14 | 14–14 | 13–16 |
| Qwen/Qwen3.6-27B | 38.0047 | 41.5028 | 73.4473 | 141/141 | 0.0000 | 15 | 15–15 | 14–16 |
| iquest-m1 | 37.8897 | 39.6676 | 73.2550 | 141/141 | 0.0000 | 16 | 16–16 | 14–16 |
| gpt-5.5 | 36.3688 | 39.9936 | 71.5816 | 141/141 | 0.3086 | 17 | 17–17 | 17–17 |
| gpt-5.6-sol | 35.1750 | 40.8805 | 64.8393 | 141/141 | 0.6944 | 18 | 18–18 | 18–18 |
| gpt-6-luna | 33.0836 | 39.2146 | 63.8492 | 141/141 | 1.1574 | 19 | 19–19 | 19–19 |
| gpt-5.6-luna | 32.5788 | 38.5210 | 62.7047 | 141/141 | 0.3086 | 20 | 20–20 | 20–20 |
| grok-4.6 | N/A | N/A | N/A | 0/141 | N/A | — | N/A | N/A |
| grok-4.7 | N/A | N/A | N/A | 0/141 | N/A | — | N/A | N/A |
| hy3-ioa | N/A | 43.9761 | N/A | 138/141 | 0.0000 | — | N/A | N/A |

Source-deletion ranges change panel composition; they are sensitivity analyses, not uncertainty from repeated model executions.
Measurement rank ranges cover recorded numeric precision within the complete cohort. Neither type is a confidence interval. Pairwise precision-unresolved relations are not merged into transitive tie groups. Point ranks use unrounded scores; exact ties share competition rank.

All-declared-model bounds include missing evidence. They are hypothetical possible competition ranks, not assigned ranks for incomplete models. Safety coverage is a fraction of fixed case weight; known declared gates do not imply comprehensive safety.

| Model | Combined Q bounds | Hypothetical rank among all declared | Known gate weight | Catastrophe modeled / unmodeled / unknown weight |
|---|---|---|---:|---|
| hy4-preview | 44.4805–44.4806 | 1–3 | 1.0000 | 0.7857 / 0.2143 / 0.0000 |
| deepseek-ai/deepseek-v4-pro-0813 | 43.1889–43.1890 | 2–4 | 1.0000 | 0.7857 / 0.2143 / 0.0000 |
| moonshotai/kimi-k3 | 43.0790–43.0791 | 3–5 | 1.0000 | 0.7857 / 0.2143 / 0.0000 |
| deepseek-ai/deepseek-v4.1-flash | 43.0715–43.0716 | 4–6 | 1.0000 | 0.7857 / 0.2143 / 0.0000 |
| claude-sonnet-5.5 | 42.3484–42.3485 | 5–7 | 1.0000 | 0.7857 / 0.2143 / 0.0000 |
| qwen/qwen3.8-max | 42.1624–42.1625 | 6–8 | 1.0000 | 0.7857 / 0.2143 / 0.0000 |
| glm-5.2 | 40.9569–40.9570 | 7–9 | 1.0000 | 0.7857 / 0.2143 / 0.0000 |
| glm-5.3-flash-ioa | 40.7469–40.7470 | 8–11 | 1.0000 | 0.7857 / 0.2143 / 0.0000 |
| gpt-6.1-sol | 40.2766–40.2767 | 9–12 | 1.0000 | 0.7857 / 0.2143 / 0.0000 |
| deepseek-ai/deepseek-v4-flash-0731 | 39.9315–39.9316 | 10–13 | 1.0000 | 0.7857 / 0.2143 / 0.0000 |
| zai-org/glm-5.3 | 39.8431–39.8432 | 11–14 | 1.0000 | 0.7857 / 0.2143 / 0.0000 |
| claude-opus-4-8 | 39.8173–39.8174 | 12–15 | 1.0000 | 0.7857 / 0.2143 / 0.0000 |
| qwen/qwen3.8-27b | 39.0799–39.0799 | 13–16 | 1.0000 | 0.7857 / 0.2143 / 0.0000 |
| gpt-6-sol | 38.1657–38.1658 | 14–17 | 1.0000 | 0.7857 / 0.2143 / 0.0000 |
| Qwen/Qwen3.6-27B | 38.0047–38.0048 | 15–18 | 1.0000 | 0.7857 / 0.2143 / 0.0000 |
| iquest-m1 | 37.8897–37.8898 | 16–19 | 1.0000 | 0.7857 / 0.2143 / 0.0000 |
| gpt-5.5 | 36.3688–36.3688 | 17–20 | 1.0000 | 0.7857 / 0.2143 / 0.0000 |
| gpt-5.6-sol | 35.1750–35.1751 | 19–21 | 1.0000 | 0.7857 / 0.2143 / 0.0000 |
| gpt-6-luna | 33.0835–33.0836 | 20–22 | 1.0000 | 0.7857 / 0.2143 / 0.0000 |
| gpt-5.6-luna | 32.5787–32.5788 | 21–23 | 1.0000 | 0.7857 / 0.2143 / 0.0000 |
| grok-4.6 | 0.0000–100.0000 | 1–23 | 0.0000 | 0.0000 / 0.0000 / 1.0000 |
| grok-4.7 | 0.0000–100.0000 | 1–23 | 0.0000 | 0.0000 / 0.0000 / 1.0000 |
| hy3-ioa | 36.2273–40.8085 | 8–20 | 1.0000 | 0.7857 / 0.2143 / 0.0000 |
