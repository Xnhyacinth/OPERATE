# Evaluation contract

OPERATE measures outcomes from executable environments and evidence records.
A plausible response or a tool call alone does not establish a successful action.

| Score | Entry point | Interpretation |
| --- | --- | --- |
| Live logical outcome | `run_full.py`, `run_lite.py` | `ranking.primary_score`, wait-relative operational outcome; live scoring `0.21.0` |
| Offline raw outcome | `scripts/evaluate_trajectories.py` | Source-normalized cost, service and hard-gate evidence; policy `0.28.0` |
| Offline anchored outcome | `scripts/evaluate_anchored_outcomes.py` | Reference cost and applicable service caps; policy `0.31.2` |
| Realtime supervision | `scripts/batch_realtime_llm_eval.py` | Independent latency, action lifecycle and supervision scorecard |

These contracts are separate and their scores must not be pooled. Qualification
version `0.15.0` describes the original admitted task checks; it is not a new
model evaluation.

## Evidence and completeness

Scoring authenticates scenario inputs, journals, run configuration and native
and provider artifacts. Original execution identities remain attached to the
measurements. A new scorer does not certify an old run under new execution
conditions. Missing evidence remains unknown with fixed-weight bounds.

The raw offline index requires all 141 Lite cases for a complete point score.
Full and Lite have different denominators. Lite was selected with development
model outcomes, so fixed-panel comparisons are not held-out generalization
estimates. Source-cluster bootstrap intervals measure task sampling rather than
repeated-run variability.

Every scored dimension needs evidence IDs or an explicit inapplicability reason.
Counterfactual prevention uses a real no-action replay or a machine-readable
opt-out. Native costs and component units stay attached to the backend that
produced them. Capability diagnostics such as proactive monitoring, correct
silence and memory remain separate from aggregate outcome quality.

## Reproduction

Follow the [model evaluation runbook](FORMAL_EVALUATION.md). For recorded runs,
use [offline scoring](OFFLINE_SCORING.md) or explicitly choose
[anchored scoring](ANCHORED_SCORING.md). Both require authenticated input
artifacts and make no provider calls or native episode replays during scoring.

[Frozen policy inputs](../evaluation/policies/README.md) preserve their original
bytes and scoring semantics. File relocation never substitutes a current runtime
scenario for an older policy's different bytes. Schema and scoring versions
remain necessary compatibility identifiers.

For evidence construction, see [the evidence contract](EVIDENCE_CONTRACT.md).
For interrupted runs and postprocessing, see [recovery](EVALUATION_RECOVERY.md).
