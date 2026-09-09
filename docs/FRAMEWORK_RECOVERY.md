# Recovery and result validity

Public `run_full.py` and `run_lite.py` do not require a particular Git commit,
a clean working tree, or the maintainer's implementation hash. They default to
`--implementation-policy provenance`. New attempts record the actual code used;
ordinary resume retains completed cells after code updates. Dataset bytes,
scenario/seed/pass, model settings and scoring protocol are still checked.
Changing code does not retroactively alter recorded results or certify them as
comparable: disclose behavior-changing repairs and remeasure affected cells.

## Requests and campaigns

The provider loop recognizes httpx/httpx2 transport errors, honors numeric and
HTTP-date Retry-After, and keeps a pending request's messages and logical
state unchanged across bounded retries. Defaults are five attempts and an
1,800-second recovery window. Set `--provider-retry-max-attempts` and
`--provider-retry-max-elapsed-s` explicitly for a new treatment. Socket waits
remain bounded by the SDK timeout; this is not a hard wall-clock kill switch.
A complete generation survives a subsequent usage-tail transport failure;
partial generations cannot execute. Observed model mismatches stop before action.

Quota cooldowns do not consume transport budgets. A deterministic framework
failure is held as `needs_repair` and is not counted as model task failure.
Campaign attempts are append-only; explicit budget extensions retain counts
and cannot bypass a provider cooldown. Other independent cells can continue.

## Episode checkpoint

To recover inside an unfinished logical episode, use `--episode-checkpoint`
with `--implementation-policy strict` and `--save-trajectories`. This optional
path uses seeded replay, not native-runtime pickle. It checks input, observation,
evidence and agent-state hashes at every recorded boundary before new requests.
Incremental JSON state, an atomic head and a cell lock detect damaged/truncated
journals and concurrent writers. Incompatible checkpoints fail closed.

A `recovery_audit` binds each attempt's provider records and counts only newly
attempted requests; reused successful prefixes are counted once. Ordinary final
LLM statistics describe the retained logical trajectory, not cumulative cost.
Missing/unsettled historical audit remains explicitly unclosed. Old trajectories
without journals cannot be retrofitted into checkpoints.

## Completed evidence and scoring

The runner saves `*.completed_runtime.json` before counterfactual processing
and complete `*.scoring_inputs.json` before scoring. A hash-verified offline
rescore invokes no backend or model and creates a separate result:

```bash
uv run python -m evaluation.scoring_snapshot \
  --input /path/to/episode.scoring_inputs.json \
  --sha256 RECORDED_SHA256 --output /path/to/new-rescore.json
```

CityLearn inspections return the last completed native storage state, while
the decision clock still names the next interval. Signed native energy costs
retain export credit; nonfinite costs are rejected. Earlier trajectories that
showed unfilled storage output slots as zero need remeasurement: rescoring alone
cannot repair decisions made from incorrect observations.
