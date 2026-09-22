# Evaluation deadlines and postprocessing recovery

Logical batch evaluation separates provider interaction from native
counterfactual/scoring work. A completed interaction is not yet a completed
score. Keep the completed-runtime snapshot and its associated evidence even
when scoring fails or a worker exceeds its deadline.

## Bounded workers

`scripts/batch_llm_eval.py` accepts:

- `--episode-timeout-s`: total wall-clock limit per attempt, including startup
  and postprocessing; default 21,600 seconds (six hours).
- `--postprocessing-timeout-s`: limit after postprocessing begins; default
  10,800 seconds (three hours).

Both must be finite and positive. They are operational limits, not attribution
caps. Set them to cover the required native workload. Expiry does not turn an
unfinished replay into a score or silently drop attribution rows.

Each scheduled job runs in an isolated child process under a surviving pool
worker. On timeout, the supervisor kills the child's process group, including
native simulator subprocesses, and records `worker_wall_clock_timeout` with
its stage and budget. Other jobs continue. The failed attempt is eligible for
retry on a later explicit resume, rather than being retried indefinitely
inside the current invocation.

Provider timeouts still apply independently. Existing already-started workers
do not hot-load these controls or backend fixes.

## Resume after interaction completes

With saved trajectories, `--resume` checks for a bound `completed_runtime`
snapshot before starting a new model interaction. It verifies the snapshot
hash, byte count, scenario/seed, agent configuration and checkpoint identity.
It also checks the original provider, semantic-ledger and trajectory bytes.
A valid snapshot resumes counterfactual replay and scoring without model calls.
A corrupt, mismatched or unsupported completed snapshot produces an explicit
repair error; it is not silently replaced by another paid interaction run.

New snapshots include the complete scoring context and can reproduce the
original result through the same postprocessing function as a live episode.
Older CityLearn snapshots support scoring recovery, with unavailable runtime
telemetry labelled explicitly. Other legacy snapshots lacking the required
context fail closed.

Same-runtime recovery still needs all ordinary evidence and eligibility
checks. Cross-runtime recovery is restricted to ordinary `provenance` runs and
is diagnostic-only. Strict/formal/checkpoint runs do not adopt a different
implementation identity through recovery. Deadline changes likewise do not
silently relax strict run compatibility.

Recovery writes new artifacts and retains both source and recomputation
identities. It does not rewrite the historical completed-runtime snapshot.

## Independent recovery command

For a stopped attempt or an archived snapshot, pin the original snapshot
identity in a JSON file and retain its independently recorded SHA-256:

```bash
uv run python scripts/recover_completed_episode.py \
  --input /path/to/episode.completed_runtime.json \
  --sha256 ORIGINAL_RECORDED_SHA256 \
  --identity /path/to/original_identity.json \
  --output /path/to/new_recovery.json \
  --timeout-s 10800
```

The identity file contains the original snapshot's full `payload.identity`,
not a newly generated current identity. The caller must verify it against the
original run record. The output must not already exist; its parent directory
must exist. Associated new scoring/evidence files are written into a separate
`.artifacts` directory next to the output.

The standalone command is supervised by the same killable deadline mechanism.
Its output never independently claims formal completion. Inspect
`same_contract_recovery`, `legacy_context_reconstructed`, the source and
recomputation identities, and attribution coverage before using its scores.
A moved archive can resolve a sibling provider/ledger artifact only when its
original hash and recorded byte count match.

A provider HTTP 400 is a separate failure. Recovery can use a previously
completed, identity-bound trajectory; it cannot manufacture the missing
interaction for an attempt that never completed.
