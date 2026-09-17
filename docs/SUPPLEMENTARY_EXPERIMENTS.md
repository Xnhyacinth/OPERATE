# Supplementary behavior experiments

OPERATE reports three matched analyses outside the primary
`logical_persistent` score. They use the same executable task, model route,
generation settings, current observation, and scoring contract within each
comparison. A pilot or incomplete pair is engineering evidence, not a paper
result.

## E1: persistent versus stateless context

E1 measures whether session continuity changes decisions over long operational
episodes. Each task is run once with `logical_persistent` and once with
`logical_stateless`. Both arms receive the same complete current observation
and provider context budget; the stateless arm differs only by starting each
decision from a fresh system/user transcript.

Report paired changes in primary score, native cost, deadline misses, safety
violations, successful evidence-consuming controls, late-stage failures, model
calls, and tool calls. A pair is invalid when either arm has a provider,
artifact, or prompt-budget failure.

The fixed all-dimension score remains the frozen E1 logical score. Also report
the applicable-dimension view within each matched task so that a task with many
declared non-applicable dimensions is not mistaken for weak model behavior.
Never compare that adaptive view across unmatched task mixes, and never pool an
E1 score with the main table unless their scorer identities match.

## E3: fixed provider delay

E3 measures sensitivity to decision latency under realtime execution. Each
task uses the same model and task state with fixed 0 s, 1 s, and 5 s delivery
delay. The environment continues while the request is in flight. Report score,
action expiry, supersession, cancellation, takeover, deadline misses, useful
controls, and provider latency. Interpret a task only when all three delay arms
pass the artifact and provider-audit gates.

E3 has no primary aggregate. Its result is the matched delay-response curve:
response misses, effected or expired actions, discarded late responses,
superseded turns, takeovers, controlled holds, and protocol-valid responses at
0, 1, and 5 seconds. This keeps transport timing separate from logical task
quality.

## E4: information reveal ablation

E4 pairs `reveal` and `withhold` conditions for the same task and deadline.
The intervention changes only whether decision-relevant evidence becomes
available. Report investigation count and time, consumed reveal evidence,
commit timing, native effect, missed action opportunity, native cost, and score.
Results with no remaining action opportunity after investigation are retained:
they measure the operational cost of information acquisition rather than a
scoring defect.

E4 has no primary aggregate. Report reveal-minus-withhold changes in
investigation actions, successful commit, query-deadline exhaustion, visible
evidence, model calls, tool calls, and native task outcomes. A pair is invalid
if its source-state, runtime, harness, or provider configuration hashes differ.

## Expansion rule

Run one complete model matrix before adding models. Expand to another model only
after every pair or delay group is complete and the shared harness has no open
validity defect. One run per condition is the initial analysis; repeated trials
are reserved for cells where stochastic provider behavior or a variance claim
requires them. Realtime supervision remains separate from the primary logical
leaderboard.
