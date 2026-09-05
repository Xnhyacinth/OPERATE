# Validation policy

Validate the changed behavior, not every historical experiment. Data quality,
diversity, executable source inputs and correct agent/environment interaction
remain the objective. Full replay is an explicit maintenance audit, not a
prerequisite for installing or starting a new evaluation after every code edit.

## Choose the affected scope

| Change | Required validation |
| --- | --- |
| New or changed scenarios/source assets | Schema, source/license/hash checks and native admission tests for those rows; refresh whole-suite uniqueness and coverage statistics. |
| Backend implementation | Regression tests and relevant existing scenarios using that backend and its shared interfaces. |
| Shared runner, scoring, event timing or safety arbitration | Tests covering the affected behavior across its consumers; widen only when the dependency scope requires it. |
| Lite selection or table export | Determinism, exact Core membership, declared coverage, exclusion reasons and reversible export; no native replay of unchanged rows. |
| Packaging or installation | Changed reader/producer contracts and one clean-consumer acceptance for the changed distribution path. |
| Documentation or unrelated release tooling | Relevant static/link/command checks, not simulator calibration. |

For data-only expansion, the existing episode cache can reuse unchanged rows
when execution dependencies are unchanged. A changed-row source suite can be
validated with the existing candidate tools; use the relevant `--stop-after`
stage rather than automatically scheduling complexity and every later stage.
Do not resume or start a stopped full pipeline merely to make its status green.

Do not weaken a cache key to claim untested code ran previously. Historical
records keep their original inputs, code identities and results. Global code
hashes remain useful provenance; they do not by themselves require a new
whole-suite experiment. A full audit is appropriate when requested or when a
broad semantic change makes the affected scope genuinely the whole suite.

## Three different identities

- Dataset qualification records describe the snapshot that was admitted.
- A new evaluation records the actual code, data, provider, configuration and
  treatment used for that run, even when its code is newer than qualification.
- Resume and merge require matching actual execution identities. Running code
  changes, altered inputs, mixed treatments and mismatched result artifacts are
  still rejected.

The verifier checks qualification artifacts against their original identities
and reports differences from live code as diagnostics. Integrity success does
not claim historical baseline scores were reproduced on the new code.
Finalization validates the actual result trees and preserves qualification
provenance separately. Historical proof bytes are never rewritten as new runs.

## Avoid redundant work

Run focused regressions after a relevant edit and review the resulting diff.
Repeat a check only if its covered path changed, it failed, or a distinct
consumer/packaging boundary remains untested. Do not repeatedly run the same
review for a cleaner banner, or treat smoke success as model performance.
Keep complete maintenance history privately; publish only the current task
data, runtime assets and compact bindings that the public readers require.
