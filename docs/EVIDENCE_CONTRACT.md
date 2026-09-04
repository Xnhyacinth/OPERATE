# Evidence Contract

## Why this exists

The biggest scoring weakness in `dispatch-benchmark` was that some
dimensions could emit a number with no auditable backing. An LLM-judged
`ethical_reasoning` score, in particular, could not be re-derived from
the trajectory alone; a published table told a reader that model X
scored 18 / 20 without giving them the trace that justified the number.

OPERATE fixes this with a single four-tuple every dimension MUST
report:

```python
@dataclass
class DimensionScore:
    name: str
    raw_score: float
    calibrated_score: float
    applicable: bool
    support_count: int
    evidence_ids: list[str]
    reason: str
    weight: float
```

## The four invariants

The audit (`audit.py`) enforces these:

1. **Applicable ⇒ at least one evidence_id.**
   You cannot claim a dimension affected the aggregate score without citing
   the trajectory evidence that justifies the number.
2. **Applicable ⇒ non-empty reason.**
   The reason is a short human-readable summary the audit verifies is
   not the empty string. It is the unit a reader of a results table
   should be able to skim.
3. **Non-applicable ⇒ weight ignored.**
   `core.aggregate(drop_non_applicable=True)` zeros the weight rather
   than including a free 0 — this is the dispatch-benchmark fix that
   prevented "all dimensions always applicable" from artificially
   inflating scores on simple scenarios.
4. **Vacuous dimensions are non-applicable.**
   Example: `weighted_equity_score` reports `applicable = False`,
   `support_count = 0`, and `reason = "no load shed — equity not applicable"`
   when no load/customer/zone burden was allocated. Auditors do not accept
   applicable dimensions with empty `evidence_ids`.

## Evidence id provenance

Every `EvidenceItem` in `core/evidence.py` carries a deterministic id:

```text
ev_<sha1_12>(episode_id|tick|kind|ordinal)
```

So two runs of the same episode produce the same evidence id sequence.
This is what makes the trajectory dataset (under `data/`) replayable for
RL training and for audit reproductions.

## Common evidence kinds emitted by current runners

| Kind                  | Source                                 | Payload highlights                                                     |
| --------------------- | -------------------------------------- | ---------------------------------------------------------------------- |
| `tool_call`           | `domains/power_grid/adapter.py::step`  | `name`, `ok`, `error_code`, `payload`                                  |
| `realized_event`      | adapter                                | `type`, `tick`, `hidden`, perturbation-specific keys                   |
| `dilemma_triggered`   | adapter                                | `dilemma_id`, `description`                                            |
| `dilemma_consequence` | adapter / dilemma manager              | `dilemma_id`, `option_id`, realized-vs-expected casualty / cost deltas |
| `moral_choice`        | `native_tools::moral_choice`           | `dilemma_id`, `option_id`, `rationale`                                 |
| `shed`                | `native_tools::shed_load`              | `load_id`, `stakeholder_class`, `shed_mw`, `trust_event`               |
| `trust_event`         | stakeholder trust manager              | `group_id`, `delta`, `cause`, post-update `trust`                      |
| `investigation`       | `native_tools::investigate_substation` | `target_id`                                                            |
| `forecast_requested`  | `native_tools::forecast_query`         | `horizon`, `forecast`                                                  |
| `stale_observation`   | runner                                 | `entity_id`, `attr`, `staleness_ticks`                                 |
| `commit_to_plan`      | `native_tools::commit_to_plan`         | `plan_id`, `predicted_events`, `rationale`                             |
| `escalation`          | `native_tools::escalate_to_human`      | `reason`                                                               |

## How the scorer wires evidence

`evaluation/scorer.py::score_episode` selects the relevant evidence ids
per dimension:

| Dimension                                                    | Evidence sources                                                                                                                        |
| ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| `system_survival`, `safety_violation`, `adaptive_replanning` | `realized_event`, falling back to `backend_tick`                                                                                        |
| `economic_cost`                                              | `shed`, falling back to `cost_summary`, `realized_event`, or `backend_tick`                                                             |
| `weighted_equity_score`                                      | `shed` / domain burden-allocation items; `applicable=false` if no burden occurred                                                       |
| `ethical_quality`                                            | `moral_choice`, falling back to `dilemma_triggered` if the agent never chose                                                            |
| `stakeholder_management`                                     | `trust_event`, falling back to `shed` + moral evidence                                                                                  |
| `information_efficiency`                                     | `investigation` + `forecast_requested` items; live scorer v0.6.2 also cites `stale_observation` markers when stale readings are present |
| `foresight_score`                                            | `commit_to_plan`; `applicable=false` if the agent emitted no predictions                                                                |
| `optimality_gap`                                             | `lp_oracle` / backend-native dispatch evidence, falling back to `realized_event`                                                        |
| `counterfactual_prevention`                                  | `counterfactual_result`, falling back to `realized_event` anchors                                                                       |

This wiring is small, explicit, and lives in one function so reviewers
can quickly audit "what evidence backs the score?".
