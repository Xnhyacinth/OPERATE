# Data provenance

OPERATE admits a scenario only when its initial state and exogenous series are
bound to real public source material or an explicitly labelled deterministic
stress overlay. An LLM never generates environment state.

## Active closure

The frozen `operate_v0_61_0` candidate closure consists of:

- `release/operate_v0_61_0/candidate_closure.json` — 2,476 terminal candidate
  dispositions with zero unresolved candidates;
- `release/operate_v0_61_0/protocol21_source_suite.json` — the closed 769-row
  replay and provenance suite spanning 502 physical sources;
- 747 inherited canonical contracts under `scenarios/operate_v0_58_0/`, 8
  additions under `scenarios/operate_v0_59_0/`, and 13 additions under
  `scenarios/operate_v0_60_0/`, plus 1 under
  `scenarios/operate_v0_61_0/`, exactly as selected by the v0.61 manifest;
- `sources/locks/` — tracked CityLearn source locks;
- `sources/alibaba/` — compact trace inputs used by released Datacenter rows;
- `sources/dynasched/` — the released DynaSched instance and event bundle;
- `sources/resco/` — the compact RESCO traffic source introduced in v0.61;
- manifest-declared upstream assets under `works/` for native local replay.

All 2,476 independent candidates have terminal dispositions and none remain
unresolved. That closure records selection decisions; it does not by itself
promote the selected rows.

Every source row binds a scenario ID, scenario signature, seed, backend, source
denominator, structural fingerprint, semantic fingerprint, and physical source
graph. The preflight recomputes scenario signatures and exercises each native
source adapter before behavioral replay.

## Source consumption

Provenance metadata alone is insufficient. Admission requires runtime evidence
that the declared source values drive backend state transitions. The replay
records source observations, typed perturbations, tool-visible state, actions,
receipts, effects, and counterfactual no-action outcomes.

Procedural stress is permitted only when it is seeded, typed, replayable,
material to the operational task, and labelled separately from source-observed
events. Unknown event types are non-actionable by default.

## Physical independence

`n_effective_sources` counts the evaluation denominator. Physical-source keys
and source-denominator keys expose shared topology, trace, feeder, instance, or
time-series ancestry so reports can publish backend/domain/source strata rather
than treating correlated rows as independent measurements.

## Distribution boundary

OPERATE code and benchmark-authored metadata are MIT-licensed. Upstream data
and simulators retain their own licenses and acquisition terms; the readable
index is [`THIRD_PARTY_LICENSES.md`](../THIRD_PARTY_LICENSES.md). The public HF
bundle includes redistributable assets and hash manifests, including the three
byte-exact M5 source tables authorized for academic redistribution, the 16
Core-required derived NREL/OEDI microgrid NPZ profiles, and their source
identifiers and attribution metadata. Assets that remain upstream-fetched are
resolved from their declared URL/revision/hash by the one-command installer
instead of being relabelled or silently mirrored.

The source suite is the bound replay and provenance input, not the formal
denominator. The promoted
`core_suite.json`, release `manifest.json`, and HF `MANIFEST.json` are the
authoritative evaluation and distribution closure. A clean download must verify
every file before installing runtime archives or beginning a formal provider
run. The compact v0.61 companion must include `candidate_closure.json`. A full
candidate-evidence archive of the 17 input ledgers is restored only when the
bundle declares `candidate_evidence_archive`; the current compact companion
omits that archive.
