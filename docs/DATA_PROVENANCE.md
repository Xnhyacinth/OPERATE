# Data provenance

OPERATE admits a scenario only when its initial state and exogenous series are
bound to real public source material or an explicitly labelled deterministic
stress overlay. An LLM never generates environment state.

The current public catalog is 769 Core rows over 502 physical sources and 193
Lite rows over 122 sources.

## Active catalog

- `benchmark/core_suite.json` — current 769-row Core catalog
- `benchmark/lite_suite.json` — current 193-row Lite catalog
- `benchmark/manifest.json` — public counts, scoring version, and file hashes
- `sources/locks/` — tracked source locks
- `sources/alibaba/` — compact trace inputs used by released Datacenter rows
- `sources/dynasched/` — the released DynaSched instance and event bundle
- `sources/resco/` — compact RESCO traffic sources
- upstream assets under `works/` restored by the installer

Every source row binds a scenario ID, scenario signature, seed, backend, source
denominator, structural fingerprint, semantic fingerprint, and physical source
graph.

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
denominator. `benchmark/core_suite.json`, `benchmark/manifest.json`, and HF
`MANIFEST.json` are the authoritative evaluation and distribution catalogs. A
clean download must verify every declared runtime file before installing
archives or beginning a provider run.
