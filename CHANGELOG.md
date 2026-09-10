# Updates

## 2026-09-10

- Live scoring is `0.17.0` (`wait_relative_outcome_v1`). Catalog qualification
  remains `0.15.0` and is not rewritten.
- Live realtime uses `native_dt_v1`: wall ticks equal each row's native plant
  quantum. The default speed card is the 37 Core rows in
  `benchmark/realtime_speed_suite.json`.
- Core remains 769 rows and Lite remains 193 rows. Membership is not driven by
  model scores.

## 2026-09-07

- Public GitHub ships one current catalog under `benchmark/` and flattened
  `scenarios/<domain>/...` paths. Maintainer mining scripts, versioned lock
  folders, and `release/` trees are not part of this checkout.
- Clone plus `bash scripts/setup_eval_env.sh` installs the public Hugging Face
  runtime companion without overlaying Git catalogs.
- Public runbook examples use `API_KEY` and `BASE_URL` only. Request `--max-tokens` equals advertised output: Hy3 192k/64k, Luna 272k/128k, GLM-5.3-Flash 1M/128k, DeepSeek Flash 1M/50k.

- Record native per-route envelopes in `docs/provider_route_bindings.json`. Formal Hy3 examples use a 192,000-token envelope and 64,000-token output; Luna's request envelope is 272,000 when that route is used.

## 2026-09-05

- Include every admitted Datacenter medium/high case in Lite: 11 medium, 7 high and the existing 9 basic cases. Lite expands from 184 to 193 rows; other domains, Full and runtime assets are unchanged.
- Retain every admitted scenario in Autonomous Driving, Building Energy, Microgrid, Power Grid and Traffic. Lite expands from 154 to 184 rows (122 physical sources), preserving the existing Logistics/Datacenter selections; Full and runtime assets are unchanged.
- Replace four JSPLIB scheduling scenarios (swv14, swv15, yn1, yn2) with source-preserving refinements that expose procedural disruptions before scheduling can finish. Full remains 769 scenarios from 502 physical sources; the old scenario files are removed from the public current tree.
- Fit persistent observations and structured memory to the complete provider request budget, reserving mission, event, memory identity and output space. Strict prompt failures cannot become model-selected waits.
- Resolve portable and legacy trajectory paths consistently for reporting, artifact verification and resume. Separate quota availability, execution completion and observed task success.
- Link score citations to proven native effects and preserve hidden-event visibility and zero-tick requests.
- Prevent command-line overrides of the fixed Lite scope; propagate installation and smoke failures. Scope SUMO leak checks to the owning episode process group.
- Carry original NGSIM checksum lists and the M5 source lock with runtime assets, so a fresh download retains native source-verification metadata.
- Separate passive-baseline terminal warnings from installation failures in an explicit installation-only profile; preserve strict diagnostics and the original warning evidence.
- Start the locked SUMO wheel's native executable directly so failed TraCI handshakes cannot strand a child behind its console launcher; verify the executable during setup.
- Document Lite compression parameters as engineering heuristics, distinct from Core quality admission. Keep historical candidate and private publication tools out of the public current tree.
- Separate immutable dataset qualification from actual new-run identity; preserve result, source, resume and merge checks without requiring full replay after every maintenance edit. Dry-run can inspect an uncommitted tree without authorizing execution.
- Replace Lite domain/stratum quotas with coverage plus independent-source enrichment: 154 rows and 122 physical sources within the explicit 150–200 development budget.
- Publish 22-column public Parquet with shared suite metadata stored once; retain complete private metadata and exact v1/v2 reconstruction.
- Regenerate Full/Lite Parquet, runtime companion and release evidence for the current code and data. Published artifact filenames remain versionless; immutable commits and content hashes identify reproducible snapshots.

Full provider evaluation is still incomplete. A successfully installed benchmark or an admitted scenario is not a completed model evaluation. Results from older implementation or scenario hashes cannot be resumed into this snapshot.
