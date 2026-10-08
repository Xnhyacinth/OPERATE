# Lite141 retrospective analysis

The offline analysis layer explains recorded outcomes without changing scoring,
source weights, episode selection, or the formal run identity. Its default scope
is the nine recorded The streaming gateway models and GPT6.1 Sol, GPT6 Sol, and GPT6 Luna.
Scoring 0.30 is the analysis endpoint; original 0.28 values are retained for
sensitivity reporting. This is a fixed, model-informed panel.

## Inputs and reproduction

Use the repository Python environment, with the optional `analysis` extra for
matplotlib. The lockfile already contains its platform-specific versions.
Generated output belongs outside tracked release and source directories.

```bash
uv sync --frozen --extra dev --extra analysis
uv run --frozen --no-sync python scripts/build_analysis_bundle.py \
  --result-dir PATH_TO_EXISTING_030_PACKAGE \
  --output-dir .hl/analysis/NEW_ANALYSIS/main
uv run --frozen --no-sync python scripts/plot_analysis_bundle.py \
  --bundle .hl/analysis/NEW_ANALYSIS/main/bundle.json \
  --supplementary PATH_TO_REVALIDATED_SUPPLEMENTARY_DIRECTORY \
  --output-dir .hl/analysis/NEW_ANALYSIS/figures
```

Keep already-needed backend extras when synchronizing an existing simulation
environment. Offline analysis itself does not construct a simulator or call a
provider. Both commands refuse existing output directories.

The first entry accepts an existing checksum-bound 0.30 package as *analysis*
input. It is not a substitute input for the public scorer, which still requires
original evidence. Package checksums are integrity records, not signatures or
independent authority. Before analysis, the reader verifies selected original
journals, configuration files, canonical whole episodes, scenario bytes, exact
case membership and frozen weights, then independently reconstructs weighted Q.
Referenced behavioral artifacts are separately hash-checked; their unavailable
measurements remain unknown. The package and original bound files are checked
again to detect concurrent changes.

## Outputs

- `bundle.json`: model/case outcomes, behavioral metrics, native time series,
  lifecycle observations, coverage issues, strata and pairwise differences.
- CSV exports: machine-readable Source Data, including original provenance.
- `checksums.json`: analysis output inventory.
- Figure directory: editable PDF/SVG, PNG previews, per-figure Source Data and
  `figure_manifest.json` with selection rules and input hashes.

Native cost units are compared within matched cases. Group scores renormalize
only the original within-group weights; they do not replace the main index.
Sensitivity ranks retain the original complete-model cohort, not a re-ranking
of the selected analysis models. Time examples are descriptive and retain all
extracted observations in the Source Data.

## Evidence and interpretation

- Missing tokens, native effects and opportunities do not become zero. Provider
  usage must cover every recorded response for an episode token total.
- Successful calls are not total inference cost. Failed calls and protocol
  repairs are reported separately.
- `proven_engine_effect_calls` is an observed lower bound. Complete effect counts
  require all calls to be resolved; successful receipts alone do not prove an
  engine effect. Unidentified calls are explicitly counted as record gaps.
- Time series come from engine backend-tick evidence, not model-visible state.
  No terminal score formula is applied to a partial trajectory.
- Historical E1 scores retain their historical versions. E3 is an independent
  realtime treatment, and E4 has native/process outcomes rather than a Lite Q.
- Strict supplementary certification means the recorded comparison passes its
  current validator. It does not certify repeatability, cross-source validity,
  or model behavior under repaired historical feedback.
- Provider/harness/backend errors and unavailable evidence remain separate from
  measured model failures. An unstarted proposed experiment is not a failed run.

No new runs are implied by analysis gaps. Additional sources, repeated trials,
proactivity controls or GPT mechanism comparisons should be scheduled only for
specific claims unsupported by the existing evidence.
