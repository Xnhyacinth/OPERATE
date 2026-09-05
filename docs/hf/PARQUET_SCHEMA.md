# Full and Lite Parquet schema

The Hugging Face export contains two configurations:

- `full/test-00000-of-00001.parquet` is the official complete evaluation
  denominator selected by `core_suite.json`.
- `lite/test-00000-of-00001.parquet` is the deterministic, diversity-oriented
  development subset selected by `lite_suite.json`. Lite results are not Full
  leaderboard results.

Both files use schema `operate-hf-parquet-v1`. Columns such as `domain`,
`backend_kind`, `family`, `difficulty_level`, `horizon_ticks`, and
`physical_source_key` support filtering in the dataset viewer. The following
columns preserve the executable contract:

- `scenario_yaml` contains the exact UTF-8 scenario contract.
- `scenario_metadata_json` contains the complete corresponding suite row.
- `suite_template_json` contains every suite-level field and its original key
  order, with a marker for the ordered scenario rows.
- `row_index`, `suite_file_sha256`, and `yaml_sha256` bind ordering and bytes.

Build the two files and their integrity manifest with:

```bash
uv run --with pyarrow python tools/build_hf_parquet.py export \
  --output-dir build/hf-parquet
```

Reconstruct a suite JSON file and its scenario tree from one Parquet file:

```bash
uv run --with pyarrow python tools/build_hf_parquet.py rebuild \
  --parquet build/hf-parquet/full/test-00000-of-00001.parquet \
  --output-dir build/reconstructed-full
```

Reconstruction fails closed on duplicate row indices, inconsistent flat and
JSON metadata, unsafe paths, changed YAML, or a suite hash mismatch. With the
published input, the reconstructed JSON and YAML are byte-identical to their
source files. Byte-identical Parquet regeneration additionally requires the
same PyArrow version; semantic reconstruction is version-independent.
