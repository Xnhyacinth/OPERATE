# Frozen scoring inputs

Policy and schema versions identify different scoring semantics. They are
retained for validation and replay compatibility; they do not select historical
runtime datasets. The default raw policy is `lite141_028.json`. The optional
anchored scorer uses `lite141_0312.json`.

`source_inputs.json` maps original policy paths to their current locations.
Inputs with identical bytes reuse `scenarios/` or `benchmark/`; three distinct
files live in `inputs/` under their SHA-256 names. Readers enforce the original
hash and repository boundary. An existing damaged declared file cannot silently
fall back to another copy.

Original policy bytes, scenario identities and weights remain unchanged.
`offline_source_inventory.json` records original inputs and external source
requirements. Setup restores the external assets under ignored `works/`.
Offline scoring additionally requires the evaluator's authenticated journals and
provider/native evidence; runtime smoke does not create those measurements.

Model selection scores and authoring ledgers remain outside this distribution.
The fixed suite retains their original provenance digests; evaluation does not
require regenerating the model-informed selection.
