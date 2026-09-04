# OPERATE command surface

The maintained command groups are:

- `run_protocol21_core_pipeline.py`: replay and qualify the current Core;
- `promote_operate_release.py` and `verify_release_integrity.py`: build and
  verify `operate_v0_61_0`;
- `batch_llm_eval.py`: formal `logical_persistent` evaluation;
- `batch_realtime_llm_eval.py`: independent `realtime_persistent` scorecard;
- `merge_formal_llm_shards.py`: merge compatible complete shards;
- `download_from_hf.py`: anonymous, manifest-backed public bundle download;
- `upload_to_hf.py`: maintainer-only private CAS publication for completed
  formal-result bundles, not the public dataset bootstrap path;
- `setup_eval_env.sh`: reproducible environment, runtime-companion, and backend setup.

Candidate-generation and historical-release utilities are not part of the
supported release surface and are removed once the corresponding source rows
are frozen.
