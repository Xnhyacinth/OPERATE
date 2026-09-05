# OPERATE command surface

The maintained command groups are:

- `run_protocol21_core_pipeline.py`: replay and qualify the current Core;
- `promote_operate_release.py` and `verify_release_integrity.py`: build and
  verify `operate_v0_61_0`;
- `batch_llm_eval.py`: formal `logical_persistent` evaluation;
- `batch_realtime_llm_eval.py`: independent `realtime_persistent` scorecard;
- `merge_formal_llm_shards.py`: merge compatible complete shards;
- `download_from_hf.py`: anonymous, manifest-backed public bundle download;
- `setup_eval_env.sh`: reproducible environment, runtime-companion, and backend setup.

Private CAS publication tools remain in the private maintainer repository;
they are not needed to download the public dataset or evaluate a model.

Candidate-generation and historical-release utilities are not part of the
supported release surface and are removed once the corresponding source rows
are frozen.
