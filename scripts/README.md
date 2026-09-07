# OPERATE command surface

The public command groups are:

- `setup_eval_env.sh`: locked environment, runtime companion, and backend setup;
- `download_from_hf.py`: anonymous, manifest-backed public bundle download;
- `verify_release_integrity.py`: verify `benchmark/` catalogs and scenario YAML;
- `run_full.py`: current 769-row Core / Full evaluation;
- `run_lite.py`: current 193-row Lite evaluation;
- `batch_llm_eval.py`: `logical_persistent` evaluation engine;
- `batch_realtime_llm_eval.py`: independent `realtime_persistent` scorecard;
- `merge_formal_llm_shards.py`: merge compatible complete shards;
- `run_protocol21_diagnostic_smoke.py`: per-backend installation smoke.

Candidate-mining, promotion, and historical-release utilities are not part of
this public tree.
