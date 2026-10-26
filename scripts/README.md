# Commands

| Command | Purpose |
| --- | --- |
| `setup_eval_env.sh` | Install the locked environment and runtime data |
| `download_from_hf.py` | Download and verify the public runtime companion |
| `verify_release_integrity.py benchmark` | Check catalogs and scenario hashes |
| `check_runtime.py` | Check installation with per-backend episodes |
| `../run_lite.py`, `../run_full.py` | Run the fixed Lite or Full track |
| `batch_llm_eval.py` | Logical evaluation engine |
| `batch_realtime_llm_eval.py` | Independent realtime evaluation engine |
| `merge_formal_llm_shards.py` | Merge compatible completed shards |
| `recover_completed_episode.py` | Recover postprocessing from recorded inputs |
| `evaluate_trajectories.py` | Score authenticated raw trajectories |
| `evaluate_anchored_outcomes.py` | Apply reference-anchored scoring |

Use `uv run --frozen --no-sync python scripts/<command>.py --help` for options.
See the [evaluation runbook](../docs/FORMAL_EVALUATION.md) for model configuration
and [scoring contract](../docs/EVALUATION.md) for choosing a score.
