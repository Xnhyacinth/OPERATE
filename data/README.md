# OPERATE data

The current public catalog contains 769 Core rows over 502 physical sources
and 144 Lite rows over 85 sources. Scenario contracts live under
`scenarios/<domain>/...`. Compact redistributable source assets live under
`sources/`. `benchmark/core_suite.json` plus `benchmark/manifest.json` define
the evaluation denominator.

The public Hugging Face repository is `Xnhyacinth/OPERATE`. A published bundle
is authoritative only when its `MANIFEST.json` passes integrity verification.
The downloader resolves the public snapshot to an immutable HF commit and
records it in `operate_data/.operate-bundle-owner.json`. Use that recorded
commit, not a mutable branch name, as the runtime binding.

## Reproduce the environment

```bash
python -m pip install uv==0.12.5
uv sync --frozen --python 3.13 --extra released-backends --extra llm --extra hf
```

Python 3.10–3.14 is supported by the framework. Use Python 3.13 for the complete
Core because CityLearn 2.5.0 does not currently support Python 3.14.

## Download and verify the runtime companion

```bash
uv run python scripts/download_from_hf.py --download-only
```

`operate_data/` is the local install root. `MANIFEST.json` binds those bytes.
Anonymous download does not require `HF_TOKEN`.

For a complete native environment and a per-backend smoke:

```bash
bash scripts/setup_eval_env.sh --smoke
```
