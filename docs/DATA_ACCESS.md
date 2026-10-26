# Data and API configuration

## Runtime data

```bash
export OPERATE_HF_REPO_ID='Xnhyacinth/OPERATE'
bash scripts/setup_eval_env.sh --smoke
```

The downloader also accepts `--repo-id owner/name`, which takes precedence over
`OPERATE_HF_REPO_ID`. Public downloads require no `HF_TOKEN`. Missing dataset
configuration is rejected before download or setup writes; `--skip-data` installs
dependencies only.

The resolved immutable revision is recorded in
`operate_data/.operate-bundle-owner.json`. Set `OPERATE_HF_REVISION` to that
commit for a pinned reproduction. The downloader verifies repository identity,
visibility, revision and manifest file hashes.

## Model API

```bash
export API_KEY='<your-api-key>'
export BASE_URL='<provider-base-url>'
```

These are the default variable names for single and batch runs.
`--api-key-env` and `--base-url-env` select different variable names when needed.
Azure transports can also use `OPERATE_API_VERSION` and
`OPERATE_RESPONSES_API_BASE_URL`. Keep real values in the process environment or
ignored local configuration.

Pass the actual model ID, advertised context/output limits and supported request
settings for your provider. Deployment endpoints and account quotas are local
configuration; the benchmark does not ship a maintainer deployment inventory.

Third-party source names, licenses, scientific identifiers and integrity hashes
retain their original meaning. See [third-party terms](../THIRD_PARTY_LICENSES.md).
