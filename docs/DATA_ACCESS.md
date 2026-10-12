# Data and API configuration

Dataset locations, provider endpoints and credentials are supplied locally.
The public files contain no maintainer account or institution-specific API
configuration.

## Runtime dataset

Set the dataset id supplied with your distribution before running setup:

```bash
export OPERATE_HF_REPO_ID='<dataset-owner>/<dataset-name>'
bash scripts/setup_eval_env.sh --smoke
```

The standalone downloader and bundle builder also accept `--repo-id owner/name`;
that argument takes precedence over `OPERATE_HF_REPO_ID`. Missing configuration
is rejected before downloads or setup writes. `--skip-data` installs dependencies
without requiring a dataset id. Public downloads do not require `HF_TOKEN`.

For exact reproduction, set `OPERATE_HF_REVISION` to the dataset commit recorded
in the distribution manifest. Download validation still binds the requested
repository, public visibility, immutable revision and every manifest file hash.

Aggregate result paths in the documentation are relative to the configured
runtime dataset. Trajectory archive paths refer to a separate dataset whose id
is supplied as `OPERATE_TRAJECTORIES_REPO_ID`; use that id with the Hugging Face
CLI or dataset browser. Restricted original evidence is supplied separately to
authorized users. No private archive location is embedded in this repository.

## Model API

```bash
export API_KEY='<your-api-key>'
export BASE_URL='<provider-base-url>'
```

Single-run and batch evaluation use these names by default. Azure transports
can additionally use `OPERATE_API_VERSION` and `OPERATE_RESPONSES_API_BASE_URL`. Explicit endpoint
arguments and `--api-key-env` / `--base-url-env` overrides remain supported.
Store actual values in the process environment or an ignored local configuration
file. Existing installations should migrate their exports to these generic names.

Provider identifiers describe SDK transport protocols. Route bindings retain
model budgets and request settings; endpoint locations come from `BASE_URL`.
Anonymous deployment labels preserve the pairing of reported numerical results.
Supply the actual model route and any tool-choice override locally when evaluating
that deployment.

Third-party dataset attribution, upstream source locks, benchmark contracts and
licenses retain their original identifiers.
