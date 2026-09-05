# Release artifacts

This directory exposes only the current formal benchmark state. The internal
`operate_v0_61_0` identifier is a reproducibility binding, not a selectable
public version.

```text
release/operate_v0_61_0/          current Core, manifest, and candidate closure
scenarios/operate_v0_58_0/        743 inherited contracts selected by the current manifest
scenarios/operate_v0_59_0/        8 inherited scenario additions
scenarios/operate_v0_60_0/        13 inherited scenario additions
scenarios/operate_v0_61_0/        5 current scenario additions
release/operate_v0_61_0/lite_suite.json  policy-derived 159-row efficiency/development track
```

Local generated trees such as `release/operate_v0_58_0_candidate/` are
gitignored. Do not resume provider results from a historical namespace.

The local runtime-companion install root is `operate_data/`. Its
`MANIFEST.json` binds the bytes
to `operate_v0_61_0`. The downloader records the resolved immutable public HF
commit in the local owner receipt.
