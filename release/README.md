# Release artifacts

This directory exposes only the current formal benchmark state. The internal
`operate_v0_62_0` identifier is a reproducibility binding, not a selectable
public version.

```text
release/operate_v0_62_0/          current Core, manifest, and candidate closure
scenarios/operate_v0_58_0/        48 inherited contracts selected by the current manifest
scenarios/operate_v0_59_0/        8 inherited scenario additions
scenarios/operate_v0_60_0/        11 retained v0.60 contracts
scenarios/operate_v0_61_0/        1 retained v0.61 contract
scenarios/operate_v0_62_0/        701 corrected v0.62 contracts
release/operate_v0_62_0/lite_suite.json  policy-derived 193-row, 122-source efficiency/development track
```

Local generated trees such as `release/operate_v0_58_0_candidate/` are
gitignored. Do not resume provider results from a historical namespace.

The local runtime-companion install root is `operate_data/`. Its
`MANIFEST.json` binds the bytes
to `operate_v0_62_0`. The downloader records the resolved immutable public HF
commit in the local owner receipt.

The historical `data_operate_v058/` compatibility path remains available for
inherited runtime assets.

The historical v0.58 package snapshot remains available only as a compatibility
reference.

Scoring is `0.15.0`; Core retains 769 rows across 502 physical sources.
The parent v0.61 admission ledger records 2,476 terminal candidate decisions
with zero unresolved. v0.62 introduces zero newly mined candidates; it preserves
that historical lineage while qualifying corrected contracts for the same
769-row, 502-source denominator. Historical admission evidence is not relabelled
as newly executed evidence.
Formal provider runs remain pending; public result release and leaderboard
eligibility remain false.
