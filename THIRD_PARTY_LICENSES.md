# Third-party data and software

The root [LICENSE](LICENSE) applies only to OPERATE-authored code and metadata.
Source datasets, simulators, and runtime packages retain their upstream terms.
The machine-readable file lists, revisions, checksums, roles, and notices in the
current release and Hugging Face `MANIFEST.json` are authoritative; this page is
the human-readable index.

| Source or runtime | Terms and redistribution boundary |
| --- | --- |
| M5 Forecasting Accuracy | M5 Competition Rules. OPERATE maintainers accepted the rules and confirmed explicit permission for academic redistribution on 2026-09-04. The public runtime companion preserves the original file hashes and source attribution; downstream use remains subject to the M5 terms. |
| OR-Gym | MIT; used as the native inventory-control environment for M5 demand streams. |
| Alibaba cluster trace | Upstream repository and research-trace terms; files not carried by the companion are fetched from the manifest-pinned commit and hash-checked. |
| CityLearn | Upstream code and dataset terms; the installer resolves the manifest-pinned runtime. |
| DynaSchedBench | Upstream repository license and bundled license notice. |
| JSPLIB and REALM-J2 | Their respective upstream data terms; the redistributed REALM-J2 source retains CC BY 4.0 attribution. |
| Grid2Op | Mozilla Public License 2.0. |
| pandapower | BSD 3-Clause. |
| PGLib-OPF and PGLib-UC | Upstream software and CC BY 4.0 data terms. |
| RTS-GMLC and NREL/OEDI-derived profiles | Applicable NREL, DOE, OEDI, NSRDB, OpenEI, and attribution terms recorded in scenario provenance and the runtime manifest. |
| OpenDSS / DSS-Python | Upstream software and feeder-data terms. |
| SUMO and RESCO | Eclipse Public License and the applicable upstream scenario/source terms. |
| NGSIM US-101 | Source release `doi:10.21949/1504477`, redistributed under CC BY-SA 4.0 with recording and source identifiers preserved. |
| PyVRP and VRPLIB | Their respective upstream software and instance-data terms. |

No third-party asset is relicensed under OPERATE's MIT license. When publishing
derived results, cite both OPERATE and the upstream sources named by the scenario
provenance records.
