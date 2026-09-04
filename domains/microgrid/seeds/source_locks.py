"""Scenario-level provenance locks for public microgrid data sources.

Mirrors ``domains.power_grid.seeds.source_locks`` /
``domains.logistics.seeds.source_locks`` exactly (same ``SourceLock``
dataclass + ``provenance_lock_kwargs`` helper) so the audit gate reads
microgrid provenance the same way it reads power-grid provenance. This
module does NOT import from another domain.

Important data-license notes (spec §3 / §11):

- **OEDI / NREL End-Use Load Profiles (ResStock/ComStock)** are released
  under the **NREL/OEDI data license** — an attribution-style license,
  NOT plain CC-BY-4.0. Label exactly as the NREL data license; verify
  redistribution terms before redistributing raw.
- **NSRDB** (solar irradiance) is NREL public with attribution; **NOAA
  HRRR** is US-Gov public domain.
- **pymgrid** ships a bundled 25-microgrid dataset under LGPL-3.0; it is
  dev/spike convenience ONLY and never enters the released corpus (mirrors
  the OR-Gym synthetic-demand exclusion). The **released** pymgrid-family
  provenance is the **baked NSRDB/OEDI** series.
- Any OSM-derived geometry carries an **ODbL** provenance caveat.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceLock:
    url: str
    commit: str
    lock_strategy: str
    version: str | None = None
    data_release: str | None = None
    license: str | None = None


SOURCE_LOCKS: dict[str, SourceLock] = {
    # NREL National Solar Radiation Database — GHI/DNI → PV output profiles.
    # Baked to .npz at generation time (never fetched at runtime).
    #
    # NOTE (2026): the lab was renamed National Laboratory of the Rockies (NLR)
    # and the nsrdb.nrel.gov host was retired 2026-05-29 in favour of
    # nsrdb.nlr.gov (same NSRDB physical-solar-model release, same api.data.gov
    # auth). This ``url`` is a canonical dataset *citation/identity* that is
    # hashed into every microgrid scenario body, so it is deliberately kept as
    # the historical nsrdb.nrel.gov citation to preserve frozen-release
    # byte-identity across the whole corpus. The *live fetch endpoint* migrated
    # to nsrdb.nlr.gov and lives only in the downloader
    # (scripts/download_v0_34_microgrid_raw_sources.py).
    "nsrdb": SourceLock(
        url="https://nsrdb.nrel.gov",
        commit="nsrdb-physical-solar-model-v3.2.2",
        lock_strategy="location+year+nsrdb_release",
        data_release="2020",
        license="NREL public (attribution required); parse-only, baked, never runtime",
    ),
    # NOAA High-Resolution Rapid Refresh — short-term forecast → noised
    # PV/wind forecast-error overlay. US-Gov public domain.
    "hrrr": SourceLock(
        url="https://rapidrefresh.noaa.gov/hrrr",
        commit="hrrr-v4",
        lock_strategy="cycle_date+grid_point",
        license="US-Gov public domain; parse-only, baked",
    ),
    # OEDI / NREL End-Use Load Profiles (ResStock / ComStock). NREL/OEDI
    # data license (attribution-style — NOT plain CC-BY-4.0; verify
    # redistribution terms).
    "oedi": SourceLock(
        url="https://data.openei.org/submissions/4520",
        commit="nrel-end-use-load-profiles-2021.1",
        lock_strategy="dataset_doi+release",
        data_release="2021.1",
        license=(
            "NREL/OEDI data license (attribution-style; NOT plain CC-BY-4.0 — "
            "verify redistribution terms); parse-only, baked"
        ),
    ),
    # pymgrid bundled 25-microgrid dataset (DOE/OpenEI reference buildings +
    # NREL). LGPL-3.0 (data shipped with the package). Dev/spike convenience
    # ONLY — never enters the released corpus.
    "pymgrid_bundled": SourceLock(
        url="https://github.com/Total-RD/pymgrid",
        commit="pymgrid==1.2.2",
        lock_strategy="package_version",
        version="1.2.2",
        license=(
            "LGPL-3.0 (data shipped w/ package); dev/spike convenience only — "
            "NOT released (mirrors OR-Gym synthetic-demand exclusion)"
        ),
    ),
}


def provenance_lock_kwargs(*source_ids: str) -> dict[str, str]:
    """Return common scenario provenance lock fields for one or more sources.

    Signature-identical to
    ``domains.power_grid.seeds.source_locks.provenance_lock_kwargs`` so the
    seed factories and audit code use one calling convention across domains.
    """
    locks = [SOURCE_LOCKS[source_id] for source_id in source_ids]
    out = {
        "url": " + ".join(lock.url for lock in locks),
        "commit": " + ".join(lock.commit for lock in locks),
        "lock_strategy": " + ".join(lock.lock_strategy for lock in locks),
    }
    licenses = [lk.license for lk in locks if lk.license]
    if licenses:
        out["license"] = " + ".join(licenses)
    return out
