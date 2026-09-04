"""Scenario-level provenance locks for public logistics / VRP data sources.

Mirrors ``domains.power_grid.seeds.source_locks`` exactly (same
``SourceLock`` dataclass + ``provenance_lock_kwargs`` helper) so the audit
gate reads logistics provenance the same way it reads power-grid
provenance. This module does NOT import from ``domains.power_grid``.

Important data-license note (verified per set, see spec §3 / §11 and the
MATPOWER-case lesson in ``.hl/failed_directions.md``): a permissive
*reader* (PyVRP / VRPLIB = MIT) does not make the *instance data*
permissive. CVRPLIB / Solomon / Augerat carry **no formal OSS license
(research-use only)** and are released as parsed structural seeds with
provenance, never redistributed raw; the MIT-licensed ``PyVRP/Instances``
mirror is the preferred released-artifact source. Amazon LMRRC is
**CC-BY-NC-4.0 (NonCommercial)** → its family is excluded from any
commercial leaderboard.
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
    # MIT-mirrored VRPLIB instance set (Augerat/Uchoa CVRP + Solomon/
    # Gehring-Homberger VRPTW). The MIRROR is MIT; the underlying classic
    # instances are research-use-only, hence released only as parsed seeds.
    "vrplib": SourceLock(
        url="https://github.com/PyVRP/Instances",
        commit="1cf23a5969fabf23c80f8002e42ed501a47aca61",
        lock_strategy="git_commit",
        license=(
            "mirror=MIT (PyVRP/Instances); instances=research-use-only "
            "(CVRPLIB/Solomon/Augerat/Uchoa — released as parsed seeds, "
            "not redistributed raw)"
        ),
    ),
    "vrplib_package_test_data": SourceLock(
        url="https://github.com/PyVRP/VRPLIB",
        commit="309f18900f54a91ec3aeaf8b840c0db351aa8c5d",
        lock_strategy="git_commit+file_sha256",
        license="MIT",
    ),
    "vrplib_package_solomon": SourceLock(
        url="https://github.com/PyVRP/VRPLIB",
        commit="309f18900f54a91ec3aeaf8b840c0db351aa8c5d",
        lock_strategy="git_commit+file_sha256",
        license="MIT",
    ),
    "vrplib_package_x_set": SourceLock(
        url="https://github.com/PyVRP/VRPLIB",
        commit="309f18900f54a91ec3aeaf8b840c0db351aa8c5d",
        lock_strategy="git_commit+file_sha256",
        license=(
            "reader/repo=MIT (PyVRP/VRPLIB); Uchoa X-set instances="
            "research-use/no formal OSS license; parsed seeds only, raw files "
            "not redistributed"
        ),
    ),
    "vrplib_package_lkh_cvrptw": SourceLock(
        url="https://github.com/PyVRP/VRPLIB",
        commit="309f18900f54a91ec3aeaf8b840c0db351aa8c5d",
        lock_strategy="git_commit+file_sha256",
        license="MIT repository test data",
    ),
    "vrplib_package_lkh_cvrp": SourceLock(
        url="https://github.com/PyVRP/VRPLIB",
        commit="309f18900f54a91ec3aeaf8b840c0db351aa8c5d",
        lock_strategy="git_commit+file_sha256",
        license="MIT repository test data",
    ),
    "vrplib_package_cvrplib_root": SourceLock(
        url="https://github.com/PyVRP/VRPLIB",
        commit="309f18900f54a91ec3aeaf8b840c0db351aa8c5d",
        lock_strategy="git_commit+file_sha256",
        license="MIT repository test data",
    ),
    # CVRPLIB canonical source (Augerat A/B/P + classic CVRP). Recorded for
    # provenance completeness; the MIT mirror above is preferred for
    # released artifacts.
    "cvrplib": SourceLock(
        url="http://vrp.galgos.inf.puc-rio.br/",
        commit="cvrplib-web",
        lock_strategy="url+access_date",
        data_release="2024-access",
        license="research-use only (no formal OSS license)",
    ),
    # Amazon Last-Mile Routing Research Challenge — real driver routes +
    # package/time-window/zone data. CC-BY-NC-4.0 (NonCommercial).
    "amazon_lmrrc": SourceLock(
        url="https://registry.opendata.aws/amazon-last-mile-challenges",
        commit="lmrrc-2021",
        lock_strategy="aws_open_data_registry+challenge_release",
        data_release="2021",
        license="CC-BY-NC-4.0 (NonCommercial)",
    ),
    # JSPLIB job-shop instances (Taillard/Lawrence/FT/ABZ/... classic OR
    # benchmarks). The current repo carries only a tiny anchored sample under
    # works/ for non-release adapter/scoring design.
    "jsplib": SourceLock(
        url="https://github.com/tamy0612/JSPLIB",
        commit="eea2b60dd7e2f5c907ff7302662c61812eb7efdf",
        lock_strategy="git_commit+file_sha256",
        license="public academic OR benchmark mirror; see works/JSPLIB-Instances/LICENSE",
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
