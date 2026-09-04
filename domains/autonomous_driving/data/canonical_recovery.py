"""Fail-closed recovery metadata for the fifteen historical NGSIM candidates.

The registry preserves source identities and deterministic derivation recipes
from the last pre-OPERATE snapshot.  It is not admission evidence: source
archives, rebuilt bundles, and fresh native replay must still be verified.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

from .contracts import (
    NGSIM_DATASET_ID,
    NGSIM_DOI,
    NGSIM_METADATA_URL,
    NGSIM_US101_ARCHIVE_SHA256,
    NGSIM_US101_ARCHIVE_URL,
    NGSIM_US101_ASSET_ID,
    NGSIM_US101_AUTHORITATIVE_MEMBER,
    NGSIM_US101_AUTHORITATIVE_SHA256,
)

HISTORICAL_RECIPE_COMMIT = "5b4d692e39ac1ed990c5a26f53ede1c9860f5f21"
CANONICAL_RECIPE_VERSION = "ngsim_phase_complete_window_v1"


@dataclass(frozen=True)
class NGSIMArchiveRequirement:
    recording_id: str
    asset_id: str
    archive_name: str
    archive_url: str
    archive_sha256: str | None
    authoritative_members: tuple[str, ...]
    authoritative_member_sha256: str | None
    provenance_status: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["authoritative_members"] = list(self.authoritative_members)
        hash_lock_complete = bool(
            self.archive_sha256 and self.authoritative_member_sha256
        )
        value.update(
            {
                "source_dataset_id": NGSIM_DATASET_ID,
                "source_release": f"doi:{NGSIM_DOI}",
                "source_metadata_url": NGSIM_METADATA_URL,
                "archive_url_sha256": hashlib.sha256(
                    self.archive_url.encode("utf-8")
                ).hexdigest(),
                "hash_lock_complete": hash_lock_complete,
                "automatic_download_allowed": hash_lock_complete,
            }
        )
        return value


@dataclass(frozen=True)
class NGSIMCanonicalCandidate:
    candidate_id: str
    recording_id: str
    hazard_kind: str
    event_time_ms: int
    window_end_time_ms_exclusive: int
    ego_actor_id: str
    bundle_id: str
    source_window_sha256: str
    source_event_chain_sha256: str
    source_evidence_sha256: str
    source_event_sequence: tuple[str, ...]
    derivation_file_name: str | None
    derivation_file_sha256: str | None
    recipe_version: str | None
    historical_stage: str

    @property
    def metadata_recipe_complete(self) -> bool:
        return bool(
            self.recipe_version == CANONICAL_RECIPE_VERSION
            and self.derivation_file_name
            and self.derivation_file_sha256
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_event_sequence"] = list(self.source_event_sequence)
        value["metadata_recipe_complete"] = self.metadata_recipe_complete
        slug = self.candidate_id.replace(":", "_", 1).replace(":", "_")
        base = f"scenarios/staging/{self.historical_stage}/{slug}"
        value["historical_commit"] = HISTORICAL_RECIPE_COMMIT
        value["historical_candidate_report"] = f"{base}/candidate_report.json"
        return value


def _asset_url(asset_id: str, archive_name: str) -> str:
    return (
        f"https://data.transportation.gov/api/views/{NGSIM_DATASET_ID}/files/"
        f"{asset_id}?download=true&filename={archive_name}"
    )


# The I-80 and Peachtree attachment identifiers and member name candidates are
# recoverable from the official USDOT attachment links and metadata documents.
# Their archive/member hashes were never retained in the repository.  They
# deliberately remain null so a download cannot be mistaken for a source lock.
OFFICIAL_ARCHIVE_REQUIREMENTS: dict[str, NGSIMArchiveRequirement] = {
    "us-101": NGSIMArchiveRequirement(
        recording_id="us-101",
        asset_id=NGSIM_US101_ASSET_ID,
        archive_name="US-101-LosAngeles-CA.zip",
        archive_url=NGSIM_US101_ARCHIVE_URL,
        archive_sha256=NGSIM_US101_ARCHIVE_SHA256,
        authoritative_members=(NGSIM_US101_AUTHORITATIVE_MEMBER,),
        authoritative_member_sha256=NGSIM_US101_AUTHORITATIVE_SHA256,
        provenance_status="hash_locked",
    ),
    "i-80": NGSIMArchiveRequirement(
        recording_id="i-80",
        asset_id="ea269540-b86c-4b2d-a9c2-c8f4c0a3d0a0",
        archive_name="I-80-Emeryville-CA.zip",
        archive_url=_asset_url(
            "ea269540-b86c-4b2d-a9c2-c8f4c0a3d0a0",
            "I-80-Emeryville-CA.zip",
        ),
        archive_sha256=None,
        authoritative_members=(
            "trajectories-0400-0415.txt",
            "trajectories-400-0415.txt",
        ),
        authoritative_member_sha256=None,
        provenance_status="official_locator_hashes_missing",
    ),
    "peachtree": NGSIMArchiveRequirement(
        recording_id="peachtree",
        asset_id="3dba3db1-dd9a-46b3-96d0-07d8c4461feb",
        archive_name="Peachtree-Street-Atlanta-GA.zip",
        archive_url=_asset_url(
            "3dba3db1-dd9a-46b3-96d0-07d8c4461feb",
            "Peachtree-Street-Atlanta-GA.zip",
        ),
        archive_sha256=None,
        authoritative_members=(
            "trajectories-1245pm-0100pm.txt",
            "trajectories-0400pm-0415pm.txt",
        ),
        authoritative_member_sha256=None,
        provenance_status="official_locator_hashes_missing",
    ),
}


def _candidate(
    candidate_id: str,
    recording_id: str,
    hazard_kind: str,
    event_time_ms: int,
    window_end_time_ms_exclusive: int,
    ego_actor_id: str,
    bundle_id: str,
    source_window_sha256: str,
    source_event_chain_sha256: str,
    source_evidence_sha256: str,
    source_event_sequence: tuple[str, ...],
    derivation_file_name: str | None,
    derivation_file_sha256: str | None,
    *,
    historical_stage: str = "autonomous_driving_multisite_v33_commonroad_v2",
) -> NGSIMCanonicalCandidate:
    return NGSIMCanonicalCandidate(
        candidate_id=candidate_id,
        recording_id=recording_id,
        hazard_kind=hazard_kind,
        event_time_ms=event_time_ms,
        window_end_time_ms_exclusive=window_end_time_ms_exclusive,
        ego_actor_id=ego_actor_id,
        bundle_id=bundle_id,
        source_window_sha256=source_window_sha256,
        source_event_chain_sha256=source_event_chain_sha256,
        source_evidence_sha256=source_evidence_sha256,
        source_event_sequence=source_event_sequence,
        derivation_file_name=derivation_file_name,
        derivation_file_sha256=derivation_file_sha256,
        recipe_version=(
            CANONICAL_RECIPE_VERSION if derivation_file_sha256 is not None else None
        ),
        historical_stage=historical_stage,
    )


_US101_EVIDENCE = "a991dfed3d2e80ad967b8710ed20c3c30776866897d82f23db92b1ef12d4c851"
_US101_MEMBER_SHA = NGSIM_US101_AUTHORITATIVE_SHA256

CANONICAL_CANDIDATES: tuple[NGSIMCanonicalCandidate, ...] = (
    _candidate(
        "ngsim:1113433176100:a1a032494881bc29",
        "i-80",
        "lane_change_conflict",
        1113433176100,
        1113433236100,
        "i-80:4",
        "ngsim-e6b5bda960853c8855e0",
        "97c1205fc18faf617c3aa040ffe0bc91e0e987d40f7a0901874ff366abc0d969",
        "61a6623433462128248be84bbf2d51d73dbb49428bedf832c2ddc985476e9bf0",
        "a2af51318c6d6df4539ee34ddf136fe558729242c5301c0b079932de975a22f9",
        ("lane_change_conflict", "cut_in_gap_boundary"),
        "i80_200s.csv",
        "7923ecdc17f0e143435440295b00a2d41da904d24c11fb97ae540b2be0502385",
    ),
    _candidate(
        "ngsim:1113433267500:2526d6e599205e7a",
        "i-80",
        "lane_change_conflict",
        1113433267500,
        1113433327500,
        "i-80:375",
        "ngsim-fa46bb498e1959121946",
        "8ef038ad52eaeac2ef164b6c32ff95d9058ac2c6277ba2b408ca46907fe4ea1c",
        "0415be7e588661170fbfde14068c20bc93fecbc6abafe530a90bb88710b2f326",
        "a2af51318c6d6df4539ee34ddf136fe558729242c5301c0b079932de975a22f9",
        ("lane_change_conflict", "cut_in_gap_boundary"),
        "i80_200s.csv",
        "7923ecdc17f0e143435440295b00a2d41da904d24c11fb97ae540b2be0502385",
    ),
    _candidate(
        "ngsim:1118846999200:e08996e3fddf68d4",
        "us-101",
        "lead_vehicle_braking",
        1118846999200,
        1118847059200,
        "us-101:42",
        "ngsim-d86eae1a85dc15212b4c",
        "408edbc9369ac834cbc05fc6211f3f4ced9ba39cba7436742c5a704409ada818",
        "ae17b0b32d736aa44723c4efc668f00b32d834a4a98756a7e2abc4ff8224bfb5",
        _US101_EVIDENCE,
        ("lead_vehicle_braking", "actor_state_update"),
        "trajectories-0750am-0805am.txt",
        _US101_MEMBER_SHA,
    ),
    _candidate(
        "ngsim:1118847062700:364428e7e12e2fee",
        "us-101",
        "lead_vehicle_braking",
        1118847067600,
        1118847122700,
        "us-101:309",
        "ngsim-1ca1e6baf16ebd6cbdc5",
        "9f611cf72b23ae3c625c3c8d27d6738886dc24ad826d3eb7fa4d05242d14a1e8",
        "4d3822f230d6329d69a93413cf3e57a3a1ae6015f019a387c8d72fefef4c4fcc",
        _US101_EVIDENCE,
        (),
        None,
        None,
        historical_stage="autonomous_driving_multisite_v25",
    ),
    _candidate(
        "ngsim:1118847070800:abc2ea840a747b78",
        "us-101",
        "minimum_time_headway_conflict",
        1118847066300,
        1118847130800,
        "us-101:399",
        "ngsim-d73d258f0181c95757e8",
        "58a36cea043d23dbf445daa826081168a99eaa374c1ca1875cbffd0c5388acd6",
        "9eb7fe5bd8251c668ada9a36a6c319bc2249e8c3def20bd182a2a32ccccc3749",
        _US101_EVIDENCE,
        ("short_time_headway_boundary",),
        "trajectories-0750am-0805am.txt",
        _US101_MEMBER_SHA,
    ),
    _candidate(
        "ngsim:1118847132300:fc9b160cb3ccb957",
        "us-101",
        "lead_vehicle_braking",
        1118847132300,
        1118847192300,
        "us-101:641",
        "ngsim-1be893566681220c441b",
        "f534571c465079dbef927445c82a3a6fdfa17ffcd173a7bab202722ab61c3907",
        "dca840354449a71782b0c29a2a7f7fd0c13fff193040a9ece966344b1cc2c7bb",
        _US101_EVIDENCE,
        (),
        None,
        None,
        historical_stage="autonomous_driving_multisite_v25",
    ),
    _candidate(
        "ngsim:1118847187100:3b6793cb928cf7fd",
        "us-101",
        "minimum_time_headway_conflict",
        1118847185100,
        1118847247100,
        "us-101:866",
        "ngsim-45e0967131ea56ac8fa4",
        "ce2b8902c10783d7384563f4d4f3fd19ca0d238c96a746b0518ebdeb0d489110",
        "8320fec55d45e3494a812c3a7542589e24a72bab37f1142d3beb495e9aa162c9",
        _US101_EVIDENCE,
        ("short_time_headway_boundary",),
        "trajectories-0750am-0805am.txt",
        _US101_MEMBER_SHA,
    ),
    _candidate(
        "ngsim:1118847260700:5aea66c4b9a5ba06",
        "us-101",
        "minimum_time_headway_conflict",
        1118847256100,
        1118847320700,
        "us-101:1109",
        "ngsim-cbab115cf709ab164a41",
        "57b0f12bee9f1992737dde6ab0a962a6bf86f2de684e477bf1a7e9ba87055d1d",
        "07c1606e127bafeded6d14b93f7494b92e68e3f8e492b08cd8a1f913ea32edbc",
        _US101_EVIDENCE,
        ("short_time_headway_boundary",),
        "trajectories-0750am-0805am.txt",
        _US101_MEMBER_SHA,
    ),
    _candidate(
        "ngsim:1118847360400:99e4d9e9718737e1",
        "us-101",
        "lead_vehicle_braking",
        1118847364800,
        1118847420400,
        "us-101:1344",
        "ngsim-bc0bd401db8947468bdc",
        "4c9e47021ce1d9cacc292ddaa239c9a20d9392ef0fd1c6c5a52ad73e24baa639",
        "e838361e0b9caf6c5bc97880f045f8c1b9faf182be0d24bbaa3968cc29ce4a34",
        _US101_EVIDENCE,
        ("lead_vehicle_braking", "actor_state_update"),
        "trajectories-0750am-0805am.txt",
        _US101_MEMBER_SHA,
    ),
    _candidate(
        "ngsim:1118847482500:626fc1b70a91943d",
        "us-101",
        "lead_vehicle_braking",
        1118847482500,
        1118847542500,
        "us-101:1753",
        "ngsim-2f726e0a8f6777a42ee6",
        "3abf469c0c08cb66452bb07eb15fe67b2527aa69db766c9bab0b795c9f523586",
        "15f3bacfeffe8d409fa44ddceb77d28bed4a870b3c49878cbbf2eedfa2fe94ea",
        _US101_EVIDENCE,
        ("lead_vehicle_braking", "actor_state_update"),
        "trajectories-0750am-0805am.txt",
        _US101_MEMBER_SHA,
    ),
    _candidate(
        "ngsim:1118847551400:ccdc6d3703d5ad43",
        "us-101",
        "lead_vehicle_braking",
        1118847561600,
        1118847611400,
        "us-101:1958",
        "ngsim-c60e32dd3be9cc8ab62b",
        "991c770b29fc4d54937ffe8b6c56bb9f1280f97e604d76c9c061ff00d123cf43",
        "e5932450ed91d06ca2d4eba44e0624b8c97bf522c3e5b9acf44d61341b39b53a",
        _US101_EVIDENCE,
        ("lead_vehicle_braking", "actor_state_update"),
        "trajectories-0750am-0805am.txt",
        _US101_MEMBER_SHA,
    ),
    _candidate(
        "ngsim:1118847616700:5cf1d4d7a4c571a4",
        "us-101",
        "lead_vehicle_braking",
        1118847616700,
        1118847676700,
        "us-101:2215",
        "ngsim-198b0b4bdf00f9d754bf",
        "6215539c74bc02bdecfed47a4ead8376defdc0fe461093328396b26d53f8ff4c",
        "ddbc3d89d86fe973a721c7876fd9f7a8d86732bf047c4a587df31f1430bc5e11",
        _US101_EVIDENCE,
        ("lead_vehicle_braking", "actor_state_update"),
        "trajectories-0750am-0805am.txt",
        _US101_MEMBER_SHA,
    ),
    _candidate(
        "ngsim:1118847677100:adc3ed02f831ff5e",
        "us-101",
        "lead_vehicle_braking",
        1118847677100,
        1118847737100,
        "us-101:2458",
        "ngsim-c60e2c40b71589890907",
        "b9df5df4fc5d688779ba016127790f8550129fa360068161187aa37b5cc60ce7",
        "17e3cbb3282f7b1d3a4611f7212e1e18863f95b95b93f40bd7a361f73bc08e24",
        _US101_EVIDENCE,
        ("lead_vehicle_braking", "actor_state_update"),
        "trajectories-0750am-0805am.txt",
        _US101_MEMBER_SHA,
    ),
    _candidate(
        "ngsim:1163040000:f0916a903b071474",
        "peachtree",
        "lane_change_conflict",
        1163040000,
        1163100000,
        "peachtree:9",
        "ngsim-fab26df1a65060711821",
        "6034b85ddbc0d88b19be46b6e2882e4f313f5662f82cf0f129be6514b3c778a1",
        "a23ff5cba80bc2c49196f1eb2ce81ece466a37362ee8ac30a9fdfa22b607339a",
        "a1430805f13076465284a6ec46287b7728a7927bc8a3536e208dad9d2c6e50ff",
        ("lane_change_conflict", "cut_in_gap_boundary"),
        "canonical.csv",
        "501f1d2a0aadbdbb88d8279fa7cdfea90d9e1f16e69674eb0aa9ac9535cf5fa7",
    ),
    _candidate(
        "ngsim:1163335200:b70b5e2d16d97895",
        "peachtree",
        "lane_change_conflict",
        1163335200,
        1163395200,
        "peachtree:1000256",
        "ngsim-40e4ba8972973def405b",
        "6ec1621f9a798af02464377cabadee7a972f2933ab1dffe75995e9d9409b39e5",
        "37f58c41c91fb53f55c733fea3f4fe9efa1c744aa0a1472d0641af4493541853",
        "7f3b6b6f74a7dd2c10e4bd5f8c11fde0c3c36644c3fffb779fa81feeecec1463",
        ("lane_change_conflict", "cut_in_gap_boundary"),
        "peachtree_1163335200_canonical.csv",
        "ea78fa0434979678c1352c6e2e6b76074607c34e9031fbdfaa1124ee9ca19e2f",
    ),
)
