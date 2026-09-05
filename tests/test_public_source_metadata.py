from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import download_from_hf as reader


def _m5_fixture(tmp_path, monkeypatch):
    spec = deepcopy(reader._EXACT_SOURCE_ASSET_SPECS["m5"])
    csvs = {path: f"fixture,{Path(path).name}\n".encode() for path in spec["paths"]}
    hashes = {path: hashlib.sha256(body).hexdigest() for path, body in csvs.items()}
    lock = {
        "source_id": "m5_forecasting",
        "source_url": "https://www.kaggle.com/competitions/m5-forecasting-accuracy",
        "license": "Kaggle competition rules",
        "license_verified": True,
        "terms_accepted": True,
        "license_or_terms_sha256": "sha256:" + "a" * 64,
        "inventory_environment_id": "InvManagement-v1",
        "package_version": "or-gym==0.5.0",
        "files": hashes,
        "orgym_runtime_source": {
            "commit": "0b18d16e569e2db70e83f09e867b53bdb4b87298",
            "license": "MIT",
        },
    }
    payload = json.dumps(lock).encode()
    metadata = "works/M5/source_lock.json"
    # Tiny fixture bytes have their own pinned identity; production identity
    # is asserted separately in test_public_hf_download.py.
    spec["metadata"] = {metadata: hashlib.sha256(payload).hexdigest()}
    monkeypatch.setitem(reader._EXACT_SOURCE_ASSET_SPECS, "m5", spec)
    data = tmp_path / "data"
    repo = tmp_path / "repo"
    repo.mkdir()
    files = {}
    for path, body in {**csvs, metadata: payload}.items():
        archive = f"backends/release_source_assets/m5/{path}"
        source = data / archive
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(body)
        files[path] = {
            "archive_path": archive, "delivery": "bundle",
            "sha256": hashlib.sha256(body).hexdigest(),
            "roles": ["metadata"] if path == metadata else spec["roles"],
            "scenario_ids": ["s"],
        }
    manifest = {"source_assets": {"m5": {
        "files": files, "n_files": 4, "scenario_ids": ["s"], "n_scenarios": 1,
        "delivery": "bundle", "redistribution": spec["redistribution"],
    }}}
    return repo, data, manifest, lock


def test_m5_metadata_installs_exactly_and_is_idempotent(tmp_path, monkeypatch):
    repo, data, manifest, lock = _m5_fixture(tmp_path, monkeypatch)
    assert reader.install_bundle_source_assets(data, manifest, repo_root=repo) == 4
    assert json.loads((repo / "works/M5/source_lock.json").read_text()) == lock
    assert reader.install_bundle_source_assets(data, manifest, repo_root=repo) == 0


@pytest.mark.parametrize("mutation", ["missing", "wrong_role", "wrong_hash", "extra"])
def test_m5_metadata_manifest_is_exact(tmp_path, monkeypatch, mutation):
    repo, data, manifest, _ = _m5_fixture(tmp_path, monkeypatch)
    contract = manifest["source_assets"]["m5"]
    files = contract["files"]
    metadata = "works/M5/source_lock.json"
    if mutation == "missing":
        del files[metadata]
    elif mutation == "wrong_role":
        files[metadata]["roles"] = ["runtime_input"]
    elif mutation == "wrong_hash":
        files[metadata]["sha256"] = "f" * 64
    else:
        files["works/M5/unlocked.json"] = deepcopy(files[metadata])
    contract["n_files"] = len(files)
    with pytest.raises(ValueError, match="m5_source_asset_contract_invalid"):
        reader.install_bundle_source_assets(data, manifest, repo_root=repo)
    assert not (repo / "works").exists()


@pytest.mark.parametrize("mutation", ["files", "terms", "runtime", "environment"])
def test_m5_native_metadata_must_match_csv_identity(tmp_path, monkeypatch, mutation):
    repo, data, manifest, lock = _m5_fixture(tmp_path, monkeypatch)
    if mutation == "files":
        lock["files"]["works/M5/calendar.csv"] = "f" * 64
    elif mutation == "terms":
        lock["terms_accepted"] = False
    elif mutation == "runtime":
        lock["orgym_runtime_source"]["commit"] = "f" * 40
    else:
        lock["inventory_environment_id"] = "wrong"
    body = json.dumps(lock).encode()
    metadata = "works/M5/source_lock.json"
    row = manifest["source_assets"]["m5"]["files"][metadata]
    (data / row["archive_path"]).write_bytes(body)
    row["sha256"] = hashlib.sha256(body).hexdigest()
    reader._EXACT_SOURCE_ASSET_SPECS["m5"]["metadata"][metadata] = row["sha256"]
    with pytest.raises(ValueError, match="m5_source_lock_metadata_invalid"):
        reader.install_bundle_source_assets(data, manifest, repo_root=repo)
    assert not (repo / "works").exists()


def test_ngsim_checksums_use_normal_install_conflict_and_rollback(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    data = tmp_path / "data"
    spec = reader._NGSIM_US101_SOURCE_SPEC
    path = f"{spec['root']}/ngsim_test/checksums.sha256"
    body = ("a" * 64 + "  bundle.json\n").encode()
    digest = hashlib.sha256(body).hexdigest()
    archive = reader._ngsim_us101_archive_path(digest).as_posix()
    source = data / archive
    source.parent.mkdir(parents=True)
    source.write_bytes(body)
    manifest = {"source_assets": {"ngsim_us101": {
        "delivery": "bundle", "redistribution": spec["redistribution"],
        "n_files": 1, "n_blobs": 1, "n_scenarios": 1, "scenario_ids": ["s"],
        "files": {path: {
            "archive_path": archive, "sha256": digest, "delivery": "bundle",
            "roles": ["metadata"], "scenario_ids": ["s"],
        }},
        "blobs": {archive: {
            "archive_path": archive, "sha256": digest,
            "source_path": path, "install_paths": [path],
        }},
    }}}
    target = repo / path
    assert not target.exists()
    targets = reader._repo_install_targets(data, manifest, repo_root=repo)
    assert target in targets
    with pytest.raises(RuntimeError, match="later failure"):
        with reader._repo_file_transaction(targets, repo_root=repo):
            assert reader.install_bundle_source_assets(data, manifest, repo_root=repo) == 1
            assert target.read_bytes() == body
            raise RuntimeError("later failure")
    assert not target.exists()
    assert reader.install_bundle_source_assets(data, manifest, repo_root=repo) == 1
    assert reader.install_bundle_source_assets(data, manifest, repo_root=repo) == 0
    target.write_bytes(b"unrelated")
    with pytest.raises(ValueError, match="bundle_source_asset_target_conflict"):
        reader.install_bundle_source_assets(data, manifest, repo_root=repo)
    assert target.read_bytes() == b"unrelated"
