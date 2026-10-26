"""Frozen input relocation preserves hashes and the repository boundary."""

import hashlib
import json
from pathlib import Path

import pytest

from evaluation.mission_contracts import _read_locked
from evaluation.source_inputs import source_input_path
from evaluation.trajectory_outcome027 import read_bound


def _relocate(root, target="inputs/scenario.yaml"):
    raw = b"scenario: frozen\n"
    sha = hashlib.sha256(raw).hexdigest()
    inventory = root / "evaluation/policies/source_inputs.json"
    inventory.parent.mkdir(parents=True)
    inventory.write_text(json.dumps({
        "schema_version": "operate-source-input-locations-v1",
        "files": {"scenarios/original.yaml": {"path": target, "sha256": sha}},
    }))
    path = root / target
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw, {"path": "scenarios/original.yaml", "sha256": sha}


def test_relocated_input_is_verified_by_both_readers(tmp_path):
    raw, ref = _relocate(tmp_path)
    assert _read_locked(tmp_path, ref, label="scenario") == raw
    assert read_bound(ref, root=tmp_path)[0] == raw


def test_present_declared_input_cannot_fall_back_after_corruption(tmp_path):
    _, ref = _relocate(tmp_path)
    path = tmp_path / ref["path"]
    path.parent.mkdir()
    path.write_text("changed")
    with pytest.raises(ValueError, match="scenario_hash_mismatch"):
        _read_locked(tmp_path, ref, label="scenario")


def test_relocated_bytes_and_binding_must_match(tmp_path):
    _, ref = _relocate(tmp_path)
    (tmp_path / "inputs/scenario.yaml").write_text("changed")
    with pytest.raises(ValueError, match="bound_artifact_hash_mismatch"):
        read_bound(ref, root=tmp_path)
    with pytest.raises(ValueError, match="source_input_binding_mismatch"):
        source_input_path(tmp_path, ref["path"], "0" * 64)


@pytest.mark.parametrize("escape", ["../outside.yaml", "/outside.yaml"])
def test_declared_input_cannot_escape_root(tmp_path, escape):
    with pytest.raises(ValueError, match="source_input_outside_root"):
        source_input_path(tmp_path, escape, "0" * 64)


def test_relocated_symlink_cannot_escape_root(tmp_path):
    _, ref = _relocate(tmp_path)
    path = tmp_path / "inputs/scenario.yaml"
    path.unlink()
    path.symlink_to(tmp_path.parent / "outside.yaml")
    with pytest.raises(ValueError, match="source_input_outside_root"):
        source_input_path(tmp_path, ref["path"], ref["sha256"])


def test_all_distributed_frozen_inputs_retain_original_hashes():
    root = Path(__file__).resolve().parents[1]
    index = json.loads((root / "evaluation/policies/source_inputs.json").read_text())
    assert len(index["files"]) == 143
    for path, ref in index["files"].items():
        _read_locked(root, {"path": path, "sha256": ref["sha256"]}, label="frozen")


def test_input_location_inventory_cannot_link_outside_root(tmp_path):
    _, ref = _relocate(tmp_path)
    index = tmp_path / "evaluation/policies/source_inputs.json"
    outside = tmp_path.parent / (tmp_path.name + "-index.json")
    index.rename(outside)
    index.symlink_to(outside)
    with pytest.raises(ValueError, match="source_input_outside_root"):
        source_input_path(tmp_path, ref["path"], ref["sha256"])
