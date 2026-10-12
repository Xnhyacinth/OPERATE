from copy import deepcopy
import hashlib
import json

import pytest

from evaluation.comparison_protocol import (
    CONCURRENCY, SHARED, _hash, build_protocol, load_protocol, validate_row,
)


def protocol_and_row():
    expected = {"class": "LLMAgent", "config": {"api_key_env": "LEGACY_KEY", "temperature": 0.0}}
    protocol = {
        "shared_conditions": {"implementation_tree_sha256": "tree", "scoring_version": "1"},
        "pass_id": "pass-0", "models": {"model": {"row_identity": {}, "agent_config": expected}},
    }
    row = {
        "model": "model", "pass_id": "pass-0", "score": {"scoring_version": "1"},
        "implementation_tree_sha256": "tree", "implementation_tree_sha256_start": "tree",
        "implementation_tree_sha256_end": "tree", "agent_config": deepcopy(expected),
    }
    row["agent_config"]["config"]["api_key_env"] = "API_KEY"
    return protocol, row


def test_comparison_accepts_generic_credential_name_without_mutating_evidence():
    protocol, row = protocol_and_row()
    original = deepcopy((protocol, row))
    validate_row(protocol, "model", row)
    assert (protocol, row) == original


def test_comparison_still_rejects_request_setting_drift():
    protocol, row = protocol_and_row()
    row["agent_config"]["config"]["temperature"] = 0.5
    with pytest.raises(ValueError, match="agent_config mismatch"):
        validate_row(protocol, "model", row)


def test_loading_historical_protocol_keeps_recorded_credential_metadata(tmp_path):
    profile = {"model": "model", "provider": "openai", "temperature": 0.0}
    cfg = {key: 1 if key in CONCURRENCY else "identity" for key in SHARED}
    cfg.update(
        agent_profile_identity_by_model={"model": profile},
        agent_profile_sha256_by_model={"model": _hash(profile)},
        agent_treatment_sha256_by_model={"model": "treatment"},
        run_semantics_fingerprint="semantics",
    )
    protocol = build_protocol({"model": cfg}, suite_sha256="suite")
    protocol["models"]["model"]["agent_config"]["config"]["api_key_env"] = "LEGACY_KEY"
    raw = json.dumps(protocol).encode()
    path = tmp_path / "protocol.json"
    path.write_bytes(raw)
    loaded, digest = load_protocol({
        "comparison_protocol": {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()},
        "runs": {},
    }, suite_sha256="suite", models=["model"])
    assert loaded == protocol
    assert digest == hashlib.sha256(raw).hexdigest()
