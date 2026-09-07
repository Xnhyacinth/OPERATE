"""Historical qualification stays immutable while fresh runs bind current code."""

import pytest

from core.implementation_identity import implementation_identity
from scripts import batch_llm_eval as logical
from tests.test_batch_llm_eval import _write_green_formal_gate


def _change_runtime(repo):
    path = repo / "core" / "maintenance_fix.py"
    path.parent.mkdir(exist_ok=True)
    path.write_text("MAINTENANCE_FIX = True\n")


def test_new_run_uses_current_tree_without_rewriting_qualification(tmp_path):
    repo = tmp_path / "repo"
    paths = _write_green_formal_gate(repo)
    original = {path: path.read_bytes() for path in paths.values() if path.is_file()}
    old_tree = implementation_identity(repo)["implementation_tree_sha256"]
    _change_runtime(repo)
    current_tree = implementation_identity(repo)["implementation_tree_sha256"]
    assert current_tree != old_tree

    binding = logical.resolve_formal_manifest_slice(paths["manifest"], repo_root=repo)
    old_treatment = logical._formal_agent_treatment_hashes(
        {"model": "a" * 64}, formal_manifest_binding=binding,
        implementation_tree_sha256=old_tree,
    )
    new_treatment = logical._formal_agent_treatment_hashes(
        {"model": "a" * 64}, formal_manifest_binding=binding,
        implementation_tree_sha256=current_tree,
    )
    assert old_treatment != new_treatment
    assert all(path.read_bytes() == body for path, body in original.items())
    paths["backend_file"].write_bytes(b"unlocked source")
    with pytest.raises(ValueError, match="backend runtime closure drift"):
        logical.resolve_formal_manifest_slice(paths["manifest"], repo_root=repo)
