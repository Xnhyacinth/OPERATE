import json
import os
from pathlib import Path
import subprocess
import sys


def test_actual_identity_collection_without_git_keeps_file_hashes(tmp_path):
    (tmp_path / "run.py").write_text('print("fixture")\n')
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "-c",
        "import json,sys;from pathlib import Path;from core.implementation_identity import implementation_identity;"
        "print(json.dumps(implementation_identity(Path(sys.argv[1]))))",
        str(tmp_path),
    ]
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env={**os.environ, "PATH": "", "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    identity = json.loads(result.stdout)
    assert identity["git_head"] == ""
    assert len(identity["implementation_tree_sha256"]) == 64
    (tmp_path / "run.py").write_text('print("changed")\n')
    changed = subprocess.run(
        command,
        cwd=tmp_path,
        env={**os.environ, "PATH": "", "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        check=True,
    )
    assert (
        json.loads(changed.stdout)["implementation_tree_sha256"]
        != identity["implementation_tree_sha256"]
    )
