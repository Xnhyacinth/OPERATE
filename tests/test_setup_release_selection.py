import json
from pathlib import Path
import subprocess
import sys


SETUP = Path(__file__).resolve().parents[1] / "scripts/setup_eval_env.sh"


def test_setup_uses_installed_bundle_release_for_smoke_and_verification(tmp_path):
    source = SETUP.read_text()
    marker = "<<'RELEASE_SELECTION'\n"
    assert marker in source
    code = source.split(marker, 1)[1].split("\nRELEASE_SELECTION", 1)[0]
    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text(json.dumps({"release_id": "operate_v0_62_0"}))
    result = subprocess.run([sys.executable, "-", str(manifest)], input=code,
                            text=True, capture_output=True, check=True)
    assert result.stdout.strip() == "operate_v0_62_0"
    assert 'core = json.loads((Path(sys.argv[1]) / "core_suite.json").read_text())' in source
    assert 'RELEASE_MANIFEST="$RELEASE_DIR/manifest.json"' in source


def test_setup_rejects_bundle_release_path_traversal(tmp_path):
    source = SETUP.read_text()
    marker = "<<'RELEASE_SELECTION'\n"
    assert marker in source
    code = source.split(marker, 1)[1].split("\nRELEASE_SELECTION", 1)[0]
    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text(json.dumps({"release_id": "../wrong"}))
    result = subprocess.run([sys.executable, "-", str(manifest)], input=code,
                            text=True, capture_output=True)
    assert result.returncode != 0
