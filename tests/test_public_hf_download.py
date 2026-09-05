from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from scripts import download_from_hf


REPO_ID = "Xnhyacinth/OPERATE"
REVISION = "a" * 40
REPO_ROOT = Path(__file__).resolve().parents[1]


def test_public_downloader_uses_versionless_install_root() -> None:
    assert download_from_hf.DATA.name == "operate_data"


def test_public_downloader_accepts_exact_m5_runtime_assets() -> None:
    spec = download_from_hf._EXACT_SOURCE_ASSET_SPECS["m5"]
    assert spec["backend_kind"] == "orgym_invmgmt"
    assert spec["delivery"] == "bundle"
    assert spec["roles"] == ["derivation_input", "runtime_input"]
    assert spec["paths"] == {
        "works/M5/calendar.csv",
        "works/M5/sales_train_evaluation.csv",
        "works/M5/sell_prices.csv",
    }
    assert spec["metadata"] == {
        "works/M5/source_lock.json":
        "271c94965d27bf74b0d66ba89e71b5bc239ddc5192ce99305bbac256a848a9b3"
    }


def test_m5_bundle_contract_requires_both_source_roles_and_exact_paths() -> None:
    spec = download_from_hf._EXACT_SOURCE_ASSET_SPECS["m5"]
    files = {
        path: {
            "archive_path": (
                f"backends/release_source_assets/m5/{path}"
            ),
            "delivery": "bundle",
            "sha256": f"{index}" * 64,
            "roles": ["derivation_input", "runtime_input"],
            "scenario_ids": ["scenario-1"],
        }
        for index, path in enumerate(sorted(spec["paths"]), start=1)
    }
    files["works/M5/source_lock.json"] = {
        "archive_path": "backends/release_source_assets/m5/works/M5/source_lock.json",
        "delivery": "bundle",
        "sha256": "271c94965d27bf74b0d66ba89e71b5bc239ddc5192ce99305bbac256a848a9b3",
        "roles": ["metadata"],
        "scenario_ids": ["scenario-1"],
    }
    contract = {
        "n_scenarios": 1,
        "n_files": 4,
        "delivery": "bundle",
        "scenario_ids": ["scenario-1"],
        "redistribution": spec["redistribution"],
        "files": files,
    }

    assert set(
        download_from_hf._source_asset_file_rows(
            {"source_assets": {"m5": contract}}
        )
    ) == spec["paths"] | {"works/M5/source_lock.json"}

    files[next(iter(files))]["roles"] = ["derivation_input"]
    with pytest.raises(ValueError, match="m5_source_asset_contract_invalid"):
        download_from_hf._source_asset_file_rows(
            {"source_assets": {"m5": contract}}
        )


def test_public_compatibility_entrypoint_is_a_thin_delegate() -> None:
    wrapper = (REPO_ROOT / "tools" / "download_public_bundle.py").read_text(
        encoding="utf-8"
    )
    assert "download_from_hf.main" in wrapper
    assert "_validate_bundle_distribution_binding =" not in wrapper


def test_setup_uses_canonical_versionless_downloader() -> None:
    setup = (REPO_ROOT / "scripts" / "setup_eval_env.sh").read_text(
        encoding="utf-8"
    )
    assert "scripts/download_from_hf.py" in setup
    assert "data_operate_v058" not in setup
    assert 'RUNTIME_DATA_DIR="$REPO/operate_data"' in setup
    assert "set OPERATE_HF_REVISION" not in setup
    assert "power-grid-lib/pglib-uc.git pglib-uc" in setup
    assert "TUM-VT/sumo_ingolstadt.git sumo_ingolstadt_upstream" in setup


def test_public_distribution_binding_is_exact() -> None:
    download_from_hf._validate_bundle_distribution_binding(
        {
            "hf_repo_id": REPO_ID,
            "visibility": "public",
            "bundle_kind": "public_runtime_companion",
        },
        repo_id=REPO_ID,
    )
    with pytest.raises(ValueError, match="bundle_download_hf_repo_id_mismatch"):
        download_from_hf._validate_bundle_distribution_binding(
            {
                "hf_repo_id": "other/OPERATE",
                "visibility": "public",
                "bundle_kind": "public_runtime_companion",
            },
            repo_id=REPO_ID,
        )
    for visibility in (None, "private", "internal", True):
        with pytest.raises(ValueError, match="bundle_download_visibility_invalid"):
            download_from_hf._validate_bundle_distribution_binding(
                {
                    "hf_repo_id": REPO_ID,
                    "visibility": visibility,
                    "bundle_kind": "public_runtime_companion",
                },
                repo_id=REPO_ID,
            )
    with pytest.raises(ValueError, match="bundle_download_kind_invalid"):
        download_from_hf._validate_bundle_distribution_binding(
            {
                "hf_repo_id": REPO_ID,
                "visibility": "public",
                "bundle_kind": "private_runtime_companion",
            },
            repo_id=REPO_ID,
        )


def test_bundle_manifest_rejects_file_hash_mismatch(tmp_path: Path) -> None:
    (tmp_path / "payload.bin").write_bytes(b"tampered")
    (tmp_path / "MANIFEST.json").write_text(
        json.dumps({"files": {"payload.bin": "a" * 64}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="bundle_hash_mismatch:payload.bin"):
        download_from_hf.verify_manifest(tmp_path)


def _fake_huggingface_hub(*, private: bool, sha: str) -> types.ModuleType:
    module = types.ModuleType("huggingface_hub")

    class HfApi:
        def __init__(self, *, token: str | None) -> None:
            assert token is None

        def dataset_info(self, repo_id: str, *, revision: str | None = None):
            assert repo_id == REPO_ID
            assert revision in {None, REVISION}
            return types.SimpleNamespace(
                private=private,
                sha=sha,
                last_modified="2026-09-04T00:00:00Z",
            )

    module.HfApi = HfApi
    module.get_token = lambda: None
    module.snapshot_download = lambda **_: (_ for _ in ()).throw(
        AssertionError("dry-run must not download")
    )
    return module


def _run_dry_run(
    monkeypatch,
    tmp_path: Path,
    *,
    private: bool,
    sha: str,
    revision: str | None,
) -> int:
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        _fake_huggingface_hub(private=private, sha=sha),
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(download_from_hf, "REPO", tmp_path)
    argv = [
        "download_from_hf.py",
        "--repo-id",
        REPO_ID,
        "--data-dir",
        str(tmp_path / "runtime"),
        "--dry-run",
    ]
    if revision is not None:
        argv.extend(("--revision", revision))
    monkeypatch.setattr(sys, "argv", argv)
    return download_from_hf.main()


def test_anonymous_current_public_revision_dry_run_is_accepted(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    assert _run_dry_run(
        monkeypatch,
        tmp_path,
        private=False,
        sha=REVISION,
        revision=None,
    ) == 0
    output = capsys.readouterr().out
    assert f"resolved revision: {REVISION}" in output


def test_anonymous_exact_public_revision_dry_run_is_accepted(
    monkeypatch, tmp_path: Path
) -> None:
    assert _run_dry_run(
        monkeypatch,
        tmp_path,
        private=False,
        sha=REVISION,
        revision=REVISION,
    ) == 0


@pytest.mark.parametrize(
    ("private", "sha", "revision", "error"),
    [
        (True, REVISION, None, "bundle_dataset_visibility_not_public"),
        (False, "f" * 40, REVISION, "bundle_dataset_revision_mismatch"),
    ],
)
def test_public_dry_run_rejects_wrong_remote_binding(
    monkeypatch,
    tmp_path: Path,
    capsys,
    private: bool,
    sha: str,
    revision: str | None,
    error: str,
) -> None:
    assert _run_dry_run(
        monkeypatch,
        tmp_path,
        private=private,
        sha=sha,
        revision=revision,
    ) == 1
    assert error in capsys.readouterr().err


@pytest.mark.parametrize(
    ("repo_id", "revision"),
    [
        ("not-a-repo", REVISION),
        (REPO_ID, "main"),
        (REPO_ID, "0" * 40),
    ],
)
def test_public_download_rejects_unbound_repo_or_revision(
    repo_id: str, revision: str
) -> None:
    with pytest.raises(ValueError):
        download_from_hf._validate_hf_binding(
            repo_id=repo_id,
            revision=revision,
        )
