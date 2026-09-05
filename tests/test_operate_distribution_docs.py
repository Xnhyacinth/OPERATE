from __future__ import annotations

import json
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_release_package_version_is_current_v061() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["version"] == "0.61.0"


def test_documented_environment_and_formal_scope_are_current() -> None:
    paths = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs" / "CURRENT_RELEASE.md",
        REPO_ROOT / "docs" / "FORMAL_EVALUATION.md",
    ]
    contents = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert ".venv313" not in contents
    assert "pre-admission" in contents
    assert "core_suite.json" in contents
    assert "manifest.json" in contents
    assert "formal_evaluation_ready" in contents
    assert "operate_v0_61_0" in contents
    assert "769" in contents
    assert "502" in contents
    core = json.loads(
        (REPO_ROOT / "release/operate_v0_61_0/core_suite.json").read_text()
    )
    inherited = sum(
        row["path"].startswith("scenarios/operate_v0_58_0/")
        for row in core["scenarios"]
    )
    assert f"{inherited} inherited" in contents
    assert "scenarios/operate_v0_59_0/" in contents
    assert "scenarios/operate_v0_60_0/" in contents
    assert "scenarios/operate_v0_61_0/" in contents
    assert "formal_logical_persistent_evaluation_pending" in contents
    assert "formal_realtime_persistent_evaluation_pending" in contents
    assert "formal_runtime_evidence_distribution_pending" in contents


def test_maintained_setup_uses_operate_runtime_bundle() -> None:
    setup = (REPO_ROOT / "scripts" / "setup_eval_env.sh").read_text(encoding="utf-8")

    assert "scripts/build_data_dir.py" not in setup
    assert "scripts/download_from_hf.py" in setup
    assert 'git -C "$WORKS/JSPLIB-Instances" check-ignore CHECKSUMS.txt' in setup
    assert "OPERATE_HF_REVISION" in setup
    assert 'DOWNLOAD_ARGS+=(--revision "$OPERATE_HF_REVISION")' in setup
    assert "NREL_API_KEY" not in setup
    assert "download_nrel_microgrid_sources.py" not in setup


def test_setup_requires_candidate_evidence_only_when_bundle_declares_it() -> None:
    setup = (REPO_ROOT / "scripts" / "setup_eval_env.sh").read_text(
        encoding="utf-8"
    )

    assert 'manifest.get("candidate_evidence_archive")' in setup
    assert 'if [ "$CANDIDATE_EVIDENCE_REQUIRED" -eq 1 ]; then' in setup
    assert '[ -d "$RUNTIME_DATA_DIR/candidate_evidence/.hl/artifacts" ]' in setup
    assert "candidate closure evidence is not required by this compact bundle" in setup
    assert "candidate closure evidence was not restored from the bundle" in setup


def test_maintained_setup_does_not_clone_companion_archive_roots() -> None:
    setup = (REPO_ROOT / "scripts" / "setup_eval_env.sh").read_text(encoding="utf-8")

    clone_lines = [
        line for line in setup.splitlines() if line.startswith("clone_pinned ")
    ]
    archive_roots = {
        "OpenDSS-IEEE13",
        "PGLib-OPF",
        "PyVRP-Instances",
        "RESCO",
        "RTS-GMLC",
        "VRPLIB",
        "sumo_ingolstadt",
    }
    assert all(line.split()[2] not in archive_roots for line in clone_lines)


def test_runtime_companion_is_not_documented_as_portable_core() -> None:
    data_readme = (REPO_ROOT / "data" / "README.md").read_text(encoding="utf-8")
    dataset_card = (REPO_ROOT / "docs" / "hf" / "OPERATE_DATASET_CARD.md").read_text(
        encoding="utf-8"
    )
    downloader = (REPO_ROOT / "scripts" / "download_from_hf.py").read_text(
        encoding="utf-8"
    )
    layout = (REPO_ROOT / "docs" / "REPO_LAYOUT_AND_DATA_USAGE.md").read_text(
        encoding="utf-8"
    )
    provenance = (REPO_ROOT / "docs" / "DATA_PROVENANCE.md").read_text(
        encoding="utf-8"
    )
    release_map = (REPO_ROOT / "release" / "README.md").read_text(encoding="utf-8")

    assert "--portable" not in data_readme
    assert "runtime companion" in data_readme.lower()
    assert "scripts/download_from_hf.py --download-only" in data_readme
    assert "data_operate_v059" not in data_readme
    assert "operate_data/" in data_readme
    assert dataset_card.startswith("---\nlicense: other\n")
    assert "does not relicense those assets under MIT" in " ".join(
        dataset_card.split()
    )
    assert "portable bundle" not in downloader.lower()
    assert "--portable" not in downloader
    assert "operate_data/" in layout
    assert "candidate_evidence_archive" in layout
    assert "candidate_evidence_archive" in provenance
    assert "operate_v0_61_0" in release_map
    assert "operate_data/" in release_map
    assert "lite_suite.json" in release_map


def test_formal_runbook_uses_resumable_logical_and_realtime_commands() -> None:
    runbook = (REPO_ROOT / "docs" / "FORMAL_EVALUATION.md").read_text(
        encoding="utf-8"
    )

    assert "--no-resume" not in runbook
    assert runbook.count("scripts/batch_realtime_llm_eval.py") >= 2
    assert "--model z-ai/glm-5.2:free" in runbook
    assert "--model hy3-ioa" in runbook
    assert "--interaction-mode logical_persistent" in runbook
    assert "--resume" in runbook
    assert ".hl/release_rebuild/" not in runbook
    assert 'manifest["formal_evidence"]["readiness"]' in runbook


def test_formal_runbook_documents_release_finalizer_and_unknown_quota() -> None:
    runbook = (REPO_ROOT / "docs" / "FORMAL_EVALUATION.md").read_text(
        encoding="utf-8"
    )

    assert "scripts/finalize_operate_release.py" in runbook
    assert "--release-manifest" in runbook
    assert "--logical-batch-manifest" in runbook
    assert "--realtime-batch-manifest" in runbook
    assert "--output-manifest" in runbook
    assert "same single model" in runbook
    assert "independent of the already-public source/runtime distribution" in runbook
    assert "Tencent quota remains unknown" in runbook
    assert "OPERATE_TENCENT_HY3_RPM_LIMIT:?" not in runbook
    assert "OPERATE_TENCENT_HY3_RPD_LIMIT:?" not in runbook


def test_active_data_docs_do_not_retain_pre_promotion_counts() -> None:
    docs = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            REPO_ROOT / "docs" / "DATA_PROVENANCE.md",
            REPO_ROOT / "docs" / "REPO_LAYOUT_AND_DATA_USAGE.md",
        )
    )

    assert "903" not in docs
    assert "635" not in docs
    assert "769" in docs
    assert "502" in docs


def test_agent_owned_wakeup_policy_is_documented_without_periodic_scan() -> None:
    docs = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            REPO_ROOT / "docs" / "AGENTIC_INTERACTION.md",
            REPO_ROOT / "docs" / "BENCHMARK_DESIGN.md",
        )
    )

    assert "agent_scheduled_v1" in docs
    assert "harness_periodic_supervisory_scan=false" in docs
    assert "maximum-silence supervisory scan" not in docs


def test_formal_bundle_builder_has_no_backendless_cli() -> None:
    builder = (REPO_ROOT / "scripts" / "build_operate_bundle.py").read_text(
        encoding="utf-8"
    )

    assert "--without-backends" not in builder
