from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_project_metadata_supports_python_314() -> None:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'requires-python = ">=3.10,<3.15"' in project
    assert "numpy>=1.24,<2.3; python_version < '3.14'" in project
    assert "numpy>=2.3.3,<2.4; python_version >= '3.14'" in project
    assert "pandas>=2.3.3,<3; python_version >= '3.14'" in project
    assert "scipy>=1.16.1; python_version >= '3.14'" in project


def test_citylearn_excludes_unused_unportable_openstudio_dependency() -> None:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert '{ name = "citylearn", version = "2.5.0" }' in project
    assert '{ name = "doe-xstock", version = "2.0.0" }' in project
    assert 'dependencies = ["openstudio"]' in project
    assert 'dependencies = ["scikit-learn>=1.6,<2"]' in project


def test_released_backends_lock_native_traffic_and_dynasched_runtimes() -> None:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")

    assert '"traci>=1.20"' in project
    assert (
        '"dsbx @ git+https://github.com/dsbx7/'
        'DynaSchedBench.git@08975bf4a0473c5dff9177393bc6743db9ddc946"'
        in project
    )
    assert 'name = "traci"' in lock
    assert (
        'DynaSchedBench.git?rev=08975bf4a0473c5dff9177393bc6743db9ddc946'
        in lock
    )


def test_ci_exercises_python_314() -> None:
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert 'python-version: ["3.10", "3.12", "3.14"]' in ci
    assert "python -m pip install uv==0.12.5" in ci
    assert "uv sync --frozen --python ${{ matrix.python-version }} --extra dev" in ci


def test_running_interpreter_is_within_declared_support() -> None:
    assert (3, 10) <= sys.version_info[:2] < (3, 15)


def test_wheel_discovers_runtime_and_data_pipeline_packages() -> None:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "[tool.setuptools.packages.find]" in project
    for package_glob in (
        '"core*"',
        '"domains*"',
        '"evaluation*"',
        '"baselines*"',
        '"runner*"',
        '"audit*"',
        '"data"',
    ):
        assert package_glob in project
    assert 'exclude = ["data.backends*"]' in project
