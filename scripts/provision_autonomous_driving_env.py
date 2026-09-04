#!/usr/bin/env python3
"""Plan, provision, and verify the isolated autonomous-driving runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil

# Commands use fixed argument vectors and never invoke a shell.
import subprocess  # nosec B404
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_DIR = REPO_ROOT / ".venv-autonomous-driving"
DEFAULT_REQUIREMENTS = REPO_ROOT / "requirements-autonomous-driving-pilot.txt"
FULL_REGRESSION_REQUIREMENTS = REPO_ROOT / "requirements-autonomous-driving-full-regression.txt"
DEFAULT_REPORT = REPO_ROOT / "reports" / "autonomous_driving_runtime.json"
MAIN_ENV_DIR = REPO_ROOT / ".venv"
FOCUSED_TESTS = (
    "tests/test_autonomous_driving_data.py",
    "tests/test_autonomous_driving_backend.py",
    "tests/test_autonomous_driving_runtime_assurance.py",
    "tests/test_autonomous_driving_adapter.py",
)
SUMO_VERSION_PROBE_TIMEOUT_SECONDS = 30


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _validate_env_dir(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == MAIN_ENV_DIR.resolve():
        raise ValueError("driving pilot may not use the main repository environment")
    if not resolved.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("driving pilot environment must stay inside the repository")
    return resolved


def _default_python() -> Path:
    override = os.environ.get("OPERATE_AUTONOMOUS_DRIVING_PYTHON")
    executable = override or shutil.which("python3.12") or "python3.12"
    return Path(executable)


def build_plan(
    *,
    python_executable: Path | None = None,
    env_dir: Path = DEFAULT_ENV_DIR,
    requirements: Path = DEFAULT_REQUIREMENTS,
    report: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    env_dir = _validate_env_dir(env_dir)
    bootstrap = python_executable or _default_python()
    env_python = env_dir / "bin" / "python"
    full_regression = requirements.resolve() == FULL_REGRESSION_REQUIREMENTS.resolve()
    test_targets = ("tests/",) if full_regression else FOCUSED_TESTS
    install_commands = (
        [
            [
                "env",
                f"VIRTUAL_ENV={env_dir}",
                "uv",
                "sync",
                "--active",
                "--frozen",
                "--python",
                str(env_python),
                "--extra",
                "dev",
                "--extra",
                "released-backends",
                "--extra",
                "llm",
                "--extra",
                "hf",
            ],
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(env_python),
                "--requirement",
                str(requirements.resolve()),
            ],
        ]
        if full_regression
        else [[
            str(env_python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "-r",
            str(requirements.resolve()),
        ]]
    )
    commands = [
        [str(bootstrap), "-m", "venv", str(env_dir)],
        *install_commands,
        [str(env_python), "-m", "pip", "check"],
        [str(env_python), "-m", "pytest", "-q", *test_targets],
        [str(env_python), "-m", "pip", "freeze", "--all"],
    ]
    return {
        "schema_version": "autonomous_driving_runtime_plan_v1",
        "environment": {"path": str(env_dir), "isolated_from_main": True},
        "python_executable": str(bootstrap),
        "requirements": {"path": str(requirements.resolve()), "sha256": _sha256(requirements)},
        "test_scope": "full_repository" if full_regression else "autonomous_driving_focused",
        "report": str(report.resolve()),
        "commands": commands,
    }


def _run(command: list[str], *, timeout: float | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(  # nosec B603
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout": str(exc.stdout or ""),
            "stderr": str(exc.stderr or ""),
            "timed_out": True,
        }
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _runtime_identity(env_python: Path) -> dict[str, Any]:
    code = (
        "import json,platform,sys;"
        "print(json.dumps({'python':platform.python_version(),"
        "'implementation':platform.python_implementation(),"
        "'platform':platform.platform(),'machine':platform.machine(),"
        "'executable':sys.executable}))"
    )
    result = _run([str(env_python), "-c", code])
    return (
        json.loads(result["stdout"]) if result["returncode"] == 0 else {"error": result["stderr"]}
    )


def _package_identity(env_python: Path) -> dict[str, Any]:
    code = (
        "import json,importlib.metadata as m;"
        "names=['eclipse-sumo','commonroad-io','numpy','scipy','shapely','lxml','pandas','PyYAML'];"
        "print(json.dumps({n:m.version(n) for n in names},sort_keys=True))"
    )
    result = _run([str(env_python), "-c", code])
    return (
        json.loads(result["stdout"]) if result["returncode"] == 0 else {"error": result["stderr"]}
    )


def _sumo_identity(env_python: Path) -> dict[str, Any]:
    code = (
        "import json,os,sys,sumo;"
        "sys.path.insert(0,os.path.join(sumo.SUMO_HOME,'tools'));"
        "import sumolib,traci;"
        "from core.sidecar.sumo_sidecar import probe_sumo_transport;"
        "print(json.dumps({'sumo_home':sumo.SUMO_HOME,"
        "'binary':os.path.join(sumo.SUMO_HOME,'bin','sumo'),"
        "'sumolib':sumolib.__file__,'traci':traci.__file__,"
        "'native_transport':probe_sumo_transport()}))"
    )
    result = _run([str(env_python), "-c", code])
    return (
        json.loads(result["stdout"]) if result["returncode"] == 0 else {"error": result["stderr"]}
    )


def _runtime_fingerprint(
    runtime: dict[str, Any],
    packages: dict[str, Any],
    sumo: dict[str, Any],
    sumo_binary: dict[str, Any],
    requirements: Path = DEFAULT_REQUIREMENTS,
) -> dict[str, Any]:
    """Build a portable, auditable runtime identity without OS admission rules."""
    value = {
        "schema_version": "autonomous_driving_runtime_fingerprint_v1",
        "runtime": {
            key: runtime.get(key) for key in ("python", "implementation", "platform", "machine")
        },
        "packages": dict(sorted(packages.items())),
        "sumo": {
            "native_transport": sumo.get("native_transport"),
            "version": next(iter(str(sumo_binary.get("stdout") or "").splitlines()), None),
        },
        "requirements_sha256": _sha256(requirements),
    }
    return {**value, "sha256": _payload_sha256(value)}


def verify(
    env_dir: Path = DEFAULT_ENV_DIR,
    report: Path = DEFAULT_REPORT,
    requirements: Path = DEFAULT_REQUIREMENTS,
) -> dict[str, Any]:
    env_dir = _validate_env_dir(env_dir)
    env_python = env_dir / "bin" / "python"
    blockers: list[str] = []
    if not env_python.is_file():
        blockers.append("isolated_environment_missing")
        payload: dict[str, Any] = {"status": "held", "blockers": blockers}
    else:
        runtime = _runtime_identity(env_python)
        if str(runtime.get("python", "")).split(".")[:2] != ["3", "12"]:
            blockers.append("cpython_3_12_required")
        package_identity = _package_identity(env_python)
        if "error" in package_identity:
            blockers.append("runtime_packages_unavailable")
        sumo = _sumo_identity(env_python)
        if "error" in sumo:
            blockers.append("sumo_python_runtime_unavailable")
        sumo_binary = _run(
            [str(sumo.get("binary") or "sumo"), "--version"],
            timeout=SUMO_VERSION_PROBE_TIMEOUT_SECONDS,
        )
        if sumo_binary["returncode"] != 0:
            blockers.append("sumo_binary_probe_failed")
        freeze = _run([str(env_python), "-m", "pip", "freeze", "--all"])
        payload = {
            "status": "verified" if not blockers else "held",
            "blockers": blockers,
            "runtime": runtime,
            "platform_policy": "cross_platform_runtime_fingerprint_v1",
            "runtime_fingerprint": _runtime_fingerprint(
                runtime, package_identity, sumo, sumo_binary, requirements
            ),
            "packages": package_identity,
            "sumo": sumo,
            "native_smoke_supported": sumo.get("native_transport") in {"libsumo", "traci"},
            "sumo_binary_probe": sumo_binary,
            "requirements_sha256": _sha256(requirements),
            "freeze": freeze["stdout"].splitlines() if freeze["returncode"] == 0 else [],
        }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def provision(
    *,
    python_executable: Path | None = None,
    env_dir: Path = DEFAULT_ENV_DIR,
    requirements: Path = DEFAULT_REQUIREMENTS,
    report: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    plan = build_plan(
        python_executable=python_executable,
        env_dir=env_dir,
        requirements=requirements,
        report=report,
    )
    env_python = env_dir / "bin" / "python"
    commands = plan["commands"] if not env_python.is_file() else plan["commands"][1:]
    results: list[dict[str, Any]] = []
    for command in commands:
        dependency_command = any(part in {"install", "sync"} for part in command)
        result = _run(
            command,
            timeout=(7_200 if "pytest" in command else 1_800 if dependency_command else 300),
        )
        results.append(result)
        if result["returncode"] != 0:
            break
    failed = next((item for item in results if item["returncode"] != 0), None)
    if failed is not None:
        payload = {
            "status": "held",
            "blockers": ["provision_command_failed"],
            "test_scope": plan["test_scope"],
            "failed": failed,
            "provision_results": results,
        }
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload
    payload = verify(env_dir, report, requirements)
    payload.update(
        {
            "test_scope": plan["test_scope"],
            "provision_results": results,
        }
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--provision", action="store_true")
    modes.add_argument("--verify", action="store_true")
    parser.add_argument("--env-dir", type=Path, default=DEFAULT_ENV_DIR)
    parser.add_argument("--python", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    args = parser.parse_args()
    if args.plan:
        payload = build_plan(
            python_executable=args.python,
            env_dir=args.env_dir,
            requirements=args.requirements,
            report=args.report,
        )
    elif args.provision:
        payload = provision(
            python_executable=args.python,
            env_dir=args.env_dir,
            requirements=args.requirements,
            report=args.report,
        )
    else:
        payload = verify(args.env_dir, args.report, args.requirements)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") != "held" or args.plan else 2


if __name__ == "__main__":
    raise SystemExit(main())
