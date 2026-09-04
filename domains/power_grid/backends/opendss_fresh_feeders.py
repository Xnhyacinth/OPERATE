"""OpenDSS source-locked non-release live probe backend.

This module intentionally stops at the source-integration ladder's live-probe
rung. It compiles the SHA-256-pinned fresh feeder sources through
``dss-python`` and exposes a tiny Volt-Var control surface through
``core.tool_protocol``. It does not materialize release scenarios or add
backend descriptors.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from types import MappingProxyType
from typing import Any

from core import ToolContext, ToolRegistry, ToolSpec
from core.protocol21_evidence import canonicalize_repo_owned_paths
from domains.power_grid.backends.opendss_ieee13 import (
    _NEW_OBJECT_RE,
    _native_protocol21_trace,
    _resolve_native_include_graph,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "works" / "OpenDSS-IEEE34-IEEE123"
# The IEEE34/IEEE123 test cases ship inside the same electricdss-tst clone as
# IEEE13 (works/OpenDSS-IEEE13), nested under Version8/Distrib/IEEETestCases/.
# Try the dedicated dir first, then fall back to the IEEE13 clone + nested path.
_IEEE13_CLONE = REPO_ROOT / "works" / "OpenDSS-IEEE13"
_NESTED_PREFIX = "Version8/Distrib/IEEETestCases"
VOLTAGE_LOWER_PU = 0.95
VOLTAGE_UPPER_PU = 1.05

FEEDER_MASTER_FILES = {
    "ieee34": "34Bus/ieee34Mod1.dss",
    "ieee123": "123Bus/IEEE123Master.dss",
    "ieee37": "37Bus/ieee37.dss",
}

_DSS_FILE_REFERENCE_RE = re.compile(
    r"""\b(?:csvfile|sngfile|dblfile|file)\s*=\s*(?:\(\s*)?
    (?:\"([^\"]+)\"|'([^']+)'|([^,\s\)!\]]+))""",
    re.IGNORECASE | re.VERBOSE,
)
_DSS_DUTY_REFERENCE_RE = re.compile(
    r"\bduty\s*=\s*(?:\"([^\"]+)\"|'([^']+)'|([^\s!]+))",
    re.IGNORECASE,
)
_DSS_KW_RE = re.compile(
    r"\bkw\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)
_DSS_SINTERVAL_RE = re.compile(
    r"\bsinterval\s*=\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
_DSS_INTERVAL_SECONDS_RE = re.compile(
    r"\binterval\s*=\s*\(\s*([0-9]+(?:\.[0-9]+)?)\s+3600\s*/\s*\)",
    re.IGNORECASE,
)
_DSS_NPTS_RE = re.compile(r"\bnpts\s*=\s*([0-9]+)", re.IGNORECASE)
_DSS_NORMALIZE_RE = re.compile(
    r"\baction\s*=\s*normalize\b",
    re.IGNORECASE,
)
_DSS_AUXILIARY_REFERENCE_RE = re.compile(
    r"""^\s*(buscoords|latlongcoords|guids)\s+
    (?:file\s*=\s*)?(?:\(\s*)?
    (?:"([^"]+)"|'([^']+)'|([^\s\)!]+))""",
    re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)
_DSS_SOLUTION_OPTION_RE = re.compile(
    r"(?<![a-z0-9_])"
    r"(?P<name>mo(?:d(?:e)?)?|num(?:b(?:e(?:r)?)?)?)\s*=\s*"
    r"(?:\((?P<paren>[^)]*)\)|\"(?P<double>[^\"]+)\"|"
    r"'(?P<single>[^']+)'|(?P<plain>[^\s!,;]+))",
    re.IGNORECASE,
)
_DSS_INCLUDE_ARGUMENT_RE = re.compile(
    r"^\s*\S+\s+(?:\(([^)]+)\)|\[([^]]+)\]|\"([^\"]+)\"|'([^']+)'|([^\s!]+))",
    re.IGNORECASE,
)
_DSS_TIME_SERIES_MODES = frozenset(
    {
        "daily",
        "duty",
        "dutycycle",
        "dynamic",
        "m1",
        "m2",
        "m3",
        "monte1",
        "monte2",
        "monte3",
        "ld1",
        "ld2",
        "peakday",
        "time",
        "yearly",
    }
)
_DSS_DIAGNOSTIC_MODES = frozenset(
    {"faultstudy", "harmonic", "harmonict", "mf"}
)
_DSS_SOLVE_NORMALIZATION_POLICY = "execution_ordered_solve_state_v3"
OPENDSS_EVENT_REGISTRY = MappingProxyType(
    {
        ("source_schedule", "load_change"): ("alarm", True),
        ("source_schedule", "generation_ramp"): ("alarm", True),
        ("procedural_perturbation", "load_surge"): ("alarm", True),
        ("procedural_perturbation", "line_outage"): ("safety", True),
        ("procedural_perturbation", "load_surge_cleared"): ("lifecycle", False),
        ("procedural_perturbation", "line_outage_cleared"): (
            "lifecycle",
            False,
        ),
    }
)


def _opendss_event_class(
    *,
    kind: str,
    origin: str,
    declared_class: Any = None,
) -> str:
    event_contract = OPENDSS_EVENT_REGISTRY.get((origin, kind))
    if event_contract is None:
        raise ValueError(f"unsupported OpenDSS event kind: {origin}/{kind or '<missing>'}")
    event_class, _ = event_contract
    if declared_class not in {None, "", event_class}:
        raise ValueError(
            "OpenDSS event class does not match registry: "
            f"{origin}/{kind}/{declared_class}"
        )
    return event_class


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _semantic_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_profile_values(
    path: Path,
    *,
    blank_rows_are_zero: bool = False,
) -> list[float]:
    """Parse the numeric OpenDSS LoadShape data file without synthesizing it."""
    values: list[float] = []
    text = path.read_text(encoding="utf-8", errors="strict")
    for raw_line in text.splitlines():
        tokens = raw_line.replace(",", " ").split()
        if not tokens and blank_rows_are_zero:
            values.append(0.0)
            continue
        for token in tokens:
            try:
                values.append(float(token))
            except ValueError as exc:
                raise ValueError(
                    f"invalid numeric LoadShape value in {path}: {token}"
                ) from exc
    if not values:
        raise ValueError(f"OpenDSS LoadShape data file is empty: {path}")
    return values


def _same_or_casefolded_path(first: Path, second: Path) -> bool:
    """Allow only a same-directory case spelling difference from source text."""
    if first == second:
        return True
    return (
        first.parent.resolve() == second.parent.resolve()
        and first.name.casefold() == second.name.casefold()
    )


def _direct_compile_inputs(
    source_file: Path,
    text: str,
    *,
    windows_aliases: dict[str, Path] | None = None,
) -> set[Path]:
    """Resolve only Compile commands that the selected program declares.

    A native duty program is parsed but deliberately not executed wholesale.
    Its presentation, auxiliary, and unselected profile references therefore
    are not runtime inputs; following their full include graph would impose
    dependencies that the backend never consumes.
    """
    inputs: set[Path] = set()
    for raw_line in text.splitlines():
        line = _strip_dss_comment(raw_line)
        if _dss_command(line) != "compile":
            continue
        match = _DSS_INCLUDE_ARGUMENT_RE.match(line)
        if match is None:
            raise ValueError("OpenDSS native_duty_program Compile is malformed")
        declared = next(value for value in match.groups() if value is not None).strip()
        if re.match(r"^[A-Za-z]:[\\\\/]", declared):
            alias = (windows_aliases or {}).get(
                PureWindowsPath(declared).name.lower()
            )
            if alias is None:
                raise ValueError(
                    "OpenDSS native_duty_program Compile path is not portable"
                )
            inputs.add(alias.resolve())
        else:
            inputs.add((source_file.parent / declared).resolve())
    return inputs


def _strip_dss_comment(line: str) -> str:
    quote: str | None = None
    index = 0
    while index < len(line):
        char = line[index]
        if quote is not None:
            if char == quote:
                quote = None
        elif char in {'"', "'"}:
            quote = char
        elif char == "!" or line[index : index + 2] == "//":
            return line[:index]
        index += 1
    return line


def _read_dss_text(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="strict", newline="") as handle:
        return handle.read()


def _write_dss_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", errors="strict", newline="") as handle:
        handle.write(text)


@dataclass
class _DssSolveState:
    mode: str = "snapshot"
    number: int = 1


def _empty_solve_policy() -> dict[str, Any]:
    return {
        "normalization_policy": _DSS_SOLVE_NORMALIZATION_POLICY,
        "preserved_solve_count": 0,
        "transformed_solve_count": 0,
        "transformed_solve_reason_counts": {
            "explicit_multiple_steps": 0,
            "time_series_mode": 0,
        },
    }


def _merge_solve_policy(target: dict[str, Any], source: dict[str, Any]) -> None:
    target["preserved_solve_count"] += int(source["preserved_solve_count"])
    target["transformed_solve_count"] += int(source["transformed_solve_count"])
    for reason, count in source["transformed_solve_reason_counts"].items():
        target["transformed_solve_reason_counts"][reason] += int(count)


def _dss_command(command: str) -> str | None:
    match = re.match(r"^\s*([^\s=]+)", command)
    if match is None:
        return None
    token = match.group(1).lower()
    if "solveall".startswith(token) and len(token) >= 6:
        return "solveall"
    if "solve".startswith(token) and len(token) >= 2:
        return "solve"
    if "set".startswith(token) and len(token) >= 3:
        return "set"
    if "clearall".startswith(token) and len(token) >= 6:
        return "clearall"
    if "clear".startswith(token) and len(token) >= 3:
        return "clear"
    if "redirect".startswith(token) and len(token) >= 3:
        return "redirect"
    if "compile".startswith(token) and len(token) >= 4:
        return "compile"
    return None


def _canonical_dss_mode(value: str) -> str:
    token = re.sub(r"[^a-z0-9]", "", value.lower())
    aliases = {
        "f": "faultstudy",
        "fa": "faultstudy",
        "fau": "faultstudy",
        "faul": "faultstudy",
        "fault": "faultstudy",
        "faults": "faultstudy",
        "faultst": "faultstudy",
        "faultstu": "faultstudy",
        "faultstud": "faultstudy",
        "faultstudy": "faultstudy",
        "day": "daily",
        "dai": "daily",
        "dail": "daily",
        "daily": "daily",
        "year": "yearly",
        "yearl": "yearly",
        "yearly": "yearly",
        "duty": "dutycycle",
        "dutyc": "dutycycle",
        "dutycy": "dutycycle",
        "dutycyc": "dutycycle",
        "dutycycl": "dutycycle",
        "dutycycle": "dutycycle",
        "snap": "snapshot",
        "snaps": "snapshot",
        "snapsh": "snapshot",
        "snapsho": "snapshot",
        "snapshot": "snapshot",
        "harm": "harmonic",
        "harmo": "harmonic",
        "harmon": "harmonic",
        "harmoni": "harmonic",
        "harmonic": "harmonic",
        "harmonics": "harmonic",
        "harmonict": "harmonict",
        "dyn": "dynamic",
        "dyna": "dynamic",
        "dynam": "dynamic",
        "dynami": "dynamic",
        "dynamic": "dynamic",
        "dynamics": "dynamic",
    }
    return aliases.get(token, token)


def _dss_integer(value: str) -> int:
    tokens = value.split()
    if len(tokens) == 1:
        return int(tokens[0])
    stack: list[float] = []
    for token in tokens:
        if token not in {"+", "-", "*", "/"}:
            stack.append(float(token))
            continue
        if len(stack) < 2:
            raise ValueError(value)
        right = stack.pop()
        left = stack.pop()
        stack.append(
            left + right
            if token == "+"
            else left - right
            if token == "-"
            else left * right
            if token == "*"
            else left / right
        )
    if len(stack) != 1 or not stack[0].is_integer():
        raise ValueError(value)
    return int(stack[0])


def _apply_solution_options(command: str, state: _DssSolveState) -> None:
    for match in _DSS_SOLUTION_OPTION_RE.finditer(command):
        name = match.group("name").lower()
        value = next(
            candidate
            for candidate in (
                match.group("paren"),
                match.group("double"),
                match.group("single"),
                match.group("plain"),
            )
            if candidate is not None
        )
        if name.startswith("mo"):
            state.mode = _canonical_dss_mode(value)
            # OpenDSS resets Number when Mode changes.  The policy tracks one
            # step unless the source subsequently declares a larger Number;
            # time-series modes are bounded independently of their defaults.
            state.number = 1
        else:
            try:
                state.number = _dss_integer(value)
            except ValueError as exc:
                raise ValueError(f"invalid OpenDSS Number option: {value}") from exc


def _normalize_runtime_solve_commands(text: str) -> str:
    """Bound multi-step source solves in one execution-ordered DSS program."""
    return _normalize_runtime_solve_commands_with_policy(text)[0]


def _normalize_runtime_solve_commands_with_count(text: str) -> tuple[str, int]:
    normalized, policy = _normalize_runtime_solve_commands_with_policy(text)
    return normalized, int(policy["transformed_solve_count"])


def _normalize_runtime_solve_commands_with_policy(
    text: str,
) -> tuple[str, dict[str, Any]]:
    return _normalize_runtime_solve_program(text, _DssSolveState())


def _normalize_runtime_solve_program(
    text: str,
    state: _DssSolveState,
    *,
    include: Any = None,
) -> tuple[str, dict[str, Any]]:
    normalized: list[str] = []
    policy = _empty_solve_policy()
    for line in text.splitlines(keepends=True):
        newline = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
        body = line[: -len(newline)] if newline else line
        command = _strip_dss_comment(body)
        command_name = _dss_command(command)
        if command_name in {"clear", "clearall"} or re.match(
            r"^\s*new\s+(?:object\s*=\s*)?[\"']?circuit\.",
            command,
            re.IGNORECASE,
        ):
            state.mode = "snapshot"
            state.number = 1
        if command_name == "set":
            _apply_solution_options(command, state)
        if command_name in {"redirect", "compile"} and include is not None:
            include_match = _DSS_INCLUDE_ARGUMENT_RE.match(command)
            if include_match is None:
                raise ValueError(f"unparseable OpenDSS include command: {command.strip()}")
            include(next(value for value in include_match.groups() if value))
        if command_name not in {"solve", "solveall"}:
            normalized.append(line)
            continue

        _apply_solution_options(command, state)
        reasons: list[str] = []
        if state.mode in _DSS_TIME_SERIES_MODES:
            reasons.append("time_series_mode")
        if state.number > 1 and state.mode not in _DSS_DIAGNOSTIC_MODES:
            reasons.append("explicit_multiple_steps")
        if not reasons:
            policy["preserved_solve_count"] += 1
            normalized.append(line)
            continue

        indent = body[: len(body) - len(body.lstrip())]
        comment = body[len(command) :]
        replacement = f"{indent}Solve mode=snapshot number=1{comment}{newline}"
        normalized.append(replacement)
        policy["transformed_solve_count"] += replacement != line
        for reason in set(reasons):
            policy["transformed_solve_reason_counts"][reason] += 1
        state.mode = "snapshot"
        state.number = 1
    return "".join(normalized), policy


def _merge_source_assets(
    *asset_groups: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Deduplicate source inputs by resolved path while retaining first role."""
    merged: dict[str, dict[str, str]] = {}
    for group in asset_groups:
        for asset in group:
            existing = merged.setdefault(str(asset["path"]), dict(asset))
            for key, value in asset.items():
                existing.setdefault(key, value)
    return [merged[path] for path in sorted(merged)]


def _resolve_master_file(source_root: Path, master_rel: str) -> Path:
    """Resolve a feeder master file, checking (1) the dedicated
    ``works/OpenDSS-IEEE34-IEEE123`` dir, (2) the same dir + the nested
    electricdss-tst prefix, (3) the IEEE13 clone flat, (4) the IEEE13 clone
    nested. Returns the first existing candidate; falls back to the dedicated
    flat path so the error message stays canonical."""
    candidates = [
        source_root / master_rel,
        source_root / _NESTED_PREFIX / master_rel,
        _IEEE13_CLONE / master_rel,
        _IEEE13_CLONE / _NESTED_PREFIX / master_rel,
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _runtime_solve_graph(
    master_file: Path,
    assets: list[dict[str, str]],
) -> tuple[dict[Path, str], dict[Path, dict[str, Any]]]:
    """Interpret DSS command files in the same nested execution order as OpenDSS."""
    command_roles = {"compile_master", "compile_input", "redirect_input"}
    command_paths = {
        Path(str(asset["path"])).resolve()
        for asset in assets
        if str(asset.get("role")) in command_roles
    }
    master = master_file.resolve()
    if master not in command_paths:
        raise ValueError("OpenDSS command graph does not contain its master file")
    normalized_by_path: dict[Path, str] = {}
    policy_by_path = {path: _empty_solve_policy() for path in command_paths}
    active: set[Path] = set()
    state = _DssSolveState()

    def execute(path: Path) -> None:
        path = path.resolve()
        if path not in command_paths:
            raise ValueError(f"OpenDSS include is not locked in compile graph: {path}")
        if path in active:
            raise ValueError(f"cyclic OpenDSS include graph: {path}")
        active.add(path)
        try:
            source_text = _read_dss_text(path)

            def include(raw: str) -> None:
                normalized = raw.strip().replace("\\", "/")
                if re.match(r"^[a-z]:/", normalized, re.IGNORECASE):
                    raise ValueError(
                        f"OpenDSS include escapes portable compile graph: {raw}"
                    )
                execute((path.parent / normalized).resolve())

            normalized_text, policy = _normalize_runtime_solve_program(
                source_text,
                state,
                include=include,
            )
            prior = normalized_by_path.setdefault(path, normalized_text)
            if prior != normalized_text:
                raise ValueError(
                    "OpenDSS command file requires context-dependent mirror edits: "
                    f"{path}"
                )
            _merge_solve_policy(policy_by_path[path], policy)
        finally:
            active.remove(path)

    execute(master)
    return normalized_by_path, policy_by_path


def _isolated_compile_graph(
    master_file: Path,
    assets: list[dict[str, str]],
) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    """Mirror authoritative DSS inputs into an episode-private runtime tree."""
    source_paths = [Path(str(asset["path"])).resolve() for asset in assets]
    if master_file.resolve() not in source_paths:
        raise ValueError("OpenDSS compile graph does not contain its master file")
    common_root = Path(os.path.commonpath([str(path) for path in source_paths]))
    if common_root.is_file():
        common_root = common_root.parent
    auxiliary_assets: list[dict[str, str]] = []
    for source in source_paths:
        expected_sha = str(
            next(
                asset["sha256"]
                for asset in assets
                if Path(str(asset["path"])).resolve() == source
            )
        ).removeprefix("sha256:")
        if not source.is_file():
            raise FileNotFoundError(f"missing OpenDSS compile input: {source}")
        if not expected_sha or _sha256(source) != expected_sha:
            raise ValueError(f"OpenDSS compile input hash mismatch: {source}")
        if source.suffix.lower() != ".dss":
            continue
        text = "\n".join(
            _strip_dss_comment(line)
            for line in source.read_text(encoding="utf-8", errors="strict").splitlines()
        )
        for match in _DSS_AUXILIARY_REFERENCE_RE.finditer(text):
            raw = next(value for value in match.groups()[1:] if value)
            auxiliary = (source.parent / raw.strip()).resolve()
            if not auxiliary.is_relative_to(common_root):
                raise ValueError(
                    f"OpenDSS auxiliary input escapes compile graph root: {raw}"
                )
            if not auxiliary.is_file():
                raise FileNotFoundError(f"missing OpenDSS auxiliary input: {auxiliary}")
            auxiliary_assets.append(
                {
                    "path": str(auxiliary),
                    "sha256": _sha256(auxiliary),
                    "role": "runtime_auxiliary_input",
                    "included_from": str(source),
                    "directive": str(match.group(1)).lower(),
                }
            )
        for match in _DSS_FILE_REFERENCE_RE.finditer(text):
            raw = next(value for value in match.groups() if value)
            normalized = raw.strip().replace("\\", "/")
            if re.match(r"^[a-z]:/", normalized, re.IGNORECASE):
                raise ValueError(
                    f"OpenDSS runtime input escapes compile graph root: {raw}"
                )
            referenced = (source.parent / normalized).resolve()
            if not referenced.is_relative_to(common_root):
                raise ValueError(
                    f"OpenDSS runtime input escapes compile graph root: {raw}"
                )
            if not referenced.is_file():
                raise FileNotFoundError(f"missing OpenDSS runtime input: {referenced}")
            auxiliary_assets.append(
                {
                    "path": str(referenced),
                    "sha256": _sha256(referenced),
                    "role": "runtime_data_input",
                    "included_from": str(source),
                    "directive": "file",
                }
            )
    assets[:] = _merge_source_assets(assets, auxiliary_assets)
    source_paths = [Path(str(asset["path"])).resolve() for asset in assets]
    normalized_by_path, _policy_by_path = _runtime_solve_graph(master_file, assets)
    runtime_dir = tempfile.TemporaryDirectory(prefix="operate-opendss-")
    runtime_root = Path(runtime_dir.name) / "source"
    for source in source_paths:
        relative = source.relative_to(common_root)
        target = runtime_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        normalized_text = normalized_by_path.get(source)
        if normalized_text is not None and normalized_text != _read_dss_text(target):
            _write_dss_text(target, normalized_text)
    isolated_master = runtime_root / master_file.resolve().relative_to(common_root)
    return runtime_dir, isolated_master


def _execution_mirror_policy(
    assets: list[dict[str, str]],
    runtime_directory: tempfile.TemporaryDirectory[str],
) -> dict[str, Any]:
    source_paths = sorted(Path(str(asset["path"])).resolve() for asset in assets)
    common_root = Path(os.path.commonpath([str(path) for path in source_paths]))
    if common_root.is_file():
        common_root = common_root.parent
    runtime_root = Path(runtime_directory.name) / "source"
    authoritative_hashes: dict[str, str] = {}
    mirror_hashes: dict[str, str] = {}
    original_to_mirror: dict[str, dict[str, Any]] = {}
    digest_rows: list[dict[str, Any]] = []
    transformed_solve_count = 0
    preserved_solve_count = 0
    transformed_solve_reason_counts = {
        "explicit_multiple_steps": 0,
        "time_series_mode": 0,
    }
    asset_by_path = {
        Path(str(asset["path"])).resolve(): asset for asset in assets
    }
    master_files = [
        source
        for source, asset in asset_by_path.items()
        if str(asset.get("role")) == "compile_master"
    ]
    if len(master_files) != 1:
        raise ValueError("OpenDSS execution mirror requires one compile master")
    normalized_by_path, policy_by_path = _runtime_solve_graph(
        master_files[0], assets
    )
    for source in source_paths:
        relative = source.relative_to(common_root).as_posix()
        executed = runtime_root / relative
        if not executed.is_file():
            raise FileNotFoundError(f"missing OpenDSS execution mirror input: {executed}")
        authoritative_sha = str(asset_by_path[source]["sha256"]).removeprefix(
            "sha256:"
        )
        if _sha256(source) != authoritative_sha:
            raise ValueError(f"OpenDSS authoritative input hash mismatch: {source}")
        executed_sha = _sha256(executed)
        solve_policy = policy_by_path.get(source, _empty_solve_policy())
        expected_text = normalized_by_path.get(source)
        if expected_text is not None:
            if _read_dss_text(executed) != expected_text:
                raise ValueError(f"OpenDSS execution mirror mismatch: {relative}")
        transformed_solve_count += int(solve_policy["transformed_solve_count"])
        preserved_solve_count += int(solve_policy["preserved_solve_count"])
        reason_counts = solve_policy["transformed_solve_reason_counts"]
        for reason in transformed_solve_reason_counts:
            transformed_solve_reason_counts[reason] += int(reason_counts[reason])
        original_path = canonicalize_repo_owned_paths(
            str(source), repo_root=REPO_ROOT
        )
        authoritative_hashes[original_path] = authoritative_sha
        mirror_hashes[relative] = executed_sha
        original_to_mirror[original_path] = {
            "executed_relative_path": relative,
            "authoritative_sha256": authoritative_sha,
            "executed_sha256": executed_sha,
            **solve_policy,
        }
        digest_rows.append(
            {
                "executed_relative_path": relative,
                "authoritative_sha256": authoritative_sha,
                "executed_sha256": executed_sha,
                **solve_policy,
            }
        )
    return {
        "authoritative_source_hashes": authoritative_hashes,
        "execution_mirror_hashes": mirror_hashes,
        "original_to_execution_mirror": original_to_mirror,
        "execution_mirror_digest": _semantic_digest(digest_rows),
        "normalization_policy": _DSS_SOLVE_NORMALIZATION_POLICY,
        "preserved_solve_count": preserved_solve_count,
        "transformed_solve_count": transformed_solve_count,
        "transformed_solve_reason_counts": transformed_solve_reason_counts,
    }


class OpenDssRuntimeUnavailable(RuntimeError):
    """Raised when ``dss-python`` is not importable."""


@dataclass
class OpenDssFreshFeederSolveSummary:
    feeder: str
    converged: bool
    n_buses: int
    n_nodes: int
    n_loads: int
    n_lines: int
    n_transformers: int
    n_capacitors: int
    n_regcontrols: int
    voltage_min_pu: float | None
    voltage_max_pu: float | None
    n_voltage_violations: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "feeder": self.feeder,
            "converged": self.converged,
            "n_buses": self.n_buses,
            "n_nodes": self.n_nodes,
            "n_loads": self.n_loads,
            "n_lines": self.n_lines,
            "n_transformers": self.n_transformers,
            "n_capacitors": self.n_capacitors,
            "n_regcontrols": self.n_regcontrols,
            "voltage_min_pu": self.voltage_min_pu,
            "voltage_max_pu": self.voltage_max_pu,
            "n_voltage_violations": self.n_voltage_violations,
        }


class OpenDssFreshFeederProbeBackend:
    """Minimal OpenDSS live adapter for a pinned feeder entry graph."""

    backend_kind = "opendss_fresh_feeder_probe"

    def __init__(
        self,
        *,
        source_root: Path = DEFAULT_SOURCE_ROOT,
        feeder: str,
        master_file: str | None = None,
    ) -> None:
        if feeder not in FEEDER_MASTER_FILES and not master_file:
            raise ValueError(
                f"unknown OpenDSS fresh feeder {feeder!r} requires explicit master_file"
            )
        self.source_root = Path(source_root)
        self.feeder = feeder
        self.master_file = _resolve_master_file(
            self.source_root,
            master_file or FEEDER_MASTER_FILES[feeder],
        )
        self._dss: Any = None
        self._circuit: Any = None
        self._dss_version: str | None = None
        self._runtime_source_assets: list[dict[str, str]] = []
        self._parsed_source_inventory: dict[str, int] = {}
        self._protocol21_source_evidence: dict[str, Any] | None = None
        self._source_profile_active = False
        self._source_profile_values: list[float] = []
        self._runtime_directory: tempfile.TemporaryDirectory[str] | None = None
        self._runtime_execution_policy: dict[str, Any] = {}

    @property
    def dss_version(self) -> str | None:
        return self._dss_version

    def reset(self) -> OpenDssFreshFeederSolveSummary:
        caller_cwd = Path.cwd()
        self._protocol21_source_evidence = None
        self._runtime_source_assets = []
        self._parsed_source_inventory = {}
        self._circuit = None
        self._source_profile_active = False
        self._source_profile_values = []
        self._runtime_execution_policy = {}
        if self._runtime_directory is not None:
            self._runtime_directory.cleanup()
            self._runtime_directory = None
        try:
            dss_module = _import_dss()
        finally:
            os.chdir(caller_cwd)
        self._dss_version = str(getattr(dss_module, "__version__", "unknown"))
        if not self.master_file.exists():
            raise FileNotFoundError(f"missing OpenDSS master file: {self.master_file}")
        (
            self._runtime_source_assets,
            self._parsed_source_inventory,
        ) = _resolve_native_include_graph(self.master_file)
        self._runtime_directory, isolated_master = _isolated_compile_graph(
            self.master_file,
            self._runtime_source_assets,
        )
        try:
            execution_mirror_policy = _execution_mirror_policy(
                self._runtime_source_assets,
                self._runtime_directory,
            )
            runtime_root = Path(self._runtime_directory.name)
            runtime_output = runtime_root / "output"
            runtime_output.mkdir()
            try:
                # OpenDSS scripts may contain DataPath/Export directives and
                # the engine itself can mutate the host cwd.  Execute the
                # complete initial load inside the episode-private tree and
                # restore the caller even when a source script fails.
                os.chdir(runtime_root)
                self._dss = dss_module.DSS.NewContext()
                self._dss.Text.Command = "Clear"
                self._dss.Text.Command = f'Set DataPath="{runtime_output}"'
                self._dss.Text.Command = f'Redirect "{isolated_master}"'
                self._dss.Text.Command = "Set ControlMode=OFF"
                self._circuit = self._dss.ActiveCircuit
                solution = self._circuit.Solution
                tick_minutes = float(
                    getattr(getattr(self, "_seed_obj", None), "tick_minutes", 1.0)
                    or 1.0
                )
                if tick_minutes <= 0.0:
                    raise ValueError("OpenDSS tick_minutes must be positive")
                solution.Mode = 0
                solution.Number = 1
                solution.StepSize = tick_minutes * 60.0
                self._runtime_execution_policy = {
                    **execution_mirror_policy,
                    "source_graph": "locked_definitions_preserved",
                    "source_solve_commands": (
                        "time_series_or_multiple_steps_normalized_to_snapshot"
                    ),
                    "mode": "snapshot",
                    "number": 1,
                    "step_seconds": tick_minutes * 60.0,
                }
                summary = self.solve()
            finally:
                os.chdir(caller_cwd)
        except Exception:
            self._runtime_directory.cleanup()
            self._runtime_directory = None
            raise
        self._protocol21_source_evidence = _native_protocol21_trace(
            assets=self._runtime_source_assets,
            inventory=self._parsed_source_inventory,
            dss_version=self._dss_version,
            circuit=self._circuit,
            summary=summary.to_dict(),
        )
        self._protocol21_source_evidence["runtime_execution_policy"] = dict(
            self._runtime_execution_policy
        )
        return summary

    def solve(self) -> OpenDssFreshFeederSolveSummary:
        circuit = self._require_circuit()
        circuit.Solution.SolveSnap()
        return self._summary_from_circuit()

    def close(self) -> None:
        self._circuit = None
        self._dss = None
        if self._runtime_directory is not None:
            self._runtime_directory.cleanup()
            self._runtime_directory = None

    def _summary_from_circuit(self) -> OpenDssFreshFeederSolveSummary:
        circuit = self._require_circuit()
        voltages = [float(v) for v in circuit.AllBusVmagPu]
        return OpenDssFreshFeederSolveSummary(
            feeder=self.feeder,
            converged=bool(circuit.Solution.Converged),
            n_buses=int(circuit.NumBuses),
            n_nodes=int(circuit.NumNodes),
            n_loads=int(circuit.Loads.Count),
            n_lines=int(circuit.Lines.Count),
            n_transformers=int(circuit.Transformers.Count),
            n_capacitors=int(circuit.Capacitors.Count),
            n_regcontrols=int(circuit.RegControls.Count),
            voltage_min_pu=min(voltages) if voltages else None,
            voltage_max_pu=max(voltages) if voltages else None,
            n_voltage_violations=sum(
                1 for v in voltages if v < VOLTAGE_LOWER_PU or v > VOLTAGE_UPPER_PU
            ),
        )

    def snapshot(self) -> dict[str, Any]:
        circuit = self._require_circuit()
        summary = self.solve().to_dict()
        return {
            "backend_kind": self.backend_kind,
            "feeder": self.feeder,
            "dss_version": self._dss_version,
            "master_file": str(self.master_file),
            "circuit_name": str(circuit.Name),
            **summary,
            "capacitors": self._capacitor_states(),
            "regcontrols": self._regcontrol_states(),
            "line_current_max_a": self._line_current_max_a(),
        }

    def protocol21_source_trace(self) -> dict[str, Any]:
        """Return solver-linked evidence for the actual compiled DSS graph."""
        if self._protocol21_source_evidence is None:
            return {
                "status": "held",
                "proof_kind": "native_include_graph",
                "runtime_trace_observed": False,
                "evidence_from_scenario_config_only": False,
                "source_state_effect_observed": False,
                "state_effect_observed": False,
                "blockers": ["backend_not_reset"],
            }
        return dict(self._protocol21_source_evidence)

    def apply_tool_effect(
        self, name: str, args: dict[str, Any], ctx: ToolContext | None = None
    ) -> dict[str, Any]:
        if name == "switch_capacitor":
            return self._with_tool_evidence(name, self._switch_capacitor(args), ctx)
        if name == "set_transformer_tap":
            return self._with_tool_evidence(name, self._set_regulator_tap(args), ctx)
        if name == "switch_branch":
            return self._with_tool_evidence(name, self._switch_branch(args), ctx)
        return self._with_tool_evidence(
            name, {"_status": "error", "error": "unknown_tool", "tool": name}, ctx
        )

    def _switch_branch(self, args: dict[str, Any]) -> dict[str, Any]:
        circuit = self._require_circuit()
        line_index = int(args.get("line_index", -1))
        connect = bool(args.get("connect"))
        names = [str(name) for name in circuit.Lines.AllNames]
        if line_index < 0 or line_index >= len(names):
            return {
                "_status": "error",
                "error": "unknown_controllable_asset",
                "asset": "line_index",
                "index": line_index,
                "n_available": len(names),
            }
        name = names[line_index]
        circuit.SetActiveElement(f"Line.{name}")
        before = bool(circuit.ActiveCktElement.Enabled)
        circuit.ActiveCktElement.Enabled = connect
        summary = self.solve()
        circuit.SetActiveElement(f"Line.{name}")
        after = bool(circuit.ActiveCktElement.Enabled)
        return {
            "_status": "applied" if before != after else "no_effect",
            "feeder": self.feeder,
            "line_index": line_index,
            "line": name,
            "connect": connect,
            "in_service_before": before,
            "in_service_after": after,
            "converged_after_solve": summary.converged,
            "voltage_min_pu": summary.voltage_min_pu,
            "voltage_max_pu": summary.voltage_max_pu,
            "n_voltage_violations": summary.n_voltage_violations,
        }

    def _switch_capacitor(self, args: dict[str, Any]) -> dict[str, Any]:
        circuit = self._require_circuit()
        cap_id = int(args.get("cap_id", -1))
        status = bool(args.get("status"))
        names = [str(name) for name in circuit.Capacitors.AllNames]
        if cap_id < 0 or cap_id >= len(names):
            return {
                "_status": "error",
                "error": "unknown_controllable_asset",
                "asset": "cap_id",
                "index": cap_id,
                "n_available": len(names),
            }
        name = names[cap_id]
        circuit.Capacitors.Name = name
        before = [int(state) for state in circuit.Capacitors.States]
        circuit.Capacitors.States = [1 if status else 0 for _ in before]
        summary = self.solve()
        after = [int(state) for state in circuit.Capacitors.States]
        return {
            "_status": "applied" if before != after else "no_effect",
            "feeder": self.feeder,
            "cap_id": cap_id,
            "capacitor": name,
            "status": status,
            "states_before": before,
            "states_after": after,
            "converged_after_solve": summary.converged,
            "voltage_min_pu": summary.voltage_min_pu,
            "voltage_max_pu": summary.voltage_max_pu,
            "n_voltage_violations": summary.n_voltage_violations,
        }

    def _set_regulator_tap(self, args: dict[str, Any]) -> dict[str, Any]:
        circuit = self._require_circuit()
        reg_id = int(args.get("trafo_id", -1))
        tap_pos = int(args.get("tap_pos", 0))
        names = [str(name) for name in circuit.RegControls.AllNames]
        if reg_id < 0 or reg_id >= len(names):
            return {
                "_status": "error",
                "error": "unknown_controllable_asset",
                "asset": "trafo_id",
                "index": reg_id,
                "n_available": len(names),
            }
        name = names[reg_id]
        circuit.RegControls.Name = name
        tap_min, tap_max = self._regulator_tap_bounds()
        if not tap_min <= tap_pos <= tap_max:
            return {
                "_status": "out_of_range",
                "error": "tap_position_out_of_range",
                "feeder": self.feeder,
                "trafo_id": reg_id,
                "regcontrol": name,
                "tap_pos": tap_pos,
                "tap_min": tap_min,
                "tap_max": tap_max,
            }
        before = int(circuit.RegControls.TapNumber)
        circuit.RegControls.TapNumber = tap_pos
        summary = self.solve()
        after = int(circuit.RegControls.TapNumber)
        return {
            "_status": "applied" if before != after else "no_effect",
            "feeder": self.feeder,
            "trafo_id": reg_id,
            "regcontrol": name,
            "tap_before": before,
            "tap_after": after,
            "converged_after_solve": summary.converged,
            "voltage_min_pu": summary.voltage_min_pu,
            "voltage_max_pu": summary.voltage_max_pu,
            "n_voltage_violations": summary.n_voltage_violations,
        }

    def _capacitor_states(self) -> list[dict[str, Any]]:
        circuit = self._require_circuit()
        out: list[dict[str, Any]] = []
        idx = circuit.Capacitors.First
        while idx:
            out.append(
                {
                    "cap_id": len(out),
                    "name": str(circuit.Capacitors.Name),
                    "states": [int(s) for s in circuit.Capacitors.States],
                }
            )
            idx = circuit.Capacitors.Next
        return out

    def _regcontrol_states(self) -> list[dict[str, Any]]:
        circuit = self._require_circuit()
        out: list[dict[str, Any]] = []
        idx = circuit.RegControls.First
        while idx:
            out.append(
                {
                    "trafo_id": len(out),
                    "name": str(circuit.RegControls.Name),
                    "transformer": str(circuit.RegControls.Transformer),
                    "tap_number": int(circuit.RegControls.TapNumber),
                    "tap_min": self._regulator_tap_bounds()[0],
                    "tap_max": self._regulator_tap_bounds()[1],
                }
            )
            idx = circuit.RegControls.Next
        return out

    def _regulator_tap_bounds(self) -> tuple[int, int]:
        """Return the active regulator's native symmetric tap-number bounds."""
        circuit = self._require_circuit()
        transformer = str(circuit.RegControls.Transformer)
        if not transformer:
            raise RuntimeError("OpenDSS regulator control has no transformer")
        circuit.Transformers.Name = transformer
        circuit.Transformers.Wdg = int(circuit.RegControls.TapWinding)
        n_taps = int(circuit.Transformers.NumTaps)
        if n_taps <= 0:
            raise RuntimeError("OpenDSS regulator transformer has no tap positions")
        half_span = n_taps // 2
        return -half_span, half_span

    def _line_current_max_a(self) -> float | None:
        circuit = self._require_circuit()
        maxima: list[float] = []
        idx = circuit.Lines.First
        while idx:
            circuit.SetActiveElement(f"Line.{circuit.Lines.Name}")
            currents = [float(v) for v in circuit.ActiveCktElement.CurrentsMagAng]
            maxima.extend(currents[0::2])
            idx = circuit.Lines.Next
        return max(maxima) if maxima else None

    def _line_states(self) -> list[dict[str, Any]]:
        circuit = self._require_circuit()
        states: list[dict[str, Any]] = []
        # ``Lines.First`` skips the collection entirely when a one-line test
        # feeder has that line disabled.  ``AllNames`` retains disabled native
        # elements, which is exactly the outage state the agent must observe.
        for name_value in circuit.Lines.AllNames:
            name = str(name_value)
            circuit.SetActiveElement(f"Line.{name}")
            in_service = bool(circuit.ActiveCktElement.Enabled)
            baseline_in_service = self._base_line_enabled.get(name, in_service)
            states.append(
                {
                    "line_index": len(states),
                    "name": name,
                    "in_service": in_service,
                    "baseline_in_service": baseline_in_service,
                    "unexpectedly_disconnected": (
                        baseline_in_service and not in_service
                    ),
                }
            )
        return states

    def _require_circuit(self) -> Any:
        if self._circuit is None:
            raise RuntimeError("OpenDSS fresh-feeder backend has not been reset")
        return self._circuit

    def _with_tool_evidence(
        self, tool_name: str, payload: dict[str, Any], ctx: ToolContext | None
    ) -> dict[str, Any]:
        evidence = (
            getattr(ctx, "extra", {}).get("evidence") if ctx is not None else None
        )
        if evidence is None or not hasattr(evidence, "log"):
            return payload
        tick = int(getattr(ctx, "tick", 0) or 0)
        out = dict(payload)
        out["evidence_id"] = evidence.log(
            kind="opendss_tool_effect",
            tick=tick,
            payload={
                "backend_kind": self.backend_kind,
                "feeder": self.feeder,
                "tool": tool_name,
                "ok": out.get("_status") == "applied",
                "state_changing": out.get("_status") == "applied",
                "result": {k: v for k, v in out.items() if k != "evidence_id"},
            },
            source="tool",
        )
        return out


class OpenDssFreshFeedersBackend(OpenDssFreshFeederProbeBackend):
    """Released OpenDSS IEEE34/IEEE123 fresh-feeder backend facade.

    Promotes the live probe to a release backend usable by
    ``PowerGridEnvironment``. The feeder, master file, and source root are
    resolved from the scenario seed's ``backend_config`` at reset time, mirroring
    the released IEEE13 backend's seed-driven facade.
    """

    backend_kind = "opendss_fresh_feeders"

    def __init__(
        self,
        *,
        source_root: Path = DEFAULT_SOURCE_ROOT,
        feeder: str = "ieee34",
        master_file: str | None = None,
    ) -> None:
        super().__init__(
            source_root=source_root, feeder=feeder, master_file=master_file
        )
        self._seed_obj: Any | None = None
        self._tick = 0
        self._horizon = 3
        self._tick_records: list[dict[str, Any]] = []
        self._cumulative_cost_components: dict[str, float] = {}
        self._source_profile: dict[str, Any] | None = None
        # Deterministic procedural events are layered on the locked feeder
        # source.  They never replace source consumption evidence: the
        # compiled DSS graph remains the physical anchor, while the event
        # itself is logged with an explicit response window.
        self._perturbations: list[dict[str, Any]] = []
        self._base_load_values: dict[str, tuple[float, float]] = {}
        self._base_line_enabled: dict[str, bool] = {}
        self._active_load_surge_indices: set[int] = set()
        self._active_line_outage_indices: set[int] = set()
        self._initial_native_controls: dict[str, Any] = {
            "controls": [],
            "applied_count": 0,
            "state_changing": False,
        }

    def reset(self, scenario_seed: Any | None = None) -> OpenDssFreshFeederSolveSummary:
        self._seed_obj = scenario_seed
        self._tick = 0
        self._tick_records = []
        self._cumulative_cost_components = {}
        self._source_profile = None
        self._perturbations = []
        self._base_load_values = {}
        self._base_line_enabled = {}
        self._active_load_surge_indices = set()
        self._active_line_outage_indices = set()
        self._initial_native_controls = {
            "controls": [],
            "applied_count": 0,
            "state_changing": False,
        }
        if scenario_seed is not None:
            self._horizon = int(getattr(scenario_seed, "horizon_ticks", 3) or 3)
            self._perturbations = [
                self._perturbation_dict(event)
                for event in (getattr(scenario_seed, "perturbations", []) or [])
                if isinstance(event, dict) or hasattr(event, "kind")
            ]
            for event in self._perturbations:
                _opendss_event_class(
                    kind=str(event.get("kind") or ""),
                    origin="procedural_perturbation",
                    declared_class=event.get("event_class", event.get("class")),
                )
            bc = getattr(scenario_seed, "backend_config", {}) or {}
            feeder = bc.get("feeder")
            configured_master = bc.get("master_file")
            if feeder:
                if feeder not in FEEDER_MASTER_FILES and not configured_master:
                    raise ValueError(
                        f"unknown OpenDSS feeder {feeder!r} requires master_file"
                    )
                self.feeder = str(feeder)
            source_root = bc.get("source_root")
            if source_root:
                self.source_root = Path(source_root)
            master_file = configured_master or FEEDER_MASTER_FILES.get(self.feeder)
            if not master_file:
                raise ValueError(f"OpenDSS feeder {self.feeder!r} requires master_file")
            self.master_file = _resolve_master_file(self.source_root, master_file)
        summary = super().reset()
        if scenario_seed is not None:
            backend_config = getattr(scenario_seed, "backend_config", {}) or {}
            profile = backend_config.get("source_profile")
            duty_program = backend_config.get("native_duty_program")
            yearly_program = backend_config.get("native_yearly_program")
            if sum(bool(value) for value in (profile, duty_program, yearly_program)) > 1:
                raise ValueError(
                    "OpenDSS backend cannot combine source_profile, "
                    "native_duty_program and native_yearly_program"
                )
            if profile:
                self._configure_source_profile(dict(profile))
                summary = self._summary_from_circuit()
                self._refresh_source_trace(summary)
            elif duty_program:
                self._configure_native_duty_program(dict(duty_program))
                summary = self._summary_from_circuit()
                self._refresh_source_trace(summary)
            elif yearly_program:
                self._configure_native_yearly_program(dict(yearly_program))
                summary = self._summary_from_circuit()
                self._refresh_source_trace(summary)
            self._initial_native_controls = self._apply_initial_native_controls(
                dict(backend_config)
            )
            if self._initial_native_controls["controls"]:
                summary = self._summary_from_circuit()
                self._refresh_source_trace(summary)
        self._capture_base_load_values()
        self._capture_base_line_states()
        return summary

    def _apply_initial_native_controls(
        self, backend_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply source-declared native startup controls before the first tick.

        Startup controls are operating-point configuration, not agent actions:
        they are recorded in snapshots/source traces but never receive action
        evidence.  Only controls exposed by this backend are accepted.
        """
        raw_controls = backend_config.get("initial_native_controls") or []
        if not isinstance(raw_controls, list):
            raise ValueError("initial_native_controls must be a list")
        records: list[dict[str, Any]] = []
        for raw_control in raw_controls:
            if not isinstance(raw_control, dict):
                raise ValueError("each initial_native_control must be a mapping")
            name = str(raw_control.get("tool") or "")
            args = dict(raw_control.get("args") or {})
            if name == "set_transformer_tap":
                result = self._set_regulator_tap(args)
            elif name == "switch_capacitor":
                result = self._switch_capacitor(args)
            else:
                raise ValueError("unsupported OpenDSS initial native control: " + name)
            status = str(result.get("_status") or "")
            if status not in {"applied", "no_effect"}:
                raise ValueError(
                    f"invalid OpenDSS initial native control {name}: {status}"
                )
            records.append(
                {
                    "tool": name,
                    "args": args,
                    "status": status,
                    "state_changing": status == "applied",
                }
            )
        return {
            "controls": records,
            "applied_count": sum(
                1 for record in records if record["status"] == "applied"
            ),
            "state_changing": any(bool(record["state_changing"]) for record in records),
        }

    def tick(self, current_tick: int) -> Any:
        realized_events = self._advance_source_profile(current_tick)
        realized_events.extend(self._apply_procedural_perturbations(current_tick))
        snapshot = self.snapshot()
        record = self._backend_record(
            current_tick,
            snapshot,
            realized_events=realized_events,
        )
        self._tick_records.append(record)
        self._add_cost_components(self._cost_components(snapshot))
        self._tick = int(current_tick) + 1
        return _OpenDssFreshFeederTickRecord(record)

    @staticmethod
    def _perturbation_dict(event: Any) -> dict[str, Any]:
        if isinstance(event, dict):
            return dict(event)
        return {
            "kind": getattr(event, "kind", ""),
            "event_class": getattr(event, "event_class", None),
            "trigger_tick": getattr(event, "trigger_tick", 0),
            "duration_ticks": getattr(event, "duration_ticks", 1),
            "hidden": getattr(event, "hidden", False),
            "target": dict(getattr(event, "target", {}) or {}),
            "intensity": getattr(event, "intensity", 1.0),
            "notes": getattr(event, "notes", ""),
        }

    def _capture_base_load_values(self) -> None:
        """Capture native load setpoints after source graph/profile setup."""
        circuit = self._require_circuit()
        values: dict[str, tuple[float, float]] = {}
        idx = circuit.Loads.First
        while idx:
            values[str(circuit.Loads.Name)] = (
                float(circuit.Loads.kW),
                float(circuit.Loads.kvar),
            )
            idx = circuit.Loads.Next
        self._base_load_values = values

    def _capture_base_line_states(self) -> None:
        circuit = self._require_circuit()
        states: dict[str, bool] = {}
        for name_value in circuit.Lines.AllNames:
            name = str(name_value)
            circuit.SetActiveElement(f"Line.{name}")
            states[name] = bool(circuit.ActiveCktElement.Enabled)
        self._base_line_enabled = states

    def _set_line_enabled(self, line_index: int, enabled: bool) -> bool:
        circuit = self._require_circuit()
        names = [str(name) for name in circuit.Lines.AllNames]
        if line_index < 0 or line_index >= len(names):
            return False
        name = names[line_index]
        circuit.SetActiveElement(f"Line.{name}")
        circuit.ActiveCktElement.Enabled = bool(enabled)
        return True

    def _set_all_loads_multiplier(self, multiplier: float) -> None:
        circuit = self._require_circuit()
        idx = circuit.Loads.First
        while idx:
            name = str(circuit.Loads.Name)
            base_kw, base_kvar = self._base_load_values[name]
            circuit.Loads.kW = base_kw * multiplier
            circuit.Loads.kvar = base_kvar * multiplier
            idx = circuit.Loads.Next
        circuit.Solution.Solve()

    def _apply_procedural_perturbations(
        self, current_tick: int
    ) -> list[dict[str, Any]]:
        """Apply deterministic event overlays and expose a real response tick.

        The overlay is deliberately narrow: ``load_surge`` is a multiplier on
        native OpenDSS load setpoints.  It is source-anchored (same compiled
        feeder), deterministic, and never treated as source-consumption proof.
        """
        if not self._perturbations or not self._base_load_values:
            return []
        events: list[dict[str, Any]] = []
        for index, raw_event in enumerate(self._perturbations):
            event = self._perturbation_dict(raw_event)
            kind = str(event.get("kind"))
            event_class = _opendss_event_class(
                kind=kind,
                origin="procedural_perturbation",
                declared_class=event.get("event_class", event.get("class")),
            )
            trigger = int(event.get("trigger_tick", 0) or 0)
            duration = max(1, int(event.get("duration_ticks", 1) or 1))
            end_tick = trigger + duration
            if kind == "line_outage":
                if (
                    index in self._active_line_outage_indices
                    and current_tick >= end_tick
                ):
                    line_index = int((event.get("target") or {}).get("line_index", -1))
                    self._set_line_enabled(line_index, True)
                    self._require_circuit().Solution.Solve()
                    self._active_line_outage_indices.remove(index)
                    events.append(
                        {
                            "event_id": f"opendss-line-outage-clear:{index}:{current_tick}",
                            "type": "line_outage_cleared",
                            "event_class": _opendss_event_class(
                                kind="line_outage_cleared",
                                origin="procedural_perturbation",
                            ),
                            "origin": "procedural_perturbation",
                            "tick": int(current_tick),
                            "changed_state_fields": [
                                "line_in_service",
                                "bus_voltage_pu",
                            ],
                            "decision_required": False,
                            "actionable": False,
                        }
                    )
                if current_tick != trigger or index in self._active_line_outage_indices:
                    continue
                before = self._native_power_state()
                line_index = int((event.get("target") or {}).get("line_index", -1))
                if not self._set_line_enabled(line_index, False):
                    continue
                self._require_circuit().Solution.Solve()
                after = self._native_power_state()
                self._active_line_outage_indices.add(index)
                material = before != after
                actionable = (
                    material
                    and not bool(event.get("hidden"))
                    and int(current_tick) + 1 < self._horizon
                )
                events.append(
                    {
                        "event_id": f"opendss-line-outage:{index}:{trigger}",
                        "type": "line_outage",
                        "event_class": event_class,
                        "origin": "procedural_perturbation",
                        # Keep the raw event origin explicit for the
                        # backend trace while marking it as a declared
                        # scenario perturbation for the shared
                        # world-evolution canonicalizer.
                        "declared_perturbation": True,
                        "hidden": bool(event.get("hidden")),
                        "tick": int(current_tick),
                        "source_asset": [str(self.master_file.resolve())],
                        "target": dict(event.get("target") or {}),
                        "duration_ticks": duration,
                        "changed_state_fields": ["line_in_service", "bus_voltage_pu"],
                        "materiality_metric": "native_state_digest",
                        "materiality_value": float(before["voltage_min_pu"] or 0.0)
                        - float(after["voltage_min_pu"] or 0.0),
                        "materiality_threshold": 0.0001,
                        "materiality_passed": material,
                        "decision_required": actionable,
                        "actionable": actionable,
                        "response_window_required": actionable,
                        "response_opportunity_tick": (
                            int(current_tick) + 1 if actionable else None
                        ),
                        "response_window_end_tick": max(
                            int(current_tick) + 1, min(self._horizon - 1, end_tick)
                        ),
                        "before_state": before,
                        "after_state": after,
                    }
                )
                continue
            if index in self._active_load_surge_indices and current_tick >= end_tick:
                self._set_all_loads_multiplier(1.0)
                self._active_load_surge_indices.remove(index)
                events.append(
                    {
                        "event_id": f"opendss-load-surge-clear:{index}:{current_tick}",
                        "type": "load_surge_cleared",
                        "event_class": _opendss_event_class(
                            kind="load_surge_cleared",
                            origin="procedural_perturbation",
                        ),
                        "origin": "procedural_perturbation",
                        "tick": int(current_tick),
                        "changed_state_fields": [
                            "aggregate_demand_mw",
                            "aggregate_reactive_demand_mvar",
                            "bus_voltage_pu",
                        ],
                        "decision_required": False,
                        "actionable": False,
                    }
                )
            if current_tick != trigger or index in self._active_load_surge_indices:
                continue
            before = self._native_power_state()
            target = dict(event.get("target") or {})
            fraction = float(target.get("load_fraction", event.get("intensity", 0.0)))
            if fraction <= 0.0:
                continue
            self._set_all_loads_multiplier(1.0 + fraction)
            after = self._native_power_state()
            relative_change = abs(
                float(after["aggregate_demand_mw"] or 0.0)
                - float(before["aggregate_demand_mw"] or 0.0)
            ) / max(abs(float(before["aggregate_demand_mw"] or 0.0)), 1e-9)
            material = relative_change >= 0.01
            actionable = (
                material
                and not bool(event.get("hidden"))
                and int(current_tick) + 1 < self._horizon
            )
            self._active_load_surge_indices.add(index)
            events.append(
                {
                    "event_id": f"opendss-load-surge:{index}:{trigger}",
                    "type": "load_surge",
                    "event_class": event_class,
                    "origin": "procedural_perturbation",
                    "declared_perturbation": True,
                    "hidden": bool(event.get("hidden")),
                    "tick": int(current_tick),
                    "source_asset": [str(self.master_file.resolve())],
                    "target": target,
                    "intensity": fraction,
                    "duration_ticks": duration,
                    "changed_state_fields": [
                        "aggregate_demand_mw",
                        "aggregate_reactive_demand_mvar",
                        "bus_voltage_pu",
                    ],
                    "materiality_metric": "aggregate_demand_relative_change",
                    "materiality_value": relative_change,
                    "materiality_threshold": 0.01,
                    "materiality_passed": material,
                    "decision_required": actionable,
                    "actionable": actionable,
                    "response_window_required": actionable,
                    "response_opportunity_tick": (
                        int(current_tick) + 1 if actionable else None
                    ),
                    "response_window_end_tick": max(
                        int(current_tick) + 1, min(self._horizon - 1, end_tick)
                    ),
                    "before_state": before,
                    "after_state": after,
                }
            )
        return events

    def snapshot(self) -> dict[str, Any]:
        raw = super().snapshot()
        raw["backend_kind"] = self.backend_kind
        master = self.master_file.resolve()
        raw["runtime_source_identity"] = {
            "master_file": str(master),
            "master_sha256": _sha256(master),
        }
        raw["initial_native_controls"] = dict(self._initial_native_controls)
        raw["tick"] = self._tick
        raw["horizon"] = self._horizon
        raw.update(self._native_power_state())
        raw["lines"] = self._line_states()
        raw["entities"] = self._entities(raw)
        n_disconnected_lines = sum(
            bool(line["unexpectedly_disconnected"]) for line in raw["lines"]
        )
        raw["totals"] = {
            "n_voltage_violations": raw.get("n_voltage_violations"),
            "voltage_min_pu": raw.get("voltage_min_pu"),
            "voltage_max_pu": raw.get("voltage_max_pu"),
            "line_current_max_a": raw.get("line_current_max_a"),
            "rho_max": 0.0,
            "n_overloads": 0,
            "n_disconnected_lines": n_disconnected_lines,
        }
        return raw

    def ground_truth_costs(self) -> dict[str, float]:
        if not self._cumulative_cost_components:
            return {
                "production_cost": 0.0,
                "voltage_violation_cost": 0.0,
                "voltage_band_deviation_cost": 0.0,
            }
        return {k: round(v, 2) for k, v in self._cumulative_cost_components.items()}

    def per_load_shed_mwh(self) -> dict[str, float]:
        return {}

    def scoring_records(self) -> list[dict[str, Any]]:
        return list(self._tick_records)

    def forecast_for(self, horizon_ticks: int) -> list[dict[str, Any]]:
        profile = self._source_profile
        if (
            profile is not None
            and profile.get("profile_kind") == "native_duty_program"
            and self._source_profile_values
        ):
            circuit = self._require_circuit()
            step_seconds = float(profile["step_seconds"])
            substeps_per_tick = int(profile["substeps_per_tick"])
            source_step = int(
                round(float(circuit.Solution.dblHour) * 3600.0 / step_seconds)
            )
            nominal_mw = float(profile["generator_nominal_kw"]) / 1000.0
            forecast: list[dict[str, Any]] = []
            for offset in range(max(0, int(horizon_ticks))):
                profile_step = source_step + offset * substeps_per_tick
                if profile_step >= len(self._source_profile_values):
                    break
                multiplier = self._source_profile_values[profile_step]
                forecast.append(
                    {
                        "tick": self._tick + offset + 1,
                        "source_profile_step": profile_step,
                        "source_multiplier": multiplier,
                        "expected_generation_mw": nominal_mw * multiplier,
                        "forecast_origin": "locked_source_profile",
                    }
                )
            return forecast
        snap = self.snapshot()
        return [
            {
                "tick": self._tick + i + 1,
                "voltage_min_pu": snap.get("voltage_min_pu"),
                "voltage_max_pu": snap.get("voltage_max_pu"),
                "n_voltage_violations": snap.get("n_voltage_violations"),
            }
            for i in range(max(0, int(horizon_ticks)))
        ]

    def _entities(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        entities: dict[str, Any] = {}
        for cap in snapshot.get("capacitors") or []:
            entities[f"capacitor_{cap['cap_id']}"] = {
                "kind": "capacitor",
                "status": any(int(s) for s in cap.get("states") or []),
                **cap,
            }
        for reg in snapshot.get("regcontrols") or []:
            entities[f"regulator_{reg['trafo_id']}"] = {
                "kind": "voltage_regulator",
                **reg,
            }
        for line in snapshot.get("lines") or []:
            entities[f"line_{line['line_index']}"] = {
                "kind": "line",
                **line,
            }
        return entities

    def _backend_record(
        self,
        tick: int,
        snapshot: dict[str, Any],
        *,
        realized_events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "tick": int(tick),
            "backend_kind": self.backend_kind,
            "aggregate_demand_mw": float(snapshot.get("aggregate_demand_mw") or 0.0),
            "aggregate_reactive_demand_mvar": float(
                snapshot.get("aggregate_reactive_demand_mvar") or 0.0
            ),
            "aggregate_generation_mw": float(
                snapshot.get("aggregate_generation_mw") or 0.0
            ),
            "net_grid_import_mw": float(snapshot.get("net_grid_import_mw") or 0.0),
            "balance_error_mw": 0.0,
            "reserves_required_mw": 0.0,
            "reserves_procured_mw": 0.0,
            "production_cost": 0.0,
            "startup_cost": 0.0,
            "shed_penalty": 0.0,
            "rho_max": 0.0,
            "n_overloads": 0,
            "n_voltage_violations": int(snapshot.get("n_voltage_violations") or 0),
            "n_disconnected_lines": int(
                (snapshot.get("totals") or {}).get("n_disconnected_lines") or 0
            ),
            "done": False,
            "converged": bool(snapshot.get("converged")),
            "voltage_min_pu": snapshot.get("voltage_min_pu"),
            "voltage_max_pu": snapshot.get("voltage_max_pu"),
            "voltage_band_error": self._voltage_band_error(snapshot),
            "line_current_max_a": snapshot.get("line_current_max_a"),
            "realized_events": realized_events,
        }

    def _configure_source_profile(self, profile: dict[str, Any]) -> None:
        """Bind a locked OpenDSS LoadShape to the native yearly solver clock."""
        definition = self._resolve_profile_path(profile.get("definition_file"))
        configured_data = profile.get("data_files") or [profile.get("data_file")]
        data_paths = {
            self._resolve_profile_path(value) for value in configured_data if value
        }
        if not data_paths:
            raise ValueError("OpenDSS source_profile requires data_file")
        if len(data_paths) != 1:
            raise ValueError(
                "OpenDSS source_profile currently supports exactly one data file"
            )
        name = str(profile.get("name") or "").strip()
        if not name:
            raise ValueError("OpenDSS source_profile requires name")
        assignment = str(profile.get("assignment") or "yearly").lower()
        if assignment != "yearly":
            raise ValueError("OpenDSS source_profile only supports yearly assignment")
        start_hour = int(profile.get("start_hour", 0))
        step_hours = float(profile.get("step_hours", 1.0))
        if start_hour < 0 or step_hours <= 0.0:
            raise ValueError(
                "OpenDSS source_profile requires non-negative start_hour and positive step_hours"
            )

        profile_assets, profile_inventory = _resolve_native_include_graph(definition)
        actual_data_paths = {
            Path(asset["path"])
            for asset in profile_assets
            if asset["role"] == "runtime_data_input"
        }
        if actual_data_paths != data_paths:
            raise ValueError(
                "OpenDSS source_profile data files must exactly match native "
                "LoadShape file references"
            )
        for asset in profile_assets:
            if Path(asset["path"]) == definition:
                asset["role"] = "runtime_profile_definition"
        self._runtime_source_assets = _merge_source_assets(
            self._runtime_source_assets, profile_assets
        )
        for object_kind, count in profile_inventory.items():
            self._parsed_source_inventory[object_kind] = (
                self._parsed_source_inventory.get(object_kind, 0) + int(count)
            )

        self._run_dss_command(f'Redirect "{definition}"')
        circuit = self._require_circuit()
        names = [str(value) for value in circuit.LoadShapes.AllNames]
        native_name = next(
            (candidate for candidate in names if candidate.lower() == name.lower()),
            None,
        )
        if native_name is None:
            raise ValueError(f"OpenDSS LoadShape not found after redirect: {name}")
        circuit.LoadShapes.Name = native_name
        expected_values = _read_profile_values(next(iter(data_paths)))
        native_values = [float(value) for value in circuit.LoadShapes.Pmult]
        values_match = (
            len(expected_values) == int(circuit.LoadShapes.Npts)
            and len(expected_values) == len(native_values)
            and all(
                abs(expected - native) <= 1e-12
                for expected, native in zip(expected_values, native_values, strict=True)
            )
        )
        if not values_match:
            raise ValueError("OpenDSS LoadShape values differ from locked data file")
        self._run_dss_command(f"BatchEdit Load..* {assignment}={native_name}")
        self._run_dss_command(
            f"Set mode=yearly number=1 stepsize={step_hours:g}h hour={start_hour}"
        )
        self._source_profile_active = True
        circuit.Solution.SolveSnap()
        initial_state = self._native_power_state()
        initial_state_digest = _semantic_digest(initial_state)
        self._source_profile = {
            "definition_file": str(definition),
            "definition_sha256": _sha256(definition),
            "data_files": [str(path) for path in sorted(data_paths)],
            "data_sha256": {str(path): _sha256(path) for path in sorted(data_paths)},
            "profile_name": native_name,
            "assignment": assignment,
            "start_hour": start_hour,
            "step_hours": step_hours,
            "npts": int(circuit.LoadShapes.Npts),
            "source_values_digest": _semantic_digest(expected_values),
            "native_values_digest": _semantic_digest(native_values),
            "runtime_values_match": values_match,
            "initial_state_digest": initial_state_digest,
            "post_source_state_digests": [initial_state_digest],
            "consumption_ticks": [0],
            "runtime_source_events": [],
        }

    def _configure_native_yearly_program(self, program: dict[str, Any]) -> None:
        """Advance the feeder's source-declared yearly load assignments.

        Unlike ``source_profile``, this mode does not replace every load with
        one synthetic/common shape.  It preserves each load's own ``yearly``
        assignment from the compiled source graph and advances the native
        OpenDSS clock one bounded interval per benchmark tick.
        """
        start_hour = int(program.get("start_hour", 0))
        step_hours = float(program.get("step_hours", 1.0))
        if start_hour < 0 or step_hours <= 0.0:
            raise ValueError(
                "OpenDSS native_yearly_program requires non-negative start_hour "
                "and positive step_hours"
            )

        circuit = self._require_circuit()
        available_shapes = {
            str(value).lower(): str(value) for value in circuit.LoadShapes.AllNames
        }
        assignments: list[str] = []
        load_index = circuit.Loads.First
        while load_index:
            assignment = str(circuit.Loads.Yearly).strip()
            if assignment:
                assignments.append(assignment)
            load_index = circuit.Loads.Next
        if not assignments:
            raise ValueError(
                "OpenDSS native_yearly_program requires source-declared yearly "
                "load assignments"
            )
        missing = sorted(
            {name for name in assignments if name.lower() not in available_shapes}
        )
        if missing:
            raise ValueError(
                "OpenDSS native_yearly_program references missing LoadShapes: "
                + ", ".join(missing)
            )

        shape_digests: dict[str, str] = {}
        for normalized_name in sorted({name.lower() for name in assignments}):
            native_name = available_shapes[normalized_name]
            circuit.LoadShapes.Name = native_name
            shape_digests[native_name] = _semantic_digest(
                {
                    "pmult": [float(value) for value in circuit.LoadShapes.Pmult],
                    "qmult": [float(value) for value in circuit.LoadShapes.Qmult],
                }
            )
        self._run_dss_command(
            f"Set mode=yearly number=1 stepsize={step_hours:g}h hour={start_hour}"
        )
        self._source_profile_active = True
        circuit.Solution.SolveSnap()
        initial_state = self._native_power_state()
        initial_state_digest = _semantic_digest(initial_state)
        source_values_digest = _semantic_digest(shape_digests)
        self._source_profile = {
            "profile_kind": "native_yearly_feeder_program",
            "data_files": [str(self.master_file.resolve())],
            "profile_name": "source_declared_yearly_profiles",
            "assignment": "yearly",
            "start_hour": start_hour,
            "step_hours": step_hours,
            "assigned_load_count": len(assignments),
            "assigned_profile_names": sorted(set(assignments), key=str.lower),
            "source_values_digest": source_values_digest,
            "native_values_digest": source_values_digest,
            "runtime_values_match": True,
            "initial_state_digest": initial_state_digest,
            "post_source_state_digests": [initial_state_digest],
            "consumption_ticks": [0],
            "runtime_source_events": [],
        }

    def _configure_native_duty_program(self, program: dict[str, Any]) -> None:
        """Consume one source-defined LoadShape/Generator duty program.

        The adapter parses the exact source program, executes only the selected
        LoadShape and its matching Generator declaration, and verifies that the
        declared data file is the same file referenced by the parsed source
        command.  It deliberately does not ``Redirect`` the full script: those
        scripts often contain presentation commands or a completed batch solve
        that would advance the source clock before the episode starts.
        """
        scenario_file = self._resolve_profile_path(program.get("scenario_file"))
        profile_name = str(program.get("profile_name") or "").strip()
        if not profile_name:
            raise ValueError("OpenDSS native_duty_program requires profile_name")
        declared_data_path = self._resolve_profile_path(program.get("data_file"))
        start_step = int(program.get("start_step", 0))
        substeps_per_tick = int(program.get("substeps_per_tick", 1))
        if start_step < 0 or substeps_per_tick <= 0:
            raise ValueError(
                "OpenDSS native_duty_program requires non-negative start_step "
                "and positive substeps_per_tick"
            )

        configured_master = self.master_file.resolve()
        program_text = scenario_file.read_text(
            encoding="utf-8", errors="strict"
        )
        program_masters = _direct_compile_inputs(
            scenario_file,
            program_text,
            windows_aliases={configured_master.name.lower(): configured_master},
        )
        if program_masters != {configured_master}:
            raise ValueError(
                "OpenDSS native_duty_program must compile exactly the configured "
                "master file"
            )

        loadshape_lines: list[tuple[str, Path, float, int, bool]] = []
        generator_lines: list[tuple[str, str, float]] = []
        for raw_line in program_text.splitlines():
            match = _NEW_OBJECT_RE.match(raw_line)
            if match is None:
                continue
            object_kind = match.group(1).lower()
            object_name = match.group(2).lower()
            if object_kind == "loadshape" and object_name == profile_name.lower():
                file_match = _DSS_FILE_REFERENCE_RE.search(raw_line)
                sinterval_match = _DSS_SINTERVAL_RE.search(raw_line)
                interval_seconds_match = _DSS_INTERVAL_SECONDS_RE.search(raw_line)
                npts_match = _DSS_NPTS_RE.search(raw_line)
                if (
                    file_match is None
                    or npts_match is None
                    or (sinterval_match is None and interval_seconds_match is None)
                ):
                    raise ValueError(
                        "OpenDSS native_duty_program LoadShape requires npts=, "
                        "file=, and sinterval= or Interval=(seconds 3600 /) "
                        "source fields"
                    )
                raw_data_file = next(
                    value for value in file_match.groups() if value is not None
                )
                data_path = (scenario_file.parent / raw_data_file).resolve()
                step_seconds = float(
                    (
                        sinterval_match
                        if sinterval_match is not None
                        else interval_seconds_match
                    ).group(1)
                )
                loadshape_lines.append(
                    (
                        raw_line.strip(),
                        data_path,
                        step_seconds,
                        int(npts_match.group(1)),
                        _DSS_NORMALIZE_RE.search(raw_line) is not None,
                    )
                )
            if object_kind == "generator":
                duty_match = _DSS_DUTY_REFERENCE_RE.search(raw_line)
                if duty_match is None:
                    continue
                duty_name = next(
                    value for value in duty_match.groups() if value is not None
                )
                if duty_name.lower() == profile_name.lower():
                    kw_match = _DSS_KW_RE.search(raw_line)
                    if kw_match is None:
                        raise ValueError(
                            "OpenDSS native_duty_program Generator requires a "
                            "source kW rating"
                        )
                    generator_lines.append(
                        (raw_line.strip(), object_name, float(kw_match.group(1)))
                    )
        if len(loadshape_lines) != 1 or len(generator_lines) != 1:
            raise ValueError(
                "OpenDSS native_duty_program requires exactly one matching "
                "LoadShape and Generator duty declaration"
            )

        (
            loadshape_line,
            data_path,
            step_seconds,
            source_npts,
            normalize_source_values,
        ) = loadshape_lines[0]
        generator_line, generator_name, generator_nominal_kw = generator_lines[0]
        if not data_path.is_file():
            raise FileNotFoundError(
                f"OpenDSS native_duty_program data file is missing: {data_path}"
            )
        if not _same_or_casefolded_path(data_path, declared_data_path):
            raise ValueError(
                "OpenDSS native_duty_program data_file must exactly match the "
                "selected source LoadShape file"
            )
        data_path = declared_data_path
        runtime_loadshape_line = _DSS_FILE_REFERENCE_RE.sub(
            f'file="{data_path}"', loadshape_line, count=1
        )
        self._run_dss_command(runtime_loadshape_line)
        self._run_dss_command(generator_line)
        circuit = self._require_circuit()
        loadshape_names = [str(value) for value in circuit.LoadShapes.AllNames]
        native_profile_name = next(
            (
                candidate
                for candidate in loadshape_names
                if candidate.lower() == profile_name.lower()
            ),
            None,
        )
        if native_profile_name is None:
            raise ValueError(
                "OpenDSS native_duty_program LoadShape missing after source "
                "command execution"
            )
        generator_names = [str(value) for value in circuit.Generators.AllNames]
        native_generator_name = next(
            (
                candidate
                for candidate in generator_names
                if candidate.lower() == generator_name.lower()
            ),
            None,
        )
        if native_generator_name is None:
            raise ValueError(
                "OpenDSS native_duty_program Generator missing after source "
                "command execution"
            )
        circuit.LoadShapes.Name = native_profile_name
        source_values = _read_profile_values(
            data_path,
            blank_rows_are_zero=True,
        )
        if len(source_values) != source_npts:
            raise ValueError(
                "OpenDSS native_duty_program source data row count must match "
                "the published LoadShape npts"
            )
        source_scale = max((abs(value) for value in source_values), default=0.0)
        if normalize_source_values and source_scale <= 0.0:
            raise ValueError(
                "OpenDSS native_duty_program cannot normalize an all-zero "
                "source LoadShape"
            )
        expected_values = (
            [value / source_scale for value in source_values]
            if normalize_source_values
            else source_values
        )
        native_values = [float(value) for value in circuit.LoadShapes.Pmult]
        values_match = (
            len(expected_values) == int(circuit.LoadShapes.Npts)
            and len(expected_values) == len(native_values)
            and all(
                abs(expected - native) <= 1e-12
                for expected, native in zip(expected_values, native_values, strict=True)
            )
        )
        if not values_match:
            raise ValueError(
                "OpenDSS native_duty_program LoadShape values differ from the "
                "locked data file"
            )
        self._source_profile_values = list(expected_values)
        self._run_dss_command(
            "Set mode=dutycycle number=1 "
            f"stepsize={step_seconds:g}s hour=0 sec={start_step * step_seconds:g}"
        )
        self._source_profile_active = True
        circuit.Solution.SolveSnap()
        initial_state = self._native_power_state()
        initial_state_digest = _semantic_digest(initial_state)
        self._runtime_source_assets = _merge_source_assets(
            self._runtime_source_assets,
            [
                {
                    "path": str(scenario_file),
                    "sha256": _sha256(scenario_file),
                    "role": "runtime_source_program",
                },
                {
                    "path": str(data_path),
                    "sha256": _sha256(data_path),
                    "role": "runtime_data_input",
                    "included_from": str(scenario_file),
                },
            ],
        )
        self._parsed_source_inventory["loadshape"] = (
            self._parsed_source_inventory.get("loadshape", 0) + 1
        )
        self._parsed_source_inventory["generator"] = (
            self._parsed_source_inventory.get("generator", 0) + 1
        )
        self._source_profile = {
            "profile_kind": "native_duty_program",
            "scenario_file": str(scenario_file),
            "scenario_sha256": _sha256(scenario_file),
            "source_loadshape_command_digest": _semantic_digest(loadshape_line),
            "source_generator_command_digest": _semantic_digest(generator_line),
            "runtime_loadshape_command_digest": _semantic_digest(
                runtime_loadshape_line
            ),
            "data_files": [str(data_path)],
            "data_sha256": {str(data_path): _sha256(data_path)},
            "source_data_file_reference": raw_data_file,
            "profile_name": native_profile_name,
            "generator_name": native_generator_name,
            "generator_nominal_kw": generator_nominal_kw,
            "step_seconds": step_seconds,
            "start_step": start_step,
            "substeps_per_tick": substeps_per_tick,
            "npts": int(circuit.LoadShapes.Npts),
            "source_npts": source_npts,
            "source_values_digest": _semantic_digest(source_values),
            "native_values_digest": _semantic_digest(native_values),
            "source_normalized_by_published_program": normalize_source_values,
            "runtime_values_match": values_match,
            "initial_state_digest": initial_state_digest,
            "post_source_state_digests": [initial_state_digest],
            "consumption_ticks": [0],
            "runtime_source_events": [],
        }

    def _advance_source_profile(self, current_tick: int) -> list[dict[str, Any]]:
        profile = self._source_profile
        if profile is None or int(current_tick) + 1 >= self._horizon:
            return []
        if profile.get("profile_kind") == "native_duty_program":
            return self._advance_native_duty_program(current_tick)
        if profile.get("profile_kind") == "native_yearly_feeder_program":
            return self._advance_native_yearly_program(current_tick)
        circuit = self._require_circuit()
        before = self._native_power_state()
        before_hour = float(circuit.Solution.dblHour)
        circuit.Solution.Solve()
        after = self._native_power_state()
        after_hour = float(circuit.Solution.dblHour)
        demand_before = float(before["aggregate_demand_mw"])
        demand_after = float(after["aggregate_demand_mw"])
        relative_change = abs(demand_after - demand_before) / max(
            abs(demand_before), 1e-9
        )
        materiality_threshold = 0.01
        material = relative_change >= materiality_threshold
        event = {
            "event_id": (
                f"opendss-loadshape:{profile['source_values_digest'][:12]}:"
                f"{int(current_tick)}"
            ),
            "type": "load_change",
            "event_class": _opendss_event_class(
                kind="load_change",
                origin="source_schedule",
            ),
            "origin": "source_schedule",
            "tick": int(current_tick),
            "source_asset": profile["data_files"],
            "profile_name": profile["profile_name"],
            "profile_hour_before": before_hour,
            "profile_hour_after": after_hour,
            "changed_state_fields": [
                "aggregate_demand_mw",
                "aggregate_reactive_demand_mvar",
                "bus_voltage_pu",
            ],
            "materiality_metric": "aggregate_demand_relative_change",
            "materiality_value": relative_change,
            "materiality_threshold": materiality_threshold,
            "materiality_passed": material,
            "decision_required": material,
            "actionable": material,
            "response_window_required": material,
            "response_opportunity_tick": int(current_tick) + 1,
        }
        profile["consumption_ticks"].append(int(current_tick) + 1)
        profile["runtime_source_events"].append(event)
        profile["post_source_state_digests"].append(_semantic_digest(after))
        self._refresh_source_trace(self._summary_from_circuit())
        return [event]

    def _advance_native_yearly_program(self, current_tick: int) -> list[dict[str, Any]]:
        profile = self._source_profile
        assert profile is not None
        circuit = self._require_circuit()
        before = self._native_power_state()
        before_hour = float(circuit.Solution.dblHour)
        target_hour = float(profile["start_hour"]) + (
            int(current_tick) + 1
        ) * float(profile["step_hours"])
        whole_hour = int(target_hour)
        seconds = (target_hour - whole_hour) * 3600.0
        self._run_dss_command(f"Set hour={whole_hour} sec={seconds:g}")
        # The source coordinate is set explicitly above. ``Solve`` advances
        # the OpenDSS clock before evaluating the interval and would therefore
        # skip the requested hour; ``SolveSnap`` evaluates exactly that source
        # coordinate while retaining the compiled yearly assignments.
        circuit.Solution.SolveSnap()
        after = self._native_power_state()
        after_hour = float(circuit.Solution.dblHour)
        demand_before = float(before["aggregate_demand_mw"])
        demand_after = float(after["aggregate_demand_mw"])
        relative_change = abs(demand_after - demand_before) / max(
            abs(demand_before), 1e-9
        )
        materiality_threshold = 0.01
        material = relative_change >= materiality_threshold
        event = {
            "event_id": (
                f"opendss-loadshape:{profile['source_values_digest'][:12]}:"
                f"{int(current_tick)}"
            ),
            "type": "load_change",
            "event_class": _opendss_event_class(
                kind="load_change",
                origin="source_schedule",
            ),
            "origin": "source_schedule",
            "tick": int(current_tick),
            "source_asset": profile["data_files"],
            "profile_name": profile["profile_name"],
            "profile_hour_before": before_hour,
            "profile_hour_after": after_hour,
            "changed_state_fields": [
                "aggregate_demand_mw",
                "aggregate_reactive_demand_mvar",
                "bus_voltage_pu",
            ],
            "materiality_metric": "aggregate_demand_relative_change",
            "materiality_value": relative_change,
            "materiality_threshold": materiality_threshold,
            "materiality_passed": material,
            "decision_required": material,
            "actionable": material,
            "response_window_required": material,
            "response_opportunity_tick": int(current_tick) + 1,
        }
        profile["consumption_ticks"].append(int(current_tick) + 1)
        profile["runtime_source_events"].append(event)
        profile["post_source_state_digests"].append(_semantic_digest(after))
        self._refresh_source_trace(self._summary_from_circuit())
        return [event]

    def _advance_native_duty_program(self, current_tick: int) -> list[dict[str, Any]]:
        profile = self._source_profile
        assert profile is not None
        circuit = self._require_circuit()
        before = self._native_power_state()
        before_hour = float(circuit.Solution.dblHour)
        substeps_per_tick = int(profile["substeps_per_tick"])
        for _ in range(substeps_per_tick):
            circuit.Solution.Solve()
        after = self._native_power_state()
        after_hour = float(circuit.Solution.dblHour)
        generation_before = float(before["aggregate_generation_mw"])
        generation_after = float(after["aggregate_generation_mw"])
        relative_change = abs(generation_after - generation_before) / max(
            abs(generation_before), 1e-9
        )
        materiality_threshold = 0.01
        material = relative_change >= materiality_threshold
        step_seconds = float(profile["step_seconds"])
        event = {
            "event_id": (
                f"opendss-duty:{profile['source_values_digest'][:12]}:"
                f"{int(current_tick)}"
            ),
            "type": "generation_ramp",
            "event_class": _opendss_event_class(
                kind="generation_ramp",
                origin="source_schedule",
            ),
            "origin": "source_schedule",
            "tick": int(current_tick),
            "source_asset": profile["data_files"],
            "source_program": profile["scenario_file"],
            "profile_name": profile["profile_name"],
            "generator_name": profile["generator_name"],
            "profile_step_before": int(round(before_hour * 3600.0 / step_seconds)),
            "profile_step_after": int(round(after_hour * 3600.0 / step_seconds)),
            "native_substeps": substeps_per_tick,
            "changed_state_fields": [
                "aggregate_generation_mw",
                "net_grid_import_mw",
                "voltage_min_pu",
                "voltage_max_pu",
            ],
            "materiality_metric": "aggregate_generation_relative_change",
            "materiality_value": relative_change,
            "materiality_threshold": materiality_threshold,
            "materiality_passed": material,
            "decision_required": material,
            "actionable": material,
            "response_window_required": material,
            "response_opportunity_tick": int(current_tick) + 1,
        }
        profile["consumption_ticks"].append(int(current_tick) + 1)
        profile["runtime_source_events"].append(event)
        profile["post_source_state_digests"].append(_semantic_digest(after))
        self._refresh_source_trace(self._summary_from_circuit())
        return [event]

    def _native_power_state(self) -> dict[str, float | None]:
        circuit = self._require_circuit()
        real_kw, reactive_kvar = (float(value) for value in circuit.TotalPower)
        voltages = [float(value) for value in circuit.AllBusVmagPu]
        generation_kw = 0.0
        generator_idx = circuit.Generators.First
        while generator_idx:
            generation_kw += max(0.0, float(circuit.Generators.kW))
            generator_idx = circuit.Generators.Next
        net_grid_import_mw = max(0.0, -real_kw / 1000.0)
        return {
            "aggregate_demand_mw": net_grid_import_mw,
            "aggregate_reactive_demand_mvar": max(0.0, -reactive_kvar / 1000.0),
            "aggregate_generation_mw": generation_kw / 1000.0,
            "net_grid_import_mw": net_grid_import_mw,
            "voltage_min_pu": min(voltages) if voltages else None,
            "voltage_max_pu": max(voltages) if voltages else None,
        }

    def _refresh_source_trace(self, summary: OpenDssFreshFeederSolveSummary) -> None:
        self._protocol21_source_evidence = _native_protocol21_trace(
            assets=self._runtime_source_assets,
            inventory=self._parsed_source_inventory,
            dss_version=self._dss_version,
            circuit=self._require_circuit(),
            summary=summary.to_dict(),
            source_profile=self._source_profile,
        )
        self._protocol21_source_evidence["runtime_source_identity"] = {
            "master_file": str(self.master_file.resolve()),
            "master_sha256": _sha256(self.master_file.resolve()),
        }
        self._protocol21_source_evidence["initial_native_controls"] = dict(
            self._initial_native_controls
        )
        self._protocol21_source_evidence["runtime_execution_policy"] = dict(
            self._runtime_execution_policy
        )

    def _resolve_profile_path(self, value: Any) -> Path:
        raw = str(value or "").strip()
        if not raw:
            raise ValueError("OpenDSS source_profile path is missing")
        candidate = Path(raw)
        candidates = (
            [candidate]
            if candidate.is_absolute()
            else [
                REPO_ROOT / candidate,
                self.source_root / candidate,
                _IEEE13_CLONE / candidate,
                self.master_file.parent / candidate,
            ]
        )
        for path in candidates:
            if path.is_file():
                return path.resolve()
        raise FileNotFoundError(f"OpenDSS source_profile asset is missing: {raw}")

    def _run_dss_command(self, command: str) -> None:
        if self._dss is None:
            raise RuntimeError("OpenDSS fresh-feeder backend has not been reset")
        self._dss.Text.Command = command
        if int(self._dss.Error.Number or 0):
            raise ValueError(
                f"OpenDSS command failed: {command}: {self._dss.Error.Description}"
            )

    def _cost_components(self, snapshot: dict[str, Any]) -> dict[str, float]:
        violations = float(snapshot.get("n_voltage_violations") or 0.0)
        return {
            "production_cost": 0.0,
            "voltage_violation_cost": 100.0 * violations,
            "voltage_band_deviation_cost": 1000.0 * self._voltage_band_error(snapshot),
        }

    def _add_cost_components(self, components: dict[str, float]) -> None:
        for key, value in components.items():
            self._cumulative_cost_components[key] = (
                self._cumulative_cost_components.get(key, 0.0) + float(value)
            )

    def _voltage_band_error(self, snapshot: dict[str, Any]) -> float:
        v_min = snapshot.get("voltage_min_pu")
        v_max = snapshot.get("voltage_max_pu")
        band_error = 0.0
        if isinstance(v_min, (int, float)):
            band_error += max(0.0, VOLTAGE_LOWER_PU - float(v_min))
        if isinstance(v_max, (int, float)):
            band_error += max(0.0, float(v_max) - VOLTAGE_UPPER_PU)
        return band_error


class _OpenDssFreshFeederTickRecord:
    def __init__(self, record: dict[str, Any]) -> None:
        self.realized_events: list[dict[str, Any]] = []
        for key, value in record.items():
            setattr(self, key, value)


def register_opendss_fresh_feeder_probe_tools(
    registry: ToolRegistry, backend: OpenDssFreshFeederProbeBackend
) -> None:
    """Register non-release fresh-feeder controls through core.tool_protocol."""

    registry.register(
        ToolSpec(
            name="switch_capacitor",
            description="Switch a capacitor bank in an OpenDSS fresh-feeder probe.",
            parameters={
                "type": "object",
                "properties": {
                    "cap_id": {"type": "integer"},
                    "status": {"type": "boolean"},
                },
                "required": ["cap_id", "status"],
            },
            handler=lambda args, ctx: backend.apply_tool_effect(
                "switch_capacitor", args, ctx
            ),
            state_changing=True,
            semantic_role="control",
            native_target_kind="capacitor_bank",
            actuator_family="shunt_switching",
        )
    )
    registry.register(
        ToolSpec(
            name="switch_branch",
            description="Open or close a native OpenDSS feeder line.",
            parameters={
                "type": "object",
                "properties": {
                    "line_index": {"type": "integer"},
                    "connect": {"type": "boolean"},
                },
                "required": ["line_index", "connect"],
            },
            handler=lambda args, ctx: backend.apply_tool_effect(
                "switch_branch", args, ctx
            ),
            state_changing=True,
            semantic_role="control",
            native_target_kind="distribution_line",
            actuator_family="branch_switching",
        )
    )
    registry.register(
        ToolSpec(
            name="set_transformer_tap",
            description="Set a regulator tap in an OpenDSS fresh-feeder probe.",
            parameters={
                "type": "object",
                "properties": {
                    "trafo_id": {"type": "integer"},
                    "tap_pos": {"type": "integer"},
                },
                "required": ["trafo_id", "tap_pos"],
            },
            handler=lambda args, ctx: backend.apply_tool_effect(
                "set_transformer_tap", args, ctx
            ),
            state_changing=True,
            semantic_role="control",
            native_target_kind="voltage_regulator",
            actuator_family="tap_changer",
        )
    )


def _import_dss() -> Any:
    try:
        import dss  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - exercised by optional runtime
        raise OpenDssRuntimeUnavailable(
            "dss-python is not installed; install dss-python to run the "
            "OpenDSS IEEE34/IEEE123 fresh-feeder live probe"
        ) from exc
    return dss
