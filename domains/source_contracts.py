"""Backend-owned source asset contracts for protocol-2.1 staging.

These builders describe which locked files a backend is expected to consume.
They do not claim that consumption occurred; runtime adapters must prove that
separately from backend trace records.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from audit.provenance import _provenance_candidates
from core.source_asset_contract import virtual_source_identity_sha256


def _paths(scenario: dict[str, Any]) -> list[str]:
    return [
        str(value)
        for value in (
            (scenario.get("provenance") or {}).get("files")
            or scenario.get("provenance_files")
            or []
        )
        if str(value)
    ]


def _contract(
    *,
    runtime: tuple[str, ...] = (),
    derivation: tuple[str, ...] = (),
    implementation: tuple[str, ...] = (),
    metadata: tuple[str, ...] = (),
    license_files: tuple[str, ...] = (),
    window_sha256: str | None = None,
    recipe_version: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "runtime_input": list(dict.fromkeys(runtime)),
        "derivation_input": list(dict.fromkeys(derivation)),
        "implementation_asset": list(dict.fromkeys(implementation)),
        "metadata": list(dict.fromkeys(metadata)),
        "license": list(dict.fromkeys(license_files)),
    }
    if window_sha256 and recipe_version:
        result["derived_window"] = {
            "sha256": window_sha256.removeprefix("sha256:"),
            "recipe_version": recipe_version,
        }
    return result


def _exact_config_path(
    scenario: dict[str, Any],
    *keys: str,
) -> str | None:
    current: Any = scenario.get("backend_config") or {}
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return str(current) if current else None


def _provenance_match(
    scenario: dict[str, Any],
    expected: str | None,
) -> tuple[str, ...]:
    if not expected:
        return ()
    expected_path = Path(expected)
    return tuple(
        value
        for value in _paths(scenario)
        if Path(value) == expected_path
        or str(Path(value)).endswith(str(expected_path))
    )


def alibaba_trace_sim(
    scenario: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    from domains.datacenter.backends.alibaba_trace_backend import (
        resolve_alibaba_source_window,
    )

    files = tuple(_paths(scenario))
    provenance = scenario.get("provenance") or {}
    window = resolve_alibaba_source_window(
        provenance_files=list(files),
        time_window=dict(provenance.get("time_window") or {}),
        backend_config=dict(scenario.get("backend_config") or {}),
        repo_root=repo_root,
    )
    return _contract(
        derivation=files,
        window_sha256=str(window["sha256"]),
        recipe_version=str(window["recipe_version"]),
    )


def alibaba_openb_gpu_placement(
    scenario: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Bind the exact OpenB node/pod files read by the placement backend."""
    del repo_root
    transform = (scenario.get("backend_config") or {}).get(
        "source_transform"
    ) or {}
    roles = transform.get("source_file_roles") or {}
    expected = (
        str(roles.get("node_inventory") or ""),
        str(roles.get("pod_trace") or ""),
    )
    runtime = tuple(
        value
        for path in expected
        if path
        for value in _provenance_match(scenario, path)
    )
    return _contract(
        runtime=runtime,
        window_sha256=str(transform.get("source_graph_sha256") or "") or None,
        recipe_version=str(transform.get("recipe_version") or "") or None,
    )


def jsplib_job_shop(
    scenario: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    del repo_root
    backend_config = scenario.get("backend_config") or {}
    assets = backend_config.get("external_source_assets") or {}
    j2_lock = assets.get("j2_event_sidecar") if isinstance(assets, dict) else None
    if (
        backend_config.get("source_mode") == "realm_j2_json"
        and isinstance(j2_lock, dict)
        and j2_lock.get("canonical_runtime_source") is True
        and j2_lock.get("path")
    ):
        runtime = (str(j2_lock["path"]),)
        metadata = tuple(value for value in _paths(scenario) if value not in runtime)
        return _contract(runtime=runtime, metadata=metadata)
    name = str(backend_config.get("instance_name") or "")
    runtime = tuple(
        value for value in _paths(scenario) if Path(value).name == name
    )
    metadata = tuple(value for value in _paths(scenario) if value not in runtime)
    return _contract(runtime=runtime, metadata=metadata)


def orgym_invmgmt(
    scenario: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    from domains.logistics.backends.orgym_invmgmt import (
        resolve_orgym_m5_source_window,
    )

    window = resolve_orgym_m5_source_window(
        backend_config=dict(scenario.get("backend_config") or {}),
        provenance=dict(scenario.get("provenance") or {}),
        repo_root=repo_root,
    )
    return _contract(
        derivation=tuple(
            window["locked_derivation_source_hashes"]
        ),
        metadata=("works/M5/source_lock.json",),
        window_sha256=str(window["source_window_sha256"]),
        recipe_version=str(window["recipe_version"]),
    )


def _vrp_contract(
    scenario: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    del repo_root
    config = scenario.get("backend_config") or {}
    network = config.get("network") or {}
    nested_name = network.get("instance_name") if isinstance(network, dict) else None
    name = str(nested_name or config.get("instance_name") or "")
    normalized_name = Path(name).with_suffix("").as_posix().lower()
    candidates = tuple(_paths(scenario))
    runtime = tuple(
        value
        for value in candidates
        if not normalized_name
        or Path(value).with_suffix("").as_posix().lower().endswith(normalized_name)
    )
    return _contract(runtime=runtime)


def pyvrp_cvrp(
    scenario: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    return _vrp_contract(scenario, repo_root)


def pyvrp_vrptw(
    scenario: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    return _vrp_contract(scenario, repo_root)


def pandapower_acopf(
    scenario: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    del repo_root
    expected = _exact_config_path(scenario, "case_file")
    return _contract(runtime=_provenance_match(scenario, expected))


def pglib_uc_synthetic(
    scenario: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    return pandapower_acopf(scenario, repo_root)


def _portable_graph_path(
    resolved: Path,
    *,
    declared_root_file: str,
    repo_root: Path,
) -> str:
    lexical = Path(declared_root_file)
    if lexical.is_absolute():
        try:
            lexical = lexical.relative_to(repo_root)
        except ValueError:
            return str(resolved)
    cursor = repo_root
    declared_prefix = Path()
    for part in lexical.parts[:-1]:
        cursor /= part
        declared_prefix /= part
        if not cursor.is_symlink():
            continue
        target = cursor.resolve()
        try:
            relative = resolved.relative_to(target)
        except ValueError:
            continue
        return (declared_prefix / relative).as_posix()
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return str(resolved)


def _opendss_contract(
    scenario: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    resolved_by_declaration = {
        declaration: next(
            (
                candidate.resolve()
                for candidate in _provenance_candidates(
                    declaration, repo_root=repo_root
                )
                if candidate.is_file()
            ),
            None,
        )
        for declaration in _paths(scenario)
    }
    declaration_by_resolved = {
        resolved: declaration
        for declaration, resolved in resolved_by_declaration.items()
        if resolved is not None
    }
    runtime: list[str] = []
    from domains.power_grid.backends.opendss_ieee13 import (
        _resolve_native_include_graph,
    )

    def append_graph(
        root_ref: Any,
        *,
        windows_compile_aliases: dict[str, Path] | None = None,
    ) -> Path | None:
        if not root_ref:
            return None
        declarations = _provenance_match(scenario, str(root_ref)) or (
            str(root_ref),
        )
        root_declaration = declarations[0]
        root = resolved_by_declaration.get(root_declaration)
        if root is None:
            root = next(
                (
                    candidate.resolve()
                    for candidate in _provenance_candidates(
                        root_declaration, repo_root=repo_root
                    )
                    if candidate.is_file()
                ),
                None,
            )
        if root is None:
            runtime.extend(declarations)
            return None
        assets, _inventory = _resolve_native_include_graph(
            root,
            windows_compile_aliases=windows_compile_aliases,
        )
        for asset in assets:
            resolved = Path(str(asset["path"])).resolve()
            declaration = declaration_by_resolved.get(resolved)
            if declaration is None:
                declaration = _portable_graph_path(
                    resolved,
                    declared_root_file=root_declaration,
                    repo_root=repo_root,
                )
            runtime.append(declaration)
        return root

    append_graph(_exact_config_path(scenario, "master_file"))
    profile = (scenario.get("backend_config") or {}).get("source_profile") or {}
    if isinstance(profile, dict):
        append_graph(profile.get("definition_file"))
        configured_data = profile.get("data_files") or [profile.get("data_file")]
        runtime.extend(
            value
            for path in configured_data
            if path
            for value in (
                _provenance_match(scenario, str(path)) or (str(path),)
            )
        )
    duty_program = (
        (scenario.get("backend_config") or {}).get("native_duty_program") or {}
    )
    if isinstance(duty_program, dict):
        scenario_file = duty_program.get("scenario_file")
        if scenario_file:
            runtime.extend(
                _provenance_match(scenario, str(scenario_file))
                or (str(scenario_file),)
            )
        data_file = duty_program.get("data_file")
        if data_file:
            runtime.extend(
                _provenance_match(scenario, str(data_file))
                or (str(data_file),)
            )
    contract = _contract(runtime=tuple(runtime))
    contract["file_sha256s"] = {
        declaration: hashlib.sha256(resolved.read_bytes()).hexdigest()
        for declaration in contract["runtime_input"]
        if (
            resolved := next(
                (
                    candidate.resolve()
                    for candidate in _provenance_candidates(
                        declaration, repo_root=repo_root
                    )
                    if candidate.is_file()
                ),
                None,
            )
        )
        is not None
    }
    return contract


def opendss_fresh_feeders(
    scenario: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    return _opendss_contract(scenario, repo_root)


def opendss_ieee13(
    scenario: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    return _opendss_contract(scenario, repo_root)


def _nrel_window_contract(
    scenario: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    del repo_root
    site = str((scenario.get("backend_config") or {}).get("site") or "")
    derivation = tuple(
        value
        for value in _paths(scenario)
        if Path(value).name == f"{site}.npz"
    )
    metadata = tuple(
        value for value in _paths(scenario) if value not in derivation
    )
    recipe = (scenario.get("backend_config") or {}).get(
        "derivation_recipe"
    ) or {}
    digest = str(
        recipe.get("source_window_sha256")
        or ((scenario.get("provenance") or {}).get("time_window") or {}).get(
            "source_window_sha256"
        )
        or ""
    )
    return _contract(
        derivation=derivation,
        metadata=metadata,
        window_sha256=digest or None,
        recipe_version=str(recipe.get("pipeline_version") or "nrel-window-v1"),
    )


def pandapower_lv(
    scenario: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    return _nrel_window_contract(scenario, repo_root)


def cigre_distribution(
    scenario: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Describe the pandapower-generated CIGRE feeder source.

    The CIGRE and Oberrhein networks are constructors in the locked
    pandapower release rather than repository files.  Keep the constructor
    URI in the derivation role so the source gate can fail closed until a
    runtime adapter proves that the locked upstream asset was consumed.
    It must not be represented as a runtime file merely because the
    scenario provenance contains a URI.
    """
    del repo_root
    source_files = tuple(_paths(scenario))
    virtual_source = next(
        (
            value
            for value in source_files
            if virtual_source_identity_sha256(value) is not None
        ),
        None,
    )
    recipe = (scenario.get("backend_config") or {}).get(
        "derivation_recipe"
    ) or {}
    runtime_profile = (scenario.get("backend_config") or {}).get(
        "long_horizon_candidate"
    ) or {}
    recipe_window = str(recipe.get("source_window_sha256") or "")
    recipe_version = str(
        runtime_profile.get("pipeline_version")
        or recipe.get("pipeline_version")
        or ""
    )
    has_profile_window = bool(recipe_window and recipe_version)
    window_sha256 = recipe_window if has_profile_window else (
        virtual_source_identity_sha256(virtual_source)
        if virtual_source is not None
        else None
    )
    return _contract(
        derivation=(virtual_source,) if virtual_source is not None else (),
        implementation=("domains/power_grid/backends/cigre_distribution.py",),
        metadata=("domains/power_grid/seeds/from_cigre.py",),
        window_sha256=window_sha256,
        recipe_version=(
            recipe_version
            if has_profile_window
            else "pandapower_constructor_runtime_v1"
            if window_sha256 is not None
            else None
        ),
    )


def pymgrid_economic_dispatch(
    scenario: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    return _nrel_window_contract(scenario, repo_root)


def sumo(
    scenario: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    config = scenario.get("backend_config") or {}
    files = config.get("sumo365_files") or {}
    direct = tuple(
        str(value)
        for value in (
            config.get("sumo_config_path") or files.get("sumocfg"),
            config.get("sumo_net_path") or files.get("network"),
            config.get("sumo_route_path") or files.get("route"),
        )
        if value
    )
    sumocfg = config.get("sumo_config_path") or files.get("sumocfg")
    if not sumocfg:
        return _contract(runtime=direct)
    declared_config = Path(str(sumocfg))
    if (
        declared_config.is_absolute()
        or len(declared_config.parts) < 3
        or declared_config.parts[0] != "works"
    ):
        raise ValueError("sumo_config_path_not_portable_works_alias")
    declared_root = Path(*declared_config.parts[:2])
    resolved_root = (repo_root / declared_root).resolve()
    path = (repo_root / declared_config).resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("sumo_config_outside_declared_root") from exc

    from domains.traffic.runtime_control_contract import (
        resolve_sumocfg_asset_graph,
    )

    graph = resolve_sumocfg_asset_graph(path)

    def portable_alias(raw: Any) -> str:
        resolved = Path(str(raw)).resolve()
        try:
            relative = resolved.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(
                "sumo_runtime_input_outside_declared_root"
            ) from exc
        alias = (declared_root / relative).as_posix()
        if (repo_root / alias).resolve() != resolved:
            raise ValueError("sumo_runtime_alias_identity_mismatch")
        return alias

    runtime = [
        portable_alias(graph["sumocfg"]["path"]),
        portable_alias(graph["network"]["path"]),
        *[portable_alias(row["path"]) for row in graph["route_files"]],
        *[portable_alias(row["path"]) for row in graph["additional_files"]],
        *[portable_alias(row["path"]) for row in graph["recursive_inputs"]],
    ]
    return _contract(runtime=tuple(runtime))


def mock_sumo(
    scenario: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    del scenario, repo_root
    return _contract()
