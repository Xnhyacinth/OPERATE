from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from core.event_protocol import (
    audit_event_decision_contract,
    resolve_event_decision,
)
from domains.power_grid.backends import opendss_ieee13


def _trace(root: Path) -> dict:
    asset = root / "works" / "OpenDSS-IEEE13" / "13Bus" / "Master.dss"
    return opendss_ieee13._native_protocol21_trace(
        assets=[
            {
                "path": str(asset),
                "sha256": "a" * 64,
                "role": "compile_master",
            }
        ],
        inventory={"circuit": 1},
        dss_version="fixture",
        circuit=SimpleNamespace(Name="fixture"),
        summary={
            "converged": True,
            "n_buses": 1,
            "n_nodes": 1,
            "n_lines": 0,
            "n_loads": 0,
            "n_transformers": 0,
            "n_capacitors": 0,
            "n_regcontrols": 0,
        },
    )


def test_opendss_trace_canonicalizes_paths_before_semantic_digests(
    tmp_path: Path, monkeypatch
) -> None:
    first_root = tmp_path / "first-clone"
    second_root = tmp_path / "second-clone"
    monkeypatch.setattr(opendss_ieee13, "REPO_ROOT", first_root)
    first = _trace(first_root)
    monkeypatch.setattr(opendss_ieee13, "REPO_ROOT", second_root)
    second = _trace(second_root)

    assert first["runtime_opened_assets"] == second["runtime_opened_assets"]
    assert first["parser_output_digest"] == second["parser_output_digest"]
    assert first["trace_semantic_digest"] == second["trace_semantic_digest"]
    assert first["opened_source_paths"] == [
        "works/OpenDSS-IEEE13/13Bus/Master.dss"
    ]
    assert all(not Path(value).is_absolute() for value in first["opened_source_sha256"])


def test_opendss_include_graph_counts_quoted_object_declarations(
    tmp_path: Path,
) -> None:
    master = tmp_path / "master.dss"
    master.write_text(
        "\n".join(
            (
                "New Circuit.fixture bus1=source",
                'New "Line.quoted_double" bus1=source bus2=load',
                "New 'Load.quoted_single' bus1=load kw=1",
                "New object=Transformer.unquoted phases=1",
            )
        ),
        encoding="utf-8",
    )

    _, inventory = opendss_ieee13._resolve_native_include_graph(master)

    assert inventory == {
        "circuit": 1,
        "line": 1,
        "load": 1,
        "transformer": 1,
    }


def test_opendss_runtime_include_graph_resolves_bracketed_compile(
    tmp_path: Path,
) -> None:
    master = tmp_path / "Run.dss"
    included = tmp_path / "Master.dss"
    master.write_text("Compile [Master.dss]\n", encoding="utf-8")
    included.write_text("New Circuit.fixture bus1=source\n", encoding="utf-8")

    assets, inventory = opendss_ieee13._resolve_native_include_graph(master)

    assert [Path(row["path"]) for row in assets] == [
        master.resolve(),
        included.resolve(),
    ]
    assert inventory == {"circuit": 1}


def test_ieee13_visible_load_surge_has_typed_immediate_wakeup() -> None:
    backend = opendss_ieee13.OpenDssIeee13Backend()
    seed = SimpleNamespace(
        horizon_ticks=4,
        perturbations=[
            {
                "kind": "load_surge",
                "trigger_tick": 1,
                "duration_ticks": 1,
                "hidden": False,
                "target": {"load_fraction": 0.2},
                "intensity": 0.2,
            }
        ],
        backend_config={},
    )

    backend.reset(seed)
    backend.tick(0)
    record = backend.tick(1)
    cleared_record = backend.tick(2)

    event = next(
        event
        for event in record.realized_events
        if event.get("type") == "load_surge"
    )
    assert event["event_class"] == "alarm"
    assert event["actionable"] is True
    assert event["decision_required"] is True
    assert audit_event_decision_contract(event, event_index=0) is None
    assert resolve_event_decision(event).requires_decision is True

    cleared = next(
        event
        for event in cleared_record.realized_events
        if event.get("type") == "load_surge_cleared"
    )
    assert cleared["event_class"] == "lifecycle"
    assert cleared["actionable"] is False
    assert cleared["decision_required"] is False
    assert audit_event_decision_contract(cleared, event_index=0) is None


def test_ieee13_hidden_load_surge_is_typed_without_wakeup() -> None:
    backend = opendss_ieee13.OpenDssIeee13Backend()
    seed = SimpleNamespace(
        horizon_ticks=4,
        perturbations=[
            {
                "kind": "load_surge",
                "trigger_tick": 1,
                "duration_ticks": 1,
                "hidden": True,
                "target": {"load_fraction": 0.2},
                "intensity": 0.2,
            }
        ],
        backend_config={},
    )

    backend.reset(seed)
    backend.tick(0)
    record = backend.tick(1)

    event = next(
        event
        for event in record.realized_events
        if event.get("type") == "load_surge"
    )
    assert event["event_class"] == "alarm"
    assert event["hidden"] is True
    assert event["actionable"] is False
    assert event["decision_required"] is False
    assert audit_event_decision_contract(event, event_index=0) is None
    assert resolve_event_decision(event).requires_decision is False


def test_ieee13_terminal_load_surge_has_no_response_window() -> None:
    backend = opendss_ieee13.OpenDssIeee13Backend()
    seed = SimpleNamespace(
        horizon_ticks=2,
        perturbations=[
            {
                "kind": "load_surge",
                "trigger_tick": 1,
                "duration_ticks": 1,
                "hidden": False,
                "target": {"load_fraction": 0.2},
                "intensity": 0.2,
            }
        ],
        backend_config={},
    )

    backend.reset(seed)
    backend.tick(0)
    record = backend.tick(1)

    event = next(
        event
        for event in record.realized_events
        if event.get("type") == "load_surge"
    )
    assert event["materiality_passed"] is True
    assert event["actionable"] is False
    assert event["decision_required"] is False
    assert event["response_window_required"] is False
    assert event["response_opportunity_tick"] is None
    assert resolve_event_decision(event).requires_decision is False


def test_ieee13_rejects_unregistered_perturbation_kind(tmp_path: Path) -> None:
    backend = opendss_ieee13.OpenDssIeee13Backend(source_root=tmp_path)
    seed = SimpleNamespace(
        horizon_ticks=2,
        perturbations=[{"kind": "unregistered_grid_event"}],
        backend_config={},
    )

    with pytest.raises(ValueError, match="unsupported OpenDSS IEEE13 event kind"):
        backend.reset(seed)


def test_ieee13_rejects_mismatched_declared_event_class(tmp_path: Path) -> None:
    backend = opendss_ieee13.OpenDssIeee13Backend(source_root=tmp_path)
    seed = SimpleNamespace(
        horizon_ticks=2,
        perturbations=[{"kind": "load_surge", "event_class": "routine"}],
        backend_config={},
    )

    with pytest.raises(
        ValueError,
        match="OpenDSS IEEE13 event class does not match registry",
    ):
        backend.reset(seed)
