#!/usr/bin/env python3
"""Build a bounded, source-first Microgrid Protocol-2.1 expansion cohort.

The pipeline treats source identity and native behavior as separate phases:

1. construct an exclusion-aware graph over the 16 locked NREL/OEDI profiles;
2. choose non-overlapping LV windows without looking at replay outcomes;
3. run bounded wait/reference screens and one exact wait replay only for the
   selected stress/controller setting;
4. emit YAML and a working-set row only for essential-gate survivors.

It never edits a frozen release and does not run the full twelve-stage gate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.implementation_identity import implementation_identity  # noqa: E402
from core.protocol21_evidence import required_semantics  # noqa: E402
from core.scenario_validator import validate_scenario_yaml  # noqa: E402
from core.source_asset_contract import (  # noqa: E402
    physical_source_lock_from_contract,
    resolve_source_asset_contract,
)
from domains.microgrid.seeds.from_nrel_microgrid import (  # noqa: E402
    baked_overlay_provenance_report,
)
from runner.resume import recompute_signature_with_seed  # noqa: E402
from scripts.audit_core_difficulty import _semantic_fingerprint  # noqa: E402
from scripts.build_microgrid_held_refine import _apply_refinement  # noqa: E402
from scripts.build_microgrid_native_state_loss_candidates import (  # noqa: E402
    _build_body as _build_ems_body,
)
from scripts.build_microgrid_protocol21_candidates import (  # noqa: E402
    _build_body as _build_lv_body,
)
from scripts.build_primary_suite import structural_fingerprint  # noqa: E402
from scripts.calibrate_grounded_microgrid_candidates import (  # noqa: E402
    NREL_DIR,
    rank_windows,
)
from scripts.calibrate_core_candidate import _episode  # noqa: E402

DEFAULT_SELECTION = (
    REPO_ROOT
    / "reports/protocol21_latest_quality_union_full_current_20260814"
    / "refined_core_selection_protocol2_v21.json"
)
DEFAULT_STAGING = (
    REPO_ROOT / "scenarios/staging/microgrid_expansion_20260814_v58"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "reports/microgrid_expansion_20260814_v58"

SITES: tuple[str, ...] = (
    "albuquerque_nm",
    "atlanta_ga",
    "boston_ma",
    "chicago_il",
    "columbus_oh",
    "denver_co",
    "las_vegas_nv",
    "miami_fl",
    "minneapolis_mn",
    "nashville_tn",
    "phoenix_az",
    "portland_or",
    "sacramento_ca",
    "salt_lake_city_ut",
    "seattle_wa",
    "tucson_az",
)

# These small grids reuse already demonstrated native stress ranges. They alter
# no source bytes, event schedule, safety threshold, or task definition.
LV_PROBES: dict[str, tuple[tuple[float, float], ...]] = {
    "high": (
        (3.0, 3.0),
        (2.5, 2.5),
        (2.0, 3.0),
        (3.0, 2.0),
        (2.5, 3.0),
        (3.0, 2.5),
    ),
    "extreme": (
        (2.0, 3.0),
        (1.5, 3.0),
        (2.5, 3.0),
        (2.0, 2.5),
        (1.5, 2.5),
        (2.5, 2.5),
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _write_immutable(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _select_disjoint_windows(
    ranked: list[dict[str, Any]],
    *,
    horizon_ticks: int,
    occupied: list[tuple[int, int]],
    limit: int,
) -> list[dict[str, Any]]:
    """Select ranked source windows without overlap or outcome feedback."""
    selected: list[dict[str, Any]] = []
    intervals = list(occupied)
    for row in ranked:
        start = int(row["start_index"])
        stop = start + horizon_ticks
        if any(start < right and left < stop for left, right in intervals):
            continue
        selected.append(dict(row))
        intervals.append((start, stop))
        if len(selected) >= limit:
            break
    return selected


def _source_consumed(episode: dict[str, Any]) -> bool:
    evidence = dict(episode.get("source_consumption_evidence") or {})
    return (
        evidence.get("status") == "passed"
        and evidence.get("source_state_effect_observed") is True
    )


def _classify_probe(
    *,
    backend_kind: str,
    difficulty_level: str,
    event_ticks: list[int],
    wait_first: dict[str, Any],
    wait_second: dict[str, Any] | None,
    reference: dict[str, Any],
) -> dict[str, Any]:
    """Apply only the essential, native candidate gates."""
    reference_task = dict(reference.get("task_completion") or {})
    wait_task = dict(wait_first.get("task_completion") or {})
    task_evidence = dict(reference_task.get("evidence") or {})
    if backend_kind == "pymgrid_economic_dispatch":
        benefit_kind = "task_loss_reduction"
        benefit_value = float(task_evidence.get("task_loss_reduction") or 0.0)
    else:
        benefit_kind = "cost_and_counterfactual_prevention"
        cost_gap = float(wait_first.get("cost") or 0.0) - float(
            reference.get("cost") or 0.0
        )
        prevented_loss = float(reference.get("prevented_loss") or 0.0)
        benefit_value = min(cost_gap, prevented_loss)

    state_ticks = sorted(
        {int(tick) for tick in reference.get("effective_state_changing_ticks") or []}
    )
    response_by_event = {
        str(tick): any(effect_tick >= tick for effect_tick in state_ticks)
        for tick in sorted(set(event_ticks))
    }
    survival = float(
        (reference.get("native_dimension_scores") or {}).get(
            "system_survival", 0.0
        )
        or 0.0
    )
    minimum_effect_ticks = 3 if difficulty_level == "extreme" else 2
    replay_episodes = [wait_first, reference]
    if wait_second is not None:
        replay_episodes.append(wait_second)
    checks = {
        "source_consumption": all(_source_consumed(row) for row in replay_episodes),
        "deterministic_wait_replay": (
            wait_second is not None and wait_first == wait_second
        ),
        "tool_protocol_state_effect": (
            int(reference.get("successful_state_changing_calls") or 0) > 0
            and bool(state_ticks)
        ),
        "positive_native_benefit": benefit_value > 1e-9,
        "reference_safety": (
            survival >= 100.0
            and reference_task.get("reason_code") != "catastrophic_outcome"
        ),
        "terminal_integrity": (
            (reference.get("terminal_integrity") or {}).get("release_ready") is True
        ),
        "native_task_completion": (
            reference_task.get("completed") is True
            and wait_task.get("completed") is not True
        ),
        "post_event_response": bool(response_by_event)
        and all(response_by_event.values()),
        "difficulty_control_depth": len(state_ticks) >= minimum_effect_ticks,
    }
    return {
        "essential_survivor": all(checks.values()),
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
        "native_benefit": {
            "kind": benefit_kind,
            "value": round(benefit_value, 9),
        },
        "reference_system_survival": survival,
        "event_ticks": sorted(set(event_ticks)),
        "post_event_response": response_by_event,
        "effective_state_changing_ticks": state_ticks,
        "minimum_effect_ticks": minimum_effect_ticks,
    }


def _compact_episode(episode: dict[str, Any]) -> dict[str, Any]:
    source = dict(episode.get("source_consumption_evidence") or {})
    return {
        "fingerprint": _digest(episode),
        "cost": episode.get("cost"),
        "prevented_loss": episode.get("prevented_loss"),
        "successful_state_changing_calls": episode.get(
            "successful_state_changing_calls"
        ),
        "effective_tool_names": episode.get("effective_tool_names"),
        "effective_state_changing_ticks": episode.get(
            "effective_state_changing_ticks"
        ),
        "native_dimension_scores": episode.get("native_dimension_scores"),
        "task_completion": episode.get("task_completion"),
        "terminal_integrity": episode.get("terminal_integrity"),
        "source_consumption": {
            "status": source.get("status"),
            "source_state_effect_observed": source.get(
                "source_state_effect_observed"
            ),
            "consumed_window_sha256": source.get("consumed_window_sha256"),
            "consumption_ticks": source.get("consumption_ticks"),
            "opened_source_sha256": source.get("opened_source_sha256"),
        },
    }


def _episode_row(body: dict[str, Any], *, path: str = "candidate-only-in-memory") -> dict[str, Any]:
    return {
        "scenario_id": body["scenario_id"],
        "scenario_signature": body["scenario_signature"],
        "path": path,
        "domain": body["domain"],
        "backend_kind": body["backend_kind"],
        "family": body["family"],
        "difficulty_mode": body["difficulty_mode"],
        "difficulty_level": body["difficulty_level"],
    }


def _event_ticks(body: dict[str, Any]) -> list[int]:
    return sorted(
        {
            int(event.get("trigger_tick") or 0)
            for event in body.get("perturbations") or []
        }
    )


def _screen_body(body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    row = _episode_row(body)
    wait = _episode(row, body, "wait_only", replay_index=0)
    reference = _episode(row, body, "oracle_offline", replay_index=0)
    classification = _classify_probe(
        backend_kind=str(body["backend_kind"]),
        difficulty_level=str(body["difficulty_level"]),
        event_ticks=_event_ticks(body),
        wait_first=wait,
        wait_second=None,
        reference=reference,
    )
    non_replay_checks = {
        key: value
        for key, value in classification["checks"].items()
        if key != "deterministic_wait_replay"
    }
    return {
        "screen_passed": all(non_replay_checks.values()),
        "classification": classification,
        "wait": _compact_episode(wait),
        "reference": _compact_episode(reference),
    }, {"wait": wait, "reference": reference}


def _confirm_body(
    body: dict[str, Any],
    *,
    wait_first: dict[str, Any],
    reference: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    row = _episode_row(body)
    wait_second = _episode(row, body, "wait_only", replay_index=1)
    classification = _classify_probe(
        backend_kind=str(body["backend_kind"]),
        difficulty_level=str(body["difficulty_level"]),
        event_ticks=_event_ticks(body),
        wait_first=wait_first,
        wait_second=wait_second,
        reference=reference,
    )
    return classification, _compact_episode(wait_second)


def _load_selection_rows(selection_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    rows = payload.get("scenarios") or payload.get("selected") or payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("selection has no scenario rows")
    return [dict(row) for row in rows if row.get("domain") == "microgrid"]


def _load_scenario(path_value: str) -> dict[str, Any]:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ValueError(f"scenario is not an object: {path}")
    return body


def _existing_contract(
    selection_path: Path,
) -> tuple[set[str], set[str], dict[str, list[tuple[int, int]]], list[dict[str, Any]]]:
    rows = _load_selection_rows(selection_path)
    source_keys = {str(row.get("source_denominator_key") or "") for row in rows}
    ems_sites: set[str] = set()
    occupied: dict[str, list[tuple[int, int]]] = defaultdict(list)
    evidence: list[dict[str, Any]] = []
    for row in rows:
        body = _load_scenario(str(row["path"]))
        config = dict(body.get("backend_config") or {})
        recipe = dict(config.get("derivation_recipe") or {})
        site = str(config.get("site") or "")
        start = int(config.get("profile_start_index") or recipe.get("profile_start_index") or 0)
        horizon = int(body.get("horizon_ticks") or 0)
        if site and horizon > 0:
            occupied[site].append((start, start + horizon))
        if body.get("backend_kind") == "pymgrid_economic_dispatch":
            ems_sites.add(site)
        evidence.append(
            {
                "scenario_id": row.get("scenario_id"),
                "backend_kind": body.get("backend_kind"),
                "site": site,
                "source_interval": [start, start + horizon],
                "source_denominator_key": row.get("source_denominator_key"),
            }
        )
    return source_keys, ems_sites, occupied, evidence


def _annotate_body(body: dict[str, Any], *, source_node_id: str) -> dict[str, Any]:
    config = dict(body.get("backend_config") or {})
    config["microgrid_expansion_v58"] = {
        "pipeline": "microgrid_source_identity_native_refine_v58",
        "source_node_id": source_node_id,
        "candidate_only": True,
        "safety_threshold_unchanged": True,
        "event_schedule_unchanged": True,
        "source_profile_unchanged": True,
    }
    body["backend_config"] = config
    body["scenario_signature"] = recompute_signature_with_seed(
        body, int(body["seed"])
    )
    return body


def build_source_identity_graph(
    *, selection_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build source nodes and candidate bodies before any native replay."""
    existing_keys, existing_ems_sites, occupied, existing_rows = _existing_contract(
        selection_path
    )
    specs: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []

    # Reserve every new EMS 24-hour physical window before selecting LV
    # windows, so the two backend lines do not reuse overlapping raw intervals.
    for site_index, site in enumerate(SITES):
        if site in existing_ems_sites:
            continue
        occupied[site].append((0, 24))
        node_id = f"nrel_oedi:{site}:0:24:pymgrid_economic_dispatch"
        variants: list[dict[str, Any]] = []
        for level in ("high", "extreme"):
            case = {
                "level": level,
                "seed": 8300 + site_index * 2 + (level == "extreme"),
                "site": site,
                "event_tick": 4 if level == "high" else 3,
                "duration_ticks": 4 if level == "high" else 6,
                "load_spike_intensity": 0.35 if level == "high" else 0.55,
                "slug": f"{site}_{level}_v58",
            }
            body = _annotate_body(_build_ems_body(case), source_node_id=node_id)
            source_key = str(body["backend_config"]["source_denominator_key"])
            if source_key in existing_keys:
                raise ValueError(f"existing EMS source was reintroduced: {source_key}")
            variants.append(body)
        specs.append(
            {
                "node_id": node_id,
                "backend_kind": "pymgrid_economic_dispatch",
                "site": site,
                "source_interval": [0, 24],
                "source_denominator_key": variants[0]["backend_config"][
                    "source_denominator_key"
                ],
                "variants": variants,
            }
        )

    for site_index, site in enumerate(SITES):
        for level, horizon in (("high", 6), ("extreme", 10)):
            ranked = rank_windows(site, horizon=horizon, limit=128)
            selected = _select_disjoint_windows(
                ranked,
                horizon_ticks=horizon,
                occupied=occupied[site],
                limit=1,
            )
            if not selected:
                raise ValueError(f"{site}: no non-overlapping {level} source window")
            window = selected[0]
            start = int(window["start_index"])
            occupied[site].append((start, start + horizon))
            node_id = f"nrel_oedi:{site}:{start}:{horizon}:pandapower_lv"
            case = {
                "site": site,
                "level": level,
                "seed": 8700 + site_index * 2 + (level == "extreme"),
                "start_index": start,
                "slug": f"{site}_{level}_p{start}_v58",
            }
            body = _annotate_body(_build_lv_body(case), source_node_id=node_id)
            source_key = str(body["backend_config"]["source_denominator_key"])
            if source_key in existing_keys:
                raise ValueError(f"existing LV source was reintroduced: {source_key}")
            specs.append(
                {
                    "node_id": node_id,
                    "backend_kind": "pandapower_lv",
                    "site": site,
                    "source_interval": [start, start + horizon],
                    "source_denominator_key": source_key,
                    "window_score": window["window_score"],
                    "variants": [body],
                }
            )

    seen_keys: set[str] = set()
    for spec in specs:
        source_key = str(spec["source_denominator_key"])
        if source_key in seen_keys:
            raise ValueError(f"duplicate candidate effective source: {source_key}")
        seen_keys.add(source_key)
        site = str(spec["site"])
        sidecar = baked_overlay_provenance_report(NREL_DIR / f"{site}.npz")
        if sidecar.get("valid") is not True:
            raise ValueError(f"{site}: invalid source sidecar")
        source_node = f"source_asset:{site}"
        nodes.extend(
            [
                {
                    "node_id": source_node,
                    "kind": "locked_source_asset",
                    "site": site,
                    "profile_path": _relative(NREL_DIR / f"{site}.npz"),
                    "profile_sha256": _sha256(NREL_DIR / f"{site}.npz"),
                    "provenance_path": _relative(
                        NREL_DIR / f"{site}.provenance.json"
                    ),
                    "provenance_sha256": _sha256(
                        NREL_DIR / f"{site}.provenance.json"
                    ),
                },
                {
                    "node_id": spec["node_id"],
                    "kind": "effective_source_window",
                    "backend_kind": spec["backend_kind"],
                    "site": site,
                    "source_interval": spec["source_interval"],
                    "source_denominator_key": source_key,
                    "candidate_levels": [
                        body["difficulty_level"] for body in spec["variants"]
                    ],
                },
            ]
        )
        edges.append(
            {
                "from": source_node,
                "to": str(spec["node_id"]),
                "relation": "derives_non_overlapping_runtime_window",
            }
        )

    graph = {
        "schema_version": "microgrid-source-identity-graph-v1",
        "status": "source_identity_graph_complete_before_native_replay",
        "input_selection": {
            "path": _relative(selection_path),
            "sha256": _sha256(selection_path),
            "n_excluded_microgrid_rows": len(existing_rows),
        },
        "policy": {
            "source_only_window_selection": True,
            "outcome_feedback_used_for_window_selection": False,
            "raw_source_intervals_non_overlapping_within_site": True,
            "existing_v57_effective_sources_excluded": True,
            "one_candidate_per_effective_source": True,
        },
        "summary": {
            "n_locked_sites": len(SITES),
            "n_existing_microgrid_rows_excluded": len(existing_rows),
            "n_effective_source_nodes": len(specs),
            "backend_counts": dict(
                sorted(Counter(spec["backend_kind"] for spec in specs).items())
            ),
        },
        "excluded_rows": existing_rows,
        "nodes": nodes,
        "edges": edges,
    }
    return graph, specs


def _scan_lv(spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    original = spec["variants"][0]
    probes: list[dict[str, Any]] = []
    bodies: dict[tuple[float, float], dict[str, Any]] = {}
    full_episodes: dict[tuple[float, float], dict[str, Any]] = {}
    level = str(original["difficulty_level"])
    for pv_scale, load_intensity in LV_PROBES[level]:
        body = _apply_refinement(
            copy.deepcopy(original),
            pv_scale=pv_scale,
            load_intensity=load_intensity,
        )
        body["scenario_signature"] = recompute_signature_with_seed(
            body, int(body["seed"])
        )
        validation_errors = validate_scenario_yaml(body)
        if validation_errors:
            probes.append(
                {
                    "pv_scale": pv_scale,
                    "load_intensity": load_intensity,
                    "screen_passed": False,
                    "schema_validation_errors": validation_errors,
                }
            )
            continue
        screen, episodes = _screen_body(body)
        screen.update(
            {
                "pv_scale": pv_scale,
                "load_intensity": load_intensity,
                "schema_validation_errors": [],
            }
        )
        probes.append(screen)
        bodies[(pv_scale, load_intensity)] = body
        full_episodes[(pv_scale, load_intensity)] = episodes

    passing = [row for row in probes if row.get("screen_passed") is True]
    if not passing:
        return {
            "source_node_id": spec["node_id"],
            "disposition": "held_essential_gate",
            "reason_codes": sorted(
                {
                    failure
                    for probe in probes
                    for failure in (probe.get("classification") or {}).get(
                        "failures", []
                    )
                    if failure != "deterministic_wait_replay"
                }
            ),
            "probes": probes,
        }, None
    selected = max(
        passing,
        key=lambda row: (
            float(row["pv_scale"]) * float(row["load_intensity"]),
            float(row["load_intensity"]),
            float(row["pv_scale"]),
        ),
    )
    key = (float(selected["pv_scale"]), float(selected["load_intensity"]))
    body = bodies[key]
    episodes = full_episodes[key]
    confirmed, wait_second = _confirm_body(
        body,
        wait_first=episodes["wait"],
        reference=episodes["reference"],
    )
    disposition = (
        "essential_survivor"
        if confirmed["essential_survivor"]
        else "held_essential_gate"
    )
    return {
        "source_node_id": spec["node_id"],
        "disposition": disposition,
        "reason_codes": [] if disposition == "essential_survivor" else confirmed["failures"],
        "selected_setting": {
            "pv_scale": selected["pv_scale"],
            "load_intensity": selected["load_intensity"],
        },
        "confirmation": confirmed,
        "wait_first": selected["wait"],
        "wait_second": wait_second,
        "reference": selected["reference"],
        "probes": probes,
    }, body if disposition == "essential_survivor" else None


def _scan_ems(spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    probes: list[dict[str, Any]] = []
    bodies: dict[str, dict[str, Any]] = {}
    full_episodes: dict[str, dict[str, Any]] = {}
    for body in spec["variants"]:
        validation_errors = validate_scenario_yaml(body)
        level = str(body["difficulty_level"])
        if validation_errors:
            probes.append(
                {
                    "difficulty_level": level,
                    "screen_passed": False,
                    "schema_validation_errors": validation_errors,
                }
            )
            continue
        screen, episodes = _screen_body(body)
        screen.update(
            {
                "difficulty_level": level,
                "schema_validation_errors": [],
            }
        )
        probes.append(screen)
        bodies[level] = body
        full_episodes[level] = episodes
    passing = [row for row in probes if row.get("screen_passed") is True]
    if not passing:
        return {
            "source_node_id": spec["node_id"],
            "disposition": "held_essential_gate",
            "reason_codes": sorted(
                {
                    failure
                    for probe in probes
                    for failure in (probe.get("classification") or {}).get(
                        "failures", []
                    )
                    if failure != "deterministic_wait_replay"
                }
            ),
            "probes": probes,
        }, None
    selected = max(
        passing,
        key=lambda row: (
            row["difficulty_level"] == "extreme",
            float(row["classification"]["native_benefit"]["value"]),
        ),
    )
    level = str(selected["difficulty_level"])
    body = bodies[level]
    episodes = full_episodes[level]
    confirmed, wait_second = _confirm_body(
        body,
        wait_first=episodes["wait"],
        reference=episodes["reference"],
    )
    disposition = (
        "essential_survivor"
        if confirmed["essential_survivor"]
        else "held_essential_gate"
    )
    return {
        "source_node_id": spec["node_id"],
        "disposition": disposition,
        "reason_codes": [] if disposition == "essential_survivor" else confirmed["failures"],
        "selected_setting": {"difficulty_level": level},
        "confirmation": confirmed,
        "wait_first": selected["wait"],
        "wait_second": wait_second,
        "reference": selected["reference"],
        "probes": probes,
    }, body if disposition == "essential_survivor" else None


def _candidate_row(body: dict[str, Any], path: Path) -> dict[str, Any]:
    config = dict(body.get("backend_config") or {})
    source_key = str(config.get("source_denominator_key") or "")
    if not source_key:
        raise ValueError(f"{body['scenario_id']}: source denominator key missing")
    contract = resolve_source_asset_contract(body, repo_root=REPO_ROOT)
    physical_source_lock = physical_source_lock_from_contract(
        contract, backend_kind=str(body["backend_kind"])
    )
    if physical_source_lock is None:
        raise ValueError(f"{body['scenario_id']}: verified physical source lock missing")
    semantic = _semantic_fingerprint(body)
    return {
        "scenario_id": body["scenario_id"],
        "path": _relative(path),
        "domain": body["domain"],
        "backend_kind": body["backend_kind"],
        "family": body["family"],
        "difficulty_mode": body["difficulty_mode"],
        "difficulty_level": body["difficulty_level"],
        "horizon_ticks": body["horizon_ticks"],
        "seed": body["seed"],
        "scenario_signature": body["scenario_signature"],
        "source_key": source_key,
        "source_denominator_key": source_key,
        "structural_fingerprint": structural_fingerprint(body),
        "semantic_fingerprint": semantic,
        "case_ledger": {
            "schema_version": "0.1",
            "source_denominator_key": source_key,
            "physical_source_lock": physical_source_lock,
            "independence_axis": "microgrid_backend_site_nonoverlapping_source_window",
            "decision_pressure_axis": "native_multistage_voltage_or_ems_recovery",
            "additional_decision_axis": (
                f"difficulty={body['difficulty_mode']}/{body['difficulty_level']}"
            ),
            "decision_variant_key": semantic,
            "complexity_tags": [
                f"n_perturbations={len(body.get('perturbations') or [])}",
                "source_consumed_native_backend",
                "post_event_response_proven",
            ],
            "event_repairs": [],
            "source_refinement": {
                "pipeline": "microgrid_source_identity_native_refine_v58",
                "candidate_path": _relative(path),
                "source_profile_unchanged": True,
            },
        },
        "protocol21_lineage": {
            "physical_identity_origin": "verified_source_asset_graph",
            "ready": True,
            "status": "ready_for_full_protocol21_replay",
            "reason_codes": [],
        },
        "status": "pending_protocol21_full_admission",
        "reason_codes": [
            "essential_native_replay_passed",
            "candidate_only_requires_full_protocol21",
        ],
    }


def build_source_suite(
    rows: list[dict[str, Any]], *, implementation_tree_sha256: str
) -> dict[str, Any]:
    source_keys = [str(row.get("source_denominator_key") or "") for row in rows]
    if len(source_keys) != len(set(source_keys)):
        raise ValueError("duplicate effective source identity in survivor suite")
    return {
        "schema_version": "protocol2.1-working-set-v1",
        "status": "working_set" if rows else "terminal_empty",
        "selection_policy": "essential_native_survivor_v58",
        "leaderboard_eligible": False,
        "release_ready": False,
        "n_scenarios": len(rows),
        "implementation_tree_sha256": implementation_tree_sha256,
        "admission_profile": "quality_core_v2",
        "evaluation_semantics": required_semantics(),
        "constraints": {
            "core_admission_profile": "quality_core_v2",
            "candidate_only": True,
            "formal_evaluation_ready": False,
            "model_outcomes_used_for_filtering": False,
            "one_per_effective_source_identity": True,
            "full_protocol21_replay_pending": True,
        },
        "scenarios": sorted(rows, key=lambda row: row["scenario_id"]),
    }


def build(
    *,
    selection_path: Path,
    staging_root: Path,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[Path, dict[str, Any]]]:
    start_identity = implementation_identity()
    graph, specs = build_source_identity_graph(selection_path=selection_path)
    outcomes: list[dict[str, Any]] = []
    survivor_bodies: dict[Path, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for spec in specs:
        outcome, body = (
            _scan_lv(spec)
            if spec["backend_kind"] == "pandapower_lv"
            else _scan_ems(spec)
        )
        outcome.update(
            {
                "backend_kind": spec["backend_kind"],
                "site": spec["site"],
                "source_interval": spec["source_interval"],
                "source_denominator_key": spec["source_denominator_key"],
            }
        )
        outcomes.append(outcome)
        if body is None:
            continue
        filename = f"{body['scenario_id'].replace('/', '__')}.yaml"
        path = staging_root / filename
        survivor_bodies[path] = body
        rows.append(_candidate_row(body, path))

    end_identity = implementation_identity()
    implementation_stable = (
        start_identity["implementation_tree_sha256"]
        == end_identity["implementation_tree_sha256"]
    )
    if not implementation_stable:
        rows = []
        survivor_bodies = {}
        for outcome in outcomes:
            if outcome["disposition"] == "essential_survivor":
                outcome["disposition"] = "held_implementation_tree_drift"
                outcome["reason_codes"] = ["implementation_tree_changed_during_replay"]

    suite = build_source_suite(
        rows,
        implementation_tree_sha256=str(
            end_identity["implementation_tree_sha256"]
        ),
    )
    rerun_command = (
        ".venv/bin/python scripts/build_microgrid_expansion_v58.py "
        f"--selection {_relative(selection_path)} "
        f"--staging-root {_relative(staging_root)} "
        f"--output-root {_relative(output_root)} --execute"
    )
    ledger = {
        "schema_version": "microgrid-expansion-terminal-ledger-v1",
        "status": "terminal_candidate_refine_complete",
        "candidate_only": True,
        "release_admission": False,
        "implementation_identity_start": start_identity,
        "implementation_identity_end": end_identity,
        "implementation_tree_stable": implementation_stable,
        "source_identity_graph_sha256": _digest(graph),
        "input_selection": graph["input_selection"],
        "policy": {
            "bounded_wait_reference_replay": True,
            "exact_wait_replay_only_for_screen_survivor": True,
            "source_consumption_required": True,
            "tool_protocol_state_effect_required": True,
            "positive_native_benefit_required": True,
            "system_survival_floor": 100.0,
            "terminal_and_task_completion_required": True,
            "post_event_response_required": True,
            "full_twelve_stage_run": False,
            "safety_threshold_lowered": False,
        },
        "summary": {
            "n_source_nodes": len(specs),
            "n_survivors": len(rows),
            "n_held": len(specs) - len(rows),
            "backend_survivor_counts": dict(
                sorted(Counter(row["backend_kind"] for row in rows).items())
            ),
            "difficulty_survivor_counts": dict(
                sorted(Counter(row["difficulty_level"] for row in rows).items())
            ),
            "site_survivor_count": len(
                {
                    str(outcome["site"])
                    for outcome in outcomes
                    if outcome["disposition"] == "essential_survivor"
                }
            ),
        },
        "artifact_contract": {
            "staging_root": _relative(staging_root),
            "source_suite": _relative(output_root / "source_suite.json"),
            "source_identity_graph": _relative(
                output_root / "source_identity_graph.json"
            ),
            "candidate_report": _relative(output_root / "candidate_report.json"),
            "rerun_command": rerun_command,
        },
        "outcomes": outcomes,
    }
    return graph, ledger, suite, survivor_bodies


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    selection_path = args.selection.resolve()
    staging_root = args.staging_root.resolve()
    output_root = args.output_root.resolve()
    for path in (staging_root, output_root):
        path.relative_to(REPO_ROOT.resolve())
    graph, ledger, suite, files = build(
        selection_path=selection_path,
        staging_root=staging_root,
        output_root=output_root,
    )
    report = {
        "schema_version": "microgrid-expansion-candidate-report-v1",
        "status": "essential_survivors_pending_full_protocol21",
        "candidate_only": True,
        "n_scenarios": suite["n_scenarios"],
        "scenarios": suite["scenarios"],
        "terminal_ledger_sha256": _digest(ledger),
    }
    if args.execute:
        for path, body in files.items():
            _write_immutable(path, yaml.safe_dump(body, sort_keys=False))
        _write_immutable(
            output_root / "source_identity_graph.json",
            json.dumps(graph, indent=2, sort_keys=True) + "\n",
        )
        _write_immutable(
            output_root / "candidate_report.json",
            json.dumps(report, indent=2, sort_keys=True) + "\n",
        )
        _write_immutable(
            output_root / "source_suite.json",
            json.dumps(suite, indent=2, sort_keys=True) + "\n",
        )
        _write_immutable(
            output_root / "terminal_ledger.json",
            json.dumps(ledger, indent=2, sort_keys=True) + "\n",
        )
    print(
        json.dumps(
            {
                "status": ledger["status"],
                "implementation_tree_stable": ledger[
                    "implementation_tree_stable"
                ],
                "summary": ledger["summary"],
                "rerun_command": ledger["artifact_contract"]["rerun_command"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
