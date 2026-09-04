"""Build dev-only logistics Job-Shop seeds from anchored JSPLIB samples.

The functions here are a release-grade skeleton for a future JSPLIB
materializer: source lock, structural axes, dimension applicability, and
canonical scorer-facing records are explicit. They deliberately do not wire
JSPLIB rows into any release registry.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from core.tool_protocol import DifficultyImperfectionProfile
from domains.logistics.backends.job_shop import JsplibJobShopBackend

from .schema import LogisticsScenarioSeed, Perturbation, Provenance
from .source_locks import provenance_lock_kwargs

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSPLIB_ROOT = REPO_ROOT / "works" / "JSPLIB-Instances"

_HONEST_ZERO_KEYS = [
    "reserves_required_mw",
    "reserves_procured_mw",
    "n_voltage_violations",
    "n_disconnected_lines",
]
_KEY_ALIASES = {
    "aggregate_demand_mw": "operations_total",
    "aggregate_generation_mw": "operations_scheduled",
    "balance_error_mw": "operations_unscheduled",
    "production_cost": "makespan",
    "startup_cost": "idle_time_proxy",
    "shed_penalty": "unscheduled_operation_penalty",
    "rho_max": "machine_queue_pressure",
    "n_overloads": "machine_conflict_count",
}


def parse_jsplib_instance(path: Path) -> dict[str, Any]:
    """Parse a classic JSPLIB job-shop instance file."""
    tokens: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        body = line.split("#", 1)[0].strip()
        if body:
            tokens.extend(body.split())
    values = [int(token) for token in tokens]
    if len(values) < 2:
        raise ValueError(f"{path} is missing the jobs/machines header")
    jobs, machines = values[0], values[1]
    if jobs <= 0 or machines <= 0:
        raise ValueError(f"{path} has non-positive jobs/machines: {jobs}/{machines}")
    expected = 2 + jobs * machines * 2
    if len(values) != expected:
        raise ValueError(
            f"{path} has {len(values)} integers; expected {expected} for "
            f"{jobs} jobs x {machines} machines"
        )

    cursor = 2
    rows: list[list[dict[str, int]]] = []
    machine_ids: set[int] = set()
    processing_times: list[int] = []
    for job_idx in range(jobs):
        row: list[dict[str, int]] = []
        seen_in_job: set[int] = set()
        for op_idx in range(machines):
            machine = values[cursor]
            duration = values[cursor + 1]
            cursor += 2
            if machine < 0 or machine >= machines:
                raise ValueError(
                    f"{path} job {job_idx} op {op_idx} references machine "
                    f"{machine}; expected 0..{machines - 1}"
                )
            if duration < 0:
                raise ValueError(
                    f"{path} job {job_idx} op {op_idx} has negative "
                    f"duration {duration}"
                )
            if machine in seen_in_job:
                raise ValueError(
                    f"{path} job {job_idx} repeats machine {machine}; JSPLIB "
                    "job rows should be permutations"
                )
            seen_in_job.add(machine)
            machine_ids.add(machine)
            processing_times.append(duration)
            row.append({"machine": machine, "duration": duration})
        rows.append(row)

    return {
        "jobs": jobs,
        "machines": machines,
        "operations": jobs * machines,
        "machine_ids": sorted(machine_ids),
        "total_processing_time": int(sum(processing_times)),
        "min_processing_time": int(min(processing_times)),
        "max_processing_time": int(max(processing_times)),
        "jobs_detail": rows,
    }


def load_jsplib_metadata(root: Path = DEFAULT_JSPLIB_ROOT) -> list[dict[str, Any]]:
    """Load the JSPLIB instances metadata list."""
    path = root / "instances.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON list")
    return [dict(item) for item in raw if isinstance(item, dict)]


def jsplib_checksums(
    root: Path = DEFAULT_JSPLIB_ROOT,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return checksum manifest entries plus repo metadata."""
    checksums: dict[str, str] = {}
    metadata = {
        "repo": "https://github.com/tamy0612/JSPLIB",
        "commit": "eea2b60dd7e2f5c907ff7302662c61812eb7efdf",
    }
    path = root / "CHECKSUMS.txt"
    if not path.exists():
        return checksums, metadata

    commit_re = re.compile(r"commit:\s*([0-9a-fA-F]{7,40})")
    repo_re = re.compile(r"repo:\s*(\S+)")
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            commit_match = commit_re.search(stripped)
            repo_match = repo_re.search(stripped)
            if commit_match:
                metadata["commit"] = commit_match.group(1)
            if repo_match:
                metadata["repo"] = repo_match.group(1)
            continue
        parts = stripped.split()
        if len(parts) >= 2 and re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
            checksums[parts[1]] = parts[0].lower()
    return checksums, metadata


def build_job_shop_dispatch_seed(
    *,
    instance: str,
    seed_id: str | None = None,
    seed: int = 42,
    difficulty_mode: str = "time_pressure",
    difficulty_level: str = "high",
    root: Path = DEFAULT_JSPLIB_ROOT,
) -> LogisticsScenarioSeed:
    """Build a non-release JSPLIB Job-Shop seed skeleton."""
    root = root.expanduser()
    metadata = _metadata_for_instance(instance, root)
    rel_path = str(metadata["path"])
    parsed = parse_jsplib_instance(root / rel_path)
    checksums, checksum_meta = jsplib_checksums(root)
    expected_sha = checksums.get(rel_path)
    actual_sha = _sha256(root / rel_path)
    if not expected_sha:
        raise ValueError(f"JSPLIB instance {instance!r} is missing CHECKSUMS entry")
    if actual_sha != expected_sha:
        raise ValueError(f"JSPLIB instance {instance!r} sha256 mismatch")

    lock = provenance_lock_kwargs("jsplib")
    reference = _reference(metadata)
    job_shop_block = {
        "instance_name": instance,
        "jobs": parsed["jobs"],
        "machines": parsed["machines"],
        "operations": parsed["operations"],
        "machine_ids": parsed["machine_ids"],
        "total_processing_time": parsed["total_processing_time"],
        "min_processing_time": parsed["min_processing_time"],
        "max_processing_time": parsed["max_processing_time"],
        "jobs_detail": parsed["jobs_detail"],
    }
    backend_config = {
        "instance_name": instance,
        "source_integration_rung": "adapted_from",
        "release_ready": False,
        "release_reentry_ready": False,
        "source_denominator_key": f"jsplib_job_shop:{instance}",
        "job_shop": job_shop_block,
        "reference": reference,
        "expected_sha256": expected_sha,
        "actual_sha256": actual_sha,
        "checksum_manifest": "works/JSPLIB-Instances/CHECKSUMS.txt",
        "source_axes": {
            "instance_name": instance,
            "jobs": parsed["jobs"],
            "machines": parsed["machines"],
            "operations": parsed["operations"],
            "reference_type": reference["type"],
        },
        "logistics_key_aliases": _KEY_ALIASES,
        "honest_zero_keys": list(_HONEST_ZERO_KEYS),
        "dimension_applicability": _dimension_applicability_for_reference(reference),
    }
    provenance = Provenance(
        data_source="jsplib",
        files=[
            f"works/JSPLIB-Instances/{rel_path}",
            "works/JSPLIB-Instances/CHECKSUMS.txt",
            "works/JSPLIB-Instances/instances.json",
        ],
        commit=checksum_meta.get("commit") or lock["commit"],
        url=checksum_meta.get("repo") or lock["url"],
        lock_strategy=lock["lock_strategy"],
        time_window={
            "objective": "minimize_makespan",
            "operations": parsed["operations"],
        },
        license=lock.get("license", "public academic OR benchmark mirror"),
        notes=(
            f"Parsed JSPLIB job-shop instance {instance!r}; "
            "non-release seed/scoring skeleton only."
        ),
    )
    imperfection = DifficultyImperfectionProfile().for_level(difficulty_level)
    fail_rate = float(imperfection["fail_rate"])
    delay_ticks = int(imperfection["delay_ticks"])
    operations = int(parsed["operations"])
    failed_calls = math.ceil(operations * fail_rate / max(0.01, 1.0 - fail_rate))
    retry_allowance = failed_calls * 2
    horizon_ticks = operations + retry_allowance + delay_ticks + 2
    backend_config["interaction_budget_basis"] = {
        "operations": operations,
        "tool_fail_rate": fail_rate,
        "tool_delay_ticks": delay_ticks,
        "retry_allowance_ticks": retry_allowance,
        "cooldown_after_failure_ticks": 1,
    }
    return LogisticsScenarioSeed(
        seed_id=seed_id
        or f"jobshop_{instance}_{difficulty_mode}_{difficulty_level}_s{seed}",
        family="job_shop_dispatch",
        backend_kind="jsplib_job_shop",
        backend_config=backend_config,
        horizon_ticks=horizon_ticks,
        tick_minutes=1,
        seed=seed,
        difficulty_mode=difficulty_mode,  # type: ignore[arg-type]
        difficulty_level=difficulty_level,  # type: ignore[arg-type]
        provenance=provenance,
    )


def build_job_shop_backend_from_seed(
    seed: LogisticsScenarioSeed,
) -> JsplibJobShopBackend:
    """Instantiate the dev-only backend from a Job-Shop seed skeleton."""
    return JsplibJobShopBackend.from_seed(seed)


def build_dynamic_job_shop_recovery_seed(
    *,
    instance: str,
    seed_id: str | None = None,
    seed: int = 42,
    difficulty_mode: str = "time_pressure",
    difficulty_level: str = "high",
    root: Path = DEFAULT_JSPLIB_ROOT,
    machine_id: int | None = None,
    trigger_tick: int | None = None,
    outage_duration_ticks: int | None = None,
) -> LogisticsScenarioSeed:
    """Build a dynamic, source-locked JSPLIB recovery candidate.

    The JSPLIB file remains the sole runtime source of job precedence,
    operation duration, and machine identity.  The outage and urgency events
    are deterministic, explicitly declared procedural perturbations over that
    locked machine/job set; they are never presented as source-observed data.
    """
    level = str(difficulty_level).lower()
    if level not in {"medium", "high", "extreme"}:
        raise ValueError("dynamic job-shop recovery requires medium/high/extreme")
    base = build_job_shop_dispatch_seed(
        instance=instance,
        seed_id=seed_id
        or f"jobshop_{instance}_dynamic_recovery_{level}_s{seed}",
        seed=seed,
        difficulty_mode=difficulty_mode,
        difficulty_level=level,
        root=root,
    )
    job_shop = base.backend_config["job_shop"]
    jobs = int(job_shop["jobs"])
    selected_machine = (
        int(machine_id)
        if machine_id is not None
        else int(job_shop["machine_ids"][0])
    )
    if selected_machine not in set(job_shop["machine_ids"]):
        raise ValueError("dynamic job-shop machine_id is not in locked source")
    default_trigger = max(1, min(base.horizon_ticks - 5, max(2, int(job_shop["operations"]) // 4)))
    outage_trigger = int(trigger_tick if trigger_tick is not None else default_trigger)
    outage_duration = int(
        outage_duration_ticks
        if outage_duration_ticks is not None
        # Extreme recovery must leave room for a deterministic retry after a
        # native tool failure.  A six-tick outage can expire before the
        # reference policy gets a successful repair under the configured
        # failure/delay contract, making the advertised two-stage task
        # impossible rather than difficult.
        else (10 if level == "extreme" else 4)
    )
    if outage_trigger <= 0 or outage_trigger + outage_duration > base.horizon_ticks:
        raise ValueError("dynamic job-shop outage must leave a response window")
    target_job = f"j{max(0, jobs - 1)}"
    events = [
        Perturbation(
            kind="machine_breakdown",
            trigger_tick=outage_trigger,
            duration_ticks=outage_duration,
            hidden=False,
            target={
                "machine_id": selected_machine,
                "source": "locked_jsplib_machine_set",
            },
            intensity=1.0,
            notes=(
                "Procedural deterministic maintenance outage over a machine "
                "identity consumed from the locked JSPLIB instance."
            ),
        )
    ]
    if level in {"high", "extreme"}:
        events.append(
            Perturbation(
                kind="demand_surge",
                trigger_tick=min(base.horizon_ticks - 2, outage_trigger + 2),
                duration_ticks=max(1, min(4, base.horizon_ticks - outage_trigger - 2)),
                hidden=True,
                target={
                    "job_id": target_job,
                    "source": "locked_jsplib_job_set",
                },
                intensity=2.0 if level == "extreme" else 1.5,
                notes=(
                    "Procedural urgent-job priority overlay over a locked "
                    "JSPLIB job identity; not a source-observed arrival."
                ),
            )
        )
    if level == "extreme":
        second_machine = int(job_shop["machine_ids"][-1])
        events.append(
            Perturbation(
                kind="machine_breakdown",
                trigger_tick=min(base.horizon_ticks - 2, outage_trigger + 5),
                # Leave a full retry window after the first repair result may
                # arrive late under the native failure/delay contract.
                duration_ticks=max(1, min(20, base.horizon_ticks - outage_trigger - 5)),
                hidden=True,
                target={
                    "machine_id": second_machine,
                    "source": "locked_jsplib_machine_set",
                },
                intensity=1.0,
                notes=(
                    "Second deterministic machine outage for extreme recovery "
                    "staging; target remains source-locked."
                ),
            )
        )
        events.append(
            Perturbation(
                kind="urgent_order",
                trigger_tick=min(base.horizon_ticks - 2, outage_trigger + 3),
                duration_ticks=max(1, min(4, base.horizon_ticks - outage_trigger - 3)),
                hidden=True,
                target={
                    "job_id": target_job,
                    "source": "locked_jsplib_job_set",
                },
                intensity=3.0,
                notes=(
                    "Deterministic urgent-order priority overlay over a locked "
                    "JSPLIB job identity for extreme recovery staging."
                ),
            )
        )
    base.perturbations = events
    base.backend_config["dynamic_job_shop"] = {
        "enabled": True,
        "event_source": "jsplib_machine_set_procedural_overlay_v1",
        "max_dispatch_batch_size": 4 if level == "extreme" else 2,
        "recovery_clearance_ticks": 1,
        "source_observed_events": False,
    }
    base.backend_config["task_contract"] = {
        "contract": "logistics.job_shop.dynamic_recovery.v1",
        "event_response_window": {
            "first_tick": outage_trigger + 1,
            "last_tick": base.horizon_ticks - 1,
        },
        "native_controls": [
            "dispatch_ready_operations",
            "dispatch_job_operation",
            "repair_machine",
        ],
    }
    # High/Extreme depth is proved from replayed native tool milestones, not
    # from the number of declared perturbations.  The first dispatch is
    # intentionally before the outage; recovery then requires a later native
    # repair (and, for Extreme, a second repair after the overlapping outage).
    # These windows are expressed in runner decision ticks (the backend event
    # trigger is observed one tick later).
    second_trigger = min(
        base.horizon_ticks - 2,
        outage_trigger + 5,
    )
    milestones: list[dict[str, Any]] = [
        {
            "tool": "dispatch_ready_operations",
            "not_after_tick": outage_trigger,
        },
        {
            "tool": "repair_machine",
            "not_before_tick": outage_trigger + 1,
            "not_after_tick": base.horizon_ticks - 1,
        },
    ]
    if level == "extreme":
        milestones.append(
            {
                "tool": "repair_machine",
                "not_before_tick": second_trigger + 1,
                "not_after_tick": base.horizon_ticks - 1,
            }
        )
    base.backend_config["task_requirements"] = {
        "min_distinct_control_ticks": len(milestones),
        "min_distinct_physical_tools": 2,
        "ordered_tool_milestones": milestones,
    }
    base.backend_config["source_event_contract"] = {
        "procedural_overlay": True,
        "source_identity_fields": ["machine_id", "job_id"],
        "source_observed": False,
        "runtime_effect_required": True,
    }
    base.backend_config["dimension_applicability"]["counterfactual_prevention"] = {
        "applicable": True,
        "reason": (
            "deterministic_no_action_replay_over_the_same_dynamic_recovery_events"
        ),
    }
    base.provenance.notes += (
        " Dynamic recovery staging uses only deterministic procedural overlays "
        "whose target IDs are validated against the consumed JSPLIB instance; "
        "the overlays are not provenance evidence."
    )
    return base


def job_shop_dimension_applicability(
    seed: LogisticsScenarioSeed,
) -> dict[str, dict[str, Any]]:
    """Dimension applicability for candidate JSPLIB rows."""
    reference = seed.backend_config.get("reference") or {}
    return _dimension_applicability_for_reference(reference)


def job_shop_complexity_metrics(seed: LogisticsScenarioSeed) -> dict[str, Any]:
    """Release-ledger complexity metrics for Job-Shop scheduling."""
    cfg = seed.backend_config.get("job_shop") or {}
    jobs = int(cfg.get("jobs", 0) or 0)
    machines = int(cfg.get("machines", 0) or 0)
    operations = int(cfg.get("operations", 0) or 0)
    total_processing = float(cfg.get("total_processing_time", 0.0) or 0.0)
    denom = max(1, machines * jobs)
    base = seed.complexity_metrics()
    base.update(
        {
            "n_jobs": jobs,
            "n_machines": machines,
            "n_operations": operations,
            "decision_depth": operations,
            "machine_conflict_density": round(operations / max(1, machines), 4),
            "processing_time_density": round(total_processing / denom, 4),
            "reference_type": (seed.backend_config.get("reference") or {}).get("type"),
        }
    )
    return base


def _metadata_for_instance(instance: str, root: Path) -> dict[str, Any]:
    for item in load_jsplib_metadata(root):
        if item.get("name") == instance:
            return item
    raise KeyError(f"JSPLIB metadata for instance {instance!r} not found")


def _reference(metadata: dict[str, Any]) -> dict[str, Any]:
    optimum = metadata.get("optimum")
    bounds = metadata.get("bounds")
    if isinstance(optimum, int) and optimum > 0:
        return {"type": "known_optimum", "makespan": optimum}
    if isinstance(bounds, dict):
        lower = bounds.get("lower")
        upper = bounds.get("upper")
        if isinstance(lower, int) and isinstance(upper, int) and lower > 0:
            return {
                "type": "best_known_bounds",
                "lower_bound": lower,
                "upper_bound": upper,
            }
    return {
        "type": "native_heuristic_policy",
        "policy": "earliest_completion_ready_operation_v1",
        "formal_optimality_applicable": False,
        "headroom_contract": "executable_policy_vs_no_action_replay",
    }


def _dimension_applicability_for_reference(
    reference: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    optimality_reason = (
        "known_optimum_or_bound_makespan_reference"
        if reference.get("type") in {"known_optimum", "best_known_bounds"}
        else "missing_makespan_reference"
    )
    optimality_applicable = reference.get("type") in {
        "known_optimum",
        "best_known_bounds",
    }
    return {
        "optimality_gap": {
            "applicable": optimality_applicable,
            "reason": optimality_reason,
        },
        "safety_violation": {
            "applicable": True,
            "reason": "precedence_machine_capacity_and_unscheduled_operation_keys",
        },
        "weighted_equity_score": {
            "applicable": False,
            "reason": "classic_jsplib_has_no_stakeholder_priority_classes",
        },
        "ethical_quality": {
            "applicable": False,
            "reason": "classic_jsplib_has_no_ethical_dilemma_payload",
        },
        "stakeholder_management": {
            "applicable": False,
            "reason": "classic_jsplib_has_no_stakeholder_trust_model",
        },
        "counterfactual_prevention": {
            "applicable": False,
            "reason": "static_job_shop_has_no_exogenous_loss_prevention_event",
        },
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
