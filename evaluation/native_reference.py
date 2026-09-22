"""Verify declared hashes of local replay files and derive fixed-policy native references."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evaluation.native_objectives import extract_native_objective
from evaluation.native_quality import REFERENCE_POLICIES, VERSION, calibrate_row


def read_verified(root: Path, relative: str, digest: str | None = None):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("reference artifact escapes declared root")
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if digest is not None and actual != digest:
        raise ValueError(f"reference artifact hash mismatch: {relative}")
    return raw, actual


def load_reference_report(root: Path, report_path: str, *, policies=None):
    selected = tuple(REFERENCE_POLICIES if policies is None else policies)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or set(selected) - set(REFERENCE_POLICIES)
    ):
        raise ValueError("unknown or duplicate reference policies")
    raw, report_hash = read_verified(root, report_path)
    report = json.loads(raw)
    manifest_raw, manifest_hash = read_verified(root, report["manifest"])
    manifest = json.loads(manifest_raw)
    start = manifest["implementation_identity"]["evaluation_runtime_sha256"]
    end = report["end_implementation_identity"]["evaluation_runtime_sha256"]
    if report.get("identity_unchanged") is not True or start != end:
        raise ValueError("reference execution changed runtime")
    if (
        manifest.get("policies") != list(REFERENCE_POLICIES)
        if policies is None
        else not set(selected).issubset(manifest.get("policies") or [])
    ) or manifest.get("repetitions") != 2:
        raise ValueError("reference policy protocol mismatch")
    specs = {(r["scenario_signature"], r["seed"]): r for r in manifest["rows"]}
    if len(specs) != len(manifest["rows"]):
        raise ValueError("duplicate reference suite rows")
    contracts = []
    seen = set()
    for row in report["results"]:
        key = (row["scenario_signature"], row["seed"])
        if key in seen or key not in specs:
            raise ValueError("duplicate or unexpected calibration row")
        seen.add(key)
        spec = specs[key]
        by_policy = {p: {} for p in selected}
        provenance = []
        paths = set()
        for item in row["episodes"]:
            policy, repetition = item["policy"], item["repetition"]
            if policies is not None and policy not in selected:
                continue
            if (
                policy not in by_policy
                or type(repetition) is not int
                or repetition not in (0, 1)
                or repetition in by_policy[policy]
            ):
                raise ValueError("invalid reference policy/repetition")
            if item["status"] != "ok":
                by_policy[policy][repetition] = {
                    "applicable": False,
                    "reason": item.get("error", "reference_failed"),
                }
                continue
            canonical_path = (root / item["episode_path"]).resolve()
            if canonical_path in paths:
                raise ValueError("reference repetitions reuse an episode artifact")
            paths.add(canonical_path)
            episode_raw, digest = read_verified(
                root, item["episode_path"], item["episode_sha256"]
            )
            if not item.get("artifacts"):
                raise ValueError("missing reference artifact inventory")
            for artifact in item["artifacts"]:
                read_verified(root, artifact["path"], artifact["sha256"])
            episode = json.loads(episode_raw)
            if episode.get("agent_name") != policy:
                raise ValueError("reference policy label differs from executed agent")
            if (episode.get("scenario_signature"), episode.get("seed")) != key:
                raise ValueError("reference episode task identity mismatch")
            for name in ("domain", "backend_kind"):
                if episode.get(name) not in (None, spec[name]):
                    raise ValueError("reference backend identity mismatch")
                episode[name] = spec[name]
            episode.setdefault("status", "ok")
            by_policy[policy][repetition] = extract_native_objective(episode)
            if (
                policies is not None
                and spec["backend_kind"] == "dynasched_flexible_job_shop"
            ):
                traces = [
                    a
                    for a in item["artifacts"]
                    if a["path"].endswith(".trajectory.jsonl")
                ]
                complete = None
                operations_verified = False
                if len(traces) == 1:
                    trace_raw, _ = read_verified(
                        root, traces[0]["path"], traces[0]["sha256"]
                    )
                    terminal = json.loads(trace_raw.splitlines()[-1])["observation"]
                    total, arrived = (
                        terminal.get("jobs_total"),
                        terminal.get("jobs_arrived"),
                    )
                    if (
                        type(total) is int
                        and total > 0
                        and type(arrived) is int
                        and 0 <= arrived <= total
                    ):
                        complete = total == arrived
                    names = (
                        "operations_total",
                        "operations_cancelled",
                        "operations_completed",
                        "operations_scheduled",
                    )
                    evidence = (episode.get("task_completion") or {}).get(
                        "evidence"
                    ) or {}
                    if all(
                        type(terminal.get(name)) is int
                        and terminal[name] >= 0
                        and terminal[name] == evidence.get(name)
                        for name in names
                    ):
                        required = (
                            terminal["operations_total"]
                            - terminal["operations_cancelled"]
                        )
                        operations_verified = (
                            required > 0
                            and required == evidence.get("operations_required")
                            and 0
                            <= terminal["operations_completed"]
                            <= terminal["operations_scheduled"]
                            <= required
                        )
                measurement = by_policy[policy][repetition]
                if complete is None:
                    measurement.update(
                        applicable=False, reason="reference_source_obligations_unproven"
                    )
                elif not complete:
                    measurement["feasible"] = False
                if complete is not None and not operations_verified:
                    measurement.update(
                        applicable=False,
                        reason="reference_terminal_operations_unproven",
                    )
            provenance.append(
                {
                    "policy": policy,
                    "repetition": repetition,
                    "episode_sha256": digest,
                    "episode_path": item["episode_path"],
                }
            )
        references = {
            p: [items[i] for i in (0, 1) if i in items]
            for p, items in by_policy.items()
        }
        contract = (
            calibrate_row(references)
            if policies is None
            else {
                "schema_version": "verified_policy_measurements.v1",
                "status": "policy_measurements_verified",
                "policy_determinism": {
                    p: (row.get("determinism", {}).get(p) or {}).get("deterministic")
                    for p in selected
                },
            }
        )
        failures = [
            {
                "policy": item["policy"],
                "repetition": item["repetition"],
                "reason": item.get("error", "reference_failed"),
            }
            for item in row["episodes"]
            if item["status"] != "ok" and item["policy"] in selected
        ]
        if contract["status"] == "ready" and not all(
            (row.get("determinism", {}).get(p) or {}).get("deterministic") is True
            for p in REFERENCE_POLICIES
        ):
            contract = {
                "schema_version": VERSION,
                "status": "unavailable",
                "reason": "reference_terminal_or_outcome_instability",
                "reference_is_optimum": False,
            }
        if failures:
            contract["reference_failures"] = failures
        contract.update(
            {
                "scenario_signature": key[0],
                "seed": key[1],
                "runtime_identity": start,
                "domain": spec["domain"],
                "backend_kind": spec["backend_kind"],
                "reference_report_sha256": report_hash,
                "reference_manifest_sha256": manifest_hash,
                "reference_artifacts": provenance,
                "policy_measurements": references,
                "qualification": "local_hashed_replay_evidence_not_provider_run_certification",
            }
        )
        if policies is not None:
            contract["source_spec"] = spec
        encoded = json.dumps(
            contract, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        contract["contract_sha256"] = hashlib.sha256(encoded).hexdigest()
        contracts.append(contract)
    if seen != set(specs):
        raise ValueError("reference report incomplete relative to manifest")
    return {
        "schema_version": VERSION,
        "contracts": contracts,
        "report_sha256": report_hash,
    }
