#!/usr/bin/env python3
"""Run resumable provider batches in locked, rate-limited background lanes.

The batch runner owns episode evidence, quotas and resume eligibility. This
supervisor only schedules bounded invocations, preserves their logs, and waits
between infrastructure retries. It never changes a model, suite or score.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import UTC, datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from uuid import UUID


def read_json(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return payload


def file_hash(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w") as handle:
        os.chmod(temp, 0o600)
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def event(path: Path, kind: str, **fields) -> None:
    with path.open("a") as handle:
        os.chmod(path, 0o600)
        handle.write(json.dumps({"time": datetime.now(UTC).isoformat(),
                                 "event": kind, **fields}, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@contextlib.contextmanager
def lane_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield handle
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def is_locked(path: Path) -> bool:
    try:
        with lane_lock(path):
            return False
    except BlockingIOError:
        return True


def load_credentials(names: list[str], rc: Path, environ=None) -> dict[str, str]:
    environ = os.environ if environ is None else environ
    lines = rc.read_text().splitlines() if rc.exists() else []
    result = {}
    for name in names:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError("invalid credential environment name")
        if environ.get(name):
            result[name] = environ[name]
            continue
        pattern = re.compile(rf"^\s*(?:export\s+)?{re.escape(name)}\s*=(.*)$")
        for line in lines:
            match = pattern.match(line)
            if not match:
                continue
            raw = match.group(1)
            if "$" in raw or "`" in raw:
                raise ValueError(f"credential {name} requires a literal assignment")
            values = shlex.split(raw, comments=True)
            if len(values) == 1:
                result[name] = values[0]
        if not result.get(name):
            raise ValueError(f"missing literal credential: {name}")
    return result


def build_command(root: Path, out: Path, job: dict, *, finalize=False) -> list[str]:
    setting = job.get("setting", "logical_persistent")
    if setting not in {"logical_persistent", "realtime_persistent"}:
        raise ValueError("campaign setting must be logical_persistent or realtime_persistent")
    effort = job.get("reasoning_effort")
    if effort not in {None, "none", "minimal", "low", "medium", "high"}:
        raise ValueError("campaign reasoning must not exceed high")
    if job["suite"] not in {"full", "lite"}:
        raise ValueError("campaign suite must be full or lite")
    output_tokens = int(job.get("output_tokens", 32000))
    if not 0 < output_tokens <= int(job["max_output"]):
        raise ValueError("output reserve exceeds route capability")
    manifest_rel = job.get("formal_manifest", "benchmark/manifest.json")
    manifest = root / manifest_rel
    if setting == "realtime_persistent":
        if not job.get("suite_path"):
            raise ValueError("realtime campaign requires an explicit suite_path")
        cmd = [str(root / ".venv/bin/python"), str(root / "scripts/batch_realtime_llm_eval.py"),
               "--output-root", str(out), "--model", job["model"],
               "--formal-manifest", str(manifest), "--suite", str(root / job["suite_path"]),
               "--suite-kind", "core" if job["suite"] == "full" else "lite",
               "--api-key-env", job["api_key_env"], "--base-url-env", "OPERATE_CAMPAIGN_BASE_URL",
               "--api-mode", job.get("api_mode", "chat_completions"),
               "--model-context-window-tokens", str(job["context_window"]),
               "--model-max-output-tokens", str(job["max_output"]),
               "--max-tokens", str(output_tokens), "--protocol-repair-max-tokens", "8192",
               "--persistent-history-max-messages", "64",
               "--persistent-context-max-chars", str(job.get("context_chars", 128000)),
               "--persistent-memory-max-items", "128",
               "--provider-timeout-s", str(job.get("timeout_s", 300)),
               "--max-workers", str(job.get("workers", 1)),
               "--max-jobs", str(job.get("chunk_jobs", 1)),
               "--pass-k", "1", "--resume", "--resume-policy", "retry-infrastructure",
               "--finalize-only" if finalize else "--no-finalize"]
        for field, flag in [("reasoning_effort", "--reasoning-effort"),
                            ("reasoning_effort_format", "--reasoning-effort-format"),
                            ("thinking_type", "--thinking-type")]:
            if job.get(field) is not None:
                cmd += [flag, str(job[field])]
        for field, flag in [("rpm", "--provider-rpm-limit"), ("rpd", "--provider-rpd-limit")]:
            if job.get(field, 0) > 0:
                cmd += [flag, str(job[field])]
        if job.get("rpm", 0) or job.get("rpd", 0):
            cmd += ["--provider-rate-limit-scope", job["quota_scope"]]
        return cmd
    entry = "scripts/batch_llm_eval.py" if job["suite"] == "full" else "run_lite.py"
    cmd = [str(root / ".venv/bin/python"), str(root / entry),
           "--output-dir", str(out), "--models", job["model"],
           "--api-key-env", job["api_key_env"],
           "--base-url-env", "OPERATE_CAMPAIGN_BASE_URL",
           "--api-mode", job.get("api_mode", "chat_completions"),
           "--stream-chat-completions", "--temperature", "0",
           "--max-tokens", str(output_tokens),
           "--protocol-repair-max-tokens", "8192",
           "--model-context-window-tokens", str(job["context_window"]),
           "--model-max-output-tokens", str(job["max_output"]),
           "--persistent-history-max-messages", "64",
           "--persistent-context-max-chars", str(job.get("context_chars", 128000)),
           "--persistent-memory-max-items", "128",
           "--provider-timeout-s", str(job.get("timeout_s", 300)),
           "--provider-failure-policy", "abort",
           "--max-consecutive-provider-failures", "1",
           "--resume-policy", "retry-infrastructure", "--resume",
           "--job-order", "shortest-first",
           "--seed-mode", "scenario", "--pass-k", "1", "--prompt-mode", "strict",
           "--interaction-mode", "logical_persistent", "--scheduler-mode", "global",
           "--max-workers", str(job.get("workers", 1)),
           "--max-jobs", str(job.get("chunk_jobs", 1)), "--save-trajectories",
           "--finalize" if finalize else "--no-finalize"]
    if job["suite"] == "full":
        cmd += ["--formal-run", "--formal-manifest",
                str(manifest)]
    elif job.get("suite_path"):
        cmd += ["--lite-suite", str(root / job["suite_path"])]
    if effort is not None:
        cmd += ["--reasoning-effort", effort]
    for field, flag in [("reasoning_effort_format", "--reasoning-effort-format"),
                        ("thinking_type", "--thinking-type")]:
        if job.get(field) is not None:
            cmd += [flag, str(job[field])]
    if job.get("rpm", 0) > 0:
        cmd += ["--provider-rpm-limit", str(job["rpm"])]
    if job.get("rpd", 0) > 0:
        cmd += ["--provider-rpd-limit", str(job["rpd"])]
    if job.get("rpm", 0) or job.get("rpd", 0):
        cmd += ["--provider-rate-limit-scope", job["quota_scope"]]
    return cmd


def choose_job(jobs: list[dict], state: dict, now: float) -> dict | None:
    if state.get("not_before", 0) > now:
        return None
    ready = []
    for index, job in enumerate(jobs):
        row = state["jobs"].get(job["id"], {})
        if not job.get("enabled", True) or row.get("status") in {
            "needs_attention", "attempts_closed", "reports_ready", "reports_need_attention"
        }:
            continue
        if max(row.get("not_before", 0), state["pools"].get(job["quota_scope"], 0)) > now:
            continue
        ready.append((row.get("last_started", 0), index, job))
    return min(ready, default=(None, None, None), key=lambda item: item[:2])[2]


def reset_epoch(value) -> float | None:
    if not value:
        return None
    raw = str(value)
    if raw.endswith(" UTC+8"):
        raw = raw.removesuffix(" UTC+8").replace(" ", "T") + "+08:00"
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=UTC).timestamp() if parsed.tzinfo is None else parsed.timestamp()
    except ValueError:
        return None


def attempt_key(result: dict) -> str:
    return json.dumps([result.get(k) for k in ["scenario_slug", "model", "seed", "pass_id"]])


def batch_directory(root: Path) -> Path:
    """Formal batches resolve one treatment subdirectory below the lane output."""
    candidates = list(root.glob("invocation_summary.json")) + list(root.glob("treatment-*/invocation_summary.json"))
    if len(candidates) > 1:
        raise ValueError("multiple batch identities in one campaign job output")
    return candidates[0].parent if candidates else root


def batch_locked(root: Path) -> bool:
    locks = [root/".run.lock", *root.glob("treatment-*/.run.lock")]
    return any(is_locked(p) for p in locks if p.exists())


def apply_invocation(state: dict, job: dict, summary: dict, *, now: float,
                     cooldown: int, max_attempts: int, success_cooldown: int | None = None) -> None:
    if summary.get("schema_version") != "batch_invocation_v1" or summary.get("status") != "completed":
        raise ValueError("batch did not write a completed invocation summary")
    row = state["jobs"].setdefault(job["id"], {})
    failures = row.setdefault("attempt_failures", {})
    row.update(status="ready", not_before=now+(cooldown if success_cooldown is None else success_cooldown),
               terminal=summary["resume_terminal"], total=summary["total_scope_jobs"],
               pending=summary["pending_after"], terminal_errors=summary["terminal_errors"])
    for result in summary.get("dispatched_results", []):
        if result.get("error_http_status") in {401, 403, 404}:
            row.update(status="needs_attention", reason="provider_access_error")
        if not result.get("retryable_infrastructure"):
            continue
        key = attempt_key(result)
        if result.get("execution_started") is not False:
            failures[key] = failures.get(key, 0) + 1
            if failures[key] >= max_attempts:
                row.update(status="needs_attention", reason="episode_retry_budget_exhausted")
        wait = min(3600, cooldown * 2 ** max(0, failures.get(key, 1)-1))
        row["not_before"] = max(row["not_before"], now+wait)
        if result.get("quota_parked"):
            reset = reset_epoch(result.get("quota_reset_at"))
            # Unknown quota resets get a conservative cooldown, never a spin loop.
            state["pools"][job["quota_scope"]] = max(
                state["pools"].get(job["quota_scope"], 0),
                (reset+30) if reset and reset > now else now+3600,
            )
    if summary.get("scope_attempts_closed") and row["status"] != "needs_attention":
        row["status"] = "attempts_closed"


def interrupted_attempts(out: Path, active: dict, summary: dict) -> tuple[list[dict], bool]:
    """Read positive worker-start proof, then exclude already settled attempts."""
    from scripts.batch_llm_eval import _retryable_infrastructure_row

    realtime = summary.get("worker_start_contract") == "realtime_worker_execution_start_v1"
    if summary.get("worker_start_contract") not in {"worker_execution_start_v1", "realtime_worker_execution_start_v1"}:
        raise ValueError("interrupted invocation has no worker-start contract")
    expected = {attempt_key(row): row for row in summary["dispatched_results"]}
    fields = (
        "scenario_signature", "agent_treatment_sha256", "implementation_tree_sha256",
        "run_semantics_fingerprint", "suite_manifest_sha256", "suite_eligibility_sha256",
    )
    if realtime:
        fields = ("scenario_signature", "batch_treatment_sha256", "implementation_tree_sha256")
    starts: dict[str, dict] = {}
    with (out / "worker_starts.jsonl").open() as handle:
        for line in handle:
            marker = json.loads(line)
            if not isinstance(marker, dict):
                raise ValueError("invalid worker-start record")
            if marker.get("invocation_started_at_utc") != summary["started_at_utc"]:
                continue
            planned = expected.get(attempt_key(marker))
            started = reset_epoch(marker.get("worker_started_at_utc"))
            if (
                marker.get("schema_version") != summary["worker_start_contract"]
                or planned is None or started is None
                or started < active["started_at"] - 1 or started > time.time() + 1
                or any(not marker.get(field) or marker[field] != planned.get(field) for field in fields)
            ):
                raise ValueError("worker-start identity mismatch")
            attempt_id = UUID(str(marker.get("execution_attempt_id") or "")).hex
            if attempt_id in starts and starts[attempt_id] != marker:
                raise ValueError("conflicting worker-start identity")
            starts[attempt_id] = marker
    terminals = {}
    journal = out / "episodes.jsonl"
    if journal.exists():
        with journal.open() as handle:
            for line in handle:
                result = json.loads(line)
                if not isinstance(result, dict):
                    raise ValueError("invalid episode journal record")
                raw_id = result.get("execution_attempt_id")
                if not raw_id:
                    continue
                attempt_id = UUID(str(raw_id)).hex
                terminal_statuses = {"ok", "ineligible", "infrastructure_error", "provider_quota_exhausted"} if realtime else {"ok", "error"}
                if attempt_id not in starts or result.get("status") not in terminal_statuses:
                    continue
                marker = starts[attempt_id]
                if (
                    result.get("invocation_started_at_utc") != summary["started_at_utc"]
                    or (result.get("job_key") != marker.get("job_key") if realtime else attempt_key(result) != attempt_key(marker))
                    or any(result.get(field) != marker[field] for field in fields)
                ):
                    raise ValueError("terminal attempt identity mismatch")
                terminals[attempt_id] = result
    if realtime:
        from scripts.batch_realtime_llm_eval import _realtime_retryable
        retryable = _realtime_retryable
    else:
        retryable = _retryable_infrastructure_row
    charge = [
        {**marker, "execution_attempt_id": attempt_id}
        for attempt_id, marker in starts.items()
        if attempt_id not in terminals or retryable(terminals[attempt_id])
    ]
    access_error = any(row.get("error_http_status") in {401, 403, 404} for row in terminals.values())
    return charge, access_error


def verify_bindings(config: dict, root: Path) -> None:
    from core.implementation_identity import implementation_identity
    actual = implementation_identity(root)["implementation_tree_sha256"]
    if actual != config["implementation_tree_sha256"]:
        raise ValueError("frozen evaluation runtime changed")
    for rel, wanted in config["file_sha256"].items():
        path = root / rel
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != wanted:
            raise ValueError(f"frozen input changed: {rel}")
    for raw, wanted in config.get("controller_files", {}).items():
        if file_hash(Path(raw)) != wanted:
            raise ValueError(f"campaign controller changed: {Path(raw).name}")


def repair_invocation_summary(config: dict, out: Path, summary: dict, log: Path) -> dict:
    """Recover missing transport metadata without modifying execution artifacts."""
    script = config.get("metadata_repair_script")
    if not script or not summary.get("terminal_errors"):
        return summary
    script = Path(script)
    env = dict(os.environ, PYTHONPATH=str(script.parent.parent))
    completed = subprocess.run(
        [sys.executable, str(script), "--output-dir", str(out), "--apply"],
        cwd=script.parent.parent, env=env, capture_output=True, text=True, check=False,
        timeout=120,
    )
    log.with_suffix(".metadata.json").write_text(completed.stdout)
    log.with_suffix(".metadata.log").write_text(completed.stderr)
    if completed.returncode:
        raise ValueError("provider metadata repair failed; inspect metadata log")
    proposed = json.loads(completed.stdout).get("proposed_invocation_summary")
    if not isinstance(proposed, dict):
        raise ValueError("provider metadata repair did not return a summary")
    fixed = ["schema_version", "status", "started_at_utc", "total_scope_jobs", "resume_policy"]
    if any(proposed.get(k) != summary.get(k) for k in fixed) or (
        [attempt_key(r) for r in proposed.get("dispatched_results", [])] !=
        [attempt_key(r) for r in summary.get("dispatched_results", [])]
    ):
        raise ValueError("provider metadata repair changed invocation scope")
    return proposed


def worker(config_path: Path, lane: str, *, once=False) -> int:
    config = read_json(config_path)
    root = Path(config["runtime_root"]).resolve()
    lane_config = config["lanes"][lane]
    jobs = lane_config["jobs"]
    directory = config_path.parent / "queue" / lane
    directory.mkdir(parents=True, exist_ok=True)
    state_path, events = directory / "state.json", directory / "events.jsonl"
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    with lane_lock(directory / "worker.lock"):
        state = read_json(state_path) if state_path.exists() else {
            "schema_version": "operate-campaign-lane-v1", "config_sha256": digest,
            "jobs": {}, "pools": {},
        }
        if state.get("config_sha256") != digest:
            raise ValueError("campaign config changed; create a new campaign namespace")
        state.update(pid=os.getpid(), worker_status="running")
        atomic_json(state_path, state)
        event(events, "worker_started", pid=os.getpid())
        while state.get("active") or not (directory / "STOP").exists():
            verify_bindings(config, root)
            active = state.get("active")
            if active:
                job = next(j for j in jobs if j["id"] == active["job_id"])
                out = Path(active["output_dir"])
                if batch_locked(out):
                    time.sleep(10)
                    continue
                summary_path = batch_directory(out) / "invocation_summary.json"
                summary = read_json(summary_path) if summary_path.exists() else {}
                fresh = reset_epoch(summary.get("started_at_utc"))
                changed = file_hash(summary_path) != active.get("summary_before")
                if summary.get("status") == "completed" and changed and fresh and fresh >= active["started_at"]-1:
                    if active.get("phase") != "finalize":
                        try:
                            summary = repair_invocation_summary(config, batch_directory(out), summary, Path(active["log"]))
                        except (ValueError, OSError, subprocess.SubprocessError):
                            state["jobs"].setdefault(job["id"], {}).update(
                                status="needs_attention", reason="provider_metadata_repair_failed")
                            event(events, "metadata_repair_failed", job_id=job["id"])
                            state.pop("active", None)
                            atomic_json(state_path, state)
                            if once:
                                break
                            continue
                    if active.get("phase") == "finalize":
                        row = state["jobs"][job["id"]]
                        row["status"] = "reports_ready" if active.get("exit_code") == 0 else "reports_need_attention"
                        row["needs_finalize"] = False
                    else:
                        apply_invocation(state, job, summary, now=time.time(),
                                         cooldown=lane_config["cooldown_s"],
                                         success_cooldown=lane_config.get("success_cooldown_s"),
                                         max_attempts=config.get("max_episode_attempts", 3))
                        if state["jobs"][job["id"]]["status"] == "attempts_closed":
                            state["jobs"][job["id"]]["needs_finalize"] = True
                        state["not_before"] = time.time()+lane_config.get("between_invocations_s", 0)
                    archived = Path(active["log"]).with_suffix(".summary.json")
                    atomic_json(archived, summary)
                    event(events, "invocation_closed", job_id=job["id"],
                          terminal=summary["resume_terminal"], pending=summary["pending_after"],
                          terminal_errors=summary["terminal_errors"], summary=archived.name,
                          exit_code=active.get("exit_code"), phase=active.get("phase"))
                elif (
                    summary.get("schema_version") == "batch_invocation_v1"
                    and summary.get("status") == "needs_attention"
                    and summary.get("reason") == "resume_artifact_integrity_failed"
                    and changed and fresh and fresh >= active["started_at"]-1
                ):
                    archived = Path(active["log"]).with_suffix(".summary.json")
                    atomic_json(archived, summary)
                    state["jobs"].setdefault(job["id"], {}).update(
                        status="needs_attention", reason="resume_artifact_integrity_failed",
                        integrity_summary=str(archived),
                    )
                    event(events, "resume_integrity_blocked", job_id=job["id"],
                          summary=archived.name,
                          n_failures=len(summary.get("integrity_failures") or []))
                else:
                    row = state["jobs"].setdefault(job["id"], {})
                    row["interrupted_invocations"] = row.get("interrupted_invocations", 0)+1
                    recoverable = changed and summary.get("status") == "running" and fresh and fresh >= active["started_at"]-1
                    if active.get("phase") == "finalize":
                        row.update(status="reports_need_attention", needs_finalize=False,
                                   reason="finalizer_interrupted")
                    elif recoverable and summary.get("dispatched_results"):
                        try:
                            started, access_error = interrupted_attempts(batch_directory(out), active, summary)
                        except (ValueError, OSError):
                            row.update(status="needs_attention", reason="interrupted_start_evidence_invalid")
                        else:
                            failures = row.setdefault("attempt_failures", {})
                            charged = set(row.get("charged_interrupted_attempt_ids") or [])
                            for result in started:
                                if result["execution_attempt_id"] not in charged:
                                    key = attempt_key(result)
                                    failures[key] = failures.get(key, 0)+1
                                    charged.add(result["execution_attempt_id"])
                            row["charged_interrupted_attempt_ids"] = sorted(charged)
                            exhausted = any(n >= config.get("max_episode_attempts", 3) for n in failures.values())
                            reason = "provider_access_error" if access_error else (
                                "episode_retry_budget_exhausted" if exhausted else "interrupted_invocation"
                            )
                            row.update(status="needs_attention" if exhausted or access_error else "ready",
                                       reason=reason, not_before=time.time()+lane_config["cooldown_s"])
                    else:
                        row.update(status="needs_attention", reason="missing_or_interrupted_batch_summary")
                    event(events, "invocation_interrupted", job_id=job["id"],
                          exit_code=active.get("exit_code"), status=row["status"])
                state.pop("active", None)
                atomic_json(state_path, state)
                if once:
                    break
                continue
            job = next((j for j in jobs if
                        state["jobs"].get(j["id"], {}).get("needs_finalize") and
                        state["jobs"][j["id"]].get("status") == "attempts_closed" and
                        state["jobs"][j["id"]].get("not_before", 0) <= time.time()), None)
            finalize = job is not None
            job = job or choose_job(jobs, state, time.time())
            if job is None:
                live = [j for j in jobs if j.get("enabled", True) and (
                        state["jobs"].get(j["id"], {}).get("status") not in {
                            "needs_attention", "attempts_closed", "reports_ready", "reports_need_attention"}
                        or (state["jobs"].get(j["id"], {}).get("status") == "attempts_closed" and
                            state["jobs"].get(j["id"], {}).get("needs_finalize")))]
                if not live or once:
                    break
                time.sleep(10)
                continue
            out = config_path.parent / "results" / job["id"]
            out.mkdir(parents=True, exist_ok=True)
            if batch_locked(out):
                # An orphaned runner must finish before this lane starts more work.
                time.sleep(10)
                continue
            credentials = load_credentials([job["api_key_env"]], Path.home()/".zshrc")
            env = dict(os.environ, **credentials, **config.get("environment", {}))
            env.update(OPERATE_CAMPAIGN_BASE_URL=job["base_url"], PYTHONPATH=str(root))
            started = time.time()
            log = directory / f"{job['id']}-{int(started*1000)}.log"
            state["jobs"].setdefault(job["id"], {})["last_started"] = started
            state["active"] = {"job_id": job["id"], "started_at": started,
                               "output_dir": str(out), "log": str(log),
                               "summary_before": file_hash(batch_directory(out)/"invocation_summary.json"),
                               "phase": "finalize" if finalize else "evaluate"}
            atomic_json(state_path, state)
            event(events, "invocation_started", job_id=job["id"], log=log.name)
            with log.open("w") as handle:
                os.chmod(log, 0o600)
                completed = subprocess.run(build_command(root, out, job, finalize=finalize), cwd=root, env=env,
                                           stdout=handle, stderr=subprocess.STDOUT, check=False)
            state["active"]["exit_code"] = completed.returncode
            atomic_json(state_path, state)
        state["worker_status"] = "stopped" if (directory/"STOP").exists() else "idle"
        atomic_json(state_path, state)
        event(events, "worker_exited", status=state["worker_status"])
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--lane", default="tencent")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--start", action="store_true")
    mode.add_argument("--worker", action="store_true")
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--stop", action="store_true", help="Pause after the current invocation")
    args = p.parse_args()
    config_path = args.config.resolve()
    config = read_json(config_path)
    if args.lane not in config["lanes"]:
        raise ValueError("unknown campaign lane")
    directory = config_path.parent/"queue"/args.lane
    directory.mkdir(parents=True, exist_ok=True)
    if args.status:
        status = read_json(directory/"state.json") if (directory/"state.json").exists() else {}
        print(json.dumps({"running": is_locked(directory/"worker.lock"),
                          "stopped": (directory/"STOP").exists(), "state": status}, indent=2))
        return 0
    if args.stop:
        (directory/"STOP").touch()
        print("Pause requested after the current invocation.")
        return 0
    if args.start:
        with lane_lock(directory/"start.lock"):
            if is_locked(directory/"worker.lock"):
                print("Campaign lane already running.")
                return 0
            if (directory/"STOP").exists():
                print("Campaign lane is manually paused; remove STOP to resume.")
                return 0
            if (directory/"state.json").exists():
                state = read_json(directory/"state.json")
                unfinished = any(
                    j.get("enabled", True) and (
                        state.get("jobs", {}).get(j["id"], {}).get("needs_finalize") or
                        state.get("jobs", {}).get(j["id"], {}).get("status") not in {
                            "needs_attention", "attempts_closed", "reports_ready", "reports_need_attention"}
                    ) for j in config["lanes"][args.lane]["jobs"]
                )
                if not unfinished and not state.get("active"):
                    print("No runnable work remains; inspect terminal and attention states.")
                    return 0
            verify_bindings(config, Path(config["runtime_root"]))
            with (directory/"worker.log").open("a") as log:
                proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                         "--config", str(config_path), "--lane", args.lane, "--worker"],
                                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
            for _ in range(40):
                if is_locked(directory/"worker.lock"):
                    print(json.dumps({"started_pid": proc.pid, "lane": args.lane}))
                    return 0
                if proc.poll() is not None:
                    raise RuntimeError("campaign worker failed at startup; inspect worker.log")
                time.sleep(0.05)
            raise RuntimeError("worker startup not confirmed; inspect worker.log before retrying")
    return worker(config_path, args.lane, once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
