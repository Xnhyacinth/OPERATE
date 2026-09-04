"""
data.trajectory_logger — Per-episode trajectory recorder.

Forked from ``dispatch-benchmark/data/trajectory_logger.py`` with these
DT-Sched-Bench-specific changes:

- ``evidence_ids`` are a first-class field on every ``TrajectoryEntry``
  so downstream replay / audit can re-derive scores from the trajectory.
- No emergency-schema-specific bookkeeping (no patients/units/hospitals).
- ``scenario_signature`` is recorded in the episode header so audits can
  pair a trajectory file with the locked release scenario.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_MAX_EPISODE_ID_BYTES = 180


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return sorted(_json_safe(v) for v in value)
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return {k: _json_safe(getattr(value, k)) for k in value.__dataclass_fields__}
    return str(value)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Durably replace one sidecar without exposing a partial artifact."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".trajectory-",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        assert temporary is not None
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _jsonl_artifact_binding(path: Path, *, schema_version: str) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "schema_version": schema_version,
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "event_count": len(payload.splitlines()),
        "byte_count": len(payload),
    }


@dataclass
class TrajectoryEntry:
    tick: int
    observation: dict[str, Any]
    action: dict[str, Any] | None
    reward: float
    tool_results: list[Any] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)
    assistant_text: str | None = None
    timestamp_utc: str = field(
        # ``timezone.utc`` is available on the project's supported Python
        # 3.10+ range; ``datetime.UTC`` was only added in Python 3.11.
        default_factory=lambda: datetime.now(timezone.utc).isoformat()  # noqa: UP017
    )

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "tick": self.tick,
                "observation": self.observation,
                "action": self.action,
                "reward": self.reward,
                "tool_results": self.tool_results,
                "evidence_ids": self.evidence_ids,
                "info": self.info,
                "assistant_text": self.assistant_text,
                "timestamp_utc": self.timestamp_utc,
            }
        )


@dataclass
class EpisodeHeader:
    episode_id: str
    scenario_id: str
    scenario_signature: str
    domain: str
    family: str
    difficulty_mode: str
    difficulty_level: str
    backend_kind: str
    horizon_ticks: int
    tick_minutes: int | None
    agent_name: str
    agent_config: dict[str, Any] | None
    seed: int
    start_time_utc: str
    end_time_utc: str | None = None
    total_ticks: int = 0
    final_score: float | None = None
    version: str = "0.1.0"
    # v0.2.2 (P1-3): optional, agent-supplied auxiliary state captured at
    # episode start (e.g. Reflexion's lessons-file fingerprint). Recorded
    # in the trajectory header so leaderboard runs are reproducible: two
    # runs with identical seeds must also share `agent_extras` to be
    # considered comparable. Purely additive — pre-existing readers that
    # do not know this key are unaffected.
    agent_extras: dict[str, Any] | None = None
    # Protocol-2.2 additive clock fields. Historical headers omit these keys.
    tick_seconds: float | None = None
    clock_contract: dict[str, Any] | None = None
    semantic_ledger_sha256: str | None = None
    semantic_ledger_events: int | None = None
    provider_audit_sha256: str | None = None
    provider_audit_events: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "episode_id": self.episode_id,
            "scenario_id": self.scenario_id,
            "scenario_signature": self.scenario_signature,
            "domain": self.domain,
            "family": self.family,
            "difficulty_mode": self.difficulty_mode,
            "difficulty_level": self.difficulty_level,
            "backend_kind": self.backend_kind,
            "horizon_ticks": self.horizon_ticks,
            "tick_minutes": self.tick_minutes,
            "agent_name": self.agent_name,
            "agent_config": self.agent_config,
            "seed": self.seed,
            "start_time_utc": self.start_time_utc,
            "end_time_utc": self.end_time_utc,
            "total_ticks": self.total_ticks,
            "final_score": self.final_score,
            "version": self.version,
            "agent_extras": self.agent_extras,
        }
        if self.tick_seconds is not None:
            payload["tick_seconds"] = self.tick_seconds
        if self.clock_contract is not None:
            payload["clock_contract"] = self.clock_contract
        if self.semantic_ledger_sha256 is not None:
            payload["semantic_ledger_sha256"] = self.semantic_ledger_sha256
        if self.semantic_ledger_events is not None:
            payload["semantic_ledger_events"] = self.semantic_ledger_events
        if self.provider_audit_sha256 is not None:
            payload["provider_audit_sha256"] = self.provider_audit_sha256
        if self.provider_audit_events is not None:
            payload["provider_audit_events"] = self.provider_audit_events
        return _json_safe(payload)


class TrajectoryLogger:
    """Append-only trajectory recorder with optional disk flush."""

    def __init__(
        self,
        episode_id: str | None = None,
        output_dir: Path | str | None = None,
        buffer_size: int = 1,
    ):
        self.episode_id = self._sanitize(episode_id or uuid.uuid4().hex)
        self.output_dir = Path(output_dir) if output_dir else None
        self.buffer_size = buffer_size
        self.header: EpisodeHeader | None = None
        self.entries: list[TrajectoryEntry] = []
        self._total_entries_logged = 0
        self._cumulative_reward = 0.0
        self._action_counts: dict[str, int] = {}
        self._tool_call_failures = 0
        self._state_changing_calls = 0
        self._total_calls = 0
        self._claimed_output = False

    @staticmethod
    def _sanitize(raw: str) -> str:
        unsafe = '/\\:*?"<>|'
        for ch in unsafe:
            raw = raw.replace(ch, "_")
        while "__" in raw:
            raw = raw.replace("__", "_")
        sanitized = raw.strip("_")
        encoded = sanitized.encode("utf-8")
        if len(encoded) <= _MAX_EPISODE_ID_BYTES:
            return sanitized
        digest = hashlib.sha256(encoded).hexdigest()[:20]
        prefix_budget = _MAX_EPISODE_ID_BYTES - len(digest) - 1
        prefix = encoded[:prefix_budget].decode("utf-8", errors="ignore").rstrip("_")
        return f"{prefix}_{digest}"

    def set_header(self, header: EpisodeHeader) -> None:
        if header.episode_id != self.episode_id:
            header.episode_id = self.episode_id
        self.header = header

    def log_step(
        self,
        tick: int,
        observation: dict[str, Any],
        action: dict[str, Any] | None,
        reward: float,
        tool_results: list[Any] | None = None,
        evidence_ids: list[str] | None = None,
        info: dict[str, Any] | None = None,
        assistant_text: str | None = None,
    ) -> None:
        entry = TrajectoryEntry(
            tick=tick,
            observation=observation,
            action=action,
            reward=reward,
            tool_results=list(tool_results or []),
            evidence_ids=list(evidence_ids or []),
            info=info or {},
            assistant_text=assistant_text,
        )
        self.entries.append(entry)
        self._total_entries_logged += 1
        self._cumulative_reward += reward
        if action:
            dom = action.get("dominant_action") or action.get("action") or "?"
            self._action_counts[dom] = self._action_counts.get(dom, 0) + 1
            for sub in action.get("actions", []) or []:
                self._total_calls += 1
                if sub.get("ok") is False:
                    self._tool_call_failures += 1
        within_tick = ((info or {}).get("extra") or {}).get(
            "within_tick_investigation"
        )
        if isinstance(within_tick, dict):
            investigation_action = within_tick.get("investigation_action") or {}
            if isinstance(investigation_action, dict):
                self._total_calls += len(investigation_action.get("actions") or [])
        for r in tool_results or []:
            if isinstance(r, dict):
                if r.get("state_changing"):
                    self._state_changing_calls += 1
                if r.get("ok") is False:
                    self._tool_call_failures += 1
        if len(self.entries) >= self.buffer_size:
            self._flush()

    def _flush(self) -> None:
        if not self.output_dir:
            return
        self._claim_output()
        path = self.output_dir / f"{self.episode_id}.trajectory.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            for entry in self.entries:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.entries.clear()

    def _claim_output(self) -> None:
        """Atomically claim one deterministic episode id for this logger.

        A logger may flush repeatedly, but a second logger must never append a
        new run to an existing trajectory while overwriting its sidecars.
        """
        if self.output_dir is None or self._claimed_output:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        prefix = self.output_dir / self.episode_id
        companion_paths = [
            Path(f"{prefix}.header.json"),
            Path(f"{prefix}.summary.json"),
            Path(f"{prefix}.evidence.jsonl"),
            Path(f"{prefix}.semantic_ledger.jsonl"),
            Path(f"{prefix}.provider_audit.jsonl"),
        ]
        existing_companion = next(
            (path for path in companion_paths if path.exists()), None
        )
        if existing_companion is not None:
            raise FileExistsError(
                f"episode {self.episode_id!r} already has artifact "
                f"{existing_companion.name}"
            )
        trajectory_path = Path(f"{prefix}.trajectory.jsonl")
        try:
            with open(trajectory_path, "x", encoding="utf-8"):
                pass
        except FileExistsError as exc:
            raise FileExistsError(
                f"episode {self.episode_id!r} already owns "
                f"{trajectory_path.name}"
            ) from exc
        self._claimed_output = True

    def write_evidence(self, items: list[dict[str, Any]]) -> Path:
        """Persist the complete evidence ledger referenced by episode scores."""
        if self.output_dir is None:
            raise ValueError("output_dir is required to persist evidence")
        self._claim_output()
        path = self.output_dir / f"{self.episode_id}.evidence.jsonl"
        payload = "".join(
            json.dumps(_json_safe(item), ensure_ascii=False) + "\n"
            for item in items
        ).encode("utf-8")
        _atomic_write_bytes(path, payload)
        return path

    def write_semantic_ledger(
        self,
        items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Persist the canonical semantic session ledger atomically.

        The artifact contains only provider-visible roles/content and is bound
        to the episode header by SHA-256 and event count. Credentials remain
        excluded by the provider-neutral prompt compiler.
        """
        if self.output_dir is None:
            raise ValueError("output_dir is required to persist semantic ledger")
        self._claim_output()
        path = self.output_dir / f"{self.episode_id}.semantic_ledger.jsonl"
        lines = [
            json.dumps(
                _json_safe(item),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for item in items
        ]
        payload = (("\n".join(lines) + "\n") if lines else "").encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        _atomic_write_bytes(path, payload)
        if self.header is not None:
            self.header.semantic_ledger_sha256 = digest
            self.header.semantic_ledger_events = len(items)
        return {
            "path": str(path),
            "sha256": digest,
            "event_count": len(items),
            "schema_version": "semantic_session_ledger_v1",
        }

    def write_provider_audit(
        self,
        items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Persist credential-free provider request/response audit records."""
        if self.output_dir is None:
            raise ValueError("output_dir is required to persist provider audit")
        self._claim_output()
        path = self.output_dir / f"{self.episode_id}.provider_audit.jsonl"
        lines = [
            json.dumps(
                _json_safe(item),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for item in items
        ]
        payload = (("\n".join(lines) + "\n") if lines else "").encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        _atomic_write_bytes(path, payload)
        if self.header is not None:
            self.header.provider_audit_sha256 = digest
            self.header.provider_audit_events = len(items)
        return {
            "path": str(path),
            "sha256": digest,
            "event_count": len(items),
            "schema_version": "provider_interaction_audit_v1",
        }

    def finalize(
        self,
        final_score: float | None = None,
        *,
        trajectory_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Flush the episode and persist its audit summary.

        ``trajectory_summary`` is kept as a nested, JSON-safe sidecar in the
        ordinary summary file.  This makes autonomy/terminal decisions
        auditable from disk instead of only from the value returned by the
        runner, while preserving the logger's aggregate counters.
        """
        total = self._total_entries_logged
        if self.header:
            self.header.end_time_utc = datetime.now(timezone.utc).isoformat()  # noqa: UP017
            self.header.total_ticks = total
            self.header.final_score = final_score
        summary = {
            "episode_id": self.episode_id,
            "total_ticks": total,
            "cumulative_reward": self._cumulative_reward,
            "mean_reward_per_step": self._cumulative_reward / max(total, 1),
            "action_distribution": dict(self._action_counts),
            "total_tool_calls": self._total_calls,
            "tool_call_failures": self._tool_call_failures,
            "tool_failure_rate": self._tool_call_failures / max(self._total_calls, 1),
            "state_changing_tool_calls": self._state_changing_calls,
            "state_changing_action_rate": self._state_changing_calls
            / max(self._total_calls, 1),
        }
        if self.output_dir:
            self._flush()
            if trajectory_summary is not None:
                prefix = self.output_dir / self.episode_id
                trajectory_summary["trajectory_artifact"] = (
                    _jsonl_artifact_binding(
                        Path(f"{prefix}.trajectory.jsonl"),
                        schema_version="episode_trajectory_jsonl_v1",
                    )
                )
                evidence_path = Path(f"{prefix}.evidence.jsonl")
                if evidence_path.is_file():
                    trajectory_summary["evidence_ledger_artifact"] = (
                        _jsonl_artifact_binding(
                            evidence_path,
                            schema_version="evidence_ledger_jsonl_v1",
                        )
                    )
            if self.header:
                header_path = self.output_dir / f"{self.episode_id}.header.json"
                header_payload = json.dumps(
                    self.header.to_dict(), indent=2, ensure_ascii=False
                ).encode("utf-8")
                _atomic_write_bytes(header_path, header_payload)
        if trajectory_summary is not None:
            summary["trajectory_summary"] = _json_safe(trajectory_summary)
        if self.output_dir:
            summary_path = self.output_dir / f"{self.episode_id}.summary.json"
            summary_payload = json.dumps(
                _json_safe(summary), indent=2, ensure_ascii=False
            ).encode("utf-8")
            _atomic_write_bytes(summary_path, summary_payload)
        return summary
