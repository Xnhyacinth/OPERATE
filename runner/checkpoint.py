"""Hash-bound logical episode recovery by replaying successful agent boundaries.

The normal runner recreates backend, RNG, delayed tools, and local scheduling
state from the same seed. This proxy supplies archived decisions only after
the inputs, environment observation, and authoritative evidence match. It
restores explicit agent JSON state instead of calling the provider again.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from functools import partial
from pathlib import Path
from typing import Any

from core import Action, ToolCall


class CheckpointIntegrityError(RuntimeError):
    """A checkpoint cannot safely be used or extended; no resampling allowed."""


_SCHEMA = "logical_episode_checkpoint_v1"
_ZERO_HASH = "0" * 64
_ACTION_METHODS = {
    "act",
    "investigate",
    "start_decision_epoch",
    "continue_decision_epoch",
    "reconcile_control_receipts",
}
_VOID_METHODS = {"observe_transition", "on_episode_end"}


def _json_default(value: Any) -> Any:
    if isinstance(value, (Action, ToolCall)):
        return value.to_dict()
    raise TypeError(f"checkpoint requires JSON values, got {type(value).__name__}")


def _encode(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_encode(value)).hexdigest()


def _clone(value: Any) -> Any:
    return json.loads(_encode(value))


def _state_patch(previous: Any, current: Any) -> dict[str, Any]:
    if (
        type(previous) is type(current)
        and previous == current
        and _encode(previous) == _encode(current)
    ):
        return {"op": "same"}
    if isinstance(previous, dict) and isinstance(current, dict):
        return {
            "op": "dict",
            "remove": sorted(previous.keys() - current.keys()),
            "changes": {
                key: _state_patch(previous[key], value)
                if key in previous
                else {"op": "set", "value": value}
                for key, value in current.items()
                if key not in previous or _encode(previous[key]) != _encode(value)
            },
        }
    if (
        isinstance(previous, list)
        and isinstance(current, list)
        and len(current) >= len(previous)
    ):
        if _encode(current[: len(previous)]) == _encode(previous):
            return {
                "op": "append",
                "offset": len(previous),
                "items": current[len(previous) :],
            }
    return {"op": "set", "value": current}


def _apply_patch(previous: Any, patch: dict[str, Any]) -> Any:
    op = patch.get("op")
    if op == "same":
        return previous
    if op == "set" and "value" in patch:
        return patch["value"]
    if op == "dict" and isinstance(previous, dict):
        current = dict(previous)
        for key in patch["remove"]:
            del current[key]
        for key, change in patch["changes"].items():
            current[key] = _apply_patch(previous.get(key), change)
        return current
    if (
        op == "append"
        and isinstance(previous, list)
        and patch.get("offset") == len(previous)
    ):
        return previous + patch["items"]
    raise CheckpointIntegrityError("invalid checkpoint state delta")


def _restore_state(previous: Any, delta: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(delta, dict) or delta.get("base_sha256") != _digest(previous):
        raise CheckpointIntegrityError("checkpoint state prefix mismatch")
    current = _apply_patch(previous, delta["patch"])
    if not isinstance(current, dict) or delta.get("sha256") != _digest(current):
        raise CheckpointIntegrityError("checkpoint restored state mismatch")
    return current


def _action_from_json(value: dict[str, Any]) -> Action:
    if not isinstance(value, dict) or set(value) != {
        "actions",
        "dominant_action",
        "assistant_text",
        "rationale",
    }:
        raise CheckpointIntegrityError("invalid checkpoint action")
    if not isinstance(value["actions"], list):
        raise CheckpointIntegrityError("invalid checkpoint tool calls")
    return Action(
        tool_calls=[ToolCall(**call) for call in _clone(value["actions"])],
        dominant=value["dominant_action"],
        assistant_text=value["assistant_text"],
        rationale=value["rationale"],
    )


class JournaledAgent:
    """Proxy a reset logical LLM agent; caller must close it in a finally block.

    A separate, atomically replaced head detects valid-line truncation as well
    as torn appends. An append/head crash is held for inspection, never silently
    repaired. The exclusive lock prevents concurrent same-cell invocations.
    """

    def __init__(
        self, agent: Any, env: Any, path: str | Path, identity: dict[str, Any]
    ) -> None:
        self._agent = agent
        self._env = env
        self._path = Path(path)
        self._head_path = self._path.with_suffix(self._path.suffix + ".head.json")
        self._lock = None
        self._failure: CheckpointIntegrityError | None = None
        self._records: list[dict[str, Any]] = []
        self._state: dict[str, Any] | None = None
        self._cursor = 0
        self._replayed = 0
        self._new = 0
        self._reused_provider_requests: int | None = 0
        self._head = {
            "schema_version": _SCHEMA,
            "record_count": 0,
            "head_sha256": _ZERO_HASH,
            "byte_count": 0,
        }
        if getattr(agent, "name", None) != "llm_agent" or getattr(
            getattr(agent, "config", None), "interaction_mode", None
        ) not in {"logical_persistent", "logical_stateless"}:
            raise CheckpointIntegrityError("checkpoint requires a logical llm_agent")
        if not isinstance(identity, dict) or not identity:
            raise CheckpointIntegrityError("checkpoint identity is required")
        for name in ("export_resume_state", "import_resume_state"):
            if not callable(getattr(agent, name, None)):
                raise CheckpointIntegrityError(
                    f"agent missing checkpoint interface: {name}"
                )
        self._identity = _clone(identity)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self._path.with_suffix(self._path.suffix + ".lock")
        self._lock = os.fdopen(os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600), "a+b")
        try:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._load_or_create()
            self._journal_stat = self._stat_signature()
            self._resume_head = dict(self._head)
        except Exception as exc:
            self.close()
            if isinstance(exc, CheckpointIntegrityError):
                raise
            raise CheckpointIntegrityError(
                "checkpoint cannot be opened or validated"
            ) from exc

    def __getattr__(self, name: str) -> Any:
        if name in {"reset", "deliberate"}:
            raise CheckpointIntegrityError(
                "checkpoint requires a reset agent and does not support multi-turn deliberation"
            )
        agent = object.__getattribute__(self, "_agent")
        value = getattr(agent, name)
        if name in _ACTION_METHODS | _VOID_METHODS and callable(value):
            return partial(self._invoke, name)
        return value

    def _load_or_create(self) -> None:
        if self._path.exists() != self._head_path.exists():
            raise CheckpointIntegrityError("checkpoint journal/head pair is incomplete")
        if not self._path.exists():
            with os.fdopen(
                os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "wb"
            ) as stream:
                stream.flush()
                os.fsync(stream.fileno())
            self._append({"kind": "header", "identity": self._identity})
            return
        head = json.loads(self._head_path.read_bytes())
        previous = _ZERO_HASH
        count = 0
        byte_count = 0
        state = None
        with self._path.open("rb") as stream:
            for line in stream:
                if not line.endswith(b"\n"):
                    raise CheckpointIntegrityError("checkpoint has an incomplete tail")
                record = json.loads(line)
                recorded_hash = record.pop("sha256")
                if (
                    record.get("previous_sha256") != previous
                    or _digest(record) != recorded_hash
                ):
                    raise CheckpointIntegrityError("checkpoint hash chain mismatch")
                if (
                    record.get("sequence") != count
                    or record.get("schema_version") != _SCHEMA
                ):
                    raise CheckpointIntegrityError(
                        "checkpoint sequence/schema mismatch"
                    )
                if count == 0:
                    if (
                        record.get("kind") != "header"
                        or record.get("identity") != self._identity
                    ):
                        raise CheckpointIntegrityError("checkpoint identity mismatch")
                else:
                    state = self._validate_boundary(record, state)
                    self._records.append(record)
                previous = recorded_hash
                count += 1
                byte_count += len(line)
        expected_head = {
            "schema_version": _SCHEMA,
            "record_count": count,
            "head_sha256": previous,
            "byte_count": byte_count,
        }
        if count == 0 or head != expected_head:
            raise CheckpointIntegrityError(
                "checkpoint head mismatch or journal truncation"
            )
        self._head = expected_head
        if state is not None:
            requests = (state.get("interaction_stats") or {}).get(
                "provider_request_records"
            )
            self._reused_provider_requests = (
                len(requests) if isinstance(requests, list) else None
            )

    @staticmethod
    def _validate_boundary(
        record: dict[str, Any], previous_state: Any
    ) -> dict[str, Any]:
        method = record.get("method")
        if (
            record.get("kind") != "boundary"
            or method not in _ACTION_METHODS | _VOID_METHODS
        ):
            raise CheckpointIntegrityError("invalid checkpoint boundary")
        for key in ("input_sha256", "environment_sha256", "evidence_sha256"):
            value = record.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise CheckpointIntegrityError(
                    f"invalid checkpoint boundary hash: {key}"
                )
        if method in _ACTION_METHODS:
            _action_from_json(record.get("result"))
        elif record.get("result") is not None:
            raise CheckpointIntegrityError("invalid checkpoint observer result")
        return _restore_state(previous_state, record.get("agent_state_delta"))

    def _append(self, payload: dict[str, Any]) -> None:
        if self._head["record_count"]:
            self._check_current_head()
        record = {
            **payload,
            "schema_version": _SCHEMA,
            "sequence": self._head["record_count"],
            "previous_sha256": self._head["head_sha256"],
        }
        record["sha256"] = _digest(record)
        encoded = _encode(record) + b"\n"
        if self._path.stat().st_size != self._head["byte_count"]:
            raise CheckpointIntegrityError("checkpoint changed during execution")
        with self._path.open("ab") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        head = {
            "schema_version": _SCHEMA,
            "record_count": self._head["record_count"] + 1,
            "head_sha256": record["sha256"],
            "byte_count": self._head["byte_count"] + len(encoded),
        }
        fd, temporary = tempfile.mkstemp(
            prefix=self._head_path.name + ".", dir=self._path.parent
        )
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(_encode(head) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._head_path)
            directory = os.open(self._path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        self._head = head
        self._journal_stat = self._stat_signature()

    def _stat_signature(self) -> tuple[int, ...]:
        stat = self._path.stat()
        return (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )

    def _check_current_head(self) -> None:
        # The chain was fully checked under the exclusive lock at open. Detect
        # out-of-band edits before another call without rereading growing state.
        try:
            changed = (
                self._stat_signature() != self._journal_stat
                or json.loads(self._head_path.read_bytes()) != self._head
            )
        except (OSError, ValueError) as exc:
            raise CheckpointIntegrityError("checkpoint head is unavailable") from exc
        if changed:
            raise CheckpointIntegrityError("checkpoint changed during execution")

    def _invoke(self, method: str, *args: Any, **kwargs: Any) -> Any:
        if self._failure is not None:
            raise self._failure
        if self._lock is None:
            raise CheckpointIntegrityError("checkpoint is closed")
        try:
            self._check_current_head()
            boundary = {
                "kind": "boundary",
                "method": method,
                "input_sha256": _digest({"args": args, "kwargs": kwargs}),
                "environment_sha256": _digest(self._env.snapshot()),
                "evidence_sha256": _digest(self._env.evidence.to_jsonable()),
            }
            if self._cursor < len(self._records):
                record = self._records[self._cursor]
                if any(record.get(key) != value for key, value in boundary.items()):
                    raise CheckpointIntegrityError(
                        f"checkpoint replay diverged at boundary {self._cursor}: {method}"
                    )
                state = _restore_state(self._state, record["agent_state_delta"])
                self._agent.import_resume_state(_clone(state))
                self._state = state
                self._cursor += 1
                self._replayed += 1
                return (
                    _action_from_json(record["result"])
                    if method in _ACTION_METHODS
                    else None
                )
        except Exception as exc:
            self._failure = (
                exc
                if isinstance(exc, CheckpointIntegrityError)
                else CheckpointIntegrityError("checkpoint replay validation failed")
            )
            raise self._failure from exc

        # Provider failures remain provider failures. Only successful boundaries
        # are durable; no returned action reaches the environment before fsync.
        try:
            result = getattr(self._agent, method)(*args, **kwargs)
        except Exception:
            # Keep the original provider error for quota/retry classification.
            # The agent may have mutated its session before raising; another
            # attempt must reconstruct the settled frontier, not reuse it.
            self._failure = CheckpointIntegrityError(
                "checkpoint requires episode reconstruction after failed agent boundary"
            )
            raise
        try:
            if method in _ACTION_METHODS and not isinstance(result, Action):
                raise CheckpointIntegrityError("checkpoint agent must return Action")
            if method in _VOID_METHODS and result is not None:
                raise CheckpointIntegrityError("checkpoint observer must return None")
            state = _clone(self._agent.export_resume_state())
            record = {
                **boundary,
                "result": result.to_dict() if isinstance(result, Action) else None,
                "agent_state_delta": {
                    "base_sha256": _digest(self._state),
                    "sha256": _digest(state),
                    "patch": _state_patch(self._state, state),
                },
            }
            self._validate_boundary(record, self._state)
            self._append(record)
            self._records.append(record)
            self._state = state
            self._cursor += 1
            self._new += 1
            return result
        except Exception as exc:
            self._failure = (
                exc
                if isinstance(exc, CheckpointIntegrityError)
                else CheckpointIntegrityError("checkpoint durability failure")
            )
            raise self._failure from exc

    def assert_replay_complete(self) -> None:
        if self._failure is not None:
            raise self._failure
        if self._cursor != len(self._records):
            raise CheckpointIntegrityError(
                "checkpoint replay ended before saved frontier"
            )
        self._check_current_head()

    def progress(self) -> dict[str, Any]:
        return {
            "path": str(self._path),
            **self._head,
            "resume_source_head_sha256": self._resume_head["head_sha256"],
            "resume_source_record_count": self._resume_head["record_count"],
            "replayed_boundaries": self._replayed,
            "new_boundaries": self._new,
            "reused_provider_request_count": self._reused_provider_requests,
        }

    def close(self) -> None:
        if self._lock is not None:
            self._lock.close()
            self._lock = None
