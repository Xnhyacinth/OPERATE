"""Cross-process provider request limiter for formal evaluation shards."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Callable


class ProviderDailyQuotaExhausted(RuntimeError):
    """Raised before transport when the configured UTC-day budget is spent."""

    def __init__(self, *, reset_at: str, audit: dict[str, object]) -> None:
        super().__init__(f"provider daily request quota exhausted; reset_at={reset_at}")
        self.reset_at = reset_at
        self.audit = dict(audit)


class ProviderLimiterStateError(RuntimeError):
    """Fail closed when shared limiter state cannot be trusted."""


@dataclass(frozen=True)
class ProviderRequestLimiter:
    """Reserve RPM/RPD slots in a scope-shared, lock-protected state file."""

    rpm_limit: int = 0
    rpd_limit: int = 0
    scope: str | None = None
    state_dir: Path | None = None
    now: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep

    def __post_init__(self) -> None:
        if int(self.rpm_limit) < 0 or int(self.rpd_limit) < 0:
            raise ValueError("provider request limits must be non-negative")
        if self.enabled and not str(self.scope or "").strip():
            raise ValueError(
                "provider_rate_limit_scope is required when a provider limit is enabled"
            )

    @property
    def enabled(self) -> bool:
        return int(self.rpm_limit) > 0 or int(self.rpd_limit) > 0

    def acquire(self) -> dict[str, object]:
        """Reserve one wire request and sleep only for its RPM schedule slot."""

        if not self.enabled:
            return {
                "schema_version": "provider_rate_limit_audit_v1",
                "status": "disabled",
                "rpm_limit": int(self.rpm_limit),
                "rpd_limit": int(self.rpd_limit),
                "scope": self.scope,
                "wait_seconds": 0.0,
            }

        state_dir = self.state_dir or _default_state_dir()
        scope = str(self.scope).strip()
        scope_sha256 = hashlib.sha256(scope.encode("utf-8")).hexdigest()
        state_path = state_dir / f"{scope_sha256}.json"
        lock_path = state_dir / f"{scope_sha256}.lock"
        try:
            state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        except OSError as exc:
            raise ProviderLimiterStateError(
                f"provider limiter state I/O failed: {state_path}"
            ) from exc
        locked = False
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            locked = True
            # Sample the reservation clock only after taking the shared lock.
            # Otherwise concurrent callers can append timestamps in lock order
            # that is the reverse of their pre-lock clock-sampling order.
            now_epoch = float(self.now())
            now_utc = datetime.fromtimestamp(now_epoch, tz=UTC)
            utc_day = now_utc.date().isoformat()
            state = _read_state(state_path, scope=scope, utc_day=utc_day)
            request_times = [
                value
                for value in state.get("request_times", [])
                if value > now_epoch - 60.0
            ]
            day_count = state["day_count"]
            if int(self.rpd_limit) > 0 and day_count >= int(self.rpd_limit):
                reset = datetime.combine(
                    now_utc.date() + timedelta(days=1),
                    datetime.min.time(),
                    tzinfo=UTC,
                )
                reset_at = reset.isoformat().replace("+00:00", "Z")
                audit = _audit_payload(
                    status="daily_quota_exhausted",
                    scope=scope,
                    scope_sha256=scope_sha256,
                    rpm_limit=int(self.rpm_limit),
                    rpd_limit=int(self.rpd_limit),
                    day_count=day_count,
                    wait_seconds=0.0,
                    reserved_at=now_epoch,
                    reset_at=reset_at,
                )
                raise ProviderDailyQuotaExhausted(
                    reset_at=reset_at,
                    audit=audit,
                )

            scheduled_at = now_epoch
            rpm_limit = int(self.rpm_limit)
            if rpm_limit > 0 and len(request_times) >= rpm_limit:
                scheduled_at = max(
                    now_epoch,
                    request_times[-rpm_limit] + 60.0,
                )
            request_times.append(scheduled_at)
            day_count += 1
            _write_state(
                state_path,
                {
                    "schema_version": "provider_rate_limit_state_v1",
                    "scope": scope,
                    "utc_day": utc_day,
                    "day_count": day_count,
                    "request_times": request_times,
                },
            )
        except (ProviderDailyQuotaExhausted, ProviderLimiterStateError):
            raise
        except (OSError, TypeError, ValueError) as exc:
            raise ProviderLimiterStateError(
                f"provider limiter state operation failed: {state_path}"
            ) from exc
        finally:
            cleanup_error: OSError | None = None
            try:
                if locked:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except OSError as exc:
                cleanup_error = exc
            try:
                os.close(lock_fd)
            except OSError as exc:
                cleanup_error = cleanup_error or exc
            if cleanup_error is not None:
                raise ProviderLimiterStateError(
                    f"provider limiter lock cleanup failed: {lock_path}"
                ) from cleanup_error

        wait_seconds = max(0.0, scheduled_at - now_epoch)
        if wait_seconds:
            self.sleep(wait_seconds)
        return _audit_payload(
            status="acquired",
            scope=scope,
            scope_sha256=scope_sha256,
            rpm_limit=int(self.rpm_limit),
            rpd_limit=int(self.rpd_limit),
            day_count=day_count,
            wait_seconds=wait_seconds,
            reserved_at=scheduled_at,
            reset_at=None,
        )


def _default_state_dir() -> Path:
    configured = os.getenv("OPERATE_PROVIDER_RATE_LIMIT_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "operate" / "provider-rate-limits"


def _read_state(path: Path, *, scope: str, utc_day: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"scope": scope, "utc_day": utc_day, "day_count": 0, "request_times": []}
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProviderLimiterStateError(
            f"invalid provider limiter state: {path}"
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "provider_rate_limit_state_v1"
        or payload.get("scope") != scope
        or not isinstance(payload.get("utc_day"), str)
    ):
        raise ProviderLimiterStateError(f"invalid provider limiter state: {path}")
    day_count = payload.get("day_count")
    raw_times = payload.get("request_times")
    if (
        isinstance(day_count, bool)
        or not isinstance(day_count, int)
        or day_count < 0
        or not isinstance(raw_times, list)
    ):
        raise ProviderLimiterStateError(f"invalid provider limiter state: {path}")
    request_times: list[float] = []
    for value in raw_times:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ProviderLimiterStateError(f"invalid provider limiter state: {path}")
        number = float(value)
        if not math.isfinite(number):
            raise ProviderLimiterStateError(f"invalid provider limiter state: {path}")
        request_times.append(number)
    if request_times != sorted(request_times):
        raise ProviderLimiterStateError(f"invalid provider limiter state: {path}")
    return {
        "schema_version": "provider_rate_limit_state_v1",
        "scope": scope,
        "utc_day": utc_day,
        "day_count": day_count if payload["utc_day"] == utc_day else 0,
        # RPM is a rolling window independent of the RPD UTC-day counter.
        # Preserve recent and future reservations across midnight; acquire()
        # applies the exact 60-second cutoff using its current clock.
        "request_times": request_times,
    }


def _write_state(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    write_error: OSError | None = None
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        write_error = exc
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as exc:
            write_error = write_error or exc
    if write_error is not None:
        raise ProviderLimiterStateError(
            f"failed to persist provider limiter state: {path}"
        ) from write_error


def _audit_payload(
    *,
    status: str,
    scope: str,
    scope_sha256: str,
    rpm_limit: int,
    rpd_limit: int,
    day_count: int,
    wait_seconds: float,
    reserved_at: float,
    reset_at: str | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "provider_rate_limit_audit_v1",
        "status": status,
        "scope": scope,
        "scope_sha256": scope_sha256,
        "rpm_limit": rpm_limit,
        "rpd_limit": rpd_limit,
        "utc_day_request_count": day_count,
        "wait_seconds": round(wait_seconds, 6),
        "reserved_at_utc": datetime.fromtimestamp(reserved_at, tz=UTC).isoformat(),
    }
    if reset_at is not None:
        payload["reset_at"] = reset_at
    return payload
