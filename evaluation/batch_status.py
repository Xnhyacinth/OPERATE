"""Execution availability counters shared by batch reports and resume logic."""

from typing import Any


def row_is_quota_exhausted(row: dict[str, Any]) -> bool:
    if str(row.get("status")) != "error":
        return False
    return bool(row.get("quota_parked")) or "ProviderQuotaExhaustedError" in str(
        row.get("error") or ""
    )


def execution_status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count execution outcomes; provider availability is not task performance."""
    unavailable = [row for row in rows if row_is_quota_exhausted(row)]
    return {
        "n_episodes_ok": sum(row.get("status") == "ok" for row in rows),
        "n_episodes_error": sum(
            row.get("status") not in {"ok", "in_flight"}
            and not row_is_quota_exhausted(row) for row in rows
        ),
        "n_episodes_quota_unavailable": len(unavailable),
        "n_episodes_quota_parked": sum(
            row.get("execution_started") is False
            or (
                "execution_started" not in row
                and str(row.get("error") or "").startswith(
                    "ProviderQuotaExhaustedError: parked after provider quota exhausted"
                )
            )
            for row in unavailable
        ),
        "n_episodes_in_flight": sum(row.get("status") == "in_flight" for row in rows),
    }
