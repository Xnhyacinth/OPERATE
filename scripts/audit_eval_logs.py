#!/usr/bin/env python3
"""Audit a batch LLM eval output directory for log contamination and API errors."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

GEMINI_FC_MARKERS = ("gemini模型fc报错", "-4333", "invalid_function_parameters")
TIMEOUT_MARKERS = ("timeout", "timed out", "readtimeout", "apitimeouterror")
# Markers that an orphan log shows a real LLM attempt was made (HTTP request /
# tool-call dispatch / model response). If a log file contains ZERO of these we
# treat it as an "empty residue" — it can be safely ignored and is reported
# under ``log_files_orphan_empty``. If it contains ANY, we treat it as an
# **interrupted attempt**: the worker started talking to the provider but its
# row never landed in episodes.jsonl. These show up under
# ``log_files_orphan_interrupted`` and are the noisiest source of stat drift.
# Two independent buckets — interrupted = (provider HTTP traffic) OR
# (LLM SDK internal call). Avoid generic strings like ``"agent"`` that
# match every well-formed run log; those produce false positives that
# would degrade an otherwise-clean batch.
PROVIDER_HTTP_MARKERS = (
    "http request",  # httpx INFO line (covers OpenAI / DeepSeek / Volcano via OpenAI SDK)
    "/chat/completions",  # OpenAI-compatible chat completions URL substring
    "/v1/responses",  # OpenAI Responses API
    "/v1/messages",  # Anthropic
    "/v1beta/models/",  # Gemini AI Studio
    "generatecontent",  # Gemini (generateContent / streamGenerateContent)
    "operation-location",  # Azure long-running operation header
)
SDK_INTERNAL_MARKERS = (
    "tool_call",
    "tool_calls",
    "function_call",
    "model_dump",
)
INTERRUPTED_ATTEMPT_MARKERS = PROVIDER_HTTP_MARKERS + SDK_INTERNAL_MARKERS
AUTH_MARKERS = ("401", "unauthorized", "no model permission", "invalid api key")
RATE_LIMIT_MARKERS = ("429", "rate limit", "ratelimit")
SERVER_ERROR_MARKERS = (
    "500",
    "502",
    "503",
    "504",
    "internal server error",
    "bad gateway",
    "service unavailable",
)
AUTH_PATTERNS = (
    re.compile(r"http/\d(?:\.\d)?\s+401\b"),
    re.compile(r"\bstatus(?:\s+code)?\s*[:=]?\s*401\b"),
    re.compile(r"\bcode\s*[:=]?\s*401\b"),
    re.compile(r"\bunauthorized\b"),
    re.compile(r"\bno model permission\b"),
    re.compile(r"\binvalid api key\b"),
    re.compile(r"\bauth(?:entication|orization)?\s+(?:failed|error)\b"),
)
RATE_LIMIT_PATTERNS = (
    re.compile(r"http/\d(?:\.\d)?\s+429\b"),
    re.compile(r"\bstatus(?:\s+code)?\s*[:=]?\s*429\b"),
    re.compile(r"\bcode\s*[:=]?\s*429\b"),
    re.compile(r"\brate[- ]?limit"),
    re.compile(r"\bratelimit\b"),
    re.compile(r"\btoo many requests\b"),
)
SERVER_ERROR_PATTERNS = (
    re.compile(r"http/\d(?:\.\d)?\s+50[0-4]\b"),
    re.compile(r"\bstatus(?:\s+code)?\s*[:=]?\s*50[0-4]\b"),
    re.compile(r"\bcode\s*[:=]?\s*50[0-4]\b"),
    re.compile(r"\binternal server error\b"),
    re.compile(r"\bbad gateway\b"),
    re.compile(r"\bservice unavailable\b"),
)
ARCHIVED_LOG_RE = re.compile(r"\.(?:orphan|stale)-\d{8}T\d{6}Z(?:\.\d+)?\.log$")


def _has_any_pattern(text_l: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(p.search(text_l) for p in patterns)


def _model_from_log_path(log_path: Path, logs_root: Path) -> str:
    rel = log_path.relative_to(logs_root)
    return rel.parts[0] if rel.parts else "unknown"


def _load_jsonl_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for idx, line in enumerate(raw_lines):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if idx == len(raw_lines) - 1:
                break
            raise
    return rows


def _expected_log_key(row: dict, output_dir: Path) -> str | None:
    raw_path = row.get("episode_log_path")
    if not raw_path:
        return None
    model = str(row.get("model") or row.get("agent_name") or "?").replace(
        "llm_agent/", ""
    )
    path = Path(str(raw_path))

    if path.is_absolute():
        try:
            return str(path.relative_to(output_dir))
        except ValueError:
            pass

    parts = path.parts
    if "logs" in parts:
        last_logs_idx = max(i for i, part in enumerate(parts) if part == "logs")
        return str(Path(*parts[last_logs_idx:]))

    return str(Path("logs") / model / path.name)


def _increment_partitioned_metric(
    report: dict[str, object], key: str, *, is_orphan: bool, amount: int
) -> None:
    if amount <= 0:
        return
    report[key] = int(report.get(key, 0)) + amount
    suffix = "orphan" if is_orphan else "backed"
    partitioned_key = f"{key}_{suffix}"
    report[partitioned_key] = int(report.get(partitioned_key, 0)) + amount


def audit(output_dir: Path) -> dict[str, object]:
    episodes_path = output_dir / "episodes.jsonl"
    logs_root = output_dir / "logs"
    report: dict[str, object] = {
        "output_dir": str(output_dir),
        "episodes_with_provider_tool_call_failures": {},
        "episodes_with_rate_limit_retries": {},
        "episodes_with_server_error_retries": {},
    }
    expected_log_keys: set[str] = set()
    terminal_ok_keys: set[tuple[Any, ...]] = set()
    in_flight_rows: list[dict[str, Any]] = []
    episodes_with_provider_tool_call_failures: Counter[str] = Counter()
    episodes_with_rate_limit_retries: Counter[str] = Counter()
    episodes_with_server_error_retries: Counter[str] = Counter()

    if episodes_path.is_file():
        rows = _load_jsonl_rows(episodes_path)
        report["episodes_total"] = len(rows)
        report["episodes_error"] = sum(1 for r in rows if r.get("status") == "error")
        llm_fail_by_model: Counter[str] = Counter()
        zero_tools = 0
        for r in rows:
            model = r.get("model") or r.get("agent_name", "?")
            ts = r.get("trajectory_summary") or {}
            llm = ts.get("llm") or {}
            model_norm = str(model).replace("llm_agent/", "")
            if llm.get("llm_calls_failed", 0) > 0:
                llm_fail_by_model[model] += 1
            if int(llm.get("provider_tool_call_failures", 0) or 0) > 0:
                episodes_with_provider_tool_call_failures[model_norm] += 1
            if int(llm.get("provider_rate_limit_failures", 0) or 0) > 0:
                episodes_with_rate_limit_retries[model_norm] += 1
            if int(llm.get("provider_server_failures", 0) or 0) > 0:
                episodes_with_server_error_retries[model_norm] += 1
            if ts.get("n_tool_calls", 0) == 0:
                zero_tools += 1
            status = str(r.get("status", "ok"))
            expected_log_key = _expected_log_key(r, output_dir)
            if expected_log_key:
                expected_log_keys.add(expected_log_key)
            slug = str(r.get("scenario_slug") or "")
            seed = r.get("seed")
            sig = r.get("scenario_signature")
            temp = r.get("temperature")
            strong: tuple[Any, ...] | None = None
            if (
                slug
                and model_norm
                and seed is not None
                and sig not in (None, "")
                and temp is not None
            ):
                strong = (
                    "strong",
                    slug,
                    model_norm,
                    int(seed),
                    str(sig),
                    f"{float(temp):.6f}",
                )
            legacy = ("legacy", slug, model_norm, int(seed) if seed is not None else -1)
            if status == "ok":
                terminal_ok_keys.add(strong or legacy)
            elif status == "in_flight":
                in_flight_rows.append(
                    {
                        "scenario_slug": slug,
                        "model": model_norm,
                        "seed": int(seed) if seed is not None else None,
                        "scenario_signature": sig,
                        "temperature": temp,
                        "episode_log_path": r.get("episode_log_path"),
                        "_identity_key": strong or legacy,
                    }
                )
        report["episodes_with_llm_failures"] = dict(llm_fail_by_model)
        report["episodes_with_provider_tool_call_failures"] = dict(
            episodes_with_provider_tool_call_failures
        )
        report["episodes_with_rate_limit_retries"] = dict(
            episodes_with_rate_limit_retries
        )
        report["episodes_with_server_error_retries"] = dict(
            episodes_with_server_error_retries
        )
        report["episodes_zero_tool_calls"] = zero_tools
    else:
        report["episodes_total"] = 0
        report["note"] = "episodes.jsonl missing"

    if not logs_root.is_dir():
        report["log_files"] = 0
        report["log_files_total"] = 0
        report["log_files_archived"] = 0
        report["log_files_backed"] = 0
        report["log_files_orphan"] = 0
        report["log_files_orphan_interrupted"] = 0
        report["log_files_orphan_empty"] = 0
        report["sample_orphan_logs"] = []
        report["sample_orphan_interrupted_logs"] = []
        report["orphan_in_flight_rows"] = 0
        report["sample_orphan_in_flight_rows"] = []
        return report

    for key in (
        "logs_with_gemini_fc_markers",
        "fallback_wait_log_lines",
        "logs_with_auth_errors",
        "logs_with_rate_limits",
        "logs_with_server_errors",
        "logs_with_timeouts",
    ):
        report[key] = 0
        report[f"{key}_backed"] = 0
        report[f"{key}_orphan"] = 0

    all_log_files = list(logs_root.rglob("*.log"))
    archived_log_files = [lp for lp in all_log_files if ARCHIVED_LOG_RE.search(lp.name)]
    log_files = [lp for lp in all_log_files if lp not in archived_log_files]
    report["log_files"] = len(log_files)
    report["log_files_total"] = len(all_log_files)
    report["log_files_archived"] = len(archived_log_files)
    report["log_files_backed"] = 0
    report["log_files_orphan"] = 0
    report["log_files_orphan_interrupted"] = 0
    report["log_files_orphan_empty"] = 0
    report["orphan_in_flight_rows"] = 0
    report["sample_orphan_in_flight_rows"] = []
    orphan_logs: list[str] = []
    interrupted_logs: list[dict[str, Any]] = []
    cross_model_gemini: list[str] = []
    for lp in log_files:
        try:
            text = lp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel_path = str(lp.relative_to(output_dir))
        is_orphan = rel_path not in expected_log_keys
        text_l = text.lower()
        looks_like_attempt = any(m in text_l for m in INTERRUPTED_ATTEMPT_MARKERS)
        if is_orphan:
            report["log_files_orphan"] = int(report["log_files_orphan"]) + 1
            if len(orphan_logs) < 8:
                orphan_logs.append(rel_path)
            if looks_like_attempt:
                report["log_files_orphan_interrupted"] = (
                    int(report["log_files_orphan_interrupted"]) + 1
                )
                if len(interrupted_logs) < 8:
                    interrupted_logs.append(
                        {
                            "path": rel_path,
                            "size_bytes": lp.stat().st_size,
                            "n_lines": len(text.splitlines()),
                        }
                    )
            else:
                report["log_files_orphan_empty"] = (
                    int(report["log_files_orphan_empty"]) + 1
                )
        else:
            report["log_files_backed"] = int(report["log_files_backed"]) + 1
        model_dir = _model_from_log_path(lp, logs_root)
        has_gemini_err = any(m in text for m in GEMINI_FC_MARKERS)
        if has_gemini_err:
            _increment_partitioned_metric(
                report, "logs_with_gemini_fc_markers", is_orphan=is_orphan, amount=1
            )
            if "gemini" not in model_dir.lower():
                cross_model_gemini.append(rel_path)
        _increment_partitioned_metric(
            report,
            "fallback_wait_log_lines",
            is_orphan=is_orphan,
            amount=text_l.count("falling back to wait"),
        )
        _increment_partitioned_metric(
            report,
            "logs_with_auth_errors",
            is_orphan=is_orphan,
            amount=int(_has_any_pattern(text_l, AUTH_PATTERNS)),
        )
        _increment_partitioned_metric(
            report,
            "logs_with_rate_limits",
            is_orphan=is_orphan,
            amount=int(_has_any_pattern(text_l, RATE_LIMIT_PATTERNS)),
        )
        _increment_partitioned_metric(
            report,
            "logs_with_server_errors",
            is_orphan=is_orphan,
            amount=int(_has_any_pattern(text_l, SERVER_ERROR_PATTERNS)),
        )
        _increment_partitioned_metric(
            report,
            "logs_with_timeouts",
            is_orphan=is_orphan,
            amount=int(any(m in text_l for m in TIMEOUT_MARKERS)),
        )

    report["sample_orphan_logs"] = orphan_logs
    report["sample_orphan_interrupted_logs"] = interrupted_logs
    orphan_in_flight = [
        {
            "scenario_slug": row["scenario_slug"],
            "model": row["model"],
            "seed": row["seed"],
            "scenario_signature": row.get("scenario_signature"),
            "temperature": row.get("temperature"),
            "episode_log_path": row.get("episode_log_path"),
        }
        for row in in_flight_rows
        if row["_identity_key"] not in terminal_ok_keys
    ]
    report["orphan_in_flight_rows"] = len(orphan_in_flight)
    report["sample_orphan_in_flight_rows"] = orphan_in_flight[:8]
    report["logs_gemini_marker_wrong_model_dir"] = len(cross_model_gemini)
    report["sample_wrong_model_logs"] = cross_model_gemini[:8]
    report["log_contamination_likely"] = len(cross_model_gemini) > 0

    # Invariance check: every orphan log must be classified as either
    # interrupted or empty. Guards against future refactors that silently
    # break the partition.
    n_orphan = int(report["log_files_orphan"])
    n_int = int(report["log_files_orphan_interrupted"])
    n_empty = int(report["log_files_orphan_empty"])
    assert n_int + n_empty == n_orphan, (
        f"orphan classification drift: interrupted({n_int}) + empty({n_empty}) "
        f"!= orphan({n_orphan})"
    )
    return report


def write_markdown(report: dict[str, object], out_path: Path) -> None:
    lines = [
        "# Batch log audit",
        "",
        f"- Output directory: `{report.get('output_dir')}`",
        f"- Episodes: **{report.get('episodes_total', 0)}**",
        f"- Episode errors: **{report.get('episodes_error', 0)}**",
        f"- Log files scanned: **{report.get('log_files_total', report.get('log_files', 0))}**",
        f"- Episode-backed logs: **{report.get('log_files_backed', 0)}**",
        f"- Orphan logs: **{report.get('log_files_orphan', 0)}**"
        f" (interrupted: {report.get('log_files_orphan_interrupted', 0)},"
        f" empty: {report.get('log_files_orphan_empty', 0)})",
        f"- In-flight rows without terminal ok row: **{report.get('orphan_in_flight_rows', 0)}**",
        "",
        "## Signals",
        "",
        f"- Episodes with LLM failures: `{report.get('episodes_with_llm_failures', {})}`",
        f"- Structured provider tool-call failures: `{report.get('episodes_with_provider_tool_call_failures', {})}`",
        f"- Structured rate-limit retries: `{report.get('episodes_with_rate_limit_retries', {})}`",
        f"- Structured server-error retries: `{report.get('episodes_with_server_error_retries', {})}`",
        f"- Episodes with zero tool calls: **{report.get('episodes_zero_tool_calls', 0)}**",
        f"- Fallback-to-wait log lines: **{report.get('fallback_wait_log_lines', 0)}**",
        f"- Auth error logs: **{report.get('logs_with_auth_errors', 0)}**",
        f"- Rate-limit logs: **{report.get('logs_with_rate_limits', 0)}**",
        f"- Server-error logs: **{report.get('logs_with_server_errors', 0)}**",
        f"- Timeout logs: **{report.get('logs_with_timeouts', 0)}**",
        f"- Gemini FC marker logs: **{report.get('logs_with_gemini_fc_markers', 0)}**",
        f"- Cross-model contamination likely: **{report.get('log_contamination_likely', False)}**",
    ]
    if int(report.get("log_files_orphan", 0) or 0) > 0:
        lines.extend(
            [
                "",
                "## Backed vs orphan split",
                "",
                (
                    "- Fallback-to-wait log lines (backed / orphan): "
                    f"**{report.get('fallback_wait_log_lines_backed', 0)} / "
                    f"{report.get('fallback_wait_log_lines_orphan', 0)}**"
                ),
                (
                    "- Rate-limit logs (backed / orphan): "
                    f"**{report.get('logs_with_rate_limits_backed', 0)} / "
                    f"{report.get('logs_with_rate_limits_orphan', 0)}**"
                ),
                (
                    "- Server-error logs (backed / orphan): "
                    f"**{report.get('logs_with_server_errors_backed', 0)} / "
                    f"{report.get('logs_with_server_errors_orphan', 0)}**"
                ),
                (
                    "- Timeout logs (backed / orphan): "
                    f"**{report.get('logs_with_timeouts_backed', 0)} / "
                    f"{report.get('logs_with_timeouts_orphan', 0)}**"
                ),
                (
                    "- Auth error logs (backed / orphan): "
                    f"**{report.get('logs_with_auth_errors_backed', 0)} / "
                    f"{report.get('logs_with_auth_errors_orphan', 0)}**"
                ),
                (
                    "- Gemini FC marker logs (backed / orphan): "
                    f"**{report.get('logs_with_gemini_fc_markers_backed', 0)} / "
                    f"{report.get('logs_with_gemini_fc_markers_orphan', 0)}**"
                ),
            ]
        )
    orphan_logs = report.get("sample_orphan_logs") or []
    if orphan_logs:
        lines.extend(["", "## Sample orphan logs", ""])
        for item in orphan_logs:
            lines.append(f"- `{item}`")
    interrupted = report.get("sample_orphan_interrupted_logs") or []
    if interrupted:
        lines.extend(
            [
                "",
                "## Sample interrupted-attempt orphan logs",
                "",
                (
                    "These orphan logs contain real provider traffic but never produced "
                    "an `episodes.jsonl` row — most likely interrupted runs that should "
                    "be re-attempted or explicitly cleaned up before the next finalize."
                ),
                "",
            ]
        )
        for item in interrupted:
            if isinstance(item, dict):
                lines.append(
                    f"- `{item.get('path')}` "
                    f"(size={item.get('size_bytes')}B, "
                    f"lines={item.get('n_lines')})"
                )
            else:
                lines.append(f"- `{item}`")
    in_flight = report.get("sample_orphan_in_flight_rows") or []
    if in_flight:
        lines.extend(["", "## Sample orphan in-flight rows", ""])
        for item in in_flight:
            if isinstance(item, dict):
                lines.append(
                    f"- model={item.get('model')} "
                    f"scenario=`{item.get('scenario_slug')}` "
                    f"seed={item.get('seed')}"
                )
            else:
                lines.append(f"- `{item}`")
    wrong_model_logs = report.get("sample_wrong_model_logs") or []
    if wrong_model_logs:
        lines.extend(["", "## Sample suspicious logs", ""])
        for item in wrong_model_logs:
            lines.append(f"- `{item}`")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output_dir", type=Path, help="batch_results/<session>")
    args = p.parse_args()
    report = audit(args.output_dir.resolve())
    write_markdown(report, args.output_dir.resolve() / "LOG_AUDIT.md")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
