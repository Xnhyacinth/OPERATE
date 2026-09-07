from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BINDINGS_PATH = REPO_ROOT / "docs" / "provider_route_bindings.json"
FORMAL_PATH = REPO_ROOT / "docs" / "FORMAL_EVALUATION.md"

_TABLE_ROW = re.compile(
    r"\| `([^`]+)` \| ([0-9,]+) / ([0-9,]+) \| ([0-9,]+) \| ([0-9]+) (?:Full|Lite) \|"
)


def _load_bindings() -> dict:
    payload = json.loads(BINDINGS_PATH.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "operate-provider-route-bindings-v1"
    return payload


def test_provider_route_bindings_are_complete_and_append_only() -> None:
    payload = _load_bindings()
    routes = payload["routes"]
    models = [row["model"] for row in routes]
    assert len(models) == len(set(models))
    assert "glm-5.3-flash-ioa" in models
    for row in routes:
        assert row["request_max_tokens"] == row["max_output_tokens"]
        assert row["context_window_tokens"] >= row["request_max_tokens"]
        assert row["provider"] in {"tencent", "openrouter"}
        assert row["workers_lite"] >= 1
        assert row["workers_full"] >= 1
    changelog = payload["changelog"]
    assert changelog
    assert all("date" in item and "change" in item for item in changelog)
    public_env = payload["public_credential_env"]
    assert public_env["tencent"] == "TENCENT_API_KEY"
    omit = set(payload.get("public_omit_keys", ["maintainer_jobs"]))
    exported = {key: value for key, value in payload.items() if key not in omit}
    assert "maintainer_jobs" not in exported
    assert "TENCENT_API_KEY" in json.dumps(public_env)


def test_formal_runbook_table_matches_provider_route_bindings() -> None:
    payload = _load_bindings()
    by_model = {row["model"]: row for row in payload["routes"]}
    runbook = FORMAL_PATH.read_text(encoding="utf-8")
    rows = _TABLE_ROW.findall(runbook)
    assert rows, "formal runbook is missing the native route table"
    for model, context, output, request, workers in rows:
        bound = by_model[model]
        assert bound["context_window_tokens"] == int(context.replace(",", ""))
        assert bound["max_output_tokens"] == int(output.replace(",", ""))
        assert bound["request_max_tokens"] == int(request.replace(",", ""))
        worker_count = int(workers)
        assert worker_count in {bound["workers_full"], bound["workers_lite"]}


def test_glm53_lite_binding_is_one_million_by_128k() -> None:
    payload = _load_bindings()
    glm = next(row for row in payload["routes"] if row["model"] == "glm-5.3-flash-ioa")
    assert glm["context_window_tokens"] == 1_000_000
    assert glm["max_output_tokens"] == 128_000
    assert glm["request_max_tokens"] == 128_000
    assert glm["thinking_type"] == "enabled"
    assert glm["reasoning_effort"] == "high"
    assert glm["reasoning_effort_format"] == "native"
    assert glm["workers_lite"] == 12
    runbook = FORMAL_PATH.read_text(encoding="utf-8")
    assert "run_lite.py" in runbook
    assert "--model-context-window-tokens 1000000" in runbook
    assert "--model-max-output-tokens 128000" in runbook
    assert "--max-tokens 128000" in runbook
    assert "--models glm-5.3-flash-ioa" in runbook
