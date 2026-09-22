"""Narrow CPU-reference reuse across audited LLM-provider-only code changes.

This never makes model executions share a runtime identity. Source/seed/native
library contracts remain required by the executed backends; this receipt checks
code and declared dependency pins, not external provider-run certification.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from core.implementation_identity import _runtime_code_files, implementation_identity

_PROVIDER_CONSTANTS = {
    "STATIC_MODEL_TOOL_CHOICE_SUPPORT",
    "STATIC_MODEL_RESPONSE_ALIASES",
}
_UNUSED_CPU_METHODS = {"_record_provider_request", "_call_openai_protocol_repair"}


def provider_only_change(before: str, after: str) -> bool:
    def normalize(source):
        tree = ast.parse(source)
        for node in tree.body:
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id in _PROVIDER_CONSTANTS
            ):
                allowed = (
                    ast.Dict,
                    ast.Set,
                    ast.List,
                    ast.Tuple,
                    ast.Constant,
                    ast.Load,
                    ast.Call,
                    ast.Name,
                )
                if any(not isinstance(item, allowed) for item in ast.walk(node.value)):
                    raise ValueError("provider map contains nonliteral expression")
                for item in ast.walk(node.value):
                    if isinstance(item, ast.Name) and item.id != "frozenset":
                        raise ValueError("provider map references executable state")
                    if isinstance(item, ast.Call) and (
                        not isinstance(item.func, ast.Name)
                        or item.func.id != "frozenset"
                        or item.keywords
                    ):
                        raise ValueError("provider map contains executable call")
                node.value = ast.Constant(value=None)
            if isinstance(node, ast.ClassDef) and node.name == "LLMAgent":
                for method in node.body:
                    if (
                        isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and method.name in _UNUSED_CPU_METHODS
                    ):
                        method.body = [ast.Pass()]
        return ast.dump(tree, include_attributes=False)

    try:
        return normalize(before) == normalize(after)
    except (SyntaxError, ValueError):
        return False


def verify_reference_compatibility(reference_root: Path, model_root: Path):
    roots = [reference_root.resolve(), model_root.resolve()]
    maps = [
        {p.relative_to(root).as_posix(): p for p in _runtime_code_files(root)}
        for root in roots
    ]
    if maps[0].keys() != maps[1].keys():
        raise ValueError("reference compatibility runtime file sets differ")
    differences = []
    for name in maps[0]:
        a, b = maps[0][name].read_bytes(), maps[1][name].read_bytes()
        if a != b:
            if name != "baselines/llm_agent.py" or not provider_only_change(
                a.decode(), b.decode()
            ):
                raise ValueError(f"CPU reference code compatibility unproven: {name}")
            differences.append(
                {
                    "path": name,
                    "reference_sha256": hashlib.sha256(a).hexdigest(),
                    "model_sha256": hashlib.sha256(b).hexdigest(),
                }
            )
    pins = {}
    for name in ("pyproject.toml", "uv.lock"):
        a, b = (roots[0] / name).read_bytes(), (roots[1] / name).read_bytes()
        if a != b:
            raise ValueError("reference dependency pins differ")
        pins[name] = hashlib.sha256(a).hexdigest()
    identities = [
        implementation_identity(root)["evaluation_runtime_sha256"] for root in roots
    ]
    receipt = {
        "schema_version": "cpu_reference_provider_only_compatibility.v1",
        "reference_runtime_identity": identities[0],
        "model_runtime_identity": identities[1],
        "differences": differences,
        "dependency_pins": pins,
        "scope": "four_fixed_CPU_policies_only_not_model_cohort_merging",
        "allowed_methods": sorted(_UNUSED_CPU_METHODS),
        "native_dependency_condition": "same_backend_enforced_native_source_and_runtime_locks",
        "formal_run_certified": False,
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return receipt
