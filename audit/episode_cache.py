"""Content-addressed episode cache and resumable audit checkpoints."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from core.implementation_identity import implementation_identity
from evaluation import SCORING_VERSION

CACHE_SCHEMA_VERSION = "0.1"
CHECKPOINT_SCHEMA_VERSION = "0.1"
# Bump whenever episode execution semantics change in a way that can alter
# deterministic audit outputs. 1.2 adds runner-parity counterfactual tick
# records for phase-aware task completion in audit episode summaries.
AUDIT_EPISODE_CONTRACT_VERSION = "1.2"
_RUNTIME_DISTRIBUTIONS = (
    "eclipse-sumo",
    "grid2op",
    "matpowercaseframes",
    "or-gym",
    "pandapower",
    "pymgrid",
    "pyvrp",
)


def _canonical_digest(value: Any) -> str:
    blob = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


class EpisodeCache:
    """Persist deterministic episode summaries under a validated cache key."""

    def __init__(self, directory: Path, *, enabled: bool = True) -> None:
        self.directory = Path(directory)
        self.enabled = bool(enabled)
        self.hits = 0
        self.misses = 0

    def _identity(
        self,
        *,
        scenario: dict[str, Any],
        row: dict[str, Any],
        agent_name: str,
        score_context: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "audit_episode_contract_version": AUDIT_EPISODE_CONTRACT_VERSION,
            "scoring_version": SCORING_VERSION,
            "implementation_tree_sha256": implementation_identity()[
                "implementation_tree_sha256"
            ],
            "runtime": {
                "python": platform.python_version(),
                "packages": {
                    name: _distribution_version(name) for name in _RUNTIME_DISTRIBUTIONS
                },
            },
            "scenario_digest": _canonical_digest(scenario),
            "scenario_signature": str(row.get("scenario_signature") or ""),
            "agent_name": str(agent_name),
            "seed": int(scenario.get("seed", 42)),
            "score_context": score_context,
        }

    def get_or_compute(
        self,
        *,
        scenario: dict[str, Any],
        row: dict[str, Any],
        agent_name: str,
        score_context: dict[str, Any],
        compute: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        if not self.enabled:
            self.misses += 1
            return compute()
        identity = self._identity(
            scenario=scenario,
            row=row,
            agent_name=agent_name,
            score_context=score_context,
        )
        key = _canonical_digest(identity)
        path = self.directory / f"{key}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("identity") == identity and isinstance(payload.get("result"), dict):
                self.hits += 1
                return payload["result"]
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass

        result = compute()
        self.misses += 1
        _atomic_write_json(
            path,
            {
                "identity": identity,
                "result": result,
                "created_at_unix": time.time(),
            },
        )
        return result

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "directory": str(self.directory),
            "enabled": self.enabled,
            "hits": self.hits,
            "misses": self.misses,
        }


class AuditCheckpoint:
    """Registry-scoped, atomic checkpoint for completed audit gates."""

    def __init__(self, path: Path, *, registry_digest: str) -> None:
        self.path = Path(path)
        self.registry_digest = registry_digest

    def _read(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}
        if (
            payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
            or payload.get("registry_digest") != self.registry_digest
        ):
            return {}
        return payload

    def load_gate(self, name: str) -> dict[str, Any] | None:
        gate = (self._read().get("gates") or {}).get(name)
        return gate if isinstance(gate, dict) else None

    def save_gate(self, name: str, result: dict[str, Any]) -> None:
        payload = self._read() or {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "registry_digest": self.registry_digest,
            "gates": {},
        }
        payload.setdefault("gates", {})[name] = result
        payload["updated_at_unix"] = time.time()
        _atomic_write_json(self.path, payload)

    def reset(self) -> None:
        with suppress(FileNotFoundError):
            self.path.unlink()


_ACTIVE_EPISODE_CACHE: ContextVar[EpisodeCache | None] = ContextVar(
    "audit_episode_cache", default=None
)
_DEFAULT_EPISODE_CACHE: EpisodeCache | None = None


def active_episode_cache() -> EpisodeCache | None:
    return _ACTIVE_EPISODE_CACHE.get() or _DEFAULT_EPISODE_CACHE


def configure_episode_cache(cache: EpisodeCache | None) -> None:
    global _DEFAULT_EPISODE_CACHE
    _DEFAULT_EPISODE_CACHE = cache


@contextmanager
def use_episode_cache(cache: EpisodeCache | None) -> Iterator[None]:
    token = _ACTIVE_EPISODE_CACHE.set(cache)
    try:
        yield
    finally:
        _ACTIVE_EPISODE_CACHE.reset(token)


def registry_digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint_scope(registry_path: Path, *, samples_per_family: int) -> str:
    return _canonical_digest(
        {
            "registry_digest": registry_digest(registry_path),
            "samples_per_family": int(samples_per_family),
            "scoring_version": SCORING_VERSION,
            "audit_episode_contract_version": AUDIT_EPISODE_CONTRACT_VERSION,
        }
    )


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
