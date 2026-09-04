"""runner.batch — multi-episode batch helpers (P3-2).

Moved verbatim from ``batch_eval.py``: ``expand_scenarios`` (was
``_expand_scenarios``) and ``run_one_safe`` (was ``_run_one_safe``),
plus the ``_episode_file_logging`` context manager they depend on.
``batch_eval.py`` re-imports these and keeps the legacy ``_``-private
aliases for backward compatibility.
"""

from __future__ import annotations

import glob
import logging
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from runner.episode import run_one

REPO_ROOT = Path(__file__).resolve().parent.parent

# ``run_one_safe`` needs ``run_one`` (imported above from ``runner.episode``)
# and ``load_scenario_yaml`` (a CLI-bound helper in ``run.py`` that depends
# on ``SCENARIOS_ROOT``). The latter is imported lazily inside
# ``run_one_safe`` to avoid a static import cycle with ``run``.
LOGGER = logging.getLogger("batch_eval")


def _append_sanitized_failure_trace(
    log_path: Path,
    *,
    public_error: str,
    error: BaseException,
) -> None:
    """Append traceback frames without re-emitting an unredacted exception."""

    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    failure_logger = logging.getLogger("batch_eval.episode_failure")
    failure_logger.propagate = False
    failure_logger.addHandler(handler)
    try:
        stack = "".join(traceback.format_list(traceback.extract_tb(error.__traceback__)))
        failure_logger.error("episode aborted: %s\n%s", public_error, stack)
    finally:
        failure_logger.removeHandler(handler)
        handler.close()


@contextmanager
def _episode_file_logging(log_path: Path) -> Iterator[None]:
    """Attach one file handler for a single episode; always remove on exit.

    ProcessPool workers run many episodes sequentially. Leaving handlers on
    the root logger would duplicate every later episode into earlier log files.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        stale = log_path.with_name(f"{log_path.stem}.stale-{ts}{log_path.suffix}")
        idx = 1
        while stale.exists():
            stale = log_path.with_name(
                f"{log_path.stem}.stale-{ts}.{idx}{log_path.suffix}"
            )
            idx += 1
        log_path.rename(stale)
    root = logging.getLogger()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    fh.setLevel(logging.INFO)
    if root.level > logging.INFO:
        root.setLevel(logging.INFO)
    root.addHandler(fh)
    try:
        yield
    finally:
        root.removeHandler(fh)
        fh.close()


def expand_scenarios(patterns: list[str]) -> list[str]:
    """Expand glob patterns into a sorted list of scenario slugs.

    Always recurses below any matched directory, picking up every .yaml
    file inside. ``"family/mode/*"`` therefore expands to the union of
    every YAML in every difficulty subdirectory.
    """
    scenarios_root = REPO_ROOT / "scenarios"
    seen: list[str] = []
    seen_set: set[str] = set()

    def _add_yaml(path: Path) -> None:
        try:
            rel = str(path.relative_to(scenarios_root))
        except ValueError:
            return
        if rel.endswith(".yaml"):
            rel = rel[:-5]
        if rel not in seen_set:
            seen.append(rel)
            seen_set.add(rel)

    for pat in patterns:
        clean = pat[:-5] if pat.endswith(".yaml") else pat
        # First try as a literal yaml file under scenarios/
        literal_matches = glob.glob(str(scenarios_root / clean) + ".yaml")
        if not literal_matches:
            # P3 archive fallback: pre-v0.48 release YAMLs were moved to
            # scenarios/releases/archive/. Try the archive-relative path;
            # if found, add the ORIGINAL (non-archive) slug so downstream
            # consumers (load_scenario_yaml) resolve it via their own
            # archive fallback. The pattern itself is always the canonical
            # release-scoped slug.
            if clean.startswith("releases/") and not clean.startswith("releases/archive/"):
                archive_pattern = str(scenarios_root / "releases" / "archive") + "/" + clean[len("releases/"):] + ".yaml"
                if glob.glob(archive_pattern):
                    _add_yaml(Path(scenarios_root / f"{clean}.yaml"))
                    continue
            # Fallback: try via the resolver to validate the slug is real
            try:
                from run import load_scenario_yaml  # noqa: PLC0415
                load_scenario_yaml(clean)
                _add_yaml(Path(scenarios_root / f"{clean}.yaml"))
                continue
            except Exception:  # noqa: BLE001
                pass
        if literal_matches:
            for m in literal_matches:
                _add_yaml(Path(m))
            continue
        # Then try as a directory or directory-glob — recurse for *.yaml
        dir_matches = glob.glob(str(scenarios_root / clean))
        if not dir_matches:
            continue
        for m in dir_matches:
            p = Path(m)
            if p.is_dir():
                for y in sorted(p.rglob("*.yaml")):
                    _add_yaml(y)
            elif p.suffix == ".yaml":
                _add_yaml(p)
    return sorted(seen)


def run_one_safe(args: tuple[str, str, int, dict[str, Any]] | tuple) -> dict[str, Any]:
    """Run one episode; optional 5th element is ``run_options`` dict."""
    # Lazy import to avoid a static cycle: run.py imports runner.episode, and
    # runner.batch imports run only for ``load_scenario_yaml`` (a pure
    # function over SCENARIOS_ROOT). Lazy import keeps the module-level
    # graph acyclic and preserves behavior exactly.
    from run import load_scenario_yaml

    run_options: dict[str, Any] = {}
    if len(args) == 5:
        scenario_slug, agent_name, seed, agent_kwargs, run_options = args
    else:
        scenario_slug, agent_name, seed, agent_kwargs = args  # type: ignore[misc]
    try:
        scenario = load_scenario_yaml(scenario_slug)
        traj_dir = run_options.get("trajectory_dir")
        log_path = run_options.get("episode_log_path")
        if log_path is not None:
            with _episode_file_logging(Path(str(log_path))):
                result = run_one(
                    scenario=scenario,
                    agent_name=agent_name,
                    agent_kwargs=agent_kwargs,
                    seed_override=seed,
                    trajectory_dir=Path(str(traj_dir)) if traj_dir else None,
                    per_action_attribution=bool(
                        run_options.get("per_action_attribution", False)
                    ),
                    per_action_cap=run_options.get("per_action_cap", 20),
                    per_action_group_attribution=bool(
                        run_options.get("per_action_group_attribution", False)
                    ),
                    per_action_group_cap=run_options.get(
                        "per_action_group_cap", 20
                    ),
                )
        else:
            result = run_one(
                scenario=scenario,
                agent_name=agent_name,
                agent_kwargs=agent_kwargs,
                seed_override=seed,
                trajectory_dir=Path(str(traj_dir)) if traj_dir else None,
                per_action_attribution=bool(
                    run_options.get("per_action_attribution", False)
                ),
                per_action_cap=run_options.get("per_action_cap", 20),
                per_action_group_attribution=bool(
                    run_options.get("per_action_group_attribution", False)
                ),
                per_action_group_cap=run_options.get(
                    "per_action_group_cap", 20
                ),
            )
        result["status"] = "ok"
        if log_path is not None:
            result["episode_log_path"] = str(log_path)
        return result
    except Exception as exc:
        from baselines.llm_agent import redact_provider_error  # noqa: PLC0415

        public_error = redact_provider_error(exc)
        stack = "".join(traceback.format_list(traceback.extract_tb(exc.__traceback__)))
        LOGGER.error(
            "episode failed scenario=%s agent=%s seed=%d: %s\n%s",
            scenario_slug,
            agent_name,
            seed,
            public_error,
            stack,
        )
        if log_path is not None:
            _append_sanitized_failure_trace(
                Path(str(log_path)),
                public_error=public_error,
                error=exc,
            )
        details = getattr(exc, "episode_error_details", None)
        if not isinstance(details, dict):
            details = {
                "error_type": type(exc).__name__,
                "error_stage": "episode",
            }
        result = {
            "status": "error",
            "scenario_slug": scenario_slug,
            "agent_name": agent_name,
            "seed": seed,
            "error": f"{type(exc).__name__}: {public_error}",
            **details,
        }
        if log_path is not None:
            result["episode_log_path"] = str(log_path)
        return result
