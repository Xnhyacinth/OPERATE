"""
baselines.reflexion_agent — Reflexion-style LLM agent for OPERATE.

Extends :class:`ReActLLMAgent` with **episode-level self-critique**:

- At the end of each episode (``on_episode_end``), prompts the LLM to
  reflect on what went wrong / what worked, given the full trajectory
  summary and the realized events.
- Persists reflection text to a family-scoped file in strict mode and a
  per-(family, mode, level) file only in explicitly diagnostic debug mode.
- On reset, loads the most recent ≤K lessons for that qualified scope.

Reflexion paper: Shinn et al., "Reflexion: Language Agents with Verbal
Reinforcement Learning" (NeurIPS 2023).

Memory file format (JSONL, one record per episode):

    {
      "scenario_id": "...",
      "seed": 42,
      "final_score": 12.34,        # optional / informative
      "lesson": "When wind dropout fires near peak, commit reserves at...",
      "ts_utc": "2026-05-27T12:34:56Z"
    }

Designed so that ``batch_eval.py --agents reflexion_llm`` automatically
accumulates lessons across the batch run without external orchestration.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core import Action

from .llm_agent import LLMConfig, _event_prompt_view, redact_provider_error
from .react_agent import ReActLLMAgent

LOGGER = logging.getLogger(__name__)

REFLEXION_DIR_ENV = "OPERATE_REFLEXION_DIR"
DEFAULT_REFLEXION_DIR = Path("reflexion_memory")
MAX_LESSONS_PER_PROMPT = 5


def _memory_path(
    family: str,
    mode: str,
    level: str,
    *,
    prompt_mode: str = "debug",
) -> Path:
    base = Path(os.environ.get(REFLEXION_DIR_ENV, DEFAULT_REFLEXION_DIR))
    if prompt_mode == "strict":
        return base / f"{family}__strict.jsonl"
    return base / f"{family}__{mode}__{level}.jsonl"


def _load_lessons(
    family: str,
    mode: str,
    level: str,
    *,
    prompt_mode: str = "debug",
) -> list[str]:
    p = _memory_path(family, mode, level, prompt_mode=prompt_mode)
    if not p.exists():
        return []
    lessons: list[str] = []
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    lesson = str(obj.get("lesson", "")).strip()
                    if lesson:
                        lessons.append(lesson)
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return lessons[-MAX_LESSONS_PER_PROMPT:]


def _append_lesson(
    family: str,
    mode: str,
    level: str,
    scenario_id: str,
    seed: int,
    lesson: str,
    *,
    prompt_mode: str = "debug",
) -> None:
    p = _memory_path(family, mode, level, prompt_mode=prompt_mode)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "scenario_id": scenario_id,
                        "seed": seed,
                        "lesson": lesson,
                        "ts_utc": datetime.now(UTC).isoformat(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    except OSError:
        pass  # silent — reflexion is best-effort


class ReflexionLLMAgent(ReActLLMAgent):
    """ReAct + episode-end self-critique + persistent lessons memory."""

    name = "reflexion_llm"
    REFLECTION_MAX_TOKENS = 400

    def __init__(self, config: LLMConfig | None = None) -> None:
        super().__init__(config=config)
        self._family = "unknown"
        self._mode = "unknown"
        self._level = "unknown"
        self._scenario_id = "unknown"
        self._lessons_loaded: list[str] = []

    def lessons_fingerprint(self) -> dict[str, Any]:
        """Deterministic fingerprint of the qualified lessons scope.

        Strict runs are family-scoped so hidden mode/level labels cannot select
        or annotate model-visible memory. Debug runs retain the legacy
        per-(family, mode, level) scope. The file is captured at episode start so
        the trajectory log
        records WHICH lessons were available to influence this episode.

        Reflexion lessons persist across episodes and ordering of
        episodes therefore affects scores. Without this fingerprint a
        leaderboard run is ambiguous: two runs with identical seeds can
        produce different scores depending on the lessons file at
        episode start.

        Returns
        -------
        dict
            ``path`` — absolute path to the lessons file (whether or
            not it exists).
            ``sha256`` — hex digest of the file contents, or ``None``
            if the file is missing.
            ``n_lessons`` — number of well-formed JSONL records
            currently present (0 if missing).
            ``tail_ids`` — list of up to the 5 most-recent lesson IDs
            (deterministically derived from each record's
            ``scenario_id`` + ``ts_utc``); empty if missing.
        """
        prompt_mode = str(self.config.prompt_mode or "strict").lower()
        path = _memory_path(
            self._family,
            self._mode,
            self._level,
            prompt_mode=prompt_mode,
        )
        info: dict[str, Any] = {
            "path": str(path),
            "sha256": None,
            "n_lessons": 0,
            "tail_ids": [],
        }
        if not path.exists():
            return info
        try:
            data = path.read_bytes()
        except OSError:
            return info
        info["sha256"] = hashlib.sha256(data).hexdigest()
        records: list[dict[str, Any]] = []
        for line in data.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        info["n_lessons"] = len(records)
        tail = records[-5:]
        ids: list[str] = []
        for rec in tail:
            seed = f"{rec.get('scenario_id', '')}|{rec.get('ts_utc', '')}"
            ids.append("ln_" + hashlib.sha1(seed.encode()).hexdigest()[:10])
        info["tail_ids"] = ids
        return info

    def reset(self, env, scenario_config, seed: int) -> None:  # type: ignore[override]
        super().reset(env, scenario_config, seed)
        self._family = str(scenario_config.get("family", "unknown"))
        self._mode = str(scenario_config.get("difficulty_mode", "unknown"))
        self._level = str(scenario_config.get("difficulty_level", "unknown"))
        self._scenario_id = str(scenario_config.get("seed_id", "unknown"))
        # Inject past lessons into the system prompt.
        prompt_mode = str(self.config.prompt_mode or "strict").lower()
        self._lessons_loaded = _load_lessons(
            self._family,
            self._mode,
            self._level,
            prompt_mode=prompt_mode,
        )
        if self._lessons_loaded:
            lesson_block = "\n".join(f"  - {lesson}" for lesson in self._lessons_loaded)
            scope = (
                f"family {self._family}"
                if prompt_mode == "strict"
                else f"{self._family} / {self._mode} / {self._level}"
            )
            self._system_prompt = (
                self._system_prompt + "\n\nLessons from previous episodes of "
                f"the same {scope}:\n"
                + lesson_block
                + "\n\nApply these lessons proactively where relevant. "
                "Do not over-commit to a single lesson if the current "
                "observation contradicts it."
            )

    def on_episode_end(
        self,
        final_observation: dict[str, Any],
        actions: list[Action],
        episode_reward: float = 0.0,
    ) -> None:
        """Ask the LLM to produce a 1-3-sentence lesson, then append it
        to the per-cell memory file."""
        if not self._has_api_key:
            return  # nothing to reflect with
        try:
            lesson = self._produce_lesson(
                actions, final_observation, episode_reward=episode_reward
            )
            if lesson:
                _append_lesson(
                    self._family,
                    self._mode,
                    self._level,
                    self._scenario_id,
                    seed=0,  # seed not surfaced post-reset; use 0
                    lesson=lesson,
                    prompt_mode=str(self.config.prompt_mode or "strict").lower(),
                )
        except Exception as exc:
            # Never fail the episode because of a bad reflection.
            self._stats["reflection_failed"] = (
                int(self._stats.get("reflection_failed", 0)) + 1
            )
            LOGGER.warning(
                "Reflexion reflection failed for (%s/%s/%s) scenario=%s: %s: %s",
                self._family,
                self._mode,
                self._level,
                self._scenario_id,
                type(exc).__name__,
                redact_provider_error(exc),
            )
            return

    def _produce_lesson(
        self,
        actions: list[Action],
        final_observation: dict[str, Any],
        episode_reward: float = 0.0,
    ) -> str:
        # Build a concise summary of the trajectory.
        tool_hist: dict[str, int] = {}
        for a in actions:
            for c in a.tool_calls:
                tool_hist[c.name] = tool_hist.get(c.name, 0) + 1
        summary = {
            "family": self._family,
            "n_ticks": len(actions),
            "tool_histogram": tool_hist,
            "scratchpad_tail": self._memory.scratchpad[-6:],
            "recent_tool_results": self._memory.recent_results[-6:],
            "recent_evidence_ids": self._memory.recent_evidence_ids[-6:],
            "realized_events": [
                _event_prompt_view(event)
                for event in self._memory.realized_events[-10:]
            ],
            "final_totals": (final_observation or {}).get("totals", {}),
            # v0.2.4: surface the accumulated per-tick reward to the
            # critic so the lesson is calibrated on outcome quality,
            # not just trajectory structure. F-03 fixed the
            # accumulation in run.py but the lesson summary still
            # discarded the signal.
            "episode_reward_total": round(float(episode_reward), 4),
        }
        prompt_mode = str(self.config.prompt_mode or "strict").lower()
        if prompt_mode != "strict":
            summary["scenario_id"] = self._scenario_id
            summary["mode"] = self._mode
            summary["level"] = self._level
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a post-episode Reflexion critic for a "
                    "real-time scheduling LLM agent. Given the "
                    "trajectory summary, output ONE short lesson "
                    "(1-3 sentences) the agent should remember for "
                    "future episodes of the same visible task family. "
                    "Focus on concrete causal patterns (e.g. 'wind "
                    "dropout at tick T precedes voltage dip — pre-commit "
                    "reserves at tick T-2'). Avoid generic advice. "
                    "Output ONLY the lesson text, no preamble."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(summary, ensure_ascii=False)[:6000],
            },
        ]
        started_ns = time.monotonic_ns()
        request_sequence = self._record_provider_request(
            messages=messages,
            tools=[],
            fallback_without_tools=False,
            max_tokens=self.REFLECTION_MAX_TOKENS,
            request_kind="post_episode_reflection",
        )
        try:
            lesson = self._request_reflection(messages)
        except Exception as exc:
            self._record_provider_action_response(
                None,
                started_ns=started_ns,
                error=exc,
                request_sequence=request_sequence,
            )
            raise
        self._record_provider_action_response(
            Action(tool_calls=[], dominant="reflection", assistant_text=lesson),
            started_ns=started_ns,
            request_sequence=request_sequence,
        )
        return lesson

    def _request_reflection(self, messages: list[dict[str, str]]) -> str:
        if self.config.provider in {"openai", "openai_compatible", "azure"}:
            if self._resolved_api_mode_public() == "responses":
                rsp = self._client.responses.create(  # type: ignore[union-attr]
                    model=self.config.model,
                    instructions=messages[0]["content"],
                    input=[messages[1]],
                    temperature=self.config.temperature,
                    max_output_tokens=self.REFLECTION_MAX_TOKENS,
                    timeout=self.config.timeout_s,
                    store=False,
                )
                self._record_provider_response_identity(rsp)
                return (getattr(rsp, "output_text", "") or "").strip()
            rsp = self._client.chat.completions.create(  # type: ignore[union-attr]
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.REFLECTION_MAX_TOKENS,
                timeout=self.config.timeout_s,
            )
            self._record_provider_response_identity(rsp)
            return (rsp.choices[0].message.content or "").strip()
        if self.config.provider == "anthropic":
            rsp = self._client.messages.create(  # type: ignore[union-attr]
                model=self.config.model,
                system=messages[0]["content"],
                messages=messages[1:],
                temperature=self.config.temperature,
                max_tokens=self.REFLECTION_MAX_TOKENS,
                timeout=self.config.timeout_s,
            )
            self._record_provider_response_identity(rsp)
            for block in rsp.content:
                if getattr(block, "type", None) == "text":
                    return getattr(block, "text", "").strip()
            return ""
        if self.config.provider == "google":
            rsp = self._client.models.generate_content(  # type: ignore[union-attr]
                model=self.config.model,
                contents=messages[-1]["content"],
                config={
                    "system_instruction": messages[0]["content"],
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.REFLECTION_MAX_TOKENS,
                },
            )
            self._record_provider_response_identity(rsp)
            return (getattr(rsp, "text", "") or "").strip()
        return ""
