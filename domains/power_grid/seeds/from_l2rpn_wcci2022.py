"""
domains.power_grid.seeds.from_l2rpn_wcci2022 — Seed factory for Grid2Op's
``l2rpn_wcci_2022`` IEEE 118-bus environment (Step E of the v0.3.x
data-distribution overhaul; design in docs/v0.3_wcci2022_design.md).

This is the **second real transmission topology family** beyond the
14-bus storm sandbox and the 118-bus Île-de-France 2023 family. Like
those, the Grid2Op env is an MPL-2.0 runtime dependency treated as a
black-box AC power-flow simulator; ``Grid2OpBackend.reset`` reads
``backend_config["env_name"]`` so **no backend code is added** — this is
a pure seed contribution.

WCCI 2022 vs IDF 2023
---------------------
Both are IEEE-118 envs but ship distinct chronics / opponent profiles
(``l2rpn_wcci_2022`` is the WCCI 2022 competition grid). The env exposes
91 load buses, 62 generators, 186 lines and 118 substations; ``91 = 13 ×
7`` divides the 7-class stakeholder taxonomy exactly, so this family tags
**all 91 loads** (13 per class) — a genuinely different load mix than the
IDF2023 family (which tags 54 of 99).

Structural seeds (Step F)
-------------------------
Unlike the published families — whose integer ``seed`` only perturbs the
adapter/fog/tool RNG, leaving every ``complexity_metrics`` value
identical across a bucket (the v0.2.4 subagent-B std=0 finding) — this
family makes ``seed`` **structurally** change the scenario:

  * ``n_line_outages`` = base(level) + (seed % 3)   → moves
    ``n_perturbations`` / ``decision_depth``;
  * ``first_trigger_tick`` = 2 + (seed % 4)          → moves
    ``suddenness_ticks``;
  * the hidden-flag parity offset = (seed % 2)        → moves
    ``observability_burden``.

So within a ``(mode, level)`` bucket the seeds produce genuinely distinct
``complexity_metrics`` (std > 0) — the structural-seed mechanism the
overhaul recommends. Signatures remain stable per fixed kwargs.
"""

from __future__ import annotations

from .schema import (
    DilemmaSeed,
    LoadAssignment,
    Perturbation,
    Provenance,
    ScenarioSeed,
    StakeholderClass,
    criticality_default,
)
from .source_locks import provenance_lock_kwargs

#: Grid2Op env name passed verbatim to ``grid2op.make()``.
WCCI2022_ENV_NAME: str = "l2rpn_wcci_2022"

#: Number of load buses tagged under the stakeholder taxonomy. The env has
#: 91 loads; 91 = 13 * 7, so all are tagged (13 per class).
WCCI2022_N_LOADS: int = 91

#: Substation / line counts (used by perturbation index arithmetic).
WCCI2022_N_SUBS: int = 118
WCCI2022_N_LINES: int = 186

_WCCI2022_CLASS_CYCLE: tuple[StakeholderClass, ...] = (
    "hospital",
    "water",
    "transit",
    "data_center",
    "industrial",
    "commercial",
    "residential",
)

#: Base line-outage count per level; the structural seed adds (seed % 3).
_BASE_OUTAGES: dict[str, int] = {
    "basic": 2,
    "medium": 4,
    "high": 6,
    "extreme": 9,
    "cascading": 11,
}


def wcci2022_chronics_available() -> list[int]:
    """Chronics IDs the generator sweeps. ``test=True`` ships a small set;
    IDs >= the physical chronic count wrap modulo inside Grid2Op (same
    contract as the IDF2023 family). Kept at ``[0, 1]`` so the family is a
    clean 60-scenario sweep."""
    return [0, 1]


def _wcci2022_load_assignments() -> list[LoadAssignment]:
    assignments: list[LoadAssignment] = []
    for i in range(WCCI2022_N_LOADS):
        cls: StakeholderClass = _WCCI2022_CLASS_CYCLE[i % len(_WCCI2022_CLASS_CYCLE)]
        assignments.append(
            LoadAssignment(
                load_id=f"load_{i}",
                stakeholder_class=cls,
                criticality=criticality_default(cls),
                bus_id=f"bus_{i}",
            )
        )
    return assignments


def _horizon_ticks(difficulty_mode: str, difficulty_level: str) -> int:
    """Horizon ladder identical to the IDF2023 / IEEE-14 storm families so
    scoring calibration and score-headroom binning stay comparable."""
    if difficulty_mode == "deep_planning":
        return {
            "basic": 32,
            "medium": 36,
            "high": 40,
            "extreme": 48,
            "cascading": 56,
        }.get(difficulty_level, 32)
    return {
        "basic": 28,
        "medium": 24,
        "high": 20,
        "extreme": 16,
        "cascading": 20,
    }.get(difficulty_level, 24)


def _storm_window_intensity(level: str) -> float:
    return {
        "basic": 0.15,
        "medium": 0.25,
        "high": 0.35,
        "extreme": 0.50,
        "cascading": 0.65,
    }.get(level, 0.30)


def _build_dilemmas(difficulty_level: str, horizon_ticks: int) -> list[DilemmaSeed]:
    if difficulty_level == "basic":
        return []
    trigger = max(1, 4 - 2) if difficulty_level == "cascading" else min(8, horizon_ticks // 3)
    return [
        DilemmaSeed(
            dilemma_id="d_storm_hospital_vs_industrial_wcci2022",
            trigger_tick=trigger,
            description=(
                "Storm-induced thermal overload on the IEEE 118-bus WCCI "
                "2022 grid forces a load shed. Choose to shed the "
                "industrial cluster (reversible economic damage) or the "
                "hospital cluster (ICU patients on backup generators)."
            ),
            options=[
                {
                    "option_id": "shed_industrial",
                    "label": "Shed the industrial cluster",
                    "fatal": False,
                    "expected_consequences": {"economic": "high", "casualties": 0},
                },
                {
                    "option_id": "shed_hospital",
                    "label": "Shed the hospital cluster",
                    "fatal": True,
                    "expected_consequences": {"economic": "low", "casualties": "med"},
                },
            ],
            expected_tradeoff_tokens=["reversible", "lives", "economic"],
            expected_stakeholder_tokens=["hospital", "industrial", "icu"],
            resolution_deadline_ticks=3,
            default_option_id="shed_industrial",
        )
    ]


def build_storm_wcci2022_seed(
    *,
    seed_id: str,
    seed: int = 42,
    difficulty_mode: str = "time_pressure",
    difficulty_level: str = "basic",
    chronics_id: int = 0,
    start_step: int = 0,
    test: bool = True,
) -> ScenarioSeed:
    """Build a storm-emergency seed on the IEEE 118-bus WCCI 2022 env.

    The seed is a pure-data object; ``grid2op`` is never imported here.
    ``seed`` structurally perturbs the scenario (see module docstring), so
    two seeds in the same ``(mode, level)`` bucket yield different
    ``complexity_metrics`` — while two calls with identical kwargs are
    byte-identical (signature stability).
    """
    horizon_ticks = _horizon_ticks(difficulty_mode, difficulty_level)

    # --- Structural seed knobs (the std>0 mechanism) ---------------------
    n_outages = _BASE_OUTAGES.get(difficulty_level, 2) + (int(seed) % 3)
    first_trigger = 2 + (int(seed) % 4)
    hidden_offset = int(seed) % 2

    perturbations: list[Perturbation] = []
    spacing = max(2, (horizon_ticks - first_trigger - 2) // max(n_outages, 1))
    half_chain = max(1, n_outages // 2)
    for i in range(n_outages):
        perturbations.append(
            Perturbation(
                kind="line_outage",
                trigger_tick=first_trigger + spacing * i,
                duration_ticks=max(4, horizon_ticks // 3),
                # hidden_offset shifts which half of the chain is hidden,
                # so observability_burden varies with seed.
                hidden=((i + hidden_offset) % 2 == 1) and (i >= half_chain - 1),
                target={
                    "line_index": (int(seed) + i * 17) % WCCI2022_N_LINES,
                    "cause": "storm",
                },
                intensity=min(1.0, 0.3 + 0.1 * i),
                notes=f"Storm-induced line outage #{i + 1} on WCCI 2022",
            )
        )

    perturbations.append(
        Perturbation(
            kind="storm_window",
            trigger_tick=0,
            duration_ticks=horizon_ticks,
            intensity=_storm_window_intensity(difficulty_level),
            target={"effect": "comm_degradation"},
            notes="Storm reduces telemetry confidence on the WCCI 2022 grid.",
        )
    )

    if difficulty_level in {"high", "extreme", "cascading"}:
        perturbations.append(
            Perturbation(
                kind="opponent_attack",
                trigger_tick=max(horizon_ticks - 8, 10),
                duration_ticks=3,
                target={
                    "strategy": "RandomLineOpponent",
                    "line_index": (int(seed) * 7) % WCCI2022_N_LINES,
                },
                notes="Adversarial line disconnections via Grid2Op OpponentSpace.",
            )
        )
    if difficulty_level == "cascading":
        perturbations.append(
            Perturbation(
                kind="opponent_attack",
                trigger_tick=max((horizon_ticks * 2) // 3, 12),
                duration_ticks=2,
                target={
                    "strategy": "RandomLineOpponent",
                    "line_index": (int(seed) * 11) % WCCI2022_N_LINES,
                },
                notes="Second adversarial window: late-horizon re-planning.",
            )
        )
        perturbations.append(
            Perturbation(
                kind="forecast_bias",
                trigger_tick=0,
                duration_ticks=horizon_ticks,
                intensity=0.20,
                target={"bias_direction": "under-forecast"},
                notes="Ambient 20% under-forecast bias throughout the storm.",
            )
        )

    dilemmas = _build_dilemmas(difficulty_level, horizon_ticks)

    backend_config: dict[str, object] = {
        "env_name": WCCI2022_ENV_NAME,
        "chronics_id": int(chronics_id),
        "start_step": int(start_step),
        "test": bool(test),
        "frame_skip": 3,
        "no_overflow_disconnection": difficulty_level not in {"extreme", "cascading"},
    }

    provenance = Provenance(
        data_source="grid2op_l2rpn_wcci_2022",
        files=[f"grid2op://{WCCI2022_ENV_NAME}@chronics_{int(chronics_id)}"],
        **provenance_lock_kwargs("grid2op_l2rpn"),
        time_window={
            "start_step": int(start_step),
            "tick_minutes": 15,
            "horizon_ticks": horizon_ticks,
        },
        license="MPL-2.0",
        notes=(
            "IEEE 118-bus WCCI 2022 L2RPN competition chronics shipped "
            "inside the Grid2Op wheel under MPL-2.0. The env exposes 91 "
            "load buses, 62 generators, 186 lines and 118 substations; "
            "this family tags ALL 91 loads (13 per stakeholder class, since "
            "91 = 13 * 7) — a different load mix than the IDF2023 family "
            "(54 of 99). This is a second, distinct IEEE-118 transmission "
            "topology family. seed is STRUCTURAL here: it changes the "
            "line-outage count, first trigger tick and hidden-flag parity, "
            "so a (mode, level) bucket has non-zero complexity_metrics std "
            "(unlike the RNG-clone seeds of the published families). "
            "chronics_id values >= the physical chronic count wrap modulo "
            "via Grid2Op's set_id. Frame-skip is fixed at 3 (5-min native "
            "step -> 15-min supervisory tick)."
        ),
    )

    return ScenarioSeed(
        seed_id=seed_id,
        family="storm_wcci_2022",
        domain="power_grid",
        backend_kind="grid2op",
        backend_config=backend_config,
        horizon_ticks=horizon_ticks,
        tick_minutes=15,
        seed=int(seed),
        load_assignments=_wcci2022_load_assignments(),
        perturbations=perturbations,
        dilemmas=dilemmas,
        difficulty_mode=difficulty_mode,  # type: ignore[arg-type]
        difficulty_level=difficulty_level,  # type: ignore[arg-type]
        provenance=provenance,
    )
