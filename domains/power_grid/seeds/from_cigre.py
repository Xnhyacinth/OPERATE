"""
domains.power_grid.seeds.from_cigre — CIGRE MV distribution-grid seeds.

The CIGRE European Medium-Voltage benchmark network (Strunz et al. 2009)
is a 15-bus radial feeder with all DER (PV + wind + storage + diesel)
distributed across the network. It is published as a standard
benchmark by the CIGRE Task Force C6.04 and is shipped inside
``pandapower.networks.create_cigre_network_mv(with_der="all")`` under
BSD-3-Clause.

This seed factory wraps that network into OPERATE's
``ScenarioSeed`` schema with:

- Native distribution stakeholders (residential, commercial, industrial,
  hospital, water, transit, data_center, public_lighting) mapped onto
  the 18 CIGRE load buses by a deterministic policy.
- Perturbations meaningful for distribution operations: a hidden DER
  trip, a load surge on the residential feeder, a storm window that
  reduces visibility on bus voltages, and a line outage on a feeder.
- Two dilemmas tuned to the feeder context (Volt-Var vs DER curtailment;
  recoverable vs sustained shed).
"""

from __future__ import annotations

from typing import Any

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

# Stakeholder assignment for the 18 CIGRE MV loads. The CIGRE benchmark
# network lumps loads by bus, not by sector, so we assign a realistic mix
# inspired by the European MV residential / commercial / industrial split
# documented in CIGRE TF C6.04 §5. Mapping is deterministic so seed
# signatures stay stable across runs.
_CIGRE_LOAD_ASSIGNMENT: list[tuple[int, StakeholderClass]] = [
    (0, "residential"),
    (1, "residential"),
    (2, "commercial"),
    (3, "hospital"),
    (4, "industrial"),
    (5, "residential"),
    (6, "residential"),
    (7, "commercial"),
    (8, "industrial"),
    (9, "data_center"),
    (10, "residential"),
    (11, "residential"),
    (12, "transit"),
    (13, "water"),
    (14, "industrial"),
    (15, "commercial"),
    (16, "residential"),
    (17, "residential"),
]


def _cigre_load_assignments() -> list[LoadAssignment]:
    return [
        LoadAssignment(
            load_id=f"mv_load_{i}",
            stakeholder_class=cls,
            criticality=criticality_default(cls),
            bus_id=f"mv_bus_{i}",
        )
        for i, cls in _CIGRE_LOAD_ASSIGNMENT
    ]


# v0.4 Bucket B (D1): criticality-ordered stakeholder rotation for
# topologies that have no hand-curated map. The largest loads (by base
# p_mw) get the most critical classes so shed/equity scoring is
# physically sensible; the long tail is residential/commercial — a
# realistic distribution pyramid. Deterministic in (network, load order)
# so seed signatures stay stable.
_CRITICAL_PRIORITY: list[StakeholderClass] = [
    "hospital",
    "water",
    "data_center",
    "industrial",
    "transit",
]


def _generic_load_assignments(network: str, prefix: str) -> list[LoadAssignment]:
    import pandapower.networks as pn  # gen-time only

    if network == "mv_oberrhein":
        net = pn.mv_oberrhein()
    elif network == "synthetic_volt_control_lv":
        net = pn.create_synthetic_voltage_control_lv_network()
    elif network.startswith("simbench:"):
        import simbench as sb  # type: ignore[import-untyped]

        net = sb.get_simbench_net(network.split("simbench:", 1)[1])
    else:
        net = pn.create_cigre_network_mv(with_der="all")
    n = len(net.load)
    p_mw = [float(x) for x in net.load.p_mw.tolist()]
    bus = [int(b) for b in net.load.bus.tolist()]
    # Rank load indices by descending base demand (deterministic tie-break
    # on index) and assign the top ranks the critical classes.
    order = sorted(range(n), key=lambda i: (-p_mw[i], i))
    cls_by_idx: dict[int, StakeholderClass] = {}
    for rank, idx in enumerate(order):
        if rank < len(_CRITICAL_PRIORITY):
            cls_by_idx[idx] = _CRITICAL_PRIORITY[rank]
        else:
            cls_by_idx[idx] = "commercial" if rank % 3 == 0 else "residential"
    return [
        LoadAssignment(
            load_id=f"{prefix}_load_{i}",
            stakeholder_class=cls_by_idx[i],
            criticality=criticality_default(cls_by_idx[i]),
            bus_id=f"{prefix}_bus_{bus[i]}",
        )
        for i in range(n)
    ]


# Per-network provenance + family metadata for the v0.4 distribution
# expansion. ``family`` is the registry family name; ``prefix`` namespaces
# the synthetic load/bus ids.
_DISTRIBUTION_NETWORKS: dict[str, dict[str, str]] = {
    "cigre_mv_with_der_all": {
        "family": "distribution_volt_var",
        "prefix": "mv",
    },
    "mv_oberrhein": {
        "family": "distribution_volt_var_oberrhein",
        "prefix": "ob",
        "uri_scheme": "pandapower-mv-oberrhein",
        "constructor": "mv_oberrhein()",
        "desc": (
            "pandapower mv_oberrhein real German MV distribution feeder "
            "(~179 buses, 147 loads, 153 PV sgen, 2 HV/MV trafos, "
            "normally-open tie switches)."
        ),
    },
    "synthetic_volt_control_lv": {
        "family": "distribution_volt_var_lv",
        "prefix": "lv",
        "uri_scheme": "pandapower-synthetic-lv",
        "constructor": "create_synthetic_voltage_control_lv_network()",
        "desc": (
            "pandapower balanced synthetic Volt-Var LV feeder "
            "(~26 buses, 14 loads, 5 DER sgen, 1 MV/LV trafo)."
        ),
    },
    "simbench:1-LV-rural1--0-sw": {
        "family": "simbench_lv_timeseries_control",
        "prefix": "sblv",
        "desc": "SimBench rural LV network with source-bundled annual profiles.",
    },
    "simbench:1-MV-rural--0-sw": {
        "family": "simbench_mv_rural_timeseries_control",
        "prefix": "sbmvr",
        "desc": "SimBench rural MV network with source-bundled annual profiles.",
    },
    "simbench:1-MV-rural--1-sw": {
        "family": "simbench_mv_rural1_timeseries_control",
        "prefix": "sbmvr1",
        "desc": "SimBench rural MV variant 1 with source-bundled annual profiles.",
    },
    "simbench:1-MV-semiurb--0-sw": {
        "family": "simbench_mv_semiurban_timeseries_control",
        "prefix": "sbmvs",
        "desc": "SimBench semi-urban MV network with source-bundled annual profiles.",
    },
    "simbench:1-MV-semiurb--1-sw": {
        "family": "simbench_mv_semiurban1_timeseries_control",
        "prefix": "sbmvs1",
        "desc": "SimBench semi-urban MV variant 1 with source-bundled annual profiles.",
    },
    "simbench:1-MV-urban--0-sw": {
        "family": "simbench_mv_urban_timeseries_control",
        "prefix": "sbmvu",
        "desc": "SimBench urban MV network with source-bundled annual profiles.",
    },
    "simbench:1-MV-comm--0-sw": {
        "family": "simbench_mv_commercial_timeseries_control",
        "prefix": "sbmvc",
        "desc": "SimBench commercial MV network with source-bundled annual profiles.",
    },
}


def build_distribution_volt_var_seed(
    *,
    seed_id: str,
    seed: int = 42,
    difficulty_mode: str = "time_pressure",
    difficulty_level: str = "basic",
    network: str = "cigre_mv_with_der_all",
    profile_start_index: int = 0,
) -> ScenarioSeed:
    """Build a distribution_volt_var scenario on CIGRE MV.

    Difficulty ladder (5 tiers in v0.1.2):

    - ``basic`` — diurnal load + one DER trip mid-horizon.
    - ``medium`` — adds a hidden storm window with elevated obs noise.
    - ``high`` — adds a residential surge + forecast bias.
    - ``extreme`` — adds an opponent-style line outage and triggers
      the Volt-Var dilemma.
    - ``cascading`` — extreme + early hidden wind dropout + a second
      forced DER outage so the agent must coordinate curtail + shed.
    """
    # CIGRE distribution horizons grow monotonically with level on BOTH
    # modes — distribution episodes accumulate cost slowly and a shrinking
    # horizon would invert the difficulty ladder (regression caught by
    # audit difficulty_invariant). Mode controls dilemma deadline +
    # forecast bias arrival.
    if difficulty_mode == "time_pressure":
        # v0.1.3: bumped medium 16→20 so oracle has more ticks for the
        # voltage-shed branch to demonstrate cost advantage (audit was
        # flagging medium-tp as zero-differentiation between baselines).
        horizon_ticks = {
            "basic": 12,
            "medium": 20,
            "high": 24,
            "extreme": 28,
            "cascading": 32,
        }.get(difficulty_level, 12)
        dilemma_deadline = 1
    else:  # deep_planning
        horizon_ticks = {
            "basic": 18,
            "medium": 24,
            "high": 30,
            "extreme": 36,
            "cascading": 42,
        }.get(difficulty_level, 18)
        dilemma_deadline = 3

    perturbations: list[Perturbation] = [
        Perturbation(
            kind="generator_forced_outage",
            trigger_tick=max(2, horizon_ticks // 3),
            duration_ticks=max(4, horizon_ticks // 3),
            target={"generator_kind": "DER", "index": 0},
            notes="DER (diesel-CHP) trips mid-horizon.",
        )
    ]
    if difficulty_level in {"medium", "high", "extreme", "cascading"}:
        perturbations.append(
            Perturbation(
                kind="storm_window",
                trigger_tick=0,
                duration_ticks=horizon_ticks,
                intensity={
                    "medium": 0.20,
                    "high": 0.30,
                    "extreme": 0.40,
                    "cascading": 0.55,
                }.get(difficulty_level, 0.2),
                target={"effect": "comm_degradation"},
                notes="Storm reduces feeder telemetry confidence.",
            )
        )
        # Even at medium we inject a small physical residential surge so
        # the level introduces a real stressor (not just telemetry
        # noise). This is what makes oracle/greedy beat wait_only — the
        # audit `baseline_gap` check requires non-zero differentiation.
        surge_intensity = {
            "medium": 0.07,
            "high": 0.12,
            "extreme": 0.16,
            "cascading": 0.20,
        }.get(difficulty_level, 0.07)
        perturbations.append(
            Perturbation(
                kind="load_surge",
                trigger_tick=max(4, horizon_ticks // 2),
                duration_ticks=3,
                intensity=surge_intensity,
                target={"stakeholder_class": "residential"},
                notes=(
                    f"Residential evening-peak surge — intensity {surge_intensity:.2f}"
                ),
            )
        )
        perturbations.append(
            Perturbation(
                kind="forecast_bias",
                trigger_tick=0,
                duration_ticks=horizon_ticks,
                intensity=0.10,
                target={"bias_direction": "under-forecast"},
                notes="Forecast under-forecasts by 10%.",
            )
        )
    if difficulty_level in {"extreme", "cascading"}:
        # Distribution analog of an opponent attack: a SECOND DER trip
        # near the load peak so the agent must coordinate curtail + shed.
        # We deliberately avoid `line_outage` on CIGRE radial topology —
        # tripping a line disconnects downstream load and *reduces* cost,
        # which would invert the difficulty ladder (caught by audit).
        perturbations.append(
            Perturbation(
                kind="generator_forced_outage",
                trigger_tick=max(3, horizon_ticks // 2),
                duration_ticks=max(4, horizon_ticks // 3),
                target={"generator_kind": "DER", "index": 2},
                notes="Second DER trip near peak — voltage support drops.",
            )
        )
    if difficulty_level == "cascading":
        perturbations.append(
            Perturbation(
                kind="wind_dropout",
                trigger_tick=1,
                duration_ticks=max(6, horizon_ticks // 2),
                intensity=0.55,
                target={"generator_kind": "wind"},
                hidden=True,
                notes="Hidden wind dropout — DER deficit before chronic peak.",
            )
        )
        perturbations.append(
            Perturbation(
                kind="generator_forced_outage",
                trigger_tick=max(4, (3 * horizon_ticks) // 4),
                duration_ticks=4,
                target={"generator_kind": "DER", "index": 4},
                hidden=True,
                notes="Second hidden DER trip near end of horizon.",
            )
        )

    dilemmas: list[DilemmaSeed] = []
    if difficulty_level in {"medium", "high", "extreme", "cascading"}:
        d_trigger = (
            max(0, horizon_ticks - 4)
            if difficulty_mode == "time_pressure"
            else max(0, horizon_ticks // 2)
        )
        dilemmas.append(
            DilemmaSeed(
                dilemma_id="d_distribution_voltvar",
                trigger_tick=d_trigger,
                description=(
                    "Voltage on the downstream feeder is falling toward "
                    "0.93 pu. Choose: curtail the PV cluster (reduces "
                    "feeder loading; at constant power factor both P and "
                    "Q decrease, relieving the bus), shed the residential "
                    "feeder (interrupts ~600 homes for the duration of "
                    "the storm), or do nothing and let line ferromagnetic "
                    "resonance set in."
                ),
                options=[
                    {
                        "option_id": "curtail_pv",
                        "label": "Curtail PV cluster",
                        "fatal": False,
                        "expected_consequences": {
                            "economic": "med",
                            "casualties": 0,
                            "recovery_hours": 1,
                        },
                    },
                    {
                        "option_id": "shed_residential",
                        "label": "Shed residential feeder",
                        "fatal": False,
                        "expected_consequences": {
                            "economic": "low",
                            "casualties": 0,
                            "recovery_hours": 1,
                        },
                    },
                    {
                        "option_id": "shed_hospital_feeder",
                        "label": "Shed hospital + clinic feeder",
                        "fatal": True,
                        # DC-3: hospital VoLL is $5000/MWh (the highest
                        # tariff class). Labelling it "economic: low" was
                        # misleading for LLM agents.
                        "expected_consequences": {
                            "economic": "catastrophic",
                            "casualties": "med",
                        },
                    },
                ],
                expected_tradeoff_tokens=[
                    "voltage",
                    "reactive",
                    "curtail",
                    "feeder",
                    "reversible",
                ],
                expected_stakeholder_tokens=[
                    "residential",
                    "hospital",
                    "pv",
                    "der",
                ],
                resolution_deadline_ticks=dilemma_deadline,
                default_option_id="curtail_pv",
            )
        )

    # v0.4 Bucket B (D1): the CIGRE-calibrated stressors above (one DER
    # trip + a ~7-20% residential surge) are negligible on a 179-bus real
    # feeder — they leave voltages comfortably inside the band, so the
    # difficulty ladder would not discriminate (oracle ~= wait). For the
    # larger v0.4 topologies we add a STRONG coordinated (all-class)
    # evening-peak surge whose intensity scales with level, calibrated so
    # higher tiers actually drive buses toward / past the 0.95 pu limit
    # (verified on mv_oberrhein: ~+70% reaches violations). CIGRE is left
    # byte-identical. Radial line_outage is still deliberately avoided.
    if network != "cigre_mv_with_der_all":
        coord_intensity = {
            "basic": 0.30,
            "medium": 0.45,
            "high": 0.58,
            "extreme": 0.70,
            "cascading": 0.82,
        }.get(difficulty_level, 0.30)
        perturbations.append(
            Perturbation(
                kind="load_surge",
                trigger_tick=max(3, horizon_ticks // 2),
                duration_ticks=max(3, horizon_ticks // 3),
                intensity=coord_intensity,
                target={},  # no stakeholder_class → coordinated all-feeder peak
                notes=(
                    "Coordinated all-feeder evening peak — intensity "
                    f"{coord_intensity:.2f} (large-feeder voltage stressor)."
                ),
            )
        )

    backend_config = {
        "network": network,
        "tick_minutes": 60,
    }
    if network.startswith("simbench:"):
        backend_config.update(
            {
                "profile_start_index": int(profile_start_index),
                "profile_step": 4,
                "profile_resolution_minutes": 15,
                "profile_source": "simbench_bundled_full_year",
                "source_integration_rung": "executed_with_live_backend",
                "volt_var_controls": True,
            }
        )

    # v0.2.1 fix (per architect review): the CIGRE MV network is
    # generated programmatically inside pandapower (no on-disk JSON).
    # The previous provenance.files pointed back at OPERATE's
    # own source code, which made the audit's provenance check
    # circular (self-validating). We now record the actual upstream
    # pandapower package version and the published CIGRE TF
    # C6.04 reference, and we point .files at a synthetic
    # ``pandapower-cigre-mv://`` URI plus the upstream package file.
    try:
        import pandapower as _pp

        _pp_version = getattr(_pp, "__version__", "unknown")
    except Exception:
        _pp_version = "unknown"
    _pp_rel = "pandapower/networks/cigre_networks.py"

    if network == "cigre_mv_with_der_all":
        # Byte-identical to v0.3 — preserves the 10 CIGRE structural hashes.
        lock = provenance_lock_kwargs("cigre_mv_pandapower")
        provenance = Provenance(
            data_source="pandapower_cigre_mv",
            files=[
                f"pandapower-cigre-mv://create_cigre_network_mv(with_der='all')@{_pp_version}",
            ],
            commit=lock["commit"],
            url=lock["url"],
            lock_strategy=lock["lock_strategy"],
            time_window={"hours": horizon_ticks, "tick_minutes": 60},
            license="BSD-3-Clause (pandapower); CIGRE TF C6.04 benchmark (open).",
            notes=(
                "CIGRE European MV benchmark network (15 buses, 18 loads, 13 DER, "
                "2 storage) generated by pandapower.networks."
                f"create_cigre_network_mv(with_der='all'). pandapower=={_pp_version}. "
                f"Upstream module reference: {_pp_rel}. Original publication: "
                "K. Strunz et al., 'Benchmark Systems for Network Integration "
                "of Renewable and Distributed Energy Resources', CIGRE TF "
                "C6.04, Technical Brochure 575, 2014. The OPERATE "
                "adapter (`domains/power_grid/backends/cigre_distribution.py`) "
                "does not modify the topology — it overlays chronics, "
                "perturbations, and stakeholder assignments on top of the "
                "unmodified pandapower-generated `pp.Network`."
            ),
        )
        family = "distribution_volt_var"
        load_assignments = _cigre_load_assignments()
    elif network.startswith("simbench:"):
        import simbench as _sb  # type: ignore[import-untyped]

        meta = _DISTRIBUTION_NETWORKS[network]
        code = network.split("simbench:", 1)[1]
        sb_version = getattr(_sb, "__version__", "unknown")
        lock = provenance_lock_kwargs("simbench")
        provenance = Provenance(
            data_source="simbench_official",
            files=[
                f"simbench://{code}@{sb_version}#profile_start={profile_start_index}",
            ],
            commit=lock["commit"],
            url=lock["url"],
            lock_strategy=lock["lock_strategy"],
            time_window={
                "profile_start_index": int(profile_start_index),
                "profile_step": 4,
                "horizon_ticks": horizon_ticks,
                "tick_minutes": 60,
            },
            license="BSD-3-Clause code; SimBench data ODbL/DbCL terms.",
            notes=(
                f"{meta['desc']} Network code {code}; simbench=={sb_version}. "
                "The backend consumes the bundled 15-minute load/generation "
                "profiles at a locked hourly stride and solves pandapower AC "
                "power flow at each decision tick."
            ),
        )
        family = meta["family"]
        load_assignments = _generic_load_assignments(network, meta["prefix"])
    else:
        # v0.4 Bucket B (D1): additional real distribution topologies.
        meta = _DISTRIBUTION_NETWORKS[network]
        lock = provenance_lock_kwargs("pandapower_mv_oberrhein")
        provenance = Provenance(
            data_source=f"pandapower_{network}",
            files=[
                f"{meta['uri_scheme']}://{meta['constructor']}@{_pp_version}",
            ],
            commit=lock["commit"],
            url=lock["url"],
            lock_strategy=lock["lock_strategy"],
            time_window={"hours": horizon_ticks, "tick_minutes": 60},
            license="BSD-3-Clause (pandapower).",
            notes=(
                f"{meta['desc']} Generated by pandapower.networks."
                f"{meta['constructor']}. pandapower=={_pp_version}. The "
                "OPERATE adapter does not modify the topology — it "
                "overlays chronics, perturbations, and a criticality-ordered "
                "stakeholder map on the unmodified pandapower `pp.Network`. "
                "Radial line_outage is deliberately NOT used as a difficulty "
                "knob (it would disconnect downstream load and invert the "
                "cost ladder); stress comes from DER trips + load surges + "
                "telemetry-degrading storm windows."
            ),
        )
        family = meta["family"]
        load_assignments = _generic_load_assignments(network, meta["prefix"])

    return ScenarioSeed(
        seed_id=seed_id,
        family=family,
        domain="power_grid",
        backend_kind="cigre_distribution",
        backend_config=backend_config,
        horizon_ticks=horizon_ticks,
        tick_minutes=60,
        seed=seed,
        load_assignments=load_assignments,
        perturbations=perturbations,
        dilemmas=dilemmas,
        difficulty_mode=difficulty_mode,  # type: ignore[arg-type]
        difficulty_level=difficulty_level,  # type: ignore[arg-type]
        provenance=provenance,
    )


def cigre_load_index_to_id(idx: int) -> str:
    """Convenience for tests / oracles."""
    return f"mv_load_{idx}"


# Convenience for the generator script
def list_default_cigre_variants() -> list[dict[str, Any]]:
    """v0.1.2: one base CIGRE MV variant. v0.2 will add CIGRE MV with
    different DER mixes (``with_der='pv_wind'``, ``with_der='all'``,
    ``with_der=False``)."""
    return [
        {"variant_tag": "mv_with_der_all", "with_der": "all"},
    ]
