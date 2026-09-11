#!/usr/bin/env python3
"""Build compact Lite quality evidence for v0.62 selection.

Joins four-model Lite 0.17 scores, three CPU baselines on the original 193-row
Lite, and Hy3 Full ok episodes. Episode trees are not read at suite build time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE = REPO_ROOT / "release" / "operate_v0_62_0"
AUDIT = REPO_ROOT / ".hl" / "lite_compress_20260911" / "audit.json"
HY3_OK = REPO_ROOT / ".hl" / "lite_compress_20260911" / "hy3_full_ok.json"
CPU = REPO_ROOT / ".hl" / "lite_compress_20260911" / "cpu_baselines_017.json"

CRITERIA = {
    "dead_headroom": "oracle primary < 5 and max 4-model primary < 5",
    "wait_parity_collapse": "oracle < 2 and wait=0 and max model < 2",
    "too_easy_saturated": "min 4-model primary >= 40 and oracle-mean < 10",
    "too_easy_ceiling": (
        "min 4-model primary >= 70, or Hy3 Full primary >= 70 on a row without "
        "CPU baselines"
    ),
    "short_and_easy": "horizon<=16, mean>=50, no model zero",
    "lite_cost_long_horizon": "horizon>=193",
    "lite_cost_extreme_horizon": "horizon>=600",
    "beats_oracle": "max model > oracle+15",
    "oracle_yardstick_broken": (
        "beats_oracle and oracle < 10 and not keep_headroom; 0.17 cannot rank"
    ),
    "keep_headroom": "oracle>=15 and mean model <= 0.55*oracle",
    "keep_discriminative": "range>=20, oracle>=10, max model>=15",
    "all_models_zero_oracle_open": "all four models 0 and oracle >= 20",
    "cpu_headroom": "oracle>=15 and greedy <= 0.55*oracle (wait is the third CPU arm)",
    "cpu_greedy_peer_strong": (
        "greedy >= 0.7*oracle and oracle>=20; keep when models still lag the heuristic"
    ),
    "cpu_oracle_dead": "CPU oracle primary < 5",
    "hy3_hard_band": "Hy3 Full ok primary in [5, 50)",
    "hy3_hard_strict": "Hy3 Full ok primary in [5, 35)",
    "hy3_zero_unresolved": (
        "Hy3 Full ok primary == 0, difficulty high/extreme, horizon >= 17; "
        "not treated as dead without an oracle"
    ),
}


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 3)


def _horizon_flags(horizon: int) -> list[str]:
    flags: list[str] = []
    if horizon >= 193:
        flags.append("lite_cost_long_horizon")
    if horizon >= 600:
        flags.append("lite_cost_extreme_horizon")
    return flags


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=RELEASE / "lite_quality_evidence.json",
    )
    args = parser.parse_args()

    core = json.loads((RELEASE / "core_suite.json").read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    hy3 = json.loads(HY3_OK.read_text(encoding="utf-8"))
    cpu = json.loads(CPU.read_text(encoding="utf-8"))

    scored_193: dict[str, dict[str, Any]] = {}
    for group in ("candidate", "dropped"):
        for rec in audit[group]:
            scored_193[rec["scenario_id"]] = rec
    if len(scored_193) != 193:
        raise SystemExit(f"expected 193 CPU/model-scored Lite rows, got {len(scored_193)}")

    rows: dict[str, dict[str, Any]] = {}
    for row in core["scenarios"]:
        sid = row["scenario_id"]
        horizon = int(row["horizon_ticks"])
        rec: dict[str, Any] = {
            "in_cpu_baseline": sid in scored_193,
            "in_parent_lite": sid in scored_193,
            "domain": row["domain"],
            "family": row["family"],
            "backend_kind": row["backend_kind"],
            "difficulty_level": row["difficulty_level"],
            "horizon": horizon,
        }
        flags: list[str] = []
        if sid in scored_193:
            src = scored_193[sid]
            rec["oracle_primary"] = _round(src.get("oracle"))
            rec["four_n_ok"] = sum(
                1 for value in (src.get("primaries") or {}).values() if value is not None
            )
            rec["four_min"] = _round(src.get("min_model"))
            rec["four_max"] = _round(src.get("max_model"))
            rec["four_mean"] = _round(src.get("mean_model"))
            flags.extend(src.get("flags") or [])
        cpu_row = cpu.get(sid) or {}
        if cpu_row:
            rec["cpu_wait"] = _round(cpu_row.get("wait_only"))
            rec["cpu_greedy"] = _round(cpu_row.get("greedy_heuristic"))
            rec["cpu_oracle"] = _round(cpu_row.get("oracle_offline"))
            oracle = rec.get("cpu_oracle")
            greedy = rec.get("cpu_greedy")
            if oracle is not None and oracle < 5:
                flags.append("cpu_oracle_dead")
            if (
                oracle is not None
                and greedy is not None
                and oracle >= 15
                and greedy <= 0.55 * oracle
            ):
                flags.append("cpu_headroom")
            if (
                oracle is not None
                and greedy is not None
                and oracle >= 20
                and greedy >= 0.7 * oracle
            ):
                flags.append("cpu_greedy_peer_strong")
        hy3_row = hy3.get(sid)
        if hy3_row is not None:
            rec["hy3_ok"] = True
            rec["hy3_primary"] = _round(hy3_row.get("primary"))
            rec["hy3_floor"] = bool(hy3_row.get("floor"))
        else:
            rec["hy3_ok"] = False
        hy3_primary = rec.get("hy3_primary")
        if sid not in scored_193:
            flags.extend(_horizon_flags(horizon))
            if "__relabel_v1" in sid:
                flags.append("relabel_alias")
            if hy3_primary is not None:
                if hy3_primary >= 70:
                    flags.append("too_easy_ceiling")
                elif 5 <= hy3_primary < 50:
                    flags.append("hy3_hard_band")
                elif (
                    hy3_primary == 0
                    and row["difficulty_level"] in {"high", "extreme"}
                    and horizon >= 17
                ):
                    flags.append("hy3_zero_unresolved")
        if hy3_primary is not None and 5 <= hy3_primary < 35:
            flags.append("hy3_hard_strict")
        oracle = rec.get("oracle_primary")
        if (
            "beats_oracle" in flags
            and oracle is not None
            and oracle < 10
            and "keep_headroom" not in flags
        ):
            flags.append("oracle_yardstick_broken")
        rec["flags"] = sorted(set(flags))
        rows[sid] = rec

    payload = {
        "schema_version": "operate-lite-quality-evidence-v1",
        "scoring_version": "0.17.0",
        "parent_release_id": core["release_id"],
        "cpu_baseline_n": 193,
        "n_core": len(rows),
        "n_hy3_ok": sum(1 for rec in rows.values() if rec.get("hy3_ok")),
        "four_model_ids": [
            "glm-5.3-flash-ioa",
            "gpt-5.6-luna",
            "deepseek-v4-flash-ioa",
            "hy3-ioa",
        ],
        "cpu_baseline_ids": ["wait_only", "greedy_heuristic", "oracle_offline"],
        "criteria": CRITERIA,
        "sources": {
            "four_model_lite_audit": str(AUDIT.relative_to(REPO_ROOT)),
            "cpu_baselines_017": str(CPU.relative_to(REPO_ROOT)),
            "hy3_full_ok": str(HY3_OK.relative_to(REPO_ROOT)),
            "hy3_full_episode_trees": [
                ".hl/new_release_20260906/campaign_hy3_biao_native_ioa_v062/results/tencent_hy3_full_logical_biao_native_ioa_v062/treatment-c718c9026957f7e2706a97835164bdf8782ee883ae65753281697a150335c2ab/episodes.jsonl",
                ".hl/retest_20260909/hy3_full_258_biao/treatment-04e2c2bd5ce0e6ff1dad4ddeff685a57b4c05e147a1e387e0452d3882215fd25/episodes.jsonl",
            ],
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {args.output}: {len(rows)} core rows, "
        f"{payload['n_hy3_ok']} hy3 ok, {payload['cpu_baseline_n']} cpu-scored"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
