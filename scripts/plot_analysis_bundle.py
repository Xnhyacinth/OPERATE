#!/usr/bin/env python3
"""Render evidence-linked offline analysis; never score or execute episodes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np


LABELS = {
    "Qwen/Qwen3.6-27B": "Qwen3.6 27B",
    "qwen/qwen3.8-27b": "Qwen3.8 27B",
    "qwen/qwen3.8-max": "Qwen3.8 Max",
    "deepseek-ai/deepseek-v4-flash-0731": "DS4 Flash 0731",
    "deepseek-ai/deepseek-v4-pro-0813": "DS4 Pro 0813",
    "deepseek-ai/deepseek-v4.1-flash": "DS4.1 Flash",
    "moonshotai/kimi-k3": "Kimi K3",
    "zai-org/glm-5.3": "GLM5.3",
    "glm-5.2": "GLM5.2",
    "gpt-6.1-sol": "GPT6.1 Sol",
    "gpt-6-sol": "GPT6 Sol",
    "gpt-6-luna": "GPT6 Luna",
}
REPOSITORY_PREFIX = str(Path(__file__).resolve().parents[1]) + "/"
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]


def number(value):
    """Unknown and nonfinite observations remain missing, never zero."""
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def label(model):
    return LABELS.get(model, model)


def export_value(value):
    """Portable source references without changing authoritative input objects."""
    if isinstance(value, dict):
        return {k: export_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [export_value(v) for v in value]
    if isinstance(value, str):
        return value.replace(REPOSITORY_PREFIX, "")
    return value


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for source_row in rows:
            row = export_value(source_row)
            writer.writerow(
                {
                    k: json.dumps(v, ensure_ascii=False)
                    if isinstance(v, (dict, list))
                    else v
                    for k, v in row.items()
                }
            )


def group_scores(cases, models, field):
    """Conditional weighted scores, only when the group's Q coverage is complete."""
    groups = sorted({str(r.get(field, "unknown")) for r in cases})
    index = defaultdict(list)
    for row in cases:
        index[row["model"], str(row.get(field, "unknown"))].append(row)
    rows = []
    for model in models:
        for group in groups:
            selected = index[model, group]
            weighted = [(number(r.get("Q")), number(r.get("weight"))) for r in selected]
            valid = [
                (q, w) for q, w in weighted if q is not None and w is not None and w > 0
            ]
            mass = sum(w for _, w in valid)
            complete = bool(selected) and len(valid) == len(selected)
            rows.append(
                {
                    "model": model,
                    field: group,
                    "n_cases": len(selected),
                    "n_numeric": len(valid),
                    "weight_mass": mass,
                    "Q": sum(q * w for q, w in valid) / mass
                    if complete and mass > 0
                    else None,
                }
            )
    return groups, rows


def load_paired_metrics(directory, *, contents=None):
    """Require every plotted value to match a strict-valid revalidation group."""
    directory = Path(directory)
    if contents is None:
        contents = {
            name: (directory / name).read_bytes()
            for name in ("revalidation.json", "paired_metrics.csv")
        }
    report = json.loads(contents["revalidation.json"])
    allowed = {}
    contrasts = {
        "E1": ["persistent_minus_stateless"],
        "E3": ["delay_1_minus_0", "delay_5_minus_0"],
        "E4": ["reveal_minus_withhold"],
    }
    for experiment, names in contrasts.items():
        for model_report in report.get(experiment, {}).values():
            for group in model_report.get("groups", []):
                if group.get("valid") is not True or (
                    experiment == "E3" and group.get("lite_overlap") is not True
                ):
                    continue
                case = group.get("pair_id") if experiment == "E4" else group["case"][0]
                for contrast in names:
                    for metric, value in group.get(contrast, {}).items():
                        key = (
                            experiment,
                            model_report["model"],
                            case,
                            contrast,
                            metric,
                        )
                        if key in allowed:
                            raise ValueError("ambiguous supplementary group identity")
                        allowed[key] = (number(value), group.get("scoring_version"))
    rows = list(
        csv.DictReader(io.StringIO(contents["paired_metrics.csv"].decode("utf-8")))
    )
    seen = set()
    for row in rows:
        key = tuple(
            row[k] for k in ("experiment", "model", "case", "contrast", "metric")
        )
        if key in seen or key not in allowed:
            raise ValueError("uncertified or duplicate supplementary metric")
        seen.add(key)
        expected, version = allowed[key]
        if expected != number(row["value"]):
            raise ValueError("supplementary metric differs from strict revalidation")
        if version is not None and row.get("scoring_version") != version:
            raise ValueError("supplementary scoring version mismatch")
    return rows


def paired_offsets(count, contrast_index, contrast_count):
    """Separate contrasts vertically while preserving every measured x value."""
    center = (
        np.linspace(-0.22, 0.22, contrast_count)[contrast_index]
        if contrast_count > 1
        else 0
    )
    spread = 0.24 / contrast_count
    return center + (
        np.linspace(-spread, spread, count) if count > 1 else np.array([0.0])
    )


def contribution_rows(rows, pairs):
    result = []
    for a, b in pairs:
        groups = defaultdict(list)
        for row in rows:
            direction = 1 if (row.get("model_a"), row.get("model_b")) == (a, b) else -1
            if (row.get("model_a"), row.get("model_b")) not in ((a, b), (b, a)):
                continue
            value = number(row.get("weighted_delta_Q"))
            groups[row["domain"]].append(
                direction * value if value is not None else None
            )
        for domain, values in sorted(groups.items()):
            result.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "domain": domain,
                    "weighted_delta_Q": sum(values)
                    if all(v is not None for v in values)
                    else None,
                    "n_cases": len(values),
                    "n_numeric": sum(v is not None for v in values),
                }
            )
    return result


class Renderer:
    def __init__(self, bundle, output):
        self.bundle = bundle
        self.output = output
        self.models = [r["model"] for r in bundle["models"]]
        if len(set(self.models)) != len(self.models):
            raise ValueError("duplicate model identity")
        self.entries = []
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": ["DejaVu Sans"],
                "font.size": 8,
                "axes.titlesize": 9,
                "axes.labelsize": 8,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "legend.fontsize": 7,
                "svg.fonttype": "none",
                "pdf.fonttype": 42,
                "axes.spines.right": False,
                "axes.spines.top": False,
                "figure.facecolor": "white",
                "savefig.facecolor": "white",
            }
        )

    def save(self, fig, name, rows, question, note):
        fig.savefig(self.output / f"{name}.pdf")
        fig.savefig(self.output / f"{name}.svg")
        fig.savefig(self.output / f"{name}.png", dpi=600)
        paths = [f"{name}.{extension}" for extension in ("pdf", "svg", "png")]
        plt.close(fig)
        source = self.output / f"{name}.csv"
        write_csv(source, rows)
        self.entries.append(
            {
                "figure": name,
                "status": "rendered",
                "question": question,
                "files": paths,
                "source_data": source.name,
                "rows": len(rows),
                "note": note,
                "interval": "None: descriptive fixed-panel observations",
            }
        )

    def skip(self, name, reason):
        self.entries.append(
            {"figure": name, "status": "not_rendered", "reason": reason}
        )

    def heatmap(self, field, name):
        groups, rows = group_scores(self.bundle.get("cases", []), self.models, field)
        if not groups:
            self.skip(name, "No case-level scores available")
            return
        matrix = np.array([np.nan if r["Q"] is None else r["Q"] for r in rows]).reshape(
            len(self.models), -1
        )
        fig, ax = plt.subplots(figsize=(7.086614, 155 / 25.4), layout="constrained")
        cmap = plt.get_cmap("cividis").with_extremes(bad="#eeeeee")
        im = ax.imshow(matrix, vmin=0, vmax=100, cmap=cmap, aspect="auto")
        ax.set_yticks(range(len(self.models)), [label(m) for m in self.models])
        ax.set_xticks(
            range(len(groups)),
            [g.replace("_", " ") for g in groups],
            rotation=65,
            ha="right",
            rotation_mode="anchor",
        )
        ax.set_title(
            f"Where do model outcomes differ?\nConditional Q030 by {field.replace('_', ' ')}"
        )
        fig.colorbar(im, ax=ax, label="Q030 (0–100)", shrink=0.65)
        self.save(
            fig,
            name,
            rows,
            "Where do model outcomes differ?",
            "Weights renormalized within each displayed group only; not a new ranking. "
            "Gray denotes missing/incomplete groups. Source CSV preserves every model/group.",
        )

    def bottlenecks(self):
        cases = self.bundle.get("cases", [])
        categories = sorted({str(r.get("bottleneck") or "unknown") for r in cases})
        if not categories:
            self.skip("F1b_bottlenecks", "No bottleneck classification")
            return
        counts = Counter(
            (r["model"], str(r.get("bottleneck") or "unknown")) for r in cases
        )
        rows = [
            {"model": m, "bottleneck": c, "n_cases": counts[m, c]}
            for m in self.models
            for c in categories
        ]
        fig, ax = plt.subplots(figsize=(7.086614, 125 / 25.4), layout="constrained")
        left = np.zeros(len(self.models))
        for i, category in enumerate(categories):
            values = np.array([counts[m, category] for m in self.models])
            ax.barh(
                range(len(self.models)),
                values,
                left=left,
                label=category.replace("_", " "),
                color=COLORS[i % len(COLORS)],
                hatch="/" if i >= len(COLORS) else None,
            )
            left += values
        ax.set_yticks(range(len(self.models)), [label(m) for m in self.models])
        ax.invert_yaxis()
        ax.set_xlabel("Number of fixed-panel cases (not weighted score)")
        ax.set_title("What limits the observed outcome?")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=2)
        self.save(
            fig,
            "F1b_bottlenecks",
            rows,
            "Which constraints limit outcomes?",
            "Case counts; categories are supplied by the analysis builder. No causal inference.",
        )

    def distributions(self):
        rows = [
            {
                k: r.get(k)
                for k in (
                    "model",
                    "scenario_signature",
                    "seed",
                    "Q",
                    "N",
                    "F",
                    "weight",
                )
            }
            for r in self.bundle.get("cases", [])
        ]
        fig, ax = plt.subplots(figsize=(7.086614, 130 / 25.4), layout="constrained")
        for i, model in enumerate(self.models):
            values = sorted(
                number(r.get("Q"))
                for r in rows
                if r["model"] == model and number(r.get("Q")) is not None
            )
            if values:
                ax.plot(
                    values,
                    np.arange(1, len(values) + 1) / len(values),
                    color=COLORS[i % 6],
                    linestyle="-" if i < 6 else "--",
                    label=label(model),
                )
        ax.set(
            xlabel="Case Q030",
            ylabel="Empirical fraction of numeric cases",
            xlim=(0, 100),
            ylim=(0, 1),
            title="Where are the low-outcome cases?",
        )
        ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17))
        self.save(
            fig,
            "A1_score_distributions",
            rows,
            "How heavy is the low-score tail?",
            "Unweighted empirical distribution, not rank uncertainty. Missing cases stay in Source Data.",
        )

    def score_versions(self):
        rows = [
            r
            for r in self.bundle.get("score_changes", {}).get("models", [])
            if r.get("model") in self.models
        ]
        if not rows:
            self.skip("A2_score_versions", "No supplied 0.28/0.30 comparison")
            return
        lookup = {r["model"]: r for r in rows}
        fig, ax = plt.subplots(figsize=(7.086614, 125 / 25.4), layout="constrained")
        for i, model in enumerate(self.models):
            r = lookup.get(model, {})
            old, new = number(r.get("Q028")), number(r.get("Q030"))
            if old is not None and new is not None:
                ax.plot([old, new], [i, i], color="#999999")
                ax.scatter(
                    [old],
                    [i],
                    facecolor="white",
                    edgecolor="#0072B2",
                    label="0.28" if i == 0 else None,
                )
                ax.scatter(
                    [new], [i], color="#0072B2", label="0.30" if i == 0 else None
                )
        ax.set_yticks(range(len(self.models)), [label(m) for m in self.models])
        ax.invert_yaxis()
        ax.set(
            xlabel="Fixed-panel Q", title="How does the reader policy change scores?"
        )
        ax.legend()
        self.save(
            fig,
            "A2_score_versions",
            rows,
            "Which scores depend on the scoring convention?",
            "Same historical evidence under two policies; not model improvement over time.",
        )

    def sensitivity(self):
        report = self.bundle.get("rank_sensitivity", {})
        rows = []
        for kind in ("physical_source_cluster", "domain"):
            for variant in report.get(kind, {}).get("variants", []):
                for model in self.models:
                    rows.append(
                        {
                            "model": model,
                            "deletion_unit": kind,
                            "excluded": variant.get("excluded"),
                            "rank": variant.get("ranks", {}).get(model),
                            "Q": variant.get("scores", {}).get(model),
                            "baseline_rank": report.get("baseline_ranks", {}).get(
                                model
                            ),
                        }
                    )
        if not rows:
            self.skip("A3_rank_sensitivity", "No precomputed deletion sensitivity")
            return
        fig, ax = plt.subplots(figsize=(7.086614, 125 / 25.4), layout="constrained")
        for i, m in enumerate(self.models):
            for offset, kind, color in [
                (-0.12, "physical_source_cluster", "#0072B2"),
                (0.12, "domain", "#D55E00"),
            ]:
                vals = [
                    number(r["rank"])
                    for r in rows
                    if r["model"] == m
                    and r["deletion_unit"] == kind
                    and number(r["rank"]) is not None
                ]
                if vals:
                    ax.plot(
                        [min(vals), max(vals)],
                        [i + offset] * 2,
                        color=color,
                        linewidth=2,
                        label=kind.replace("_", " ") if i == 0 else None,
                    )
            base = number(report.get("baseline_ranks", {}).get(m))
            if base is not None:
                ax.plot(base, i, "o", color="#333333", markersize=3)
        ax.set_yticks(range(len(self.models)), [label(m) for m in self.models])
        ax.invert_yaxis()
        ax.set(
            xlabel="Rank in original complete cohort",
            title="How sensitive are ranks to panel composition?",
        )
        ax.legend()
        self.save(
            fig,
            "A3_rank_sensitivity",
            rows,
            "How source-dependent is the ranking?",
            "Ranges are deterministic deletion extrema, not confidence intervals. Original cohort ranks retained.",
        )

    def behavior(self):
        rows = self.bundle.get("behavior", [])
        if not rows:
            self.skip("F3_behavior", "No authenticated trajectory summaries")
            self.skip("F4_efficiency", "No observed usage or calls")
            return
        metrics = [
            "unique_calls",
            "successful_receipts",
            "failed_receipts",
            "proven_engine_effect_calls",
            "protocol_repair_attempts",
            "session_compactions",
        ]
        exported = []
        fig, axes = plt.subplots(
            3, 2, figsize=(7.086614, 220 / 25.4), layout="constrained"
        )
        for ax, metric in zip(axes.flat, metrics):
            for i, model in enumerate(self.models):
                selected = [r for r in rows if r.get("model") == model]
                vals = [
                    number(r.get(metric))
                    for r in selected
                    if number(r.get(metric)) is not None
                ]
                mean = sum(vals) / len(vals) if vals else None
                exported.append(
                    {
                        "model": model,
                        "metric": metric,
                        "mean": mean,
                        "n_numeric": len(vals),
                        "n_cases": len(selected),
                    }
                )
                if mean is not None:
                    ax.barh(i, mean, color="#0072B2")
            ax.set_yticks(range(len(self.models)), [label(m) for m in self.models])
            ax.invert_yaxis()
            ax.set_title(
                metric.replace("_", " ").replace(
                    "proven engine effect calls", "Observed effect calls (lower bound)"
                ),
                wrap=True,
            )
            ax.set_xlabel("Mean per observed episode")
        self.save(
            fig,
            "F3_behavior",
            exported,
            "How do tool interaction and repair patterns differ?",
            "Separate event summaries, not a nested success funnel. Coverage in Source Data; "
            "means over observed episodes only. Observed effect calls are lower bounds, not task success.",
        )
        model_q = {r["model"]: number(r.get("Q")) for r in self.bundle["models"]}
        fig, axes = plt.subplots(
            1, 2, figsize=(7.086614, 125 / 25.4), layout="constrained"
        )
        exported = []
        for ax, metric in zip(axes, ["llm_calls_ok", "total_tokens"]):
            for i, model in enumerate(self.models):
                selected = [r for r in rows if r.get("model") == model]
                vals = [
                    number(r.get(metric))
                    for r in selected
                    if number(r.get(metric)) is not None
                ]
                # No partial-cost point is comparable with the full-panel outcome.
                complete = bool(selected) and len(vals) == len(selected)
                expected = next(
                    r.get("n_expected", len(selected))
                    for r in self.bundle["models"]
                    if r["model"] == model
                )
                complete = complete and len(selected) == expected
                if metric == "total_tokens":
                    complete = complete and all(
                        r.get("provider_usage_complete") is True for r in selected
                    )
                value = sum(vals) / len(vals) if vals and complete else None
                exported.append(
                    {
                        "model": model,
                        "metric": metric,
                        "mean": value,
                        "Q": model_q[model],
                        "n_numeric": len(vals),
                        "n_cases": len(selected),
                        "complete": complete,
                    }
                )
                if value is not None and model_q[model] is not None:
                    ax.scatter(
                        value,
                        model_q[model],
                        color=COLORS[i % 6],
                        marker="o" if i < 6 else "s",
                        label=f"{i + 1} {label(model)}",
                    )
                    ax.annotate(
                        str(i + 1),
                        (value, model_q[model]),
                        xytext=(3, 3),
                        textcoords="offset points",
                        fontsize=7,
                    )
            ax.set(
                xlabel="Successful model calls / episode"
                if metric == "llm_calls_ok"
                else "Actual tokens / episode",
                ylabel="Fixed-panel Q030",
                title="Successful calls"
                if metric == "llm_calls_ok"
                else "Actual usage",
            )
            plotted = sum(
                r["metric"] == metric and r["mean"] is not None for r in exported
            )
            ax.set_title(ax.get_title() + f" ({plotted}/{len(self.models)} models)")
            if not ax.collections:
                ax.text(
                    0.5,
                    0.5,
                    "No complete usage coverage",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=3)
        self.save(
            fig,
            "F4_efficiency",
            exported,
            "What observed computation accompanies quality?",
            "Only complete per-model observed cost coverage plotted. Missing usage is not zero. "
            "Association, not budget intervention. Model numbers identify close points.",
        )

    def timelines(self):
        rows = self.bundle.get("timeline", [])
        if not rows:
            self.skip("F2_native_time", "No authenticated native backend tick series")
            return
        groups = defaultdict(list)
        for r in rows:
            key = (
                r.get("backend", r.get("backend_kind")),
                r.get("scenario_signature", r.get("key")),
                r.get("seed"),
                r.get("metric"),
                r.get("unit"),
            )
            if number(r.get("tick")) is not None and number(r.get("value")) is not None:
                groups[key].append(r)
        if not groups:
            self.skip("F2_native_time", "No finite native tick values")
            return
        # One deterministic example per backend, maximizing model coverage, then key.
        chosen = {}
        for key in sorted(groups, key=str):
            backend = key[0]
            metric = str(key[3]).lower()
            keywords = (
                "cost",
                "violation",
                "unserved",
                "voltage",
                "risk",
                "queue",
                "load",
                "energy",
                "service",
            )
            priority = next(
                (
                    len(keywords) - i
                    for i, word in enumerate(keywords)
                    if word in metric
                ),
                0,
            )
            if priority == 0:
                continue
            n = len({r["model"] for r in groups[key]})
            varying = len({r["value"] for r in groups[key]}) > 1
            quality = (varying, priority, n)
            if backend not in chosen or quality > chosen[backend][0]:
                chosen[backend] = (quality, key)
        for index, (_, key) in enumerate(chosen.values(), 1):
            selected = groups[key]
            fig, ax = plt.subplots(figsize=(7.086614, 125 / 25.4), layout="constrained")
            ticks = sorted({float(r["tick"]) for r in selected})
            if all(t.is_integer() for t in ticks):
                ticks = list(range(int(ticks[0]), int(ticks[-1]) + 1))
            matrix = np.full((len(self.models), len(ticks)), np.nan)
            tick_index = {t: i for i, t in enumerate(ticks)}
            for row in selected:
                mi, ti = self.models.index(row["model"]), tick_index[row["tick"]]
                if np.isfinite(matrix[mi, ti]) and matrix[mi, ti] != float(
                    row["value"]
                ):
                    raise ValueError("ambiguous native value at model/tick")
                matrix[mi, ti] = float(row["value"])
            edges = (
                [ticks[0] - 0.5, ticks[0] + 0.5]
                if len(ticks) == 1
                else [ticks[0] - (ticks[1] - ticks[0]) / 2]
                + [(a + b) / 2 for a, b in zip(ticks, ticks[1:])]
                + [ticks[-1] + (ticks[-1] - ticks[-2]) / 2]
            )
            observed = matrix[np.isfinite(matrix)]
            constant = bool(len(observed)) and np.all(observed == observed[0])
            cmap = (
                ListedColormap(["#0072B2"]) if constant else plt.get_cmap("cividis")
            ).with_extremes(bad="#eeeeee")
            im = ax.pcolormesh(
                edges,
                np.arange(len(self.models) + 1) - 0.5,
                matrix,
                cmap=cmap,
                shading="flat",
            )
            ax.set_yticks(range(len(self.models)), [label(m) for m in self.models])
            ax.invert_yaxis()
            ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))
            ax.set(
                xlabel="Native tick",
                title=f"Native evolution: {key[0]}\nSame scenario and seed; {key[1]}",
            )
            if constant:
                ax.set_title(
                    ax.get_title()
                    + f"\n{key[3]}: All observed values = {observed[0]:g}"
                )
                ax.text(
                    0.0,
                    -0.13,
                    "Blue: observed constant; gray: missing",
                    transform=ax.transAxes,
                    fontsize=7,
                    ha="left",
                )
            else:
                fig.colorbar(
                    im,
                    ax=ax,
                    label=f"{key[3]} ({key[4] or 'backend-native units'})",
                    shrink=0.7,
                )
            self.save(
                fig,
                f"F2_native_time_{index:02d}",
                selected,
                "How do native trajectories diverge?",
                "Illustrative case per backend: nonconstant series, then "
                "cost/violation/unserved/voltage/risk/queue/load/energy/service priority, "
                "model coverage, lexicographic "
                "scenario/metric. No cross-source averaging or extrapolation beyond termination. "
                "Metric units only asserted when supplied by evidence. Gray cells are missing; no temporal interpolation.",
            )
        write_csv(self.output / "F2_all_native_time.csv", rows)
        self.entries.append(
            {
                "figure": "F2_all_native_time",
                "status": "source_data_only",
                "source_data": "F2_all_native_time.csv",
                "rows": len(rows),
                "note": "All extracted observations retained; displayed examples do not establish recovery causality.",
            }
        )

    def tradeoff(self):
        rows = self.bundle.get("cases", [])
        groups = defaultdict(list)
        for r in rows:
            groups[r.get("domain"), r.get("scenario_signature"), r.get("seed")].append(
                r
            )
        chosen = {}
        for key in sorted(groups, key=str):
            selected = [
                r
                for r in groups[key]
                if number(r.get("C")) is not None and number(r.get("F")) is not None
            ]
            if len(selected) < 2:
                continue
            spread = max(float(r["C"]) for r in selected) - min(
                float(r["C"]) for r in selected
            )
            if key[0] not in chosen or spread > chosen[key[0]][0]:
                chosen[key[0]] = (spread, key, selected)
        if not chosen:
            self.skip("A4_cost_fulfillment", "No matched native cost/fulfillment pairs")
            return
        fig, axes = plt.subplots(
            math.ceil(len(chosen) / 2),
            2,
            figsize=(7.086614, 2.4 * math.ceil(len(chosen) / 2)),
            layout="constrained",
            squeeze=False,
        )
        source = []
        for ax, (_, key, selected) in zip(axes.flat, chosen.values()):
            for r in selected:
                i = self.models.index(r["model"])
                ax.scatter(
                    r["C"], r["F"], color=COLORS[i % 6], marker="o" if i < 6 else "s"
                )
            ax.set(
                title=f"{key[0]}\n{key[1]}",
                xlabel="Native cost C (same case)",
                ylabel="Fulfillment F",
            )
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.ticklabel_format(
                axis="x", style="sci", scilimits=(-3, 4), useOffset=False
            )
            source.extend(selected)
        spare = list(axes.flat)[len(chosen) :]
        for ax in spare:
            ax.set_visible(False)
        handles = [
            Line2D(
                [],
                [],
                color=COLORS[i % 6],
                marker="o" if i < 6 else "s",
                linestyle="None",
                label=label(model),
            )
            for i, model in enumerate(self.models)
        ]
        if spare:
            spare[0].set_visible(True)
            spare[0].axis("off")
            spare[0].legend(handles=handles, loc="center", ncol=1, frameon=False)
        else:
            fig.legend(
                handles=handles, loc="outside lower center", ncol=3, frameon=False
            )
        self.save(
            fig,
            "A4_cost_fulfillment",
            source,
            "Can equal-looking outcomes hide native tradeoffs?",
            "Illustrative largest native-C spread case per domain; selection is descriptive, not "
            "representative. No cross-source C pooling. Coincident observations may overlap; all values in Source Data.",
        )

    def supplementary(self, directory, *, rows=None):
        if directory is None:
            self.skip(
                "F5_context_information",
                "No revalidated supplementary metrics supplied",
            )
            self.skip("F6_delay", "No revalidated supplementary metrics supplied")
            return
        path = Path(directory) / "paired_metrics.csv"
        if rows is None and not path.is_file():
            self.skip("F5_F6_supplementary", "Missing paired_metrics.csv")
            return
        if rows is None:
            rows = load_paired_metrics(directory)
        rows = [r for r in rows if r.get("model") in self.models]
        for experiment in ("E1", "E3", "E4"):
            selected = [r for r in rows if r.get("experiment") == experiment]
            display = {
                "E1": {"model_calls", "tool_calls"},
                "E3": {
                    "actions_effected",
                    "actions_expired",
                    "response_missed",
                    "takeovers",
                },
                "E4": {
                    "investigation_actions",
                    "native_actual_cost",
                    "native_cost_component:compute_cost",
                    "native_cost_component:sla_violation_cost",
                    "native_cost_component:unfinished_work_penalty",
                },
            }
            write_csv(self.output / f"{experiment}_all_paired_metrics.csv", selected)
            metrics = sorted(
                {
                    (r["metric"], r.get("scoring_version", ""))
                    for r in selected
                    if r["metric"] in display[experiment]
                }
            )
            if not metrics:
                self.skip(
                    f"F{'6' if experiment == 'E3' else '5'}_{experiment}",
                    "No strictly certified paired numeric metrics",
                )
                continue
            for index, (metric, version) in enumerate(metrics, 1):
                values = [
                    r
                    for r in selected
                    if r["metric"] == metric and r.get("scoring_version", "") == version
                ]
                if not any(number(r.get("value")) is not None for r in values):
                    continue
                fig, ax = plt.subplots(
                    figsize=(7.086614, 125 / 25.4), layout="constrained"
                )
                contrasts = sorted({r["contrast"] for r in values})
                for ci, contrast in enumerate(contrasts):
                    for mi, model in enumerate(self.models):
                        observed = [
                            number(r["value"])
                            for r in values
                            if r["model"] == model and r["contrast"] == contrast
                        ]
                        observed = [x for x in observed if x is not None]
                        if observed:
                            offsets = paired_offsets(len(observed), ci, len(contrasts))
                            ax.scatter(
                                observed,
                                mi + np.array(offsets),
                                color=COLORS[ci % 6],
                                marker="o" if ci % 2 == 0 else "s",
                                s=13,
                                label=contrast
                                if mi
                                == next(
                                    (
                                        j
                                        for j, m in enumerate(self.models)
                                        if any(
                                            r["model"] == m
                                            and r["contrast"] == contrast
                                            for r in values
                                        )
                                    ),
                                    -1,
                                )
                                else None,
                            )
                ax.axvline(0, color="#777777", linewidth=0.7)
                tick_labels = []
                for model in self.models:
                    counts = [
                        sum(
                            r["model"] == model
                            and r["contrast"] == c
                            and number(r["value"]) is not None
                            for r in values
                        )
                        for c in contrasts
                    ]
                    tick_labels.append(
                        f"{label(model)} (n={'/'.join(map(str, counts))})"
                    )
                ax.set_yticks(range(len(self.models)), tick_labels)
                ax.invert_yaxis()
                ax.set(
                    xlabel=f"Paired difference: {metric}",
                    title=f"{experiment}: {metric}\nScoring: {version or 'native / process metric'}",
                )
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1)
                self.save(
                    fig,
                    f"F{'6' if experiment == 'E3' else '5'}_{experiment}_{index:02d}",
                    values,
                    f"How does {experiment} change {metric}?",
                    "Every dot is a certified pair/contrast from the supplied revalidation. Vertical "
                    "offsets separate contrasts and observations; x is unchanged. Row n follows legend contrast order. "
                    "No independence or repeated-run uncertainty claim.",
                )

    def contributions(self):
        pairs = [
            ("deepseek-ai/deepseek-v4.1-flash", "moonshotai/kimi-k3"),
            ("gpt-6.1-sol", "gpt-6-sol"),
            ("gpt-6-sol", "gpt-6-luna"),
        ]
        rows = contribution_rows(
            self.bundle.get("pairwise_case_differences", []), pairs
        )
        if not rows:
            self.skip(
                "A7_domain_contributions",
                "No paired case differences for prespecified model pairs",
            )
            return
        domains = sorted({r["domain"] for r in rows})
        fig, axes = plt.subplots(
            1, 3, figsize=(7.086614, 105 / 25.4), layout="constrained", sharey=True
        )
        for ax, (a, b) in zip(axes, pairs):
            selected = {
                r["domain"]: r for r in rows if r["model_a"] == a and r["model_b"] == b
            }
            for i, domain in enumerate(domains):
                value = selected.get(domain, {}).get("weighted_delta_Q")
                if value is not None:
                    ax.barh(i, value, color="#0072B2" if value >= 0 else "#D55E00")
            ax.axvline(0, color="#777777", linewidth=0.7)
            ax.set_title(f"{label(a)} minus\n{label(b)}")
            ax.set_xlabel("Contribution to Q gap")
            ax.set_yticks(range(len(domains)), [d.replace("_", "\n") for d in domains])
        axes[0].invert_yaxis()
        self.save(
            fig,
            "A7_domain_contributions",
            rows,
            "Which domains contribute to matched-model gaps?",
            "Three model pairs fixed before inspecting effect size. Sum of original case weight "
            "times paired Q difference; domains are not renormalized. Not a causal attribution.",
        )

    def run(self):
        self.heatmap("task_family", "F1a_family_outcomes")
        self.bottlenecks()
        self.heatmap("domain", "A0_domain_outcomes")
        self.distributions()
        self.score_versions()
        self.sensitivity()
        self.heatmap("difficulty_mode", "A5_difficulty_mode")
        self.heatmap("difficulty_level", "A6_difficulty_level")
        self.contributions()
        self.behavior()
        self.timelines()
        self.tradeoff()


def render_bundle(bundle_path, output_dir, supplementary=None):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {output_dir}"
        )
    bundle_path = Path(bundle_path)
    bundle_bytes = bundle_path.read_bytes()
    bundle = json.loads(bundle_bytes)
    script_bytes = Path(__file__).read_bytes()
    supplementary_bytes = {}
    paired_rows = None
    if supplementary:
        supplementary_bytes = {
            name: (Path(supplementary) / name).read_bytes()
            if name in ("paired_metrics.csv", "revalidation.json")
            or (Path(supplementary) / name).is_file()
            else None
            for name in (
                "paired_metrics.csv",
                "revalidation.json",
                "input_hashes.json",
                "artifact_locators.json",
            )
        }
        paired_rows = load_paired_metrics(supplementary, contents=supplementary_bytes)
    renderer = Renderer(bundle, output_dir)
    output_dir.mkdir(parents=True)
    renderer.run()
    renderer.supplementary(supplementary, rows=paired_rows)
    inputs = {bundle_path: bundle_bytes, Path(__file__): script_bytes}
    inputs.update(
        {
            Path(supplementary) / name: content
            for name, content in supplementary_bytes.items()
        }
    )
    for path, content in inputs.items():
        current = path.read_bytes() if path.is_file() else None
        if current != content:
            raise ValueError(f"input_changed_during_render: {path}")
    manifest = {
        "schema_version": "operate_analysis_figures.v1",
        "bundle_sha256": hashlib.sha256(bundle_bytes).hexdigest(),
        "plot_script_sha256": hashlib.sha256(script_bytes).hexdigest(),
        "models": renderer.models,
        "figures": renderer.entries,
        "font_minimum_pt": 7,
        "backend": "matplotlib",
        "provider_calls": 0,
        "native_episode_replays": 0,
        "supplementary_sha256": hashlib.sha256(
            supplementary_bytes["paired_metrics.csv"]
        ).hexdigest()
        if supplementary
        else None,
        "supplementary_bindings": {
            name: hashlib.sha256(content).hexdigest()
            for name, content in supplementary_bytes.items()
            if content is not None
        },
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--supplementary", type=Path)
    args = parser.parse_args()
    manifest = render_bundle(args.bundle, args.output_dir, args.supplementary)
    print(
        json.dumps(
            {"output_dir": str(args.output_dir), "figures": len(manifest["figures"])}
        )
    )


if __name__ == "__main__":
    main()
