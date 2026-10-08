import csv
import json

import pytest

from scripts import plot_analysis_bundle as plots


@pytest.mark.parametrize("value", [None, "unknown", float("nan"), float("inf"), True])
def test_missing_or_nonfinite_is_not_zero(value):
    assert plots.number(value) is None


def test_group_scores_preserves_all_models_and_missing_weight():
    models = [f"model-{i}" for i in range(12)]
    cases = [
        {"model": m, "task_family": "energy", "Q": 50, "weight": 0.5} for m in models
    ]
    cases.append(
        {"model": models[0], "task_family": "energy", "Q": None, "weight": 0.5}
    )
    _, rows = plots.group_scores(cases, models, "task_family")
    assert {r["model"] for r in rows} == set(models)
    assert rows[0]["Q"] is None
    assert rows[0]["n_numeric"] == 1
    assert rows[1]["Q"] == 50


def test_plot_source_data_keeps_all_twelve_models(tmp_path):
    models = [f"model-{i}" for i in range(12)]
    bundle = {
        "models": [{"model": m} for m in models],
        "cases": [
            {
                "model": m,
                "task_family": "energy",
                "Q": None if i == 11 else i,
                "weight": 1.0,
            }
            for i, m in enumerate(models)
        ],
    }
    renderer = plots.Renderer(bundle, tmp_path)
    renderer.heatmap("task_family", "test_heatmap")
    with (tmp_path / "test_heatmap.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [r["model"] for r in rows] == models
    assert rows[-1]["Q"] == ""
    svg = (tmp_path / "test_heatmap.svg").read_text()
    assert "<text" in svg
    assert (tmp_path / "test_heatmap.pdf").is_file()


def test_refuses_to_overwrite_existing_output(tmp_path):
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps({"models": []}))
    with pytest.raises(FileExistsError):
        plots.render_bundle(path, tmp_path)


def test_missing_supplementary_is_explicit(tmp_path):
    renderer = plots.Renderer({"models": []}, tmp_path)
    renderer.supplementary(None)
    assert {r["figure"] for r in renderer.entries} == {
        "F5_context_information",
        "F6_delay",
    }
    assert all(r["status"] == "not_rendered" and r["reason"] for r in renderer.entries)


def test_partial_usage_does_not_produce_comparable_efficiency_point(tmp_path):
    bundle = {
        "models": [{"model": "m", "Q": 50}],
        "behavior": [
            {
                "model": "m",
                "llm_calls_ok": 4,
                "total_tokens": 100,
                "provider_usage_complete": True,
            },
            {
                "model": "m",
                "llm_calls_ok": 5,
                "total_tokens": None,
                "provider_usage_complete": False,
            },
        ],
    }
    renderer = plots.Renderer(bundle, tmp_path)
    renderer.behavior()
    with (tmp_path / "F4_efficiency.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    usage = next(r for r in rows if r["metric"] == "total_tokens")
    assert usage["mean"] == ""
    assert usage["n_numeric"] == "1"


def test_supplementary_value_must_match_valid_group(tmp_path):
    report = {
        "E1": {
            "m": {
                "model": "m",
                "groups": [
                    {
                        "case": ["s", 42],
                        "valid": True,
                        "scoring_version": "0.21.0",
                        "persistent_minus_stateless": {"model_calls": 2},
                    }
                ],
            }
        }
    }
    (tmp_path / "revalidation.json").write_text(json.dumps(report))
    rows = [
        {
            "experiment": "E1",
            "model": "m",
            "case": "s",
            "contrast": "persistent_minus_stateless",
            "metric": "model_calls",
            "value": 2,
            "scoring_version": "0.21.0",
        }
    ]
    plots.write_csv(tmp_path / "paired_metrics.csv", rows)
    assert len(plots.load_paired_metrics(tmp_path)) == 1
    rows[0]["value"] = 3
    plots.write_csv(tmp_path / "paired_metrics.csv", rows)
    with pytest.raises(ValueError, match="differs from strict"):
        plots.load_paired_metrics(tmp_path)
    rows[0]["value"] = 2
    report["E1"]["m"]["groups"][0]["valid"] = False
    (tmp_path / "revalidation.json").write_text(json.dumps(report))
    plots.write_csv(tmp_path / "paired_metrics.csv", rows)
    with pytest.raises(ValueError, match="uncertified"):
        plots.load_paired_metrics(tmp_path)


def test_timeline_never_combines_scenarios_or_backends(tmp_path):
    rows = [
        {
            "model": "m",
            "backend_kind": backend,
            "scenario_signature": case,
            "seed": 42,
            "metric": "cost",
            "unit": None,
            "tick": tick,
            "value": tick + 1,
        }
        for backend, case in [("a", "one"), ("a", "two"), ("b", "one")]
        for tick in [0, 1]
    ]
    renderer = plots.Renderer({"models": [{"model": "m"}], "timeline": rows}, tmp_path)
    renderer.timelines()
    displayed = [e for e in renderer.entries if e["status"] == "rendered"]
    assert len(displayed) == 2
    for entry in displayed:
        with (tmp_path / entry["source_data"]).open() as stream:
            selected = list(csv.DictReader(stream))
        assert (
            len(
                {
                    (r["backend_kind"], r["scenario_signature"], r["seed"])
                    for r in selected
                }
            )
            == 1
        )
    with (tmp_path / "F2_all_native_time.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == len(rows)


def test_domain_contributions_reverse_sign_and_keep_missing():
    rows = [
        {"model_a": "b", "model_b": "a", "domain": "energy", "weighted_delta_Q": 2},
        {"model_a": "b", "model_b": "a", "domain": "energy", "weighted_delta_Q": -0.5},
        {"model_a": "b", "model_b": "a", "domain": "traffic", "weighted_delta_Q": None},
    ]
    result = plots.contribution_rows(rows, [("a", "b")])
    assert result[0]["weighted_delta_Q"] == -1.5
    assert result[0]["n_cases"] == 2
    assert result[1]["weighted_delta_Q"] is None


def test_source_export_paths_are_portable_without_mutating_inputs(tmp_path):
    root = str(plots.Path(plots.__file__).resolve().parents[1])
    row = {
        "artifact_path": root + "/.hl/raw.jsonl",
        "origin": {"journal": {"path": root + "/.hl/journal.jsonl", "sha256": "abc"}},
        "Q": 50,
    }
    plots.write_csv(tmp_path / "source.csv", [row])
    assert row["artifact_path"].startswith(root)
    assert root not in (tmp_path / "source.csv").read_text()
    with (tmp_path / "source.csv").open() as stream:
        exported = next(csv.DictReader(stream))
    assert exported["artifact_path"] == ".hl/raw.jsonl"
    assert json.loads(exported["origin"])["journal"]["sha256"] == "abc"
    assert exported["Q"] == "50"


def test_constant_native_values_have_no_artificial_colorbar(tmp_path, monkeypatch):
    rows = [
        {
            "model": "m",
            "backend_kind": "sumo",
            "scenario_signature": "s",
            "seed": 42,
            "metric": "safety_violation_severity",
            "tick": 0,
            "value": 0,
        }
    ]
    renderer = plots.Renderer(
        {"models": [{"model": "m"}, {"model": "missing"}], "timeline": rows}, tmp_path
    )
    figures = []
    monkeypatch.setattr(renderer, "save", lambda fig, *args: figures.append(fig))
    renderer.timelines()
    fig = figures[0]
    assert len(fig.axes) == 1
    assert "All observed values = 0" in fig.axes[0].get_title()
    fig.canvas.draw()
    colors = fig.axes[0].collections[0].get_facecolors()
    assert not plots.np.allclose(colors[0], colors[1])
    plots.plt.close(fig)


def test_paired_contrast_offsets_do_not_hide_identical_zero_effects():
    first = plots.paired_offsets(11, 0, 2)
    second = plots.paired_offsets(11, 1, 2)
    assert max(first) < min(second)
    assert len(first) == len(second) == 11


@pytest.mark.parametrize(
    "changed",
    [
        "bundle.json",
        "paired_metrics.csv",
        "revalidation.json",
        "input_hashes.json",
        "artifact_locators.json",
    ],
)
def test_render_rejects_changed_inputs_before_manifest(tmp_path, monkeypatch, changed):
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({"models": []}))
    supplementary = tmp_path / "supplementary"
    supplementary.mkdir()
    (supplementary / "paired_metrics.csv").write_text(
        "experiment,model,case,contrast,metric,value\n"
    )
    for name in ("revalidation.json", "input_hashes.json", "artifact_locators.json"):
        (supplementary / name).write_text("{}")
    target = bundle if changed == "bundle.json" else supplementary / changed
    monkeypatch.setattr(
        plots.Renderer,
        "run",
        lambda self: target.write_bytes(target.read_bytes() + b"\n"),
    )
    output = tmp_path / "figures"
    with pytest.raises(ValueError, match="input_changed_during_render"):
        plots.render_bundle(bundle, output, supplementary)
    assert not (output / "figure_manifest.json").exists()


def test_render_manifest_hashes_exact_consumed_bytes(tmp_path, monkeypatch):
    import hashlib

    bundle = tmp_path / "bundle.json"
    content = b'{"models": []}\n'
    bundle.write_bytes(content)
    monkeypatch.setattr(plots.Renderer, "run", lambda self: None)
    manifest = plots.render_bundle(bundle, tmp_path / "figures")
    assert manifest["bundle_sha256"] == hashlib.sha256(content).hexdigest()
    assert manifest["supplementary_bindings"] == {}
