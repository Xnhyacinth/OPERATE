#!/usr/bin/env python3
"""Build the unified ``data/`` directory — the HF-distributable eval bundle.

``data/`` is the canonical, self-contained eval dataset: the 351 v0.51.0 core
scenario YAMLs, the v0.51.0 release manifest + suite JSONs, and the backend
runtime data each released backend reads. A checkout with ``data/`` populated
can run the full core without ``works/`` (the setup script symlinks
``works/<name>`` -> ``data/backends/<name>`` so backend code that reads
``works/`` resolves transparently).

Idempotent. Writes ``data/MANIFEST.json`` with SHA-256s of every copied file.

Usage:
    python scripts/build_data_dir.py
    python scripts/build_data_dir.py --link-backends   # also symlink works/ -> data/backends/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
DATA = REPO / "data"
WORKS = REPO / "works"
RELEASE_ID = "dt_sched_bench_v0_51_0"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_tree(src: Path, dst: Path, manifest: dict, prefix: str) -> int:
    """Copy ``src`` -> ``dst`` (recursive), recording SHA-256s. Returns n files."""
    n = 0
    for p in sorted(src.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts or ".git" in p.parts:
            continue
        rel = p.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
        manifest[f"{prefix}/{rel}"] = sha256_file(p)
        n += 1
    return n


def build_scenarios(manifest: dict) -> int:
    """Copy the 351 core scenario YAMLs into data/scenarios/ (flattened, release-agnostic)."""
    from run import load_scenario_yaml  # noqa: PLC0415
    from scripts import batch_llm_eval as m  # noqa: PLC0415

    slugs = m._release_suite_scenarios("v0_51_core_suite")
    out_root = DATA / "scenarios"
    n = 0
    for slug in slugs:
        # slug is like "releases/dt_sched_bench_v0_6_0/power_grid/.../<id>"
        # Flatten to data/scenarios/<domain>/<family>/<mode>/<level>/<id>.yaml
        try:
            scenario = load_scenario_yaml(slug)
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP {slug}: {exc}", file=sys.stderr)
            continue
        domain = scenario.get("domain") or "power_grid"
        family = scenario.get("family", "unknown")
        mode = scenario.get("difficulty_mode", "unknown")
        level = scenario.get("difficulty_level", "unknown")
        sid = scenario.get("seed_id") or Path(slug).stem
        out = out_root / domain / family / mode / level / f"{sid}.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        # Re-serialize from the loaded dict so every file is canonical YAML.
        import yaml  # noqa: PLC0415
        out.write_text(yaml.safe_dump(scenario, sort_keys=False, allow_unicode=True), encoding="utf-8")
        manifest[f"scenarios/{domain}/{family}/{mode}/{level}/{sid}.yaml"] = sha256_file(out)
        n += 1
    return n


def build_release(manifest: dict) -> int:
    """Copy the v0.51.0 release manifest + suite JSONs into data/release/."""
    src = REPO / "release" / RELEASE_ID
    dst = DATA / "release"
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for name in ("manifest.json", "registry.json", "primary_suite.json", "core_suite.json"):
        p = src / name
        if not p.exists():
            continue
        shutil.copy2(p, dst / name)
        manifest[f"release/{name}"] = sha256_file(p)
        n += 1
    return n


# works/ subdir -> data/backends/<name> mapping. Names match what backends read
# (e.g. opendss_ieee13 reads works/OpenDSS-IEEE13; pglib_uc reads works/pglib-uc).
# The M5, VRPLIB, and RESCO mirrors are required by released rows and the
# current Protocol-2.1 candidate; omitting them makes a bundle look complete
# while leaving those samples unable to start.
BACKEND_DATA_MAP = {
    "pglib-uc": "pglib_uc",
    "RTS-GMLC": "rts_gmlc",
    "PGLib-OPF": "pglib_opf",
    "OpenDSS-IEEE13": "opendss_ieee13",
    "JSPLIB-Instances": "jsplib",
    "PyVRP-Instances": "pyvrp_instances",
    "OR-Gym": "or_gym",
    "M5": "m5",
    "VRPLIB": "vrplib",
    "RESCO": "resco",
    "sumo_ingolstadt": "sumo_ingolstadt",
    "nrel-microgrid": "nrel_microgrid",
}


def build_backends(manifest: dict) -> int:
    """Copy each present works/ backend-data dir into data/backends/<name>."""
    dst_root = DATA / "backends"
    n_files = 0
    for works_name, data_name in BACKEND_DATA_MAP.items():
        src = WORKS / works_name
        if not src.exists():
            print(f"  SKIP data/backends/{data_name} (works/{works_name} not present)")
            continue
        dst = dst_root / data_name
        n = copy_tree(src, dst, manifest, f"backends/{data_name}")
        print(f"  data/backends/{data_name}: {n} files")
        n_files += n
    return n_files


def link_backends() -> int:
    """Symlink works/<name> -> data/backends/<name> so backend code reading
    ``works/`` resolves to the curated ``data/`` copy. Only links dirs that
    exist in data/backends/ and aren't already real dirs in works/."""
    n = 0
    for works_name, data_name in BACKEND_DATA_MAP.items():
        data_dst = DATA / "backends" / data_name
        works_link = WORKS / works_name
        if not data_dst.exists():
            continue
        if works_link.exists() and works_link.is_symlink():
            works_link.unlink()  # refresh stale symlink
        if works_link.exists():
            continue  # real dir in works/, leave it
        works_link.symlink_to(data_dst.resolve())
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--link-backends", action="store_true",
                    help="also symlink works/<name> -> data/backends/<name>")
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}

    print("=== data/scenarios/ (351 core YAMLs) ===")
    n_scen = build_scenarios(manifest)
    print(f"  {n_scen} scenario YAMLs")

    print("=== data/release/ (v0.51.0 manifest + suites) ===")
    n_rel = build_release(manifest)
    print(f"  {n_rel} release artifacts")

    print("=== data/backends/ (backend runtime data) ===")
    n_be = build_backends(manifest)
    print(f"  {n_be} backend data files")

    if args.link_backends:
        n_link = link_backends()
        print(f"=== linked {n_link} works/ -> data/backends/ symlinks ===")

    manifest_path = DATA / "MANIFEST.json"
    manifest_obj = {
        "release_id": RELEASE_ID,
        "n_scenarios": n_scen,
        "n_release_artifacts": n_rel,
        "n_backend_files": n_be,
        "files": dict(sorted(manifest.items())),
    }
    manifest_path.write_text(json.dumps(manifest_obj, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== wrote {manifest_path} ({len(manifest)} files, {n_scen} scenarios) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
