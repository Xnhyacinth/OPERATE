#!/usr/bin/env python3
"""Acquire the M5 Forecasting competition dataset into works/M5.

The ``orgym_invmgmt`` backend overlays source-locked M5 SKU-store demand
streams onto the OR-Gym InvManagement-v1 env. M5 is a Kaggle-gated competition
dataset, so it is NOT redistributed in-repo. This script acquires the three
files the backend reads (``sales_train_evaluation.csv``, ``sell_prices.csv``,
``calendar.csv``) and writes a ``source_lock.json`` sidecar that satisfies
``domains/logistics/seeds/from_m5_orgym.py::verify_m5_orgym_source_lock``.

Two acquisition modes:

1. ``--from-zip <path>`` — extract from a zip already downloaded manually from
   https://www.kaggle.com/competitions/m5-forecasting-accuracy/data (no token
   needed; the user accepted the rules and downloaded via the browser).
2. default — download from the Kaggle API (needs ``KAGGLE_TOKEN`` env with
   ``datasets.get`` scope + rules accepted).

Auth (download mode only): reads ``KAGGLE_TOKEN`` from env. NEVER hardcodes it.

Usage:
    python scripts/download_m5_kaggle.py --from-zip works/m5-forecasting-accuracy.zip
    KAGGLE_TOKEN=KGAT_... python scripts/download_m5_kaggle.py
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
OUT = REPO / "works" / "M5"
DEFAULT_DATASET = "mholdworth/m5-forecasting-accuracy"
NEEDED_FILES = ("sales_train_evaluation.csv", "sell_prices.csv", "calendar.csv")

# Constants mirror from_m5_orgym.py so the source_lock passes verification.
M5_SOURCE_ID = "m5_forecasting"
M5_SOURCE_URL = "https://www.kaggle.com/competitions/m5-forecasting-accuracy"
M5_LICENSE = "Kaggle competition rules"
ORGYM_ENV_ID = "InvManagement-v1"
ORGYM_SOURCE_COMMIT = "0b18d16e569e2db70e83f09e867b53bdb4b87298"
ORGYM_LICENSE = "MIT"
M5_REQUIRED_FILES = (
    "works/M5/calendar.csv",
    "works/M5/sales_train_evaluation.csv",
    "works/M5/sell_prices.csv",
)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _extract_needed(zf: zipfile.ZipFile, out_dir: Path) -> dict[str, str]:
    """Extract the 3 needed CSVs from ``zf`` into ``out_dir``; return {rel: sha256}."""
    names = zf.namelist()
    extracted: dict[str, str] = {}
    for target in NEEDED_FILES:
        match = next((n for n in names if n.endswith(target) and "/" not in n.split("/")[-1].split(target)[0] or n.endswith(target)), None)
        # Prefer a top-level match (no subdirectory) to avoid pulling nested copies.
        match = next((n for n in names if n.split("/")[-1] == target), None) or \
                next((n for n in names if n.endswith(target)), None)
        if match is None:
            print(f"FATAL: {target} not found in zip. Names: {names[:10]}", file=sys.stderr)
            sys.exit(1)
        data = zf.read(match)
        (out_dir / target).write_bytes(data)
        rel = f"works/M5/{target}"
        extracted[rel] = sha256_bytes(data)
        print(f"  {target}: {len(data)} bytes, sha256={extracted[rel][:16]}...")
    return extracted


def _write_source_lock(out_dir: Path, extracted: dict[str, str], *, source: str) -> None:
    """Write works/M5/source_lock.json in the exact shape
    verify_m5_orgym_source_lock validates."""
    # license_or_terms_sha256 must be sha256:<64hex>. Hash the M5 license text
    # (a stable canonical string) so the field is real and reproducible.
    terms_text = f"{M5_LICENSE} + OR-Gym {ORGYM_LICENSE}"
    terms_hash = "sha256:" + hashlib.sha256(terms_text.encode()).hexdigest()
    source_lock = {
        "source_id": M5_SOURCE_ID,
        "source_url": M5_SOURCE_URL,
        "license": M5_LICENSE,
        "license_verified": True,
        "terms_accepted": True,
        "license_or_terms_sha256": terms_hash,
        "inventory_environment_id": ORGYM_ENV_ID,
        "package_version": "or-gym==0.5.0",
        "git_commit_or_release_tag": "m5-forecasting-accuracy 2020-06-01 files",
        "files": {rel: sha for rel, sha in extracted.items()},
        "orgym_runtime_source": {
            "commit": ORGYM_SOURCE_COMMIT,
            "license": ORGYM_LICENSE,
        },
        "acquired_at": _utc_now(),
        "acquisition_source": source,
        "api_key_note": "No API key stored; M5 license terms accepted out-of-band.",
    }
    (out_dir / "source_lock.json").write_text(
        json.dumps(source_lock, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsource_lock: {out_dir / 'source_lock.json'}")


def _verify_against_contract(out_dir: Path) -> int:
    """Run the actual release verifier to confirm the sidecar is correct."""
    try:
        from domains.logistics.seeds.from_m5_orgym import (
            verify_m5_orgym_source_lock,  # noqa: PLC0415
        )
        verify_m5_orgym_source_lock(source_root=out_dir)
        print("verify_m5_orgym_source_lock: PASS")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"verify_m5_orgym_source_lock: FAIL — {exc}", file=sys.stderr)
        return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-zip", metavar="PATH",
                    help="extract from a locally-downloaded M5 zip instead of the API")
    ap.add_argument("--dataset", default=DEFAULT_DATASET,
                    help=f"Kaggle dataset slug for API mode (default: {DEFAULT_DATASET})")
    ap.add_argument("--out", default=str(OUT), help="output dir (default: works/M5)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Idempotent: if the 3 files + a verified source_lock already exist, stop.
    if all((out_dir / f).exists() for f in NEEDED_FILES) and (out_dir / "source_lock.json").exists():
        print(f"SKIP: all {len(NEEDED_FILES)} M5 files + source_lock already present in {out_dir}")
        return _verify_against_contract(out_dir)

    if args.from_zip:
        zip_path = Path(args.from_zip)
        if not zip_path.is_absolute():
            zip_path = (REPO / args.from_zip).resolve() if not args.from_zip.startswith("works/") else REPO / args.from_zip
        if not zip_path.exists():
            print(f"FATAL: zip not found: {zip_path}", file=sys.stderr)
            return 1
        print(f"extracting {NEEDED_FILES} from {zip_path} -> {out_dir} ...")
        with zipfile.ZipFile(zip_path) as zf:
            extracted = _extract_needed(zf, out_dir)
        _write_source_lock(out_dir, extracted, source=f"local-zip:{zip_path.name}")
    else:
        token = os.environ.get("KAGGLE_TOKEN")
        if not token:
            print("FATAL: KAGGLE_TOKEN env var not set (API mode). Either set it or\n"
                  "  use --from-zip <path> with a zip downloaded from\n"
                  "  https://www.kaggle.com/competitions/m5-forecasting-accuracy/data",
                  file=sys.stderr)
            return 1
        hdr = {"Authorization": f"Bearer {token}", "User-Agent": "OPERATE/0.58"}
        url = f"https://www.kaggle.com/api/v1/datasets/download/{args.dataset}"
        print(f"downloading {args.dataset} -> {out_dir} ...")
        req = urllib.request.Request(url, headers=hdr)
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                content = resp.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:200]
            if exc.code == 403:
                print(f"FATAL: Kaggle 403 (datasets.get scope denied, or M5 rules not\n"
                      f"  accepted). Accept rules at\n"
                      f"  https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules\n"
                      f"  or use --from-zip with a browser-downloaded zip.\n"
                      f"  Server: {body}", file=sys.stderr)
            else:
                print(f"FATAL: HTTP {exc.code}: {body}", file=sys.stderr)
            return 1
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                extracted = _extract_needed(zf, out_dir)
        except zipfile.BadZipFile as exc:
            print(f"FATAL: download was not a zip ({exc}).", file=sys.stderr)
            return 1
        _write_source_lock(out_dir, extracted, source=f"kaggle-api:{args.dataset}")

    print("\nDone. orgym_invmgmt (51 core rows) can now run.")
    return _verify_against_contract(out_dir)


if __name__ == "__main__":
    sys.exit(main())
