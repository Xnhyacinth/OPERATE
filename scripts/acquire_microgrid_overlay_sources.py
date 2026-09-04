#!/usr/bin/env python3
"""Acquire public OEDI, PVWatts, and URDB inputs for Microgrid overlays."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "works/nrel-microgrid"
OEDI_ROOT = (
    "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/"
    "end-use-load-profiles-for-us-building-stock/2021/"
    "comstock_amy2018_release_1/timeseries_aggregates/by_state"
)
PVWATTS_URL = "https://developer.nlr.gov/api/pvwatts/v8.json"
URDB_URL = "https://api.openei.org/utility_rates"

SITES: dict[str, dict[str, Any]] = {
    "phoenix_az": {"lat": 33.45, "lon": -112.07, "state": "AZ"},
    "denver_co": {"lat": 39.74, "lon": -104.99, "state": "CO"},
    "boston_ma": {"lat": 42.36, "lon": -71.06, "state": "MA"},
    "seattle_wa": {"lat": 47.61, "lon": -122.33, "state": "WA"},
    "miami_fl": {"lat": 25.76, "lon": -80.19, "state": "FL"},
    "minneapolis_mn": {"lat": 44.98, "lon": -93.27, "state": "MN"},
    "chicago_il": {"lat": 41.88, "lon": -87.63, "state": "IL"},
    "atlanta_ga": {"lat": 33.749, "lon": -84.388, "state": "GA"},
    "sacramento_ca": {"lat": 38.582, "lon": -121.494, "state": "CA"},
    "portland_or": {"lat": 45.515, "lon": -122.679, "state": "OR"},
    "salt_lake_city_ut": {"lat": 40.761, "lon": -111.891, "state": "UT"},
    "albuquerque_nm": {"lat": 35.084, "lon": -106.65, "state": "NM"},
    "las_vegas_nv": {"lat": 36.17, "lon": -115.14, "state": "NV"},
    "nashville_tn": {"lat": 36.163, "lon": -86.782, "state": "TN"},
    "raleigh_nc": {"lat": 35.78, "lon": -78.639, "state": "NC"},
    "columbus_oh": {"lat": 39.961, "lon": -82.999, "state": "OH"},
    "tucson_az": {"lat": 32.223, "lon": -110.975, "state": "AZ"},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_urls(site: str, *, api_key: str = "DEMO_KEY") -> dict[str, str]:
    spec = SITES[site]
    state = str(spec["state"])
    oedi_name = f"{state.lower()}-largeoffice.csv"
    pvwatts_query = urlencode(
        {
            "api_key": api_key,
            "lat": spec["lat"],
            "lon": spec["lon"],
            "system_capacity": 1,
            "azimuth": 180,
            "tilt": spec["lat"],
            "array_type": 1,
            "module_type": 1,
            "losses": 10,
            "dataset": "nsrdb",
            "timeframe": "hourly",
        }
    )
    urdb_query = urlencode(
        {
            "version": "latest",
            "format": "json",
            "api_key": api_key,
            "lat": spec["lat"],
            "lon": spec["lon"],
            "radius": 50,
            "sector": "Commercial",
            "approved": "true",
            "detail": "full",
            "limit": 500,
        }
    )
    return {
        "oedi": f"{OEDI_ROOT}/state={state}/{oedi_name}",
        "pvwatts": f"{PVWATTS_URL}?{pvwatts_query}",
        "urdb": f"{URDB_URL}?{urdb_query}",
    }


def _download(url: str, target: Path, *, force: bool) -> None:
    if target.is_file() and not force:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.part")
    request = Request(url, headers={"User-Agent": "OPERATE/0.58"})
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=120) as response, temporary.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            temporary.replace(target)
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            temporary.unlink(missing_ok=True)
            if attempt < 2:
                time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def _validate_json(path: Path, *, output_key: str) -> None:
    body = json.loads(path.read_text(encoding="utf-8"))
    if body.get("errors"):
        raise ValueError(f"{path}: API errors: {body['errors']}")
    values = body.get("outputs", {}).get(output_key) if output_key else body.get("items")
    if not values:
        raise ValueError(f"{path}: missing non-empty {output_key or 'items'}")


def acquire(*, root: Path, sites: list[str], api_key: str, force: bool) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for site in sites:
        spec = SITES[site]
        state = str(spec["state"]).lower()
        urls = source_urls(site, api_key=api_key)
        paths = {
            "oedi": root / "sources/oedi" / f"{state}-largeoffice.csv",
            "pvwatts": root / "sources/nsrdb" / f"pvwatts_{site}_hourly.json",
            "urdb": root / "sources/openei" / f"urdb_{site}_commercial.json",
        }
        try:
            for key in ("oedi", "pvwatts", "urdb"):
                _download(urls[key], paths[key], force=force)
            _validate_json(paths["pvwatts"], output_key="ac")
            _validate_json(paths["urdb"], output_key="")
        except Exception as exc:  # noqa: BLE001
            failures.append({"site": site, "error": f"{type(exc).__name__}: {exc}"})
            continue
        rows.append(
            {
                "site": site,
                "files": {
                    key: {
                        "path": str(path.relative_to(REPO_ROOT)),
                        "url": urls[key].replace(api_key, "<redacted>"),
                        "sha256": _sha256(path),
                        "bytes": path.stat().st_size,
                    }
                    for key, path in paths.items()
                },
            }
        )
    report = {
        "schema_version": "0.1",
        "retrieved_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "sources": {
            "oedi": "OEDI ComStock AMY2018 release 1",
            "pvwatts": "NLR PVWatts v8 / NSRDB TMY 2020",
            "urdb": "OpenEI URDB latest approved commercial rates",
        },
        "sites": rows,
        "failures": failures,
    }
    lock_path = root / "sources/source_lock_vnext.json"
    lock_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--sites", nargs="*", choices=sorted(SITES))
    parser.add_argument("--api-key", default="DEMO_KEY")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    sites = args.sites or list(SITES)
    report = acquire(root=args.root, sites=sites, api_key=args.api_key, force=args.force)
    print(
        f"acquired {len(report['sites'])} Microgrid source bundles; "
        f"{len(report['failures'])} incomplete"
    )
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
