#!/usr/bin/env python3
"""Download NSRDB TMY + PVWatts hourly data for Boston/Seattle from NREL API.

Usage:
    python scripts/download_nrel_microgrid_sources.py

Requires: network access to developer.nlr.gov (the NREL→NLR host migrated
2026-05-29; see domains/microgrid/seeds/source_locks.py). The frozen
nsrdb.nrel.gov citation strings in the release manifests are unchanged.
Produces: works/nrel-microgrid/sources/{boston_ma,seattle_wa}_{nsrdb_tmy.csv,pvwatts.json}
"""
import hashlib
import json
import os
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

# API key comes from the environment; never hardcode a live key. Register a free
# key at https://developer.nlr.gov/signup/ and export NREL_API_KEY, or fall back
# to the rate-limited public DEMO_KEY for a quick smoke test.
API_KEY = os.environ.get('NREL_API_KEY', 'DEMO_KEY')
REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'works' / 'nrel-microgrid' / 'sources'
OUT.mkdir(parents=True, exist_ok=True)

# Live fetch host. NREL was renamed National Laboratory of the Rockies (NLR) on
# 2026-05-29; the developer.nrel.gov host was retired in favour of
# developer.nlr.gov (same NSRDB/PVWatts APIs, same api.data.gov auth). The
# frozen nsrdb.nrel.gov citation URLs in source_locks.py / the release
# manifests are byte-identity-locked and deliberately NOT changed here.
NREL_API_HOST = os.environ.get('NREL_API_HOST', 'https://developer.nlr.gov')

SITES = {
    'boston_ma':  {'lat': 42.36, 'lon': -71.06, 'nsrdb_site': '1433277'},
    'seattle_wa': {'lat': 47.61, 'lon': -122.33, 'nsrdb_site': '1632347'},
}

def _utc_now():
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace('+00:00', 'Z')

def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

# ── Step 1: PVWatts v8 hourly ──
print('=== PVWatts v8 Hourly ===')
for site_id, coords in SITES.items():
    url = (
        f'{NREL_API_HOST}/api/pvwatts/v8.json'
        f'?api_key={API_KEY}&lat={coords["lat"]}&lon={coords["lon"]}'
        '&system_capacity=100&azimuth=180&tilt=' + str(coords['lat']) +
        '&array_type=1&module_type=1&losses=10&dataset=nsrdb&timeframe=hourly'
    )
    json_path = OUT / f'{site_id}_pvwatts.json'
    req = urllib.request.Request(url, headers={'User-Agent': 'OPERATE/0.58'})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    if 'errors' in data and data['errors']:
        print(f'{site_id}: API errors: {data["errors"]}')
        continue
    ac = data.get('outputs', {}).get('ac', [])
    if not ac:
        print(f'{site_id}: no AC output')
        continue
    data['_downloaded_at'] = _utc_now()
    data['_sha256'] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    json_path.write_text(json.dumps(data, indent=2))
    print(f'{site_id}: {len(ac)}h, annual={sum(ac):.0f}Wh, peak={max(ac):.0f}W')

# ── Step 2: NSRDB PSM3 TMY ──
print('\n=== NSRDB PSM3 TMY ===')
for site_id, coords in SITES.items():
    url = (
        f'{NREL_API_HOST}/api/nsrdb/v2/solar/psm3-download.csv'
        f'?api_key={API_KEY}&wkt=POINT({coords["lon"]}+{coords["lat"]})'
        '&names=tmy&leap_day=false&interval=60&utc=true'
        '&full_name=dt_sched_bench&email=dt_sched_bench@example.com'
        '&mailing_list=false'
        '&attributes=ghi,dni,dhi,air_temperature,wind_speed,surface_albedo,solar_zenith_angle'
    )
    csv_path = OUT / f'{site_id}_nsrdb_tmy.csv'
    req = urllib.request.Request(url, headers={'User-Agent': 'OPERATE/0.58'})
    # NSRDB PSM3 is a restricted-access API that may 404 on some keys/hosts
    # (see .hl/failed_directions.md: microgrid acquisition is narrowed to the
    # two PVWatts hourly JSON files; the raw NSRDB TMY CSV is secondary).
    # Treat a fetch failure as a non-fatal warning so the PVWatts source-lock
    # (the load-bearing artifact) still completes.
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            content = resp.read()
        csv_path.write_bytes(content)
        lines = [line for line in content.decode('utf-8').split('\n') if line.strip()]
        print(f'{site_id}: {len(lines)} lines, sha256={sha256_file(csv_path)}')
    except Exception as exc:  # noqa: BLE001
        print(f'{site_id}: NSRDB PSM3 fetch failed ({type(exc).__name__}: {exc}); '
              'PVWatts JSON is the load-bearing source-lock, continuing without TMY CSV')

# ── Step 3: Source lock ──
print('\n=== Source Lock ===')
source_lock = {
    'generated_at': _utc_now(),
    'api_key_note': 'NREL API key used at download time; key not stored in output',
    'sources': {},
}
for site_id, coords in SITES.items():
    entry = {}
    pv_path = OUT / f'{site_id}_pvwatts.json'
    nsrdb_path = OUT / f'{site_id}_nsrdb_tmy.csv'
    if pv_path.exists():
        entry['pvwatts_v8'] = {
            'url': f'{NREL_API_HOST}/api/pvwatts/v8.json?lat={coords["lat"]}&lon={coords["lon"]}&system_capacity=100',
            'local_file': str(pv_path.relative_to(REPO)),
            'sha256': sha256_file(pv_path),
            'license': 'NREL/DOE public data',
        }
    if nsrdb_path.exists():
        entry['nsrdb_psm3_tmy'] = {
            'url': f'{NREL_API_HOST}/api/nsrdb/v2/solar/psm3-download.csv?wkt=POINT({coords["lon"]}+{coords["lat"]})&names=tmy',
            'local_file': str(nsrdb_path.relative_to(REPO)),
            'sha256': sha256_file(nsrdb_path),
            'license': 'NREL/DOE public data',
        }
    source_lock['sources'][site_id] = entry

lock_path = OUT / 'source_lock_nrel.json'
lock_path.write_text(json.dumps(source_lock, indent=2, ensure_ascii=False))
print(f'Source lock: {lock_path}')
print('Done!')
