"""
domains.microgrid.seeds.from_nrel_microgrid — baked NSRDB/OEDI/HRRR overlays.

Reads the **baked** (version-locked) NSRDB solar, OEDI building-load and
HRRR forecast-error series into deterministic per-tick profile arrays
(``load_mw`` / ``pv_mw`` / ``wind_mw`` / ``price`` / ``forecast_error``).
These are the **released** pymgrid-family provenance series (NOT pymgrid's
bundled/synthetic series; spec §3) and the rooftop-PV series for the LV
family.

Anchoring posture (spec §10 ``anchored`` rung):

- If a baked ``.npz`` is anchored under ``works/`` the loader reads it and
  records the file + checksum in provenance.
- If nothing is anchored (network blocked at build time) the loader
  synthesizes a **deterministic** physically-shaped profile (diurnal load,
  solar bell curve, diurnal price) keyed to ``(location, year, seed)`` so
  the build is fully reproducible offline. The parser path is exercised the
  moment a real ``.npz`` is dropped in; tests that need a real anchor
  ``pytest.skip`` when none is present.

No live fetch ever happens at runtime (counterfactual replay determinism,
spec §9).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .source_locks import SOURCE_LOCKS

# Where a baked overlay would live if anchored. Mirrors the power-grid
# ``works/pglib-uc`` / logistics ``works/PyVRP-Instances`` convention.
NREL_MICROGRID_ROOT_REL = "works/nrel-microgrid"

# Representative baked sites (location keys). A real anchor would carry the
# NSRDB site id + OEDI building profile id; offline we synthesize per key.
#
# ``peak_load_mw`` is an aggregate-feeder MW-equivalent proxy (community /
# campus microgrid aggregate), intentionally scaled into the hundreds so the
# EMS imbalance feeding ``balance_error_mw`` lands the scorer's hard-coded
# literals meaningfully (spec §7: ``balance_error_mw>200`` = real
# catastrophe, ``>100`` = stress). pymgrid module sizing on the optional
# cross-check path uses the same scale.
BAKED_SITES: dict[str, dict[str, Any]] = {
    "phoenix_az": {"nsrdb_site": "1433277", "lat": 33.45, "peak_load_mw": 340.0},
    "denver_co": {"nsrdb_site": "1212448", "lat": 39.74, "peak_load_mw": 260.0},
    "boston_ma": {"nsrdb_site": "1493281", "lat": 42.36, "peak_load_mw": 300.0},
    "seattle_wa": {"nsrdb_site": "1632347", "lat": 47.61, "peak_load_mw": 220.0},
    "miami_fl": {"nsrdb_site": "1060699", "lat": 25.76, "peak_load_mw": 320.0},
    "minneapolis_mn": {"nsrdb_site": "778153", "lat": 44.98, "peak_load_mw": 280.0},
    "chicago_il": {"nsrdb_site": "903523", "lat": 41.88, "peak_load_mw": 250.0},
    "atlanta_ga": {"nsrdb_site": "970013", "lat": 33.749, "peak_load_mw": 290.0},
    "sacramento_ca": {"nsrdb_site": "131636", "lat": 38.582, "peak_load_mw": 270.0},
    "portland_or": {"nsrdb_site": "215938", "lat": 45.515, "peak_load_mw": 230.0},
    "salt_lake_city_ut": {
        "nsrdb_site": "1404985",
        "lat": 40.761,
        "peak_load_mw": 240.0,
    },
    "albuquerque_nm": {"nsrdb_site": "690725", "lat": 35.084, "peak_load_mw": 235.0},
    "las_vegas_nv": {"nsrdb_site": "1164573", "lat": 36.170, "peak_load_mw": 310.0},
    "nashville_tn": {"nsrdb_site": "912402", "lat": 36.163, "peak_load_mw": 265.0},
    "raleigh_nc": {"nsrdb_site": "937815", "lat": 35.780, "peak_load_mw": 255.0},
    "columbus_oh": {"nsrdb_site": "1164573", "lat": 39.961, "peak_load_mw": 245.0},
    "tucson_az": {"nsrdb_site": "1060699", "lat": 32.223, "peak_load_mw": 250.0},
}

REQUIRED_BAKED_OVERLAY_ARRAYS = ("load_mw", "pv_mw", "wind_mw", "price")
REQUIRED_BAKED_OVERLAY_SIDECAR_FIELDS = (
    "url",
    "license",
    "lock_strategy",
    "sha256",
)

# Documented forecast-error regimes (bias, sigma) for the noised forecast.
FORECAST_REGIMES: list[dict[str, float]] = [
    {"bias": 0.00, "sigma": 0.05},  # well-calibrated
    {"bias": 0.10, "sigma": 0.12},  # under-forecast renewables
    {"bias": -0.08, "sigma": 0.10},  # over-forecast renewables
    {"bias": 0.05, "sigma": 0.20},  # high-variance (storm front)
]


def _runtime_lock_signature(*, pymgrid_version: str | None) -> str:
    """Return the spec §3 ``lock_strategy`` string.

    ``"nsrdb=<release> oedi=<doi> pymgrid=<ver|unavailable>"`` — pymgrid
    absent records ``pymgrid=unavailable``.
    """
    nsrdb = SOURCE_LOCKS["nsrdb"].data_release or "unknown"
    oedi = SOURCE_LOCKS["oedi"].data_release or "unknown"
    pm = pymgrid_version or "unavailable"
    return f"nsrdb={nsrdb} oedi={oedi} pymgrid={pm}"


def _pymgrid_version() -> str | None:
    try:
        import importlib.metadata as m

        return m.version("pymgrid")
    except Exception:
        return None


def _baked_npz_path(site: str) -> Path:
    # parents[3] is the repository root (same convention as the
    # power-grid case_file resolution).
    return Path(__file__).resolve().parents[3] / NREL_MICROGRID_ROOT_REL / f"{site}.npz"


def _baked_provenance_sidecar_path(npz_path: Path) -> Path:
    return npz_path.with_name(f"{npz_path.stem}.provenance.json")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdefABCDEF" for ch in value)
    )


def _valid_public_source_url(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def baked_overlay_provenance_report(path: Path) -> dict[str, Any]:
    """Validate the file-local source-lock sidecar for a baked overlay."""
    sidecar = _baked_provenance_sidecar_path(path)
    if not sidecar.exists():
        return {
            "path": str(sidecar),
            "exists": False,
            "valid": False,
            "validation_errors": ["missing_provenance_sidecar"],
            "metadata": {},
        }

    try:
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - exact JSON error varies
        return {
            "path": str(sidecar),
            "exists": True,
            "valid": False,
            "validation_errors": [f"invalid_provenance_sidecar:{type(exc).__name__}"],
            "metadata": {},
        }
    if not isinstance(raw, dict):
        return {
            "path": str(sidecar),
            "exists": True,
            "valid": False,
            "validation_errors": ["invalid_provenance_sidecar:not_object"],
            "metadata": {},
        }

    metadata = dict(raw)
    errors: list[str] = []
    for field in REQUIRED_BAKED_OVERLAY_SIDECAR_FIELDS:
        if not metadata.get(field):
            errors.append(f"missing_provenance_field:{field}")
    if metadata.get("url") and not _valid_public_source_url(metadata.get("url")):
        errors.append("invalid_provenance_field:url")
    if not (metadata.get("data_release") or metadata.get("version")):
        errors.append("missing_provenance_field:data_release_or_version")

    expected_site = path.stem
    expected_site_spec = BAKED_SITES.get(expected_site)
    actual_site = metadata.get("site")
    if not actual_site:
        errors.append("missing_provenance_field:site")
    elif str(actual_site) != expected_site:
        errors.append(f"provenance_site_mismatch:{actual_site}!={expected_site}")

    source_ids = metadata.get("source_ids")
    if not isinstance(source_ids, dict):
        errors.append("missing_provenance_field:source_ids")
    elif expected_site_spec is not None:
        expected_nsrdb = str(expected_site_spec["nsrdb_site"])
        actual_nsrdb = source_ids.get("nsrdb_site")
        if not actual_nsrdb:
            errors.append("missing_provenance_field:source_ids.nsrdb_site")
        elif str(actual_nsrdb) != expected_nsrdb:
            errors.append(
                f"provenance_source_id_mismatch:nsrdb_site:{actual_nsrdb}!={expected_nsrdb}"
            )

    expected_sha = metadata.get("sha256")
    if expected_sha and not _valid_sha256(expected_sha):
        errors.append("invalid_provenance_field:sha256")
    elif expected_sha and path.exists():
        actual_sha = _sha256_file(path)
        if str(expected_sha).lower() != actual_sha.lower():
            errors.append(f"sha256_mismatch:{expected_sha}!={actual_sha}")

    return {
        "path": str(sidecar),
        "exists": True,
        "valid": not errors,
        "validation_errors": errors,
        "metadata": metadata,
    }


def baked_overlay_validation_errors(
    path: Path, *, min_horizon_ticks: int = 24, require_sidecar: bool = False
) -> list[str]:
    """Return release-gate validation errors for a baked overlay.

    The dev loader remains tolerant so local experiments can use partial files,
    but release packaging must prove the anchored overlay is complete enough to
    support both 24h EMS and 6h LV families.
    """
    if not path.exists():
        return ["missing_file"]
    if min_horizon_ticks <= 0:
        return ["invalid_min_horizon_ticks"]

    try:
        import numpy as np  # local import; numpy is in tree

        data = np.load(path, allow_pickle=False)
    except Exception as exc:  # pragma: no cover - exact numpy error varies
        return [f"invalid_npz:{type(exc).__name__}"]

    errors: list[str] = []
    for name in REQUIRED_BAKED_OVERLAY_ARRAYS:
        if name not in data:
            errors.append(f"missing_array:{name}")
            continue
        try:
            arr = np.asarray(data[name], dtype=float).ravel()
        except Exception as exc:  # pragma: no cover - exact numpy error varies
            errors.append(f"invalid_array:{name}:{type(exc).__name__}")
            continue
        if len(arr) < min_horizon_ticks:
            errors.append(f"short_array:{name}:{len(arr)}<{min_horizon_ticks}")
            continue
        window = arr[:min_horizon_ticks]
        if not np.all(np.isfinite(window)):
            errors.append(f"non_finite_array:{name}")
        if name in {"load_mw", "pv_mw", "wind_mw"} and np.any(window < 0):
            errors.append(f"negative_array:{name}")
        if name == "price" and np.any(window <= 0):
            errors.append(f"non_positive_array:{name}")
    if require_sidecar:
        errors.extend(
            baked_overlay_provenance_report(path).get("validation_errors") or []
        )
    return errors


def site_is_anchored(
    site: str, *, strict: bool = False, min_horizon_ticks: int = 24
) -> bool:
    path = _baked_npz_path(site)
    if not strict:
        return path.exists()
    return not baked_overlay_validation_errors(
        path, min_horizon_ticks=min_horizon_ticks, require_sidecar=True
    )


def baked_overlay_provenance_files(site: str) -> list[str]:
    """Return provenance handles for a baked overlay or offline synthesis."""
    npz_path = _baked_npz_path(site)
    if not npz_path.exists():
        return [f"<offline-synthesized:{site}>"]

    files = [f"{NREL_MICROGRID_ROOT_REL}/{site}.npz"]
    if baked_overlay_provenance_report(npz_path).get("valid"):
        files.append(f"{NREL_MICROGRID_ROOT_REL}/{site}.provenance.json")
    return files


def _det_unit(seed: int, key: str) -> float:
    """Deterministic value in [0,1) from (seed, key)."""
    body = f"{int(seed)}|{key}".encode()
    return int.from_bytes(hashlib.sha256(body).digest()[:6], "big") / float(1 << 48)


def load_overlay(
    site: str,
    *,
    horizon_ticks: int,
    seed: int,
    forecast_regime_idx: int,
    pv_scale: float = 1.0,
    wind_scale: float = 0.0,
    start_index: int = 0,
) -> dict[str, Any]:
    """Return a baked (or deterministically synthesized) overlay.

    Output keys: ``load_mw``, ``pv_mw``, ``wind_mw``, ``price``,
    ``forecast_bias``, ``forecast_sigma``, ``anchored``, ``files``.

    All series have length ``horizon_ticks``. The synthesis is fully
    deterministic in ``(site, horizon_ticks, seed, forecast_regime_idx,
    pv_scale, wind_scale)``.
    """
    meta = BAKED_SITES.get(site, BAKED_SITES["denver_co"])
    peak = float(meta["peak_load_mw"])
    regime = FORECAST_REGIMES[forecast_regime_idx % len(FORECAST_REGIMES)]

    if start_index < 0:
        raise ValueError("start_index must be non-negative")

    npz_path = _baked_npz_path(site)
    if npz_path.exists():  # pragma: no cover - exercised only when anchored
        overlay = _read_npz(npz_path, horizon_ticks, start_index=start_index)
        overlay["anchored"] = True
        overlay["files"] = baked_overlay_provenance_files(site)
        overlay["forecast_bias"] = regime["bias"]
        overlay["forecast_sigma"] = regime["sigma"]
        overlay["start_index"] = start_index
        return overlay

    # Deterministic offline synthesis (physically shaped).
    load_mw: list[float] = []
    pv_mw: list[float] = []
    wind_mw: list[float] = []
    price: list[float] = []
    for t in range(max(1, horizon_ticks)):
        # Diurnal load: evening peak around 0.75*horizon.
        peak_tick = max(1, int(horizon_ticks * 0.75))
        diurnal = 0.55 + 0.45 * math.sin(math.pi * t / max(1, peak_tick))
        diurnal = max(0.35, min(1.0, diurnal))
        jitter = 0.05 * (_det_unit(seed, f"load|{t}") - 0.5)
        load_mw.append(round(peak * (diurnal + jitter), 5))

        # Solar bell curve (zero at night, peak at midday ~ 0.5*horizon).
        solar = max(0.0, math.sin(math.pi * t / max(1, horizon_ticks)))
        pv_mw.append(round(peak * pv_scale * solar, 5))

        # Wind: noisier, non-zero baseline.
        w = 0.4 + 0.5 * _det_unit(seed, f"wind|{t}")
        wind_mw.append(round(peak * wind_scale * w, 5))

        # Grid-import price: low overnight, peak in evening.
        p_diurnal = 0.6 + 0.6 * math.sin(math.pi * (t - 1) / max(1, peak_tick))
        price.append(round(40.0 * max(0.4, p_diurnal), 4))

    return {
        "load_mw": load_mw,
        "pv_mw": pv_mw,
        "wind_mw": wind_mw,
        "price": price,
        "forecast_bias": regime["bias"],
        "forecast_sigma": regime["sigma"],
        "anchored": False,
        "files": [f"<offline-synthesized:{site}>"],
        "start_index": start_index,
    }


def _read_npz(
    path: Path, horizon_ticks: int, *, start_index: int = 0
) -> dict[str, Any]:  # pragma: no cover
    """Read a baked NSRDB/OEDI overlay ``.npz`` into per-tick lists."""
    import numpy as np  # local import; numpy is in tree

    data = np.load(path, allow_pickle=False)

    def _slice(name: str, default: float) -> list[float]:
        if name in data:
            arr = np.asarray(data[name], dtype=float).ravel()
            stop = start_index + horizon_ticks
            if stop > len(arr):
                raise ValueError(
                    f"{path.name}:{name} source window [{start_index}:{stop}] "
                    f"exceeds series length {len(arr)}"
                )
            return [
                round(float(x), 5)
                for x in arr[start_index:stop]
            ]
        return [default] * horizon_ticks

    return {
        "load_mw": _slice("load_mw", 0.5),
        "pv_mw": _slice("pv_mw", 0.0),
        "wind_mw": _slice("wind_mw", 0.0),
        "price": _slice("price", 40.0),
    }
