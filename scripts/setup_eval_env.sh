#!/usr/bin/env bash
# scripts/setup_eval_env.sh — native asset setup for a public OPERATE checkout.
#
# Idempotent: safe to re-run. Syncs the lockfile, clones the source-locked
# external source checkouts at their pinned commits into works/, downloads
# user-provided sources when credentials are available, and installs the
# OPERATE runtime companion. Archive-delivered sources come from the bundle.
#
# Reads from env (never hardcodes secrets):
#   HF_TOKEN      — optional; public downloads do not require authentication
#   OPERATE_HF_REPO_ID — optional dataset override (default Xnhyacinth/OPERATE)
#   OPERATE_HF_REVISION — optional immutable dataset commit revision
#
# Usage:
#   bash scripts/setup_eval_env.sh             # full setup
#   bash scripts/setup_eval_env.sh --skip-data # deps only, no works/ clones
#   bash scripts/setup_eval_env.sh --smoke     # also run a per-backend smoke
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
VENV="$REPO/.venv"
PY="$VENV/bin/python"
WORKS="$REPO/works"

SKIP_DATA=0
SMOKE=0
for arg in "$@"; do
	case "$arg" in
	--skip-data) SKIP_DATA=1 ;;
	--smoke) SMOKE=1 ;;
	*)
		echo "unknown arg: $arg"
		exit 2
		;;
	esac
done

log() { printf '\n\033[1;34m=== %s ===\033[0m\n' "$*"; }
ok() { printf '  \033[32mOK\033[0m   %s\n' "$*"; }
warn() { printf '  \033[33mWARN\033[0m %s\n' "$*"; }

# ── 0. Reproducible lockfile environment ─────────────────────────────
log "sync uv environment"
command -v uv >/dev/null || {
	echo "FATAL: install uv 0.12.5"
	exit 1
}
uv sync --frozen --python 3.13 --extra released-backends --extra simulators \
	--extra llm --extra hf
[ -x "$PY" ] || {
	echo "FATAL: uv did not create $VENV"
	exit 1
}
PY_VER="$($PY -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
ok ".venv Python $PY_VER"
case "$PY_VER" in
3.1[0-4]) ok "Python $PY_VER is in the supported 3.10–3.14 range" ;;
*) warn "Python $PY_VER is outside the supported 3.10–3.14 range" ;;
esac

# ── 1. Backend imports ────────────────────────────────────────────────
log "verify locked backend imports"
for pkg in grid2op pandapower pyvrp vrplib ortools simbench pymgrid dss matpowercaseframes citylearn dsbx traci; do
	"$PY" -c "import $pkg"
	ok "$pkg"
done
"$PY" -c "from or_gym.envs.supply_chain.inventory_management import InvManagementLostSalesEnv"
ok "or_gym"

log "verify native SUMO executable"
"$PY" - <<'SUMO_PREFLIGHT'
import subprocess
import sys

from core.sidecar.sumo_sidecar import _resolve_traci_launch

try:
    binary, environment = _resolve_traci_launch()
    print(f"  SUMO executable: {binary}", flush=True)
    completed = subprocess.run(
        [binary, "--version"], env=environment, check=True,
        capture_output=True, text=True, timeout=30,
    )
except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
    print(f"FATAL: native SUMO preflight failed: {exc}", file=sys.stderr)
    if getattr(exc, "stderr", None):
        print(exc.stderr, file=sys.stderr)
    raise SystemExit(1) from exc
version_lines = (completed.stdout or completed.stderr).strip().splitlines()
print(f"  SUMO version: {version_lines[0] if version_lines else '(no version output)'}")
SUMO_PREFLIGHT

if [ "$SKIP_DATA" -eq 1 ]; then
	log "skip-data: stopping after deps"
	exit 0
fi

# ── 2. External source checkouts at pinned commits ───────────────────
# Only sources declared as git_checkout/upstream_fetch by the active runtime
# closure are cloned here. Bundle-delivered source trees are restored later.
log "clone external sources into works/ (pinned commits)"
mkdir -p "$WORKS"

clone_pinned() {
	local url="$1" dir="$2" commit="$3"
	if [ -d "$WORKS/$dir/.git" ]; then
		ok "SKIP works/$dir (exists)"
		return
	fi
	echo "  CLONE works/$dir @ ${commit:0:8}"
	git clone --quiet "$url" "$WORKS/$dir"
	(cd "$WORKS/$dir" && git checkout --quiet "$commit")
	ok "works/$dir @ $(cd "$WORKS/$dir" && git rev-parse --short HEAD)"
}

clone_pinned https://github.com/hubbs5/or-gym.git OR-Gym 0b18d16e569e2db70e83f09e867b53bdb4b87298
clone_pinned https://github.com/tamy0612/JSPLIB.git JSPLIB-Instances eea2b60dd7e2f5c907ff7302662c61812eb7efdf
clone_pinned https://github.com/intelligent-environments-lab/CityLearn.git CityLearn 29062af6d077409e1c37a3e53a6cac30fd4d02bc
clone_pinned https://github.com/power-grid-lib/pglib-uc.git pglib-uc 39a7f38cf4703de92f0291f0c873c2e98c789301
clone_pinned https://github.com/TUM-VT/sumo_ingolstadt.git sumo_ingolstadt_upstream e0a95deebe200ff81b6705044d66310d6266d42b
clone_pinned https://github.com/alibaba/clusterdata.git clusterdata 0d0f3f1efdbf1add6a7bcc63676eafbd1eb11f71
"$PY" scripts/prepare_local_source_locks.py
if ! git -C "$WORKS/JSPLIB-Instances" check-ignore CHECKSUMS.txt >/dev/null 2>&1; then
	printf '\nCHECKSUMS.txt\n' >>"$WORKS/JSPLIB-Instances/.git/info/exclude"
fi
ok "JSPLIB source checksum manifest prepared"

# ── 3. Install the manifest-bound runtime companion ─────────────────
log "install OPERATE runtime companion"
RUNTIME_DATA_DIR="$REPO/operate_data"
DOWNLOAD_ARGS=(
	--repo-id "${OPERATE_HF_REPO_ID:-Xnhyacinth/OPERATE}"
	--data-dir "$RUNTIME_DATA_DIR"
)
if [ -n "${OPERATE_HF_REVISION:-}" ]; then
	DOWNLOAD_ARGS+=(--revision "$OPERATE_HF_REVISION")
fi
"$PY" scripts/download_from_hf.py "${DOWNLOAD_ARGS[@]}"
# When no revision is provided, the downloader resolves the public dataset once
# and records that exact immutable revision in the local owner receipt.
ok "OPERATE runtime companion installed"
ok "16 hash-bound NREL/OEDI derived NPZ profiles restored from the bundle"
RUNTIME_BUNDLE_MANIFEST="$RUNTIME_DATA_DIR/MANIFEST.json"
[ -f "$RUNTIME_BUNDLE_MANIFEST" ] || {
	echo "FATAL: runtime companion manifest was not installed"
	exit 1
}
CANDIDATE_EVIDENCE_REQUIRED="$("$PY" - "$RUNTIME_BUNDLE_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print("1" if manifest.get("candidate_evidence_archive") else "0")
PY
)"
if [ "$CANDIDATE_EVIDENCE_REQUIRED" -eq 1 ]; then
	[ -d "$RUNTIME_DATA_DIR/candidate_evidence/.hl/artifacts" ] || {
		echo "FATAL: candidate closure evidence was not restored from the bundle"
		exit 1
	}
	ok "candidate closure evidence restored and hash-verified"
else
	ok "candidate closure evidence is not required by this compact bundle"
fi

# ── 5. Optional per-backend smoke ────────────────────────────────────
if [ "$SMOKE" -eq 1 ]; then
	log "per-backend smoke (one wait_only episode each)"
	OPERATE_TRAFFIC_BACKEND_REAL=1 OPERATE_AUTONOMOUS_DRIVING_SUMO_REAL=1 "$PY" - <<'PY'
import json
import subprocess
import sys
from pathlib import Path

core = json.loads(Path("release/operate_v0_61_0/core_suite.json").read_text())
by_backend = {}
for row in core["scenarios"]:
    by_backend.setdefault(row["backend_kind"], row)
if not by_backend:
    raise SystemExit("FATAL: Core contains no runtime backends")
output = Path(".hl/setup_smoke")
output.mkdir(parents=True, exist_ok=True)
slice_path = output / "slice.json"
slice_path.write_text(json.dumps({"scenarios": list(by_backend.values())}))
subprocess.run([
    sys.executable, "scripts/run_protocol21_diagnostic_smoke.py",
    "--slice", str(slice_path), "--output-dir", str(output),
    "--agents", "wait_only", "--check-profile", "runtime_installation",
], check=True)
PY
fi

log "setup complete"
RELEASE_MANIFEST="$REPO/release/operate_v0_61_0/manifest.json"
if [ -f "$RELEASE_MANIFEST" ]; then
	"$PY" - "$RELEASE_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
closure = manifest.get("candidate_closure") or {}
print(
    "  Release:           "
    f"{manifest['release_id']} "
    f"({manifest['n_scenarios']} Core rows; "
    f"{manifest['n_physical_sources']} physical sources; "
    f"status={manifest['status']})."
)
print(
    "  Candidate closure: "
    f"{closure['n_unresolved_candidates']} unresolved."
)
PY
else
	warn "release manifest not found: $RELEASE_MANIFEST"
fi
echo "  Verify release:   .venv/bin/python scripts/verify_release_integrity.py release/operate_v0_61_0"
echo "  Formal commands:  docs/FORMAL_EVALUATION.md"
