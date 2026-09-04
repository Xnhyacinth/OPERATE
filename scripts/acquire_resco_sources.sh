#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="${repo_root}/works/RESCO"
source_url="https://github.com/Pi-Star-Lab/RESCO.git"
source_commit="f1ed9a174f8de41fc9d8689373b836bc882570dc"

if [[ ! -d "${source_dir}/.git" ]]; then
  git clone "${source_url}" "${source_dir}"
fi
git -C "${source_dir}" fetch origin "${source_commit}"
git -C "${source_dir}" checkout --detach "${source_commit}"
"${repo_root}/.venv/bin/python" "${repo_root}/scripts/audit_resco_replacement_sources.py"
