#!/usr/bin/env bash
# Clone external dependencies required by the PhaseI scripts into the expected folder layout.
#
# This is needed if your Dreamcloth repo does NOT already include these directories:
#   - ../econ/ECON
#   - ../InstantMesh/InstantMesh
#   - ../Garment3d/Garment3DGen
#
# Usage:
#   ./clone_deps.sh
#
# Env vars (override URLs/branches if needed):
#   ECON_REPO_URL="https://github.com/YuliangXiu/ECON.git"
#   ECON_REF="main"
#   INSTANTMESH_REPO_URL="https://github.com/TencentARC/InstantMesh.git"
#   INSTANTMESH_REF="main"
#   GARMENT3D_REPO_URL="<set-me>"
#   GARMENT3D_REF="main"
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

log() { echo "[INFO] $*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

command -v git >/dev/null 2>&1 || die "git not found in PATH"

ECON_REPO_URL="${ECON_REPO_URL:-https://github.com/YuliangXiu/ECON.git}"
ECON_REF="${ECON_REF:-main}"

INSTANTMESH_REPO_URL="${INSTANTMESH_REPO_URL:-https://github.com/TencentARC/InstantMesh.git}"
INSTANTMESH_REF="${INSTANTMESH_REF:-main}"

# Garment3DGen upstream can vary by project; require explicit URL so we don't clone the wrong repo.
GARMENT3D_REPO_URL="${GARMENT3D_REPO_URL:-}"
GARMENT3D_REF="${GARMENT3D_REF:-main}"

mkdir -p "${REPO_ROOT}/econ" "${REPO_ROOT}/InstantMesh" "${REPO_ROOT}/Garment3d"

if [ ! -d "${REPO_ROOT}/econ/ECON" ]; then
  log "Cloning ECON -> ${REPO_ROOT}/econ/ECON"
  git clone --depth 1 --branch "${ECON_REF}" "${ECON_REPO_URL}" "${REPO_ROOT}/econ/ECON"
else
  log "ECON already present: ${REPO_ROOT}/econ/ECON"
fi

if [ ! -d "${REPO_ROOT}/InstantMesh/.git" ]; then
  log "Cloning InstantMesh -> ${REPO_ROOT}/InstantMesh"
  git clone --depth 1 --branch "${INSTANTMESH_REF}" "${INSTANTMESH_REPO_URL}" "${REPO_ROOT}/InstantMesh"
else
  log "InstantMesh already present: ${REPO_ROOT}/InstantMesh"
fi

# Validate expected InstantMesh layout
if [ ! -d "${REPO_ROOT}/InstantMesh/InstantMesh" ]; then
  die "InstantMesh repo cloned, but expected folder missing: ${REPO_ROOT}/InstantMesh/InstantMesh"
fi

if [ ! -d "${REPO_ROOT}/Garment3d/Garment3DGen" ]; then
  if [ -z "${GARMENT3D_REPO_URL}" ]; then
    die "Garment3DGen not found at ${REPO_ROOT}/Garment3d/Garment3DGen. Set GARMENT3D_REPO_URL and re-run ./clone_deps.sh"
  fi
  log "Cloning Garment3DGen -> ${REPO_ROOT}/Garment3d/Garment3DGen"
  git clone --depth 1 --branch "${GARMENT3D_REF}" "${GARMENT3D_REPO_URL}" "${REPO_ROOT}/Garment3d/Garment3DGen"
else
  log "Garment3DGen already present: ${REPO_ROOT}/Garment3d/Garment3DGen"
fi

log "Dependency clone step complete."


